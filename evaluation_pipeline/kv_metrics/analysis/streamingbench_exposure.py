"""StreamingBench ``real`` exposure scan (structure-only, no correctness).

Purpose: measure the REAL effective-compaction exposure per query before
choosing the 20-video experiment set. Selection never uses correctness or any
fidelity metric -- only structural information available before answering.

Key optimisation (verified separately by
``verify_query_independent_schedule``): HERMES compaction happens while frames
are encoded, i.e. BEFORE the query prompt is injected, so the compaction
schedule up to ``t`` does not depend on the question text. One compressed-only
run to the LATEST query of a video therefore yields ``k_q`` for all of its
queries (``k_q = #{effective compactions with end_time <= t_q}``).
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .loader import read_jsonl


def parse_timestamp(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parts = [float(p) for p in str(value).split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def build_candidates(
    csv_path: str,
    max_latest_query_time: float = 260.0,
) -> List[dict]:
    """Group the Real-Time Visual Understanding CSV by video, keep <= cap."""

    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8", newline="")))
    per: Dict[str, List[dict]] = {}
    for row in rows:
        match = re.match(r"^(.*)_sample_(\d+)_(\d+)$", row.get("question_id", ""))
        if not match:
            continue
        video_id, index = match.group(2), int(match.group(3))
        row["_index"] = index
        row["_time"] = parse_timestamp(row.get("time_stamp"))
        per.setdefault(video_id, []).append(row)

    out = []
    for video_id, items in sorted(per.items(), key=lambda kv: int(kv[0])):
        items.sort(key=lambda r: r["_index"])
        times = [r["_time"] for r in items]
        if any(t is None for t in times):
            continue
        latest = max(times)
        if latest > max_latest_query_time:
            continue
        star = max(items, key=lambda r: (r["_time"], r["question_id"]))
        out.append({
            "video_id": video_id,
            "n_queries": len(items),
            "q_times": times,
            "q_ids": [r["question_id"] for r in items],
            "q_task_types": [r.get("task_type") for r in items],
            "q_star_id": star["question_id"],
            "q_star_index": star["_index"],
            "q_star_time": star["_time"],
            "q_star_task": star.get("task_type"),
            "q_star_reason": (
                f"argmax(time_stamp)={star['_time']}s "
                f"(row index {star['_index']}, tie-break on question_id)"
            ),
        })
    return out


def compactions_by_video(run_dir: Path) -> Dict[str, List[dict]]:
    """Read a completed run's compactions.jsonl and group by source video."""

    compactions = read_jsonl(Path(run_dir) / "compactions.jsonl")
    grouped: Dict[str, List[dict]] = {}
    for record in compactions:
        video = record.get("video_id") or str(record.get("event_id", "")).split("|")[0]
        grouped.setdefault(str(video), []).append(record)
    return grouped


def _video_key(video_id: str) -> str:
    return f"sample_{video_id}" if not str(video_id).startswith("sample_") else str(video_id)


def effective_k_at(t_query: float, compactions: List[dict]) -> int:
    """Number of EFFECTIVE compactions completed at/before ``t_query``."""

    count = 0
    for record in compactions:
        if not record.get("is_effective_compaction"):
            continue
        end = record.get("end_time", record.get("stream_time"))
        if end is not None and float(end) <= float(t_query) + 1e-9:
            count += 1
    return count


def trigger_k_at(t_query: float, compactions: List[dict]) -> int:
    count = 0
    for record in compactions:
        end = record.get("end_time", record.get("stream_time"))
        if end is not None and float(end) <= float(t_query) + 1e-9:
            count += 1
    return count


def build_exposure_rows(candidates: List[dict], run_dir: Path) -> List[dict]:
    grouped = compactions_by_video(run_dir)
    rows = []
    for cand in candidates:
        comps = grouped.get(_video_key(cand["video_id"]), [])
        row = {"video_id": cand["video_id"], "q_star_time": cand["q_star_time"],
               "q_star_id": cand["q_star_id"]}
        for i in range(5):
            t = cand["q_times"][i] if i < len(cand["q_times"]) else None
            row[f"q{i + 1}_time"] = t
            row[f"q{i + 1}_task_type"] = (
                cand["q_task_types"][i] if i < len(cand["q_task_types"]) else None
            )
            row[f"k_q{i + 1}"] = (
                effective_k_at(t, comps) if t is not None else None
            )
            row[f"triggers_q{i + 1}"] = (
                trigger_k_at(t, comps) if t is not None else None
            )
        k_all = [row[f"k_q{i + 1}"] for i in range(5) if row[f"k_q{i + 1}"] is not None]
        row["k_q_star"] = max(k_all) if k_all else None
        rows.append(row)
    return rows


FIELDS = [
    "video_id",
    "q1_time", "q2_time", "q3_time", "q4_time", "q5_time",
    "q_star_time", "q_star_id",
    "k_q1", "k_q2", "k_q3", "k_q4", "k_q5", "k_q_star",
    "triggers_q1", "triggers_q2", "triggers_q3", "triggers_q4", "triggers_q5",
    "q1_task_type", "q2_task_type", "q3_task_type", "q4_task_type", "q5_task_type",
]


def write_exposure_csv(rows: List[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDS})


def k_histogram(rows: List[dict], field: str = "k_q") -> Dict[int, int]:
    hist: Dict[int, int] = {}
    for row in rows:
        for i in range(1, 6):
            value = row.get(f"{field}{i}")
            if value is None:
                continue
            hist[int(value)] = hist.get(int(value), 0) + 1
    return dict(sorted(hist.items()))
