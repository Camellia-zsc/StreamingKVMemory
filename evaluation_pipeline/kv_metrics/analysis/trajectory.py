"""Single-video Query x cumulative-compaction-count fidelity trajectory.

This is a **cumulative operational fidelity** view, not a per-compaction causal
effect: each point answers "when this query arrived, the cache had already
undergone n completed HERMES compactions -- what was the Full-vs-Compressed
fidelity then?". The query content and the video content also change between
points, so the trajectory must not be read as the effect of compactions n1..n2.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from . import statistics as st
from .loader import ExperimentData, RunData


def find_run_for_budget(experiment: ExperimentData, budget: Optional[int]) -> Optional[RunData]:
    if budget is None:
        # pick the run with the most compactions
        runs = experiment.ordered_runs()
        return max(runs, key=lambda r: len(r.compactions)) if runs else None
    wanted = f"hermes_budget_{int(budget)}"
    if wanted in experiment.runs:
        return experiment.runs[wanted]
    # fall back: any compressed run whose config budget matches
    for run in experiment.ordered_runs():
        memory = (run.config or {}).get("memory", {}) or {}
        try:
            if int(memory.get("compression", {}).get("budget", -1)) == int(budget):
                return run
        except (TypeError, ValueError):
            continue
    return None


def _longest_query(questions, field: str = "query_time"):
    """q_star = argmax(query_time) with a deterministic tie-break.

    StreamingBench ``real`` has non-monotone and duplicated timestamps
    (18 duplicates / 66 non-increasing over 500 videos), so the latest query
    must be found by timestamp, never by row index. Ties break on
    ``question_id`` (stable, deterministic).
    """

    candidates = [q for q in questions if q.get(field) is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda q: (float(q[field]), str(q.get("query_id") or q.get("question_id") or "")),
    )


def _anchor_time(event: dict) -> Optional[float]:
    value = event.get("stream_time")
    if value is None:
        value = event.get("query_time")
    return float(value) if value is not None else None


def _compaction_end(record: dict) -> Optional[float]:
    value = record.get("end_time")
    if value is None:
        value = record.get("trigger_time")
    if value is None:
        value = record.get("stream_time")
    return float(value) if value is not None else None


def compaction_count_at_query(event: dict, compactions_for_event: List[dict]) -> int:
    """Number of compactions that had **completed** at the query anchor time.

    Uses ``compaction_end_time <= query_anchor_time``; a compaction that was
    triggered but whose end time is after the anchor does not count.
    """

    anchor = _anchor_time(event)
    if anchor is None:
        return 0
    count = 0
    for record in compactions_for_event:
        end = _compaction_end(record)
        if end is not None and end <= anchor + 1e-9:
            count += 1
    return count


def _js_display(value):
    """Visualisation clamp for tiny negative JS floating-point artifacts."""

    if value is None:
        return None
    return max(float(value), 0.0)


def build_query_trajectory(
    experiment: ExperimentData,
    sample_id: str,
    budget: Optional[int] = None,
) -> dict:
    run = find_run_for_budget(experiment, budget)
    if run is None:
        raise ValueError(f"no run found for budget={budget} in {experiment.name}")

    events = [
        e for e in run.events if str(e.get("sample_id")) == str(sample_id)
    ]
    events.sort(key=lambda e: float(e.get("query_time") or 0.0))

    compactions_for_sample = [
        c for c in run.compactions if str(c.get("sample_id")) == str(sample_id)
    ]
    by_event: Dict[str, List[dict]] = {}
    for record in compactions_for_sample:
        by_event.setdefault(str(record.get("event_id")), []).append(record)
    for values in by_event.values():
        values.sort(key=lambda r: (_compaction_end(r) or 0.0, r.get("compaction_index") or 0))

    rows = []
    for event in events:
        eid = str(event.get("event_id"))
        count = compaction_count_at_query(event, by_event.get(eid, []))
        rows.append({
            "sample_id": str(sample_id),
            "query_id": event.get("query_id"),
            "event_id": eid,
            "query_time": event.get("query_time"),
            "stream_time": event.get("stream_time"),
            "compaction_count_at_query": count,
            "compaction_index_reported": event.get("compaction_index"),
            "RAM": event.get("ram_mean"),
            "AORE": event.get("aore_mean"),
            "JS": event.get("js"),
            "JS_display": _js_display(event.get("js")),
            "KL": event.get("kl"),
            "CR": event.get("compression_ratio"),
            "current_kv_bytes": event.get("compressed_kv_bytes"),
            "full_kv_bytes": event.get("full_kv_bytes"),
            "compressed_kv_bytes": event.get("compressed_kv_bytes"),
            "task_prediction": event.get("task_prediction"),
            "task_correct": event.get("task_correct"),
        })

    ratios = [r["CR"] for r in rows if r["CR"] is not None]
    meta = {
        "experiment": experiment.name,
        "run": run.name,
        "benchmark": (run.config or {}).get("benchmark"),
        "sample_id": str(sample_id),
        "budget": budget,
        "n_queries": len(rows),
        "total_compactions": len(compactions_for_sample),
        "n_events_with_compactions": len(by_event),
        "mean_compression_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "max_compaction_count_at_query": max(
            (r["compaction_count_at_query"] for r in rows), default=0
        ),
        "duration_sec": max(
            [r["stream_time"] for r in rows if r["stream_time"] is not None]
            or [r["query_time"] for r in rows if r["query_time"] is not None]
            or [0.0]
        ),
    }
    return {"rows": rows, "meta": meta}


# --------------------------------------------------------------------- #
# candidate selection
# --------------------------------------------------------------------- #
def select_trajectory_candidates(
    experiment: ExperimentData,
    min_queries: int = 5,
    min_compactions: int = 3,
    min_unique: int = 4,
    include_below_threshold: bool = True,
) -> List[dict]:
    candidates = []
    for run in experiment.ordered_runs():
        if not run.compactions:
            continue
        sample_ids = sorted({str(e.get("sample_id")) for e in run.events})
        for sid in sample_ids:
            events = [e for e in run.events if str(e.get("sample_id")) == sid]
            events.sort(key=lambda e: float(e.get("query_time") or 0.0))
            comps = [c for c in run.compactions if str(c.get("sample_id")) == sid]
            by_event: Dict[str, List[dict]] = {}
            for record in comps:
                by_event.setdefault(str(record.get("event_id")), []).append(record)
            counts = [
                compaction_count_at_query(e, by_event.get(str(e.get("event_id")), []))
                for e in events
            ]
            qtimes = [float(e.get("query_time")) for e in events if e.get("query_time") is not None]
            candidates.append({
                "sample_id": sid,
                "video_id": sid,
                "benchmark_subset": (run.config or {}).get("benchmark"),
                "experiment": run.name,
                "duration_sec": max(qtimes) if qtimes else None,
                "num_queries": len(events),
                "num_compactions": len(comps),
                "num_unique_query_compaction_counts": len(set(counts)),
                "min_query_time": min(qtimes) if qtimes else None,
                "max_query_time": max(qtimes) if qtimes else None,
                "min_compaction_count_at_query": min(counts) if counts else 0,
                "max_compaction_count_at_query": max(counts) if counts else 0,
                "meets_criteria": bool(
                    len(events) >= min_queries
                    and len(comps) >= min_compactions
                    and len(set(counts)) >= min_unique
                ),
            })

    candidates.sort(
        key=lambda c: (
            c["num_unique_query_compaction_counts"],
            c["num_queries"],
            c["num_compactions"],
        ),
        reverse=True,
    )
    if not include_below_threshold:
        candidates = [c for c in candidates if c["meets_criteria"]]
    return candidates


def recommended_sample(candidates: List[dict]) -> Optional[str]:
    for cand in candidates:
        if cand.get("meets_criteria"):
            return str(cand["sample_id"])
    return str(candidates[0]["sample_id"]) if candidates else None


# --------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------- #
def write_candidates_csv(candidates: List[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "video_id", "benchmark_subset", "experiment", "duration_sec",
        "num_queries", "num_compactions", "num_unique_query_compaction_counts",
        "min_query_time", "max_query_time",
        "min_compaction_count_at_query", "max_compaction_count_at_query",
        "meets_criteria",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cand in candidates:
            writer.writerow({k: cand.get(k) for k in fields})


def write_trajectory_csv(rows: List[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "query_id", "event_id", "query_time", "stream_time",
        "compaction_count_at_query", "compaction_index_reported",
        "RAM", "AORE", "JS", "JS_display", "KL", "CR",
        "full_kv_bytes", "compressed_kv_bytes", "current_kv_bytes",
        "task_prediction", "task_correct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def trajectory_stats(rows: List[dict]) -> dict:
    """Trend summary for the report (monotonicity + volatility)."""

    out = {}
    for key in ("RAM", "AORE", "JS", "KL"):
        values = [r.get(key) for r in rows if r.get(key) is not None]
        means = st.sample_level_means(
            [{"sample_id": r["query_id"], key: r.get(key)} for r in rows], key
        )
        out[key] = {
            "first": values[0] if values else None,
            "last": values[-1] if values else None,
            "summary": st.summarize(values),
        }
    return out
