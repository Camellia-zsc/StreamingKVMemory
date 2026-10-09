"""Select benchmark samples that actually trigger HERMES KV compression.

Selection is grounded in **real** data whenever available: if a previous run
left ``compactions.jsonl`` / ``paired/*.jsonl`` sidecars, compaction counts and
compression ratios are read from them. Only when no history exists do we fall
back to an explicit, labelled *estimate* from video duration and sample rate.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from . import loader


@dataclass
class SampleCandidate:
    sample_id: str
    video_path: str
    task: Optional[str] = None
    duration: Optional[float] = None
    num_queries: int = 0
    query_times: List[float] = field(default_factory=list)
    max_kv_tokens: Optional[int] = None
    compaction_count: Optional[int] = None
    max_compression_ratio: Optional[float] = None
    mean_compression_ratio: Optional[float] = None
    estimated_visual_tokens: Optional[int] = None
    effective_compaction_count: Optional[int] = None
    task_group: Optional[str] = None
    source: str = "estimated"  # "observed" or "estimated"

    def to_dict(self) -> dict:
        return asdict(self)


def _video_duration(video_path: str) -> Optional[float]:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or count <= 0:
            return None
        return count / fps
    except Exception:  # noqa: BLE001
        return None


def _estimate_tokens(duration: Optional[float], sample_fps: float, max_side: int) -> Optional[int]:
    if duration is None:
        return None
    tokens_per_frame = max(1, (int(max_side) // 28) ** 2)
    frames = max(1, int(duration * float(sample_fps)))
    return frames * tokens_per_frame


def build_candidates(
    samples,
    sample_fps: float = 0.5,
    max_side: int = 336,
) -> List[SampleCandidate]:
    candidates = []
    for sample in samples:
        duration = _video_duration(sample.video_path)
        query_times = [
            float(q.timestamp) for q in sample.questions if q.timestamp is not None
        ]
        task = None
        if sample.questions:
            task = sample.questions[0].task_type
        cand = SampleCandidate(
            sample_id=str(sample.sample_id),
            video_path=sample.video_path,
            task=task,
            duration=duration,
            num_queries=len(sample.questions),
            query_times=query_times,
            estimated_visual_tokens=_estimate_tokens(duration, sample_fps, max_side),
        )
        candidates.append(cand)
    return candidates


def apply_history(candidates: List[SampleCandidate], metrics_root: Path) -> None:
    """Overlay real compaction statistics from existing sidecar runs."""

    metrics_root = Path(metrics_root)
    if not metrics_root.is_dir():
        return
    stats = {}
    for run_dir in sorted(metrics_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "experiments":
            continue
        compactions = loader.read_jsonl(run_dir / "compactions.jsonl")
        for record in compactions:
            sample_id = str(record.get("sample_id") or loader.video_id_from_event(record.get("event_id", "")))
            entry = stats.setdefault(
                sample_id,
                {
                    "compaction_count": 0,
                    "effective_compaction_count": 0,
                    "max_kv_tokens": 0,
                    "max_compression_ratio": None,
                    "compression_ratios": [],
                },
            )
            entry["compaction_count"] += 1
            if record.get("is_effective_compaction"):
                entry["effective_compaction_count"] += 1
            if record.get("kv_before") is not None:
                entry["max_kv_tokens"] = max(entry["max_kv_tokens"], int(record["kv_before"]))
            ratio = record.get("compression_ratio")
            if ratio is not None:
                entry["compression_ratios"].append(float(ratio))
                if entry["max_compression_ratio"] is None:
                    entry["max_compression_ratio"] = float(ratio)
                else:
                    entry["max_compression_ratio"] = max(entry["max_compression_ratio"], float(ratio))

    for cand in candidates:
        entry = stats.get(cand.sample_id)
        if not entry:
            continue
        cand.compaction_count = entry["compaction_count"]
        cand.effective_compaction_count = entry["effective_compaction_count"]
        cand.max_kv_tokens = entry["max_kv_tokens"] or None
        cand.max_compression_ratio = entry["max_compression_ratio"]
        ratios = entry["compression_ratios"]
        cand.mean_compression_ratio = sum(ratios) / len(ratios) if ratios else None
        cand.source = "observed"


def _filter_and_rank(
    candidates: List[SampleCandidate],
    min_compactions: Optional[int],
    min_video_duration: Optional[float],
    min_max_kv_tokens: Optional[int],
    min_compression_ratio: Optional[float],
    include_zero_compaction: bool,
    max_video_duration: Optional[float] = None,
):
    kept = []
    for cand in candidates:
        if min_video_duration is not None and (cand.duration or 0) < min_video_duration:
            continue
        if max_video_duration is not None and (cand.duration or 0) > max_video_duration:
            continue
        if min_max_kv_tokens is not None and (cand.max_kv_tokens or 0) < min_max_kv_tokens:
            continue
        if min_compression_ratio is not None and (cand.max_compression_ratio or 0) < min_compression_ratio:
            continue
        if min_compactions is not None and cand.compaction_count is not None:
            if cand.compaction_count < min_compactions and not include_zero_compaction:
                continue
        kept.append(cand)
    return kept


def _stratified(candidates: List[SampleCandidate], num_samples: int) -> List[SampleCandidate]:
    """Stratify by duration into medium / long / very_long thirds."""

    if len(candidates) <= num_samples:
        return list(candidates)
    ordered = sorted(candidates, key=lambda c: (c.duration or 0))
    third = len(ordered) // 3 or 1
    strata = [ordered[:third], ordered[third:2 * third], ordered[2 * third:]]
    out = []
    per = max(1, num_samples // 3)
    for group in strata:
        out.extend(group[-per:] if len(group) > per else group)
    # fill remaining with the longest not yet picked
    seen = {c.sample_id for c in out}
    for cand in reversed(ordered):
        if len(out) >= num_samples:
            break
        if cand.sample_id not in seen:
            out.append(cand)
            seen.add(cand.sample_id)
    return out[:num_samples]


SCALE_DEFAULTS = {
    "smoke": {"num_samples": 3, "min_compactions": 3},
    "medium": {"num_samples": 20, "min_compactions": 3},
    "full": {"num_samples": None, "min_compactions": 3},
}


def select_samples(
    samples,
    metrics_root: str = "outputs/kv_metrics",
    min_compactions: Optional[int] = 3,
    min_video_duration: Optional[float] = None,
    max_video_duration: Optional[float] = None,
    min_max_kv_tokens: Optional[int] = None,
    min_compression_ratio: Optional[float] = None,
    num_samples: Optional[int] = None,
    top_k_longest: Optional[int] = None,
    top_k_most_compactions: Optional[int] = None,
    scale: str = "smoke",
    sample_fps: float = 0.5,
    max_side: int = 336,
    include_zero_compaction: bool = False,
) -> List[SampleCandidate]:
    defaults = SCALE_DEFAULTS.get(scale, {})
    if num_samples is None:
        num_samples = defaults.get("num_samples")
    if min_compactions is None:
        min_compactions = defaults.get("min_compactions", 3)

    candidates = build_candidates(samples, sample_fps=sample_fps, max_side=max_side)
    apply_history(candidates, Path(metrics_root))

    filtered = _filter_and_rank(
        candidates,
        min_compactions=min_compactions,
        min_video_duration=min_video_duration,
        max_video_duration=max_video_duration,
        min_max_kv_tokens=min_max_kv_tokens,
        min_compression_ratio=min_compression_ratio,
        include_zero_compaction=include_zero_compaction,
    )
    # If the strict compaction filter removes everything (no history yet),
    # fall back to the longest videos so a first experiment can still run.
    if not filtered:
        filtered = sorted(candidates, key=lambda c: (c.duration or 0), reverse=True)

    if top_k_most_compactions:
        filtered = sorted(
            filtered,
            key=lambda c: (c.compaction_count or 0),
            reverse=True,
        )[:top_k_most_compactions]
    elif top_k_longest:
        filtered = sorted(filtered, key=lambda c: (c.duration or 0), reverse=True)[:top_k_longest]
    elif num_samples:
        filtered = _stratified(filtered, num_samples)

    return filtered


def write_selection(candidates: List[SampleCandidate], path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "num_selected": len(candidates),
        "samples": [c.to_dict() for c in candidates],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_selection(path: Path) -> List[dict]:
    data = loader.read_json(Path(path), {}) or {}
    if isinstance(data, list):
        return data
    return data.get("samples", [])


# --------------------------------------------------------------------- #
# Compaction-accuracy experiment: backward/realtime samples with nested queries
# --------------------------------------------------------------------- #
def select_compaction_accuracy_samples(
    samples,
    target_groups=("backward", "realtime"),
    num_samples: int = 20,
    target_total_queries: int = 100,
    seed: int = 20260930,
    metrics_root: str = "outputs/kv_metrics",
    sample_fps: float = 0.5,
    max_side: int = 336,
    min_queries: int = 2,
    min_effective_compactions: int = 2,
    budget: int = 2000,
    max_query_time: Optional[float] = None,
) -> List[dict]:
    """Pick samples for the compaction-accuracy study.

    Rules (no cherry-picking on correctness / fidelity):
      * task must belong to a target group (groups come from the adapter's
        ``OVO_TASK_GROUPS``, never hard-coded here);
      * the sample must expose several queries (nested prefixes);
      * prefer samples whose longest query reaches >= ``min_effective_compactions``
        effective compactions (observed history, else an explicit estimate).
    """

    import random

    from ...adapters import OVO_TASK_GROUPS

    allowed = set()
    group_of = {}
    for group in target_groups:
        for task in OVO_TASK_GROUPS.get(group, []):
            allowed.add(task)
            group_of[task] = group

    candidates = build_candidates(samples, sample_fps=sample_fps, max_side=max_side)
    apply_history(candidates, Path(metrics_root))

    tokens_per_frame = max(1, (int(max_side) // 28) ** 2)
    rows = []
    for cand, sample in zip(candidates, samples):
        task = sample.questions[0].task_type if sample.questions else None
        if task not in allowed:
            continue
        cand.task_group = group_of.get(task)
        qtimes = [float(q.timestamp) for q in sample.questions if q.timestamp is not None]
        if not qtimes:
            continue
        if max_query_time is not None and max(qtimes) > float(max_query_time):
            # feasibility guard: the eager Full-KV oracle must fit in memory
            continue
        if cand.effective_compaction_count is None:
            est_tokens = _estimate_tokens(cand.duration, sample_fps, max_side) or 0
            cand.effective_compaction_count = int(
                max(0, est_tokens // max(budget, 1) - 1)
            )
            cand.source = "estimated"
        rows.append({
            "sample_id": cand.sample_id,
            "task_type": task,
            "task_group": cand.task_group,
            "num_queries": len(sample.questions),
            "query_times": sorted(qtimes),
            "longest_query_id": (
                max(sample.questions, key=lambda q: (q.timestamp or -1)).question_id
            ),
            "longest_query_time": max(qtimes),
            "duration_sec": cand.duration,
            "estimated_or_observed_effective_compactions": cand.effective_compaction_count,
            "compaction_source": cand.source,
            "selection_reason": (
                f"task_group={cand.task_group}; nested_queries={len(sample.questions)}; "
                f"effective_compactions({cand.source})={cand.effective_compaction_count}"
            ),
        })

    # deterministic ordering: shuffle seeded, then take the qualifying ones
    rng = random.Random(seed)
    qualifying = [
        r for r in rows
        if r["num_queries"] >= min_queries
        and r["estimated_or_observed_effective_compactions"] >= min_effective_compactions
    ]
    others = [
        r for r in rows
        if r not in qualifying and r["num_queries"] >= min_queries
    ]
    rng.shuffle(qualifying)
    rng.shuffle(others)

    selected = []
    total_queries = 0
    for row in qualifying + others:
        if len(selected) >= num_samples:
            break
        selected.append(row)
        total_queries += row["num_queries"]
    if total_queries < target_total_queries:
        # keep adding (seeded order) until the query budget is met or exhausted
        for row in qualifying + others:
            if row in selected or len(selected) >= num_samples:
                continue
            selected.append(row)
            total_queries += row["num_queries"]
            if total_queries >= target_total_queries:
                break
    return selected


def write_compaction_selection(rows, path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"meta": meta, "num_selected": len(rows),
             "total_queries": sum(r["num_queries"] for r in rows),
             "samples": rows},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
