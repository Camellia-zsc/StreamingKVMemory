"""Deterministic coverage-aware selection of 20 StreamingBench real videos.

This stage only READS the exposure scan (observed effective-compaction counts).
It never reads correctness / prediction / RAM / AORE / JS / KL / task_transition.

Selection objective (documented, reproducible)
----------------------------------------------
Greedy for 20 rounds. At each round every unselected video ``v`` is scored by
its MARGINAL contribution given the already-selected set ``S``:

    score(v | S) =
          w_exact  * new_exact_k(v | S)          # + unique integer k values
        + w_nonzero* nonzero_queries(v) / 5      # + queries with k >= 1
        + w_bin    * new_bins(v | S)             # + B1..B4 bins that were empty
        + w_qstar  * min(k_q_star(v), 40) / 40   # + q_star trajectory richness
        + w_task   * new_task_types(v | S)       # + task-type diversity

Hard preference: ``k_q_star >= 5`` (relaxed to ``>= 3`` then to any value only
if fewer than 20 candidates qualify). Ties break deterministically on the
numeric ``video_id``; the RNG seed (20260930) only fixes that deterministic
ordering, never a random draw that could not be reproduced.

The resulting subset is a **coverage_balanced_diagnostic_subset**, NOT a
random sample of StreamingBench real -- its accuracy must never be reported as
the benchmark's overall accuracy.
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


SEED = 20260930
SELECTION_VERSION = "sb_real_coverage_v1"
SELECTION_TYPE = "coverage_balanced_diagnostic_subset"

BINS = [("B0", 0, 0), ("B1", 1, 10), ("B2", 11, 20), ("B3", 21, 30), ("B4", 31, 10**9)]
NONZERO_BINS = [name for name, lo, _ in BINS if lo > 0]

# q_star trajectory richness buckets (coverage target for the 20 q_star runs)
QSTAR_BUCKETS = [("5-10", 5, 10), ("11-20", 11, 20), ("21-30", 21, 30), ("31+", 31, 10**9)]


def bin_of(k: int) -> str:
    for name, lo, hi in BINS:
        if lo <= k <= hi:
            return name
    return "B4"


def qstar_bucket_of(k: int) -> Optional[str]:
    for name, lo, hi in QSTAR_BUCKETS:
        if lo <= k <= hi:
            return name
    return None


# q_star bucket quotas for the 20 videos (availability-aware: the pool only
# has 1 video in 5-10 and 4 in 11-20, so the quota is capped by availability)
QSTAR_QUOTA = {"5-10": 1, "11-20": 3, "21-30": 8, "31+": 8}


PLANS = {
    # Plan A: exposure coverage first (exact-k, non-zero, query bins, q_star bins)
    "plan_a": {"exact": 1.0, "nonzero": 1.0, "bin": 1.5, "qstar_bucket": 2.5,
               "task": 0.0, "qstar_rich": 0.15},
    # Plan B: same coverage backbone + task-type diversity / balance
    "plan_b": {"exact": 1.0, "nonzero": 1.0, "bin": 1.5, "qstar_bucket": 2.5,
               "task": 2.0, "qstar_rich": 0.15},
}


def _task_balance(task_counts: Dict[str, int]) -> float:
    """Lower is better; used to spread task types in Plan B."""

    total = sum(task_counts.values()) or 1
    return max(task_counts.values()) / total if task_counts else 0.0


def load_candidates(csv_path) -> List[dict]:
    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8", newline="")))
    out = []
    for row in rows:
        ks = [int(row[f"k_q{i}"]) for i in range(1, 6)]
        star_id = row.get("q_star_id") or ""
        # the exposure CSV has no per-query id column; derive it from the
        # q_star id (``..._sample_<vid>_<idx>``) which is stable in the CSV
        prefix = star_id.rsplit("_", 1)[0] if star_id else ""
        query_ids = [f"{prefix}_{i}" if prefix else "" for i in range(1, 6)]
        out.append({
            "video_id": row["video_id"],
            "query_ids": query_ids,
            "query_times": [float(row[f"q{i}_time"]) for i in range(1, 6)],
            "task_types": [row.get(f"q{i}_task_type", "") for i in range(1, 6)],
            "effective_k_vector": ks,
            "q_star_time": float(row["q_star_time"]),
            "q_star_id": row.get("q_star_id"),
            "k_q_star": int(row["k_q_star"]),
        })
    return out


def _unique_ks(cands) -> set:
    return {k for c in cands for k in c["effective_k_vector"]}


def _bin_counts(cands) -> Dict[str, int]:
    counts = {name: 0 for name, _, _ in BINS}
    for c in cands:
        for k in c["effective_k_vector"]:
            counts[bin_of(k)] += 1
    return counts


def select(candidates: List[dict], plan: str = "plan_a", n_videos: int = 20,
           min_q_star_k: int = 5) -> List[dict]:
    weights = PLANS[plan]
    pool = [c for c in candidates if c["k_q_star"] >= min_q_star_k]
    if len(pool) < n_videos:
        pool = [c for c in candidates if c["k_q_star"] >= 3]
    if len(pool) < n_videos:
        pool = list(candidates)

    # deterministic order; the seed only fixes this ordering
    rng = random.Random(SEED)
    ordered = sorted(pool, key=lambda c: int(c["video_id"]))
    rng.shuffle(ordered)
    ordered.sort(key=lambda c: (c["k_q_star"], int(c["video_id"])), reverse=True)

    selected: List[dict] = []
    selected_ids = set()
    seen_ks: set = set()
    seen_tasks: set = set()
    seen_bins: set = set()
    seen_qstar_buckets: set = set()
    qstar_counts: Dict[str, int] = {name: 0 for name, _, _ in QSTAR_BUCKETS}
    task_counts: Dict[str, int] = {}

    for _ in range(min(n_videos, len(pool))):
        best = None
        for cand in ordered:
            if cand["video_id"] in selected_ids:
                continue
            ks = cand["effective_k_vector"]
            new_exact = len((seen_ks | set(ks)) - seen_ks)
            nonzero = sum(1 for k in ks if k > 0) / 5.0
            new_bins = len((seen_bins | {bin_of(k) for k in ks}) - seen_bins)
            qb = qstar_bucket_of(cand["k_q_star"])
            # quota-filling: reward candidates from a q_star bucket that is
            # still below its availability-aware quota
            under_quota = (
                1 if (qb is not None and qstar_counts.get(qb, 0) < QSTAR_QUOTA.get(qb, 0))
                else 0
            )
            new_qstar_bucket = 1 if (qb is not None and qb not in seen_qstar_buckets) else 0
            qstar = min(cand["k_q_star"], 40) / 40.0
            new_tasks = len((seen_tasks | set(cand["task_types"])) - seen_tasks)
            # task balance term: prefer candidates whose task types are still
            # under-represented (Plan B only, w_task > 0)
            balance_bonus = 0.0
            if weights["task"] > 0:
                for t in set(cand["task_types"]):
                    if t:
                        balance_bonus += 1.0 / (1.0 + task_counts.get(t, 0))
                balance_bonus /= 5.0
            score = (weights["exact"] * new_exact
                     + weights["nonzero"] * nonzero
                     + weights["bin"] * new_bins
                     + weights["qstar_bucket"] * new_qstar_bucket
                     + weights["qstar_bucket"] * 0.5 * under_quota
                     + weights["qstar_rich"] * qstar
                     + weights["task"] * (new_tasks + balance_bonus))
            key = (score, -int(cand["video_id"]))
            if best is None or key > best[0]:
                best = (key, cand, {
                    "score": score,
                    "new_exact_k": new_exact,
                    "nonzero_queries": int(nonzero * 5),
                    "new_bins": new_bins,
                    "new_qstar_bucket": new_qstar_bucket,
                    "under_qstar_quota": under_quota,
                    "new_task_types": new_tasks,
                })
        if best is None:
            break
        _, cand, detail = best
        selected_ids.add(cand["video_id"])
        seen_ks |= set(cand["effective_k_vector"])
        seen_tasks |= set(cand["task_types"])
        seen_bins |= {bin_of(k) for k in cand["effective_k_vector"]}
        if qstar_bucket_of(cand["k_q_star"]):
            seen_qstar_buckets.add(qstar_bucket_of(cand["k_q_star"]))
            qstar_counts[qstar_bucket_of(cand["k_q_star"])] += 1
        for t in cand["task_types"]:
            if t:
                task_counts[t] = task_counts.get(t, 0) + 1

        entry = dict(cand)
        entry["selection_score"] = round(detail["score"], 6)
        entry["coverage_bins_contributed"] = sorted(
            {bin_of(k) for k in cand["effective_k_vector"]}
        )
        entry["selection_reason"] = (
            f"greedy({plan}): +{detail['new_exact_k']} exact-k, "
            f"{detail['nonzero_queries']}/5 non-zero queries, "
            f"+{detail['new_bins']} query-bin(s), "
            f"+{detail['new_qstar_bucket']} q_star-bucket, "
            f"+{detail['new_task_types']} task type(s), k_q_star={cand['k_q_star']}"
        )
        selected.append(entry)
    return selected


# --------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------- #
def subset_stats(cands: List[dict]) -> dict:
    ks = [k for c in cands for k in c["effective_k_vector"]]
    exact: Dict[int, List[str]] = {}
    for c in cands:
        for k in c["effective_k_vector"]:
            exact.setdefault(k, []).append(c["video_id"])
    bins = {name: 0 for name, _, _ in BINS}
    for k in ks:
        bins[bin_of(k)] += 1
    qstar = sorted(c["k_q_star"] for c in cands)
    tasks = [t for c in cands for t in c["task_types"] if t]
    return {
        "num_videos": len(cands),
        "num_queries": len(ks),
        "k0_queries": sum(1 for k in ks if k == 0),
        "k0_fraction": (sum(1 for k in ks if k == 0) / len(ks)) if ks else None,
        "nonzero_queries": sum(1 for k in ks if k > 0),
        "unique_exact_k": len(exact),
        "exact_k": dict(sorted(exact.items())),
        "bins": bins,
        "q_star_median_k": qstar[len(qstar) // 2] if qstar else None,
        "q_star_max_k": max(qstar) if qstar else None,
        "q_star_buckets": {
            "5-10": sum(1 for k in qstar if 5 <= k <= 10),
            "11-20": sum(1 for k in qstar if 11 <= k <= 20),
            "21-30": sum(1 for k in qstar if 21 <= k <= 30),
            "31+": sum(1 for k in qstar if k >= 31),
        },
        "task_type_counts": {t: tasks.count(t) for t in sorted(set(tasks))},
        "unique_task_types": len(set(tasks)),
    }


def bins_from_queries(cands: List[dict]) -> Dict[str, dict]:
    """Per-bin stats computed DIRECTLY from each video's 5 query k values.

    This is the single source of truth for the report: ``N_query`` counts query
    records, ``N_video`` counts distinct videos with >=1 query in the bin.
    Never derive these from any summary/aggregate field.
    """

    out: Dict[str, dict] = {}
    for name, _, _ in BINS:
        rows = [
            (str(c["video_id"]), int(k))
            for c in cands
            for k in c["effective_k_vector"]
            if bin_of(int(k)) == name
        ]
        videos = sorted({vid for vid, _ in rows}, key=lambda v: int(v))
        out[name] = {
            "bin": name,
            "n_query": len(rows),
            "n_video": len(videos),
            "video_ids": videos,
        }
    return out


def qstar_bucket_videos(cands: List[dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {name: [] for name, _, _ in QSTAR_BUCKETS}
    for c in cands:
        name = qstar_bucket_of(int(c["k_q_star"]))
        if name:
            out[name].append(str(c["video_id"]))
    for name in out:
        out[name] = sorted(out[name], key=lambda v: int(v))
    return out


def check_invariants(cands: List[dict]) -> List[str]:
    """Structural invariants that previously caught a report aggregation bug.

    1. per bin: ``N_video <= N_query``;
    2. per bin: ``#{videos with q_star_k in bin} <= N_video(bin)`` -- q_star is
       itself one of the video's queries, so a video whose q_star falls in a
       bin MUST be counted in that bin's video set;
    3. ``k_q_star == k_vector[q_star_index]`` (the selected JSON must not use a
       stale / estimated / trigger-count field).
    """

    problems: List[str] = []
    bins = bins_from_queries(cands)
    qstar = qstar_bucket_videos(cands)

    for name, stats in bins.items():
        if stats["n_video"] > stats["n_query"]:
            problems.append(
                f"{name}: N_video={stats['n_video']} > N_query={stats['n_query']}"
            )
    for name, videos in qstar.items():
        # q_star buckets 5-10/11-20/21-30/31+ map onto B1/B2/B3/B4
        bin_name = {"5-10": "B1", "11-20": "B2", "21-30": "B3", "31+": "B4"}[name]
        have = set(bins[bin_name]["video_ids"])
        missing = sorted(set(videos) - have, key=lambda v: int(v))
        if missing:
            problems.append(
                f"q_star bucket {name}: {len(videos)} videos but {bin_name} only "
                f"lists {len(have)}; missing={missing}"
            )
    for c in cands:
        star_id = str(c.get("q_star_id") or "")
        try:
            index = int(star_id.rsplit("_", 1)[1]) - 1
        except (IndexError, ValueError):
            continue
        vec = c["effective_k_vector"]
        if 0 <= index < len(vec) and int(c["k_q_star"]) != int(vec[index]):
            problems.append(
                f"video {c['video_id']}: k_q_star={c['k_q_star']} != "
                f"k_vector[{index}]={vec[index]}"
            )
    return problems


def write_selection_json(selected: List[dict], path: Path, plan: str) -> None:
    payload = {
        "selection_type": SELECTION_TYPE,
        "selection_version": SELECTION_VERSION,
        "plan": plan,
        "seed": SEED,
        "n_videos": len(selected),
        "n_queries": sum(len(c["effective_k_vector"]) for c in selected),
        "note": ("Coverage-aware diagnostic subset selected from the "
                 "compressed-only exposure scan. No correctness / fidelity "
                 "metric was read. NOT a random sample of StreamingBench real."),
        "samples": selected,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_audit_csv(selected: List[dict], path: Path) -> None:
    fields = [
        "video_id", "q1_time", "q2_time", "q3_time", "q4_time", "q5_time",
        "k_q1", "k_q2", "k_q3", "k_q4", "k_q5",
        "q_star_id", "q_star_time", "k_q_star",
        "t1", "t2", "t3", "t4", "t5",
        "coverage_bins_contributed", "selection_score", "selection_reason",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for c in selected:
            row = {f"q{i}_time": c["query_times"][i - 1] for i in range(1, 6)}
            row.update({f"k_q{i}": c["effective_k_vector"][i - 1] for i in range(1, 6)})
            row.update({f"t{i}": c["task_types"][i - 1] for i in range(1, 6)})
            row.update({
                "video_id": c["video_id"],
                "q_star_id": c["q_star_id"],
                "q_star_time": c["q_star_time"],
                "k_q_star": c["k_q_star"],
                "coverage_bins_contributed": "|".join(c["coverage_bins_contributed"]),
                "selection_score": c["selection_score"],
                "selection_reason": c["selection_reason"],
            })
            writer.writerow(row)


def build_manifest(
    plan: List[dict],
    plan_name: str,
    exposure_csv,
    base_config: str,
    budget: int,
    bootstrap_seed: int = 42,
    n_boot: int = 2000,
) -> dict:
    """Frozen experiment manifest (written BEFORE any model run)."""

    import hashlib
    from datetime import datetime, timezone

    video_ids = [str(c["video_id"]) for c in plan]
    digest = hashlib.sha256(",".join(video_ids).encode()).hexdigest()[:16]
    return {
        "experiment_id": f"sb_real_{plan_name}_{digest}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked": True,
        "lock_note": (
            "Sample set frozen before any correctness/fidelity observation. "
            "It must NOT be re-selected after inspecting model outcomes."
        ),
        "selection_type": SELECTION_TYPE,
        "selection_version": SELECTION_VERSION,
        "plan": plan_name,
        "plan_file": f"selected_samples_{plan_name}.json",
        "exposure_csv": str(exposure_csv),
        "benchmark": "streamingbench",
        "category": "real_time_visual_understanding",
        "context_mode": "full_prefix",   # [0, query_time]
        "budget": int(budget),
        "selection_seed": SEED,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap": {"unit": "video", "n_boot": int(n_boot), "ci": 0.95},
        "bins": [
            {"name": "B0", "range": "k==0"},
            {"name": "B1", "range": "1<=k<=10"},
            {"name": "B2", "range": "11<=k<=20"},
            {"name": "B3", "range": "21<=k<=30"},
            {"name": "B4", "range": "k>=31"},
        ],
        "qstar_buckets": ["5-10", "11-20", "21-30", "31+"],
        "base_config": base_config,
        "video_ids": video_ids,
        "n_videos": len(video_ids),
        "n_queries": sum(len(c["effective_k_vector"]) for c in plan),
        "analysis": {
            "task_only": "all 5 queries per video, independent [0, query_time] runs",
            "detailed_trajectory": "q_star only, fixed-query diagnostic probe "
                                   "after every effective compaction",
            "metrics": ["Accuracy_compressed", "Accuracy_full", "DeltaAcc",
                        "Preservation", "CorrectToWrongRate"],
            "fidelity": ["RAM", "AORE", "JS", "KL", "compression_ratio"],
        },
    }
