"""Accuracy vs effective-compaction-count (cross-query task analysis).

Every number here is a *task* outcome of real benchmark queries, not a
fidelity metric. The unit of inference is the sample (cluster bootstrap), and
the Full-KV pass is the baseline that separates "the question was already too
hard" from "compression made it wrong".
"""

import math
from typing import Dict, List, Optional

from . import statistics as st


def _norm(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _usable(row: dict) -> bool:
    if row.get("task_missing"):
        return False
    return _norm(row.get("task_correct_compressed")) is not None


def sample_key(row: dict) -> str:
    """Cluster unit for the bootstrap.

    StreamingBench emits one benchmark sample per QUESTION, so ``sample_id`` is
    a question id there; the cluster (and the "N_video" column) must use the
    SOURCE VIDEO id instead, otherwise every query would be its own cluster.
    """

    return str(row.get("video_id") or row.get("sample_id") or "")


def accuracy_by_compaction_count(
    rows: List[dict],
    k_field: str = "effective_compaction_count_at_query",
    n_boot: int = 2000,
    seed: int = 42,
    bin_edges: Optional[List[int]] = None,
) -> List[dict]:
    """Exact-count table (``bin_edges=None``) or a binned secondary table."""

    usable = [r for r in rows if _usable(r) and r.get(k_field) is not None]
    if bin_edges is None:
        keys = sorted({int(r[k_field]) for r in usable})
        key_of = lambda r: int(r[k_field])  # noqa: E731
        label_of = lambda k: str(k)  # noqa: E731
    else:
        edges = sorted(bin_edges)
        keys = list(range(len(edges) + 1))

        def key_of(row):
            value = int(row[k_field])
            for i, edge in enumerate(edges):
                if value <= edge:
                    return i
            return len(edges)

        def label_of(index):
            lo = 0 if index == 0 else edges[index - 1] + 1
            hi = None if index >= len(edges) else edges[index]
            return f"{lo}" if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")

    out = []
    for key in keys:
        group = [r for r in usable if key_of(r) == key]
        if not group:
            continue
        row = {"effective_compaction_count": key, "bin_label": label_of(key)}
        by_sample: Dict[str, List[dict]] = {}
        for r in group:
            by_sample.setdefault(sample_key(r), []).append(r)
        row["num_queries"] = len(group)
        row["num_samples"] = len(by_sample)

        comp = [int(_norm(r["task_correct_compressed"])) for r in group]
        row["num_correct_compressed"] = sum(comp)
        row["accuracy_compressed"] = sum(comp) / len(comp)

        full_rows = [r for r in group if _norm(r.get("task_correct_full")) is not None]
        if full_rows:
            full = [int(_norm(r["task_correct_full"])) for r in full_rows]
            row["num_correct_full"] = sum(full)
            row["accuracy_full"] = sum(full) / len(full)
            row["num_full_correct"] = len(full)
            base = [
                r for r in full_rows if _norm(r.get("task_correct_full")) is True
            ]
            row["num_full_correct_with_compression"] = len(base)
            preserved = sum(
                1 for r in base if _norm(r.get("task_correct_compressed")) is True
            )
            row["preservation_accuracy"] = (
                preserved / len(base) if base else None
            )
            row["compression_failure_rate"] = (
                1.0 - row["preservation_accuracy"]
                if row["preservation_accuracy"] is not None
                else None
            )
        else:
            row["num_correct_full"] = None
            row["accuracy_full"] = None
            row["num_full_correct"] = 0
            row["preservation_accuracy"] = None
            row["compression_failure_rate"] = None

        transitions = {
            "correct_to_correct": 0, "correct_to_wrong": 0,
            "wrong_to_correct": 0, "wrong_to_wrong": 0,
        }
        for r in group:
            name = r.get("task_transition") or "missing"
            if name in transitions:
                transitions[name] += 1
        for name, value in transitions.items():
            row[f"num_{name}"] = value
        row["protocol_violation_count"] = sum(
            1 for r in group
            if (r.get("task_status_compressed") or r.get("task_error_type_compressed"))
        )
        row["missing_count"] = sum(1 for r in group if r.get("task_missing"))
        # alias required by the spec
        row["correct_to_wrong_rate"] = (
            transitions["correct_to_wrong"] / len(group) if group else None
        )

        # cluster bootstrap (unit = sample)
        comp_clusters = {
            sid: [int(_norm(r["task_correct_compressed"])) for r in rs]
            for sid, rs in by_sample.items()
        }
        ci = st.cluster_bootstrap_ci(list(comp_clusters.values()),
                                     n_boot=n_boot, seed=seed)
        row["accuracy_compressed_ci_low"] = ci["ci_low"]
        row["accuracy_compressed_ci_high"] = ci["ci_high"]
        if full_rows:
            full_clusters: Dict[str, List[int]] = {}
            for r in full_rows:
                full_clusters.setdefault(sample_key(r), []).append(
                    int(_norm(r["task_correct_full"]))
                )
            ci_full = st.cluster_bootstrap_ci(list(full_clusters.values()),
                                              n_boot=n_boot, seed=seed)
            row["accuracy_full_ci_low"] = ci_full["ci_low"]
            row["accuracy_full_ci_high"] = ci_full["ci_high"]
            base_clusters: Dict[str, List[int]] = {}
            for r in full_rows:
                if _norm(r.get("task_correct_full")) is not True:
                    continue
                base_clusters.setdefault(sample_key(r), []).append(
                    int(_norm(r.get("task_correct_compressed")) is True)
                )
            if base_clusters:
                ci_pres = st.cluster_bootstrap_ci(
                    list(base_clusters.values()), n_boot=n_boot, seed=seed
                )
                row["preservation_ci_low"] = ci_pres["ci_low"]
                row["preservation_ci_high"] = ci_pres["ci_high"]
            else:
                row["preservation_ci_low"] = None
                row["preservation_ci_high"] = None
        row["exploratory_low_n"] = bool(row["num_samples"] < 5)
        out.append(row)
    return out
