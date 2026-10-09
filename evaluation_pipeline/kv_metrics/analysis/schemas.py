"""Derived tables for each experiment.

Everything here is read from the existing sidecar records; metric math
(RAM/AORE/JS/KL) is never recomputed. Aggregation always follows
``events -> sample-level statistic -> cross-sample statistic``.
"""

from typing import Dict, List, Optional, Tuple

from . import statistics as st
from .loader import ExperimentData, RunData, compactions_by_sample, events_by_sample


COMPACTION_BIN_EDGES = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, None)]


def js_display(value):
    """JS is >= 0 in theory; clamp tiny negative floating-point artifacts.

    The raw jsonl keeps the original value; only analysis/plotting use this.
    """

    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


def bin_label_for_index(index: Optional[int]) -> Optional[str]:
    if index is None:
        return None
    for lo, hi in COMPACTION_BIN_EDGES:
        if hi is None:
            if index >= lo:
                return f"{lo}+"
        elif lo <= index <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return None


def _sample_metric(run: RunData, metric: str) -> Dict[str, float]:
    return st.sample_level_means(run.events, metric)


def _compression_samples(run: RunData) -> List[str]:
    return sorted({str(c.get("sample_id")) for c in run.compactions})


# --------------------------------------------------------------------- #
# Experiment 1
# --------------------------------------------------------------------- #
def build_compression_fidelity(experiment: ExperimentData):
    points = []
    csv_rows = []
    for run in experiment.ordered_runs():
        comp_samples = set(_compression_samples(run))
        if not comp_samples:
            points.append({
                "name": run.name,
                "x": None,
                "effective_compression": False,
                "n_samples": 0,
                "n_events": 0,
            })
            continue
        events = [e for e in run.events if str(e.get("sample_id")) in comp_samples]
        sample_ratio = {
            sid: r
            for sid, r in _sample_metric(run, "compression_ratio").items()
            if sid in comp_samples
        }
        x_values = [v for v in sample_ratio.values() if v is not None]
        point = {
            "name": run.name,
            "x": (sum(x_values) / len(x_values)) if x_values else None,
            "effective_compression": True,
            "n_samples": len(comp_samples),
            "n_events": len(events),
        }
        for metric in ("ram_mean", "aore_mean", "js", "kl"):
            values = st.sample_level_means(events, metric)
            if metric == "js":
                values = {k: js_display(v) for k, v in values.items()}
            stat = st.bootstrap_ci(list(values.values()))
            point[metric.replace("_mean", "")] = stat
        points.append(point)

        for sid in sorted(comp_samples):
            row = {"experiment": run.name, "sample_id": sid}
            for metric in ("compression_ratio", "ram_mean", "aore_mean", "js", "kl"):
                if metric == "compression_ratio":
                    row[metric] = sample_ratio.get(sid)
                    continue
                value = st.sample_level_means(
                    [e for e in events if str(e.get("sample_id")) == sid], metric
                ).get(sid)
                row[metric] = js_display(value) if metric == "js" else value
            csv_rows.append(row)
    return points, csv_rows


# --------------------------------------------------------------------- #
# Experiment 2 : first-compaction PRE vs POST
# --------------------------------------------------------------------- #
def build_pre_post(experiment: ExperimentData):
    rows = []
    for run in experiment.ordered_runs():
        if not run.compactions:
            continue
        cbs = compactions_by_sample(run.compactions)
        ebs = events_by_sample(run.events)
        for sid, events in ebs.items():
            comps = cbs.get(sid)
            if not comps:
                continue
            first_t = comps[0].get("stream_time")
            if first_t is None:
                continue
            pre = [e for e in events if (e.get("query_time") or 0) < first_t]
            post = [e for e in events if (e.get("query_time") or 0) >= first_t]
            if not pre or not post:
                continue
            row = {
                "experiment": run.name,
                "sample_id": sid,
                "first_compaction_stream_time": first_t,
                "n_pre": len(pre),
                "n_post": len(post),
                "query_times": [e.get("query_time") for e in events],
            }
            for metric, key in (
                ("kl", "kl"),
                ("js", "js"),
                ("aore", "aore_mean"),
                ("ram", "ram_mean"),
            ):
                pre_vals = [e.get(key) for e in pre if e.get(key) is not None]
                post_vals = [e.get(key) for e in post if e.get(key) is not None]
                row[f"{metric}_pre"] = (sum(pre_vals) / len(pre_vals)) if pre_vals else None
                row[f"{metric}_post"] = (sum(post_vals) / len(post_vals)) if post_vals else None
            rows.append(row)

    stats = {}
    for metric in ("kl", "js", "aore", "ram"):
        deltas = [
            r[f"{metric}_post"] - r[f"{metric}_pre"]
            for r in rows
            if r.get(f"{metric}_pre") is not None and r.get(f"{metric}_post") is not None
        ]
        stats[metric] = {
            "delta": st.paired_bootstrap_ci(deltas),
            "wilcoxon": st.wilcoxon_signed_rank(
                [r[f"{metric}_pre"] for r in rows],
                [r[f"{metric}_post"] for r in rows],
            ),
        }
    return rows, stats


# --------------------------------------------------------------------- #
# Experiment 3 : repeated compaction drift
# --------------------------------------------------------------------- #
def build_repeated_compaction(experiment: ExperimentData):
    points: Dict[str, dict] = {}
    for run in experiment.ordered_runs():
        for event in run.events:
            label = bin_label_for_index(event.get("compaction_index"))
            if label is None:
                continue
            point = points.setdefault(label, {"bin_label": label})
            point.setdefault("_events", []).append(event)

    ordered = []
    csv_rows = []
    for label in _bin_order(points.keys()):
        entry = points[label]
        events = entry["_events"]
        point = {"bin_label": label, "n_events": len(events)}
        for metric, key in (
            ("ram", "ram_mean"),
            ("aore", "aore_mean"),
            ("js", "js"),
            ("kl", "kl"),
        ):
            sample_means = st.sample_level_means(events, key)
            if metric == "js":
                sample_means = {k: js_display(v) for k, v in sample_means.items()}
            stat = st.bootstrap_ci(list(sample_means.values()))
            point[metric] = stat["mean"]
            point[f"{metric}_ci_low"] = stat["ci_low"]
            point[f"{metric}_ci_high"] = stat["ci_high"]
            point[f"{metric}_n"] = stat["n"]
            point[f"{metric}_n_events"] = len(events)
            for sid, value in sample_means.items():
                csv_rows.append({
                    "bin_label": label, "sample_id": sid, "metric": metric, "value": value,
                })
        ordered.append(point)
    return ordered, csv_rows


def _bin_order(labels) -> List[str]:
    def key(label):
        lo = int(str(label).split("-")[0].rstrip("+"))
        return lo
    return sorted(labels, key=key)


# --------------------------------------------------------------------- #
# Experiment 4 : temporal RAM
# --------------------------------------------------------------------- #
def build_temporal_ram(experiment: ExperimentData):
    # collect per-bin sample-level means (unit = sample)
    per_bin: Dict[str, Dict[str, float]] = {}
    bin_order: List[str] = []
    for run in experiment.ordered_runs():
        for event in run.events:
            temporal = event.get("temporal_ram")
            if not isinstance(temporal, dict):
                continue
            sid = str(event.get("sample_id"))
            for label, value in temporal.items():
                if value is None:
                    continue
                if label not in bin_order:
                    bin_order.append(label)
                per_bin.setdefault(label, {}).setdefault(sid, []).append(float(value))
    bin_order = sorted(per_bin.keys(), key=_age_bin_key)

    points = []
    csv_rows = []
    for label in bin_order:
        sample_series = per_bin[label]
        sample_means = {sid: sum(v) / len(v) for sid, v in sample_series.items()}
        stat = st.bootstrap_ci(list(sample_means.values()))
        points.append({"bin_label": label, **stat})
        for sid, value in sample_means.items():
            csv_rows.append({"bin_label": label, "sample_id": sid, "temporal_ram": value})
    return points, csv_rows


def _age_bin_key(label: str) -> float:
    return float(str(label).split("_")[0].replace("inf", "1e9"))


# --------------------------------------------------------------------- #
# Experiment 5 : age vs retention
# --------------------------------------------------------------------- #
def build_age_retention(experiment: ExperimentData):
    """Retention probability by KV age.

    Only compaction events that actually **evicted** something
    (kept < total) carry a retention signal, so no-op compactions are
    excluded. The number of contributing events is reported per experiment.
    """

    series = []
    csv_rows = []
    for run in experiment.ordered_runs():
        per_bin: Dict[str, dict] = {}
        n_events = 0
        for record in run.compactions:
            retention = record.get("retention_by_age")
            if not isinstance(retention, dict):
                continue
            ev_kept = sum(int(e.get("kept") or 0) for e in retention.values())
            ev_total = sum(int(e.get("total") or 0) for e in retention.values())
            if ev_total == 0 or ev_kept >= ev_total:
                # compaction kept everything -> no eviction signal
                continue
            n_events += 1
            for label, entry in retention.items():
                b = per_bin.setdefault(label, {"kept": 0, "total": 0})
                b["kept"] += int(entry.get("kept") or 0)
                b["total"] += int(entry.get("total") or 0)
        if not per_bin:
            continue
        ordered = sorted(per_bin.items(), key=lambda kv: _age_bin_key(kv[0]))
        points = [
            {
                "bin_label": label,
                "kept": v["kept"],
                "total": v["total"],
                "rate": (v["kept"] / v["total"]) if v["total"] else None,
            }
            for label, v in ordered
        ]
        series.append({"name": run.name, "points": points, "n_events": n_events})
        for p in points:
            csv_rows.append({"experiment": run.name, "n_evicting_compactions": n_events, **p})
    return series, csv_rows


# --------------------------------------------------------------------- #
# Experiment 6 : system efficiency
# --------------------------------------------------------------------- #
def build_system_efficiency(experiment: ExperimentData):
    points = []
    csv_rows = []
    for run in experiment.ordered_runs():
        comp_samples = set(_compression_samples(run))
        if not comp_samples:
            continue
        ratios = st.sample_level_means(
            [e for e in run.events if str(e.get("sample_id")) in comp_samples],
            "compression_ratio",
        )
        walls = [c.get("wall_clock_sec") for c in run.compactions if c.get("wall_clock_sec") is not None]
        rhos = [c.get("rho_realtime") for c in run.compactions if c.get("rho_realtime") is not None]
        # overhead per sample: sum of compaction wall-clock
        overhead_by_sample: Dict[str, float] = {}
        for c in run.compactions:
            sid = str(c.get("sample_id"))
            overhead_by_sample[sid] = overhead_by_sample.get(sid, 0.0) + float(c.get("wall_clock_sec") or 0.0)
        duration_by_sample: Dict[str, float] = {}
        for e in run.events:
            sid = str(e.get("sample_id"))
            qt = e.get("query_time")
            if qt is not None:
                duration_by_sample[sid] = max(duration_by_sample.get(sid, 0.0), float(qt))
        point = {
            "name": run.name,
            "compression_ratio": (sum(ratios.values()) / len(ratios)) if ratios else None,
            "mean_wallclock": (sum(walls) / len(walls)) if walls else None,
            "max_wallclock": max(walls) if walls else None,
            "mean_rho": (sum(rhos) / len(rhos)) if rhos else None,
            "total_overhead": sum(overhead_by_sample.values()) if overhead_by_sample else None,
            "n_compactions": len(run.compactions),
        }
        points.append(point)
        for sid, overhead in overhead_by_sample.items():
            duration = duration_by_sample.get(sid)
            csv_rows.append({
                "experiment": run.name,
                "sample_id": sid,
                "total_compaction_overhead_sec": overhead,
                "stream_duration_sec": duration,
                "overhead_ratio": (overhead / duration) if duration else None,
            })
    return points, csv_rows


# --------------------------------------------------------------------- #
# Experiment 7 : fidelity vs task accuracy
# --------------------------------------------------------------------- #
def task_map(run: RunData) -> Dict[str, dict]:
    block = (run.summary or {}).get("task") or {}
    return {
        row["event_id"]: row
        for row in block.get("per_event", [])
        if row.get("event_id")
    }


def build_metric_task(experiment: ExperimentData):
    rows = []
    for run in experiment.ordered_runs():
        tasks = task_map(run)
        for event in run.events:
            correct = event.get("task_correct")
            if correct is None:
                task = tasks.get(event.get("event_id"))
                correct = task.get("task_correct") if task else None
            if correct is None:
                continue
            rows.append({
                "experiment": run.name,
                "event_id": event.get("event_id"),
                "sample_id": event.get("sample_id"),
                "correct": int(bool(correct)),
                "ram": event.get("ram_mean"),
                "aore": event.get("aore_mean"),
                "js": event.get("js"),
                "kl": event.get("kl"),
            })

    bins: Dict[str, List[dict]] = {}
    groups: Dict[str, dict] = {}
    csv_rows = []
    for metric in ("ram", "aore", "js", "kl"):
        values = [(r[metric], r["correct"]) for r in rows if r.get(metric) is not None]
        if metric == "js":
            values = [(js_display(v), c) for v, c in values]
        if not values:
            continue
        values.sort(key=lambda x: x[0])
        n = len(values)
        q = max(1, n // 5)
        metric_bins = []
        for i in range(5):
            lo = i * q
            hi = n if i == 4 else min(n, (i + 1) * q)
            chunk = values[lo:hi]
            if not chunk:
                continue
            acc = sum(c for _, c in chunk) / len(chunk)
            metric_bins.append({
                "bin_label": f"Q{i + 1}",
                "accuracy": acc,
                "n": len(chunk),
                "metric_range": [chunk[0][0], chunk[-1][0]],
            })
            csv_rows.append({
                "metric": metric, "bin": f"Q{i + 1}", "accuracy": acc, "n": len(chunk),
            })
        bins[metric] = metric_bins
        groups[metric] = {
            "correct": [v for v, c in values if c == 1],
            "incorrect": [v for v, c in values if c == 0],
        }

    stats = {
        metric: {
            "point_biserial": st.point_biserial(
                [r["correct"] for r in rows if r.get(metric) is not None],
                [r[metric] for r in rows if r.get(metric) is not None],
            ),
            "n": sum(1 for r in rows if r.get(metric) is not None),
        }
        for metric in ("ram", "aore", "js", "kl")
    }
    return bins, groups, csv_rows, stats, rows


# --------------------------------------------------------------------- #
# Experiment 8 : case selection
# --------------------------------------------------------------------- #
def select_cases(experiment: ExperimentData, limit: int = 3):
    candidates = []
    for run in experiment.ordered_runs():
        comp_by_sample = compactions_by_sample(run.compactions)
        events_by_sid = events_by_sample(run.events)
        for sid, events in events_by_sid.items():
            n_comp = len(comp_by_sample.get(sid, []))
            if n_comp == 0:
                continue
            ram = st.sample_level_means(events, "ram_mean").get(sid)
            aore = st.sample_level_means(events, "aore_mean").get(sid)
            ratio = st.sample_level_means(events, "compression_ratio").get(sid)
            candidates.append({
                "experiment": run.name, "sample_id": sid,
                "n_compactions": n_comp, "ram": ram, "aore": aore,
                "compression_ratio": ratio,
            })
    if not candidates:
        return []
    # A: high compression & good fidelity ; B: high compression & poor fidelity
    # C: most compactions
    high_ratio = sorted(candidates, key=lambda c: (c.get("compression_ratio") or 0), reverse=True)
    good = [c for c in high_ratio if c.get("aore") is not None]
    cases = []
    if good:
        cases.append({**min(good, key=lambda c: c["aore"]), "kind": "A_high_compression_good_fidelity"})
        cases.append({**max(good, key=lambda c: c["aore"]), "kind": "B_high_compression_poor_fidelity"})
    most = max(candidates, key=lambda c: c["n_compactions"])
    cases.append({**most, "kind": "C_most_repeated_compaction"})
    # de-duplicate by (experiment, sample)
    seen = set()
    unique = []
    for case in cases:
        key = (case["experiment"], case["sample_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)
    return unique[:limit]
