"""Tables, statistics.json and the auto-generated experiment report."""

import csv
import json
from pathlib import Path
from typing import Dict, List

from . import schemas, statistics as st
from .loader import ExperimentData


# --------------------------------------------------------------------- #
# compute everything once
# --------------------------------------------------------------------- #
def compute_all(experiment: ExperimentData) -> dict:
    fid_points, fid_rows = schemas.build_compression_fidelity(experiment)
    pre_post_rows, pre_post_stats = schemas.build_pre_post(experiment)
    rc_points, rc_rows = schemas.build_repeated_compaction(experiment)
    temporal_points, temporal_rows = schemas.build_temporal_ram(experiment)
    age_series, age_rows = schemas.build_age_retention(experiment)
    system_points, system_rows = schemas.build_system_efficiency(experiment)
    task_bins, task_groups, task_rows, task_stats, task_raw = schemas.build_metric_task(experiment)
    cases = schemas.select_cases(experiment)
    return {
        "compression_fidelity": {"points": fid_points, "rows": fid_rows},
        "pre_post": {"rows": pre_post_rows, "stats": pre_post_stats},
        "repeated_compaction": {"points": rc_points, "rows": rc_rows},
        "temporal_ram": {"points": temporal_points, "rows": temporal_rows},
        "age_retention": {"series": age_series, "rows": age_rows},
        "system_efficiency": {"points": system_points, "rows": system_rows},
        "metric_task": {
            "bins": task_bins,
            "groups": task_groups,
            "rows": task_rows,
            "stats": task_stats,
            "raw": task_raw,
        },
        "cases": cases,
    }


# --------------------------------------------------------------------- #
# CSV tables
# --------------------------------------------------------------------- #
def _write_csv(path: Path, rows: List[dict], fieldnames: List[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_tables(experiment_dir: Path, data: dict, experiment: ExperimentData) -> None:
    tables = Path(experiment_dir) / "tables"
    _write_csv(tables / "compression_fidelity.csv", data["compression_fidelity"]["rows"])
    _write_csv(tables / "pre_post_stats.csv", data["pre_post"]["rows"])
    _write_csv(tables / "repeated_compaction_drift.csv", data["repeated_compaction"]["rows"])
    _write_csv(tables / "temporal_ram_curve.csv", data["temporal_ram"]["rows"])
    _write_csv(tables / "age_vs_retention.csv", data["age_retention"]["rows"])
    _write_csv(tables / "system_efficiency.csv", data["system_efficiency"]["rows"])
    _write_csv(tables / "metric_task_relation.csv", data["metric_task"]["rows"])

    # per-sample summary table
    sample_rows = []
    for run in experiment.ordered_runs():
        by_sample = {}
        for event in run.events:
            sid = str(event.get("sample_id"))
            bucket = by_sample.setdefault(sid, {"ram": [], "aore": [], "js": [], "cr": []})
            for key, target in (("ram_mean", "ram"), ("aore_mean", "aore"), ("js", "js"), ("compression_ratio", "cr")):
                if event.get(key) is not None:
                    bucket[target].append(float(event[key]))
        comp_by_sample = {}
        for c in run.compactions:
            comp_by_sample.setdefault(str(c.get("sample_id")), []).append(c)
        for sid, bucket in by_sample.items():
            comps = comp_by_sample.get(sid, [])
            sample_rows.append({
                "experiment": run.name,
                "sample_id": sid,
                "n_events": sum(1 for e in run.events if str(e.get("sample_id")) == sid),
                "compaction_count": len(comps),
                "max_kv_tokens": max([c.get("kv_before") or 0 for c in comps] or [0]),
                "mean_compression_ratio": _mean(bucket["cr"]),
                "max_compression_ratio": max(bucket["cr"]) if bucket["cr"] else None,
                "ram_mean": _mean(bucket["ram"]),
                "aore_mean": _mean(bucket["aore"]),
                "js_mean": _mean(bucket["js"]),
            })
    _write_csv(tables / "sample_summary.csv", sample_rows)


def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------- #
# statistics.json
# --------------------------------------------------------------------- #
def write_statistics(experiment_dir: Path, data: dict, experiment: ExperimentData) -> dict:
    stats = {
        "experiment": experiment.name,
        "runs": {
            run.name: {
                "n_events": len(run.events),
                "n_compactions": len(run.compactions),
                "n_samples": len(run.sample_ids()),
                "effective_compression": run.effective_compression(),
            }
            for run in experiment.ordered_runs()
        },
        "compression_fidelity": data["compression_fidelity"]["points"],
        "pre_post": data["pre_post"]["stats"],
        "repeated_compaction": data["repeated_compaction"]["points"],
        "temporal_ram": data["temporal_ram"]["points"],
        "metric_task": data["metric_task"]["stats"],
        "bootstrap": {"n_boot": 2000, "ci": 0.95, "seed": 42, "unit": "sample"},
        "notes": {
            "temporal_aore_supported": False,
            "temporal_js_supported": False,
            "pre_post_definition": (
                "first HERMES compaction within a sample; PRE = queries before, "
                "POST = queries at/after. Not an immediate causal pre/post."
            ),
        },
    }
    out = Path(experiment_dir) / "statistics.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return stats


# --------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------- #
def write_report(experiment_dir: Path, data: dict, experiment: ExperimentData, stats: dict) -> Path:
    lines: List[str] = []
    runs = experiment.ordered_runs()
    total_events = sum(len(r.events) for r in runs)
    total_compactions = sum(len(r.compactions) for r in runs)
    all_samples = sorted({str(e.get("sample_id")) for r in runs for e in r.events})

    lines.append(f"# KV compression fidelity experiment: `{experiment.name}`\n")
    lines.append("## 1. Scale\n")
    lines.append(
        f"- Benchmark videos (samples): **{len(all_samples)}**\n"
        f"- Query events summed over runs: {total_events}\n"
        f"- Compaction events summed over runs: {total_compactions}\n"
        f"- Experiment runs (budgets): {', '.join(r.name for r in runs)}\n\n"
    )
    lines.append("| run | N samples | N query events | N compactions |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for run in runs:
        lines.append(
            f"| {run.name} | {len(run.sample_ids())} | {len(run.events)} | "
            f"{len(run.compactions)} |\n"
        )
    lines.append(
        "\n> Note: query events are counted per run (each run re-executes every "
        "sample's queries), so early-stop / multi-query samples contribute one "
        "event per query.\n"
    )
    lines.append("## 2. Selection criteria\n")
    meta = (experiment.selected_samples or [])
    meta_info = dict(getattr(experiment, "selection_meta", {}) or {})
    sel_path = Path(experiment_dir) / "selected_samples.json"
    if sel_path.exists() and not meta_info:
        try:
            meta_info = json.loads(sel_path.read_text(encoding="utf-8")).get("meta", {})
        except json.JSONDecodeError:
            meta_info = {}
    lines.append(
        f"- benchmark: {meta_info.get('benchmark')}\n"
        f"- scale: {meta_info.get('scale')}\n"
        f"- min_compactions: {meta_info.get('min_compactions')}\n"
        f"- explicit sample ids: {meta_info.get('explicit_sample_ids')}\n"
        f"- samples selected: {len(meta)}\n"
        f"- sample ids: {', '.join(str(s.get('sample_id')) for s in meta)}\n"
    )

    lines.append("## 3. Effective compression per budget\n")
    lines.append("| experiment | effective_compression | mean compression ratio | N samples | N events |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for point in data["compression_fidelity"]["points"]:
        cr = point.get("x")
        lines.append(
            f"| {point['name']} | {point.get('effective_compression')} | "
            f"{_fmt(cr)} | {point.get('n_samples')} | {point.get('n_events')} |\n"
        )

    def trend(key: str, higher_is_worse: bool) -> str:
        pts = [
            (p["x"], (p.get(key) or {}).get("mean"))
            for p in data["compression_fidelity"]["points"]
            if p.get("effective_compression") and p.get("x") is not None and (p.get(key) or {}).get("mean") is not None
        ]
        pts.sort()
        if len(pts) < 2:
            return "insufficient effective-compression budgets to judge a trend."
        direction = "increase" if pts[-1][1] > pts[0][1] else "decrease"
        expect = "increase" if higher_is_worse else "decrease"
        return (
            f"{key.upper()} tends to {direction} as compression grows "
            f"({_fmt(pts[0][1])} @ CR {_fmt(pts[0][0])} -> {_fmt(pts[-1][1])} @ CR {_fmt(pts[-1][0])}); "
            f"expected direction = {expect}."
        )

    lines.append("## 4-6. Fidelity vs compression\n")
    lines.append(f"- RAM: {trend('ram', higher_is_worse=False)}\n")
    lines.append(f"- AORE: {trend('aore', higher_is_worse=True)}\n")
    lines.append(f"- JS: {trend('js', higher_is_worse=True)}\n")
    kl_points = [
        (p["x"], (p.get("kl") or {}).get("mean"))
        for p in data["compression_fidelity"]["points"]
        if p.get("effective_compression") and (p.get("kl") or {}).get("mean") is not None
    ]
    if len(kl_points) >= 2:
        kl_points.sort()
        lines.append(f"- KL: {_fmt(kl_points[0][1])} -> {_fmt(kl_points[-1][1])} across budgets.\n")

    lines.append("## 7. First-compaction PRE vs POST\n")
    pp = data["pre_post"]
    prepost_supported = len(pp["rows"]) > 0
    lines.append(
        f"- `paired_prepost_supported = {str(prepost_supported).lower()}`\n"
        f"- samples with both PRE and POST queries: **{len(pp['rows'])}**\n"
    )
    if not pp["rows"]:
        lines.append(
            "- No sample provides queries on both sides of its first compaction "
            "in this benchmark. OVO timestamp tasks (EPM/ASI/...) have exactly "
            "one query per sample, and the forward tasks (REC/SSR/CRR) ignore the "
            "requested `realtime` and always stream to the end of the clip. "
            "A genuine PRE/POST therefore requires **extra probe anchors** "
            "(re-running the same video with additional query timestamps), which "
            "this stage does not fabricate.\n"
        )
    for metric in ("kl", "js", "aore", "ram"):
        stat = pp["stats"].get(metric, {})
        delta = stat.get("delta", {})
        wl = stat.get("wilcoxon", {})
        lines.append(
            f"- Δ{metric.upper()}: mean {_fmt(delta.get('mean'))}, "
            f"95% CI [{_fmt(delta.get('ci_low'))}, {_fmt(delta.get('ci_high'))}], "
            f"paired N={delta.get('n')}"
            + (f", Wilcoxon p={_fmt(wl.get('p_value'))}" if not wl.get("skipped") else f" ({wl.get('reason')})")
            + "\n"
        )
    lines.append(
        "\n> This compares queries **before** vs **after the first compaction** in the "
        "same video. It is *not* an immediate causal measurement of a single prune "
        "operation (no probe anchor is inserted around the prune).\n"
    )

    lines.append("## 8. Repeated compaction drift\n")
    for metric in ("ram", "aore", "js"):
        pts = [
            (p["bin_label"], p.get(metric), p.get(f"{metric}_n"))
            for p in data["repeated_compaction"]["points"]
            if p.get(metric) is not None
        ]
        if len(pts) >= 2:
            lines.append(
                f"- {metric.upper()}(n): {pts[0][0]} -> {pts[-1][0]}: "
                f"{_fmt(pts[0][1])} -> {_fmt(pts[-1][1])} "
                f"(N samples {pts[0][2]} -> {pts[-1][2]})\n"
            )
        else:
            lines.append(f"- {metric.upper()}(n): insufficient populated bins.\n")

    lines.append("## 9. Temporal RAM\n")
    tp = data["temporal_ram"]["points"]
    if len(tp) >= 2:
        first, last = tp[0], tp[-1]
        lines.append(
            f"- {first['bin_label']}: {_fmt(first.get('mean'))} (N={first.get('n')}) "
            f"-> {last['bin_label']}: {_fmt(last.get('mean'))} (N={last.get('n')})\n"
        )
        lines.append(
            "- Empty buckets (no full-attention mass) are excluded, never counted as 0.\n"
        )
    else:
        lines.append("- insufficient temporal bins.\n")

    lines.append("## 10. Age vs retention (does HERMES prefer recent KV?)\n")
    for series in data["age_retention"]["series"]:
        pts = [p for p in series["points"] if p.get("rate") is not None]
        if len(pts) >= 2:
            lines.append(
                f"- {series['name']}: retention {_fmt(pts[0]['rate'])} at {pts[0]['bin_label']} "
                f"-> {_fmt(pts[-1]['rate'])} at {pts[-1]['bin_label']} "
                f"(pooled over {series.get('n_events')} evicting compactions; "
                f"no-op compactions excluded)\n"
            )
    if not data["age_retention"]["series"]:
        lines.append(
            "- no age/retention data present. This field is only emitted by runs "
            "produced after the retention-by-age instrumentation; older sidecars "
            "must be re-run.\n"
        )

    lines.append("## 11. Budget comparison (factual, no 'best' claim)\n")
    lines.append("| budget | mean CR | RAM | AORE | JS | mean wall-clock (s) | mean rho |\n")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    sys_by_name = {p["name"]: p for p in data["system_efficiency"]["points"]}
    for point in data["compression_fidelity"]["points"]:
        se = sys_by_name.get(point["name"], {})
        lines.append(
            f"| {point['name']} | {_fmt(point.get('x'))} | "
            f"{_fmt((point.get('ram') or {}).get('mean'))} | "
            f"{_fmt((point.get('aore') or {}).get('mean'))} | "
            f"{_fmt((point.get('js') or {}).get('mean'))} | "
            f"{_fmt(se.get('mean_wallclock'))} | {_fmt(se.get('mean_rho'))} |\n"
        )
    lines.append(
        "\n> This is a factual Pareto comparison. The report does **not** declare "
        "a best configuration.\n"
    )

    lines.append("## 12. Real-time feasibility\n")
    rho_values = [se.get("mean_rho") for se in data["system_efficiency"]["points"] if se.get("mean_rho") is not None]
    if rho_values:
        worst = max(rho_values)
        lines.append(
            f"- worst mean rho_realtime across budgets = {_fmt(worst)} "
            f"({'< 1: one compaction fits before the next trigger' if worst < 1 else '>= 1: not real-time feasible'}).\n"
        )
        lines.append(
            "- rho uses video stream time for the trigger interval; compaction "
            "wall-clock uses cuda.synchronize. This is a legit system measurement "
            "(the eager fidelity backend is not used for timing).\n"
        )
    else:
        lines.append("- no wall-clock / rho data.\n")

    lines.append("## 13. Fidelity vs task correctness\n")
    if data["metric_task"]["rows"]:
        outcomes = [int(r["correct"]) for r in data["metric_task"]["raw"]]
        n_pos = sum(outcomes)
        lines.append(
            f"- task outcomes: {n_pos} correct / {len(outcomes) - n_pos} incorrect "
            f"(N={len(outcomes)} query events)\n"
        )
        for metric, stat in data["metric_task"]["stats"].items():
            rho = stat.get("point_biserial")
            if rho is None:
                lines.append(
                    f"- {metric.upper()}: point-biserial r = n/a "
                    f"(N={stat.get('n')}) -- undefined when the outcome has zero "
                    "variance (all correct or all incorrect)\n"
                )
            else:
                lines.append(
                    f"- {metric.upper()}: point-biserial r = {_fmt(rho)} (N={stat.get('n')})\n"
                )
        lines.append("- Association only; not causation.\n")
    else:
        lines.append(
            "- no task_correct values available in the paired sidecars "
            "(OVO forward samples without a gold answer are excluded).\n"
        )

    lines.extend(_trajectory_section(Path(experiment_dir)))
    lines.extend(_runtime_section(Path(experiment_dir)))

    lines.append("\n## Caveats\n")
    lines.append(
        "- Aggregation unit is the **sample** (events are first averaged per "
        "sample) to avoid pseudo-replication.\n"
        "- Empty bins are excluded and reported with N, never silently zeroed.\n"
        "- Temporal AORE/JS are unsupported (no evidence timestamps).\n"
        "- JS is >= 0 in theory; raw jsonl keeps the original value, while "
        "analysis/plots use `max(js, 0)`. Tiny negative values around 1e-8 are "
        "floating-point artifacts of the Full-Oracle (identical-branch) case "
        "and are treated as zero for visualization only.\n"
    )

    out = Path(experiment_dir) / "experiment_report.md"
    out.write_text("".join(lines), encoding="utf-8")
    return out


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _trajectory_section(experiment_dir: Path) -> List[str]:
    """Answer Q2-Q4: single-video drift with cumulative compaction count."""

    out = ["\n## 14. Single-video trajectory (Query x cumulative compaction count)\n"]
    tables_dir = experiment_dir / "tables"
    csvs = sorted(tables_dir.glob("query_compaction_trajectory_*.csv"))
    if not csvs:
        out.append(
            "- no trajectory table yet. Run `trajectory-select` then "
            "`trajectory --sample_id <id> --budget <b>`.\n"
        )
        return out
    for path in csvs:
        rows = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        if not rows:
            continue
        sample = rows[0].get("sample_id")
        out.append(f"\n### sample {sample} (`{path.name}`)\n")
        out.append("| query_id | query_time | compaction_count | RAM | AORE | JS | KL | CR |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            out.append(
                f"| {row.get('query_id')} | {_fmt(_as_float(row.get('query_time')))} | "
                f"{row.get('compaction_count_at_query')} | "
                f"{_fmt(_as_float(row.get('RAM')))} | {_fmt(_as_float(row.get('AORE')))} | "
                f"{row.get('JS_display')} | {_fmt(_as_float(row.get('KL')))} | "
                f"{_fmt(_as_float(row.get('CR')))} |\n"
            )
        ram = [_as_float(r.get("RAM")) for r in rows]
        aore = [_as_float(r.get("AORE")) for r in rows]
        cc = [_as_float(r.get("compaction_count_at_query")) for r in rows]
        pairs = [(c, r) for c, r in zip(cc, ram) if c is not None and r is not None]
        aore_pairs = [(c, a) for c, a in zip(cc, aore) if c is not None and a is not None]
        if len(pairs) >= 2:
            out.append(
                f"\n- RAM: {_fmt(pairs[0][1])} @ cc={int(pairs[0][0])} -> "
                f"{_fmt(pairs[-1][1])} @ cc={int(pairs[-1][0])} "
                f"({'declining' if pairs[-1][1] < pairs[0][1] else 'not declining'}).\n"
            )
        if len(aore_pairs) >= 2:
            out.append(
                f"- AORE: {_fmt(aore_pairs[0][1])} @ cc={int(aore_pairs[0][0])} -> "
                f"{_fmt(aore_pairs[-1][1])} @ cc={int(aore_pairs[-1][0])} "
                f"({'rising' if aore_pairs[-1][1] > aore_pairs[0][1] else 'not rising'}).\n"
            )
        # monotonicity (approximately)
        if len(pairs) >= 3:
            inversions = sum(
                1 for i in range(1, len(pairs)) if pairs[i][1] > pairs[i - 1][1] + 1e-6
            )
            out.append(
                f"- monotonicity: {inversions} increase(s) in RAM along increasing "
                f"compaction count -> "
                f"{'near-monotone decline' if inversions <= 1 else 'noticeably non-monotone'}.\n"
            )
        out.append(
            "\n> Interpretation: cumulative operational fidelity. Each point is a "
            "real query; the query content and the video content also change between "
            "points, so this is **not** a per-compaction causal effect.\n"
        )
    return out


def _runtime_section(experiment_dir: Path) -> List[str]:
    out = ["\n## 15. Runtime breakdown\n"]
    summaries = sorted((experiment_dir / "raw").glob("*/runtime_summary.json"))
    tables = sorted((experiment_dir / "raw").glob("*/tables/runtime_breakdown.csv"))
    if not summaries:
        out.append(
            "- not profiled. Re-run with `--profile_runtime_breakdown` to populate "
            "this section.\n"
        )
        return out
    for summary_path in summaries:
        run_name = summary_path.parent.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        out.append(f"\n### {run_name}\n")
        out.append(
            f"- T_measured_model_compute_total = "
            f"{_fmt(summary.get('measured_model_compute_total_sec'))} s\n"
            f"- T_HERMES_total = {_fmt(summary.get('hermes_total_sec'))} s "
            f"(prune = {_fmt(summary.get('hermes_prune_sec'))} s, importance = "
            f"{_fmt(summary.get('hermes_importance_sec'))} s, selection = "
            f"{_fmt(summary.get('hermes_selection_sec'))} s)\n"
            f"- hermes_compute_fraction = {_fmt(summary.get('hermes_compute_fraction'))}\n"
            f"- rho_prune_realtime = {_fmt(summary.get('rho_prune_realtime'))}, "
            f"rho_hermes_realtime = {_fmt(summary.get('rho_hermes_realtime'))}\n"
        )
        out.append(
            "- NOTE: when `--profile_runtime_breakdown` is on, every measured call "
            "is surrounded by `cuda.synchronize`, which inflates wall-clock. "
            "`rho_prune_realtime` / `rho_hermes_realtime` from this mode are "
            "**upper bounds**, not deployment numbers; the deployment rho "
            "(`compression_vs_rho`) is measured without per-call syncs.\n"
        )
    if tables:
        out.append(
            "\nPer-component breakdown is in `raw/*/tables/runtime_breakdown.csv`. "
            "`fraction_of_model_compute` uses the non-double-counted total "
            "(vision_encoder + streaming_prefill + hermes_total + query_prefill + "
            "decode); HERMES sub-components are never added again. "
            "`stream_time_overhead_ratio` (per video duration) is a separate, "
            "different quantity.\n"
        )
    return out


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
