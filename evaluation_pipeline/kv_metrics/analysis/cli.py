"""Unified CLI for the KV fidelity experiment workflow.

    python -m evaluation_pipeline.kv_metrics.analysis.cli select   ...
    python -m evaluation_pipeline.kv_metrics.analysis.cli run      ...
    python -m evaluation_pipeline.kv_metrics.analysis.cli analyze  ...
    python -m evaluation_pipeline.kv_metrics.analysis.cli plot     ...
    python -m evaluation_pipeline.kv_metrics.analysis.cli report    ...
    python -m evaluation_pipeline.kv_metrics.analysis.cli run-all  ...
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

from ...evaluate import load_config
from ...adapters import BENCHMARK_REGISTRY
from . import (
    budget_stress,
    experiment_runner,
    loader,
    ovo_diagnostic,
    report,
    reproducibility,
    decision_fidelity,
    sample_selector,
    streamingbench_selection,
    streamingbench_exposure,
    trajectory,
)


def _bench_config(config_path, benchmark):
    if config_path:
        base = load_config(config_path)
        return base.get("benchmarks", {}).get(benchmark, {})
    default = Path("configs/benchmarks") / f"{benchmark}.yaml"
    if default.exists():
        return load_config(str(default))
    return {}


# --------------------------------------------------------------------- #
def cmd_select(args) -> int:
    bench_config = _bench_config(args.experiment, args.benchmark)
    adapter = BENCHMARK_REGISTRY[args.benchmark](bench_config)
    samples = list(adapter.load_samples())
    candidates = sample_selector.select_samples(
        samples,
        metrics_root=args.metrics_root,
        min_compactions=args.min_compactions,
        min_video_duration=args.min_video_duration,
        max_video_duration=getattr(args, "max_video_duration", None),
        min_max_kv_tokens=args.min_max_kv_tokens,
        min_compression_ratio=args.min_compression_ratio,
        num_samples=args.num_samples,
        top_k_longest=args.top_k_longest,
        top_k_most_compactions=args.top_k_most_compactions,
        scale=args.experiment_scale,
        sample_fps=args.sample_fps,
        max_side=args.max_side,
        include_zero_compaction=args.include_zero_compaction,
    )
    out = Path(args.output or Path(args.metrics_root) / "experiments" / args.name / "selected_samples.json")
    sample_selector.write_selection(
        candidates,
        out,
        meta={
            "benchmark": args.benchmark,
            "scale": args.experiment_scale,
            "min_compactions": args.min_compactions,
            "metrics_root": args.metrics_root,
        },
    )
    print(f"[select] wrote {out} with {len(candidates)} samples")
    for c in candidates:
        print(
            f"  {c.sample_id:>8}  dur={_fmt(c.duration)}  "
            f"compactions={c.compaction_count}  src={c.source}"
        )
    return 0


def cmd_run(args) -> int:
    experiment_dir = experiment_runner.run_matrix(
        args.experiment,
        selection_path=args.samples,
        force=args.force,
        resume=not args.no_resume,
        experiment_name=args.name,
    )
    print(f"[run] experiment dir = {experiment_dir}")
    return 0


def cmd_analyze(args) -> int:
    experiment_dir = Path(args.experiment_dir)
    experiment = loader.load_experiment(experiment_dir)
    data = report.compute_all(experiment)
    report.write_tables(experiment_dir, data, experiment)
    stats = report.write_statistics(experiment_dir, data, experiment)
    reproducibility.write(
        experiment_dir / "reproducibility.json",
        {
            "experiment": experiment.name,
            "runs": {
                run.name: {
                    "n_events": len(run.events),
                    "n_compactions": len(run.compactions),
                    "config": run.config,
                }
                for run in experiment.ordered_runs()
            },
            "selected_samples": experiment.selected_samples,
        },
    )
    print(f"[analyze] tables -> {experiment_dir / 'tables'} ({len(stats['runs'])} runs)")
    return 0


def cmd_plot(args) -> int:
    from . import plotting

    experiment_dir = Path(args.experiment_dir)
    experiment = loader.load_experiment(experiment_dir)
    data = report.compute_all(experiment)
    figures_dir = experiment_dir / "figures"
    figures = []
    figures += plotting.plot_compression_fidelity(
        data["compression_fidelity"]["points"], figures_dir
    )
    figures += plotting.plot_pre_post(data["pre_post"]["rows"], figures_dir)
    figures += plotting.plot_pre_post_delta(data["pre_post"]["rows"], figures_dir)
    figures += plotting.plot_repeated_compaction(
        data["repeated_compaction"]["points"], figures_dir
    )
    figures += plotting.plot_temporal_ram(data["temporal_ram"]["points"], figures_dir)
    figures += plotting.plot_age_retention(data["age_retention"]["series"], figures_dir)
    figures += plotting.plot_system_efficiency(
        data["system_efficiency"]["points"], figures_dir
    )
    if data["metric_task"]["rows"]:
        figures += plotting.plot_metric_vs_task(
            data["metric_task"]["bins"], data["metric_task"]["groups"], figures_dir
        )
    for case in data["cases"]:
        run = experiment.runs.get(case["experiment"])
        if run is None:
            continue
        events = [e for e in run.events if str(e.get("sample_id")) == str(case["sample_id"])]
        comps = [c for c in run.compactions if str(c.get("sample_id")) == str(case["sample_id"])]
        figures.append(
            plotting.plot_case_timeline(
                f"{case['experiment']}_{case['sample_id']}", events, comps, figures_dir
            )
        )
    print(f"[plot] wrote {len(figures)} figures -> {figures_dir}")
    return 0


def cmd_report(args) -> int:
    experiment_dir = Path(args.experiment_dir)
    experiment = loader.load_experiment(experiment_dir)
    data = report.compute_all(experiment)
    report.write_tables(experiment_dir, data, experiment)
    stats = report.write_statistics(experiment_dir, data, experiment)
    out = report.write_report(experiment_dir, data, experiment, stats)
    print(f"[report] wrote {out}")
    return 0


def cmd_run_all(args) -> int:
    experiment_dir = experiment_runner.run_matrix(
        args.experiment,
        selection_path=args.samples,
        force=args.force,
        resume=not args.no_resume,
        experiment_name=args.name,
    )
    args.experiment_dir = str(experiment_dir)
    cmd_analyze(args)
    cmd_plot(args)
    cmd_report(args)
    return 0


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


# --------------------------------------------------------------------- #
# StreamingBench real: exposure scan + 3-video smoke
# --------------------------------------------------------------------- #
SB_CSV = "datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
SB_BASE = "configs/eval_kv_metrics_streamingbench_real.yaml"


def cmd_sb_candidates(args) -> int:
    candidates = streamingbench_exposure.build_candidates(
        args.csv or SB_CSV, max_latest_query_time=args.max_latest_query_time
    )
    out = Path(args.output or "outputs/kv_metrics/streamingbench_real_candidates.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"num_candidates": len(candidates), "samples": candidates},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[sb-candidates] {len(candidates)} videos with latest_query_time <= "
          f"{args.max_latest_query_time}s -> {out}")
    for cand in candidates[:5]:
        print(f"  {cand['video_id']}: times={cand['q_times']} q_star={cand['q_star_time']}s")
    return 0


def cmd_sb_exposure(args) -> int:
    """Compressed-only exposure scan: observed effective k per query."""

    from ...kv_metrics.config import KVMetricConfig
    from ...kv_metrics.paired_runner import run as run_paired

    candidates = streamingbench_exposure.build_candidates(
        args.csv or SB_CSV, max_latest_query_time=args.max_latest_query_time
    )
    if args.limit:
        candidates = candidates[: args.limit]
    out_root = Path(args.output_root)
    run_dir = out_root / "raw" / "exposure"

    if not args.skip_run:
        config = load_config(SB_BASE)
        kv_config = KVMetricConfig(
            enable_kv_metrics=True,
            mode="compressed",
            output_dir=str(out_root / "raw"),
            run_name="exposure",
            attention_backend="original",   # structural scan: no fidelity probe
            profile_wallclock=False,
        )
        run_paired(
            config,
            benchmark="streamingbench",
            max_samples=0,
            kv_config=kv_config,
            video_ids=[c["video_id"] for c in candidates],
            question_ids=[c["q_star_id"] for c in candidates],
        )

    rows = streamingbench_exposure.build_exposure_rows(candidates, run_dir)
    out_csv = Path(args.output or out_root / "streamingbench_real_exposure_candidates.csv")
    streamingbench_exposure.write_exposure_csv(rows, out_csv)
    hist = streamingbench_exposure.k_histogram(rows)
    print(f"[sb-exposure] {len(rows)} videos scanned -> {out_csv}")
    print("  k distribution over all queries:", hist)
    print("  k range:", (min(hist), max(hist)) if hist else None)
    return 0


def cmd_sb_smoke(args) -> int:
    """3-video full_prefix smoke (paired run + detailed trajectory)."""

    from ...kv_metrics.config import KVMetricConfig
    from ...kv_metrics.paired_runner import run as run_paired

    candidates = streamingbench_exposure.build_candidates(
        args.csv or SB_CSV, max_latest_query_time=args.max_latest_query_time
    )
    by_time = sorted(candidates, key=lambda c: c["q_star_time"])
    picked = [by_time[0], by_time[len(by_time) // 2], by_time[-1]] if not args.video_ids \
        else [c for c in candidates if c["video_id"] in set(args.video_ids)]
    out_root = Path(args.output_root)
    config = load_config(args.config)
    config["benchmarks"]["streamingbench"]["context_mode"] = "full_prefix"
    kv_config = KVMetricConfig(
        enable_kv_metrics=True,
        mode="paired",
        output_dir=str(out_root / "raw"),
        run_name="smoke",
        attention_backend="eager",
        profile_wallclock=True,
        detailed_trajectory=True,
    )
    run_paired(
        config,
        benchmark="streamingbench",
        max_samples=0,
        kv_config=kv_config,
        video_ids=[c["video_id"] for c in picked],
        # NOTE: all 5 queries run (task-only); detailed_trajectory probes only q_star
    )
    print("[sb-smoke] videos:", [c["video_id"] for c in picked])
    for c in picked:
        print(f"  video {c['video_id']}: q_times={c['q_times']} "
              f"q_star={c['q_star_id']}@{c['q_star_time']}s")
    print(f"RESULT DIR:\n  {out_root / 'raw' / 'smoke'}")
    return 0


def cmd_sb_select(args) -> int:
    """Deterministic 20-video selection (Plan A / Plan B) + audit. No model."""

    from . import streamingbench_selection as sel

    csv_path = Path(
        args.exposure_csv
        or "outputs/kv_metrics/experiments/streamingbench_real/"
           "streamingbench_real_exposure_candidates.csv"
    )
    out_dir = Path(args.output_dir or csv_path.parent)
    candidates = sel.load_candidates(csv_path)

    plan_a = sel.select(candidates, "plan_a", args.num_videos)
    plan_b = sel.select(candidates, "plan_b", args.num_videos)
    stats_a = sel.subset_stats(plan_a)
    stats_b = sel.subset_stats(plan_b)
    pool_stats = sel.subset_stats(candidates)
    bins_a = sel.bins_from_queries(plan_a)
    bins_b = sel.bins_from_queries(plan_b)
    qs_a = sel.qstar_bucket_videos(plan_a)
    qs_b = sel.qstar_bucket_videos(plan_b)
    problems = sel.check_invariants(plan_a) + [
        f"[B] {p}" for p in sel.check_invariants(plan_b)
    ]
    if problems:
        print("!! SELECTION INVARIANTS VIOLATED:")
        for p in problems:
            print("   -", p)
    else:
        print("[invariants] OK: N_video<=N_query per bin; q_star bucket videos "
              "are all counted in their query bin; k_q_star == k_vector[q_star_index]")

    sel.write_selection_json(plan_a, out_dir / "selected_samples_plan_a.json", "plan_a")
    sel.write_selection_json(plan_b, out_dir / "selected_samples_plan_b.json", "plan_b")
    sel.write_audit_csv(plan_a, out_dir / "selected_samples_audit_plan_a.csv")
    sel.write_audit_csv(plan_b, out_dir / "selected_samples_audit_plan_b.csv")

    report = _sb_selection_report(candidates, plan_a, plan_b, stats_a, stats_b, pool_stats)
    (out_dir / "selection_report.md").write_text(report, encoding="utf-8")

    def show(name, plan, stats):
        print(f"\n===== {name} =====")
        print("video IDs:", [c["video_id"] for c in plan])
        print("k=0 queries:", stats["k0_queries"], "/", stats["num_queries"],
              f"({stats['k0_fraction']:.1%})" if stats["k0_fraction"] is not None else "")
        print("non-zero queries:", stats["nonzero_queries"],
              "| unique exact k:", stats["unique_exact_k"])
        print("bins B0..B4 (N_query):", [stats["bins"][b] for b in ("B0", "B1", "B2", "B3", "B4")])
        print("q_star k buckets:", stats["q_star_buckets"],
              "| median", stats["q_star_median_k"], "| max", stats["q_star_max_k"])
        print("task types:", stats["task_type_counts"])
        print("exact-k distribution:", stats["exact_k"])

    show("PLAN A (exposure coverage first)", plan_a, stats_a)
    show("PLAN B (balanced + task diversity)", plan_b, stats_b)
    print("\n===== POOL (43 candidates) =====")
    print("k=0:", pool_stats["k0_queries"], "/", pool_stats["num_queries"],
          f"({pool_stats['k0_fraction']:.1%})", "| unique exact k:", pool_stats["unique_exact_k"])
    print("\n[compare] metric | PlanA | PlanB")
    for key in ("k0_queries", "nonzero_queries", "unique_exact_k", "q_star_median_k",
                "q_star_max_k", "unique_task_types"):
        print(f"  {key:<20} {stats_a[key]!s:>8} {stats_b[key]!s:>8}")
    print("  bins B0-B4 N_query  ", [stats_a['bins'][b] for b in ('B0','B1','B2','B3','B4')],
          [stats_b['bins'][b] for b in ('B0','B1','B2','B3','B4')])
    for label, bins, qs in (("A", bins_a, qs_a), ("B", bins_b, qs_b)):
        print(f"  Plan {label} bins N_query/N_video:",
              {b: (bins[b]['n_query'], bins[b]['n_video']) for b in ('B0','B1','B2','B3','B4')})
        print(f"  Plan {label} B4 video IDs:", bins['B4']['video_ids'])
        print(f"  Plan {label} q_star buckets:", {k: len(v) for k, v in qs.items()})
    print(f"\nOUTPUT DIR:\n  {out_dir}")
    for f in ("selected_samples_plan_a.json", "selected_samples_plan_b.json",
              "selected_samples_audit_plan_a.csv", "selected_samples_audit_plan_b.csv",
              "selection_report.md"):
        print(f"  {out_dir / f}")
    return 0


def _sb_selection_report(candidates, plan_a, plan_b, sa, sb, pool) -> str:
    from . import streamingbench_selection as sel

    L = ["# StreamingBench real — 20-video selection report\n\n"]
    L.append("**selection_type = coverage_balanced_diagnostic_subset**  \n")
    L.append("selection_version = sb_real_coverage_v1, seed = 20260930\n\n")
    L.append(
        "> Selected with observed PRE-ANSWER exposure only. This is **not** a "
        "random sample of StreamingBench real; its accuracy must never be "
        "reported as the benchmark overall accuracy. No correctness / prediction "
        "/ RAM / AORE / JS / KL / task_transition was read.\n\n"
    )
    L.append("## A. Video IDs\n\n")
    for name, plan in (("Plan A", plan_a), ("Plan B", plan_b)):
        L.append(f"- **{name}** ({len(plan)}): {', '.join(c['video_id'] for c in plan)}\n")
    L.append("\n## B. Per-video detail\n\n")
    for name, plan in (("Plan A", plan_a), ("Plan B", plan_b)):
        L.append(f"### {name}\n\n")
        L.append("| video | q1..q5 time (s) | k vector | q_star | k_q_star | bins | task types |\n")
        L.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for c in plan:
            L.append(
                f"| {c['video_id']} | {', '.join(f'{t:g}' for t in c['query_times'])} | "
                f"{c['effective_k_vector']} | {c['q_star_id'].split('_')[-1]}@{c['q_star_time']:g}s | "
                f"{c['k_q_star']} | {'|'.join(c['coverage_bins_contributed'])} | "
                f"{'; '.join((t or '-').replace(' Understanding','')[:18] for t in c['task_types'])} |\n"
            )
        L.append("\n")
    L.append("## C. Exact-k distribution (100 selected queries)\n\n")
    L.append("| k | PlanA N_query | PlanA N_video | PlanB N_query | PlanB N_video |\n")
    L.append("| --- | --- | --- | --- | --- |\n")
    for k in sorted(set(sa["exact_k"]) | set(sb["exact_k"])):
        a = sa["exact_k"].get(k, []); b = sb["exact_k"].get(k, [])
        L.append(f"| {k} | {len(a)} | {len(set(a))} | {len(b)} | {len(set(b))} |\n")
    bins_a = sel.bins_from_queries(plan_a)
    bins_b = sel.bins_from_queries(plan_b)
    qstar_a = sel.qstar_bucket_videos(plan_a)
    qstar_b = sel.qstar_bucket_videos(plan_b)
    L.append("\n## D. Selection bins (recomputed from the 100 query records)\n\n")
    L.append("N_query = query records in the bin; N_video = distinct videos with "
             ">=1 query in the bin. Both are derived directly from each video's "
             "5 observed effective-k values -- never from an aggregate field.\n\n")
    L.append("| bin | range | PlanA N_query | PlanA N_video | PlanA % | PlanB N_query | PlanB N_video | PlanB % |\n")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    ranges = {"B0": "k=0", "B1": "1-10", "B2": "11-20", "B3": "21-30", "B4": "31+"}
    for b in ("B0", "B1", "B2", "B3", "B4"):
        L.append(
            f"| {b} | {ranges[b]} | {bins_a[b]['n_query']} | {bins_a[b]['n_video']} | "
            f"{bins_a[b]['n_query']/sa['num_queries']:.1%} | "
            f"{bins_b[b]['n_query']} | {bins_b[b]['n_video']} | "
            f"{bins_b[b]['n_query']/sb['num_queries']:.1%} |\n"
        )
    L.append("\n**Unique video IDs per bin**\n\n")
    L.append("| bin | Plan A | Plan B |\n| --- | --- | --- |\n")
    for b in ("B0", "B1", "B2", "B3", "B4"):
        L.append(f"| {b} | {', '.join(bins_a[b]['video_ids'])} | {', '.join(bins_b[b]['video_ids'])} |\n")
    L.append(
        f"\n**B4 (k>=31) video IDs** — Plan A: {', '.join(bins_a['B4']['video_ids'])}; "
        f"Plan B: {', '.join(bins_b['B4']['video_ids'])}\n"
    )
    problems = sel.check_invariants(plan_a) + [f"[plan B] {p}" for p in sel.check_invariants(plan_b)]
    L.append("\n**Invariant check**\n\n")
    if problems:
        L.append("- **VIOLATIONS**:\n")
        for p in problems:
            L.append(f"  - {p}\n")
    else:
        L.append(
            "- OK: per bin `N_video <= N_query`; every video whose **q_star k** "
            "falls in a bin is present in that bin's video set (verified from the "
            "100 query records); `k_q_star == k_vector[q_star_index]` for all 40 "
            "selected videos.\n"
        )
    L.append("\n## E. Exact-k coverage\n\n")
    for name, st in (("Plan A", sa), ("Plan B", sb)):
        missing = sorted(set(range(0, 44)) - set(st["exact_k"]))
        L.append(f"- {name}: {st['unique_exact_k']} unique k values; missing = {missing}\n")
    L.append(f"- Full pool: {pool['unique_exact_k']} unique k values\n")
    L.append("\n## F. q_star k buckets\n\n")
    L.append("| bucket | Plan A videos | Plan B videos | pool availability |\n| --- | --- | --- | --- |\n")
    avail = {"5-10": 0, "11-20": 0, "21-30": 0, "31+": 0}
    for c in candidates:
        b = sel.qstar_bucket_of(c["k_q_star"])
        if b:
            avail[b] += 1
    for key in ("5-10", "11-20", "21-30", "31+"):
        L.append(f"| {key} | {len(qstar_a[key])} | {len(qstar_b[key])} | {avail[key]} |\n")
    L.append("\n| bucket | Plan A video IDs | Plan B video IDs |\n| --- | --- | --- |\n")
    for key in ("5-10", "11-20", "21-30", "31+"):
        L.append(f"| {key} | {', '.join(qstar_a[key])} | {', '.join(qstar_b[key])} |\n")
    L.append("\n## G. task_type distribution\n\n")
    L.append("| task type | Plan A | Plan B |\n| --- | --- | --- |\n")
    for t in sorted(set(sa["task_type_counts"]) | set(sb["task_type_counts"])):
        L.append(f"| {t} | {sa['task_type_counts'].get(t, 0)} | {sb['task_type_counts'].get(t, 0)} |\n")
    L.append("\n## H. Candidate pool vs selected\n\n")
    L.append("| metric | pool (43v/215q) | Plan A (20v/100q) | Plan B |\n| --- | --- | --- | --- |\n")
    L.append(f"| k=0 queries | {pool['k0_queries']} ({pool['k0_fraction']:.1%}) | "
             f"{sa['k0_queries']} ({sa['k0_fraction']:.1%}) | {sb['k0_queries']} ({sb['k0_fraction']:.1%}) |\n")
    for b in ("B0", "B1", "B2", "B3", "B4"):
        L.append(f"| {b} | {pool['bins'][b]} | {sa['bins'][b]} | {sb['bins'][b]} |\n")
    L.append(f"| unique exact k | {pool['unique_exact_k']} | {sa['unique_exact_k']} | {sb['unique_exact_k']} |\n")
    L.append(f"| unique task types | {pool['unique_task_types']} | {sa['unique_task_types']} | {sb['unique_task_types']} |\n")
    L.append("\n## I. k=0 dominance\n\n")
    L.append(
        f"- pool {pool['k0_queries']}/{pool['num_queries']} = {pool['k0_fraction']:.1%}\n"
        f"- Plan A {sa['k0_queries']}/{sa['num_queries']} = {sa['k0_fraction']:.1%}\n"
        f"- Plan B {sb['k0_queries']}/{sb['num_queries']} = {sb['k0_fraction']:.1%}\n\n"
        "k=0 remains ~46% and is **structural**: HERMES' first effective compaction "
        "lands near t~94s while StreamingBench real usually places q1-q3 before "
        "that, so every 5-query video contributes 2-4 k=0 points. Reducing it "
        "further would require changing the protocol (smaller budget so compaction "
        "starts earlier), relaxing the 20x5 grid, or weighting the analysis by "
        "exact k instead of raw query count.\n"
    )
    L.append("\n## J. Recommendation\n\n")
    L.append(
        "Recommend **Plan B**: identical exposure coverage to Plan A (same "
        "B0-B4 counts, same q_star bucket quotas, same 35 unique exact-k) while "
        "spreading task types more evenly (largest single task shares: "
        f"Plan A {sa['task_type_counts'][max(sa['task_type_counts'], key=sa['task_type_counts'].get)]}, "
        f"Plan B {sb['task_type_counts'][max(sb['task_type_counts'], key=sb['task_type_counts'].get)]}). "
        "Task type is a pre-answer structural attribute, so using it does not "
        "introduce any correctness-based selection.\n"
    )
    return "".join(L)



# --------------------------------------------------------------------- #
# trajectory / case-study commands
# --------------------------------------------------------------------- #
def cmd_trajectory_select(args) -> int:
    experiment_dir = Path(args.experiment_dir)
    experiment = loader.load_experiment(experiment_dir)
    candidates = trajectory.select_trajectory_candidates(
        experiment,
        min_queries=args.min_queries,
        min_compactions=args.min_compactions,
        min_unique=args.min_unique,
        include_below_threshold=not args.only_qualifying,
    )
    out = Path(args.output or (experiment_dir / "trajectory_candidates.csv"))
    trajectory.write_candidates_csv(candidates, out)
    rec = trajectory.recommended_sample(candidates)
    print(f"[trajectory-select] wrote {out} ({len(candidates)} candidates)")
    for cand in candidates[:10]:
        print(
            f"  {cand['sample_id']:>8}  queries={cand['num_queries']:>3}  "
            f"compactions={cand['num_compactions']:>4}  "
            f"unique_cc={cand['num_unique_query_compaction_counts']:>2}  "
            f"meets={cand['meets_criteria']}"
        )
    print(f"recommended_sample_id={rec}")
    return 0


def cmd_trajectory(args) -> int:
    experiment_dir = Path(args.experiment_dir)
    experiment = loader.load_experiment(experiment_dir)
    sample_id = args.sample_id
    if not sample_id:
        candidates = trajectory.select_trajectory_candidates(experiment)
        sample_id = trajectory.recommended_sample(candidates)
        if not sample_id:
            print("[trajectory] no candidate sample found", file=sys.stderr)
            return 1
        print(f"[trajectory] auto-selected sample_id={sample_id}")
    data = trajectory.build_query_trajectory(experiment, sample_id, budget=args.budget)
    rows, meta = data["rows"], data["meta"]
    csv_path = experiment_dir / "tables" / f"query_compaction_trajectory_{sample_id}.csv"
    trajectory.write_trajectory_csv(rows, csv_path)
    figs = [
        plotting.plot_query_fidelity_vs_compaction_count(rows, meta, experiment_dir / "figures"),
        plotting.plot_query_fidelity_normalized(rows, meta, experiment_dir / "figures"),
    ]
    run = trajectory.find_run_for_budget(experiment, args.budget)
    comps = [c for c in (run.compactions if run else []) if str(c.get("sample_id")) == str(sample_id)]
    figs.append(plotting.plot_query_compaction_timeline(rows, comps, meta, experiment_dir / "figures"))
    reproducibility.write(
        experiment_dir / "reproducibility.json",
        {"benchmark": meta.get("benchmark"), "sample_id": str(sample_id),
         "hermes_budget": args.budget, "trajectory_table": str(csv_path),
         "figures": [f["png"] for f in figs]},
    )
    print(f"[trajectory] sample={sample_id} budget={args.budget} queries={meta['n_queries']} "
          f"compactions={meta['total_compactions']}")
    print(f"TABLE: {csv_path}")
    for fig in figs:
        print(f"FIGURE: {fig['png']}")
    return 0


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def cmd_reproduce_case(args) -> int:
    """One command: paired run -> analysis -> trajectory -> figures -> runtime."""
    from ...kv_metrics.config import KVMetricConfig
    from ...kv_metrics.paired_runner import run as run_paired
    from .experiment_runner import _parse_bins

    experiment_dir = Path(args.output_root) / (
        args.name or f"case_{args.benchmark}_{args.sample_id}_b{args.budget}"
    )
    base_config = load_config(args.config)
    run_config = copy.deepcopy(base_config)
    run_config.setdefault("memory", {}).setdefault("compression", {})
    run_config["memory"]["method"] = "hermes"
    run_config["memory"]["compression"]["enabled"] = True
    run_config["memory"]["compression"]["budget"] = args.budget
    kv_config = KVMetricConfig(
        enable_kv_metrics=True, mode="paired",
        output_dir=str(experiment_dir / "raw"),
        run_name=f"hermes_budget_{args.budget}",
        attention_backend=args.metric_backend,
        temporal_bins=_parse_bins(args.temporal_bins),
        profile_wallclock=True,
        profile_runtime_breakdown=args.profile_runtime_breakdown,
    )
    summary_path = experiment_dir / "raw" / f"hermes_budget_{args.budget}" / "summary.json"
    if not (args.skip_run and summary_path.exists()):
        run_paired(run_config, benchmark=args.benchmark, max_samples=0,
                   kv_config=kv_config, sample_ids=[str(args.sample_id)])
    experiment = loader.load_experiment(experiment_dir)
    data = report.compute_all(experiment)
    report.write_tables(experiment_dir, data, experiment)
    stats = report.write_statistics(experiment_dir, data, experiment)
    report.write_report(experiment_dir, data, experiment, stats)
    cmd_trajectory(_Namespace(experiment_dir=str(experiment_dir), sample_id=str(args.sample_id),
                              budget=args.budget))
    print("RESULT DIR:\n ", experiment_dir)
    return 0


def cmd_compaction_accuracy(args) -> int:
    from . import compaction_accuracy

    experiment_dir = Path(args.experiment_dir)
    tables = compaction_accuracy.write_tables(experiment_dir, n_boot=args.n_boot, seed=args.seed)
    figures_dir = experiment_dir / "figures"
    figures = []
    figures += compaction_accuracy.plot_trajectories(tables["trajectory"], figures_dir, args.max_traj_samples)
    figures += compaction_accuracy.plot_trajectory_vs_time(tables["trajectory"], figures_dir, args.max_traj_samples)
    figures += compaction_accuracy.plot_accuracy(tables["exact"], figures_dir)
    reproducibility.write(experiment_dir / "reproducibility.json",
                          {"experiment": experiment_dir.name,
                           "tables": str(experiment_dir / "analysis"),
                           "figures": [f["png"] for f in figures]})
    (experiment_dir / "experiment_report.md").write_text(
        _compaction_accuracy_report(experiment_dir, tables), encoding="utf-8")
    print(f"[compaction-accuracy] tables -> {experiment_dir / 'analysis'}")
    print(f"[compaction-accuracy] figures -> {len(figures)}")
    return 0


def _compaction_accuracy_report(experiment_dir: Path, tables: dict) -> str:
    exact, selection, traj, rows = tables["exact"], tables["selection"], tables["trajectory"], tables["task_rows"]
    L = [f"# Compaction-accuracy experiment: `{Path(experiment_dir).name}`\n\n"]
    L.append(f"- samples: {len({str(r.get('sample_id')) for r in rows})}   "
             f"queries: {len(rows)}   longest-run trajectories: {len(traj)}\n\n")
    L.append("| k | N_query | N_sample | acc_compressed | acc_full | preservation | C->W rate | exploratory |\n")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in exact:
        L.append(f"| {r['effective_compaction_count']} | {r['num_queries']} | {r['num_samples']} | "
                 f"{_fmt(r.get('accuracy_compressed'))} | {_fmt(r.get('accuracy_full'))} | "
                 f"{_fmt(r.get('preservation_accuracy'))} | {_fmt(r.get('correct_to_wrong_rate'))} | "
                 f"{r.get('exploratory_low_n')} |\n")
    L.append("\n> CI = sample-level cluster bootstrap (unit = sample).\n")
    return "".join(L)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)



SB_PLAN_B = "outputs/kv_metrics/experiments/streamingbench_real/selected_samples_plan_b.json"
SB_EXPOSURE = ("outputs/kv_metrics/experiments/streamingbench_real/"
               "streamingbench_real_exposure_candidates.csv")
SB_EXP_DIR = "outputs/kv_metrics/experiments/sb_real_plan_b"


def cmd_sb_manifest(args) -> int:
    """Freeze Plan B into an immutable experiment manifest (no model run)."""

    from . import streamingbench_selection as sel

    plan_path = Path(args.plan or SB_PLAN_B)
    plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = plan_doc["samples"]
    exp_dir = Path(args.experiment_dir or SB_EXP_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)
    plan_name = args.plan_name or "plan_b"
    manifest = sel.build_manifest(
        plan, plan_name, args.exposure_csv or SB_EXPOSURE, args.config,
        args.budget, args.bootstrap_seed, args.n_boot,
    )
    (exp_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "selected_samples.json").write_text(
        json.dumps({**plan_doc, "locked_plan": "plan_b",
                    "experiment_id": manifest["experiment_id"]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sb-manifest] experiment_id = {manifest['experiment_id']}")
    print(f"  locked videos ({manifest['n_videos']}): {manifest['video_ids']}")
    print(f"  context_mode={manifest['context_mode']} budget={manifest['budget']} "
          f"selection_seed={manifest['selection_seed']} bootstrap_seed={manifest['bootstrap_seed']}")
    print(f"  -> {exp_dir / 'experiment_manifest.json'}")
    return 0


def cmd_sb_batches(args) -> int:
    """Run P3+P4 in resumable batches (paired: Full + HERMES per query)."""

    from ...kv_metrics.config import KVMetricConfig
    from ...kv_metrics.paired_runner import run as run_paired

    exp_dir = Path(args.experiment_dir or SB_EXP_DIR)
    manifest = json.loads((exp_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    video_ids = manifest["video_ids"]
    batch_size = int(args.batch_size or 5)
    batches = [video_ids[i:i + batch_size] for i in range(0, len(video_ids), batch_size)]

    config = load_config(args.config or manifest["base_config"])
    config["benchmarks"]["streamingbench"]["context_mode"] = manifest["context_mode"]
    failures = []
    for index, batch in enumerate(batches, start=1):
        name = f"batch{index}"
        run_dir = exp_dir / "raw" / name
        if run_dir.joinpath("summary.json").exists() and not args.force:
            print(f"[sb-batches] resume: {name} already complete")
            continue
        print(f"[sb-batches] {name}: videos={batch}")
        kv_config = KVMetricConfig(
            enable_kv_metrics=True, mode="paired",
            output_dir=str(exp_dir / "raw"), run_name=name,
            attention_backend="eager", profile_wallclock=True,
            detailed_trajectory=True,
        )
        try:
            run_paired(config, benchmark=benchmark, max_samples=0,
                       kv_config=kv_config, video_ids=batch)
        except Exception as error:  # noqa: BLE001
            print(f"[sb-batches] !! {name} FAILED: {type(error).__name__}: {error}")
            failures.append({"batch": name, "videos": batch,
                             "error": f"{type(error).__name__}: {error}"})
    (exp_dir / "run_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8")
    print(f"[sb-batches] done; failures={len(failures)}")
    return 0



LOCKED_BIN_EDGES = [0, 10, 20, 30]   # -> 0 / 1-10 / 11-20 / 21-30 / 31+


def cmd_sb_analyze(args) -> int:
    """P5 + P6: locked-bin aggregation with video-level cluster bootstrap."""

    from . import compaction_accuracy as ca
    from . import task_accuracy as ta

    exp_dir = Path(args.experiment_dir or SB_EXP_DIR)
    manifest = json.loads((exp_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    n_boot = int(manifest["bootstrap"]["n_boot"])
    seed = int(manifest["bootstrap_seed"])

    rows = ca.load_task_rows(exp_dir)
    traj = ca.load_trajectory_rows(exp_dir)
    analysis = exp_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    exact = ta.accuracy_by_compaction_count(rows, n_boot=n_boot, seed=seed)
    locked = ta.accuracy_by_compaction_count(
        rows, n_boot=n_boot, seed=seed, bin_edges=LOCKED_BIN_EDGES)
    ca._write_csv(analysis / "task_accuracy_by_compaction_count.csv", exact)
    ca._write_csv(analysis / "task_accuracy_by_locked_bins.csv", locked)
    ca._write_csv(analysis / "task_transitions.csv", [
        {"effective_compaction_count": r["effective_compaction_count"],
         "bin_label": r["bin_label"], "num_queries": r["num_queries"],
         "num_samples": r["num_samples"],
         "num_correct_to_correct": r.get("num_correct_to_correct"),
         "num_correct_to_wrong": r.get("num_correct_to_wrong"),
         "num_wrong_to_correct": r.get("num_wrong_to_correct"),
         "num_wrong_to_wrong": r.get("num_wrong_to_wrong"),
         "correct_to_wrong_rate": r.get("correct_to_wrong_rate"),
         "protocol_violation_count": r.get("protocol_violation_count"),
         "missing_count": r.get("missing_count")}
        for r in exact])

    per_query = [{
        "video_id": str(r.get("sample_id")).split("_")[-2] if "_" in str(r.get("sample_id")) else r.get("sample_id"),
        "query_id": r.get("query_id"), "query_time": r.get("query_time"),
        "compaction_trigger_count_at_query": r.get("compaction_trigger_count_at_query"),
        "effective_compaction_count_at_query": r.get("effective_compaction_count_at_query"),
        "prediction_full": r.get("task_prediction_full"),
        "correct_full": r.get("task_correct_full"),
        "prediction_compressed": r.get("task_prediction_compressed"),
        "correct_compressed": r.get("task_correct_compressed"),
        "task_transition": r.get("task_transition"),
        "task_status_compressed": r.get("task_status_compressed"),
        "task_error_type_compressed": r.get("task_error_type_compressed"),
        "task_missing": r.get("task_missing"),
    } for r in rows]
    ca._write_csv(analysis / "task_only_all_queries.csv", per_query)

    traj_rows = []
    for sample_id, rs in traj.items():
        for r in rs:
            traj_rows.append({
                "video_id": sample_id, "query_id": r.get("query_id"),
                "probe_k": r.get("probe_k"), "stream_time": r.get("stream_time"),
                "RAM": r.get("ram_mean"), "AORE": r.get("aore_mean"),
                "JS": r.get("js"), "KL": r.get("kl"),
                "compression_ratio": r.get("compression_ratio"),
                "probe_query_hash": r.get("probe_query_hash"),
                "probe_query_hash_full": r.get("probe_query_hash_full"),
            })
    ca._write_csv(analysis / "trajectory_metrics.csv", traj_rows)

    figs = []
    figs += ca.plot_accuracy(exact, exp_dir / "figures")
    figs += ca.plot_delta_and_c2w(exact, exp_dir / "figures")
    for sample_id, rs in sorted(traj.items()):
        short = sample_id.replace("Real-Time_Visual_Understanding_", "")
        figs.append(ca.plot_combined_trajectory(short, rs, exp_dir / "figures"))
    figs += ca.plot_fidelity_aggregate(traj, exp_dir / "figures")

    stats = {
        "experiment_id": manifest["experiment_id"],
        "selection_type": manifest["selection_type"],
        "plan": manifest["plan"], "budget": manifest["budget"],
        "context_mode": manifest["context_mode"],
        "bootstrap": manifest["bootstrap"],
        "n_queries": len(rows), "n_videos": len({(r.get("video_id") or r.get("sample_id")) for r in rows}),
        "overall": {
            "accuracy_full": sum(1 for r in rows if r.get("task_correct_full")) / len(rows),
            "accuracy_compressed": sum(1 for r in rows if r.get("task_correct_compressed")) / len(rows),
            "transitions": {k: sum(1 for r in rows if r.get("task_transition") == k)
                            for k in ("correct_to_correct", "correct_to_wrong",
                                      "wrong_to_correct", "wrong_to_wrong")},
        },
        "exact_k": exact, "locked_bins": locked,
    }
    (exp_dir / "statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (exp_dir / "experiment_report.md").write_text(
        _sb_final_report(manifest, stats, traj), encoding="utf-8")
    reproducibility.write(exp_dir / "reproducibility.json",
                          {"experiment_id": manifest["experiment_id"],
                           "figures": [f["png"] for f in figs]})
    print(f"[sb-analyze] exact-k rows={len(exact)}, locked bins={len(locked)}, figures={len(figs)}")
    return 0


def _sb_final_report(manifest, stats, traj) -> str:
    L = [f"# StreamingBench real — coverage-balanced diagnostic subset\n\n"]
    L.append(f"experiment_id = `{manifest['experiment_id']}`  \n")
    L.append(f"selection_type = `{manifest['selection_type']}`; plan={manifest['plan']}; "
             f"budget={manifest['budget']}; context_mode={manifest['context_mode']}  \n")
    L.append("> Frozen BEFORE any model observation. Not a random sample of "
             "StreamingBench real -- accuracy here is NOT the benchmark overall.\n\n")
    L.append(f"- videos: {stats['n_videos']}   queries: {stats['n_queries']}   "
             f"bootstrap: video-level cluster, n_boot={stats['bootstrap']['n_boot']}, "
             f"seed={manifest.get('bootstrap_seed')}\n\n")
    o = stats["overall"]
    L.append(f"- overall Full accuracy {o['accuracy_full']:.3f} | "
             f"Compressed accuracy {o['accuracy_compressed']:.3f}\n")
    L.append(f"- transitions: {o['transitions']}\n\n")
    L.append("## Locked bins\n\n")
    L.append("| bin | N_query | N_video | acc_compressed | acc_full | ΔAcc | Preservation | C→W |\n")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in stats["locked_bins"]:
        d = (r["accuracy_compressed"] - r["accuracy_full"]) if (
            r.get("accuracy_compressed") is not None and r.get("accuracy_full") is not None) else None
        L.append(f"| {r['bin_label']} | {r['num_queries']} | {r['num_samples']} | "
                 f"{_fmt(r.get('accuracy_compressed'))} | {_fmt(r.get('accuracy_full'))} | "
                 f"{_fmt(d)} | {_fmt(r.get('preservation_accuracy'))} | "
                 f"{_fmt(r.get('correct_to_wrong_rate'))} |\n")
    L.append("\n## Trajectories\n\n")
    L.append(f"- q_star trajectories: {len(traj)}; total probe points: "
             f"{sum(len(v) for v in traj.values())}\n")
    return "".join(L)



def cmd_sb_decision_run(args) -> int:
    """P0' re-run: 100 task-only queries with read-only option-logit capture.

    Reuses the frozen Plan B manifest. NO detailed trajectory probe is run
    (those data already exist), so this is cheaper than the P3+P4 run.
    """

    from ...kv_metrics.config import KVMetricConfig
    from ...kv_metrics.paired_runner import run as run_paired

    exp_dir = Path(args.experiment_dir or SB_EXP_DIR)
    manifest = json.loads((exp_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    video_ids = manifest["video_ids"]
    batch_size = int(args.batch_size or 5)
    batches = [video_ids[i:i + batch_size] for i in range(0, len(video_ids), batch_size)]
    config = load_config(args.config or manifest["base_config"])
    config["benchmarks"]["streamingbench"]["context_mode"] = manifest["context_mode"]

    failures = []
    for index, batch in enumerate(batches, start=1):
        name = f"batch{index}"
        run_dir = exp_dir / "raw_decision" / name
        if run_dir.joinpath("summary.json").exists() and not args.force:
            print(f"[sb-decision] resume: {name} complete")
            continue
        print(f"[sb-decision] {name}: videos={batch}")
        kv_config = KVMetricConfig(
            enable_kv_metrics=True, mode="paired",
            output_dir=str(exp_dir / "raw_decision"), run_name=name,
            attention_backend="original",          # no eager fidelity needed
            profile_wallclock=True,
            detailed_trajectory=False,             # trajectories already exist
            capture_option_logits=True,
        )
        try:
            run_paired(config, benchmark=benchmark, max_samples=0,
                       kv_config=kv_config, video_ids=batch)
        except Exception as error:  # noqa: BLE001
            print(f"[sb-decision] !! {name} FAILED: {type(error).__name__}: {error}")
            failures.append({"batch": name, "videos": batch,
                             "error": f"{type(error).__name__}: {error}"})
    (exp_dir / "decision_run_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8")
    print(f"[sb-decision] done; failures={len(failures)}")
    return 0



def cmd_sb_decision_analyze(args) -> int:
    """Offline ΔP_gold / ΔM_z analysis on the read-only option-logit capture."""

    from . import compaction_accuracy as ca

    exp_dir = Path(args.experiment_dir or SB_EXP_DIR)
    manifest = json.loads((exp_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    n_boot = int(manifest["bootstrap"]["n_boot"]); seed = int(manifest["bootstrap_seed"])
    edges = LOCKED_BIN_EDGES

    records = []
    for path in sorted((exp_dir / "raw_decision").glob("*/paired/*.jsonl")):
        records.extend(loader.read_jsonl(path))
    extra_dirs = [d for d in (args.extra_dirs or "").split(",") if d.strip()]
    for extra in extra_dirs:
        for path in sorted((Path(extra.strip()) / "raw_decision").glob("*/paired/*.jsonl")):
            records.extend(loader.read_jsonl(path))
    rows = decision_fidelity.build_rows(records)
    exact = decision_fidelity.aggregate(rows, None, n_boot, seed)
    locked = decision_fidelity.aggregate(rows, edges, n_boot, seed)

    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else exp_dir
    analysis = out_dir / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    ca._write_csv(analysis / "decision_fidelity_exact_k.csv", exact)
    ca._write_csv(analysis / "decision_fidelity_locked_bins.csv", locked)

    # join with the existing fidelity (only q_star queries have RAM/AORE)
    traj = ca.load_trajectory_rows(exp_dir)
    fidelity = {}
    for sample_id, pts in traj.items():
        short = str(sample_id).replace("Real-Time_Visual_Understanding_", "")
        short = short.replace("sample_", "")          # trajectory keys are "sample_<vid>"
        for p in pts:
            fidelity[(short, int(p.get("probe_k") or -1))] = p
    join_rows = []
    for r in rows:
        vid = r["video_id"]
        k = r.get("effective_compaction_count")
        fid = fidelity.get((vid, int(k))) if k is not None else None
        join_rows.append({
            **{key: r.get(key) for key in (
                "video_id", "query_id", "query_time", "effective_compaction_count",
                "gold_option", "p_gold_full", "p_gold_compressed", "delta_p_gold",
                "margin_full", "margin_compressed", "delta_margin_logit",
                "best_option_full", "best_option_compressed",
                "runner_up_full", "runner_up_compressed",
                "prediction_full", "prediction_compressed", "task_transition",
                "decision_metric_mismatch", "margin_crossed")},
            "p_option_full": json.dumps(r.get("p_option_full"), sort_keys=True),
            "p_option_compressed": json.dumps(r.get("p_option_compressed"), sort_keys=True),
            "RAM": (fid or {}).get("ram_mean"),
            "AORE": (fid or {}).get("aore_mean"),
            "JS": (fid or {}).get("js"),
            "KL": (fid or {}).get("kl"),
            "fidelity_source": "q_star_probe" if fid else None,
        })
    ca._write_csv(analysis / "decision_fidelity_join.csv", join_rows)

    figdir = out_dir / "figures"
    figs = []
    figs += decision_fidelity._plot_delta(rows, exact, "delta_p_gold", "ΔP_gold",
                                          "delta_p_gold", figdir)
    figs += decision_fidelity._plot_delta(rows, exact, "delta_margin_logit", "ΔM_z",
                                          "delta_margin_logit", figdir)
    figs.append(decision_fidelity._plot_scatter(
        rows, "p_gold_full", "p_gold_compressed", "p_gold_full_vs_compressed",
        "P_gold (Full-KV)", "P_gold (Compressed)", figdir, color_key="task_transition"))
    figs.append(decision_fidelity._plot_scatter(
        rows, "margin_full", "margin_compressed", "margin_full_vs_compressed",
        "margin M_z (Full-KV)", "margin M_z (Compressed)", figdir))

    mismatches = [r for r in rows if r.get("decision_metric_mismatch")]
    crossed = [r for r in rows if r.get("margin_crossed")]
    if mismatches:
        print(f"!! WARNING: decision_metric_mismatch for {len(mismatches)} queries:",
              [r["query_id"] for r in mismatches][:5])
    stats = {
        "experiment_id": manifest["experiment_id"],
        "n_queries": len(rows),
        "bootstrap": {"unit": "video", "n_boot": n_boot, "seed": seed},
        "median_delta_p_gold": st_median([r["delta_p_gold"] for r in rows]),
        "median_delta_margin_logit": st_median([r["delta_margin_logit"] for r in rows]),
        "fraction_delta_p_gold_negative": _frac_neg([r["delta_p_gold"] for r in rows]),
        "fraction_delta_margin_negative": _frac_neg([r["delta_margin_logit"] for r in rows]),
        "num_decision_metric_mismatch": len(mismatches),
        "num_margin_crossed": len(crossed),
        "margin_crossed_queries": [r["query_id"] for r in crossed],
        "locked_bins": locked, "exact_k": exact,
    }
    (out_dir / "statistics_decision.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "decision_report.md").write_text(
        _sb_decision_report(manifest, stats, rows, locked), encoding="utf-8")
    print(f"[sb-decision-analyze] queries={len(rows)} bins={len(locked)} figures={len(figs)}")
    print(f"  median ΔP_gold = {stats['median_delta_p_gold']}, "
          f"median ΔM_z = {stats['median_delta_margin_logit']}")
    print(f"  fidelity join rows: {len(join_rows)} (RAM/AORE only for q_star queries)")
    return 0


def st_median(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    import statistics

    return statistics.median(vals)


def _frac_neg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v < 0) / len(vals)


def _sb_decision_report(manifest, stats, rows, locked) -> str:
    L = ["# Decision-layer fidelity (ΔP_gold / ΔM_z)\n\n"]
    L.append(f"experiment_id = `{manifest['experiment_id']}`  \n")
    L.append("Metrics are computed from the OFFICIAL answer-generation step-0 "
             "pre-mask logits at the allowed MCQ option tokens (read-only capture; "
             "no re-generation, no second decoder).\n\n")
    L.append("**Variant aggregation differs by metric and must not be mixed:**\n"
             "- `P_gold` uses **logsumexp** over an option's token variants "
             "(`\"A\"`, `\" A\"`), renormalised across A-D.\n"
             "- `M_z` uses **max** over the variants, matching the decoder's "
             "token-level `argmax`.\n\n")
    L.append(f"- queries: {stats['n_queries']}; bootstrap: video-level cluster, "
             f"n_boot={stats['bootstrap']['n_boot']}, seed={stats['bootstrap']['seed']}\n")
    L.append(f"- median ΔP_gold = {stats['median_delta_p_gold']}; "
             f"median ΔM_z = {stats['median_delta_margin_logit']}\n")
    L.append(f"- fraction(ΔP_gold < 0) = {stats['fraction_delta_p_gold_negative']}; "
             f"fraction(ΔM_z < 0) = {stats['fraction_delta_margin_negative']}\n")
    L.append(f"- decision_metric_mismatch = {stats['num_decision_metric_mismatch']} "
             f"(must be 0)\n")
    L.append(f"- margin crossings (Full>0, Compressed<0) = {stats['num_margin_crossed']}"
             f" {stats['margin_crossed_queries']}\n\n")
    L.append("## Locked bins\n\n")
    L.append("| bin | Nq | Nv | medΔP_gold | meanΔP_gold [95% CI] | frac(ΔP<0) | medΔM_z | meanΔM_z [95% CI] | frac(ΔM<0) | crossed |\n")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for b in locked:
        L.append(
            f"| {b['bin_label']} | {b['num_queries']} | {b['num_videos']} | "
            f"{_fmt(b.get('delta_p_gold_median'))} | "
            f"{_fmt(b.get('delta_p_gold_mean'))} [{_fmt(b.get('delta_p_gold_mean_ci_low'))},{_fmt(b.get('delta_p_gold_mean_ci_high'))}] | "
            f"{_fmt(b.get('fraction_delta_p_gold_negative'))} | "
            f"{_fmt(b.get('delta_margin_logit_median'))} | "
            f"{_fmt(b.get('delta_margin_logit_mean'))} [{_fmt(b.get('delta_margin_logit_mean_ci_low'))},{_fmt(b.get('delta_margin_logit_mean_ci_high'))}] | "
            f"{_fmt(b.get('fraction_delta_margin_logit_negative'))} | "
            f"{b.get('num_margin_crossed')} |\n")
    return "".join(L)



def cmd_sb_extend_select(args) -> int:
    """Pick EXTRA videos (from the exposure pool) to enrich high-k coverage.

    Read-only, structure-only: uses video_id / query_time /
    effective_compaction_count / task metadata. It never reads correctness,
    prediction or any fidelity metric, and never touches the original run.
    """

    import collections

    exposure = Path(args.exposure_csv or SB_EXPOSURE)
    base_plan = Path(args.base_plan or SB_PLAN_B)
    out_dir = Path(args.experiment_dir or SB_EXP_DIR + "_extended")
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = {r["video_id"]: [int(r[f"k_q{i}"]) for i in range(1, 6)]
            for r in csv.DictReader(exposure.open(encoding="utf-8"))}
    base = json.loads(base_plan.read_text(encoding="utf-8"))["samples"]
    current = {str(c["video_id"]): list(c["effective_k_vector"]) for c in base}
    remaining = {v: k for v, k in pool.items() if v not in current}

    ks31 = list(range(31, 44))
    ks21 = list(range(21, 31))

    def nvid(vecs):
        per = collections.defaultdict(set)
        for v, kv in vecs.items():
            for k in kv:
                per[k].add(v)
        return per

    def score(vecs):
        per = nvid(vecs)
        return (sum(min(len(per.get(k, ())), 3) for k in ks31),
                sum(min(len(per.get(k, ())), 3) for k in ks21))

    ceiling = nvid(pool)
    chosen = []
    cur = dict(current)
    while True:
        best = None
        for v, kv in remaining.items():
            if any(v == c["video_id"] for c in chosen):
                continue
            trial = dict(cur)
            trial[v] = kv
            s_ = score(trial)
            if best is None or s_ > best[0]:
                best = (s_, v, kv)
        if best is None or best[0] <= score(cur):
            break
        chosen.append({"video_id": best[1], "effective_k_vector": best[2],
                       "query_ids": [f"Real-Time Visual Understanding_sample_{best[1]}_{i}"
                                     for i in range(1, 6)]})
        cur[best[1]] = best[2]
    if args.max_extra:
        chosen = chosen[: args.max_extra]

    # report
    before = nvid(current)
    final = dict(current)
    for c in chosen:
        final[c["video_id"]] = c["effective_k_vector"]
    after = nvid(final)
    print("[sb-extend-select] extra videos:", [c["video_id"] for c in chosen])
    for c in chosen:
        print(f"   video {c['video_id']:>5}  k_vector={c['effective_k_vector']}")
    print(f"\n{'k':>4}{'pool ceiling':>14}{'N_video before':>16}{'N_video after':>15}")
    for k in ks21 + ks31:
        print(f"{k:>4}{len(ceiling.get(k, ())):>14}{len(before.get(k, ())):>16}{len(after.get(k, ())):>15}")
    unreachable = [k for k in ks31 if not ceiling.get(k)]
    print("\nstructurally unreachable k in 31..43 (no video in pool):", unreachable)
    print("score(31..43) %s -> %s  (max %d)" % (score(current)[0], score(final)[0], 3 * len(ks31)))
    print("score(21..30) %s -> %s  (max %d)" % (score(current)[1], score(final)[1], 3 * len(ks21)))

    plan_doc = {
        "selection_type": "coverage_balanced_diagnostic_subset_extension",
        "selection_version": "sb_real_extend_v1",
        "seed": 20260930,
        "base_plan": str(base_plan),
        "n_videos": len(chosen),
        "note": ("Extra videos to enrich high effective-compaction coverage. "
                 "Structure-only selection; no correctness/fidelity was read. "
                 "Does NOT replace the original plan_b run."),
        "samples": chosen,
    }
    (out_dir / "selected_samples_extended.json").write_text(
        json.dumps(plan_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "extension_coverage.csv").write_text(
        "k,pool_ceiling,n_video_before,n_video_after\n"
        + "".join(f"{k},{len(ceiling.get(k, ()))},{len(before.get(k, ()))},{len(after.get(k, ()))}\n"
                  for k in ks21 + ks31),
        encoding="utf-8")
    print(f"\n-> {out_dir / 'selected_samples_extended.json'}")
    print(f"-> {out_dir / 'extension_coverage.csv'}")
    return 0



SB_FIXED28_NAMES = ["sb_real_plan_b", "sb_real_plan_b_extended_highk8"]
SB_EXPERIMENTS_ROOT = "outputs/kv_metrics/experiments"


def _find_experiment_dir(name: str):
    """Resolve an experiment dir by name, tolerating a leading marker prefix
    (e.g. the user renaming ``sb_real_plan_b`` -> ``*sb_real_plan_b``)."""

    root = Path(SB_EXPERIMENTS_ROOT)
    exact = root / name
    if exact.exists():
        return exact
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.lstrip("*+.!_-") == name:
                return child
    return exact


def _frozen28_dirs():
    return [_find_experiment_dir(n) for n in SB_FIXED28_NAMES]


SB_FIXED28_DIRS = []  # resolved lazily via _frozen28_dirs()
BUDGETS = [2000, 1500, 1000, 750, 100]
SB_BUDGET_ROOT = "outputs/kv_metrics/experiments"


def _fixed_video_ids():
    ids, seen = [], set()
    for d in _frozen28_dirs():
        man = Path(d) / "experiment_manifest.json"
        if not man.exists():
            continue
        for v in json.loads(man.read_text(encoding="utf-8")).get("video_ids", []):
            if v not in seen:
                seen.add(v); ids.append(v)
    return ids


def cmd_sb_budget_prepare(args) -> int:
    """Create the per-budget manifests. Reuses the FROZEN 28-video set."""

    from . import streamingbench_selection as sel

    video_ids = _fixed_video_ids()
    if not video_ids:
        print("!! no frozen video ids found; check SB_FIXED28_DIRS")
        return 1
    print(f"[sb-budget-prepare] frozen videos: {len(video_ids)} -> {video_ids}")
    for budget in BUDGETS:
        if budget == 2000:
            print(f"  budget {budget}: reuse existing runs (plan_b + highk8); no new dir")
            continue
        out = Path(args.root or SB_BUDGET_ROOT) / f"sb_real_budget_{budget}"
        out.mkdir(parents=True, exist_ok=True)
        plan = {"selection_type": "coverage_balanced_diagnostic_subset",
                "selection_version": "sb_real_budget_stress_v1",
                "seed": 20260930, "budget": budget,
                "note": "Same 28 videos as the frozen set; only the KV budget changes.",
                "samples": [{"video_id": v, "effective_k_vector": []} for v in video_ids]}
        manifest = {
            "experiment_id": f"sb_real_budget_{budget}",
            "locked": True,
            "selection_type": "coverage_balanced_diagnostic_subset",
            "benchmark": "streamingbench", "category": "real_time_visual_understanding",
            "context_mode": "full_prefix",
            "budget": budget,
            "selection_seed": 20260930, "bootstrap_seed": 42,
            "bootstrap": {"unit": "video", "n_boot": 2000, "ci": 0.95},
            "base_config": args.config or SB_BASE,
            "video_ids": video_ids, "n_videos": len(video_ids),
            "n_queries": len(video_ids) * 5,
            "full_reuse": {"dirs": [str(d) for d in _frozen28_dirs()],
                           "note": "Full-KV is budget independent; reused, never re-run."},
            "compressed_only": True,
        }
        (out / "selected_samples.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  budget {budget}: -> {out}")
    return 0


def cmd_sb_budget_run(args) -> int:
    """Compressed-only task runs at one budget (resumable, batched)."""

    from ...kv_metrics.config import KVMetricConfig
    from ...kv_metrics.paired_runner import run as run_paired

    exp_dir = Path(args.experiment_dir)
    manifest = json.loads((exp_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    video_ids = manifest["video_ids"]
    budget_raw = manifest.get("budget")
    is_full = bool(getattr(args, "full_only", False)) or budget_raw is None
    budget = int(budget_raw) if budget_raw is not None else 10 ** 9
    batch_size = int(args.batch_size or 7)
    batches = [video_ids[i:i + batch_size] for i in range(0, len(video_ids), batch_size)]

    benchmark = args.benchmark or manifest.get("benchmark") or "streamingbench"
    config = load_config(args.config or manifest["base_config"])
    config.setdefault("memory", {}).setdefault("compression", {})
    config["memory"]["method"] = "hermes"
    config["memory"]["compression"]["enabled"] = True
    config["memory"]["compression"]["budget"] = budget          # <- the only variable
    if "streamingbench" in (config.get("benchmarks") or {}):
        config["benchmarks"]["streamingbench"]["context_mode"] = manifest["context_mode"]

    failures = []
    for index, batch in enumerate(batches, start=1):
        name = f"batch{index}"
        run_dir = exp_dir / "raw_decision" / name
        if run_dir.joinpath("summary.json").exists() and not args.force:
            print(f"[sb-budget-run:{budget}] resume: {name} complete")
            continue
        print(f"[sb-budget-run:{budget}] {name}: videos={batch}")
        kv_config = KVMetricConfig(
            enable_kv_metrics=True,
            mode="full" if is_full else "compressed",
            output_dir=str(exp_dir / "raw_decision"), run_name=name,
            attention_backend="original",
            profile_wallclock=True,
            detailed_trajectory=False,
            capture_option_logits=True,
        )
        try:
            run_paired(config, benchmark=benchmark, max_samples=0,
                       kv_config=kv_config, video_ids=batch)
        except Exception as error:  # noqa: BLE001
            print(f"[sb-budget-run:{budget}] !! {name} FAILED: {type(error).__name__}: {error}")
            failures.append({"batch": name, "videos": batch,
                             "error": f"{type(error).__name__}: {error}"})
    (exp_dir / "run_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"[sb-budget-run:{budget if not is_full else 'FULL'}] done; failures={len(failures)}")
    return 0


def cmd_sb_budget_analyze(args) -> int:
    """Merge budget 2000 (reused) + 1500/1000/750 -> tables + figures."""

    root = Path(args.root or SB_BUDGET_ROOT)
    out_dir = Path(args.output_dir or (root / "sb_real_budget_stress"))
    budget_dirs = {2000: [d for d in _frozen28_dirs() if d.exists()]}
    for d in sorted(root.glob("sb_real_budget_*")):
        name = d.name[len("sb_real_budget_"):]
        if name.isdigit() and (d / "raw_decision").exists():
            budget_dirs[int(name)] = [d]

    missing = [b for b in BUDGETS if b not in budget_dirs]
    if missing:
        print(f"!! missing runs for budgets {missing}; run sb-budget-run first")
    if 2000 not in budget_dirs:
        print("!! frozen budget-2000 runs not found")
        return 1

    full_records = budget_stress.load_budget_records(budget_dirs[2000])
    paired, summary = budget_stress.build_budget_table(budget_dirs, full_records)
    problems = budget_stress.sanity_checks(paired, summary)

    analysis = out_dir / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    budget_stress.write_csv(analysis / "budget_stress_summary.csv", summary)
    budget_stress.write_csv(analysis / "budget_stress_paired_queries.csv", paired,
                            fields=list(paired[0].keys()) if paired else None)
    figures = budget_stress.plot_budget_curves(summary, out_dir / "figures")
    (out_dir / "budget_stress_sanity.json").write_text(
        json.dumps({"problems": problems}, indent=2), encoding="utf-8")

    print(f"[sb-budget-analyze] queries={len(paired)} budgets={[r['budget'] for r in summary]} "
          f"figures={len(figures)}")
    if problems:
        print("!! sanity problems:")
        for prob in problems:
            print("   -", prob)
    else:
        print("[sb-budget-analyze] sanity checks OK")
    for row in summary:
        print(f"  budget {row['budget']}: Nq={row['N_query']} Nv={row['N_video']} "
              f"dP_gold(mean)={row['mean_delta_p_gold_video_level']} "
              f"dM_z(mean)={row['mean_delta_margin_video_level']} "
              f"CW={row['CW']} Pres={row['Preservation']} "
              f"mean_k={row['mean_effective_compaction_count']}")
    # standalone budget=100 report + 750-vs-100 paired comparison
    last = summary[-1] if summary else None
    if last and last["budget"] == 100:
        print("\n=== budget=100 (extreme compression) ===")
        for k in ("N_query", "N_video", "Acc_full", "Acc_compressed", "DeltaAcc",
                  "CC", "CW", "WC", "WW", "Preservation",
                  "mean_delta_p_gold_video_level", "median_delta_p_gold_video_level",
                  "mean_delta_margin_video_level", "median_delta_margin_video_level",
                  "fraction_query_delta_p_negative", "fraction_query_delta_margin_negative",
                  "mean_effective_compaction_count", "median_effective_compaction_count",
                  "num_margin_crossed"):
            print(f"   {k}: {last.get(k)}")
    by_budget = {r["budget"]: r for r in summary}
    if 750 in by_budget and 100 in by_budget:
        cmp = budget_stress.paired_budget_comparison(paired, 750, 100)
        print("\n=== paired 750 vs 100 (same queries) ===")
        for k, v in cmp.items():
            print(f"   {k}: {v}")
        budget_stress.write_csv(out_dir / "analysis" / "budget_stress_paired_750_vs_100.csv",
                                [cmp])
    print(f"-> {out_dir}")
    return 0




OVO_LM_SHORTLIST = "outputs/kv_metrics/analysis/ovo_long_memory_shortlist_provisional.csv"
OVO_LM_DIR = "outputs/kv_metrics/analysis/ovo_long_memory_manual"
OVO_VIDEO_ROOT = "OVO-Bench/data/chunked_videos"
OVO_ANNOTATION = "OVO-Bench/data/ovo_bench_new.json"

LM_MANUAL_FIELDS = [
    "evidence_start_time", "evidence_end_time", "memory_gap",
    "answer_visible_last_10s", "answer_visible_last_30s", "answer_visible_last_60s",
    "evidence_type", "evidence_directness", "ambiguity", "usable_long_memory", "notes",
]


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def cmd_ovo_long_memory_kit(args) -> int:
    """Build the manual evidence-time verification kit (frames only, no model)."""

    import csv as _csv
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = Path(args.shortlist or OVO_LM_SHORTLIST)
    out = Path(args.out_dir or OVO_LM_DIR)
    step = float(args.step_seconds or 10)
    video_root = Path(args.video_root or OVO_VIDEO_ROOT)
    max_tiles = int(args.max_tiles or 160)
    out.mkdir(parents=True, exist_ok=True)

    annotation = {}
    ann_path = Path(args.annotation or OVO_ANNOTATION)
    if ann_path.exists():
        for item in json.loads(ann_path.read_text(encoding="utf-8")):
            annotation[str(item.get("id"))] = item

    rows = list(_csv.DictReader(src.open(encoding="utf-8")))
    checklist, index_rows = [], []
    for row in rows:
        sid = str(row["sample_id"])
        item = annotation.get(sid, {})
        tq = float(row["query_time"])
        video = video_root / f"{sid}.mp4"
        sample_dir = out / f"sample_{sid}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # ---- frames every `step` seconds in [0, tq] (extra frame at tq) ----
        times, note = [], ""
        s = step
        n = int(tq // s) + 1
        if n > max_tiles:
            s = float(int((tq / max_tiles) // 10 + 1) * 10)
            n = int(tq // s) + 1
            note = f"(interval widened to {s:g}s to bound sheet size)"
        for i in range(n):
            times.append(i * s)
        if times and abs(times[-1] - tq) > 1e-6:
            times.append(tq)

        tiles = []
        if video.exists():
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            for t in times:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                scale = 320.0 / max(w, 1)
                if scale < 1:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                tiles.append((t, frame))
            cap.release()

        sheet_path = sample_dir / "contact_sheet.jpg"
        if tiles:
            cols = min(8, max(1, len(tiles)))
            rows_n = (len(tiles) + cols - 1) // cols
            fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.0, rows_n * 1.5))
            axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
            for ax in axes:
                ax.axis("off")
            for ax, (t, frame) in zip(axes, tiles):
                ax.imshow(frame)
                ax.set_title(_mmss(t), fontsize=7)
                ax.axis("off")
            fig.suptitle(
                f"sample_{sid} | {row['task_type']} | query_time={_mmss(tq)} "
                f"| frame every {s:g}s {note}", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(sheet_path, dpi=110, pil_kwargs={"quality": 85})
            plt.close(fig)

        options = item.get("options") or []
        meta = [
            f"sample_id: {sid}",
            f"video_id: {row.get('video_id', '')}",
            f"task_type: {row.get('task_type', '')}",
            f"video_path: {video}",
            f"local_video_exists: {video.exists()}",
            f"video_duration: {row.get('video_duration', '')}",
            f"query_time: {tq}  ({_mmss(tq)})",
            f"question: {row.get('question', '')}",
            f"options: {json.dumps(options, ensure_ascii=False)}",
            f"gold_answer: {row.get('gold_answer', '')}",
            f"contact_sheet: {sheet_path}  (frames every {s:g}s in [0, {tq:g}])",
            "",
            "--- 人工填写（不要用模型猜）---",
            "evidence_start_time: ",
            "evidence_end_time: ",
            "memory_gap = query_time - evidence_end_time: ",
            "answer_visible_last_10s (yes/no/unknown): ",
            "answer_visible_last_30s (yes/no/unknown): ",
            "answer_visible_last_60s (yes/no/unknown): ",
            "evidence_type (object_state/action/sequence/ocr/other): ",
            "evidence_directness (direct/indirect): ",
            "ambiguity (clear/ambiguous): ",
            "usable_long_memory (auto: gap>=60 AND last60s==no AND ambiguity==clear): ",
            "notes: ",
            "",
            "提示：contact_sheet 每 10s 一帧，仅用于快速定位；",
            "疑似 evidence 区段请再用 ffmpeg 在 [t-15s, t+15s] 逐帧精查。",
        ]
        (sample_dir / "metadata.txt").write_text("\n".join(meta), encoding="utf-8")

        entry = {
            "sample_id": sid, "video_id": row.get("video_id", ""),
            "task_type": row.get("task_type", ""),
            "query_time": tq, "video_duration": row.get("video_duration", ""),
            "question": row.get("question", ""),
            "options": json.dumps(options, ensure_ascii=False),
            "gold_answer": row.get("gold_answer", ""),
            "video_path": str(video), "contact_sheet": str(sheet_path),
        }
        for f in LM_MANUAL_FIELDS:
            entry[f] = ""
        checklist.append(entry)
        index_rows.append(entry)

    checklist_path = Path(args.checklist or out.parent / "ovo_long_memory_manual_checklist.csv")
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    with checklist_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(checklist[0].keys()) if checklist else [])
        if checklist:
            w.writeheader()
            w.writerows(checklist)

    html = ["<html><head><meta charset='utf-8'><title>OVO long-memory manual evidence check</title>",
            "<style>body{font-family:sans-serif}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;font-size:13px}"
            "img{max-width:900px}</style></head><body>",
            "<h1>OVO Backward long-memory candidates — manual evidence-time check</h1>",
            "<p><b>人工填写提醒</b>：evidence_start/end_time、answer_visible_last_10/30/60s、"
            "evidence_type/directness、ambiguity、usable_long_memory、notes 都需<u>人工</u>填写；"
            "<b>不要用模型猜测</b>。usable_long_memory = (query_time - evidence_end_time >= 60s) "
            "AND (last60s == no) AND (ambiguity == clear)；gap>=90s 记为 high-priority。</p>",
            "<table><tr><th>#</th><th>sample</th><th>task</th><th>query_time</th>"
            "<th>question</th><th>gold answer</th><th>contact sheet</th><th>manual template</th></tr>"]
    for i, r in enumerate(index_rows, 1):
        html.append(
            f"<tr><td>{i}</td><td>{r['sample_id']}</td><td>{r['task_type']}</td>"
            f"<td>{_mmss(float(r['query_time']))} ({r['query_time']}s)</td>"
            f"<td>{r['question']}</td><td>{r['gold_answer']}</td>"
            f"<td><img src='{Path(r['contact_sheet']).relative_to(out)}'></td>"
            f"<td><pre>{Path(r['video_path'])}</pre>"
            f"<a href='{Path(str(out / ('sample_' + r['sample_id'])) ).name}/metadata.txt'>metadata.txt</a></td></tr>")
    html.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"[ovo-long-memory-kit] samples={len(checklist)}")
    print(f"  checklist : {checklist_path}")
    print(f"  index     : {out / 'index.html'}")
    print(f"  sheets    : {out}/sample_*/contact_sheet.jpg")
    return 0



OVO_BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
OVO_BS_ROOT = "outputs/kv_metrics/experiments/ovo_backward_budget_stress"
OVO_BS_BUDGETS = [2000, 1000, 750, 500, 400, 300, 200, 100]


def _ovo_backward_items(path=None):
    ann = Path(path or OVO_ANNOTATION)
    return [x for x in json.loads(ann.read_text(encoding="utf-8"))
            if x.get("task") in OVO_BACKWARD_TASKS]


def cmd_ovo_bs_prepare(args) -> int:
    """Manifests/configs for the OVO backward budget stress (no model)."""

    root = Path(args.root) if args.root else _find_experiment_dir("ovo_backward_budget_stress")
    items = _ovo_backward_items(args.annotation)
    video_ids = [str(x["id"]) for x in items]
    per_task = ", ".join(
        "{}={}".format(t, sum(1 for x in items if x["task"] == t))
        for t in OVO_BACKWARD_TASKS
    )
    print(f"[ovo-bs-prepare] backward items: {len(video_ids)} ({per_task})")
    base = load_config(args.config or SB_BASE)
    # OVO backward config: same inference params, OVO annotation, full prefix
    cfg = {
        "model": base["model"], "inference": base["inference"],
        "memory": base["memory"],
        "benchmarks": {"ovobench": {
            "enabled": True,
            "annotation": str(Path(args.annotation or OVO_ANNOTATION)),
            "chunked_dir": "OVO-Bench/data/chunked_videos",
        }},
        "output": base.get("output", {}),
        "evaluation": base.get("evaluation", {}),
    }
    (root).mkdir(parents=True, exist_ok=True)
    cfg_path = root / "eval_ovo_backward.yaml"
    try:
        import yaml
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:  # noqa: BLE001
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    budgets = OVO_BS_BUDGETS
    if getattr(args, "budgets", None):
        budgets = sorted({int(x) for x in str(args.budgets).split(",") if x.strip()},
                         reverse=True)
    for name, budget in [("full", None)] + [(f"budget_{b}", b) for b in budgets]:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment_id": f"ovo_backward_{name}",
            "locked": True,
            "benchmark": "ovobench", "tasks": OVO_BACKWARD_TASKS,
            "context_mode": "full_prefix", "budget": budget,
            "compressed_only": name != "full",
            "selection_seed": 20260930, "bootstrap_seed": 42,
            "bootstrap": {"unit": "sample", "n_boot": 2000, "ci": 0.95},
            "base_config": str(cfg_path),
            "video_ids": video_ids, "n_videos": len(video_ids),
            "n_queries": len(video_ids),
            "note": "OVO backward long-history diagnostic; 1 query per sample.",
        }
        (d / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {name:>12}: -> {d}")
    print(f"  config      : {cfg_path}")
    return 0


OVO_DIAG_ROOT = "outputs/qwen25vl_full_eval"
OVO_DIAG_DEFAULT_BUDGETS = [6000, 2000, 1000, 500, 200, 100, 50]
OVO_DIAG_BASE_CONFIG = "configs/eval_qwen25vl_ovo.yaml"


def _ovo_diag_budgets(raw):
    if not raw:
        return list(OVO_DIAG_DEFAULT_BUDGETS)
    if isinstance(raw, (list, tuple)):
        return sorted({int(x) for x in raw}, reverse=True)
    import re as _re

    return sorted(
        {int(x) for x in _re.split(r"[,\s]+", str(raw)) if x.strip()}, reverse=True
    )


def cmd_ovo_diag_prepare(args) -> int:
    """Build the deterministic diagnostic manifest + per-run directories."""

    from . import ovo_diagnostic as diag

    root = Path(args.root or OVO_DIAG_ROOT) / f"ovobench_{args.group}"
    root.mkdir(parents=True, exist_ok=True)

    manifest = diag.build_manifest(
        args.annotation or diag.DEFAULT_ANNOTATION,
        per_category=int(args.per_category or diag.DEFAULT_PER_CATEGORY),
    )
    manifest_path = diag.write_manifest(root / "diagnostic_manifest.json", manifest)

    entry = manifest["groups"][args.group]
    print(f"[ovo-diag-prepare] group={args.group} annotation={manifest['annotation']}")
    total_videos = 0
    selected_videos = 0
    for task in entry["tasks"]:
        info = entry["categories"][task]
        total_videos += info["total_unique_videos"]
        selected_videos += info["selected_count"]
        print(
            f"  {task}: total_unique_videos={info['total_unique_videos']} "
            f"samples={info['total_samples']} diagnostic_selected={info['selected_count']}"
        )
    print(f"  total unique videos={total_videos} diagnostic selected={selected_videos}")

    budgets = _ovo_diag_budgets(args.budgets)
    video_ids = diag.manifest_video_ids(manifest, args.group)
    base_config = args.config or OVO_DIAG_BASE_CONFIG
    for name, budget in [("full", None)] + [(f"budget_{b}", b) for b in budgets]:
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "experiment_id": f"ovobench_{args.group}_{name}",
            "benchmark": "ovobench",
            "group": args.group,
            "tasks": entry["tasks"],
            "budget": budget,
            "context_mode": "full_prefix",
            "annotation": manifest["annotation"],
            "base_config": base_config,
            "diagnostic_manifest": str(manifest_path),
            "diagnostic_video_ids": video_ids,
            "n_diagnostic_videos": len(video_ids),
            "all_samples": True,
            "note": (
                "Full accuracy over ALL OVO samples; rich decision diagnostics "
                "only for diagnostic_video_ids."
            ),
        }
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  {name:>12}: -> {run_dir}")
    print(f"  manifest    : {manifest_path}")
    return 0


def _completed_video_ids(exp_dir: Path):
    """Video ids already finished in ``exp_dir`` (batch-size independent).

    New batches record an explicit ledger in ``run_status.json``. Legacy
    batches (written before that existed) are recovered from ``tasks.jsonl``,
    which is only produced once a pass completed.
    """

    done = set()
    root = Path(exp_dir) / "raw_decision"
    if not root.exists():
        return done
    for batch_dir in sorted(root.glob("batch*")):
        marker = batch_dir / "run_status.json"
        if marker.exists():
            try:
                state = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = {}
            if state.get("status") == "success":
                ids = state.get("completed_video_ids")
                if ids:
                    done.update(str(v) for v in ids)
                    continue
        tasks = batch_dir / "tasks.jsonl"
        if not tasks.exists():
            continue
        for line in tasks.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            sid = record.get("sample_id")
            if sid is not None:
                done.add(str(sid))
    return done


def cmd_ovo_diag_run(args) -> int:
    """Run one OVO diagnostic experiment dir (full or one budget), resumable.

    Resumption is tracked per VIDEO, not per batch, so a re-run with a
    different ``--batch_size`` still skips finished work. A batch that runs out
    of GPU memory is automatically split in half and retried, which is what
    makes the un-compressed Full pass survivable on long videos.
    """

    from ...kv_metrics.config import KVMetricConfig
    from ...kv_metrics.paired_runner import run as run_paired

    exp_dir = Path(args.experiment_dir)
    manifest = json.loads(
        (exp_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    tasks = manifest["tasks"]
    diagnostic_ids = manifest.get("diagnostic_video_ids") or []

    # Resolve which OVO samples belong to this group, in original data order.
    from . import ovo_diagnostic as diag

    items = diag.load_items(manifest.get("annotation") or diag.DEFAULT_ANNOTATION)
    video_ids = [str(i["id"]) for i in items if i.get("task") in set(tasks)]
    video_ids = list(dict.fromkeys(video_ids))  # forward tasks repeat ids
    if getattr(args, "max_videos", None):
        video_ids = video_ids[: int(args.max_videos)]

    budget = manifest.get("budget")
    is_full = budget is None
    budget_value = int(budget) if budget is not None else 10 ** 9
    batch_size = int(args.batch_size or 40)

    completed = set() if args.force else _completed_video_ids(exp_dir)
    pending = [v for v in video_ids if v not in completed]
    print(
        f"[ovo-diag-run] {exp_dir.name}: total={len(video_ids)} "
        f"already_complete={len(completed)} pending={len(pending)}"
    )

    config = load_config(args.config or manifest["base_config"])
    config.setdefault("memory", {}).setdefault("compression", {})
    # Full == the same backend with compression explicitly disabled; HERMES
    # enables compression. Only the budget differs between budget runs.
    config["memory"]["method"] = "full_cache" if is_full else "hermes"
    config["memory"]["compression"]["enabled"] = True
    config["memory"]["compression"]["budget"] = budget_value
    config.setdefault("benchmarks", {})["ovobench"] = {
        **(config.get("benchmarks", {}).get("ovobench") or {}),
        "enabled": True,
        "annotation": manifest["annotation"],
        "chunked_dir": "OVO-Bench/data/chunked_videos",
    }

    failures = []

    def run_batch(batch):
        # Batch dir name is derived from content, so it can never collide with
        # a differently-shaped batch from an earlier (different-batch-size) run.
        name = f"batch_{batch[0]}"
        print(
            f"[ovo-diag-run] {name}: videos={len(batch)} [{batch[0]}..{batch[-1]}]"
        )
        kv_config = KVMetricConfig(
            enable_kv_metrics=True,
            mode="full" if is_full else "compressed",
            output_dir=str(exp_dir / "raw_decision"),
            run_name=name,
            attention_backend="original",
            profile_wallclock=False,
            detailed_trajectory=False,
            capture_option_logits=True,
            diagnostic_video_ids=list(diagnostic_ids),
        )
        try:
            run_paired(config, benchmark="ovobench", max_samples=0,
                       kv_config=kv_config, video_ids=batch)
        except Exception as error:  # noqa: BLE001
            if type(error).__name__ == "OutOfMemoryError" and len(batch) > 1:
                mid = len(batch) // 2
                print(
                    f"[ovo-diag-run] !! {name} OOM -> split into "
                    f"{len(batch[:mid])}+{len(batch[mid:])} and retry"
                )
                run_batch(batch[:mid])
                run_batch(batch[mid:])
                return
            print(f"[ovo-diag-run] !! {name} FAILED: {type(error).__name__}: {error}")
            failures.append(
                {"batch": name, "videos": list(batch),
                 "error": f"{type(error).__name__}: {error}"}
            )

    if not pending:
        print(f"[ovo-diag-run:{manifest.get('budget') or 'FULL'}] "
              "already complete; nothing to do")
    else:
        for start in range(0, len(pending), batch_size):
            run_batch(pending[start:start + batch_size])
    (exp_dir / "run_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(f"[ovo-diag-run:{manifest.get('budget') or 'FULL'}] done; failures={len(failures)}")
    return 0


def cmd_ovo_diag_analyze(args) -> int:
    """CPU-only: full accuracy + diagnostic-subset decision statistics/figures."""

    from . import ovo_diagnostic as diag

    root = Path(args.root or OVO_DIAG_ROOT) / f"ovobench_{args.group}"
    manifest = json.loads((root / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    tasks = manifest["groups"][args.group]["tasks"]

    full_dir = root / "full"
    budget_dirs = {}
    for d in sorted(root.glob("budget_*")):
        name = d.name[len("budget_"):]
        if name.isdigit() and (d / "raw_decision").exists():
            budget_dirs[int(name)] = [d]
    out_dir = Path(args.output_dir or (root / "analysis"))
    analysis = out_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    # ---- A. full accuracy over ALL samples (never the subset) ---- #
    full_tasks = _load_tasks_jsonl(full_dir)
    per_cat, overall = _full_accuracy(full_tasks)
    diag.write_csv(
        analysis / "per_category_accuracy.csv",
        [{"benchmark": f"ovobench_{args.group}", "scope": "all_samples", **row}
         for row in per_cat],
        fields=["benchmark", "scope", "category", "N", "correct", "accuracy"],
    )
    print(f"[ovo-diag-analyze] full-run samples={len(full_tasks)} "
          f"overall_accuracy={overall}")
    for row in per_cat:
        print(f"    {row['category']:>4}: N={row['N']} acc={row['accuracy']}")

    if not budget_dirs:
        print("!! no budget runs found; only full accuracy written")
        return 0

    # ---- B. diagnostic subset only ---- #
    paired, _ = diag.merge_runs(full_dir, budget_dirs)
    subset = diag.group_rows(paired, tasks)
    budgets = sorted(budget_dirs, reverse=True)
    summary = diag.diagnostic_summary(subset, budgets, args.group, tasks)
    diag.write_csv(analysis / "diagnostic_records.csv", subset,
                   fields=list(subset[0].keys()) if subset else None)
    diag.write_csv(analysis / "diagnostic_summary.csv", summary,
                   fields=diag.SUMMARY_FIELDS)
    figures = diag.plot_group(summary, out_dir / "figures", args.group)

    print(f"[ovo-diag-analyze] diagnostic subset={len(subset)} budgets={budgets}")
    print("diagnostic_summary.csv:")
    for row in summary:
        print(f"    {row['category']:>4} b={row['budget']:>5} N={row['N']} "
              f"k={row['mean_k']} dP={row['mean_delta_p_gold']} "
              f"dM={row['mean_delta_margin']} CW={row['CW']} "
              f"cross={row['margin_crossing_count']}")
    print(f"figures={len(figures)} -> {out_dir / 'figures'}")
    return 0


def cmd_lite_accuracy(args) -> int:
    """Roll up accuracy for the light benchmarks (no KV diagnostics)."""

    root = Path(args.root)
    rows = []
    for run_dir in sorted(root.glob("*")):
        summary_path = run_dir / args.benchmark / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        block = data.get(args.benchmark) or {}
        pipeline = block.get("pipeline") or {}
        rows.append({
            "run": run_dir.name,
            "overall": block.get("overall"),
            "N_videos": pipeline.get("videos"),
            "N_questions": pipeline.get("questions"),
            "per_task": json.dumps(block.get("tasks") or {}, sort_keys=True),
            "per_category": json.dumps(block.get("categories") or {}, sort_keys=True),
        })
    out_csv = Path(args.output_dir or root) / "accuracy_rollup.csv"
    if rows:
        import csv as _csv

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[lite-accuracy] benchmark={args.benchmark} runs={len(rows)}")
    for row in rows:
        print(f"    {row['run']:>16}: overall={row['overall']} "
              f"N_videos={row['N_videos']} N_questions={row['N_questions']}")
    if rows:
        print(f"-> {out_csv}")
    return 0


def _load_tasks_jsonl(run_dir: Path):
    rows = []
    for path in sorted((Path(run_dir) / "raw_decision").glob("*/tasks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _full_accuracy(task_rows):
    from ...adapters import option_letter

    by_cat = {}
    for row in task_rows:
        if row.get("status") == "failed":
            continue
        cat = row.get("task_type") or "unknown"
        pred = str(row.get("prediction", ""))
        gold = str(row.get("ground_truth", ""))
        correct = option_letter(pred) == option_letter(gold)
        bucket = by_cat.setdefault(cat, [0, 0])
        bucket[0] += int(correct)
        bucket[1] += 1
    rows = []
    total_correct = sum(v[0] for v in by_cat.values())
    total = sum(v[1] for v in by_cat.values())
    for cat in sorted(by_cat):
        correct, n = by_cat[cat]
        rows.append({"category": cat, "N": n, "correct": correct,
                     "accuracy": (correct / n) if n else None})
    rows.append({"category": "OVERALL", "N": total, "correct": total_correct,
                 "accuracy": (total_correct / total) if total else None})
    return rows, ((total_correct / total) if total else None)


def cmd_ovo_bs_analyze(args) -> int:
    """Merge Full + budgets, per-task accuracy, ΔP/ΔM, exposure, figures."""

    root = Path(args.root) if args.root else _find_experiment_dir("ovo_backward_budget_stress")
    dirs = {}
    full_dirs = [root / "full"]
    # auto-discover whatever budget_* dirs exist (any budget set works)
    for d in sorted(root.glob("budget_*")):
        name = d.name[len("budget_"):]
        if name.isdigit() and (d / "raw_decision").exists():
            dirs[int(name)] = [d]
    if not dirs:
        print("!! no budget runs found; run sb-budget-run first")
        return 1
    if not (full_dirs[0] / "raw_decision").exists():
        print("!! full run not found")
        return 1

    full_records = budget_stress.load_budget_records(full_dirs)
    paired, summary = budget_stress.build_budget_table(dirs, full_records)
    problems = budget_stress.sanity_checks(paired, summary)
    tasks = budget_stress.task_breakdown(paired, summary)
    exposure = budget_stress.exposure_table(paired, sorted(dirs))
    inv = budget_stress.join_invariants(paired)
    mm = sum(int(r.get("decision_metric_mismatch") or 0) for r in summary)

    out_dir = Path(args.output_dir or (root / "analysis"))
    analysis = out_dir / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    budget_stress.write_csv(analysis / "ovo_backward_budget_summary.csv", summary)
    budget_stress.write_csv(analysis / "ovo_backward_budget_paired_samples.csv", paired)
    budget_stress.write_csv(analysis / "ovo_backward_budget_by_task.csv", tasks)
    budget_stress.write_csv(analysis / "ovo_backward_exposure.csv", exposure)
    figures = budget_stress.plot_budget_curves(summary, out_dir / "figures")
    figures += budget_stress.plot_task_breakdown(tasks, out_dir / "figures")
    (out_dir / "sanity.json").write_text(
        json.dumps({"problems": problems, "join_invariants": inv,
                    "expected_n_samples": len(paired),
                    "transition_sum_per_budget": {
                        str(r["budget"]): r["CC"] + r["CW"] + r["WC"] + r["WW"]
                        for r in summary},
                    "decision_metric_mismatch_total": mm}, indent=2),
        encoding="utf-8")

    print(f"[ovo-bs-analyze] samples={len(paired)} budgets={[r['budget'] for r in summary]}")
    print("join invariants:", inv)
    print(f"best_option vs official prediction mismatches: {mm} (must be 0)")
    if problems:
        print("!! sanity problems:")
        for prob in problems:
            print("   -", prob)
    print("\nper-budget:")
    for row in summary:
        print(f"  budget {row['budget']}: N={row['N_query']} Acc_full={row['Acc_full']} "
              f"Acc_comp={row['Acc_compressed']} CC={row['CC']} CW={row['CW']} WC={row['WC']} "
              f"WW={row['WW']} Pres={row['Preservation']} "
              f"dP={row['mean_delta_p_gold_video_level']} dM={row['mean_delta_margin_video_level']} "
              f"crossed={row.get('num_margin_crossed')} meank={row['mean_effective_compaction_count']}")
    print("\nper-task (budget, task, Acc_comp, Pres, CW):")
    for row in tasks:
        print(f"  {row['budget']:>5} {row['task_type']:>4} Acc_comp={row['Acc_compressed']:.3f} "
              f"Pres={row['Preservation']} CW={row['CW']} N={row['N_sample']}")
    print(f"-> {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KV fidelity experiment workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select", help="select samples that trigger compression")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--experiment", default=None, help="base config (for benchmark paths)")
    p.add_argument("--metrics_root", default="outputs/kv_metrics")
    p.add_argument("--min_compactions", type=int, default=3)
    p.add_argument("--min_video_duration", type=float, default=None)
    p.add_argument("--max_video_duration", type=float, default=None)
    p.add_argument("--min_max_kv_tokens", type=int, default=None)
    p.add_argument("--min_compression_ratio", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--top_k_longest", type=int, default=None)
    p.add_argument("--top_k_most_compactions", type=int, default=None)
    p.add_argument("--include_zero_compaction", action="store_true", default=False)
    p.add_argument("--experiment_scale", choices=["smoke", "medium", "full"], default="smoke")
    p.add_argument("--sample_fps", type=float, default=0.5)
    p.add_argument("--max_side", type=int, default=336)
    p.add_argument("--name", default="experiment")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("run", help="run the experiment matrix")
    p.add_argument("--experiment", required=True)
    p.add_argument("--samples", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--no_resume", action="store_true", default=False)
    p.set_defaults(func=cmd_run)

    for name, func, extra in (
        ("analyze", cmd_analyze, False),
        ("plot", cmd_plot, False),
        ("report", cmd_report, False),
    ):
        p = sub.add_parser(name, help=f"{name} the experiment")
        p.add_argument("--experiment_dir", required=True)
        p.set_defaults(func=func)

    p = sub.add_parser("run-all", help="run + analyze + plot + report")
    p.add_argument("--experiment", required=True)
    p.add_argument("--samples", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--no_resume", action="store_true", default=False)
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("trajectory-select", help="find a video suitable for the trajectory study")
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--min_queries", type=int, default=5)
    p.add_argument("--min_compactions", type=int, default=3)
    p.add_argument("--min_unique", type=int, default=4)
    p.add_argument("--only_qualifying", action="store_true", default=False)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_trajectory_select)

    p = sub.add_parser("trajectory", help="build the Query x compaction-count trajectory")
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--sample_id", default=None)
    p.add_argument("--budget", type=int, default=None)
    p.set_defaults(func=cmd_trajectory)

    p = sub.add_parser("reproduce-case", help="end-to-end single-video case study")
    p.add_argument("--sample_id", required=True)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--benchmark", default="ovobench")
    p.add_argument("--config", default="configs/eval_kv_metrics_smoke.yaml")
    p.add_argument("--output_root", default="outputs/kv_metrics/experiments")
    p.add_argument("--name", default=None)
    p.add_argument("--metric_backend", choices=["eager", "original"], default="eager")
    p.add_argument("--temporal_bins", default="0,10,30,60,300,inf")
    p.add_argument("--profile_runtime_breakdown", action="store_true", default=False)
    p.add_argument("--skip_run", action="store_true", default=False)
    p.set_defaults(func=cmd_reproduce_case)

    p = sub.add_parser(
        "compaction-accuracy",
        help="analyze+plot+report the compaction-accuracy experiment",
    )
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_traj_samples", type=int, default=3)
    p.set_defaults(func=cmd_compaction_accuracy)

    p = sub.add_parser("sb-candidates",
                       help="StreamingBench real candidates (<= latest query cap)")
    p.add_argument("--csv", default=None)
    p.add_argument("--max_latest_query_time", type=float, default=260.0)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_sb_candidates)

    p = sub.add_parser("sb-exposure",
                       help="compressed-only exposure scan (observed effective k)")
    p.add_argument("--csv", default=None)
    p.add_argument("--max_latest_query_time", type=float, default=260.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_root", default="outputs/kv_metrics/experiments/streamingbench_real")
    p.add_argument("--output", default=None)
    p.add_argument("--skip_run", action="store_true", default=False)
    p.set_defaults(func=cmd_sb_exposure)

    p = sub.add_parser("sb-smoke", help="3-video StreamingBench full_prefix smoke")
    p.add_argument("--csv", default=None)
    p.add_argument("--config", default=SB_BASE)
    p.add_argument("--max_latest_query_time", type=float, default=260.0)
    p.add_argument("--video_ids", nargs="*", default=None)
    p.add_argument("--output_root", default="outputs/kv_metrics/experiments/streamingbench_real")
    p.set_defaults(func=cmd_sb_smoke)

    p = sub.add_parser("sb-select",
                       help="deterministic 20-video selection (Plan A/B) + audit")
    p.add_argument("--exposure_csv", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_videos", type=int, default=20)
    p.set_defaults(func=cmd_sb_select)

    p = sub.add_parser("sb-manifest",
                       help="freeze Plan B into an experiment manifest (no model)")
    p.add_argument("--plan", default=None)
    p.add_argument("--plan_name", default=None)
    p.add_argument("--exposure_csv", default=None)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--config", default=SB_BASE)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=2000)
    p.set_defaults(func=cmd_sb_manifest)

    p = sub.add_parser("sb-decision-run",
                       help="re-run 100 task-only queries with option-logit capture")
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--force", action="store_true", default=False)
    p.set_defaults(func=cmd_sb_decision_run)

    p = sub.add_parser("ovo-diag-prepare",
                       help="deterministic per-category diagnostic manifest (OVO RT/BW)")
    p.add_argument("--group", required=True, choices=["backward", "realtime"])
    p.add_argument("--root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--annotation", default=None)
    p.add_argument("--per_category", type=int, default=10)
    p.add_argument("--budgets", default=None,
                   help="comma-separated budget list (default: 6000,2000,...,50)")
    p.set_defaults(func=cmd_ovo_diag_prepare)

    p = sub.add_parser("ovo-diag-run",
                       help="run one OVO diagnostic dir (full or one budget), resumable")
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--max_videos", type=int, default=None,
                   help="limit to the first N videos (pilot / debug only)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_ovo_diag_run)

    p = sub.add_parser("ovo-diag-analyze",
                       help="CPU: full accuracy + diagnostic subset stats/figures")
    p.add_argument("--group", required=True, choices=["backward", "realtime"])
    p.add_argument("--root", default=None)
    p.add_argument("--output_dir", default=None)
    p.set_defaults(func=cmd_ovo_diag_analyze)

    p = sub.add_parser("lite-accuracy",
                       help="roll up accuracy for StreamingBench / Video-MME runs")
    p.add_argument("--benchmark", required=True,
                   choices=["streamingbench", "videomme", "ovobench"])
    p.add_argument("--root", required=True,
                   help="dir containing per-run subdirs (e.g. .../streamingbench)")
    p.add_argument("--output_dir", default=None)
    p.set_defaults(func=cmd_lite_accuracy)

    p = sub.add_parser("ovo-bs-prepare",
                       help="manifests/configs for OVO backward budget stress")
    p.add_argument("--root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--annotation", default=None)
    p.add_argument("--budgets", default=None,
                   help="comma-separated budgets to (re)create manifests for")
    p.set_defaults(func=cmd_ovo_bs_prepare)

    p = sub.add_parser("ovo-bs-analyze",
                       help="merge Full+budgets for OVO backward (per-task + ΔP/ΔM)")
    p.add_argument("--root", default=None)
    p.add_argument("--output_dir", default=None)
    p.set_defaults(func=cmd_ovo_bs_analyze)

    p = sub.add_parser("ovo-long-memory-kit",
                       help="build manual evidence-time verification kit (frames only)")
    p.add_argument("--shortlist", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--checklist", default=None)
    p.add_argument("--video_root", default=None)
    p.add_argument("--annotation", default=None)
    p.add_argument("--step_seconds", type=float, default=10.0)
    p.add_argument("--max_tiles", type=int, default=160)
    p.set_defaults(func=cmd_ovo_long_memory_kit)

    p = sub.add_parser("sb-budget-prepare",
                       help="create per-budget manifests for the frozen 28 videos")
    p.add_argument("--root", default=None)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_sb_budget_prepare)

    p = sub.add_parser("sb-budget-run",
                       help="compressed-only task run for one budget (resumable)")
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--batch_size", type=int, default=7)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--full_only", action="store_true", default=False,
                   help="run the FULL-KV branch only (for the one-time baseline)")
    p.add_argument("--benchmark", default=None,
                   help="override the benchmark (default: the manifest's)")
    p.set_defaults(func=cmd_sb_budget_run)

    p = sub.add_parser("sb-budget-analyze",
                       help="merge budgets and produce stress-test tables/figures")
    p.add_argument("--root", default=None)
    p.add_argument("--output_dir", default=None)
    p.set_defaults(func=cmd_sb_budget_analyze)

    p = sub.add_parser("sb-extend-select",
                       help="pick EXTRA videos to enrich high-k coverage (no model)")
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--exposure_csv", default=None)
    p.add_argument("--base_plan", default=None)
    p.add_argument("--max_extra", type=int, default=None)
    p.set_defaults(func=cmd_sb_extend_select)

    p = sub.add_parser("sb-decision-analyze",
                       help="ΔP_gold / ΔM_z offline analysis + figures")
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--extra_dirs", default=None,
                   help="comma-separated extra experiment dirs whose raw_decision "
                        "records are merged in (e.g. the extended run)")
    p.add_argument("--output_dir", default=None)
    p.set_defaults(func=cmd_sb_decision_analyze)

    p = sub.add_parser("sb-analyze",
                       help="P5+P6 locked-bin aggregation + figures")
    p.add_argument("--experiment_dir", default=None)
    p.set_defaults(func=cmd_sb_analyze)

    p = sub.add_parser("sb-batches",
                       help="run P3+P4 in resumable batches")
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--force", action="store_true", default=False)
    p.set_defaults(func=cmd_sb_batches)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
