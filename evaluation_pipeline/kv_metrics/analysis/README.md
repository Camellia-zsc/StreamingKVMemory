# KV fidelity experiment + analysis layer

Multi-sample experiments, statistical comparison and auto-generated figures
built **on top of** the verified `evaluation_pipeline/kv_metrics` sidecars.
It never recomputes RAM/AORE/JS/KL and never edits the model / benchmark code.

## Workflow

```bash
# 1. pick samples that actually trigger compression
python -m evaluation_pipeline.kv_metrics.analysis.cli select \
    --benchmark ovobench --metrics_root outputs/kv_metrics \
    --min_compactions 3 --num_samples 20 --experiment_scale medium

# 2. run the experiment matrix (budgets) with resume
python -m evaluation_pipeline.kv_metrics.analysis.cli run \
    --experiment configs/kv_metrics_experiment.yaml

# 3. tables + statistics.json
python -m evaluation_pipeline.kv_metrics.analysis.cli analyze \
    --experiment_dir outputs/kv_metrics/experiments/<name>

# 4. figures (PNG + PDF)
python -m evaluation_pipeline.kv_metrics.analysis.cli plot \
    --experiment_dir outputs/kv_metrics/experiments/<name>

# 5. experiment_report.md
python -m evaluation_pipeline.kv_metrics.analysis.cli report \
    --experiment_dir outputs/kv_metrics/experiments/<name>

# or all of the above in one shot
python -m evaluation_pipeline.kv_metrics.analysis.cli run-all \
    --experiment configs/kv_metrics_experiment.yaml
```

### Single-video Query x compaction-count trajectory (P0)

```bash
# 1. find a video with enough queries and compaction variation
python -m evaluation_pipeline.kv_metrics.analysis.cli trajectory-select \
    --experiment_dir outputs/kv_metrics/experiments/stage1_ovo_smoke \
    --min_queries 5 --min_compactions 3
# -> trajectory_candidates.csv + recommended_sample_id=<id>

# 2. build the trajectory table + the 4-panel figure + the timeline
python -m evaluation_pipeline.kv_metrics.analysis.cli trajectory \
    --experiment_dir outputs/kv_metrics/experiments/stage1_ovo_smoke \
    --sample_id 1516 --budget 2000

# 3. one command for the whole paper-style case study
python -m evaluation_pipeline.kv_metrics.analysis.cli reproduce-case \
    --sample_id 1516 --budget 2000 --benchmark ovobench \
    --profile_runtime_breakdown
```

`compaction_count_at_query` counts compactions with
`end_time <= query_anchor_time` (a compaction that was triggered but not yet
completed does not count). Every point is a real query; multiple queries
sharing a compaction count are all kept.

> **Interpretation**: this is *cumulative operational fidelity* -- "when this
> query arrived, the cache had already undergone n compactions". It is NOT a
> per-compaction causal effect, because the query text and the video content
> also change between points.

### Runtime breakdown (P2)

```bash
python -m evaluation_pipeline.kv_metrics.paired_runner \
    --config configs/eval_kv_metrics_smoke.yaml \
    --benchmark ovobench --sample_id 1516 \
    --kv_metrics_mode paired --kv_metrics_profile_wallclock \
    --profile_runtime_breakdown
```

Writes `tables/runtime_breakdown.csv` + `runtime_summary.json`. Three different
real-time numbers are kept separate:

| name | definition |
| --- | --- |
| `rho_prune_realtime` | `T_prune / T_trigger_interval` |
| `rho_hermes_realtime` | `T_HERMES_total / T_trigger_interval` |
| `hermes_compute_fraction` | `T_HERMES_total / T_measured_model_compute_total` |

`T_measured_model_compute_total = vision_encoder + streaming_prefill +`
`hermes_total + query_prefill + decode` (HERMES sub-components are **not**
added again). `stream_time_overhead_ratio = T_component / video_duration` is a
different, separate quantity.

> `--profile_runtime_breakdown` wraps every measured call in
> `cuda.synchronize`, so its wall-clock and rho values are **upper bounds**,
> not deployment numbers. Deployment rho is measured without per-call syncs.

## Statistical contract

* Unit of inference is the **sample**, never the event: events are first
  averaged per sample, then samples are bootstrapped (`n_boot=2000`,
  `ci=0.95`, `seed=42`).
* PRE/POST uses a **paired** bootstrap; Wilcoxon signed-rank only runs with
  `>= 5` valid paired samples.
* Empty/NaN bins are excluded and reported with their `N`, never silently
  zeroed.

## Experiment definitions

| # | experiment | x | y | unit |
| --- | --- | --- | --- | --- |
| 1 | compression vs fidelity | sample-mean compression ratio | RAM / AORE / JS | sample |
| 2 | PRE/POST first compaction | before/after 1st compaction | KL/JS/AORE/RAM | sample (paired) |
| 3 | repeated compaction drift | compaction-count bin | RAM/AORE/JS | sample |
| 4 | temporal RAM | KV age bucket | RAM | sample |
| 5 | age vs retention | KV age bucket | retention probability | token |
| 6 | system efficiency | compression ratio | wall-clock / rho | sample |
| 7 | fidelity vs task | metric quintile | task accuracy | event |
| 8 | case timelines | stream time | KV size + metrics | event |

## Limitations

* **PRE/POST (experiment 2)** requires queries on both sides of the first
  compaction within one video. OVO timestamp tasks have a single query and the
  forward tasks (REC/SSR/CRR) ignore `realtime` and always stream to the end of
  the clip, so Stage 1 has `N=0`. A real PRE/POST needs **extra probe
  anchors**, which are intentionally not fabricated here.
* **Temporal AORE / JS** are unsupported (no evidence timestamps) and are never
  plotted.
* **Experiment 7** needs `task_correct`, which is only available when the
  runner persists adapter predictions (added; Stage-1 sidecars predate it).
* Retention-by-age (experiment 5) is only emitted by runs produced after the
  retention instrumentation was added; older sidecars are skipped, not faked.
