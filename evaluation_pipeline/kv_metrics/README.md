# KV Cache Compression Fidelity Evaluation

Sidecar evaluation for **MMDuet2 + HERMES** streaming video inference.

It measures how much fidelity the HERMES KV compression preserves, **without
touching any existing inference / benchmark code**. Every new artefact lives
under `evaluation_pipeline/kv_metrics/` and `outputs/kv_metrics/<run_name>/`.

## Design (non-invasive)

Nothing under `OVO-Bench/` or `HERMES/` is modified. Everything is installed by
wrapping bound methods at runtime (`hooks.py`) plus one torch forward hook on
`qa_model.lm_head`-free path (we instead reproduce the anchor eagerly).

Hook points:

| what | where | why |
| --- | --- | --- |
| `EvalMMDuet2Hermes._reset_stream` | wrapper | reset provenance, register the prefix |
| module `iter_chunks_until_query` | wrapper | attach per-frame stream timestamps |
| `qa_model.encode_image_chunk`/`encode_video_chunk` | wrapper | register visual token provenance |
| `qa_model.apply_kv_cache_pruning_strict` | wrapper | **the real compaction**: time it, mirror `keep_indices` on UIDs, measure bytes |
| `qa_model.question_answering` | wrapper | snapshot pre-query cache, reproduce the anchor eagerly |

## Two-pass evaluation

```
Pass A (compressed) : normal MMDuet2 + HERMES
Pass B (full)       : SAME runtime, SAME weights, SAME frames/query,
                      compaction disabled by setting kv_size ~ 1e12
```

Both passes use identical weights, frame schedule, query schedule,
preprocessing, `do_sample=False` (temperature=0) and seeds. Events are paired
by a stable `event_id = "{video_stem}|{sample_id}:{test_index}|{query_time}"`.

## Metrics

1. **RAM** – Retained Attention Mass, region-normalised:
   `RAM = Σ_{i∈S_keep∩region} A_full[i] / Σ_{i∈region} A_full[i]`.
   `region` = the HERMES-managed visual KV region, so transient text KV never
   enters the RAM denominator (`scope=hermes_managed`). The unnormalised value
   is reported as `ram_per_layer[layer]["raw_mean"]`.
2. **AORE** – Attention Output Relative Error:
   `||O_comp - O_full||_F / (||O_full||_F + eps)`, computed in fp32 on the
   per-head **pre-o_proj** attention context of the video region.
   `aore_level = "per_head_pre_o_proj"`.
3. **JS / KL** – from the first-decode-step logits at the anchor.
4. **Temporal RAM** – bucket-conditioned (`0-10,10-30,30-60,60-300,300+`).
   `AORE`/`JS` temporal binning is gated behind `get_evidence_timestamp`; the
   current benchmarks expose no evidence timestamp, so
   `temporal_aore_supported=false`, `temporal_js_supported=false`.
5. **Compression ratio / wall-clock / ρ_realtime** from real persistent KV
   bytes and synchronised compaction timing.
6. **Task performance** reused from the existing evaluators (never recomputed).

## Deliverables (answers A–M)

**A. Files added**

* `evaluation_pipeline/kv_metrics/config.py` – `KVMetricConfig` + CLI.
* `evaluation_pipeline/kv_metrics/provenance.py` – `KVTokenUID`,
  `KVProvenanceTracker`.
* `evaluation_pipeline/kv_metrics/hooks.py` – `KVInstrumentation`,
  `AnchorArtifact` (all method wrapping).
* `evaluation_pipeline/kv_metrics/fidelity.py` – RAM / AORE / JS / KL math and
  `capture_anchor_attention`.
* `evaluation_pipeline/kv_metrics/temporal.py` – Temporal RAM + binning.
* `evaluation_pipeline/kv_metrics/system_metrics.py` – `tensor_bytes`,
  `kv_cache_bytes`, `KVSystemProfiler`, `CompactionEvent`.
* `evaluation_pipeline/kv_metrics/aggregate.py` – `MetricAggregator`,
  `compute_paired_metrics`.
* `evaluation_pipeline/kv_metrics/paired_runner.py` – two-pass CLI.
* `evaluation_pipeline/kv_metrics/recorder.py` – sidecar jsonl helpers.
* `evaluation_pipeline/kv_metrics/runtime_breakdown.py` – component-level
  runtime profiling (`--profile_runtime_breakdown`).
* `evaluation_pipeline/kv_metrics/analysis/` – multi-sample experiments,
  statistics, the single-video Query x compaction-count trajectory, plotting
  and the auto-generated report (see `analysis/README.md`).
* `tests/test_kv_metrics.py` – 18 unit/sanity tests.
* `configs/eval_kv_metrics_smoke.yaml`, `configs/eval_kv_metrics_no_compression.yaml`.

**B. HERMES compaction trigger** – `OVO-Bench/models/MMDuet2_HERMES.py`
`_encode_frames()` calls `qa_model.predict_and_compress()` when
`max(cache_len) > kv_size`; the actual eviction is
`QwenVL_Hermes.apply_kv_cache_pruning_strict()` →
`_shrink_positions_and_rerotate_keys()`. We wrap the latter.

**C. Hook locations** – attention weights/output: an eager q_len reproduction of
the query prefill (`fidelity.capture_anchor_attention`) — no FlashAttention
patch. Logits: the last position of that same eager forward.

**D. kv_uid ↔ timestamp** – `KVProvenanceTracker` mirrors the cache per layer;
every visual token gets an immutable `vis:<i>` UID (per-kind counter, stable
across passes) plus `frame_index`, `frame_timestamp_sec`, `visual_token_index`.
Timestamps come from `iter_chunks_until_query` (queue → encoder hook).

**E. GQA mapping** – `region_context()` maps query head `h` to KV head
`h // (num_attention_heads // num_key_value_heads)` (here 16 → 2, group 8).
HERMES keeps the same token set across heads per layer, so provenance is stored
per layer and expanded through the GQA map at metric time.

**F. RAM tensor shape** – `attn_row [num_heads, kv_len]` (full-KV) and
`kept_mask [kv_len]`; `kept_mass = (probs·mask).sum(-1)` →
`[num_heads]`, divided by the region mass.

**G. AORE comparison point** – per-head pre-`o_proj` context of the video
region (`aore_level="per_head_pre_o_proj"`).

**H. JS/KL token position** – the LAST query prefill token (anchor), before any
autoregressive divergence and before the first-token constraint.

**I. Compression bytes** – the persistent cache K and V tensors
(`tensor_bytes = numel * element_size`), summed over layers. Python strings and
debug metadata are never counted.

**J. Wall-clock** – `apply_kv_cache_pruning_strict` is timed with
`torch.cuda.synchronize()` before/after **only when**
`--kv_metrics_profile_wallclock` is set; ρ_realtime uses stream-time deltas.

**K. Minimal test command**

```bash
python -m evaluation_pipeline.kv_metrics.paired_runner \
  --config configs/eval_kv_metrics_smoke.yaml \
  --benchmark ovobench --max_samples 1 \
  --kv_metrics_mode paired --kv_metrics_profile_wallclock
```

Sanity suite (no GPU/model needed):

```bash
python -m pytest tests/test_kv_metrics.py -q
```

**L. Example metrics json** – `outputs/kv_metrics/smoke_paired3/summary.json`
(and `paired/0.jsonl`). Real values from a 1-sample OVO-Bench smoke run at
budget=2000:

```json
{
  "ram_mean": 0.0288, "aore_mean": 0.3105, "js_mean": 7.3e-05,
  "kl_mean": 3.1e-04, "compression_ratio_mean": 2.808,
  "last_compaction_wallclock_mean": 0.0512,
  "temporal_ram_mean": {"0_10": 0.903, "10_30": 0.841, "30_60": 0.699, "60_300": 0.620}
}
```

**M. Limitations / approximations**

* **RAM normalisation**: the spec's §4 formula is `Σ_{i∈S_keep} A_full[i]`,
  which on a text anchor equals the *total video attention mass* (here ≈0.03)
  and never reaches 1 even with no compression. Spec §9 asks not to leak
  transient text KV into the RAM denominator, so the primary `ram_mean` is
  **region-normalised** (TEST 1 → ≈1). The unnormalised value is preserved as
  `ram_per_layer[layer]["raw_mean"]`.
* **Summary tokens** (`summary_token_mode`): long-term layers pool pruned
  tokens into one token. `identity_only` (default) counts only original
  positions that survive verbatim; `absorbed` also counts pooled parents.
  The choice is recorded in `config.json` and never silently changed.
* **AORE region scope**: compares the video-region attention contribution, not
  the whole `A@V`. The anchor query's RoPE position also differs between
  branches (inherent to compression + re-rotation), so AORE is an end-to-end
  functional difference, not a pure "same-Q" comparison.
* **Eager profiling**: RAM/AORE use an eager attention forward. This is
  FIDELITY PROFILING only — its latency must never be used as deployment
  wall-clock. Deployment/system timing keeps the original attention backend.
* **Temporal AORE/JS**: unsupported until a benchmark provides an evidence
  timestamp (`get_evidence_timestamp`).
* **Teacher-forced span** (`--kv_metrics_teacher_forced_span`): reserved; not
  yet wired (both branches currently stop at the first decode step).
* **Repeated compaction drift**: `compaction_index` and per-event
  `AORE/JS/RAM` are recorded so drift can be aggregated offline; a dedicated
  `AORE(n)` curve is not yet emitted.
