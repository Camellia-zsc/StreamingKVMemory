# Pluggable visual-KV memory methods

One config key selects the streaming memory method:

```yaml
memory:
  method: hermes          # default (backward compatible), or...
  # method: infokv_stream
```

The upper pipeline only ever talks to the `KVMemoryMethod` interface:

```python
memory = build_memory_method(config)
memory.reset()
for chunk in stream:
    memory.observe(chunk)
    if memory.should_compact():
        memory.compact()
stats = memory.get_stats()
```

No `if method == "hermes" / elif method == "infokv"` is scattered across the
pipeline; dispatch lives in exactly one place (`factory.py`).

## Files

| File | Role |
| --- | --- |
| `base.py` | `KVMemoryMethod` interface (`reset/observe/should_compact/compact/get_stats`) + `ObserveResult` / `CompactResult`. |
| `factory.py` | `build_memory_method(config)`, `resolve_memory_config(config)`, registry, `register_memory_method`. |
| `hermes_adapter.py` | `HermesMemoryMethod` -- thin adapter over the **untouched** HERMES backend. Never imports torch/HERMES; the real pruning stays in `HERMES/inference/qwenvl_hermes*.py`. |
| `infokv_stream.py` | `InfoKVStreamMemory` -- the new InfoKV-inspired streaming video memory (CPU first version). |
| `v2v.py` | `ContextImportanceProvider`, `VisionToVisionAttentionProvider`, `vision_to_vision_attention` -- the LiveVLM-inspired A^V2V provider. |
| `position.py` | `RetainedIndexSet`, `PositionAdapter`, `Qwen25VLMRoPEAdapter`, `KVCacheCompactor` -- interface-only reuse plan for the GPU stage. |
| `qwen25vl_gpu.py` | **GPU stage**: `VisualTokenLedger`, `InfoKVCapture`, real `Qwen25VLMRoPEAdapter` / `KVCacheCompactor`, `InfoKVGPUIntegrator`. Pure-Python parts are CPU-testable; torch is imported lazily. |
| `qwen25vl_backend.py` | **GPU stage**: `EvalQwen25VLStreamInfoKV`, a subclass of the Qwen2.5-VL streaming backend that funnels every chunk through the integrator. |
| `signals.py` | `temporal_novelty`, `layer_evolution`, `min_max_normalize`, `fuse_scores`, `deterministic_topk`. |
| `schemas.py` | `MemoryMethodConfig`, `Observation`, `TokenRecord`, `PositionMetadata`, `MemoryStats`, validation. |
| `debug.py` | `python -m evaluation_pipeline.memory.debug ...` CPU synthetic CLI. |

> Importing this package never imports `torch`, HERMES or OVO-Bench.

## This is **InfoKV-inspired**, not the original text InfoKV

The paper's text InfoKV scores tokens by **predictive entropy** over a next-token
vocabulary distribution. Visual tokens are continuous embeddings with no natural
next-token distribution, so the first version **does not fabricate a "visual
entropy"**. It composes three explicit signals instead:

```
S_i = w_N * TemporalNovelty_i
    + w_D * LayerRepresentationEvolution_i
    + w_A * A^V2V_i
```

### Granularity (explicit)

| signal | granularity |
| --- | --- |
| `N` Temporal Novelty | **chunk-level** (shared by every token of the chunk) |
| `D` Layer Evolution | **token-level** |
| `A^V2V` | **token-level** |

### A^V2V provenance (LiveVLM-inspired)

Vision-to-Vision context importance is **inspired by LiveVLM v2** Vision Sink
Bucketing (VSB):

```
recent visual queries -> historical visual keys -> attention-based importance
```

Reused principle only (see `modules/cache_utils.py::StreamCache.cache_compression_diversity`
in the official repo):

* `Q_recent = Q[:, :, -window_len:, :]` (window ≈ 32 in LiveVLM);
* `K = repeat_kv(K_history, Hq // Hkv)` (GQA/MQA);
* `softmax(Q_recent @ K^T / sqrt(D))` over history, then mean over the recent
  query window and mean over attention heads -> one score per historical token.

We do **NOT** reproduce LiveVLM's VSB scatter selection, Position-agnostic
Retrieval (PaR), page retrieval, or its full cache architecture. Our selection
stays the fused `N + D + A^V2V` Top-K.

* Paper: arXiv:2505.15269
* Code: https://github.com/sjtu-zhao-lab/LiveVLM

**Strictly vision-to-vision.** The recent window and the historical keys contain
**only visual tokens** (`Observation.is_visual`). System / question / answer /
NO-REPLY / text tokens are never fed to the provider, so compression stays
query-agnostic (it does not depend on a question that has not arrived yet).

## State flow (`InfoKVStreamMemory`)

```
new chunk ──observe──► Recent Buffer (last B_recent visual tokens, high fidelity)
                              │ overflow ("retired", never dropped)
                              ▼
                    Long-term candidate pool
                              │  every compression.interval_chunks
                              ▼
   normalize(N, D, A) ──► fuse ──► deterministic Top-K(B_long) ──► evict rest
```

* **Recent Buffer** keeps the most recent `B_recent` visual tokens verbatim.
  No information-aware pruning here; overflow retires into the candidate pool.
* **Long-term Cache** stores only tokens that already left the Recent Buffer,
  keeping `TopK(final_score, B_long)`.

These are a *temporal* two-level memory; they are unrelated to HERMES's
shallow/middle/deep Transformer-layer memory.

## The three signals

* **Temporal Novelty (N)** -- `N_t = 1 - cos(g_t, reference)` where `g_t` is the
  chunk embedding (mean-pooled token embeddings). `reference` is either the
  previous chunk or the mean of the recent `K` chunks (default). Shared by every
  token of the chunk.
* **Layer Representation Evolution (D)** -- `D_i = 1 - cos(h_i^shallow,
  h_i^deep)` for the same visual token across two selected layers. Token identity
  is kept row-aligned.
* **Recent-context / Attention Importance (A)** -- an injected per-token score
  (`attention_importance`, a sequence or `{token_id: score}` mapping). A future
  GPU path can feed HERMES-style importance here; the CPU debug path feeds
  synthetic scores. No model is loaded for CPU tests.

## Normalization & fusion

The three raw signals have different scales, so each is **min-max normalized over
the current candidate pool** before fusion:

```
x_norm = (x - min(x)) / (max(x) - min(x) + eps)
```

Weights must be `>= 0`; a sum other than 1 is renormalized with a warning.
Disabled signals contribute weight 0.

## Deterministic Top-K

The long-term selection key is total and explicit:
`(-score, -original_position, token_id)` -- higher score wins; on a tie the
**more recent** token wins; on a further tie the smaller `token_id` wins. It never
relies on Python's unstable sort.

## Trigger count vs effective compaction count

Two counters are kept strictly separate:

* `compression_trigger_count` -- incremented every time `compact()` is called,
  whether or not tokens were removed;
* `effective_compaction_count` -- incremented only when a real eviction happened
  (`evicted > 0`).

The periodic policy accepts a transient overshoot between two compactions (no
hard-overflow trigger). This is tracked read-only:

* `peak_managed_tokens`
* `peak_budget_overshoot_tokens`
* `peak_budget_overshoot_ratio`

For the reference scenario (20 chunks × 8 tokens, budget 64 / recent 16 / long 48,
interval 4): `trigger_count = 5`, `effective_compaction_count = 3`, pool
transitions `16→16`, `48→48`, `80→48`, `80→48`, `80→48`, peak managed `96`
(overshoot `32`, ratio `0.5`).

Downstream convention: fidelity trajectory uses `k = effective_compaction_count`;
runtime overhead uses `compression_trigger_count`.

## Position metadata

`TokenRecord` keeps `original_position`, `chunk_id`, `frame_time`, `token_index`.
Compaction output is re-sorted by `original_position`, so retained tokens stay in
time order (no reversal). No M-RoPE / position re-index transform is invented
here -- that is reserved for `PositionAdapter` / `Qwen25VLMRoPEAdapter` during GPU
integration (see `position.py`).

### Selection semantics: compatibility and layerwise modes

With `scoring.layer_evolution.layerwise: false`, `N / D / A^V2V` produce one
global set of retained visual token ids (`retained_visual_indices()`), and every
Transformer layer gathers the same visual positions. With `layerwise: true`, N
and A remain shared, D is computed from adjacent decoder-layer representations,
and every layer runs its own deterministic Top-K. The GPU ledger and compactor
then pass a separate keep-index list to each layer, matching HERMES's
heterogeneous retention semantics. `RetainedIndexSet` supports both shapes.

## Eviction

Eviction is a **real eviction**: no summary token is generated
(`compression.summary_strategy` must be `none`). This keeps the analysis of
InfoKV selection clean and the HERMES ablation uncontaminated.

## Sidecar / metrics schema

Method-agnostic (`MemoryStats.as_dict()`):
`memory_method, budget_total, budget_recent, budget_long_term,
compression_trigger_count, effective_compaction_count, tokens_before,
tokens_after, recent_size, long_term_size, evicted_count, retention_ratio,
compression_wallclock, compression_interval_chunks, peak_managed_tokens,
peak_budget_overshoot_tokens, peak_budget_overshoot_ratio`.

Per-compaction event (`compaction_events()`):
`trigger_index, effective_index, effective, before, after, evicted`.

Per-token debug rows (`TokenRecord.as_sidecar()`):
`token_id, chunk_id, novelty_raw, novelty_norm, evolution_raw, evolution_norm,
v2v_raw, v2v_norm, final_score, selected, is_visual` (plus deprecated
`attention_raw/attention_norm` aliases).

HERMES is never forced to emit InfoKV-specific fields.

## What is still interface-only (needs GPU integration)

The GPU integration is implemented (see below); the remaining gaps are:

* diversity-aware retention / LiveVLM scatter selection -- deliberately not v1;
* summary ("KE") tokens -- v1 uses real eviction, no summary;
* cross-checking real keys (not the write-time capture) for V2V is partially
  handled by refreshing keys from the live cache before each compaction.

## GPU integration (Qwen2.5-VL)

`memory.method: infokv_stream` is wired through a **new branch**: the runtime
builds `EvalQwen25VLStreamInfoKV` (a subclass) and attaches an
`InfoKVGPUIntegrator`. HERMES/Full backends are untouched.

* **D layers**: `layer_evolution.layer_a` / `layer_b` (defaults `(2, N-4)`),
  captured via read-only hooks on `language_model.layers[i]`.
* **V2V layer**: `scoring.v2v.layer` (default `N//2`), captured from the
  already-computed `self_attn.q_proj` / `k_proj` outputs (no re-projection).
* **Strictly visual**: every `Observation` from the integrator has
  `is_visual=True`, and the provider only ever receives visual Q/K.
* **Identity mapping**: `VisualTokenLedger` keeps `token_id`,
  `original_stream_position`, `cache_physical_index` and `mrope_position`
  separate; `token_id -> cache_physical_index` is resolved before any gather.
* **Global broadcast**: the same ascending keep-index list is used for every
  layer.
* **M-RoPE reuse**: `KVCacheCompactor.apply` delegates to the backend's validated
  `apply_kv_cache_pruning_strict` -> `_shrink_positions_and_rerotate_keys`; the
  RoPE math lives in `HERMES/inference/reindex_3d.py` and is **never reimplemented**.
  For InfoKV v1 (real eviction, no summary) the backend's long-term summary
  branch is suppressed for the duration of the call and immediately restored, so
  the backend code is not modified.

### GPU smoke

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n hermes-qwen python -m \
  evaluation_pipeline.models.qwen25_vl_smoke qwen-infokv-smoke \
  --model_path models/Qwen2.5-VL-7B-Instruct --budget 300 \
  --compression_interval 2 --chunks 6
```

The smoke checks: Full/InfoKV both run, N/D/A finite, V2V shape, retained ids map
to cache indices, per-layer KV shapes valid, M-RoPE re-index + decode, effective
compaction shrinks the cache, the next chunk appends, and capture-on vs
capture-off generation is identical.

Reuse the verified HERMES components rather than re-implementing M-RoPE:
`HERMES/inference/reindex_3d.py` (`compute_cos_sin_for_positions`,
`rotary_delta`, `apply_rotary_delta_to_keys_only`, `_get_mrope_section`,
`contiguous_kv`) and the backend methods
`_shrink_positions_and_rerotate_keys` / `apply_kv_cache_pruning_strict`
(with `visual_start_idx` as the protected prefix).

## Commands

```bash
# CPU tests (InfoKV memory)
python -m pytest tests/test_memory_infokv.py -q

# CPU debug CLI
python -m evaluation_pipeline.memory.debug \
    --method infokv_stream \
    --num_chunks 20 \
    --tokens_per_chunk 8 \
    --budget 64 \
    --recent_budget 16 \
    --compression_interval 4

# GPU smoke (requires the local Qwen checkpoint and a CUDA environment)
CUDA_VISIBLE_DEVICES=0 python -m evaluation_pipeline.models.qwen25_vl_smoke \
    qwen-infokv-smoke --model_path models/Qwen2.5-VL-7B-Instruct \
    --budget 300 --compression_interval 2 --chunks 6

# Paired evaluation after the smoke passes
python -m evaluation_pipeline.evaluate \
    --config configs/eval_kv_metrics_infokv_stream_example.yaml \
    --benchmarks streamingbench --max_samples 3 --save_raw_prediction
```
