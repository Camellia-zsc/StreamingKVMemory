# Model backends

The streaming evaluation pipeline supports two backends, selected by
`model.backend` in the config (default `mmduet2`, so legacy configs are
untouched):

| backend | class | file |
|---|---|---|
| `mmduet2` | `models.MMDuet2.EvalMMDuet2` / `models.MMDuet2_HERMES.EvalMMDuet2Hermes` | `OVO-Bench/models/` |
| `qwen2_5_vl` | `models.Qwen25VL_Stream.EvalQwen25VLStream` | `OVO-Bench/models/Qwen25VL_Stream.py` |

## Qwen2.5-VL streaming backend (non-proactive)

New research protocol — long/infinite streaming video memory:

```
video stream -> incremental chunk prefill (KV grows) -> HERMES compression
             -> query_time -> inject question -> decode answer
```

Before the query arrives this backend **never** calls `generate`. There is no
proactive decision, no response-timing policy and no NO REPLY parsing.

### Core interface (`EvalQwen25VLStream`)

* `reset()` — drop KV cache + accounting; no text.
* `stream_video_chunk(frames)` — incremental visual-KV prefill, no text.
* `ask_question(text, options=..., max_new_tokens=...)` — inject + decode.

The kv_metrics instrumentation (`KVInstrumentation`) still attaches to
`qa_model` exactly as on the MMDuet2 path because `EvalQwen25VLStream`
subclasses `EvalMMDuet2Hermes` and exposes the same attribute surface
(`num_layers`, `visual_start_idx`, `kv_cache`,
`apply_kv_cache_pruning_strict`, `question_answering`).

### Full vs HERMES

Both modes use the **same** loaded model / processor / frames / decode config.
`mode=full` sets the HERMES budget to a sentinel (`FULL_CACHE_BUDGET`), so no
compaction ever triggers and the code path stays identical to `mode=hermes`.

### MCQ decoding

`mcq_decode_mode` (config, or `--mcq_decode_mode`):

* `official` (default) — free decode, option letter extracted afterwards;
* `constrained` — historical MMDuet2-style hard first-token A–E mask.

A read-only `lm_head` forward hook captures the **step-0** option logits
(`P_gold`, `ΔP_gold`, `ΔM_z`). It only reads the logits tensor and never
alters the mask, decoder or formal prediction.

### Cache accounting / streaming proof / profiling

`evaluation_pipeline/models/qwen25_vl.py` (no torch import) provides:

* `CacheAccounting` — runtime `total_kv_before/after`, `protected_prefix_tokens`,
  `managed_visual_before/after`, `effective_compaction(_index)`, `visual_budget`,
  and a `managed_visual_after <= budget` check. No hard-coded 63/89.
* `IncrementalStreamAudit` — records per-chunk pixels and cache lengths and
  fails if a chunk's `cache_len_before` is not the running prefix+seen total
  (i.e. if the full prefix were re-encoded, the "fake streaming" case).
* `StreamingProfiler` — per-phase wall clock and
  `rho_realtime = compression_time / trigger_interval`.

## GPU smoke tests

```bash
python -m evaluation_pipeline.models.qwen25_vl_smoke qwen-load-smoke
python -m evaluation_pipeline.models.qwen25_vl_smoke qwen-stream-smoke
python -m evaluation_pipeline.models.qwen25_vl_smoke qwen-hermes-smoke --budget 200
```
