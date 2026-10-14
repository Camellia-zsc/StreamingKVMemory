# InfoKVStream and controlled HERMES comparisons

InfoKVStream supports two retention modes for Qwen2.5-VL streaming video:

- `memory.scoring.layer_evolution.layerwise: false`: original common token set;
- `memory.scoring.layer_evolution.layerwise: true`: independent token set for
  each decoder layer. Temporal novelty N and V2V importance A are shared; each
  layer's D measures the same token's representation change through that layer
  (`1 - cosine(input_hidden, output_hidden)`, clipped to `[0, 1]`).

Recent and long-term describe video time regions. They apply within every
Transformer layer and are independent of the layer number. Each layer keeps
`B_recent + B_long <= B` visual slots; the protected prompt prefix is additional.
Tokens absent from a layer cannot re-enter that layer's candidate pool. The CPU
metadata keeps the union of identities across layers, not an extra model KV
cache. N/A are normalized over this shared metadata pool and D over each layer's
live candidate pool. V2V refreshes live keys in its configured anchor layer;
tokens evicted from that layer retain their last captured key in the shared
CPU signal bank. InfoKV produces no summary tokens.

`qwen25vl_gpu.py` performs physical per-layer gathers using the existing HERMES
compactor and M-RoPE handling. Forward hooks compute D on GPU and transfer only
one scalar per token per layer; they do not retain all hidden states.

## Reproducible RT20 evaluation

The frozen 20-video/100-question selection is
`configs/streamingbench_rt20_selection.json`. Model weights, videos and generated
results are deliberately excluded from this repository. Supply the local model
and dataset paths in `configs/eval_kv_metrics_streamingbench_real_qwen25vl.yaml`.

Use a single visible GPU and the same FA2-enabled Python environment for every
arm. FA2 and SDPA in the inherited backend have different image/video encoding
entry points; silently switching between them changes visual token counts.
Force the FA2 guard for controlled comparisons:

```bash
export CUDA_VISIBLE_DEVICES=1
export HERMES_ABLATION_BACKEND=fa2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON=/home/workspace/ypq/miniconda3/envs/hermes-qwen/bin/python

# Generate the official Full / HERMES / original InfoKV actual-query references.
$PYTHON -m InfoKVStream.run_streamingbench_rt all

# Then collect trajectories, evaluate all seven budgets, and run timing.
$PYTHON -m InfoKVStream.run_layerwise_queue
```

The layerwise budgets are exactly `6000, 2000, 1000, 500, 200, 100, 50`.
The queue runs InfoKV's seven budget sweeps. HERMES sweeps use the same runner:

```bash
for B in 6000 2000 1000 500 200 100 50; do
  $PYTHON -m InfoKVStream.run_docstyle_diagnostics stress --method hermes --budget "$B"
done
$PYTHON -m InfoKVStream.analyze_docstyle_diagnostics
```

Each successful sample has an atomic JSON record. Source/config hashes prevent
mixing altered implementations into a resumed experiment. Reuse is allowed only
for matching method, budget, model, prompts, sampling and encoding protocol.
Trajectory probes use a fixed official question, cloned cache, and matching Full
video times. Reports retain trigger counts separately from effective evictions.

## Compression timing

```bash
$PYTHON -m InfoKVStream.run_compression_timing
```

At budget 2000, the same RT20 video `sample_23`, query 5 is streamed for HERMES,
original InfoKV and layerwise InfoKV. Synchronized wall-clock time surrounds
`predict_and_compress()` for HERMES and `maybe_compact()` for both InfoKV modes.
Only calls that actually remove managed KV slots enter first/mean/median/std/total
statistics. Input token and chunk schedules and effective counts must match.

HERMES includes pseudo-question prediction, attention scoring, per-layer score
refinement/Top-K, KV gather, position handling and summary creation. InfoKV
includes key refresh, shared V2V scoring, normalization/fusion/Top-K, keep-index
construction, KV gather, position handling, ledger and sidecar updates. Both
exclude video IO, visual prefill, QA and diagnostic fidelity probes. Cached
N/D/Q/K capture happens during InfoKV visual prefill and is also excluded from
this compression-only metric; end-to-end stream time is recorded separately.
Physical-gather time is retained as a narrower breakdown, not substituted for
the full compression metric.

Outputs are under `outputs/InfoKVStream/`; PNG/PDF plots and CSV/JSON summaries
stay local. The implementation and CPU regression tests contain no model weights
or large artifacts.
