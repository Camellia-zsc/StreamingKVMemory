"""Synchronized end-to-end and nested compression timing on one RT20 query.

This runner installs timing wrappers only on the loaded model. It does not
change the memory policy, official prompt, video sampler, or decoder.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import json
import os
from pathlib import Path
from statistics import mean
import time

from InfoKVStream.run_ovo import ROOT, memory_config, now

SAMPLE_ID = "Real-Time Visual Understanding_sample_23_5"
REFERENCE_ROOT = ROOT / "outputs/InfoKVStream/streamingbench_rt20_20261013"
DEFAULT_OUT = ROOT / "outputs/InfoKVStream/end_to_end_timing_20261015"


class StageClock:
    def __init__(self, torch):
        self.torch = torch
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.current = None
        self.events = []
        self.context = None
        self.qa_forward_count = 0
        self.in_shrink = False
        self.gather_events = []
        self.restore = []

    def tick(self):
        self.torch.cuda.synchronize()
        return time.perf_counter()

    def add(self, key, seconds):
        self.totals[key] += seconds
        self.counts[key] += 1
        if self.current is not None:
            self.current["parts"][key] += seconds

    def patch(self, obj, name, factory):
        original = getattr(obj, name)
        setattr(obj, name, factory(original))
        self.restore.append((obj, name, original))

    def wrap(self, obj, name, key):
        def factory(original):
            def timed(*args, **kwargs):
                start = self.tick()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.add(key, self.tick() - start)
            return timed
        self.patch(obj, name, factory)

    def close(self):
        for obj, name, original in reversed(self.restore):
            setattr(obj, name, original)


def install(clock, model, method):
    import models.Qwen25VL_Stream as stream_module
    import torch

    qa = model.qa_model
    # The iterator's next() performs OpenCV read, frame sampling, resize and
    # PIL conversion. Its time is disjoint from stream_video_chunk time.
    original_iterator = stream_module.iter_chunks_until_query

    def timed_iterator(*args, **kwargs):
        source = iter(original_iterator(*args, **kwargs))
        while True:
            start = clock.tick()
            try:
                item = next(source)
            except StopIteration:
                clock.add("video_read_resize", clock.tick() - start)
                return
            clock.add("video_read_resize", clock.tick() - start)
            yield item

    stream_module.iter_chunks_until_query = timed_iterator
    clock.restore.append((stream_module, "iter_chunks_until_query", original_iterator))

    def stream_factory(original):
        def timed(*args, **kwargs):
            previous = clock.context
            clock.context = "visual"
            start = clock.tick()
            try:
                return original(*args, **kwargs)
            finally:
                clock.add("stream_chunks", clock.tick() - start)
                clock.context = previous
        return timed

    clock.patch(model, "stream_video_chunk", stream_factory)
    clock.wrap(qa, "get_image_features", "vision_encoder")

    original_forward = qa.language_model.forward

    def timed_forward(*args, **kwargs):
        context = clock.context
        if context == "visual":
            key = "visual_lm_forward"
        elif context == "question":
            key = "qa_prefill_forward" if clock.qa_forward_count == 0 else "qa_decode_forwards"
            clock.qa_forward_count += 1
        else:
            return original_forward(*args, **kwargs)
        start = clock.tick()
        try:
            return original_forward(*args, **kwargs)
        finally:
            clock.add(key, clock.tick() - start)

    qa.language_model.forward = timed_forward
    clock.restore.append((qa.language_model, "forward", original_forward))

    def question_factory(original):
        def timed(*args, **kwargs):
            previous = clock.context
            clock.context = "question"
            clock.qa_forward_count = 0
            start = clock.tick()
            try:
                return original(*args, **kwargs)
            finally:
                clock.add("qa_answering", clock.tick() - start)
                clock.context = previous
        return timed

    clock.patch(qa, "question_answering", question_factory)

    # Physical KV gather is a nested CUDA-kernel measurement. It is reported
    # separately from the synchronized wall-clock stage, not added to it.
    original_select = torch.index_select

    def timed_select(source, dim, index, *args, **kwargs):
        if not clock.in_shrink or source.ndim != 4:
            return original_select(source, dim, index, *args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = original_select(source, dim, index, *args, **kwargs)
        end.record()
        clock.gather_events.append((clock.current, start, end))
        return value

    torch.index_select = timed_select
    clock.restore.append((torch, "index_select", original_select))

    def shrink_factory(original):
        def timed(*args, **kwargs):
            clock.in_shrink = True
            try:
                return original(*args, **kwargs)
            finally:
                clock.in_shrink = False
        return timed

    clock.patch(qa, "_shrink_positions_and_rerotate_keys", shrink_factory)

    # This streaming run should stay below the M-RoPE position shift threshold.
    # Wrapping the primitives makes any actual re-rotation visible if triggered.
    import inference.qwenvl_hermes_fa2 as hermes_module
    for name in ("compute_cos_sin_for_positions", "rotary_delta", "apply_rotary_delta_to_keys_only"):
        clock.wrap(hermes_module, name, "mrope_primitives")

    if method == "hermes":
        for name, key in (
            ("predict_next_question", "hermes_predict_question"),
            ("_compute_attention_scores_manually", "hermes_attention"),
            ("prune_kv_cache_by_attention", "hermes_score_topk"),
            ("apply_kv_cache_pruning_strict", "physical_prune"),
        ):
            clock.wrap(qa, name, key)

        def compression_factory(original):
            def timed(*args, **kwargs):
                before = model._cache_lengths()
                entry = {"parts": defaultdict(float), "effective": False}
                clock.current = entry
                start = clock.tick()
                try:
                    return original(*args, **kwargs)
                finally:
                    entry["duration"] = clock.tick() - start
                    entry["effective"] = any(a < b for a, b in zip(model._cache_lengths(), before))
                    clock.events.append(entry)
                    clock.add("compression_all", entry["duration"])
                    clock.current = None
            return timed
        clock.patch(qa, "predict_and_compress", compression_factory)
    else:
        integrator = model._integrator
        memory = integrator.memory
        # Split the selection stage into score fusion and deterministic Top-K.
        # These helpers are module-level references inside infokv_stream.py,
        # so patch the module symbols used by both global and layerwise paths.
        import evaluation_pipeline.memory.infokv_stream as memory_module
        clock.wrap(memory_module, "fuse_scores", "infokv_scoring")
        clock.wrap(memory_module, "deterministic_topk", "infokv_topk")
        # ``EvalQwen25VLStreamInfoKV.reset`` recreates the ledger at the start
        # of every sample. Patch its class method so the timing wrapper follows
        # that fresh instance instead of retaining the pre-reset ledger.
        ledger_cls = type(integrator.ledger)
        for obj, name, key in (
            (integrator, "_refresh_keys_from_cache", "infokv_key_refresh"),
            (memory, "_refresh_v2v", "infokv_attention_refresh"),
            (memory, "_select_long_term_layerwise" if memory.layerwise_retention else "_select_long_term", "infokv_score_topk"),
            (memory, "compact", "infokv_memory_compact"),
            (integrator, "_build_sidecar", "infokv_sidecar"),
            (integrator.compactor, "apply_per_layer", "physical_prune"),
            (integrator, "_mrope_positions", "infokv_mrope_metadata"),
        ):
            clock.wrap(obj, name, key)
        clock.wrap(ledger_cls, "rebuild_after_compaction", "infokv_ledger_rebuild")
        import evaluation_pipeline.memory.qwen25vl_gpu as gpu_module
        clock.wrap(gpu_module, "build_keep_indices", "infokv_keep_indices")

        def compression_factory(original):
            def timed(*args, **kwargs):
                entry = {"parts": defaultdict(float), "effective": False}
                clock.current = entry
                start = clock.tick()
                try:
                    result = original(*args, **kwargs)
                    entry["effective"] = bool(result and result.effective)
                    return result
                finally:
                    entry["duration"] = clock.tick() - start
                    clock.events.append(entry)
                    clock.add("compression_all", entry["duration"])
                    clock.current = None
            return timed
        clock.patch(integrator, "maybe_compact", compression_factory)


def analyze(clock, total, method, layerwise, prediction, reference, model_load):
    clock.tick()
    for entry, start, end in clock.gather_events:
        elapsed = start.elapsed_time(end) / 1000.0
        clock.totals["kv_gather_gpu"] += elapsed
        if entry is not None:
            entry["parts"]["kv_gather_gpu"] += elapsed

    effective = [e for e in clock.events if e["effective"]]
    inactive = [e for e in clock.events if not e["effective"]]
    parts = defaultdict(float)
    for event in effective:
        for key, value in event["parts"].items():
            if key != "compression_all":
                parts[key] += value

    # Rows at this level are disjoint; nested detailed rows below must not be
    # added to their parent stream/compression/QA stage a second time.
    stream = clock.totals["stream_chunks"]
    read = clock.totals["video_read_resize"]
    qa = clock.totals["qa_answering"]
    compression = clock.totals["compression_all"]
    broad = {
        "video_read_resize": read,
        "visual_stream_excluding_compression": max(0.0, stream - compression),
        "compression_all_triggers": compression,
        "qa_answering": qa,
        "adapter_cleanup_and_other": total - read - stream - qa,
    }
    return {
        "method": method, "variant": "layerwise" if layerwise else "global",
        "sample_id": SAMPLE_ID, "budget": 2000,
        "prediction": prediction, "reference_prediction": reference,
        "matches_reference": prediction == reference,
        "model_load_sec": model_load, "end_to_end_sec": total,
        "trigger_count": len(clock.events), "effective_count": len(effective),
        "non_effective_count": len(inactive),
        "first_effective_ms": effective[0]["duration"] * 1000 if effective else None,
        "mean_effective_ms": mean(e["duration"] for e in effective) * 1000 if effective else None,
        "total_effective_sec": sum(e["duration"] for e in effective),
        "total_non_effective_sec": sum(e["duration"] for e in inactive),
        "broad_sec": broad,
        "nested_sec": dict(clock.totals),
        "effective_compression_nested_sec": dict(parts),
        "counts": dict(clock.counts),
        "effective_events": [
            {"duration_ms": e["duration"] * 1000,
             "parts_ms": {k: v * 1000 for k, v in e["parts"].items()}}
            for e in effective
        ],
    }


def run_one(method, layerwise, base, sample):
    import torch
    from evaluation_pipeline.evaluate import make_runtime_config
    from evaluation_pipeline.runtime import ModelRuntime

    config = copy.deepcopy(base)
    config["memory"] = (memory_config(2000, layerwise=layerwise) if method == "infokv_stream"
                        else {"method": "hermes", "compression": {"enabled": True, "budget": 2000}})
    torch.cuda.synchronize()
    load_start = time.perf_counter()
    runtime = ModelRuntime(make_runtime_config(config)).load()
    torch.cuda.synchronize()
    model_load = time.perf_counter() - load_start
    # Runtime normalizes the FA2 backend to ``fa2`` in the current loader,
    # while older revisions exposed the full ``flash_attention_2`` name.
    if runtime.attention_backend not in {"fa2", "flash_attention_2"}:
        raise RuntimeError(f"Expected FA2, got {runtime.attention_backend}")
    runtime.reset()
    model = runtime.model
    clock = StageClock(torch)
    try:
        install(clock, model, method)
        from evaluation_pipeline.adapters import StreamingBenchAdapter
        adapter = StreamingBenchAdapter(config["benchmarks"]["streamingbench"])
        start = clock.tick()
        produced = adapter.run_sample(runtime, sample)
        total = clock.tick() - start
        if len(produced) != 1:
            raise RuntimeError(f"Expected one answer for {SAMPLE_ID}")
        record = produced[0]
        reference_method = "hermes" if method == "hermes" else "infokv_stream"
        reference_root = (REFERENCE_ROOT if not layerwise else
                          ROOT / "outputs/InfoKVStream/rt_docstyle_layerwise_20261014_fa2/budget_stress/budget_2000")
        reference_path = reference_root / reference_method / "samples" / f"{SAMPLE_ID}.json"
        reference = json.loads(reference_path.read_text())["record"]["prediction"]
        result = analyze(clock, total, method, layerwise, record["prediction"], reference, model_load)
        result["visual_tokens_per_chunk"] = list(model.visual_tokens_per_chunk)
        result["seconds_per_chunk"] = list(model.seconds_per_chunk)
        return result
    finally:
        clock.close()
        del runtime, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main(out, only=None):
    import torch
    import yaml
    from evaluation_pipeline.adapters import StreamingBenchAdapter

    os.chdir(ROOT)
    os.environ["HERMES_ABLATION_BACKEND"] = "fa2"
    torch.set_num_threads(4)
    base = yaml.safe_load((ROOT / "configs/eval_kv_metrics_streamingbench_real_qwen25vl.yaml").read_text())
    adapter = StreamingBenchAdapter(base["benchmarks"]["streamingbench"])
    samples = {s.sample_id: s for s in adapter.load_samples()}
    sample = samples[SAMPLE_ID]
    methods = (("hermes", False), ("infokv_stream", False), ("infokv_stream", True))
    if only:
        methods = tuple(item for item in methods if item[0] == only or
                        (only == "layerwise" and item == ("infokv_stream", True)) or
                        (only == "global" and item == ("infokv_stream", False)))
        if not methods:
            raise ValueError(f"unknown --only value: {only}")
    rows = [run_one(method, layerwise, base, sample) for method, layerwise in methods]
    if len(rows) > 1 and len({r["effective_count"] for r in rows}) != 1:
        raise RuntimeError("Effective compression counts differ")
    if not all(r["visual_tokens_per_chunk"] == rows[0]["visual_tokens_per_chunk"] and
               r["seconds_per_chunk"] == rows[0]["seconds_per_chunk"] for r in rows):
        raise RuntimeError("Video token/chunk schedules differ")
    if not all(r["matches_reference"] for r in rows):
        raise RuntimeError("Prediction changed under timing instrumentation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_timing.json").write_text(json.dumps({"created_at": now(), "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    with (out / "stage_summary.csv").open("w", newline="") as handle:
        fields = ["method", "variant", "model_load_sec", "end_to_end_sec", "trigger_count",
                  "effective_count", "first_effective_ms", "mean_effective_ms", "total_effective_sec",
                  "total_non_effective_sec", "video_read_resize", "visual_stream_excluding_compression",
                  "compression_all_triggers", "qa_answering", "adapter_cleanup_and_other"]
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, row["broad_sec"].get(key)) for key in fields})
    with (out / "compression_parts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["method", "variant", "part", "total_ms", "mean_per_effective_ms"])
        writer.writeheader()
        for row in rows:
            for key, seconds in sorted(row["effective_compression_nested_sec"].items()):
                writer.writerow({"method": row["method"], "variant": row["variant"],
                                 "part": key, "total_ms": seconds * 1000,
                                 "mean_per_effective_ms": seconds * 1000 / row["effective_count"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["HERMES" if r["method"] == "hermes" else
              ("InfoKV layerwise" if r["variant"] == "layerwise" else "InfoKV global")
              for r in rows]
    keys = list(rows[0]["broad_sec"])
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#bab0ac"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bottom = [0.0] * len(rows)
    for key, color in zip(keys, colors):
        values = [row["broad_sec"][key] for row in rows]
        axes[0].bar(labels, values, bottom=bottom, label=key, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set_ylabel("Seconds"); axes[0].set_title("Video input to answer returned")
    axes[0].legend(fontsize=7)
    axes[1].bar(labels, [r["mean_effective_ms"] for r in rows],
                color=["#4c78a8" if r["method"] == "hermes" else
                       ("#54a24b" if r["variant"] == "layerwise" else "#f58518")
                       for r in rows])
    axes[1].set_ylabel("Milliseconds"); axes[1].set_title("Mean effective compaction")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=12)
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "stage_comparison.png", dpi=180)
    fig.savefig(out / "stage_comparison.pdf")
    print(json.dumps({"output": str(out), "rows": [{k: r[k] for k in
          ("method", "variant", "end_to_end_sec", "effective_count", "mean_effective_ms")}
          for r in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", choices=("hermes", "global", "layerwise"),
                        help="run one method in a fresh process for isolated timing")
    args = parser.parse_args()
    main(args.output, only=args.only)
