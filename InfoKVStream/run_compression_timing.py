"""Compare full effective-compression wall-clock time on one frozen RT video."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import time
from statistics import median

from InfoKVStream.run_ovo import ROOT, memory_config, now

VIDEO_ID = "Real-Time Visual Understanding_sample_23_5"
DEFAULT_OUT = ROOT / "outputs/InfoKVStream/compression_timing_20261014"
RT_ROOT = ROOT / "outputs/InfoKVStream/streamingbench_rt20_20261013"


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_one(method: str, *, layerwise: bool, sample, base: dict) -> dict:
    import torch
    from evaluation_pipeline.adapters import StreamingBenchAdapter
    from evaluation_pipeline.evaluate import make_runtime_config
    from evaluation_pipeline.runtime import ModelRuntime

    config = copy.deepcopy(base)
    if method == "hermes":
        config["memory"] = {"method": "hermes", "compression": {"enabled": True, "budget": 2000}}
    else:
        config["memory"] = memory_config(2000, layerwise=layerwise)
    runtime = ModelRuntime(make_runtime_config(config)).load()
    model = runtime.model
    timings = []
    physical_timings = []
    original_prune = model.qa_model.apply_kv_cache_pruning_strict

    def physical(*args, **kwargs):
        _sync(torch); t0 = time.perf_counter()
        result = original_prune(*args, **kwargs)
        _sync(torch)
        physical_timings.append((time.perf_counter() - t0) * 1000.0)
        return result

    model.qa_model.apply_kv_cache_pruning_strict = physical
    if method == "hermes":
        qa = model.qa_model
        original = qa.predict_and_compress

        def timed(*args, **kwargs):
            before = list(model._cache_lengths())
            _sync(torch); t0 = time.perf_counter()
            result = original(*args, **kwargs)
            _sync(torch)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if any(a < b for a, b in zip(model._cache_lengths(), before)):
                timings.append(elapsed)
            return result

        qa.predict_and_compress = timed
    else:
        integrator = model._integrator
        original = integrator.maybe_compact

        def timed():
            _sync(torch); t0 = time.perf_counter()
            event = original()
            _sync(torch)
            if event is not None and event.effective:
                timings.append((time.perf_counter() - t0) * 1000.0)
            return event

        integrator.maybe_compact = timed
    runtime.reset()
    adapter = StreamingBenchAdapter(config["benchmarks"]["streamingbench"])
    produced = adapter.run_sample(runtime, sample)
    if len(produced) != 1:
        raise RuntimeError(f"expected one answer for {VIDEO_ID}, got {len(produced)}")
    stats = model.profiler.as_dict()
    return {
        "method": method,
        "variant": "layerwise" if layerwise else "global",
        "video_id": VIDEO_ID,
        "budget": 2000,
        "prediction": produced[0].get("prediction"),
        "effective_count": len(timings),
        "first_effective_ms": timings[0] if timings else None,
        "mean_effective_ms": sum(timings) / len(timings) if timings else None,
        "median_effective_ms": median(timings) if timings else None,
        "std_effective_ms": (sum((x - sum(timings)/len(timings)) ** 2 for x in timings) / len(timings)) ** 0.5 if timings else None,
        "total_effective_ms": sum(timings),
        "event_times_ms": timings,
        "physical_times_ms": physical_timings,
        "mean_physical_ms": sum(physical_timings) / len(physical_timings) if physical_timings else None,
        "visual_tokens_per_chunk": list(model.visual_tokens_per_chunk),
        "seconds_per_chunk": list(model.seconds_per_chunk),
        "profiler_phase_seconds": stats.get("phase_seconds", {}),
        "profiler_compression_count": stats.get("phase_counts", {}).get("hermes_compression", 0),
        "definition": "Synchronized wall clock of predict_and_compress (HERMES) or maybe_compact (InfoKV), conditioned on actual eviction; includes scoring/Top-K, KV gather, M-RoPE handling, ledger and sidecar. Excludes visual prefill, capture of cached N/D/Q/K signals, video IO and QA; no fidelity probes are installed.",
    }


def main(out: Path = DEFAULT_OUT):
    import yaml
    base = yaml.safe_load((ROOT / "configs/eval_kv_metrics_streamingbench_real_qwen25vl.yaml").read_text())
    manifest = json.loads((RT_ROOT / "manifest.json").read_text())
    from evaluation_pipeline.adapters import StreamingBenchAdapter
    adapter = StreamingBenchAdapter(base["benchmarks"]["streamingbench"])
    samples = {s.sample_id: s for s in adapter.load_samples()}
    if VIDEO_ID not in samples:
        raise KeyError(f"{VIDEO_ID} is missing from the frozen RT20 adapter")
    rows = []
    for method, layerwise in (("hermes", False), ("infokv_stream", False), ("infokv_stream", True)):
        rows.append(run_one(method, layerwise=layerwise, sample=samples[VIDEO_ID], base=base))
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    if len({r['effective_count'] for r in rows}) != 1:
        raise RuntimeError('Three methods have different effective compression counts')
    if not all(r['visual_tokens_per_chunk'] == rows[0]['visual_tokens_per_chunk'] and
               r['seconds_per_chunk'] == rows[0]['seconds_per_chunk'] for r in rows):
        raise RuntimeError('Compression timing input token/chunk schedules differ')
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing_summary.json").write_text(json.dumps({"created_at": now(), "video_id": VIDEO_ID, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    with (out / "timing_summary.csv").open("w", newline="") as f:
        fields = ["method", "variant", "video_id", "budget", "effective_count", "first_effective_ms", "mean_effective_ms", "median_effective_ms", "std_effective_ms", "total_effective_ms"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["HERMES", "InfoKV-global", "InfoKV-layerwise"]
    colors = ["#2677b8", "#dc6b24", "#55a868"]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, [r["mean_effective_ms"] or 0 for r in rows], color=colors, alpha=.85)
    ax.set_xticks(x, labels); ax.set_ylabel("Effective compression time (ms)")
    ax.set_title(f"Compression wall clock | {VIDEO_ID} | budget=2000")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "compression_time_comparison.png", dpi=180); fig.savefig(out / "compression_time_comparison.pdf")
    print(json.dumps({"output": str(out), "rows": [{k: r.get(k) for k in ["method", "variant", "effective_count", "first_effective_ms", "mean_effective_ms"]} for r in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
