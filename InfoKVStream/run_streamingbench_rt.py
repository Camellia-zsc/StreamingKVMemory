"""Matched Qwen Full/HERMES/InfoKV diagnostics on the frozen Plan-B RT set.

The historical Plan-B run used MMDuet2. It supplies the video/question set,
not the model baseline: this runner measures all three Qwen arms afresh.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from InfoKVStream.run_ovo import ROOT, atomic_json, now, memory_config, fingerprints

DEFAULT_OUTPUT = ROOT / "outputs/InfoKVStream/streamingbench_rt20_20261013"
SELECTION = ROOT / "configs/streamingbench_rt20_selection.json"
METHODS = ("full_cache", "hermes", "infokv_stream")
PILOT_IDS = [f"Real-Time Visual Understanding_sample_1_{i}" for i in (1, 5)]


def prepare(output):
    import yaml
    base = yaml.safe_load((ROOT / "configs/eval_kv_metrics_streamingbench_real_qwen25vl.yaml").read_text())
    selected = json.loads(SELECTION.read_text())
    sources = fingerprints()
    for relative in ["InfoKVStream/run_streamingbench_rt.py", "evaluation_pipeline/adapters.py",
                     *[str(p.relative_to(ROOT)) for p in (ROOT / "evaluation_pipeline/kv_metrics").glob("*.py")]]:
        sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    contract = {"base_config": base, "selected_samples": selected, "source_sha256": sources,
                "infokv_memory": memory_config(2000), "budget": 2000,
                "methods": METHODS, "fidelity": "all 100 actual query anchors; no future-query probes"}
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["contract_id"] != contract_id:
            raise RuntimeError("Source/protocol changed; use a new output directory")
        return saved
    manifest = {**contract, "contract_id": contract_id, "created_at": now(),
                "sample_ids": [q for s in selected["samples"] for q in s["query_ids"]],
                "video_ids": [s["video_id"] for s in selected["samples"]],
                "pilot_ids": PILOT_IDS, "historical_reference_model": "MMDuet2",
                "notes": ["RT full_prefix diagnostic subset, not overall StreamingBench accuracy.",
                          "RAM is retained attention mass, not host memory.",
                          "HERMES and InfoKV are freshly evaluated on the same Qwen2.5-VL-7B.",
                          "Eager anchor profiling time is excluded from inference timing."]}
    atomic_json(path, manifest)
    for relative in sources:
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / relative).read_bytes())
    return manifest


class Timing:
    """Installed before KV hooks: QA timing excludes the eager reproduction."""
    def __init__(self, model):
        import functools
        import torch
        self.torch = torch
        self.model = model
        original_stream = model.stream_video_chunk
        original_qa = model.qa_model.question_answering

        @functools.wraps(original_stream)
        def stream(frames, **kwargs):
            start = self.clock()
            result = original_stream(frames, **kwargs)
            self.stream_sec += self.clock() - start
            self.frames += len(frames)
            return result

        @functools.wraps(original_qa)
        def ask(*args, **kwargs):
            self.lengths = [int(k.shape[2]) for k, _ in model.qa_model.kv_cache]
            self.prefix = int(model.qa_model.visual_start_idx)
            start = self.clock()
            result = original_qa(*args, **kwargs)
            self.qa_sec += self.clock() - start
            self.raw_answer = result
            return result

        model.stream_video_chunk = stream
        model.qa_model.question_answering = ask
        self.reset()

    def clock(self):
        self.torch.cuda.synchronize()
        return time.perf_counter()

    def reset(self):
        self.stream_sec = self.qa_sec = 0.0
        self.frames = self.prefix = 0
        self.lengths = []
        self.raw_answer = None
        self.torch.cuda.reset_peak_memory_stats()


def worker(args, manifest):
    import gc
    import math
    import torch
    import transformers
    from evaluation_pipeline.adapters import StreamingBenchAdapter
    from evaluation_pipeline.evaluate import make_runtime_config
    from evaluation_pipeline.runtime import ModelRuntime
    from evaluation_pipeline.kv_metrics.config import KVMetricConfig
    from evaluation_pipeline.kv_metrics.hooks import KVInstrumentation
    from evaluation_pipeline.kv_metrics.paired_runner import _load_chunk_module, _wrap_runtime_events, _official_correct
    from evaluation_pipeline.kv_metrics.aggregate import compute_paired_metrics

    torch.set_num_threads(4)
    method = args.method
    directory = args.output / method
    directory.mkdir(parents=True, exist_ok=True)
    expected = manifest["pilot_ids"] if args.pilot else manifest["sample_ids"]
    pending = [s for s in expected if not (directory / "samples" / f"{s}.json").exists()]
    if not pending:
        return
    config = copy.deepcopy(manifest["base_config"])
    config["memory"] = (manifest["infokv_memory"] if method == "infokv_stream" else
                        {"method": method, "compression": {"enabled": method != "full_cache", "budget": 2000}})
    adapter = StreamingBenchAdapter(config["benchmarks"]["streamingbench"])
    samples = {s.sample_id: s for s in adapter.load_samples()}
    if any(s not in samples for s in manifest["sample_ids"]):
        raise RuntimeError("Frozen selection does not match the adapter")
    runtime = ModelRuntime(make_runtime_config(config)).load()
    timing = Timing(runtime.model)
    kv_config = KVMetricConfig(enable_kv_metrics=True, mode="paired", attention_backend="eager",
                               capture_option_logits=True, profile_wallclock=True,
                               detailed_trajectory=False, scope="hermes_managed", summary_token_mode="identity_only")
    instrumentation = KVInstrumentation(kv_config)
    instrumentation.install(runtime.model, _load_chunk_module(runtime.backend))
    captured, set_targets = _wrap_runtime_events(runtime, instrumentation, True)
    atomic_json(directory / "config.json", config)
    atomic_json(directory / "environment.json", {"python": sys.version, "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "attention_backend": runtime.attention_backend,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "created_at": now()})
    for sid in pending:
        try:
            runtime.reset()
            instrumentation.reset_collected()
            set_targets({})
            timing.reset()
            start = timing.clock()
            rows = adapter.run_sample(runtime, samples[sid])
            if len(rows) != 1 or len(instrumentation.collected_events) != 1:
                raise RuntimeError("Expected one official answer and one query anchor")
            row = rows[0]
            artifact = instrumentation.collected_events[0]
            if len(artifact.layers) != runtime.model.qa_model.num_layers:
                raise RuntimeError("Incomplete attention capture")
            instrumentation.tracker.assert_sync(runtime.model.qa_model.kv_cache)
            elapsed = timing.clock() - start
            answer = captured.get(artifact.event_id, {})
            info_events = []
            if method == "infokv_stream":
                stats = runtime.model.infokv_stats()
                info_events = [e.as_dict() for e in runtime.model.infokv_events()]
                triggers = stats["compression_trigger_count"]
                effective = stats["effective_compaction_count"]
                if effective != artifact.effective_compaction_count:
                    raise RuntimeError("InfoKV count differs from physical compaction observation")
                if any(n - timing.prefix > 2000 for n in timing.lengths):
                    raise RuntimeError("InfoKV exceeds query-time budget")
                if triggers != len(info_events) or effective != sum(e["effective"] for e in info_events):
                    raise RuntimeError("InfoKV event accounting mismatch")
            else:
                triggers = artifact.compaction_index
                effective = artifact.effective_compaction_count
            metrics = {"stream_processing_sec": timing.stream_sec, "qa_sec": timing.qa_sec,
                       "stream_fps": timing.frames / timing.stream_sec if timing.stream_sec else None,
                       "frames": timing.frames, "total_with_diagnostics_sec": elapsed,
                       "peak_gpu_allocated_mb_with_diagnostics": torch.cuda.max_memory_allocated() / 1024**2,
                       "kv_bytes_at_query": artifact.kv_bytes, "cache_lengths_at_query": timing.lengths,
                       "protected_prefix_tokens": timing.prefix,
                       "compression_trigger_count": triggers, "effective_compaction_count": effective,
                       "physical_pruning_call_count": artifact.compaction_index,
                       "physical_pruning_total_sec": sum(e.get("wall_clock_sec", 0) for e in instrumentation.collected_compactions),
                       "infokv_management_total_sec": sum(e["wallclock_ms"] for e in info_events)/1000 if info_events else None}
            if method == "full_cache":
                paired = compute_paired_metrics(artifact, artifact, kv_config)
                if abs(paired["ram_mean"] - 1) > 1e-5 or abs(paired["aore_mean"]) > 1e-6:
                    raise RuntimeError("Full/full sanity failed")
                dest = args.output / "full_artifacts" / f"{sid}.pt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Values originate in BF16; storing in BF16 preserves them exactly.
                for layer in artifact.layers.values():
                    layer.value_states = layer.value_states.to(torch.bfloat16)
                torch.save(artifact, dest.with_suffix(".tmp"))
                dest.with_suffix(".tmp").replace(dest)
            else:
                full = torch.load(args.output / "full_artifacts" / f"{sid}.pt", map_location="cpu", weights_only=False)
                if artifact.event_id != full.event_id or artifact.query_input_ids != full.query_input_ids:
                    raise RuntimeError("Query pairing mismatch")
                paired = compute_paired_metrics(artifact, full, kv_config)
                full_record = json.loads((args.output / "full_cache/samples" / f"{sid}.json").read_text())["record"]
                if effective == 0:
                    if abs(paired["ram_mean"] - 1) > 1e-5 or paired["aore_mean"] > 1e-4 or paired["js"] > 1e-5:
                        raise RuntimeError("No-compaction fidelity control failed")
                    if row["prediction"] != full_record["prediction"]:
                        raise RuntimeError("No-compaction answer differs from Full")
                paired["full_prediction"] = full_record["prediction"]
                paired["full_correct"] = full_record["correct"]
                del full
            for key in ("ram_mean", "aore_mean", "js", "kl"):
                if paired.get(key) is None or not math.isfinite(paired[key]):
                    raise RuntimeError(f"Missing or nonfinite {key}")
            record = {**row, "correct": _official_correct(adapter, row), "raw_output": timing.raw_answer,
                      "method": method, "budget": None if method == "full_cache" else 2000,
                      "contract_id": manifest["contract_id"], "finished_at": now(), "status": "success",
                      "runtime": metrics, "option_token_logits": answer.get("option_token_logits"),
                      "option_token_ids": answer.get("option_token_ids"), "fidelity": paired}
            atomic_json(directory / "samples" / f"{sid}.json", {"record": record,
                        "physical_compactions": instrumentation.collected_compactions, "infokv_events": info_events})
            completed = len(list((directory / "samples").glob("*.json")))
            atomic_json(directory / "progress.json", {"completed": completed, "expected": 100,
                        "method": method, "last_sample": sid, "updated_at": now(), "pid": os.getpid()})
            print(f"[completed] {method} {completed}/100 {sid} correct={record['correct']} "
                  f"k={effective} RAM={paired['ram_mean']:.5f} AORE={paired['aore_mean']:.5f} sec={elapsed:.2f}", flush=True)
            instrumentation.reset_collected()
            del artifact, paired
            runtime.reset()
            gc.collect()
        except Exception as error:
            atomic_json(directory / "last_failure.json", {"sample_id": sid, "error": str(error),
                        "traceback": traceback.format_exc(), "updated_at": now()})
            raise


def supervise(args, manifest):
    import fcntl
    lock = (args.output / "supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("StreamingBench diagnostic queue is already running; see status.json.")
        return
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    for pilot in (True, False):
        for method in METHODS:
            label = ("pilot_" if pilot else "all_") + method
            command = [sys.executable, "-u", "-m", "InfoKVStream.run_streamingbench_rt", "worker",
                       "--output", str(args.output), "--method", method]
            if pilot:
                command.append("--pilot")
            atomic_json(args.output / "status.json", {"status": "running", "job": label,
                        "pid": os.getpid(), "updated_at": now()})
            with (logs / f"{label}.log").open("a") as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(args.output / "worker.json", {"pid": proc.pid, "job": label, "started_at": now()})
                rc = proc.wait()
            if rc:
                atomic_json(args.output / "status.json", {"status": "failed", "job": label,
                            "exit_code": rc, "updated_at": now()})
                raise RuntimeError(f"{label} failed; inspect {logs / (label + '.log')}")
        if pilot:
            for method in METHODS:
                records = [json.loads((args.output / method / "samples" / f"{sid}.json").read_text())["record"] for sid in PILOT_IDS]
                if method != "full_cache" and not any(r["runtime"]["effective_compaction_count"] > 0 for r in records):
                    raise RuntimeError(f"No effective compaction in {method} pilot")
            atomic_json(args.output / "pilot_gate.json", {"passed": True, "sample_ids": PILOT_IDS, "checked_at": now()})
    atomic_json(args.output / "status.json", {"status": "complete", "updated_at": now()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "worker", "all"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--method", choices=METHODS, default="infokv_stream")
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    manifest = prepare(args.output)
    if args.command == "worker":
        worker(args, manifest)
    elif args.command == "all":
        supervise(args, manifest)
    else:
        print(json.dumps({"n_videos": len(manifest["video_ids"]), "n_queries": len(manifest["sample_ids"]),
                          "contract_id": manifest["contract_id"]}, indent=2))


if __name__ == "__main__":
    main()
