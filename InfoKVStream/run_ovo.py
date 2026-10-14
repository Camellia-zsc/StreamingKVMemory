"""Resumable OVO InfoKV experiment, using the existing official adapter/runtime.

Run with the hermes-qwen Python and CUDA_VISIBLE_DEVICES=1. The supervisor only
advances from pilot to full after every budget passes the saved pilot gate.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/InfoKVStream/ovo_20261013"
BUDGETS = [6000, 2000, 1000, 500, 200, 100, 50]
GROUPS = {"backward": ["EPM", "ASI", "HLD"],
          "realtime": ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]}
SOURCES = ["InfoKVStream/run_ovo.py", "evaluation_pipeline/runtime.py",
           "OVO-Bench/models/Qwen25VL_Stream.py", "OVO-Bench/models/MMDuet2_HERMES.py",
           "HERMES/inference/qwenvl_hermes_fa2.py", "HERMES/inference/reindex_3d.py"]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    append_rows(path, [value])


def append_rows(path, values):
    with Path(path).open("a", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def fingerprints():
    paths = [ROOT / name for name in SOURCES]
    paths += sorted((ROOT / "evaluation_pipeline/memory").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def memory_config(budget, *, layerwise=False):
    recent = max(1, budget // 5)
    return {
        "method": "infokv_stream", "total_visual_budget": budget,
        "recent": {"enabled": True, "budget": recent},
        "long_term": {"budget": budget - recent},
        "compression": {"interval_chunks": 1, "summary_strategy": "none"},
        "scoring": {
            "temporal_novelty": {"enabled": True, "weight": 0.35,
                                 "reference": "recent_mean", "reference_chunks": 4},
            "layer_evolution": {"enabled": True, "weight": 0.35,
                                 "layerwise": bool(layerwise)},
            "v2v": {"enabled": True, "weight": 0.30,
                    "observation_tokens": min(32, recent)},
        },
    }


def select_pilot(items):
    """One short video per category plus a >=160s video per group; no labels."""
    out = {}
    for group, tasks in GROUPS.items():
        selected = []
        for task in tasks:
            candidates = [r for r in items if r["task"] == task and float(r["realtime"]) >= 12]
            chosen = min(candidates, key=lambda r: (float(r["realtime"]), int(r["id"])))
            selected.append(str(chosen["id"]))
        long = [r for r in items if r["task"] in tasks and float(r["realtime"]) >= 160]
        chosen = min(long, key=lambda r: (float(r["realtime"]), int(r["id"])))
        selected.append(str(chosen["id"]))
        out[group] = list(dict.fromkeys(selected))
    return out


def prepare(output):
    import yaml

    output.mkdir(parents=True, exist_ok=True)
    items = json.loads((ROOT / "OVO-Bench/data/ovo_bench_new.json").read_text())
    base = yaml.safe_load((ROOT / "configs/eval_qwen25vl_ovo.yaml").read_text())
    source = fingerprints()
    contract = {"base_config": base, "memories": {str(b): memory_config(b) for b in BUDGETS},
                "source_sha256": source,
                "annotation_sha256": hashlib.sha256((ROOT / base["benchmarks"]["ovobench"]["annotation"]).read_bytes()).hexdigest()}
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    path = output / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["contract_id"] != contract_id:
            raise RuntimeError("Experiment source/config changed; use a new output root to preserve comparability")
        return saved
    ids = {g: [str(r["id"]) for r in items if r["task"] in ts] for g, ts in GROUPS.items()}
    selected = set(sum(ids.values(), []))
    missing = [str(ROOT / base["benchmarks"]["ovobench"]["chunked_dir"] / (sid + ".mp4"))
               for sid in selected if not (ROOT / base["benchmarks"]["ovobench"]["chunked_dir"] / (sid + ".mp4")).is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} videos: {missing[:5]}")
    manifest = {**contract, "contract_id": contract_id, "created_at": now(),
                "python": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "budgets": BUDGETS, "groups": GROUPS, "pilot_ids": select_pilot(items),
                "full_ids": ids, "pilot_selection": "shortest >=12s per category plus shortest >=160s per group",
                "nominal_budget": "visual tokens per layer; excludes protected prompt prefix",
                "compression_interval_chunks": 1,
                "notes": ["InfoKV uses one global keep set; HERMES may keep different positions/summary tokens by layer.",
                          "Prefill may transiently exceed budget; record peak and enforce budget at every query.",
                          "Existing RT Full has missing samples; compare Full only on common successful IDs."]}
    atomic_json(path, manifest)
    for relative in source:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    for b in BUDGETS:
        cfg = copy.deepcopy(base)
        cfg["memory"] = memory_config(b)
        atomic_json(output / "configs" / f"budget_{b}.json", cfg)
    return manifest


def read_predictions(path):
    if not Path(path).exists():
        return {}
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def rebuild_predictions(directory):
    # Per-sample atomic files are the commit record; recover after interruption.
    rows = [json.loads(p.read_text())["record"] for p in sorted((directory / "samples").glob("*.json"))]
    path = directory / "predictions.jsonl"
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    temp.replace(path)
    return {str(row["sample_id"]): row for row in rows}


class Observer:
    """Read-only hooks around the official stream and question entry points."""
    def __init__(self, model):
        import torch
        self.torch = torch
        self.model = model
        stream = model.stream_video_chunk
        ask = model.ask_question

        def stream_measured(frames, **kwargs):
            start = self.clock()
            result = stream(frames, **kwargs)
            self.stream_seconds += self.clock() - start
            self.frames += len(frames)
            return result

        def ask_measured(*args, **kwargs):
            cache = model.qa_model.kv_cache
            self.cache_lengths = [int(k.shape[2]) for k, _ in cache]
            self.kv_bytes = sum(t.numel() * t.element_size() for kv in cache for t in kv)
            prefix = int(model.qa_model.visual_start_idx)
            budget = model._integrator.memory.total_budget
            if any(n - prefix > budget for n in self.cache_lengths):
                raise RuntimeError(f"InfoKV exceeds budget at query: {self.cache_lengths}, prefix={prefix}, B={budget}")
            if len(set(self.cache_lengths)) != 1:
                raise RuntimeError("InfoKV global keep set produced unequal per-layer lengths")
            if self.cache_lengths[0] != prefix + len(model._integrator.ledger):
                raise RuntimeError("InfoKV token ledger disagrees with actual cache")
            self.question_start = self.clock()
            self.armed = True
            try:
                self.raw_answer = ask(*args, **kwargs)
                return self.raw_answer
            finally:
                self.question_seconds = self.clock() - self.question_start
                self.armed = False

        def logits_observed(module, args, output):
            if self.armed:
                if self.first_token_seconds is None:
                    self.first_token_seconds = self.clock() - self.question_start
                self.decode_steps += 1

        model.stream_video_chunk = stream_measured
        model.ask_question = ask_measured
        self.handle = model.qa_model.lm_head.register_forward_hook(logits_observed)
        self.reset()

    def clock(self):
        self.torch.cuda.synchronize()
        return time.perf_counter()

    def reset(self):
        self.stream_seconds = self.question_seconds = 0.0
        self.first_token_seconds = None
        self.frames = self.decode_steps = self.kv_bytes = 0
        self.cache_lengths = []
        self.raw_answer = None
        self.armed = False
        self.torch.cuda.reset_peak_memory_stats()

    def metrics(self, elapsed):
        m = self.model._integrator.memory
        events = self.model.infokv_events()
        stats = m.get_stats().as_dict()
        compression_sec = sum(e.wallclock_ms for e in events) / 1000
        signals = [x for r in m.managed_tokens() for x in (r.novelty_raw, r.evolution_raw, r.v2v_raw)]
        if not all(math.isfinite(x) for x in signals):
            raise RuntimeError("Non-finite memory signals")
        return {
            **stats, "compression_total_sec": compression_sec,
            "mean_compression_sec": compression_sec / len(events) if events else 0.0,
            "question_to_first_token_sec": self.first_token_seconds,
            "first_token_metric": "question start to first lm_head logits, including question prefill",
            "question_answer_sec": self.question_seconds,
            "decode_steps_including_first": self.decode_steps,
            "tpot_sec": ((self.question_seconds - self.first_token_seconds) / (self.decode_steps - 1)
                         if self.decode_steps > 1 and self.first_token_seconds is not None else None),
            "stream_processing_sec": self.stream_seconds,
            "stream_fps": self.frames / self.stream_seconds if self.stream_seconds else None,
            "frames_observed": self.frames,
            "peak_gpu_memory_mb": self.torch.cuda.max_memory_allocated() / 1024**2,
            "kv_bytes_at_query": self.kv_bytes,
            "kv_lengths_at_query": self.cache_lengths,
            "protected_prefix_tokens": self.model._integrator.prefix_len,
            "managed_visual_tokens_at_query": len(m.managed_tokens()),
            "total_seen_visual_tokens": m.total_seen_tokens,
            "retention_ratio": len(m.managed_tokens()) / m.total_seen_tokens if m.total_seen_tokens else None,
            "total_wallclock_sec": elapsed,
        }


def worker(args, manifest):
    import torch
    import transformers
    from evaluation_pipeline.adapters import OVOBenchAdapter
    from evaluation_pipeline.evaluate import make_runtime_config
    from evaluation_pipeline.memory import resolve_memory_config
    from evaluation_pipeline.runtime import ModelRuntime

    # Keep small NumPy attention matmuls from spawning dozens of CPU threads.
    torch.set_num_threads(4)
    adapter = OVOBenchAdapter(manifest["base_config"]["benchmarks"]["ovobench"])
    samples = {s.sample_id: s for s in adapter.load_samples()}
    runtime = None
    for budget in args.budgets:
        for group in args.groups:
            directory = args.output / args.stage / group / f"budget_{budget}"
            directory.mkdir(parents=True, exist_ok=True)
            done = rebuild_predictions(directory)
            expected = manifest[f"{args.stage}_ids"][group]
            if args.stage == "full":
                pilot_dir = args.output / "pilot" / group / f"budget_{budget}" / "samples"
                for sid in expected:
                    p = pilot_dir / f"{sid}.json"
                    if sid not in done and p.exists():
                        bundle = json.loads(p.read_text())
                        if bundle["record"]["status"] == "success":
                            bundle["record"]["reused_from"] = "identical pilot configuration"
                            atomic_json(directory / "samples" / p.name, bundle)
                done = rebuild_predictions(directory)
            pending = [sid for sid in expected if sid not in done or done[sid]["status"] != "success"]
            config = copy.deepcopy(manifest["base_config"])
            config["memory"] = manifest["memories"][str(budget)]
            atomic_json(directory / "config.json", config)
            if pending:
                if runtime is None:
                    runtime = ModelRuntime(make_runtime_config(config)).load()
                    observer = Observer(runtime.model)
                    atomic_json(args.output / "environment.json", {
                        "python": sys.version, "executable": sys.executable,
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "attention_backend": getattr(runtime, "attention_backend", None),
                        "model_dtype": str(next(runtime.model.qa_model.parameters()).dtype),
                        "recorded_at": now()})
                else:
                    runtime.model.attach_infokv(resolve_memory_config(config))
                for sid in pending:
                    observer.reset()
                    start = observer.clock()
                    try:
                        rows = adapter.run_sample(runtime, samples[sid])
                        if len(rows) != 1:
                            raise RuntimeError("Expected exactly one OVO BW/RT answer")
                        elapsed = observer.clock() - start
                        row = {**rows[0], "raw_output": observer.raw_answer,
                               "status": "success", "budget": budget, "group": group,
                               "memory_method": "infokv_stream", "stage": args.stage,
                               "contract_id": manifest["contract_id"], "finished_at": now(),
                               "runtime": observer.metrics(elapsed),
                               "option_logits_by_letter": runtime.model.option_logits_by_letter()}
                        events = [e.as_dict() for e in runtime.model.infokv_events()]
                        atomic_json(directory / "samples" / f"{sid}.json", {"record": row, "events": events})
                        done[sid] = row
                        append_json(directory / "predictions.jsonl", row)
                        append_rows(directory / "compactions.jsonl", (
                            {"sample_id": sid, "budget": budget,
                             "finished_at": row["finished_at"], **event} for event in events))
                        print(f"[completed] {args.stage}/{group}/{budget} {len(done)}/{len(expected)} "
                              f"id={sid} pred={row['prediction']} gold={row['ground_truth']} "
                              f"k={row['runtime']['effective_compaction_count']} sec={elapsed:.2f}", flush=True)
                    except Exception as error:
                        failure = {"sample_id": sid, "task_type": samples[sid].questions[0].task_type,
                                   "status": "failed", "budget": budget, "group": group,
                                   "error": str(error), "error_type": type(error).__name__,
                                   "traceback": traceback.format_exc(), "finished_at": now()}
                        atomic_json(directory / "last_failure.json", failure)
                        append_json(directory / "failures.jsonl", failure)
                        raise
                    atomic_json(directory / "progress.json", {
                        "expected": len(expected), "completed": len(done), "updated_at": now(),
                        "status": "complete" if len(done) == len(expected) else "running",
                        "pid": os.getpid()})
                    if args.stage == "full" and len(done) % 50 == 0:
                        analyze(args.output, args.stage)
            atomic_json(directory / "progress.json", {"expected": len(expected), "completed": len(done),
                        "updated_at": now(), "status": "complete", "pid": os.getpid()})
            analyze(args.output, args.stage)


def analyze(output, stage):
    script = ROOT / "InfoKVStream/analyze_ovo.py"
    if script.exists():
        subprocess.run([sys.executable, str(script), "--root", str(output), "--stage", stage], check=True)


def check_pilot(output, manifest):
    checks = []
    for group, ids in manifest["pilot_ids"].items():
        for b in BUDGETS:
            rows = read_predictions(output / "pilot" / group / f"budget_{b}" / "predictions.jsonl")
            complete = all(s in rows and rows[s]["status"] == "success" for s in ids)
            good = complete and all(rows[s]["runtime"]["managed_visual_tokens_at_query"] <= b for s in ids)
            compressed = complete and any(rows[s]["runtime"]["effective_compaction_count"] > 0 for s in ids)
            checks.append({"group": group, "budget": b, "expected": len(ids),
                           "complete": complete, "budget_ok": good, "effective_compaction_seen": compressed,
                           "pass": complete and good and compressed})
    result = {"passed": all(c["pass"] for c in checks), "checked_at": now(), "checks": checks}
    atomic_json(output / "pilot_gate.json", result)
    return result["passed"]


def supervise(args, manifest):
    import fcntl
    lock = (args.output / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logdir = args.output / "logs"
    logdir.mkdir(exist_ok=True)
    jobs = [("pilot", BUDGETS, list(GROUPS))]
    jobs += [("full", [b], [g]) for b in BUDGETS for g in GROUPS]
    for stage, budgets, groups in jobs:
        if stage == "full" and not check_pilot(args.output, manifest):
            raise RuntimeError("Pilot gate failed; full evaluation was not started")
        label = stage + "_" + "_".join(groups) + "_" + "_".join(map(str, budgets))
        command = [sys.executable, "-u", "-m", "InfoKVStream.run_ovo", "worker",
                   "--output", str(args.output), "--stage", stage, "--budgets", *map(str, budgets),
                   "--groups", *groups]
        atomic_json(args.output / "status.json", {"status": "running", "job": label,
                    "pid": os.getpid(), "command": command, "updated_at": now()})
        with (logdir / f"{label}.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(args.output / "worker.json", {"pid": child.pid, "job": label, "started_at": now()})
            rc = child.wait()
        if rc:
            atomic_json(args.output / "status.json", {"status": "failed", "job": label,
                        "exit_code": rc, "updated_at": now(), "log": str(logdir / f"{label}.log")})
            raise RuntimeError(f"Worker {label} failed; inspect its log. Full queue is stopped.")
    atomic_json(args.output / "status.json", {"status": "complete", "updated_at": now()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "worker", "all", "check-pilot"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    p.add_argument("--budgets", nargs="+", type=int, choices=BUDGETS, default=BUDGETS)
    p.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = p.parse_args()
    os.chdir(ROOT)
    manifest = prepare(args.output)
    if args.command == "prepare":
        print(json.dumps({"full_counts": {k: len(v) for k, v in manifest['full_ids'].items()},
                          "pilot_ids": manifest['pilot_ids'], "contract_id": manifest['contract_id']}, indent=2))
    elif args.command == "worker":
        worker(args, manifest)
    elif args.command == "all":
        supervise(args, manifest)
    elif not check_pilot(args.output, manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
