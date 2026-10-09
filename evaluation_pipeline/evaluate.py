import argparse
import copy
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict

from .adapters import BENCHMARK_REGISTRY
from .models.qwen25_vl import (
    resolve_backend,
    resolve_mcq_decode_mode,
    resolve_model_path,
)
from .runtime import ModelRuntime, Profiler, cleanup_after_error
from .storage import append_jsonl, read_jsonl_recover, safe_float, write_json, write_summary_csv


LOG = logging.getLogger("evaluate_all")


# Method -> human/model label written into prediction records. The HERMES and
# Full-KV labels are intentionally unchanged so existing result files / resume
# state stay byte-identical; only the new method adds a label.
MEMORY_METHOD_LABELS = {
    "hermes": "MMDuet2-HERMES",
    "full_cache": "MMDuet2-FullKV",
    "infokv_stream": "InfoKVStream",
}


def memory_method_label(method: str) -> str:
    return MEMORY_METHOD_LABELS.get(method, f"Memory-{method}")


def load_config(path: str) -> Dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        import json
        return json.loads(text)


def configure_logging(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")],
    )


def apply_defaults(config):
    config.setdefault("model", {})
    config.setdefault("inference", {})
    config.setdefault("memory", {})
    config.setdefault("benchmarks", {})
    config.setdefault("output", {})
    config["memory"].setdefault("method", "full_cache")
    config["output"].setdefault("root", "outputs/evaluation")
    config["output"].setdefault("resume", True)
    config["output"].setdefault("save_raw_prediction", True)
    config["inference"].setdefault("max_new_tokens", 16)
    # OVO Backward/Realtime MCQ ablation switches. The defaults reproduce the
    # official path (mandatory-reply prefix appended, first generated token
    # constrained to an option letter); they only affect OVO reply_required
    # MCQ questions.
    config.setdefault("ovo", {})
    config["ovo"].setdefault("must_reply", True)
    config["ovo"].setdefault("constrain_mcq_first_token", True)
    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("fail_on_missing_video", True)
    # Backend selection defaults to the legacy MMDuet2 so existing configs are
    # byte-identical; new configs opt in with ``model.backend: qwen2_5_vl``.
    config["model"].setdefault("backend", resolve_backend(config["model"]))
    resolved_path = resolve_model_path(config["model"])
    if resolved_path and not config["model"].get("model_path"):
        config["model"]["model_path"] = resolved_path
    config.setdefault("mcq_decode_mode", resolve_mcq_decode_mode(config))
    return config


def make_runtime_config(config):
    model = copy.deepcopy(config["model"])
    model["memory"] = config["memory"]
    model["inference"] = config["inference"]
    return model


def make_run_id(config, benchmark_name, benchmark_config):
    inference = config.get("inference", {})
    memory = config.get("memory", {})
    identity = {
        "model_name": config.get("model", {}).get("name", "MMDuet2"),
        "model_path": config.get("model", {}).get("model_path"),
        "benchmark": benchmark_name,
        "memory_method": memory.get("method", "full_cache"),
        "memory_budget": memory.get("compression", {}).get("budget"),
        "sample_fps": inference.get("sample_fps"),
        "frames_per_turn": inference.get("frames_per_turn"),
        "max_side": inference.get("max_side", 336),
        "offline_fps": inference.get("offline_fps"),
        "offline_max_frames": inference.get("offline_max_frames"),
        "offline_max_side": inference.get("offline_max_side", 336),
        "max_new_tokens": inference.get("max_new_tokens"),
        "context_mode": benchmark_config.get("context_mode"),
        "context_time": benchmark_config.get("context_time"),
        "proactive_protocol": benchmark_config.get("proactive_protocol"),
        "input_modality": benchmark_config.get("input_modality"),
        "annotation": benchmark_config.get("annotation"),
        "annotations": benchmark_config.get("annotations"),
        "archive_map": benchmark_config.get("archive_map"),
        "allowed_archives": benchmark_config.get("allowed_archives"),
    }
    # Only surface the new backend identity when it actually deviates, so old
    # MMDuet2 run_ids / resume state remain byte-identical.
    backend = resolve_backend(config.get("model", {}))
    if backend != "mmduet2":
        identity["model_backend"] = backend
        identity["model_path"] = config.get("model", {}).get("model_path") or identity.get(
            "model_path"
        )
        mode = resolve_mcq_decode_mode(config)
        if mode != "official":
            identity["mcq_decode_mode"] = mode
    if benchmark_name == "ovobench":
        ovo = config.get("ovo", {}) or {}
        must_reply = bool(ovo.get("must_reply", True))
        constrained = bool(ovo.get("constrain_mcq_first_token", True))
        # Keep the official default identity byte-identical so existing
        # run_ids / resume state stay valid. Only surface the ablation
        # switches when they deviate, so each of the A/B/C configs gets its
        # own run_id and cannot silently resume from another config's rows.
        if not (must_reply and constrained):
            identity["ovo_must_reply"] = must_reply
            identity["ovo_mcq_constrained"] = constrained
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], identity


def efficiency_metrics(predictions):
    runtimes = [
        item.get("runtime", {})
        for item in predictions
        if item.get("status") in {"ok", "error"}
    ]
    def mean(key):
        values = [float(item[key]) for item in runtimes if item.get(key) is not None]
        return sum(values) / len(values) if values else None
    return {
        "mean_latency": mean("latency"),
        "peak_gpu_memory_mb": max(
            [float(item["peak_gpu_memory_mb"]) for item in runtimes
             if item.get("peak_gpu_memory_mb") is not None],
            default=None,
        ),
        "peak_kv_tokens": max(
            [float(item["peak_kv_tokens"]) for item in runtimes
             if item.get("peak_kv_tokens") is not None],
            default=None,
        ),
        "compression_events": sum(
            int(item.get("compression_events", 0)) for item in runtimes
        ),
        "total_removed_tokens": sum(
            int(item.get("total_removed_tokens", 0)) for item in runtimes
        ),
        "mean_kv_retention_ratio": mean("mean_kv_retention_ratio"),
        "mean_kv_compression_ratio": mean("mean_kv_compression_ratio"),
    }


def select_samples(samples, max_samples=None, start_idx=0, end_idx=None,
                   strategy="ordered", seed=0):
    """Select samples deterministically, optionally by random order."""
    indexed = list(enumerate(samples))
    if strategy == "random":
        random.Random(seed).shuffle(indexed)
    elif strategy == "stratified":
        rng = random.Random(seed)
        groups = {}
        for index, sample in indexed:
            task = str(sample.metadata.get("task", "")).upper()
            if task in {"EPM", "ASI", "HLD"}:
                group = "backward"
            elif task in {"REC", "SSR", "CRR"}:
                group = "forward"
            elif task:
                group = "realtime"
            else:
                group = "default"
            groups.setdefault(group, []).append((index, sample))
        for values in groups.values():
            rng.shuffle(values)
        indexed = []
        while groups:
            for group in list(groups):
                values = groups[group]
                if values:
                    indexed.append(values.pop())
                if not values:
                    del groups[group]
    elif strategy != "ordered":
        raise ValueError(f"Unknown sample strategy: {strategy}")

    selected = []
    for index, sample in indexed:
        if strategy == "ordered":
            if index < start_idx:
                continue
            if end_idx is not None and index >= end_idx:
                break
        if max_samples is not None and len(selected) >= max_samples:
            break
        selected.append(sample)
    return selected


def run(
    config,
    benchmarks,
    max_samples=None,
    start_idx=0,
    end_idx=None,
    resume=None,
    output_dir=None,
    dry_run=False,
    sample_strategy="ordered",
    seed=0,
):
    config = apply_defaults(config)
    selected = benchmarks or [name for name, item in config["benchmarks"].items() if item.get("enabled", True)]
    root = Path(output_dir or config["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    configure_logging(root / "evaluation.log")
    write_config = copy.deepcopy(config)
    write_json(root / "config.json", write_config)

    runtime = None
    summary = {}
    profiler = Profiler()

    for name in selected:
        if name not in BENCHMARK_REGISTRY:
            raise ValueError(f"Unknown benchmark {name!r}; registered={sorted(BENCHMARK_REGISTRY)}")
        bench_config = copy.deepcopy(config["benchmarks"].get(name, {}))
        if name == "ovobench":
            # Surface the top-level `ovo:` ablation switches to the adapter.
            # A benchmark-local key wins if one is present.
            bench_config.setdefault(
                "must_reply",
                bool(config.get("ovo", {}).get("must_reply", True)),
            )
            bench_config.setdefault(
                "constrain_mcq_first_token",
                bool(config.get("ovo", {}).get("constrain_mcq_first_token", True)),
            )
        adapter = BENCHMARK_REGISTRY[name](bench_config)
        run_id, run_identity = make_run_id(config, name, bench_config)
        prediction_path = root / name / "predictions.jsonl"
        existing = read_jsonl_recover(prediction_path) if (resume if resume is not None else config["output"]["resume"]) else []
        active_memory_method = config["memory"].get("method", "full_cache")
        active_existing = [
            item for item in existing
            if item.get("run_id") == run_id
        ]
        completed = {
            str(item.get("question_id", item.get("sample_id")))
            for item in active_existing
            if item.get("status", "ok") in {"ok", "failed", "error"}
        }
        predictions = list(active_existing)
        LOG.info(
            "Benchmark: %s mode=%s run_id=%s completed=%d",
            name, adapter.mode, run_id, len(completed),
        )
        samples = list(adapter.load_samples())
        selected_samples = select_samples(
            samples,
            max_samples=max_samples,
            start_idx=start_idx,
            end_idx=end_idx,
            strategy=sample_strategy,
            seed=seed,
        )
        if sample_strategy == "random":
            LOG.info(
                "Sampling: benchmark=%s strategy=random seed=%d selected_ids=%s",
                name,
                seed,
                ",".join(sample.sample_id for sample in selected_samples),
            )
        adapter._last_samples = selected_samples
        preflight = adapter.preflight(selected_samples)
        LOG.info(
            "Preflight: benchmark=%s videos=%d samples=%d questions=%d missing_videos=%d",
            name,
            preflight["videos"],
            preflight["samples"],
            preflight["questions"],
            len(preflight["missing_videos"]),
        )
        if preflight["missing_videos"] and config["evaluation"]["fail_on_missing_video"] and not dry_run:
            missing_preview = ", ".join(preflight["missing_videos"][:5])
            raise FileNotFoundError(
                f"{name} has {len(preflight['missing_videos'])} missing primary videos; "
                f"examples: {missing_preview}"
            )
        if dry_run:
            LOG.info(
                "Dry run: benchmark=%s samples=%d questions=%d missing_primary_video=%d",
                name,
                preflight["samples"],
                preflight["questions"],
                len(preflight["missing_videos"]),
            )
            summary[name] = {
                "dry_run": True,
                "samples": preflight["samples"],
                "questions": preflight["questions"],
                "videos": preflight["videos"],
                "missing_primary_video": len(preflight["missing_videos"]),
                "missing_video_examples": preflight["missing_videos"][:10],
            }
            continue
        if runtime is None:
            runtime = ModelRuntime(make_runtime_config(config)).load()
        for sample in selected_samples:
            ids = {q.question_id for q in sample.questions}
            if ids and ids.issubset(completed):
                continue
            LOG.info("Progress: %s sample=%s video=%s", name, sample.sample_id, sample.video_path)
            adapter.config["_skip_question_ids"] = completed
            try:
                if not Path(sample.video_path).exists():
                    raise FileNotFoundError(sample.video_path)
                runtime.reset()
                profiler.start(runtime)
                records = adapter.run_sample(runtime, sample)
                runtime_stats = profiler.stop(runtime)
                for record in records:
                    record.setdefault("status", "ok")
                    record["run_id"] = run_id
                    record["run_identity"] = run_identity
                    record["model_label"] = (
                        memory_method_label(active_memory_method)
                    )
                    record.setdefault("protocol", {})
                    record.setdefault("inference", {
                        "sample_fps": config["inference"].get("sample_fps"),
                        "frames_per_turn": config["inference"].get("frames_per_turn"),
                        "max_side": config["inference"].get("max_side", 336),
                    })
                    record["runtime"] = {
                        key: (
                            safe_float(value)
                            if isinstance(value, (int, float)) or value is None
                            else value
                        )
                        for key, value in runtime_stats.items()
                    }
                    record["memory"] = {
                        "method": config["memory"].get("method"),
                        "budget": config["memory"].get("compression", {}).get("budget"),
                    }
                    append_jsonl(prediction_path, record)
                    predictions.append(record)
                    completed.add(str(record.get("question_id", record.get("sample_id"))))
                runtime.reset()
            except Exception as error:
                profiler.stop()
                runtime.reset()
                cleanup_after_error()
                for question in sample.questions or [None]:
                    question_id = question.question_id if question else sample.sample_id
                    if question_id in completed:
                        continue
                    record = {
                        "benchmark": name,
                        "sample_id": sample.sample_id,
                        "question_id": question_id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "video_path": sample.video_path,
                        "memory": {
                            "method": config["memory"].get("method"),
                            "budget": config["memory"].get("compression", {}).get("budget"),
                        },
                        "run_id": run_id,
                        "run_identity": run_identity,
                        "model_label": (
                            memory_method_label(active_memory_method)
                        ),
                        "protocol": {
                            "context_mode": sample.metadata.get("context_mode"),
                            "proactive_protocol": sample.metadata.get("proactive_protocol"),
                        },
                    }
                    append_jsonl(prediction_path, record)
                    predictions.append(record)
                    completed.add(str(record["question_id"]))
                LOG.exception("Sample failed: %s", sample.sample_id)
        metrics = adapter.evaluate(predictions)
        metrics["accuracy"] = {
            key: value for key, value in metrics.items()
            if key not in {"pipeline", "timing", "efficiency", "accuracy"}
            and isinstance(value, (int, float, type(None), dict))
        }
        metrics["efficiency"] = efficiency_metrics(predictions)
        metrics.setdefault("pipeline", {})
        metrics["pipeline"].update({
            "selected_samples": preflight["samples"],
            "selected_videos": preflight["videos"],
            "selected_questions": preflight["questions"],
            "missing_primary_videos": len(preflight["missing_videos"]),
        })
        write_json(root / name / "metrics.json", metrics)
        summary[name] = metrics

    write_json(root / "summary.json", summary)
    write_summary_csv(root / "summary.csv", summary)
    LOG.info("Evaluation finished: %s", root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run multiple video benchmark evaluations with one model load.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmarks", nargs="+")
    parser.add_argument("--model_config", default=None)
    parser.add_argument(
        "--model_backend",
        choices=["mmduet2", "qwen2_5_vl"],
        help=(
            "Streaming model backend. Defaults to the config's model.backend, "
            "or mmduet2 for legacy configs."
        ),
    )
    parser.add_argument(
        "--mcq_decode_mode",
        choices=["official", "constrained"],
        help=(
            "MCQ decoding for the Qwen2.5-VL backend: official = free decode + "
            "option extraction (default); constrained = hard A-E first-token mask."
        ),
    )
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--memory_method",
        choices=[
            "full_cache",
            "hermes",
            "infokv_stream",
            "sliding_window",
            "streammem",
            "custom_method",
        ],
        help=(
            "Visual-KV memory method. 'hermes' (default) and 'infokv_stream' "
            "are the pluggable CPU-verified methods; the InfoKV-inspired "
            "streaming method's GPU KV integration is a separate stage."
        ),
    )
    parser.add_argument(
        "--memory_budget",
        type=int,
        help="HERMES KV budget; included in run_id and resume identity.",
    )
    parser.add_argument("--offline_fps", type=float, help="Uniform FPS for Video-MME offline video sampling.")
    parser.add_argument("--offline_max_frames", type=int, help="Maximum frames per Video-MME offline question.")
    parser.add_argument(
        "--streaming_context_mode",
        choices=["main_60s", "all_context"],
        help="Override StreamingBench context mode.",
    )
    parser.add_argument(
        "--streaming_context_time",
        type=float,
        help="Override StreamingBench context length in seconds.",
    )
    parser.add_argument(
        "--proactive_protocol",
        choices=["official_polling", "mmduet_native"],
        help="Override StreamingBench proactive execution protocol.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "sdpa", "flash_attention_2"],
        help="Attention backend; auto falls back to SDPA when flash_attn is unavailable.",
    )
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument(
        "--ovo_must_reply",
        choices=["true", "false"],
        help=(
            "OVO Backward/Realtime MCQ ablation: append the mandatory-reply "
            "assistant prefix. Default true (official behavior)."
        ),
    )
    parser.add_argument(
        "--ovo_constrain_mcq_first_token",
        choices=["true", "false"],
        help=(
            "OVO Backward/Realtime MCQ ablation: constrain the first generated "
            "token to an option letter. Default true (official behavior)."
        ),
    )
    parser.add_argument(
        "--sample_strategy",
        choices=["ordered", "random", "stratified"],
        default="ordered",
        help="Select samples in annotation order, random order, or task-family-stratified order.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Validate annotations and video paths without loading the model.")
    args = parser.parse_args(argv)
    if not args.config:
        raise SystemExit("--config is required; use configs/eval_all.yaml")
    config = load_config(args.config)
    if args.model_config:
        model_config = load_config(args.model_config)
        config["model"] = {**config.get("model", {}), **model_config.get("model", model_config)}
    if args.model_backend:
        config.setdefault("model", {})["backend"] = args.model_backend
    if args.mcq_decode_mode:
        config["mcq_decode_mode"] = args.mcq_decode_mode
    if args.memory_method:
        config.setdefault("memory", {})["method"] = args.memory_method
    if args.memory_budget is not None:
        config.setdefault("memory", {}).setdefault("compression", {})[
            "budget"
        ] = args.memory_budget
    if args.offline_fps is not None:
        config.setdefault("inference", {})["offline_fps"] = args.offline_fps
    if args.offline_max_frames is not None:
        config.setdefault("inference", {})["offline_max_frames"] = args.offline_max_frames
    if args.streaming_context_mode:
        config.setdefault("benchmarks", {}).setdefault("streamingbench", {})[
            "context_mode"
        ] = args.streaming_context_mode
    if args.streaming_context_time is not None:
        config.setdefault("benchmarks", {}).setdefault("streamingbench", {})[
            "context_time"
        ] = args.streaming_context_time
    if args.proactive_protocol:
        config.setdefault("benchmarks", {}).setdefault("streamingbench", {})[
            "proactive_protocol"
        ] = args.proactive_protocol
    if args.attn_implementation:
        config.setdefault("inference", {})["attn_implementation"] = args.attn_implementation
    if args.max_new_tokens is not None:
        config.setdefault("inference", {})["max_new_tokens"] = args.max_new_tokens
    if args.ovo_must_reply is not None:
        config.setdefault("ovo", {})["must_reply"] = (
            args.ovo_must_reply == "true"
        )
    if args.ovo_constrain_mcq_first_token is not None:
        config.setdefault("ovo", {})["constrain_mcq_first_token"] = (
            args.ovo_constrain_mcq_first_token == "true"
        )
    run(
        config,
        args.benchmarks,
        args.max_samples,
        args.start_idx,
        args.end_idx,
        args.resume,
        args.output_dir,
        args.dry_run,
        args.sample_strategy,
        args.seed,
    )


if __name__ == "__main__":
    main()
