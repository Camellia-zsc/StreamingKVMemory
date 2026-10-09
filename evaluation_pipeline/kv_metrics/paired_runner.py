"""Two-pass KV fidelity evaluation runner.

Pass A (compressed): normal MMDuet2 + HERMES.
Pass B (full): the *same* HERMES runtime with compaction disabled.

Both passes use identical weights, frames, query schedule, preprocessing and
greedy decoding. Events are paired by a stable ``event_id``
(``video_id | question_id | query_time``), never by Python object identity.

This runner is opt-in and never mutates benchmark answer files.
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..adapters import BENCHMARK_REGISTRY
from ..evaluate import apply_defaults, load_config, make_runtime_config
from ..runtime import ModelRuntime, cleanup_after_error
from .aggregate import MetricAggregator, compute_paired_metrics
from .config import KVMetricConfig, add_cli_arguments, parse_temporal_bins
from .hooks import KVInstrumentation
from .runtime_breakdown import RuntimeBreakdown
from .system_metrics import is_effective_compaction


LOG = logging.getLogger("kv_metrics")


def event_id_for(video_path: str, question: dict) -> str:
    video_id = video_key_for(video_path)
    question_id = question.get("question_id")
    if question_id is None:
        sample_id = question.get("sample_id")
        test_index = question.get("test_index")
        question_id = (
            f"{sample_id}:{test_index}" if test_index is not None else str(sample_id)
        )
    query_time = question.get("timestamp")
    return f"{video_id}|{question_id}|{query_time}"


def video_key_for(video_path) -> str:
    """Stable per-video key.

    OVO clips are ``<id>.mp4`` (stem is the id) while StreamingBench clips are
    ``sample_<id>/video.mp4`` (stem is the useless literal "video", so use the
    parent directory).
    """

    path = Path(str(video_path))
    if path.name == "video.mp4" and path.parent.name:
        return path.parent.name
    return path.stem


def _wrap_runtime_events(runtime: ModelRuntime, instrumentation: KVInstrumentation,
                         capture_option_logits: bool = False,
                         diagnostic_ids=None):
    """Wrap the runtime's run_* entry points to tag stable event ids and
    capture the produced answer (for task correlation)."""

    captured: Dict[str, dict] = {}
    state = {"targets": {}}
    diagnostic_ids = {str(v) for v in (diagnostic_ids or []) if v is not None}

    def _is_diagnostic(question) -> bool:
        if not diagnostic_ids:
            return True
        token = question.get("sample_id")
        if token is None:
            token = question.get("video_id")
        return str(token) in diagnostic_ids

    def set_targets(targets: Dict[tuple, dict]) -> None:
        """``targets`` maps source-video stem -> probe metadata.

        Only the q_star query of that video gets an expensive detailed
        fidelity trajectory. Keying by video (not by benchmark sample) matters
        for benchmarks that emit one sample per question (StreamingBench).
        """

        state["targets"] = dict(targets or {})
        instrumentation.clear_probe()

    def make(name):
        original = getattr(runtime, name)

        def wrapped(video_path, prompt, question):
            eid = event_id_for(video_path, question)
            query_id = question.get("question_id")
            if query_id is None:
                sample_id = question.get("sample_id")
                test_index = question.get("test_index")
                query_id = (
                    f"{sample_id}:{test_index}"
                    if test_index is not None
                    else str(sample_id)
                )
            instrumentation.set_pending_event(
                eid,
                query_id,
                question.get("timestamp"),
                sample_id=question.get("sample_id"),
            )
            target = state["targets"].get(
                video_key_for(video_path)
            )
            is_target = target is not None and (
                _round_time(target.get("query_time"))
                == _round_time(question.get("timestamp"))
            )
            if is_target and not target.get("_armed"):
                target["_armed"] = True
                instrumentation.set_probe_query(
                    target.get("sample_id"),
                    target.get("query_id"),
                    target.get("query_time"),
                )
                instrumentation.set_probe_target(eid)
                if target.get("probe_time_plan"):
                    instrumentation.set_probe_time_plan(target["probe_time_plan"])
            # read-only decision-layer observation for EVERY task query
            if capture_option_logits and _is_diagnostic(question):
                option_map = runtime._mcq_option_token_ids_map(question)
                instrumentation.arm_option_capture(option_map)
            answer = original(video_path, prompt, question)
            option_logits = instrumentation.take_option_capture()
            option_ids = instrumentation.option_token_ids
            captured[eid] = {
                "question_id": query_id,
                "task": question.get("task"),
                "answer": answer,
                "answer_gold": question.get("answer"),
                "option_token_logits": option_logits,
                "option_token_ids": option_ids,
            }
            return answer

        setattr(runtime, name, wrapped)

    for name in ("run_ovo", "run_streaming_mcq", "run_streaming_text"):
        if hasattr(runtime, name):
            make(name)
    return captured, set_targets


def _official_correct(adapter, record) -> Optional[bool]:
    """Correctness computed by the *benchmark's own* evaluator.

    Reuses ``BenchmarkAdapter.evaluate`` on a single record so kv_metrics never
    re-implements the MCQ / REC / SSR answer-normalisation rules.
    """

    if record is None:
        return None
    try:
        result = adapter.evaluate([dict(record)])
    except Exception as error:  # noqa: BLE001
        LOG.warning("kv_metrics: official evaluate() failed: %s", error)
        return None
    tasks = (result or {}).get("tasks") or {}
    task = record.get("task_type")
    if task in tasks and tasks[task] is not None:
        return bool(tasks[task] >= 0.5)
    overall = (result or {}).get("overall")
    if overall is None:
        return None
    return bool(overall >= 0.5)


def task_answers_from_records(records, adapter=None) -> Dict[tuple, dict]:
    """Map ``(sample_id, query_time)`` -> task fields from adapter records."""

    out: Dict[tuple, dict] = {}
    for record in records or []:
        sample_id = record.get("sample_id")
        timestamp = record.get("question_timestamp")
        out[(str(sample_id), _round_time(timestamp))] = {
            "prediction": record.get("prediction"),
            "ground_truth": record.get("ground_truth"),
            "task_type": record.get("task_type"),
            "question_id": record.get("question_id"),
            "status": record.get("status"),
            "error_type": record.get("error_type"),
            "correct": _official_correct(adapter, record) if adapter is not None else None,
            "video_path": record.get("video_path"),
        }
    return out


def _task_transition(full_correct, comp_correct) -> str:
    if full_correct is None or comp_correct is None:
        return "missing"
    if full_correct and comp_correct:
        return "correct_to_correct"
    if full_correct and not comp_correct:
        return "correct_to_wrong"
    if not full_correct and comp_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _pair_probes(comp_probes, full_probes, kv_config, aggregator) -> None:
    """Pair Full/Compressed diagnostic probes by (sample, query, k)."""

    full_map = {
        (str(p.sample_id), str(p.query_id), int(p.probe_k)): p
        for p in full_probes
        if p.probe_k is not None
    }
    for comp in comp_probes:
        if comp.probe_k is None:
            continue
        key = (str(comp.sample_id), str(comp.query_id), int(comp.probe_k))
        full = full_map.get(key)
        if full is None:
            LOG.warning("kv_metrics: no full probe for %s", key)
            continue
        record = compute_paired_metrics(comp, full, kv_config)
        record["probe_k"] = int(comp.probe_k)
        record["probe_kind"] = "diagnostic_probe"
        record["probe_query_hash"] = comp.probe_query_hash
        record["probe_query_hash_full"] = full.probe_query_hash
        record["full_kv_tokens_probe"] = full.kv_tokens
        record["compressed_kv_tokens_probe"] = comp.kv_tokens
        aggregator.add_probe(record)


def _round_time(value):
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value


def _build_runtime(
    config: dict, memory_method: str, compression_enabled: Optional[bool] = None
) -> ModelRuntime:
    runtime_config = make_runtime_config(config)
    runtime_config["memory"] = copy.deepcopy(config.get("memory", {}))
    runtime_config["memory"]["method"] = memory_method
    if compression_enabled is not None:
        runtime_config["memory"]["compression_enabled"] = bool(compression_enabled)
    runtime = ModelRuntime(runtime_config).load()
    return runtime


def _load_chunk_module(backend: str = "mmduet2"):
    import importlib

    if backend == "qwen2_5_vl":
        # The Qwen backend re-exports the same chunk iterators, so the existing
        # kv_metrics instrumentation can wrap it unchanged.
        return importlib.import_module("models.Qwen25VL_Stream")
    return importlib.import_module("models.MMDuet2_HERMES")


def _run_pass(
    config: dict,
    kv_config: KVMetricConfig,
    adapter,
    samples,
    mode: str,
    probe_targets=None,
) -> dict:
    """Run one pass (compressed or full) and return captured artifacts."""

    runtime = _build_runtime(config, "hermes", compression_enabled=(mode != "full"))
    chunk_module = _load_chunk_module(getattr(runtime, "backend", "mmduet2"))

    instrumentation = KVInstrumentation(kv_config)
    instrumentation.install(runtime.model, chunk_module)
    breakdown = RuntimeBreakdown(enabled=kv_config.profile_runtime_breakdown)
    breakdown.install(runtime.model, chunk_module, runtime.model.qa_model)
    if mode == "full":
        instrumentation.disable_compression()

    captured_answers, set_targets = _wrap_runtime_events(
        runtime,
        instrumentation,
        kv_config.capture_option_logits,
        getattr(kv_config, "diagnostic_video_ids", None),
    )

    events: List[dict] = []
    compactions: List[dict] = []
    task_records: List[dict] = []
    probes: List[dict] = []
    uid_dumps: List[dict] = []
    chunk_records: List[dict] = []
    run_t0 = time.perf_counter()
    try:
        for sample in samples:
            runtime.reset()
            instrumentation.reset_collected()
            # fresh copy per sample so the one-shot `_armed` flag resets
            target_map = {k: dict(v) for k, v in (probe_targets or {}).items()}
            set_targets(target_map)
            try:
                produced = adapter.run_sample(runtime, sample)
                if produced:
                    task_records.extend(produced)
            except Exception as error:  # noqa: BLE001
                LOG.exception("kv_metrics pass=%s sample failed: %s", mode, sample.sample_id)
                raise error
            finally:
                runtime.reset()
            for artifact in instrumentation.collected_events:
                artifact.video_id = artifact.video_id or artifact.event_id.split("|", 1)[0]
                if artifact.sample_id is None:
                    artifact.sample_id = str(sample.sample_id)
                events.append(artifact)
            compactions.extend(instrumentation.collected_compactions)
            for dump in instrumentation.collected_uid_dumps:
                dump.setdefault("sample_id", str(sample.sample_id))
                uid_dumps.append(dump)
            for chunk in instrumentation.collected_chunks:
                chunk.setdefault("sample_id", str(sample.sample_id))
                chunk_records.append(chunk)
            for probe in instrumentation.collected_probes:
                probe.sample_id = probe.sample_id or str(sample.sample_id)
                probe.video_id = probe.video_id or probe.event_id.split("|", 1)[0]
                probes.append(probe)
    finally:
        # ALWAYS free this pass's model, even on failure, so the next batch
        # cannot stack a second 3B model on the same GPU.
        try:
            breakdown.restore()
        except Exception:  # noqa: BLE001
            pass
        try:
            instrumentation.release()
        except Exception:  # noqa: BLE001
            pass
        del runtime
        cleanup_after_error()

    breakdown.set_run_wallclock(time.perf_counter() - run_t0)
    stream_times = [
        getattr(e, "stream_time", None)
        for e in events
        if getattr(e, "stream_time", None) is not None
    ]
    if stream_times:
        breakdown.set_stream_duration(max(stream_times))
    intervals = [
        c.get("trigger_interval_sec") for c in compactions if c.get("trigger_interval_sec")
    ]
    if intervals:
        breakdown.set_mean_trigger_interval(sum(intervals) / len(intervals))
    runtime_breakdown_stats = breakdown.stats()
    runtime_summary = breakdown.summary()

    write_run_outputs(kv_config, mode, samples, task_records)
    if uid_dumps or chunk_records:
        write_ablation_sidecars(kv_config, uid_dumps, chunk_records)

    return {
        "instrumentation": instrumentation,
        "mode": mode,
        "events": events,
        "compactions": compactions,
        "answers": captured_answers,
        "task_records": task_records,
        "probes": probes,
        "uid_dumps": uid_dumps,
        "chunk_records": chunk_records,
        "runtime_breakdown": runtime_breakdown_stats,
        "runtime_summary": runtime_summary,
    }





def write_run_outputs(kv_config, mode: str, samples, task_records) -> dict:
    """Persist per-sample task records + a resume marker.

    Every sample (rich OR lightweight) is written, so overall / per-category
    accuracy covers the full benchmark, while paired decision sidecars only
    exist for the diagnostic subset. A batch counts as done only when every
    expected sample produced a record.
    """

    import json as _json
    from pathlib import Path as _Path

    out = _Path(getattr(kv_config, "output_dir", "outputs/kv_metrics")) / str(
        getattr(kv_config, "run_name", "default")
    )
    out.mkdir(parents=True, exist_ok=True)

    with (out / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for record in task_records:
            handle.write(_json.dumps(record, ensure_ascii=False) + "\n")

    completed = {
        str(r.get("sample_id"))
        for r in task_records
        if r.get("status") != "failed"
    }
    expected = [str(s.sample_id) for s in samples]
    missing = [sid for sid in expected if sid not in completed]
    summary = {
        "mode": mode,
        "n_expected": len(expected),
        "n_completed": len(completed),
        "n_records": len(task_records),
        "missing_sample_ids": missing,
        # Video-level ledger so a re-run with a DIFFERENT batch size can still
        # skip the videos that are already finished.
        "completed_video_ids": sorted(completed),
        "status": "success" if not missing else "partial",
        "kv_config": {
            "mode": getattr(kv_config, "mode", None),
            "diagnostic_video_ids": list(
                getattr(kv_config, "diagnostic_video_ids", []) or []
            ),
            "capture_option_logits": getattr(kv_config, "capture_option_logits", False),
        },
    }
    # NOTE: ``summary.json`` is owned by the metric aggregator (it overwrites
    # it at the end of the run), so the resume marker uses its own filename.
    (out / "run_status.json").write_text(
        _json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def write_ablation_sidecars(kv_config, uid_dumps, chunk_records) -> None:
    """Stage-ablation JSONL sidecars (stable-UID provenance + chunk schedule).

    Written to separate files so the production ``compactions.jsonl`` /
    ``summary.json`` stay slim and are never widened by the ablation.
    """

    import json as _json
    from pathlib import Path as _Path

    out = _Path(getattr(kv_config, "output_dir", "outputs/kv_metrics")) / str(
        getattr(kv_config, "run_name", "default")
    )
    out.mkdir(parents=True, exist_ok=True)
    if uid_dumps:
        with (out / "compaction_uids.jsonl").open("w", encoding="utf-8") as fh:
            for row in uid_dumps:
                fh.write(_json.dumps(row, ensure_ascii=False) + "\n")
    if chunk_records:
        with (out / "chunk_consistency.jsonl").open("w", encoding="utf-8") as fh:
            for row in chunk_records:
                fh.write(_json.dumps(row, ensure_ascii=False) + "\n")


def apply_single_pass_task_fields(record: dict, task, answer_record,
                                  capture_option_logits: bool = False,
                                  side: str = "compressed") -> dict:
    """Canonical compressed-branch task schema + legacy aliases.

    The compressed-only path must emit the SAME canonical names as the paired
    path (``task_prediction_compressed`` etc.), keeping the legacy keys as
    aliases for backwards compatibility.
    """

    suffix = "full" if side == "full" else "compressed"
    record[f"task_prediction_{suffix}"] = task.get("prediction") if task else None
    record[f"task_correct_{suffix}"] = task.get("correct") if task else None
    record[f"task_status_{suffix}"] = task.get("status") if task else None
    record[f"task_error_type_{suffix}"] = task.get("error_type") if task else None
    record["task_type"] = (task or {}).get("task_type")
    record["task_gold"] = task.get("ground_truth") if task else None
    record["task_missing"] = task is None
    # legacy aliases (always mirror the branch that actually ran)
    record["task_prediction"] = record[f"task_prediction_{suffix}"]
    record["task_correct"] = record[f"task_correct_{suffix}"]
    record["task_status"] = record[f"task_status_{suffix}"]
    record["task_error_type"] = record[f"task_error_type_{suffix}"]
    if capture_option_logits:
        ans = answer_record or {}
        record[f"option_token_logits_{suffix}"] = ans.get("option_token_logits")
        record["option_token_ids"] = ans.get("option_token_ids")
    return record


def run(
    config: dict,
    benchmark: Optional[str] = None,
    max_samples: int = 1,
    kv_config: Optional[KVMetricConfig] = None,
    output_dir: Optional[str] = None,
    sample_ids: Optional[List[str]] = None,
    video_ids: Optional[List[str]] = None,
    question_ids: Optional[List[str]] = None,
) -> dict:
    config = apply_defaults(config)
    kv_config = kv_config or KVMetricConfig(enable_kv_metrics=True)
    if output_dir:
        kv_config.output_dir = output_dir

    selected = [benchmark] if benchmark else list(config["benchmarks"].keys())
    benchmark = selected[0]
    bench_config = copy.deepcopy(config["benchmarks"].get(benchmark, {}))
    if video_ids:
        # video-level filter for benchmarks that emit one sample per question
        bench_config["_video_ids"] = [str(v) for v in video_ids]
    if question_ids:
        bench_config["_only_question_ids"] = {str(q) for q in question_ids}
    adapter = BENCHMARK_REGISTRY[benchmark](bench_config)
    samples = list(adapter.load_samples())
    if sample_ids:
        wanted = {str(s) for s in sample_ids}
        samples = [s for s in samples if str(s.sample_id) in wanted]
    elif max_samples and max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(
            f"kv_metrics: no samples selected for benchmark={benchmark} "
            f"(sample_ids={sample_ids})"
        )
    kv_config.benchmark = benchmark
    kv_config.sample_ids = [s.sample_id for s in samples]

    LOG.info(
        "kv_metrics: benchmark=%s samples=%d mode=%s output=%s",
        benchmark, len(samples), kv_config.mode, kv_config.output_dir,
    )

    compressed = None
    full = None
    probe_targets = {}
    if kv_config.detailed_trajectory:
        # group by SOURCE VIDEO: benchmarks may emit one CanonicalSample per
        # question (StreamingBench) or one per video (OVO/OMME), so q_star must
        # be chosen across all questions that share the same video.
        grouped: Dict[str, list] = {}
        for sample in samples:
            if not sample.questions:
                continue
            video_key = video_key_for(sample.video_path)
            grouped.setdefault(video_key, []).extend(sample.questions)
        for video_key, questions in grouped.items():
            q_star = max(
                [q for q in questions if q.timestamp is not None] or questions,
                key=lambda item: (
                    item.timestamp if item.timestamp is not None else -1.0,
                    str(item.question_id),
                ),
            )
            probe_targets[video_key] = {
                "sample_id": video_key,
                "query_id": q_star.question_id,
                "query_time": q_star.timestamp,
            }

    if kv_config.mode in {"compressed", "paired"}:
        compressed = _run_pass(
            config, kv_config, adapter, samples, "compressed",
            probe_targets=probe_targets,
        )
    if kv_config.mode in {"full", "paired"}:
        if kv_config.detailed_trajectory and compressed:
            # the Full-KV pass probes at the SAME stream times as the compressed
            # pass (it has no compactions of its own to trigger on)
            for probe in compressed.get("probes", []):
                info = probe_targets.get(str(probe.video_id))
                if info is None or str(info.get("query_id")) != str(probe.query_id):
                    continue
                info.setdefault("probe_time_plan", [])
                info["probe_time_plan"].append((probe.probe_k, probe.stream_time))
        full = _run_pass(
            config, kv_config, adapter, samples, "full",
            probe_targets=probe_targets,
        )

    aggregator = MetricAggregator(kv_config)
    task_by_event: Dict[str, dict] = {}

    if compressed:
        aggregator.write_pass_sidecar(compressed["events"], "compressed")
        aggregator.write_compactions(compressed["compactions"])
        if kv_config.detailed_trajectory:
            aggregator.write_probe_sidecars(compressed.get("probes", []), "compressed")
        if kv_config.profile_runtime_breakdown:
            aggregator.write_runtime_breakdown(
                compressed.get("runtime_breakdown"),
                compressed.get("runtime_summary"),
            )
    if full:
        aggregator.write_pass_sidecar(full["events"], "full")
        if kv_config.detailed_trajectory:
            aggregator.write_probe_sidecars(full.get("probes", []), "full")
        if kv_config.profile_runtime_breakdown and not compressed:
            aggregator.write_runtime_breakdown(
                full.get("runtime_breakdown"),
                full.get("runtime_summary"),
            )

    if kv_config.mode == "paired" and compressed and full:
        task_lookup = task_answers_from_records(
            compressed.get("task_records"), adapter
        )
        task_lookup_full = task_answers_from_records(
            full.get("task_records"), adapter
        )
        full_by_id = {e.event_id: e for e in full["events"]}
        for comp_artifact in compressed["events"]:
            full_artifact = full_by_id.get(comp_artifact.event_id)
            if full_artifact is None:
                LOG.warning("kv_metrics: no full match for %s", comp_artifact.event_id)
                continue
            record = compute_paired_metrics(comp_artifact, full_artifact, kv_config)
            key = (str(comp_artifact.sample_id), _round_time(comp_artifact.query_time))
            comp_task = task_lookup.get(key)
            full_task = task_lookup_full.get(key)
            _attach_task_fields(record, comp_task, full_task, key)
            if kv_config.capture_option_logits:
                comp_ans = compressed["answers"].get(comp_artifact.event_id) or {}
                full_ans = full["answers"].get(comp_artifact.event_id) or {}
                record["option_token_logits_compressed"] = comp_ans.get("option_token_logits")
                record["option_token_logits_full"] = full_ans.get("option_token_logits")
                record["option_token_ids"] = (
                    comp_ans.get("option_token_ids") or full_ans.get("option_token_ids")
                )
            aggregator.add_paired(record)
            answer = compressed["answers"].get(comp_artifact.event_id)
            if answer:
                task_by_event[comp_artifact.event_id] = {
                    "task_answer": answer.get("answer"),
                    "task_answer_gold": answer.get("answer_gold"),
                    "task_correct": _is_correct(answer),
                }
        if kv_config.detailed_trajectory:
            _pair_probes(
                compressed.get("probes", []), full.get("probes", []),
                kv_config, aggregator,
            )
    else:
        pass_artifacts = compressed or full
        pass_tasks = task_answers_from_records(
            (pass_artifacts or {}).get("task_records"), adapter
        )
        for artifact in (pass_artifacts or {}).get("events", []):
            record = _single_pass_record(artifact, kv_config)
            key = (str(artifact.sample_id), _round_time(artifact.query_time))
            task = pass_tasks.get(key)
            if task is None:
                LOG.warning("kv_metrics: no task record for %s", key)
            answers = (pass_artifacts or {}).get("answers", {}) or {}
            apply_single_pass_task_fields(
                record, task, answers.get(artifact.event_id),
                capture_option_logits=kv_config.capture_option_logits,
                # authoritative mode of the pass these records came from
                side="full" if str(
                    (pass_artifacts or {}).get("mode") or kv_config.mode
                ) == "full" else "compressed",
            )
            aggregator.add_paired(record)

    summary = aggregator.write(task_by_event)
    return summary


def _single_pass_record(artifact, kv_config: KVMetricConfig) -> dict:
    return {
        "event_id": artifact.event_id,
        "video_id": artifact.video_id,
        "sample_id": artifact.sample_id or artifact.video_id,
        "query_id": artifact.query_id,
        "query_time": artifact.query_time,
        "stream_time": artifact.stream_time,
        "compaction_index": artifact.compaction_index,
        "compaction_trigger_count_at_query": artifact.compaction_index,
        "effective_compaction_count_at_query": artifact.effective_compaction_count,
        "compressed_kv_bytes": artifact.kv_bytes,
        "kept_kv_tokens": artifact.kv_tokens,
        "last_compaction_wallclock": artifact.last_compaction_wallclock,
        "ram_mean": None,
        "aore_mean": None,
        "js": None,
        "kl": None,
        "temporal_ram": None,
    }


def _is_correct(answer: dict) -> Optional[bool]:
    gold = answer.get("answer_gold")
    pred = answer.get("answer")
    if gold is None or pred is None:
        return None
    return str(gold).strip().upper() == str(pred).strip().upper()


def _attach_task_fields(record: dict, comp_task, full_task, key) -> None:
    """Split full/compressed task outcomes and classify the transition.

    Never falls back to a synthetic prediction such as "NO REPLY": a missing
    lookup yields ``null`` predictions plus ``task_missing = true``.
    """

    missing = comp_task is None or full_task is None
    if missing:
        LOG.warning(
            "kv_metrics: no task record for sample/query %s "
            "(compressed=%s, full=%s)", key, comp_task is not None, full_task is not None,
        )

    comp_pred = comp_task.get("prediction") if comp_task else None
    comp_correct = comp_task.get("correct") if comp_task else None
    full_pred = full_task.get("prediction") if full_task else None
    full_correct = full_task.get("correct") if full_task else None

    record["task_prediction_compressed"] = comp_pred
    record["task_prediction_full"] = full_pred
    record["task_correct_compressed"] = comp_correct
    record["task_correct_full"] = full_correct
    record["task_status_compressed"] = comp_task.get("status") if comp_task else None
    record["task_status_full"] = full_task.get("status") if full_task else None
    record["task_error_type_compressed"] = comp_task.get("error_type") if comp_task else None
    record["task_error_type_full"] = full_task.get("error_type") if full_task else None
    record["task_type"] = (comp_task or full_task or {}).get("task_type")
    record["task_gold"] = (comp_task or full_task or {}).get("ground_truth")
    record["task_transition"] = _task_transition(full_correct, comp_correct)
    record["task_missing"] = bool(missing)

    # legacy aliases (compressed branch) kept for backwards compatibility
    record["task_prediction"] = comp_pred
    record["task_correct"] = comp_correct


def _task_correct(prediction, ground_truth) -> Optional[bool]:
    if prediction is None or ground_truth is None:
        return None
    return str(prediction).strip().upper() == str(ground_truth).strip().upper()


def main(argv=None):
    parser = argparse.ArgumentParser(description="KV cache fidelity evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument(
        "--sample_id",
        action="append",
        default=None,
        help="Restrict the run to these benchmark sample ids (repeatable).",
    )
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument(
        "--video_id",
        action="append",
        default=None,
        help="Restrict to these source video ids (repeatable).",
    )
    add_cli_arguments(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    kv_config = KVMetricConfig(
        enable_kv_metrics=True,
        mode=args.kv_metrics_mode,
        output_dir=args.kv_metrics_dir,
        run_name=args.kv_metrics_run_name,
        attention_backend=args.kv_metrics_attention_backend,
        anchor=args.kv_metrics_anchor,
        temporal_bins=parse_temporal_bins(args.kv_metrics_temporal_bins),
        save_per_head=args.kv_metrics_save_per_head,
        teacher_forced_span=args.kv_metrics_teacher_forced_span,
        profile_wallclock=args.kv_metrics_profile_wallclock,
        scope=args.kv_metrics_scope,
        summary_token_mode=args.kv_metrics_summary_token_mode,
        profile_runtime_breakdown=args.profile_runtime_breakdown,
        detailed_trajectory=args.kv_metrics_detailed_trajectory,
    )
    summary = run(
        config,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        kv_config=kv_config,
        sample_ids=args.sample_id,
        video_ids=args.video_id,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
