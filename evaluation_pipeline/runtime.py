import gc
import importlib
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch

from .models.qwen25_vl import (
    BACKEND_QWEN25VL,
    resolve_backend,
    resolve_mcq_decode_mode,
    resolve_model_path,
)
from .video_sampling import sample_video_frames


LOG = logging.getLogger("evaluate_all")

# Sentinel budget used by ``mode=full``: large enough that HERMES never
# triggers a compaction, while keeping the exact same code path / weights /
# frames as the compressed run.
FULL_CACHE_BUDGET = int(1e12)


class Profiler:
    def __init__(self):
        self._started = None

    @staticmethod
    def _kv_tokens(runtime):
        if runtime is None or runtime.model is None:
            return None
        for owner_name in ("qa_model", "client"):
            owner = getattr(runtime.model, owner_name, None)
            cache = getattr(owner, "kv_cache", None) if owner is not None else None
            if cache is None:
                cache = getattr(owner, "past_key_values", None) if owner is not None else None
            if cache is None:
                continue
            try:
                if hasattr(cache, "get_seq_length"):
                    return int(cache.get_seq_length())
                if isinstance(cache, (list, tuple)) and cache:
                    return int(cache[0][0].shape[2])
            except (AttributeError, IndexError, TypeError):
                continue
        return None

    def start(self, runtime=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._started = time.perf_counter()
        self._kv_before = self._kv_tokens(runtime)

    def stop(self, runtime=None) -> Dict[str, Optional[float]]:
        if self._started is None:
            return {"latency": None, "peak_gpu_memory_mb": None}
        result = {
            "latency": time.perf_counter() - self._started,
            "peak_gpu_memory_mb": None,
        }
        if runtime is not None and hasattr(runtime.model, "get_compression_stats"):
            result.update(runtime.model.get_compression_stats())
        else:
            result.update({
                "peak_kv_tokens": None,
                "compression_events": 0,
                "total_removed_tokens": 0,
                "mean_kv_retention_ratio": None,
                "mean_kv_compression_ratio": None,
                "compression_events_detail": [],
            })
        if torch.cuda.is_available():
            result["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        self._started = None
        self._kv_before = None
        return result


def cleanup_after_error() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelRuntime:
    """Loads the selected model exactly once and exposes benchmark-neutral calls."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.backend = resolve_backend(config)
        self.model_name = config.get("name", "MMDuet2")
        self.memory_method = config.get("memory", {}).get("method", "full_cache")
        self.mcq_decode_mode = resolve_mcq_decode_mode(config)
        self.last_inference_metadata = {}

    def load(self):
        if self.model is not None:
            return self
        # Validate the ``memory:`` block up front (method-agnostic; this never
        # changes HERMES behaviour). The factory is the single source of truth
        # for which memory methods exist -- see ``evaluation_pipeline/memory``.
        from .memory import available_memory_methods, resolve_memory_config

        self.memory_method_config = resolve_memory_config(self.config)
        if self.memory_method not in {"full_cache", "hermes", "infokv_stream"}:
            raise NotImplementedError(
                f"Memory policy {self.memory_method!r} has no GPU runtime wiring "
                "yet. The pluggable CPU memory methods are "
                f"{available_memory_methods()}; the InfoKV-inspired streaming "
                "method (memory.method=infokv_stream) has a Qwen2.5-VL GPU "
                "integration; it requires the Qwen backend and a live model. "
                "Use the CPU debug CLI for configuration-only checks: "
                "`python -m evaluation_pipeline.memory.debug --help`."
            )
        if self.memory_method == "infokv_stream" and self.backend != BACKEND_QWEN25VL:
            raise NotImplementedError(
                "memory.method=infokv_stream is only wired for the Qwen2.5-VL "
                "backend (set model.backend: qwen2_5_vl). The MMDuet2/HERMES "
                "backend is untouched."
            )
        root = os.path.abspath(self.config.get("ovo_root", "OVO-Bench"))
        if root not in sys.path:
            sys.path.insert(0, root)

        model_path = resolve_model_path(self.config)
        if not model_path:
            raise ValueError("No model_path configured for the selected backend")
        attention = self.config.get("inference", {}).get(
            "attn_implementation",
            "auto",
        )
        os.environ["MMDUET2_ATTN_IMPLEMENTATION"] = str(attention)
        inference = self.config.get("inference", {})
        for env_name, key in (
            ("MMDUET2_SAMPLE_FPS", "sample_fps"),
            ("MMDUET2_FRAMES_PER_TURN", "frames_per_turn"),
            ("MMDUET2_MAX_SIDE", "max_side"),
        ):
            if key in inference:
                os.environ[env_name] = str(inference[key])
        args = SimpleNamespace(
            model_path=model_path,
            model=self.model_name,
            result_dir="",
            chunked_dir="",
            save_results=False,
        )
        if self.backend == BACKEND_QWEN25VL:
            # New default streaming backend: non-proactive Qwen2.5-VL.
            os.environ["QWEN25VL_MCQ_DECODE_MODE"] = self.mcq_decode_mode
            capture = (self.config.get("inference", {}) or {}).get(
                "capture_option_logits", True
            )
            os.environ["QWEN25VL_CAPTURE_OPTION_LOGITS"] = "1" if capture else "0"
            if self.memory_method == "hermes":
                budget = int(
                    self.config.get("memory", {})
                    .get("compression", {})
                    .get("budget", 6000)
                )
            else:
                budget = FULL_CACHE_BUDGET
            os.environ["HERMES_KV_SIZE"] = str(budget)
            # Explicit Full-vs-HERMES switch for the new backend: ``full``
            # disables the compression step rather than relying on a sentinel.
            # An explicit ``memory.compression_enabled`` override wins, so a
            # Full pass can load the HERMES wrapper but never prune.
            explicit = (self.config.get("memory", {}) or {}).get(
                "compression_enabled"
            )
            compression_enabled = (
                False
                if self.memory_method == "infokv_stream"
                else (
                    bool(explicit)
                    if explicit is not None
                    else self.memory_method == "hermes"
                )
            )
            os.environ["QWEN25VL_COMPRESSION_ENABLED"] = (
                "1" if compression_enabled else "0"
            )
            module = importlib.import_module("models.Qwen25VL_Stream")
            if self.memory_method == "infokv_stream":
                # New branch: the InfoKVStream subclass funnels every streamed
                # chunk through the memory method + KVCacheCompactor. HERMES's
                # own per-layer selection is bypassed (compression disabled).
                from .memory.qwen25vl_backend import build_infokv_wrapper_class

                cls = build_infokv_wrapper_class()
            else:
                cls = module.EvalQwen25VLStream
            self.attention_backend = getattr(
                importlib.import_module("models.MMDuet2_HERMES"),
                "HERMES_ATTENTION_BACKEND",
                None,
            )
            if self.attention_backend:
                LOG.info("Qwen25VL backend: attn=%s", self.attention_backend.upper())
        elif self.memory_method == "hermes":
            try:
                module = importlib.import_module("models.MMDuet2_HERMES")
            except ModuleNotFoundError as error:
                if error.name == "logzero":
                    raise ModuleNotFoundError(
                        "HERMES requires logzero. Install it with "
                        "`python -m pip install logzero==1.7.0` "
                        "or `python -m pip install -r HERMES/requirements_qwen.txt`."
                    ) from error
                raise
            cls = module.EvalMMDuet2Hermes
            os.environ["HERMES_KV_SIZE"] = str(
                self.config.get("memory", {}).get("compression", {}).get("budget", 6000)
            )
            # Surface the actually selected HERMES attention backend so it does
            # not have to be inferred from a traceback.
            backend = getattr(module, "HERMES_ATTENTION_BACKEND", None)
            self.attention_backend = backend
            if backend:
                LOG.info("OVO backend: %s", backend.upper())
        elif self.model_name in ("MMDuet2", "MMDuet2_native"):
            module = importlib.import_module("models.MMDuet2")
            cls = module.EvalMMDuet2
        else:
            raise ValueError(
                f"Unsupported unified runtime model={self.model_name!r}; "
                "use MMDuet2 or set a project-specific runtime."
            )

        self.model = cls(args)
        if self.memory_method == "infokv_stream":
            # Attach the memory method + capture hooks + KV compactor.
            self.model.attach_infokv(self.memory_method_config)
        return self

    def reset(self):
        if self.model is None:
            return
        client = getattr(self.model, "client", None)
        if client is not None and hasattr(client, "reset"):
            client.reset()
        qa_model = getattr(self.model, "qa_model", None)
        if qa_model is not None and hasattr(qa_model, "clear_cache"):
            qa_model.clear_cache()
        if hasattr(self.model, "reset_compression_stats"):
            self.model.reset_compression_stats()

    def run_ovo(self, video_path: str, prompt: str, question: Dict[str, Any]):
        task = question["task"]
        if task in getattr(self.model, "FORWARD_TASKS", set()):
            return self.model.inference_forward(
                video_file_name=video_path,
                prompt=prompt,
                realtime=float(question.get("timestamp", 0.0)),
                task=task,
                sample_id=question.get("sample_id"),
                test_index=question.get("test_index"),
            )
        # OVO Backward/Realtime MCQ ablation switches.
        # Defaults (both True) reproduce the official path: the first token is
        # constrained to an option letter and the mandatory-reply prefix is
        # appended. These only affect reply_required MCQ questions; Forward
        # tasks return above and StreamingBench/Video-MME use other entry
        # points.
        if self.backend == BACKEND_QWEN25VL:
            # The new backend owns its own MCQ policy: ``official`` (default,
            # free decode + option extraction) or ``constrained`` (hard A-E
            # first-token mask), selected by config, never implicitly.
            return self.model.inference(
                video_path,
                prompt,
                start_time=0.0,
                end_time=float(question.get("timestamp", 0.0)),
                task=task,
                sample_id=question.get("sample_id"),
                options=question.get("options"),
                prompt_source="official_benchmark",
            )
        must_reply = bool(question.get("must_reply", True))
        constrain_mcq_first_token = bool(
            question.get("constrain_mcq_first_token", True)
        )
        allowed_first_token_ids = (
            self._allowed_mcq_token_ids(question)
            if constrain_mcq_first_token
            else None
        )
        return self.model.inference(
            video_path,
            prompt,
            start_time=0.0,
            end_time=float(question.get("timestamp", 0.0)),
            task=task,
            sample_id=question.get("sample_id"),
            allowed_first_token_ids=allowed_first_token_ids,
            must_reply=must_reply,
        )

    def run_streaming_mcq(self, video_path: str, prompt: str, question: Dict[str, Any]):
        if self.backend == BACKEND_QWEN25VL:
            # Non-proactive streaming backend: prefill [start, query] then ask
            # the question once. No proactive decision and no task hack.
            return self.model.inference(
                video_path,
                prompt,
                start_time=float(question.get("start_time", 0.0)),
                end_time=float(question.get("timestamp", 0.0)),
                task=question.get("task") or "STU",
                sample_id=question.get("sample_id"),
                options=question.get("options"),
                prompt_source=question.get("prompt_source", "official_benchmark"),
            )
        # MMDuet2's timestamp entry point still requires an internal task
        # selector. This is an execution detail, not StreamingBench metadata.
        return self.model.inference(
            video_path,
            prompt,
            start_time=float(question.get("start_time", 0.0)),
            end_time=float(question.get("timestamp", 0.0)),
            task="STU",
            sample_id=question.get("sample_id"),
            allowed_first_token_ids=self._allowed_mcq_token_ids(question),
        )

    def _mcq_option_token_ids_map(self, question: Dict[str, Any]):
        """Per-option first-token variants used by the official MCQ constraint.

        Returns ``{"A": [id("A"), id(" A")], ...}`` -- the exact token set the
        constrained decoder allows. Nothing here is hard-coded: the variants
        come from the live tokenizer, matching ``_allowed_mcq_token_ids``.
        """

        if question.get("response_policy") != "reply_required":
            return None
        options = question.get("options") or []
        if not options:
            return None
        tokenizer = getattr(
            getattr(self.model, "qa_processor", None),
            "tokenizer",
            None,
        )
        if tokenizer is None:
            tokenizer = getattr(
                getattr(self.model, "client", None),
                "processor",
                None,
            )
            tokenizer = getattr(tokenizer, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Cannot resolve tokenizer for MCQ constraint")
        letters = "ABCDE"[: min(len(options), 5)]
        mapping = {}
        for letter in letters:
            ids_for_letter = []
            for candidate in (letter, f" {letter}"):
                ids = tokenizer(candidate, add_special_tokens=False).input_ids
                if ids:
                    ids_for_letter.append(int(ids[0]))
            mapping[letter] = list(dict.fromkeys(ids_for_letter))
        return mapping

    def _allowed_mcq_token_ids(self, question: Dict[str, Any]):
        mapping = self._mcq_option_token_ids_map(question)
        if not mapping:
            return None
        allowed = []
        for letter in mapping:
            allowed.extend(mapping[letter])
        allowed = list(dict.fromkeys(allowed))
        if not allowed:
            raise RuntimeError("No allowed first-token IDs for MCQ options")
        return allowed

    @staticmethod
    def _is_no_reply(value) -> bool:
        normalized = str(value or "").strip().upper()
        return not normalized or normalized == "NO" or normalized.startswith("NO REPLY")

    def run_streaming_text(self, video_path: str, prompt: str, question: Dict[str, Any]):
        """Run a timestamped streaming answer without OVO's MCQ constraint."""
        import gc

        query_time = float(question.get("timestamp", 0.0))
        start_time = float(question.get("start_time", 0.0))
        response_policy = question.get("response_policy")
        if self.memory_method == "hermes":
            module = importlib.import_module("models.MMDuet2_HERMES")
            is_proactive_output = (
                response_policy
                in {
                    "standing_trigger",
                    "proactive_output_final",
                }
            )

            self.model._reset_stream(
                forward=False,
                proactive_output=is_proactive_output,
            )
            try:
                stream = module.iter_chunks_until_query(
                    video_path=video_path,
                    query_time=query_time - start_time,
                    start_time=start_time,
                    sample_fps=self.model.sample_fps,
                    frames_per_turn=self.model.frames_per_turn,
                    max_side=self.model.max_side,
                )
                for frames, _, _ in stream:
                    self.model._encode_frames(frames)
                allowed_first_token_ids = None

                max_new_tokens = int(
                    self.config.get(
                        "inference",
                        {},
                    ).get(
                        "max_new_tokens",
                        32,
                    )
                )

                # StreamingBench PO:
                # let the model generate naturally.
                # Do not hard-mask the vocabulary to Yes/No.
                if response_policy == "standing_trigger":
                    allowed_first_token_ids = None

                effective_prompt = (
                    prompt
                    if response_policy
                    in {
                        "standing_trigger",
                        "proactive_output_final",
                    }
                    else (
                        prompt
                        + "\nAnswer now using only the video observed so far."
                    )
                )

                return self.model._ask(
                    effective_prompt,
                    max_new_tokens=max_new_tokens,
                    mc=False,
                    must_reply=(
                        response_policy
                        not in {
                            "standing_trigger",
                            "proactive_output_final",
                        }
                    ),
                    allowed_first_token_ids=(
                        allowed_first_token_ids
                    ),
                )
            finally:
                self.model.qa_model.clear_cache()

        module = importlib.import_module("models.MMDuet2")

        is_proactive_output = (
            response_policy
            in {
                "standing_trigger",
                "proactive_output_final",
            }
        )

        # PO 使用与 HERMES 相同的 neutral system prompt。
        # 普通 StreamingBench timestamp QA 保持原行为。
        self.model._reset_client(
            forward=False,
            proactive_output=is_proactive_output,
        )

        try:
            stream = module.iter_chunks_until_query(
                video_path=video_path,
                query_time=query_time - start_time,
                start_time=start_time,
                sample_fps=self.model.sample_fps,
                frames_per_turn=self.model.frames_per_turn,
                max_side=self.model.max_side,
            )

            answer = None

            for (
                frames,
                is_query_turn,
                relative_time,
            ) in stream:

                turn_time = (
                    start_time
                    + float(relative_time)
                )

                content = [
                    {
                        "type": "image",
                        "image": frame,
                    }
                    for frame in frames
                ]

                if not is_query_turn:
                    self.model._prefill_video_turn(
                        {
                            "role": "user",
                            "content": content,
                        },
                        turn_time,
                    )
                    continue

                # run_proactive_polling() 已经构造好了
                # official-style PO prompt。
                #
                # 所以 PO 不再追加
                # "Answer now using only..."。
                #
                # 普通 timestamp QA 则维持原来的行为。
                effective_prompt = (
                    prompt
                    if is_proactive_output
                    else (
                        prompt
                        + "\nAnswer now using only "
                        "the video observed so far."
                    )
                )

                content.append(
                    {
                        "type": "text",
                        "text": effective_prompt,
                    }
                )

                answer = self.model._answer_query_turn(
                    {
                        "role": "user",
                        "content": content,
                    },
                    turn_time,

                    # PO 不加入 "I must reply."
                    # 普通 timestamp QA 仍然强制回答。
                    must_reply=(
                        not is_proactive_output
                    ),

                    # StreamingBench PO 使用自然生成，
                    # 不做人为 Yes/No vocabulary mask。
                    allowed_first_token_ids=None,
                )

                break

            if answer is None:
                raise RuntimeError(
                    "No query turn was executed"
                )

            return answer

        finally:
            self.model.client.reset()
            gc.collect()


    def _video_duration(self, video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return frames / fps if frames > 0 else 0.0
        finally:
            cap.release()

    def run_proactive_polling(
        self,
        video_path,
        question_text,
        question,
        start_time,
        sample_id,
    ):
        """StreamingBench official-style proactive polling."""

        if self.backend == BACKEND_QWEN25VL:
            raise NotImplementedError(
                "The Qwen2.5-VL streaming backend has no proactive-response "
                "protocol; use the MMDuet2 backend for StreamingBench "
                "Proactive Output."
            )

        gt_time = float(
            question.metadata.get(
                "ground_truth_timestamp"
            )
            or 0.0
        )

        gt_output = str(
            question.metadata.get(
                "ground_truth_output"
            )
            or question.answer
            or ""
        )

        poll_start = int(start_time) + 1
        poll_end = gt_time + 4.0

        predicted = None
        poll_outputs = []

        # StreamingBench official proactive QA template.
        def build_po_prompt(query):
            return (
                "You are an advanced image question-answering "
                "AI assistant. You have been provided with "
                "images and a question related to the images. "
                "Your task is to carefully analyze the images "
                "and provide the answer to the question. "
                "You need to carefully confirm whether the "
                "images content meet the conditions of the "
                "question, and then output the correct content."
                "\n\n"
                f"Question: {query}"
                "\n\n"
                "The answer is:"
            )

        for poll_time in range(
            poll_start,
            int(poll_end) + 1,
        ):
            query = (
                f'{question_text} '
                f'Is it the right time to output '
                f'"{gt_output}"? '
                f'You can only answer yes or no.'
            )

            poll_prompt = build_po_prompt(
                query
            )

            answer = self.run_streaming_text(
                video_path,
                poll_prompt,
                {
                    "timestamp": float(
                        poll_time
                    ),
                    "start_time": start_time,
                    "sample_id": sample_id,
                    "response_policy": (
                        "standing_trigger"
                    ),
                },
            )

            print(
                f"[PO POLL] "
                f"id={sample_id} "
                f"t={poll_time} "
                f"answer={answer!r}"
            )

            poll_outputs.append(
                {
                    "timestamp": poll_time,
                    "answer": answer,
                }
            )

            # StreamingBench official behavior:
            # trigger if the response contains "yes".
            if "yes" in str(answer).strip().lower():

                predicted = float(
                    poll_time
                )

                final_prompt = build_po_prompt(
                    question_text
                )

                final = self.run_streaming_text(
                    video_path,
                    final_prompt,
                    {
                        "timestamp": float(
                            poll_time
                        ),
                        "start_time": start_time,
                        "sample_id": sample_id,
                        "response_policy": (
                            "proactive_output_final"
                        ),
                    },
                )

                return {
                    "prediction": final,
                    "predicted_timestamp": (
                        predicted
                    ),
                    "poll_outputs": (
                        poll_outputs
                    ),
                }

        return {
            "prediction": "",
            "predicted_timestamp": None,
            "poll_outputs": poll_outputs,
        }


    def run_proactive_native(self, video_path, question_text, question, start_time, sample_id):
        """Optional research mode: poll until the video ends, without GT timing."""
        if self.backend == BACKEND_QWEN25VL:
            raise NotImplementedError(
                "The Qwen2.5-VL streaming backend has no proactive-response "
                "protocol."
            )
        poll_start = int(start_time) + 1
        poll_end = int(self._video_duration(video_path))
        poll_outputs = []
        for poll_time in range(poll_start, poll_end + 1):
            answer = self.run_streaming_text(
                video_path,
                f"{question_text} Should you output now? Answer yes or no.",
                {
                    "timestamp": float(poll_time),
                    "start_time": start_time,
                    "sample_id": sample_id,
                    "response_policy": "standing_trigger",
                },
            )
            poll_outputs.append({"timestamp": poll_time, "answer": answer})
            if not str(answer).strip().lower().startswith("no"):
                final = self.run_streaming_text(
                    video_path,
                    question_text,
                    {
                        "timestamp": float(poll_time),
                        "start_time": start_time,
                        "sample_id": sample_id,
                        "response_policy": "reply_required",
                    },
                )
                return {
                    "prediction": final,
                    "predicted_timestamp": float(poll_time),
                    "poll_outputs": poll_outputs,
                }
        return {
            "prediction": "",
            "predicted_timestamp": None,
            "poll_outputs": poll_outputs,
        }

    def run_offline(self, video_path: str, prompt: str, question: Dict[str, Any]):
        """Run full-video QA while keeping the benchmark's offline protocol."""
        sampled = sample_video_frames(
            video_path,
            int(self.config.get("inference", {}).get("offline_max_frames", 4)),
        )
        self.last_inference_metadata = {
            key: sampled[key]
            for key in (
                "frame_indices",
                "frame_timestamps",
                "source_fps",
                "total_frames",
                "sample_strategy",
            )
        }
        self.last_inference_metadata["num_frames"] = len(sampled["frames"])
        self.last_inference_metadata["max_side"] = self.config.get(
            "inference", {}
        ).get("offline_max_side", 336)
        if self.backend == BACKEND_QWEN25VL:
            # Same model/processor/frames as the streaming runs; offline QA is
            # just "stream the sampled frames, then ask once".
            reset = getattr(self.model, "reset", None)
            if callable(reset):
                reset()
            else:
                self.model._reset_stream(forward=False)
            try:
                frames = sampled["frames"]
                chunk_size = int(getattr(self.model, "frames_per_turn", 2)) or 2
                for start in range(0, len(frames), chunk_size):
                    chunk = frames[start:start + chunk_size]
                    if chunk:
                        self.model.stream_video_chunk(chunk)
                return self.model.ask_question(
                    prompt,
                    options=None,
                    max_new_tokens=int(
                        self.config.get("inference", {}).get("max_new_tokens", 16)
                    ),
                    # No MMDuet2 "Best option: (" suffix: the benchmark prompt
                    # is the only instruction for the Qwen backend.
                    mc=False,
                )
            finally:
                self.model.qa_model.clear_cache()
        if self.memory_method == "hermes":
            self.model._reset_stream(forward=False)
            try:
                frames = sampled["frames"]
                chunk_size = int(getattr(self.model, "frames_per_turn", 2))
                for start in range(0, len(frames), chunk_size):
                    chunk = frames[start:start + chunk_size]
                    if chunk:
                        self.model._encode_frames(chunk)
                return self.model._ask(
                    prompt,
                    max_new_tokens=int(self.config.get("inference", {}).get("max_new_tokens", 16)),
                    mc=True,
                )
            finally:
                self.model.qa_model.clear_cache()

        client = getattr(self.model, "client", None)
        if client is None:
            raise RuntimeError("Selected runtime has no offline model client")
        from qwen_vl_utils import process_vision_info

        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": sampled["frames"],
                        "max_pixels": int(
                            self.config.get("inference", {}).get(
                                "offline_max_side",
                                336,
                            )
                        ) ** 2,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = client.processor.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(history)
        inputs = client.processor(
            text=[text], images=images, videos=videos, return_tensors="pt", padding=True
        ).to(next(client.model.parameters()).device)
        output = client.model.generate(
            **inputs,
            max_new_tokens=int(self.config.get("inference", {}).get("max_new_tokens", 16)),
            do_sample=False,
        )
        generated = output[:, inputs.input_ids.shape[1] :]
        return client.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
