"""Qwen2.5-VL streaming memory backend (non-proactive).

This is the new default streaming backend. It shares the verified HERMES
Qwen2.5-VL machinery (incremental visual-KV prefill + ``predict_and_compress``
attention pruning) but exposes a *non-proactive* protocol:

    reset()
    stream_video_chunk(chunk)      # prefill only, no text generation
    stream_video_chunk(chunk)      # ... until the query time
    ask_question(question)         # inject the question, decode the answer

There is no proactive decision, no response-timing policy and no NO REPLY
parsing anywhere in this path: before the query arrives the model never
calls ``generate``.

MCQ decoding has two explicitly configured modes:

* ``official``    -- free decode + option extraction (default, benchmark-faithful);
* ``constrained`` -- historical MMDuet2-style hard first-token A-E mask.

A read-only ``lm_head`` sidecar captures the step-0 option logits for the
decision-fidelity metrics. It only *reads* the logits tensor; it never changes
the mask, the decoder or the formal prediction.
"""

import gc
import inspect
import os
from typing import Dict, List, Optional

import torch
from PIL import Image

from models.MMDuet2_HERMES import (
    EvalMMDuet2Hermes,
    clean_reply,
    extract_option,
    is_no_reply,
    iter_chunks_until_query,
    iter_frame_chunks,
    mark_last_chunk,
    normalize_forward_reply,
)

from evaluation_pipeline.models.qwen25_vl import (
    BACKEND_QWEN25VL,
    BudgetViolation,
    CacheAccounting,
    IncrementalStreamAudit,
    StreamingProfiler,
    mcq_first_token_ids,
    resolve_mcq_decode_mode,
)


# Re-exported so ``models.Qwen25VL_Stream`` can serve as the kv_metrics
# "chunk module" exactly like ``models.MMDuet2_HERMES`` does.
__all__ = [
    "EvalQwen25VLStream",
    "iter_chunks_until_query",
    "iter_frame_chunks",
    "mark_last_chunk",
    "clean_reply",
    "extract_option",
    "is_no_reply",
    "normalize_forward_reply",
]


NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."


class EvalQwen25VLStream(EvalMMDuet2Hermes):
    """Qwen2.5-VL (full or HERMES budget) with a non-proactive stream API."""

    backend = BACKEND_QWEN25VL
    supports_timestamp_inference = True
    supports_forward_inference = True

    def __init__(self, args):
        super().__init__(args)

        # Replace the MMDuet2 proactive/"NO REPLY" system prompt with a
        # neutral instruction-following prompt. There is no NO REPLY path here.
        self.system_prompt = NEUTRAL_SYSTEM_PROMPT
        self.forward_system_prompt = NEUTRAL_SYSTEM_PROMPT
        self.po_system_prompt = NEUTRAL_SYSTEM_PROMPT

        device = next(self.qa_model.parameters()).device
        init_text = (
            "<|im_start|>system\n"
            + self.system_prompt
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
        )
        init_ids = (
            self.qa_processor.tokenizer(init_text, return_tensors="pt")
            .input_ids.to(device)
        )
        self._init_prompt_ids = init_ids
        self._forward_init_prompt_ids = init_ids
        self._po_init_prompt_ids = init_ids
        self.qa_model.init_prompt_ids = self._init_prompt_ids

        # Explicit MCQ policy: default is the benchmark-official free protocol.
        self.mcq_decode_mode = resolve_mcq_decode_mode(
            {
                "mcq_decode_mode": os.environ.get("QWEN25VL_MCQ_DECODE_MODE"),
                "inference": {"mcq_decode_mode": os.environ.get("QWEN25VL_MCQ_DECODE_MODE")},
            }
        )
        self.capture_option_logits = (
            os.environ.get("QWEN25VL_CAPTURE_OPTION_LOGITS", "1") == "1"
        )
        self.last_option_logits: Optional[Dict[str, float]] = None
        self.last_step0_top: Optional[List[dict]] = None

        # Runtime accounting / profiling (all values measured, never inherited).
        self.accounting = CacheAccounting()
        self.profiler = StreamingProfiler()
        self.stream_audit = IncrementalStreamAudit()
        self._visual_tokens_seen = 0
        self._sidecar_armed = False
        self._sidecar_ids: List[int] = []
        self._sidecar_map: Dict[str, List[int]] = {}

        # Explicit Full-vs-HERMES switch. ``compression_enabled=False`` (Full)
        # never prunes; the huge sentinel budget is only an internal detail so
        # the shared HERMES trigger condition stays literally identical.
        self.compression_enabled = (
            os.environ.get("QWEN25VL_COMPRESSION_ENABLED", "1") == "1"
        )
        self.proactive_mode = False
        self._decode_calls = 0
        self.visual_tokens_per_chunk: List[int] = []
        self.seconds_per_chunk: List[float] = []
        if not self.compression_enabled:
            # Internal sentinel only: keeps the shared HERMES trigger condition
            # literally unchanged while guaranteeing it can never fire.
            self.kv_size = int(1e12)
            self.qa_model.kv_size = int(1e12)
        self._install_compression_timer()
        self._install_option_sidecar()

        print(
            "[qwen] backend=qwen2_5_vl",
            f"mode={'hermes' if self.compression_enabled else 'full'}",
            f"compression_enabled={self.compression_enabled}",
            "proactive_mode=False",
            f"attn={os.environ.get('MMDUET2_ATTN_IMPLEMENTATION', 'auto')}",
            f"mcq_decode_mode={self.mcq_decode_mode}",
            f"model={args.model_path}",
        )

    # ------------------------------------------------------------------ #
    # core streaming interface
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Drop KV cache + per-sample accounting. No text is generated."""

        # Delegates to the inherited (and instrumented) _reset_stream so the
        # kv_metrics hooks observe the reset exactly like on the MMDuet2 path.
        self._reset_stream()
        self.accounting = CacheAccounting()
        self.profiler.reset()
        self.stream_audit.start_stream()
        self._visual_tokens_seen = 0
        self.last_option_logits = None
        self._decode_calls = 0
        self.visual_tokens_per_chunk = []
        self.seconds_per_chunk = []

    def stream_video_chunk(
        self, frames: List[Image.Image], *, seconds: Optional[float] = None
    ) -> dict:
        """Incrementally prefill one chunk. Never generates any text."""

        before = self._cache_lengths()
        self._encode_frames(frames)
        after = self._cache_lengths()
        if seconds is None:
            seconds = len(frames) / float(self.sample_fps or 1.0)
        self.seconds_per_chunk.append(float(seconds))
        return {
            "frames": len(frames),
            "cache_before": max(before) if before else None,
            "cache_after": max(after) if after else None,
        }

    def effective_compaction_count(self) -> int:
        if not self.compression_enabled:
            return 0
        return int(self.accounting.effective_compaction)

    def visual_token_rate(self) -> dict:
        """Visual KV produced per chunk / per second (explanatory estimate)."""

        tokens = [t for t in self.visual_tokens_per_chunk if t > 0]
        seconds = [s for s in self.seconds_per_chunk if s > 0]
        if not tokens:
            return {"tokens_per_chunk": None, "seconds_per_chunk": None,
                    "tokens_per_second": None}
        per_chunk = sum(tokens) / len(tokens)
        per_sec_frac = sum(s for s in seconds) / len(seconds) if seconds else None
        per_second = per_chunk / per_sec_frac if per_sec_frac else None
        return {
            "tokens_per_chunk": per_chunk,
            "seconds_per_chunk": per_sec_frac,
            "tokens_per_second": per_second,
        }

    def ask_question(
        self,
        question: str,
        *,
        options: Optional[List[str]] = None,
        max_new_tokens: int = 16,
        mc: bool = False,
    ) -> str:
        """Inject the question and decode normally.

        ``mc`` only controls whether HERMES appends its legacy ``"Best option: ("``
        suffix; it defaults to ``False`` so the benchmark's own prompt is the
        only instruction given to the model.
        """

        # No proactive protocol: nothing was generated before this point.
        print(
            f"[qwen] proactive_mode={self.proactive_mode} "
            f"decode_calls_before_query={self._decode_calls}"
        )
        assert self.proactive_mode is False
        assert self._decode_calls == 0, "text was generated before the query"

        token_ids = None
        if self.mcq_decode_mode == "constrained":
            token_ids = self._allowed_mcq_token_ids(options)
        with self.profiler.phase("question_prefill"):
            raw = self._ask(
                question,
                max_new_tokens=max_new_tokens,
                mc=mc,
                allowed_first_token_ids=token_ids,
                options=options,
            )
        return raw

    # ------------------------------------------------------------------ #
    # prompt + MCQ policy
    # ------------------------------------------------------------------ #
    def _option_letters(self, options: Optional[List[str]]) -> List[str]:
        if not options:
            return []
        return list("ABCDE"[: min(len(options), 5)])

    def _option_token_ids_map(
        self, options: Optional[List[str]]
    ) -> Optional[Dict[str, List[int]]]:
        letters = self._option_letters(options)
        if not letters:
            return None
        tokenizer = self.qa_processor.tokenizer
        mapping: Dict[str, List[int]] = {}
        for letter in letters:
            ids = []
            for candidate in (letter, f" {letter}"):
                got = tokenizer(candidate, add_special_tokens=False).input_ids
                if got:
                    ids.append(int(got[0]))
            mapping[letter] = list(dict.fromkeys(ids))
        return mapping

    def _allowed_mcq_token_ids(self, options: Optional[List[str]]):
        letters = self._option_letters(options)
        if not letters:
            return None
        return mcq_first_token_ids(
            letters, self.qa_processor.tokenizer, mode="constrained"
        )

    # ------------------------------------------------------------------ #
    # read-only option-logit sidecar
    # ------------------------------------------------------------------ #
    def _install_compression_timer(self) -> None:
        """Time the HERMES compaction separately from the visual prefill."""

        original = self.qa_model.predict_and_compress

        def timed(*args, **kwargs):
            with self.profiler.phase("hermes_compression"):
                return original(*args, **kwargs)

        self.qa_model.predict_and_compress = timed

    def _install_option_sidecar(self) -> None:
        if not self.capture_option_logits:
            return
        lm_head = getattr(self.qa_model, "lm_head", None)
        if lm_head is None:
            return

        def hook(module, args, output):  # noqa: ARG001 - read-only observer
            if not getattr(self, "_sidecar_armed", False):
                return
            ids = getattr(self, "_sidecar_ids", None)
            if not ids:
                return
            row = output[..., -1, :]
            with torch.no_grad():
                picked = row[0, list(ids)].detach().float().cpu()
            self.last_option_logits = {
                str(int(i)): float(v) for i, v in zip(ids, picked.tolist())
            }
            self._sidecar_armed = False

        self._sidecar_handle = lm_head.register_forward_hook(hook)

    def _arm_sidecar(self, options: Optional[List[str]]) -> None:
        if not self.capture_option_logits:
            return
        mapping = self._option_token_ids_map(options)
        if not mapping:
            self._sidecar_armed = False
            return
        ids = [int(i) for group in mapping.values() for i in group]
        self._sidecar_ids = ids
        self.last_option_logits = None
        self._sidecar_armed = True

    def option_logits_by_letter(self) -> Optional[Dict[str, float]]:
        """Map the raw sidecar capture onto ``{letter: logit}``."""

        if not self.last_option_logits:
            return None
        mapping = getattr(self, "_sidecar_map", None) or {}
        out: Dict[str, float] = {}
        for letter, ids in mapping.items():
            vals = [self.last_option_logits.get(str(i)) for i in ids]
            vals = [v for v in vals if v is not None]
            if vals:
                out[letter] = max(vals)
        return out or None

    # ------------------------------------------------------------------ #
    # encode / ask overrides: accounting + profiling
    # ------------------------------------------------------------------ #
    def _encode_frames(self, frames: List[Image.Image]) -> None:
        if not frames:
            return
        before = self._cache_lengths()
        max_before = max(before) if before else 0
        pixels = 0
        for frame in frames:
            w, h = frame.size
            pixels += int(w) * int(h)
        events_before = len(self.compression_stats.get("events", []))
        with self.profiler.phase("stream_prefill"):
            super()._encode_frames(frames)
        after = self._cache_lengths()
        max_after = max(after) if after else 0

        if self.stream_audit.records or max_before:
            self.stream_audit.record(
                n_frames=len(frames),
                pixels=pixels,
                visual_tokens=max(0, max_after - max_before),
                cache_len_before=max_before,
                cache_len_after=max_after,
            )
        self.accounting.set_prefix_and_budget(
            int(getattr(self.qa_model, "visual_start_idx", 0) or 0),
            int(self.kv_size),
        )
        events_after = len(self.compression_stats.get("events", []))
        if events_after > events_before:
            # A HERMES compaction fired during this chunk.
            event = self.compression_stats["events"][-1]
            # Raw visual tokens PRODUCED by this chunk (pre-compaction).
            self.visual_tokens_per_chunk.append(
                max(0, int(event["before"]) - max_before)
            )
            try:
                self.accounting.on_compaction(
                    int(event["before"]), int(event["after"])
                )
            except BudgetViolation as violation:
                print(f"[QWEN25VL BUDGET VIOLATION] {violation}")
                if os.environ.get("QWEN25VL_STRICT_BUDGET", "0") == "1":
                    raise
        else:
            self.visual_tokens_per_chunk.append(max(0, max_after - max_before))
            self.accounting.on_chunk(max_before, max_after)
        self._visual_tokens_seen = max(
            0, max_after - self.accounting.protected_prefix_tokens
        )

    def _ask(self, text, *, max_new_tokens, mc=False, must_reply=False,
             allowed_first_token_ids=None, options=None, **kwargs):
        """Deterministic, benchmark-faithful decode.

        This deliberately does NOT call the inherited MMDuet2 ``_ask``:

        * no ``"I must reply."`` suffix (and it raises if one is requested);
        * no legacy ``temperature``/``repetition_penalty`` carry-over -- the
          decode is greedy ``do_sample=False`` with ``repetition_penalty=1.0``;
        * the first token is masked ONLY in ``constrained`` mode.
        """

        if must_reply:
            raise ValueError(
                "The Qwen2.5-VL backend has no mandatory-reply protocol; "
                "must_reply must stay False."
            )
        prompt = self.qa_model.get_prompt(text, mc=mc)
        input_text = {
            "question": text,
            "formatted_question": text if mc else None,
            "prompt": prompt,
        }
        answer_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "repetition_penalty": 1.0,
        }
        if "allowed_first_token_ids" in inspect.signature(
            self.qa_model.question_answering
        ).parameters:
            answer_kwargs["allowed_first_token_ids"] = allowed_first_token_ids

        if options is not None:
            self._arm_sidecar(options)
            self._sidecar_map = self._option_token_ids_map(options) or {}
        before = self._cache_lengths()
        with self.profiler.phase("decode"):
            reply = self.qa_model.question_answering(input_text, **answer_kwargs)
            self._decode_calls += 1
        after = self._cache_lengths()
        self.accounting.set_prefix_and_budget(
            int(getattr(self.qa_model, "visual_start_idx", 0) or 0),
            int(self.kv_size),
        )
        self.accounting.on_chunk(
            max(before) if before else 0,
            max(after) if after else 0,
        )
        self.last_step0_top = self._debug_step0(dict(self.last_option_logits or {}))
        return clean_reply(reply)

    def _debug_step0(self, logits: Dict[str, float]) -> Optional[List[dict]]:
        if not logits:
            return None
        tokenizer = self.qa_processor.tokenizer
        rows = []
        for key, value in sorted(logits.items(), key=lambda kv: -kv[1]):
            try:
                text = tokenizer.decode([int(key)], skip_special_tokens=False)
            except Exception:  # noqa: BLE001
                text = "?"
            rows.append({"token_id": int(key), "text": text, "logit": float(value)})
        return rows

    # ------------------------------------------------------------------ #
    # OVO backward / realtime MCQ
    # ------------------------------------------------------------------ #
    def inference(
        self,
        video_file_name: str,
        prompt: str,
        start_time: float = 0.0,
        end_time: float = 0.0,
        task: Optional[str] = None,
        sample_id: Optional[int] = None,
        **kwargs,
    ) -> str:
        # ``task`` is only used for logging here: the prompt is built by the
        # benchmark adapter (official prompt) and passed in verbatim. Forward
        # tasks (REC/SSR/CRR) have their own entry point.
        if task in self.FORWARD_TASKS:
            raise NotImplementedError(
                f"Forward task {task!r} must use inference_forward()"
            )
        start_time = float(start_time)
        query_time = float(end_time)
        relative_query_time = query_time - start_time

        options = kwargs.get("options")
        prompt_source = kwargs.get("prompt_source", "official_benchmark")

        print(
            f"\n[QWEN25VL START] id={sample_id}, task={task}, "
            f"query_time={query_time:.3f}s, video={video_file_name}, "
            f"prompt_source={prompt_source}, "
            f"compression_enabled={self.compression_enabled}"
        )
        self.reset()
        prefill_turns = 0
        final_reply = None
        try:
            stream = iter_chunks_until_query(
                video_path=video_file_name,
                query_time=relative_query_time,
                sample_fps=self.sample_fps,
                frames_per_turn=self.frames_per_turn,
                max_side=self.max_side,
                start_time=start_time,
            )
            for frames, is_query_turn, relative_turn_time in stream:
                turn_time = start_time + float(relative_turn_time)
                self.stream_video_chunk(frames, seconds=len(frames) / float(self.sample_fps or 1.0))
                if not is_query_turn:
                    prefill_turns += 1
                    continue
                final_reply = self.ask_question(
                    prompt, options=options, max_new_tokens=16
                )
                break

            if final_reply is None:
                raise RuntimeError("No query turn was executed")

            # Official protocol: the model answers freely. We only map the free
            # text onto an option letter for scoring; the decoder is never forced.
            final_answer = extract_option(final_reply)

            print(
                f"[QWEN25VL RESULT] id={sample_id}, task={task}, "
                f"prefill_turns={prefill_turns}, "
                f"kv={self._cache_length_summary()}, "
                f"effective_compaction={self.effective_compaction_count()}, "
                f"accounting={self.accounting.as_dict()}, "
                f"raw={final_reply!r}, answer={final_answer!r}"
            )
            return final_answer
        except torch.cuda.OutOfMemoryError:
            print(f"[QWEN25VL OOM] id={sample_id}, task={task}")
            raise
        finally:
            self.qa_model.clear_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # OVO forward tasks (REC / SSR / CRR)
    # ------------------------------------------------------------------ #
    def inference_forward(
        self,
        video_file_name: str,
        prompt: str,
        realtime: float,
        task=None,
        sample_id=None,
        test_index=None,
        **kwargs,
    ) -> str:
        if task not in self.FORWARD_TASKS:
            raise NotImplementedError(f"Forward does not support: {task!r}")
        print(
            f"\n[QWEN25VL FORWARD START] id={sample_id}, index={test_index}, "
            f"task={task}, realtime={realtime:.3f}s"
        )
        self.reset()
        turn_count = 0
        try:
            chunks = iter_frame_chunks(
                video_path=video_file_name,
                sample_fps=self.sample_fps,
                frames_per_turn=self.frames_per_turn,
                max_side=self.max_side,
            )
            for frames, is_last in mark_last_chunk(chunks):
                turn_count += 1
                self.stream_video_chunk(frames)
            if turn_count == 0:
                raise RuntimeError(f"Video has no valid frames: {video_file_name}")
            raw_reply = self.ask_question(
                prompt, options=None, max_new_tokens=64, mc=False
            )
            response = normalize_forward_reply(reply=raw_reply, task=task)
            print(
                f"[QWEN25VL FORWARD RESULT] id={sample_id}, task={task}, "
                f"turns={turn_count}, raw={raw_reply!r}, answer={response!r}"
            )
            return response
        except torch.cuda.OutOfMemoryError:
            print(f"[QWEN25VL FORWARD OOM] id={sample_id}, task={task}")
            raise
        finally:
            self.qa_model.clear_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
