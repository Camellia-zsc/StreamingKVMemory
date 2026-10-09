"""Component-level runtime breakdown for MMDuet2 + HERMES streaming inference.

Only active when ``profile_runtime_breakdown`` is enabled. It wraps existing
bound methods (no source edits) and accumulates wall-clock per component with
``torch.cuda.synchronize`` around each measured call.

Important accounting rule: ``hermes_total`` *contains* importance + selection +
prune, so when computing ``T_measured_model_compute_total`` only ``hermes_total``
is added -- the sub-components are reported for the breakdown but must never be
summed again (no double counting).
"""

import functools
import inspect
import time
from typing import Dict, List, Optional

import torch


def _preserve_signature(wrapped, original):
    try:
        wrapped = functools.wraps(original)(wrapped)
    except Exception:  # noqa: BLE001
        pass
    try:
        wrapped.__signature__ = inspect.signature(original)
    except (TypeError, ValueError):
        pass
    return wrapped


COMPONENTS = [
    "video_decode",
    "vision_encoder",
    "streaming_prefill",
    "hermes_importance",
    "hermes_selection",
    "hermes_prune",
    "hermes_total",
    "query_prefill",
    "decode",
]

# components that form the (non-double-counted) model-compute total
COMPUTE_TOTAL_MEMBERS = [
    "vision_encoder",
    "streaming_prefill",
    "hermes_total",
    "query_prefill",
    "decode",
]


class RuntimeBreakdown:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.samples: Dict[str, List[float]] = {name: [] for name in COMPONENTS}
        self.run_wallclock_sec: Optional[float] = None
        self.stream_duration_sec: Optional[float] = None
        self._model = None
        self._qa_model = None
        self._chunk_module = None
        self._originals: Dict[str, object] = {}
        self._question_start: Optional[float] = None
        self._first_lm_head_seen = False
        self._vision_accumulated = 0.0

    # ------------------------------------------------------------------ #
    def _sync_time(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def record(self, name: str, elapsed: float) -> None:
        if self.enabled:
            self.samples.setdefault(name, []).append(float(elapsed))

    # ------------------------------------------------------------------ #
    def install(self, model, chunk_module, qa_model) -> "RuntimeBreakdown":
        if not self.enabled:
            return self
        self._model = model
        self._qa_model = qa_model
        self._chunk_module = chunk_module

        self._wrap_chunk_generators(chunk_module)
        self._wrap_vision_encoder(qa_model)
        self._wrap_encoder(qa_model)
        self._wrap_hermes(qa_model)
        self._wrap_question_answering(qa_model)
        self._wrap_lm_head(qa_model)
        return self

    # ------------------------------------------------------------------ #
    def _wrap_chunk_generators(self, module) -> None:
        for name in ("iter_chunks_until_query", "iter_frame_chunks"):
            original = getattr(module, name, None)
            if original is None or not callable(original):
                continue
            self._originals[name] = original

            def make(orig):
                def wrapped(*args, **kwargs):
                    inner = orig(*args, **kwargs)
                    while True:
                        t0 = self._sync_time()
                        try:
                            item = next(inner)
                        except StopIteration:
                            break
                        t1 = self._sync_time()
                        self.record("video_decode", t1 - t0)
                        yield item

                return wrapped

            setattr(module, name, make(original))

    def _wrap_vision_encoder(self, qa_model) -> None:
        for name in ("get_video_features", "get_image_features"):
            original = getattr(qa_model, name, None)
            if original is None or not callable(original):
                continue
            self._originals[name] = original

            def make(orig):
                def wrapped(*args, **kwargs):
                    t0 = self._sync_time()
                    out = orig(*args, **kwargs)
                    t1 = self._sync_time()
                    elapsed = t1 - t0
                    self._vision_accumulated += elapsed
                    self.record("vision_encoder", elapsed)
                    return out

                return wrapped

            setattr(qa_model, name, make(original))

    def _wrap_encoder(self, qa_model) -> None:
        for name in ("encode_image_chunk", "encode_video_chunk"):
            original = getattr(qa_model, name, None)
            if original is None or not callable(original):
                continue
            self._originals[name] = original

            def make(orig):
                def wrapped(*args, **kwargs):
                    vision_before = self._vision_accumulated
                    t0 = self._sync_time()
                    out = orig(*args, **kwargs)
                    t1 = self._sync_time()
                    total = t1 - t0
                    # subtract this chunk's vision-encoder time to isolate the
                    # streaming prefill of the visual tokens into the LLM.
                    vision = self._vision_accumulated - vision_before
                    self.record("streaming_prefill", max(0.0, total - vision))
                    return out

                return wrapped

            setattr(qa_model, name, make(original))

    def _wrap_hermes(self, qa_model) -> None:
        import time as _time

        def wrap(name, component=None):
            original = getattr(qa_model, name, None)
            if original is None or not callable(original):
                return
            self._originals[name] = original

            def make(orig):
                def wrapped(*args, **kwargs):
                    t0 = self._sync_time()
                    out = orig(*args, **kwargs)
                    t1 = self._sync_time()
                    elapsed = t1 - t0
                    if component and self.enabled:
                        self.samples.setdefault(component, []).append(elapsed)
                    return out

                return wrapped

            setattr(qa_model, name, make(original))

        # hermes_total = full predict_and_compress
        wrap("predict_and_compress", "hermes_total")
        # importance estimation (attention-score passes + next-question probe)
        wrap("_compute_attention_scores_manually", "hermes_importance")
        wrap("predict_next_question", "hermes_importance")
        wrap("prune_kv_cache_by_attention", "hermes_selection")
        # apply_kv_cache_pruning_strict is wrapped by KVInstrumentation for
        # prune wall-clock; add a light timer here too (profiling only).
        original_apply = qa_model.apply_kv_cache_pruning_strict
        self._originals["_rb_apply_kv_cache_pruning_strict"] = original_apply

        def apply_wrapped(*args, **kwargs):
            t0 = self._sync_time()
            out = original_apply(*args, **kwargs)
            t1 = self._sync_time()
            self.record("hermes_prune", t1 - t0)
            return out

        qa_model.apply_kv_cache_pruning_strict = _preserve_signature(
            apply_wrapped, original_apply
        )

    def _wrap_question_answering(self, qa_model) -> None:
        original = qa_model.question_answering
        self._originals["question_answering"] = original

        def wrapped(*args, **kwargs):
            self._question_start = self._sync_time()
            self._first_lm_head_seen = False
            out = original(*args, **kwargs)
            end = self._sync_time()
            start = self._question_start or end
            prefill = getattr(self, "_query_prefill_elapsed", None)
            if prefill is not None:
                self.record("query_prefill", prefill)
                self.record("decode", max(0.0, (end - start) - prefill))
            self._question_start = None
            return out

        qa_model.question_answering = _preserve_signature(wrapped, original)

    def _wrap_lm_head(self, qa_model) -> None:
        lm_head = getattr(qa_model, "lm_head", None)
        if lm_head is None:
            return

        def hook(module, args, output):
            if not self.enabled or self._question_start is None:
                return
            if not self._first_lm_head_seen:
                now = self._sync_time()
                self._query_prefill_elapsed = now - self._question_start
                self._first_lm_head_seen = True

        self._lm_head_handle = lm_head.register_forward_hook(hook)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.samples = {name: [] for name in COMPONENTS}

    def set_stream_duration(self, seconds: Optional[float]) -> None:
        if seconds is not None:
            self.stream_duration_sec = max(self.stream_duration_sec or 0.0, float(seconds))

    def set_run_wallclock(self, seconds: float) -> None:
        self.run_wallclock_sec = float(seconds)

    # ------------------------------------------------------------------ #
    def stats(self) -> List[dict]:
        rows = []
        compute_total = sum(
            sum(self.samples.get(name, [])) for name in COMPUTE_TOTAL_MEMBERS
        )
        for name in COMPONENTS:
            values = self.samples.get(name, [])
            total = sum(values)
            row = {
                "component": name,
                "call_count": len(values),
                "total_sec": total if values else None,
                "mean_sec": (total / len(values)) if values else None,
                "p50_sec": _percentile(values, 50) if values else None,
                "p95_sec": _percentile(values, 95) if values else None,
                "fraction_of_model_compute": (
                    total / compute_total
                    if values and compute_total > 0 and name in COMPUTE_TOTAL_MEMBERS
                    else None
                ),
                "stream_time_overhead_ratio": (
                    total / self.stream_duration_sec
                    if values and self.stream_duration_sec
                    else None
                ),
            }
            rows.append(row)
        return rows

    def summary(self) -> dict:
        compute_total = sum(
            sum(self.samples.get(name, [])) for name in COMPUTE_TOTAL_MEMBERS
        )
        hermes_total = sum(self.samples.get("hermes_total", []))
        prune_total = sum(self.samples.get("hermes_prune", []))
        interval = self._mean_trigger_interval or 0.0
        return {
            "measured_model_compute_total_sec": compute_total or None,
            "hermes_total_sec": hermes_total or None,
            "hermes_prune_sec": prune_total or None,
            "hermes_importance_sec": sum(self.samples.get("hermes_importance", [])) or None,
            "hermes_selection_sec": sum(self.samples.get("hermes_selection", [])) or None,
            "hermes_compute_fraction": (
                hermes_total / compute_total if compute_total > 0 else None
            ),
            "rho_prune_realtime": (
                prune_total / interval if interval > 0 and prune_total else None
            ),
            "rho_hermes_realtime": (
                hermes_total / interval if interval > 0 and hermes_total else None
            ),
            "run_wallclock_sec": self.run_wallclock_sec,
            "stream_duration_sec": self.stream_duration_sec,
        }

    # trigger interval is supplied by the caller (mean stream-time between compactions)
    _mean_trigger_interval: Optional[float] = None

    def set_mean_trigger_interval(self, seconds: Optional[float]) -> None:
        self._mean_trigger_interval = seconds

    def restore(self) -> None:
        if not self.enabled:
            return
        for name, original in self._originals.items():
            if name.startswith("_rb_"):
                continue
            target = self._chunk_module if name in ("iter_chunks_until_query", "iter_frame_chunks") else self._qa_model
            if target is not None and original is not None:
                try:
                    setattr(target, name, original)
                except Exception:  # noqa: BLE001
                    pass
        handle = getattr(self, "_lm_head_handle", None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    import math

    pos = (q / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac
