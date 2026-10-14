"""Non-invasive instrumentation.

All instrumentation is installed by wrapping existing bound methods at
runtime. No file under ``OVO-Bench/`` or ``HERMES/`` is edited.

Hook points
-----------
* ``EvalMMDuet2Hermes._reset_stream``           -> reset provenance, register prefix
* module ``iter_chunks_until_query``            -> attach per-frame timestamps
* ``qa_model.encode_image_chunk`` / ``encode_video_chunk``
                                                -> register visual token provenance
* ``qa_model.apply_kv_cache_pruning_strict``    -> the REAL compaction: time it,
                                                  mirror keep_indices on UIDs,
                                                  measure bytes before/after
* ``qa_model.question_answering``               -> snapshot pre-query cache, then
                                                  reproduce the anchor in eager mode
"""

import functools
import hashlib
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import KVMetricConfig
from .fidelity import LayerAttention, capture_anchor_attention
from .provenance import KVProvenanceTracker, VisualRegion
from .system_metrics import KVSystemProfiler, cache_token_count, kv_cache_bytes


class TimestampedFrames(list):
    """A list of frames that also carries per-frame stream timestamps."""

    frame_times: List[float]
    frame_indices: List[int]


def _is_monitoring_reply(answer) -> bool:
    """True for intermediate monitoring turns that must not be anchored.

    Mirrors the benchmark's own NO REPLY policy: a deferred turn produces
    ``NO REPLY`` (or an empty / bare ``NO``), while a real answer is a count,
    option letter or short phrase.
    """

    text = str(answer or "").strip().upper()
    if not text:
        return True
    if text == "NO" or text.startswith("NO REPLY") or text.startswith("NONE"):
        return True
    return False


def _is_forced_reply_prompt(prompt: str) -> bool:
    """True when ``_ask`` appended the mandatory-reply marker.

    ``EvalMMDuet2Hermes._ask`` appends ``"I must reply.\\n"`` when
    ``must_reply=True``; that is exactly the final answering turn of a
    streaming/monitoring protocol.
    """

    return "I must reply" in str(prompt or "")


def _query_hash(input_ids) -> str:
    """Stable hash of the diagnostic-probe query token ids.

    Used to assert that the SAME Q_star drives every trajectory point and both
    passes (hash(k=0) == ... == hash(k=K), hash_full == hash_compressed).
    """

    text = ",".join(str(int(t)) for t in (input_ids or []))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _preserve_signature(wrapped, original):
    """Make ``inspect.signature(wrapped)`` identical to the original.

    ``MMDuet2_HERMES._ask`` decides whether to forward
    ``allowed_first_token_ids`` by introspecting the signature of
    ``qa_model.question_answering``. Wrapping that method with
    ``(*args, **kwargs)`` would silently drop the argument (and thus the
    MUST-REPLY MCQ first-token constraint), so we restore the signature.
    """

    try:
        wrapped = functools.wraps(original)(wrapped)
    except Exception:  # noqa: BLE001
        pass
    try:
        wrapped.__signature__ = inspect.signature(original)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        pass
    return wrapped


@dataclass
class AnchorArtifact:
    """Everything captured at the fidelity anchor for one query event."""

    event_id: str
    video_id: Optional[str]
    query_id: Optional[str]
    query_time: Optional[float]
    stream_time: Optional[float]
    query_input_ids: List[int]
    sample_id: Optional[str] = None
    logits: Optional[torch.Tensor] = None
    # layer_idx -> (attn_row [H, kv_len], value_states [KVH, kv_len, D])
    layers: Dict[int, LayerAttention] = field(default_factory=dict)
    # provenance snapshot
    retained_uids: Dict[int, set] = field(default_factory=dict)
    visual_regions: Dict[int, VisualRegion] = field(default_factory=dict)
    uid_timestamps: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict)
    kv_bytes: int = 0
    kv_tokens: Optional[int] = None
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    summary_parent_counts: Dict[int, int] = field(default_factory=dict)
    compaction_index: int = 0
    effective_compaction_count: int = 0
    probe_k: Optional[int] = None
    probe_kind: str = "query_anchor"
    probe_query_hash: Optional[str] = None
    option_token_logits: Optional[Dict[str, float]] = None
    option_token_ids: Optional[Dict[str, list]] = None
    last_compaction_wallclock: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> dict:
        return {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "sample_id": self.sample_id,
            "query_id": self.query_id,
            "query_time": self.query_time,
            "stream_time": self.stream_time,
            "logits": self.logits,
            "kv_bytes": self.kv_bytes,
            "kv_tokens": self.kv_tokens,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "compaction_index": self.compaction_index,
            "effective_compaction_count": self.effective_compaction_count,
            "probe_k": self.probe_k,
            "probe_kind": self.probe_kind,
            "probe_query_hash": self.probe_query_hash,
            "option_token_logits": self.option_token_logits,
            "option_token_ids": self.option_token_ids,
            "last_compaction_wallclock": self.last_compaction_wallclock,
            "retained_uids": self.retained_uids,
            "visual_regions": self.visual_regions,
            "uid_timestamps": self.uid_timestamps,
            "summary_parent_counts": self.summary_parent_counts,
            "layers": self.layers,
            "extra": self.extra,
        }


class KVInstrumentation:
    """Installs and owns all hooks for one loaded HERMES model."""

    def __init__(
        self,
        config: KVMetricConfig,
        video_id: Optional[str] = None,
    ):
        config.validate()
        self.config = config
        self.tracker = KVProvenanceTracker(
            video_id=video_id, summary_token_mode=config.summary_token_mode
        )
        self.profiler = KVSystemProfiler(
            profile_wallclock=config.profile_wallclock, eps=config.eps
        )
        self.video_id = video_id
        self.model = None
        self.qa_model = None
        self.chunk_module = None

        # per-sample state
        self._frame_counter = 0
        self._current_stream_time: Optional[float] = None
        self._pending_frame_meta: List[tuple] = []
        self._pending_event_id: Optional[str] = None
        self._pending_sample_id: Optional[str] = None
        self._pending_query_id: Optional[str] = None
        self._pending_query_time: Optional[float] = None
        self.events: List[AnchorArtifact] = []
        self.compaction_events: List[dict] = []
        # Accumulators that survive per-question _reset_stream calls so the
        # runner can collect every question of a sample (not just the last).
        self.collected_events: List[AnchorArtifact] = []
        self.collected_compactions: List[dict] = []
        # Stage-ablation sidecars (stable-UID provenance + per-chunk schedule).
        self.collected_uid_dumps: List[dict] = []
        self.collected_chunks: List[dict] = []
        self._chunk_index = 0
        # cumulative per-run compaction accounting
        self._trigger_count = 0
        self._effective_count = 0
        # diagnostic probe (fixed-query fidelity probe on a long run)
        self.probe_query: Optional[dict] = None
        self.probe_target_event: Optional[str] = None
        self.probe_time_plan: List[tuple] = []
        self.probes: List[AnchorArtifact] = []
        self.collected_probes: List[AnchorArtifact] = []
        self._probe_k0_candidate: Optional[AnchorArtifact] = None
        self._probe_k0_done = False
        self._probe_time_idx = 0
        self._probe_snapshots: List[dict] = []
        # read-only decision-layer observation (official step-0 logits)
        self.option_token_ids: Optional[Dict[str, list]] = None
        self._option_capture: Optional[Dict[str, float]] = None
        self._option_armed = False
        self._compression_disabled = False
        # Rich diagnostics are per-sample: only samples in ``diagnostic_ids``
        # pay for cache snapshots / anchors / option-logit capture. Every other
        # sample in the same pass is lightweight (prediction + correctness
        # only). Empty set = original behaviour (everything is rich).
        self.diagnostic_ids = {str(v) for v in getattr(config, "diagnostic_video_ids", []) or []}
        self._original_methods: Dict[str, Any] = {}

    def _is_diagnostic(self) -> bool:
        """Whether the sample currently being answered is in the rich subset."""

        if not self.diagnostic_ids:
            return True
        current = (
            self._pending_sample_id
            if self._pending_sample_id is not None
            else self._pending_event_id
        )
        return str(current) in self.diagnostic_ids

    # ------------------------------------------------------------------ #
    # install / teardown
    # ------------------------------------------------------------------ #
    def install(self, model, chunk_module) -> "KVInstrumentation":
        """``model`` is ``EvalMMDuet2Hermes``; ``chunk_module`` is
        ``models.MMDuet2_HERMES``."""

        self.model = model
        self.qa_model = model.qa_model
        self.chunk_module = chunk_module

        self._wrap_iter_chunks(chunk_module)
        self._wrap_reset_stream(model)
        self._wrap_encoder(self.qa_model)
        self._wrap_compaction(self.qa_model)
        self._wrap_question_answering(self.qa_model)
        self._wrap_option_logits(self.qa_model)
        return self

    def disable_compression(self) -> None:
        """Force HERMES to never trigger compaction (full-KV oracle pass).

        We only touch ``kv_size`` on the already-loaded model so the code path,
        weights, frames and query schedule stay identical to the compressed run.
        """

        self._compression_disabled = True
        self._saved_kv_size = getattr(self.qa_model, "kv_size", None)
        self.qa_model.kv_size = int(1e12)
        if getattr(self.model, "kv_size", None) is not None:
            self.model.kv_size = int(1e12)

    # ------------------------------------------------------------------ #
    # wrappers
    # ------------------------------------------------------------------ #
    def _wrap_iter_chunks(self, module) -> None:
        original = module.iter_chunks_until_query
        import cv2

        def wrapped(video_path, query_time, sample_fps, frames_per_turn, max_side, start_time=0.0):
            cap = cv2.VideoCapture(video_path)
            source_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 30.0
            if source_fps <= 0:
                source_fps = 30.0
            cap.release()
            frame_step = max(1, int(round(source_fps / float(sample_fps))))
            dt = frame_step / source_fps

            for frames, is_query, t in original(
                video_path=video_path,
                query_time=query_time,
                sample_fps=sample_fps,
                frames_per_turn=frames_per_turn,
                max_side=max_side,
                start_time=start_time,
            ):
                n = len(frames)
                abs_end = float(start_time) + float(t)
                times = [abs_end - (n - 1 - i) * dt for i in range(n)]
                tf = TimestampedFrames(frames)
                tf.frame_times = times
                tf.frame_indices = list(range(self._frame_counter, self._frame_counter + n))
                # _encode_frames re-wraps frames into a plain list, losing the
                # attribute, so also queue the metadata for the encoder hook.
                self._pending_frame_meta.append((times, tf.frame_indices))
                yield tf, is_query, t

        module.iter_chunks_until_query = _preserve_signature(wrapped, original)
        self._original_methods["iter_chunks_until_query"] = original

        # Forward tasks (REC/SSR/CRR) use iter_frame_chunks instead; wrap it too
        # so visual tokens get real video timestamps there as well.
        frame_original = getattr(module, "iter_frame_chunks", None)
        if frame_original is not None:
            self._original_methods["iter_frame_chunks"] = frame_original

            def frame_wrapped(
                video_path,
                sample_fps,
                frames_per_turn,
                max_side,
                max_frames=None,
            ):
                cap = cv2.VideoCapture(video_path)
                source_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 30.0
                if source_fps <= 0:
                    source_fps = 30.0
                cap.release()
                frame_step = max(1, int(round(source_fps / float(sample_fps))))
                dt = frame_step / source_fps
                index = 0
                for frames in frame_original(
                    video_path=video_path,
                    sample_fps=sample_fps,
                    frames_per_turn=frames_per_turn,
                    max_side=max_side,
                    max_frames=max_frames,
                ):
                    n = len(frames)
                    times = [(index + i) * dt for i in range(n)]
                    indices = list(range(index, index + n))
                    index += n
                    tf = TimestampedFrames(frames)
                    tf.frame_times = times
                    tf.frame_indices = indices
                    self._pending_frame_meta.append((times, indices))
                    yield tf

            module.iter_frame_chunks = _preserve_signature(frame_wrapped, frame_original)

    def _wrap_reset_stream(self, model) -> None:
        original = model._reset_stream
        self._original_methods["_reset_stream"] = original

        def wrapped(*args, **kwargs):
            self.tracker.reset()
            self.profiler.reset()
            self.events = []
            self.compaction_events = []
            self._frame_counter = 0
            self._current_stream_time = None
            self._pending_frame_meta = []
            self._trigger_count = 0
            self._effective_count = 0
            self._probe_k0_candidate = None
            self._probe_k0_done = False
            self._probe_snapshots = []
            self.probes = []
            result = original(*args, **kwargs)
            num_layers = int(getattr(self.qa_model, "num_layers", 0))
            prefix_len = int(getattr(self.qa_model, "visual_start_idx", 0))
            self.tracker.register_init(num_layers, prefix_len)
            return result

        model._reset_stream = _preserve_signature(wrapped, original)

    def _wrap_encoder(self, qa_model) -> None:
        for name in ("encode_image_chunk", "encode_video_chunk"):
            original = getattr(qa_model, name, None)
            if original is None:
                continue
            self._original_methods[name] = original

            def make(orig):
                def wrapped(chunk, *args, **kwargs):
                    before = self._lengths()
                    result = orig(chunk, *args, **kwargs)
                    after = self._lengths()
                    if self._is_diagnostic():
                        self._register_chunk_visual(chunk, before, after)
                    return result

                return _preserve_signature(wrapped, orig)

            setattr(qa_model, name, make(original))

    def _register_chunk_visual(self, chunk, before, after) -> None:
        if not before or not after:
            return
        # frames appended = delta on the shortest layer (all layers grow equally
        # for an encode step; long-term summary tokens only appear on compaction)
        deltas = [a - b for a, b in zip(after, before)]
        appended = max(deltas)
        if appended <= 0:
            return

        n = len(chunk)
        frame_times = list(getattr(chunk, "frame_times", []) or [])
        frame_indices = list(getattr(chunk, "frame_indices", []) or [])
        if (not frame_times or len(frame_times) != n) and self._pending_frame_meta:
            queued_times, queued_indices = self._pending_frame_meta.pop(0)
            frame_times = list(queued_times)
            frame_indices = list(queued_indices)
        if not frame_times:
            frame_times = [float(self._current_stream_time or 0.0)] * n
        if not frame_indices:
            frame_indices = list(range(self._frame_counter, self._frame_counter + n))
        self._frame_counter += n

        base = appended // max(n, 1)
        counts = [base] * n
        counts[-1] += appended - base * n

        self.tracker.register_visual_chunk(counts, frame_indices, frame_times)
        self._current_stream_time = frame_times[-1] if frame_times else self._current_stream_time
        # ---- stage-ablation: per-chunk schedule record -------------------
        # Used to prove the two arms saw the same frames / same token counts
        # before any UID-level comparison is trusted.
        if os.environ.get("HERMES_DUMP_KEEP_PROVENANCE", "0") == "1":
            encode_meta = getattr(self.qa_model, "last_encode_meta", None) or {}
            self.collected_chunks.append({
                "sample_id": self._pending_sample_id,
                "chunk_index": self._chunk_index,
                "n_frames": n,
                "frame_indices": list(frame_indices),
                "frame_times": [float(t) for t in frame_times],
                "start_sec": float(min(frame_times)) if frame_times else None,
                "end_sec": float(max(frame_times)) if frame_times else None,
                "visual_tokens": int(appended),
                "encode_style": encode_meta.get("encode_style"),
                "grid_type": encode_meta.get("grid_type"),
                "grid_shape": encode_meta.get("grid_shape"),
                "pixels_shape": encode_meta.get("pixels_shape"),
                "cache_len_before": int(max(before)) if before else None,
                "cache_len_after": int(max(after)) if after else None,
            })
        self._chunk_index += 1
        # full-pass probes are driven by the compressed-pass stream times
        self._maybe_time_probe()

    def _wrap_compaction(self, qa_model) -> None:
        original = qa_model.apply_kv_cache_pruning_strict
        self._original_methods["apply_kv_cache_pruning_strict"] = original
        num_layers = int(getattr(qa_model, "num_layers", 0))

        def wrapped(keep_indices_all_layers):
            # Lightweight samples do not need compaction provenance / k.
            if not self._is_diagnostic():
                return original(keep_indices_all_layers)
            # InfoKV temporarily suppresses summary creation for its real
            # eviction. Observe the threshold at call time, not installation.
            threshold = int(getattr(qa_model, "long_term_threshold", num_layers))
            long_term_flags = [i >= threshold for i in range(num_layers)]
            trigger_time = self._current_stream_time
            bytes_before = kv_cache_bytes(qa_model.kv_cache)
            kv_before = cache_token_count(qa_model.kv_cache)
            lengths_before = self._lengths()
            # snapshot provenance BEFORE the compaction for age/retention stats
            pre_uids = [list(layer) for layer in self.tracker.per_layer]

            # k=0 baseline for the diagnostic probe: capture the PRE-prune cache
            # of the first trigger that turns out to be effective. Discarded if
            # this trigger is a no-op.
            self._probe_k0_candidate = None
            if self._probing_active() and not self._probe_k0_done:
                self._probe_k0_candidate = self._snapshot_for_probe(0)

            import time as _time

            cuda = torch.cuda.is_available()
            if self.config.profile_wallclock and cuda:
                torch.cuda.synchronize()
            t0 = _time.perf_counter()
            original(keep_indices_all_layers)
            if self.config.profile_wallclock and cuda:
                torch.cuda.synchronize()
            elapsed = _time.perf_counter() - t0

            # Mirror the compaction on the UID store.
            sanitized = [
                self._sanitize(keep_indices_all_layers[i], lengths_before[i])
                for i in range(min(len(keep_indices_all_layers), len(lengths_before)))
            ]
            self.tracker.apply_keep_indices(sanitized, long_term_flags)

            bytes_after = kv_cache_bytes(qa_model.kv_cache)
            kv_after = cache_token_count(qa_model.kv_cache)

            prefix = int(getattr(qa_model, "visual_start_idx", 0))
            managed_before = (kv_before - prefix) if kv_before is not None else None
            managed_after = (kv_after - prefix) if kv_after is not None else None
            # effective == something was actually evicted from the managed
            # visual region. Falls back to total-token comparison, which is
            # equivalent while the protected prefix length is constant in a run.
            if managed_before is not None and managed_after is not None:
                effective = managed_after < managed_before
            elif kv_before is not None and kv_after is not None:
                effective = kv_after < kv_before
            else:
                effective = False
            self._trigger_count += 1
            if effective:
                self._effective_count += 1

            try:
                self.tracker.assert_sync(qa_model.kv_cache)
            except AssertionError as err:
                self.compaction_events.append({"provenance_error": str(err)})
                raise

            event_id = self._current_event_id()
            event = self.profiler.record_compaction(
                event_id=event_id,
                compaction_index=self.tracker.compaction_index,
                stream_time=self._current_stream_time,
                bytes_before=bytes_before,
                bytes_after=bytes_after,
                kv_before=kv_before,
                kv_after=kv_after,
                wall_clock_sec=elapsed,
            )
            self.compaction_events.append(event.to_dict())
            # Video-time trigger/end. HERMES compaction is synchronous, so in
            # *stream* time trigger == end; the real cost is wall_clock_sec.
            self.compaction_events[-1]["trigger_time"] = trigger_time
            self.compaction_events[-1]["end_time"] = trigger_time
            self.compaction_events[-1]["is_effective_compaction"] = effective
            self.compaction_events[-1]["compaction_trigger_count"] = self._trigger_count
            self.compaction_events[-1]["effective_compaction_count"] = self._effective_count
            self.compaction_events[-1]["managed_visual_tokens_before"] = managed_before
            self.compaction_events[-1]["managed_visual_tokens_after"] = managed_after
            self.compaction_events[-1]["protected_prefix_tokens"] = prefix
            self.compaction_events[-1]["sample_id"] = self._pending_sample_id
            self.compaction_events[-1]["video_id"] = (
                event_id.split("|", 1)[0] if event_id else None
            )
            # Experiment 5 data: retention rate as a function of KV age.
            self.compaction_events[-1].update(
                self._retention_by_age(
                    pre_uids,
                    sanitized,
                    self._current_stream_time,
                )
            )
            self.collected_compactions.append(dict(self.compaction_events[-1]))

            # ---- stage-ablation: stable-UID provenance dump --------------
            # ``tracker.per_layer`` mirrors the POST-compaction cache, so the
            # UID strings here are the STABLE identities (vis:N assigned once,
            # carried across compactions). Only kind=="visual" enters the main
            # Jaccard; summary/system/text are counted separately.
            if os.environ.get("HERMES_DUMP_KEEP_PROVENANCE", "0") == "1":
                bin_w = float(os.environ.get("HERMES_TIME_BIN_SEC", "4"))
                per_layer_uids, per_layer_frames, per_layer_bins = {}, {}, {}
                per_layer_ages = {}
                n_summary_total = 0
                for layer_idx, uids in enumerate(self.tracker.per_layer):
                    vis = [u for u in uids if u.kind == "visual"]
                    per_layer_uids[str(layer_idx)] = [u.uid for u in vis]
                    stamps = sorted(
                        float(u.frame_timestamp_sec) for u in vis
                        if u.frame_timestamp_sec is not None
                    )
                    if stamps:
                        try:
                            import numpy as _np

                            q = _np.percentile(stamps, [10, 25, 50, 75, 90])
                            qs = [float(v) for v in q]
                        except Exception:  # noqa: BLE001
                            qs = [None] * 5
                        per_layer_ages[str(layer_idx)] = {
                            "oldest_ts": stamps[0],
                            "newest_ts": stamps[-1],
                            "mean_ts": float(sum(stamps) / len(stamps)),
                            "p10": qs[0], "p25": qs[1], "p50": qs[2],
                            "p75": qs[3], "p90": qs[4],
                            "n": len(stamps),
                        }
                    else:
                        per_layer_ages[str(layer_idx)] = {"n": 0}
                    per_layer_frames[str(layer_idx)] = sorted(
                        {int(u.frame_index) for u in vis if u.frame_index is not None}
                    )
                    per_layer_bins[str(layer_idx)] = sorted(
                        {
                            int(float(u.frame_timestamp_sec) // bin_w)
                            for u in vis
                            if u.frame_timestamp_sec is not None
                        }
                    )
                    n_summary_total += sum(1 for u in uids if u.kind == "summary")
                self.collected_uid_dumps.append({
                    "sample_id": self._pending_sample_id,
                    "video_id": (
                        event_id.split("|", 1)[0] if event_id else None
                    ),
                    "query_id": self._pending_query_id,
                    "query_time": self._pending_query_time,
                    "compaction_index": self.tracker.compaction_index,
                    "effective": bool(effective),
                    "stream_time": trigger_time,
                    "cache_len_before": [int(v) for v in lengths_before],
                    "cache_len_after": [int(v) for v in self._lengths()],
                    "managed_visual_before": managed_before,
                    "managed_visual_after": managed_after,
                    "retained_visual_uids_by_layer": per_layer_uids,
                    "retained_frames_by_layer": per_layer_frames,
                    "retained_seconds_by_layer": per_layer_bins,
                    "retained_age_stats_by_layer": per_layer_ages,
                    "observed_bins": sorted(
                        {
                            int(float(f) // bin_w)
                            for chunk in self.collected_chunks
                            for f in (chunk.get("frame_times") or [])
                        }
                    ),
                    "n_summary_uids_total": n_summary_total,
                    "time_bin_sec": bin_w,
                    "raw_keep_indices": [list(map(int, k)) for k in sanitized],
                })

            # ---- diagnostic probe bookkeeping -------------------------------
            if effective and self._probing_active():
                if not self._probe_k0_done and self._probe_k0_candidate is not None:
                    # commit the pre-prune candidate as the k=0 point
                    self._probe_snapshots.append(self._probe_k0_candidate)
                    self._probe_k0_done = True
                candidate = self._snapshot_for_probe(self._effective_count)
                if candidate is not None:
                    self._probe_snapshots.append(candidate)
            self._probe_k0_candidate = None
            return None

        qa_model.apply_kv_cache_pruning_strict = _preserve_signature(wrapped, original)

    def _retention_by_age(self, pre_uids, keep_indices_per_layer, stream_time) -> dict:
        """Per-age-bin kept/total counts of HERMES-managed visual tokens.

        Purely additive sidecar data for the "KV age vs retention" analysis.
        Retention uses the same identity semantics as RAM (a token is kept iff
        its position is in ``keep_indices``).
        """

        from .config import INF
        from .temporal import bin_index, bin_label

        bins = list(self.config.temporal_bins)
        n_bins = len(bins) - 1
        kept = [0] * n_bins
        total = [0] * n_bins
        if stream_time is None:
            return {"retention_by_age": None, "retention_age_bins": None}

        for layer_idx, uids in enumerate(pre_uids):
            if layer_idx >= len(keep_indices_per_layer):
                break
            keep_set = set(int(i) for i in keep_indices_per_layer[layer_idx])
            for pos, uid in enumerate(uids):
                if uid.kind != "visual" or uid.frame_timestamp_sec is None:
                    continue
                age = float(stream_time) - float(uid.frame_timestamp_sec)
                idx = bin_index(age, bins)
                if idx < 0:
                    continue
                total[idx] += 1
                if pos in keep_set:
                    kept[idx] += 1

        return {
            "retention_by_age": {
                bin_label(bins, i): {
                    "kept": kept[i],
                    "total": total[i],
                    "retention_rate": (kept[i] / total[i]) if total[i] else None,
                }
                for i in range(n_bins)
            },
            "retention_age_bins": [bin_label(bins, i) for i in range(n_bins)],
        }

    def _wrap_question_answering(self, qa_model) -> None:
        original = qa_model.question_answering
        self._original_methods["question_answering"] = original

        def wrapped(input_text, *args, **kwargs):
            if not self.config.enable_kv_metrics:
                return original(input_text, *args, **kwargs)
            # Lightweight samples skip the expensive full-cache snapshot and
            # anchor capture entirely; they still answer normally.
            if not self._is_diagnostic():
                return original(input_text, *args, **kwargs)

            snapshot = self._snapshot_cache()
            prompt = (input_text or {}).get("prompt", "")
            ids = qa_model.processor.tokenizer(prompt).input_ids
            offsets = qa_model._get_next_global_offset_per_layer()

            # The diagnostic probe must use the EXACT official prompt, so we
            # capture it here (the model built it itself) instead of rebuilding
            # the benchmark prompt format inside kv_metrics.
            if self.probe_query is not None and self._probing_active():
                if not self.probe_query.get("input_ids"):
                    self.probe_query["input_ids"] = list(ids)

            answer = original(input_text, *args, **kwargs)

            # Forward/monitoring protocols issue many `_ask` calls per query.
            # The anchor we want is the FINAL answering turn. MMDuet2 forces
            # that turn with a trailing "I must reply." marker (must_reply),
            # so we capture whenever we see it (the dedupe keeps the latest).
            # For protocols that never force a reply we fall back to capturing
            # the first call so an anchor is still recorded.
            pending = self._pending_event_id
            already = bool(pending) and any(
                a.event_id == pending for a in self.collected_events
            )
            forced_final = _is_forced_reply_prompt(prompt)
            if snapshot is not None and ids and (forced_final or not already):
                self._capture_anchor(
                    snapshot=snapshot,
                    input_ids=ids,
                    offsets=offsets,
                    input_text=input_text,
                )
            # materialise the deferred probe snapshots for this long run
            if self.probe_query is not None and self._probing_active():
                self._finalize_probes()
            return answer

        qa_model.question_answering = _preserve_signature(wrapped, original)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _lengths(self) -> List[int]:
        cache = self.qa_model.kv_cache
        if cache is None:
            return []
        return [int(entry[0].shape[2]) for entry in cache]

    def _sanitize(self, keep_indices, seq_len: int) -> List[int]:
        if isinstance(keep_indices, torch.Tensor):
            values = keep_indices.tolist()
        else:
            values = list(keep_indices)
        cleaned = sorted({int(i) for i in values if 0 <= int(i) < seq_len})
        return cleaned

    def _snapshot_cache(self):
        cache = self.qa_model.kv_cache
        if cache is None:
            return None
        legacy = [
            (entry[0].detach().clone(), entry[1].detach().clone())
            for entry in cache
        ]
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(legacy)
        except Exception:  # noqa: BLE001
            return legacy

    # ------------------------------------------------------------------ #
    # read-only decision-layer observation
    # ------------------------------------------------------------------ #
    def _wrap_option_logits(self, qa_model) -> None:
        """Copy the official step-0 pre-mask logits at the allowed option ids.

        Pure observer: it only READS ``lm_head``'s output for the last prompt
        position and stores a detached CPU copy of a handful of scalars. It
        never modifies the tensor, the mask, the decoder or the prediction.
        """

        lm_head = getattr(qa_model, "lm_head", None)
        if lm_head is None:
            return

        def hook(module, args, output):
            if not self._option_armed or self.option_token_ids is None:
                return
            ids = [int(i) for group in self.option_token_ids.values() for i in group]
            if not ids:
                return
            row = output[..., -1, :]
            with torch.no_grad():
                picked = row[0, ids].detach().float().cpu()
            self._option_capture = {
                str(int(i)): float(v) for i, v in zip(ids, picked.tolist())
            }
            self._option_armed = False

        self._lm_head_handle = lm_head.register_forward_hook(hook)

    def arm_option_capture(self, option_token_ids) -> None:
        self.option_token_ids = {
            str(k): [int(i) for i in v] for k, v in (option_token_ids or {}).items()
        } or None
        self._option_capture = None
        self._option_armed = self.option_token_ids is not None

    def take_option_capture(self):
        capture, self._option_capture = self._option_capture, None
        return capture
        legacy = [
            (entry[0].detach().clone(), entry[1].detach().clone())
            for entry in cache
        ]
        # transformers >=4.57 requires a Cache object (get_seq_length) for a
        # forward pass, so normalise the snapshot into a DynamicCache.
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(legacy)
        except Exception:  # noqa: BLE001 - fall back to legacy tuples
            return legacy

    def _current_event_id(self) -> str:
        return self._pending_event_id or f"{self.video_id}:compaction"

    def set_pending_event(
        self,
        event_id: str,
        query_id: Optional[str],
        query_time: Optional[float],
        sample_id: Optional[str] = None,
    ):
        self._pending_event_id = event_id
        self._pending_sample_id = sample_id
        self._pending_query_id = query_id
        self._pending_query_time = query_time

    def _build_artifact(
        self,
        *,
        event_id,
        cache,
        input_ids,
        offsets,
        probe_k=None,
        probe_kind="query_anchor",
        provenance=None,
    ) -> AnchorArtifact:
        ids = torch.tensor([list(input_ids)], device=self.qa_model.device, dtype=torch.long)
        want_attention = self.config.attention_backend == "eager"
        saved_pos = dict(getattr(self.qa_model, "_layer_position_ids", {}) or {})
        layers, logits = capture_anchor_attention(
            qa_model=self.qa_model,
            input_ids=ids,
            per_layer_offsets=offsets,
            past_key_values=cache,
            num_layers=int(getattr(self.qa_model, "num_layers", 0)) or None,
        )
        # capture_anchor_attention clears _layer_position_ids; restore the
        # official streaming state exactly (diagnostic probe must be side-effect free)
        self.qa_model._layer_position_ids.clear()
        self.qa_model._layer_position_ids.update(saved_pos)

        artifact = AnchorArtifact(
            event_id=event_id,
            video_id=self.video_id,
            sample_id=self._pending_sample_id,
            query_id=self._pending_query_id,
            query_time=self._pending_query_time,
            stream_time=self._current_stream_time,
            query_input_ids=list(input_ids),
            logits=logits,
            kv_bytes=kv_cache_bytes(cache),
            kv_tokens=cache_token_count(cache),
            num_layers=len(cache),
            num_heads=(layers[0].num_heads if layers else 0),
            num_kv_heads=(layers[0].num_kv_heads if layers else 0),
            compaction_index=self.tracker.compaction_index,
            effective_compaction_count=(
                int(probe_k) if probe_k is not None else self._effective_count
            ),
            probe_k=probe_k,
            probe_kind=probe_kind,
            probe_query_hash=_query_hash(input_ids),
            last_compaction_wallclock=(
                self.profiler.events[-1].wall_clock_sec if self.profiler.events else None
            ),
        )

        if want_attention:
            for layer_att in layers:
                layer_idx = layer_att.layer_idx
                # move to CPU so a second (full) model can be loaded later
                artifact.layers[layer_idx] = LayerAttention(
                    layer_idx=layer_idx,
                    attn_row=layer_att.attn_row.detach().cpu(),
                    value_states=layer_att.value_states.detach().cpu(),
                    num_heads=layer_att.num_heads,
                    num_kv_heads=layer_att.num_kv_heads,
                )
                if provenance is not None and layer_idx in provenance:
                    snap = provenance[layer_idx]
                    region = snap["region"]
                    retained = snap["retained"]
                    timestamps = snap["timestamps"]
                    summary_count = sum(
                        1 for uid in region.uids if uid.kind == "summary"
                    )
                else:
                    region = self.tracker.visual_region(layer_idx)
                    retained = self.tracker.retained_uid_set(layer_idx)
                    timestamps = {
                        uid.uid: uid.frame_timestamp_sec for uid in region.uids
                    }
                    summary_count = sum(
                        1 for uid in self.tracker.layer_uids(layer_idx)
                        if uid.kind == "summary"
                    )
                artifact.visual_regions[layer_idx] = region
                artifact.retained_uids[layer_idx] = retained
                artifact.uid_timestamps[layer_idx] = timestamps
                artifact.summary_parent_counts[layer_idx] = summary_count

        if artifact.logits is not None:
            artifact.logits = artifact.logits.detach().cpu()
        return artifact

    def _capture_anchor(self, snapshot, input_ids, offsets, input_text) -> None:
        event_id = self._pending_event_id or f"{self.video_id}:query"
        artifact = self._build_artifact(
            event_id=event_id, cache=snapshot, input_ids=input_ids, offsets=offsets,
        )
        self.events.append(artifact)
        # Keep only the LAST anchor per event_id: in monitoring protocols the
        # final answering turn is the one we want.
        self.collected_events = [
            a for a in self.collected_events if a.event_id != event_id
        ]
        self.collected_events.append(artifact)

    # ------------------------------------------------------------------ #
    # fixed-query diagnostic probe (sidecar only; never affects streaming)
    # ------------------------------------------------------------------ #
    def set_probe_query(self, sample_id, query_id, query_time, input_ids=None) -> None:
        self.probe_query = {
            "sample_id": None if sample_id is None else str(sample_id),
            "query_id": query_id,
            "query_time": query_time,
            "input_ids": list(input_ids) if input_ids else None,
        }
        self.probe_target_event = None

    def clear_probe(self) -> None:
        self.probe_query = None
        self.probe_target_event = None
        self.probe_time_plan = []
        self._probe_time_idx = 0
        self._probe_snapshots = []

    def set_probe_time_plan(self, plan) -> None:
        """plan = [(k, stream_time), ...] from the compressed pass (full pass)."""

        self.probe_time_plan = [(int(k), float(t)) for k, t in (plan or [])]
        self._probe_time_idx = 0

    def _probing_active(self) -> bool:
        if not self.probe_query or self.config.attention_backend != "eager":
            return False
        if self.probe_target_event is None:
            return True
        return self._pending_event_id == self.probe_target_event

    def set_probe_target(self, event_id: Optional[str]) -> None:
        """Restrict probing to a single query run (the longest query)."""

        self.probe_target_event = event_id

    def _run_probe(self, k, kind, cache, offsets=None, provenance=None) -> Optional[AnchorArtifact]:
        if not self._probing_active():
            return None
        if offsets is None:
            offsets = self.qa_model._get_next_global_offset_per_layer()
        return self._build_artifact(
            event_id=self._pending_event_id or f"{self.video_id}:probe",
            cache=cache,
            input_ids=self.probe_query["input_ids"],
            offsets=offsets,
            probe_k=int(k),
            probe_kind=kind,
            provenance=provenance,
        )

    def _clone_cache(self):
        cache = self.qa_model.kv_cache
        if cache is None:
            return None
        legacy = [
            (entry[0].detach().clone(), entry[1].detach().clone())
            for entry in cache
        ]
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(legacy)
        except Exception:  # noqa: BLE001
            return legacy

    def _snapshot_for_probe(self, k):
        """Clone the cache now; the probe forward is deferred until the exact
        official prompt is known (captured from question_answering)."""

        if not self._probing_active():
            return None
        cache = self._clone_cache()
        if cache is None:
            return None
        # Capture position offsets AND provenance NOW: both are mutated by the
        # ongoing stream (offsets grow with every chunk, provenance changes on
        # every compaction), so deferring them to finalize would mis-place the
        # probe query and use the wrong retained-token mask.
        offsets = list(self.qa_model._get_next_global_offset_per_layer())
        provenance = {}
        for layer in range(len(self.tracker.per_layer)):
            region = self.tracker.visual_region(layer)
            provenance[layer] = {
                "region": region,
                "retained": self.tracker.retained_uid_set(layer),
                "timestamps": {uid.uid: uid.frame_timestamp_sec for uid in region.uids},
            }
        return {
            "k": int(k),
            "stream_time": self._current_stream_time,
            "compaction_index": self.tracker.compaction_index,
            "offsets": offsets,
            "provenance": provenance,
            "cache": cache,
        }

    def _finalize_probes(self) -> None:
        """Compute RAM/AORE/JS/KL probes from the snapshots using the exact
        official prompt captured from the model's own question_answering call."""

        if not self._probe_snapshots:
            return
        if not self.probe_query or not self.probe_query.get("input_ids"):
            self._probe_snapshots = []
            return
        seen = set()
        for snap in sorted(self._probe_snapshots, key=lambda s: s["k"]):
            if snap["k"] in seen:
                continue
            seen.add(snap["k"])
            saved_stream = self._current_stream_time
            self._current_stream_time = snap["stream_time"]
            artifact = self._run_probe(
                k=snap["k"], kind="diagnostic_probe", cache=snap["cache"],
                offsets=snap.get("offsets"), provenance=snap.get("provenance"),
            )
            if artifact is not None and snap.get("compaction_index") is not None:
                artifact.compaction_index = int(snap["compaction_index"])
            self._current_stream_time = saved_stream
            self._commit_probe(artifact)
        self._probe_snapshots = []

    def _commit_probe(self, artifact) -> None:
        if artifact is None:
            return
        self.probes.append(artifact)
        self.collected_probes.append(artifact)

    def _maybe_time_probe(self) -> None:
        """Full pass: probe at the same stream times as the compressed pass."""

        if not self._probing_active() or not self.probe_time_plan:
            return
        if self._current_stream_time is None:
            return
        while self._probe_time_idx < len(self.probe_time_plan):
            k, t = self.probe_time_plan[self._probe_time_idx]
            if self._current_stream_time + 1e-9 < t:
                break
            snap = self._snapshot_for_probe(k)
            if snap is not None:
                self._probe_snapshots.append(snap)
            self._probe_time_idx += 1

    def reset_collected(self) -> None:
        """Clear the cross-question collectors (called by the runner)."""

        self.collected_events = []
        self.collected_compactions = []
        self.collected_probes = []
        self.collected_uid_dumps = []
        self.collected_chunks = []

    def mark_query_start(self, event_id: str) -> None:
        """Called by the runner when a query run begins (probe targeting)."""

        if self.probe_query and self.probe_query.get("query_id") == self._pending_query_id:
            self.set_probe_target(event_id)

    def restore(self) -> None:
        """Restore all wrapped methods (used when swapping run modes)."""

        model = self.model
        qa_model = self.qa_model
        if model is not None and "_reset_stream" in self._original_methods:
            model._reset_stream = self._original_methods["_reset_stream"]
        for name in ("encode_image_chunk", "encode_video_chunk"):
            if name in self._original_methods and qa_model is not None:
                setattr(qa_model, name, self._original_methods[name])
        if qa_model is not None and "apply_kv_cache_pruning_strict" in self._original_methods:
            qa_model.apply_kv_cache_pruning_strict = self._original_methods[
                "apply_kv_cache_pruning_strict"
            ]
        if qa_model is not None and "question_answering" in self._original_methods:
            qa_model.question_answering = self._original_methods["question_answering"]
        if self.chunk_module is not None and "iter_chunks_until_query" in self._original_methods:
            self.chunk_module.iter_chunks_until_query = self._original_methods[
                "iter_chunks_until_query"
            ]
        if self._compression_disabled and getattr(self, "_saved_kv_size", None) is not None:
            self.qa_model.kv_size = self._saved_kv_size
            if getattr(self.model, "kv_size", None) is not None:
                self.model.kv_size = self._saved_kv_size

    def release(self) -> None:
        """Drop strong references to the model so it can be garbage collected
        before a second pass loads its own copy."""

        try:
            self.restore()
        except Exception:  # noqa: BLE001
            pass
        self._original_methods = {}
        self.model = None
        self.qa_model = None
        self.chunk_module = None
