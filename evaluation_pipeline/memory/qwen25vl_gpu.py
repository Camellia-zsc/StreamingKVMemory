"""Real GPU integration for InfoKVStream on Qwen2.5-VL.

This module holds everything that needs a live model / CUDA:

* :class:`VisualTokenLedger` -- ``token_id -> cache_physical_index`` mapping
  (pure Python, CPU-testable);
* :class:`InfoKVCapture` -- read-only forward hooks capturing, for the current
  visual chunk, the two ``layer_evolution`` hidden states and the single
  ``v2v`` attention layer's already-computed Q/K;
* :class:`Qwen25VLMRoPEAdapter` -- thin wrapper over the validated
  ``HERMES/inference/reindex_3d.py`` M-RoPE primitives;
* :class:`KVCacheCompactor` -- gathers the retained KV and re-indexes positions
  by delegating to the backend's validated compaction;
* :class:`InfoKVGPUIntegrator` -- glue: observe() per chunk, compact() on the
  periodic schedule, update the ledger.

Nothing here is imported on the CPU path; the module defers ``torch`` imports so
it stays importable (and its pure-Python parts unit-testable) without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# token identity / cache-index ledger  (pure Python -> CPU testable)
# --------------------------------------------------------------------------- #
@dataclass
class TokenMetadata:
    token_id: int
    original_stream_position: int
    cache_physical_index: int
    cache_physical_indices: Dict[int, int] = field(default_factory=dict)
    mrope_position: Optional[List[int]] = None
    visual_rank: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token_id": int(self.token_id),
            "original_stream_position": int(self.original_stream_position),
            "cache_physical_index": int(self.cache_physical_index),
            "cache_physical_indices": {
                str(k): int(v) for k, v in self.cache_physical_indices.items()
            },
            "mrope_position": (
                None
                if self.mrope_position is None
                else [int(x) for x in self.mrope_position]
            ),
            "visual_rank": int(self.visual_rank),
        }


class VisualTokenLedger:
    """Maps a stable ``token_id`` to its *current* physical cache index.

    ``token_id`` is InfoKVStream's internal identity; ``cache_physical_index`` is
    where the token currently lives in the KV cache; ``original_stream_position``
    is the token's ordinal in the incoming visual stream; ``mrope_position`` is
    the model's real 3-D M-RoPE coordinate. These are kept strictly separate.
    """

    def __init__(self, prefix_len: int):
        self.prefix_len = int(prefix_len)
        self._entries: Dict[int, TokenMetadata] = {}

    def register_chunk(
        self,
        token_ids: Sequence[int],
        original_stream_positions: Sequence[int],
        cache_indices: Sequence[int],
        mrope_positions: Optional[Sequence[Sequence[int]]] = None,
        cache_indices_by_layer: Optional[Dict[int, Sequence[int]]] = None,
    ) -> None:
        for i, token_id in enumerate(token_ids):
            layer_indices = (
                {int(layer): int(values[i]) for layer, values in cache_indices_by_layer.items()}
                if cache_indices_by_layer is not None else {}
            )
            self._entries[int(token_id)] = TokenMetadata(
                token_id=int(token_id),
                original_stream_position=int(original_stream_positions[i]),
                cache_physical_index=int(cache_indices[i]),
                mrope_position=(
                    None
                    if mrope_positions is None
                    else [int(x) for x in mrope_positions[i]]
                ),
                visual_rank=int(cache_indices[i]) - self.prefix_len,
                cache_physical_indices=layer_indices,
            )

    def __contains__(self, token_id: int) -> bool:
        return int(token_id) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, token_id: int) -> TokenMetadata:
        return self._entries[int(token_id)]

    def cache_index(self, token_id: int, layer_idx: Optional[int] = None) -> int:
        entry = self._entries[int(token_id)]
        if layer_idx is not None and entry.cache_physical_indices:
            return int(entry.cache_physical_indices[int(layer_idx)])
        return entry.cache_physical_index

    def cache_indices(self, token_ids: Sequence[int], layer_idx: Optional[int] = None) -> List[int]:
        return [self.cache_index(t, layer_idx=layer_idx) for t in token_ids]

    def token_ids(self) -> List[int]:
        return list(self._entries)

    def rebuild_after_compaction(
        self, kept_token_ids: Sequence[int], layer_idx: Optional[int] = None
    ) -> None:
        """Re-index survivors to ``prefix + rank`` and drop evicted tokens.

        ``kept_token_ids`` MUST already be in temporal order (the same order the
        KV gather uses), so the new physical indices are ascending and the cache
        order matches the original stream order.
        """

        kept = [int(t) for t in kept_token_ids]
        for rank, token_id in enumerate(kept):
            entry = self._entries.get(token_id)
            if entry is None:
                continue
            new_index = self.prefix_len + rank
            if layer_idx is None:
                entry.cache_physical_index = new_index
                entry.visual_rank = rank
                for layer in list(entry.cache_physical_indices):
                    entry.cache_physical_indices[layer] = new_index
            else:
                entry.cache_physical_indices[int(layer_idx)] = new_index
        keep_set = set(kept)
        if layer_idx is None:
            self._entries = {t: e for t, e in self._entries.items() if t in keep_set}
        else:
            for token_id, entry in list(self._entries.items()):
                if token_id in keep_set:
                    continue
                entry.cache_physical_indices.pop(int(layer_idx), None)
                if not entry.cache_physical_indices:
                    self._entries.pop(token_id, None)

    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            self._entries[t].as_dict()
            for t in sorted(
                self._entries,
                key=lambda t: self._entries[t].cache_physical_index,
            )
        ]


def build_keep_indices(
    prefix_len: int,
    kept_token_ids: Sequence[int],
    ledger: VisualTokenLedger,
    *,
    cache_len: Optional[int] = None,
    layer_idx: Optional[int] = None,
) -> List[int]:
    """Ascending, prefix-preserving physical indices for one InfoKV compaction.

    Top-K decides *who* stays; this decides *in which cache order*. The selected
    physical indices are sorted ascending (temporal order), never by score, and
    the protected prefix is always kept.
    """

    kept = sorted(int(t) for t in kept_token_ids)
    physical = sorted(ledger.cache_index(t, layer_idx=layer_idx) for t in kept)
    if len(physical) != len(set(physical)):
        raise ValueError("duplicate cache indices in retained set")
    indices = list(range(int(prefix_len))) + physical
    if indices != sorted(indices):
        raise ValueError("keep indices must be ascending")
    if len(set(indices)) != len(indices):
        raise ValueError("keep indices must be unique")
    if cache_len is not None and indices and indices[-1] >= int(cache_len):
        raise ValueError(
            f"keep index {indices[-1]} out of range for cache_len={cache_len}"
        )
    if cache_len is not None and len(indices) >= int(cache_len):
        raise ValueError("keep indices cover the whole cache (no eviction)")
    return indices


# --------------------------------------------------------------------------- #
# capture (GPU only)
# --------------------------------------------------------------------------- #
class InfoKVCapture:
    """Read-only forward hooks for the InfoKV signals.

    * ``layer_a`` / ``layer_b`` decoder-layer outputs -> the D signal;
    * the ``v2v`` layer's ``q_proj`` / ``k_proj`` outputs -> the already-computed
      Q/K for the V2V signal (no re-projection).

    Hooks only *read* ``output``; they never modify inputs/outputs, so the model
    forward is identical whether the capture is enabled or not.
    """

    def __init__(self, model, *, layer_a: int, layer_b: int, v2v_layer: int,
                 layerwise: bool = False):
        self.model = model
        self.layer_a = int(layer_a)
        self.layer_b = int(layer_b)
        self.v2v_layer = int(v2v_layer)
        self.layerwise = bool(layerwise)
        self._enabled = False
        self._buffers: Dict[str, List[Any]] = {}
        self._previous_hidden = None
        self._handles: List[Any] = []
        self._install()

    def _install(self) -> None:
        language_model = getattr(self.model, "language_model", None) or getattr(
            self.model, "model", None
        )
        if language_model is None or not hasattr(language_model, "layers"):
            raise AttributeError(
                "cannot locate decoder layers on "
                f"{type(self.model).__name__} (expected .language_model.layers)"
            )
        self.language_model = language_model
        layers = language_model.layers
        # Decoder input is the projected visual embedding z, before contextual
        # processing. Pool here so novelty does not depend on deep-layer D.
        self._handles.append(layers[0].register_forward_pre_hook(
            self._capture_visual_input, with_kwargs=True
        ))
        if self.layerwise:
            for layer_idx, layer in enumerate(layers):
                self._handles.append(layer.register_forward_hook(
                    self._make_layerwise_hook(layer_idx)
                ))
        else:
            self._handles.append(
                layers[self.layer_a].register_forward_hook(
                    self._make_hidden_hook("hidden_a")
                )
            )
            self._handles.append(
                layers[self.layer_b].register_forward_hook(
                    self._make_hidden_hook("hidden_b")
                )
            )
        attn = layers[self.v2v_layer].self_attn
        self.num_q_heads = int(getattr(attn, "num_heads"))
        self.num_kv_heads = int(getattr(attn, "num_key_value_heads"))
        self.head_dim = int(getattr(attn, "head_dim"))
        self._handles.append(
            attn.q_proj.register_forward_hook(
                self._make_qk_hook("q", self.num_q_heads)
            )
        )
        self._handles.append(
            attn.k_proj.register_forward_hook(
                self._make_qk_hook("k", self.num_kv_heads)
            )
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _capture_visual_input(self, module, args, kwargs):
        if not self._enabled:
            return
        hidden = args[0] if args else kwargs["hidden_states"]
        self._buffers.setdefault("visual_sum", []).append(
            hidden.detach().float().sum(dim=(0, 1))
        )
        self._visual_count += int(hidden.shape[0] * hidden.shape[1])
        if self.layerwise:
            self._previous_hidden = hidden.detach()

    def _make_layerwise_hook(self, layer_idx: int):
        def hook(module, inputs, output):  # noqa: ARG001
            if not self._enabled:
                return
            import torch

            hidden = output[0] if isinstance(output, tuple) else output
            previous = self._previous_hidden
            if previous is not None:
                score = 1.0 - torch.nn.functional.cosine_similarity(
                    previous.float(), hidden.float(), dim=-1
                )
                self._buffers.setdefault("layer_d", {}).setdefault(
                    int(layer_idx), []
                ).append(score.clamp(0.0, 1.0).detach().reshape(-1).cpu().numpy())
            self._previous_hidden = hidden.detach()

        return hook

    def _make_hidden_hook(self, name: str):
        def hook(module, inputs, output):  # noqa: ARG001 - read-only
            if not self._enabled:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            self._buffers.setdefault(name, []).append(hidden.detach())

        return hook

    def _make_qk_hook(self, name: str, heads: int):
        def hook(module, inputs, output):  # noqa: ARG001 - read-only
            if not self._enabled:
                return
            batch, seq_len, _ = output.shape
            reshaped = (
                output.detach()
                .view(batch, seq_len, heads, self.head_dim)
                .transpose(1, 2)
            )
            self._buffers.setdefault(name, []).append(reshaped)

        return hook

    def begin_chunk(self) -> None:
        self._buffers = {}
        self._visual_count = 0
        self._previous_hidden = None
        self._enabled = True

    @staticmethod
    def _concat(parts: List[Any], dim: int):
        import torch

        return torch.cat(parts, dim=dim) if len(parts) > 1 else parts[0]

    def end_chunk(self) -> Dict[str, Optional[np.ndarray]]:
        self._enabled = False
        import torch

        out: Dict[str, Optional[np.ndarray]] = {
            "q": None,
            "k": None,
            "hidden_a": None,
            "hidden_b": None,
            "chunk_embedding": None,
            "layer_evolution_by_layer": None,
        }
        if self._buffers.get("visual_sum"):
            pooled = torch.stack(self._buffers["visual_sum"]).sum(0) / self._visual_count
            out["chunk_embedding"] = pooled.cpu().numpy()
        for key, dim in (("q", 2), ("k", 2), ("hidden_a", 1), ("hidden_b", 1)):
            parts = self._buffers.get(key)
            if not parts:
                continue
            merged = self._concat(parts, dim)
            if key.startswith("hidden") and merged.dim() == 3 and merged.shape[0] == 1:
                # [B=1, T, D] -> [T, D] so it lines up with layer_evolution.
                merged = merged[0]
            out[key] = merged.detach().to(torch.float32).cpu().numpy()
        if self._buffers.get("layer_d"):
            out["layer_evolution_by_layer"] = {
                int(layer): np.concatenate(parts, axis=0).astype(np.float32)
                for layer, parts in self._buffers["layer_d"].items()
            }
        self._buffers = {}
        self._previous_hidden = None
        return out


# --------------------------------------------------------------------------- #
# M-RoPE adapter / KV compactor (GPU only)
# --------------------------------------------------------------------------- #
class Qwen25VLMRoPEAdapter:
    """Thin wrapper over the validated HERMES M-RoPE primitives.

    It does **not** re-implement rotary math: it delegates to
    ``HERMES/inference/reindex_3d.py`` (``compute_cos_sin_for_positions`` /
    ``rotary_delta`` / ``apply_rotary_delta_to_keys_only`` /
    ``_get_mrope_section`` / ``contiguous_kv`` / ``get_cache_seq_len``).
    """

    def __init__(self, backend):
        from inference import reindex_3d  # HERMES on sys.path

        self._r = reindex_3d
        self.backend = backend
        self.mrope_section = reindex_3d._get_mrope_section(backend.language_model)

    def rotate_queries(self, raw_queries, positions):
        """Rotate pre-RoPE Q at the current cache coordinates, just like K."""
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb,
        )
        cos, sin = self._r.compute_cos_sin_for_positions(
            self.backend.language_model, raw_queries.shape[2], positions,
            raw_queries.dtype, raw_queries.device,
        )
        rotated, _ = apply_multimodal_rotary_pos_emb(
            raw_queries, raw_queries, cos, sin, self.mrope_section
        )
        return rotated

    def rerotate_keys(self, keys, position_ids_old, position_ids_new, dtype, device):
        cos_old, sin_old = self._r.compute_cos_sin_for_positions(
            self.backend.language_model,
            keys.shape[2],
            position_ids_old,
            dtype,
            device,
        )
        cos_new, sin_new = self._r.compute_cos_sin_for_positions(
            self.backend.language_model,
            keys.shape[2],
            position_ids_new,
            dtype,
            device,
        )
        cos_delta, sin_delta = self._r.rotary_delta(cos_old, sin_old, cos_new, sin_new)
        return self._r.apply_rotary_delta_to_keys_only(
            keys, cos_delta, sin_delta, self.mrope_section
        )


@dataclass
class CompactionEvent:
    trigger_index: int
    effective: bool
    visual_before: int
    visual_after: int
    evicted: int
    wallclock_ms: float
    cache_lens_before: List[int] = field(default_factory=list)
    cache_lens_after: List[int] = field(default_factory=list)
    effective_index: Optional[int] = None
    description: str = ""
    visual_before_by_layer: Dict[int, int] = field(default_factory=dict)
    visual_after_by_layer: Dict[int, int] = field(default_factory=dict)
    retained_sets_distinct: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trigger_index": int(self.trigger_index),
            "effective_index": self.effective_index,
            "effective": bool(self.effective),
            "visual_before": int(self.visual_before),
            "visual_after": int(self.visual_after),
            "evicted": int(self.evicted),
            "wallclock_ms": float(self.wallclock_ms),
            "cache_lens_before": [int(x) for x in self.cache_lens_before],
            "cache_lens_after": [int(x) for x in self.cache_lens_after],
            "visual_before_by_layer": self.visual_before_by_layer,
            "visual_after_by_layer": self.visual_after_by_layer,
            "retained_sets_distinct": self.retained_sets_distinct,
            "description": self.description,
        }


class KVCacheCompactor:
    """Applies a global InfoKV retained set to the real KV cache.

    Reuses the backend's validated ``apply_kv_cache_pruning_strict`` ->
    ``_shrink_positions_and_rerotate_keys`` gather + M-RoPE re-index path. The
    same (global) keep-index list is broadcast to every layer, so InfoKVStream
    keeps identical visual positions across all layers.

    InfoKV v1 uses **real eviction with no summary token**: the backend's
    long-term *summary* branch is suppressed for the duration of the call (and
    restored immediately) so a single global token->cache-index mapping stays
    valid. The backend code itself is not modified.
    """

    def __init__(self, backend, *, mrope_adapter=None):
        self.backend = backend
        # Built lazily: ``apply`` delegates the whole gather+reindex to the
        # backend, so the adapter is only needed if a caller rerotates itself.
        self._mrope_adapter = mrope_adapter
        self.summary_free = True

    @property
    def mrope_adapter(self):
        if self._mrope_adapter is None:
            self._mrope_adapter = Qwen25VLMRoPEAdapter(self.backend)
        return self._mrope_adapter

    def apply(self, keep_indices: Sequence[int], *, num_layers: Optional[int] = None):
        num_layers = int(num_layers or self.backend.num_layers)
        keep = [int(i) for i in keep_indices]
        keep_per_layer = [list(keep) for _ in range(num_layers)]
        saved = self.backend.long_term_threshold
        try:
            if self.summary_free:
                self.backend.long_term_threshold = num_layers
            self.backend.apply_kv_cache_pruning_strict(keep_per_layer)
        finally:
            self.backend.long_term_threshold = saved
        return self.backend._get_cache_seq_len_per_layer()

    def apply_per_layer(self, keep_indices_by_layer: Dict[int, Sequence[int]]):
        num_layers = int(self.backend.num_layers)
        keep_per_layer = [list(keep_indices_by_layer[int(layer)]) for layer in range(num_layers)]
        saved = self.backend.long_term_threshold
        try:
            if self.summary_free:
                self.backend.long_term_threshold = num_layers
            self.backend.apply_kv_cache_pruning_strict(keep_per_layer)
        finally:
            self.backend.long_term_threshold = saved
        return self.backend._get_cache_seq_len_per_layer()


# --------------------------------------------------------------------------- #
# integrator (GPU only)
# --------------------------------------------------------------------------- #
class InfoKVGPUIntegrator:
    """Ties together capture, the InfoKVStream method, the ledger and the compactor."""

    def __init__(
        self,
        wrapper,
        memory,
        *,
        layer_a: int,
        layer_b: int,
        v2v_layer: int,
        log=print,
    ):
        self.wrapper = wrapper
        self.qa_model = wrapper.qa_model
        self.memory = memory
        self.log = log
        self.layer_a = int(layer_a)
        self.layer_b = int(layer_b)
        self.v2v_layer = int(v2v_layer)
        self.prefix_len = int(self.qa_model.visual_start_idx)
        self.ledger = VisualTokenLedger(self.prefix_len)
        self.capture = InfoKVCapture(
            self.qa_model,
            layer_a=self.layer_a,
            layer_b=self.layer_b,
            v2v_layer=self.v2v_layer,
            layerwise=bool(memory.layerwise_retention),
        )
        self.compactor = KVCacheCompactor(self.qa_model)
        self._next_token_id = 0
        self._next_stream_position = 0
        self.events: List[CompactionEvent] = []
        self.last_sidecar: List[Dict[str, Any]] = []
        self._effective_index = 0
        self._raw_queries: Dict[int, np.ndarray] = {}
        self._stream_time = 0.0

    def reset(self) -> None:
        self.memory.reset()
        # The Qwen wrapper replaces the inherited system prompt. Its true
        # prefix length is known only after encode_init_prompt() in reset().
        self.prefix_len = int(self.qa_model.visual_start_idx)
        self.ledger = VisualTokenLedger(self.prefix_len)
        self._next_token_id = 0
        self._next_stream_position = 0
        self.events = []
        self.last_sidecar = []
        self._effective_index = 0
        self._raw_queries = {}
        self._stream_time = 0.0

    def close(self) -> None:
        self.capture.close()

    # -- one streamed chunk -------------------------------------------- #
    def observe_chunk(
        self, frames, *, seconds=None, frame_time: Optional[float] = None, encode_fn=None
    ):
        from .schemas import Observation

        encode = encode_fn or self.wrapper.stream_video_chunk
        before_by_layer = list(self.wrapper._cache_lengths() or [])
        before = max(before_by_layer or [0])
        self.capture.begin_chunk()
        try:
            meta = encode(frames, seconds=seconds)
        finally:
            captured = self.capture.end_chunk()
        after = max(self.wrapper._cache_lengths() or [0])

        n = int(after - before)
        if n <= 0:
            return meta, None

        cache_start = int(before)
        token_ids = list(range(self._next_token_id, self._next_token_id + n))
        stream_positions = list(
            range(self._next_stream_position, self._next_stream_position + n)
        )
        cache_indices = list(range(cache_start, cache_start + n))
        cache_indices_by_layer = {
            int(layer): list(range(int(start), int(start) + n))
            for layer, start in enumerate(before_by_layer)
        }
        mrope_positions = self._mrope_positions(cache_start, cache_start + n)

        q = captured.get("q")
        k = captured.get("k")
        hidden_a = captured.get("hidden_a")
        hidden_b = captured.get("hidden_b")

        chunk_embedding = captured.get("chunk_embedding")
        if self.memory.config.scoring.temporal_novelty_enabled and chunk_embedding is None:
            raise RuntimeError("InfoKV novelty requires projected visual input capture")
        # Observation wants per-token states as [H, D]; captured arrays are
        # [B=1, H, T, D] -> [T, H, D].
        query_states = None if q is None else np.transpose(q[0], (1, 0, 2))
        key_states = None if k is None else np.transpose(k[0], (1, 0, 2))

        observation = Observation(
            chunk_id=self.memory.chunk_counter,
            chunk_embedding=chunk_embedding,
            token_ids=token_ids,
            shallow_hidden=hidden_a,
            deep_hidden=hidden_b,
            layer_evolution_by_layer=captured.get("layer_evolution_by_layer"),
            query_states=query_states,
            key_states=key_states,
            is_visual=[True] * n,
            original_positions=stream_positions,
            frame_time=self._stream_time if frame_time is None else float(frame_time),
            seconds=float(seconds or 0.0),
        )
        self.memory.observe(observation)
        self.ledger.register_chunk(
            token_ids, stream_positions, cache_indices, mrope_positions,
            cache_indices_by_layer=cache_indices_by_layer,
        )
        if query_states is not None:
            self._raw_queries.update(zip(token_ids, query_states))
        # Only the observation window can be queried at the next compaction.
        recent_ids = {r.token_id for r in self.memory.recent[-self.memory.observation_tokens:]}
        self._raw_queries = {t: q for t, q in self._raw_queries.items() if t in recent_ids}
        self._stream_time += float(seconds or 0.0)

        self._next_token_id += n
        self._next_stream_position += n
        return meta, captured

    def _mrope_positions(self, start: int, end: int) -> List[List[int]]:
        cache = getattr(self.qa_model, "_position_ids_cache", None)
        if not cache or cache[0] is None:
            return [[i, i, i] for i in range(start, end)]
        positions = cache[self.v2v_layer][:, start:end]
        return positions.detach().to("cpu").T.tolist()

    # -- periodic compaction ------------------------------------------- #
    def maybe_compact(self) -> Optional[CompactionEvent]:
        import time

        if not self.memory.should_compact():
            return None
        profiler = getattr(self.wrapper, "profiler", None)
        timer = profiler.phase("hermes_compression") if profiler is not None else None
        if timer is not None:
            timer.__enter__()
        try:
            return self._maybe_compact_timed()
        finally:
            if timer is not None:
                timer.__exit__(None, None, None)

    def _maybe_compact_timed(self) -> Optional[CompactionEvent]:
        import time
        cis_before = list(self.wrapper._cache_lengths() or [])
        visual_before = max(cis_before or [self.prefix_len]) - self.prefix_len

        t0 = time.perf_counter()
        self._refresh_keys_from_cache()
        candidates_before = list(self.memory.managed_tokens())
        self.memory.compact()
        keep_token_ids = self.memory.retained_visual_token_ids()
        per_layer = (self.memory.retained_visual_token_ids_by_layer()
                     if self.memory.layerwise_retention else
                     {layer: keep_token_ids for layer in range(self.qa_model.num_layers)})
        # A periodic trigger below budget is valid; never prune in that case.
        effective = any(len(per_layer[layer]) < n - self.prefix_len
                        for layer, n in enumerate(cis_before))
        self.last_sidecar = self._build_sidecar(keep_token_ids, candidates_before)
        cis_after = cis_before
        if effective:
            indices_by_layer = {
                layer: build_keep_indices(self.prefix_len, ids, self.ledger,
                                          cache_len=cis_before[layer], layer_idx=layer)
                for layer, ids in per_layer.items()
            }
            cis_after = self.compactor.apply_per_layer(indices_by_layer)
            for layer, ids in per_layer.items():
                self.ledger.rebuild_after_compaction(ids, layer_idx=layer)
            for entry in self.ledger._entries.values():
                available = entry.cache_physical_indices
                entry.cache_physical_index = available.get(self.v2v_layer, next(iter(available.values())))
                entry.visual_rank = entry.cache_physical_index - self.prefix_len
            # The representative coordinates belong to the V2V signal layer.
            ids = per_layer[self.v2v_layer]
            positions = self._mrope_positions(self.prefix_len, self.prefix_len + len(ids))
            for token_id, position in zip(ids, positions):
                self.ledger.get(token_id).mrope_position = position
        for row in self.last_sidecar:
            entry = self.ledger._entries.get(row["token_id"])
            row["cache_index_after"] = entry.cache_physical_index if entry else None
        wallclock_ms = (time.perf_counter() - t0) * 1000.0

        visual_after = max((len(ids) for ids in per_layer.values()), default=0)
        evicted = max(0, visual_before - visual_after)
        effective = evicted > 0
        if effective:
            self._effective_index += 1
        event = CompactionEvent(
            trigger_index=len(self.events) + 1,
            effective=effective,
            visual_before=visual_before,
            visual_after=visual_after,
            evicted=evicted,
            wallclock_ms=wallclock_ms,
            cache_lens_before=[int(x) for x in cis_before],
            cache_lens_after=[int(x) for x in (cis_after or [])],
            effective_index=self._effective_index if effective else None,
            description=("infokv_stream layerwise retained indices" if self.memory.layerwise_retention
                         else "infokv_stream global retained indices"),
            visual_before_by_layer={layer: n-self.prefix_len for layer, n in enumerate(cis_before)},
            visual_after_by_layer={layer: len(ids) for layer, ids in per_layer.items()},
            retained_sets_distinct=len({tuple(ids) for ids in per_layer.values()}),
        )
        self.events.append(event)
        return event

    def _managed_visual_total(self):
        return sum(1 for r in self.memory.managed_tokens() if r.is_visual)

    def _refresh_keys_from_cache(self) -> None:
        import torch

        if not self.memory.config.scoring.v2v_enabled:
            return
        cache = self.qa_model.kv_cache
        if cache is None:
            return
        keys = cache[self.v2v_layer][0]  # [B, Hkv, T, D]
        records = list(self.memory.managed_tokens())
        if self.memory.layerwise_retention:
            # A remains shared across layers.  Keep the last observed anchor-
            # layer key for tokens no longer live there, and refresh live keys.
            # This signal bank is CPU metadata, not an additional model KV slot.
            records = [r for r in records if self.v2v_layer in
                       self.ledger.get(r.token_id).cache_physical_indices]
        if not records:
            return
        indices = torch.tensor(self.ledger.cache_indices([r.token_id for r in records], layer_idx=self.v2v_layer),
                               device=keys.device, dtype=torch.long)
        # One device transfer for the managed K, instead of one per token.
        selected_keys = keys[0].index_select(1, indices).transpose(0, 1).detach().float().cpu().numpy()
        for record, key in zip(records, selected_keys):
            record.key_state = key

        recent = [r for r in self.memory.recent[-self.memory.observation_tokens:]
                  if r.token_id in self._raw_queries]
        if recent:
            positions = self.qa_model._position_ids_cache[self.v2v_layer]
            q_indices = torch.tensor(self.ledger.cache_indices([r.token_id for r in recent], layer_idx=self.v2v_layer),
                                     device=positions.device, dtype=torch.long)
            raw = np.stack([self._raw_queries[r.token_id] for r in recent])
            raw = torch.as_tensor(raw, device=keys.device, dtype=keys.dtype).permute(1, 0, 2).unsqueeze(0)
            rotated = self.compactor.mrope_adapter.rotate_queries(
                raw, positions.index_select(1, q_indices)
            )[0].transpose(0, 1).detach().float().cpu().numpy()
            for record, query in zip(recent, rotated):
                record.query_state = query

    def _build_sidecar(self, keep_token_ids: Sequence[int], records=None) -> List[Dict[str, Any]]:
        selected = set(int(t) for t in keep_token_ids)
        rows = []
        for record in self.memory.managed_tokens() if records is None else records:
            entry = self.ledger._entries.get(int(record.token_id))
            row = record.as_sidecar()
            row.update(
                {
                    "original_position": (
                        entry.original_stream_position
                        if entry
                        else row["original_position"]
                    ),
                    "cache_index_before": (
                        entry.cache_physical_index if entry else None
                    ),
                    "mrope_position": None if entry is None else entry.mrope_position,
                    "selected": int(record.token_id) in selected,
                }
            )
            rows.append(row)
        return rows


def default_layer_choices(num_layers: int) -> Dict[str, int]:
    """Defaults when the config leaves layer_a/layer_b/v2v.layer null."""

    return {
        "layer_a": min(2, num_layers - 1),
        # Keep a deep representation while leaving a small tail for the
        # language-model output; this is the established Qwen default and is
        # compatible with the CPU/GPU layer-resolution contract.
        "layer_b": max(0, num_layers - 4),
        "v2v_layer": num_layers // 2,
    }
