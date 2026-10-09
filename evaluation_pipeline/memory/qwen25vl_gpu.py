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
    mrope_position: Optional[List[int]] = None
    visual_rank: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token_id": int(self.token_id),
            "original_stream_position": int(self.original_stream_position),
            "cache_physical_index": int(self.cache_physical_index),
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
    ) -> None:
        for i, token_id in enumerate(token_ids):
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
            )

    def __contains__(self, token_id: int) -> bool:
        return int(token_id) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, token_id: int) -> TokenMetadata:
        return self._entries[int(token_id)]

    def cache_index(self, token_id: int) -> int:
        return self._entries[int(token_id)].cache_physical_index

    def cache_indices(self, token_ids: Sequence[int]) -> List[int]:
        return [self.cache_index(t) for t in token_ids]

    def token_ids(self) -> List[int]:
        return list(self._entries)

    def rebuild_after_compaction(self, kept_token_ids: Sequence[int]) -> None:
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
            entry.cache_physical_index = self.prefix_len + rank
            entry.visual_rank = rank
        keep_set = set(kept)
        self._entries = {t: e for t, e in self._entries.items() if t in keep_set}

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
) -> List[int]:
    """Ascending, prefix-preserving physical indices for one InfoKV compaction.

    Top-K decides *who* stays; this decides *in which cache order*. The selected
    physical indices are sorted ascending (temporal order), never by score, and
    the protected prefix is always kept.
    """

    kept = sorted(int(t) for t in kept_token_ids)
    physical = sorted(ledger.cache_index(t) for t in kept)
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

    def __init__(self, model, *, layer_a: int, layer_b: int, v2v_layer: int):
        self.model = model
        self.layer_a = int(layer_a)
        self.layer_b = int(layer_b)
        self.v2v_layer = int(v2v_layer)
        self._enabled = False
        self._buffers: Dict[str, List[Any]] = {}
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
        }
        for key, dim in (("q", 2), ("k", 2), ("hidden_a", 1), ("hidden_b", 1)):
            parts = self._buffers.get(key)
            if not parts:
                continue
            merged = self._concat(parts, dim)
            if key.startswith("hidden") and merged.dim() == 3 and merged.shape[0] == 1:
                # [B=1, T, D] -> [T, D] so it lines up with layer_evolution.
                merged = merged[0]
            out[key] = merged.detach().to(torch.float32).cpu().numpy()
        self._buffers = {}
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
        )
        self.compactor = KVCacheCompactor(self.qa_model)
        self._next_token_id = 0
        self._next_stream_position = 0
        self.events: List[CompactionEvent] = []
        self.last_sidecar: List[Dict[str, Any]] = []
        self._effective_index = 0

    def reset(self) -> None:
        self.memory.reset()
        self.ledger = VisualTokenLedger(self.prefix_len)
        self._next_token_id = 0
        self._next_stream_position = 0
        self.events = []
        self.last_sidecar = []
        self._effective_index = 0

    def close(self) -> None:
        self.capture.close()

    # -- one streamed chunk -------------------------------------------- #
    def observe_chunk(
        self, frames, *, seconds=None, frame_time: float = 0.0, encode_fn=None
    ):
        from .schemas import Observation

        encode = encode_fn or self.wrapper.stream_video_chunk
        before = max(self.wrapper._cache_lengths() or [0])
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
        mrope_positions = self._mrope_positions(cache_start, cache_start + n)

        q = captured.get("q")
        k = captured.get("k")
        hidden_a = captured.get("hidden_a")
        hidden_b = captured.get("hidden_b")

        token_embeddings = hidden_b if hidden_b is not None else hidden_a
        chunk_embedding = (
            token_embeddings.mean(axis=0) if token_embeddings is not None else None
        )
        # Observation wants per-token states as [H, D]; captured arrays are
        # [B=1, H, T, D] -> [T, H, D].
        query_states = None if q is None else np.transpose(q[0], (1, 0, 2))
        key_states = None if k is None else np.transpose(k[0], (1, 0, 2))

        observation = Observation(
            chunk_id=self.memory.chunk_counter,
            chunk_embedding=chunk_embedding,
            token_ids=token_ids,
            token_embeddings=token_embeddings,
            shallow_hidden=hidden_a,
            deep_hidden=hidden_b,
            query_states=query_states,
            key_states=key_states,
            is_visual=[True] * n,
            original_positions=stream_positions,
            frame_time=float(frame_time),
            seconds=seconds,
        )
        self.memory.observe(observation)
        self.ledger.register_chunk(
            token_ids, stream_positions, cache_indices, mrope_positions
        )

        self._next_token_id += n
        self._next_stream_position += n
        return meta, captured

    def _mrope_positions(self, start: int, end: int) -> List[List[int]]:
        cache = getattr(self.qa_model, "_position_ids_cache", None)
        if not cache or cache[0] is None:
            return [[i, i, i] for i in range(start, end)]
        positions = cache[0][:, start:end]
        return [[int(x) for x in positions[:, i]] for i in range(positions.shape[1])]

    # -- periodic compaction ------------------------------------------- #
    def maybe_compact(self) -> Optional[CompactionEvent]:
        import time

        if not self.memory.should_compact():
            return None
        self._refresh_keys_from_cache()

        cis_before = list(self.wrapper._cache_lengths() or [])
        visual_before = self._managed_visual_total()

        t0 = time.perf_counter()
        self.memory.compact()
        keep_token_ids = self.memory.retained_visual_token_ids()
        keep_indices = build_keep_indices(
            self.prefix_len,
            keep_token_ids,
            self.ledger,
            cache_len=max(cis_before) if cis_before else None,
        )
        cis_after = self.compactor.apply(
            keep_indices, num_layers=self.qa_model.num_layers
        )
        wallclock_ms = (time.perf_counter() - t0) * 1000.0

        self.ledger.rebuild_after_compaction(keep_token_ids)
        visual_after = len(keep_token_ids)
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
            description="infokv_stream global retained indices",
        )
        self.events.append(event)
        self.last_sidecar = self._build_sidecar(keep_token_ids)
        return event

    def _managed_visual_total(self):
        return sum(1 for r in self.memory.managed_tokens() if r.is_visual)

    def _refresh_keys_from_cache(self) -> None:
        cache = self.qa_model.kv_cache
        if cache is None:
            return
        keys = cache[self.v2v_layer][0]  # [B, Hkv, T, D]

        def provider(record):
            entry = self.ledger._entries.get(int(record.token_id))
            if entry is None:
                return None, None
            index = entry.cache_physical_index
            if index < 0 or index >= keys.shape[2]:
                return None, None
            return keys[0, :, index, :].detach().to("cpu").float().numpy(), None

        self.memory.refresh_context_states(provider)

    def _build_sidecar(self, keep_token_ids: Sequence[int]) -> List[Dict[str, Any]]:
        selected = set(int(t) for t in keep_token_ids)
        rows = []
        for record in self.memory.managed_tokens():
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
        "layer_b": max(0, num_layers - 4),
        "v2v_layer": num_layers // 2,
    }
