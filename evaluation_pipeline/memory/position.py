"""Method-agnostic KV compaction / position-adapter interfaces (CPU stage).

This module contains **interface only** -- no CUDA, no torch, no real cache
surgery. It documents and constrains how a future GPU integration should turn a
selection result into a new DynamicCache, and it names the *verified* HERMES
components that must be reused rather than re-implemented.

Reusable HERMES components (do NOT copy-paste a second M-RoPE implementation):

* ``HERMES/inference/reindex_3d.py``
    - ``get_cache_seq_len``
    - ``contiguous_kv``
    - ``_get_mrope_section``
    - ``compute_cos_sin_for_positions``
    - ``rotary_delta``
    - ``apply_rotary_delta_to_keys_only``
* ``HERMES/inference/qwenvl_hermes_fa2.py`` (backend class methods)
    - ``_get_cache_seq_len_per_layer`` / ``_get_next_global_offset_per_layer``
    - ``_sanitize_keep_indices``
    - ``_shrink_positions_and_rerotate_keys`` (retained gather + position
      re-index + M-RoPE re-rotation + protected-prefix handling)
    - ``apply_kv_cache_pruning_strict``
    - ``visual_start_idx`` (the protected non-visual prefix length)

Selection semantics vs. compaction semantics
--------------------------------------------
HERMES produces ``retained_indices_per_layer`` (layer-wise heterogeneous).
InfoKVStream v1 produces a single ``global_retained_token_ids`` set that is
applied identically to every layer. :class:`PositionAdapter` must accept both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class RetainedIndexSet:
    """Retained visual positions expressed either globally or per layer.

    * ``global_indices``  -- one set shared by all layers (InfoKVStream v1);
    * ``per_layer_indices`` -- one list per Transformer layer (HERMES).

    Exactly one of the two is populated. ``protected_prefix_len`` is never
    evicted.
    """

    global_indices: Optional[List[int]] = None
    per_layer_indices: Optional[List[List[int]]] = None
    protected_prefix_len: int = 0

    def validate(self) -> None:
        if (self.global_indices is None) == (self.per_layer_indices is None):
            raise ValueError(
                "provide exactly one of global_indices / per_layer_indices"
            )
        if self.protected_prefix_len < 0:
            raise ValueError("protected_prefix_len must be >= 0")

    @property
    def is_global(self) -> bool:
        return self.global_indices is not None


class PositionAdapter:
    """Reserved interface for KV position re-indexing after eviction.

    The CPU first version never invents an M-RoPE transform: it only guarantees
    that retained tokens keep their ``original_position`` order. A GPU
    integration subclasses this and delegates to the HERMES primitives listed in
    the module docstring.
    """

    def remap(self, records: Sequence[Any]) -> List[int]:
        raise NotImplementedError(
            "Position re-indexing is out of scope for the CPU first version; "
            "implement it during GPU integration by delegating to HERMES's "
            "_shrink_positions_and_rerotate_keys / reindex_3d helpers."
        )


class Qwen25VLMRoPEAdapter(PositionAdapter):
    """Qwen2.5-VL M-RoPE correction adapter (interface only, CPU stage).

    Future implementation must reuse ``HERMES/inference/reindex_3d.py``:
    ``compute_cos_sin_for_positions`` + ``rotary_delta`` +
    ``apply_rotary_delta_to_keys_only`` with ``_get_mrope_section(llm)``.
    """

    def __init__(self, mrope_section: Optional[Sequence[int]] = None):
        self.mrope_section = (
            None if mrope_section is None else tuple(int(x) for x in mrope_section)
        )

    def remap(self, records: Sequence[Any]) -> List[int]:  # pragma: no cover
        raise NotImplementedError(
            "Qwen2.5-VL M-RoPE re-index runs on the GPU cache and is not part "
            "of the CPU first version."
        )


class KVCacheCompactor:
    """Turns a selection result into a new KV cache (interface only).

    Contract for the future GPU implementation::

        retained = RetainedIndexSet(...)          # global OR per-layer
        retained.validate()
        compactor.apply(kv_cache, retained, position_adapter=...)
        # -> gather K/V at retained positions
        # -> re-index positions (M-RoPE correction)
        # -> return a new DynamicCache, protected prefix untouched
    """

    def apply(
        self,
        kv_cache: Any,
        retained: Union[RetainedIndexSet, Sequence[int]],
        *,
        position_adapter: Optional[PositionAdapter] = None,
    ) -> Any:
        raise NotImplementedError(
            "Real KV-cache surgery is a GPU-integration step. Reuse HERMES's "
            "apply_kv_cache_pruning_strict / _shrink_positions_and_rerotate_keys "
            "rather than implementing a second gather/position path."
        )
