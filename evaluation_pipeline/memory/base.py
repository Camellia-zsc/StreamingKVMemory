"""Unified, pluggable interface for visual-KV memory methods.

The upper pipeline (``runtime.py`` / ``evaluate.py``) must not know whether a
method is HERMES or InfoKV. It only talks to this interface::

    memory = build_memory_method(config, ...)
    memory.reset()
    for chunk in stream:
        memory.observe(chunk)
        if memory.should_compact():
            memory.compact()
    stats = memory.get_stats()

Adding a future method (``sliding_window``, ``full_kv``, ...) means adding one
implementation here and one registry entry -- never another scattered
``if method == ...`` branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schemas import MemoryMethodConfig, MemoryStats, Observation, TokenRecord


@dataclass
class ObserveResult:
    """What a single :meth:`KVMemoryMethod.observe` call changed."""

    chunk_id: int
    n_tokens: int = 0
    retired_tokens: int = 0
    recent_size: int = 0
    candidate_size: int = 0
    chunk_embedding: Optional[List[float]] = None


@dataclass
class CompactResult:
    """What a single :meth:`KVMemoryMethod.compact` call changed."""

    triggered: bool = False
    tokens_before: int = 0
    tokens_after: int = 0
    kept_token_ids: List[int] = field(default_factory=list)
    evicted_token_ids: List[int] = field(default_factory=list)
    retention_ratio: Optional[float] = None
    wallclock: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class KVMemoryMethod:
    """Minimal unified interface every memory method implements.

    Subclasses must not require a GPU to be *constructed* or to run the CPU
    debug path.
    """

    method_name: str = "base"

    def __init__(self, config: MemoryMethodConfig):
        self.config = config

    # -- lifecycle ----------------------------------------------------- #
    def reset(self) -> None:
        raise NotImplementedError

    def observe(self, observation: Observation) -> ObserveResult:
        raise NotImplementedError

    def should_compact(self) -> bool:
        raise NotImplementedError

    def compact(self) -> CompactResult:
        raise NotImplementedError

    # -- reporting ----------------------------------------------------- #
    def get_stats(self) -> MemoryStats:
        raise NotImplementedError

    def sidecar_rows(self) -> List[Dict[str, Any]]:
        """Per-token debug/sidecar rows (method-agnostic schema)."""

        return [record.as_sidecar() for record in self.managed_tokens()]

    def managed_tokens(self) -> List[TokenRecord]:
        """All tokens the method currently keeps (recent + long-term)."""

        raise NotImplementedError

    def retained_token_ids(self) -> List[int]:
        """Retained token ids in retained (time) order."""

        records = sorted(
            self.managed_tokens(), key=lambda r: r.original_position
        )
        return [int(r.token_id) for r in records]

    # -- optional GPU hook --------------------------------------------- #
    def bind_backend(self, backend: Any) -> None:
        """Attach the live backend object (GPU integration stage only).

        The CPU first version does not need this; it exists so a future GPU
        adapter can drive the real KV cache without changing the pipeline.
        """

        self._backend = backend
