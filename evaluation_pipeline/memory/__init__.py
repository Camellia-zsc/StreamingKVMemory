"""Pluggable visual-KV memory methods for the streaming evaluation pipeline.

This package contains the *unified interface* between the streaming pipeline and
the concrete KV-memory/compression strategy:

* :class:`~evaluation_pipeline.memory.base.KVMemoryMethod` -- the interface;
* :func:`build_memory_method` -- config-driven factory (``memory.method``);
* :class:`~evaluation_pipeline.memory.hermes_adapter.HermesMemoryMethod` --
  adapter over the existing, untouched HERMES backend.

Importing this package never imports ``torch``, HERMES or OVO-Bench, so it is
safe inside the lightweight CPU test environment.

This baseline ``__init__`` intentionally only exposes the *generic*
infrastructure plus the HERMES adapter. The InfoKV-inspired streaming method
(``evaluation_pipeline.memory.infokv_stream``) lives in a follow-up change and
registers itself through :func:`register_memory_method`.
"""

from .base import CompactResult, KVMemoryMethod, ObserveResult
from .factory import (
    available_memory_methods,
    build_memory_method,
    register_memory_method,
    resolve_memory_config,
)
from .hermes_adapter import HermesMemoryMethod
from .schemas import (
    CompressionConfig,
    LongTermConfig,
    MemoryMethodConfig,
    MemoryStats,
    Observation,
    PositionMetadata,
    RecentConfig,
    ScoringConfig,
    TokenRecord,
)

__all__ = [
    "KVMemoryMethod",
    "ObserveResult",
    "CompactResult",
    "build_memory_method",
    "register_memory_method",
    "resolve_memory_config",
    "available_memory_methods",
    "HermesMemoryMethod",
    "MemoryMethodConfig",
    "MemoryStats",
    "Observation",
    "PositionMetadata",
    "TokenRecord",
    "RecentConfig",
    "LongTermConfig",
    "CompressionConfig",
    "ScoringConfig",
]
