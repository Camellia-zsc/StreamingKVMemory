"""Pluggable visual-KV memory methods for the streaming evaluation pipeline.

This package contains the *unified interface* between the streaming pipeline and
the concrete KV-memory/compression strategy:

* :class:`~evaluation_pipeline.memory.base.KVMemoryMethod` -- the interface;
* :func:`build_memory_method` -- config-driven factory (``memory.method``);
* :class:`~evaluation_pipeline.memory.hermes_adapter.HermesMemoryMethod` --
  adapter over the existing, untouched HERMES backend;
* :class:`~evaluation_pipeline.memory.infokv_stream.InfoKVStreamMemory` --
  the new **InfoKV-inspired** streaming video memory (CPU first version).

Importing this package never imports ``torch``, HERMES or OVO-Bench, so it is
safe inside the lightweight CPU test environment. See
``evaluation_pipeline/memory/README.md``.
"""

from .base import CompactResult, KVMemoryMethod, ObserveResult
from .factory import (
    available_memory_methods,
    build_memory_method,
    register_memory_method,
    resolve_memory_config,
)
from .hermes_adapter import HermesMemoryMethod
from .infokv_stream import InfoKVStreamMemory
from .position import (
    KVCacheCompactor,
    PositionAdapter,
    Qwen25VLMRoPEAdapter,
    RetainedIndexSet,
)
from .schemas import (
    CompressionConfig,
    LongTermConfig,
    MemoryMethodConfig,
    MemoryStats,
    Observation,
    RecentConfig,
    ScoringConfig,
    TokenRecord,
)
from .schemas import PositionMetadata
from .v2v import (
    ContextImportanceProvider,
    VisionToVisionAttentionProvider,
    vision_to_vision_attention,
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
    "InfoKVStreamMemory",
    "ContextImportanceProvider",
    "VisionToVisionAttentionProvider",
    "vision_to_vision_attention",
    "PositionAdapter",
    "Qwen25VLMRoPEAdapter",
    "KVCacheCompactor",
    "RetainedIndexSet",
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
