"""Factory: one config key selects the memory method.

The pipeline calls :func:`build_memory_method` and then only ever uses the
:class:`~evaluation_pipeline.memory.base.KVMemoryMethod` interface::

    memory = build_memory_method(config)
    memory.observe(chunk)
    if memory.should_compact():
        memory.compact()

Switching HERMES <-> InfoKV is a single config change::

    memory:
      method: hermes          # or: infokv_stream

Adding another method later is one subclass + one :func:`register_memory_method`
call -- no new ``if method == ...`` branches in ``runtime.py`` / ``evaluate.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from .base import KVMemoryMethod
from .hermes_adapter import HermesMemoryMethod
from .schemas import DEFAULT_METHOD, MemoryMethodConfig


LOG = logging.getLogger("evaluate_all")


_METHOD_REGISTRY: Dict[str, Type[KVMemoryMethod]] = {
    "hermes": HermesMemoryMethod,
}


def register_memory_method(name: str, cls: Type[KVMemoryMethod]) -> None:
    """Register a new memory method (extension point for future methods)."""

    key = str(name).strip().lower()
    if not key:
        raise ValueError("memory method name must be non-empty")
    _METHOD_REGISTRY[key] = cls


def available_memory_methods():
    return tuple(sorted(_METHOD_REGISTRY))


def resolve_memory_config(
    config: Optional[Dict[str, Any]],
    **overrides: Any,
) -> MemoryMethodConfig:
    """Build + validate a :class:`MemoryMethodConfig` from a pipeline config.

    ``config`` may be the full pipeline config (with a ``memory:`` block) or the
    ``memory:`` block directly. Backward compatible: a HERMES-era config without
    ``memory.method`` resolves to ``method == "hermes"``.
    """

    if config is None:
        block: Dict[str, Any] = {}
    elif "memory" in config:
        block = config
    else:
        block = {"memory": config}
    method_config = MemoryMethodConfig.from_config(block)
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(method_config, key):
            raise ValueError(f"unknown memory config override {key!r}")
        setattr(method_config, key, value)
    method_config.validate()
    if method_config.weight_sum() and abs(method_config.weight_sum() - 1.0) > 1e-6:
        LOG.warning(
            "memory scoring weights sum to %.6f != 1.0; they will be "
            "renormalized.",
            method_config.weight_sum(),
        )
    return method_config


def build_memory_method(
    config: Optional[Dict[str, Any]] = None,
    *,
    method_config: Optional[MemoryMethodConfig] = None,
    **kwargs: Any,
) -> KVMemoryMethod:
    """Construct the memory method selected by ``memory.method``.

    Either pass the full ``config`` (a ``memory:`` block is read from it) or a
    pre-built ``method_config``.
    """

    if method_config is None:
        method_config = resolve_memory_config(config)
    method = method_config.method or DEFAULT_METHOD
    cls = _METHOD_REGISTRY.get(method)
    if cls is None:
        raise NotImplementedError(
            f"memory method {method!r} is not registered. Available: "
            f"{available_memory_methods()}. Implement it and call "
            "register_memory_method()."
        )
    return cls(method_config, **kwargs)
