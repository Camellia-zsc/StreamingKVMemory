"""Data schemas for the pluggable visual-KV memory methods.

Everything here is plain Python + numpy: importing this module must never pull
in ``torch``, ``transformers``, HERMES or OVO-Bench, so the whole memory layer
stays CPU-importable and unit-testable without weights.

The two dataclasses that matter at the pipeline boundary are:

* :class:`Observation` -- one streamed video chunk handed to a memory method;
* :class:`TokenRecord` -- the per-token bookkeeping a method keeps internally
  and the row it emits to the method-agnostic debug sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# per-token identity / position
# --------------------------------------------------------------------------- #
@dataclass
class PositionMetadata:
    """Original temporal position of a visual token.

    This is deliberately *descriptive only*: it records where a token came from
    so that compaction can re-emit tokens in time order. No M-RoPE / position
    re-index transform is applied here -- that belongs to the (future) GPU
    integration stage, behind the ``PositionAdapter`` interface.
    """

    original_position: int
    chunk_id: int
    frame_time: float = 0.0
    token_index: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_position": int(self.original_position),
            "chunk_id": int(self.chunk_id),
            "frame_time": float(self.frame_time),
            "token_index": int(self.token_index),
        }


@dataclass
class TokenRecord:
    """One visual token tracked by a memory method."""

    token_id: int
    chunk_id: int
    original_position: int
    token_index: int = 0
    frame_time: float = 0.0
    # Vision-to-Vision importance only ever applies to real visual tokens.
    is_visual: bool = True
    # Raw (un-normalized) signals.
    novelty_raw: float = 0.0
    evolution_raw: float = 0.0
    v2v_raw: float = 0.0
    # Normalized / fused outputs (filled at compaction time).
    novelty_norm: Optional[float] = None
    evolution_norm: Optional[float] = None
    v2v_norm: Optional[float] = None
    final_score: Optional[float] = None
    selected: Optional[bool] = None
    retained_order: Optional[int] = None
    # Per-token representation used by the V2V provider (populated only when
    # the caller supplies real/synthetic visual Q/K states). For the true GPU
    # path these are slices of the actual KV cache; on CPU they are synthetic.
    key_state: Optional[Any] = None
    query_state: Optional[Any] = None

    @property
    def position(self) -> PositionMetadata:
        return PositionMetadata(
            original_position=self.original_position,
            chunk_id=self.chunk_id,
            frame_time=self.frame_time,
            token_index=self.token_index,
        )

    def as_sidecar(self) -> Dict[str, Any]:
        """Method-agnostic debug row (see spec section I)."""

        return {
            "token_id": int(self.token_id),
            "chunk_id": int(self.chunk_id),
            "original_position": int(self.original_position),
            "token_index": int(self.token_index),
            "frame_time": float(self.frame_time),
            "is_visual": bool(self.is_visual),
            "novelty_raw": float(self.novelty_raw),
            "novelty_norm": _opt_float(self.novelty_norm),
            "evolution_raw": float(self.evolution_raw),
            "evolution_norm": _opt_float(self.evolution_norm),
            "v2v_raw": float(self.v2v_raw),
            "v2v_norm": _opt_float(self.v2v_norm),
            # Deprecated aliases kept so earlier sidecar consumers do not break.
            "attention_raw": float(self.v2v_raw),
            "attention_norm": _opt_float(self.v2v_norm),
            "final_score": _opt_float(self.final_score),
            "selected": (
                None if self.selected is None else bool(self.selected)
            ),
        }


def _opt_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


# --------------------------------------------------------------------------- #
# a single streamed chunk handed to a method
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """One streamed video chunk.

    Only ``chunk_embedding`` (or ``token_embeddings``) is required to compute
    temporal novelty. The layer-evolution and V2V signals are optional so the
    CPU debug path can synthesize them, and a future GPU path can feed real
    hidden states / KV-cache slices without changing the interface.
    """

    chunk_id: Optional[int] = None
    chunk_embedding: Optional[Sequence[float]] = None
    token_ids: Optional[Sequence[int]] = None
    token_embeddings: Optional[Any] = None  # array [n_tokens, dim]
    shallow_hidden: Optional[Any] = None  # array [n_tokens, dim]
    deep_hidden: Optional[Any] = None  # array [n_tokens, dim]
    # Direct A^V2V injection: either a sequence aligned with the chunk's tokens
    # or a ``{token_id: score}`` mapping. Used when no V2V provider is wired
    # (HERMES-style importance or synthetic scores).
    attention_importance: Optional[Any] = None
    # Vision-to-Vision provider inputs (visual tokens only). Layout:
    #   query_states: [n, Hq, D] or [n, D]
    #   key_states:   [n, Hkv, D] or [n, D]
    query_states: Optional[Any] = None
    key_states: Optional[Any] = None
    # Per-token visual mask. ``None`` == all tokens are visual. Text / prefix /
    # question tokens MUST be marked ``False`` so V2V stays strictly
    # vision-to-vision.
    is_visual: Optional[Sequence[bool]] = None
    original_positions: Optional[Sequence[int]] = None
    frame_time: float = 0.0
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DEFAULT_METHOD = "hermes"
KNOWN_METHODS = ("hermes", "infokv_stream", "full_cache", "sliding_window")


@dataclass
class RecentConfig:
    enabled: bool = True
    budget: int = 0


@dataclass
class LongTermConfig:
    budget: int = 0


@dataclass
class CompressionConfig:
    interval_chunks: int = 1
    summary_strategy: str = "none"


@dataclass
class ScoringConfig:
    """Weights for the fused InfoKV-inspired score (spec sections D and I)."""

    temporal_novelty_enabled: bool = True
    temporal_novelty_weight: float = 0.35
    # ``previous`` or ``recent_mean``.
    temporal_novelty_reference: str = "recent_mean"
    temporal_novelty_reference_chunks: int = 4

    layer_evolution_enabled: bool = True
    layer_evolution_weight: float = 0.35
    # The two Transformer layers whose hidden states define D. GPU integration
    # captures ONLY these two (never ``output_hidden_states=True``).
    layer_a: Optional[int] = None
    layer_b: Optional[int] = None

    # A^V2V: Vision-to-Vision recent-context importance (LiveVLM-inspired).
    v2v_enabled: bool = True
    v2v_weight: float = 0.30
    v2v_observation_tokens: int = 32
    v2v_aggregate_queries: str = "mean"
    v2v_aggregate_heads: str = "mean"
    # Single Transformer layer whose Q/K are used (None -> a default chosen at
    # GPU-integration time). We never aggregate across all layers.
    v2v_layer: Optional[int] = None

    def weights(self) -> Dict[str, float]:
        return {
            "temporal_novelty": float(self.temporal_novelty_weight),
            "layer_evolution": float(self.layer_evolution_weight),
            "v2v": float(self.v2v_weight),
        }

    def enabled(self) -> Dict[str, bool]:
        return {
            "temporal_novelty": bool(self.temporal_novelty_enabled),
            "layer_evolution": bool(self.layer_evolution_enabled),
            "v2v": bool(self.v2v_enabled),
        }


@dataclass
class MemoryMethodConfig:
    """Normalized ``memory:`` block shared by every memory method.

    Backward compatibility: an existing HERMES config that only has a
    ``memory: {compression: {budget: ...}}`` block (no ``method``) resolves to
    ``method == "hermes"`` -- see :meth:`from_config`.
    """

    method: str = DEFAULT_METHOD
    # Total managed VISUAL token budget (excludes any protected prompt prefix).
    total_visual_budget: Optional[int] = None
    recent: RecentConfig = field(default_factory=RecentConfig)
    long_term: LongTermConfig = field(default_factory=LongTermConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    # HERMES passthrough knobs (kept so a HERMES config is fully described).
    hermes_budget: Optional[int] = None
    hermes_compression_enabled: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "MemoryMethodConfig":
        block = dict(((config or {}).get("memory") or {}))
        method = str(block.get("method", DEFAULT_METHOD)).strip().lower()
        if not method:
            method = DEFAULT_METHOD

        compression_block = dict(block.get("compression") or {})
        recent_block = dict(block.get("recent") or {})
        long_term_block = dict(block.get("long_term") or {})
        scoring_block = dict(block.get("scoring") or {})
        tn = dict(scoring_block.get("temporal_novelty") or {})
        le = dict(scoring_block.get("layer_evolution") or {})
        # ``v2v`` is the frozen name; ``recent_context_importance`` is accepted
        # as a backward-compatible alias from the previous CPU round.
        v2v = dict(
            scoring_block.get("v2v")
            or scoring_block.get("recent_context_importance")
            or {}
        )

        total_visual_budget = block.get("total_visual_budget")
        hermes_budget = compression_block.get(
            "budget", block.get("budget")
        )
        if total_visual_budget is None and method == "infokv_stream":
            # Allow InfoKV to inherit a legacy ``compression.budget`` if set.
            total_visual_budget = hermes_budget

        scoring = ScoringConfig(
            temporal_novelty_enabled=bool(tn.get("enabled", True)),
            temporal_novelty_weight=float(tn.get("weight", 0.35)),
            temporal_novelty_reference=str(
                tn.get("reference", "recent_mean")
            ),
            temporal_novelty_reference_chunks=int(
                tn.get(
                    "reference_chunks",
                    tn.get("recent_window", 4),
                )
            ),
            layer_evolution_enabled=bool(le.get("enabled", True)),
            layer_evolution_weight=float(le.get("weight", 0.35)),
            layer_a=_opt_int(
                le.get("layer_a", le.get("shallow_layer"))
            ),
            layer_b=_opt_int(
                le.get("layer_b", le.get("deep_layer"))
            ),
            v2v_enabled=bool(v2v.get("enabled", True)),
            v2v_weight=float(v2v.get("weight", 0.30)),
            v2v_observation_tokens=int(v2v.get("observation_tokens", 32) or 32),
            v2v_aggregate_queries=str(v2v.get("aggregate_queries", "mean")),
            v2v_aggregate_heads=str(v2v.get("aggregate_heads", "mean")),
            v2v_layer=_opt_int(v2v.get("layer")),
        )

        return cls(
            method=method,
            total_visual_budget=_opt_int(total_visual_budget),
            recent=RecentConfig(
                enabled=bool(recent_block.get("enabled", True)),
                budget=int(recent_block.get("budget", 0) or 0),
            ),
            long_term=LongTermConfig(
                budget=int(long_term_block.get("budget", 0) or 0),
            ),
            compression=CompressionConfig(
                interval_chunks=int(
                    compression_block.get("interval_chunks", 1) or 1
                ),
                summary_strategy=str(
                    compression_block.get("summary_strategy", "none")
                ),
            ),
            scoring=scoring,
            hermes_budget=_opt_int(hermes_budget),
            hermes_compression_enabled=_opt_bool(
                compression_block.get(
                    "enabled", block.get("compression_enabled")
                )
            ),
        )

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if self.method not in KNOWN_METHODS:
            raise ValueError(
                f"Unknown memory method {self.method!r}; expected one of "
                f"{KNOWN_METHODS}. (Add it to the memory factory registry.)"
            )
        if self.compression.interval_chunks < 1:
            raise ValueError(
                "memory.compression.interval_chunks must be >= 1, got "
                f"{self.compression.interval_chunks}"
            )
        if self.compression.summary_strategy != "none":
            raise NotImplementedError(
                "The InfoKV streaming first version never fabricates summary "
                "tokens; memory.compression.summary_strategy must be 'none'."
            )
        if self.method == "infokv_stream":
            self._validate_infokv()

    def _validate_infokv(self) -> None:
        total = self.total_visual_budget
        if total is None:
            raise ValueError(
                "memory.total_visual_budget is required for infokv_stream"
            )
        if int(total) <= 0:
            raise ValueError(
                f"memory.total_visual_budget must be > 0, got {total}"
            )
        if not self.recent.enabled:
            raise ValueError(
                "infokv_stream requires memory.recent.enabled = true "
                "(the Recent Buffer is the structural first stage)"
            )
        recent = int(self.recent.budget)
        long_term = int(self.long_term.budget)
        if recent < 0 or long_term < 0:
            raise ValueError(
                "memory.recent.budget and memory.long_term.budget must be >= 0"
            )
        if recent + long_term > int(total):
            raise ValueError(
                "memory.recent.budget + memory.long_term.budget "
                f"({recent} + {long_term} = {recent + long_term}) must be "
                f"<= memory.total_visual_budget ({int(total)})"
            )
        weights = self.scoring.weights()
        for name, value in weights.items():
            if value < 0:
                raise ValueError(
                    f"memory.scoring.{name}.weight must be >= 0, got {value}"
                )
        if not any(self.scoring.enabled().values()):
            raise ValueError(
                "infokv_stream requires at least one scoring signal enabled"
            )
        if self.scoring.temporal_novelty_enabled:
            reference = self.scoring.temporal_novelty_reference
            if reference not in {"previous", "recent_mean"}:
                raise ValueError(
                    "memory.scoring.temporal_novelty.reference must be "
                    f"'previous' or 'recent_mean', got {reference!r}"
                )
            if int(self.scoring.temporal_novelty_reference_chunks) < 1:
                raise ValueError(
                    "memory.scoring.temporal_novelty.reference_chunks must be >= 1"
                )
        if self.scoring.v2v_enabled:
            if int(self.scoring.v2v_observation_tokens) < 1:
                raise ValueError(
                    "memory.scoring.v2v.observation_tokens must be >= 1"
                )
            if self.scoring.v2v_layer is not None and self.scoring.v2v_layer < 0:
                raise ValueError(
                    "memory.scoring.v2v.layer must be >= 0 or null"
                )
            if self.scoring.v2v_aggregate_queries not in {"mean"}:
                raise ValueError(
                    "memory.scoring.v2v.aggregate_queries must be 'mean' "
                    f"(first version), got {self.scoring.v2v_aggregate_queries!r}"
                )
            if self.scoring.v2v_aggregate_heads not in {"mean"}:
                raise ValueError(
                    "memory.scoring.v2v.aggregate_heads must be 'mean' "
                    f"(first version), got {self.scoring.v2v_aggregate_heads!r}"
                )
        if self.scoring.layer_evolution_enabled:
            a, b = self.scoring.layer_a, self.scoring.layer_b
            if (a is None) != (b is None):
                raise ValueError(
                    "memory.scoring.layer_evolution.layer_a and layer_b must be "
                    "set together (or both null for the CPU/synthetic path)"
                )
            if a is not None and a == b:
                raise ValueError(
                    "memory.scoring.layer_evolution.layer_a and layer_b must differ"
                )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def weight_sum(self) -> float:
        return float(sum(self.scoring.weights().values()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "total_visual_budget": self.total_visual_budget,
            "recent": {"enabled": self.recent.enabled, "budget": self.recent.budget},
            "long_term": {"budget": self.long_term.budget},
            "compression": {
                "interval_chunks": self.compression.interval_chunks,
                "summary_strategy": self.compression.summary_strategy,
            },
            "scoring": {
                "temporal_novelty": {
                    "enabled": self.scoring.temporal_novelty_enabled,
                    "weight": self.scoring.temporal_novelty_weight,
                    "reference": self.scoring.temporal_novelty_reference,
                    "reference_chunks": self.scoring.temporal_novelty_reference_chunks,
                },
                "layer_evolution": {
                    "enabled": self.scoring.layer_evolution_enabled,
                    "weight": self.scoring.layer_evolution_weight,
                    "layer_a": self.scoring.layer_a,
                    "layer_b": self.scoring.layer_b,
                },
                "v2v": {
                    "enabled": self.scoring.v2v_enabled,
                    "weight": self.scoring.v2v_weight,
                    "observation_tokens": self.scoring.v2v_observation_tokens,
                    "aggregate_queries": self.scoring.v2v_aggregate_queries,
                    "aggregate_heads": self.scoring.v2v_aggregate_heads,
                    "layer": self.scoring.v2v_layer,
                },
            },
        }


def _opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _opt_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
@dataclass
class MemoryStats:
    """Method-agnostic sidecar (spec section P).

    HERMES-only knobs (``scope=hermes_managed``) are *not* forced here; the
    generic fields are produced by every method.
    """

    memory_method: str
    budget_total: Optional[int] = None
    budget_recent: Optional[int] = None
    budget_long_term: Optional[int] = None
    compression_trigger_count: int = 0
    effective_compaction_count: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    recent_size: int = 0
    long_term_size: int = 0
    evicted_count: int = 0
    retention_ratio: Optional[float] = None
    compression_wallclock: List[float] = field(default_factory=list)
    compression_interval_chunks: Optional[int] = None
    # Transient overshoot between two periodic compactions (read-only stats).
    peak_managed_tokens: int = 0
    peak_budget_overshoot_tokens: int = 0
    peak_budget_overshoot_ratio: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "memory_method": self.memory_method,
            "budget_total": self.budget_total,
            "budget_recent": self.budget_recent,
            "budget_long_term": self.budget_long_term,
            "compression_trigger_count": int(self.compression_trigger_count),
            "effective_compaction_count": int(self.effective_compaction_count),
            "tokens_before": int(self.tokens_before),
            "tokens_after": int(self.tokens_after),
            "recent_size": int(self.recent_size),
            "long_term_size": int(self.long_term_size),
            "evicted_count": int(self.evicted_count),
            "retention_ratio": _opt_float(self.retention_ratio),
            "compression_wallclock": [float(t) for t in self.compression_wallclock],
            "compression_interval_chunks": self.compression_interval_chunks,
            "peak_managed_tokens": int(self.peak_managed_tokens),
            "peak_budget_overshoot_tokens": int(self.peak_budget_overshoot_tokens),
            "peak_budget_overshoot_ratio": _opt_float(
                self.peak_budget_overshoot_ratio
            ),
        }
        data.update(self.extra)
        return data


def as_array(value: Any, *, dim: int = 1) -> np.ndarray:
    """Coerce ``value`` to a float32 numpy array (never imports torch)."""

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim < dim:
        arr = arr.reshape(-1) if dim == 1 else arr
    return arr
