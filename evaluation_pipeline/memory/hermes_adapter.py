"""HERMES adapter for the unified memory interface.

This is a *thin* adapter. It does **not** reimplement or alter HERMES: the real
attention-importance scoring, shallow/middle/deep layer budgets, recency
weighting, pruning and KV-cache surgery all stay exactly where they are
(``HERMES/inference/qwenvl_hermes*.py`` driven by
``OVO-Bench/models/MMDuet2_HERMES.py``).

What the adapter provides is the method-agnostic surface the pipeline needs:

* the HERMES trigger semantics (``managed_visual_tokens > budget``) so the
  generic pipeline can decide *when* to compact;
* generic accounting (trigger count, tokens before/after, retention);
* an optional :meth:`bind_backend` hook that, during GPU integration, can call
  the live ``predict_and_compress`` -- never here, and never on import.

Importing this module pulls in nothing heavy, so HERMES experiments are
unaffected and the adapter is fully unit-testable on CPU.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import CompactResult, KVMemoryMethod, ObserveResult
from .schemas import MemoryStats, Observation, TokenRecord


class HermesMemoryMethod(KVMemoryMethod):
    """Adapter exposing the existing HERMES compaction through the interface."""

    method_name = "hermes"

    def __init__(self, config, *_, **__):
        super().__init__(config)
        self.budget = int(
            config.hermes_budget
            if config.hermes_budget is not None
            else (config.total_visual_budget or 0)
        )
        self.compression_enabled = config.hermes_compression_enabled
        self._backend = None
        self._compression_callable = None
        self.reset()

    # -- lifecycle ----------------------------------------------------- #
    def reset(self) -> None:
        self.managed_visual_tokens = 0
        self.total_seen_tokens = 0
        self.chunk_counter = 0
        self.trigger_count = 0
        self.effective_compaction_count = 0
        self.last_tokens_before = 0
        self.last_tokens_after = 0
        self.last_retention_ratio: Optional[float] = None
        self.compression_wallclock: List[float] = []

    def observe(self, observation: Observation) -> ObserveResult:
        n = self._count(observation)
        self.managed_visual_tokens += n
        self.total_seen_tokens += n
        self.chunk_counter += 1
        return ObserveResult(
            chunk_id=(
                int(observation.chunk_id)
                if observation.chunk_id is not None
                else self.chunk_counter - 1
            ),
            n_tokens=n,
            recent_size=0,
            candidate_size=0,
        )

    def should_compact(self) -> bool:
        """HERMES trigger: fire when the managed visual region exceeds budget."""

        if self.compression_enabled is False:
            return False
        if self.budget <= 0:
            return False
        return self.managed_visual_tokens > self.budget

    def compact(self) -> CompactResult:
        import time

        t0 = time.perf_counter()
        self.trigger_count += 1
        before = self.managed_visual_tokens
        after = min(before, self.budget) if self.budget > 0 else before

        # When (and only when) a live backend is bound during GPU integration,
        # delegate to the untouched HERMES entry point. On CPU this is a no-op.
        if self._compression_callable is not None:
            self._compression_callable()
        elif self._backend is not None:
            qa_model = getattr(self._backend, "qa_model", None)
            if qa_model is not None and hasattr(qa_model, "predict_and_compress"):
                qa_model.predict_and_compress()

        self.managed_visual_tokens = after
        self.last_tokens_before = before
        self.last_tokens_after = after
        self.last_retention_ratio = after / before if before > 0 else None
        if after < before:
            self.effective_compaction_count += 1
        wallclock = time.perf_counter() - t0
        self.compression_wallclock.append(wallclock)
        return CompactResult(
            triggered=True,
            tokens_before=before,
            tokens_after=after,
            retention_ratio=self.last_retention_ratio,
            wallclock=wallclock,
            extra={"delegated": self._compression_callable is not None
                   or self._backend is not None},
        )

    # -- reporting ----------------------------------------------------- #
    def get_stats(self) -> MemoryStats:
        return MemoryStats(
            memory_method=self.method_name,
            budget_total=self.budget or None,
            budget_recent=None,
            budget_long_term=None,
            compression_trigger_count=self.trigger_count,
            effective_compaction_count=self.effective_compaction_count,
            tokens_before=self.last_tokens_before,
            tokens_after=self.last_tokens_after,
            recent_size=0,
            long_term_size=self.managed_visual_tokens,
            evicted_count=0,
            retention_ratio=self.last_retention_ratio,
            compression_wallclock=list(self.compression_wallclock),
            extra={
                "scope": "hermes_managed",
                "total_seen_tokens": int(self.total_seen_tokens),
                "chunks_observed": int(self.chunk_counter),
                "compression_enabled": self.compression_enabled,
            },
        )

    def managed_tokens(self) -> List[TokenRecord]:
        # HERMES keeps its own KV state inside the backend; the adapter does not
        # shadow it. Return an empty list so generic callers stay safe.
        return []

    # -- GPU integration hook ------------------------------------------ #
    def set_compression_callable(self, fn) -> None:
        """Point the adapter at a zero-arg ``predict_and_compress`` wrapper."""

        self._compression_callable = fn

    @staticmethod
    def _count(observation: Observation) -> int:
        if observation.token_ids is not None:
            return len(observation.token_ids)
        for value in (
            observation.token_embeddings,
            observation.shallow_hidden,
            observation.deep_hidden,
            observation.original_positions,
        ):
            if value is not None:
                try:
                    return int(len(value))
                except TypeError:
                    continue
        if observation.attention_importance is not None:
            try:
                return int(len(observation.attention_importance))
            except TypeError:
                return 0
        return 0
