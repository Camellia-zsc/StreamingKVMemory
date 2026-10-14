"""InfoKV-inspired streaming video memory (CPU first version).

.. warning::

    This is an **InfoKV-inspired streaming video memory**, NOT a verbatim
    reproduction of the paper's text InfoKV. The original InfoKV scores tokens
    by *predictive entropy* over a next-token vocabulary distribution; visual
    tokens are continuous embeddings and have no such distribution. This
    version therefore does **not** fabricate a "visual entropy". It composes
    three well-defined signals instead:

        S_i = w_N * N_i                  (Temporal Novelty,            chunk-level)
            + w_D * D_i                  (Layer Representation Evolution, token-level)
            + w_A * A_i^V2V              (Vision-to-Vision importance,  token-level)

Structure (spec sections E, J, K)::

    Recent Buffer  (high fidelity, most recent B_recent visual tokens)
          |  overflow ("retired")
          v
    Long-term candidates --+
                           |  every compression_interval_chunks
                           v
    score (normalize -> fuse) -> deterministic Top-K(B_long) -> evict the rest

The Recent Buffer and the Long-term Cache are a *temporal* two-level memory;
they are unrelated to HERMES's shallow/middle/deep Transformer-layer memory.

``A^V2V`` is computed by a :class:`~evaluation_pipeline.memory.v2v.ContextImportanceProvider`
(default: :class:`VisionToVisionAttentionProvider`, LiveVLM-inspired). When no
visual Q/K states are available the method falls back to an injected
``Observation.attention_importance`` (or zeros).

Eviction is a real eviction: no summary tokens are generated
(``summary_strategy: none``).

Selection semantics (v1): **global token identity**. ``N / D / A^V2V`` produce a
single set of retained visual token ids ``I_keep``; every Transformer layer
gathers the *same* visual positions. (HERMES, by contrast, keeps layer-wise
heterogeneous retained indices.)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import CompactResult, KVMemoryMethod, ObserveResult
from .schemas import MemoryStats, Observation, TokenRecord
from .signals import (
    deterministic_topk,
    fuse_scores,
    layer_evolution,
    mean_pool,
    min_max_normalize,
    normalize_weights,
    reference_embedding,
    temporal_novelty,
)
from .v2v import VisionToVisionAttentionProvider


class InfoKVStreamMemory(KVMemoryMethod):
    """Recent Buffer + information-aware Long-term Cache for streaming video."""

    method_name = "infokv_stream"

    def __init__(self, config, *, context_provider=None, eps: float = 1e-8):
        super().__init__(config)
        self.eps = float(eps)
        self.recent_budget = int(config.recent.budget)
        self.long_term_budget = int(config.long_term.budget)
        self.total_budget = int(config.total_visual_budget or 0)
        self.interval = int(config.compression.interval_chunks)
        self.observation_tokens = int(config.scoring.v2v_observation_tokens)
        self.layerwise_retention = bool(config.scoring.layerwise_retention_enabled)

        # Effective weights: disabled signals contribute nothing, then the
        # remaining weights are renormalized to 1.0 (with a warning if they did
        # not already sum to 1).
        raw_weights = {
            name: (weight if config.scoring.enabled()[name] else 0.0)
            for name, weight in config.scoring.weights().items()
        }
        self.weights, self.weight_warning = normalize_weights(raw_weights)

        if context_provider is None and config.scoring.v2v_enabled:
            context_provider = VisionToVisionAttentionProvider(
                aggregate_queries=config.scoring.v2v_aggregate_queries,
                aggregate_heads=config.scoring.v2v_aggregate_heads,
            )
        self.context_provider = context_provider

        self.reset()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.recent: List[TokenRecord] = []
        self.long_term: List[TokenRecord] = []
        self.candidates: List[TokenRecord] = []
        self._chunk_history: List[np.ndarray] = []
        self._next_token_id = 0
        self._next_chunk_id = 0
        self._chunks_since_compaction = 0
        self.chunk_counter = 0
        self.total_seen_tokens = 0
        self.retired_tokens = 0
        self.evicted_count = 0
        self.trigger_count = 0
        self.effective_compaction_count = 0
        self.compression_wallclock: List[float] = []
        self.last_compaction_tokens_before = 0
        self.last_compaction_tokens_after = 0
        self.last_retention_ratio: Optional[float] = None
        self.compaction_log: List[Dict[str, Any]] = []
        self._last_compaction_sidecar: List[Dict[str, Any]] = []
        # In layerwise mode each Transformer layer owns an independent
        # long-term retained set.  The Recent Buffer remains shared/protected.
        self.layer_ids: List[int] = []
        self.layer_retained_ids: Dict[int, set[int]] = {}
        self.last_compaction_tokens_before_by_layer: Dict[int, int] = {}
        self.last_compaction_tokens_after_by_layer: Dict[int, int] = {}
        # Transient-overshoot bookkeeping (read-only statistics).
        self.peak_managed_tokens = 0
        self.peak_budget_overshoot_tokens = 0
        self._update_peak()

    # ------------------------------------------------------------------ #
    # observe
    # ------------------------------------------------------------------ #
    def observe(self, observation: Observation) -> ObserveResult:
        chunk_id = (
            int(observation.chunk_id)
            if observation.chunk_id is not None
            else self._next_chunk_id
        )
        self._next_chunk_id = max(self._next_chunk_id, chunk_id + 1)

        token_ids, n = self._resolve_token_ids(observation)
        positions = self._resolve_positions(observation, token_ids)
        visual_mask = self._resolve_visual_mask(observation, n)
        query_states, key_states = self._resolve_kv_states(observation, n)

        # --- chunk embedding g_t (mean pooling fallback - spec F) -------- #
        token_embeddings = observation.token_embeddings
        if observation.chunk_embedding is not None:
            chunk_embedding = np.asarray(
                observation.chunk_embedding, dtype=np.float32
            ).reshape(-1)
        elif token_embeddings is not None and n > 0:
            chunk_embedding = mean_pool(token_embeddings, eps=self.eps)
        else:
            chunk_embedding = None

        # --- N: temporal novelty (shared by every token of the chunk) ---- #
        reference = None
        if self.config.scoring.temporal_novelty_enabled:
            reference = reference_embedding(
                self._chunk_history,
                reference=self.config.scoring.temporal_novelty_reference,
                window=self.config.scoring.temporal_novelty_reference_chunks,
            )
        novelty = temporal_novelty(chunk_embedding, reference, eps=self.eps)

        # --- D: layer-wise representation evolution (per token) ---------- #
        evolution_by_layer: Dict[int, np.ndarray] = {}
        if self.layerwise_retention and observation.layer_evolution_by_layer:
            evolution_by_layer = {
                int(layer): np.asarray(values, dtype=np.float32).reshape(-1)
                for layer, values in observation.layer_evolution_by_layer.items()
            }
            for layer, score in evolution_by_layer.items():
                if score.shape[0] != n:
                    raise ValueError(
                        f"layer_evolution produced {score.shape[0]} values for {n} tokens"
                    )
            self.layer_ids = sorted(evolution_by_layer)
            for layer in self.layer_ids:
                self.layer_retained_ids.setdefault(layer, set())
        elif self.layerwise_retention and observation.layer_hidden:
            hidden = {
                int(layer): np.asarray(value, dtype=np.float32)
                for layer, value in observation.layer_hidden.items()
            }
            layer_ids = sorted(hidden)
            for left, right in zip(layer_ids, layer_ids[1:]):
                score = layer_evolution(hidden[left], hidden[right], eps=self.eps)
                if score.shape[0] != n:
                    raise ValueError(
                        f"layer_evolution produced {score.shape[0]} values for {n} tokens"
                    )
                # Associate the representation change entering `right` with
                # the layer whose KV will be scored.  This gives every layer a
                # genuine per-token D signal while keeping N/A shared.
                evolution_by_layer[right] = score
            if layer_ids:
                self.layer_ids = layer_ids
                for layer in layer_ids:
                    self.layer_retained_ids.setdefault(layer, set())

        if (
            self.config.scoring.layer_evolution_enabled
            and observation.shallow_hidden is not None
            and observation.deep_hidden is not None
        ):
            evolution = layer_evolution(
                observation.shallow_hidden,
                observation.deep_hidden,
                eps=self.eps,
            )
            if n > 0 and evolution.shape[0] != n:
                raise ValueError(
                    "layer_evolution produced "
                    f"{evolution.shape[0]} values for {n} tokens"
                )
        else:
            evolution = np.zeros(n, dtype=np.float32)
        if self.layerwise_retention and evolution_by_layer:
            # Retain a useful method-agnostic sidecar value while selection
            # uses the true per-layer values below.
            evolution = np.mean(
                np.stack(list(evolution_by_layer.values()), axis=0), axis=0
            ).astype(np.float32)

        # --- A^V2V: last-window IMPORTANCE is computed at compaction time -- #
        # At observe time we only record any directly-injected score (HERMES-
        # style importance / synthetic). The true V2V score is derived from the
        # Recent Buffer queries + candidate keys inside ``compact``.
        injected = self._resolve_injected_importance(observation, n, token_ids)

        records: List[TokenRecord] = []
        for i in range(n):
            records.append(
                TokenRecord(
                    token_id=int(token_ids[i]),
                    chunk_id=int(chunk_id),
                    original_position=int(positions[i]),
                    token_index=int(i),
                    frame_time=float(observation.frame_time),
                    is_visual=bool(visual_mask[i]),
                    novelty_raw=float(novelty),
                    evolution_raw=float(evolution[i]),
                    evolution_by_layer={
                        int(layer): float(values[i])
                        for layer, values in evolution_by_layer.items()
                    },
                    v2v_raw=float(injected[i]),
                    key_state=None if key_states is None else key_states[i],
                    query_state=None if query_states is None else query_states[i],
                )
            )

        self.recent.extend(records)
        self.total_seen_tokens += n

        retired = 0
        while len(self.recent) > self.recent_budget:
            self.candidates.append(self.recent.pop(0))
            retired += 1
        self.retired_tokens += retired

        if chunk_embedding is not None:
            self._chunk_history.append(chunk_embedding)
            keep = max(
                1, int(self.config.scoring.temporal_novelty_reference_chunks)
            )
            if len(self._chunk_history) > keep:
                self._chunk_history = self._chunk_history[-keep:]

        self.chunk_counter += 1
        self._chunks_since_compaction += 1
        # The transient peak is reached BETWEEN compactions (periodic policy).
        self._update_peak()

        return ObserveResult(
            chunk_id=chunk_id,
            n_tokens=n,
            retired_tokens=retired,
            recent_size=len(self.recent),
            candidate_size=len(self.candidates),
            chunk_embedding=(
                None if chunk_embedding is None else chunk_embedding.tolist()
            ),
        )

    # ------------------------------------------------------------------ #
    # periodic compaction
    # ------------------------------------------------------------------ #
    def should_compact(self) -> bool:
        """True when ``compression.interval_chunks`` chunks have accumulated.

        No extra hard-overflow trigger: like HERMES, we accept a transient
        overshoot between two periodic compactions (tracked read-only).
        """

        return self._chunks_since_compaction >= self.interval

    def compact(self) -> CompactResult:
        t0 = time.perf_counter()
        self.trigger_count += 1

        pool = list(self.long_term) + list(self.candidates)
        pool_before = len(pool)
        tokens_before = len(self.recent) + len(pool)
        before_by_layer = {}
        if self.layerwise_retention and self.layer_ids:
            candidate_ids = {r.token_id for r in self.candidates}
            before_by_layer = {
                layer: len(self.recent) + len(self.layer_retained_ids[layer] | candidate_ids)
                for layer in self.layer_ids
            }
            tokens_before = max(before_by_layer.values())

        if pool:
            self._refresh_v2v(pool)
            if self.layerwise_retention and self.layer_ids:
                kept, evicted, sidecar = self._select_long_term_layerwise(pool)
            else:
                kept, evicted, sidecar = self._select_long_term(pool)
        else:
            kept, evicted, sidecar = [], [], []

        # Recent tokens are protected; annotate them for the debug sidecar.
        # These display-only scores share one candidate scale. Rebuilding
        # three arrays for every recent token makes a 6K cache quadratic.
        scales = {}
        if pool:
            for attr in ("novelty_raw", "evolution_raw", "v2v_raw"):
                raw = np.asarray([getattr(r, attr) for r in pool], dtype=np.float32)
                scales[attr] = (float(raw.min()), float(raw.max()))
        for record in self.recent:
            record.selected = True
            record.retained_order = None
            self._annotate_norms(record, pool, scales=scales)

        self.long_term = sorted(kept, key=lambda r: r.original_position)
        self.candidates = []
        self.evicted_count += len(evicted)
        pool_after = len(self.long_term)

        tokens_after = len(self.recent) + len(self.long_term)
        after_by_layer = {}
        if self.layerwise_retention and self.layer_ids:
            after_by_layer = {
                layer: len(self.recent) + len(self.layer_retained_ids[layer])
                for layer in self.layer_ids
            }
            tokens_after = max(after_by_layer.values())
        self.last_compaction_tokens_before = tokens_before
        self.last_compaction_tokens_after = tokens_after
        self.last_retention_ratio = (
            tokens_after / tokens_before if tokens_before > 0 else None
        )
        effective = (any(before_by_layer[l] > after_by_layer[l] for l in before_by_layer)
                     if before_by_layer else len(evicted) > 0)
        if effective:
            self.effective_compaction_count += 1

        wallclock = time.perf_counter() - t0
        self.compression_wallclock.append(wallclock)
        self._last_compaction_sidecar = sidecar
        if self.layerwise_retention and self.layer_ids:
            self.last_compaction_tokens_before_by_layer = before_by_layer
            self.last_compaction_tokens_after_by_layer = after_by_layer
        self.compaction_log.append(
            {
                "trigger_index": self.trigger_count,
                "effective_index": (
                    self.effective_compaction_count if effective else None
                ),
                "effective": bool(effective),
                # ``before`` / ``after`` describe the compacted long-term pool
                # (the thing that grows between compactions).
                "before": pool_before,
                "after": pool_after,
                "evicted": len(evicted),
                # Managed totals (recent + long-term) for accounting parity with
                # the generic MemoryStats fields.
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "recent": len(self.recent),
                "long_term": len(self.long_term),
                "retention_ratio": self.last_retention_ratio,
                "wallclock": wallclock,
            }
        )

        self._chunks_since_compaction = 0
        self._update_peak()
        return CompactResult(
            triggered=True,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            kept_token_ids=[int(r.token_id) for r in self.long_term],
            evicted_token_ids=[int(r.token_id) for r in evicted],
            retention_ratio=self.last_retention_ratio,
            wallclock=wallclock,
            extra={
                "recent_size": len(self.recent),
                "long_term_size": len(self.long_term),
                "before": pool_before,
                "after": pool_after,
                "effective": bool(effective),
                "evicted": len(evicted),
            },
        )

    # ------------------------------------------------------------------ #
    # V2V computation
    # ------------------------------------------------------------------ #
    def _refresh_v2v(self, pool: Sequence[TokenRecord]) -> None:
        """Compute ``A^V2V`` for every long-term candidate from real Q/K if
        available, else keep the injected per-token score."""

        if not self.config.scoring.v2v_enabled or self.context_provider is None:
            return

        # Recent Buffer tail = the observation window (visual only).
        recent_visual = [
            r
            for r in self.recent
            if r.is_visual and r.query_state is not None
        ]
        queries = [r.query_state for r in recent_visual[-self.observation_tokens :]]

        # Historical keys = the managed visual candidate pool, in pool order.
        key_idx = [
            i
            for i, r in enumerate(pool)
            if r.is_visual and r.key_state is not None
        ]
        keys = [pool[i].key_state for i in key_idx]

        if not queries or not keys:
            return  # fall back to injected / zero scores

        q_arr = self._stack_states(queries)  # [Hq, R, D]
        k_arr = self._stack_states(keys)  # [Hkv, T, D]
        scores = np.asarray(
            self.context_provider.score(q_arr[None], k_arr[None]),
            dtype=np.float32,
        ).reshape(-1)
        if scores.shape[0] != len(key_idx):
            raise ValueError(
                f"V2V provider returned {scores.shape[0]} scores for "
                f"{len(key_idx)} historical keys"
            )
        for local_i, pool_i in enumerate(key_idx):
            pool[pool_i].v2v_raw = float(scores[local_i])

    @staticmethod
    def _stack_states(states: Sequence[Any]) -> np.ndarray:
        """Stack per-token Q/K states to ``[H, N, D]`` for provider input.

        Each per-token state is ``[D]`` (bare) or ``[H, D]``. The provider
        expects ``[B, H, N, D]``, so we return ``[H, N, D]`` here for the caller
        to add the batch dimension.
        """

        arrays = [np.asarray(s, dtype=np.float32) for s in states]
        if arrays[0].ndim == 1:
            arrays = [a[None, :] for a in arrays]  # [D] -> [1, D]
        if any(a.ndim != 2 for a in arrays):
            raise ValueError(
                "per-token Q/K states must be [D] or [H, D]; got "
                f"{[a.shape for a in arrays]}"
            )
        heads = {a.shape[0] for a in arrays}
        if len(heads) != 1:
            raise ValueError(
                f"inconsistent head counts across tokens: {sorted(heads)}"
            )
        return np.stack(arrays, axis=1)  # [H, N, D]

    # ------------------------------------------------------------------ #
    # selection internals
    # ------------------------------------------------------------------ #
    def _select_long_term(self, pool: Sequence[TokenRecord]):
        novelty = [r.novelty_raw for r in pool]
        evolution = [r.evolution_raw for r in pool]
        v2v = [r.v2v_raw for r in pool]

        novelty_norm = min_max_normalize(novelty, eps=self.eps)
        evolution_norm = min_max_normalize(evolution, eps=self.eps)
        v2v_norm = min_max_normalize(v2v, eps=self.eps)

        # Disabled signals contribute a flat zero (their weight is 0 anyway).
        if not self.config.scoring.temporal_novelty_enabled:
            novelty_norm = np.zeros_like(novelty_norm)
        if not self.config.scoring.layer_evolution_enabled:
            evolution_norm = np.zeros_like(evolution_norm)
        if not self.config.scoring.v2v_enabled:
            v2v_norm = np.zeros_like(v2v_norm)

        scores = fuse_scores(novelty_norm, evolution_norm, v2v_norm, self.weights)

        keep_idx = deterministic_topk(
            scores,
            [r.original_position for r in pool],
            [r.token_id for r in pool],
            self.long_term_budget,
        )
        keep_set = set(keep_idx)

        kept: List[TokenRecord] = []
        evicted: List[TokenRecord] = []
        for i, record in enumerate(pool):
            record.novelty_norm = float(novelty_norm[i])
            record.evolution_norm = float(evolution_norm[i])
            record.v2v_norm = float(v2v_norm[i])
            record.final_score = float(scores[i])
            record.selected = i in keep_set
            if record.selected:
                kept.append(record)
            else:
                evicted.append(record)

        for order, record in enumerate(
            sorted(kept, key=lambda r: r.original_position)
        ):
            record.retained_order = order
        for record in evicted:
            record.retained_order = None

        sidecar = [r.as_sidecar() for r in kept]
        return kept, evicted, sidecar

    def _select_long_term_layerwise(self, pool: Sequence[TokenRecord]):
        """Select an independent long-term Top-K set for every Transformer layer.

        N and A are shared signals (one normalization over the common pool),
        while D is normalized and ranked separately for each layer.  The
        managed CPU ledger stores the union so a token remains available while
        any layer still retains it; the GPU ledger receives each set directly.
        """
        novelty = min_max_normalize([r.novelty_raw for r in pool], eps=self.eps)
        v2v = min_max_normalize([r.v2v_raw for r in pool], eps=self.eps)
        if not self.config.scoring.temporal_novelty_enabled:
            novelty = np.zeros(len(pool), dtype=np.float32)
        if not self.config.scoring.v2v_enabled:
            v2v = np.zeros(len(pool), dtype=np.float32)

        union: set[int] = set()
        candidates = {int(r.token_id) for r in self.candidates}
        layer_sidecars: Dict[int, set[int]] = {}
        for layer in self.layer_ids:
            eligible = self.layer_retained_ids[layer] | candidates
            local_pool = [r for r in pool if r.token_id in eligible]
            local_indices = [i for i, r in enumerate(pool) if r.token_id in eligible]
            d = np.asarray(
                [r.evolution_by_layer.get(int(layer), r.evolution_raw) for r in local_pool],
                dtype=np.float32,
            )
            d_norm = min_max_normalize(d, eps=self.eps)
            if not self.config.scoring.layer_evolution_enabled:
                d_norm = np.zeros_like(d_norm)
            scores = fuse_scores(novelty[local_indices], d_norm, v2v[local_indices], self.weights)
            keep_idx = deterministic_topk(
                scores,
                [r.original_position for r in local_pool],
                [r.token_id for r in local_pool],
                self.long_term_budget,
            )
            ids = {int(local_pool[i].token_id) for i in keep_idx}
            layer_sidecars[int(layer)] = ids
            self.layer_retained_ids[int(layer)] = ids
            union.update(ids)
            for i, record in enumerate(local_pool):
                # Sidecar fields describe the union score; retain the D value
                # for the layer-specific audit in ``selected_by_layer`` below.
                if int(record.token_id) in ids:
                    record.selected = True
                record.evolution_norm = float(d_norm[i])
                record.final_score = float(scores[i])

        kept = [r for r in pool if int(r.token_id) in union]
        evicted = [r for r in pool if int(r.token_id) not in union]
        for r in kept:
            r.selected = True
        for r in evicted:
            r.selected = False
            r.retained_order = None
        for order, record in enumerate(sorted(kept, key=lambda r: r.original_position)):
            record.retained_order = order
        # Shared N/A values are useful for plotting, while D/final_score above
        # follows the last layer.  Consumers needing exact layer membership use
        # ``selected_by_layer``.
        sidecar = []
        for r in kept:
            row = r.as_sidecar()
            row["selected_by_layer"] = {
                str(layer): int(r.token_id in ids)
                for layer, ids in layer_sidecars.items()
            }
            sidecar.append(row)
        return kept, evicted, sidecar

    def _annotate_norms(
        self, record: TokenRecord, pool: Sequence[TokenRecord], *, scales=None
    ) -> None:
        """Give a protected recent token a display-only normalized score.

        Recent tokens never participate in the long-term Top-K, so their norm is
        computed against the candidate pool's scale purely for the sidecar.
        """

        if not pool:
            record.novelty_norm = 0.0
            record.evolution_norm = 0.0
            record.v2v_norm = 0.0
            record.final_score = 0.0
            return
        record.novelty_norm = self._apply(record.novelty_raw, pool, "novelty_raw", scales)
        record.evolution_norm = self._apply(
            record.evolution_raw, pool, "evolution_raw", scales
        )
        record.v2v_norm = self._apply(record.v2v_raw, pool, "v2v_raw", scales)
        record.final_score = float(
            self.weights["temporal_novelty"] * record.novelty_norm
            + self.weights["layer_evolution"] * record.evolution_norm
            + self.weights["v2v"] * record.v2v_norm
        )

    @staticmethod
    def _apply(value: float, pool: Sequence[TokenRecord], attr: str, scales=None) -> float:
        if scales is None:
            raw = np.asarray([getattr(r, attr) for r in pool], dtype=np.float32)
            lo, hi = float(raw.min()), float(raw.max())
        else:
            lo, hi = scales[attr]
        span = hi - lo
        if span < 1e-8:
            return 0.0
        return float(min(1.0, max(0.0, (float(value) - lo) / (span + 1e-8))))

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def get_stats(self) -> MemoryStats:
        return MemoryStats(
            memory_method=self.method_name,
            budget_total=self.total_budget,
            budget_recent=self.recent_budget,
            budget_long_term=self.long_term_budget,
            compression_trigger_count=self.trigger_count,
            effective_compaction_count=self.effective_compaction_count,
            tokens_before=self.last_compaction_tokens_before,
            tokens_after=self.last_compaction_tokens_after,
            recent_size=len(self.recent),
            long_term_size=len(self.long_term),
            evicted_count=self.evicted_count,
            retention_ratio=self.last_retention_ratio,
            compression_wallclock=list(self.compression_wallclock),
            compression_interval_chunks=self.interval,
            peak_managed_tokens=self.peak_managed_tokens,
            peak_budget_overshoot_tokens=self.peak_budget_overshoot_tokens,
            peak_budget_overshoot_ratio=(
                self.peak_budget_overshoot_tokens / self.total_budget
                if self.total_budget > 0
                else None
            ),
            extra={
                "pending_candidates": len(self.candidates),
                "total_seen_tokens": int(self.total_seen_tokens),
                "retired_tokens": int(self.retired_tokens),
                "chunks_observed": int(self.chunk_counter),
                "scoring_weights": dict(self.weights),
                "weight_warning": self.weight_warning,
                "num_compaction_events": len(self.compaction_log),
                "v2v_provider": getattr(self.context_provider, "name", None),
            },
        )

    def managed_tokens(self) -> List[TokenRecord]:
        return list(self.recent) + list(self.long_term) + list(self.candidates)

    def compaction_events(self) -> List[Dict[str, Any]]:
        return list(self.compaction_log)

    def last_compaction_sidecar(self) -> List[Dict[str, Any]]:
        return list(self._last_compaction_sidecar)

    def retained_visual_indices(self) -> List[int]:
        """v1 selection semantics: one global set of retained visual token ids.

        Returns the retained long-term visual token ids in time order. Every
        Transformer layer gathers exactly these positions (unlike HERMES's
        per-layer indices).
        """

        return [
            int(r.token_id)
            for r in sorted(self.long_term, key=lambda r: r.original_position)
            if r.is_visual
        ]

    def retained_visual_token_ids(self) -> List[int]:
        """All visual token ids currently kept, in temporal order.

        This is the Recent Buffer tail plus the long-term selection -- i.e. the
        exact set the KV cache must retain after a compaction (prefix excluded).
        """

        return [
            int(r.token_id)
            for r in sorted(self.managed_tokens(), key=lambda r: r.original_position)
            if r.is_visual
        ]

    def retained_visual_token_ids_by_layer(self) -> Dict[int, List[int]]:
        """Return recent plus layer-specific long-term visual IDs."""
        if not (self.layerwise_retention and self.layer_ids):
            ids = self.retained_visual_token_ids()
            return {0: ids}
        recent = {int(r.token_id) for r in self.recent if r.is_visual}
        positions = {r.token_id: r.original_position for r in self.managed_tokens()}
        return {
            int(layer): sorted(
                recent | set(self.layer_retained_ids.get(layer, set())),
                key=lambda token_id: positions[token_id],
            )
            for layer in self.layer_ids
        }

    def refresh_context_states(self, provider) -> None:
        """Refresh stored per-token Q/K states from a live source.

        Used by the GPU integration to re-read the *current* KV keys after a
        compaction re-rotated them, so the next V2V pass uses up-to-date K.
        ``provider(record)`` returns ``(key_state, query_state)``; either may be
        ``None`` to leave the stored value unchanged.
        """

        for record in self.managed_tokens():
            key_state, query_state = provider(record)
            if key_state is not None:
                record.key_state = key_state
            if query_state is not None:
                record.query_state = query_state

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    def _update_peak(self) -> None:
        managed = len(self.recent) + len(self.long_term) + len(self.candidates)
        if self.layerwise_retention and self.layer_ids:
            managed = len(self.recent) + len(self.candidates) + max(
                (len(ids) for ids in self.layer_retained_ids.values()), default=0
            )
        self.peak_managed_tokens = max(self.peak_managed_tokens, managed)
        overshoot = max(0, managed - self.total_budget)
        self.peak_budget_overshoot_tokens = max(
            self.peak_budget_overshoot_tokens, overshoot
        )

    def _resolve_token_ids(self, observation: Observation):
        if observation.token_ids is not None:
            token_ids = [int(t) for t in observation.token_ids]
            return token_ids, len(token_ids)
        n = self._infer_n_tokens(observation)
        token_ids = list(range(self._next_token_id, self._next_token_id + n))
        self._next_token_id += n
        return token_ids, n

    @staticmethod
    def _infer_n_tokens(observation: Observation) -> int:
        for value in (
            observation.token_embeddings,
            observation.shallow_hidden,
            observation.deep_hidden,
            observation.original_positions,
            observation.query_states,
            observation.key_states,
        ):
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] > 0:
                return int(arr.shape[0])
        attention = observation.attention_importance
        if isinstance(attention, dict):
            return len(attention)
        if attention is not None:
            arr = np.asarray(attention)
            if arr.ndim >= 1:
                return int(arr.shape[0])
        return 0

    def _resolve_positions(self, observation: Observation, token_ids):
        if observation.original_positions is not None:
            return [int(p) for p in observation.original_positions]
        return [int(t) for t in token_ids]

    @staticmethod
    def _resolve_visual_mask(observation: Observation, n: int):
        if observation.is_visual is None:
            return np.ones(n, dtype=bool)
        mask = np.asarray(observation.is_visual, dtype=bool).reshape(-1)
        if mask.shape[0] != n:
            raise ValueError(
                f"is_visual has {mask.shape[0]} entries for {n} tokens"
            )
        return mask

    def _resolve_kv_states(self, observation: Observation, n: int):
        query_states = observation.query_states
        key_states = observation.key_states
        if query_states is not None:
            query_states = list(np.asarray(query_states, dtype=np.float32))
            if len(query_states) != n:
                raise ValueError(
                    f"query_states has {len(query_states)} entries for {n} tokens"
                )
        if key_states is not None:
            key_states = list(np.asarray(key_states, dtype=np.float32))
            if len(key_states) != n:
                raise ValueError(
                    f"key_states has {len(key_states)} entries for {n} tokens"
                )
        return query_states, key_states

    def _resolve_injected_importance(self, observation: Observation, n: int, token_ids):
        attention = observation.attention_importance
        if attention is None:
            return np.zeros(n, dtype=np.float32)
        if isinstance(attention, dict):
            return np.asarray(
                [float(attention.get(int(t), 0.0)) for t in token_ids],
                dtype=np.float32,
            )
        arr = np.asarray(attention, dtype=np.float32).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"attention_importance has {arr.shape[0]} values for {n} tokens"
            )
        return arr
