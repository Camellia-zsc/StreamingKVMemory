"""Scoring signals for the InfoKV-inspired streaming video memory.

These are *not* the original text InfoKV predictive-entropy scores. Visual
tokens are continuous embeddings and have no natural next-token vocabulary
distribution, so no "visual entropy" is fabricated. Instead three
well-defined, independently testable signals are composed:

    S_i = w_N * N_i + w_D * D_i + w_A * A_i

* ``N`` Temporal Novelty          -- how different the current chunk is from
  recent history (chunk embedding mean pooling + cosine distance);
* ``D`` Layer-wise Representation Evolution -- how much one visual token's
  hidden state changed between a shallow and a deep layer;
* ``A`` Recent-context / attention importance -- an injected per-token score
  (HERMES-style importance, or any provider).

All raw signals are min-max normalized over the current candidate pool before
fusing, because the three quantities have different scales.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8


# --------------------------------------------------------------------------- #
# embeddings helpers
# --------------------------------------------------------------------------- #
def mean_pool(embeddings, *, eps: float = EPS) -> np.ndarray:
    """Mean-pool ``[n, dim]`` token embeddings into a single ``[dim]`` vector."""

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=0)


def _as_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def cosine_similarity(a, b, *, eps: float = EPS) -> float:
    va = _as_vector(a)
    vb = _as_vector(b)
    if va.shape != vb.shape:
        raise ValueError(
            f"cosine_similarity shape mismatch: {va.shape} vs {vb.shape}"
        )
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def cosine_distance(a, b, *, eps: float = EPS) -> float:
    """``1 - cos(a, b)`` clipped to ``[0, 1]`` (a raw novelty score)."""

    return float(min(1.0, max(0.0, 1.0 - cosine_similarity(a, b, eps=eps))))


# --------------------------------------------------------------------------- #
# N -- temporal novelty
# --------------------------------------------------------------------------- #
def reference_embedding(
    history: Sequence[np.ndarray],
    *,
    reference: str = "recent_mean",
    window: int = 4,
) -> Optional[np.ndarray]:
    """Reference embedding for temporal novelty.

    ``history`` is the list of *previous* chunk embeddings (oldest first).

    * ``previous``    -> the immediately previous chunk;
    * ``recent_mean`` -> mean of the last ``window`` chunks (default).
    """

    if not history:
        return None
    if reference == "previous":
        return _as_vector(history[-1])
    if reference == "recent_mean":
        k = max(1, int(window))
        recent = history[-k:]
        return np.stack([_as_vector(h) for h in recent], axis=0).mean(axis=0)
    raise ValueError(
        f"reference must be 'previous' or 'recent_mean', got {reference!r}"
    )


def temporal_novelty(
    chunk_embedding,
    reference,
    *,
    eps: float = EPS,
) -> float:
    """``N = 1 - cos(g_t, reference)`` in ``[0, 1]``.

    ``reference is None`` (e.g. the very first chunk) yields 0.0: with no
    history there is nothing to be novel *relative to*. A duplicate chunk gives
    a value close to 0; a clearly different chunk gives a clearly higher value.
    """

    if reference is None:
        return 0.0
    return cosine_distance(chunk_embedding, reference, eps=eps)


# --------------------------------------------------------------------------- #
# D -- layer-wise representation evolution
# --------------------------------------------------------------------------- #
def layer_evolution(
    shallow_hidden,
    deep_hidden,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """Per-token ``D_i = 1 - cos(h_i^(shallow), h_i^(deep))``.

    Token identity must stay aligned row-by-row: element ``i`` of the shallow
    array is compared with element ``i`` of the deep array. Identical
    representations give ``D ~ 0``; strongly changed ones give ``D -> 1``.
    """

    shallow = np.asarray(shallow_hidden, dtype=np.float32)
    deep = np.asarray(deep_hidden, dtype=np.float32)
    if shallow.ndim == 1:
        shallow = shallow.reshape(1, -1)
    if deep.ndim == 1:
        deep = deep.reshape(1, -1)
    if shallow.shape != deep.shape:
        raise ValueError(
            "layer_evolution expects aligned shallow/deep hidden states, got "
            f"{shallow.shape} vs {deep.shape}"
        )
    a_norm = np.linalg.norm(shallow, axis=1)
    b_norm = np.linalg.norm(deep, axis=1)
    denom = np.clip(a_norm * b_norm, eps, None)
    cos = (shallow * deep).sum(axis=1) / denom
    cos = np.clip(cos, -1.0, 1.0)
    return np.clip(1.0 - cos, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# normalization / fusion
# --------------------------------------------------------------------------- #
def min_max_normalize(values, *, eps: float = EPS) -> np.ndarray:
    """``(x - min) / (max - min + eps)`` over the current candidate pool.

    A degenerate (all-equal or empty) pool maps to all zeros.
    """

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    span = hi - lo
    if span < eps:
        return np.zeros_like(arr)
    return (arr - lo) / (span + eps)


def normalize_weights(
    weights: Dict[str, float],
    *,
    eps: float = 1e-9,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Return non-negative weights summing to 1 plus an optional warning.

    Negative weights are an error. A non-unit sum is renormalized and reported
    so callers can surface the adjustment (rather than silently biasing the
    score).
    """

    for name, value in weights.items():
        if value < 0:
            raise ValueError(f"weight {name!r} must be >= 0, got {value}")
    total = float(sum(weights.values()))
    if total < eps:
        raise ValueError("at least one scoring weight must be > 0")
    if abs(total - 1.0) > 1e-6:
        warning = (
            f"memory scoring weights sum to {total:.6f} != 1.0; "
            "renormalizing to 1.0"
        )
        return {name: value / total for name, value in weights.items()}, warning
    return dict(weights), None


def fuse_scores(
    novelty_norm,
    evolution_norm,
    v2v_norm,
    weights: Dict[str, float],
) -> np.ndarray:
    """Weighted sum of already-normalized signals."""

    n = np.asarray(novelty_norm, dtype=np.float32).reshape(-1)
    d = np.asarray(evolution_norm, dtype=np.float32).reshape(-1)
    a = np.asarray(v2v_norm, dtype=np.float32).reshape(-1)
    if not (n.shape == d.shape == a.shape):
        raise ValueError(
            f"fuse_scores needs aligned signals, got {n.shape}, {d.shape}, {a.shape}"
        )
    return (
        float(weights["temporal_novelty"]) * n
        + float(weights["layer_evolution"]) * d
        + float(weights["v2v"]) * a
    ).astype(np.float32)


def deterministic_topk(
    scores: Iterable[float],
    original_positions: Iterable[int],
    token_ids: Iterable[int],
    k: int,
) -> list:
    """Top-``k`` indices with a stable, documented tie-break.

    Ordering key is ``(-score, -original_position, token_id)``:

    1. higher fused score wins;
    2. on a tie, the *more recent* token (larger original position) wins;
    3. on a further tie, the smaller ``token_id`` wins.

    Never relies on Python's unstable sort: the key is total and explicit.
    """

    scores = [float(s) for s in scores]
    positions = [int(p) for p in original_positions]
    token_ids = [int(t) for t in token_ids]
    if not (len(scores) == len(positions) == len(token_ids)):
        raise ValueError("deterministic_topk arguments must have equal length")
    order = sorted(
        range(len(scores)),
        key=lambda i: (-scores[i], -positions[i], token_ids[i]),
    )
    if k <= 0:
        return []
    return order[: min(k, len(order))]
