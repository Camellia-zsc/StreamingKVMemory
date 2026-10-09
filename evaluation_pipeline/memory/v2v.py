"""Vision-to-Vision (V2V) recent-context importance.

Provenance
----------
Vision-to-Vision context importance is **inspired by LiveVLM v2**
(Vision Sink Bucketing, VSB):

    recent visual observation window  (queries)
        -> historical visual KV        (keys)
        -> attention-based per-position importance

Reference (v2 implementation):

* ``modules/cache_utils.py::StreamCache.cache_compression_diversity`` in
  https://github.com/sjtu-zhao-lab/LiveVLM
* Paper: *LiveVLM: Efficient Online Video Understanding via Streaming-Oriented
  KV Cache and Retrieval*, arXiv:2505.15269

What is reused here is **only the V2V importance principle**: recent visual
queries attend to historical visual keys; the query-averaged, head-averaged
softmax attention over history is the importance of each historical visual
position. We explicitly do **not** reproduce:

* LiveVLM VSB *scatter* selection (uniform scattered retention);
* Position-agnostic Retrieval (PaR);
* page retrieval / query-time retrieval;
* LiveVLM's full cache architecture.

Selection in this project is always the fused ``N + D + A^V2V`` Top-K; the V2V
provider only *scores* historical visual positions.

The math mirrors LiveVLM's ``cache_compression_diversity``::

    Q_recent = Q[:, :, -window_len:, :]                 # last window_len query tokens
    K = repeat_kv(K_history, Hq // Hkv)                 # GQA/MQA compatibility
    logits = Q_recent @ K^T / sqrt(D)                   # [B, Hq, R, T]
    attn = softmax(logits, dim=-1)                      # over history T
    A = attn.mean(query_dim=-2).mean(head_dim=1)        # [B, T]

Strictly vision-to-vision: callers must pass only visual tokens (see
``Observation.is_visual``). Never system / question / answer / text tokens.

This module is numpy-only so it stays importable and unit-testable without a
GPU. A GPU integration supplies a torch implementation of the same
``ContextImportanceProvider`` protocol (or pre-computed per-token scores).
"""

from __future__ import annotations

import math
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

EPS = 1e-8


def repeat_kv(hidden_states, n_rep: int):
    """NumPy port of ``transformers...repeat_kv`` (GQA/MQA head expansion).

    ``[B, Hkv, T, D]`` -> ``[B, Hkv * n_rep, T, D]`` equivalent to
    ``repeat_interleave`` along the head axis.
    """

    arr = np.asarray(hidden_states, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(
            f"repeat_kv expects [B, Hkv, T, D], got shape {arr.shape}"
        )
    if n_rep == 1:
        return arr
    if n_rep < 1:
        raise ValueError(f"n_rep must be >= 1, got {n_rep}")
    batch, num_kv_heads, slen, head_dim = arr.shape
    arr = np.repeat(arr, n_rep, axis=1)
    return arr.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / np.clip(denom, EPS, None)


def _as_queries(recent_queries) -> np.ndarray:
    arr = np.asarray(recent_queries, dtype=np.float32)
    if arr.ndim == 2:  # [R, D] -> [1, 1, R, D]
        arr = arr[None, None]
    elif arr.ndim == 3:  # [B, R, D] -> [B, 1, R, D]
        arr = arr[:, None]
    if arr.ndim != 4:
        raise ValueError(
            "recent_queries must be [B, Hq, R, D] (or [B, R, D] / [R, D]), "
            f"got shape {arr.shape}"
        )
    return arr


def _as_keys(historical_keys) -> np.ndarray:
    arr = np.asarray(historical_keys, dtype=np.float32)
    if arr.ndim == 2:  # [T, D] -> [1, 1, T, D]
        arr = arr[None, None]
    elif arr.ndim == 3:  # [B, T, D] -> [B, 1, T, D]
        arr = arr[:, None]
    if arr.ndim != 4:
        raise ValueError(
            "historical_keys must be [B, Hkv, T, D] (or [B, T, D] / [T, D]), "
            f"got shape {arr.shape}"
        )
    return arr


def vision_to_vision_attention(
    recent_queries,
    historical_keys,
    *,
    aggregate_queries: str = "mean",
    aggregate_heads: str = "mean",
    return_per_head: bool = False,
    eps: float = EPS,
):
    """Per-historical-token V2V importance ``A^V2V``.

    Parameters
    ----------
    recent_queries : array-like
        ``[B, Hq, R, D]`` (or ``[B, R, D]`` / ``[R, D]``) recent *visual*
        query states.
    historical_keys : array-like
        ``[B, Hkv, T, D]`` (or ``[B, T, D]`` / ``[T, D]``) historical *visual*
        key states. ``T`` is the long-term candidate/history size.

    Returns
    -------
    importance : ``[B, T]``
        Query-averaged, head-averaged softmax attention over history.
    per_head : ``[B, Hkv, T]`` (only when ``return_per_head=True``)
    """

    queries = _as_queries(recent_queries)
    keys = _as_keys(historical_keys)

    batch_q, num_heads, r_len, head_dim = queries.shape
    batch_k, num_kv_heads, t_len, key_dim = keys.shape
    if batch_q != batch_k:
        raise ValueError(
            f"batch mismatch: queries {batch_q} vs keys {batch_k}"
        )
    if head_dim != key_dim:
        raise ValueError(
            f"head_dim mismatch: queries {head_dim} vs keys {key_dim}"
        )
    if t_len == 0:
        raise ValueError("historical_keys is empty (T == 0)")
    if r_len == 0:
        raise ValueError("recent_queries is empty (R == 0)")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num query heads {num_heads} is not a multiple of num KV heads "
            f"{num_kv_heads} (GQA/MQA requires Hq % Hkv == 0)"
        )

    keys_repeated = repeat_kv(keys, num_heads // num_kv_heads)

    logits = np.matmul(queries, keys_repeated.transpose(0, 1, 3, 2)) / math.sqrt(
        head_dim
    )
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    attn = _softmax(logits, axis=-1)  # [B, Hq, R, T]

    # Aggregate over the recent observation window (R).
    if aggregate_queries != "mean":
        raise ValueError(
            f"aggregate_queries={aggregate_queries!r} is not supported "
            "(first version: 'mean')"
        )
    head_scores = attn.mean(axis=-2)  # [B, Hq, T]

    # Aggregate over attention heads -> [B, T]. If the caller disabled head
    # averaging we still report the head-mean as the fusion score, but expose
    # the per-KV-head tensor for debugging.
    per_head = head_scores.reshape(
        batch_q, num_kv_heads, num_heads // num_kv_heads, t_len
    ).mean(axis=2)  # [B, Hkv, T]
    if aggregate_heads != "mean":
        raise ValueError(
            f"aggregate_heads={aggregate_heads!r} is not supported "
            "(first version: 'mean')"
        )
    importance = per_head.mean(axis=1)  # [B, T]

    if return_per_head:
        return importance, per_head
    return importance


@runtime_checkable
class ContextImportanceProvider(Protocol):
    """Unified interface for per-token context-importance sources.

    A provider maps recent visual queries + historical visual keys to a raw
    importance score per historical token. Implementing this protocol is what
    lets a future GPU path plug real KV-cache slices into InfoKVStream without
    changing the fusion code.
    """

    def score(self, recent_queries, historical_keys, **kwargs) -> np.ndarray:
        ...  # pragma: no cover - protocol


class VisionToVisionAttentionProvider:
    """``ContextImportanceProvider`` implementing the LiveVLM-inspired V2V score."""

    name = "vision_to_vision_attention"

    def __init__(
        self,
        *,
        aggregate_queries: str = "mean",
        aggregate_heads: str = "mean",
        eps: float = EPS,
    ):
        self.aggregate_queries = aggregate_queries
        self.aggregate_heads = aggregate_heads
        self.eps = eps
        self.last_per_head: Optional[np.ndarray] = None

    def score(
        self,
        recent_queries,
        historical_keys,
        *,
        return_per_head: bool = False,
    ):
        importance, per_head = vision_to_vision_attention(
            recent_queries,
            historical_keys,
            aggregate_queries=self.aggregate_queries,
            aggregate_heads=self.aggregate_heads,
            return_per_head=True,
            eps=self.eps,
        )
        self.last_per_head = per_head
        if return_per_head:
            return importance, per_head
        return importance
