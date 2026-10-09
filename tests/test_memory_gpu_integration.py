"""CPU-only tests for the InfoKV GPU-integration *plumbing*.

No GPU, no model: these exercise the pure-Python pieces (ledger, keep-index
construction, compactor broadcast, layer resolution) and the torch hook
mechanism with a tiny stub module.
"""

from __future__ import annotations

import numpy as np
import pytest

import torch
import torch.nn as nn

from evaluation_pipeline.memory.qwen25vl_gpu import (
    InfoKVCapture,
    KVCacheCompactor,
    VisualTokenLedger,
    build_keep_indices,
    default_layer_choices,
)


# --------------------------------------------------------------------------- #
# ledger: token_id -> cache index mapping
# --------------------------------------------------------------------------- #
def test_visual_token_ledger_mapping_and_rebuild():
    ledger = VisualTokenLedger(prefix_len=4)
    ledger.register_chunk(
        token_ids=[10, 11, 12],
        original_stream_positions=[0, 1, 2],
        cache_indices=[4, 5, 6],
        mrope_positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]],
    )
    assert ledger.cache_index(11) == 5
    assert ledger.get(12).mrope_position == [0, 0, 2]
    assert ledger.get(10).original_stream_position == 0
    assert len(ledger) == 3

    # Keep 10 and 12 (drop 11): survivors re-index to prefix + rank.
    ledger.rebuild_after_compaction([10, 12])
    assert ledger.cache_index(10) == 4
    assert ledger.cache_index(12) == 5
    assert 11 not in ledger
    assert len(ledger) == 2


def test_token_id_and_cache_index_are_distinct():
    ledger = VisualTokenLedger(prefix_len=14)
    ledger.register_chunk([100, 101, 102], [0, 1, 2], [14, 15, 16])
    # internal identity != physical index != stream position
    assert ledger.cache_index(100) == 14
    assert ledger.get(100).original_stream_position == 0
    assert ledger.cache_index(100) != ledger.get(100).original_stream_position


# --------------------------------------------------------------------------- #
# retained-order invariant: Top-K decides *who*, cache order decides *where*
# --------------------------------------------------------------------------- #
def test_build_keep_indices_is_ascending_prefix_preserving():
    ledger = VisualTokenLedger(prefix_len=2)
    ledger.register_chunk(
        token_ids=[0, 1, 2, 3, 4],
        original_stream_positions=[0, 1, 2, 3, 4],
        cache_indices=[2, 3, 4, 5, 6],
    )
    # A score-ranked Top-K {4, 0, 2} must be written in cache order {0, 2, 4}.
    keep = build_keep_indices(2, [4, 0, 2], ledger)
    assert keep == [0, 1, 2, 4, 6]
    assert keep == sorted(keep)  # never score-ordered


def test_build_keep_indices_rejects_out_of_range_and_noop():
    ledger = VisualTokenLedger(prefix_len=1)
    ledger.register_chunk([0, 1], [0, 1], [1, 2])
    with pytest.raises(ValueError):
        build_keep_indices(1, [0, 1], ledger, cache_len=3)  # keeps everything


def test_build_keep_indices_rejects_duplicate_physical():
    ledger = VisualTokenLedger(prefix_len=0)
    ledger.register_chunk([0, 1], [0, 1], [5, 5])  # impossible duplicate
    with pytest.raises(ValueError):
        build_keep_indices(0, [0, 1], ledger)


# --------------------------------------------------------------------------- #
# global broadcast + summary-free threshold
# --------------------------------------------------------------------------- #
class _FakeBackend:
    def __init__(self, num_layers=4, threshold=3, cache_len=100):
        self.num_layers = num_layers
        self.long_term_threshold = threshold
        self.cache_len = cache_len
        self.calls = []
        self.threshold_at_call = []

    def apply_kv_cache_pruning_strict(self, keep_per_layer):
        self.threshold_at_call.append(self.long_term_threshold)
        self.calls.append([list(x) for x in keep_per_layer])
        self.cache_len = len(keep_per_layer[0])

    def _get_cache_seq_len_per_layer(self):
        return [self.cache_len] * self.num_layers


def test_compactor_broadcasts_one_global_index_set():
    backend = _FakeBackend(num_layers=4, threshold=3)
    compactor = KVCacheCompactor(backend)
    keep = [0, 1, 5, 9, 12]
    lens = compactor.apply(keep)
    assert lens == [len(keep)] * 4
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 4
    for layer_keep in backend.calls[0]:
        assert layer_keep == keep  # same indices everywhere
    # summary branch suppressed during the call, threshold restored after
    assert backend.threshold_at_call[0] == 4
    assert backend.long_term_threshold == 3


# --------------------------------------------------------------------------- #
# layer resolution / defaults
# --------------------------------------------------------------------------- #
def test_default_layer_choices_are_in_range():
    for n in (28, 12, 4):
        d = default_layer_choices(n)
        for value in d.values():
            assert 0 <= value < n


def test_resolve_layers_validates_range():
    from evaluation_pipeline.memory.qwen25vl_backend import _resolve_layers
    from evaluation_pipeline.memory.schemas import MemoryMethodConfig

    cfg = MemoryMethodConfig(method="infokv_stream", total_visual_budget=10)
    a, b, v = _resolve_layers(cfg, 28)
    assert (a, b, v) == (2, 24, 14)

    from evaluation_pipeline.memory.schemas import ScoringConfig

    cfg2 = MemoryMethodConfig(
        method="infokv_stream", total_visual_budget=10, scoring=ScoringConfig(v2v_layer=99)
    )
    with pytest.raises(ValueError):
        _resolve_layers(cfg2, 28)


# --------------------------------------------------------------------------- #
# capture transparency (torch hook mechanism, stub model)
# --------------------------------------------------------------------------- #
class _StubAttn(nn.Module):
    def __init__(self, hidden=16, heads=4, kv_heads=2, head_dim=4):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)


class _StubLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _StubAttn()
        self.lin = nn.Linear(16, 16)

    def forward(self, hidden):
        q = self.self_attn.q_proj(hidden)
        # Real attention also computes K; call it so the k_proj hook fires.
        self.self_attn.k_proj(hidden)
        return (self.lin(hidden) + q.mean(-1, keepdim=True),)


class _StubLM(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([_StubLayer() for _ in range(num_layers)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class _StubModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.language_model = _StubLM(num_layers)


def test_capture_does_not_change_model_output():
    torch.manual_seed(0)
    model = _StubModel(num_layers=5)
    capture = InfoKVCapture(model, layer_a=1, layer_b=3, v2v_layer=2)
    x = torch.randn(1, 7, 16)

    with torch.no_grad():
        out_disabled = model.language_model(x).clone()

        capture.begin_chunk()
        out_enabled = model.language_model(x).clone()
        captured = capture.end_chunk()

    assert torch.allclose(out_disabled, out_enabled)  # hooks are read-only
    assert captured["hidden_a"].shape == (7, 16)
    assert captured["hidden_b"].shape == (7, 16)
    assert captured["q"].shape == (1, 4, 7, 4)
    assert captured["k"].shape == (1, 2, 7, 4)
    capture.close()


def test_capture_is_inert_when_not_streaming():
    model = _StubModel(num_layers=5)
    capture = InfoKVCapture(model, layer_a=1, layer_b=3, v2v_layer=2)
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        model.language_model(x)  # no begin_chunk -> buffers stay empty
        captured = capture.end_chunk()
    assert all(v is None for v in captured.values())
    capture.close()


# --------------------------------------------------------------------------- #
# no-op compaction must not count as effective
# --------------------------------------------------------------------------- #
def test_noop_compaction_is_not_effective():
    from evaluation_pipeline.memory import (
        CompressionConfig,
        InfoKVStreamMemory,
        LongTermConfig,
        MemoryMethodConfig,
        Observation,
        RecentConfig,
    )

    cfg = MemoryMethodConfig(
        method="infokv_stream",
        total_visual_budget=10,
        recent=RecentConfig(enabled=True, budget=2),
        long_term=LongTermConfig(budget=8),
        compression=CompressionConfig(interval_chunks=1),
    )
    cfg.validate()
    memory = InfoKVStreamMemory(cfg)
    # Only two tokens ever seen -> long-term holds both, nothing is evicted.
    memory.observe(
        Observation(
            chunk_id=0,
            chunk_embedding=np.ones(4, dtype=np.float32),
            token_ids=[0, 1],
            token_embeddings=np.ones((2, 4), dtype=np.float32),
            original_positions=[0, 1],
        )
    )
    result = memory.compact()
    assert result.tokens_before == result.tokens_after
    assert result.extra["evicted"] == 0
    assert result.extra["effective"] is False
    stats = memory.get_stats()
    assert stats.compression_trigger_count == 1
    assert stats.effective_compaction_count == 0
