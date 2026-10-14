"""Layerwise retention invariants: identities, budget and irreversible eviction."""
import numpy as np
import pytest
import torch

from evaluation_pipeline.memory import InfoKVStreamMemory, resolve_memory_config, Observation
from evaluation_pipeline.memory.qwen25vl_gpu import VisualTokenLedger, KVCacheCompactor, InfoKVCapture
from test_memory_gpu_integration import _StubModel, _FakeBackend


def memory():
    return InfoKVStreamMemory(resolve_memory_config({
        "method": "infokv_stream", "total_visual_budget": 2,
        "recent": {"enabled": True, "budget": 1}, "long_term": {"budget": 1},
        "scoring": {"temporal_novelty": {"enabled": False, "weight": 0},
                    "layer_evolution": {"enabled": True, "weight": 1, "layerwise": True},
                    "v2v": {"enabled": False, "weight": 0}},
    }))


def test_layer_specific_topk_does_not_resurrect_evicted_token():
    m = memory()
    m.observe(Observation(token_ids=[0, 1, 2], layer_evolution_by_layer={
        0: [1., 0., 0.], 1: [0., 1., 0.]}, chunk_embedding=[1., 0.]))
    first = m.compact()
    assert m.retained_visual_token_ids_by_layer() == {0: [0, 2], 1: [1, 2]}
    # Union ledger exceeds per-layer budget and must never be treated as a
    # physical cache. Neither layer may regain the other's historical token.
    assert len(m.retained_visual_token_ids()) == 3
    assert first.tokens_before == 3 and first.tokens_after == 2
    m.observe(Observation(token_ids=[3], layer_evolution_by_layer={
        0: [0.], 1: [0.]}, chunk_embedding=[1., 0.]))
    m.compact()
    sets = m.retained_visual_token_ids_by_layer()
    assert sets == {0: [0, 3], 1: [1, 3]}
    assert m.effective_compaction_count == 2
    assert m.get_stats().peak_managed_tokens == 3


def test_layer_ledger_preserves_union_and_distinct_physical_indices():
    l = VisualTokenLedger(2)
    l.register_chunk([0, 1, 2], [0, 1, 2], [2, 3, 4],
                     cache_indices_by_layer={0: [2, 3, 4], 1: [2, 3, 4]})
    l.rebuild_after_compaction([0, 2], layer_idx=0)
    l.rebuild_after_compaction([1, 2], layer_idx=1)
    assert len(l) == 3
    assert l.cache_indices([0, 2], layer_idx=0) == [2, 3]
    assert l.cache_indices([1, 2], layer_idx=1) == [2, 3]
    with pytest.raises(KeyError):
        l.cache_index(0, layer_idx=1)
    l.rebuild_after_compaction([2], layer_idx=0)
    assert 0 not in l and 1 in l


def test_compactor_keeps_layer_sets_and_restores_summary_threshold():
    b = _FakeBackend(num_layers=2, threshold=1)
    c = KVCacheCompactor(b)
    c.apply_per_layer({0: [0, 2, 4], 1: [0, 1, 3]})
    assert b.calls == [[[0, 2, 4], [0, 1, 3]]]
    assert b.threshold_at_call == [2] and b.long_term_threshold == 1


def test_layer_capture_matches_input_output_cosine_without_changing_forward():
    torch.manual_seed(6)
    model = _StubModel(num_layers=4)
    cap = InfoKVCapture(model, layer_a=1, layer_b=3, v2v_layer=2, layerwise=True)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        expected = model.language_model(x)
        cap.begin_chunk()
        actual = model.language_model(x)
        data = cap.end_chunk()
    assert torch.equal(expected, actual)
    assert set(data['layer_evolution_by_layer']) == set(range(4))
    current = x
    with torch.no_grad():
        for i, layer in enumerate(model.language_model.layers):
            nxt = layer(current)[0]
            d = (1-torch.nn.functional.cosine_similarity(current.float(), nxt.float(), dim=-1)).clamp(0, 1)
            np.testing.assert_allclose(data['layer_evolution_by_layer'][i], d.flatten().numpy(), atol=1e-6)
            current = nxt
    cap.close()
