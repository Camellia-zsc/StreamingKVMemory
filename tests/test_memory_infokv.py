"""CPU-only tests for the pluggable memory methods (no GPU / no weights).

Covers the synthetic scenarios from the InfoKV task spec (TEST 1..10), the
Vision-to-Vision tests (V2V-1..6), and trigger/effective/overshoot bookkeeping.
Everything here imports only numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation_pipeline.memory import (
    CompressionConfig,
    HermesMemoryMethod,
    InfoKVStreamMemory,
    LongTermConfig,
    MemoryMethodConfig,
    Observation,
    RecentConfig,
    ScoringConfig,
    build_memory_method,
    resolve_memory_config,
)
from evaluation_pipeline.memory.signals import (
    deterministic_topk,
    layer_evolution,
    min_max_normalize,
    normalize_weights,
    temporal_novelty,
)
from evaluation_pipeline.memory.v2v import (
    VisionToVisionAttentionProvider,
    repeat_kv,
    vision_to_vision_attention,
)


def make_infokv(
    *,
    total: int,
    recent: int,
    long_term: int,
    interval: int = 1,
    scoring: ScoringConfig | None = None,
    context_provider=None,
) -> InfoKVStreamMemory:
    config = MemoryMethodConfig(
        method="infokv_stream",
        total_visual_budget=total,
        recent=RecentConfig(enabled=True, budget=recent),
        long_term=LongTermConfig(budget=long_term),
        compression=CompressionConfig(interval_chunks=interval),
        scoring=scoring or ScoringConfig(),
    )
    config.validate()
    memory = build_memory_method(
        method_config=config, context_provider=context_provider
    )
    assert isinstance(memory, InfoKVStreamMemory)
    return memory


def chunk_obs(
    chunk_id: int,
    *,
    tokens: int,
    embedding,
    attention=None,
    shallow=None,
    deep=None,
    query_states=None,
    key_states=None,
    is_visual=None,
    start_token: int | None = None,
):
    start_token = chunk_id * tokens if start_token is None else start_token
    token_ids = list(range(start_token, start_token + tokens))
    return Observation(
        chunk_id=chunk_id,
        chunk_embedding=np.asarray(embedding, dtype=np.float32),
        token_ids=token_ids,
        token_embeddings=np.tile(
            np.asarray(embedding, dtype=np.float32), (tokens, 1)
        ),
        shallow_hidden=shallow,
        deep_hidden=deep,
        attention_importance=attention,
        query_states=query_states,
        key_states=key_states,
        is_visual=is_visual,
        original_positions=token_ids,
    )


def run_stream(memory, observations):
    for observation in observations:
        memory.observe(observation)
        if memory.should_compact():
            memory.compact()
    return memory


def only_novelty():
    return ScoringConfig(
        temporal_novelty_enabled=True,
        temporal_novelty_weight=1.0,
        layer_evolution_enabled=False,
        layer_evolution_weight=0.0,
        v2v_enabled=False,
        v2v_weight=0.0,
    )


# --------------------------------------------------------------------------- #
# signal unit tests
# --------------------------------------------------------------------------- #
def test_temporal_novelty_duplicate_vs_different():
    a = np.ones(8, dtype=np.float32)
    assert temporal_novelty(a, a.copy()) == pytest.approx(0.0, abs=1e-6)
    assert temporal_novelty(a, -a) == pytest.approx(1.0, abs=1e-6)
    assert temporal_novelty(a, None) == 0.0


def test_layer_evolution_identical_vs_different():
    h = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    same = layer_evolution(h, h.copy())
    assert np.allclose(same, 0.0, atol=1e-5)
    different = layer_evolution(h, h + 3.0)
    assert np.all(different > same)
    with pytest.raises(ValueError):
        layer_evolution(h, h[:2])


def test_min_max_normalize_and_weights():
    normed = min_max_normalize([2.0, 4.0, 6.0])
    assert normed[0] == pytest.approx(0.0)
    assert normed[-1] == pytest.approx(1.0)
    assert np.allclose(min_max_normalize([5.0, 5.0]), 0.0)
    weights, warning = normalize_weights({"a": 2.0, "b": 2.0})
    assert warning is not None
    assert sum(weights.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_weights({"a": -1.0})


def test_deterministic_topk_tie_break():
    idx = deterministic_topk([0.5, 0.5, 0.5], [10, 20, 30], [1, 2, 3], k=1)
    assert idx == [2]
    assert deterministic_topk([0.5, 0.5, 0.5], [10, 20, 30], [1, 2, 3], 3) == [2, 1, 0]


# --------------------------------------------------------------------------- #
# TEST 1: static repeated video
# --------------------------------------------------------------------------- #
def test_static_repeated_video_low_novelty():
    memory = make_infokv(total=20, recent=8, long_term=12)
    base = np.ones(16, dtype=np.float32)
    rng = np.random.default_rng(1)
    observations = [
        chunk_obs(
            chunk_id,
            tokens=4,
            embedding=base + 0.001 * rng.normal(size=16).astype(np.float32),
        )
        for chunk_id in range(10)
    ]
    run_stream(memory, observations)
    novelties = [r.novelty_raw for r in memory.managed_tokens()]
    assert max(novelties) < 0.05
    assert len(memory.long_term) <= memory.long_term_budget
    assert len(memory.recent) <= 8


# --------------------------------------------------------------------------- #
# TEST 2: salient event (chunk 5 stands out)
# --------------------------------------------------------------------------- #
def test_salient_event_chunk_survives_tight_budget():
    memory = make_infokv(total=1, recent=0, long_term=1, scoring=only_novelty())
    base = np.ones(16, dtype=np.float32)
    observations = [
        chunk_obs(
            chunk_id, tokens=2, embedding=base * (-1.0 if chunk_id == 5 else 1.0)
        )
        for chunk_id in range(10)
    ]
    run_stream(memory, observations)
    event_novelty = {r.chunk_id: r.novelty_raw for r in memory.managed_tokens()}
    assert event_novelty.get(5, 0.0) > 0.9
    assert {r.chunk_id for r in memory.long_term} == {5}


# --------------------------------------------------------------------------- #
# TEST 3: representation evolution
# --------------------------------------------------------------------------- #
def test_layer_evolution_prefers_changed_token():
    memory = make_infokv(total=1, recent=0, long_term=1)
    shallow = np.zeros((2, 8), dtype=np.float32)
    deep = shallow.copy()
    deep[1] = 5.0
    memory.observe(
        Observation(
            chunk_id=0,
            chunk_embedding=np.ones(8, dtype=np.float32),
            token_ids=[0, 1],
            token_embeddings=np.ones((2, 8), dtype=np.float32),
            shallow_hidden=shallow,
            deep_hidden=deep,
            attention_importance=[0.5, 0.5],
            original_positions=[0, 1],
        )
    )
    memory.compact()
    assert [r.token_id for r in memory.long_term] == [1]


# --------------------------------------------------------------------------- #
# TEST 4: injected context importance (no Q/K provider available)
# --------------------------------------------------------------------------- #
def test_injected_context_importance_prefers_high_token():
    memory = make_infokv(total=1, recent=0, long_term=1)
    memory.observe(
        Observation(
            chunk_id=0,
            chunk_embedding=np.ones(8, dtype=np.float32),
            token_ids=[0, 1],
            token_embeddings=np.ones((2, 8), dtype=np.float32),
            shallow_hidden=np.zeros((2, 8), dtype=np.float32),
            deep_hidden=np.zeros((2, 8), dtype=np.float32),
            attention_importance=[0.9, 0.1],
            original_positions=[0, 1],
        )
    )
    memory.compact()
    assert [r.token_id for r in memory.long_term] == [0]


# --------------------------------------------------------------------------- #
# TEST 5: recent-buffer protection
# --------------------------------------------------------------------------- #
def test_recent_buffer_protection():
    memory = make_infokv(total=10, recent=4, long_term=6, interval=1)
    rng = np.random.default_rng(3)
    seen: list[int] = []
    for chunk_id in range(12):
        seen.append(chunk_id)
        memory.observe(
            chunk_obs(
                chunk_id,
                tokens=1,
                embedding=rng.normal(size=8).astype(np.float32),
                start_token=chunk_id,
            )
        )
        if memory.should_compact():
            memory.compact()
        assert len(memory.recent) <= 4
        assert len(memory.long_term) <= 6
        assert len(memory.recent) + len(memory.long_term) <= 10
    assert [r.token_id for r in memory.recent] == seen[-4:]


# --------------------------------------------------------------------------- #
# TEST 6: periodic compression
# --------------------------------------------------------------------------- #
def test_periodic_compression_trigger_count():
    memory = make_infokv(total=8, recent=2, long_term=6, interval=4)
    rng = np.random.default_rng(4)
    triggers = 0
    for chunk_id in range(12):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=2, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
            triggers += 1
    assert triggers == 3
    assert memory.get_stats().compression_trigger_count == 3


def test_periodic_compression_interval_two_triggers():
    memory = make_infokv(total=8, recent=2, long_term=6, interval=4)
    rng = np.random.default_rng(5)
    triggers = 0
    for chunk_id in range(10):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=2, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
            triggers += 1
    assert triggers == 2


def test_cache_does_not_grow_unbounded():
    memory = make_infokv(total=16, recent=4, long_term=12, interval=4)
    rng = np.random.default_rng(6)
    for chunk_id in range(40):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=4, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
            assert len(memory.recent) + len(memory.long_term) <= 16
    assert len(memory.long_term) <= 12


# --------------------------------------------------------------------------- #
# TEST 7: reset
# --------------------------------------------------------------------------- #
def test_reset_clears_all_state():
    memory = make_infokv(total=8, recent=2, long_term=6, interval=1)
    rng = np.random.default_rng(7)
    for chunk_id in range(6):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=2, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
    assert memory.total_seen_tokens > 0
    memory.reset()
    stats = memory.get_stats()
    assert memory.recent == [] and memory.long_term == [] and memory.candidates == []
    assert memory.compaction_log == []
    assert stats.compression_trigger_count == 0
    assert stats.effective_compaction_count == 0
    assert memory.total_seen_tokens == 0
    assert memory.chunk_counter == 0
    assert memory.peak_managed_tokens == 0


# --------------------------------------------------------------------------- #
# TEST 8: determinism
# --------------------------------------------------------------------------- #
def test_selection_is_deterministic():
    def run(seed: int):
        memory = make_infokv(total=16, recent=4, long_term=12, interval=3)
        rng = np.random.default_rng(seed)
        for chunk_id in range(20):
            memory.observe(
                chunk_obs(
                    chunk_id,
                    tokens=4,
                    embedding=rng.normal(size=8).astype(np.float32),
                    attention=rng.uniform(size=4),
                )
            )
            if memory.should_compact():
                memory.compact()
        return memory.retained_token_ids()

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_retained_tokens_keep_time_order():
    memory = make_infokv(total=16, recent=4, long_term=12, interval=1)
    rng = np.random.default_rng(21)
    for chunk_id in range(20):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=4, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
    ids = memory.retained_token_ids()
    assert ids == sorted(ids)


# --------------------------------------------------------------------------- #
# TEST 9: method switch (factory / type / interface only, no GPU)
# --------------------------------------------------------------------------- #
def test_method_switch_builds_correct_types():
    hermes = build_memory_method(
        {"memory": {"method": "hermes", "compression": {"budget": 123}}}
    )
    assert isinstance(hermes, HermesMemoryMethod) and hermes.budget == 123
    infokv = build_memory_method(
        {
            "memory": {
                "method": "infokv_stream",
                "total_visual_budget": 100,
                "recent": {"enabled": True, "budget": 20},
                "long_term": {"budget": 80},
                "compression": {"interval_chunks": 4},
                "scoring": {
                    "v2v": {
                        "enabled": True,
                        "weight": 0.3,
                        "observation_tokens": 32,
                    }
                },
            }
        }
    )
    assert isinstance(infokv, InfoKVStreamMemory)
    assert infokv.observation_tokens == 32
    for method in (hermes, infokv):
        for name in ("reset", "observe", "should_compact", "compact", "get_stats"):
            assert callable(getattr(method, name))


def test_hermes_adapter_never_needs_gpu():
    adapter = build_memory_method(
        {"memory": {"method": "hermes", "compression": {"budget": 5}}}
    )
    for chunk_id in range(4):
        adapter.observe(
            Observation(
                chunk_id=chunk_id,
                token_ids=list(range(chunk_id * 2, chunk_id * 2 + 2)),
            )
        )
    assert adapter.should_compact() is True
    result = adapter.compact()
    assert result.tokens_before == 8 and result.tokens_after == 5
    assert result.extra["delegated"] is False


# --------------------------------------------------------------------------- #
# TEST 10: legacy HERMES config compatibility
# --------------------------------------------------------------------------- #
def test_legacy_hermes_config_defaults_to_hermes():
    resolved = resolve_memory_config(
        {"memory": {"compression": {"enabled": True, "budget": 2000}}}
    )
    assert resolved.method == "hermes" and resolved.hermes_budget == 2000
    assert resolve_memory_config({}).method == "hermes"
    method = build_memory_method(
        {"memory": {"compression": {"enabled": True, "budget": 2000}}}
    )
    assert isinstance(method, HermesMemoryMethod)


def test_legacy_recent_window_and_scoring_aliases():
    resolved = resolve_memory_config(
        {
            "memory": {
                "method": "infokv_stream",
                "total_visual_budget": 10,
                "recent": {"enabled": True, "budget": 4},
                "long_term": {"budget": 6},
                "scoring": {
                    "temporal_novelty": {
                        "enabled": True,
                        "weight": 0.5,
                        "recent_window": 3,
                    },
                    "recent_context_importance": {"enabled": True, "weight": 0.5},
                },
            }
        }
    )
    assert resolved.scoring.temporal_novelty_reference_chunks == 3
    assert resolved.scoring.v2v_enabled is True


# --------------------------------------------------------------------------- #
# P3/P4: trigger vs effective + transient overshoot
# --------------------------------------------------------------------------- #
def test_trigger_count_exceeds_effective_compaction_count():
    memory = make_infokv(total=64, recent=16, long_term=48, interval=4)
    rng = np.random.default_rng(0)
    for chunk_id in range(20):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=8, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
    stats = memory.get_stats()
    assert stats.compression_trigger_count == 5
    assert stats.effective_compaction_count == 3
    assert stats.compression_trigger_count > stats.effective_compaction_count
    events = memory.compaction_events()
    assert [e["effective"] for e in events] == [False, False, True, True, True]
    assert [e["evicted"] for e in events] == [0, 0, 32, 32, 32]
    assert [e["effective_index"] for e in events] == [None, None, 1, 2, 3]
    # Pool-level before -> after (spec P4: 16->16, 48->48, 80->48 ...).
    assert [e["before"] for e in events] == [16, 48, 80, 80, 80]
    assert [e["after"] for e in events] == [16, 48, 48, 48, 48]


def test_transient_overshoot_stats():
    memory = make_infokv(total=64, recent=16, long_term=48, interval=4)
    rng = np.random.default_rng(0)
    for chunk_id in range(20):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=8, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
    stats = memory.get_stats()
    assert stats.peak_managed_tokens == 96
    assert stats.peak_budget_overshoot_tokens == 32
    assert stats.peak_budget_overshoot_ratio == pytest.approx(0.5)
    assert stats.compression_interval_chunks == 4


# --------------------------------------------------------------------------- #
# V2V-1: matching historical key scores higher
# --------------------------------------------------------------------------- #
def test_v2v_1_high_similarity_key_scores_higher():
    rng = np.random.default_rng(0)
    d = 8
    q = rng.normal(size=(1, 1, 3, d)).astype(np.float32)
    k_match = q[0, 0, 0].reshape(1, 1, 1, d)
    k_far = -k_match
    keys = np.concatenate([k_match, k_far], axis=2).astype(np.float32)
    scores = vision_to_vision_attention(q, keys)
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]


# ---------------------------------------------------------------------------
# V2V-2: aggregation equals softmax -> query mean -> head mean
# ---------------------------------------------------------------------------
def test_v2v_2_query_and_head_aggregation_matches_manual():
    rng = np.random.default_rng(1)
    b, h, r, t, d = 1, 4, 5, 6, 8
    q = rng.normal(size=(b, h, r, d)).astype(np.float32)
    k = rng.normal(size=(b, h, t, d)).astype(np.float32)
    scores = vision_to_vision_attention(q, k)
    logits = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(d)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    attn = exp / exp.sum(axis=-1, keepdims=True)
    manual = attn.mean(axis=2).mean(axis=1)
    assert np.allclose(scores, manual, atol=1e-5)


# ---------------------------------------------------------------------------
# V2V-3: GQA/MQA head handling
# ---------------------------------------------------------------------------
def test_v2v_3_gqa_repeat_kv():
    rng = np.random.default_rng(2)
    hq, hkv, r, t, d = 8, 2, 4, 6, 16
    q = rng.normal(size=(1, hq, r, d)).astype(np.float32)
    k = rng.normal(size=(1, hkv, t, d)).astype(np.float32)
    assert repeat_kv(k, hq // hkv).shape == (1, hq, t, d)
    scores = vision_to_vision_attention(q, k)
    assert scores.shape == (1, t)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores.sum(axis=-1), 1.0, atol=1e-4)
    with pytest.raises(ValueError):
        vision_to_vision_attention(q, np.repeat(k, 3, axis=1)[:, :7])


# ---------------------------------------------------------------------------
# V2V-4: the observation window is the Recent Buffer tail
# ---------------------------------------------------------------------------
def test_v2v_4_query_source_is_recent_buffer_tail():
    rng = np.random.default_rng(3)
    d = 8
    scoring = ScoringConfig(v2v_weight=1.0)
    memory = make_infokv(total=8, recent=2, long_term=6, interval=1, scoring=scoring)
    q_recent = rng.normal(size=(2, 1, d)).astype(np.float32)
    k0 = rng.normal(size=(2, 1, d)).astype(np.float32)
    memory.observe(
        chunk_obs(0, tokens=2, embedding=rng.normal(size=d), key_states=k0)
    )
    memory.observe(
        chunk_obs(1, tokens=2, embedding=rng.normal(size=d), query_states=q_recent)
    )
    memory.compact()
    q_canonical = q_recent.transpose(1, 0, 2)[None]  # [1, 1, R, D]
    k_canonical = k0.transpose(1, 0, 2)[None]  # [1, 1, T, D]
    expected = VisionToVisionAttentionProvider().score(q_canonical, k_canonical)[0]
    by_id = {r.token_id: r for r in memory.managed_tokens()}
    for i, tid in enumerate([0, 1]):
        assert by_id[tid].v2v_raw == pytest.approx(expected[i], abs=1e-6)


def test_v2v_4b_pool_query_states_are_ignored():
    d = 8
    scoring = ScoringConfig(v2v_weight=1.0)

    def run(pool_query_states):
        rng = np.random.default_rng(4)
        memory = make_infokv(total=8, recent=2, long_term=6, interval=1, scoring=scoring)
        q_recent = rng.normal(size=(2, 1, d)).astype(np.float32)
        k0 = rng.normal(size=(2, 1, d)).astype(np.float32)
        memory.observe(
            chunk_obs(
                0,
                tokens=2,
                embedding=rng.normal(size=d),
                key_states=k0,
                query_states=pool_query_states,
            )
        )
        memory.observe(
            chunk_obs(1, tokens=2, embedding=rng.normal(size=d), query_states=q_recent)
        )
        memory.compact()
        return {
            r.token_id: r.v2v_raw
            for r in memory.managed_tokens()
            if r.chunk_id == 0
        }

    a = run(np.zeros((2, 1, d), dtype=np.float32))
    b = run(np.ones((2, 1, d), dtype=np.float32) * 99.0)
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# V2V-5: strictly no text leakage
# ---------------------------------------------------------------------------
def test_v2v_5_no_text_leakage():
    rng = np.random.default_rng(5)
    d = 8
    seen_keys = []

    class SpyProvider:
        name = "spy"

        def score(self, recent_queries, historical_keys, **kwargs):
            seen_keys.append(np.asarray(historical_keys))
            return vision_to_vision_attention(recent_queries, historical_keys)

    scoring = ScoringConfig(v2v_weight=1.0)
    memory = make_infokv(
        total=8,
        recent=2,
        long_term=6,
        interval=1,
        scoring=scoring,
        context_provider=SpyProvider(),
    )
    k_visual = rng.normal(size=(2, 1, d)).astype(np.float32)
    text_key = np.full((1, 1, d), 777.0, dtype=np.float32)
    keys = np.concatenate([k_visual, text_key], axis=0)
    memory.observe(
        Observation(
            chunk_id=0,
            chunk_embedding=rng.normal(size=d).astype(np.float32),
            token_ids=[0, 1, 2],
            token_embeddings=np.ones((3, d), dtype=np.float32),
            key_states=keys,
            query_states=rng.normal(size=(3, 1, d)).astype(np.float32),
            is_visual=[True, True, False],
            original_positions=[0, 1, 2],
        )
    )
    memory.observe(
        chunk_obs(
            1,
            tokens=2,
            embedding=rng.normal(size=d),
            query_states=rng.normal(size=(2, 1, d)).astype(np.float32),
        )
    )
    memory.compact()
    assert seen_keys
    for keys_seen in seen_keys:
        assert not np.any(np.isclose(keys_seen, 777.0))
    text_record = [r for r in memory.managed_tokens() if r.token_id == 2]
    assert text_record and text_record[0].v2v_raw == 0.0


# ---------------------------------------------------------------------------
# V2V-6: fusion -> higher A^V2V increases final_score and retention
# ---------------------------------------------------------------------------
def test_v2v_6_fusion_prefers_high_v2v_token():
    d = 8
    scoring = ScoringConfig(
        temporal_novelty_weight=0.0,
        layer_evolution_weight=0.0,
        v2v_weight=1.0,
    )

    def run(long_term):
        rng = np.random.default_rng(6)
        memory = make_infokv(
            total=8, recent=2, long_term=long_term, interval=1, scoring=scoring
        )
        q_recent = rng.normal(size=(2, 1, d)).astype(np.float32)
        k_match = np.repeat(q_recent[:1], 2, axis=0)
        k_far = -k_match
        candidate_keys = np.stack([k_match[0], k_far[0]], axis=0)
        memory.observe(
            chunk_obs(
                0, tokens=2, embedding=rng.normal(size=d), key_states=candidate_keys
            )
        )
        memory.observe(
            chunk_obs(
                1, tokens=2, embedding=rng.normal(size=d), query_states=q_recent
            )
        )
        memory.compact()
        return memory

    # With room for both, the strongly-attended token scores higher.
    roomy = run(2)
    by_id = {r.token_id: r for r in roomy.long_term}
    assert by_id[0].v2v_raw > by_id[1].v2v_raw
    assert by_id[0].final_score > by_id[1].final_score
    # Under a tight budget the strongly-attended token is the one retained.
    tight = run(1)
    assert [r.token_id for r in tight.long_term] == [0]


# --------------------------------------------------------------------------- #
# P13: v1 selection is a single global retained set
# --------------------------------------------------------------------------- #
def test_global_retained_indices_semantics():
    memory = make_infokv(total=8, recent=2, long_term=4, interval=1)
    rng = np.random.default_rng(9)
    for chunk_id in range(10):
        memory.observe(
            chunk_obs(
                chunk_id, tokens=2, embedding=rng.normal(size=8).astype(np.float32)
            )
        )
        if memory.should_compact():
            memory.compact()
    global_ids = memory.retained_visual_indices()
    assert global_ids == sorted(global_ids)
    assert set(global_ids) == {r.token_id for r in memory.long_term if r.is_visual}


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_infokv_budget_sum_must_respect_total():
    with pytest.raises(ValueError):
        MemoryMethodConfig(
            method="infokv_stream",
            total_visual_budget=10,
            recent=RecentConfig(enabled=True, budget=8),
            long_term=LongTermConfig(budget=8),
        ).validate()


def test_infokv_requires_total_budget():
    with pytest.raises(ValueError):
        MemoryMethodConfig(
            method="infokv_stream",
            recent=RecentConfig(enabled=True, budget=1),
            long_term=LongTermConfig(budget=1),
        ).validate()


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        MemoryMethodConfig(method="does_not_exist").validate()


def test_summary_strategy_must_be_none():
    with pytest.raises(NotImplementedError):
        MemoryMethodConfig(
            method="infokv_stream",
            total_visual_budget=10,
            recent=RecentConfig(enabled=True, budget=4),
            long_term=LongTermConfig(budget=6),
            compression=CompressionConfig(
                interval_chunks=1, summary_strategy="mean_token"
            ),
        ).validate()


def test_v2v_observation_tokens_validated():
    with pytest.raises(ValueError):
        MemoryMethodConfig(
            method="infokv_stream",
            total_visual_budget=10,
            recent=RecentConfig(enabled=True, budget=4),
            long_term=LongTermConfig(budget=6),
            scoring=ScoringConfig(v2v_observation_tokens=0),
        ).validate()


def test_layer_evolution_layers_must_pair():
    with pytest.raises(ValueError):
        MemoryMethodConfig(
            method="infokv_stream",
            total_visual_budget=10,
            recent=RecentConfig(enabled=True, budget=4),
            long_term=LongTermConfig(budget=6),
            scoring=ScoringConfig(layer_a=3, layer_b=None),
        ).validate()


# --------------------------------------------------------------------------- #
# sidecar / stats schema
# --------------------------------------------------------------------------- #
def test_sidecar_rows_and_stats_schema():
    memory = make_infokv(total=8, recent=2, long_term=6, interval=1)
    rng = np.random.default_rng(31)
    for chunk_id in range(6):
        memory.observe(
            chunk_obs(
                chunk_id,
                tokens=2,
                embedding=rng.normal(size=8).astype(np.float32),
                attention=rng.uniform(size=2),
            )
        )
        if memory.should_compact():
            memory.compact()
    rows = memory.sidecar_rows()
    assert rows
    for key in (
        "token_id",
        "chunk_id",
        "novelty_raw",
        "novelty_norm",
        "evolution_raw",
        "evolution_norm",
        "v2v_raw",
        "v2v_norm",
        "final_score",
        "selected",
        "is_visual",
    ):
        assert key in rows[0]
    stats = memory.get_stats().as_dict()
    for key in (
        "memory_method",
        "budget_total",
        "budget_recent",
        "budget_long_term",
        "compression_trigger_count",
        "effective_compaction_count",
        "tokens_before",
        "tokens_after",
        "recent_size",
        "long_term_size",
        "evicted_count",
        "retention_ratio",
        "compression_wallclock",
        "compression_interval_chunks",
        "peak_managed_tokens",
        "peak_budget_overshoot_tokens",
        "peak_budget_overshoot_ratio",
    ):
        assert key in stats
