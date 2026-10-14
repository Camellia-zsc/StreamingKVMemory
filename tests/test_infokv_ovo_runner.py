from types import SimpleNamespace

import numpy as np

from InfoKVStream.run_ovo import BUDGETS, GROUPS, atomic_json, check_pilot, memory_config, rebuild_predictions, select_pilot
from evaluation_pipeline.memory import build_memory_method, Observation, resolve_memory_config
from evaluation_pipeline.memory.qwen25vl_gpu import InfoKVGPUIntegrator, VisualTokenLedger


def test_all_experiment_budgets_have_recent_window_and_exact_total():
    for budget in BUDGETS:
        cfg = resolve_memory_config(memory_config(budget))
        assert cfg.recent.budget + cfg.long_term.budget == budget
        assert cfg.scoring.v2v_observation_tokens <= cfg.recent.budget
        assert cfg.compression.interval_chunks == 1


def test_pilot_selection_covers_every_task_and_long_context_without_gold():
    rows = []
    for i, task in enumerate(sum(GROUPS.values(), [])):
        rows += [{"id": i * 10, "task": task, "realtime": 13},
                 {"id": i * 10 + 1, "task": task, "realtime": 170}]
    picked = select_pilot(rows)
    by_id = {str(r["id"]): r for r in rows}
    for group, ids in picked.items():
        assert {by_id[s]["task"] for s in ids} == set(GROUPS[group])
        assert any(by_id[s]["realtime"] >= 160 for s in ids)


def test_rebuild_uses_atomic_sample_commits_not_partial_jsonl(tmp_path):
    atomic_json(tmp_path / "samples/0.json", {"record": {"sample_id": "0", "status": "success"}})
    (tmp_path / "predictions.jsonl").write_text('{"sample_id":')
    done = rebuild_predictions(tmp_path)
    assert done["0"]["status"] == "success"
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 1


def test_pilot_gate_requires_all_budgets_and_actual_compaction(tmp_path):
    manifest = {"pilot_ids": {"backward": ["0"]}}
    for b in BUDGETS:
        p = tmp_path / "pilot/backward" / f"budget_{b}"
        atomic_json(p / "samples/0.json", {"record": {"sample_id": "0", "status": "success",
                    "runtime": {"managed_visual_tokens_at_query": b, "effective_compaction_count": 2}}})
        rebuild_predictions(p)
    assert check_pilot(tmp_path, manifest)
    (tmp_path / "pilot/backward/budget_50/predictions.jsonl").write_text("")
    assert not check_pilot(tmp_path, manifest)


def test_periodic_integrator_noop_does_not_gather_and_reset_updates_prefix():
    memory = build_memory_method(memory_config(50))
    memory.observe(Observation(token_ids=[0, 1], original_positions=[0, 1],
                               chunk_embedding=np.ones(4)))
    integ = InfoKVGPUIntegrator.__new__(InfoKVGPUIntegrator)
    integ.memory = memory
    integ.prefix_len = 14
    integ.ledger = VisualTokenLedger(14)
    integ.ledger.register_chunk([0, 1], [0, 1], [14, 15])
    integ.wrapper = SimpleNamespace(_cache_lengths=lambda: [16] * 4)
    integ.qa_model = SimpleNamespace(visual_start_idx=7, num_layers=4)
    integ._refresh_keys_from_cache = lambda: None
    integ.events = []
    integ._effective_index = 0
    event = integ.maybe_compact()
    assert not event.effective
    assert event.cache_lens_after == [16] * 4
    assert memory.get_stats().compression_trigger_count == 1
    integ.reset()
    assert integ.prefix_len == integ.ledger.prefix_len == 7
    assert len(integ.ledger) == 0


def test_bulk_recent_annotation_preserves_selection_and_all_scores():
    import copy
    import types
    memory = build_memory_method(memory_config(100))
    rng = np.random.default_rng(13)
    for i in range(3):
        memory.observe(Observation(token_ids=list(range(i * 60, (i + 1) * 60)),
                                   chunk_embedding=rng.normal(size=8),
                                   shallow_hidden=rng.normal(size=(60, 8)),
                                   deep_hidden=rng.normal(size=(60, 8)),
                                   attention_importance=rng.random(60)))
    legacy = copy.deepcopy(memory)
    original = type(memory)._annotate_norms
    legacy._annotate_norms = types.MethodType(
        lambda self, record, pool, **kwargs: original(self, record, pool), legacy)
    memory.compact()
    legacy.compact()
    assert memory.retained_token_ids() == legacy.retained_token_ids()
    assert memory.sidecar_rows() == legacy.sidecar_rows()
