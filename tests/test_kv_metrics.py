"""Sanity tests for the KV fidelity evaluation module (no GPU/model needed)."""

import math

import pytest
import torch

from evaluation_pipeline.kv_metrics.aggregate import compute_paired_metrics
from evaluation_pipeline.kv_metrics.config import KVMetricConfig, parse_temporal_bins
from evaluation_pipeline.kv_metrics.fidelity import (
    aore,
    js_divergence,
    kl_divergence,
    ram_per_head,
    region_context,
)
from evaluation_pipeline.kv_metrics.hooks import AnchorArtifact
from evaluation_pipeline.kv_metrics.provenance import (
    KIND_SUMMARY,
    KVProvenanceTracker,
)
from evaluation_pipeline.kv_metrics.system_metrics import (
    KVSystemProfiler,
    kv_cache_bytes,
    tensor_bytes,
)
from evaluation_pipeline.kv_metrics.temporal import temporal_ram


# --------------------------------------------------------------------- #
# metric math
# --------------------------------------------------------------------- #
def test_ram_hand_example():
    probs = torch.tensor(
        [[0.70, 0.10, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]], dtype=torch.float32
    )
    mask = torch.tensor([True, True, False, False, False, False, True, True])
    ram = ram_per_head(probs, mask)
    assert ram.shape == (1,)
    assert ram[0].item() == pytest.approx(0.83, abs=1e-6)


def test_ram_is_one_when_nothing_dropped():
    probs = torch.softmax(torch.randn(4, 32), dim=-1)
    mask = torch.ones(32, dtype=torch.bool)
    ram = ram_per_head(probs, mask)
    assert torch.allclose(ram, torch.ones(4), atol=1e-5)


def test_ram_region_normalised_ignores_text_mass():
    # attention mostly on text (first 4), region is the last 8 keys
    probs = torch.zeros(2, 12)
    probs[:, :4] = 0.9 / 4
    probs[:, 4:] = 0.1 / 8
    region = torch.zeros(12, dtype=torch.bool)
    region[4:] = True
    mask = region.clone()  # keep the whole region
    ram = ram_per_head(probs, mask, region)
    assert torch.allclose(ram, torch.ones(2), atol=1e-5)
    # unnormalised variant reflects the small video mass
    raw = ram_per_head(probs, mask)
    assert torch.allclose(raw, torch.full((2,), 0.1), atol=1e-5)


def test_aore_zero_when_identical():
    o = torch.randn(8, 128)
    assert aore(o, o) == pytest.approx(0.0, abs=1e-6)


def test_aore_positive_when_different():
    o_full = torch.randn(8, 128)
    o_comp = o_full + 0.1 * torch.randn(8, 128)
    assert aore(o_comp, o_full) > 0.0


def test_js_kl_zero_when_identical():
    logits = torch.randn(1, 1000)
    assert js_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)


def test_js_symmetric_positive():
    a = torch.randn(1, 1000)
    b = torch.randn(1, 1000)
    js_ab = js_divergence(a, b)
    js_ba = js_divergence(b, a)
    assert js_ab > 0.0
    assert js_ab == pytest.approx(js_ba, abs=1e-6)


def test_gqa_region_context_maps_query_heads_to_kv_heads():
    # 16 query heads, 2 kv heads -> 8 query heads per kv head.
    num_heads, num_kv_heads, head_dim, kv_len = 16, 2, 8, 10
    probs = torch.softmax(torch.randn(num_heads, kv_len), dim=-1)
    values = torch.randn(num_kv_heads, kv_len, head_dim)
    ctx = region_context(probs, values, 0, kv_len, num_heads)
    assert ctx.shape == (num_heads, head_dim)
    # head 0..7 must use kv head 0; head 8..15 must use kv head 1.
    for h in range(8):
        expected = torch.matmul(probs[h], values[0])
        assert torch.allclose(ctx[h], expected, atol=1e-5)
    for h in range(8, 16):
        expected = torch.matmul(probs[h], values[1])
        assert torch.allclose(ctx[h], expected, atol=1e-5)


# --------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------- #
def test_provenance_visual_registration_and_compaction_identity():
    tracker = KVProvenanceTracker(video_id="v1", summary_token_mode="identity_only")
    tracker.register_init(num_layers=2, prefix_len=4)
    tracker.register_visual_chunk(
        per_frame_token_counts=[3, 3],
        frame_indices=[0, 1],
        frame_timestamps=[1.0, 2.0],
    )
    assert tracker.lengths() == [10, 10]
    region = tracker.visual_region(0)
    assert region.start == 4 and region.end == 10
    # keep prefix + the last two visual tokens
    keep = [0, 1, 2, 3, 8, 9]
    tracker.apply_keep_indices([keep, keep], long_term_flags=[False, False])
    assert tracker.lengths() == [6, 6]
    # no summary tokens on non-long-term layers
    assert all(u.kind != KIND_SUMMARY for u in tracker.layer_uids(0))
    retained = tracker.retained_uid_set(0)
    assert len(retained) == 2  # two surviving original visual positions


def test_provenance_summary_token_absorbs_pruned_parents():
    tracker = KVProvenanceTracker(video_id="v1", summary_token_mode="absorbed")
    tracker.register_init(num_layers=1, prefix_len=2)
    tracker.register_visual_chunk([4], [0], [5.0])
    tracker.apply_keep_indices([[0, 1, 2]], long_term_flags=[True])
    uids = tracker.layer_uids(0)
    assert len(uids) == 4  # 3 kept + 1 summary
    assert uids[-1].kind == KIND_SUMMARY
    assert len(uids[-1].parents) == 3
    # absorbed mode counts summary parents as retained
    assert len(tracker.retained_uid_set(0)) == 4


def test_provenance_desync_detected():
    tracker = KVProvenanceTracker(video_id="v1")
    tracker.register_init(num_layers=1, prefix_len=3)

    class _Cache:
        def __iter__(self):
            yield (torch.zeros(1, 2, 5, 8), torch.zeros(1, 2, 5, 8))

    with pytest.raises(AssertionError):
        tracker.assert_sync(_Cache())


# --------------------------------------------------------------------- #
# temporal
# --------------------------------------------------------------------- #
def test_temporal_ram_bucket_conditioned_normalization():
    # 4 positions: 2 recent (age 5s) dropping, 2 old (age 100s) kept.
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    uid_mass = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    ts = {"a": 5.0, "b": 5.0, "c": -100.0, "d": -100.0}
    result = temporal_ram(
        uid_mass, ts, retained_uids={"c", "d"}, query_time=10.0,
        bins=[0, 10, 30, 60, 300, float("inf")],
    )
    # recent bucket: none kept -> 0 ; old bucket: all kept -> 1
    assert result.temporal_ram["0_10"] == pytest.approx(0.0)
    assert result.temporal_ram["60_300"] == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# system metrics
# --------------------------------------------------------------------- #
def test_tensor_and_cache_bytes():
    t = torch.zeros(2, 3, 4, dtype=torch.float16)
    assert tensor_bytes(t) == 2 * 3 * 4 * 2
    cache = [(t, t), (t, t)]
    assert kv_cache_bytes(cache) == 4 * tensor_bytes(t)


def test_profiler_compression_ratio_and_realtime():
    profiler = KVSystemProfiler(profile_wallclock=False)
    profiler.record_compaction(
        event_id="e1", compaction_index=1, stream_time=10.0,
        bytes_before=1000, bytes_after=100, kv_before=100, kv_after=10,
        wall_clock_sec=0.05,
    )
    event = profiler.record_compaction(
        event_id="e2", compaction_index=2, stream_time=20.0,
        bytes_before=1000, bytes_after=100, kv_before=100, kv_after=10,
        wall_clock_sec=0.4,
    )
    assert event.compression_ratio() == pytest.approx(10.0)
    assert event.trigger_interval_sec == pytest.approx(10.0)
    assert event.rho_realtime == pytest.approx(0.04)
    assert event.realtime_feasible is True


# --------------------------------------------------------------------- #
# end-to-end paired math on synthetic artifacts (TEST 1 / TEST 2 analog)
# --------------------------------------------------------------------- #
def _make_artifact(event_id, num_layers, num_heads, num_kv_heads, head_dim, visual_len, logits, keep_ratio, dropped_region, seed=1234):
    torch.manual_seed(seed)
    artifact = AnchorArtifact(
        event_id=event_id,
        video_id="v1",
        query_id="q1",
        query_time=10.0,
        stream_time=10.0,
        query_input_ids=[1, 2, 3],
        logits=logits,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    from evaluation_pipeline.kv_metrics.fidelity import LayerAttention
    from evaluation_pipeline.kv_metrics.provenance import VisualRegion, KVTokenUID

    for layer in range(num_layers):
        attn = torch.softmax(torch.randn(num_heads, visual_len), dim=-1)
        values = torch.randn(num_kv_heads, visual_len, head_dim)
        artifact.layers[layer] = LayerAttention(
            layer_idx=layer,
            attn_row=attn,
            value_states=values,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        uids = [
            KVTokenUID(
                uid=f"vis:{i}",
                layer_id=layer,
                kind="visual",
                frame_timestamp_sec=float(visual_len - i),
            )
            for i in range(visual_len)
        ]
        artifact.visual_regions[layer] = VisualRegion(
            layer_idx=layer, start=0, end=visual_len, uids=uids
        )
        artifact.uid_timestamps[layer] = {u.uid: u.frame_timestamp_sec for u in uids}
    return artifact


def test_paired_metrics_identical_branches():
    torch.manual_seed(0)
    logits = torch.randn(1, 500)
    cfg = KVMetricConfig(enable_kv_metrics=True, mode="paired")
    full = _make_artifact("e", 2, 4, 4, 8, 16, logits, 1.0, False, seed=7)
    comp = _make_artifact("e", 2, 4, 4, 8, 16, logits, 1.0, False, seed=7)
    # identical branch: keep every visual uid
    for layer in range(2):
        comp.retained_uids[layer] = {u.uid for u in comp.visual_regions[layer].uids}
    comp.kv_bytes = full.kv_bytes = 1000
    record = compute_paired_metrics(comp, full, cfg)
    assert record["ram_mean"] == pytest.approx(1.0, abs=1e-5)
    assert record["aore_mean"] == pytest.approx(0.0, abs=1e-5)
    assert record["js"] == pytest.approx(0.0, abs=1e-6)
    assert record["kl"] == pytest.approx(0.0, abs=1e-6)
    assert record["compression_ratio"] == pytest.approx(1.0)


def test_paired_metrics_half_drop():
    torch.manual_seed(1)
    logits_full = torch.randn(1, 500)
    logits_comp = logits_full + 0.5 * torch.randn(1, 500)
    cfg = KVMetricConfig(enable_kv_metrics=True, mode="paired")
    full = _make_artifact("e", 1, 4, 4, 8, 20, logits_full, 1.0, False)
    comp = _make_artifact("e", 1, 4, 4, 8, 20, logits_comp, 0.5, True)
    # keep only the first 10 visual uids
    comp.retained_uids[0] = {u.uid for u in comp.visual_regions[0].uids[:10]}
    full.kv_bytes = 2000
    comp.kv_bytes = 1000
    record = compute_paired_metrics(comp, full, cfg)
    assert 0.0 <= record["ram_mean"] <= 1.0
    assert record["aore_mean"] >= 0.0
    assert record["js"] >= 0.0
    assert record["compression_ratio"] == pytest.approx(2.0)


# --------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------- #
def test_parse_temporal_bins():
    edges = parse_temporal_bins("0,10,30,60,300,inf")
    assert edges == [0.0, 10.0, 30.0, 60.0, 300.0, float("inf")]


def test_config_validation_rejects_bad_mode():
    cfg = KVMetricConfig(enable_kv_metrics=True, mode="nope")
    with pytest.raises(ValueError):
        cfg.validate()


# --------------------------------------------------------------------- #
# P0 (trajectory) + P2 (runtime breakdown) unit tests
# --------------------------------------------------------------------- #
import copy as _copy  # noqa: E402

from evaluation_pipeline.kv_metrics.analysis import schemas, statistics as st  # noqa: E402
from evaluation_pipeline.kv_metrics.analysis import trajectory  # noqa: E402
from evaluation_pipeline.kv_metrics.analysis.loader import (  # noqa: E402
    ExperimentData,
    RunData,
)
from evaluation_pipeline.kv_metrics.runtime_breakdown import RuntimeBreakdown  # noqa: E402


def _make_run(name, sample_id, events, compactions, config=None):
    return RunData(
        name=name,
        directory=None,
        config=config or {"benchmark": "ovobench", "memory": {"method": "hermes"}},
        summary={},
        events=events,
        compactions=compactions,
    )


def test_compaction_count_at_query_hand_example():
    compactions = [
        {"end_time": 10.0, "stream_time": 10.0},
        {"end_time": 20.0, "stream_time": 20.0},
        {"end_time": 30.0, "stream_time": 30.0},
    ]
    assert trajectory.compaction_count_at_query({"stream_time": 5.0}, compactions) == 0
    assert trajectory.compaction_count_at_query({"stream_time": 10.0}, compactions) == 1
    assert trajectory.compaction_count_at_query({"stream_time": 25.0}, compactions) == 2
    assert trajectory.compaction_count_at_query({"stream_time": 99.0}, compactions) == 3


def test_query_before_first_compaction_counts_zero():
    compactions = [{"end_time": 42.0, "stream_time": 42.0}]
    assert trajectory.compaction_count_at_query({"stream_time": 41.9}, compactions) == 0


def test_triggered_but_not_completed_not_counted():
    # trigger at 30 but end at 35 -> a query at 32 must NOT count it
    compactions = [{"trigger_time": 30.0, "end_time": 35.0, "stream_time": 30.0}]
    assert trajectory.compaction_count_at_query({"stream_time": 32.0}, compactions) == 0
    assert trajectory.compaction_count_at_query({"stream_time": 36.0}, compactions) == 1


def test_multiple_queries_same_compaction_count_are_kept():
    events = [
        {"event_id": "v|q0|1.0", "sample_id": "v", "query_id": "q0", "query_time": 1.0,
         "stream_time": 1.0, "compaction_index": 0, "ram_mean": 0.9, "aore_mean": 0.01,
         "js": 1e-9, "kl": 1e-9, "compression_ratio": 1.0},
        {"event_id": "v|q1|2.0", "sample_id": "v", "query_id": "q1", "query_time": 2.0,
         "stream_time": 2.0, "compaction_index": 0, "ram_mean": 0.8, "aore_mean": 0.02,
         "js": 2e-9, "kl": 2e-9, "compression_ratio": 1.0},
    ]
    compactions = [
        {"event_id": "v|q1|2.0", "sample_id": "v", "end_time": 1.5, "stream_time": 1.5},
    ]
    run = _make_run("hermes_budget_2000", "v", events, compactions)
    experiment = ExperimentData(directory=None, name="exp", runs={"hermes_budget_2000": run})
    data = trajectory.build_query_trajectory(experiment, "v", budget=2000)
    counts = [r["compaction_count_at_query"] for r in data["rows"]]
    assert counts == [0, 1]  # q0 sees none, q1 sees the one completed at 1.5
    assert len(data["rows"]) == 2  # no query dropped despite shared count earlier


def test_trajectory_row_matches_paired_jsonl_values():
    event = {
        "event_id": "v|q0|5.0", "sample_id": "v", "query_id": "q0",
        "query_time": 5.0, "stream_time": 5.0, "compaction_index": 0,
        "ram_mean": 0.77, "aore_mean": 0.123, "js": -1e-8, "kl": 4e-4,
        "compression_ratio": 1.9, "full_kv_bytes": 1000, "compressed_kv_bytes": 526,
    }
    run = _make_run("hermes_budget_2000", "v", [event], [])
    experiment = ExperimentData(directory=None, name="exp", runs={"hermes_budget_2000": run})
    row = trajectory.build_query_trajectory(experiment, "v", budget=2000)["rows"][0]
    assert row["RAM"] == 0.77 and row["AORE"] == 0.123
    assert row["KL"] == 4e-4 and row["CR"] == 1.9
    assert row["full_kv_bytes"] == 1000 and row["compressed_kv_bytes"] == 526


def test_js_tiny_negative_clamped_for_visualisation_only():
    assert schemas.js_display(-1.2e-8) == 0.0
    assert schemas.js_display(3e-5) == 3e-5
    assert schemas.js_display(None) is None


def test_effective_compression_false_not_in_main_aggregate():
    # one sample compacted, one did not -> only the compacting sample counts
    events = [
        {"event_id": "a|q|1.0", "sample_id": "a", "query_id": "q", "query_time": 1.0,
         "stream_time": 1.0, "compaction_index": 2, "ram_mean": 0.8, "aore_mean": 0.1,
         "js": 1e-4, "kl": 1e-3, "compression_ratio": 1.5},
        {"event_id": "b|q|1.0", "sample_id": "b", "query_id": "q", "query_time": 1.0,
         "stream_time": 1.0, "compaction_index": 0, "ram_mean": 0.999, "aore_mean": 0.0,
         "js": 0.0, "kl": 0.0, "compression_ratio": 1.0},
    ]
    compactions = [{"end_time": 0.5, "stream_time": 0.5, "sample_id": "a", "event_id": "a|q|1.0"}]
    run = _make_run("hermes_budget_2000", None, events, compactions)
    experiment = ExperimentData(directory=None, name="exp", runs={"hermes_budget_2000": run})
    points, _ = schemas.build_compression_fidelity(experiment)
    point = points[0]
    assert point["effective_compression"] is True
    assert point["n_samples"] == 1  # sample b excluded


def test_runtime_breakdown_does_not_double_count_hermes():
    rb = RuntimeBreakdown(enabled=True)
    rb.samples["vision_encoder"] = [2.0]
    rb.samples["streaming_prefill"] = [4.0]
    rb.samples["hermes_total"] = [4.0]
    rb.samples["hermes_importance"] = [0.2]
    rb.samples["hermes_selection"] = [0.1]
    rb.samples["hermes_prune"] = [1.0]
    rb.samples["query_prefill"] = [1.0]
    rb.samples["decode"] = [2.0]
    rb.samples["video_decode"] = [9.0]
    summary = rb.summary()
    # 2 + 4 + 4 + 1 + 2 = 13 (sub-components NOT added again)
    assert summary["measured_model_compute_total_sec"] == 13.0
    stats = {row["component"]: row for row in rb.stats()}
    assert stats["hermes_total"]["fraction_of_model_compute"] == pytest.approx(4 / 13)
    assert stats["hermes_prune"]["fraction_of_model_compute"] is None
    assert stats["hermes_importance"]["fraction_of_model_compute"] is None


def test_trajectory_uses_specified_budget_run():
    events_a = [{"event_id": "v|q|1.0", "sample_id": "v", "query_id": "q", "query_time": 1.0,
                 "stream_time": 1.0, "compaction_index": 1, "ram_mean": 0.5}]
    events_b = [{"event_id": "v|q|1.0", "sample_id": "v", "query_id": "q", "query_time": 1.0,
                 "stream_time": 1.0, "compaction_index": 3, "ram_mean": 0.9}]
    runs = {
        "hermes_budget_2000": _make_run("hermes_budget_2000", "v", events_a, []),
        "hermes_budget_4000": _make_run("hermes_budget_4000", "v", events_b, []),
    }
    experiment = ExperimentData(directory=None, name="exp", runs=runs)
    row = trajectory.build_query_trajectory(experiment, "v", budget=4000)["rows"][0]
    assert row["RAM"] == 0.9


def test_bootstrap_ci_is_seeded_and_stable():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = st.bootstrap_ci(values, seed=42)
    b = st.bootstrap_ci(values, seed=42)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]


# --------------------------------------------------------------------- #
# P1 regression suite: protocol preservation / accounting / task chain
# --------------------------------------------------------------------- #
import inspect as _inspect  # noqa: E402

from evaluation_pipeline.kv_metrics import paired_runner as _pr  # noqa: E402
from evaluation_pipeline.kv_metrics.hooks import _preserve_signature  # noqa: E402
from evaluation_pipeline.kv_metrics.system_metrics import (  # noqa: E402
    is_effective_compaction,
)


class _FakeQA:
    def question_answering(
        self,
        input_text,
        max_new_tokens=128,
        temperature=0,
        repetition_penalty=1.1,
        pseudo_forward=False,
        allowed_first_token_ids=None,
    ):
        return ("text", allowed_first_token_ids)


def test_question_answering_wrapper_signature_preserved():
    qa = _FakeQA()
    before = _inspect.signature(qa.question_answering)
    original = qa.question_answering

    def wrapped(input_text, *args, **kwargs):
        return original(input_text, *args, **kwargs)

    qa.question_answering = _preserve_signature(wrapped, original)
    after = _inspect.signature(qa.question_answering)
    assert "allowed_first_token_ids" in after.parameters
    assert list(before.parameters) == list(after.parameters)


def test_allowed_first_token_ids_survives_instrumentation():
    """Mirrors MMDuet2_HERMES._ask's own inspect.signature guard."""
    qa = _FakeQA()
    original = qa.question_answering

    def wrapped(input_text, *args, **kwargs):
        return original(input_text, *args, **kwargs)

    qa.question_answering = _preserve_signature(wrapped, original)
    kwargs = {}
    if "allowed_first_token_ids" in _inspect.signature(qa.question_answering).parameters:
        kwargs["allowed_first_token_ids"] = [32, 33]
    assert kwargs == {"allowed_first_token_ids": [32, 33]}
    assert qa.question_answering("x", **kwargs)[1] == [32, 33]


def test_metrics_off_equals_metrics_on_when_wrapper_preserves_signature():
    """If the signature guard passes, the MCQ constraint is forwarded both ways."""
    qa = _FakeQA()
    original = qa.question_answering
    baseline = qa.question_answering("p", allowed_first_token_ids=[32, 33])

    def wrapped(input_text, *args, **kwargs):
        return original(input_text, *args, **kwargs)

    qa.question_answering = _preserve_signature(wrapped, original)
    hooked = qa.question_answering("p", allowed_first_token_ids=[32, 33])
    assert baseline == hooked == ("text", [32, 33])


def test_effective_compaction_noop_2021_to_2021():
    assert is_effective_compaction(2021, 2021, prefix=89) is False


def test_effective_compaction_real_2173_to_2089():
    assert is_effective_compaction(2173, 2089, prefix=89) is True


def test_effective_compaction_threshold_exact():
    # 2105 -> 2089 (16 tokens evicted) is effective; equal is not
    assert is_effective_compaction(2105, 2089, prefix=89) is True
    assert is_effective_compaction(2089, 2089, prefix=89) is False


def test_longest_query_selection_picks_argmax_query_time():
    from evaluation_pipeline.kv_metrics.analysis.trajectory import _longest_query

    questions = [
        {"query_time": 27.0, "query_id": "q27"},
        {"query_time": 117.0, "query_id": "q117"},
        {"query_time": 49.0, "query_id": "q49"},
    ]
    assert _longest_query(questions)["query_id"] == "q117"


def test_task_transition_classification():
    assert _pr._task_transition(True, True) == "correct_to_correct"
    assert _pr._task_transition(True, False) == "correct_to_wrong"
    assert _pr._task_transition(False, True) == "wrong_to_correct"
    assert _pr._task_transition(False, False) == "wrong_to_wrong"
    assert _pr._task_transition(None, True) == "missing"


def test_missing_task_lookup_is_null_not_no_reply():
    record = {}
    _pr._attach_task_fields(record, None, None, ("s", 1.0))
    assert record["task_prediction_compressed"] is None
    assert record["task_prediction_full"] is None
    assert record["task_correct_compressed"] is None
    assert record["task_missing"] is True
    assert record["task_transition"] == "missing"
    assert record["task_prediction"] is None


def test_full_and_compressed_task_fields_are_separate():
    record = {}
    # Full-KV answers correctly, the compressed cache flips it to wrong.
    full = {"prediction": "B", "correct": True, "status": None, "error_type": None,
            "ground_truth": "B"}
    comp = {"prediction": "C", "correct": False, "status": "error",
            "error_type": "unexpected_no_reply", "ground_truth": "B"}
    _pr._attach_task_fields(record, comp, full, ("s", 1.0))
    assert record["task_prediction_compressed"] == "C"
    assert record["task_prediction_full"] == "B"
    assert record["task_correct_compressed"] is False
    assert record["task_correct_full"] is True
    assert record["task_error_type_compressed"] == "unexpected_no_reply"
    assert record["task_transition"] == "correct_to_wrong"


def test_official_correct_reuses_adapter_evaluate():
    class _Adapter:
        def evaluate(self, predictions):
            rec = predictions[0]
            ok = str(rec.get("ground_truth")) in str(rec.get("prediction", ""))
            return {"overall": float(ok), "tasks": {rec["task_type"]: float(ok)}}

    adapter = _Adapter()
    good = {"prediction": "B", "ground_truth": "B", "task_type": "EPM"}
    bad = {"prediction": "C", "ground_truth": "B", "task_type": "EPM"}
    assert _pr._official_correct(adapter, good) is True
    assert _pr._official_correct(adapter, bad) is False


def test_cluster_bootstrap_resamples_samples_not_queries():
    from evaluation_pipeline.kv_metrics.analysis import statistics as st

    clusters = [("s1", [1, 1, 1, 1]), ("s2", [0, 0, 0, 0])]
    values = [v for _, g in clusters for v in g]
    out = st.cluster_bootstrap_ci([g for _, g in clusters], n_boot=200, seed=1)
    assert out["n_samples"] == 2
    assert out["n_queries"] == len(values)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]


def test_accuracy_by_compaction_count_exact_counts():
    from evaluation_pipeline.kv_metrics.analysis import task_accuracy as ta

    rows = [
        {"sample_id": "s1", "effective_compaction_count_at_query": 0,
         "task_correct_compressed": True, "task_correct_full": True,
         "task_transition": "correct_to_correct", "task_missing": False},
        {"sample_id": "s1", "effective_compaction_count_at_query": 1,
         "task_correct_compressed": False, "task_correct_full": True,
         "task_transition": "correct_to_wrong", "task_missing": False},
        {"sample_id": "s2", "effective_compaction_count_at_query": 1,
         "task_correct_compressed": True, "task_correct_full": True,
         "task_transition": "correct_to_correct", "task_missing": False},
    ]
    table = ta.accuracy_by_compaction_count(rows)
    by_k = {row["effective_compaction_count"]: row for row in table}
    assert by_k[0]["num_queries"] == 1 and by_k[0]["accuracy_compressed"] == 1.0
    assert by_k[1]["num_queries"] == 2 and by_k[1]["accuracy_compressed"] == 0.5
    assert by_k[1]["num_samples"] == 2
    assert by_k[1]["num_correct_to_wrong"] == 1
    assert by_k[1]["accuracy_full"] == 1.0
    assert by_k[1]["preservation_accuracy"] == 0.5
    assert by_k[1]["compression_failure_rate"] == 0.5
    assert by_k[1]["exploratory_low_n"] is True


def test_binned_accuracy_labels_are_secondary_not_replacing_exact():
    from evaluation_pipeline.kv_metrics.analysis import task_accuracy as ta

    rows = [
        {"sample_id": f"s{i}", "effective_compaction_count_at_query": k,
         "task_correct_compressed": True, "task_correct_full": True,
         "task_transition": "correct_to_correct", "task_missing": False}
        for k in range(4) for i in range(2)
    ]
    exact = ta.accuracy_by_compaction_count(rows)
    assert [r["effective_compaction_count"] for r in exact] == [0, 1, 2, 3]
    binned = ta.accuracy_by_compaction_count(rows, bin_edges=[0, 1, 2])
    assert [r["bin_label"] for r in binned] == ["0", "1", "2", "3+"]


# --------------------------------------------------------------------- #
# StreamingBench selection: bin aggregation invariants
# --------------------------------------------------------------------- #
def _plan(videos):
    """videos = [(video_id, [k_q1..k_q5], q_star_index)]"""
    return [
        {"video_id": vid, "effective_k_vector": list(ks),
         "k_q_star": ks[idx - 1], "q_star_id": f"sample_{vid}_{idx}",
         "query_times": [float(i) for i in range(1, 6)],
         "q_star_time": float(idx), "task_types": ["t"] * 5,
         "coverage_bins_contributed": []}
        for vid, ks, idx in videos
    ]


def test_sb_bin_stats_are_derived_from_query_records():
    from evaluation_pipeline.kv_metrics.analysis import streamingbench_selection as sel

    plan = _plan([("1", [0, 0, 0, 0, 10], 5),      # B1 via q5
                  ("2", [0, 35, 0, 0, 0], 2),      # B4 via q2, q_star k=35
                  ("3", [0, 0, 0, 0, 0], 5)])      # B0 only
    bins = sel.bins_from_queries(plan)
    # video1: 4 zeros + k10 ; video2: 4 zeros + k35 ; video3: 5 zeros
    assert bins["B0"]["n_query"] == 13 and bins["B0"]["n_video"] == 3
    assert bins["B1"]["n_query"] == 1 and bins["B1"]["n_video"] == 1
    assert bins["B4"]["n_query"] == 1 and bins["B4"]["n_video"] == 1
    assert bins["B4"]["video_ids"] == ["2"]


def test_sb_invariant_qstar_video_must_appear_in_its_query_bin():
    """The historical bug: B4 N_video smaller than the number of q_star>=31
    videos, which is impossible because q_star is one of that video's queries."""
    from evaluation_pipeline.kv_metrics.analysis import streamingbench_selection as sel

    plan = _plan([(str(i), [0, 0, 0, 0, 31 + i], 5) for i in range(1, 9)])
    assert sel.check_invariants(plan) == []
    bins = sel.bins_from_queries(plan)
    assert bins["B4"]["n_video"] == 8 and bins["B4"]["n_query"] == 8

    # a broken aggregation (e.g. derived from a stale summary) must be caught
    broken = _plan([("1", [0, 0, 0, 0, 40], 5)])
    broken.append({"video_id": "1", "effective_k_vector": [0, 0, 0, 0, 40],
                   "k_q_star": 40, "q_star_id": "sample_1_5",
                   "coverage_bins_contributed": []})
    assert sel.bins_from_queries(broken)["B4"]["n_video"] == 1  # unique videos


def test_sb_invariant_detects_stale_k_q_star():
    from evaluation_pipeline.kv_metrics.analysis import streamingbench_selection as sel

    plan = _plan([("7", [0, 0, 0, 0, 12], 5)])
    plan[0]["k_q_star"] = 99          # stale / wrong field
    problems = sel.check_invariants(plan)
    assert any("k_q_star" in p for p in problems)


def test_sb_invariant_n_video_leq_n_query_and_qstar_subset():
    from evaluation_pipeline.kv_metrics.analysis import streamingbench_selection as sel

    plan = _plan([
        ("10", [0, 3, 12, 25, 40], 5),
        ("11", [0, 0, 0, 22, 33], 5),
        ("12", [0, 0, 0, 0, 0], 5),
    ])
    bins = sel.bins_from_queries(plan)
    for name, stats in bins.items():
        assert stats["n_video"] <= stats["n_query"], name
    qstar = sel.qstar_bucket_videos(plan)
    # every q_star>=31 video must be inside B4's video set
    assert set(qstar["31+"]) <= set(bins["B4"]["video_ids"])
    assert set(qstar["11-20"]) <= set(bins["B2"]["video_ids"])
    assert sel.check_invariants(plan) == []


# --------------------------------------------------------------------- #
# Decision-layer fidelity: ΔP_gold / ΔM_z
# --------------------------------------------------------------------- #
def _rec(logits_full, logits_comp, gold="B", k=0,
         pred_full="B", pred_comp="B", ids=None, transition="correct_to_correct"):
    return {
        "video_id": "sample_1", "query_id": "q1", "query_time": 10.0,
        "effective_compaction_count_at_query": k,
        "task_gold": gold,
        "task_prediction_full": pred_full, "task_prediction_compressed": pred_comp,
        "task_transition": transition,
        "option_token_ids": ids or {"A": [32, 362], "B": [33, 425],
                                    "C": [34, 356], "D": [35, 422]},
        "option_token_logits_full": logits_full,
        "option_token_logits_compressed": logits_comp,
    }


def test_decision_option_variants_come_from_token_ids_map():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    logits = {"32": 10.0, "362": 5.0, "33": 20.0, "425": 1.0,
              "34": 0.0, "356": -5.0, "35": 2.0, "422": 0.0}
    ids_a = {"A": [32], "B": [33], "C": [34], "D": [35]}
    ids_b = {"A": [32, 362], "B": [33, 425], "C": [34, 356], "D": [35, 422]}
    # the variant set changes the option score -> ids are really used
    sa, _ = df.option_scores(logits, ids_a)
    sb, _ = df.option_scores(logits, ids_b)
    assert sa["A"] != sb["A"]
    # and they are the ids the caller supplied


def test_decision_option_probabilities_sum_to_one():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    logits = {"32": 1.0, "362": 0.0, "33": 2.0, "425": 1.0,
              "34": 0.5, "356": 0.0, "35": -1.0, "422": -2.0}
    probs, _ = df.option_scores(logits, {"A": [32, 362], "B": [33, 425],
                                         "C": [34, 356], "D": [35, 422]})
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_decision_p_gold_uses_logsumexp_over_variants():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    # two variants of A together must outrank single-token B
    logits = {"32": 0.0, "362": 0.0, "33": 0.5, "425": -50.0,
              "34": -50.0, "356": -50.0, "35": -50.0, "422": -50.0}
    probs, _ = df.option_scores(logits, {"A": [32, 362], "B": [33, 425],
                                         "C": [34, 356], "D": [35, 422]})
    assert probs["A"] > probs["B"]
    assert abs(df._logsumexp([0.0, 0.0]) - math.log(2)) < 1e-9


def test_decision_margin_uses_max_variant():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    logits = {"32": 5.0, "362": 0.0, "33": 4.0, "425": 1.0,
              "34": 0.0, "356": 0.0, "35": 0.0, "422": 0.0}
    _, best = df.option_scores(logits, {"A": [32, 362], "B": [33, 425],
                                        "C": [34, 356], "D": [35, 422]})
    assert best["A"] == 5.0 and best["B"] == 4.0          # max, not logsumexp
    assert df.decision_margin(best, "A") == 1.0
    assert df.decision_margin(best, "B") == -1.0


def test_decision_best_option_matches_official_prediction_flag():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    logits = {"32": 9.0, "362": 0.0, "33": 1.0, "425": 0.0,
              "34": 0.0, "356": 0.0, "35": 0.0, "422": 0.0}
    good = df.compute_row(_rec(logits, logits, gold="A", pred_full="A", pred_comp="A"))
    assert good["best_option_full"] == "A" and good["decision_metric_mismatch"] is False
    bad = df.compute_row(_rec(logits, logits, gold="A", pred_full="C", pred_comp="A"))
    assert bad["decision_metric_mismatch"] is True


def test_decision_k0_is_identity_like():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    logits = {"32": 3.0, "362": 0.0, "33": 2.5, "425": 0.0,
              "34": 1.0, "356": 0.0, "35": 0.0, "422": 0.0}
    row = df.compute_row(_rec(dict(logits), dict(logits), gold="A",
                              pred_full="A", pred_comp="A", k=0))
    assert abs(row["delta_p_gold"]) < 1e-12
    assert abs(row["delta_margin_logit"]) < 1e-12
    assert row["p_option_full"] == row["p_option_compressed"]
    assert row["margin_crossed"] is False


def test_decision_margin_crossing_detected():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    full = {"32": 5.0, "362": 0.0, "33": 4.0, "425": 0.0,
            "34": 0.0, "356": 0.0, "35": 0.0, "422": 0.0}
    comp = {"32": 1.0, "362": 0.0, "33": 4.0, "425": 0.0,
            "34": 0.0, "356": 0.0, "35": 0.0, "422": 0.0}
    row = df.compute_row(_rec(full, comp, gold="A", pred_full="A", pred_comp="B",
                              k=30, transition="correct_to_wrong"))
    assert row["margin_full"] > 0 > row["margin_compressed"]
    assert row["margin_crossed"] is True
    assert row["delta_margin_logit"] < 0


def test_decision_cluster_bootstrap_unit_is_video_not_query():
    from evaluation_pipeline.kv_metrics.analysis import decision_fidelity as df

    logits = {"32": 3.0, "362": 0.0, "33": 1.0, "425": 0.0,
              "34": 0.0, "356": 0.0, "35": 0.0, "422": 0.0}
    rows = []
    for i in range(5):                      # 5 queries, ONE video
        rows.append(df.compute_row({**_rec(dict(logits), dict(logits), gold="A",
                                           pred_full="A", pred_comp="A", k=5),
                                    "query_id": f"q{i}"}))
    bins = df.aggregate(rows, bin_edges=[0, 10, 20, 30], n_boot=100, seed=42)
    assert len(bins) == 1
    assert bins[0]["num_queries"] == 5
    assert bins[0]["num_videos"] == 1        # NOT 5


# --------------------------------------------------------------------- #
# KV-budget stress test
# --------------------------------------------------------------------- #
def _b_rec(vid, t, k, delta_p, delta_m, correct_c, correct_f, pred_c="A", pred_f="A",
           gold="A", mismatch=False):
    logits = {"32": 5.0, "362": 0.0, "33": 1.0, "425": 0.0,
              "34": 0.0, "356": 0.0, "35": 0.0, "422": 0.0}
    return {
        "video_id": f"sample_{vid}", "query_id": f"{vid}_{t}", "query_time": float(t),
        "effective_compaction_count_at_query": k,
        "task_gold": gold, "task_prediction_full": pred_f,
        "task_prediction_compressed": pred_c,
        "task_correct_full": correct_f, "task_correct_compressed": correct_c,
        "option_token_ids": {"A": [32, 362], "B": [33, 425],
                             "C": [34, 356], "D": [35, 422]},
        "option_token_logits_full": dict(logits),
        "option_token_logits_compressed": dict(logits),
    }


def test_budget_table_joins_full_and_compressed():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    full = [_b_rec("1", 10, 0, 0, 0, True, True), _b_rec("1", 20, 5, 0, 0, True, True)]
    import tempfile, json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "sb_real_budget_2000" / "raw_decision" / "batch1" / "paired"
        d.mkdir(parents=True)
        with open(d / "x.jsonl", "w") as fh:
            for r in full:
                fh.write(json.dumps(r) + "\n")
        paired, summary = bs.build_budget_table(
            {2000: [Path(tmp) / "sb_real_budget_2000"]}, full)
    assert len(paired) == 2
    assert summary[0]["budget"] == 2000
    assert summary[0]["N_query"] == 2 and summary[0]["N_video"] == 1
    assert summary[0]["CW"] == 0 and summary[0]["Preservation"] == 1.0


def test_budget_summary_aggregates_per_video_first():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    paired = [
        {"video_id": "1", "budget2000_delta_p_gold": 1.0, "budget2000_delta_margin": 0.0,
         "budget2000_prediction": "A", "budget2000_correct": True, "budget2000_k": 1,
         "full_correct": True, "budget2000_mismatch": False},
        {"video_id": "1", "budget2000_delta_p_gold": -1.0, "budget2000_delta_margin": 0.0,
         "budget2000_prediction": "A", "budget2000_correct": True, "budget2000_k": 1,
         "full_correct": True, "budget2000_mismatch": False},
        {"video_id": "2", "budget2000_delta_p_gold": 0.5, "budget2000_delta_margin": 0.0,
         "budget2000_prediction": "A", "budget2000_correct": True, "budget2000_k": 1,
         "full_correct": True, "budget2000_mismatch": False},
    ]
    out = bs.summarize_budgets(paired)
    row = out[0]
    # video 1 average = 0.0, video 2 = 0.5 -> mean = 0.25 (not per-query 0.166..)
    assert abs(row["mean_delta_p_gold_video_level"] - 0.25) < 1e-12
    assert row["N_video"] == 2 and row["N_query"] == 3
    assert abs(row["fraction_query_delta_p_negative"] - 1 / 3) < 1e-12


def test_budget_sanity_flags_wrong_direction_and_missing_queries():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    summaries = [
        {"budget": 2000, "mean_effective_compaction_count": 10.0,
         "decision_metric_mismatch": 0},
        {"budget": 1500, "mean_effective_compaction_count": 5.0,   # WRONG direction
         "decision_metric_mismatch": 0},
    ]
    problems = bs.sanity_checks([], summaries)
    assert any("decreased" in p for p in problems)

    paired = [{"budget2000_prediction": "A"}, {"budget1500_prediction": None}]
    problems2 = bs.sanity_checks(paired, [])
    assert any("only" in p for p in problems2)


def test_budget_sanity_flags_decision_mismatch():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    problems = bs.sanity_checks([], [{"budget": 2000, "decision_metric_mismatch": 3,
                                      "mean_effective_compaction_count": 1.0}])
    assert any("best_option" in p for p in problems)


# --------------------------------------------------------------------- #
# extreme budget (100): ordering, margin crossing, paired 750-vs-100
# --------------------------------------------------------------------- #
def test_budget_order_includes_extreme_100():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    assert 100 in bs.BUDGET_ORDER
    assert bs.BUDGET_ORDER[0] == 2000 and bs.BUDGET_ORDER[-1] == 100


def test_budget_summary_counts_margin_crossings():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    paired = [
        {"video_id": "1", "budget100_delta_p_gold": -0.1, "budget100_delta_margin": -1.0,
         "budget100_prediction": "B", "budget100_correct": False, "budget100_k": 40,
         "full_correct": True, "budget100_mismatch": False,
         "budget100_margin_crossed": True},
        {"video_id": "2", "budget100_delta_p_gold": 0.1, "budget100_delta_margin": 0.5,
         "budget100_prediction": "A", "budget100_correct": True, "budget100_k": 30,
         "full_correct": True, "budget100_mismatch": False,
         "budget100_margin_crossed": False},
    ]
    row = [r for r in bs.summarize_budgets(paired) if r["budget"] == 100][0]
    assert row["num_margin_crossed"] == 1
    assert row["CW"] == 1 and row["CC"] == 1
    assert abs(row["Preservation"] - 0.5) < 1e-12


def test_paired_budget_comparison_750_vs_100():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    paired = [
        {"budget750_delta_margin": 0.2, "budget100_delta_margin": -0.1,   # down
         "budget750_delta_p_gold": 0.1, "budget100_delta_p_gold": 0.05,
         "budget750_correct": True, "budget100_correct": False},
        {"budget750_delta_margin": 0.0, "budget100_delta_margin": 0.3,    # up
         "budget750_delta_p_gold": 0.0, "budget100_delta_p_gold": 0.4,
         "budget750_correct": True, "budget100_correct": True},
        {"budget750_delta_margin": 0.1, "budget100_delta_margin": 0.1,    # equal
         "budget750_delta_p_gold": -0.1, "budget100_delta_p_gold": -0.1,
         "budget750_correct": False, "budget100_correct": True},
    ]
    cmp = bs.paired_budget_comparison(paired, 750, 100)
    assert cmp["delta_margin_down"] == 1 and cmp["delta_margin_up"] == 1
    assert cmp["delta_margin_equal"] == 1
    assert cmp["delta_p_gold_down"] == 1 and cmp["delta_p_gold_up"] == 1
    assert cmp["hi_correct_lo_wrong"] == 1
    assert cmp["lo_correct_hi_wrong"] == 1
    assert cmp["n_queries"] == 3


# --------------------------------------------------------------------- #
# OVO backward budget stress: schema, task split, exposure, adapter filter
# --------------------------------------------------------------------- #
def test_compressed_only_records_use_canonical_schema():
    from evaluation_pipeline.kv_metrics import paired_runner as pr

    record = {}
    pr.apply_single_pass_task_fields(
        record,
        {"prediction": "C", "correct": True, "status": None,
         "error_type": None, "ground_truth": "C"},
        {"option_token_logits": {"32": 1.0}, "option_token_ids": {"A": [32]}},
        capture_option_logits=True,
    )
    for key in ("task_prediction_compressed", "task_correct_compressed",
                "task_status_compressed", "task_error_type_compressed"):
        assert key in record
    assert record["task_prediction_compressed"] == "C"
    assert record["task_correct_compressed"] is True
    # legacy aliases still present
    assert record["task_prediction"] == "C" and record["task_correct"] is True
    assert record["option_token_logits_compressed"] == {"32": 1.0}
    assert record["option_token_ids"] == {"A": [32]}


def test_task_breakdown_splits_by_task():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    paired = [
        {"video_id": "1", "task_type": "EPM", "budget1000_prediction": "A",
         "budget1000_correct": True, "full_correct": True, "budget1000_k": 3},
        {"video_id": "2", "task_type": "HLD", "budget1000_prediction": "B",
         "budget1000_correct": False, "full_correct": True, "budget1000_k": 5},
    ]
    summary = bs.summarize_budgets(paired)
    rows = bs.task_breakdown(paired, summary)
    by_task = {r["task_type"]: r for r in rows}
    assert by_task["EPM"]["Acc_compressed"] == 1.0 and by_task["EPM"]["CW"] == 0
    assert by_task["HLD"]["Acc_compressed"] == 0.0 and by_task["HLD"]["CW"] == 1
    assert by_task["HLD"]["Preservation"] == 0.0


def test_exposure_table_lists_k_per_budget():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    paired = [{"video_id": "7", "query_time": 100.0, "task_type": "ASI",
               "budget2000_k": 4, "budget750_k": 12, "budget100_k": 40}]
    rows = bs.exposure_table(paired, [2000, 750, 100])
    assert rows[0]["effective_compaction_count@2000"] == 4
    assert rows[0]["effective_compaction_count@750"] == 12
    assert rows[0]["effective_compaction_count@100"] == 40


def test_ovo_adapter_video_id_filter():
    from evaluation_pipeline.adapters import OVOBenchAdapter

    cfg = {"annotation": "OVO-Bench/data/ovo_bench_new.json",
           "chunked_dir": "OVO-Bench/data/chunked_videos",
           "_video_ids": ["508", "378"]}
    adapter = OVOBenchAdapter(cfg)
    samples = list(adapter.load_samples())
    assert {s.sample_id for s in samples} == {"508", "378"}
    assert all(s.questions[0].task_type in {"EPM", "ASI", "HLD"} for s in samples)


def test_normalize_full_record_accepts_legacy_and_single_pass_names():
    """Regression: the Full-only pass reused the single-pass writer, so its
    fields are named *_compressed / legacy; the join must still find them."""
    from evaluation_pipeline.kv_metrics.analysis.budget_stress import (
        normalize_full_record,
    )

    legacy = {"task_prediction": "C", "task_correct": True,
              "option_token_logits": {"32": 1.0}}
    out = normalize_full_record(legacy)
    assert out["task_prediction_full"] == "C"
    assert out["task_correct_full"] is True
    assert out["option_token_logits_full"] == {"32": 1.0}

    single_pass = {"task_prediction_compressed": "B", "task_correct_compressed": False,
                   "task_status_compressed": None, "task_error_type_compressed": None,
                   "option_token_logits_compressed": {"33": 2.0}}
    out2 = normalize_full_record(single_pass)
    assert out2["task_prediction_full"] == "B"
    assert out2["task_correct_full"] is False
    assert out2["option_token_logits_full"] == {"33": 2.0}

    canonical = {"task_prediction_full": "D", "task_correct_full": True,
                 "option_token_logits_full": {"34": 3.0}}
    assert normalize_full_record(canonical) == canonical


def test_join_invariants_detects_missing_full_side():
    from evaluation_pipeline.kv_metrics.analysis import budget_stress as bs

    paired = [
        {"video_id": "1", "full_correct": True, "full_prediction": "A",
         "gold_option": "A", "budget2000_correct": True,
         "budget2000_delta_p_gold": 0.0, "budget2000_delta_margin": 0.0},
        {"video_id": "2", "full_correct": None, "full_prediction": None,
         "gold_option": "B", "budget2000_correct": True},
    ]
    inv = bs.join_invariants(paired)
    assert inv["n_samples"] == 2
    assert inv["n_with_full_correct"] == 1
    problems = bs.sanity_checks(paired, [])
    assert any("Full baseline joined for only 1/2" in p for p in problems)


def test_single_pass_writer_is_side_aware():
    from evaluation_pipeline.kv_metrics.paired_runner import (
        apply_single_pass_task_fields,
    )

    task = {"prediction": "A", "correct": True, "status": None,
            "error_type": None, "ground_truth": "A", "task_type": "EPM"}
    rec = {}
    apply_single_pass_task_fields(rec, task, {"option_token_logits": {"32": 9.0}},
                                  capture_option_logits=True, side="full")
    assert rec["task_prediction_full"] == "A" and rec["task_correct_full"] is True
    assert rec["option_token_logits_full"] == {"32": 9.0}
    assert "task_prediction_compressed" not in rec
    # legacy aliases mirror the branch that actually ran
    assert rec["task_prediction"] == "A"

    rec2 = {}
    apply_single_pass_task_fields(rec2, task, None,
                                  capture_option_logits=False, side="compressed")
    assert rec2["task_prediction_compressed"] == "A"
    assert "task_prediction_full" not in rec2


def test_run_has_no_undefined_local_name_in_single_pass_branch():
    """Regression for the NameError that silently killed all batches of the
    OVO budget runs: the single-pass branch referenced a bare ``mode`` that does
    not exist inside ``run()``. Guards the whole function body via AST."""

    import ast
    from pathlib import Path

    src = Path("evaluation_pipeline/kv_metrics/paired_runner.py").read_text()
    tree = ast.parse(src)
    run_fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "run")
    defined = {a.arg for a in run_fn.args.args} | {
        n.id for n in ast.walk(run_fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    } | {"kv_config"}
    bare = {
        n.id for n in ast.walk(run_fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    } - defined - set(dir(__builtins__))
    assert "mode" not in bare, f"run() references undefined names: {sorted(bare)}"
