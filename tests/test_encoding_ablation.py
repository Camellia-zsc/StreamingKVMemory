"""CPU tests for the image-vs-video encoding ablation (no GPU)."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

from evaluation_pipeline.ablation import encoding_analyze as ea
from evaluation_pipeline.ablation import encoding_common as ec


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OVO-Bench"))
sys.path.insert(0, str(ROOT / "HERMES"))


# --------------------------------------------------------------------- #
# encoding switch
# --------------------------------------------------------------------- #
class _FakeQAModel:
    def __init__(self, has_image=True, has_video=True):
        self.calls = []
        self._has_image = has_image
        self._has_video = has_video

    def _rec(self, name):
        self.calls.append(name)

    def __getattr__(self, item):  # pragma: no cover - attribute plumbing
        if item == "encode_image_chunk" and self._has_image:
            return lambda chunk: self._rec("image")
        if item == "encode_video_chunk" and self._has_video:
            return lambda chunk: self._rec("video")
        raise AttributeError(item)


def _driver(qa):
    module = pytest.importorskip("models.MMDuet2_HERMES")
    obj = module.EvalMMDuet2Hermes.__new__(module.EvalMMDuet2Hermes)
    obj.qa_model = qa
    obj.kv_size = 10 ** 12
    obj.compression_stats = {"events": [], "peak_kv_tokens": 0}
    obj._cache_lengths = lambda: [14, 14]
    return obj


def _frames():
    from PIL import Image

    return [Image.new("RGB", (8, 8)) for _ in range(2)]


def _run_with(env_value, has_image=True):
    qa = _FakeQAModel(has_image=has_image)
    obj = _driver(qa)
    old = os.environ.get("HERMES_ENCODE_STYLE")
    try:
        if env_value is None:
            os.environ.pop("HERMES_ENCODE_STYLE", None)
        else:
            os.environ["HERMES_ENCODE_STYLE"] = env_value
        obj._encode_frames(_frames())
    finally:
        if old is None:
            os.environ.pop("HERMES_ENCODE_STYLE", None)
        else:
            os.environ["HERMES_ENCODE_STYLE"] = old
    return qa.calls, obj.last_encode_style


def test_encode_style_default_prefers_image():
    calls, style = _run_with(None)
    assert calls == ["image"]
    assert style == "image"


def test_encode_style_default_falls_back_to_video():
    calls, style = _run_with(None, has_image=False)
    assert calls == ["video"]
    assert style == "video"


def test_encode_style_explicit_video():
    calls, style = _run_with("video")
    assert calls == ["video"] and style == "video"


def test_encode_style_explicit_image():
    calls, style = _run_with("image")
    assert calls == ["image"] and style == "image"


def test_encode_style_invalid_rejected():
    with pytest.raises(ValueError):
        _run_with("frames")


def test_encode_style_image_without_entrypoint_fails():
    with pytest.raises(RuntimeError):
        _run_with("image", has_image=False)


def test_fa2_backend_forced_for_encoding_ablation():
    ec.apply_encoding_env("video")
    assert os.environ["HERMES_ABLATION_BACKEND"] == "fa2"
    assert os.environ["HERMES_ENCODE_STYLE"] == "video"
    assert os.environ["HERMES_DUMP_KEEP_PROVENANCE"] == "1"
    ec.apply_encoding_env("image")
    assert os.environ["HERMES_ENCODE_STYLE"] == "image"


# --------------------------------------------------------------------- #
# temporal bins / metrics
# --------------------------------------------------------------------- #
def test_time_bin_mapping_is_fixed_width():
    assert ec.TIME_BIN_SEC == 4.0
    assert int(0.0 // ec.TIME_BIN_SEC) == 0
    assert int(3.99 // ec.TIME_BIN_SEC) == 0
    assert int(4.0 // ec.TIME_BIN_SEC) == 1
    assert int(135.9 // ec.TIME_BIN_SEC) == 33


def test_temporal_jaccard_and_empty_bins():
    j, inter, union = ea._jacc({1, 2, 3}, {2, 3, 4})
    assert j == pytest.approx(0.5) and inter == 2 and union == 4
    j, inter, union = ea._jacc(set(), set())
    assert j is None and union == 0


def test_layer_taxonomy_matches_hermes():
    assert ea._lt(0, 28) == "short" and ea._lt(1, 28) == "short"
    assert ea._lt(2, 28) == "mid" and ea._lt(18, 28) == "mid"
    assert ea._lt(19, 28) == "long" and ea._lt(27, 28) == "long"


def test_coverage_and_ages_from_dump(tmp_path, monkeypatch):
    dump = {
        "video_id": "300", "compaction_index": 1, "stream_time": 100.0,
        "observed_bins": [0, 1, 2, 3],
        "retained_seconds_by_layer": {"0": [2, 3], "1": [0, 1, 2, 3]},
        "retained_age_stats_by_layer": {
            "0": {"oldest_ts": 80.0, "mean_ts": 88.0, "p50": 88.0, "n": 10},
            "1": {"oldest_ts": 20.0, "mean_ts": 60.0, "p50": 60.0, "n": 10},
        },
    }
    monkeypatch.setattr(ea, "arm_dir", lambda s: tmp_path / f"enc_{s}")
    for s in ("image", "video"):
        d = tmp_path / f"enc_{s}" / "raw" / "main"
        d.mkdir(parents=True)
        (d / "compaction_uids.jsonl").write_text(json.dumps(dump), encoding="utf-8")
    rows = ea.temporal_retention(tmp_path, {s: ea.load(s) for s in ec.STYLES})
    r0 = [r for r in rows if r["arm"] == "image" and r["layer_idx"] == 0][0]
    assert r0["temporal_coverage_ratio"] == pytest.approx(0.5)
    assert r0["oldest_retained_age"] == pytest.approx(20.0)
    assert r0["median_retained_age"] == pytest.approx(12.0)


def test_stream_time_alignment_requires_one_chunk_tolerance(tmp_path, monkeypatch):
    def dump(t, k, bins):
        return {"video_id": "3", "compaction_index": k, "stream_time": t,
                "observed_bins": [0, 1, 2], "retained_seconds_by_layer": {"0": bins},
                "retained_age_stats_by_layer": {"0": {"oldest_ts": t - 8, "mean_ts": t - 4,
                                                      "p50": t - 4, "n": 5}}}

    imgs = [dump(40.0, 1, [0, 1]), dump(80.0, 2, [1, 2])]
    vids = [dump(42.0, 1, [1, 2]), dump(120.0, 2, [0, 2])]
    monkeypatch.setattr(ea, "arm_dir", lambda s: tmp_path / f"enc_{s}")
    for s, rows in (("image", imgs), ("video", vids)):
        d = tmp_path / f"enc_{s}" / "raw" / "main"
        d.mkdir(parents=True)
        (d / "compaction_uids.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ov = ea.temporal_overlap(tmp_path, {s: ea.load(s) for s in ec.STYLES})
    # only the (40, 42) pair is within one 4 s chunk; 80 vs 120 is not
    assert len(ov) == 1
    assert ov[0]["time_image"] == 40.0 and ov[0]["time_video"] == 42.0
    assert ov[0]["temporal_bin_jaccard"] == pytest.approx(1 / 3)


def test_input_consistency_detects_frame_mismatch(tmp_path, monkeypatch):
    base = {"sample_id": "298", "n_frames": 2, "frame_indices": [4, 6],
            "frame_times": [8.0, 12.0], "start_sec": 8.0, "end_sec": 12.0,
            "visual_tokens": 100, "chunk_index": 0}
    monkeypatch.setattr(ea, "arm_dir", lambda s: tmp_path / f"enc_{s}")
    for s, rec in (("image", base), ("video", dict(base, frame_indices=[4, 7]))):
        d = tmp_path / f"enc_{s}" / "raw" / "main"
        d.mkdir(parents=True)
        (d / "chunk_consistency.jsonl").write_text(json.dumps(rec), encoding="utf-8")
    rows, valid = ea.input_consistency(tmp_path, {s: ea.load(s) for s in ec.STYLES})
    assert rows[0]["consistent"] is False
    assert valid["298"] is False


def test_input_consistency_allows_different_token_counts(tmp_path, monkeypatch):
    base = {"sample_id": "298", "n_frames": 2, "frame_indices": [4, 6],
            "frame_times": [8.0, 12.0], "chunk_index": 0, "visual_tokens": 100}
    monkeypatch.setattr(ea, "arm_dir", lambda s: tmp_path / f"enc_{s}")
    for s, vt in (("image", 100), ("video", 250)):
        d = tmp_path / f"enc_{s}" / "raw" / "main"
        d.mkdir(parents=True)
        (d / "chunk_consistency.jsonl").write_text(
            json.dumps(dict(base, visual_tokens=vt)), encoding="utf-8")
    rows, valid = ea.input_consistency(tmp_path, {s: ea.load(s) for s in ec.STYLES})
    assert rows[0]["consistent"] is True and valid["298"] is True


def test_transition_direction_uses_video_as_reference():
    data = {s: {"paired": []} for s in ec.STYLES}
    data["video"]["paired"] = [{"sample_id": "3", "task_gold": "A", "task_correct_compressed": True,
                                "task_prediction_compressed": "A", "option_token_ids": {"A": [32], "B": [33]},
                                "option_token_logits_compressed": {"32": 5.0, "33": 1.0}}]
    data["image"]["paired"] = [{"sample_id": "3", "task_gold": "A", "task_correct_compressed": False,
                                "task_prediction_compressed": "B", "option_token_ids": {"A": [32], "B": [33]},
                                "option_token_logits_compressed": {"32": 1.0, "33": 5.0}}]
    row = ea.decision_rows(data)[0]
    assert row["transition_CW"] is True          # video correct, image wrong
    assert row["dP_gold"] < 0                    # image minus video
    assert row["dM_z"] < 0


def test_predeclared_thresholds_are_frozen():
    assert ec.TH_REP_JS == 1e-4 and ec.TH_REP_TOP10 == 0.9 and ec.TH_REP_TOKEN_RATE == 0.10
    assert ec.TH_RET_JACCARD == 0.90 and ec.TH_RET_COVERAGE == 0.05 and ec.TH_RET_OLDEST_SEC == 10.0
    assert ec.TH_DIV_JACCARD == 0.80 and ec.TH_DIV_COVERAGE == 0.10
    assert ec.TH_DIV_OLDEST_SEC == 30.0 and ec.TH_DIV_K_RATIO == 1.5
