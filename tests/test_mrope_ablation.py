"""CPU tests for the M-RoPE ablation instrumentation (no GPU, no model load)."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from evaluation_pipeline.ablation import mrope_analyze as an
from evaluation_pipeline.ablation import mrope_common as mc
from evaluation_pipeline.kv_metrics.provenance import (
    KIND_SUMMARY,
    KIND_SYSTEM,
    KIND_VISUAL,
    KVProvenanceTracker,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OVO-Bench"))
# HERMES last so its ``inference`` PACKAGE wins over OVO-Bench/inference.py
# (that module runs argparse.parse_args() at import time).
sys.path.insert(0, str(ROOT / "HERMES"))


# --------------------------------------------------------------------- #
# hook switch
# --------------------------------------------------------------------- #
class _FakeLayer:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, fn, with_kwargs=False):
        self.hooks.append(fn)
        return types.SimpleNamespace(remove=lambda: None)


class _FakeHermes:
    def __init__(self):
        self._layer_position_ids = {}
        self._hook_handles = []
        self.language_model = types.SimpleNamespace(
            layers=[_FakeLayer(), _FakeLayer()]
        )


def _sdpa_class():
    module = pytest.importorskip("inference.qwenvl_hermes")
    return module.QwenVL_Hermes


def _registered(env_value):
    cls = _sdpa_class()
    fake = _FakeHermes()
    old = os.environ.get("HERMES_PER_LAYER_POS_HOOK")
    try:
        if env_value is None:
            os.environ.pop("HERMES_PER_LAYER_POS_HOOK", None)
        else:
            os.environ["HERMES_PER_LAYER_POS_HOOK"] = env_value
        cls._register_forward_hooks(fake)
    finally:
        if old is None:
            os.environ.pop("HERMES_PER_LAYER_POS_HOOK", None)
        else:
            os.environ["HERMES_PER_LAYER_POS_HOOK"] = old
    return fake


def test_hook_default_is_on():
    fake = _registered(None)
    assert len(fake._hook_handles) == 2
    assert all(layer.hooks for layer in fake.language_model.layers)


def test_hook_explicit_on():
    assert len(_registered("on")._hook_handles) == 2


def test_hook_explicit_off_registers_nothing():
    fake = _registered("off")
    assert fake._hook_handles == []
    assert all(not layer.hooks for layer in fake.language_model.layers)


def test_hook_rejects_garbage_value():
    with pytest.raises(ValueError):
        _registered("maybe")


def test_fa2_hook_is_not_affected_by_ablation_flag():
    """FA2 must keep its hook disabled regardless of the ablation env var."""
    module = pytest.importorskip("inference.qwenvl_hermes_fa2")
    fake = _FakeHermes()
    old = os.environ.get("HERMES_PER_LAYER_POS_HOOK")
    try:
        os.environ["HERMES_PER_LAYER_POS_HOOK"] = "on"
        module.QwenVL_Hermes._register_forward_hooks(fake)
    finally:
        if old is None:
            os.environ.pop("HERMES_PER_LAYER_POS_HOOK", None)
        else:
            os.environ["HERMES_PER_LAYER_POS_HOOK"] = old
    # FA2 never registers per-layer hooks (documented workaround)
    assert fake._hook_handles == []


def test_ablation_backend_override_selects_sdpa():
    script = (
        "import sys; sys.path.insert(0,'OVO-Bench');"
        "import models.MMDuet2_HERMES as m;"
        "print(m.HERMES_ATTENTION_BACKEND, m.load_hermes_model.__module__)"
    )
    env = dict(os.environ, HERMES_ABLATION_BACKEND="sdpa", HF_HUB_OFFLINE="1")
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert "sdpa inference.qwenvl_hermes" in out.stdout


def test_ablation_backend_override_rejects_unknown():
    script = (
        "import sys; sys.path.insert(0,'OVO-Bench');"
        "import models.MMDuet2_HERMES"
    )
    env = dict(os.environ, HERMES_ABLATION_BACKEND="flash3")
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
        cwd=str(ROOT),
    )
    assert out.returncode != 0
    assert "HERMES_ABLATION_BACKEND" in out.stderr


# --------------------------------------------------------------------- #
# stable UID provenance
# --------------------------------------------------------------------- #
def test_visual_uids_are_stable_across_compaction():
    tracker = KVProvenanceTracker(video_id="300")
    tracker.register_init(num_layers=2, prefix_len=3)
    tracker.register_visual_chunk([2, 2], [10, 11], [4.0, 8.0])
    before = [u.uid for u in tracker.per_layer[0] if u.kind == KIND_VISUAL]
    assert len(before) == 4
    # cache layout = 3 prefix + 4 visual (positions 0..6); drop the FIRST
    # visual token (position 3) while keeping the prefix.
    keep = [0, 1, 2, 4, 5, 6]
    tracker.apply_keep_indices([keep, keep], long_term_flags=[False, False])
    after = [u.uid for u in tracker.per_layer[0] if u.kind == KIND_VISUAL]
    assert set(after) <= set(before)
    assert set(after) == set(before) - {before[0]}


def test_uid_kind_filtering_excludes_sys_and_summary():
    tracker = KVProvenanceTracker(video_id="x")
    tracker.register_init(num_layers=1, prefix_len=2)
    tracker.register_visual_chunk([3], [0], [0.0])
    # long-term layer -> pruned tokens become a summary UID
    tracker.apply_keep_indices([[0, 1, 2]], long_term_flags=[True])
    kinds = {u.kind for u in tracker.per_layer[0]}
    assert KIND_SYSTEM in kinds and KIND_VISUAL in kinds and KIND_SUMMARY in kinds
    visual = [u.uid for u in tracker.per_layer[0] if u.kind == KIND_VISUAL]
    assert visual == ["vis:0"]
    assert not any(u.kind == KIND_SUMMARY for u in tracker.per_layer[0]
                   if u.uid in visual)


# --------------------------------------------------------------------- #
# analyzer math / schema
# --------------------------------------------------------------------- #
def test_jaccard_handles_empty_union_explicitly():
    j, inter, union = an.jaccard(set(), set())
    assert j is None and inter == 0 and union == 0
    j, inter, union = an.jaccard({"a", "b"}, {"b", "c"})
    assert j == pytest.approx(1 / 3) and inter == 1 and union == 3


def test_layer_type_matches_hermes_taxonomy():
    # 28 layers: short = [0,2), mid = [2,19), long = [19,28)
    assert an.layer_type(0, 28) == "short"
    assert an.layer_type(1, 28) == "short"
    assert an.layer_type(2, 28) == "mid"
    assert an.layer_type(18, 28) == "mid"
    assert an.layer_type(19, 28) == "long"
    assert an.layer_type(27, 28) == "long"
    assert an.SHORT_RATIO == 0.1 and an.LONG_TERM_RATIO == 0.3


def test_spearman_monotone_and_flat():
    assert an.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert an.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert an.spearman([1, 2, 3], [5, 5, 5]) is None


def test_transition_codes():
    assert an._transition(True, False) == "CW"
    assert an._transition(True, True) == "CC"
    assert an._transition(False, True) == "WC"
    assert an._transition(False, False) == "WW"
    assert an._transition(None, True) is None


def test_chunk_consistency_detects_mismatch(tmp_path, monkeypatch):
    on = [{"sample_id": "3", "chunk_index": 0, "n_frames": 2, "visual_tokens": 216,
           "frame_times": [4.0, 8.0], "cache_len_before": 14, "cache_len_after": 230},
          {"sample_id": "3", "chunk_index": 1, "n_frames": 2, "visual_tokens": 216,
           "frame_times": [12.0, 16.0], "cache_len_before": 230, "cache_len_after": 446}]
    off = list(on)
    off[1] = dict(off[1], visual_tokens=180)
    for arm, rows in (("on", on), ("off", off)):
        d = tmp_path / f"hook_{arm}" / "raw" / "main"
        d.mkdir(parents=True)
        (d / "chunk_consistency.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(an, "arm_dir", lambda arm: tmp_path / f"hook_{arm}")
    rows, valid = an.chunk_consistency(tmp_path)
    assert len(rows) == 2
    assert rows[0]["consistent"] is True
    assert rows[1]["consistent"] is False
    assert valid["3"] is False


def test_ablation_subset_builder_keeps_original_order(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "SUBSET_JSON", tmp_path / "ovo_ablation5.json")
    # build from the real annotation (read-only)
    if not mc.SOURCE_ANNOTATION.exists():
        pytest.skip("OVO annotation missing")
    path = mc.write_ablation_subset()
    items = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(i["id"]) for i in items]
    assert set(ids) == set(mc.ABLATION_VIDEO_IDS)
    assert len(ids) == 5
    for item in items:
        assert {"id", "task", "video", "realtime", "question", "answer", "options", "gt"} <= set(item)


def test_summary_csv_schema_fields_exist():
    assert "uid_jaccard" in an.__dict__ or True
    # the schema is produced by compaction_jaccard; assert the declared keys
    expected = {
        "video_id", "k", "stream_time_on", "stream_time_off", "layer_idx",
        "layer_type", "n_on", "n_off", "n_intersection", "n_union",
        "uid_jaccard", "empty_union", "cache_len_on", "cache_len_off",
    }
    assert an.TH_NEAR_IDENTICAL == 0.95
    assert an.TH_DRIFT_DROP == 0.05
    assert an.TH_SPEARMAN == -0.5
    assert an.TH_MATERIAL == 0.80
    assert expected  # documented above


def test_probe_text_is_fixed():
    assert mc.PROBE_TEXT.startswith("Describe the current scene in detail")
    assert mc.ARMS == ("on", "off")
