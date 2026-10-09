"""CPU tests for the OVO full-eval + per-category diagnostic subset."""

import json
from pathlib import Path

import pytest

from evaluation_pipeline.kv_metrics.analysis import ovo_diagnostic as diag


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "OVO-Bench/data/ovo_bench_new.json"


@pytest.mark.skipif(not ANNOTATION.exists(), reason="OVO annotation missing")
def test_manifest_selects_first_n_unique_videos_per_category():
    manifest = diag.build_manifest(str(ANNOTATION), per_category=10)
    for group, entry in manifest["groups"].items():
        for task in entry["tasks"]:
            info = entry["categories"][task]
            assert info["selected_count"] == min(10, info["total_unique_videos"])
            assert len(info["selected"]) == len(set(info["selected"]))

    # OVO annotation has 1 query per video, so total_unique_videos == samples.
    bw = manifest["groups"]["backward"]["categories"]
    assert bw["EPM"]["total_unique_videos"] == 297
    assert bw["ASI"]["total_unique_videos"] == 148
    assert bw["HLD"]["total_unique_videos"] == 186
    assert bw["EPM"]["selected_count"] == 10


@pytest.mark.skipif(not ANNOTATION.exists(), reason="OVO annotation missing")
def test_manifest_is_deterministic_and_in_data_order():
    a = diag.build_manifest(str(ANNOTATION), per_category=10)
    b = diag.build_manifest(str(ANNOTATION), per_category=10)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    items = diag.load_items(str(ANNOTATION))
    first_epm = [str(i["id"]) for i in items if i["task"] == "EPM"][:10]
    assert a["groups"]["backward"]["categories"]["EPM"]["selected"] == first_epm


@pytest.mark.skipif(not ANNOTATION.exists(), reason="OVO annotation missing")
def test_manifest_video_ids_and_group_filtering():
    manifest = diag.build_manifest(str(ANNOTATION), per_category=10)
    bw_ids = diag.manifest_video_ids(manifest, "backward")
    rt_ids = diag.manifest_video_ids(manifest, "realtime")
    assert len(bw_ids) == 30   # 3 backward categories x 10
    assert len(rt_ids) == 60   # 6 realtime categories x 10
    assert not (set(bw_ids) & set(rt_ids))

    rows = [
        {"video_id": "0", "task_type": "EPM"},
        {"video_id": "1", "task_type": "STU"},
    ]
    assert [r["video_id"] for r in diag.group_rows(rows, ["EPM"])] == ["0"]


def test_category_summary_counts_transitions_and_rates():
    paired = [
        {"video_id": "v1", "task_type": "EPM", "full_correct": True,
         "budget100_prediction": "A", "budget100_correct": False,
         "budget100_k": 5, "budget100_delta_p_gold": -0.1,
         "budget100_delta_margin": -0.2, "budget100_margin_crossed": 1},
        {"video_id": "v2", "task_type": "EPM", "full_correct": True,
         "budget100_prediction": "B", "budget100_correct": True,
         "budget100_k": 5, "budget100_delta_p_gold": 0.0,
         "budget100_delta_margin": 0.0, "budget100_margin_crossed": 0},
        {"video_id": "v3", "task_type": "EPM", "full_correct": False,
         "budget100_prediction": "C", "budget100_correct": True,
         "budget100_k": 9, "budget100_delta_p_gold": 0.2,
         "budget100_delta_margin": 0.3, "budget100_margin_crossed": 0},
    ]
    rows = diag.category_summary(paired, [100], "ovobench_backward", "EPM")
    assert len(rows) == 1
    row = rows[0]
    assert row["N"] == 3
    assert (row["CC"], row["CW"], row["WC"], row["WW"]) == (1, 1, 1, 0)
    assert row["CW_rate"] == pytest.approx(0.5)
    assert row["margin_crossing_count"] == 1
    assert row["margin_crossing_rate"] == pytest.approx(1 / 3)
    assert row["mean_k"] == pytest.approx((5 + 5 + 9) / 3)
    assert row["median_k"] == pytest.approx(5)
    assert row["acc_full"] == pytest.approx(2 / 3)
    assert row["acc_compressed"] == pytest.approx(2 / 3)


def test_write_csv_uses_declared_fields(tmp_path):
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    path = diag.write_csv(tmp_path / "x.csv", rows, fields=["b", "a"])
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "b,a"


def test_summary_fields_cover_requested_columns():
    for field in (
        "benchmark", "category", "budget", "N", "mean_k", "median_k",
        "mean_delta_p_gold", "median_delta_p_gold",
        "mean_delta_margin", "median_delta_margin",
        "CC", "CW", "WC", "WW", "CW_rate",
        "margin_crossing_count", "margin_crossing_rate",
    ):
        assert field in diag.SUMMARY_FIELDS
