import pytest

from InfoKVStream.analyze_streamingbench_rt import cluster_ci, video_id


def test_cluster_summary_weights_each_video_once():
    rows = [{"video_id": "a", "score": 0.0}] * 5 + [{"video_id": "b", "score": 1.0}]
    avg, low, high = cluster_ci(rows, "score")
    assert avg == 0.5
    assert low <= avg <= high


def test_missing_temporal_mass_is_excluded():
    rows = [{"video_id": "a", "ram": None}, {"video_id": "b", "ram": 0.7}]
    assert cluster_ci(rows, "ram") == pytest.approx((0.7, 0.7, 0.7))
    assert cluster_ci(rows[:1], "ram") == (None, None, None)


def test_paired_cluster_difference_zero_control():
    rows = [{"video_id": str(v), "delta_accuracy": 0} for v in range(20) for _ in range(5)]
    assert cluster_ci(rows, "delta_accuracy") == (0, 0, 0)
    assert video_id("Real-Time Visual Understanding_sample_155_5") == "155"
