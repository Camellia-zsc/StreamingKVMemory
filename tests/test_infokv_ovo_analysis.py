"""Regression checks for fair OVO joins and incremental result reporting."""

import json
import math

import pytest

from InfoKVStream import analyze_ovo as analysis


def record(sample_id, prediction="A", gold="A", **fields):
    return {"sample_id": str(sample_id), "prediction": prediction,
            "ground_truth": gold, "task_type": "EPM", "status": "success", **fields}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_official_extractor_keeps_the_benchmark_rule():
    assert analysis.extract_option("The answer is (b).") == "B"
    assert analysis.extract_option(" D: explanation") == "D"
    assert analysis.extract_option("No reply") == "No reply"
    assert analysis.official_correct(record(1, "The answer is (b).", "B"))
    assert not analysis.official_correct(record(1, "A. C is another choice", "C"))
    assert not analysis.official_correct(record(1, "", ""))


def test_dedup_preserves_success_when_a_later_retry_failed(tmp_path):
    path = write_rows(tmp_path / "predictions.jsonl", [
        record(1, "B"), record(1, "A"), record(1, None, status="failed"),
        record(2, None, status="failed"), record(2, "B", "B"),
    ])
    warnings = []
    rows = analysis.load_records([path], warnings)
    assert len(rows) == 2
    assert rows["1"]["prediction"] == "A"
    assert rows["2"]["prediction"] == "B"
    assert any("3 repeated" in note and "1 successful conflicts" in note for note in warnings)


def test_truncated_live_jsonl_is_reported_and_not_counted_as_a_wrong_answer(tmp_path):
    path = write_rows(tmp_path / "predictions.jsonl", [record(1)])
    with path.open("a") as stream:
        stream.write('{"sample_id": "2"')
    warnings = []
    rows = analysis.load_records([path], warnings)
    assert len(rows) == 1
    assert len(warnings) == 1
    assert "malformed JSON" in warnings[0]


def test_pairing_uses_matching_successful_samples_only_and_transition_direction():
    info = {str(i): row for i, row in enumerate([
        record(0, "A"), record(1, "B"), record(2, "A"), record(3, "B"),
        record(4, None, status="failed"), record(5), record(6), record(7),
    ])}
    base = {str(i): row for i, row in enumerate([
        record(0, "A"), record(1, "A"), record(2, "B"), record(3, "B"),
        record(4), record(5, None, status="failed"), record(6, "B", "B"),
    ])}
    result = analysis.paired_summary(info, base)
    assert result["N"] == 4
    assert tuple(result[key] for key in ("CC", "CW", "WC", "WW")) == (1, 1, 1, 1)
    assert result["accuracy_infokv"] == result["accuracy_baseline"] == 0.5
    assert result["delta_accuracy"] == 0
    assert result["baseline_missing"] == result["baseline_failed"] == result["label_mismatch"] == 1


def test_micro_macro_wilson_and_failed_denominators():
    rows = [record(0), record(1), record(2), record(3, "B", task_type="ASI"),
            record(4, None, status="failed", task_type="ASI")]
    result = analysis.accuracy_summary(rows, ("EPM", "ASI"))
    assert result["N"] == 4
    assert result["failed"] == 1
    assert result["accuracy"] == 0.75
    assert result["official_macro_accuracy"] == 0.5
    assert result["accuracy_wilson_low"] == pytest.approx(0.3006418426)
    assert result["accuracy_wilson_high"] == pytest.approx(0.9544127392)
    assert analysis.wilson(0, 0) == (None, None)
    assert analysis.accuracy_summary(rows, ("EPM", "ASI", "HLD"))["official_macro_accuracy"] is None


def test_option_diagnostics_use_max_token_variant_and_condition_on_letters():
    result = analysis.option_metrics({
        "task_gold": "A", "option_token_ids": {"A": [1, 2], "B": [3, 4]},
        "option_token_logits_full": {"1": 1000, "2": 999, "3": 998, "4": 997},
    })
    assert result["margin"] == 2
    assert result["p_gold"] == pytest.approx(1 / (1 + math.exp(-2)))
    assert analysis.option_metrics({"ground_truth": "B",
                                    "option_logits_by_letter": {"A": 1, "B": float("nan")}}) == {
        "p_gold": None, "margin": None}


def test_missing_runtime_measurements_do_not_become_zero():
    result = analysis.measurement_summary([{"count": 0}, {"count": 4}, {}, {"count": float("nan")}], ["count"])
    assert result["n_count"] == 2
    assert result["mean_count"] == 2


def test_analysis_emits_pending_rows_and_explicit_full_pair_coverage(tmp_path):
    root, baseline = tmp_path / "run", tmp_path / "baseline"
    anno = tmp_path / "annotation.json"
    anno.write_text(json.dumps([{"id": str(i), "task": "EPM"} for i in range(3)]))
    write_rows(root / "full/backward/budget_50/predictions.jsonl", [
        record(0, runtime={"compression_trigger_count": 10, "effective_compaction_count": 8,
                           "kv_bytes_at_query": 100},
               option_logits_by_letter={"A": 2, "B": 0}),
        record(1, "B", runtime={"compression_trigger_count": 20}),
        record(2, None, status="failed"),
    ])
    write_rows(baseline / "ovobench_backward/full/raw_decision/batch1/tasks.jsonl", [record(0, "B")])
    write_rows(baseline / "ovobench_backward/budget_50/raw_decision/batch1/tasks.jsonl", [record(0), record(1)])
    write_rows(baseline / "ovobench_backward/budget_50/raw_decision/batch1/paired/0.jsonl", [{
        "sample_id": "0", "task_gold": "A", "compaction_trigger_count_at_query": 5,
        "effective_compaction_count_at_query": 4, "compressed_kv_bytes": 200,
        "option_token_ids": {"A": [1], "B": [2]},
        "option_token_logits_compressed": {"1": 1, "2": 0},
    }])
    result = analysis.analyze(root, "full", baseline, anno)
    assert len(result["paired"]) == 3
    row = next(r for r in result["summary"] if r["group"] == "backward"
               and r["budget"] == 50 and r["category"] == "ALL")
    assert row["N"] == 2 and row["failed"] == 1 and row["missing"] == 0
    assert row["hermes_N"] == 2 and row["hermes_delta_accuracy"] == -0.5
    assert row["full_N"] == 1 and row["full_delta_accuracy"] == 1
    assert row["mean_compression_trigger_count"] == 15
    assert row["mean_effective_compaction_count"] == 8
    assert row["n_effective_compaction_count"] == 1
    assert row["mean_hermes_effective_compaction_count"] == 4
    assert row["n_hermes_effective_compaction_count"] == 1
    assert row["mean_delta_kv_bytes_at_query_vs_hermes"] == -100
    assert any(r["status"] == "pending" and r["N"] == 0 and r["accuracy"] is None
               for r in result["summary"])
    assert (root / "analysis/full_summary.csv").exists()
    assert "Full paired N" in (root / "analysis/full_report.md").read_text()
    assert "diagnostic subset" in (root / "analysis/full_report.md").read_text()


def test_empty_pilot_still_writes_headers_for_paired_csv(tmp_path):
    analysis.analyze(tmp_path / "run", "pilot", tmp_path / "base", tmp_path / "missing.json")
    content = (tmp_path / "run/analysis/pilot_paired.csv").read_text()
    assert content.startswith("stage,group,budget,sample_id,")


def test_pilot_manifest_controls_pending_counts_and_completion(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "pilot_ids": {"backward": ["0", "1"], "realtime": []},
    }))
    anno = tmp_path / "annotation.json"
    anno.write_text(json.dumps([{"id": str(i), "task": "EPM"} for i in range(20)]))
    write_rows(root / "pilot/backward/budget_50/predictions.jsonl", [record(0), record(1)])
    result = analysis.analyze(root, "pilot", tmp_path / "base", anno)
    row = next(r for r in result["summary"] if r["group"] == "backward"
               and r["budget"] == 50 and r["category"] == "ALL")
    assert row["expected"] == 2
    assert row["missing"] == 0
    assert row["status"] == "complete"
    pending = next(r for r in result["summary"] if r["group"] == "backward"
                   and r["budget"] == 100 and r["category"] == "ALL")
    assert pending["status"] == "pending"
    assert pending["expected"] == pending["missing"] == 2


def test_incomplete_baseline_logit_mapping_is_unavailable():
    assert analysis.option_metrics({
        "task_gold": "A", "option_token_ids": {"A": [1], "B": [2], "C": [3]},
        "option_token_logits_full": {"1": 2, "2": 1},
    }) == {"p_gold": None, "margin": None}
