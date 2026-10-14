"""CPU-only, resumable OVO BW/RT analysis with explicit paired denominators.

Run from the repository root, for example::

    python InfoKVStream/analyze_ovo.py \
        --root outputs/InfoKVStream/ovo_20261013 --stage pilot

Prediction files are append-only. The latest successful result for a sample
is preferred to failed retries. Original records are never rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (6000, 2000, 1000, 500, 200, 100, 50)
GROUPS = {
    "backward": ("EPM", "ASI", "HLD"),
    "realtime": ("OCR", "ACR", "ATR", "STU", "FPD", "OJR"),
}
RUNTIME_METRICS = (
    "compression_trigger_count", "effective_compaction_count",
    "compression_total_sec", "mean_compression_sec",
    "question_to_first_token_sec", "question_answer_sec",
    "stream_processing_sec", "stream_fps", "peak_gpu_memory_mb",
    "kv_bytes_at_query", "managed_visual_tokens_at_query",
    "total_seen_visual_tokens", "retention_ratio",
    "peak_budget_overshoot_tokens", "total_wallclock_sec",
)
DIAGNOSTIC_METRICS = (
    "compression_trigger_count", "effective_compaction_count",
    "kv_bytes_at_query", "kv_tokens_at_query", "last_compression_sec",
    "p_gold", "margin",
)
PAIRED_BASE_FIELDS = (
    "stage", "group", "budget", "sample_id", "task_type", "ground_truth",
    "infokv_status", "infokv_prediction", "infokv_correct",
    "hermes_status", "hermes_prediction", "hermes_correct", "hermes_pair_status",
    "hermes_transition", "full_status", "full_prediction", "full_correct",
    "full_pair_status", "full_transition",
)


def finite(value):
    """A missing, nonnumeric or nonfinite measurement is never treated as zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def extract_option(reply) -> str:
    """Exact CPU-only equivalent of MMDuet2_HERMES.extract_option.

    Importing the model helper itself would load GPU/model dependencies.
    Qwen25VL_Stream uses this helper for its official/free MCQ output.
    """
    reply = "" if reply is None else str(reply)
    normalized = reply.strip().upper()
    match = re.match(r"^\s*([A-E])(?:\s|[.):,;]|$)", normalized)
    if match:
        return match.group(1)
    match = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", normalized)
    return match.group(1) if match else reply.strip()


def official_correct(record: dict) -> bool:
    """Official OVO BW/RT scorer: gold occurs in the extracted response."""
    gold = str(record.get("ground_truth", "")).strip().upper()
    response = record.get("prediction")
    if response is None:
        response = record.get("raw_output")
    return gold in "ABCDE" and len(gold) == 1 and gold in extract_option(response)


def is_success(record: dict) -> bool:
    status = record.get("status")
    return (
        status in (None, "", "success", "ok", "completed")
        and (record.get("prediction") is not None or record.get("raw_output") is not None)
        and str(record.get("ground_truth", "")).strip().upper() in tuple("ABCDE")
        and record.get("task_type") in {t for tasks in GROUPS.values() for t in tasks}
    )


def read_jsonl(path: Path, warnings: list[str]) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not a JSON object")
            except (ValueError, TypeError) as exc:
                warnings.append(f"{path}:{line_number}: skipped malformed JSON ({exc})")
                continue
            yield row


def load_records(paths: Iterable[Path], warnings: list[str]) -> dict[str, dict]:
    """Deduplicate deterministically, retaining success across failed retries."""
    records = {}
    duplicates = conflicts = 0
    for path in sorted(paths, key=lambda p: (p.stat().st_mtime_ns, str(p))):
        for row in read_jsonl(path, warnings):
            sample_id = row.get("sample_id")
            if sample_id is None:
                warnings.append(f"{path}: skipped record without sample_id")
                continue
            key = str(sample_id)
            row["sample_id"] = key
            previous = records.get(key)
            if previous is not None:
                duplicates += 1
                if is_success(previous) and is_success(row):
                    conflicts += any(previous.get(k) != row.get(k) for k in
                                     ("prediction", "ground_truth", "task_type"))
                if is_success(previous) and not is_success(row):
                    continue
            records[key] = row
    if duplicates:
        warnings.append(
            f"Deduplicated {duplicates} repeated sample records; "
            f"{conflicts} successful conflicts (latest success wins)."
        )
    return records


def option_metrics(row: dict) -> dict:
    logits = row.get("option_logits_by_letter")
    if not isinstance(logits, dict):
        raw = (row.get("option_token_logits_full")
               or row.get("option_token_logits_compressed")
               or row.get("option_token_logits") or {})
        logits = {}
        mapping = row.get("option_token_ids") or {}
        for letter, token_ids in mapping.items():
            values = [finite(raw.get(str(token_id))) for token_id in token_ids]
            values = [v for v in values if v is not None]
            if values:
                logits[letter] = max(values)
        if set(logits) != set(mapping):
            return {"p_gold": None, "margin": None}
    logits = {str(k): finite(v) for k, v in logits.items()}
    # Partial or NaN option measurements must not silently renormalize to fewer options.
    if len(logits) < 2 or any(v is None for v in logits.values()):
        return {"p_gold": None, "margin": None}
    gold = str(row.get("ground_truth", row.get("task_gold", ""))).strip().upper()
    if gold not in logits:
        return {"p_gold": None, "margin": None}
    peak = max(logits.values())
    denominator = sum(math.exp(v - peak) for v in logits.values())
    return {
        "p_gold": math.exp(logits[gold] - peak) / denominator,
        "margin": logits[gold] - max(v for k, v in logits.items() if k != gold),
    }


def load_diagnostics(run_dir: Path, warnings: list[str]) -> dict[str, dict]:
    """Read the existing diagnostic subset; it never changes accuracy coverage."""
    results = {}
    paths = (run_dir / "raw_decision").glob("*/paired/*.jsonl")
    aliases = {
        "compression_trigger_count": "compaction_trigger_count_at_query",
        "effective_compaction_count": "effective_compaction_count_at_query",
        "kv_bytes_at_query": "compressed_kv_bytes",
        "kv_tokens_at_query": "kept_kv_tokens",
        "last_compression_sec": "last_compaction_wallclock",
    }
    for path in sorted(paths, key=lambda p: (p.stat().st_mtime_ns, str(p))):
        for row in read_jsonl(path, warnings):
            key = str(row.get("sample_id", row.get("video_id", "")))
            status = row.get("task_status_full", row.get(
                "task_status_compressed", row.get("task_status")))
            if not key or row.get("task_missing") or status not in (None, "", "success", "ok"):
                continue
            results[key] = {
                **{name: finite(row.get(alias)) for name, alias in aliases.items()},
                **option_metrics(row),
            }
    return results


def wilson(correct: int, n: int, z: float = 1.959963984540054):
    if not n:
        return None, None
    fraction = correct / n
    denominator = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def measurement_summary(rows: list[dict], fields: Iterable[str]) -> dict:
    result = {}
    for field in fields:
        values = [finite(row.get(field)) for row in rows]
        values = [v for v in values if v is not None]
        result[f"n_{field}"] = len(values)
        result[f"mean_{field}"] = mean(values) if values else None
        result[f"median_{field}"] = median(values) if values else None
    return result


def accuracy_summary(rows: list[dict], categories: Iterable[str]) -> dict:
    successful = [r for r in rows if is_success(r)]
    n = len(successful)
    correct = sum(official_correct(r) for r in successful)
    low, high = wilson(correct, n)
    category_scores = []
    for category in categories:
        subset = [r for r in successful if r["task_type"] == category]
        if subset:
            category_scores.append(sum(official_correct(r) for r in subset) / len(subset))
    return {
        "recorded": len(rows), "N": n, "failed": len(rows) - n,
        "correct": correct, "accuracy": correct / n if n else None,
        "accuracy_wilson_low": low, "accuracy_wilson_high": high,
        "categories_observed": len(category_scores),
        "macro_accuracy_observed_categories": mean(category_scores) if category_scores else None,
        "official_macro_accuracy": mean(category_scores)
        if category_scores and len(category_scores) == len(tuple(categories)) else None,
        "unparsed_responses": sum(extract_option(r.get("prediction", r.get("raw_output")))
                                  not in tuple("ABCDE") for r in successful),
    }


def pair_records(info: dict[str, dict], baseline: dict[str, dict]) -> dict[str, str]:
    statuses = {}
    for key, row in info.items():
        reference = baseline.get(key)
        if not is_success(row):
            statuses[key] = "infokv_failed"
        elif reference is None:
            statuses[key] = "baseline_missing"
        elif not is_success(reference):
            statuses[key] = "baseline_failed"
        elif (str(row["ground_truth"]).strip().upper() != str(reference["ground_truth"]).strip().upper()
              or row["task_type"] != reference["task_type"]):
            statuses[key] = "label_mismatch"
        else:
            statuses[key] = "paired"
    return statuses


def paired_summary(info: dict[str, dict], baseline: dict[str, dict]) -> dict:
    states = pair_records(info, baseline)
    transitions = {key: 0 for key in ("CC", "CW", "WC", "WW")}
    paired_info = []
    paired_base = []
    for key, status in states.items():
        if status != "paired":
            continue
        base_ok = official_correct(baseline[key])
        info_ok = official_correct(info[key])
        transitions[("C" if base_ok else "W") + ("C" if info_ok else "W")] += 1
        paired_info.append(info[key])
        paired_base.append(baseline[key])
    n = sum(transitions.values())
    categories = tuple(dict.fromkeys(r["task_type"] for r in paired_info))
    info_stats = accuracy_summary(paired_info, categories)
    base_stats = accuracy_summary(paired_base, categories)
    return {
        "N": n, **transitions,
        "accuracy_infokv": info_stats["accuracy"],
        "accuracy_baseline": base_stats["accuracy"],
        "macro_accuracy_infokv": info_stats["macro_accuracy_observed_categories"],
        "macro_accuracy_baseline": base_stats["macro_accuracy_observed_categories"],
        "accuracy_infokv_wilson_low": info_stats["accuracy_wilson_low"],
        "accuracy_infokv_wilson_high": info_stats["accuracy_wilson_high"],
        "accuracy_baseline_wilson_low": base_stats["accuracy_wilson_low"],
        "accuracy_baseline_wilson_high": base_stats["accuracy_wilson_high"],
        "delta_accuracy": (transitions["WC"] - transitions["CW"]) / n if n else None,
        "delta_macro_accuracy": (
            info_stats["macro_accuracy_observed_categories"]
            - base_stats["macro_accuracy_observed_categories"]
        ) if n else None,
        "categories_observed": len(categories),
        **{status: sum(value == status for value in states.values())
           for status in ("baseline_missing", "baseline_failed", "label_mismatch")},
    }


def make_paired_rows(stage, group, budget, info, baselines, diagnostics):
    states = {name: pair_records(info, base) for name, base in baselines.items()}
    rows = []
    for key, record in info.items():
        row = {
            "stage": stage, "group": group, "budget": budget, "sample_id": key,
            "task_type": record.get("task_type"), "ground_truth": record.get("ground_truth"),
            "infokv_status": "success" if is_success(record) else "failed",
            "infokv_prediction": extract_option(record.get("prediction", record.get("raw_output"))),
            "infokv_correct": int(official_correct(record)) if is_success(record) else None,
        }
        metrics = {field: finite((record.get("runtime") or {}).get(field))
                   for field in RUNTIME_METRICS}
        metrics.update(option_metrics(record))
        row.update({f"infokv_{name}": value for name, value in metrics.items()})
        for name, baseline in baselines.items():
            reference = baseline.get(key)
            valid = reference is not None and is_success(reference)
            paired = states[name][key] == "paired"
            correct = int(official_correct(reference)) if valid else None
            row.update({
                f"{name}_status": "success" if valid else ("failed" if reference else "missing"),
                f"{name}_prediction": extract_option(reference.get("prediction", reference.get("raw_output")))
                if reference else None,
                f"{name}_correct": correct,
                f"{name}_pair_status": states[name][key],
                f"{name}_transition": (("C" if correct else "W")
                                        + ("C" if row["infokv_correct"] else "W")) if paired else None,
            })
            diagnostic = diagnostics[name].get(key, {}) if paired else {}
            row.update({f"{name}_{metric}": diagnostic.get(metric) for metric in DIAGNOSTIC_METRICS})
            for metric in ("p_gold", "margin", "kv_bytes_at_query"):
                current, previous = metrics.get(metric), diagnostic.get(metric)
                row[f"delta_{metric}_vs_{name}"] = current - previous \
                    if current is not None and previous is not None else None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fields: Iterable[str] = ()):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def percent(value):
    return "pending" if value is None else f"{100 * value:.2f}%"


def annotation_rows(annotation: Path) -> dict[str, dict]:
    if not annotation.exists():
        return {}
    return {str(row["id"]): row for row in json.loads(annotation.read_text(encoding="utf-8"))}


def analyze(root: Path, stage: str, baseline_root: Path,
            annotation: Path = REPO_ROOT / "OVO-Bench/data/ovo_bench_new.json") -> dict:
    warnings = []
    annotations = annotation_rows(annotation)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary_rows, paired_rows, baseline_rows = [], [], []
    for group, tasks in GROUPS.items():
        baseline_runs = {"full": baseline_root / f"ovobench_{group}" / "full"}
        baseline_runs.update({str(b): baseline_root / f"ovobench_{group}" / f"budget_{b}" for b in BUDGETS})
        records, diagnostics = {}, {}
        for name, run_dir in baseline_runs.items():
            records[name] = load_records((run_dir / "raw_decision").glob("*/tasks.jsonl"), warnings)
            diagnostics[name] = load_diagnostics(run_dir, warnings)
            for category in ("ALL", *tasks):
                wanted = tasks if category == "ALL" else (category,)
                subset = [r for r in records[name].values() if r.get("task_type") in wanted]
                expected_ids = {key for key, row in annotations.items() if row["task"] in wanted}
                expected = len(expected_ids) if annotations else None
                diagnostic_subset = [diagnostics[name][r["sample_id"]] for r in subset
                                     if is_success(r) and r["sample_id"] in diagnostics[name]]
                baseline_rows.append({
                    "group": group, "category": category,
                    "method": "full" if name == "full" else "hermes",
                    "budget": name, "expected": expected,
                    "missing": len(expected_ids - {r["sample_id"] for r in subset})
                    if expected is not None else None,
                    **accuracy_summary(subset, wanted), "diagnostic_N": len(diagnostic_subset),
                    **measurement_summary(diagnostic_subset, DIAGNOSTIC_METRICS),
                })
        for budget in BUDGETS:
            path = root / stage / group / f"budget_{budget}" / "predictions.jsonl"
            info = load_records([path] if path.exists() else [], warnings)
            baselines = {"hermes": records[str(budget)], "full": records["full"]}
            sidecars = {"hermes": diagnostics[str(budget)], "full": diagnostics["full"]}
            budget_pairs = make_paired_rows(stage, group, budget, info, baselines, sidecars)
            paired_rows.extend(budget_pairs)
            for category in ("ALL", *tasks):
                wanted = tasks if category == "ALL" else (category,)
                subset = {key: row for key, row in info.items() if row.get("task_type") in wanted}
                successful = [r for r in subset.values() if is_success(r)]
                planned_group = (manifest.get(f"{stage}_ids") or {}).get(group)
                if planned_group is not None:
                    planned_group = {str(key) for key in planned_group}
                    expected_ids = planned_group if category == "ALL" else {
                        key for key in planned_group if annotations.get(key, {}).get("task") in wanted
                    }
                    expected = len(expected_ids) if category == "ALL" or annotations else None
                else:
                    expected_ids = {key for key, row in annotations.items() if row["task"] in wanted}
                    expected = len(expected_ids) if annotations and stage == "full" else None
                stats = {
                    "stage": stage, "group": group, "category": category, "budget": budget,
                    "status": "pending" if not subset else (
                        "complete" if expected is not None and expected_ids == {
                            r["sample_id"] for r in successful} else "partial"),
                    "expected": expected,
                    "missing": len(expected_ids - set(subset)) if expected is not None else None,
                    **accuracy_summary(list(subset.values()), wanted),
                    **measurement_summary([r.get("runtime") or {} for r in successful], RUNTIME_METRICS),
                    **measurement_summary([option_metrics(r) for r in successful], ("p_gold", "margin")),
                }
                for name, baseline in baselines.items():
                    stats.update({f"{name}_{key}": value for key, value in paired_summary(subset, baseline).items()})
                    paired_subset = [r for r in budget_pairs if r.get("task_type") in wanted]
                    metric_fields = [f"{name}_{m}" for m in DIAGNOSTIC_METRICS]
                    metric_fields += [f"delta_{m}_vs_{name}" for m in ("p_gold", "margin", "kv_bytes_at_query")]
                    stats.update(measurement_summary(paired_subset, metric_fields))
                summary_rows.append(stats)
    destination = root / "analysis"
    write_csv(destination / f"{stage}_summary.csv", summary_rows)
    write_csv(destination / f"{stage}_paired.csv", paired_rows, PAIRED_BASE_FIELDS)
    write_csv(destination / "baseline_coverage.csv", baseline_rows)
    lines = [
        f"# InfoKVStream OVO {stage} comparison", "",
        "All values describe files available when this report was generated. "
        "A pilot or partial full run is not a final benchmark result.", "",
        "Accuracy follows Qwen25VL_Stream's official option extraction and the OVO BW/RT scorer "
        "(gold in extracted response). `accuracy` is sample-weighted micro accuracy; "
        "`official_macro_accuracy` is the unweighted mean over all categories in the group "
        "and remains empty if a category has no successful samples. "
        "95% Wilson intervals apply to each micro accuracy, not to the paired difference or macro average.", "",
        "Failed or missing records are listed separately and excluded from accuracy. "
        "Retries are deduplicated by sample_id, retaining the latest success. "
        "Each baseline comparison uses its own common successful sample set with matching labels. "
        "In CC/CW/WC/WW the first letter is baseline correctness and the second is InfoKV correctness. "
        "Differences are InfoKV minus baseline.", "",
        "Equal nominal visual token budgets do not guarantee equal physical KV bytes: HERMES uses "
        "per-layer selection and summaries, while InfoKV uses a common retained token set. "
        "Question-time KV byte measurements are provided where both methods recorded them. "
        "InfoKV's prefill overshoot is recorded separately from question-time memory.", "",
        "Baseline compression and option-logit diagnostics cover only the saved diagnostic subset; "
        "`diagnostic_N` and every `n_*` column show the actual measurement denominator. "
        "Unavailable measurements remain empty. Baseline `last_compression_sec` is one event, "
        "not total compression time. P_gold is softmax conditional on the available MCQ letters; "
        "margin is gold logit minus the largest alternative logit. They describe first-token "
        "preference and may differ from the free-decoded answer.", "",
        "| Group | Budget | Success / failed / missing | InfoKV micro | InfoKV official macro | "
        "HERMES paired N | HERMES paired accuracy | Δ vs HERMES | Full paired N | Δ vs Full |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["category"] != "ALL":
            continue
        missing = "unknown" if row["missing"] is None else row["missing"]
        lines.append(
            f"| {row['group']} | {row['budget']} | {row['N']} / {row['failed']} / {missing} "
            f"| {percent(row['accuracy'])} | {percent(row['official_macro_accuracy'])} "
            f"| {row['hermes_N']} | {percent(row['hermes_accuracy_baseline'])} "
            f"| {percent(row['hermes_delta_accuracy'])} | {row['full_N']} "
            f"| {percent(row['full_delta_accuracy'])} |"
        )
    lines += ["", "Baseline coverage (micro / official macro; diagnostic subset size):", ""]
    for row in baseline_rows:
        if row["category"] == "ALL":
            lines.append(f"- {row['group']} {row['method']} {row['budget']}: "
                         f"{row['N']}/{row['expected']}, failed={row['failed']}, "
                         f"{percent(row['accuracy'])} / {percent(row['official_macro_accuracy'])}; "
                         f"diagnostic N={row['diagnostic_N']}.")
    lines += ["", "Per-category results, Wilson intervals, paired transitions and compression/timing "
              "means and medians are in the summary CSV. Sample-level joins are in the paired CSV.", ""]
    if warnings:
        lines += ["Data loading notes:", "", *[f"- {warning}" for warning in warnings], ""]
    (destination / f"{stage}_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"summary": summary_rows, "paired": paired_rows,
            "baseline_coverage": baseline_rows, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--baseline-root", type=Path,
                        default=REPO_ROOT / "outputs/qwen25vl_full_eval")
    parser.add_argument("--annotation", type=Path,
                        default=REPO_ROOT / "OVO-Bench/data/ovo_bench_new.json")
    args = parser.parse_args()
    results = analyze(args.root, args.stage, args.baseline_root, args.annotation)
    print(json.dumps({"stage": args.stage, "sample_budget_records": len(results["paired"]),
                      "warnings": len(results["warnings"]), "output": str(args.root / "analysis")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
