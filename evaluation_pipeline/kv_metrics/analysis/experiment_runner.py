"""Run an experiment matrix (several HERMES budgets) over a sample set.

Each matrix entry is executed by the existing two-pass runner
(``evaluation_pipeline.kv_metrics.paired_runner``); this module only resolves
configs, selects samples, handles resume, and lays out the output directory.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

from ..config import KVMetricConfig
from ...evaluate import load_config
from ..paired_runner import run as run_paired
from . import sample_selector


FULL_BUDGET = 100000000


def resolve_matrix(config: dict) -> List[dict]:
    entries = []
    if config.get("budgets"):
        return [
            {"name": f"hermes_budget_{int(b)}", "hermes_enabled": True, "budget": int(b)}
            for b in config["budgets"]
        ]
    for item in config.get("experiments", []) or []:
        name = item.get("name")
        if not name:
            continue
        hermes_enabled = bool(item.get("hermes_enabled", True))
        budget = item.get("budget", FULL_BUDGET if not hermes_enabled else 2000)
        entries.append(
            {
                "name": str(name),
                "hermes_enabled": hermes_enabled,
                "budget": FULL_BUDGET if not hermes_enabled else int(budget),
            }
        )
    if not entries:
        # sensible default matrix
        entries = [
            {"name": "full", "hermes_enabled": False, "budget": FULL_BUDGET},
            {"name": "hermes_budget_2000", "hermes_enabled": True, "budget": 2000},
            {"name": "hermes_budget_4000", "hermes_enabled": True, "budget": 4000},
        ]
    return entries


def _load_samples(config: dict, benchmark: str, base_config: dict):
    from ...adapters import BENCHMARK_REGISTRY

    bench_config = copy.deepcopy(base_config.get("benchmarks", {}).get(benchmark, {}))
    adapter = BENCHMARK_REGISTRY[benchmark](bench_config)
    return list(adapter.load_samples())


def resolve_samples(
    config: dict,
    base_config: dict,
    benchmark: str,
    experiment_dir: Path,
    force: bool = False,
) -> List[dict]:
    selection_path = experiment_dir / "selected_samples.json"
    explicit = config.get("sample_ids") or []
    if explicit:
        samples = _load_samples(config, benchmark, base_config)
        by_id = {str(s.sample_id): s for s in samples}
        selected = [
            {
                "sample_id": sid,
                "video_path": by_id[sid].video_path if sid in by_id else None,
                "source": "explicit",
            }
            for sid in explicit
        ]
    elif selection_path.exists() and not force:
        selected = sample_selector.read_selection(selection_path)
    elif config.get("task_scope"):
        samples = _load_samples(config, benchmark, base_config)
        inference = base_config.get("inference", {})
        selected = sample_selector.select_compaction_accuracy_samples(
            samples,
            target_groups=config.get("task_scope") or ("backward", "realtime"),
            num_samples=int(config.get("num_samples", 20)),
            target_total_queries=int(config.get("target_total_queries", 100)),
            seed=int(config.get("sample_selection_seed", 20260930)),
            metrics_root=str(config.get("metrics_root", "outputs/kv_metrics")),
            sample_fps=float(inference.get("sample_fps", 0.5)),
            max_side=int(inference.get("max_side", 336)),
            min_queries=int(config.get("min_queries", 2)),
            min_effective_compactions=int(
                config.get("min_effective_compactions", 2)
            ),
            budget=int((config.get("budgets") or [2000])[0]),
            max_query_time=config.get("max_query_time"),
        )
        sample_selector.write_compaction_selection(
            selected,
            experiment_dir / "selected_samples.json",
            meta={
                "benchmark": benchmark,
                "task_scope": config.get("task_scope"),
                "num_samples": config.get("num_samples"),
                "target_total_queries": config.get("target_total_queries"),
                "sample_selection_seed": config.get("sample_selection_seed"),
            },
        )
    else:
        samples = _load_samples(config, benchmark, base_config)
        inference = base_config.get("inference", {})
        candidates = sample_selector.select_samples(
            samples,
            metrics_root=str(config.get("metrics_root", "outputs/kv_metrics")),
            min_compactions=config.get("min_compactions", 3),
            min_video_duration=config.get("min_video_duration"),
            max_video_duration=config.get("max_video_duration"),
            min_max_kv_tokens=config.get("min_max_kv_tokens"),
            min_compression_ratio=config.get("min_compression_ratio"),
            num_samples=config.get("num_samples"),
            top_k_longest=config.get("top_k_longest"),
            top_k_most_compactions=config.get("top_k_most_compactions"),
            scale=config.get("sample_scale", "smoke"),
            sample_fps=float(inference.get("sample_fps", 0.5)),
            max_side=int(inference.get("max_side", 336)),
        )
        selected = [c.to_dict() for c in candidates]
        sample_selector.write_selection(
            candidates,
            selection_path,
            meta={
                "benchmark": benchmark,
                "scale": config.get("sample_scale", "smoke"),
                "min_compactions": config.get("min_compactions", 3),
            },
        )
    return selected


def run_matrix(
    experiment_config_path: str,
    selection_path: Optional[str] = None,
    force: bool = False,
    resume: bool = True,
    experiment_name: Optional[str] = None,
) -> Path:
    config = load_config(experiment_config_path)
    base_config_path = config.get("base_config")
    if not base_config_path:
        raise ValueError("experiment config needs a 'base_config' path")
    base_config = load_config(base_config_path)

    benchmark = config.get("benchmark") or next(iter(base_config.get("benchmarks", {})), None)
    if not benchmark:
        raise ValueError("no benchmark resolved for the experiment")

    exp_name = experiment_name or config.get("experiment_name", "experiment")
    output_root = Path(config.get("output_root", "outputs/kv_metrics/experiments"))
    experiment_dir = output_root / exp_name
    (experiment_dir / "raw").mkdir(parents=True, exist_ok=True)

    selected = resolve_samples(config, base_config, benchmark, experiment_dir, force=force)
    if selection_path:
        selected = sample_selector.read_selection(Path(selection_path))
    # Always persist the resolved selection with its provenance so the report
    # can state exactly how samples were chosen.
    meta = {
        "benchmark": benchmark,
        "scale": config.get("sample_scale", "smoke"),
        "min_compactions": config.get("min_compactions", 3),
        "metrics_root": config.get("metrics_root", "outputs/kv_metrics"),
        "source": str(selection_path) if selection_path else "config/sample_ids_or_selector",
        "explicit_sample_ids": bool(config.get("sample_ids")),
    }
    (experiment_dir / "selected_samples.json").write_text(
        json.dumps(
            {"meta": meta, "num_selected": len(selected), "samples": selected},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sample_ids = [str(s["sample_id"]) for s in selected]

    matrix = resolve_matrix(config)
    resolved = {
        "experiment_name": exp_name,
        "benchmark": benchmark,
        "sample_ids": sample_ids,
        "experiments": matrix,
        "base_config": base_config_path,
        "metric_backend": config.get("metric_backend", "eager"),
        "temporal_bins": config.get("temporal_bins", [0, 10, 30, 60, 300, "inf"]),
        "profile_wallclock": bool(config.get("profile_wallclock", True)),
    }
    (experiment_dir / "experiment_config_resolved.yaml").write_text(
        _to_yaml(resolved), encoding="utf-8"
    )

    for entry in matrix:
        name = entry["name"]
        run_dir = experiment_dir / "raw" / name
        summary_path = run_dir / "summary.json"
        if resume and not force and summary_path.exists():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("num_events", 0) > 0:
                print(f"[experiment] resume: skip {name} (already complete)")
                continue

        run_config = copy.deepcopy(base_config)
        run_config.setdefault("memory", {})
        run_config["memory"]["method"] = "hermes"
        run_config["memory"].setdefault("compression", {})
        run_config["memory"]["compression"]["enabled"] = entry["hermes_enabled"]
        run_config["memory"]["compression"]["budget"] = entry["budget"]

        kv_config = KVMetricConfig(
            enable_kv_metrics=True,
            mode="paired",
            output_dir=str(experiment_dir / "raw"),
            run_name=name,
            detailed_trajectory=bool(
                (config.get("detailed_trajectory") or {}).get("strategy")
                == "longest_query_only"
            ),
            attention_backend=config.get("metric_backend", "eager"),
            anchor=config.get("anchor", "last_user_token"),
            temporal_bins=_parse_bins(config.get("temporal_bins", [0, 10, 30, 60, 300, "inf"])),
            save_per_head=bool(config.get("save_per_head", False)),
            profile_wallclock=bool(config.get("profile_wallclock", True)),
            scope=config.get("scope", "hermes_managed"),
            summary_token_mode=config.get("summary_token_mode", "identity_only"),
        )
        print(f"[experiment] run {name}: budget={entry['budget']} samples={len(sample_ids)}")
        run_paired(
            run_config,
            benchmark=benchmark,
            max_samples=0,
            kv_config=kv_config,
            sample_ids=sample_ids,
        )
    return experiment_dir


def _parse_bins(bins) -> List[float]:
    if isinstance(bins, str):
        return [
            float("inf") if b.strip().lower() in {"inf", "+inf"} else float(b)
            for b in bins.split(",")
            if b.strip()
        ]
    return [
        float("inf") if str(b).lower() in {"inf", "+inf"} else float(b)
        for b in bins
    ]


def _to_yaml(data: dict) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except Exception:  # noqa: BLE001
        return json.dumps(data, indent=2, ensure_ascii=False)
