"""OVO-Bench full-eval + per-category diagnostic subset.

Two clearly separated products:

* **Full accuracy** -- every OVO RT/BW sample runs the normal streaming
  protocol and contributes to overall / per-category accuracy.
* **Diagnostic subset** -- the first ``per_category`` *unique videos* of each
  task in original annotation order (deterministic, never random) additionally
  carry the paired decision diagnostics: effective compaction count ``k``,
  ``P_gold``, ``ΔP_gold``, ``margin``, ``ΔM_z``, CC/CW/WC/WW and margin
  crossing.

Everything here is pure CPU: manifest building, joins and plotting. No model
is loaded and no GPU work happens in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import budget_stress


# Canonical OVO taxonomy (see OVO-Bench/constant.py). Kept here so the
# manifest does not depend on importing the OVO-Bench package.
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

GROUPS = {
    "backward": BACKWARD_TASKS,
    "realtime": REALTIME_TASKS,
    "forward": FORWARD_TASKS,
}

DEFAULT_ANNOTATION = "OVO-Bench/data/ovo_bench_new.json"
DEFAULT_PER_CATEGORY = 10


# --------------------------------------------------------------------- #
# manifest (deterministic, original data order)
# --------------------------------------------------------------------- #
def load_items(annotation: str = DEFAULT_ANNOTATION) -> List[dict]:
    return json.loads(Path(annotation).read_text(encoding="utf-8"))


def build_manifest(
    annotation: str = DEFAULT_ANNOTATION,
    per_category: int = DEFAULT_PER_CATEGORY,
) -> dict:
    """First ``per_category`` unique videos of every task, in data order."""

    items = load_items(annotation)
    manifest = {
        "annotation": str(annotation),
        "per_category": int(per_category),
        "selection": "first_n_unique_videos_in_original_data_order",
        "deterministic": True,
        "groups": {},
    }
    for group, tasks in GROUPS.items():
        entry = {"tasks": list(tasks), "categories": {}}
        for task in tasks:
            seen: List[str] = []
            seen_set = set()
            total = 0
            for item in items:
                if item.get("task") != task:
                    continue
                total += 1
                vid = str(item["id"])
                if vid not in seen_set:
                    seen_set.add(vid)
                    seen.append(vid)
            selected = seen[:per_category]
            entry["categories"][task] = {
                "total_samples": total,
                "total_unique_videos": len(seen),
                "selected": selected,
                "selected_count": len(selected),
            }
        manifest["groups"][group] = entry
    return manifest


def manifest_video_ids(manifest: dict, group: str) -> List[str]:
    entry = (manifest.get("groups") or {}).get(group) or {}
    out: List[str] = []
    for task in entry.get("tasks", []):
        out.extend(entry["categories"].get(task, {}).get("selected", []))
    return list(dict.fromkeys(out))


def write_manifest(path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------- #
# diagnostic records (join Full reference with each budget)
# --------------------------------------------------------------------- #
def load_run_records(run_dir) -> List[dict]:
    return budget_stress.load_budget_records([run_dir])


def merge_runs(
    full_dir,
    budget_dirs: Dict[int, object],
    n_boot: int = 2000,
    seed: int = 42,
):
    """``{budget: dir}`` + the single Full dir -> (paired, summary)."""

    full_records = load_run_records(full_dir)
    return budget_stress.build_budget_table(budget_dirs, full_records, n_boot=n_boot, seed=seed)


def group_rows(paired: List[dict], tasks: Iterable[str]) -> List[dict]:
    wanted = {str(t) for t in tasks}
    return [r for r in paired if str(r.get("task_type")) in wanted]


def index_by_video(rows: List[dict]) -> Dict[str, dict]:
    return {str(r.get("video_id")): r for r in rows}


# --------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------- #
SUMMARY_FIELDS = [
    "benchmark", "category", "budget", "N",
    "mean_k", "median_k",
    "mean_delta_p_gold", "median_delta_p_gold",
    "mean_delta_margin", "median_delta_margin",
    "CC", "CW", "WC", "WW",
    "CW_rate", "margin_crossing_count", "margin_crossing_rate",
    "acc_full", "acc_compressed",
]


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0


def category_summary(
    paired: List[dict],
    budgets: List[int],
    benchmark: str,
    task: str,
) -> List[dict]:
    rows = [r for r in paired if str(r.get("task_type")) == str(task)]
    out = []
    for budget in budgets:
        pref = f"budget{budget}"
        have = [r for r in rows if r.get(f"{pref}_prediction") is not None]
        if not have:
            continue
        cc = cw = wc = ww = 0
        for r in have:
            full_ok = bool(r.get("full_correct"))
            comp_ok = bool(r.get(f"{pref}_correct"))
            if full_ok and comp_ok:
                cc += 1
            elif full_ok and not comp_ok:
                cw += 1
            elif not full_ok and comp_ok:
                wc += 1
            else:
                ww += 1
        n = len(have)
        crossing = sum(1 for r in have if r.get(f"{pref}_margin_crossed"))
        full_ok_n = sum(1 for r in have if r.get("full_correct"))
        comp_ok_n = sum(1 for r in have if r.get(f"{pref}_correct"))
        out.append({
            "benchmark": benchmark,
            "category": task,
            "budget": budget,
            "N": n,
            "mean_k": _mean([r.get(f"{pref}_k") for r in have]),
            "median_k": _median([r.get(f"{pref}_k") for r in have]),
            "mean_delta_p_gold": _mean([r.get(f"{pref}_delta_p_gold") for r in have]),
            "median_delta_p_gold": _median([r.get(f"{pref}_delta_p_gold") for r in have]),
            "mean_delta_margin": _mean([r.get(f"{pref}_delta_margin") for r in have]),
            "median_delta_margin": _median([r.get(f"{pref}_delta_margin") for r in have]),
            "CC": cc, "CW": cw, "WC": wc, "WW": ww,
            "CW_rate": (cw / full_ok_n) if full_ok_n else None,
            "margin_crossing_count": crossing,
            "margin_crossing_rate": (crossing / n) if n else None,
            "acc_full": (full_ok_n / n) if n else None,
            "acc_compressed": (comp_ok_n / n) if n else None,
        })
    return out


def diagnostic_summary(
    paired: List[dict],
    budgets: List[int],
    group: str,
    tasks: Iterable[str],
) -> List[dict]:
    rows = []
    for task in tasks:
        rows.extend(category_summary(paired, budgets, f"ovobench_{group}", task))
    return rows


def write_csv(path, rows: List[dict], fields: Optional[List[str]] = None) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    return path


# --------------------------------------------------------------------- #
# plots (diagnostic subset only)
# --------------------------------------------------------------------- #
PLOT_EXCLUDE_BUDGETS = {80}
PLOT_FIGSIZE = (10.0, 4.6)
PLOT_TICK_ROTATION = 40


def _plot_rows(rows: List[dict]) -> List[int]:
    seen = {
        int(r["budget"])
        for r in rows
        if r.get("budget") is not None and int(r["budget"]) not in PLOT_EXCLUDE_BUDGETS
    }
    return sorted(seen, reverse=True)


def _style(ax, budgets):
    ax.set_xscale("linear")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(b)) for b in budgets], rotation=PLOT_TICK_ROTATION,
                       ha="right", fontsize=8)


def plot_group(summary_rows: List[dict], out_dir, group: str) -> List[dict]:
    """One figure per metric; one line per category. Diagnostic subset only."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import _apply_style, _save

    if not summary_rows:
        return []
    _apply_style()
    out_dir = Path(out_dir)
    budgets = _plot_rows(summary_rows)
    categories = sorted({r["category"] for r in summary_rows})
    figs = []

    metrics = (
        ("mean_k", "mean effective compaction count", "budget_vs_effective_compaction_count"),
        ("mean_delta_p_gold", "mean ΔP_gold", "budget_vs_delta_p_gold"),
        ("mean_delta_margin", "mean ΔM_z", "budget_vs_delta_margin"),
        ("CW_rate", "Correct→Wrong rate", "budget_vs_cw_rate"),
        ("margin_crossing_rate", "margin crossing rate", "budget_vs_margin_crossing_rate"),
    )
    for key, ylabel, name in metrics:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        plotted = False
        for category in categories:
            series = {int(r["budget"]): r.get(key) for r in summary_rows
                      if r["category"] == category}
            xs = [b for b in budgets if series.get(b) is not None]
            ys = [series[b] for b in xs]
            if xs:
                ax.plot(xs, ys, "-o", label=category)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        if key in {"mean_delta_p_gold", "mean_delta_margin"}:
            ax.axhline(0.0, color="0.8")
        ax.set_xlabel("HERMES visual KV budget")
        ax.set_ylabel(ylabel)
        _style(ax, budgets)
        ax.set_title(f"{ylabel} — OVO {group} diagnostic subset")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        figs.append(_save(
            fig, out_dir, f"{name}_{group}",
            f"OVO-Bench {group} diagnostic subset (first 10 unique videos per "
            "category, original data order).",
        ))

    # transitions as a stacked bar per category
    fig, ax = plt.subplots(figsize=(max(PLOT_FIGSIZE[0], 0.9 * len(budgets) * len(categories)), 4.6))
    labels = []
    bottoms = {"CC": [], "CW": [], "WC": [], "WW": []}
    for category in categories:
        for budget in budgets:
            row = next((r for r in summary_rows
                        if r["category"] == category and int(r["budget"]) == budget), None)
            labels.append(f"{category}@{budget}")
            for key in bottoms:
                bottoms[key].append((row or {}).get(key, 0))
    import numpy as np

    x = np.arange(len(labels))
    base = np.zeros(len(labels))
    colors = {"CC": "#4c78a8", "CW": "#e45756", "WC": "#54a24b", "WW": "#b0b0b0"}
    for key in ("CC", "CW", "WC", "WW"):
        values = np.asarray(bottoms[key], dtype=float)
        ax.bar(x, values, bottom=base, label=key, color=colors[key])
        base = base + values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("count")
    ax.set_title(f"CC/CW/WC/WW — OVO {group} diagnostic subset")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    figs.append(_save(
        fig, out_dir, f"budget_vs_transitions_{group}",
        f"CC/CW/WC/WW per category and budget, OVO {group} diagnostic subset.",
    ))
    return figs
