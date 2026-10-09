"""Compaction-accuracy experiment: tables, figures and report section.

Semantics kept strictly separate:

A. longest-run fidelity trajectory  -- ONE query per sample (q_star), probed
   with the SAME official prompt after every *effective* compaction.
B. cross-query task accuracy        -- real benchmark queries, grouped by the
   number of effective compactions their own run had completed.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from . import loader, task_accuracy


def _plt():
    """Import matplotlib lazily so the CLI imports without a plotting stack."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plotting():
    from . import plotting

    return plotting


def load_task_rows(experiment_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted((Path(experiment_dir) / "raw").glob("*/paired/*.jsonl")):
        for record in loader.read_jsonl(path):
            record.setdefault("sample_id", loader.video_id_from_event(record.get("event_id", "")))
            record["experiment"] = path.parent.parent.name
            rows.append(record)
    return rows


def load_trajectory_rows(experiment_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    base = Path(experiment_dir) / "raw"
    for path in sorted(base.glob("*/detailed_trajectory/*/probe_metrics_*.jsonl")):
        for record in loader.read_jsonl(path):
            record["experiment"] = path.parents[2].name
            key = str(record.get("video_id") or record.get("sample_id"))
            out.setdefault(key, []).append(record)
    for values in out.values():
        values.sort(key=lambda r: int(r.get("probe_k") or 0))
    return out


def load_sample_selection(experiment_dir: Path) -> List[dict]:
    data = loader.read_json(Path(experiment_dir) / "selected_samples.json", {}) or {}
    if isinstance(data, dict):
        return data.get("samples", [])
    return data


# --------------------------------------------------------------------- #
# CSV tables
# --------------------------------------------------------------------- #
def _write_csv(path: Path, rows: List[dict], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_tables(experiment_dir: Path, n_boot: int = 2000, seed: int = 42) -> dict:
    experiment_dir = Path(experiment_dir)
    tables = experiment_dir / "analysis"
    task_rows = load_task_rows(experiment_dir)
    traj = load_trajectory_rows(experiment_dir)
    selection = load_sample_selection(experiment_dir)

    exact = task_accuracy.accuracy_by_compaction_count(
        task_rows, n_boot=n_boot, seed=seed
    )
    binned = task_accuracy.accuracy_by_compaction_count(
        task_rows, n_boot=n_boot, seed=seed, bin_edges=[0, 1, 2, 3, 5]
    )
    _write_csv(tables / "task_accuracy_by_compaction_count.csv", exact)
    _write_csv(tables / "task_accuracy_by_compaction_count_binned.csv", binned)

    transitions = []
    for row in exact:
        transitions.append({
            "effective_compaction_count": row["effective_compaction_count"],
            "num_queries": row["num_queries"],
            "num_correct_to_correct": row.get("num_correct_to_correct"),
            "num_correct_to_wrong": row.get("num_correct_to_wrong"),
            "num_wrong_to_correct": row.get("num_wrong_to_correct"),
            "num_wrong_to_wrong": row.get("num_wrong_to_wrong"),
            "correct_to_wrong_rate": row.get("correct_to_wrong_rate"),
            "protocol_violation_count": row.get("protocol_violation_count"),
            "missing_count": row.get("missing_count"),
        })
    _write_csv(tables / "task_transitions.csv", transitions)

    _write_csv(tables / "sample_selection.csv", [
        {
            "sample_id": s.get("sample_id"),
            "task_type": s.get("task_type"),
            "task_group": s.get("task_group"),
            "num_queries": s.get("num_queries"),
            "longest_query_id": s.get("longest_query_id"),
            "longest_query_time": s.get("longest_query_time"),
            "estimated_or_observed_effective_compactions":
                s.get("estimated_or_observed_effective_compactions"),
            "compaction_source": s.get("compaction_source"),
            "selection_reason": s.get("selection_reason"),
        }
        for s in selection
    ])

    traj_rows = []
    for sample_id, rows in traj.items():
        for row in rows:
            traj_rows.append({
                "sample_id": sample_id,
                "query_id": row.get("query_id"),
                "probe_k": row.get("probe_k"),
                "stream_time": row.get("stream_time"),
                "RAM": row.get("ram_mean"),
                "AORE": row.get("aore_mean"),
                "JS": row.get("js"),
                "KL": row.get("kl"),
                "compression_ratio": row.get("compression_ratio"),
                "compressed_kv_tokens": row.get("compressed_kv_tokens_probe"),
                "full_kv_tokens": row.get("full_kv_tokens_probe"),
            })
    _write_csv(tables / "trajectory_metrics.csv", traj_rows)

    per_query = []
    for row in task_rows:
        per_query.append({
            "experiment": row.get("experiment"),
            "sample_id": row.get("sample_id"),
            "query_id": row.get("query_id"),
            "query_time": row.get("query_time"),
            "compaction_trigger_count_at_query": row.get("compaction_trigger_count_at_query"),
            "effective_compaction_count_at_query": row.get("effective_compaction_count_at_query"),
            "prediction_full": row.get("task_prediction_full"),
            "correct_full": row.get("task_correct_full"),
            "prediction_compressed": row.get("task_prediction_compressed"),
            "correct_compressed": row.get("task_correct_compressed"),
            "task_transition": row.get("task_transition"),
            "task_status_compressed": row.get("task_status_compressed"),
            "task_error_type_compressed": row.get("task_error_type_compressed"),
            "task_missing": row.get("task_missing"),
        })
    _write_csv(tables / "task_only_all_queries.csv", per_query)

    return {
        "exact": exact,
        "binned": binned,
        "task_rows": task_rows,
        "trajectory": traj,
        "selection": selection,
    }


# --------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------- #
def plot_trajectories(traj: Dict[str, List[dict]], out_dir: Path, max_samples: int = 3) -> List[dict]:
    _plotting()._apply_style()
    figs = []
    specs = [
        ("RAM", "ram_mean", "ram", False),
        ("AORE", "aore_mean", "aore", False),
        ("JS", "js", "js", True),
        ("KL", "kl", "kl", True),
        ("Compression ratio", "compression_ratio", "cr", False),
    ]
    for sample_id in sorted(traj)[:max_samples]:
        rows = traj[sample_id]
        for ylabel, key, slug, sci in specs:
            fig, ax = _plt().subplots(figsize=(5.4, 3.8))
            xs = [r.get("probe_k") for r in rows]
            ys = [r.get(key) for r in rows]
            valid = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if valid:
                vx, vy = zip(*valid)
                ax.plot(vx, vy, "-o")
                if sci:
                    from matplotlib.ticker import ScalarFormatter

                    fmt = ScalarFormatter(useMathText=True)
                    fmt.set_powerlimits((0, 0))
                    ax.yaxis.set_major_formatter(fmt)
            ax.set_xlabel("effective_compaction_count (k)")
            ax.set_ylabel(ylabel)
            qid = rows[0].get("query_id") if rows else ""
            ax.set_title(f"{ylabel} vs effective compactions -- {sample_id}:{qid}")
            ax.grid(True, alpha=0.3)
            cap = (
                f"Longest-run diagnostic probe (same official query prompt at every "
                f"post-effective-compaction state). N points = {len(valid)}. "
                f"Retrospective probe: the model did not know the question at those times."
            )
            figs.append(_plotting()._save(fig, out_dir, f"traj_{slug}_{sample_id}", cap))
    return figs


def plot_trajectory_vs_time(traj: Dict[str, List[dict]], out_dir: Path, max_samples: int = 3) -> List[dict]:
    _plotting()._apply_style()
    figs = []
    for sample_id in sorted(traj)[:max_samples]:
        rows = traj[sample_id]
        fig, axes = _plt().subplots(2, 1, figsize=(6.0, 6.0), sharex=True)
        xs = [r.get("stream_time") for r in rows]
        for ax, key, label in zip(
            axes,
            ["ram_mean", "aore_mean"],
            ["RAM", "AORE"],
        ):
            ys = [r.get(key) for r in rows]
            valid = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if valid:
                vx, vy = zip(*valid)
                ax.plot(vx, vy, "-o")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"Longest-run fidelity vs stream time -- {sample_id}")
        axes[-1].set_xlabel("stream time (s)")
        cap = ("Only the longest query run's own probe events are drawn; "
               "nested runs are NOT overlaid.")
        figs.append(_plotting()._save(fig, out_dir, f"traj_vs_time_{sample_id}", cap))
    return figs


def plot_accuracy(exact: List[dict], out_dir: Path) -> List[dict]:
    _plotting()._apply_style()
    figs = []
    xs = [r["effective_compaction_count"] for r in exact]

    def _valid(rows, key, lo_key=None, hi_key=None):
        out = []
        for r in rows:
            y = r.get(key)
            if y is None:
                continue
            lo = r.get(lo_key) if lo_key else None
            hi = r.get(hi_key) if hi_key else None
            out.append((r["effective_compaction_count"], y,
                        lo if lo is not None else y,
                        hi if hi is not None else y))
        return out

    # FIGURE 6: compressed accuracy
    fig, ax = _plt().subplots(figsize=(5.6, 4.0))
    valid = _valid(exact, "accuracy_compressed",
                   "accuracy_compressed_ci_low", "accuracy_compressed_ci_high")
    if valid:
        vx, vy, vlo, vhi = zip(*valid)
        yerr = [[y - l, h - y] for y, l, h in zip(vy, vlo, vhi)]
        ax.errorbar(vx, vy, yerr=list(zip(*yerr)), marker="o", capsize=4)
        for x, y, r in zip(vx, vy, [r for r in exact if r.get("accuracy_compressed") is not None]):
            ax.annotate(f"Nq={r['num_queries']}\nNs={r['num_samples']}", (x, y),
                        textcoords="offset points", xytext=(5, -12), fontsize=6.5)
    ax.set_xlabel("effective compaction count at query (k)")
    ax.set_ylabel("Compressed accuracy")
    ax.set_title("Compressed accuracy vs effective compaction count")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    figs.append(_plotting()._save(fig, out_dir, "accuracy_compressed_vs_k",
                      "Unit: benchmark query; CI = sample-level cluster bootstrap 95%."))

    # FIGURE 7: full vs compressed
    fig, ax = _plt().subplots(figsize=(5.6, 4.0))
    if valid:
        vx, vy, vlo, vhi = zip(*valid)
        ax.errorbar(vx, vy, yerr=list(zip(*[[y - l, h - y] for y, l, h in zip(vy, vlo, vhi)])),
                    marker="o", capsize=4, label="Compressed")
    full_valid = _valid(exact, "accuracy_full",
                        "accuracy_full_ci_low", "accuracy_full_ci_high")
    if full_valid:
        fx, fy, flo, fhi = zip(*full_valid)
        ax.errorbar(fx, fy, yerr=list(zip(*[[y - l, h - y] for y, l, h in zip(fy, flo, fhi)])),
                    marker="s", capsize=4, label="Full-KV")
    ax.set_xlabel("effective compaction count at query (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Full-KV vs Compressed accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    figs.append(_plotting()._save(fig, out_dir, "accuracy_full_vs_compressed_vs_k",
                      "Full-KV is the difficulty baseline: if both fall together the "
                      "questions simply get harder with k."))

    # FIGURE 8: preservation / correct->wrong
    fig, ax = _plt().subplots(figsize=(5.6, 4.0))
    pres_valid = _valid(exact, "preservation_accuracy",
                        "preservation_ci_low", "preservation_ci_high")
    if pres_valid:
        px, py, plo, phi = zip(*pres_valid)
        ax.errorbar(px, py, yerr=list(zip(*[[y - l, h - y] for y, l, h in zip(py, plo, phi)])),
                    marker="o", capsize=4, label="Preservation accuracy")
    else:
        ax.text(0.5, 0.5, "no Full-correct queries to preserve",
                transform=ax.transAxes, ha="center", va="center", color="0.4")
    c2w_valid = _valid(exact, "correct_to_wrong_rate")
    if c2w_valid:
        cx, cy, _, _ = zip(*c2w_valid)
        ax.plot(cx, cy, "--s", label="Correct→Wrong rate (all queries)")
    ax.set_xlabel("effective compaction count at query (k)")
    ax.set_ylabel("Rate")
    ax.set_title("Compression preservation vs effective compaction count")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    figs.append(_plotting()._save(fig, out_dir, "preservation_vs_k",
                      "Preservation = #(Full correct AND Compressed correct) / #(Full correct). "
                      "Low-N points are exploratory."))
    return figs


# --------------------------------------------------------------------- #
# StreamingBench real: locked-bin accuracy figures + fidelity trajectories
# --------------------------------------------------------------------- #
def _series(rows, key, lo_key=None, hi_key=None):
    out = []
    for r in rows:
        y = r.get(key)
        if y is None:
            continue
        lo = r.get(lo_key) if lo_key else None
        hi = r.get(hi_key) if hi_key else None
        out.append((r["effective_compaction_count"], y,
                    lo if lo is not None else y, hi if hi is not None else y))
    return out


def _err(vals):
    return list(zip(*[[y - l, h - y] for _, y, l, h in vals]))


def plot_delta_and_c2w(exact: List[dict], out_dir: Path) -> List[dict]:
    _plotting()._apply_style()
    figs = []
    fig, ax = _plt().subplots(figsize=(5.6, 4.0))
    xs = [r["effective_compaction_count"] for r in exact]
    delta = [
        (r["accuracy_compressed"] - r["accuracy_full"])
        if (r.get("accuracy_compressed") is not None and r.get("accuracy_full") is not None)
        else None
        for r in exact
    ]
    valid = [(x, d) for x, d in zip(xs, delta) if d is not None]
    if valid:
        vx, vy = zip(*valid)
        ax.plot(vx, vy, "-o")
        ax.axhline(0.0, color="0.7", linestyle="--")
    ax.set_xlabel("effective compaction count at query (k)")
    ax.set_ylabel("ΔAcc = acc_compressed − acc_full")
    ax.set_title("Compressed − Full accuracy")
    ax.grid(True, alpha=0.3)
    figs.append(_plotting()._save(fig, out_dir, "delta_accuracy_vs_k",
                      "Positive = the compressed cache answered MORE queries correctly than Full-KV."))

    fig, ax = _plt().subplots(figsize=(5.6, 4.0))
    c2w = [r.get("correct_to_wrong_rate") for r in exact]
    pres = [r.get("preservation_accuracy") for r in exact]
    if any(v is not None for v in c2w):
        vx = [x for x, v in zip(xs, c2w) if v is not None]
        vy = [v for v in c2w if v is not None]
        ax.plot(vx, vy, "-o", label="Correct→Wrong rate")
    if any(v is not None for v in pres):
        vx = [x for x, v in zip(xs, pres) if v is not None]
        vy = [v for v in pres if v is not None]
        ax.plot(vx, vy, "--s", label="Preservation")
    ax.set_xlabel("effective compaction count at query (k)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    figs.append(_plotting()._save(fig, out_dir, "correct_to_wrong_vs_k",
                      "Correct→Wrong = (Full correct AND Compressed wrong) / (Full correct). "
                      "Preservation = (both correct) / (Full correct)."))
    return figs


def plot_combined_trajectory(sample_id: str, rows: List[dict], out_dir: Path) -> dict:
    _plotting()._apply_style()
    specs = [("ram_mean", "RAM"), ("aore_mean", "AORE"), ("js", "JS"),
             ("kl", "KL"), ("compression_ratio", "CR")]
    fig, axes = _plt().subplots(len(specs), 1, figsize=(6.0, 10.0), sharex=True)
    xs = [int(r.get("probe_k") or 0) for r in rows]
    for ax, (key, label) in zip(axes, specs):
        ys = [r.get(key) for r in rows]
        valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if valid:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, "-o", ms=3)
            if key in ("js", "kl"):
                from matplotlib.ticker import ScalarFormatter

                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(fmt)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"q_star fidelity trajectory — video {sample_id}")
    axes[-1].set_xlabel("effective_compaction_count (k)")
    cap = ("Longest-run diagnostic probe: the SAME official q_star prompt is "
           "replayed after every effective compaction. Retrospective probe, not "
           "the model knowing the question at that time.")
    return _save(fig, out_dir, f"trajectory_{sample_id}", cap)


def plot_fidelity_aggregate(traj: Dict[str, List[dict]], out_dir: Path) -> List[dict]:
    """Median (IQR band) across videos, per exact k."""
    import statistics

    _plotting()._apply_style()
    figs = []
    specs = [("ram_mean", "RAM"), ("aore_mean", "AORE"), ("js", "JS"),
             ("kl", "KL"), ("compression_ratio", "CR")]
    for key, label in specs:
        by_k: Dict[int, List[float]] = {}
        for rows in traj.values():
            for r in rows:
                k = r.get("probe_k")
                v = r.get(key)
                if k is None or v is None:
                    continue
                by_k.setdefault(int(k), []).append(float(v))
        fig, ax = _plt().subplots(figsize=(5.8, 4.0))
        if by_k:
            ks = sorted(by_k)
            med = [statistics.median(by_k[k]) for k in ks]
            lo = [min(by_k[k]) for k in ks]
            hi = [max(by_k[k]) for k in ks]
            ax.fill_between(ks, lo, hi, alpha=0.18, label="min-max across videos")
            ax.plot(ks, med, "-o", ms=3, label="median across videos")
            if key in ("js", "kl"):
                from matplotlib.ticker import ScalarFormatter

                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(fmt)
            ax.legend(fontsize=8)
        ax.set_xlabel("effective_compaction_count (k)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs effective compaction count (all q_star runs)")
        ax.grid(True, alpha=0.3)
        figs.append(_plotting()._save(fig, out_dir, f"fidelity_aggregate_{key.split('_')[0]}",
                          f"Per-k statistics across {len(traj)} q_star trajectories."))
    return figs
