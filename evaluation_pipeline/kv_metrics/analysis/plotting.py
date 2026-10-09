"""Publication-quality figures for the fidelity experiments.

Conventions (enforced here):
* matplotlib only, no seaborn
* single axes per figure, no 3D
* 300 DPI PNG + matching PDF
* every figure records N samples / N events and the statistic definition
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 110,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def _save(fig, out_dir: Path, name: str, caption: str, dpi: int = 300) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.005, caption, fontsize=7.5, color="0.35", va="bottom")
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return {"name": name, "png": str(png), "pdf": str(pdf), "caption": caption}


def _finite(values: Sequence[Optional[float]]) -> List[float]:
    return [
        float(v)
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]


# --------------------------------------------------------------------- #
# Experiment 1: compression ratio vs fidelity
# --------------------------------------------------------------------- #
def plot_compression_fidelity(points: List[dict], out_dir: Path) -> List[dict]:
    """``points``: one dict per experiment with keys
    name, x (mean compression ratio), metric stats for ram/aore/js
    (each = {"mean","ci_low","ci_high","n"}), plus x_err_low/high."""

    _apply_style()
    figures = []
    specs = [
        ("ram", "RAM", "compression_vs_ram"),
        ("aore", "AORE", "compression_vs_aore"),
        ("js", "JS divergence", "compression_vs_js"),
    ]
    for key, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        xs, ys, yerrs, labels = [], [], [], []
        total_samples = 0
        total_events = 0
        for point in points:
            stat = point.get(key) or {}
            if stat.get("mean") is None:
                continue
            xs.append(point["x"])
            ys.append(stat["mean"])
            lo = stat.get("ci_low")
            hi = stat.get("ci_high")
            yerrs.append([
                (stat["mean"] - lo) if lo is not None else 0.0,
                (hi - stat["mean"]) if hi is not None else 0.0,
            ])
            labels.append(point["name"])
            total_samples = max(total_samples, stat.get("n") or 0)
            total_events += point.get("n_events") or 0
        if xs:
            yerr = list(zip(*yerrs)) if yerrs else None
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, linestyle="-")
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=7.5)
        ax.set_xlabel("Mean compression ratio (full / compressed KV bytes)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Compression ratio vs {ylabel}")
        ax.grid(True, alpha=0.3)
        caption = (
            f"Unit: sample. Each point = one experiment/budget "
            f"(x = sample-mean compression ratio). Error bar = bootstrap 95% CI "
            f"of the sample-mean. max N samples = {total_samples}, "
            f"total events = {total_events}. Compression-triggering samples only."
        )
        figures.append(_save(fig, out_dir, fname, caption))
    return figures


# --------------------------------------------------------------------- #
# Experiment 2: first-compaction PRE vs POST
# --------------------------------------------------------------------- #
def plot_pre_post(rows: List[dict], out_dir: Path) -> List[dict]:
    _apply_style()
    figures = []
    specs = [
        ("kl", "KL (forward)", "pre_post_kl"),
        ("js", "JS divergence", "pre_post_js"),
        ("aore", "AORE", "pre_post_aore"),
        ("ram", "RAM", "pre_post_ram"),
    ]
    for key, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        n = 0
        deltas = []
        for row in rows:
            pre = row.get(f"{key}_pre")
            post = row.get(f"{key}_post")
            if pre is None or post is None:
                continue
            n += 1
            deltas.append(post - pre)
            ax.plot([0, 1], [pre, post], marker="o", alpha=0.6, linewidth=1.0)
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            ax.plot([0, 1], [0, 0], color="0.8", zorder=0)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["PRE\n(before 1st compaction)", "POST\n(after 1st compaction)"])
            ax.text(0.5, 0.02, f"mean Δ = {mean_delta:+.4g}  (N={n})",
                    transform=ax.transAxes, ha="center", fontsize=9)
        else:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["PRE\n(before 1st compaction)", "POST\n(after 1st compaction)"])
            ax.text(
                0.5, 0.5,
                "N = 0 paired samples\nno sample has queries on both sides\n"
                "of its first compaction (needs probe anchors)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.35",
            )
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}: PRE vs POST first compaction")
        ax.grid(True, alpha=0.3)
        caption = (
            f"Within-sample paired comparison. PRE = queries at query_time < "
            f"first_compaction_stream_time; POST = >=. Only samples with both "
            f"sides participate. Unit: sample (sample-level mean first). N = {n}."
        )
        figures.append(_save(fig, out_dir, fname, caption))
    return figures


def plot_pre_post_delta(rows: List[dict], out_dir: Path) -> List[dict]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    specs = ["kl", "js", "aore", "ram"]
    data, labels = [], []
    for key in specs:
        deltas = [
            row[f"{key}_post"] - row[f"{key}_pre"]
            for row in rows
            if row.get(f"{key}_pre") is not None and row.get(f"{key}_post") is not None
        ]
        if deltas:
            data.append(deltas)
            labels.append(key.upper())
    caption = (
        f"Δ = POST - PRE, unit: sample. Box = IQR, points = individual samples. "
        f"N = {[len(d) for d in data]}."
    )
    if data:
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=False)
        for i, deltas in enumerate(data, start=1):
            xs = [i + (j % 5 - 2) * 0.02 for j in range(len(deltas))]
            ax.scatter(xs, deltas, s=14, alpha=0.7, zorder=3)
        ax.axhline(0.0, color="0.8", zorder=0)
    ax.set_ylabel("Δ metric (POST - PRE)")
    ax.set_title("Paired Δ distribution after first compaction")
    ax.grid(True, alpha=0.3)
    return [_save(fig, out_dir, "pre_post_delta", caption)]


# --------------------------------------------------------------------- #
# Experiment 3: repeated compaction drift
# --------------------------------------------------------------------- #
def plot_repeated_compaction(points: List[dict], out_dir: Path) -> List[dict]:
    _apply_style()
    figures = []
    specs = [
        ("aore", "AORE", "repeated_compaction_aore"),
        ("js", "JS divergence", "repeated_compaction_js"),
        ("ram", "RAM", "repeated_compaction_ram"),
        ("kl", "KL (forward)", "repeated_compaction_kl"),
    ]
    for key, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        xs = [p["bin_label"] for p in points]
        ys = [p.get(key) for p in points]
        los = [p.get(f"{key}_ci_low") for p in points]
        his = [p.get(f"{key}_ci_high") for p in points]
        ns = [p.get(f"{key}_n") or 0 for p in points]
        valid = [
            (x, y, lo, hi, n)
            for x, y, lo, hi, n in zip(xs, ys, los, his, ns)
            if y is not None
        ]
        if valid:
            vx, vy, vlo, vhi, vn = zip(*valid)
            yerr = [
                [(y - lo) if lo is not None else 0.0, (hi - y) if hi is not None else 0.0]
                for y, lo, hi in zip(vy, vlo, vhi)
            ]
            ax.errorbar(range(len(vx)), vy, yerr=list(zip(*yerr)), marker="o", capsize=4)
            ax.set_xticks(range(len(vx)))
            ax.set_xticklabels(vx, rotation=30, ha="right")
            for i, n in enumerate(vn):
                ax.annotate(f"N={n}", (i, vy[i]), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7.5)
        ax.set_xlabel("Compaction-count bin")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Repeated compaction: {ylabel}(n)")
        ax.grid(True, alpha=0.3)
        caption = (
            "Unit: sample (events -> sample mean -> cross-sample bootstrap 95% CI). "
            "Bin label = number of compactions before the query anchor. "
            "N = number of samples contributing to the bin."
        )
        figures.append(_save(fig, out_dir, fname, caption))
    return figures


# --------------------------------------------------------------------- #
# Experiment 4: temporal RAM curve
# --------------------------------------------------------------------- #
def plot_temporal_ram(points: List[dict], out_dir: Path) -> List[dict]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    labels = [p["bin_label"] for p in points]
    ys = [p.get("mean") for p in points]
    los = [p.get("ci_low") for p in points]
    his = [p.get("ci_high") for p in points]
    ns = [p.get("n") or 0 for p in points]
    valid = [
        (i, y, lo, hi, n)
        for i, (y, lo, hi, n) in enumerate(zip(ys, los, his, ns))
        if y is not None
    ]
    if valid:
        idx, vy, vlo, vhi, vn = zip(*valid)
        yerr = [
            [(y - lo) if lo is not None else 0.0, (hi - y) if hi is not None else 0.0]
            for y, lo, hi in zip(vy, vlo, vhi)
        ]
        ax.errorbar(idx, vy, yerr=list(zip(*yerr)), marker="o", capsize=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0)
        for i, y, n in zip(idx, vy, vn):
            ax.annotate(f"N={n}", (i, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.5)
    ax.set_xlabel("History age bucket (seconds before query)")
    ax.set_ylabel("Temporal RAM")
    ax.set_title("Temporal fidelity: RAM vs KV age")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    caption = (
        "Bucket-conditioned RAM = retained full-attention mass / total "
        "full-attention mass inside the bucket. Unit: sample (bootstrap 95% CI). "
        "Bins with no full-attention mass are excluded (not zero)."
    )
    return [_save(fig, out_dir, "temporal_ram_curve", caption)]


# --------------------------------------------------------------------- #
# Experiment 5: age vs retention probability
# --------------------------------------------------------------------- #
def plot_age_retention(series: List[dict], out_dir: Path) -> List[dict]:
    """``series``: one entry per experiment: {name, points:[{bin,total,kept,rate}]}."""

    _apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    cmap = plt.get_cmap("viridis")
    usable = [s for s in series if any(p.get("total") for p in s["points"])]
    for i, entry in enumerate(usable):
        color = cmap(i / max(1, len(usable) - 1)) if len(usable) > 1 else None
        labels = [p["bin_label"] for p in entry["points"]]
        rates = [p.get("rate") for p in entry["points"]]
        ax.plot(range(len(labels)), rates, marker="o", label=entry["name"], color=color)
    if usable:
        ax.set_xticks(range(len(usable[0]["points"])))
        ax.set_xticklabels([p["bin_label"] for p in usable[0]["points"]])
        labels = [
            f"{s['name']} (evicting compactions={s.get('n_events', '?')})"
            for s in usable
        ]
        ax.legend(labels)
    ax.set_xlabel("KV age bucket (seconds before compaction)")
    ax.set_ylabel("Retention probability (kept / total)")
    ax.set_title("HERMES retention probability by KV age")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    caption = (
        "Token-level retention of HERMES-managed visual KV, pooled per "
        "experiment. rate = kept_count / total_count within the age bucket, "
        "pooled over compaction events that actually evicted tokens "
        "(no-op compactions excluded). Not attention-weighted."
    )
    return [_save(fig, out_dir, "age_vs_retention", caption)]


# --------------------------------------------------------------------- #
# Experiment 6: system efficiency
# --------------------------------------------------------------------- #
def plot_system_efficiency(points: List[dict], out_dir: Path) -> List[dict]:
    _apply_style()
    figures = []

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [p.get("compression_ratio") for p in points]
    ys = [p.get("mean_wallclock") for p in points]
    valid = [(x, y, p["name"]) for x, y, p in zip(xs, ys, points) if x and y]
    if valid:
        vx, vy, names = zip(*valid)
        ax.scatter(vx, vy, s=30)
        for x, y, nm in valid:
            ax.annotate(nm, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("Mean compression ratio")
    ax.set_ylabel("Mean compaction wall-clock (s)")
    ax.set_title("Compression ratio vs compaction wall-clock")
    ax.grid(True, alpha=0.3)
    figures.append(_save(
        fig, out_dir, "compression_vs_wallclock",
        "Unit: sample. Wall-clock measured around the real "
        "apply_kv_cache_pruning_strict call with cuda.synchronize (profiling only).",
    ))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [p.get("compression_ratio") for p in points]
    ys = [p.get("mean_rho") for p in points]
    valid = [(x, y, p["name"]) for x, y, p in zip(xs, ys, points) if x and y]
    if valid:
        vx, vy, names = zip(*valid)
        ax.scatter(vx, vy, s=30)
        if any(v >= 1.0 for v in vy):
            ax.axhline(1.0, color="0.8", linestyle="--")
        for x, y, nm in valid:
            ax.annotate(nm, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("Mean compression ratio")
    ax.set_ylabel("rho_realtime (T_compact / T_trigger_interval)")
    ax.set_title("Compression ratio vs real-time ratio")
    ax.grid(True, alpha=0.3)
    figures.append(_save(
        fig, out_dir, "compression_vs_rho",
        "rho_realtime < 1 means one compaction fits before the next trigger. "
        "Trigger interval uses video stream time, not CPU wall time.",
    ))
    return figures


# --------------------------------------------------------------------- #
# Experiment 7: fidelity vs task accuracy
# --------------------------------------------------------------------- #
def plot_metric_vs_task(bins: Dict[str, List[dict]], groups: Dict[str, dict], out_dir: Path) -> List[dict]:
    _apply_style()
    figures = []

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for metric in ("ram", "aore", "js", "kl"):
        rows = bins.get(metric) or []
        xs = [r["bin_label"] for r in rows]
        ys = [r.get("accuracy") for r in rows]
        ns = [r.get("n") or 0 for r in rows]
        if any(y is not None for y in ys):
            ax.plot(range(len(rows)), ys, marker="o", label=metric.upper())
            for i, (y, n) in enumerate(zip(ys, ns)):
                if y is not None:
                    ax.annotate(f"N={n}", (i, y), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=7)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels(xs, rotation=25, ha="right", fontsize=8)
    ax.set_xlabel("Metric quintile bin (Q1 -> Q5)")
    ax.set_ylabel("Task accuracy")
    ax.set_title("Fidelity metric vs task accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    figures.append(_save(
        fig, out_dir, "metric_vs_task_accuracy",
        "Unit: query/event. Accuracy per metric quintile (Q1=lowest metric "
        "value). Association only, not causation.",
    ))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    data, labels = [], []
    for metric in ("ram", "aore", "js", "kl"):
        entry = groups.get(metric) or {}
        correct = entry.get("correct") or []
        incorrect = entry.get("incorrect") or []
        if correct or incorrect:
            # two panels per metric: correct then incorrect
            data.append(correct)
            labels.append(f"{metric.upper()}\ncorrect")
            data.append(incorrect)
            labels.append(f"{metric.upper()}\nincorrect")
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Metric value")
    ax.set_title("Correct vs incorrect task outcomes")
    ax.grid(True, alpha=0.3)
    figures.append(_save(
        fig, out_dir, "correct_vs_incorrect_metrics",
        "Unit: query/event. Distribution of each fidelity metric split by task "
        "correctness.",
    ))
    return figures


# --------------------------------------------------------------------- #
# Experiment 8: case timelines
# --------------------------------------------------------------------- #
def plot_case_timeline(video_id: str, events: List[dict], compactions: List[dict], out_dir: Path) -> dict:
    _apply_style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    cx = [c.get("stream_time") for c in compactions if c.get("stream_time") is not None]
    cy = [c.get("kv_after") for c in compactions if c.get("kv_after") is not None]
    if cx:
        ax.plot(cx, cy, marker=".", linestyle="-", color="0.6", label="KV size after compaction")
    qx = [e.get("query_time") for e in events if e.get("query_time") is not None]
    for metric, marker in (("ram_mean", "o"), ("aore_mean", "s"), ("js", "^")):
        ys = [e.get(metric) for e in events]
        valid = [(x, y) for x, y in zip(qx, ys) if x is not None and y is not None]
        if valid:
            vx, vy = zip(*valid)
            ax2.plot(vx, vy, marker=marker, linestyle="none", label=metric)
    if cx:
        ax2.vlines(cx, 0, 1, transform=ax2.get_xaxis_transform(), color="0.85", zorder=0)
    ax.set_ylabel("KV size (tokens)")
    ax.set_title(f"Case timeline: {video_id}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2.set_xlabel("Video stream time (s)")
    ax2.set_ylabel("Fidelity metric")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    caption = (
        f"Grey verticals = compaction events; points = query anchors. "
        f"events={len(events)}, compactions={len(compactions)}. "
        f"Missing metrics are omitted, never interpolated."
    )
    return _save(fig, out_dir, f"case_{video_id}_timeline", caption)


# --------------------------------------------------------------------- #
# Single-video Query x compaction-count trajectory
# --------------------------------------------------------------------- #
def _sci_axis(ax):
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)


def plot_query_fidelity_vs_compaction_count(
    rows: List[dict],
    meta: dict,
    out_dir: Path,
) -> dict:
    """Main 4-panel figure: compaction_count_at_query vs RAM/AORE/JS/KL."""

    _apply_style()
    panel_specs = [
        ("RAM", "RAM", True),
        ("AORE", "AORE", False),
        ("JS_display", r"JS divergence", False),
        ("KL", r"KL (forward)", False),
    ]
    xs = [r.get("compaction_count_at_query") for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(6.0, 9.0), sharex=True)
    for ax, (key, ylabel, cap_one) in zip(axes, panel_specs):
        ys = [r.get(key) for r in rows]
        valid = [
            (x, y, r.get("query_id"))
            for x, y, r in zip(xs, ys, rows)
            if x is not None and y is not None
        ]
        if valid:
            vx, vy, vids = zip(*valid)
            # connect in query-time order (rows are already time-sorted)
            ax.plot(vx, vy, "-", color="0.7", linewidth=1.0, zorder=1)
            ax.scatter(vx, vy, s=32, zorder=2)
            for x, y, vid in valid:
                ax.annotate(str(vid), (x, y), textcoords="offset points",
                            xytext=(5, 4), fontsize=6.5, color="0.35")
            if key in ("JS_display", "KL"):
                _sci_axis(ax)
        if cap_one:
            ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f"Fidelity vs cumulative compactions -- sample {meta.get('sample_id')}",
        fontsize=12,
    )
    info = (
        f"N queries = {meta.get('n_queries')}   "
        f"total compactions = {meta.get('total_compactions')}   "
        f"budget = {meta.get('budget')}   "
        f"mean CR = {_fmt_num(meta.get('mean_compression_ratio'))}"
    )
    axes[0].text(0.01, 0.97, info, transform=axes[0].transAxes,
                 fontsize=8, va="top", ha="left",
                 bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
    axes[-1].set_xlabel("compaction_count_at_query (completed compactions before the query)")
    caption = (
        "Each point = one real query. x = number of HERMES compactions completed "
        "before that query's anchor. Points are connected in query-time order; "
        "queries sharing a compaction count are all plotted. "
        "This is cumulative operational fidelity, NOT a per-compaction causal effect."
    )
    name = f"query_fidelity_vs_compaction_count_{meta.get('sample_id')}"
    return _save(fig, out_dir, name, caption)


def plot_query_fidelity_normalized(rows: List[dict], meta: dict, out_dir: Path) -> dict:
    """Supplementary shape-only view (explicitly NOT the raw metrics)."""

    _apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xs = [r.get("compaction_count_at_query") for r in rows]
    specs = [("RAM", "RAM"), ("AORE", "AORE"), ("JS_display", "JS"), ("KL", "KL")]
    for key, label in specs:
        values = [r.get(key) for r in rows]
        pairs = [(x, v) for x, v in zip(xs, values) if x is not None and v is not None]
        if len(pairs) < 2:
            continue
        first = pairs[0][1]
        span = max(abs(max(v for _, v in pairs) - min(v for _, v in pairs)), 1e-12)
        ax.plot(
            [x for x, _ in pairs],
            [(v - first) / span for _, v in pairs],
            marker="o", label=label,
        )
    ax.set_xlabel("compaction_count_at_query")
    ax.set_ylabel("(value - first) / range   [shape only]")
    ax.set_title(f"Normalized trend (supplementary, sample {meta.get('sample_id')})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    caption = (
        "SUPPLEMENTARY / SHAPE-ONLY: each metric is rescaled to "
        "(value - first) / range purely to compare trends. "
        "These are NOT the raw metric values; use the main 4-panel figure."
    )
    name = f"query_fidelity_normalized_trend_{meta.get('sample_id')}"
    return _save(fig, out_dir, name, caption)


def plot_query_compaction_timeline(
    rows: List[dict],
    compactions: List[dict],
    meta: dict,
    out_dir: Path,
) -> dict:
    """Timeline putting the trajectory into video-time context."""

    _apply_style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ctimes = [c.get("trigger_time", c.get("stream_time")) for c in compactions]
    ckv = [c.get("kv_after") for c in compactions]
    pairs = [(t, k, i + 1) for i, (t, k) in enumerate(zip(ctimes, ckv)) if t is not None]
    if pairs:
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "-o",
                ms=3, color="0.6", label="KV size after compaction")
    qtimes = [r.get("query_time") for r in rows]
    qkv = [r.get("current_kv_bytes") for r in rows]
    qpairs = [(t, k) for t, k in zip(qtimes, qkv) if t is not None and k is not None]
    if qpairs:
        ax.scatter([p[0] for p in qpairs], [p[1] for p in qpairs],
                   marker="*", s=90, color="C3", zorder=3, label="query anchor")
    for q in rows:
        if q.get("query_time") is not None:
            ax2.axvline(q["query_time"], color="0.9", zorder=0)
    for c in compactions:
        t = c.get("trigger_time", c.get("stream_time"))
        if t is not None:
            ax.axvline(t, color="0.85", linewidth=0.6, zorder=0)
    ax.set_ylabel("KV tokens at anchor")
    ax.set_title(f"Query / compaction timeline (sample {meta.get('sample_id')})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    cr = [(r.get("query_time"), r.get("CR")) for r in rows if r.get("CR") is not None]
    if cr:
        ax2.plot([p[0] for p in cr], [p[1] for p in cr], "-o", color="C0")
    ax2.set_xlabel("video stream time (s)")
    ax2.set_ylabel("compression ratio at query")
    ax2.grid(True, alpha=0.3)
    caption = (
        f"Grey verticals = compaction events; stars = query anchors. "
        f"N queries = {meta.get('n_queries')}, total compactions = "
        f"{meta.get('total_compactions')}."
    )
    name = f"query_compaction_timeline_{meta.get('sample_id')}"
    return _save(fig, out_dir, name, caption)


def _fmt_num(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
