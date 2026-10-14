"""Plot HERMES/InfoKV comparisons in the style of the KV-metrics summary.

This is deliberately CPU-only.  It consumes the frozen StreamingBench RT
``per_query.csv``, ``paired.csv`` and the saved per-sample event JSON files;
it never loads the VLM or changes the running OVO evaluation.

The frozen RT run has one fidelity anchor at the real question for each query
(``detailed_trajectory=False``).  Therefore the six RAM/AORE/JS/KL/Delta plots
aggregate query anchors by effective compaction count.  They are labelled as
query-anchor plots rather than being presented as the repeated read-only probe
trajectory from the older report.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


METHODS = ("hermes", "infokv_stream")
LABEL = {"hermes": "HERMES", "infokv_stream": "InfoKV"}
COLOR = {"hermes": "#2677b8", "infokv_stream": "#dc6b24"}
ROOT = Path(__file__).resolve().parents[1]


def f(value):
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(rows, path):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def video_id(sample_id):
    return sample_id.rsplit("_", 2)[-2]


def grouped(rows, x_field, y_field):
    groups = defaultdict(list)
    for row in rows:
        x, y = f(row.get(x_field)), f(row.get(y_field))
        if x is not None and y is not None:
            groups[int(round(x))].append(y)
    out = []
    for x in sorted(groups):
        values = groups[x]
        out.append({"x": x, "n": len(values), "mean": mean(values),
                    "median": median(values), "min": min(values), "max": max(values),
                    "q10": float(np.quantile(values, .10)),
                    "q90": float(np.quantile(values, .90))})
    return out


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_line(ax, points, label, color, ylabel, *, symlog=False):
    if not points:
        return
    x = np.asarray([p["x"] for p in points])
    y = np.asarray([p["median"] for p in points])
    lo = np.asarray([p["min"] for p in points])
    hi = np.asarray([p["max"] for p in points])
    ax.plot(x, y, marker="o", markersize=3.4, linewidth=1.5, color=color,
            label=label, zorder=3)
    ax.fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0,
                    label=f"{label} min–max")
    if symlog:
        ax.set_yscale("symlog", linthresh=.01)
    ax.set_xlabel("Effective compaction count (k)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=.25)


def load_token_events(root, sample_id):
    rows = {}
    for method in METHODS:
        path = root / method / "samples" / f"{sample_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if method == "hermes":
            events = data.get("physical_compactions", [])
            out = []
            for e in events:
                out.append({
                    "method": method,
                    "event": int(e["compaction_index"]),
                    "stream_time": f(e.get("stream_time")),
                    "effective": bool(e.get("is_effective_compaction", True)),
                    "visual_before": int(e["managed_visual_tokens_before"]),
                    "visual_after": int(e["managed_visual_tokens_after"]),
                    "cache_before": int(e["kv_before"]),
                    "cache_after": int(e["kv_after"]),
                    "removed": int(e["managed_visual_tokens_before"] - e["managed_visual_tokens_after"]),
                })
        else:
            events = data.get("infokv_events", [])
            out = []
            for e in events:
                before = e.get("cache_lens_before") or []
                after = e.get("cache_lens_after") or []
                out.append({
                    "method": method,
                    "event": int(e["trigger_index"]),
                    "stream_time": None,
                    "effective": bool(e.get("effective", False)),
                    "visual_before": int(e["visual_before"]),
                    "visual_after": int(e["visual_after"]),
                    "cache_before": max(map(int, before)) if before else None,
                    "cache_after": max(map(int, after)) if after else None,
                    "removed": int(e.get("evicted", 0)),
                })
        rows[method] = out
    effective_chunk_indices = [e['event'] for e in rows['infokv_stream'] if e['effective']]
    if len(effective_chunk_indices) != len(rows['hermes']):
        raise ValueError('Example replay has unequal effective schedules')
    for event, chunk_index in zip(rows['hermes'], effective_chunk_indices):
        event['chunk_index'] = chunk_index
    for event in rows['infokv_stream']:
        event['chunk_index'] = event['event']
    return rows


def write_token_csv(rows, path):
    fields = ["method", "event", "chunk_index", "stream_time", "effective", "visual_before",
              "visual_after", "cache_before", "cache_after", "removed"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    analysis = root / "analysis"
    out = root / "figures" / "doc_style_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(analysis / "per_query.csv")
    method_rows = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    full = {r["sample_id"]: r for r in rows if r["method"] == "full_cache"}

    # Add option-level shifts relative to the same Full query.  The full run is
    # only a reference; no new model inference is performed here.
    shifts = {m: [] for m in METHODS}
    for m in METHODS:
        for r in method_rows[m]:
            base = full.get(r["sample_id"], {})
            x = f(r.get("effective_compaction_count"))
            pg, pg0 = f(r.get("p_gold")), f(base.get("p_gold"))
            mg, mg0 = f(r.get("margin")), f(base.get("margin"))
            if x is None:
                continue
            shifts[m].append({"effective_compaction_count": x,
                              "delta_p_gold": None if pg is None or pg0 is None else pg - pg0,
                              "delta_margin": None if mg is None or mg0 is None else mg - mg0})

    specs = [
        ("ram_mean", "RAM", False),
        ("aore_mean", "AORE", True),
        ("js", "JS divergence", False),
        ("kl", "Forward KL", False),
    ]
    all_points = {}
    for field, title, symlog in specs:
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        for method in METHODS:
            points = grouped(method_rows[method], "effective_compaction_count", field)
            all_points[(method, field)] = points
            plot_line(ax, points, LABEL[method], COLOR[method], title, symlog=symlog)
        ax.set_title(f"{title} vs effective compaction count\nStreamingBench RT query anchors (2K, 20 videos)")
        if ax.lines:
            ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        save(fig, out, f"{field}_vs_effective_compaction")

    delta_specs = [("delta_p_gold", r"$\Delta P_{gold}$"),
                   ("delta_margin", r"$\Delta M_z$")]
    for field, title in delta_specs:
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        for method in METHODS:
            points = grouped(shifts[method], "effective_compaction_count", field)
            all_points[(method, field)] = points
            for statistic, style in [('mean', '-'), ('median', '--')]:
                ax.plot([p['x'] for p in points], [p[statistic] for p in points],
                        marker='o', markersize=3, linestyle=style, color=COLOR[method],
                        label=f'{LABEL[method]} {statistic}')
        ax.set_xlabel('Effective compaction count (k)')
        ax.set_ylabel(title)
        ax.grid(True, alpha=.25)
        ax.axhline(0, color="#555", linewidth=.8, linestyle="--")
        ax.set_title(f"{title} vs effective compaction count\nrelative to Full-KV, matched query anchors")
        if ax.lines:
            ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        save(fig, out, f"{field}_vs_effective_compaction")

    # One six-panel figure matching the document's compact indicator layout.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panel_specs = [("ram_mean", "RAM", False), ("aore_mean", "AORE", True),
                   ("js", "JS divergence", False), ("kl", "Forward KL", False),
                   ("delta_p_gold", r"$\Delta P_{gold}$", False),
                   ("delta_margin", r"$\Delta M_z$", False)]
    for ax, (field, title, symlog) in zip(axes.flat, panel_specs):
        for method in METHODS:
            points = all_points[(method, field)]
            if field.startswith('delta_'):
                for statistic, style in [('mean', '-'), ('median', '--')]:
                    ax.plot([p['x'] for p in points], [p[statistic] for p in points],
                            marker='o', markersize=3, linestyle=style, color=COLOR[method],
                            label=f'{LABEL[method]} {statistic}')
                ax.set_xlabel('Effective compaction count (k)')
                ax.set_ylabel(title)
                ax.grid(True, alpha=.25)
            else:
                plot_line(ax, points, LABEL[method], COLOR[method], title, symlog=symlog)
        if field in ("delta_p_gold", "delta_margin"):
            ax.axhline(0, color="#555", linewidth=.8, linestyle="--")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("HERMES vs InfoKV | StreamingBench RT | 2K visual budget\n"
                 "query-anchor aggregates by effective compaction count")
    fig.tight_layout()
    save(fig, out, "doc_style_indicator_comparison")

    # Accuracy and Full-relative transition plots (document figures 7/8 style).
    acc = {m: mean(float(r["correct"]) for r in method_rows[m]) for m in METHODS}
    acc["full_cache"] = mean(float(r["correct"]) for r in rows if r["method"] == "full_cache")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["Full", "HERMES", "InfoKV"]
    vals = [acc["full_cache"], acc["hermes"], acc["infokv_stream"]]
    axes[0].plot(labels, vals, marker="o", color="#2677b8")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Task accuracy")
    axes[0].grid(True, alpha=.25)
    transitions = []
    for method in METHODS:
        counts = {x: 0 for x in ("CC", "CW", "WC", "WW")}
        for sid, r in full.items():
            comp = next((x for x in method_rows[method] if x["sample_id"] == sid), None)
            if comp is None: continue
            counts[("C" if int(r["correct"]) else "W") + ("C" if int(comp["correct"]) else "W")] += 1
        transitions.append(counts)
    x = np.arange(2); bottom = np.zeros(2)
    for key, color in [("CC", "#2878b5"), ("CW", "#e67e22"), ("WC", "#55a868"), ("WW", "#c44e52")]:
        values = [d[key] for d in transitions]
        axes[1].bar(x, values, bottom=bottom, label=key, color=color)
        bottom += np.asarray(values)
    axes[1].set_xticks(x, ["HERMES", "InfoKV"]); axes[1].set_ylabel("Number of queries")
    axes[1].set_title("Full → compressed decision transitions")
    axes[1].legend(frameon=False, ncol=4, fontsize=8)
    fig.suptitle("Task layer comparison on the same 100 matched queries")
    fig.tight_layout()
    save(fig, out, "accuracy_and_transitions")

    # Compression exposure and retained tokens at the query, document figure 9
    # analogue without inventing a cross-budget experiment.
    summary = {r["method"]: r for r in read_csv(analysis / "summary.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(2)
    trig = [f(summary[m].get("mean_compression_trigger_count")) for m in METHODS]
    eff = [f(summary[m].get("mean_effective_compaction_count")) for m in METHODS]
    axes[0].bar(x-.18, trig, .36, label="Trigger", color="#7aa6c2")
    axes[0].bar(x+.18, eff, .36, label="Effective", color="#dc6b24")
    axes[0].set_xticks(x, [LABEL[m] for m in METHODS]); axes[0].set_ylabel("Count / query")
    axes[0].set_title("Compression exposure"); axes[0].legend(frameon=False)
    kept = [f(summary[m].get("median_kv_bytes_at_query")) / 1024**2 for m in METHODS]
    axes[1].bar(x, kept, color=[COLOR[m] for m in METHODS])
    axes[1].set_xticks(x, [LABEL[m] for m in METHODS]); axes[1].set_ylabel("KV at query (MiB)")
    axes[1].set_title("Median physical KV at query")
    fig.suptitle("Compression exposure and physical KV | 2K visual budget")
    fig.tight_layout()
    save(fig, out, "compression_exposure_and_kv")

    # One concrete token trajectory, including InfoKV's non-effective periodic
    # triggers.  This is also written as a CSV for inspection.
    sid = "Real-Time Visual Understanding_sample_23_5"
    token = load_token_events(root, sid)
    token_rows = token["hermes"] + token["infokv_stream"]
    write_token_csv(token_rows, out / "token_changes_video23_5.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for method in METHODS:
        ev = token[method]
        x = np.asarray([e["chunk_index"] for e in ev])
        before = np.asarray([e["visual_before"] for e in ev])
        after = np.asarray([e["visual_after"] for e in ev])
        axes[0].plot(x, before, color=COLOR[method], alpha=.55, linewidth=1,
                     linestyle="--", label=f"{LABEL[method]} before")
        axes[0].plot(x, after, color=COLOR[method], linewidth=1.7,
                     marker="o", markersize=2.8, label=f"{LABEL[method]} after")
        effective = np.asarray([e["effective"] for e in ev], dtype=bool)
        axes[0].scatter(x[effective], after[effective], s=24, color=COLOR[method], zorder=4)
        cb, ca = [e["cache_before"] for e in ev], [e["cache_after"] for e in ev]
        axes[1].plot(x, cb, color=COLOR[method], alpha=.55, linewidth=1, linestyle="--")
        axes[1].plot(x, ca, color=COLOR[method], linewidth=1.7, marker="o", markersize=2.8,
                     label=LABEL[method])
    axes[0].set_ylabel("Managed visual tokens")
    axes[0].set_title("Video 23, query at 258 s: visual-token count before/after")
    axes[0].grid(True, alpha=.25); axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("Common incoming chunk index"); axes[1].set_ylabel("Total cache length")
    axes[1].set_title("Same events including the 14-token protected prefix")
    axes[1].grid(True, alpha=.25); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Token changes during one identical video replay")
    fig.tight_layout()
    save(fig, out, "token_changes_video23_5")

    # Save the grouped data used by all indicator plots.
    rows_out = []
    for (method, field), points in all_points.items():
        for p in points:
            rows_out.append({"method": method, "metric": field, **p})
    write_csv(rows_out, out / "comparison_by_effective_compaction.csv")

    report = out / "README_zh.md"
    report.write_text(
        "# HERMES / InfoKV 文档风格对比图\n\n"
        "这些图从冻结的 StreamingBench RT 20 视频、100 个匹配问题的既有 JSON/CSV 生成，没有重新推理，也没有改变模型、采样、预算或问题集合。颜色和布局参照《KV缓存压缩行为评价指标》中的折线、范围带和堆叠柱图。\n\n"
        "当前 RT 运行保存的是每个真实问题到达时的一个 fidelity anchor（`detailed_trajectory=False`），所以 RAM/AORE/JS/KL/ΔP/ΔM 图按 effective compaction count 对问题 anchor 做中位数和 min–max 聚合；它们不能冒充文档中每次 compaction 后 read-only probe 的 repeated-compaction trajectory。若要完全复现该轨迹，需要单独开启 detailed trajectory 的 GPU 重放。\n\n"
        "ΔP_gold 和 ΔM_z 均按同一 sample_id 的 Full-KV 结果作 reference，分别计算压缩方法值减 Full 值。`token_changes_video23_5.csv` 展示视频 23、258 秒问题的一次完整回放：HERMES 55 次均有效；InfoKV 66 次周期触发，其中 55 次实际淘汰 token。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), "plots": len(list(out.glob("*.png"))),
                      "token_example": sid}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT / "outputs/InfoKVStream/streamingbench_rt20_20261013")
    main(parser.parse_args().root)
