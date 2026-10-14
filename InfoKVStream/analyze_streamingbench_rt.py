"""Tables and publication-ready figures for the frozen RT diagnostic set."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

METHODS = ("full_cache", "hermes", "infokv_stream")
LABELS = {"full_cache": "Full", "hermes": "HERMES", "infokv_stream": "InfoKV"}
COLORS = {"full_cache": "#7f8c8d", "hermes": "#2677b8", "infokv_stream": "#dc6b24"}
FIDELITY = ("ram_mean", "aore_mean", "js", "kl", "compression_ratio")
RUNTIME = ("compression_trigger_count", "effective_compaction_count", "kv_bytes_at_query",
           "stream_processing_sec", "stream_fps", "qa_sec", "physical_pruning_total_sec",
           "infokv_management_total_sec", "peak_gpu_allocated_mb_with_diagnostics")
DECISION = ("p_gold", "margin")
SUPPORT = ("ram_valid_heads", "ram_undefined_heads")


def decision_values(record):
    from evaluation_pipeline.kv_metrics.analysis.decision_fidelity import option_scores, decision_margin
    logits = record.get("option_token_logits") or {}
    option_ids = record.get("option_token_ids") or {}
    if not option_ids or any(str(i) not in logits or not math.isfinite(logits[str(i)])
                             for ids in option_ids.values() for i in ids):
        return {"p_gold": None, "margin": None}
    probs, best = option_scores(logits, option_ids)
    gold = str(record["ground_truth"]).strip().upper()
    return {"p_gold": probs.get(gold), "margin": decision_margin(best, gold)}


def video_id(sid):
    return sid.rsplit("_", 2)[-2]


def cluster_ci(rows, field, n_boot=2000):
    groups = {}
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            groups.setdefault(row["video_id"], []).append(float(value))
    if not groups:
        return None, None, None
    values = np.array([mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(42)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cluster_median_ci(rows, field, n_boot=2000):
    groups = {}
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            groups.setdefault(row["video_id"], []).append(float(value))
    if not groups:
        return None, None
    groups = [np.array(v) for _, v in sorted(groups.items())]
    rng = np.random.default_rng(42)
    draws = [np.median(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
             for _ in range(n_boot)]
    low, high = np.quantile(draws, [.025, .975])
    return float(low), float(high)


def analyze(root):
    manifest = json.loads((root / "manifest.json").read_text())
    expected = set(manifest["sample_ids"])
    records = {}
    all_rows = []
    for method in METHODS:
        arm = {}
        for path in (root / method / "samples").glob("*.json"):
            record = json.loads(path.read_text())["record"]
            sid = record["sample_id"]
            if sid not in expected or record["contract_id"] != manifest["contract_id"]:
                raise RuntimeError(f"Unexpected source/sample in {path}")
            if record["correct"] is None:
                raise RuntimeError(f"Missing official correctness in {path}")
            arm[sid] = record
            flat = {"method": method, "sample_id": sid, "video_id": video_id(sid),
                    "prediction": record["prediction"], "gold": record["ground_truth"],
                    "correct": int(record["correct"]), "task_type": record["task_type"],
                    "query_time": record["question_timestamp"],
                    **decision_values(record),
                    **{k: record["fidelity"].get(k) for k in FIDELITY},
                    **{k: record["fidelity"].get(k) for k in SUPPORT},
                    **{k: record["runtime"].get(k) for k in RUNTIME}}
            all_rows.append(flat)
        records[method] = arm
    complete = all(set(records[m]) == expected for m in METHODS)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    write_csv(out / "per_query.csv", all_rows)
    summaries = []
    for method in METHODS:
        rows = [r for r in all_rows if r["method"] == method]
        s = {"method": method, "N": len(rows), "missing": len(expected) - len(rows),
             "videos": len({r["video_id"] for r in rows}), "correct": sum(r["correct"] for r in rows)}
        s["accuracy_micro"] = s["correct"] / len(rows) if rows else None
        for field in ("correct", *FIDELITY, *RUNTIME, *DECISION, *SUPPORT):
            avg, lo, hi = cluster_ci(rows, field)
            values = [r[field] for r in rows if r.get(field) is not None]
            s.update({f"mean_{field}": avg, f"ci_low_{field}": lo, f"ci_high_{field}": hi,
                      f"median_{field}": median(values) if values else None, f"n_{field}": len(values)})
        for field in ("compression_trigger_count", "effective_compaction_count"):
            s[f"sum_{field}"] = sum(r[field] for r in rows)
        low, high = cluster_median_ci(rows, "aore_mean")
        s["ci_low_median_aore_mean"], s["ci_high_median_aore_mean"] = low, high
        summaries.append(s)
    write_csv(out / "summary.csv", summaries)
    aore_diagnostics = []
    for method in METHODS:
        vals = [r["aore_mean"] for r in all_rows if r["method"] == method]
        if vals:
            aore_diagnostics.append({"method": method, "N": len(vals), "mean": mean(vals),
                "median": median(vals), "p90": float(np.quantile(vals, .9)),
                "p95": float(np.quantile(vals, .95)), "max": max(vals),
                "n_greater_than_10": sum(v > 10 for v in vals)})
    write_csv(out / "aore_diagnostics.csv", aore_diagnostics)
    common = set.intersection(*(set(records[m]) for m in METHODS))
    compressed_common = {sid for sid in common if any(
        records[m][sid]["runtime"]["effective_compaction_count"] > 0 for m in ("hermes", "infokv_stream"))}
    compressed_summary = []
    for method in METHODS:
        rows = [r for r in all_rows if r["method"] == method and r["sample_id"] in compressed_common]
        entry = {"method": method, "N": len(rows), "videos": len({r["video_id"] for r in rows}),
                 "accuracy_micro": mean(r["correct"] for r in rows) if rows else None}
        for key in (*FIDELITY, *DECISION):
            value, low, high = cluster_ci(rows, key)
            entry.update({f"mean_{key}": value, f"ci_low_{key}": low, f"ci_high_{key}": high})
        compressed_summary.append(entry)
    write_csv(out / "compressed_common_summary.csv", compressed_summary)
    pairs = []
    for sid in sorted(common):
        h, i, f = (records[m][sid] for m in ("hermes", "infokv_stream", "full_cache"))
        for key in ("ground_truth", "question_timestamp", "task_type"):
            if h[key] != f[key] or i[key] != f[key]:
                raise RuntimeError(f"Matched query has inconsistent {key}: {sid}")
        if any(r["runtime"]["frames"] != f["runtime"]["frames"] for r in (h, i)):
            raise RuntimeError(f"Matched query has inconsistent sampled frame count: {sid}")
        row = {"sample_id": sid, "video_id": video_id(sid),
               "full_correct": int(f["correct"]), "hermes_correct": int(h["correct"]),
               "infokv_correct": int(i["correct"]), "delta_accuracy": int(i["correct"]) - int(h["correct"]),
               "transition_hermes_to_infokv": ("C" if h["correct"] else "W") + ("C" if i["correct"] else "W")}
        for key in FIDELITY:
            row[f"delta_{key}"] = i["fidelity"][key] - h["fidelity"][key]
        for key in DECISION:
            iv, hv, fv = decision_values(i)[key], decision_values(h)[key], decision_values(f)[key]
            row[f"infokv_{key}"] = iv
            row[f"hermes_{key}"] = hv
            row[f"full_{key}"] = fv
            row[f"delta_{key}"] = iv-hv if iv is not None and hv is not None else None
        pairs.append(row)
    write_csv(out / "paired.csv", pairs)
    transitions = {t: sum(r["transition_hermes_to_infokv"] == t for r in pairs) for t in ("CC", "CW", "WC", "WW")}
    d_avg, d_lo, d_hi = cluster_ci(pairs, "delta_accuracy")
    audit = {"complete": complete, "expected_per_method": len(expected), "matched_N": len(common),
             "compressed_common_N": len(compressed_common), "compressed_common_summary": compressed_summary,
             "bootstrap": {"unit": "video", "n_boot": 2000, "seed": 42},
             "transitions_hermes_to_infokv": transitions,
             "delta_accuracy": {"mean": d_avg, "ci_low": d_lo, "ci_high": d_hi}, "summary": summaries}
    (out / "summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    temporal = []
    for method in ("hermes", "infokv_stream"):
        for bucket in ("0_10", "10_30", "30_60", "60_300", "300_inf"):
            rows = [{"video_id": video_id(sid), "ram": (r["fidelity"].get("temporal_ram") or {}).get(bucket)}
                    for sid, r in records[method].items()]
            avg, lo, hi = cluster_ci(rows, "ram")
            temporal.append({"method": method, "age_bucket_sec": bucket, "mean_ram": avg,
                             "ci_low": lo, "ci_high": hi, "N_query": sum(r["ram"] is not None for r in rows)})
    write_csv(out / "temporal_ram.csv", temporal)
    plot(figures, summaries, all_rows, records, common, temporal, complete)
    def fmt(x):
        return "—" if x is None else ("0.0000" if abs(x) < .00005 else f"{x:.4f}")
    lines = ["# StreamingBench RT：同模型、同视频集对照", "",
             f"完成状态：{'全部完成' if complete else '部分结果，仍在运行'}；每方法预期 100 个问题，20 个视频。",
             "", "模型：Qwen2.5-VL-7B；HERMES / InfoKV 视觉预算 2000；Full 不压缩。",
             "视频和问题完全复用既有 Plan B；0.5 FPS、2 帧/chunk、max_side=336、full_prefix、官方 prompt。",
             "旧 Plan B 使用 MMDuet2，其结果只作为历史参考；本表的 HERMES 与 Full 均重新实测。",
             "", "| 方法 | 完成 | 正确率 | RAM ↑ | AORE 逐题中位数 ↓ | JS ↓ | KL ↓ | 有效压缩总次数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        acc = f"{s['correct']}/{s['N']} ({s['accuracy_micro']:.1%})" if s["N"] else "—"
        aore = fmt(s['median_aore_mean']) if s['N'] else "—"
        lines.append(f"| {LABELS[s['method']]} | {s['N']}/100 | {acc} | {fmt(s['mean_ram_mean'])} | "
                     f"{aore} | {fmt(s['mean_js'])} | {fmt(s['mean_kl'])} | {s['sum_effective_compaction_count']} |")
    lines += ["", f"共同完成 {len(common)} 题；HERMES→InfoKV 的 CC/CW/WC/WW = {transitions}。",
              f"其中 {len(compressed_common)} 题至少一种方法发生过有效压缩；这组相同问题的指标另存 compressed_common_summary.csv，避免大量未压缩问题掩盖差异。",
              f"InfoKV−HERMES 正确率差：{fmt(d_avg)}，视频聚类 bootstrap 95% CI [{fmt(d_lo)}, {fmt(d_hi)}]。",
              "", "RAM 是视觉区域内保留的 Full 注意力质量；AORE 比较每层每头、o_proj 前的视觉注意力输出。",
              "正的微小注意力质量按真实分母归一化；完全零质量的头没有定义 RAM，已从均值排除，有效/无定义头数分别保存。Temporal RAM 的正分母也不添加绝对 epsilon。",
              "JS/KL 使用同一问题首个生成位置的完整词表 logits。每个真实问题到达时都进行诊断，共 100 个 anchor。",
              "RAM/AORE/JS/KL 使用 eager 诊断重放；任务正确率和选项 logits 来自原有推理路径。AORE 是相对误差，Full 视觉输出范数接近零的头会产生极端值；主表使用逐题 AORE 中位数。原始均值、P90/P95、最大值与极端值数量保存在 aore_diagnostics.csv；逐题值、逐层统计全部保留。AORE 散点图使用 symlog 坐标保留零与极端值。",
              "Full/self 的 JS 可能有约 1e-8 的正负舍入误差，原始值保留；主表四位小数显示为 0。",
              "CSV 另有 P_gold（A-D 选项集合条件概率，token 变体用 logsumexp）和 margin（变体取最大分数）；它们可能与自由生成答案不同。",
              "Temporal RAM 按 token 年龄分桶；没有 Full 注意力质量的桶留空。时间归属沿用既有 provenance 的帧间等分近似。",
              "", "表中正确率按问题统计；其余均值先在视频内平均，再对视频平均。图中误差线为视频聚类 bootstrap 95% 区间（2000 次，seed=42）。每视频五题全部完成时，两种准确率权重相同。当前视频集按压缩覆盖选择，不代表完整 StreamingBench 的准确率。",
              "相同名义视觉预算不保证相同 KV 字节：HERMES 按层裁剪并可能生成 summary，InfoKV 保留跨层共同 token。",
              "压缩触发次数和真正淘汰 token 的有效次数分别保存；InfoKV 的整理触发还包括未裁剪的 chunk。",
              "每个问题独立从视频起点重放到问题时刻；压缩总次数为 100 次独立重放的合计。",
              "stream_processing_sec 包括采样后帧的编码/prefill/压缩及观测开销；qa_sec 不含 eager 诊断重放。",
              "峰值 GPU allocated memory 包含诊断开销，不能作为纯部署显存结论。physical_pruning_total_sec 仅物理裁剪；InfoKV 完整管理耗时另列。",
              "", "结果文件：analysis/summary.csv、per_query.csv、paired.csv、temporal_ram.csv；逐题 events 保存在各方法 samples/*.json。",
              "图：figures/overview.png/pdf、fidelity_vs_compactions.png/pdf、paired_fidelity.png/pdf、temporal_ram.png/pdf。", ""]
    (out / "report_zh.md").write_text("\n".join(lines))
    import matplotlib
    (out / "reproducibility.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_contract_id": manifest["contract_id"],
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "complete": complete, "records_per_method": {m: len(records[m]) for m in METHODS},
    }, indent=2) + "\n")
    print(json.dumps({"complete": complete, "counts": {m: len(records[m]) for m in METHODS}, "report": str(out / "report_zh.md")}))


def plot(out, summaries, rows, records, common, temporal, complete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    state = "20 videos / 100 queries per method" if complete else "PARTIAL results"
    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    specs = [("correct", "Accuracy (video mean)", 100), ("ram_mean", "RAM (higher is better)", 1),
             ("aore_mean", "AORE median (lower is better)", 1), ("js", "JS divergence", 1),
             ("kv_bytes_at_query", "KV at query (MiB)", 1/1024**2),
             ("effective_compaction_count", "Effective compactions / query", 1)]
    for ax, (key, title, scale) in zip(axes.flat, specs):
        for x, s in enumerate(summaries):
            value = s.get(f"median_{key}") if key == "aore_mean" else s.get(f"mean_{key}")
            if value is None:
                continue
            low, high = (s[f"ci_low_{key}"], s[f"ci_high_{key}"]) if key != "aore_mean" else (s["ci_low_median_aore_mean"], s["ci_high_median_aore_mean"])
            ax.bar(x, value*scale, color=COLORS[s["method"]], width=.65)
            ax.errorbar(x, value*scale, yerr=[[max(0,value-low)*scale],[max(0,high-value)*scale]], fmt="none", color="black", capsize=4)
        ax.set_xticks(range(3), [f"{LABELS[s['method']]}\nN={s['N']}" for s in summaries])
        ax.set_title(title + (" (%)" if key == "correct" else ""))
        ax.set_ylim(bottom=0)
    fig.suptitle("StreamingBench RT | Qwen2.5-VL-7B | 2K visual budget\n" + state)
    fig.tight_layout()
    save(fig, "overview")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, field in zip(axes.flat, ("ram_mean", "aore_mean", "js", "kl")):
        for method in ("hermes", "infokv_stream"):
            arm = [r for r in rows if r["method"] == method]
            ax.scatter([r["effective_compaction_count"] for r in arm], [r[field] for r in arm],
                       s=22, alpha=.65, color=COLORS[method], label=LABELS[method])
        ax.set_xlabel("Effective compactions before query")
        ax.set_ylabel({"ram_mean":"RAM", "aore_mean":"AORE", "js":"JS", "kl":"KL"}[field])
        if field == "aore_mean":
            ax.set_yscale("symlog", linthresh=.01)
        ax.legend()
    fig.suptitle("Actual query anchors; query content varies across points\n" + state)
    fig.tight_layout()
    save(fig, "fidelity_vs_compactions")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, title in zip(axes, ("ram_mean", "aore_mean"), ("RAM", "AORE")):
        x = [records["hermes"][sid]["fidelity"][field] for sid in sorted(common)]
        y = [records["infokv_stream"][sid]["fidelity"][field] for sid in sorted(common)]
        ax.scatter(x, y, s=25, color=COLORS["infokv_stream"], alpha=.7)
        if x:
            lo, hi = min(x+y), max(x+y)
            if hi == lo: hi += .01
            ax.plot([lo,hi], [lo,hi], "--", color="gray")
        ax.set_xlabel("HERMES " + title)
        ax.set_ylabel("InfoKV " + title)
        if field == "aore_mean":
            ax.set_xscale("symlog", linthresh=.01)
            ax.set_yscale("symlog", linthresh=.01)
    fig.suptitle(f"Matched queries (N={len(common)}) | 2K")
    fig.tight_layout()
    save(fig, "paired_fidelity")
    fig, ax = plt.subplots(figsize=(8, 4))
    for method in ("hermes", "infokv_stream"):
        arm = [r for r in temporal if r["method"] == method]
        valid = [(j,r) for j,r in enumerate(arm) if r["mean_ram"] is not None]
        if not valid:
            continue
        xs, ys = [j for j,r in valid], [r["mean_ram"] for j,r in valid]
        low = [max(0,r["mean_ram"]-r["ci_low"]) for j,r in valid]
        high = [max(0,r["ci_high"]-r["mean_ram"]) for j,r in valid]
        ax.errorbar(xs, ys, yerr=[low,high], marker="o", capsize=4, label=LABELS[method], color=COLORS[method])
    ax.set_xticks(range(5), ["0–10", "10–30", "30–60", "60–300", "300+"])
    ax.set_xlabel("Visual token age (seconds)")
    ax.set_ylabel("Temporal RAM")
    ax.set_ylim(0, 1.05)
    if ax.lines: ax.legend()
    ax.set_title("Empty attention buckets excluded | " + state)
    fig.tight_layout()
    save(fig, "temporal_ram")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/InfoKVStream/streamingbench_rt20_20261013"))
    analyze(parser.parse_args().root)
