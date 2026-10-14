"""Merge isolated stage-timing runs into one reviewable comparison artifact."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/InfoKVStream/end_to_end_timing_20261015_final"
SOURCES = {
    "hermes": ROOT / "outputs/InfoKVStream/end_to_end_timing_20261015_hermes_isolated",
    "infokv_global": ROOT / "outputs/InfoKVStream/end_to_end_timing_20261015_v2_global",
    "infokv_layerwise": ROOT / "outputs/InfoKVStream/end_to_end_timing_20261015_v2_layerwise",
}

COMPONENTS = {
    "hermes": [
        ("hermes_predict_question", "HERMES question prediction", False),
        ("hermes_attention", "HERMES attention", False),
        ("hermes_score_topk", "HERMES score and Top-K", False),
        ("kv_gather_gpu", "KV gather CUDA kernel", False),
        ("mrope_primitives", "M-RoPE primitives", False),
        ("physical_prune", "physical KV prune", False),
    ],
    "infokv": [
        ("infokv_key_refresh", "InfoKV key/query signal refresh", False),
        ("infokv_attention_refresh", "InfoKV V2V signal refresh", False),
        ("infokv_scoring", "InfoKV score fusion", False),
        ("infokv_topk", "InfoKV deterministic Top-K", False),
        ("infokv_keep_indices", "keep-index construction", False),
        ("kv_gather_gpu", "KV gather CUDA kernel", False),
        ("infokv_mrope_metadata", "M-RoPE metadata update", False),
        ("infokv_ledger_rebuild", "ledger update", False),
        ("infokv_sidecar", "sidecar update", False),
        ("physical_prune", "physical KV prune", False),
        ("infokv_memory_compact", "InfoKV compact aggregate", True),
        ("infokv_score_topk", "InfoKV selection aggregate", True),
    ],
}


def load_rows():
    rows = []
    for label, path in SOURCES.items():
        payload = json.loads((path / "stage_timing.json").read_text())
        row = payload["rows"][0]
        row["comparison_label"] = {
            "hermes": "HERMES",
            "infokv_global": "InfoKV global",
            "infokv_layerwise": "InfoKV layerwise",
        }[label]
        row["comparison_key"] = label
        rows.append(row)
    if {r["effective_count"] for r in rows} != {55}:
        raise RuntimeError("isolated timing runs do not all contain 55 effective compressions")
    if not all(r["matches_reference"] for r in rows):
        raise RuntimeError("timing instrumentation changed a reference prediction")
    schedules = {(tuple(r["visual_tokens_per_chunk"]), tuple(r["seconds_per_chunk"])) for r in rows}
    if len(schedules) != 1:
        raise RuntimeError("isolated runs used different chunk schedules")
    return rows


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source_manifest.json").write_text(json.dumps(
        {"sources": {k: str(v) for k, v in SOURCES.items()},
         "sample_id": rows[0]["sample_id"], "budget": rows[0]["budget"],
         "isolated_processes": True}, indent=2, ensure_ascii=False) + "\n")
    (OUT / "stage_timing.json").write_text(json.dumps(
        {"rows": rows, "definition": {
            "end_to_end": "sample execution start through answer returned; model loading excluded",
            "compression": "effective and trigger totals are synchronized wall clock",
            "nested": "detailed compression rows are nested measurements and must not be summed",
            "kv_gather_gpu": "CUDA event kernel time, reported separately from wall clock",
        }}, indent=2, ensure_ascii=False) + "\n")

    stage_fields = ["method", "variant", "comparison_label", "model_load_sec", "end_to_end_sec",
                    "trigger_count", "effective_count", "non_effective_count",
                    "first_effective_ms", "mean_effective_ms", "total_effective_sec",
                    "total_non_effective_sec", "video_read_resize", "vision_encoder",
                    "visual_lm_forward", "visual_capture_and_adapter", "compression_all_triggers",
                    "qa_answering", "qa_prefill_forward", "qa_decode_forwards",
                    "adapter_cleanup_and_other"]
    stage_rows = []
    for r in rows:
        nested = r["nested_sec"]
        visual_rest = r["broad_sec"]["visual_stream_excluding_compression"] - nested.get("vision_encoder", 0.0) - nested.get("visual_lm_forward", 0.0)
        stage_rows.append({
            "method": r["method"], "variant": r["variant"], "comparison_label": r["comparison_label"],
            "model_load_sec": r["model_load_sec"], "end_to_end_sec": r["end_to_end_sec"],
            "trigger_count": r["trigger_count"], "effective_count": r["effective_count"],
            "non_effective_count": r["non_effective_count"], "first_effective_ms": r["first_effective_ms"],
            "mean_effective_ms": r["mean_effective_ms"], "total_effective_sec": r["total_effective_sec"],
            "total_non_effective_sec": r["total_non_effective_sec"],
            "video_read_resize": r["broad_sec"]["video_read_resize"],
            "vision_encoder": nested.get("vision_encoder", 0.0),
            "visual_lm_forward": nested.get("visual_lm_forward", 0.0),
            "visual_capture_and_adapter": max(0.0, visual_rest),
            "compression_all_triggers": r["broad_sec"]["compression_all_triggers"],
            "qa_answering": r["broad_sec"]["qa_answering"],
            "qa_prefill_forward": nested.get("qa_prefill_forward", 0.0),
            "qa_decode_forwards": nested.get("qa_decode_forwards", 0.0),
            "adapter_cleanup_and_other": r["broad_sec"]["adapter_cleanup_and_other"],
        })
    write_csv(OUT / "stage_summary.csv", stage_fields, stage_rows)

    part_rows = []
    for r in rows:
        family = "hermes" if r["method"] == "hermes" else "infokv"
        values = r["effective_compression_nested_sec"]
        for key, label, aggregate in COMPONENTS[family]:
            seconds = float(values.get(key, 0.0))
            part_rows.append({"comparison_label": r["comparison_label"], "method": r["method"],
                              "variant": r["variant"], "part": key, "label": label,
                              "aggregate_nested": int(aggregate), "total_ms": seconds * 1000.0,
                              "mean_per_effective_ms": seconds * 1000.0 / r["effective_count"]})
    write_csv(OUT / "compression_parts.csv",
              ["comparison_label", "method", "variant", "part", "label", "aggregate_nested", "total_ms", "mean_per_effective_ms"],
              part_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["comparison_label"] for r in rows]
    broad_keys = ["video_read_resize", "visual_stream_excluding_compression", "compression_all_triggers",
                  "qa_answering", "adapter_cleanup_and_other"]
    broad_labels = ["video read/resize", "visual prefill + capture", "compression triggers", "QA", "other"]
    broad_colors = ["#9ecae1", "#6baed6", "#fd8d3c", "#74c476", "#bdbdbd"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.0, 0.8, 1.6]})
    bottom = [0.0] * len(rows)
    for key, title, color in zip(broad_keys, broad_labels, broad_colors):
        values = [r["broad_sec"][key] for r in rows]
        axes[0].bar(labels, values, bottom=bottom, label=title, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set_title("Video input to answer")
    axes[0].set_ylabel("Seconds")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=.25)
    axes[1].bar(labels, [r["mean_effective_ms"] for r in rows], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_title("Mean effective compression")
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(axis="y", alpha=.25)
    plot_parts = [
        ("infokv_key_refresh", "key refresh"), ("infokv_attention_refresh", "V2V refresh"),
        ("infokv_scoring", "score fusion"), ("infokv_topk", "Top-K"),
        ("infokv_keep_indices", "keep-index"), ("kv_gather_gpu", "KV gather"),
        ("infokv_mrope_metadata", "M-RoPE"), ("infokv_ledger_rebuild", "ledger"),
        ("infokv_sidecar", "sidecar"), ("physical_prune", "physical prune"),
    ]
    x = list(range(len(plot_parts)))
    width = .25
    for j, r in enumerate(rows):
        family = "hermes" if r["method"] == "hermes" else "infokv"
        vals = r["effective_compression_nested_sec"]
        values = []
        for key, _ in plot_parts:
            if family == "hermes" and key.startswith("infokv_"):
                values.append(0.0)
            elif family == "hermes":
                values.append(vals.get({"kv_gather_gpu": "kv_gather_gpu", "physical_prune": "physical_prune"}.get(key, key), 0.0) * 1000 / r["effective_count"])
            else:
                values.append(vals.get(key, 0.0) * 1000 / r["effective_count"])
        axes[2].bar([v + (j - 1) * width for v in x], values, width=width, label=labels[j])
    axes[2].set_xticks(x, [b for _, b in plot_parts], rotation=45, ha="right")
    axes[2].set_title("Nested compression components (mean ms)")
    axes[2].set_ylabel("Milliseconds")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "timing_comparison.png", dpi=180)
    fig.savefig(OUT / "timing_comparison.pdf")

    report = [
        "# RT20 单视频分阶段耗时",
        "",
        f"样本：`{rows[0]['sample_id']}`；budget：`{rows[0]['budget']}`；三次运行均为独立进程、FA2、相同 66 个视觉 chunk。三种方法均输出参考答案 `{rows[0]['prediction']}`，有效压缩次数均为 55。",
        "",
        "## 端到端结果",
        "",
        "| 方法 | 模型加载（单列） | 视频输入到答案 | 触发/有效 | 首次有效压缩 | 平均有效压缩 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        report.append(f"| {r['comparison_label']} | {r['model_load_sec']:.2f} s | {r['end_to_end_sec']:.2f} s | {r['trigger_count']}/{r['effective_count']} | {r['first_effective_ms']:.2f} ms | {r['mean_effective_ms']:.2f} ms |")
    report += [
        "",
        "端到端阶段的墙钟时间从样本执行开始计，到答案返回结束；模型加载没有计入端到端列。视频读取/缩放、视觉编码器、视觉 LM prefill、压缩、QA 和清理阶段在 `stage_summary.csv` 中给出，阶段之和等于端到端时间。",
        "",
        "## 压缩内部计时",
        "",
        "计时范围包括信号刷新（InfoKV 的 key/query 与 V2V）、评分融合、Top-K、keep-index 构造、KV gather、M-RoPE、ledger、sidecar 和物理 KV 裁剪；HERMES 对应 attention、score/Top-K、M-RoPE primitive 和物理裁剪。视频读取、视觉 prefill、QA 解码和 fidelity probe 不在压缩计时中。",
        "",
        "`compression_parts.csv` 的细分值是有效压缩事件内的嵌套测量。`InfoKV compact aggregate` 和 `InfoKV selection aggregate` 包含其子步骤，不能与子步骤相加；`KV gather CUDA kernel` 是 CUDA event 的内核时间，物理裁剪列是同步墙钟时间，也不能相加作为端到端时间。有效压缩的可比总耗时使用表中的平均有效压缩列。",
        "",
        "## 文件",
        "",
        "- `stage_summary.csv`：端到端阶段、QA prefill/decode、首次/平均压缩耗时。",
        "- `compression_parts.csv`：每个压缩子步骤的总耗时和每次有效压缩平均耗时。",
        "- `timing_comparison.png/pdf`：端到端堆叠图、平均有效压缩图和压缩细分图。",
        "- `stage_timing.json`：完整事件级数据，包括每次有效压缩的细分。",
    ]
    (OUT / "report_zh.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(OUT), "rows": [
        {k: r[k] for k in ("comparison_label", "end_to_end_sec", "effective_count", "mean_effective_ms")}
        for r in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
