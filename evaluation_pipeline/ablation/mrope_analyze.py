"""CPU analysis for the M-RoPE ablation (run in the ``base`` env).

    python -m evaluation_pipeline.ablation.mrope_analyze

Reads ``hook_on`` / ``hook_off`` sidecars and produces the CSVs + figures
required by the ablation spec. Pure CPU: no model, no GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .mrope_common import ABLATION_ROOT, ARMS, arm_dir


# HERMES layer taxonomy, taken from load_model(): short_term_ratio=0.1,
# long_term_ratio=0.3 -> short = [0, 0.1L), mid = [0.1L, 0.7L), long = [0.7L, L)
SHORT_RATIO = 0.1
LONG_TERM_RATIO = 0.3

# Pre-declared decision thresholds (do NOT tune after seeing results).
TH_NEAR_IDENTICAL = 0.95
TH_DRIFT_DROP = 0.05
TH_SPEARMAN = -0.5
TH_MATERIAL = 0.80


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def layer_type(layer: int, num_layers: int) -> str:
    short_end = int(num_layers * SHORT_RATIO)
    long_start = int(num_layers * (1 - LONG_TERM_RATIO))
    if layer < short_end:
        return "short"
    if layer >= long_start:
        return "long"
    return "mid"


def jaccard(a: set, b: set) -> Tuple[Optional[float], int, int]:
    union = a | b
    if not union:
        return None, len(a & b), 0
    return len(a & b) / len(union), len(a & b), len(union)


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _write_csv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return path


# --------------------------------------------------------------------- #
def chunk_consistency(analysis: Path) -> Tuple[List[dict], Dict[str, bool]]:
    rows = []
    valid = {}
    chunks = {arm: _read_jsonl(arm_dir(arm) / "raw" / "main" / "chunk_consistency.jsonl")
              for arm in ARMS}
    by_arm = {
        arm: {(str(r["sample_id"]), int(r["chunk_index"])): r for r in recs}
        for arm, recs in chunks.items()
    }
    keys = sorted(set(by_arm["on"]) | set(by_arm["off"]))
    for key in keys:
        on = by_arm["on"].get(key, {})
        off = by_arm["off"].get(key, {})
        same = (
            on.get("n_frames") == off.get("n_frames")
            and on.get("visual_tokens") == off.get("visual_tokens")
            and on.get("frame_times") == off.get("frame_times")
        )
        rows.append({
            "video_id": key[0], "chunk_id": key[1],
            "n_frames_on": on.get("n_frames"), "n_frames_off": off.get("n_frames"),
            "visual_tokens_on": on.get("visual_tokens"),
            "visual_tokens_off": off.get("visual_tokens"),
            "cache_delta_on": (on.get("cache_len_after") or 0) - (on.get("cache_len_before") or 0) if on else None,
            "cache_delta_off": (off.get("cache_len_after") or 0) - (off.get("cache_len_before") or 0) if off else None,
            "consistent": bool(same),
        })
        valid[key[0]] = valid.get(key[0], True) and bool(same)
    _write_csv(analysis / "chunk_consistency.csv", rows)
    return rows, valid


def compaction_alignment(analysis: Path) -> Tuple[List[dict], Dict[str, bool]]:
    rows = []
    valid = {}
    dumps = {arm: _read_jsonl(arm_dir(arm) / "raw" / "main" / "compaction_uids.jsonl")
             for arm in ARMS}
    for arm in ARMS:
        for d in dumps[arm]:
            vid = str(d.get("video_id") or d.get("sample_id"))
            rows.append({
                "arm": arm, "video_id": vid, "k": d["compaction_index"],
                "stream_time": d.get("stream_time"),
                "cache_len_before": max(d["cache_len_before"]) if d.get("cache_len_before") else None,
                "cache_len_after": max(d["cache_len_after"]) if d.get("cache_len_after") else None,
                "effective": d.get("effective"),
            })
    by = {(r["arm"], r["video_id"], r["k"]): r for r in rows}
    videos = sorted({r["video_id"] for r in rows})
    align_rows = []
    for vid in videos:
        on = [r for r in rows if r["arm"] == "on" and r["video_id"] == vid]
        off = [r for r in rows if r["arm"] == "off" and r["video_id"] == vid]
        ok = len(on) == len(off) and all(
            abs((by[("on", vid, k)]["stream_time"] or 0) - (by[("off", vid, k)]["stream_time"] or 0)) < 1e-6
            and by[("on", vid, k)]["cache_len_after"] == by[("off", vid, k)]["cache_len_after"]
            for k in range(1, min(len(on), len(off)) + 1)
            if ("on", vid, k) in by and ("off", vid, k) in by
        )
        valid[vid] = ok
        align_rows.append({
            "video_id": vid, "k_on": len(on), "k_off": len(off),
            "k_equal": len(on) == len(off), "alignment_valid": ok,
        })
    _write_csv(analysis / "A_compaction_alignment.csv", align_rows)
    return align_rows, valid


def compaction_jaccard(analysis: Path, num_layers: int) -> List[dict]:
    dumps = {arm: _read_jsonl(arm_dir(arm) / "raw" / "main" / "compaction_uids.jsonl")
             for arm in ARMS}
    index = {}
    for arm in ARMS:
        for d in dumps[arm]:
            vid = str(d.get("video_id") or d.get("sample_id"))
            index[(arm, vid, int(d["compaction_index"]))] = d
    rows = []
    videos = sorted({k[1] for k in index})
    for vid in videos:
        ks = sorted({k[2] for k in index if k[1] == vid})
        for k in ks:
            on = index.get(("on", vid, k))
            off = index.get(("off", vid, k))
            if on is None or off is None:
                continue
            for layer in range(num_layers):
                key = str(layer)
                u_on = set(on["retained_visual_uids_by_layer"].get(key, []))
                u_off = set(off["retained_visual_uids_by_layer"].get(key, []))
                j, inter, union = jaccard(u_on, u_off)
                rows.append({
                    "video_id": vid, "k": k,
                    "stream_time_on": on.get("stream_time"),
                    "stream_time_off": off.get("stream_time"),
                    "layer_idx": layer,
                    "layer_type": layer_type(layer, num_layers),
                    "n_on": len(u_on), "n_off": len(u_off),
                    "n_intersection": inter, "n_union": union,
                    "uid_jaccard": j,
                    "empty_union": union == 0,
                    "n_summary_uids_on": on.get("n_summary_uids_total"),
                    "n_summary_uids_off": off.get("n_summary_uids_total"),
                    "cache_len_on": max(on["cache_len_after"]) if on.get("cache_len_after") else None,
                    "cache_len_off": max(off["cache_len_after"]) if off.get("cache_len_after") else None,
                })
    _write_csv(analysis / "A_compaction_summary.csv", rows)
    return rows


def jaccard_trend(rows: List[dict]) -> List[dict]:
    out = []
    for vid in sorted({r["video_id"] for r in rows}):
        sub = [r for r in rows if r["video_id"] == vid and r["uid_jaccard"] is not None]
        if not sub:
            continue
        ks = sorted({r["k"] for r in sub})
        per_k = {k: statistics.mean([r["uid_jaccard"] for r in sub if r["k"] == k]) for k in ks}
        vals = [per_k[k] for k in ks]
        q = max(1, len(vals) // 4)
        first_q = statistics.mean(vals[:q])
        last_q = statistics.mean(vals[-q:])
        rec = {
            "video_id": vid, "task": sub[0].get("task") or TASK_BY_VIDEO.get(vid),
            "k_total": len(ks),
            "all_layer_jaccard_mean": statistics.mean(vals),
            "all_layer_jaccard_min": min(vals),
            "first_quartile_mean": first_q,
            "last_quartile_mean": last_q,
            "last_minus_first": last_q - first_q,
            "spearman_j_vs_k": spearman([float(k) for k in ks], vals),
        }
        for lt in ("short", "mid", "long"):
            lt_rows = [r for r in sub if r["layer_type"] == lt]
            if not lt_rows:
                rec[f"{lt}_layer_mean"] = None
                rec[f"{lt}_spearman"] = None
                continue
            lt_per_k = {k: statistics.mean([r["uid_jaccard"] for r in lt_rows if r["k"] == k])
                        for k in sorted({r["k"] for r in lt_rows})}
            rec[f"{lt}_layer_mean"] = statistics.mean(lt_per_k.values())
            rec[f"{lt}_spearman"] = spearman(
                [float(k) for k in lt_per_k], list(lt_per_k.values())
            )
        out.append(rec)
    return out


TASK_BY_VIDEO = {"300": "HLD", "299": "HLD", "298": "HLD", "3": "EPM", "483": "ASI"}


def decision_rows() -> List[dict]:
    from evaluation_pipeline.kv_metrics.config import KVMetricConfig  # noqa: F401

    def load(arm: str) -> Dict[str, dict]:
        out = {}
        for path in sorted((arm_dir(arm) / "raw" / "main" / "paired").glob("*.jsonl")):
            for r in _read_jsonl(path):
                out[str(r.get("sample_id"))] = r
        return out

    on, off = load("on"), load("off")
    rows = []
    for sid in sorted(set(on) | set(off), key=lambda s: (len(s), s)):
        a, b = on.get(sid, {}), off.get(sid, {})

        def score(logits, gold):
            if not logits:
                return None, None
            letters = a.get("option_token_ids") or b.get("option_token_ids") or {}
            per = {}
            for letter, ids in letters.items():
                vals = [logits.get(str(i)) for i in ids]
                vals = [v for v in vals if v is not None]
                if vals:
                    per[letter] = max(vals)
            if gold not in per:
                return None, None
            wrong = [v for l, v in per.items() if l != gold]
            if not wrong:
                return None, None
            gold_logit = per[gold]
            import math

            exp = {l: math.exp(v - max(per.values())) for l, v in per.items()}
            total = sum(exp.values())
            p_gold = exp[gold] / total
            return p_gold, gold_logit - max(wrong)

        gold = a.get("task_gold") or b.get("task_gold")
        p_on, m_on = score(a.get("option_token_logits_compressed"), gold)
        p_off, m_off = score(b.get("option_token_logits_compressed"), gold)
        rows.append({
            "sample_id": sid,
            "video_id": a.get("video_id") or b.get("video_id") or sid,
            "task": a.get("task_type") or b.get("task_type"),
            "gold": gold,
            "prediction_on": a.get("task_prediction_compressed"),
            "prediction_off": b.get("task_prediction_compressed"),
            "correct_on": a.get("task_correct_compressed"),
            "correct_off": b.get("task_correct_compressed"),
            "prediction_changed": a.get("task_prediction_compressed") != b.get("task_prediction_compressed"),
            "P_gold_on": p_on, "P_gold_off": p_off,
            "dP_gold": (p_off - p_on) if (p_on is not None and p_off is not None) else None,
            "M_z_on": m_on, "M_z_off": m_off,
            "dM_z": (m_off - m_on) if (m_on is not None and m_off is not None) else None,
            "margin_crossed": (
                None if (m_on is None or m_off is None)
                else (m_on > 0) != (m_off > 0)
            ),
            "transition": _transition(a.get("task_correct_compressed"),
                                      b.get("task_correct_compressed")),
        })
    return rows


def _transition(on_correct, off_correct) -> Optional[str]:
    if on_correct is None or off_correct is None:
        return None
    return ("C" if on_correct else "W") + ("C" if off_correct else "W")


def plot_all(rows: List[dict], trend: List[dict], decisions: List[dict], out: Path) -> List[Path]:
    """The five required figures (matplotlib, no smoothing, nothing hidden)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    figs: List[Path] = []
    videos = sorted({r["video_id"] for r in rows}, key=lambda v: (len(v), v))

    # 1) Jaccard vs compaction index, one line per video (all-layer mean)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for vid in videos:
        sub = [r for r in rows if r["video_id"] == vid and r["uid_jaccard"] is not None]
        ks = sorted({r["k"] for r in sub})
        ys = [statistics.mean([r["uid_jaccard"] for r in sub if r["k"] == k]) for k in ks]
        ax.plot(ks, ys, "-o", ms=3, label=f"{vid} ({TASK_BY_VIDEO.get(vid,'')})")
    ax.set_xlabel("effective compaction index k")
    ax.set_ylabel("all-layer mean UID Jaccard")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Retained-set Jaccard vs compaction index (hook ON vs OFF)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    for ext in ("png", "pdf"):
        p = out / f"jaccard_vs_compaction_index_per_video.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        figs.append(p)
    plt.close(fig)

    # 2) Jaccard vs k by layer type
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for lt in ("short", "mid", "long"):
        xs, ys = [], []
        for k in sorted({r["k"] for r in rows}):
            vals = [r["uid_jaccard"] for r in rows if r["k"] == k and r["layer_type"] == lt
                    and r["uid_jaccard"] is not None]
            if vals:
                xs.append(k)
                ys.append(statistics.mean(vals))
        if xs:
            ax.plot(xs, ys, "-", label=lt)
    ax.set_xlabel("effective compaction index k")
    ax.set_ylabel("mean UID Jaccard")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Jaccard vs k by HERMES layer type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    for ext in ("png", "pdf"):
        p = out / f"jaccard_vs_k_by_layer_type.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        figs.append(p)
    plt.close(fig)

    # 3) per-layer mean Jaccard
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    layers = sorted({r["layer_idx"] for r in rows})
    means = [statistics.mean([r["uid_jaccard"] for r in rows
                              if r["layer_idx"] == l and r["uid_jaccard"] is not None])
             for l in layers]
    ax.bar([str(l) for l in layers], means)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean UID Jaccard")
    ax.set_title("Per-layer mean retained-set Jaccard")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    for l, m in zip(layers, means):
        if m < 0.999:
            ax.annotate(f"{m:.3f}", (str(l), m), fontsize=6, rotation=90)
    ax.grid(True, alpha=0.3, axis="y")
    for ext in ("png", "pdf"):
        p = out / f"per_layer_mean_jaccard.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        figs.append(p)
    plt.close(fig)

    # 4) early vs late
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels, firsts, lasts = [], [], []
    for t in trend:
        labels.append(str(t["video_id"]))
        firsts.append(t["first_quartile_mean"])
        lasts.append(t["last_quartile_mean"])
    idx = range(len(labels))
    ax.bar([i - 0.2 for i in idx], firsts, width=0.4, label="first quartile")
    ax.bar([i + 0.2 for i in idx], lasts, width=0.4, label="last quartile")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("video id")
    ax.set_ylabel("mean UID Jaccard")
    ax.set_title("Early vs late Jaccard (cumulative-drift check)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for ext in ("png", "pdf"):
        p = out / f"early_vs_late_jaccard.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        figs.append(p)
    plt.close(fig)

    # 5) decision shift
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    xs = [str(d["video_id"]) for d in decisions]
    for ax, key, title in ((axes[0], "dP_gold", "ΔP_gold (OFF − ON)"),
                           (axes[1], "dM_z", "ΔM_z (OFF − ON)")):
        ys = [d.get(key) or 0.0 for d in decisions]
        ax.bar(xs, ys)
        ax.axhline(0.0, color="0.6")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    figs_dir = out
    for ext in ("png", "pdf"):
        p = figs_dir / f"decision_shift.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        figs.append(p)
    plt.close(fig)
    return figs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ABLATION_ROOT / "analysis"))
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args(argv)
    analysis = Path(args.out)
    analysis.mkdir(parents=True, exist_ok=True)

    num_layers = 28
    dumps = _read_jsonl(arm_dir("on") / "raw" / "main" / "compaction_uids.jsonl")
    if dumps:
        num_layers = len(dumps[0]["retained_visual_uids_by_layer"])
    print(f"[mrope-analyze] num_layers={num_layers} "
          f"short<{int(num_layers*SHORT_RATIO)} "
          f"long>={int(num_layers*(1-LONG_TERM_RATIO))}")

    chunk_rows, chunk_valid = chunk_consistency(analysis)
    align_rows, align_valid = compaction_alignment(analysis)
    jrows = compaction_jaccard(analysis, num_layers)
    trend = jaccard_trend(jrows)
    _write_csv(analysis / "A_jaccard_trend.csv", trend)
    dec = decision_rows()
    _write_csv(analysis / "A_decision.csv", dec)

    valid_js = [r["uid_jaccard"] for r in jrows if r["uid_jaccard"] is not None]
    empty = sum(1 for r in jrows if r["empty_union"])
    summary = {
        "n_pairs": len(jrows),
        "n_empty_union": empty,
        "overall_mean_jaccard": statistics.mean(valid_js) if valid_js else None,
        "overall_median_jaccard": statistics.median(valid_js) if valid_js else None,
        "chunk_consistency_all_valid": all(chunk_valid.values()) if chunk_valid else None,
        "compaction_alignment_all_valid": all(align_valid.values()) if align_valid else None,
        "jaccard_trend": trend,
        "decisions": dec,
    }
    (analysis / "A_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    figures = []
    if not args.no_plots and jrows:
        figures = plot_all(jrows, trend, dec, analysis / "figures")
    print(f"[mrope-analyze] pairs={len(jrows)} meanJ={summary['overall_mean_jaccard']}")
    print(f"  chunk consistency valid: {summary['chunk_consistency_all_valid']}")
    print(f"  compaction alignment valid: {summary['compaction_alignment_all_valid']}")
    for t in trend:
        print(f"  video {t['video_id']} ({t['task']}): k={t['k_total']} "
              f"meanJ={t['all_layer_jaccard_mean']:.4f} "
              f"first={t['first_quartile_mean']:.4f} last={t['last_quartile_mean']:.4f} "
              f"rho={t['spearman_j_vs_k']}")
    print(f"-> {analysis}")
    if figures:
        print(f"figures={len(figures)} -> {analysis / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
