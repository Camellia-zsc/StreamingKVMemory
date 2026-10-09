"""CPU analysis for the image-vs-video encoding ablation (run in ``base``)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .encoding_common import (
    ENC_ROOT,
    STYLES,
    TASK_BY_VIDEO,
    TH_DIV_COVERAGE,
    TH_DIV_JACCARD,
    TH_DIV_K_RATIO,
    TH_DIV_OLDEST_SEC,
    TH_REP_JS,
    TH_REP_TOP10,
    TH_REP_TOKEN_RATE,
    TH_RET_COVERAGE,
    TH_RET_JACCARD,
    TH_RET_OLDEST_SEC,
    TIME_BIN_SEC,
    arm_dir,
)

SHORT_RATIO, LONG_TERM_RATIO = 0.1, 0.3


def _jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _csv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return path


def _lt(layer: int, n: int) -> str:
    if layer < int(n * SHORT_RATIO):
        return "short"
    if layer >= int(n * (1 - LONG_TERM_RATIO)):
        return "long"
    return "mid"


def _jacc(a: set, b: set) -> Tuple[Optional[float], int, int]:
    u = a | b
    return (None, len(a & b), 0) if not u else (len(a & b) / len(u), len(a & b), len(u))


def load(style: str) -> Dict[str, List[dict]]:
    base = arm_dir(style) / "raw" / "main"
    return {
        "chunks": _jsonl(base / "chunk_consistency.jsonl"),
        "dumps": _jsonl(base / "compaction_uids.jsonl"),
        "paired": [r for p in sorted((base / "paired").glob("*.jsonl")) for r in _jsonl(p)],
    }


# --------------------------------------------------------------------- #
def input_consistency(analysis: Path, data) -> Tuple[List[dict], Dict[str, bool]]:
    idx = {s: {(str(c["sample_id"]), int(c["chunk_index"])): c for c in data[s]["chunks"]}
           for s in STYLES}
    rows, valid = [], {}
    for key in sorted(set(idx["image"]) | set(idx["video"])):
        a, b = idx["image"].get(key, {}), idx["video"].get(key, {})
        ok = (a.get("n_frames") == b.get("n_frames")
              and a.get("frame_indices") == b.get("frame_indices")
              and a.get("frame_times") == b.get("frame_times"))
        rows.append({
            "video_id": key[0], "chunk_id": key[1],
            "n_frames_image": a.get("n_frames"), "n_frames_video": b.get("n_frames"),
            "frame_indices_image": json.dumps(a.get("frame_indices")),
            "frame_indices_video": json.dumps(b.get("frame_indices")),
            "frame_times_image": json.dumps(a.get("frame_times")),
            "frame_times_video": json.dumps(b.get("frame_times")),
            "start_sec": a.get("start_sec"), "end_sec": a.get("end_sec"),
            "consistent": bool(ok),
        })
        valid[key[0]] = valid.get(key[0], True) and bool(ok)
    _csv(analysis / "B_input_consistency.csv", rows)
    return rows, valid


def tokens(analysis: Path, data) -> List[dict]:
    rows = []
    for style in STYLES:
        for c in data[style]["chunks"]:
            n = c.get("n_frames") or 0
            vt = c.get("visual_tokens")
            secs = (c.get("end_sec") or 0) - (c.get("start_sec") or 0)
            if secs <= 0:
                secs = 2 * n  # frames at 0.5 fps
            rows.append({
                "video_id": str(c.get("sample_id")), "task": TASK_BY_VIDEO.get(str(c.get("sample_id"))),
                "arm": style, "chunk_id": c.get("chunk_index"),
                "n_frames": n, "start_sec": c.get("start_sec"), "end_sec": c.get("end_sec"),
                "encode_style": c.get("encode_style"), "grid_type": c.get("grid_type"),
                "grid_shape": json.dumps(c.get("grid_shape")),
                "pixels_shape": json.dumps(c.get("pixels_shape")),
                "visual_tokens": vt,
                "visual_tokens_per_frame": (vt / n) if (vt and n) else None,
                "seconds_per_chunk": secs,
                "visual_tokens_per_sec": (vt / secs) if (vt and secs) else None,
                "cache_len_before": c.get("cache_len_before"),
                "cache_len_after": c.get("cache_len_after"),
                "cache_delta": ((c.get("cache_len_after") or 0) - (c.get("cache_len_before") or 0)),
            })
    _csv(analysis / "B_tokens.csv", rows)
    return rows


def schedule(analysis: Path, data) -> List[dict]:
    rows = []
    for style in STYLES:
        dumps = data[style]["dumps"]
        for vid in sorted({str(d.get("video_id") or d.get("sample_id")) for d in dumps}):
            ev = sorted([d for d in dumps if str(d.get("video_id") or d.get("sample_id")) == vid],
                        key=lambda d: d["compaction_index"])
            times = [e.get("stream_time") for e in ev if e.get("stream_time") is not None]
            gaps = [b - a for a, b in zip(times, times[1:])]
            chunks = [c for c in data[style]["chunks"] if str(c.get("sample_id")) == vid]
            seen = sum((c.get("visual_tokens") or 0) for c in chunks)
            last = ev[-1] if ev else {}
            retained = None
            if last.get("retained_visual_uids_by_layer"):
                retained = len(last["retained_visual_uids_by_layer"]["0"]) if "0" in last["retained_visual_uids_by_layer"] else None
            rows.append({
                "video_id": vid, "task": TASK_BY_VIDEO.get(vid), "arm": style,
                "budget": len(ev),  # placeholder overwritten below if known
                "n_chunks": len(chunks),
                "n_effective_compactions": len(ev),
                "first_compaction_time": times[0] if times else None,
                "last_compaction_time": times[-1] if times else None,
                "mean_interval_between_compactions": statistics.mean(gaps) if gaps else None,
                "median_interval": statistics.median(gaps) if gaps else None,
                "cache_len_at_query": max(last.get("cache_len_after") or [0]) if last else None,
                "visual_tokens_seen": seen,
                "visual_tokens_retained": retained,
            })
    for r in rows:
        r["budget"] = json.loads((arm_dir(r["arm"]) / "run_config.json").read_text()).get("budget") \
            if (arm_dir(r["arm"]) / "run_config.json").exists() else None
    _csv(analysis / "B_compaction_schedule.csv", rows)
    return rows


def temporal_retention(analysis: Path, data) -> List[dict]:
    rows = []
    for style in STYLES:
        for d in data[style]["dumps"]:
            vid = str(d.get("video_id") or d.get("sample_id"))
            t = float(d.get("stream_time") or 0.0)
            observed = set(int(b) for b in (d.get("observed_bins") or []))
            stats = d.get("retained_age_stats_by_layer") or {}
            bins_by_layer = d.get("retained_seconds_by_layer") or {}
            for layer in sorted(bins_by_layer, key=lambda x: int(x)):
                bins = set(int(b) for b in bins_by_layer[layer])
                st = stats.get(layer) or {}
                oldest_ts = st.get("oldest_ts")
                med_ts = st.get("p50")
                rows.append({
                    "video_id": vid, "task": TASK_BY_VIDEO.get(vid), "arm": style,
                    "k": d["compaction_index"], "stream_time": t,
                    "layer_idx": int(layer), "layer_type": _lt(int(layer), len(bins_by_layer)),
                    "n_retained_bins": len(bins),
                    "n_observed_bins": len(observed),
                    "temporal_coverage_ratio": (len(bins & observed) / len(observed)) if observed else None,
                    "retained_bins": json.dumps(sorted(bins & observed)),
                    "oldest_retained_timestamp": oldest_ts,
                    "oldest_retained_age": (t - oldest_ts) if oldest_ts is not None else None,
                    "median_retained_age": (t - med_ts) if med_ts is not None else None,
                    "mean_retained_age": (t - st["mean_ts"]) if st.get("mean_ts") is not None else None,
                    "p10_age": (t - st["p10"]) if st.get("p10") is not None else None,
                    "p25_age": (t - st["p25"]) if st.get("p25") is not None else None,
                    "p50_age": (t - st["p50"]) if st.get("p50") is not None else None,
                    "p75_age": (t - st["p75"]) if st.get("p75") is not None else None,
                    "p90_age": (t - st["p90"]) if st.get("p90") is not None else None,
                    "n_visual_uids": st.get("n"),
                })
    _csv(analysis / "B_temporal_retention.csv", rows)
    return rows


def temporal_overlap(analysis: Path, data) -> List[dict]:
    """Nearest-neighbour (both directions) pairing by stream_time within 1 chunk."""
    rows = []
    tol = TIME_BIN_SEC + 1e-6
    for vid in sorted({str(d.get("video_id") or d.get("sample_id"))
                       for s in STYLES for d in data[s]["dumps"]}):
        ev = {s: sorted([d for d in data[s]["dumps"]
                         if str(d.get("video_id") or d.get("sample_id")) == vid],
                        key=lambda d: d.get("stream_time") or 0.0) for s in STYLES}
        if not ev["image"] or not ev["video"]:
            continue
        pairs = []
        for a in ev["image"]:
            b = min(ev["video"], key=lambda x: abs((x.get("stream_time") or 0) - (a.get("stream_time") or 0)))
            if abs((b.get("stream_time") or 0) - (a.get("stream_time") or 0)) <= tol:
                pairs.append((a, b))
        for b in ev["video"]:
            a = min(ev["image"], key=lambda x: abs((x.get("stream_time") or 0) - (b.get("stream_time") or 0)))
            if abs((b.get("stream_time") or 0) - (a.get("stream_time") or 0)) <= tol and (a, b) not in pairs:
                pairs.append((a, b))
        for a, b in pairs:
            ia = a.get("retained_seconds_by_layer") or {}
            ib = b.get("retained_seconds_by_layer") or {}
            for layer in sorted(set(ia) & set(ib), key=lambda x: int(x)):
                bins_a = set(int(x) for x in ia[layer])
                bins_b = set(int(x) for x in ib[layer])
                j, inter, union = _jacc(bins_a, bins_b)
                sa = (a.get("retained_age_stats_by_layer") or {}).get(layer) or {}
                sb = (b.get("retained_age_stats_by_layer") or {}).get(layer) or {}
                ta, tb = float(a.get("stream_time") or 0), float(b.get("stream_time") or 0)
                obs_a = set(int(x) for x in (a.get("observed_bins") or []))
                obs_b = set(int(x) for x in (b.get("observed_bins") or []))
                rows.append({
                    "video_id": vid, "task": TASK_BY_VIDEO.get(vid),
                    "time_image": ta, "time_video": tb, "time_diff": tb - ta,
                    "k_image": a.get("compaction_index"), "k_video": b.get("compaction_index"),
                    "layer_idx": int(layer), "layer_type": _lt(int(layer), len(ia)),
                    "bins_image": json.dumps(sorted(bins_a)), "bins_video": json.dumps(sorted(bins_b)),
                    "n_inter": inter, "n_union": union, "temporal_bin_jaccard": j,
                    "coverage_image": (len(bins_a & obs_a) / len(obs_a)) if obs_a else None,
                    "coverage_video": (len(bins_b & obs_b) / len(obs_b)) if obs_b else None,
                    "oldest_age_image": (ta - sa["oldest_ts"]) if sa.get("oldest_ts") is not None else None,
                    "oldest_age_video": (tb - sb["oldest_ts"]) if sb.get("oldest_ts") is not None else None,
                })
    _csv(analysis / "B_temporal_overlap.csv", rows)
    return rows


def layer_summary(retention: List[dict], overlap: List[dict]) -> List[dict]:
    rows = []
    for style in STYLES:
        for lt in ("short", "mid", "long"):
            sub = [r for r in retention if r["arm"] == style and r["layer_type"] == lt]
            ov = [r for r in overlap if r["layer_type"] == lt]
            cov = [r["temporal_coverage_ratio"] for r in sub if r["temporal_coverage_ratio"] is not None]
            old = [r["oldest_retained_age"] for r in sub if r["oldest_retained_age"] is not None]
            med = [r["median_retained_age"] for r in sub if r["median_retained_age"] is not None]
            rows.append({
                "arm": style, "layer_type": lt, "n_events": len(sub),
                "mean_coverage": statistics.mean(cov) if cov else None,
                "mean_oldest_age": statistics.mean(old) if old else None,
                "median_oldest_age": statistics.median(old) if old else None,
                "mean_median_age": statistics.mean(med) if med else None,
                "n_paired_events": len(ov),
                "mean_temporal_bin_jaccard": statistics.mean(
                    [r["temporal_bin_jaccard"] for r in ov if r["temporal_bin_jaccard"] is not None]
                ) if ov else None,
            })
    return rows


def decision_rows(data) -> List[dict]:
    img = {str(r.get("sample_id")): r for r in data["image"]["paired"]}
    vid = {str(r.get("sample_id")): r for r in data["video"]["paired"]}
    out = []
    for sid in sorted(set(img) | set(vid), key=lambda s: (len(s), s)):
        a, b = img.get(sid, {}), vid.get(sid, {})
        letters = a.get("option_token_ids") or b.get("option_token_ids") or {}

        def score(logits, gold):
            if not logits or gold not in letters:
                return None, None
            per = {}
            for L, ids in letters.items():
                vals = [logits.get(str(i)) for i in ids]
                vals = [v for v in vals if v is not None]
                if vals:
                    per[L] = max(vals)
            if gold not in per:
                return None, None
            wrong = [v for L, v in per.items() if L != gold]
            if not wrong:
                return None, None
            mx = max(per.values())
            exps = {L: math.exp(v - mx) for L, v in per.items()}
            tot = sum(exps.values())
            return exps[gold] / tot, per[gold] - max(wrong)

        gold = a.get("task_gold") or b.get("task_gold")
        p_i, m_i = score(a.get("option_token_logits_compressed"), gold)
        p_v, m_v = score(b.get("option_token_logits_compressed"), gold)
        ci = a.get("task_correct_compressed")
        cv = b.get("task_correct_compressed")
        out.append({
            "sample_id": sid, "video_id": sid, "task": a.get("task_type") or b.get("task_type"),
            "gold": gold,
            "prediction_image": a.get("task_prediction_compressed"),
            "prediction_video": b.get("task_prediction_compressed"),
            "correct_image": ci, "correct_video": cv,
            "prediction_changed": a.get("task_prediction_compressed") != b.get("task_prediction_compressed"),
            "P_gold_image": p_i, "P_gold_video": p_v,
            "dP_gold": (p_i - p_v) if (p_i is not None and p_v is not None) else None,
            "M_z_image": m_i, "M_z_video": m_v,
            "dM_z": (m_i - m_v) if (m_i is not None and m_v is not None) else None,
            "margin_crossed": (None if (m_i is None or m_v is None) else (m_i > 0) != (m_v > 0)),
            "transition_CC": bool(cv and ci),
            "transition_CW": bool(cv and not ci),
            "transition_WC": bool((not cv) and ci),
            "transition_WW": bool((not cv) and (not ci)),
        })
    return out


def plot_all(tok, ret, ov, dec, lsum, sched, out: Path) -> List[Path]:
    """The nine required experiment-B figures (nothing smoothed/hidden)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs: List[Path] = []
    videos = sorted({r["video_id"] for r in tok}, key=lambda v: (len(v), v))

    def save(fig, name):
        for ext in ("png", "pdf"):
            p = figdir / f"{name}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            figs.append(p)
        plt.close(fig)

    # 1/2) tokens per chunk & per sec
    for key, name, ylab in (("visual_tokens", "tokens_per_chunk_image_vs_video",
                             "visual tokens / chunk"),
                            ("visual_tokens_per_sec", "tokens_per_sec_image_vs_video",
                             "visual tokens / sec")):
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        width = 0.38
        for i, style in enumerate(STYLES):
            ys = [statistics.mean([float(r[key]) for r in tok
                                   if r["arm"] == style and r["video_id"] == v and r[key] is not None])
                  for v in videos]
            ax.bar([x + (i - 0.5) * width for x in range(len(videos))], ys, width, label=style)
            for x, y in zip(range(len(videos)), ys):
                ax.annotate(f"{y:.0f}", (x + (i - 0.5) * width, y), ha="center",
                            va="bottom", fontsize=7)
        ax.set_xticks(range(len(videos)))
        ax.set_xticklabels([f"{v}\n{TASK_BY_VIDEO.get(v,'')}" for v in videos], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}: image vs video encoding")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        save(fig, name)

    # 3) compaction count
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ks = {s: {str(r["video_id"]): int(r["n_effective_compactions"]) for r in sched
              if r["arm"] == s} for s in STYLES}
    width = 0.38
    for i, style in enumerate(STYLES):
        ys = [ks[style].get(v, 0) for v in videos]
        ax.bar([x + (i - 0.5) * width for x in range(len(videos))], ys, width, label=style)
        for x, y in zip(range(len(videos)), ys):
            ax.annotate(str(y), (x + (i - 0.5) * width, y), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(videos)))
    ax.set_xticklabels([f"{v}\n{TASK_BY_VIDEO.get(v,'')}" for v in videos], fontsize=8)
    ax.set_ylabel("effective compactions (k)")
    ax.set_title("Compression count at the same numerical budget (2000)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "compaction_count_image_vs_video")

    # 4/5/6) coverage / oldest age / median age vs stream time
    for key, name, ylab in (("temporal_coverage_ratio", "temporal_coverage_vs_stream_time_per_video",
                             "temporal coverage ratio"),
                            ("oldest_retained_age", "oldest_retained_age_vs_stream_time",
                             "oldest retained age (s)"),
                            ("median_retained_age", "median_retained_age_vs_stream_time",
                             "median retained age (s)")):
        for v in videos:
            fig, ax = plt.subplots(figsize=(7.6, 3.8))
            for style in STYLES:
                pts = {}
                for r in ret:
                    if r["arm"] != style or r["video_id"] != v or r[key] is None:
                        continue
                    pts.setdefault(float(r["stream_time"]), []).append(float(r[key]))
                xs = sorted(pts)
                ys = [statistics.mean(pts[x]) for x in xs]
                ax.plot(xs, ys, "-", label=style)
            ax.set_xlabel("stream time (s)")
            ax.set_ylabel(ylab)
            ax.set_title(f"{ylab} — video {v} ({TASK_BY_VIDEO.get(v,'')})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            save(fig, f"{name}_video{v}")

    # 7) temporal overlap vs stream time (paired events only)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for v in videos:
        pts = {}
        for r in ov:
            if r["video_id"] != v or r["temporal_bin_jaccard"] is None:
                continue
            pts.setdefault(float(r["time_video"]), []).append(float(r["temporal_bin_jaccard"]))
        xs = sorted(pts)
        if xs:
            ax.plot(xs, [statistics.mean(pts[x]) for x in xs], "-", label=f"{v}")
    ax.axhline(TH_RET_JACCARD, color="0.6", ls="--", lw=1,
               label=f"near-equivalent threshold {TH_RET_JACCARD}")
    ax.axhline(TH_DIV_JACCARD, color="r", ls=":", lw=1,
               label=f"material threshold {TH_DIV_JACCARD}")
    ax.set_xlabel("stream time (s)")
    ax.set_ylabel("temporal-bin Jaccard")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Temporal-bin overlap at stream-time-aligned compactions")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    save(fig, "temporal_overlap_vs_stream_time")

    # 8) coverage by layer type
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    lts = ["short", "mid", "long"]
    width = 0.38
    for i, style in enumerate(STYLES):
        ys = []
        for lt in lts:
            vals = [r["mean_coverage"] for r in lsum
                    if r["arm"] == style and r["layer_type"] == lt and r["mean_coverage"] is not None]
            ys.append(vals[0] if vals else 0.0)
        ax.bar([x + (i - 0.5) * width for x in range(len(lts))], ys, width, label=style)
        for x, y in zip(range(len(lts)), ys):
            ax.annotate(f"{y:.3f}", (x + (i - 0.5) * width, y), ha="center",
                        va="bottom", fontsize=7)
    ax.set_xticks(range(len(lts)))
    ax.set_xticklabels(lts)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mean temporal coverage ratio")
    ax.set_title("Temporal coverage by HERMES layer type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "coverage_by_layer_type")

    # 9) decision shift (image - video)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    xs = [str(d["video_id"]) for d in dec]
    for ax, key, title in ((axes[0], "dP_gold", "ΔP_gold (IMAGE − VIDEO)"),
                           (axes[1], "dM_z", "ΔM_z (IMAGE − VIDEO)")):
        ys = [float(d.get(key) or 0.0) for d in dec]
        ax.bar(xs, ys)
        ax.axhline(0.0, color="0.6")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    save(fig, "decision_shift_image_minus_video")
    return figs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ENC_ROOT / "analysis"))
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args(argv)
    analysis = Path(args.out)
    analysis.mkdir(parents=True, exist_ok=True)

    data = {s: load(s) for s in STYLES}
    n_layers = 28
    for s in STYLES:
        if data[s]["dumps"]:
            n_layers = len(data[s]["dumps"][0]["retained_visual_uids_by_layer"])
            break

    cons_rows, cons_valid = input_consistency(analysis, data)
    tok = tokens(analysis, data)
    sched = schedule(analysis, data)
    ret = temporal_retention(analysis, data)
    ov = temporal_overlap(analysis, data)
    lsum = layer_summary(ret, ov)
    _csv(analysis / "B_layer_summary.csv", lsum)
    dec = decision_rows(data)
    _csv(analysis / "B_decision.csv", dec)

    # aggregates
    def mean(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(v) if v else None

    k = {s: {r["video_id"]: r["n_effective_compactions"] for r in sched if r["arm"] == s}
         for s in STYLES}
    k_ratio = None
    if k["image"] and k["video"]:
        ratios = [k["image"][v] / k["video"][v] for v in k["image"]
                  if k["video"].get(v)]
        k_ratio = statistics.median(ratios) if ratios else None
    summary = {
        "num_layers": n_layers,
        "time_bin_sec": TIME_BIN_SEC,
        "input_consistency_valid": all(cons_valid.values()) if cons_valid else None,
        "tokens": {
            "mean_visual_tokens_per_chunk_image": mean([r for r in tok if r["arm"] == "image"], "visual_tokens"),
            "mean_visual_tokens_per_chunk_video": mean([r for r in tok if r["arm"] == "video"], "visual_tokens"),
            "mean_visual_tokens_per_sec_image": mean([r for r in tok if r["arm"] == "image"], "visual_tokens_per_sec"),
            "mean_visual_tokens_per_sec_video": mean([r for r in tok if r["arm"] == "video"], "visual_tokens_per_sec"),
        },
        "k_image": k["image"], "k_video": k["video"],
        "median_k_ratio_image_over_video": k_ratio,
        "mean_temporal_bin_jaccard": mean(ov, "temporal_bin_jaccard"),
        "mean_coverage_image": mean([r for r in ret if r["arm"] == "image"], "temporal_coverage_ratio"),
        "mean_coverage_video": mean([r for r in ret if r["arm"] == "video"], "temporal_coverage_ratio"),
        "mean_oldest_age_image": mean([r for r in ret if r["arm"] == "image"], "oldest_retained_age"),
        "mean_oldest_age_video": mean([r for r in ret if r["arm"] == "video"], "oldest_retained_age"),
        "median_oldest_age_image": statistics.median(
            [r["oldest_retained_age"] for r in ret if r["arm"] == "image" and r["oldest_retained_age"] is not None] or [0]),
        "median_oldest_age_video": statistics.median(
            [r["oldest_retained_age"] for r in ret if r["arm"] == "video" and r["oldest_retained_age"] is not None] or [0]),
        "n_paired_events": len(ov),
        "layer_summary": lsum,
        "decisions": dec,
    }
    (analysis / "B_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                             encoding="utf-8")

    if not args.no_plots and ret:
        figs = plot_all(tok, ret, ov, dec, lsum, sched, analysis)
        print(f"figures={len(figs)} -> {analysis / 'figures'}")

    print(f"[enc-analyze] layers={n_layers} bin={TIME_BIN_SEC}s "
          f"input_consistency={summary['input_consistency_valid']}")
    def _f(v, nd=1):
        return "n/a" if v is None else f"{v:.{nd}f}"

    print(f"  tokens/chunk  image={_f(summary['tokens']['mean_visual_tokens_per_chunk_image'])} "
          f"video={_f(summary['tokens']['mean_visual_tokens_per_chunk_video'])}")
    print(f"  tokens/sec    image={_f(summary['tokens']['mean_visual_tokens_per_sec_image'],2)} "
          f"video={_f(summary['tokens']['mean_visual_tokens_per_sec_video'],2)}")
    print(f"  k  image={k['image']}")
    print(f"  k  video={k['video']}")
    print(f"  median k ratio (image/video) = {k_ratio}")
    print(f"  paired events={len(ov)} mean temporal-bin J={summary['mean_temporal_bin_jaccard']}")
    print(f"  coverage image={summary['mean_coverage_image']} video={summary['mean_coverage_video']}")
    print(f"  oldest age image={summary['mean_oldest_age_image']} video={summary['mean_oldest_age_video']}")
    print(f"-> {analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
