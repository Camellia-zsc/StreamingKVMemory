"""Task-decision-layer fidelity: ΔP_gold and ΔM_z.

Both are computed from the OFFICIAL answer-generation step-0 logits (captured
read-only, restricted to the allowed MCQ option tokens). No re-generation, no
second decoder, no changes to the inference path.

Variant aggregation (important, do NOT mix them up):

* option probability uses **logsumexp** over the option's token variants
  (``"A"`` and ``" A"``), then re-normalises across the legal options -- this
  is the probability that the model answers that option GIVEN the constraint.
* decision margin uses **max** over the variants, because the real decoder
  takes ``argmax`` over allowed TOKEN ids, not over summed option scores.
  ``M_z = score(gold) - max_{c != gold} score(c)`` therefore shares the exact
  decision boundary of the constrained decoder.
"""

import math
from typing import Dict, List, Optional, Tuple

from . import statistics as st
from .task_accuracy import sample_key


def _logsumexp(values: List[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def option_scores(logits_by_token: Dict[str, float], option_ids: Dict[str, List[int]]):
    """Return (probabilities, max_variant_scores) per option letter."""

    logsum: Dict[str, float] = {}
    best: Dict[str, float] = {}
    for letter, ids in option_ids.items():
        vals = [float(logits_by_token[str(int(i))]) for i in ids
                if str(int(i)) in logits_by_token]
        if not vals:
            continue
        logsum[letter] = _logsumexp(vals)
        best[letter] = max(vals)
    if not logsum:
        return {}, {}
    z = max(logsum.values())
    denom = sum(math.exp(v - z) for v in logsum.values())
    probs = {k: math.exp(v - z) / denom for k, v in logsum.items()}
    return probs, best


def decision_margin(best: Dict[str, float], gold: str) -> Optional[float]:
    if gold not in best:
        return None
    others = [v for k, v in best.items() if k != gold]
    if not others:
        return None
    return best[gold] - max(others)


def _side(record: dict, side: str):
    return record.get(f"option_token_logits_{side}")


def compute_side(record: dict, side: str) -> dict:
    """Per-side option probabilities / margin / argmax for one query."""

    logits = _side(record, side)
    option_ids = record.get("option_token_ids") or {}
    gold = str(record.get("task_gold") or "").strip().upper()[:1]
    out = {
        f"p_option_{side}": None,
        f"p_gold_{side}": None,
        f"margin_{side}": None,
        f"best_option_{side}": None,
        f"runner_up_{side}": None,
        f"runner_up_score_{side}": None,
    }
    if not logits or not option_ids:
        return out
    probs, best = option_scores(logits, option_ids)
    if not probs:
        return out
    out[f"p_option_{side}"] = probs
    out[f"p_gold_{side}"] = probs.get(gold) if gold in probs else None
    out[f"margin_{side}"] = decision_margin(best, gold)
    order = sorted(best, key=lambda k: (-best[k], k))
    out[f"best_option_{side}"] = order[0] if order else None
    if len(order) > 1:
        out[f"runner_up_{side}"] = order[1]
        out[f"runner_up_score_{side}"] = best[order[1]]
    return out


def compute_row(record: dict) -> dict:
    full = compute_side(record, "full")
    comp = compute_side(record, "compressed")
    row = {
        "video_id": str(record.get("video_id") or "").replace("sample_", ""),
        "query_id": record.get("query_id"),
        "query_time": record.get("query_time"),
        "effective_compaction_count": record.get(
            "effective_compaction_count_at_query"
        ),
        "gold_option": str(record.get("task_gold") or "").strip().upper()[:1] or None,
        "prediction_full": record.get("task_prediction_full"),
        "prediction_compressed": record.get("task_prediction_compressed"),
        "task_transition": record.get("task_transition"),
        **full, **comp,
    }
    if row["p_gold_full"] is not None and row["p_gold_compressed"] is not None:
        row["delta_p_gold"] = row["p_gold_compressed"] - row["p_gold_full"]
    else:
        row["delta_p_gold"] = None
    if row["margin_full"] is not None and row["margin_compressed"] is not None:
        row["delta_margin_logit"] = row["margin_compressed"] - row["margin_full"]
    else:
        row["delta_margin_logit"] = None

    mismatch = False
    for side in ("full", "compressed"):
        pred = row[f"prediction_{side}"]
        besto = row[f"best_option_{side}"]
        if pred is not None and besto is not None:
            if str(pred).strip().upper()[:1] != besto:
                mismatch = True
    row["decision_metric_mismatch"] = mismatch
    # margin crossing: gold wins on Full but loses after compression
    row["margin_crossed"] = bool(
        row["margin_full"] is not None and row["margin_compressed"] is not None
        and row["margin_full"] > 0 > row["margin_compressed"]
    )
    return row


def build_rows(records: List[dict]) -> List[dict]:
    return [compute_row(r) for r in records]


# --------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------- #
def _bin_label(k: int, edges) -> str:
    if edges is None:
        return str(k)
    lo = 0
    for i, edge in enumerate(edges):
        if k <= edge:
            return f"{lo}" if lo == edge else f"{lo}-{edge}"
        lo = edge + 1
    return f"{lo}+"


def aggregate(rows: List[dict], bin_edges=None, n_boot: int = 2000, seed: int = 42):
    """Per-bin ΔP_gold / ΔM_z, aggregated with the FROZEN rule:

    1. average multiple queries of the SAME video inside the same (bin / exact k)
       -> one value per (video, bin);
    2. then take mean / median across videos, with a VIDEO-level cluster
       bootstrap CI (n_boot, seed).

    ``num_videos`` therefore counts distinct SOURCE VIDEOS, and
    ``fraction_*_negative`` is the fraction of (video, bin) units that are < 0.
    """

    usable = [r for r in rows if r.get("delta_p_gold") is not None
              or r.get("delta_margin_logit") is not None]
    if bin_edges is not None:
        labels = []
        lo = 0
        for edge in bin_edges:
            labels.append((lo, edge))
            lo = edge + 1
        labels.append((lo, None))
    else:
        keys = sorted({int(r["effective_compaction_count"]) for r in usable
                       if r.get("effective_compaction_count") is not None})
        labels = [(k, k) for k in keys]

    metrics = ("p_gold_full", "p_gold_compressed", "delta_p_gold",
               "margin_full", "margin_compressed", "delta_margin_logit")
    out = []
    for lo, hi in labels:
        group = [
            r for r in usable
            if r.get("effective_compaction_count") is not None
            and lo <= int(r["effective_compaction_count"])
            and (hi is None or int(r["effective_compaction_count"]) <= hi)
        ]
        if not group:
            continue
        label = f"{lo}" if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")

        # step 1: one value per (video, metric)
        per_video: Dict[str, Dict[str, float]] = {}
        for r in group:
            vid = sample_key({"video_id": r["video_id"]})
            bucket = per_video.setdefault(vid, {})
            for key in metrics:
                v = r.get(key)
                if v is None:
                    continue
                bucket.setdefault(key, []).append(float(v))
        per_video_mean = {
            vid: {k: sum(v) / len(v) for k, v in vals.items()}
            for vid, vals in per_video.items()
        }

        row = {"bin_label": label, "k_lo": lo, "k_hi": hi,
               "num_queries": len(group), "num_videos": len(per_video_mean)}
        for key in metrics:
            vals = [m[key] for m in per_video_mean.values() if key in m]
            row[f"{key}_median"] = st.summarize(vals)["median"]
            row[f"{key}_mean"] = st.summarize(vals)["mean"]
        for key in ("delta_p_gold", "delta_margin_logit"):
            vals = [m[key] for m in per_video_mean.values() if key in m]
            ci = st.cluster_bootstrap_ci([[v] for v in vals], n_boot=n_boot, seed=seed)
            row[f"{key}_mean_ci_low"] = ci["ci_low"]
            row[f"{key}_mean_ci_high"] = ci["ci_high"]
            row[f"fraction_{key}_negative"] = (
                sum(1 for v in vals if v < 0) / len(vals) if vals else None
            )
        row["num_margin_crossed"] = sum(1 for r in group if r.get("margin_crossed"))
        row["num_decision_metric_mismatch"] = sum(
            1 for r in group if r.get("decision_metric_mismatch")
        )
        out.append(row)
    return out


# --------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------- #
def _plot_delta(rows, bins, key, ylabel, name, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import _apply_style, _save

    _apply_style()
    figs = []

    # exact-k raw curve
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    by_k: Dict[int, List[float]] = {}
    for r in rows:
        v = r.get(key)
        k = r.get("effective_compaction_count")
        if v is None or k is None:
            continue
        by_k.setdefault(int(k), []).append(float(v))
    if by_k:
        ks = sorted(by_k)
        med = [st.summarize(by_k[k])["median"] for k in ks]
        mean = [st.summarize(by_k[k])["mean"] for k in ks]
        ax.plot(ks, mean, "-o", ms=3, label="mean")
        ax.plot(ks, med, "--s", ms=3, label="median")
        ax.axhline(0.0, color="0.8")
        ax.legend(fontsize=8)
        # annotate n_video(k) so sparse high-k points are obvious
        per_video = {}
        for r in rows:
            k = r.get("effective_compaction_count")
            if k is None:
                continue
            per_video.setdefault(int(k), set()).add(str(r.get("video_id")))
        for k in ks:
            ax.annotate(f"{len(per_video.get(k, ()))}", (k, mean[ks.index(k)]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6, color="0.35")
    ax.set_xlabel("effective_compaction_count (exact k)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs exact k")
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, out_dir, f"{name}_exact_k",
                      "Raw per-query statistics (no smoothing, no binning). "
                      "Small numbers above each point = n_video(k)."))

    # locked-bin with CI
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    labels = [b["bin_label"] for b in bins]
    mean = [b.get(f"{key}_mean") for b in bins]
    lo = [b.get(f"{key}_mean_ci_low") for b in bins]
    hi = [b.get(f"{key}_mean_ci_high") for b in bins]
    ns = [b["num_queries"] for b in bins]
    valid = [(i, m, l, h, n) for i, (m, l, h, n) in enumerate(zip(mean, lo, hi, ns))
             if m is not None]
    if valid:
        idx, m, l, h, n = zip(*valid)
        yerr = [[mi - (li if li is not None else mi), (hi_ if hi_ is not None else mi) - mi]
                for mi, li, hi_ in zip(m, l, h)]
        ax.errorbar(idx, m, yerr=list(zip(*yerr)), marker="o", capsize=4)
        ax.axhline(0.0, color="0.8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        for i, mi, ni in zip(idx, m, n):
            ax.annotate(f"Nq={ni}", (i, mi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("locked effective-compaction bin")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by locked bin")
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, out_dir, f"{name}_locked_bins",
                      "Video-level cluster bootstrap, n_boot=2000, seed=42."))
    return figs


def _plot_scatter(rows, xkey, ykey, name, xlabel, ylabel, out_dir, color_key=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import _apply_style, _save

    _apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    pts = [(r[xkey], r[ykey], r.get(color_key)) for r in rows
           if r.get(xkey) is not None and r.get(ykey) is not None]
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color="0.7", linestyle="--", label="y = x")
        if color_key:
            groups: Dict[str, List[tuple]] = {}
            for x, y, c in pts:
                groups.setdefault(str(c), []).append((x, y))
            for key, values in sorted(groups.items()):
                ax.scatter([v[0] for v in values], [v[1] for v in values], s=18, label=key)
            ax.legend(fontsize=7)
        else:
            ax.scatter(xs, ys, s=18)
        below = sum(1 for x, y, _ in pts if y < x)
        ax.set_title(f"{ylabel.split(chr(40))[0].strip()} vs {xlabel.split(chr(40))[0].strip()}", fontsize=12)
        ax.text(0.02, 0.02, f"{below}/{len(pts)} below y=x", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, name,
                 "Points below the diagonal = compression REDUCED the quantity.")
