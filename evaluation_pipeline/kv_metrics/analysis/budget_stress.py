"""KV-budget stress test: the same 140 queries at several HERMES budgets.

Fixed sample set (28 videos x 5 real queries = 140), fixed full_prefix
[0, tq], fixed everything except ``memory.compression.budget``. The Full-KV
branch is budget-independent and is REUSED, never re-run.

Main aggregation rule (frozen): average a video's queries first, then take
mean / median across videos; the cluster bootstrap unit is the video.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from . import loader, statistics as st
from .decision_fidelity import compute_side
from .task_accuracy import sample_key


BUDGET_ORDER = [2000, 1500, 1000, 750, 100]  # default only; analysis is dynamic

# ---- figure-only scope -------------------------------------------------- #
# budget=80 is kept in the summary/CSV but dropped from every budget-vs-*
# figure so the low-budget x labels stay readable. This affects plotting ONLY.
PLOT_EXCLUDE_BUDGETS = {80}
# Every budget figure shares this width / tick style so the x-axis matches.
PLOT_FIGSIZE = (10.5, 4.6)
PLOT_TICK_ROTATION = 40
PLOT_TICK_FONTSIZE = 8


def _plot_budgets(rows: List[dict]) -> List[int]:
    """Budgets drawn on budget-vs-* figures (figures only, never the CSV).

    Drops :data:`PLOT_EXCLUDE_BUDGETS` and orders the rest from the largest to
    the smallest budget, so every budget figure shows the identical x set.
    """

    seen = {
        int(r["budget"])
        for r in rows
        if r.get("budget") is not None and int(r["budget"]) not in PLOT_EXCLUDE_BUDGETS
    }
    return sorted(seen, reverse=True)


def _rows_at(rows: List[dict], budget: int) -> List[dict]:
    return [r for r in rows if r.get("budget") is not None and int(r["budget"]) == budget]


def _style_budget_axis(ax, budgets: List[int]) -> None:
    """Real numeric x positions + rotated, small ticks to avoid overlap."""

    ax.set_xscale("linear")
    ax.set_xticks(budgets)
    ax.set_xticklabels(
        [str(int(b)) for b in budgets],
        rotation=PLOT_TICK_ROTATION,
        ha="right",
        fontsize=PLOT_TICK_FONTSIZE,
    )


def _video(row: dict) -> str:
    return sample_key({"video_id": row.get("video_id")})


FULL_SIDE_ALIASES = {
    "task_prediction_full": ("task_prediction_full", "task_prediction",
                             "task_prediction_compressed"),
    "task_correct_full": ("task_correct_full", "task_correct",
                          "task_correct_compressed"),
    "task_status_full": ("task_status_full", "task_status",
                         "task_status_compressed"),
    "task_error_type_full": ("task_error_type_full", "task_error_type",
                             "task_error_type_compressed"),
    "option_token_logits_full": ("option_token_logits_full",
                                 "option_token_logits",
                                 "option_token_logits_compressed"),
}


def normalize_full_record(record: dict) -> dict:
    """Expose a Full-branch record under canonical ``*_full`` names.

    The Full-only pass reuses the single-pass writer, which historically named
    its fields ``*_compressed`` (and before that the bare legacy names). The
    raw files are NEVER rewritten -- we only alias on read so the join can find
    the Full branch. A genuinely canonical record is returned unchanged.
    """

    out = dict(record)
    for canonical, sources in FULL_SIDE_ALIASES.items():
        if out.get(canonical) is not None:
            continue
        for src in sources:
            if out.get(src) is not None:
                out[canonical] = out[src]
                break
    return out


def load_budget_records(exp_dirs) -> List[dict]:
    """Records from one or more experiment dirs (``raw_decision/*/paired``)."""

    if isinstance(exp_dirs, (str, Path)):
        exp_dirs = [exp_dirs]
    out = []
    for exp_dir in exp_dirs:
        for path in sorted((Path(exp_dir) / "raw_decision").glob("*/paired/*.jsonl")):
            out.extend(loader.read_jsonl(path))
    return out


def build_budget_table(budget_runs: Dict[int, object], full_records: List[dict],
                       n_boot: int = 2000, seed: int = 42):
    """Per-query table across budgets + per-budget summary."""

    # Full baseline keyed by (video, query_time) -- budget independent
    full_by_key = {}
    for r in full_records:
        key = (_video(r), round(float(r.get("query_time") or 0.0), 3))
        full_by_key[key] = normalize_full_record(r)

    order = sorted(budget_runs.keys(), reverse=True)
    per_query: Dict[tuple, dict] = {}
    for budget in order:
        exp = budget_runs.get(budget)
        if exp is None:
            continue
        for r in load_budget_records(exp):
            key = (_video(r), round(float(r.get("query_time") or 0.0), 3))
            entry = per_query.setdefault(key, {
                "video_id": _video(r),
                "query_id": r.get("query_id"),
                "query_time": r.get("query_time"),
            })
            base = full_by_key.get(key, {})
            merged = {
                **base,
                "task_gold": r.get("task_gold") or base.get("task_gold"),
                "option_token_ids": r.get("option_token_ids") or base.get("option_token_ids"),
                "option_token_logits_compressed": r.get("option_token_logits_compressed"),
                "option_token_logits_full": base.get("option_token_logits_full"),
                "task_prediction_compressed": r.get("task_prediction_compressed")
                or r.get("task_prediction"),
                "task_prediction_full": base.get("task_prediction_full"),
                "task_correct_compressed": r.get("task_correct_compressed")
                if r.get("task_correct_compressed") is not None else r.get("task_correct"),
                "task_correct_full": base.get("task_correct_full"),
            }
            from .decision_fidelity import compute_row

            row = compute_row(merged)
            row["effective_compaction_count"] = r.get(
                "effective_compaction_count_at_query"
            )
            prefix = f"budget{budget}"
            entry[f"{prefix}_k"] = row["effective_compaction_count"]
            entry[f"{prefix}_p_gold"] = row["p_gold_compressed"]
            entry[f"{prefix}_margin"] = row["margin_compressed"]
            entry[f"{prefix}_delta_p_gold"] = row["delta_p_gold"]
            entry[f"{prefix}_delta_margin"] = row["delta_margin_logit"]
            entry[f"{prefix}_prediction"] = merged.get("task_prediction_compressed")
            entry[f"{prefix}_correct"] = merged.get("task_correct_compressed")
            entry[f"{prefix}_best_option"] = row["best_option_compressed"]
            entry[f"{prefix}_mismatch"] = row["decision_metric_mismatch"]
            entry[f"{prefix}_margin_crossed"] = row["margin_crossed"]
            entry[f"{prefix}_full_margin"] = row["margin_full"]
            entry[f"{prefix}_comp_margin"] = row["margin_compressed"]
            entry.setdefault("gold_option", row.get("gold_option"))
            entry.setdefault("task_type", merged.get("task_type"))
            entry.setdefault("full_prediction", merged.get("task_prediction_full"))
            entry.setdefault("full_correct", merged.get("task_correct_full"))

    paired = sorted(per_query.values(), key=lambda r: (r["video_id"], r["query_time"] or 0))
    summary = summarize_budgets(paired, n_boot=n_boot, seed=seed, budgets=order)
    return paired, summary


def _budgets_in(paired: List[dict]) -> List[int]:
    found = set()
    for r in paired:
        for k in r:
            if k.startswith("budget") and k.endswith("_prediction"):
                mid = k[len("budget"):-len("_prediction")]
                if mid.isdigit():
                    found.add(int(mid))
    return sorted(found, reverse=True)


def summarize_budgets(paired: List[dict], n_boot: int = 2000, seed: int = 42,
                      budgets: Optional[List[int]] = None) -> List[dict]:
    rows = []
    for budget in (budgets if budgets is not None else _budgets_in(paired)):
        pref = f"budget{budget}"
        have = [r for r in paired if r.get(f"{pref}_prediction") is not None]
        if not have:
            continue
        # step 1: one value per video
        per_video: Dict[str, Dict[str, List[float]]] = {}
        trans = {"correct_to_correct": 0, "correct_to_wrong": 0,
                 "wrong_to_correct": 0, "wrong_to_wrong": 0}
        ks = []
        crossed = 0
        n_correct_c = n_correct_f = 0
        neg_p = neg_m = tot_p = tot_m = 0
        mismatch = 0
        for r in have:
            if r.get(f"{pref}_margin_crossed"):
                crossed += 1
            vid = r["video_id"]
            bucket = per_video.setdefault(vid, {"dp": [], "dm": []})
            if r.get(f"{pref}_delta_p_gold") is not None:
                bucket["dp"].append(float(r[f"{pref}_delta_p_gold"]))
                tot_p += 1
                if r[f"{pref}_delta_p_gold"] < 0:
                    neg_p += 1
            if r.get(f"{pref}_delta_margin") is not None:
                bucket["dm"].append(float(r[f"{pref}_delta_margin"]))
                tot_m += 1
                if r[f"{pref}_delta_margin"] < 0:
                    neg_m += 1
            if r.get(f"{pref}_k") is not None:
                ks.append(int(r[f"{pref}_k"]))
            c = r.get(f"{pref}_correct")
            f = r.get("full_correct")
            if c is not None:
                n_correct_c += int(bool(c))
            if f is not None:
                n_correct_f += int(bool(f))
            if c is not None and f is not None:
                if f and c:
                    trans["correct_to_correct"] += 1
                elif f and not c:
                    trans["correct_to_wrong"] += 1
                elif not f and c:
                    trans["wrong_to_correct"] += 1
                else:
                    trans["wrong_to_wrong"] += 1
            if r.get(f"{pref}_mismatch"):
                mismatch += 1
        dp_video = [sum(v["dp"]) / len(v["dp"]) for v in per_video.values() if v["dp"]]
        dm_video = [sum(v["dm"]) / len(v["dm"]) for v in per_video.values() if v["dm"]]
        ci_p = st.cluster_bootstrap_ci([[v] for v in dp_video], n_boot=n_boot, seed=seed)
        ci_m = st.cluster_bootstrap_ci([[v] for v in dm_video], n_boot=n_boot, seed=seed)
        full_correct_total = trans["correct_to_correct"] + trans["correct_to_wrong"]
        rows.append({
            "budget": budget,
            "N_video": len(per_video),
            "N_query": len(have),
            "mean_delta_p_gold_video_level": st.summarize(dp_video)["mean"],
            "median_delta_p_gold_video_level": st.summarize(dp_video)["median"],
            "delta_p_gold_ci_low": ci_p["ci_low"],
            "delta_p_gold_ci_high": ci_p["ci_high"],
            "mean_delta_margin_video_level": st.summarize(dm_video)["mean"],
            "median_delta_margin_video_level": st.summarize(dm_video)["median"],
            "delta_margin_ci_low": ci_m["ci_low"],
            "delta_margin_ci_high": ci_m["ci_high"],
            "fraction_query_delta_p_negative": (neg_p / tot_p) if tot_p else None,
            "fraction_query_delta_margin_negative": (neg_m / tot_m) if tot_m else None,
            "CW": trans["correct_to_wrong"], "CC": trans["correct_to_correct"],
            "WC": trans["wrong_to_correct"], "WW": trans["wrong_to_wrong"],
            "Preservation": (trans["correct_to_correct"] / full_correct_total)
            if full_correct_total else None,
            "CorrectToWrongRate": (trans["correct_to_wrong"] / full_correct_total)
            if full_correct_total else None,
            "Acc_full": (n_correct_f / len(have)) if have else None,
            "Acc_compressed": (n_correct_c / len(have)) if have else None,
            "DeltaAcc": ((n_correct_c - n_correct_f) / len(have)) if have else None,
            "mean_effective_compaction_count": st.summarize(ks)["mean"],
            "median_effective_compaction_count": st.summarize(ks)["median"],
            "decision_metric_mismatch": mismatch,
            "num_margin_crossed": crossed,
            "n_margin_full_pos_comp_neg": crossed,
        })
    return rows


def join_invariants(paired: List[dict]) -> dict:
    """Join-level invariants: N_full == N_compressed, transition sum, Δ coverage."""

    n = len(paired)
    have_full = sum(1 for r in paired if r.get("full_correct") is not None)
    have_full_pred = sum(1 for r in paired if r.get("full_prediction") is not None)
    both = [r for r in paired
            if r.get("full_correct") is not None and r.get("gold_option") is not None]
    trans_total = 0
    mismatch = 0
    budgets = sorted({k.split("_")[0] for r in paired for k in r if k.startswith("budget")})
    for pref in budgets:
        trans_total += sum(1 for r in paired if r.get(f"{pref}_correct") is not None
                           and r.get("full_correct") is not None)
    n_dp = sum(1 for r in paired if r.get("budget2000_delta_p_gold") is not None)
    n_dm = sum(1 for r in paired if r.get("budget2000_delta_margin") is not None)
    return {
        "n_samples": n,
        "n_with_full_correct": have_full,
        "n_with_full_prediction": have_full_pred,
        "n_with_both_sides": len(both),
        "n_delta_p_budget2000": n_dp,
        "n_delta_margin_budget2000": n_dm,
    }


def sanity_checks(paired: List[dict], summaries: List[dict]) -> List[str]:
    problems: List[str] = []
    n = len(paired)
    inv = join_invariants(paired)
    if n and inv["n_with_full_correct"] != n:
        problems.append(
            f"Full baseline joined for only {inv['n_with_full_correct']}/{n} samples "
            "(check Full-side canonical normalization)")
    for row in summaries:
        if row.get("N_query") != n:
            problems.append(f"budget {row['budget']}: N_query={row.get('N_query')} != {n}")
    for budget in BUDGET_ORDER:
        pref = f"budget{budget}"
        have = sum(1 for r in paired if r.get(f"{pref}_prediction") is not None)
        if have and have != n:
            problems.append(f"budget {budget}: only {have}/{n} queries have a prediction")
    for row in summaries:
        if row["decision_metric_mismatch"]:
            problems.append(
                f"budget {row['budget']}: best_option != official prediction for "
                f"{row['decision_metric_mismatch']} queries"
            )
    # lower budget should mean >= compaction exposure
    by_budget = {r["budget"]: r for r in summaries}
    ordered = [b for b in BUDGET_ORDER if b in by_budget]
    for hi, lo in zip(ordered, ordered[1:]):
        a = by_budget[hi]["mean_effective_compaction_count"]
        b = by_budget[lo]["mean_effective_compaction_count"]
        if a is not None and b is not None and b < a:
            problems.append(
                f"mean effective k decreased from budget {hi} ({a:.1f}) to "
                f"budget {lo} ({b:.1f})"
            )
    return problems


def write_csv(path: Path, rows: List[dict], fields=None) -> None:
    import csv as _csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


# --------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------- #
def plot_budget_curves(summaries: List[dict], out_dir: Path) -> List[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import _apply_style, _save

    _apply_style()
    rows = [r for r in summaries if r.get("budget") is not None
            and int(r["budget"]) not in PLOT_EXCLUDE_BUDGETS]
    budgets = _plot_budgets(rows)
    rows = sorted(rows, key=lambda r: -int(r["budget"]))
    figs = []

    def _fig(key_mean, key_med, ylabel, name, y0=False, notes=""):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        mean = [r.get(key_mean) for r in rows]
        med = [r.get(key_med) for r in rows]
        if any(v is not None for v in mean):
            ax.plot(budgets, mean, "-o", label="video-level mean")
        if any(v is not None for v in med):
            ax.plot(budgets, med, "--s", label="video-level median")
        if y0:
            ax.axhline(0.0, color="0.8")
        ax.set_xlabel("HERMES visual KV budget")
        ax.set_ylabel(ylabel)
        _style_budget_axis(ax, budgets)
        ax.set_title(f"{ylabel} vs KV budget")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        figs.append(_save(fig, out_dir, name,
                          "Frozen 28-video / 140-query set; only the KV budget changes. "
                          "Unit = video (queries averaged within a video first). " + notes))

    _fig("mean_delta_p_gold_video_level", "median_delta_p_gold_video_level",
         "ΔP_gold", "budget_vs_delta_p_gold", y0=True)
    _fig("mean_delta_margin_video_level", "median_delta_margin_video_level",
         "ΔM_z", "budget_vs_delta_margin", y0=True)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    cw = [r.get("CW") for r in rows]
    rate = [r.get("CorrectToWrongRate") for r in rows]
    ax.plot(budgets, cw, "-o", label="Correct→Wrong count")
    ax.plot(budgets, rate, "--s", label="Correct→Wrong rate")
    ax.set_xlabel("HERMES visual KV budget")
    ax.set_ylabel("Correct→Wrong")
    _style_budget_axis(ax, budgets)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, out_dir, "budget_vs_correct_to_wrong",
                      "rate = CW / (# queries where Full-KV was correct)."))

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    acc = [r.get("Acc_compressed") for r in rows]
    accf = [r.get("Acc_full") for r in rows]
    if any(v is not None for v in acc):
        ax.plot(budgets, acc, "-o", label="Compressed")
        ax.plot(budgets, accf, "--s", label="Full-KV (baseline)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    ax.set_xlabel("HERMES visual KV budget")
    ax.set_ylabel("Task accuracy")
    _style_budget_axis(ax, budgets)
    ax.set_title("Accuracy vs KV budget")
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, out_dir, "budget_vs_accuracy",
                      "Full-KV is budget independent; the same 140 queries are used everywhere."))

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    pres = [r.get("Preservation") for r in rows]
    ax.plot(budgets, pres, "-o")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("HERMES visual KV budget")
    ax.set_ylabel("Preservation")
    _style_budget_axis(ax, budgets)
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, out_dir, "budget_vs_preservation",
                      "Preservation = (Full correct AND Compressed correct) / (Full correct)."))

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    m = [r.get("mean_effective_compaction_count") for r in rows]
    md = [r.get("median_effective_compaction_count") for r in rows]
    ax.plot(budgets, m, "-o", label="mean k")
    ax.plot(budgets, md, "--s", label="median k")
    ax.set_xlabel("HERMES visual KV budget")
    ax.set_ylabel("effective compaction count at query")
    _style_budget_axis(ax, budgets)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, out_dir, "budget_vs_effective_compaction_count",
                      "Lower budget => more effective compactions before the query."))
    return figs


def paired_budget_comparison(paired: List[dict], hi: int, lo: int) -> dict:
    """Same-query comparison between two budgets (e.g. 750 vs 100)."""

    hp, lp = f"budget{hi}", f"budget{lo}"
    up_m = dn_m = eq_m = 0
    up_p = dn_p = eq_p = 0
    hi_ok_lo_bad = lo_ok_hi_bad = 0
    n = 0
    for r in paired:
        a, b = r.get(f"{hp}_delta_margin"), r.get(f"{lp}_delta_margin")
        if a is not None and b is not None:
            n += 1
            d = b - a
            if d > 1e-12: up_m += 1
            elif d < -1e-12: dn_m += 1
            else: eq_m += 1
        a2, b2 = r.get(f"{hp}_delta_p_gold"), r.get(f"{lp}_delta_p_gold")
        if a2 is not None and b2 is not None:
            d2 = b2 - a2
            if d2 > 1e-12: up_p += 1
            elif d2 < -1e-12: dn_p += 1
            else: eq_p += 1
        ch, cl = r.get(f"{hp}_correct"), r.get(f"{lp}_correct")
        if ch is not None and cl is not None:
            if ch and not cl: hi_ok_lo_bad += 1
            if cl and not ch: lo_ok_hi_bad += 1
    return {
        "hi_budget": hi, "lo_budget": lo, "n_queries": n,
        "delta_margin_up": up_m, "delta_margin_down": dn_m, "delta_margin_equal": eq_m,
        "delta_p_gold_up": up_p, "delta_p_gold_down": dn_p, "delta_p_gold_equal": eq_p,
        "hi_correct_lo_wrong": hi_ok_lo_bad,
        "lo_correct_hi_wrong": lo_ok_hi_bad,
    }


def task_breakdown(paired: List[dict], summaries: List[dict]) -> List[dict]:
    """Per-(budget, task_type) accuracy / preservation / CW rate."""

    out = []
    for row in summaries:
        budget = row["budget"]
        pref = f"budget{budget}"
        by_task: Dict[str, List[dict]] = {}
        for r in paired:
            by_task.setdefault(str(r.get("task_type") or "?"), []).append(r)
        for task, rs in sorted(by_task.items()):
            have = [r for r in rs if r.get(f"{pref}_prediction") is not None]
            if not have:
                continue
            cc = cw = wc = ww = 0
            for r in have:
                c = r.get(f"{pref}_correct"); f = r.get("full_correct")
                if c is None or f is None:
                    continue
                cc += int(f and c); cw += int(f and not c)
                wc += int((not f) and c); ww += int((not f) and (not c))
            totf = cc + cw
            out.append({
                "budget": budget, "task_type": task,
                "N_sample": len(have), "N_video": len(have),
                "Acc_full": sum(1 for r in have if r.get("full_correct")) / len(have),
                "Acc_compressed": sum(1 for r in have if r.get(f"{pref}_correct")) / len(have),
                "CC": cc, "CW": cw, "WC": wc, "WW": ww,
                "Preservation": cc / totf if totf else None,
                "CorrectToWrongRate": cw / totf if totf else None,
            })
    return out


def exposure_table(paired: List[dict], budgets: List[int]) -> List[dict]:
    """Per sample: effective compaction count at each budget (+ task type)."""

    rows = []
    for r in paired:
        row = {"sample_id": r.get("video_id"), "query_time": r.get("query_time"),
               "task_type": r.get("task_type")}
        for b in budgets:
            row[f"effective_compaction_count@{b}"] = r.get(f"budget{b}_k")
        rows.append(row)
    return rows


def plot_task_breakdown(task_rows: List[dict], out_dir: Path) -> List[dict]:
    """One line per task type for accuracy / preservation / CW rate."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import _apply_style, _save

    _apply_style()
    task_rows = [r for r in task_rows if r.get("budget") is not None
                 and int(r["budget"]) not in PLOT_EXCLUDE_BUDGETS]
    tasks = sorted({r["task_type"] for r in task_rows})
    budgets = _plot_budgets(task_rows)
    figs = []
    for key, ylabel, name, y0 in (
        ("Acc_compressed", "Compressed accuracy", "budget_vs_accuracy_by_task", False),
        ("Preservation", "Preservation", "budget_vs_preservation_by_task", False),
        ("CorrectToWrongRate", "Correct→Wrong rate", "budget_vs_cw_rate_by_task", True),
    ):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for task in tasks:
            xs = [b for b in budgets if any(r["task_type"] == task and r["budget"] == b for r in task_rows)]
            ys = []
            for b in xs:
                row = next(r for r in task_rows if r["task_type"] == task and r["budget"] == b)
                ys.append(row.get(key))
            if any(v is not None for v in ys):
                ax.plot(xs, ys, "-o", label=task)
        if y0:
            ax.axhline(0.0, color="0.8")
        ax.set_xlabel("HERMES visual KV budget")
        ax.set_ylabel(ylabel)
        _style_budget_axis(ax, budgets)
        ax.set_title(f"{ylabel} by task")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        figs.append(_save(fig, out_dir, name,
                          "OVO backward tasks; 1 query per sample; unit = sample/video."))
    return figs
