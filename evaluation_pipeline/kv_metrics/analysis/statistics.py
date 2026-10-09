"""Statistics helpers for the fidelity experiment analysis.

All bootstrap / significance units are **samples**, never events, to avoid
pseudo-replication: events are first aggregated to one statistic per sample,
then samples are resampled.
"""

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------- #
# descriptive
# --------------------------------------------------------------------- #
def _clean(values: Sequence[Optional[float]]) -> List[float]:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        out.append(f)
    return out


def summarize(values: Sequence[Optional[float]]) -> dict:
    vals = _clean(values)
    n_total = len(values)
    if not vals:
        return {
            "n_valid": 0,
            "n_missing": n_total,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    s = sorted(vals)
    return {
        "n_valid": len(vals),
        "n_missing": n_total - len(vals),
        "mean": sum(vals) / len(vals),
        "median": _quantile(s, 0.5),
        "p10": _quantile(s, 0.1),
        "p90": _quantile(s, 0.9),
        "min": s[0],
        "max": s[-1],
    }


def _quantile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


# --------------------------------------------------------------------- #
# bootstrap (unit = sample)
# --------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[Optional[float]],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI of the **mean**, resampling units (samples)."""

    vals = _clean(values)
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_boot": 0}
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": sum(vals) / n,
        "median": _quantile(sorted(vals), 0.5),
        "ci_low": _quantile(means, alpha),
        "ci_high": _quantile(means, 1 - alpha),
        "n": n,
        "n_boot": n_boot,
        "ci": ci,
        "seed": seed,
    }


def paired_bootstrap_ci(
    deltas: Sequence[Optional[float]],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI of the mean of (already computed) paired deltas."""

    return bootstrap_ci(deltas, n_boot=n_boot, ci=ci, seed=seed)


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence[float]],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI with the **sample** as the resampling unit.

    ``clusters`` is a sequence of per-sample value lists (all queries of a
    sample stay together). The statistic is the pooled mean across all values,
    which is exactly accuracy when the values are 0/1.
    """

    groups = [[float(v) for v in g if v is not None] for g in clusters]
    groups = [g for g in groups if g]
    if not groups:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_samples": 0,
                "n_queries": 0, "n_boot": 0}

    def pooled(sample_groups):
        total = [v for g in sample_groups for v in g]
        return sum(total) / len(total) if total else None

    rng = random.Random(seed)
    n = len(groups)
    means = []
    for _ in range(n_boot):
        draw = [groups[rng.randrange(n)] for _ in range(n)]
        value = pooled(draw)
        if value is not None:
            means.append(value)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": pooled(groups),
        "ci_low": _quantile(means, alpha) if means else None,
        "ci_high": _quantile(means, 1 - alpha) if means else None,
        "n_samples": n,
        "n_queries": sum(len(g) for g in groups),
        "n_boot": n_boot,
        "ci": ci,
        "seed": seed,
    }


# --------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------- #
def sample_level_means(
    records: Sequence[dict],
    metric: str,
    sample_key: str = "sample_id",
) -> Dict[str, float]:
    """events -> one mean per sample (avoids event over-weighting)."""

    buckets: Dict[str, List[float]] = {}
    for rec in records:
        value = rec.get(metric)
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        key = str(rec.get(sample_key) or rec.get("video_id"))
        buckets.setdefault(key, []).append(f)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


# --------------------------------------------------------------------- #
# significance
# --------------------------------------------------------------------- #
def wilcoxon_signed_rank(
    x: Sequence[float],
    y: Sequence[float],
    min_pairs: int = 5,
) -> dict:
    """Two-sided Wilcoxon signed-rank test, only when >= ``min_pairs`` pairs."""

    pairs = [
        (a, b)
        for a, b in zip(x, y)
        if a is not None and b is not None
    ]
    if len(pairs) < min_pairs:
        return {
            "test": "wilcoxon_signed_rank",
            "n_pairs": len(pairs),
            "skipped": True,
            "reason": f"need >= {min_pairs} valid paired samples",
        }
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(
            [a for a, _ in pairs], [b for _, b in pairs]
        )
        return {
            "test": "wilcoxon_signed_rank",
            "n_pairs": len(pairs),
            "skipped": False,
            "statistic": float(stat),
            "p_value": float(p),
        }
    except Exception as error:  # noqa: BLE001
        return {
            "test": "wilcoxon_signed_rank",
            "n_pairs": len(pairs),
            "skipped": True,
            "reason": f"{type(error).__name__}: {error}",
        }


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    pairs = [
        (a, b) for a, b in zip(x, y) if a is not None and b is not None
    ]
    if len(pairs) < 3:
        return None
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        if math.isnan(rho):
            return None
        return float(rho)
    except Exception:  # noqa: BLE001
        return None


def point_biserial(binary: Sequence[float], metric: Sequence[float]) -> Optional[float]:
    """Point-biserial correlation (= Pearson with a 0/1 variable)."""

    pairs = [
        (a, b)
        for a, b in zip(binary, metric)
        if a is not None and b is not None
    ]
    if len(pairs) < 3:
        return None
    try:
        from scipy.stats import pointbiserialr

        r, _ = pointbiserialr(
            [a for a, _ in pairs], [b for _, b in pairs]
        )
        if math.isnan(r):
            return None
        return float(r)
    except Exception:  # noqa: BLE001
        return None
