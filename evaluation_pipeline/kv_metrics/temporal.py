"""Temporal Fidelity Curve.

Only *Temporal RAM* is computed (the reliable variant). AORE / JS temporal
bucketing is only meaningful when the benchmark exposes an evidence timestamp,
so it is gated behind :func:`get_evidence_timestamp` and reported as
``supported=False`` otherwise. We never invent an evidence timestamp.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .config import INF


def bin_index(age: float, bins: Sequence[float]) -> int:
    """Right-open bucket index for an age in seconds."""

    if age != age:  # NaN
        return -1
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if age >= lo and age < hi:
            return i
    # age >= last finite edge falls into final bin
    if age >= bins[-2]:
        return len(bins) - 2
    return -1


def bin_label(bins: Sequence[float], index: int) -> str:
    lo, hi = bins[index], bins[index + 1]
    lo_s = "inf" if lo == INF else (str(int(lo)) if float(lo).is_integer() else str(lo))
    hi_s = "inf" if hi == INF else (str(int(hi)) if float(hi).is_integer() else str(hi))
    return f"{lo_s}_{hi_s}"


@dataclass
class TemporalRAMResult:
    temporal_ram: Dict[str, Optional[float]]
    full_attention_mass: Dict[str, float]
    num_full_tokens: Dict[str, int]
    num_kept_tokens: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "temporal_ram": self.temporal_ram,
            "temporal_full_attention_mass": self.full_attention_mass,
            "temporal_num_full_tokens": self.num_full_tokens,
            "temporal_num_kept_tokens": self.num_kept_tokens,
        }


def temporal_ram(
    attention_mass_by_uid: Dict[str, float],
    timestamp_by_uid: Dict[str, Optional[float]],
    retained_uids: set,
    query_time: float,
    bins: Sequence[float],
    eps: float = 1e-8,
) -> TemporalRAMResult:
    """Bucket-wise retained attention mass with bucket-conditioned normalisation.

    ``attention_mass_by_uid`` must come from the **full-KV** attention
    distribution, one scalar per original (full-KV) visual position.
    """

    n_bins = len(bins) - 1
    full_mass = [0.0] * n_bins
    kept_mass = [0.0] * n_bins
    n_full = [0] * n_bins
    n_kept = [0] * n_bins

    for uid, mass in attention_mass_by_uid.items():
        ts = timestamp_by_uid.get(uid)
        if ts is None:
            continue
        age = float(query_time) - float(ts)
        idx = bin_index(age, bins)
        if idx < 0:
            continue
        full_mass[idx] += float(mass)
        n_full[idx] += 1
        if uid in retained_uids:
            kept_mass[idx] += float(mass)
            n_kept[idx] += 1

    temporal = {}
    full_map = {}
    n_full_map = {}
    n_kept_map = {}
    for idx in range(n_bins):
        label = bin_label(bins, idx)
        denom = full_mass[idx]
        temporal[label] = (
            kept_mass[idx] / denom if full_mass[idx] > 0 else None
        )
        full_map[label] = full_mass[idx]
        n_full_map[label] = n_full[idx]
        n_kept_map[label] = n_kept[idx]

    return TemporalRAMResult(
        temporal_ram=temporal,
        full_attention_mass=full_map,
        num_full_tokens=n_full_map,
        num_kept_tokens=n_kept_map,
    )


# --------------------------------------------------------------------- #
# Evidence-timestamp adapter interface (reserved; not used by default)
# --------------------------------------------------------------------- #
EvidenceResolver = Callable[[dict], Optional[float]]


def get_evidence_timestamp(sample: dict) -> Optional[float]:
    """Return the answer-relevant evidence timestamp for a sample, if any.

    Current OVO-Bench / StreamingBench adapters do not expose a trustworthy
    evidence timestamp, so this returns ``None``. Benchmarks that do should
    override it via ``sample["metadata"]["evidence_time"]`` downstream.
    """

    metadata = (sample or {}).get("metadata") or {}
    value = metadata.get("evidence_time")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bucket_metric_by_evidence_age(
    values: Sequence[float],
    evidence_ages: Sequence[float],
    bins: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Mean of ``values`` per evidence-age bucket (only when callable)."""

    n_bins = len(bins) - 1
    acc = [[] for _ in range(n_bins)]
    for value, age in zip(values, evidence_ages):
        if value is None or age is None:
            continue
        idx = bin_index(float(age), bins)
        if idx >= 0:
            acc[idx].append(float(value))
    out = {}
    for idx in range(n_bins):
        label = bin_label(bins, idx)
        out[label] = (sum(acc[idx]) / len(acc[idx])) if acc[idx] else None
    return out
