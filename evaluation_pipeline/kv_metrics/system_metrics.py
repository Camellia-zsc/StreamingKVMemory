"""System-level metrics: real persistent KV bytes, compaction wall-clock,
compression ratio and real-time ratio.

Only *tensors that live in GPU memory* count towards KV bytes. Python strings
and debug metadata never count.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


def tensor_bytes(t: torch.Tensor) -> int:
    """Bytes actually occupied by a tensor (numel * element_size)."""

    if not isinstance(t, torch.Tensor):
        return 0
    return int(t.numel()) * int(t.element_size())


def is_effective_compaction(
    kv_before: Optional[int],
    kv_after: Optional[int],
    prefix: int = 0,
) -> bool:
    """True only when the compaction actually evicted cached tokens.

    HERMES triggers whenever ``total > budget``, but the budget only bounds the
    *managed visual* region while a protected prompt prefix is not counted, so
    many triggers are no-ops (e.g. ``2021 -> 2021``). Such triggers must never
    be counted as a compression step.

    Uses the managed-region comparison ``after - prefix < before - prefix`` and
    falls back to the total-token comparison (equivalent while the protected
    prefix length is constant within a run).
    """

    if kv_before is None or kv_after is None:
        return False
    managed_before = int(kv_before) - int(prefix)
    managed_after = int(kv_after) - int(prefix)
    if managed_before >= 0 and managed_after >= 0:
        return managed_after < managed_before
    return int(kv_after) < int(kv_before)


def kv_cache_bytes(cache) -> int:
    """Sum the bytes of a (key, value) cache, per layer.

    ``cache`` may be a list/tuple of ``(k, v)`` tuples, or a
    ``DynamicCache``-like object supporting iteration over layers.
    """

    if cache is None:
        return 0
    total = 0
    try:
        iterator = cache
        for entry in iterator:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                total += tensor_bytes(entry[0]) + tensor_bytes(entry[1])
    except TypeError:
        return total
    return total


def cache_token_count(cache) -> Optional[int]:
    if cache is None:
        return None
    lengths = []
    try:
        for entry in cache:
            if isinstance(entry, (tuple, list)) and len(entry) >= 1:
                lengths.append(int(entry[0].shape[2]))
    except (TypeError, IndexError):
        return None
    return max(lengths) if lengths else None


@dataclass
class CompactionEvent:
    event_id: str
    compaction_index: int
    stream_time: Optional[float]
    kv_before: Optional[int]
    kv_after: Optional[int]
    bytes_before: int
    bytes_after: int
    wall_clock_sec: Optional[float]
    trigger_interval_sec: Optional[float] = None
    rho_realtime: Optional[float] = None
    realtime_feasible: Optional[bool] = None

    def compression_ratio(self) -> Optional[float]:
        if self.bytes_after and self.bytes_after > 0:
            return self.bytes_before / self.bytes_after
        return None

    def retained_size(self) -> Optional[float]:
        if self.bytes_before and self.bytes_before > 0:
            return self.bytes_after / self.bytes_before
        return None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "compaction_index": self.compaction_index,
            "stream_time": self.stream_time,
            "kv_before": self.kv_before,
            "kv_after": self.kv_after,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "compression_ratio": self.compression_ratio(),
            "retained_size": self.retained_size(),
            "wall_clock_sec": self.wall_clock_sec,
            "trigger_interval_sec": self.trigger_interval_sec,
            "rho_realtime": self.rho_realtime,
            "realtime_feasible": self.realtime_feasible,
        }


class KVSystemProfiler:
    """Times HERMES compaction and tracks persistent KV bytes.

    Timing is only synchronized when ``profile_wallclock`` is True so that the
    default inference path never pays for an extra ``cuda.synchronize``.
    """

    def __init__(self, profile_wallclock: bool = True, eps: float = 1e-8):
        self.profile_wallclock = profile_wallclock
        self.eps = eps
        self.events: List[CompactionEvent] = []
        self._last_stream_time: Optional[float] = None
        self._last_full_bytes: Optional[int] = None

    def reset(self) -> None:
        self.events = []
        self._last_stream_time = None
        self._last_full_bytes = None

    @torch.inference_mode()
    def time_compaction(self, fn, *args, **kwargs) -> float:
        """Run ``fn`` and return the (optionally synchronized) elapsed seconds."""

        cuda = torch.cuda.is_available()
        if self.profile_wallclock and cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if self.profile_wallclock and cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed, result

    def record_compaction(
        self,
        event_id: str,
        compaction_index: int,
        stream_time: Optional[float],
        bytes_before: int,
        bytes_after: int,
        kv_before: Optional[int],
        kv_after: Optional[int],
        wall_clock_sec: Optional[float],
    ) -> CompactionEvent:
        interval = None
        if stream_time is not None and self._last_stream_time is not None:
            interval = float(stream_time) - float(self._last_stream_time)
        if stream_time is not None:
            self._last_stream_time = float(stream_time)

        rho = None
        feasible = None
        if wall_clock_sec is not None and interval and interval > 0:
            rho = wall_clock_sec / interval
            feasible = bool(rho < 1.0)

        event = CompactionEvent(
            event_id=event_id,
            compaction_index=compaction_index,
            stream_time=stream_time,
            kv_before=kv_before,
            kv_after=kv_after,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            wall_clock_sec=wall_clock_sec,
            trigger_interval_sec=interval,
            rho_realtime=rho,
            realtime_feasible=feasible,
        )
        self.events.append(event)
        return event

    def set_full_bytes(self, full_bytes: int) -> None:
        self._last_full_bytes = full_bytes

    def summary(self) -> Dict[str, Any]:
        if not self.events:
            return {
                "compaction_events": 0,
                "mean_compression_ratio": None,
                "min_compression_ratio": None,
                "last_compression_ratio": None,
                "mean_wallclock_sec": None,
                "max_wallclock_sec": None,
                "mean_rho_realtime": None,
                "realtime_feasible_fraction": None,
            }
        ratios = [
            e.compression_ratio()
            for e in self.events
            if e.compression_ratio() is not None
        ]
        walls = [
            e.wall_clock_sec for e in self.events if e.wall_clock_sec is not None
        ]
        rhos = [
            e.rho_realtime for e in self.events if e.rho_realtime is not None
        ]
        feas = [
            e.realtime_feasible
            for e in self.events
            if e.realtime_feasible is not None
        ]
        return {
            "compaction_events": len(self.events),
            "mean_compression_ratio": _mean(ratios),
            "min_compression_ratio": min(ratios) if ratios else None,
            "last_compression_ratio": ratios[-1] if ratios else None,
            "mean_wallclock_sec": _mean(walls),
            "max_wallclock_sec": max(walls) if walls else None,
            "mean_rho_realtime": _mean(rhos),
            "realtime_feasible_fraction": (
                sum(1 for f in feas if f) / len(feas) if feas else None
            ),
        }


def _mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)
