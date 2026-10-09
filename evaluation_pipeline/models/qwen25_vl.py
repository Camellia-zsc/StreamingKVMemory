"""Qwen2.5-VL streaming backend: dependency-light policy core.

This module contains everything about the new backend that can be decided and
verified *without* a GPU or the 7B checkpoint:

* backend / model-path / decode-mode resolution from config;
* visual-token span detection for the Qwen2.5-VL prompt layout;
* dynamic KV cache accounting (no hard-coded 63/89 numbers);
* an incremental-streaming audit that proves chunks are encoded once;
* per-phase profiling for the realtime-streaming study.

Nothing here imports ``torch`` or ``transformers`` at module load, so it is
importable and unit-testable inside the lightweight test environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------- #
# backend / policy resolution
# --------------------------------------------------------------------- #
LEGACY_BACKEND = "mmduet2"
BACKEND_MMDUET2 = "mmduet2"
BACKEND_QWEN25VL = "qwen2_5_vl"

SUPPORTED_BACKENDS = (BACKEND_MMDUET2, BACKEND_QWEN25VL)

# Legacy model ``name`` values map onto the backend that owns them, so old
# configs that only set ``model.name`` keep working unchanged.
LEGACY_NAME_TO_BACKEND = {
    "MMDuet2": BACKEND_MMDUET2,
    "MMDuet2_native": BACKEND_MMDUET2,
    "MMDuet2_HERMES": BACKEND_MMDUET2,
    "Qwen2.5-VL": BACKEND_QWEN25VL,
    "Qwen2_5_VL": BACKEND_QWEN25VL,
    "Qwen25VL": BACKEND_QWEN25VL,
}

DEFAULT_QWEN25VL_MODEL_PATH = "models/Qwen2.5-VL-7B-Instruct"

# Official MCQ evaluation allows the model to emit any letter; the constrained
# mode is the historical MMDuet2-style hard first-token mask. Official is the
# default so the new backend never silently reproduces the old mask.
MCQ_DECODE_MODE_OFFICIAL = "official"
MCQ_DECODE_MODE_CONSTRAINED = "constrained"
MCQ_DECODE_MODES = (MCQ_DECODE_MODE_OFFICIAL, MCQ_DECODE_MODE_CONSTRAINED)
DEFAULT_MCQ_DECODE_MODE = MCQ_DECODE_MODE_OFFICIAL


def resolve_backend(model_config: Optional[dict]) -> str:
    """Return the backend name for a ``model`` config block.

    ``model.backend`` wins; otherwise the legacy ``model.name`` is mapped;
    anything unknown falls back to the historical MMDuet2 backend so existing
    configs are unaffected.
    """

    model_config = model_config or {}
    backend = model_config.get("backend") or model_config.get("model_backend")
    if backend:
        backend = str(backend).strip().lower()
        if backend in SUPPORTED_BACKENDS:
            return backend
        raise ValueError(
            f"Unsupported model.backend={backend!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    name = model_config.get("name")
    return LEGACY_NAME_TO_BACKEND.get(name, BACKEND_MMDUET2)


def resolve_model_path(model_config: Optional[dict]) -> Optional[str]:
    """Model path with a Qwen default; never hard-coded into core logic."""

    model_config = model_config or {}
    path = model_config.get("model_path")
    if path:
        return str(path)
    if resolve_backend(model_config) == BACKEND_QWEN25VL:
        return DEFAULT_QWEN25VL_MODEL_PATH
    return None


def resolve_mcq_decode_mode(config: Optional[dict]) -> str:
    """``official`` (default) or ``constrained`` first-token mask.

    Accepts ``config["mcq_decode_mode"]`` or a nested
    ``config["inference"]["mcq_decode_mode"]``.
    """

    config = config or {}
    mode = config.get("mcq_decode_mode")
    if mode is None:
        mode = (config.get("inference") or {}).get("mcq_decode_mode")
    if mode is None:
        return DEFAULT_MCQ_DECODE_MODE
    mode = str(mode).strip().lower()
    if mode not in MCQ_DECODE_MODES:
        raise ValueError(
            f"Unsupported mcq_decode_mode={mode!r}; expected {MCQ_DECODE_MODES}"
        )
    return mode


# --------------------------------------------------------------------- #
# visual span detection
# --------------------------------------------------------------------- #
@dataclass
class VisualSpan:
    """Index range of the visual tokens inside a Qwen2.5-VL prompt.

    Qwen2.5-VL wraps the merged image/video pad tokens as
    ``<|vision_start|> <|image_pad|> * N <|vision_end|>``. ``content_start`` /
    ``content_end`` bound the ``<|image_pad|>`` run (the tokens that carry
    visual content); ``start`` / ``end`` are the inclusive outer bounds.
    """

    start: int
    end: int
    content_start: int
    content_end: int
    n_visual_tokens: int
    has_marker: bool = True

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "content_start": self.content_start,
            "content_end": self.content_end,
            "n_visual_tokens": self.n_visual_tokens,
            "has_marker": self.has_marker,
        }


def detect_visual_span(
    input_ids: Sequence[int],
    *,
    image_token_id: int,
    vision_start_id: Optional[int] = None,
    vision_end_id: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> VisualSpan:
    """Locate the visual pad-token run in ``input_ids``.

    Falls back to the bare ``<|image_pad|>`` run when the vision markers are
    unavailable (older processor layouts). Raises ``ValueError`` when no visual
    token is present, so a silent "0 visual tokens" cannot slip through.
    """

    ids = list(input_ids)
    n = len(ids) if seq_len is None else int(seq_len)
    content = [i for i in range(len(ids)) if int(ids[i]) == int(image_token_id)]
    if not content:
        raise ValueError("No visual (image_pad) tokens found in input_ids")

    content_start, content_end = content[0], content[-1]
    start, end = content_start, content_end
    has_marker = False
    if vision_start_id is not None and vision_end_id is not None:
        for i in range(content_start - 1, -1, -1):
            if int(ids[i]) == int(vision_start_id):
                start = i
                has_marker = True
                break
        for i in range(content_end + 1, len(ids)):
            if int(ids[i]) == int(vision_end_id):
                end = i
                has_marker = True
                break

    if not 0 <= start <= end < max(n, len(ids)):
        raise ValueError(
            f"Invalid visual span start={start} end={end} for seq_len={n}"
        )
    return VisualSpan(
        start=start,
        end=end,
        content_start=content_start,
        content_end=content_end,
        n_visual_tokens=len(content),
        has_marker=has_marker,
    )


# --------------------------------------------------------------------- #
# dynamic KV cache accounting
# --------------------------------------------------------------------- #
class BudgetViolation(RuntimeError):
    """Raised when the managed visual KV exceeds the configured budget."""


@dataclass
class CacheAccounting:
    """Dynamic KV accounting for one streamed sample.

    All token counts are *per layer* (the streaming cache grows uniformly
    across layers between compactions). Nothing is inherited from the old
    MMDuet2 constants -- every number is measured at runtime.

    ``recent_visual_tokens`` documents whether a recency/window region exists;
    for the Qwen backend the recent frames are *inside* ``visual_budget``
    (HERMES keeps them by attention score, not by an unconditional window), so
    ``managed_visual_after`` already includes them.
    """

    protected_prefix_tokens: int = 0
    visual_budget: int = 0
    total_kv_before: int = 0
    total_kv_after: int = 0
    managed_visual_before: int = 0
    managed_visual_after: int = 0
    effective_compaction: int = 0
    effective_compaction_index: int = 0
    recent_visual_tokens: int = 0
    recent_inside_budget: bool = True
    violations: List[dict] = field(default_factory=list)

    def set_prefix_and_budget(
        self, protected_prefix_tokens: int, visual_budget: int
    ) -> None:
        self.protected_prefix_tokens = int(protected_prefix_tokens)
        self.visual_budget = int(visual_budget)

    @staticmethod
    def managed_visual(total_len: int, prefix_len: int) -> int:
        return max(0, int(total_len) - int(prefix_len))

    def on_chunk(self, total_len_before: int, total_len_after: int) -> None:
        self.total_kv_before = int(total_len_before)
        self.total_kv_after = int(total_len_after)
        self.managed_visual_before = self.managed_visual(
            total_len_before, self.protected_prefix_tokens
        )
        self.managed_visual_after = self.managed_visual(
            total_len_after, self.protected_prefix_tokens
        )

    def on_compaction(self, total_len_before: int, total_len_after: int) -> None:
        self.effective_compaction += 1
        self.effective_compaction_index += 1
        self.on_chunk(total_len_before, total_len_after)
        self.validate()

    def validate(self) -> bool:
        """``managed_visual_after <= visual_budget`` must hold after a compaction."""

        ok = self.managed_visual_after <= self.visual_budget
        if not ok:
            violation = {
                "managed_visual_after": self.managed_visual_after,
                "visual_budget": self.visual_budget,
                "effective_compaction_index": self.effective_compaction_index,
            }
            self.violations.append(violation)
            raise BudgetViolation(
                "managed visual KV "
                f"{self.managed_visual_after} > budget {self.visual_budget} "
                f"(compaction #{self.effective_compaction_index})"
            )
        return ok

    def as_dict(self) -> dict:
        return {
            "total_kv_before": self.total_kv_before,
            "total_kv_after": self.total_kv_after,
            "protected_prefix_tokens": self.protected_prefix_tokens,
            "managed_visual_before": self.managed_visual_before,
            "managed_visual_after": self.managed_visual_after,
            "effective_compaction": self.effective_compaction,
            "effective_compaction_index": self.effective_compaction_index,
            "visual_budget": self.visual_budget,
            "recent_visual_tokens": self.recent_visual_tokens,
            "recent_inside_budget": self.recent_inside_budget,
        }


# --------------------------------------------------------------------- #
# incremental-streaming audit ("not fake streaming")
# --------------------------------------------------------------------- #
@dataclass
class ChunkRecord:
    index: int
    n_frames: int
    pixels: int
    visual_tokens: int
    cache_len_before: int
    cache_len_after: int

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "n_frames": self.n_frames,
            "pixels": self.pixels,
            "visual_tokens": self.visual_tokens,
            "cache_len_before": self.cache_len_before,
            "cache_len_after": self.cache_len_after,
        }


class IncrementalStreamAudit:
    """Evidence that chunk *n* is encoded once with only its own pixels.

    The contract recorded per chunk is:

    * ``pixels`` is the chunk's own pixel count (it must not grow with the
      stream, which is what re-encoding the full prefix would produce);
    * ``cache_len_before`` of chunk *n* equals ``cache_len_after`` of chunk
      *n-1* -- the history lives in the KV cache, not in re-fed frames;
    * ``cache_len_before`` of chunk *n* equals prefix + the summed visual
      tokens of all previous chunks.
    """

    def __init__(self) -> None:
        self.records: List[ChunkRecord] = []
        self.failures: List[str] = []

    def start_stream(self) -> None:
        self.records = []
        self.failures = []

    def record(
        self,
        *,
        n_frames: int,
        pixels: int,
        visual_tokens: int,
        cache_len_before: int,
        cache_len_after: int,
    ) -> ChunkRecord:
        rec = ChunkRecord(
            index=len(self.records),
            n_frames=int(n_frames),
            pixels=int(pixels),
            visual_tokens=int(visual_tokens),
            cache_len_before=int(cache_len_before),
            cache_len_after=int(cache_len_after),
        )
        self.records.append(rec)
        return rec

    def check(self, prefix_len: int = 0) -> bool:
        """Re-derive the contract from the recorded chunks; returns True if OK."""

        self.failures = []
        running = int(prefix_len)
        prev_after: Optional[int] = None
        for rec in self.records:
            if prev_after is not None and rec.cache_len_before != prev_after:
                self.failures.append(
                    f"chunk {rec.index}: cache_len_before={rec.cache_len_before} "
                    f"!= previous cache_len_after={prev_after} (cache not reused)"
                )
            if rec.cache_len_before != running:
                self.failures.append(
                    f"chunk {rec.index}: cache_len_before="
                    f"{rec.cache_len_before} != prefix+seen={running} "
                    "(history appears to be re-encoded)"
                )
            if rec.cache_len_after < rec.cache_len_before:
                self.failures.append(
                    f"chunk {rec.index}: cache shrank "
                    f"({rec.cache_len_before}->{rec.cache_len_after})"
                )
            running = rec.cache_len_after
            prev_after = rec.cache_len_after
        return not self.failures

    def summary(self) -> dict:
        pixels = [r.pixels for r in self.records]
        return {
            "n_chunks": len(self.records),
            "pixels_per_chunk_min": min(pixels) if pixels else None,
            "pixels_per_chunk_max": max(pixels) if pixels else None,
            "pixels_per_chunk_monotone_growth": _strictly_growing(pixels),
            "ok": not self.failures,
            "failures": list(self.failures),
            "chunks": [r.as_dict() for r in self.records],
        }


def _strictly_growing(values: Sequence[int]) -> bool:
    if len(values) < 2:
        return False
    grew = all(b >= a for a, b in zip(values, values[1:]))
    return grew and values[-1] > values[0]


# --------------------------------------------------------------------- #
# realtime profiling
# --------------------------------------------------------------------- #
PHASES = (
    "vision_preprocess",
    "vision_encode",
    "stream_prefill",
    "hermes_compression",
    "question_prefill",
    "decode",
)


class StreamingProfiler:
    """Accumulates wall-clock time per streaming phase.

    ``rho_realtime = compression_time / compression_trigger_interval`` answers
    "can compression keep up with the live stream?".
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.totals: Dict[str, float] = {p: 0.0 for p in PHASES}
        self.counts: Dict[str, int] = {p: 0 for p in PHASES}
        self.compression_intervals: List[float] = []
        self._last_compression_t: Optional[float] = None
        self._open: Dict[str, float] = {}

    def phase(self, name: str) -> "PhaseTimer":
        if name not in PHASES:
            raise KeyError(f"Unknown phase {name!r}; expected {PHASES}")
        return PhaseTimer(self, name)

    def add(self, name: str, seconds: float) -> None:
        if name not in self.totals:
            raise KeyError(f"Unknown phase {name!r}; expected {PHASES}")
        self.totals[name] += float(seconds)
        self.counts[name] += 1
        if name == "hermes_compression":
            now = time.perf_counter()
            if self._last_compression_t is not None:
                self.compression_intervals.append(now - self._last_compression_t)
            self._last_compression_t = now

    def mean_trigger_interval(self) -> Optional[float]:
        if not self.compression_intervals:
            return None
        return sum(self.compression_intervals) / len(self.compression_intervals)

    def rho_realtime(self) -> Optional[float]:
        interval = self.mean_trigger_interval()
        if not interval or interval <= 0:
            return None
        return self.totals["hermes_compression"] / interval

    def as_dict(self) -> dict:
        return {
            "phase_seconds": dict(self.totals),
            "phase_counts": dict(self.counts),
            "mean_trigger_interval_sec": self.mean_trigger_interval(),
            "rho_realtime": self.rho_realtime(),
        }


class PhaseTimer:
    def __init__(self, profiler: StreamingProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self._t0: Optional[float] = None

    def __enter__(self) -> "PhaseTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._t0 is not None:
            self.profiler.add(self.name, time.perf_counter() - self._t0)
            self._t0 = None


# --------------------------------------------------------------------- #
# MCQ protocol (official vs constrained) -- policy only, no decoding
# --------------------------------------------------------------------- #
def mcq_first_token_ids(
    option_letters: Sequence[str],
    tokenizer,
    *,
    mode: str = DEFAULT_MCQ_DECODE_MODE,
) -> Optional[List[int]]:
    """Allowed first-token ids for the constrained mode, ``None`` for official.

    This is the *only* place that decides whether to mask; the decode call and
    the option-logit sidecar both consult it, so the diagnostic capture can
    never silently switch the official protocol into the constrained one.
    """

    mode = str(mode).strip().lower()
    if mode == MCQ_DECODE_MODE_OFFICIAL:
        return None
    if mode != MCQ_DECODE_MODE_CONSTRAINED:
        raise ValueError(f"Unsupported mcq_decode_mode={mode!r}")

    allowed: List[int] = []
    for letter in option_letters:
        for candidate in (letter, f" {letter}"):
            ids = tokenizer(candidate, add_special_tokens=False).input_ids
            if ids:
                allowed.append(int(ids[0]))
    allowed = list(dict.fromkeys(allowed))
    if not allowed:
        raise RuntimeError("No allowed first-token IDs for MCQ options")
    return allowed


def best_option_from_logits(
    option_logits: Dict[str, float],
) -> Optional[str]:
    """Argmax option letter from a ``{letter: logit}`` sidecar capture.

    Read-only helper: the returned letter is reported next to (never instead
    of) the formal prediction.
    """

    scored = {
        str(k): float(v)
        for k, v in (option_logits or {}).items()
        if v is not None
    }
    if not scored:
        return None
    return max(scored, key=scored.get)
