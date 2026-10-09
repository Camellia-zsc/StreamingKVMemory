"""Configuration for the KV Cache Compression Fidelity Evaluation module.

This module is intentionally standalone: importing it must not import the
HERMES backend, OVO-Bench, or any heavy dependency. It only defines the
dataclass that drives the metrics and the CLI flags.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


INF = float("inf")

DEFAULT_TEMPORAL_BINS: List[float] = [0.0, 10.0, 30.0, 60.0, 300.0, INF]


@dataclass
class KVMetricConfig:
    """Runtime configuration for the KV fidelity evaluation.

    Defaults are chosen so that *nothing* runs unless ``enable_kv_metrics``
    is explicitly set, matching the "existing inference must be unaffected"
    requirement.
    """

    # Master switch. When False the whole package is a no-op.
    enable_kv_metrics: bool = False

    # compressed | full | paired
    mode: str = "paired"

    # Where the sidecar files go. Never touches the benchmark answer files.
    output_dir: str = "outputs/kv_metrics"
    run_name: str = "default"

    # eager  -> recompute attention probabilities in fp32 for fidelity
    # original-> keep the deployment attention path untouched (no RAM/AORE)
    attention_backend: str = "eager"

    # last_user_token | all_query_tokens
    anchor: str = "last_user_token"

    # Temporal bin edges in seconds, right-open except the last bin.
    temporal_bins: List[float] = field(
        default_factory=lambda: list(DEFAULT_TEMPORAL_BINS)
    )

    # Emit per-head RAM/AORE (large). Summary always keeps per-layer stats.
    save_per_head: bool = False

    # Optional teacher-forced continuation span for KL/JS. 0 = disabled.
    teacher_forced_span: int = 0

    # Add torch.cuda.synchronize() around compaction for honest wall-clock.
    profile_wallclock: bool = True

    # Component-level runtime breakdown (vision encoder / prefill / HERMES /
    # decode). Off by default: it adds cuda.synchronize around many calls.
    profile_runtime_breakdown: bool = False

    # Sample ids to restrict the run to (used by the trajectory / case study).
    sample_ids_filter: List[str] = field(default_factory=list)

    # Detailed fidelity trajectory: probe the LONGEST query of each sample
    # after every *effective* compaction (plus a k=0 pre-first-compaction
    # baseline). Other queries only get a task-only record.
    detailed_trajectory: bool = False

    # Read-only capture of the official step-0 pre-mask logits at the allowed
    # MCQ option tokens (for ΔP_gold / ΔM_z). Never modifies decoding.
    capture_option_logits: bool = False

    # Restrict the *rich* per-sample diagnostics (cache snapshots, anchor
    # capture, option-logit sidecar) to this subset of sample/video ids.
    # Everything else in the same run stays ``lightweight``: it still produces
    # the normal prediction / correctness record, but pays none of the extra
    # fidelity instrumentation. Empty = no restriction (original behaviour).
    diagnostic_video_ids: List[str] = field(default_factory=list)

    # hermes_managed (visual region only) | all_cache
    scope: str = "hermes_managed"

    # identity_only | absorbed
    #   identity_only: a KE (summary) token never counts as "retained original".
    #   absorbed     : original tokens folded into a summary count as retained.
    summary_token_mode: str = "identity_only"

    # Float tolerance used for sanity checks (TEST 1 etc.).
    eps: float = 1e-8

    # Only used when writing per-event jsonl; keeps sidecar dumps bounded.
    max_paired_events: Optional[int] = None

    # Benchmark/sample identification is filled by the runner.
    benchmark: Optional[str] = None
    sample_ids: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.mode not in {"compressed", "full", "paired"}:
            raise ValueError(f"unknown kv_metrics mode: {self.mode!r}")
        if self.attention_backend not in {"eager", "original"}:
            raise ValueError(
                f"unknown attention backend: {self.attention_backend!r}"
            )
        if self.anchor not in {"last_user_token", "all_query_tokens"}:
            raise ValueError(f"unknown anchor: {self.anchor!r}")
        if self.scope not in {"hermes_managed", "all_cache"}:
            raise ValueError(f"unknown scope: {self.scope!r}")
        if self.summary_token_mode not in {"identity_only", "absorbed"}:
            raise ValueError(
                f"unknown summary_token_mode: {self.summary_token_mode!r}"
            )
        if len(self.temporal_bins) < 2:
            raise ValueError("temporal_bins needs at least two edges")
        if list(self.temporal_bins) != sorted(self.temporal_bins):
            raise ValueError("temporal_bins must be ascending")

    @property
    def bin_labels(self) -> List[str]:
        labels = []
        for lo, hi in zip(self.temporal_bins[:-1], self.temporal_bins[1:]):
            lo_s = _fmt_edge(lo)
            hi_s = _fmt_edge(hi)
            labels.append(f"{lo_s}_{hi_s}")
        return labels

    def as_dict(self) -> dict:
        data = asdict(self)
        # json-safe: replace inf with the string "inf"
        data["temporal_bins"] = [_json_edge(edge) for edge in self.temporal_bins]
        return data


def _fmt_edge(value: float) -> str:
    if value == INF:
        return "inf"
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _json_edge(value: float):
    return "inf" if value == INF else value


def add_cli_arguments(parser, default_output_dir: str = "outputs/kv_metrics"):
    """Register the KV-metric flags on an argparse parser."""

    group = parser.add_argument_group("kv_metrics")
    group.add_argument(
        "--enable_kv_metrics",
        action="store_true",
        default=False,
        help="Enable KV cache compression fidelity evaluation.",
    )
    group.add_argument(
        "--kv_metrics_mode",
        choices=["compressed", "full", "paired"],
        default="paired",
    )
    group.add_argument("--kv_metrics_dir", default=default_output_dir)
    group.add_argument("--kv_metrics_run_name", default="default")
    group.add_argument(
        "--kv_metrics_attention_backend",
        choices=["eager", "original"],
        default="eager",
    )
    group.add_argument(
        "--kv_metrics_anchor",
        choices=["last_user_token", "all_query_tokens"],
        default="last_user_token",
    )
    group.add_argument(
        "--kv_metrics_temporal_bins",
        default="0,10,30,60,300,inf",
        help="Comma separated bin edges in seconds; use 'inf' for the last.",
    )
    group.add_argument(
        "--kv_metrics_save_per_head",
        action="store_true",
        default=False,
    )
    group.add_argument(
        "--kv_metrics_teacher_forced_span",
        type=int,
        default=0,
    )
    group.add_argument(
        "--kv_metrics_profile_wallclock",
        action="store_true",
        default=False,
    )
    group.add_argument(
        "--profile_runtime_breakdown",
        action="store_true",
        default=False,
        help="Enable component-level runtime breakdown profiling.",
    )
    group.add_argument(
        "--kv_metrics_detailed_trajectory",
        action="store_true",
        default=False,
        help="Probe ONLY the longest query of each sample after every effective "
             "compaction (detailed fidelity trajectory).",
    )
    group.add_argument(
        "--kv_metrics_capture_option_logits",
        action="store_true",
        default=False,
        help="Read-only capture of the official step-0 MCQ option logits.",
    )
    group.add_argument(
        "--kv_metrics_scope",
        choices=["hermes_managed", "all_cache"],
        default="hermes_managed",
    )
    group.add_argument(
        "--kv_metrics_summary_token_mode",
        choices=["identity_only", "absorbed"],
        default="identity_only",
    )
    return group


def parse_temporal_bins(text: str) -> List[float]:
    edges = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"inf", "+inf", "infinity"}:
            edges.append(INF)
        else:
            edges.append(float(part))
    return edges


def from_cli_args(args) -> KVMetricConfig:
    """Build a :class:`KVMetricConfig` from parsed CLI arguments."""

    return KVMetricConfig(
        enable_kv_metrics=bool(getattr(args, "enable_kv_metrics", False)),
        mode=getattr(args, "kv_metrics_mode", "paired"),
        output_dir=getattr(args, "kv_metrics_dir", "outputs/kv_metrics"),
        run_name=getattr(args, "kv_metrics_run_name", "default"),
        attention_backend=getattr(
            args, "kv_metrics_attention_backend", "eager"
        ),
        anchor=getattr(args, "kv_metrics_anchor", "last_user_token"),
        temporal_bins=parse_temporal_bins(
            getattr(args, "kv_metrics_temporal_bins", "0,10,30,60,300,inf")
        ),
        save_per_head=bool(getattr(args, "kv_metrics_save_per_head", False)),
        teacher_forced_span=int(
            getattr(args, "kv_metrics_teacher_forced_span", 0) or 0
        ),
        profile_wallclock=bool(
            getattr(args, "kv_metrics_profile_wallclock", False)
        ),
        profile_runtime_breakdown=bool(
            getattr(args, "profile_runtime_breakdown", False)
        ),
        detailed_trajectory=bool(
            getattr(args, "kv_metrics_detailed_trajectory", False)
        ),
        capture_option_logits=bool(
            getattr(args, "kv_metrics_capture_option_logits", False)
        ),
        scope=getattr(args, "kv_metrics_scope", "hermes_managed"),
        summary_token_mode=getattr(
            args, "kv_metrics_summary_token_mode", "identity_only"
        ),
    )


def from_config_dict(config: dict) -> KVMetricConfig:
    """Build a config from a YAML ``kv_metrics:`` block."""

    block = dict((config or {}).get("kv_metrics", {}) or {})
    bins = block.get("temporal_bins")
    if isinstance(bins, str):
        bins = parse_temporal_bins(bins)
    elif isinstance(bins, list):
        bins = [INF if str(b).lower() in {"inf", "+inf"} else float(b) for b in bins]
    kwargs = dict(
        enable_kv_metrics=block.get("enable", block.get("enable_kv_metrics", False)),
        mode=block.get("mode", "paired"),
        output_dir=block.get("dir", block.get("output_dir", "outputs/kv_metrics")),
        run_name=block.get("run_name", "default"),
        attention_backend=block.get("attention_backend", "eager"),
        anchor=block.get("anchor", "last_user_token"),
        save_per_head=bool(block.get("save_per_head", False)),
        teacher_forced_span=int(block.get("teacher_forced_span", 0) or 0),
        profile_wallclock=bool(block.get("profile_wallclock", False)),
        scope=block.get("scope", "hermes_managed"),
        summary_token_mode=block.get("summary_token_mode", "identity_only"),
    )
    if bins is not None:
        kwargs["temporal_bins"] = bins
    return KVMetricConfig(**kwargs)
