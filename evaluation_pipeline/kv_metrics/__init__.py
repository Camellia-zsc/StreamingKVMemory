"""KV Cache Compression Fidelity Evaluation for MMDuet2 + HERMES.

Independent sidecar module. Importing it never imports HERMES, OVO-Bench or
any heavy backend, and it never mutates benchmark answer files.
"""

from .config import KVMetricConfig
from .provenance import KVProvenanceTracker, KVTokenUID
from .fidelity import (
    aore,
    capture_anchor_attention,
    js_divergence,
    kl_divergence,
    ram_per_head,
)
from .system_metrics import KVSystemProfiler, kv_cache_bytes, tensor_bytes
from .temporal import temporal_ram
from .hooks import KVInstrumentation, AnchorArtifact
from .aggregate import MetricAggregator, compute_paired_metrics

__all__ = [
    "KVMetricConfig",
    "KVProvenanceTracker",
    "KVTokenUID",
    "KVInstrumentation",
    "AnchorArtifact",
    "MetricAggregator",
    "compute_paired_metrics",
    "KVSystemProfiler",
    "temporal_ram",
    "aore",
    "js_divergence",
    "kl_divergence",
    "ram_per_head",
    "capture_anchor_attention",
    "kv_cache_bytes",
    "tensor_bytes",
]
