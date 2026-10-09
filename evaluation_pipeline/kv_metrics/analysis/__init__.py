"""Multi-sample KV compression fidelity experiment + analysis layer.

Reads the sidecar outputs produced by ``evaluation_pipeline.kv_metrics`` and
never re-implements RAM/AORE/JS/KL. Importing this package must not import
torch, HERMES or OVO-Bench.
"""

__all__ = [
    "statistics",
    "loader",
    "sample_selector",
    "plotting",
    "report",
    "experiment_runner",
]
