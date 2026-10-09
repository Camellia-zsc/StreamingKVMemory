"""Model backends for the streaming video memory pipeline.

``qwen25_vl`` holds the dependency-light policy core for the Qwen2.5-VL
streaming backend. It deliberately avoids importing ``torch`` /
``transformers`` at module import time so the policy logic is unit-testable on
CPU without the 7B checkpoint.
"""

from .qwen25_vl import (  # noqa: F401
    BACKEND_MMDUET2,
    BACKEND_QWEN25VL,
    CacheAccounting,
    BudgetViolation,
    IncrementalStreamAudit,
    StreamingProfiler,
    VisualSpan,
    detect_visual_span,
    resolve_backend,
    resolve_mcq_decode_mode,
    resolve_model_path,
)

__all__ = [
    "BACKEND_MMDUET2",
    "BACKEND_QWEN25VL",
    "CacheAccounting",
    "BudgetViolation",
    "IncrementalStreamAudit",
    "StreamingProfiler",
    "VisualSpan",
    "detect_visual_span",
    "resolve_backend",
    "resolve_mcq_decode_mode",
    "resolve_model_path",
]
