"""Low-level sidecar recording helpers.

Sidecars never touch benchmark answer files: everything goes under
``<kv_metrics_dir>/<run_name>/``.
"""

import json
from pathlib import Path
from typing import Iterable, Optional

import torch


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def save_layer_tensor(path: Path, name: str, tensor: Optional[torch.Tensor]) -> None:
    """Persist a large per-layer tensor to .npz instead of summary.json."""

    if tensor is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    existing = {}
    if path.exists():
        try:
            with np.load(path) as data:
                existing = {k: data[k] for k in data.files}
        except (OSError, ValueError):
            existing = {}
    existing[name] = tensor.detach().cpu().numpy()
    np.savez_compressed(path, **existing)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj
