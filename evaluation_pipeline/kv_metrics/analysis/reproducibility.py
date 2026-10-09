"""Reproducibility manifest (no secrets)."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _git_hash(path: str) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def collect(extra: Optional[dict] = None) -> dict:
    import torch

    gpu = None
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        gpu = None

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": _pkg_version("transformers"),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_model": gpu,
        "git": {
            "workspace": _git_hash("."),
            "ovobench": _git_hash("OVO-Bench"),
            "hermes": _git_hash("HERMES"),
            "local_code_repo": _git_hash("local-code-repo"),
        },
        "cwd": os.getcwd(),
        "command_line": " ".join(sys.argv),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write(path: Path, extra: Optional[dict] = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect(extra)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
