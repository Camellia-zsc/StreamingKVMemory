"""Shared helpers for the M-RoPE ablation (sample list, env wiring, paths)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List


ABLATION_VIDEO_IDS: List[str] = ["300", "299", "298", "3", "483"]
EXPECTED_TASKS = {"300": "HLD", "299": "HLD", "298": "HLD", "3": "EPM", "483": "ASI"}

ABLATION_ROOT = Path("outputs/qwen25vl_ablation/mrope")
SUBSET_JSON = ABLATION_ROOT / "ovo_ablation5.json"
SOURCE_ANNOTATION = Path("OVO-Bench/data/ovo_bench_new.json")
BASE_CONFIG = "configs/eval_qwen25vl_ovo.yaml"

PROBE_TEXT = (
    "Describe the current scene in detail, focusing on specific objects, "
    "fine-grained actions, and spatial relationships."
)

ARMS = ("on", "off")


def arm_dir(arm: str) -> Path:
    return ABLATION_ROOT / f"hook_{arm}"


def write_ablation_subset() -> Path:
    """Build the ablation-only annotation subset (original order + fields)."""

    items = json.loads(SOURCE_ANNOTATION.read_text(encoding="utf-8"))
    wanted = set(ABLATION_VIDEO_IDS)
    subset = [item for item in items if str(item["id"]) in wanted]
    found = {str(item["id"]) for item in subset}
    missing = wanted - found
    if missing:
        raise FileNotFoundError(f"ablation videos missing from annotation: {sorted(missing)}")
    SUBSET_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_JSON.write_text(
        json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return SUBSET_JSON


def apply_ablation_env(arm: str) -> None:
    """Arm wiring: SDPA backend + hook switch + provenance dump."""

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    os.environ["HERMES_ABLATION_BACKEND"] = "sdpa"
    os.environ["HERMES_PER_LAYER_POS_HOOK"] = arm
    os.environ["HERMES_DUMP_KEEP_PROVENANCE"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def runtime_fingerprint(extra: dict | None = None) -> dict:
    import torch
    import transformers

    import flash_attn  # noqa: F401  (presence check only)

    info = {
        "backend": os.environ.get("HERMES_ABLATION_BACKEND"),
        "per_layer_position_hook": os.environ.get("HERMES_PER_LAYER_POS_HOOK"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if extra:
        info.update(extra)
    return info
