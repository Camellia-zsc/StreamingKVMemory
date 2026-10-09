"""Shared helpers for the image-vs-video encoding ablation (experiment B)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List


ENC_ROOT = Path("outputs/qwen25vl_ablation/encoding")
SUBSET_JSON = ENC_ROOT / "ovo_ablation5.json"
SOURCE_ANNOTATION = Path("OVO-Bench/data/ovo_bench_new.json")
BASE_CONFIG = "configs/eval_qwen25vl_ovo.yaml"
VIDEO_IDS: List[str] = ["300", "299", "298", "3", "483"]
TASK_BY_VIDEO = {"300": "HLD", "299": "HLD", "298": "HLD", "3": "EPM", "483": "ASI"}

STYLES = ("image", "video")
PROBE_TEXT = (
    "Describe the current scene in detail, focusing on specific objects, "
    "fine-grained actions, and spatial relationships."
)
TIME_BIN_SEC = 4.0  # == frames_per_turn / sample_fps == one streaming chunk

TH_REP_JS = 1e-4
TH_REP_TOP10 = 0.9
TH_REP_TOKEN_RATE = 0.10
TH_RET_JACCARD = 0.90
TH_RET_COVERAGE = 0.05
TH_RET_OLDEST_SEC = 10.0
TH_DIV_JACCARD = 0.80
TH_DIV_COVERAGE = 0.10
TH_DIV_OLDEST_SEC = 30.0
TH_DIV_K_RATIO = 1.5


def arm_dir(style: str) -> Path:
    return ENC_ROOT / f"enc_{style}"


def write_subset() -> Path:
    items = json.loads(SOURCE_ANNOTATION.read_text(encoding="utf-8"))
    wanted = set(VIDEO_IDS)
    subset = [i for i in items if str(i["id"]) in wanted]
    missing = wanted - {str(i["id"]) for i in subset}
    if missing:
        raise FileNotFoundError(f"missing ablation videos: {sorted(missing)}")
    SUBSET_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_JSON.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    return SUBSET_JSON


def apply_encoding_env(style: str) -> None:
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")
    os.environ["HERMES_ABLATION_BACKEND"] = "fa2"
    os.environ["HERMES_ENCODE_STYLE"] = style
    os.environ["HERMES_DUMP_KEEP_PROVENANCE"] = "1"
    os.environ["HERMES_TIME_BIN_SEC"] = str(TIME_BIN_SEC)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def fingerprint(extra: dict | None = None) -> dict:
    import torch
    import transformers

    info = {
        "backend": os.environ.get("HERMES_ABLATION_BACKEND"),
        "encode_style": os.environ.get("HERMES_ENCODE_STYLE"),
        "time_bin_sec": os.environ.get("HERMES_TIME_BIN_SEC"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if extra:
        info.update(extra)
    return info
