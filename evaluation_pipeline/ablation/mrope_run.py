"""Main M-RoPE ablation run: 5 OVO-Backward samples, ONE arm per process.

    python -m evaluation_pipeline.ablation.mrope_run --arm on  --budget 6000
    python -m evaluation_pipeline.ablation.mrope_run --arm off --budget 6000

Everything except ``HERMES_PER_LAYER_POS_HOOK`` is identical between arms.
Results are written under ``outputs/qwen25vl_ablation/mrope/hook_{on,off}/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from .mrope_common import (
    ABLATION_VIDEO_IDS,
    BASE_CONFIG,
    apply_ablation_env,
    arm_dir,
    runtime_fingerprint,
    write_ablation_subset,
)


def _config_for(annotation: Path, budget: int):
    from evaluation_pipeline.evaluate import load_config

    config = copy.deepcopy(load_config(BASE_CONFIG))
    config.setdefault("benchmarks", {})["ovobench"] = {
        **(config.get("benchmarks", {}).get("ovobench") or {}),
        "enabled": True,
        "annotation": str(annotation),
        "chunked_dir": "OVO-Bench/data/chunked_videos",
        "_video_ids": [str(v) for v in ABLATION_VIDEO_IDS],
    }
    config.setdefault("memory", {}).setdefault("compression", {})
    config["memory"]["method"] = "hermes"
    config["memory"]["compression"]["enabled"] = True
    config["memory"]["compression"]["budget"] = int(budget)
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["on", "off"])
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model_path", default="models/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.abspath("OVO-Bench"))
    apply_ablation_env(args.arm)

    from evaluation_pipeline.kv_metrics.config import KVMetricConfig
    from evaluation_pipeline.kv_metrics.paired_runner import run as run_paired

    subset = write_ablation_subset()
    out = arm_dir(args.arm)
    raw_dir = out / "raw"
    marker = raw_dir / "main" / "run_status.json"
    if marker.exists() and not args.force:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("status") == "success" and int(state.get("n_expected", -1)) == len(
            ABLATION_VIDEO_IDS
        ):
            print(f"[mrope-run:{args.arm}] already complete; nothing to do")
            return 0

    config = _config_for(subset, args.budget)
    kv_config = KVMetricConfig(
        enable_kv_metrics=True,
        mode="compressed",
        output_dir=str(raw_dir),
        run_name="main",
        attention_backend="original",
        profile_wallclock=False,
        detailed_trajectory=False,
        capture_option_logits=True,
        diagnostic_video_ids=list(ABLATION_VIDEO_IDS),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "budget": args.budget,
                "model_path": args.model_path,
                "video_ids": ABLATION_VIDEO_IDS,
                "annotation": str(subset),
                "memory": config["memory"],
                "inference": config.get("inference"),
                "mcq_decode_mode": config.get("mcq_decode_mode"),
                "runtime": runtime_fingerprint(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[mrope-run:{args.arm}] budget={args.budget} videos={ABLATION_VIDEO_IDS}")
    run_paired(
        config,
        benchmark="ovobench",
        max_samples=0,
        kv_config=kv_config,
        video_ids=[str(v) for v in ABLATION_VIDEO_IDS],
    )
    print(f"[mrope-run:{args.arm}] done -> {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
