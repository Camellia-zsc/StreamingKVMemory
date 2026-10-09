"""Main image-vs-video encoding run (5 OVO samples, ONE arm per process)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from .encoding_common import (
    BASE_CONFIG,
    STYLES,
    VIDEO_IDS,
    apply_encoding_env,
    arm_dir,
    fingerprint,
    write_subset,
)


def _config(annotation: Path, budget: int):
    from evaluation_pipeline.evaluate import load_config

    config = copy.deepcopy(load_config(BASE_CONFIG))
    config.setdefault("benchmarks", {})["ovobench"] = {
        **(config.get("benchmarks", {}).get("ovobench") or {}),
        "enabled": True,
        "annotation": str(annotation),
        "chunked_dir": "OVO-Bench/data/chunked_videos",
        "_video_ids": [str(v) for v in VIDEO_IDS],
    }
    config.setdefault("memory", {}).setdefault("compression", {})
    config["memory"]["method"] = "hermes"
    config["memory"]["compression"]["enabled"] = True
    config["memory"]["compression"]["budget"] = int(budget)
    return config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--style", required=True, choices=list(STYLES))
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--videos", default=None, help="comma ids (default: all 5)")
    args = p.parse_args(argv)

    sys.path.insert(0, os.path.abspath("OVO-Bench"))
    apply_encoding_env(args.style)

    from evaluation_pipeline.kv_metrics.config import KVMetricConfig
    from evaluation_pipeline.kv_metrics.paired_runner import run as run_paired

    videos = [v.strip() for v in args.videos.split(",")] if args.videos else list(VIDEO_IDS)
    subset = write_subset()
    out = arm_dir(args.style)
    raw = out / "raw"
    marker = raw / "main" / "run_status.json"
    if marker.exists() and not args.force:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("status") == "success" and int(state.get("n_expected", -1)) == len(videos):
            print(f"[encoding-run:{args.style}] already complete")
            return 0

    config = _config(subset, args.budget)
    config["benchmarks"]["ovobench"]["_video_ids"] = videos
    kv = KVMetricConfig(
        enable_kv_metrics=True, mode="compressed",
        output_dir=str(raw), run_name="main",
        attention_backend="original", profile_wallclock=False,
        detailed_trajectory=False, capture_option_logits=True,
        diagnostic_video_ids=list(videos),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({
        "style": args.style, "budget": args.budget, "videos": videos,
        "annotation": str(subset), "memory": config["memory"],
        "inference": config.get("inference"),
        "runtime": fingerprint(),
    }, indent=2), encoding="utf-8")

    print(f"[encoding-run:{args.style}] budget={args.budget} videos={videos}")
    run_paired(config, benchmark="ovobench", max_samples=0, kv_config=kv,
               video_ids=videos)
    print(f"[encoding-run:{args.style}] done -> {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
