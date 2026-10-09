"""Pre-compression sanity: does hook ON vs OFF already change the forward?

Runs ONE arm per process (the hook is registered at model load):

    python -m evaluation_pipeline.ablation.mrope_sanity --arm on
    python -m evaluation_pipeline.ablation.mrope_sanity --arm off
    python -m evaluation_pipeline.ablation.mrope_sanity --compare

No compression happens in this run (budget is set to a sentinel), so any
difference measured here is NOT attributable to pruning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from .mrope_common import (
    ABLATION_ROOT,
    PROBE_TEXT,
    apply_ablation_env,
    arm_dir,
    runtime_fingerprint,
)


NO_COMPRESSION_BUDGET = int(1e12)


def _load_backend(model_path: str):
    """Load the SDPA HERMES backend directly (no benchmark adapter)."""

    from types import SimpleNamespace

    from models.Qwen25VL_Stream import EvalQwen25VLStream

    args = SimpleNamespace(
        model_path=model_path,
        model="Qwen2.5-VL",
        result_dir="",
        chunked_dir="",
        save_results=False,
    )
    model = EvalQwen25VLStream(args)
    # Sentinel budget: pseudo_forward can never trigger a compaction.
    model.kv_size = NO_COMPRESSION_BUDGET
    model.qa_model.kv_size = NO_COMPRESSION_BUDGET
    return model


def _capture_full_vocab_logits(model, text: str):
    """One forward with the fixed probe; return the FULL last-position logits."""

    captured = {}
    lm_head = model.qa_model.lm_head

    def hook(module, args, output):
        if "logits" in captured:
            # Only the FIRST forward (prefill / step 0) is the sanity target;
            # later decode steps would overwrite it.
            return
        row = output[..., -1, :]
        with __import__("torch").no_grad():
            captured["logits"] = row[0].detach().float().cpu()

    handle = lm_head.register_forward_hook(hook)
    before = max(model._cache_lengths() or [0])
    try:
        model.qa_model.question_answering(
            {
                "question": text,
                "formatted_question": None,
                "prompt": model.qa_model.get_prompt(text, mc=False),
            },
            max_new_tokens=1,
            temperature=0.0,
            repetition_penalty=1.0,
        )
    finally:
        handle.remove()
    after = max(model._cache_lengths() or [0])
    return captured.get("logits"), before, after


def run_arm(args) -> int:
    apply_ablation_env(args.arm)
    arm_dir(args.arm).mkdir(parents=True, exist_ok=True)
    out_path = arm_dir(args.arm) / "precompression_sanity.json"

    model = _load_backend(args.model_path)
    from models.Qwen25VL_Stream import iter_frame_chunks

    model.reset()
    n_chunks = 0
    for chunk in iter_frame_chunks(
        video_path=args.video,
        sample_fps=args.sample_fps,
        frames_per_turn=args.frames_per_turn,
        max_side=args.max_side,
        max_frames=args.frames_per_turn * args.n_chunks,
    ):
        model.stream_video_chunk(chunk, seconds=args.frames_per_turn / args.sample_fps)
        n_chunks += 1
        if n_chunks >= args.n_chunks:
            break

    effective = model.effective_compaction_count()
    if effective != 0:
        raise RuntimeError(
            f"[sanity] expected 0 compactions, got {effective}; "
            "the sanity run must be pre-compression."
        )

    logits, cache_before, cache_after = _capture_full_vocab_logits(model, PROBE_TEXT)
    if logits is None:
        raise RuntimeError("[sanity] failed to capture full-vocab logits")

    tokenizer = model.qa_processor.tokenizer
    top = logits.topk(20)
    payload = {
        "arm": args.arm,
        "video": args.video,
        "n_chunks": n_chunks,
        "frames_per_turn": args.frames_per_turn,
        "sample_fps": args.sample_fps,
        "probe_text": PROBE_TEXT,
        "effective_compaction": effective,
        "cache_len_before_probe": cache_before,
        "cache_len_after_probe": cache_after,
        "cache_len_after_truncate": max(model._cache_lengths() or [0]),
        "vocab_size": int(logits.shape[0]),
        "logits": [float(v) for v in logits.tolist()],
        "top20_token_ids": [int(i) for i in top.indices.tolist()],
        "top20_texts": [
            tokenizer.decode([int(i)], skip_special_tokens=False)
            for i in top.indices.tolist()
        ],
        "top20_logits": [float(v) for v in top.values.tolist()],
        "runtime": runtime_fingerprint(
            {
                "video": args.video,
                "n_chunks": n_chunks,
                "frames_per_turn": args.frames_per_turn,
                "sample_fps": args.sample_fps,
                "max_side": args.max_side,
            }
        ),
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    if payload["cache_len_after_truncate"] != cache_before:
        raise RuntimeError(
            "[sanity] probe did not roll back the QA KV: "
            f"{cache_before} -> {payload['cache_len_after_truncate']}"
        )
    print(
        f"[sanity] arm={args.arm} chunks={n_chunks} cache={cache_before} "
        f"top1={payload['top20_texts'][0]!r} -> {out_path}"
    )
    return 0


def _js_divergence(p, q) -> float:
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(x, y):
        total = 0.0
        for a, b in zip(x, y):
            if a > 0 and b > 0:
                total += a * math.log(a / b)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_compare(args) -> int:
    import torch

    on = json.loads((arm_dir("on") / "precompression_sanity.json").read_text())
    off = json.loads((arm_dir("off") / "precompression_sanity.json").read_text())
    a = torch.tensor(on["logits"], dtype=torch.float64)
    b = torch.tensor(off["logits"], dtype=torch.float64)
    assert a.shape == b.shape, "vocab mismatch between arms"

    diff = (a - b).abs()
    pa = torch.softmax(a, dim=-1)
    pb = torch.softmax(b, dim=-1)
    m = (pa + pb) / 2.0
    js = 0.5 * float((pa * (pa / m).log()).sum()) + 0.5 * float(
        (pb * (pb / m).log()).sum()
    )
    kl_on_off = float((pa * (pa / pb).log()).sum())
    kl_off_on = float((pb * (pb / pa).log()).sum())
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    top1_a, top1_b = int(a.argmax()), int(b.argmax())
    top10_a = set(torch.topk(a, 10).indices.tolist())
    top10_b = set(torch.topk(b, 10).indices.tolist())
    result = {
        "vocab_size": int(a.shape[0]),
        "cache_len_before_probe_on": on["cache_len_before_probe"],
        "cache_len_before_probe_off": off["cache_len_before_probe"],
        "top1_on": on["top20_token_ids"][0],
        "top1_off": off["top20_token_ids"][0],
        "top1_text_on": on["top20_texts"][0],
        "top1_text_off": off["top20_texts"][0],
        "top1_agreement": top1_a == top1_b,
        "top10_overlap": len(top10_a & top10_b) / 10.0,
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "js_divergence": js,
        "kl_on_to_off": kl_on_off,
        "kl_off_to_on": kl_off_on,
        "cosine_logits": cosine,
    }
    out = ABLATION_ROOT / "A_precompression_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"-> {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["on", "off"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--model_path", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video", default="OVO-Bench/data/chunked_videos/300.mp4")
    parser.add_argument("--n_chunks", type=int, default=2)
    parser.add_argument("--frames_per_turn", type=int, default=2)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--max_side", type=int, default=336)
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.abspath("OVO-Bench"))
    if args.compare:
        return run_compare(args)
    if not args.arm:
        parser.error("--arm is required unless --compare")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
