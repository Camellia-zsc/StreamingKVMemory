"""No-compression representation sanity for image vs video encoding.

    python -m evaluation_pipeline.ablation.encoding_sanity --style image
    python -m evaluation_pipeline.ablation.encoding_sanity --style video
    python -m evaluation_pipeline.ablation.encoding_sanity --compare

Same 2 chunks / 4 frames, no compression (sentinel budget) => any logit
difference is attributable to the ENCODING, not to HERMES pruning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

from .encoding_common import (
    ENC_ROOT,
    PROBE_TEXT,
    STYLES,
    apply_encoding_env,
    arm_dir,
    fingerprint,
)


NO_COMPRESSION_BUDGET = int(1e12)


def _load(model_path: str):
    from types import SimpleNamespace

    from models.Qwen25VL_Stream import EvalQwen25VLStream

    model = EvalQwen25VLStream(
        SimpleNamespace(model_path=model_path, model="Qwen2.5-VL", result_dir="",
                        chunked_dir="", save_results=False)
    )
    model.kv_size = NO_COMPRESSION_BUDGET
    model.qa_model.kv_size = NO_COMPRESSION_BUDGET
    return model


def run_style(args) -> int:
    apply_encoding_env(args.style)
    out_dir = arm_dir(args.style)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load(args.model_path)
    from models.Qwen25VL_Stream import iter_frame_chunks

    model.reset()
    chunks = []
    n = 0
    for chunk in iter_frame_chunks(
        video_path=args.video,
        sample_fps=args.sample_fps,
        frames_per_turn=args.frames_per_turn,
        max_side=args.max_side,
        max_frames=args.frames_per_turn * args.n_chunks,
    ):
        before = model._cache_lengths()
        model.stream_video_chunk(chunk, seconds=args.frames_per_turn / args.sample_fps)
        after = model._cache_lengths()
        meta = getattr(model.qa_model, "last_encode_meta", None) or {}
        chunks.append({
            "chunk_id": n,
            "n_frames": len(chunk),
            "visual_tokens": (max(after) - max(before)) if before and after else None,
            "cache_len_before": max(before) if before else None,
            "cache_len_after": max(after) if after else None,
            **meta,
        })
        n += 1
        if n >= args.n_chunks:
            break

    if model.effective_compaction_count() != 0:
        raise RuntimeError("[sanity] expected 0 compactions")

    # full-vocab step-0 logits on the fixed probe (first lm_head call only)
    captured = {}

    def hook(module, hook_args, output):
        if "logits" in captured:
            return
        row = output[..., -1, :]
        import torch

        with torch.no_grad():
            captured["logits"] = row[0].detach().float().cpu()

    handle = model.qa_model.lm_head.register_forward_hook(hook)
    cache_before = max(model._cache_lengths() or [0])
    try:
        model.qa_model.question_answering(
            {"question": PROBE_TEXT, "formatted_question": None,
             "prompt": model.qa_model.get_prompt(PROBE_TEXT, mc=False)},
            max_new_tokens=1, temperature=0.0, repetition_penalty=1.0,
        )
    finally:
        handle.remove()
    cache_after_trunc = max(model._cache_lengths() or [0])
    if cache_after_trunc != cache_before:
        raise RuntimeError(
            f"[sanity] probe KV not rolled back: {cache_before} -> {cache_after_trunc}"
        )

    logits = captured["logits"]
    tok = model.qa_processor.tokenizer
    top = logits.topk(20)
    payload = {
        "style": args.style,
        "video": args.video,
        "n_chunks": n,
        "chunks": chunks,
        "probe_text": PROBE_TEXT,
        "cache_len_before_probe": cache_before,
        "cache_len_after_truncate": cache_after_trunc,
        "vocab_size": int(logits.shape[0]),
        "logits": [float(v) for v in logits.tolist()],
        "top20_token_ids": [int(i) for i in top.indices.tolist()],
        "top20_texts": [tok.decode([int(i)], skip_special_tokens=False)
                        for i in top.indices.tolist()],
        "runtime": fingerprint({"n_chunks": n, "frames_per_turn": args.frames_per_turn,
                                "sample_fps": args.sample_fps, "max_side": args.max_side}),
    }
    (out_dir / "precompression_sanity.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    print(f"[enc-sanity] style={args.style} chunks={n} cache={cache_before} "
          f"top1={payload['top20_texts'][0]!r} grid={chunks[0].get('grid_type')} "
          f"grid_shape={chunks[0].get('grid_shape')} "
          f"tokens/chunk={chunks[0].get('visual_tokens')}")
    return 0


def _js(p, q) -> float:
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x, y):
        return sum(a * math.log(a / b) for a, b in zip(x, y) if a > 0 and b > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_compare(args) -> int:
    import torch

    data = {s: json.loads((arm_dir(s) / "precompression_sanity.json").read_text())
            for s in STYLES}
    a = torch.tensor(data["image"]["logits"], dtype=torch.float64)
    b = torch.tensor(data["video"]["logits"], dtype=torch.float64)
    assert a.shape == b.shape
    pa, pb = torch.softmax(a, -1), torch.softmax(b, -1)
    m = (pa + pb) / 2
    js = 0.5 * float((pa * (pa / m).log()).sum()) + 0.5 * float((pb * (pb / m).log()).sum())
    img_tok = data["image"]["chunks"][0].get("visual_tokens")
    vid_tok = data["video"]["chunks"][0].get("visual_tokens")
    res = {
        "vocab_size": int(a.shape[0]),
        "chunks": data["image"]["n_chunks"],
        "grid_type_image": data["image"]["chunks"][0].get("grid_type"),
        "grid_type_video": data["video"]["chunks"][0].get("grid_type"),
        "grid_shape_image": data["image"]["chunks"][0].get("grid_shape"),
        "grid_shape_video": data["video"]["chunks"][0].get("grid_shape"),
        "pixels_shape_image": data["image"]["chunks"][0].get("pixels_shape"),
        "pixels_shape_video": data["video"]["chunks"][0].get("pixels_shape"),
        "visual_tokens_per_chunk_image": img_tok,
        "visual_tokens_per_chunk_video": vid_tok,
        "token_rate_ratio_video_over_image": (vid_tok / img_tok) if img_tok and vid_tok else None,
        "visual_tokens_total_image": max(
            (c.get("cache_len_after") or 0) - (c.get("cache_len_before") or 0)
            for c in data["image"]["chunks"]
        ) * data["image"]["n_chunks"] if data["image"]["chunks"] else None,
        "cache_len_image": data["image"]["cache_len_before_probe"],
        "cache_len_video": data["video"]["cache_len_before_probe"],
        "top1_image": data["image"]["top20_token_ids"][0],
        "top1_video": data["video"]["top20_token_ids"][0],
        "top1_text_image": data["image"]["top20_texts"][0],
        "top1_text_video": data["video"]["top20_texts"][0],
        "top1_agreement": data["image"]["top20_token_ids"][0] == data["video"]["top20_token_ids"][0],
        "top10_overlap": len(set(data["image"]["top20_token_ids"][:10])
                             & set(data["video"]["top20_token_ids"][:10])) / 10.0,
        "max_abs_logit_diff": float((a - b).abs().max()),
        "mean_abs_logit_diff": float((a - b).abs().mean()),
        "js_divergence": js,
        "kl_image_to_video": float((pa * (pa / pb).log()).sum()),
        "kl_video_to_image": float((pb * (pb / pa).log()).sum()),
        "cosine_logits": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }
    ENC_ROOT.mkdir(parents=True, exist_ok=True)
    (ENC_ROOT / "B_precompression_sanity.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    with (ENC_ROOT / "B_precompression_sanity.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res.keys()))
        w.writeheader()
        w.writerow(res)
    print(json.dumps(res, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--style", choices=list(STYLES))
    p.add_argument("--compare", action="store_true")
    p.add_argument("--model_path", default="models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--video", default="OVO-Bench/data/chunked_videos/300.mp4")
    p.add_argument("--n_chunks", type=int, default=2)
    p.add_argument("--frames_per_turn", type=int, default=2)
    p.add_argument("--sample_fps", type=float, default=0.5)
    p.add_argument("--max_side", type=int, default=336)
    args = p.parse_args(argv)
    sys.path.insert(0, os.path.abspath("OVO-Bench"))
    if args.compare:
        return run_compare(args)
    if not args.style:
        p.error("--style is required unless --compare")
    return run_style(args)


if __name__ == "__main__":
    raise SystemExit(main())
