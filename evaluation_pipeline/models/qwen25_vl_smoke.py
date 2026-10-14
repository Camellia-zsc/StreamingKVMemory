"""GPU smoke tests for the Qwen2.5-VL streaming backend.

These are **GPU** commands: they load the 7B checkpoint. They are intentionally
separate from the CPU unit tests so ``pytest`` stays model-free.

Run inside the ``hermes-qwen`` (or ``mmduet2-infer``) environment:

    python -m evaluation_pipeline.models.qwen25_vl_smoke qwen-load-smoke
    python -m evaluation_pipeline.models.qwen25_vl_smoke qwen-stream-smoke
    python -m evaluation_pipeline.models.qwen25_vl_smoke qwen-hermes-smoke --budget 200

Every command prints an explicit OK/FAIL banner so the outcome is unambiguous.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODEL_PATH = "models/Qwen2.5-VL-7B-Instruct"
DEFAULT_VIDEO = "OVO-Bench/data/chunked_videos/0.mp4"


def _bootstrap(ovo_root: str) -> None:
    """Put the OVO-Bench ``models`` package on sys.path like the runtime does."""

    root = os.path.abspath(ovo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_backend(model_path: str, budget: int, attn: str):
    os.environ["MMDUET2_ATTN_IMPLEMENTATION"] = attn
    os.environ["HERMES_KV_SIZE"] = str(budget)
    os.environ["QWEN25VL_CAPTURE_OPTION_LOGITS"] = "1"
    from models.Qwen25VL_Stream import EvalQwen25VLStream
    from types import SimpleNamespace

    args = SimpleNamespace(
        model_path=model_path,
        model="Qwen2.5-VL",
        result_dir="",
        chunked_dir="",
        save_results=False,
    )
    return EvalQwen25VLStream(args)


def _ok(name: str, extra: str = "") -> int:
    print(f"\n===== {name}: OK ===== {extra}")
    return 0


def _fail(name: str, error: Exception) -> int:
    print(f"\n===== {name}: FAIL ===== {type(error).__name__}: {error}")
    return 1


def cmd_qwen_load_smoke(args) -> int:
    name = "qwen-load-smoke"
    try:
        model = _load_backend(args.model_path, args.budget, args.attn)
        cfg = model.qa_model.config
        model.reset()
        lengths = model._cache_lengths()
        print(
            "[load] model_type=", getattr(cfg, "model_type", None),
            "layers=", getattr(cfg, "num_hidden_layers", None),
            "kv_heads=", getattr(cfg, "num_key_value_heads", None),
            "hidden=", getattr(cfg, "hidden_size", None),
        )
        print("[load] protected_prefix_tokens=", model.qa_model.visual_start_idx)
        print("[load] cache_lens_after_init=", lengths)
        if not lengths:
            raise RuntimeError("cache empty after reset()")
        # Pure-text forward must work with no visual tokens.
        reply = model._ask("Say OK.", max_new_tokens=4, mc=False)
        print("[load] text_reply=", repr(reply))
        return _ok(name, f"prefix={model.qa_model.visual_start_idx}")
    except Exception as error:  # noqa: BLE001
        return _fail(name, error)


def cmd_qwen_stream_smoke(args) -> int:
    name = "qwen-stream-smoke"
    try:
        model = _load_backend(args.model_path, 10**12, args.attn)
        from models.Qwen25VL_Stream import iter_frame_chunks

        model.reset()
        prefix = int(model.qa_model.visual_start_idx)
        n = 0
        seconds = args.frames_per_turn / float(args.sample_fps)
        for chunk in iter_frame_chunks(
            video_path=args.video,
            sample_fps=args.sample_fps,
            frames_per_turn=args.frames_per_turn,
            max_side=args.max_side,
            max_frames=args.frames_per_turn * args.chunks,
        ):
            before = max(model._cache_lengths() or [0])
            model.stream_video_chunk(chunk, seconds=seconds)
            after = max(model._cache_lengths() or [0])
            print(
                f"[stream] chunk={n} frames={len(chunk)} "
                f"visual_tokens={after - before} "
                f"cache_len_before={before} cache_len_after={after} "
                f"cache_delta={after - before}"
            )
            n += 1
            if n >= args.chunks:
                break
        ok = model.stream_audit.check(prefix_len=prefix)
        summary = model.stream_audit.summary()
        print("[stream] pixels_per_chunk_min=", summary["pixels_per_chunk_min"])
        print("[stream] pixels_per_chunk_max=", summary["pixels_per_chunk_max"])
        print("[stream] pixels_growth=", summary["pixels_per_chunk_monotone_growth"])
        print("[stream] rate=", model.visual_token_rate())
        print(f"[stream] incremental_audit = {'PASS' if ok else 'FAIL'}")
        if not ok:
            for failure in summary["failures"]:
                print("   ", failure)
            raise RuntimeError("incremental audit failed: not genuinely streaming")
        if model.stream_audit.records and model.stream_audit.records[-1].cache_len_after <= prefix:
            raise RuntimeError("cache did not grow")
        return _ok(name, f"chunks={n} final_len={model._cache_lengths()}")
    except Exception as error:  # noqa: BLE001
        return _fail(name, error)


def cmd_qwen_hermes_smoke(args) -> int:
    name = "qwen-hermes-smoke"
    try:
        model = _load_backend(args.model_path, args.budget, args.attn)
        from models.Qwen25VL_Stream import iter_frame_chunks

        model.reset()
        prefix = int(model.qa_model.visual_start_idx)
        print("[hermes] protected_prefix_tokens =", prefix)
        print("[hermes] visual_budget =", model.kv_size)
        n = 0
        triggered = False
        seconds = args.frames_per_turn / float(args.sample_fps)
        for chunk in iter_frame_chunks(
            video_path=args.video,
            sample_fps=args.sample_fps,
            frames_per_turn=args.frames_per_turn,
            max_side=args.max_side,
            max_frames=args.frames_per_turn * args.chunks,
        ):
            before = max(model._cache_lengths() or [0])
            model.stream_video_chunk(chunk, seconds=seconds)
            after = max(model._cache_lengths() or [0])
            n += 1
            managed = model.accounting.managed_visual_after
            print(
                f"[hermes] chunk={n} cache {before}->{after} "
                f"managed_visual={managed} budget={model.kv_size}"
            )
            if model.accounting.effective_compaction:
                acc = model.accounting.as_dict()
                total_after = acc["total_kv_after"]
                print(
                    "[hermes] compaction: "
                    f"total_kv_before={acc['total_kv_before']} "
                    f"total_kv_after={total_after} "
                    f"protected_prefix_tokens={acc['protected_prefix_tokens']} "
                    f"managed_visual_before={acc['managed_visual_before']} "
                    f"managed_visual_after={acc['managed_visual_after']} "
                    f"visual_budget={acc['visual_budget']} "
                    f"effective_compaction_index={acc['effective_compaction_index']} "
                    f"recent_inside_budget={acc['recent_inside_budget']}"
                )
                print(
                    "[hermes] check: "
                    f"total_after({total_after}) == "
                    f"prefix({acc['protected_prefix_tokens']}) + "
                    f"managed_after({acc['managed_visual_after']}) -> "
                    f"{total_after == acc['protected_prefix_tokens'] + acc['managed_visual_after']}"
                )
                print(
                    "[hermes] check: managed_visual_after <= budget -> "
                    f"{acc['managed_visual_after'] <= acc['visual_budget']}"
                )
                triggered = True
                break
        if not triggered:
            print(
                "[hermes] WARNING: no compaction triggered; stream more chunks "
                "or lower the budget."
            )
        acc = model.accounting.as_dict()
        print("[hermes] accounting=", acc)
        print("[hermes] rate=", model.visual_token_rate())
        rate = model.visual_token_rate()
        if rate["tokens_per_second"]:
            for budget_value in (200, 100, 50):
                print(
                    f"[hermes] explain: budget={budget_value} ~= "
                    f"{budget_value / rate['tokens_per_second']:.1f}s of recent visual history"
                )
        if triggered and acc["managed_visual_after"] > acc["visual_budget"]:
            raise RuntimeError("managed visual KV above budget after compaction")
        reply = model.ask_question(
            "Question: What is happening in the video?\nOptions:\n"
            "A. something\nB. nothing\nAnswer with the option letter only.",
            options=["something", "nothing"],
            max_new_tokens=8,
        )
        print("[hermes] reply=", repr(reply))
        print("[hermes] step0_option_logits=", model.option_logits_by_letter())
        print("[hermes] profile=", model.profiler.as_dict())
        return _ok(name, f"compactions={acc['effective_compaction']}")
    except Exception as error:  # noqa: BLE001
        return _fail(name, error)


def _js_divergence(p, q) -> float:
    import math

    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(x, y):
        total = 0.0
        for a, b in zip(x, y):
            if a > 0 and b > 0:
                total += a * math.log(a / b)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _softmax(values):
    import math

    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def cmd_qwen_vanilla_equiv(args) -> int:
    """One-shot full-prefix Qwen2.5-VL vs the incremental Full streaming wrapper.

    Only the wrapper is supposed to add accounting / sidecar machinery; the
    model behaviour must be unchanged. We therefore compare the greedy
    prediction and the step-0 option distribution. If they diverge we report
    the measured divergence rather than hiding it.
    """

    name = "qwen-vanilla-equiv"
    try:
        import torch

        model = _load_backend(args.model_path, 10**12, args.attn)
        model.compression_enabled = False
        model.kv_size = int(1e12)
        model.qa_model.kv_size = int(1e12)

        from models.Qwen25VL_Stream import iter_frame_chunks

        frames = []
        for chunk in iter_frame_chunks(
            video_path=args.video,
            sample_fps=args.sample_fps,
            frames_per_turn=args.frames_per_turn,
            max_side=args.max_side,
            max_frames=args.frames_per_turn * args.chunks,
        ):
            frames.extend(chunk)
        print(f"[equiv] frames={len(frames)}")

        question = (
            "Question: What is shown in the video?\n"
            "Options:\nA. something happens\nB. nothing happens\n"
            "Please select the correct answer."
        )
        options = ["something happens", "nothing happens"]

        # ---- A: vanilla, single full-prefix forward ------------------ #
        from qwen_vl_utils import process_vision_info

        processor = model.qa_processor
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "max_pixels": int(args.max_side) ** 2,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(next(model.qa_model.parameters()).device)

        option_ids = {
            letter: processor.tokenizer(letter, add_special_tokens=False).input_ids[0]
            for letter in ("A", "B")
        }
        with torch.inference_mode():
            out = model.qa_model(**inputs, use_cache=False)
            vanilla_row = out.logits[0, -1, :].float()
            vanilla_logits = {
                letter: float(vanilla_row[tid]) for letter, tid in option_ids.items()
            }
            vanilla_gen = model.qa_model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        vanilla_pred = processor.tokenizer.decode(
            vanilla_gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        print(f"[equiv] vanilla logits={vanilla_logits}")
        print(f"[equiv] vanilla prediction={vanilla_pred!r}")
        del inputs, out, vanilla_gen
        torch.cuda.empty_cache()

        # ---- B: Full streaming wrapper (compression disabled) -------- #
        model.reset()
        seconds = args.frames_per_turn / float(args.sample_fps)
        for start in range(0, len(frames), args.frames_per_turn):
            model.stream_video_chunk(
                frames[start:start + args.frames_per_turn], seconds=seconds
            )
        wrapper_pred_raw = model.ask_question(
            question, options=options, max_new_tokens=8
        )
        wrapper_logits = model.option_logits_by_letter() or {}
        print(f"[equiv] wrapper logits={wrapper_logits}")
        print(f"[equiv] wrapper prediction={wrapper_pred_raw!r}")
        print(f"[equiv] wrapper compaction={model.effective_compaction_count()}")

        # ---- compare -------------------------------------------------- #
        letters = [l for l in ("A", "B") if l in vanilla_logits and l in wrapper_logits]
        if not letters:
            raise RuntimeError("no comparable option logits captured")
        p = _softmax([vanilla_logits[l] for l in letters])
        q = _softmax([wrapper_logits[l] for l in letters])
        js = _js_divergence(p, q)
        max_abs = max(
            abs(vanilla_logits[l] - wrapper_logits[l]) for l in letters
        )
        same = vanilla_pred.strip()[:1].upper() == wrapper_pred_raw.strip()[:1].upper()
        print(f"[equiv] option_js_divergence={js:.6f}")
        print(f"[equiv] option_logit_max_abs_diff={max_abs:.4f}")
        print(f"[equiv] prediction_agreement={same}")
        if js > 0.05 or not same:
            print(
                "[equiv] WARNING: wrapper deviates from one-shot vanilla "
                "beyond the tolerance (likely incremental vs one-shot temporal "
                "RoPE framing); report honestly."
            )
            return _fail(
                name,
                RuntimeError(
                    f"equivalence outside tolerance (js={js:.4f}, agree={same})"
                ),
            )
        return _ok(name, f"js={js:.6f} agree={same}")
    except Exception as error:  # noqa: BLE001
        return _fail(name, error)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ovo_root", default="OVO-Bench")
    parser.add_argument(
        "--attn", default="auto", choices=["auto", "sdpa", "flash_attention_2"]
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--frames_per_turn", type=int, default=2)
    parser.add_argument("--max_side", type=int, default=336)
    parser.add_argument("--chunks", type=int, default=8)


def _infokv_runtime(args):
    """Build the real ModelRuntime with memory.method=infokv_stream."""

    from evaluation_pipeline.runtime import ModelRuntime

    budget = int(args.budget)
    recent = int(args.recent_budget) if args.recent_budget is not None else budget // 3
    recent = max(0, min(recent, budget))
    long_term = budget - recent
    scoring = {
        "temporal_novelty": {"enabled": True, "weight": 0.35, "reference": "recent_mean", "reference_chunks": 4},
        "layer_evolution": {"enabled": True, "weight": 0.35,
                            "layer_a": args.layer_a, "layer_b": args.layer_b},
        "v2v": {"enabled": True, "weight": 0.30, "observation_tokens": args.observation_tokens,
                "layer": args.v2v_layer},
    }
    config = {
        "name": "Qwen2.5-VL",
        "backend": "qwen2_5_vl",
        "model_path": args.model_path,
        "ovo_root": args.ovo_root,
        "inference": {
            "sample_fps": args.sample_fps,
            "frames_per_turn": args.frames_per_turn,
            "max_side": args.max_side,
            "attn_implementation": args.attn,
        },
        "memory": {
            "method": "infokv_stream",
            "total_visual_budget": budget,
            "recent": {"enabled": True, "budget": recent},
            "long_term": {"budget": long_term},
            "compression": {"interval_chunks": args.compression_interval},
            "scoring": scoring,
        },
    }
    return ModelRuntime(config).load()


def _stream_once(m, chunk, seconds):
    """Drive the real wired path (subclass -> integrator -> observe/compact)."""

    before = max(m._cache_lengths() or [0])
    m.stream_video_chunk(chunk, seconds=seconds)
    return before, max(m._cache_lengths() or [0])


def cmd_qwen_infokv_smoke(args) -> int:
    """Minimal GPU smoke for the real InfoKVStream KV integration."""

    name = "qwen-infokv-smoke"
    try:
        import numpy as np

        from models.Qwen25VL_Stream import iter_frame_chunks

        runtime = _infokv_runtime(args)
        m = runtime.model
        integ = m._integrator
        print(
            "[infokv] layers: D=(%d,%d) V2V=%d prefix=%d budget=%d recent=%d "
            "long=%d interval=%d"
            % (integ.layer_a, integ.layer_b, integ.v2v_layer, integ.prefix_len,
               m.kv_size if isinstance(m.kv_size, int) and m.kv_size < 1e9 else integ.memory.total_budget,
               integ.memory.recent_budget, integ.memory.long_term_budget,
               integ.memory.interval)
        )
        seconds = args.frames_per_turn / float(args.sample_fps)
        chunks = list(iter_frame_chunks(
            video_path=args.video, sample_fps=args.sample_fps,
            frames_per_turn=args.frames_per_turn, max_side=args.max_side,
            max_frames=args.frames_per_turn * args.chunks,
        ))
        print(f"[infokv] chunks_available={len(chunks)}")

        # ---- Phase T: capture transparency (no compaction) ----------- #
        saved_interval = integ.memory.interval
        integ.memory.interval = 10 ** 9
        question = "Describe what happens in this video in one short sentence."
        m.reset()
        for chunk in chunks:
            _stream_once(m, chunk, seconds)
        answer_capture_on = m.ask_question(question, options=None, max_new_tokens=16, mc=False)
        m._integrator = None  # plain parent path, no capture
        m.reset()
        for chunk in chunks:
            _stream_once(m, chunk, seconds)
        answer_capture_off = m.ask_question(question, options=None, max_new_tokens=16, mc=False)
        m._integrator = integ
        integ.memory.interval = saved_interval
        transparent = answer_capture_on.strip() == answer_capture_off.strip()
        print(f"[transparency] capture_on ={answer_capture_on!r}")
        print(f"[transparency] capture_off={answer_capture_off!r}")
        print(f"[transparency] identical={transparent}")
        if not transparent:
            raise RuntimeError("capture changed the uncompressed answer")

        # ---- Phase S: real compaction -------------------------------- #
        m.reset()
        effective_events = []
        for i, chunk in enumerate(chunks):
            n_events_before = len(integ.events)
            before, after = _stream_once(m, chunk, seconds)
            lens = m._cache_lengths()
            managed = len(integ.memory.managed_tokens())
            print(
                f"[stream] chunk={i} cache {before}->{after} "
                f"visual={after - before} managed={managed} "
                f"uniform={len(set(lens)) == 1} "
                f"max_emitted_len={lens[0] if lens else None}"
            )
            new_events = integ.events[n_events_before:]
            for event in new_events:
                print("[compact]", event.as_dict())
            for layer_idx, (k, v) in enumerate(m.qa_model.kv_cache):
                if k.shape[2] != v.shape[2]:
                    raise RuntimeError(f"layer {layer_idx}: len(K) != len(V)")
            if any(e.effective for e in new_events):
                lens_check = m._cache_lengths()
                if len(set(lens_check)) != 1:
                    raise RuntimeError("InfoKV v1 must keep equal cache length per layer")
                if managed > integ.memory.total_budget:
                    raise RuntimeError(
                        f"managed {managed} > budget "
                        f"{integ.memory.total_budget} after effective compaction"
                    )
                effective_events = [e for e in integ.events if e.effective]

        events = integ.events
        triggers = len(events)
        effective = [e for e in events if e.effective]
        print(f"[compact] trigger_count={triggers} effective_count={len(effective)}")
        if not effective:
            raise RuntimeError("no effective compaction triggered; lower --budget")

        # invariants after compaction
        stats = integ.memory.get_stats()
        print(
            "[infokv] peak_managed_tokens=%d peak_overshoot=%d (%.3f)"
            % (stats.peak_managed_tokens, stats.peak_budget_overshoot_tokens,
               stats.peak_budget_overshoot_ratio or 0.0)
        )
        lens = m._cache_lengths()
        print(f"[infokv] final_cache_lens min={min(lens)} max={max(lens)} uniform={len(set(lens)) == 1}")

        # retained order ascending (temporal) + global broadcast check
        keep_ids = integ.memory.retained_visual_token_ids()
        keep_positions = [integ.ledger.get(t).cache_physical_index for t in keep_ids]
        if keep_positions != sorted(keep_positions):
            raise RuntimeError("retained cache positions are not ascending")
        print(f"[infokv] retained_visual={len(keep_ids)} ascending=True")

        # N / D / A ranges + finiteness
        rows = integ.memory.managed_tokens()
        n_vals = np.array([r.novelty_raw for r in rows])
        d_vals = np.array([r.evolution_raw for r in rows])
        a_vals = np.array([r.v2v_raw for r in rows])
        finite = all(np.isfinite(x).all() for x in (n_vals, d_vals, a_vals))
        if not finite:
            raise RuntimeError("non-finite InfoKV signal")
        print(
            "[infokv] N[%.4f,%.4f] D[%.4f,%.4f] A_v2v[%.5f,%.5f] finite=%s"
            % (n_vals.min(), n_vals.max(), d_vals.min(), d_vals.max(),
               a_vals.min(), a_vals.max(), finite)
        )

        # sidecar sample: the rows that actually got a V2V score
        rows_by_a = sorted(
            (integ.last_sidecar or []), key=lambda r: -(r.get("v2v_raw") or 0.0)
        )
        for row in rows_by_a[:3]:
            print(
                "[sidecar] token=%s chunk=%s cache_idx=%s N=%.4f D=%.4f A=%.5f "
                "final=%.4f selected=%s mrope=%s"
                % (row["token_id"], row["chunk_id"], row.get("cache_index_before"),
                   row["novelty_raw"], row["evolution_raw"], row["v2v_raw"],
                   row["final_score"] or 0.0, row["selected"], row.get("mrope_position"))
            )

        # decode after compaction, then continue appending KV
        reply = m.ask_question(
            "What is happening in the video? Answer in one short sentence.",
            options=None, max_new_tokens=16, mc=False,
        )
        print(f"[infokv] reply_after_compaction={reply!r}")
        if not (reply or "").strip():
            print("[infokv] WARNING: post-compaction reply was empty")
        seen_before = integ.memory.total_seen_tokens
        before, after = _stream_once(m, chunks[-1], seconds)
        print(f"[infokv] post-compaction append: {before}->{after}")
        if integ.memory.total_seen_tokens <= seen_before:
            raise RuntimeError("no new visual tokens observed after compaction")
        if max(m._cache_lengths()) != integ.prefix_len + len(integ.ledger):
            raise RuntimeError("cache and ledger disagree after appending")
        print("[infokv] sidecar rows=%d events=%d" % (len(integ.last_sidecar), len(events)))
        return _ok(name, f"triggers={triggers} effective={len(effective)}")
    except Exception as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return _fail(name, error)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("qwen-load-smoke")
    _common(p1)
    p1.add_argument("--budget", type=int, default=200)
    p1.set_defaults(func=cmd_qwen_load_smoke)

    p2 = sub.add_parser("qwen-stream-smoke")
    _common(p2)
    p2.set_defaults(func=cmd_qwen_stream_smoke)

    p3 = sub.add_parser("qwen-hermes-smoke")
    _common(p3)
    p3.add_argument("--budget", type=int, default=200)
    p3.set_defaults(func=cmd_qwen_hermes_smoke)

    p4 = sub.add_parser("qwen-vanilla-equiv")
    _common(p4)
    p4.set_defaults(func=cmd_qwen_vanilla_equiv)

    p5 = sub.add_parser("qwen-infokv-smoke")
    _common(p5)
    p5.add_argument("--budget", type=int, default=300)
    p5.add_argument("--recent_budget", type=int, default=None)
    p5.add_argument("--compression_interval", type=int, default=2)
    p5.add_argument("--observation_tokens", type=int, default=32)
    p5.add_argument("--layer_a", type=int, default=None)
    p5.add_argument("--layer_b", type=int, default=None)
    p5.add_argument("--v2v_layer", type=int, default=None)
    p5.set_defaults(func=cmd_qwen_infokv_smoke)

    args = parser.parse_args(argv)
    _bootstrap(args.ovo_root)
    if not Path(args.model_path).exists():
        print(f"[error] model path does not exist: {args.model_path}")
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
