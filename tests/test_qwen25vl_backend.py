"""CPU unit tests for the Qwen2.5-VL streaming backend (no GPU / no weights).

Everything here runs without loading the 7B checkpoint: config/policy
resolution, visual-span detection, cache accounting, the incremental-streaming
audit, profiling, the MCQ sidecar policy, and the runtime dispatch. The
processor-output test only loads the local tokenizer/image-processor config
with ``local_files_only=True`` (no model weights).
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation_pipeline.evaluate import apply_defaults, make_run_id
from evaluation_pipeline.models.qwen25_vl import (
    BACKEND_MMDUET2,
    BACKEND_QWEN25VL,
    BudgetViolation,
    CacheAccounting,
    IncrementalStreamAudit,
    StreamingProfiler,
    best_option_from_logits,
    detect_visual_span,
    mcq_first_token_ids,
    resolve_backend,
    resolve_mcq_decode_mode,
    resolve_model_path,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"


# --------------------------------------------------------------------- #
# 1. backend config selection
# --------------------------------------------------------------------- #
def test_backend_defaults_to_mmduet2_for_legacy_configs():
    assert resolve_backend({}) == BACKEND_MMDUET2
    assert resolve_backend({"name": "MMDuet2"}) == BACKEND_MMDUET2


def test_backend_explicit_qwen_selection():
    assert resolve_backend({"backend": "qwen2_5_vl"}) == BACKEND_QWEN25VL
    assert resolve_backend({"name": "Qwen2.5-VL"}) == BACKEND_QWEN25VL


def test_backend_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_backend({"backend": "nope"})


def test_model_path_default_only_for_qwen():
    assert resolve_model_path({"backend": "qwen2_5_vl"}).endswith(
        "Qwen2.5-VL-7B-Instruct"
    )
    assert resolve_model_path({"name": "MMDuet2"}) is None
    assert resolve_model_path({"model_path": "x/y"}) == "x/y"


def test_mcq_decode_mode_default_is_official():
    assert resolve_mcq_decode_mode({}) == "official"
    assert resolve_mcq_decode_mode({"mcq_decode_mode": "constrained"}) == "constrained"
    assert (
        resolve_mcq_decode_mode({"inference": {"mcq_decode_mode": "constrained"}})
        == "constrained"
    )
    with pytest.raises(ValueError):
        resolve_mcq_decode_mode({"mcq_decode_mode": "auto"})


# --------------------------------------------------------------------- #
# 2. Qwen processor output parsing + visual span
# --------------------------------------------------------------------- #
@pytest.mark.skipif(not QWEN_PATH.exists(), reason="local Qwen2.5-VL checkpoint missing")
def test_processor_output_parsing_and_visual_span():
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(str(QWEN_PATH), local_files_only=True)
    image = Image.fromarray(np.zeros((64, 96, 3), dtype=np.uint8))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Q"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    out = processor(
        text=[text], images=images, videos=videos, return_tensors="pt", padding=True
    )

    for field in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        assert field in out, field
    assert out["input_ids"].shape[0] == 1
    assert out["pixel_values"].ndim == 2

    image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    v_start = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    v_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    span = detect_visual_span(
        out["input_ids"][0].tolist(),
        image_token_id=image_token,
        vision_start_id=v_start,
        vision_end_id=v_end,
    )
    assert span.has_marker
    assert span.n_visual_tokens > 0
    assert span.start < span.content_start <= span.content_end < span.end


# --------------------------------------------------------------------- #
# 3. visual span detection
# --------------------------------------------------------------------- #
def test_detect_visual_span_with_and_without_markers():
    ids = [1, 2, 151652, 151655, 151655, 151655, 151653, 7]
    span = detect_visual_span(
        ids, image_token_id=151655, vision_start_id=151652, vision_end_id=151653
    )
    assert (span.start, span.end) == (2, 6)
    assert span.n_visual_tokens == 3

    bare = detect_visual_span([5, 151655, 151655, 9], image_token_id=151655)
    assert bare.n_visual_tokens == 2
    assert not bare.has_marker


def test_detect_visual_span_raises_without_visual_tokens():
    with pytest.raises(ValueError):
        detect_visual_span([1, 2, 3], image_token_id=151655)


# --------------------------------------------------------------------- #
# 4. cache accounting
# --------------------------------------------------------------------- #
def test_cache_accounting_tracks_prefix_and_managed_visual():
    acc = CacheAccounting()
    acc.set_prefix_and_budget(63, 200)
    acc.on_chunk(63, 347)
    assert acc.managed_visual_after == 284
    acc.on_compaction(347, 263)
    assert acc.effective_compaction == 1
    assert acc.managed_visual_after == 200
    d = acc.as_dict()
    assert d["protected_prefix_tokens"] == 63
    assert d["visual_budget"] == 200
    assert d["total_kv_after"] == 263


def test_cache_accounting_raises_when_above_budget():
    acc = CacheAccounting()
    acc.set_prefix_and_budget(63, 100)
    with pytest.raises(BudgetViolation):
        acc.on_compaction(400, 263)
    assert acc.violations


# --------------------------------------------------------------------- #
# 5. incremental-streaming audit ("not fake streaming")
# --------------------------------------------------------------------- #
def _feed(audit, prefix, chunks):
    for n_frames, pixels, vt in chunks:
        before = prefix if not audit.records else audit.records[-1].cache_len_after
        audit.record(
            n_frames=n_frames,
            pixels=pixels,
            visual_tokens=vt,
            cache_len_before=before,
            cache_len_after=before + vt,
        )


def test_incremental_audit_accepts_true_streaming():
    audit = IncrementalStreamAudit()
    audit.start_stream()
    _feed(audit, 63, [(2, 1000, 100), (2, 1000, 100), (2, 1000, 100)])
    assert audit.check(prefix_len=63)
    summary = audit.summary()
    assert summary["ok"]
    assert summary["pixels_per_chunk_monotone_growth"] is False
    assert summary["n_chunks"] == 3


def test_incremental_audit_detects_reencoded_prefix():
    audit = IncrementalStreamAudit()
    audit.start_stream()
    audit.record(
        n_frames=2, pixels=1000, visual_tokens=100,
        cache_len_before=63, cache_len_after=163,
    )
    audit.record(
        n_frames=4, pixels=2500, visual_tokens=200,
        cache_len_before=63, cache_len_after=263,
    )
    assert not audit.check(prefix_len=63)
    assert audit.summary()["failures"]


def test_incremental_audit_detects_growing_pixels():
    audit = IncrementalStreamAudit()
    audit.start_stream()
    _feed(audit, 0, [(1, 500, 50), (1, 1000, 50), (1, 1500, 50)])
    assert audit.summary()["pixels_per_chunk_monotone_growth"] is True


# --------------------------------------------------------------------- #
# 6. profiling
# --------------------------------------------------------------------- #
def test_profiler_records_phases_and_rho():
    profiler = StreamingProfiler()
    with profiler.phase("stream_prefill"):
        pass
    profiler.add("hermes_compression", 0.10)
    profiler.add("hermes_compression", 0.10)
    assert profiler.totals["hermes_compression"] == pytest.approx(0.20)
    assert profiler.counts["stream_prefill"] == 1
    interval = profiler.mean_trigger_interval()
    assert interval is not None and interval >= 0
    assert profiler.rho_realtime() is not None
    assert "rho_realtime" in profiler.as_dict()
    profiler.reset()
    assert profiler.totals["hermes_compression"] == 0.0
    with pytest.raises(KeyError):
        profiler.phase("not_a_phase")


# --------------------------------------------------------------------- #
# 7. MCQ protocol policy + sidecar
# --------------------------------------------------------------------- #
class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class _Out:
            input_ids = [ord(text[-1]) % 5000]

        return _Out()


def test_official_mode_never_masks_first_token():
    assert mcq_first_token_ids(["A", "B"], _FakeTokenizer(), mode="official") is None


def test_constrained_mode_returns_deduped_ids():
    ids = mcq_first_token_ids(["A", "B", "C"], _FakeTokenizer(), mode="constrained")
    assert ids and len(ids) == len(set(ids))
    with pytest.raises(ValueError):
        mcq_first_token_ids(["A"], _FakeTokenizer(), mode="nonsense")


def test_best_option_from_logits_is_read_only():
    logits = {"A": -1.0, "B": 2.5, "C": 0.0}
    snapshot = dict(logits)
    assert best_option_from_logits(logits) == "B"
    assert logits == snapshot
    assert best_option_from_logits({}) is None


# --------------------------------------------------------------------- #
# 8. runtime dispatch / Full-HERMES mode selection
# --------------------------------------------------------------------- #
def test_runtime_backend_dispatch_defaults_legacy():
    from evaluation_pipeline.runtime import FULL_CACHE_BUDGET, ModelRuntime

    legacy = ModelRuntime({"name": "MMDuet2", "model_path": "models/MMDuet2"})
    assert legacy.backend == BACKEND_MMDUET2
    assert legacy.mcq_decode_mode == "official"

    qwen = ModelRuntime(
        {
            "backend": "qwen2_5_vl",
            "memory": {"method": "hermes", "compression": {"budget": 200}},
        }
    )
    assert qwen.backend == BACKEND_QWEN25VL
    assert qwen.memory_method == "hermes"
    assert FULL_CACHE_BUDGET >= 10 ** 11


def test_apply_defaults_fills_backend_and_run_id_identity():
    legacy = apply_defaults(
        {"model": {"name": "MMDuet2", "model_path": "models/MMDuet2"}}
    )
    rid_legacy, identity_legacy = make_run_id(legacy, "ovobench", {})
    assert identity_legacy["model_name"] == "MMDuet2"
    assert "model_backend" not in identity_legacy

    qwen = apply_defaults({"model": {"backend": "qwen2_5_vl"}})
    assert qwen["model"]["model_path"].endswith("Qwen2.5-VL-7B-Instruct")
    rid_qwen, identity_qwen = make_run_id(qwen, "ovobench", {})
    assert identity_qwen["model_backend"] == "qwen2_5_vl"
    assert rid_qwen != rid_legacy


# --------------------------------------------------------------------- #
# 9. model-class surface (no weights loaded)
# --------------------------------------------------------------------- #
def test_qwen25vl_model_class_surface():
    import sys

    sys.path.insert(0, str(ROOT / "OVO-Bench"))
    module = pytest.importorskip("models.Qwen25VL_Stream")
    cls = module.EvalQwen25VLStream
    for method in (
        "reset",
        "stream_video_chunk",
        "ask_question",
        "inference",
        "inference_forward",
    ):
        assert callable(getattr(cls, method))
    for name in ("iter_chunks_until_query", "iter_frame_chunks"):
        assert callable(getattr(module, name))
    assert cls.FORWARD_TASKS == {"REC", "SSR", "CRR"}
    assert "EPM" in cls.TIMESTAMP_TASKS


def test_smoke_cli_exits_cleanly_on_missing_model_path():
    from evaluation_pipeline.models.qwen25_vl_smoke import main

    for command in ("qwen-load-smoke", "qwen-stream-smoke", "qwen-hermes-smoke"):
        assert main([command, "--model_path", "/nonexistent/path"]) == 2


# --------------------------------------------------------------------- #
# 10. non-proactive / no-mask protocol guarantees (no model instantiation)
# --------------------------------------------------------------------- #
def _qwen_class():
    import sys
    import types  # noqa: F401

    sys.path.insert(0, str(ROOT / "OVO-Bench"))
    module = pytest.importorskip("models.Qwen25VL_Stream")
    return module.EvalQwen25VLStream


def test_must_reply_is_rejected_on_qwen_backend():
    import types

    cls = _qwen_class()
    fake = types.SimpleNamespace()
    with pytest.raises(ValueError):
        cls._ask(fake, "question", max_new_tokens=4, must_reply=True)


def test_full_mode_reports_zero_compaction():
    import types

    cls = _qwen_class()
    fake = types.SimpleNamespace(
        compression_enabled=False,
        accounting=CacheAccounting(
            effective_compaction=7
        ),
        visual_tokens_per_chunk=[100, 100, 100],
        seconds_per_chunk=[4.0, 4.0, 4.0],
    )
    assert cls.effective_compaction_count(fake) == 0
    rate = cls.visual_token_rate(fake)
    assert rate["tokens_per_chunk"] == pytest.approx(100.0)
    assert rate["seconds_per_chunk"] == pytest.approx(4.0)
    assert rate["tokens_per_second"] == pytest.approx(25.0)


def test_hermes_mode_reports_measured_compaction():
    import types

    cls = _qwen_class()
    fake = types.SimpleNamespace(
        compression_enabled=True,
        accounting=CacheAccounting(effective_compaction=3),
    )
    assert cls.effective_compaction_count(fake) == 3
