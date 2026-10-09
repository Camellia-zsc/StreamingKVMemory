import gc
import importlib.util
import inspect
import os
import re
import sys
import math
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from utils.OVOBench import OVOBenchOffline


# ============================================================
# 1. Load HERMES Qwen2.5-VL backend
# ============================================================

HERMES_CODE_ROOT = os.environ.get(
    "HERMES_CODE_ROOT",
    "/home/workspace/zsc/HERMES",
)

if not os.path.isdir(HERMES_CODE_ROOT):
    raise FileNotFoundError(
        f"HERMES code root does not exist: {HERMES_CODE_ROOT}"
    )

# Important: put HERMES before OVO-Bench itself in sys.path,
# otherwise OVO-Bench/inference.py can shadow HERMES/inference package.
if HERMES_CODE_ROOT not in sys.path:
    sys.path.insert(0, HERMES_CODE_ROOT)

_requested_attention = os.environ.get(
    "MMDUET2_ATTN_IMPLEMENTATION",
    "auto",
)
# ---- ablation-only backend override ------------------------------------ #
# The M-RoPE ablation must run on the SDPA implementation; the FA2 variant
# crashes with the per-layer 3D position_ids hook (transformers 4.57.1).
# Unset => the historical selector below is unchanged.
_ablation_backend = os.environ.get("HERMES_ABLATION_BACKEND", "").strip().lower()
if _ablation_backend:
    if _ablation_backend not in {"sdpa", "fa2"}:
        raise ValueError(
            "HERMES_ABLATION_BACKEND must be 'sdpa' or 'fa2', "
            f"got {_ablation_backend!r}"
        )
    _requested_attention = (
        "flash_attention_2" if _ablation_backend == "fa2" else "sdpa"
    )
    print(f"[ABLATION] backend override -> {_ablation_backend}")
_has_flash_attention = importlib.util.find_spec("flash_attn") is not None
if _requested_attention == "flash_attention_2" and not _has_flash_attention:
    raise ModuleNotFoundError(
        "OVO backend: FA2 requested "
        "(MMDUET2_ATTN_IMPLEMENTATION=flash_attention_2) but flash_attn is "
        "not installed in this environment. Run with the hermes-qwen env or "
        "pass --attn_implementation sdpa."
    )
if _requested_attention == "flash_attention_2" or (
    _requested_attention == "auto" and _has_flash_attention
):
    from inference.qwenvl_hermes_fa2 import load_model as load_hermes_model

    HERMES_ATTENTION_BACKEND = "fa2"
else:
    from inference.qwenvl_hermes import load_model as load_hermes_model

    HERMES_ATTENTION_BACKEND = "sdpa"

print(f"OVO backend: {HERMES_ATTENTION_BACKEND.upper()}")
if _ablation_backend:
    # Hard guard: never let an ablation arm silently run on the wrong backend.
    if HERMES_ATTENTION_BACKEND != _ablation_backend:
        raise RuntimeError(
            f"[ABLATION] requested backend={_ablation_backend} but loaded "
            f"{HERMES_ATTENTION_BACKEND}; aborting."
        )
    print(
        f"[ABLATION] backend={HERMES_ATTENTION_BACKEND} "
        f"module={load_hermes_model.__module__} "
        f"per_layer_hook={os.environ.get('HERMES_PER_LAYER_POS_HOOK', 'on')}"
    )


# ============================================================
# 2. Video reading
# ============================================================

def resize_frame(
    frame_bgr,
    max_side: int,
) -> Image.Image:
    height, width = frame_bgr.shape[:2]

    scale = min(
        max_side / max(height, width),
        1.0,
    )

    if scale < 1.0:
        new_width = max(
            1,
            int(round(width * scale)),
        )
        new_height = max(
            1,
            int(round(height * scale)),
        )
        frame_bgr = cv2.resize(
            frame_bgr,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )
    return Image.fromarray(frame_rgb)


FrameChunk = Tuple[
    List[Image.Image],
    bool,
    float,
]


def iter_chunks_until_query(
    video_path: str,
    query_time: float,
    sample_fps: float,
    frames_per_turn: int,
    max_side: int,
    start_time: float = 0.0,
) -> Iterator[FrameChunk]:
    """
    Backward / Realtime:
    stream frames only up to the query time.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )
    if source_fps <= 0:
        source_fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    start_frame_index = max(0, int(round(start_time * source_fps)))
    query_frame_index = start_frame_index + max(
        0,
        int(math.ceil(query_time * source_fps) - 1),
    )

    if total_frames > 0:
        query_frame_index = min(
            query_frame_index,
            total_frames - 1,
        )

    frame_step = max(
        1,
        int(round(source_fps / sample_fps)),
    )

    buffer: List[Image.Image] = []
    buffer_time = 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    frame_index = start_frame_index
    last_frame_bgr = None
    last_frame_index = -1
    last_sampled_index = -1

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            last_frame_bgr = frame_bgr
            last_frame_index = frame_index

            if frame_index < query_frame_index:
                if (frame_index - start_frame_index) % frame_step == 0:
                    buffer.append(
                        resize_frame(
                            frame_bgr,
                            max_side,
                        )
                    )
                    buffer_time = (
                        (frame_index - start_frame_index) / source_fps
                    )
                    last_sampled_index = frame_index

                    if len(buffer) >= frames_per_turn:
                        yield (
                            buffer,
                            False,
                            buffer_time,
                        )
                        buffer = []
            else:
                if buffer:
                    yield (
                        buffer,
                        False,
                        buffer_time,
                    )
                    buffer = []

                query_image = resize_frame(
                    frame_bgr,
                    max_side,
                )
                query_frame_time = (
                    (frame_index - start_frame_index) / source_fps
                )

                yield (
                    [query_image],
                    True,
                    query_frame_time,
                )
                return

            frame_index += 1

        if last_frame_bgr is None:
            raise RuntimeError(
                f"Video has no valid frames: {video_path}"
            )

        if (
            buffer
            and last_sampled_index == last_frame_index
        ):
            buffer.pop()

        if buffer:
            yield (
                buffer,
                False,
                buffer_time,
            )

        yield (
            [
                resize_frame(
                    last_frame_bgr,
                    max_side,
                )
            ],
            True,
            max(
                0.0,
                (last_frame_index - start_frame_index) / source_fps,
            ),
        )

    finally:
        cap.release()


def iter_frame_chunks(
    video_path: str,
    sample_fps: float,
    frames_per_turn: int,
    max_side: int,
    max_frames: Optional[int] = None,
) -> Iterator[List[Image.Image]]:
    """
    Forward:
    sample frames through the whole already-cut evaluation clip.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )
    if source_fps <= 0:
        source_fps = 30.0

    frame_step = max(
        1,
        int(round(source_fps / sample_fps)),
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = None
    if max_frames is not None and max_frames > 0 and total_frames > 0:
        target_indices = set(
            np.linspace(
                0,
                total_frames - 1,
                max_frames,
            ).round().astype(int).tolist()
        )

    buffer: List[Image.Image] = []
    frame_index = 0

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            if (
                target_indices is not None
                and frame_index in target_indices
            ) or (
                target_indices is None
                and frame_index % frame_step == 0
            ):
                buffer.append(
                    resize_frame(
                        frame_bgr,
                        max_side,
                    )
                )

                if len(buffer) >= frames_per_turn:
                    yield buffer
                    buffer = []

            frame_index += 1

        if buffer:
            yield buffer

    finally:
        cap.release()


def mark_last_chunk(
    chunks: Iterator[List[Image.Image]],
) -> Iterator[
    Tuple[List[Image.Image], bool]
]:
    iterator = iter(chunks)

    try:
        current = next(iterator)
    except StopIteration:
        return

    for following in iterator:
        yield current, False
        current = following

    yield current, True


# ============================================================
# 3. Reply processing
# ============================================================

def clean_reply(
    reply: Optional[str],
) -> str:
    if reply is None:
        return ""

    reply = reply.strip()

    prefix = "I must reply."
    if reply.startswith(prefix):
        reply = reply[len(prefix):].strip()

    return reply


def is_no_reply(
    reply: Optional[str],
) -> bool:
    if reply is None:
        return True

    normalized = reply.strip().upper()

    return (
        normalized == ""
        or normalized == "NO"
        or normalized.startswith("NO REPLY")
    )


def extract_option(
    reply: str,
) -> str:
    normalized = reply.strip().upper()

    start_match = re.match(
        r"^\s*([A-E])(?:\s|[.):,;]|$)",
        normalized,
    )
    if start_match:
        return start_match.group(1)

    option_match = re.search(
        r"(?<![A-Z])([A-E])(?![A-Z])",
        normalized,
    )
    if option_match:
        return option_match.group(1)

    return reply.strip()


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def normalize_forward_reply(
    reply: Optional[str],
    task: str,
) -> str:
    reply = clean_reply(reply)

    if task == "REC":
        number_match = re.search(
            r"\b\d+\b",
            reply,
        )
        if number_match:
            return number_match.group(0)

        normalized = reply.lower()
        for word, number in NUMBER_WORDS.items():
            if re.search(
                rf"\b{word}\b",
                normalized,
            ):
                return number

        return reply

    if task in {"SSR", "CRR"}:
        if is_no_reply(reply):
            return "No"

        normalized = reply.lower()

        yes_match = re.search(
            r"\byes\b",
            normalized,
        )
        no_match = re.search(
            r"\bno\b",
            normalized,
        )

        if yes_match and not no_match:
            return "Yes"
        if no_match and not yes_match:
            return "No"
        if normalized.startswith("yes"):
            return "Yes"
        if normalized.startswith("no"):
            return "No"

        return reply

    return reply


def build_forward_initial_prompt(
    prompt: str,
    task: str,
) -> str:
    # if task == "REC":
    #     return (
    #         prompt
    #         + "\nTrack this activity while the video "
    #         "continues. If the current evidence is not "
    #         "sufficient, output exactly \"NO REPLY\". "
    #         "When you do answer, output only the total "
    #         "count as an integer. Do not describe the video."
    #     )
    if task == "REC":
        return (
            prompt
            + "\nTrack this activity while the video "
            "continues. When asked for the current result, "
            "output only the total count as an integer. "
            "Do not describe the video."
        )
    # if task == "SSR":
    #     return (
    #         prompt
    #         + "\nDetermine whether the specified step is "
    #         "happening at the current evaluation moment. "
    #         "If the current evidence is not sufficient, "
    #         "output exactly \"NO REPLY\". "
    #         "When you do answer, output exactly \"Yes\" or "
    #         "\"No\". Do not describe the video."
    #     )
    if task == "SSR":
        return (
            prompt
            + "\nDetermine whether the specified step is "
            "happening at the current evaluation moment. "
            "At the final evaluation moment, output exactly "
            "\"Yes\" or \"No\". "
            "Do not describe the video."
        )
    # if task == "CRR":
    #     return (
    #         prompt
    #         + "\nYour task is not to answer the original "
    #         "question itself. Decide whether the video "
    #         "observed so far contains enough evidence to "
    #         "answer that question. If there is not enough "
    #         "evidence yet, output exactly \"NO REPLY\". "
    #         "Otherwise output exactly \"Yes\". "
    #         "Do not answer the original question and do not "
    #         "describe the video."
    #     )
    if task == "CRR":
         return (
            prompt
            + "\nYour task is not to answer the original "
            "question itself. Your task is to decide whether "
            "the video observed so far contains enough "
            "evidence to answer that question. "
            "At the final evaluation moment, output exactly "
            "\"Yes\" or \"No\". "
            "Do not answer the original question and do not "
            "describe the video."
    )
    return prompt


def build_forward_final_instruction(
    task: str,
) -> str:
    # if task == "REC":
    #     return (
    #         "The video has now reached the evaluation time. "
    #         "Answer the counting question now. "
    #         "Output only one non-negative integer. "
    #         "Do not output NO REPLY and do not explain."
    #     )

    # if task == "SSR":
    #     return (
    #         "The video has now reached the exact evaluation "
    #         "time. Is the specified step happening at this "
    #         "moment? Output exactly \"Yes\" or \"No\". "
    #         "Do not output NO REPLY and do not explain."
    #     )

    # if task == "CRR":
    #     return (
    #         "The video has now reached the exact evaluation "
    #         "time. Based only on the video observed so far, "
    #         "is there enough evidence to answer the original "
    #         "question? Output exactly \"Yes\" or \"No\". "
    #         "Do not answer the original question. "
    #         "Do not describe the video."
    #     )
    if task == "REC":
        return (
            "The video has now reached the final evaluation time. "
            "You must answer the task now. "
            "Based only on the video observed so far, report the "
            "current total count. "
            "Output exactly one non-negative integer. "
            "Do not output NO REPLY. "
            "Do not explain."
        )

    if task == "SSR":
        return (
            "The video has now reached the final evaluation time. "
            "You must answer the task now. "
            "Based only on the video observed so far, determine "
            "whether the specified step is happening at this exact "
            "moment. "
            "Output exactly \"Yes\" or \"No\". "
            "Do not output NO REPLY. "
            "Do not explain."
        )

    if task == "CRR":
        return (
            "The video has now reached the final evaluation time. "
            "You must answer the task now. "
            "Based only on the video observed so far, determine "
            "whether there is enough evidence to answer the original "
            "question. "
            "Output exactly \"Yes\" or \"No\". "
            "Do not output NO REPLY. "
            "Do not answer the original question. "
            "Do not describe the video. "
            "Do not explain."
        )
    return ""


def get_forward_allowed_first_token_ids(
    tokenizer,
    task: str,
):
    """
    OVO Forward final-turn format constraint.

    REC:
        first generated token must start with a digit.

    SSR / CRR:
        first generated token must be Yes or No.

    Only the first token is constrained.
    """
    if task == "REC":
        candidates = [
            str(i)
            for i in range(10)
        ]
    elif task in {"SSR", "CRR"}:
        candidates = [
            "Yes",
            "No",
        ]
    else:
        return None

    allowed_ids = []

    for candidate in candidates:
        ids = tokenizer(
            candidate,
            add_special_tokens=False,
        ).input_ids

        if not ids:
            continue

        allowed_ids.append(
            int(ids[0])
        )

    allowed_ids = list(
        dict.fromkeys(allowed_ids)
    )

    if not allowed_ids:
        raise RuntimeError(
            f"No allowed first-token IDs for task={task}"
        )

    return allowed_ids


# ============================================================
# 4. MMDuet2 + HERMES adapter
# ============================================================

class EvalMMDuet2Hermes(
    OVOBenchOffline
):
    """
    MMDuet2 post-trained checkpoint + HERMES KV memory.

    Division of responsibilities:
      - MMDuet2 weights / prompting decide whether to answer or NO REPLY.
      - HERMES stores and compresses the long-term visual KV cache.
      - QA text is transient: HERMES question_answering() removes QA KV
        after generation, so persistent memory remains video-centric.
    """

    supports_timestamp_inference = True
    supports_forward_inference = True

    TIMESTAMP_TASKS = {
        "EPM",
        "ASI",
        "HLD",
        "STU",
        "OJR",
        "ATR",
        "ACR",
        "OCR",
        "FPD",
    }

    FORWARD_TASKS = {
        "REC",
        "SSR",
        "CRR",
    }

    def __init__(
        self,
        args,
    ):
        super().__init__(args)

        self.sample_fps = float(
            os.environ.get(
                "MMDUET2_SAMPLE_FPS",
                "0.5",
            )
        )

        self.frames_per_turn = int(
            os.environ.get(
                "MMDUET2_FRAMES_PER_TURN",
                "2",
            )
        )

        self.max_side = int(
            os.environ.get(
                "MMDUET2_MAX_SIDE",
                "336",
            )
        )

        self.kv_size = int(
            os.environ.get(
                "HERMES_KV_SIZE",
                "6000",
            )
        )
        self.compression_stats = {
            "events": [],
            "peak_kv_tokens": 0,
        }

        self.system_prompt = (
            "You are processing a continuously arriving "
            "video stream. If a user question is present "
            "but the current video evidence is not sufficient, "
            "output exactly \"NO REPLY\". Once enough evidence "
            "is available, answer the question. Follow the "
            "output format requested by the user and do not "
            "add explanations."
        )

        self.forward_system_prompt = (
            "You are processing a continuously arriving video stream. "
            "During intermediate monitoring turns, you may output exactly "
            "\"NO REPLY\" when no task answer is required yet. "
            "At the final evaluation time, you must answer the requested "
            "task based only on the video observed so far. "
            "\"NO REPLY\" is not allowed at the final evaluation time. "
            "Follow the exact output format requested by the user and do not "
            "add explanations."
        )

        self.po_system_prompt = (
            "You are a helpful visual question-answering assistant. "
            "Answer the current question based only on the video "
            "content provided to you. Follow the requested output "
            "format exactly and do not add explanations."
        )

        print(
            "HERMES code root:",
            HERMES_CODE_ROOT,
        )
        print(
            "MMDuet2 checkpoint:",
            args.model_path,
        )
        print(
            "HERMES KV budget:",
            self.kv_size,
        )
        print(
            "Streaming sample FPS:",
            self.sample_fps,
        )
        print(
            "Streaming frames per turn:",
            self.frames_per_turn,
        )
        print(
            "Maximum image side:",
            self.max_side,
        )

        self.qa_model, self.qa_processor = (
            load_hermes_model(
                model_path=args.model_path,
                kv_size=self.kv_size,
                streaming=True,
                sample_fps=self.sample_fps,
            )
        )

        # Replace HERMES's generic system prompt with MMDuet2's
        # proactive-response system prompt. Keep the open user turn
        # because HERMES get_prompt()/question_answering() expects it.
        init_text = (
            "<|im_start|>system\n"
            + self.system_prompt
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
        )

        model_device = next(
            self.qa_model.parameters()
        ).device

        self._init_prompt_ids = (
            self.qa_processor.tokenizer(
                init_text,
                return_tensors="pt",
            )
            .input_ids
            .to(model_device)
        )

        forward_init_text = (
            "<|im_start|>system\n"
            + self.forward_system_prompt
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
        )

        self._forward_init_prompt_ids = (
            self.qa_processor.tokenizer(
                forward_init_text,
                return_tensors="pt",
            )
            .input_ids
            .to(model_device)
        )

        po_init_text = (
            "<|im_start|>system\n"
            + self.po_system_prompt
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
        )

        self._po_init_prompt_ids = (
            self.qa_processor.tokenizer(
                po_init_text,
                return_tensors="pt",
            )
            .input_ids
            .to(model_device)
        )

        self.qa_model.init_prompt_ids = (
            self._init_prompt_ids
        )

    def _reset_stream(
        self,
        forward: bool = False,
        proactive_output: bool = False,
    ):
        self.qa_model.clear_cache()

        if proactive_output:
            self.qa_model.init_prompt_ids = (
                self._po_init_prompt_ids
            )
        elif forward:
            self.qa_model.init_prompt_ids = (
                self._forward_init_prompt_ids
            )
        else:
            self.qa_model.init_prompt_ids = (
                self._init_prompt_ids
            )

        if hasattr(
            self.qa_model,
            "conv_history",
        ):
            self.qa_model.conv_history = []

        self.qa_model.encode_init_prompt()

    def _cache_lengths(self):
        if self.qa_model.kv_cache is None:
            return []

        return [
            int(k.shape[2])
            for k, _ in self.qa_model.kv_cache
        ]

    def reset_compression_stats(self):
        self.compression_stats = {
            "events": [],
            "peak_kv_tokens": 0,
        }

    def get_compression_stats(self):
        events = list(self.compression_stats.get("events", []))
        retentions = [
            event["retention_ratio"]
            for event in events
            if event.get("retention_ratio") is not None
        ]
        compressions = [
            event["compression_ratio"]
            for event in events
            if event.get("compression_ratio") is not None
        ]
        return {
            "peak_kv_tokens": self.compression_stats.get("peak_kv_tokens") or None,
            "compression_events": len(events),
            "total_removed_tokens": sum(
                event["removed"] for event in events
                if event.get("removed") is not None
            ),
            "mean_kv_retention_ratio": (
                sum(retentions) / len(retentions) if retentions else None
            ),
            "mean_kv_compression_ratio": (
                sum(compressions) / len(compressions) if compressions else None
            ),
            "compression_events_detail": events,
        }

    def _cache_length_summary(self):
        lengths = self._cache_lengths()

        if not lengths:
            return "empty"

        return (
            f"min={min(lengths)}, "
            f"max={max(lengths)}"
        )

    def _encode_frames(
        self,
        frames: List[Image.Image],
    ):
        if not frames:
            return

        # 与 Native MMDuet2 保持一致：默认每一帧都是独立 image。
        image_chunk = [
            frame.convert("RGB")
            for frame in frames
        ]

        before = (
            self._cache_lengths()
        )

        # ---- ablation-only encoding switch -------------------------------
        # HERMES_ENCODE_STYLE=image|video. Unset => historical behaviour
        # (prefer per-frame images when the backend exposes that entry point).
        _style = os.environ.get("HERMES_ENCODE_STYLE", "").strip().lower()
        if _style and _style not in {"image", "video"}:
            raise ValueError(
                f"HERMES_ENCODE_STYLE must be 'image' or 'video', got {_style!r}"
            )
        path = "image" if hasattr(self.qa_model, "encode_image_chunk") else "video"
        style = _style or path
        if style == "image" and not hasattr(self.qa_model, "encode_image_chunk"):
            raise RuntimeError(
                "[ABLATION] encode_style=image requested but encode_image_chunk "
                "is unavailable on this backend"
            )
        if style != path and _style:
            print(
                f"[ABLATION] encode_style={style} (forced); backend offers "
                f"{path} entry point"
            )
        self.last_encode_style = style
        if style == "image":
            self.qa_model.encode_image_chunk(image_chunk)
        else:
            video_chunk = torch.from_numpy(np.stack(
                [np.asarray(frame) for frame in image_chunk],
                axis=0,
            ))
            self.qa_model.encode_video_chunk(video_chunk)
        if os.environ.get("HERMES_ENCODE_STYLE") and not getattr(
            self, "_encode_style_logged", False
        ):
            self._encode_style_logged = True
            print(
                f"[ABLATION] backend={HERMES_ATTENTION_BACKEND} "
                f"encode_style={self.last_encode_style}"
            )
        # -------------------------------------------------------------------

        after = (
            self._cache_lengths()
        )
        if after:
            self.compression_stats["peak_kv_tokens"] = max(
                self.compression_stats.get("peak_kv_tokens", 0),
                max(after),
            )

        # smoke test 时打开：
        # export HERMES_INPUT_DEBUG=1
        if (
            os.environ.get(
                "HERMES_INPUT_DEBUG",
                "0",
            )
            == "1"
        ):
            delta = (
                after[0] - before[0]
                if before and after
                else None
            )

            kv_text = (
                f"min={min(after)}, max={max(after)}"
                if after
                else "empty"
            )

            print(
                "[HERMES IMAGE PREFILL] "
                f"frames={len(image_chunk)}, "
                f"kv_delta={delta}, "
                f"kv={kv_text}"
            )

        # HERMES 原来的压缩触发逻辑不变
        if (
            after
            and max(after) > self.kv_size
        ):
            before = max(after)
            self.qa_model.predict_and_compress()
            compressed = self._cache_lengths()
            after_compression = max(compressed) if compressed else None
            event = {
                "before": before,
                "after": after_compression,
                "removed": (
                    before - after_compression
                    if after_compression is not None
                    else None
                ),
                "retention_ratio": (
                    after_compression / before
                    if after_compression is not None and before > 0
                    else None
                ),
                "compression_ratio": (
                    1 - after_compression / before
                    if after_compression is not None and before > 0
                    else None
                ),
            }
            self.compression_stats["events"].append(event)
            print(
                "HERMES compression: "
                f"before = {before}, "
                f"after = {after_compression}, "
                f"retention = "
                f"{event['retention_ratio']:.2%}"
                if event["retention_ratio"] is not None
                else (
                    "HERMES compression: "
                    f"before = {before}, after = {after_compression}, "
                    "retention = None"
                )
            )



    def _ask(
        self,
        text: str,
        *,
        max_new_tokens: int,
        mc: bool = False,
        must_reply: bool = False,
        allowed_first_token_ids=None,
    ) -> str:
        prompt = self.qa_model.get_prompt(
            text,
            mc=mc,
        )

        # MMDuet2 native inference appends this after the assistant
        # generation prompt when a reply is mandatory.
        if must_reply:
            prompt += "I must reply.\n"

        input_text = {
            "question": text,
            "formatted_question": (
                text if mc else None
            ),
            "prompt": prompt,
        }

        answer_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
        }
        if "allowed_first_token_ids" in inspect.signature(
            self.qa_model.question_answering
        ).parameters:
            answer_kwargs["allowed_first_token_ids"] = allowed_first_token_ids
        reply = self.qa_model.question_answering(input_text, **answer_kwargs)

        return clean_reply(reply)

    # ========================================================
    # Backward + Realtime
    # ========================================================

    @torch.inference_mode()
    def inference(
        self,
        video_file_name: str,
        prompt: str,
        start_time: float = 0.0,
        end_time: float = 0.0,
        task: Optional[str] = None,
        sample_id: Optional[int] = None,
        **kwargs,
    ) -> str:
        if task not in self.TIMESTAMP_TASKS:
            raise NotImplementedError(
                "Timestamp inference does not support: "
                f"{task!r}"
            )

        start_time = float(start_time)
        query_time = float(end_time)

        relative_query_time = (
            query_time - start_time
        )
        allowed_first_token_ids = kwargs.get(
            "allowed_first_token_ids"
        )
        # Experimental switch (OVO Backward/Realtime MCQ ablation only).
        # Default True keeps the official behavior: the mandatory-reply
        # assistant prefix is appended before generation.
        must_reply = bool(
            kwargs.get("must_reply", True)
        )

        print(
            f"\n[HERMES START] "
            f"id={sample_id}, "
            f"task={task}, "
            f"query_time={query_time:.3f}s, "
            f"video={video_file_name}"
        )

        self._reset_stream()

        prefill_turns = 0
        final_reply = None
        query_frame_time = None

        try:
            stream = iter_chunks_until_query(
                video_path=video_file_name,
                query_time=relative_query_time,
                sample_fps=self.sample_fps,
                frames_per_turn=(
                    self.frames_per_turn
                ),
                max_side=self.max_side,
                start_time=start_time,
            )

            for (
                frames,
                is_query_turn,
                relative_turn_time,
            ) in stream:
                turn_time = (
                    start_time
                    + float(relative_turn_time)
                )

                # Query frame is also visual evidence and must be
                # encoded before answering.
                self._encode_frames(frames)

                if not is_query_turn:
                    prefill_turns += 1
                    continue

                question_prompt = (
                    prompt
                    + "\nThe question is available now. "
                    "Answer it immediately. "
                    "Do not output NO REPLY. "
                    "Return only the option letter."
                )

                final_reply = self._ask(
                    question_prompt,
                    max_new_tokens=16,
                    mc=True,
                    must_reply=must_reply,
                    allowed_first_token_ids=allowed_first_token_ids,
                )

                query_frame_time = turn_time
                break

            if final_reply is None:
                raise RuntimeError(
                    "No query turn was executed"
                )

            final_answer = extract_option(
                final_reply
            )

            print(
                f"[HERMES RESULT] "
                f"id={sample_id}, "
                f"task={task}, "
                f"query_time={query_time:.3f}s, "
                f"query_frame_time="
                f"{query_frame_time:.3f}s, "
                f"prefill_turns={prefill_turns}, "
                f"kv={self._cache_length_summary()}, "
                f"raw={final_reply!r}, "
                f"answer={final_answer!r}"
            )

            return final_answer

        except torch.cuda.OutOfMemoryError:
            print(
                f"[HERMES OOM] "
                f"id={sample_id}, "
                f"task={task}, "
                f"query_time={query_time:.3f}s"
            )
            raise

        finally:
            self.qa_model.clear_cache()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # Forward: REC / SSR / CRR
    # ========================================================

    @torch.inference_mode()
    def inference_forward(
        self,
        video_file_name: str,
        prompt: str,
        realtime: float,
        task: Optional[str] = None,
        sample_id: Optional[int] = None,
        test_index: Optional[int] = None,
        **kwargs,
    ) -> str:
        if task not in self.FORWARD_TASKS:
            raise NotImplementedError(
                "Forward inference does not support: "
                f"{task!r}"
            )

        print(
            f"\n[HERMES FORWARD START] "
            f"id={sample_id}, "
            f"index={test_index}, "
            f"task={task}, "
            f"realtime={realtime:.3f}s, "
            f"video={video_file_name}"
        )

        self._reset_stream(
            forward=True
        )

        turn_count = 0
        final_reply = None

        active_prompt = (
            build_forward_initial_prompt(
                prompt=prompt,
                task=task,
            )
        )

        try:
            chunks = iter_frame_chunks(
                video_path=video_file_name,
                sample_fps=self.sample_fps,
                frames_per_turn=(
                    self.frames_per_turn
                ),
                max_side=self.max_side,
            )

            for (
                frames,
                is_last,
            ) in mark_last_chunk(chunks):
                turn_count += 1

                # Persistent memory contains video KV only.
                self._encode_frames(frames)

                # HERMES intentionally truncates QA KV after each
                # answer. Therefore the active question is resent
                # every turn while the video KV keeps accumulating.
                current_prompt = active_prompt

                if is_last:
                    current_prompt = (
                        active_prompt
                        + "\n"
                        + build_forward_final_instruction(
                            task=task,
                        )
                    )

                allowed_first_token_ids = None

                if is_last:
                    allowed_first_token_ids = (
                        get_forward_allowed_first_token_ids(
                            self.qa_processor.tokenizer,
                            task,
                        )
                    )

                reply = self._ask(
                    current_prompt,
                    max_new_tokens=64,
                    mc=False,
                    must_reply=is_last,
                    allowed_first_token_ids=(
                        allowed_first_token_ids
                    ),
                )

                print(
                    f"[HERMES FORWARD TURN] "
                    f"id={sample_id}, "
                    f"index={test_index}, "
                    f"turn={turn_count}, "
                    f"is_last={is_last}, "
                    f"reply={reply!r}, "
                    f"kv={self._cache_length_summary()}"
                )

                if is_last:
                    final_reply = reply
                    break

            if turn_count == 0:
                raise RuntimeError(
                    f"Video has no valid frames: "
                    f"{video_file_name}"
                )

            response = normalize_forward_reply(
                reply=final_reply,
                task=task,
            )

            print(
                f"[HERMES FORWARD RESULT] "
                f"id={sample_id}, "
                f"index={test_index}, "
                f"task={task}, "
                f"realtime={realtime:.3f}s, "
                f"turns={turn_count}, "
                f"kv={self._cache_length_summary()}, "
                f"raw_answer={final_reply!r}, "
                f"answer={response!r}"
            )

            return response

        except torch.cuda.OutOfMemoryError:
            print(
                f"[HERMES FORWARD OOM] "
                f"id={sample_id}, "
                f"index={test_index}, "
                f"task={task}"
            )
            raise

        finally:
            self.qa_model.clear_cache()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
