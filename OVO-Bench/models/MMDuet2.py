import copy
import gc
import importlib.util
import math
import os
import re
import sys
from types import SimpleNamespace
from typing import Iterator, List, Optional, Tuple

import cv2
import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from utils.OVOBench import OVOBenchOffline


# ============================================================
# 1. 加载 MMDuet2
# ============================================================

MMDUET2_CODE_ROOT = os.environ.get(
    "MMDUET2_CODE_ROOT",
    "/home/workspace/zsc/MMDuet2/proactive_eval",
)

if not os.path.isdir(MMDUET2_CODE_ROOT):
    raise FileNotFoundError(
        "MMDuet2 代码目录不存在："
        f"{MMDUET2_CODE_ROOT}"
    )

if MMDUET2_CODE_ROOT not in sys.path:
    sys.path.insert(
        0,
        MMDUET2_CODE_ROOT,
    )

inference_path = os.path.join(
    MMDUET2_CODE_ROOT,
    "inference.py",
)

if not os.path.isfile(inference_path):
    raise FileNotFoundError(
        "MMDuet2 inference.py 不存在："
        f"{inference_path}"
    )

spec = importlib.util.spec_from_file_location(
    "mmduet2_proactive_inference",
    inference_path,
)

if spec is None or spec.loader is None:
    raise ImportError(
        "无法加载 MMDuet2 inference.py："
        f"{inference_path}"
    )

mmduet2_inference = (
    importlib.util.module_from_spec(spec)
)

sys.modules[spec.name] = (
    mmduet2_inference
)

spec.loader.exec_module(
    mmduet2_inference
)

ProactiveInferenceClient = (
    mmduet2_inference
    .ProactiveInferenceClient
)

process_vision_info = (
    mmduet2_inference
    .process_vision_info
)


# ============================================================
# 2. 视频读取
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
            (
                new_width,
                new_height,
            ),
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
    用于 Backward 和 Realtime。

    query_time 之前分批返回视频帧；
    到 query_time 时返回问题帧并停止，
    不读取问题时刻之后的视频。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频：{video_path}"
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
        int(
            math.ceil(
                query_time * source_fps
            )
            - 1
        ),
    )

    if total_frames > 0:
        query_frame_index = min(
            query_frame_index,
            total_frames - 1,
        )

    frame_step = max(
        1,
        int(round(
            source_fps / sample_fps
        )),
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

                    last_sampled_index = (
                        frame_index
                    )

                    if (
                        len(buffer)
                        >= frames_per_turn
                    ):
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
                f"视频没有有效帧：{video_path}"
            )

        if (
            buffer
            and last_sampled_index
            == last_frame_index
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
                (last_frame_index - start_frame_index)
                / source_fps,
            ),
        )

    finally:
        cap.release()


def iter_frame_chunks(
    video_path: str,
    sample_fps: float,
    frames_per_turn: int,
    max_side: int,
) -> Iterator[List[Image.Image]]:
    """
    用于 Forward。

    按照 sample_fps 抽帧，
    每 frames_per_turn 张组成一批。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频：{video_path}"
        )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if source_fps <= 0:
        source_fps = 30.0

    frame_step = max(
        1,
        int(round(
            source_fps / sample_fps
        )),
    )

    buffer: List[Image.Image] = []
    frame_index = 0

    try:
        while True:
            success, frame_bgr = cap.read()

            if not success:
                break

            if frame_index % frame_step == 0:
                buffer.append(
                    resize_frame(
                        frame_bgr,
                        max_side,
                    )
                )

                if (
                    len(buffer)
                    >= frames_per_turn
                ):
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
    """
    给 Forward 的最后一批帧标记 is_last=True。
    """
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
# 3. 回复处理
# ============================================================

def clean_reply(
    reply: Optional[str],
) -> str:
    if reply is None:
        return ""

    reply = reply.strip()

    prefix = "I must reply."

    if reply.startswith(prefix):
        reply = reply[
            len(prefix):
        ].strip()

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
        or normalized.startswith(
            "NO REPLY"
        )
    )


def extract_option(
    reply: str,
) -> str:
    normalized = reply.strip().upper()

    start_match = re.match(
        r"^\s*([A-E])"
        r"(?:\s|[.):,;]|$)",
        normalized,
    )

    if start_match:
        return start_match.group(1)

    option_match = re.search(
        r"(?<![A-Z])"
        r"([A-E])"
        r"(?![A-Z])",
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
    """
    将 Forward 输出整理成评分需要的格式。

    REC：
        只保留整数。

    SSR、CRR：
        只保留 Yes 或 No。
        NO REPLY 表示模型认为当前条件不成立，
        因此转换为 No。
    """
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
    """
    第一次向模型提供 Forward 问题时，
    明确任务目标和输出格式。
    """
    if task == "REC":
        return (
            prompt
            + "\nTrack this activity while the video "
            "continues. When asked for the current result, "
            "output only the total count as an integer. "
            "Do not describe the video."
        )

    if task == "SSR":
        return (
            prompt
            + "\nDetermine whether the specified step is "
            "happening at the current evaluation moment. "
            "At the final evaluation moment, output exactly "
            "\"Yes\" or \"No\". "
            "Do not describe the video."
        )

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
    """
    At the final evaluation time, require a benchmark-format
    answer instead of NO REPLY.
    """
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

def get_cache_length(
    client,
):
    if client.past_key_values is None:
        return None

    try:
        return (
            client
            .past_key_values
            .get_seq_length()
        )
    except Exception:
        return "unknown"


class FirstTokenAllowedProcessor(
    LogitsProcessor
):
    def __init__(
        self,
        prompt_length: int,
        allowed_token_ids,
    ):
        self.prompt_length = int(prompt_length)
        self.allowed_token_ids = list(
            dict.fromkeys(
                int(x)
                for x in allowed_token_ids
            )
        )

    def __call__(
        self,
        input_ids,
        scores,
    ):
        # generate() calls the processor before every token.
        # Constrain only the first newly generated token.
        if input_ids.shape[1] != self.prompt_length:
            return scores

        constrained = torch.full_like(
            scores,
            -float("inf"),
        )

        constrained[
            :,
            self.allowed_token_ids,
        ] = scores[
            :,
            self.allowed_token_ids,
        ]

        return constrained


def get_forward_allowed_first_token_ids(
    tokenizer,
    task: str,
):
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

        if ids:
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
# 4. MMDuet2 适配器
# ============================================================

class EvalMMDuet2(
    OVOBenchOffline
):
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

        system_prompt = (
            "You are processing a continuously "
            "arriving video stream. "
            "If a user question is present but "
            "the current video evidence is not "
            "sufficient, output exactly "
            "\"NO REPLY\". "
            "Once enough evidence is available, "
            "answer the question. "
            "Follow the output format requested "
            "by the user and do not add explanations."
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

        attn_implementation = os.environ.get(
            "MMDUET2_ATTN_IMPLEMENTATION",
            "auto",
        )
        if attn_implementation == "auto":
            attn_implementation = (
                "flash_attention_2"
                if importlib.util.find_spec("flash_attn") is not None
                else "sdpa"
            )
            if attn_implementation == "sdpa":
                print(
                    "flash_attn is unavailable; "
                    "falling back to PyTorch SDPA attention."
                )

        client_args = SimpleNamespace(
            llm_pretrained=args.model_path,
            attn_implementation=attn_implementation,
            system_prompt=system_prompt,
            input_assistant_turns=False,
            do_sample=False,
            temperature=1.0,
            top_k=40,
        )

        print(
            "MMDuet2 code root:",
            MMDUET2_CODE_ROOT,
        )

        print(
            "MMDuet2 model path:",
            args.model_path,
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

        self.client = (
            ProactiveInferenceClient(
                client_args
            )
        )

    def _reset_client(
        self,
        forward: bool = False,
        proactive_output: bool = False,
    ):
        self.client.reset()

        self.client.set_fps(
            fps=self.sample_fps
        )

        if proactive_output:
            active_system_prompt = (
                self.po_system_prompt
            )
        elif forward:
            active_system_prompt = (
                self.forward_system_prompt
            )
        else:
            active_system_prompt = (
                self.client.system_prompt
            )

        self.client.history = [
            {
                "role": "system",
                "content": active_system_prompt,
            }
        ]

    def _apply_keep_mask(
        self,
        inputs,
    ):
        all_keep_masks = (
            self.client
            .model
            .model
            .all_keep_masks
        )

        if not all_keep_masks:
            return inputs

        keep_mask = torch.ones_like(
            inputs.input_ids,
            dtype=torch.bool,
        )

        old_keep_mask = torch.cat(
            all_keep_masks,
            dim=1,
        )

        old_length = min(
            old_keep_mask.size(1),
            keep_mask.size(1),
        )

        keep_mask[
            :,
            :old_length,
        ] = old_keep_mask[
            :,
            :old_length,
        ]

        inputs["input_ids"] = (
            inputs.input_ids[
                keep_mask
            ].unsqueeze(0)
        )

        inputs["attention_mask"] = (
            inputs.attention_mask[
                keep_mask
            ].unsqueeze(0)
        )

        return inputs

    def _prepare_inputs(
        self,
        query,
        turn_time: float,
        add_generation_prompt: bool,
        must_reply: bool = False,
    ):
        client = self.client

        client.history.append(query)

        client.history[-1][
            "time"
        ] = turn_time

        text = (
            client.processor
            .apply_chat_template(
                client.history,
                tokenize=False,
                add_generation_prompt=(
                    add_generation_prompt
                ),
            )
        )

        if must_reply:
            text += client.must_reply_prompt

        (
            new_image_inputs,
            new_video_inputs,
        ) = process_vision_info(
            [query]
        )

        if new_image_inputs is not None:
            client.prev_image_inputs.extend(
                new_image_inputs
            )

        if new_video_inputs is not None:
            client.prev_video_inputs.extend(
                new_video_inputs
            )

        image_inputs = (
            copy.deepcopy(
                client.prev_image_inputs
            )
            if client.prev_image_inputs
            else None
        )

        video_inputs = (
            copy.deepcopy(
                client.prev_video_inputs
            )
            if client.prev_video_inputs
            else None
        )

        inputs = client.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

        return self._apply_keep_mask(
            inputs
        )

    @torch.inference_mode()
    def _prefill_video_turn(
        self,
        query,
        turn_time: float,
    ):
        """
        问题尚未出现时只更新视频 KV Cache。
        """
        client = self.client

        inputs = self._prepare_inputs(
            query=query,
            turn_time=turn_time,
            add_generation_prompt=False,
        )

        expected_cache_length = (
            inputs.input_ids.size(1)
        )

        output = client.model.generate(
            **inputs,
            max_new_tokens=1,
            past_key_values=(
                client.past_key_values
            ),
            return_dict_in_generate=True,
            use_cache=True,
            drop_method="none",
            drop_threshold=1.0,
            drop_absolute=True,
            do_sample=False,
        )

        new_cache = (
            output.past_key_values
        )

        actual_cache_length = (
            new_cache.get_seq_length()
        )

        if (
            actual_cache_length
            > expected_cache_length
            and hasattr(new_cache, "crop")
        ):
            new_cache.crop(
                expected_cache_length
            )

        client.past_key_values = new_cache
        client.video_time = turn_time

        del output
        del inputs

    @torch.inference_mode()
    def _answer_query_turn(
        self,
        query,
        turn_time: float,
        must_reply: bool = False,
        allowed_first_token_ids=None,
    ) -> str:
        """
        在 Backward/Realtime 的问题时刻生成答案。
        """
        client = self.client

        inputs = self._prepare_inputs(
            query=query,
            turn_time=turn_time,
            add_generation_prompt=True,
            must_reply=must_reply,
        )

        input_length = (
            inputs.input_ids.size(1)
        )

        generate_kwargs = {
            "max_new_tokens": 16,
            "min_new_tokens": 1,
            "past_key_values": (
                client.past_key_values
            ),
            "return_dict_in_generate": True,
            "use_cache": True,
            "drop_method": "none",
            "drop_threshold": 1.0,
            "drop_absolute": True,
            "do_sample": False,
        }

        if allowed_first_token_ids is not None:
            generate_kwargs["logits_processor"] = (
                LogitsProcessorList(
                    [
                        FirstTokenAllowedProcessor(
                            prompt_length=input_length,
                            allowed_token_ids=(
                                allowed_first_token_ids
                            ),
                        )
                    ]
                )
            )

        output = client.model.generate(
            **inputs,
            **generate_kwargs,
        )

        client.past_key_values = (
            output.past_key_values
        )

        generated_ids = (
            output.sequences[
                :,
                input_length:,
            ]
        )

        reply = (
            client.processor
            .batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        if not reply:
            reply = "NO REPLY"

        client.history.append(
            {
                "role": "assistant",
                "content": reply,
                "time": turn_time,
            }
        )

        client.video_time = turn_time

        del output
        del inputs

        return reply

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
                "时间戳推理不支持："
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

        print(
            f"\n[START] "
            f"id={sample_id}, "
            f"task={task}, "
            f"query_time={query_time:.3f}s, "
            f"video={video_file_name}"
        )

        self._reset_client()

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
                    + float(
                        relative_turn_time
                    )
                )

                content = [
                    {
                        "type": "image",
                        "image": frame,
                    }
                    for frame in frames
                ]

                query = {
                    "role": "user",
                    "content": content,
                }

                if not is_query_turn:
                    self._prefill_video_turn(
                        query=query,
                        turn_time=turn_time,
                    )

                    prefill_turns += 1
                    continue

                question_prompt = (
                    prompt
                    + "\nThe question is available "
                    "now. Answer it immediately. "
                    "Do not output NO REPLY. "
                    "Return only the option letter."
                )

                content.append(
                    {
                        "type": "text",
                        "text": question_prompt,
                    }
                )

                final_reply = (
                    self._answer_query_turn(
                        query=query,
                        turn_time=turn_time,
                        must_reply=True,
                        allowed_first_token_ids=allowed_first_token_ids,
                    )
                )

                query_frame_time = turn_time
                break

            if final_reply is None:
                raise RuntimeError(
                    "没有执行问题轮次"
                )

            final_answer = extract_option(
                final_reply
            )

            cache_length = get_cache_length(
                self.client
            )

            print(
                f"[RESULT] "
                f"id={sample_id}, "
                f"task={task}, "
                f"query_time={query_time:.3f}s, "
                f"query_frame_time="
                f"{query_frame_time:.3f}s, "
                f"prefill_turns={prefill_turns}, "
                f"kv_length={cache_length}, "
                f"answer={final_answer!r}"
            )

            return final_answer

        except torch.cuda.OutOfMemoryError:
            print(
                f"[OOM] "
                f"id={sample_id}, "
                f"task={task}, "
                f"query_time={query_time:.3f}s"
            )
            raise

        finally:
            self.client.reset()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # Forward：REC / SSR / CRR
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
                "Forward 推理不支持："
                f"{task!r}"
            )

        print(
            f"\n[FORWARD START] "
            f"id={sample_id}, "
            f"index={test_index}, "
            f"task={task}, "
            f"realtime={realtime:.3f}s, "
            f"video={video_file_name}"
        )

        self._reset_client(
            forward=True
        )

        turn_count = 0
        final_reply = None

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

                content = [
                    {
                        "type": "image",
                        "image": frame,
                    }
                    for frame in frames
                ]

                # 第一轮提供原始问题和任务格式。
                if turn_count == 1:
                    initial_prompt = (
                        build_forward_initial_prompt(
                            prompt=prompt,
                            task=task,
                        )
                    )

                    content.append(
                        {
                            "type": "text",
                            "text": initial_prompt,
                        }
                    )

                # 最后一轮要求给出该检查时刻的最终判断。
                if is_last:
                    final_instruction = (
                        build_forward_final_instruction(
                            task=task,
                        )
                    )

                    content.append(
                        {
                            "type": "text",
                            "text": final_instruction,
                        }
                    )

                query = {
                    "role": "user",
                    "content": content,
                }

                if is_last:
                    allowed_first_token_ids = (
                        get_forward_allowed_first_token_ids(
                            self.client.processor.tokenizer,
                            task,
                        )
                    )

                    final_reply = (
                        self._answer_query_turn(
                            query=query,
                            turn_time=float(realtime),
                            must_reply=True,
                            allowed_first_token_ids=(
                                allowed_first_token_ids
                            ),
                        )
                    )

                    final_reply = clean_reply(
                        final_reply
                    )
                    break

                self.client.query_queue.append(
                    query
                )

                self.client._encode_query(
                    debug_print=False
                )

                reply = clean_reply(
                    self.client
                    .history[-1]["content"]
                )

            if turn_count == 0:
                raise RuntimeError(
                    f"视频没有有效帧："
                    f"{video_file_name}"
                )

            response = normalize_forward_reply(
                reply=final_reply,
                task=task,
            )

            cache_length = get_cache_length(
                self.client
            )

            print(
                f"[FORWARD RESULT] "
                f"id={sample_id}, "
                f"index={test_index}, "
                f"task={task}, "
                f"realtime={realtime:.3f}s, "
                f"turns={turn_count}, "
                f"kv_length={cache_length}, "
                f"raw_answer={final_reply!r}, "
                f"answer={response!r}"
            )

            return response

        except torch.cuda.OutOfMemoryError:
            print(
                f"[FORWARD OOM] "
                f"id={sample_id}, "
                f"index={test_index}, "
                f"task={task}"
            )
            raise

        finally:
            self.client.reset()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
