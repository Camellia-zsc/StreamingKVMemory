import os
import sys
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from utils.OVOBench import OVOBenchOffline


# 决定使用原始 MMDuet2 代码，还是 MMDuet2_Think 代码。
# 运行前可以通过环境变量覆盖。
MMDUET2_CODE_ROOT = os.environ.get(
    "MMDUET2_CODE_ROOT",
    "/home/workspace/zsc/MMDuet2/proactive_eval",
)

if not os.path.isdir(MMDUET2_CODE_ROOT):
    raise FileNotFoundError(
        f"MMDuet2 代码目录不存在：{MMDUET2_CODE_ROOT}"
    )

# 放到最前面，确保加载指定项目中的 qwen2_5_vl 和 qwen_vl_utils。
if MMDUET2_CODE_ROOT not in sys.path:
    sys.path.insert(0, MMDUET2_CODE_ROOT)

from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def sample_video_frames(
    video_path: str,
    sample_fps: float = 1.0,
    max_frames: int = 768,
) -> List[Image.Image]:
    """
    按固定 FPS 从切分视频中采样，并返回 PIL RGB 图片。

    sample_fps=1.0：
        每秒抽取约 1 帧。

    max_frames：
        超过该数量时，在已采样帧中做均匀下采样，
        防止一次输入过长导致显存不足。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps is None or source_fps <= 0:
        source_fps = 30.0

    frame_step = max(1, int(round(source_fps / sample_fps)))

    frames: List[Image.Image] = []
    frame_index = 0

    try:
        while True:
            success, frame_bgr = cap.read()

            if not success:
                break

            if frame_index % frame_step == 0:
                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )
                frames.append(Image.fromarray(frame_rgb))

            frame_index += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"视频中没有读取到有效帧：{video_path}")

    if max_frames > 0 and len(frames) > max_frames:
        selected_indices = np.linspace(
            0,
            len(frames) - 1,
            max_frames,
            dtype=int,
        )
        frames = [frames[i] for i in selected_indices]

    return frames


class EvalMMDuet2(OVOBenchOffline):
    """
    第一阶段兼容适配器：

    将 OVO-Bench 的一个切分视频按 1 FPS 抽帧，
    然后作为一组多图输入一次性交给 MMDuet2。

    该版本用于先验证：
    数据 → 模型 → 结果文件 → 评分
    的完整链路。

    它暂时不代表最终的流式 KV Cache 评测。
    """

    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.device = "cuda:0"

        self.sample_fps = float(
            os.environ.get("MMDUET2_SAMPLE_FPS", "1.0")
        )
        self.max_frames = int(
            os.environ.get("MMDUET2_MAX_FRAMES", "768")
        )

        print("MMDuet2 code root:", MMDUET2_CODE_ROOT)
        print("MMDuet2 model path:", args.model_path)
        print("Sample FPS:", self.sample_fps)
        print("Max frames:", self.max_frames)

        self.processor = AutoProcessor.from_pretrained(
            args.model_path
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .eval()
            .to(self.device)
        )

    @torch.inference_mode()
    def inference(
        self,
        video_file_name: str,
        prompt: str,
        start_time: float = 0,
        end_time: float = 0,
    ) -> str:
        frames = sample_video_frames(
            video_file_name,
            sample_fps=self.sample_fps,
            max_frames=self.max_frames,
        )

        print(
            f"\nVideo: {video_file_name}\n"
            f"Sampled frames: {len(frames)}"
        )

        content = [
            {
                "type": "image",
                "image": frame,
            }
            for frame in frames
        ]

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        conversation = [
            {
                "role": "system",
                "content": (
                    "Answer the question using the provided video frames. "
                    "Follow the output format requested by the user exactly."
                ),
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(
            conversation
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        model_output = self.model.generate(
            **inputs,
            max_new_tokens=16,
            past_key_values=None,
            return_dict_in_generate=True,
            do_sample=False,
            drop_method="none",
            drop_threshold=1.0,
            drop_absolute=True,
        )

        generated_ids = model_output.sequences[
            :,
            inputs.input_ids.size(1):
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        print("Response:", response)

        return response
