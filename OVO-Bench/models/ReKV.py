import os
import re
import sys

import torch
from decord import VideoReader, cpu

from utils.OVOBench import OVOBenchOffline


REKV_ROOT = os.environ.get(
    "REKV_ROOT",
    "/home/workspace/zsc/ReKV",
)

if REKV_ROOT not in sys.path:
    sys.path.insert(0, REKV_ROOT)

from model.llava_onevision_rekv import load_model


class EvalReKV(OVOBenchOffline):
    """使用 ReKV 在 OVO-Bench 的 offline 协议上进行评测。"""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._model_init()

    def _model_init(self):
        print("Loading ReKV model:", self.args.model_path)

        self.qa_model, self.qa_processor = load_model(
            model_path=self.args.model_path,
            n_local=self.args.n_local,
            topk=self.args.retrieve_size,
            chunk_size=self.args.retrieve_chunk_size,
        )

    def _load_video(self, video_path):
        """按照 sample_fps 从截断后的 MP4 中抽帧。"""
        vr = VideoReader(
            video_path,
            ctx=cpu(0),
            num_threads=1,
        )

        source_fps = float(vr.get_avg_fps())
        sample_fps = float(self.args.sample_fps)

        if sample_fps <= 0:
            raise ValueError(
                f"sample_fps 必须大于 0，当前为 {sample_fps}"
            )

        frame_step = max(
            1,
            int(round(source_fps / sample_fps)),
        )

        frame_indices = list(
            range(0, len(vr), frame_step)
        )

        if not frame_indices:
            raise RuntimeError(
                f"视频中没有可读取的帧：{video_path}"
            )

        video = vr.get_batch(frame_indices).asnumpy()

        print(
            f"video={os.path.basename(video_path)}, "
            f"frames={len(frame_indices)}, "
            f"source_fps={source_fps:.2f}, "
            f"sample_fps={sample_fps}"
        )

        return video

    @staticmethod
    def _extract_question(prompt):
        """
        OVO 的多选 prompt 形如：
        Question: ...
        Options: ...
        只取 Question 部分用于 ReKV 检索。
        """
        match = re.search(
            r"Question:\s*(.*?)\s*Options:",
            prompt,
            flags=re.DOTALL,
        )

        if match:
            return match.group(1).strip()

        return prompt.strip()

    @torch.inference_mode()
    def inference(
        self,
        video_file_name,
        prompt,
        start_time=0,
        end_time=0,
    ):
        video = self._load_video(video_file_name)
        video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        retrieval_question = self._extract_question(prompt)

        input_text = {
            "question": retrieval_question,
            "prompt": self.qa_model.get_prompt(
                prompt,
                mc=True,
            ),
        }

        response = self.qa_model.question_answering(
            input_text,
            max_new_tokens=8,
        )

        response = response.strip()

        # 将 “D). explanation” 或 “D)” 统一转换为 “D”
        if ")" in response:
            before_bracket = response[:response.index(")")].strip()
            candidate = before_bracket[-1:].upper()

            if candidate in {"A", "B", "C", "D"}:
                return candidate

        candidate = response[:1].upper()

        if candidate in {"A", "B", "C", "D"}:
            return candidate

        return response
