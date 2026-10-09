"""
Qwen2.5-VL-Instruct + HERMES adapter for OVO-Bench.

Forward protocol:
- encode the complete cumulative video prefix into HERMES visual memory;
- do not ask intermediate proactive questions;
- ask the official OVO Forward prompt once at the end;
- do not append "I must reply.";
- do not use first-token constrained decoding;
- normalize only after free-form generation.
"""

import gc
import os

import torch

from models.MMDuet2_HERMES import (
    EvalMMDuet2Hermes,
    iter_frame_chunks,
    mark_last_chunk,
    normalize_forward_reply,
)


class EvalQwenHermes(EvalMMDuet2Hermes):
    """Qwen2.5-VL-Instruct + HERMES OVO adapter."""

    def __init__(self, args):
        super().__init__(args)

        # Neutral instruction-following system prompt for base Qwen.
        self.system_prompt = "You are a helpful assistant."
        self.forward_system_prompt = self.system_prompt

        model_device = next(self.qa_model.parameters()).device

        init_text = (
            "<|im_start|>system\n"
            + self.system_prompt
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
        )

        init_ids = (
            self.qa_processor.tokenizer(
                init_text,
                return_tensors="pt",
            )
            .input_ids
            .to(model_device)
        )

        self._init_prompt_ids = init_ids
        self._forward_init_prompt_ids = init_ids
        self.qa_model.init_prompt_ids = self._init_prompt_ids

        print(
            "Qwen+HERMES baseline checkpoint:",
            args.model_path,
        )
        print(
            "Qwen+HERMES Forward protocol:",
            "encode prefix -> ask official OVO prompt once",
        )

    @torch.inference_mode()
    def inference_forward(
        self,
        video_file_name: str,
        prompt: str,
        realtime: float,
        task=None,
        sample_id=None,
        test_index=None,
        **kwargs,
    ) -> str:
        if task not in self.FORWARD_TASKS:
            raise NotImplementedError(
                "Qwen_HERMES Forward does not support: "
                f"{task!r}"
            )

        print(
            f"\n[QWEN HERMES FORWARD START] "
            f"id={sample_id}, "
            f"index={test_index}, "
            f"task={task}, "
            f"realtime={realtime:.3f}s, "
            f"video={video_file_name}"
        )

        self._reset_stream(forward=True)
        turn_count = 0

        try:
            chunks = iter_frame_chunks(
                video_path=video_file_name,
                sample_fps=self.sample_fps,
                frames_per_turn=self.frames_per_turn,
                max_side=self.max_side,
            )

            # Encode the whole prefix. No intermediate _ask().
            for frames, is_last in mark_last_chunk(chunks):
                turn_count += 1
                self._encode_frames(frames)

                if os.environ.get(
                    "HERMES_INPUT_DEBUG",
                    "0",
                ) == "1":
                    print(
                        f"[QWEN HERMES PREFILL] "
                        f"id={sample_id}, "
                        f"index={test_index}, "
                        f"turn={turn_count}, "
                        f"is_last={is_last}, "
                        f"kv={self._cache_length_summary()}"
                    )

            if turn_count == 0:
                raise RuntimeError(
                    f"Video has no valid frames: "
                    f"{video_file_name}"
                )

            # Ask official OVO prompt once.
            raw_reply = self._ask(
                prompt,
                max_new_tokens=64,
                mc=False,
                must_reply=False,
                allowed_first_token_ids=None,
            )

            response = normalize_forward_reply(
                reply=raw_reply,
                task=task,
            )

            print(
                f"[QWEN HERMES FORWARD RESULT] "
                f"id={sample_id}, "
                f"index={test_index}, "
                f"task={task}, "
                f"realtime={realtime:.3f}s, "
                f"turns={turn_count}, "
                f"kv={self._cache_length_summary()}, "
                f"raw_answer={raw_reply!r}, "
                f"answer={response!r}"
            )

            return response

        except torch.cuda.OutOfMemoryError:
            print(
                f"[QWEN HERMES FORWARD OOM] "
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
