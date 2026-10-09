"""
Qwen2.5-VL-Instruct + HERMES adapter for OVO-Bench, with intermediate ASK.

Controlled difference from Qwen_HERMES.py:
- same Qwen checkpoint;
- same HERMES image-aligned visual-KV path;
- same neutral system prompt;
- same official OVO Forward prompt;
- no MMDuet2 "NO REPLY" protocol;
- no "I must reply.";
- no first-token constrained decoding.

Only difference:
- Qwen_HERMES.py: encode all chunks, ask once at the final checkpoint.
- Qwen_HERMES_Ask.py: after every chunk, ask the official OVO prompt once.
  HERMES question_answering() truncates transient QA KV, while conv_history
  remains available to guide later HERMES query prediction/compression.
"""

import gc
import os

import torch

from models.Qwen_HERMES import EvalQwenHermes
from models.MMDuet2_HERMES import (
    iter_frame_chunks,
    mark_last_chunk,
    normalize_forward_reply,
)


class EvalQwenHermesAsk(EvalQwenHermes):
    """Qwen2.5-VL-Instruct + HERMES with intermediate per-chunk QA."""

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
                "Qwen_HERMES_Ask Forward does not support: "
                f"{task!r}"
            )

        print(
            f"\n[QWEN HERMES ASK FORWARD START] "
            f"id={sample_id}, "
            f"index={test_index}, "
            f"task={task}, "
            f"realtime={realtime:.3f}s, "
            f"video={video_file_name}"
        )

        self._reset_stream(forward=True)

        turn_count = 0
        final_raw_reply = None

        try:
            chunks = iter_frame_chunks(
                video_path=video_file_name,
                sample_fps=self.sample_fps,
                frames_per_turn=self.frames_per_turn,
                max_side=self.max_side,
            )

            for frames, is_last in mark_last_chunk(chunks):
                turn_count += 1

                # 1) Add new visual evidence to persistent HERMES memory.
                self._encode_frames(frames)

                # 2) ASK after every chunk using exactly the official
                #    OVO Forward prompt. No MMDuet2-specific control.
                raw_reply = self._ask(
                    prompt,
                    max_new_tokens=64,
                    mc=False,
                    must_reply=False,
                    allowed_first_token_ids=None,
                )

                print(
                    f"[QWEN HERMES ASK TURN] "
                    f"id={sample_id}, "
                    f"index={test_index}, "
                    f"turn={turn_count}, "
                    f"is_last={is_last}, "
                    f"raw_reply={raw_reply!r}, "
                    f"kv={self._cache_length_summary()}"
                )

                if is_last:
                    final_raw_reply = raw_reply
                    break

            if turn_count == 0:
                raise RuntimeError(
                    f"Video has no valid frames: {video_file_name}"
                )

            response = normalize_forward_reply(
                reply=final_raw_reply,
                task=task,
            )

            print(
                f"[QWEN HERMES ASK FORWARD RESULT] "
                f"id={sample_id}, "
                f"index={test_index}, "
                f"task={task}, "
                f"realtime={realtime:.3f}s, "
                f"turns={turn_count}, "
                f"kv={self._cache_length_summary()}, "
                f"raw_answer={final_raw_reply!r}, "
                f"answer={response!r}"
            )

            return response

        except torch.cuda.OutOfMemoryError:
            print(
                f"[QWEN HERMES ASK FORWARD OOM] "
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
