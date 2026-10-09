"""Shared deterministic video sampling for offline benchmark comparisons."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


def sample_video_frames(
    video_path: str,
    num_frames: int,
    max_side: Optional[int] = None,
) -> Dict[str, Any]:
    """Uniformly sample the same RGB frames for all offline backends.

    The index rule intentionally matches Qwen's ``torch.linspace(...).round``.
    Frames are returned as PIL images so FullKV and HERMES consume identical
    decoded RGB content before backend-specific preprocessing.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames,
        ).round().astype(int).tolist()
        frames: List[Image.Image] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {index} from {video_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            if max_side is not None and max(image.size) > max_side:
                scale = max_side / max(image.size)
                image = image.resize(
                    (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            frames.append(image)
    finally:
        cap.release()

    return {
        "frames": frames,
        "frame_indices": indices,
        "frame_timestamps": [
            round(index / source_fps, 6) for index in indices
        ],
        "source_fps": source_fps,
        "total_frames": total_frames,
        "video_path": str(Path(video_path)),
        "sample_strategy": "uniform_linspace",
    }
