from __future__ import annotations

from pathlib import Path

import cv2

from .io import ensure_directory


def extract_frames(input_video: Path, output_dir: Path, every_n: int = 5) -> int:
    ensure_directory(output_dir)
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open input video: {input_video}")

    saved = 0
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id % every_n == 0:
                output_path = output_dir / f"frame_{frame_id:06d}.png"
                cv2.imwrite(str(output_path), frame)
                saved += 1
            frame_id += 1
    finally:
        capture.release()

    return saved
