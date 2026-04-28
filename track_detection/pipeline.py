from __future__ import annotations

import time
from pathlib import Path

import cv2

from detectors.factory import create_detector

from .io import ensure_directory, iter_image_files, save_debug_frame, write_result_jsonl
from .types import DetectionResult, FrameInput


def run_on_path(method: str, input_path: Path, output_dir: Path) -> list[DetectionResult]:
    detector = create_detector(method)
    ensure_directory(output_dir)

    if input_path.is_dir():
        return _run_on_images(detector, input_path, output_dir)
    return _run_on_video(detector, input_path, output_dir)


def _run_on_images(detector, input_dir: Path, output_dir: Path) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    debug_dir = output_dir / "debug_frames"
    ensure_directory(debug_dir)

    for frame_id, image_path in enumerate(iter_image_files(input_dir)):
        frame = cv2.imread(str(image_path))
        result = detector.detect(FrameInput(frame=frame, frame_id=frame_id))
        result.metadata["source"] = str(image_path)
        results.append(result)

        if result.debug_frame is not None:
            save_debug_frame(debug_dir / image_path.name, result.debug_frame)

    write_result_jsonl(output_dir / "results.jsonl", results)
    return results


def _run_on_video(detector, video_path: Path, output_dir: Path) -> list[DetectionResult]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open input video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    video_output = output_dir / "debug_overlay.mp4"
    results: list[DetectionResult] = []

    try:
        frame_id = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_s = frame_id / fps
            start = time.perf_counter()
            result = detector.detect(FrameInput(frame=frame, frame_id=frame_id, timestamp_s=timestamp_s))
            result.metadata["source"] = str(video_path)
            result.metadata["runtime_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
            results.append(result)

            if result.debug_frame is not None:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(video_output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (result.debug_frame.shape[1], result.debug_frame.shape[0]),
                    )
                writer.write(result.debug_frame)

            frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    write_result_jsonl(output_dir / "results.jsonl", results)
    return results
