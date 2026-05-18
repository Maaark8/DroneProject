from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from detectors.factory import create_detector

from .io import ensure_directory, iter_image_files, result_to_payload, save_debug_frame, write_result_jsonl
from .live_capture import normalize_capture_source, open_live_capture
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


def run_live_camera(
    method: str,
    camera_index: int = 0,
    source: str | int | None = None,
    output_dir: Path | None = None,
    display: bool = True,
    max_frames: int | None = None,
) -> None:
    detector = create_detector(method)
    capture_source = normalize_capture_source(source if source is not None else camera_index)
    capture = open_live_capture(capture_source)
    if not capture.isOpened():
        raise ValueError(
            f"Unable to open live video source: {capture_source!r}. "
            "Use --source with a camera index, video path, or IP camera URL."
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    result_handle = None
    writer = None
    started = time.perf_counter()

    if output_dir is not None:
        ensure_directory(output_dir)
        result_handle = (output_dir / "results.jsonl").open("w", encoding="utf-8")

    try:
        frame_id = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s = time.perf_counter() - started
            start = time.perf_counter()
            result = detector.detect(FrameInput(frame=frame, frame_id=frame_id, timestamp_s=timestamp_s))
            result.metadata["source"] = f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source)
            result.metadata["runtime_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
            result.metadata["capture_fps"] = round(float(fps), 3)

            if result_handle is not None:
                result_handle.write(json.dumps(result_to_payload(result)) + "\n")
                result_handle.flush()

            if output_dir is not None and result.debug_frame is not None:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output_dir / "debug_overlay.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (result.debug_frame.shape[1], result.debug_frame.shape[0]),
                    )
                writer.write(result.debug_frame)

            if display and result.debug_frame is not None:
                cv2.imshow(f"{method} live", result.debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_id += 1
            if max_frames is not None and frame_id >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if result_handle is not None:
            result_handle.close()
        if display:
            cv2.destroyAllWindows()
