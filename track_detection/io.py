from __future__ import annotations

import json
from pathlib import Path

import cv2

from .types import DetectionResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_image_files(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_result_jsonl(path: Path, results: list[DetectionResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            payload = {
                "method": result.method,
                "centerline": result.centerline,
                "heading_rad": result.heading_rad,
                "lateral_offset_px": result.lateral_offset_px,
                "confidence": result.confidence,
                "valid": result.valid,
                "metadata": result.metadata,
            }
            handle.write(json.dumps(payload) + "\n")


def save_debug_frame(path: Path, frame) -> None:
    cv2.imwrite(str(path), frame)
