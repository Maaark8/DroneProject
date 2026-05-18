from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from track_detection.geometry import centerline_from_mask, heading_and_offset, overlay_detection
from track_detection.preprocessing import preprocess_frame
from track_detection.types import DetectionResult, FrameInput, PreprocessConfig


@dataclass(slots=True)
class DetectorContext:
    frame_input: FrameInput
    prepared_frame: Any
    working_mask: np.ndarray
    centerline: list[tuple[float, float]]


class BaseTrackDetector(ABC):
    method_name = "base"
    overlay_color = (0, 255, 0)

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def detect(self, frame_input: FrameInput) -> DetectionResult:
        prepared = preprocess_frame(frame_input.frame, self.config)
        working_mask, metadata = self._detect_mask(prepared.frame)
        centerline = centerline_from_mask(
            working_mask,
            stride=self.config.centerline_stride,
            min_points=self.config.min_centerline_points,
        )
        mapped_centerline = [prepared.point_to_original(point) for point in centerline]
        heading_rad, lateral_offset_px = heading_and_offset(mapped_centerline, frame_input.frame.shape[1])
        valid = len(mapped_centerline) >= self.config.min_centerline_points
        confidence = self._compute_confidence(working_mask, mapped_centerline)

        mask_original = _resize_mask_to_original(working_mask, frame_input.frame.shape[1], prepared.roi_height)
        full_mask = np.zeros(frame_input.frame.shape[:2], dtype=np.uint8)
        full_mask[prepared.roi_top_px:prepared.roi_bottom_px, :] = mask_original
        debug_frame = overlay_detection(frame_input.frame, full_mask, mapped_centerline, self.overlay_color)
        metadata["mask_pixels"] = int(np.count_nonzero(working_mask))
        metadata["frame_id"] = frame_input.frame_id

        return DetectionResult(
            method=self.method_name,
            centerline=mapped_centerline,
            heading_rad=heading_rad,
            lateral_offset_px=lateral_offset_px,
            confidence=confidence,
            valid=valid,
            debug_frame=debug_frame,
            metadata=metadata,
        )

    @abstractmethod
    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def _compute_confidence(self, mask: np.ndarray, centerline: list[tuple[float, float]]) -> float:
        mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        point_ratio = min(1.0, len(centerline) / float(max(self.config.min_centerline_points, 1)))
        confidence = min(1.0, 0.5 * point_ratio + 2.0 * mask_ratio)
        return round(confidence, 3)


def _resize_mask_to_original(mask: np.ndarray, original_width: int, roi_height: int) -> np.ndarray:
    if mask.shape == (roi_height, original_width):
        return mask.astype(np.uint8)
    resized = cv2.resize(mask, (original_width, roi_height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)
