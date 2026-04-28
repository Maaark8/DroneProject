from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import PreprocessConfig, Point


@dataclass(slots=True)
class PreprocessedFrame:
    frame: np.ndarray
    original_shape: tuple[int, int, int]
    roi_top_px: int
    roi_bottom_px: int
    working_width: int
    working_height: int

    @property
    def roi_height(self) -> int:
        return self.roi_bottom_px - self.roi_top_px

    def point_to_original(self, point: Point) -> Point:
        x, y = point
        height_scale = self.roi_height / float(self.working_height)
        width_scale = self.original_shape[1] / float(self.working_width)
        return (
            x * width_scale,
            self.roi_top_px + y * height_scale,
        )


def preprocess_frame(frame: np.ndarray, config: PreprocessConfig) -> PreprocessedFrame:
    height, width = frame.shape[:2]
    roi_top_px = max(0, min(height - 1, int(height * config.roi_top)))
    roi_bottom_px = max(roi_top_px + 1, min(height, int(height * config.roi_bottom)))
    cropped = frame[roi_top_px:roi_bottom_px, :]

    resized = cv2.resize(
        cropped,
        (config.working_width, config.working_height),
        interpolation=cv2.INTER_AREA,
    )

    kernel = _normalized_kernel(config.blur_kernel)
    if kernel > 1:
        resized = cv2.GaussianBlur(resized, (kernel, kernel), 0)

    return PreprocessedFrame(
        frame=resized,
        original_shape=frame.shape,
        roi_top_px=roi_top_px,
        roi_bottom_px=roi_bottom_px,
        working_width=config.working_width,
        working_height=config.working_height,
    )


def _normalized_kernel(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1
