from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detectors.base import BaseTrackDetector
from track_detection.geometry import largest_component
from track_detection.types import PreprocessConfig


@dataclass(slots=True)
class ThresholdMorphConfig(PreprocessConfig):
    hsv_lower: tuple[int, int, int] = (5, 40, 40)
    hsv_upper: tuple[int, int, int] = (35, 255, 230)
    lab_a_lower: int = 120
    lab_a_upper: int = 170
    close_kernel: int = 9
    open_kernel: int = 5


class ThresholdMorphDetector(BaseTrackDetector):
    method_name = "threshold_morph"
    overlay_color = (60, 180, 255)

    def __init__(self, config: ThresholdMorphConfig | None = None) -> None:
        super().__init__(config or ThresholdMorphConfig())

    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        hsv = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(working_frame, cv2.COLOR_BGR2LAB)

        hsv_mask = cv2.inRange(hsv, self.config.hsv_lower, self.config.hsv_upper)
        a_channel = lab[:, :, 1]
        warm_mask = cv2.inRange(a_channel, self.config.lab_a_lower, self.config.lab_a_upper)
        mask = cv2.bitwise_and(hsv_mask, warm_mask)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.close_kernel, self.config.close_kernel))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.open_kernel, self.config.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = largest_component(mask)

        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        return mask, {"coverage_ratio": round(coverage, 4)}
