from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detectors.base import BaseTrackDetector
from track_detection.geometry import largest_component
from track_detection.types import PreprocessConfig


@dataclass(slots=True)
class EdgeGeometryConfig(PreprocessConfig):
    canny_low: int = 60
    canny_high: int = 160
    dilate_kernel: int = 5
    close_kernel: int = 11
    min_contour_area: int = 1200


class EdgeGeometryDetector(BaseTrackDetector):
    method_name = "edge_geometry"
    overlay_color = (90, 255, 120)

    def __init__(self, config: EdgeGeometryConfig | None = None) -> None:
        super().__init__(config or EdgeGeometryConfig())

    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        normalized = cv2.equalizeHist(gray)
        edges = cv2.Canny(normalized, self.config.canny_low, self.config.canny_high)
        _, silhouette = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.dilate_kernel, self.config.dilate_kernel))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.close_kernel, self.config.close_kernel))
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)
        filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, close_kernel)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, dilate_kernel)
        filled = cv2.bitwise_or(filled, silhouette)

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_mask = np.zeros_like(gray)
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_distance = abs((x + w / 2.0) - (gray.shape[1] / 2.0))
            vertical_span = h / float(gray.shape[0])
            score = area * 0.001 + vertical_span * 4.0 - center_distance * 0.01

            if score <= best_score:
                continue

            candidate_mask = np.zeros_like(gray)
            cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=cv2.FILLED)
            best_mask = candidate_mask
            best_score = score

        best_mask = largest_component(best_mask)
        coverage = float(np.count_nonzero(best_mask)) / float(best_mask.size)
        return best_mask, {"coverage_ratio": round(coverage, 4), "contour_score": round(max(best_score, 0.0), 3)}
