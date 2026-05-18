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
    bright_percentile: float = 88.0
    bright_open_kernel: int = 3
    bright_close_kernel: int = 7
    max_mask_area_ratio: float = 0.28
    max_border_touches: int = 2
    width_outlier_scale: float = 1.8
    width_outlier_margin_px: int = 12


class EdgeGeometryDetector(BaseTrackDetector):
    method_name = "edge_geometry"
    overlay_color = (90, 255, 120)

    def __init__(self, config: EdgeGeometryConfig | None = None) -> None:
        super().__init__(config or EdgeGeometryConfig())

    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        edge_mask, edge_meta = self._edge_candidate_mask(working_frame)
        bright_mask, bright_meta = self._bright_candidate_mask(working_frame)

        candidates = [
            ("edge", edge_mask, edge_meta),
            ("bright", bright_mask, bright_meta),
        ]
        best_name, best_mask, best_meta = max(candidates, key=lambda item: float(item[2]["quality_score"]))
        coverage = float(np.count_nonzero(best_mask)) / float(best_mask.size)
        meta: dict[str, float] = {
            "coverage_ratio": round(coverage, 4),
            "selected_candidate": best_name,
        }
        for prefix, _, candidate_meta in candidates:
            meta[f"{prefix}_quality_score"] = round(float(candidate_meta["quality_score"]), 3)
            meta[f"{prefix}_area_ratio"] = round(float(candidate_meta["area_ratio"]), 4)
            meta[f"{prefix}_aspect_ratio"] = round(float(candidate_meta["aspect_ratio"]), 3)
            meta[f"{prefix}_border_touches"] = float(candidate_meta["border_touches"])
        meta.update(best_meta)
        return best_mask, meta

    def _edge_candidate_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
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

        return self._best_contour_mask(
            filled,
            min_area=self.config.min_contour_area,
            quality_bias="edge",
        )

    def _bright_candidate_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        threshold = int(np.percentile(gray, self.config.bright_percentile))
        mask = cv2.inRange(gray, threshold, 255)
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.bright_open_kernel, self.config.bright_open_kernel),
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.bright_close_kernel, self.config.bright_close_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask, meta = self._best_contour_mask(
            mask,
            min_area=max(self.config.min_contour_area * 0.15, 200.0),
            quality_bias="bright",
        )
        if np.count_nonzero(mask) == 0:
            meta["threshold_value"] = float(threshold)
            return mask, meta
        pruned = self._prune_wide_rows(mask)
        pruned = largest_component(pruned)
        metrics = _mask_metrics(pruned)
        meta.update(
            {
                "threshold_value": float(threshold),
                "quality_score": _candidate_score(metrics, "bright", self.config),
                "area_ratio": metrics["area_ratio"],
                "aspect_ratio": metrics["aspect_ratio"],
                "border_touches": metrics["border_touches"],
            }
        )
        return pruned, meta

    def _best_contour_mask(
        self,
        mask: np.ndarray,
        min_area: float,
        quality_bias: str,
    ) -> tuple[np.ndarray, dict[str, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_mask = np.zeros_like(mask)
        best_meta = {
            "quality_score": -1e9,
            "area_ratio": 0.0,
            "aspect_ratio": 0.0,
            "border_touches": 4.0,
            "raw_candidate_score": 0.0,
        }

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(min_area):
                continue
            candidate_mask = np.zeros_like(mask)
            cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=cv2.FILLED)
            metrics = _mask_metrics(candidate_mask)
            raw_score = _candidate_score(metrics, quality_bias, self.config)
            if raw_score <= best_meta["quality_score"]:
                continue
            best_mask = candidate_mask
            best_meta = {
                "quality_score": raw_score,
                "area_ratio": metrics["area_ratio"],
                "aspect_ratio": metrics["aspect_ratio"],
                "border_touches": metrics["border_touches"],
                "raw_candidate_score": raw_score,
            }

        if np.count_nonzero(best_mask) > 0:
            best_mask = largest_component(best_mask)
            metrics = _mask_metrics(best_mask)
            best_meta.update(
                {
                    "quality_score": _candidate_score(metrics, quality_bias, self.config),
                    "area_ratio": metrics["area_ratio"],
                    "aspect_ratio": metrics["aspect_ratio"],
                    "border_touches": metrics["border_touches"],
                }
            )
        return best_mask, best_meta

    def _prune_wide_rows(self, mask: np.ndarray) -> np.ndarray:
        row_widths = np.count_nonzero(mask > 0, axis=1)
        positive = row_widths[row_widths > 0]
        if positive.size == 0:
            return mask
        median_width = float(np.median(positive))
        limit = max(
            median_width * float(self.config.width_outlier_scale),
            median_width + float(self.config.width_outlier_margin_px),
        )
        pruned = mask.copy()
        for row_index, width in enumerate(row_widths):
            if float(width) > limit:
                pruned[row_index, :] = 0
        return pruned


def _mask_metrics(mask: np.ndarray) -> dict[str, float]:
    height, width = mask.shape[:2]
    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area_ratio": 0.0,
            "aspect_ratio": 0.0,
            "border_touches": 4.0,
            "long_side_ratio": 0.0,
            "center_distance_ratio": 1.0,
        }
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    long_side = float(max(rw, rh))
    short_side = float(max(min(rw, rh), 1e-6))
    aspect_ratio = long_side / short_side
    border_touches = float(int(x <= 1) + int(y <= 1) + int(x + w >= width - 1) + int(y + h >= height - 1))
    center_x = x + (w / 2.0)
    center_distance_ratio = abs(center_x - (width / 2.0)) / float(width / 2.0)
    return {
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "border_touches": border_touches,
        "long_side_ratio": long_side / float(max(height, width)),
        "center_distance_ratio": center_distance_ratio,
    }


def _candidate_score(metrics: dict[str, float], quality_bias: str, config: EdgeGeometryConfig) -> float:
    area_ratio = metrics["area_ratio"]
    aspect_ratio = min(metrics["aspect_ratio"], 8.0)
    border_touches = metrics["border_touches"]
    long_side_ratio = metrics["long_side_ratio"]
    center_distance_ratio = metrics["center_distance_ratio"]
    oversize_penalty = max(0.0, area_ratio - float(config.max_mask_area_ratio)) * 140.0
    border_penalty = max(0.0, border_touches - float(config.max_border_touches)) * 18.0

    if quality_bias == "bright":
        return (
            (aspect_ratio * 9.0)
            + (long_side_ratio * 18.0)
            + (min(area_ratio, 0.18) * 45.0)
            - (center_distance_ratio * 3.0)
            - oversize_penalty
            - border_penalty
        )

    return (
        (aspect_ratio * 3.0)
        + (long_side_ratio * 8.0)
        + (min(area_ratio, 0.2) * 12.0)
        - (center_distance_ratio * 4.0)
        - oversize_penalty
        - (border_touches * 10.0)
    )
