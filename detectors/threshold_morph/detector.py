from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import math
import numpy as np

from detectors.base import BaseTrackDetector
from track_detection.geometry import overlay_detection
from track_detection.preprocessing import preprocess_frame
from track_detection.types import DetectionResult, FrameInput, PreprocessConfig


@dataclass(slots=True)
class ThresholdMorphConfig(PreprocessConfig):
    # Color thresholds tuned for light beech wood.
    hsv_lower: tuple[int, int, int] = (5, 20, 80)
    hsv_upper: tuple[int, int, int] = (30, 200, 240)
    lab_a_lower: int = 128
    lab_a_upper: int = 150
    # Morphology kernels.
    close_kernel: int = 9
    open_kernel: int = 5
    # A track is long and thin. Reject blobs whose long/short axis ratio is
    # below this. A wooden floor patch tends to be ~1:1.
    min_aspect_ratio: float = 1.5
    # Reject blobs smaller than this many pixels of the working frame.
    min_blob_pixels: int = 800


class ThresholdMorphDetector(BaseTrackDetector):
    method_name = "threshold_morph"
    overlay_color = (60, 180, 255)

    def __init__(self, config: ThresholdMorphConfig | None = None) -> None:
        super().__init__(config or ThresholdMorphConfig())

    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        hsv = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(working_frame, cv2.COLOR_BGR2LAB)

        hsv_mask = cv2.inRange(hsv, self.config.hsv_lower, self.config.hsv_upper)
        a_channel = lab[:, :, 1]
        warm_mask = cv2.inRange(a_channel, self.config.lab_a_lower, self.config.lab_a_upper)
        mask = cv2.bitwise_and(hsv_mask, warm_mask)

        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.close_kernel, self.config.close_kernel))
        ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.open_kernel, self.config.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ok)

        mask, shape_info = _largest_elongated_component(
            mask,
            min_pixels=self.config.min_blob_pixels,
            min_aspect_ratio=self.config.min_aspect_ratio,
        )
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        meta: dict[str, Any] = {"coverage_ratio": round(coverage, 4)}
        meta.update(shape_info)
        return mask, meta

    def detect(self, frame_input: FrameInput) -> DetectionResult:
        prepared = preprocess_frame(frame_input.frame, self.config)
        working_mask, metadata = self._detect_mask(prepared.frame)

        if np.count_nonzero(working_mask) == 0:
            full_mask = np.zeros(frame_input.frame.shape[:2], dtype=np.uint8)
            debug_frame = overlay_detection(frame_input.frame, full_mask, [], self.overlay_color)
            metadata["mask_pixels"] = 0
            metadata["frame_id"] = frame_input.frame_id
            metadata["orientation"] = None
            return DetectionResult(
                method=self.method_name, centerline=[],
                heading_rad=None, lateral_offset_px=None,
                confidence=0.0, valid=False,
                debug_frame=debug_frame, metadata=metadata,
            )

        is_vertical = _track_is_mostly_vertical(working_mask)
        if is_vertical:
            centerline = _centerline_rowwise(
                working_mask,
                stride=self.config.centerline_stride,
                min_points=self.config.min_centerline_points,
            )
        else:
            centerline = _centerline_columnwise(
                working_mask,
                stride=self.config.centerline_stride,
                min_points=self.config.min_centerline_points,
            )

        mapped_centerline = [prepared.point_to_original(point) for point in centerline]
        heading_rad, lateral_offset_px = _heading_and_offset_oriented(
            mapped_centerline, frame_input.frame.shape[1], is_vertical
        )
        valid = len(mapped_centerline) >= self.config.min_centerline_points
        confidence = self._compute_confidence(working_mask, mapped_centerline)

        mask_orig = _resize_mask_to_original(
            working_mask, frame_input.frame.shape[1], prepared.roi_height
        )
        full_mask = np.zeros(frame_input.frame.shape[:2], dtype=np.uint8)
        full_mask[prepared.roi_top_px:prepared.roi_bottom_px, :] = mask_orig
        debug_frame = overlay_detection(
            frame_input.frame, full_mask, mapped_centerline, self.overlay_color
        )
        metadata["mask_pixels"] = int(np.count_nonzero(working_mask))
        metadata["frame_id"] = frame_input.frame_id
        metadata["orientation"] = "vertical" if is_vertical else "horizontal"

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


# --- helpers ---------------------------------------------------------------

def _largest_elongated_component(
    mask: np.ndarray, min_pixels: int, min_aspect_ratio: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    info: dict[str, Any] = {"rejected_reason": None, "aspect_ratio": 0.0, "best_area": 0}
    if count <= 1:
        info["rejected_reason"] = "no_components"
        return np.zeros_like(mask), info
    component_ids = range(1, count)
    best_id = max(component_ids, key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]))
    area = int(stats[best_id, cv2.CC_STAT_AREA])
    info["best_area"] = area
    if area < min_pixels:
        info["rejected_reason"] = "too_small"
        return np.zeros_like(mask), info
    component = np.zeros_like(mask)
    component[labels == best_id] = 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        info["rejected_reason"] = "no_contour"
        return np.zeros_like(mask), info
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 5:
        return component, info
    (_, _), (w, h), _ = cv2.minAreaRect(biggest)
    if w < 1 or h < 1:
        info["rejected_reason"] = "degenerate_rect"
        return np.zeros_like(mask), info
    long_side, short_side = max(w, h), min(w, h)
    aspect = float(long_side / short_side)
    info["aspect_ratio"] = round(aspect, 2)
    if aspect < min_aspect_ratio:
        info["rejected_reason"] = "not_elongated"
        return np.zeros_like(mask), info
    return component, info


def _track_is_mostly_vertical(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 5:
        return True
    (_, _), (w, h), angle = cv2.minAreaRect(biggest)
    long_angle = angle if w >= h else angle + 90
    long_angle = long_angle % 180
    return 45 <= long_angle < 135


def _centerline_rowwise(mask: np.ndarray, stride: int, min_points: int) -> list[tuple[float, float]]:
    height, _ = mask.shape
    points: list[tuple[float, float]] = []
    step = max(1, stride)
    for y in range(height - 1, -1, -step):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size < 3:
            continue
        points.append((float(xs.mean()), float(y)))
    if len(points) < min_points:
        return []
    points.reverse()
    return _smooth(points, axis=0)


def _centerline_columnwise(mask: np.ndarray, stride: int, min_points: int) -> list[tuple[float, float]]:
    _, width = mask.shape
    points: list[tuple[float, float]] = []
    step = max(1, stride)
    for x in range(0, width, step):
        ys = np.flatnonzero(mask[:, x] > 0)
        if ys.size < 3:
            continue
        points.append((float(x), float(ys.mean())))
    if len(points) < min_points:
        return []
    return _smooth(points, axis=1)


def _smooth(points: list[tuple[float, float]], axis: int) -> list[tuple[float, float]]:
    if len(points) < 5:
        return points
    coords = np.array(points, dtype=np.float32)
    target = coords[:, axis]
    kernel = np.ones(5, dtype=np.float32) / 5.0
    padded = np.pad(target, (2, 2), mode="edge")
    coords[:, axis] = np.convolve(padded, kernel, mode="valid")
    return [(float(x), float(y)) for x, y in coords]


def _heading_and_offset_oriented(
    centerline: list[tuple[float, float]], frame_width: int, is_vertical: bool
) -> tuple[float | None, float | None]:
    if len(centerline) < 2:
        return None, None
    pts = np.array(centerline, dtype=np.float32)
    if is_vertical:
        bottom = pts[pts[:, 1].argsort()][-min(8, len(pts)):]
        ys = bottom[:, 1]; xs = bottom[:, 0]
        if np.allclose(ys, ys[0]):
            return None, None
        slope, intercept = np.polyfit(ys, xs, 1)
        y_ref = float(ys.max())
        x_ref = float(slope * y_ref + intercept)
        offset_px = x_ref - (frame_width / 2.0)
        heading_rad = math.atan(float(slope))
        return heading_rad, offset_px
    right = pts[pts[:, 0].argsort()][-min(8, len(pts)):]
    xs = right[:, 0]; ys = right[:, 1]
    if np.allclose(xs, xs[0]):
        return None, None
    slope, _ = np.polyfit(xs, ys, 1)
    offset_px = float(ys.mean() - pts[:, 1].mean())
    heading_rad = math.atan(float(slope))
    return heading_rad, offset_px


def _resize_mask_to_original(mask: np.ndarray, original_width: int, roi_height: int) -> np.ndarray:
    if mask.shape == (roi_height, original_width):
        return mask.astype(np.uint8)
    resized = cv2.resize(mask, (original_width, roi_height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)