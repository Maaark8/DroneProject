from __future__ import annotations

import math

import cv2
import numpy as np

from .types import Point


def largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)

    component_ids = range(1, count)
    best_id = max(component_ids, key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]))
    output = np.zeros_like(mask)
    output[labels == best_id] = 255
    return output


def centerline_from_mask(
    mask: np.ndarray,
    stride: int = 8,
    min_points: int = 8,
) -> list[Point]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    height, _ = mask.shape
    points: list[Point] = []
    row_step = max(1, stride)

    for y in range(height - 1, -1, -row_step):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size < 3:
            continue
        points.append((float(xs.mean()), float(y)))

    if len(points) < min_points:
        return []

    points.reverse()
    xs = np.array([point[0] for point in points], dtype=np.float32)
    ys = np.array([point[1] for point in points], dtype=np.float32)

    if len(points) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        padded = np.pad(xs, (2, 2), mode="edge")
        smoothed_xs = np.convolve(padded, kernel, mode="valid")
    else:
        smoothed_xs = xs

    return [(float(x), float(y)) for x, y in zip(smoothed_xs, ys, strict=True)]


def heading_and_offset(centerline: list[Point], frame_width: int) -> tuple[float | None, float | None]:
    if len(centerline) < 2:
        return None, None

    bottom_points = sorted(centerline, key=lambda point: point[1], reverse=True)[: min(8, len(centerline))]
    ys = np.array([point[1] for point in bottom_points], dtype=np.float32)
    xs = np.array([point[0] for point in bottom_points], dtype=np.float32)

    if np.allclose(ys, ys[0]):
        return None, None

    slope, intercept = np.polyfit(ys, xs, 1)
    y_ref = float(np.max(ys))
    x_ref = float(slope * y_ref + intercept)
    offset_px = x_ref - (frame_width / 2.0)
    heading_rad = math.atan(float(slope))
    return heading_rad, offset_px


def overlay_detection(frame: np.ndarray, mask: np.ndarray, centerline: list[Point], color: tuple[int, int, int]) -> np.ndarray:
    overlay = frame.copy()
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    tinted = np.zeros_like(frame)
    tinted[:, :] = color
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(overlay, 0.4, tinted, 0.6, 0)[mask_bool]

    if centerline:
        polyline = np.array([[int(x), int(y)] for x, y in centerline], dtype=np.int32)
        cv2.polylines(overlay, [polyline], isClosed=False, color=(0, 255, 255), thickness=2)
        x, y = polyline[-1]
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), -1)

    return overlay
