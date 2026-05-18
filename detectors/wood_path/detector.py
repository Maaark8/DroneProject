"""Topology-based wooden-track tracer.

Segments the wood, thins the blob to a 1px medial skeleton, then walks the
longest geodesic through the skeleton graph. This follows curves, S-bends and
U-turns faithfully -- where a per-row mean centerline collapses both arms of a
curve into the gap between them -- and reports what route the track takes
(shape, net turn, curvature, endpoints, junctions). Output uses the shared
DetectionResult contract so it feeds the mission/follow pipeline directly.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from detectors.base import BaseTrackDetector
from track_detection.geometry import heading_and_offset, overlay_detection
from track_detection.types import DetectionResult, FrameInput, PreprocessConfig

Pixel = tuple[int, int]


@dataclass(slots=True)
class _Prepared:
    """Uniform-scale working frame plus the transform back to pixels.

    The shared `preprocess_frame` resizes to a fixed 320x240, which
    anisotropically squashes tall/wide photos -- a long vertical track piece
    becomes a near-square blob (failing the elongation gate) and every
    curvature angle is warped. Scaling by a single factor avoids both.
    """

    frame: np.ndarray
    scale_x: float
    scale_y: float
    roi_top_px: int
    roi_bottom_px: int

    @property
    def roi_height(self) -> int:
        return self.roi_bottom_px - self.roi_top_px

    def point_to_original(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (x * self.scale_x, self.roi_top_px + y * self.scale_y)


def _prepare(frame: np.ndarray, config: "WoodPathConfig") -> _Prepared:
    height, width = frame.shape[:2]
    roi_top_px = max(0, min(height - 1, int(height * config.roi_top)))
    roi_bottom_px = max(roi_top_px + 1, min(height, int(height * config.roi_bottom)))
    crop = frame[roi_top_px:roi_bottom_px, :]
    crop_h, crop_w = crop.shape[:2]

    scale = min(1.0, float(config.working_long_side) / float(max(crop_h, crop_w)))
    target_w = max(1, int(round(crop_w * scale)))
    target_h = max(1, int(round(crop_h * scale)))
    work = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    kernel = config.blur_kernel if config.blur_kernel % 2 == 1 else config.blur_kernel + 1
    if kernel > 1:
        work = cv2.GaussianBlur(work, (kernel, kernel), 0)

    return _Prepared(
        frame=work,
        scale_x=crop_w / float(target_w),
        scale_y=crop_h / float(target_h),
        roi_top_px=roi_top_px,
        roi_bottom_px=roi_bottom_px,
    )


@dataclass(slots=True)
class WoodPathConfig(PreprocessConfig):
    # Trace the whole frame: the track can fill the image in a top-down view,
    # so unlike the lane-style detectors we do not crop the upper ROI away.
    roi_top: float = 0.0
    roi_bottom: float = 1.0
    # Beech wood is warm and moderately saturated against a near-gray desk.
    hsv_lower: tuple[int, int, int] = (5, 25, 70)
    hsv_upper: tuple[int, int, int] = (35, 220, 245)
    lab_a_lower: int = 128
    lab_a_upper: int = 165
    # Hard floor for the b* split so a near-uniform frame cannot threshold
    # the neutral desk as "wood"; `bg_margin` is how far above the measured
    # background b* the wood must sit.
    lab_b_min: int = 132
    lab_b_bg_margin: int = 5
    min_saturation: int = 22
    close_kernel: int = 13
    open_kernel: int = 5
    min_blob_pixels: int = 600
    min_aspect_ratio: float = 1.4
    # Path post-processing.
    path_stride: int = 3
    smooth_window: int = 7
    min_path_points: int = 12
    straight_turn_deg: float = 20.0
    max_thinning_iterations: int = 200
    # Longest side of the uniform-scaled working frame.
    working_long_side: int = 512


class WoodPathDetector(BaseTrackDetector):
    method_name = "wood_path"
    overlay_color = (255, 120, 60)

    def __init__(self, config: WoodPathConfig | None = None) -> None:
        super().__init__(config or WoodPathConfig())

    def detect(self, frame_input: FrameInput) -> DetectionResult:
        prepared = _prepare(frame_input.frame, self.config)
        mask, mask_meta = self._wood_mask(prepared.frame)
        metadata: dict[str, Any] = {
            "detection_type": "skeleton_path",
            "frame_id": frame_input.frame_id,
            "timestamp_s": frame_input.timestamp_s,
            "frame_size": {
                "width": int(frame_input.frame.shape[1]),
                "height": int(frame_input.frame.shape[0]),
            },
        }
        metadata.update(mask_meta)

        skeleton = (
            _zhang_suen_thinning(mask, self.config.max_thinning_iterations)
            if np.count_nonzero(mask)
            else np.zeros_like(mask)
        )
        endpoints, junctions = _skeleton_nodes(skeleton)
        path_px = _longest_geodesic(skeleton)

        if len(path_px) < self.config.min_path_points:
            return self._empty_result(frame_input, metadata, "path_too_short")

        ordered = _order_bottom_first(path_px)
        sampled = ordered[:: max(1, self.config.path_stride)]
        if sampled[-1] != ordered[-1]:
            sampled.append(ordered[-1])
        working_centerline = _smooth_polyline(
            [(float(x), float(y)) for y, x in sampled], self.config.smooth_window
        )
        centerline = [prepared.point_to_original(point) for point in working_centerline]

        heading_rad, lateral_offset_px = heading_and_offset(
            centerline, frame_input.frame.shape[1]
        )
        metadata.update(_describe_path(centerline, self.config.straight_turn_deg))
        metadata["skeleton_pixels"] = int(np.count_nonzero(skeleton))
        metadata["endpoint_count"] = len(endpoints)
        metadata["junction_count"] = len(junctions)
        metadata["junctions_px"] = [
            [float(prepared.point_to_original((x, y))[0]),
             float(prepared.point_to_original((x, y))[1])]
            for y, x in junctions[:8]
        ]

        full_mask = _mask_to_original(
            skeleton, frame_input.frame.shape[1], prepared.roi_height
        )
        canvas = np.zeros(frame_input.frame.shape[:2], dtype=np.uint8)
        canvas[prepared.roi_top_px:prepared.roi_bottom_px, :] = full_mask
        debug_frame = overlay_detection(
            frame_input.frame, canvas, centerline, self.overlay_color
        )

        valid = len(centerline) >= self.config.min_path_points
        confidence = self._path_confidence(mask, skeleton, centerline)
        return DetectionResult(
            method=self.method_name,
            centerline=centerline,
            heading_rad=heading_rad,
            lateral_offset_px=lateral_offset_px,
            confidence=confidence,
            valid=valid,
            debug_frame=debug_frame,
            metadata=metadata,
        )

    # BaseTrackDetector requires this, but detect() is fully overridden above.
    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return self._wood_mask(working_frame)

    def _wood_mask(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        # Wood is strongly yellow; a gray/white desk is colour-neutral. The
        # Lab b* channel (yellow<->blue) separates them regardless of how
        # light or orange the wood is, so an Otsu split on b* generalises
        # across beech and pale maple far better than a fixed HSV window.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b_channel = lab[:, :, 2]
        a_channel = lab[:, :, 1]

        otsu_t, _ = cv2.threshold(
            b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Estimate the desk's neutral b* from a border ring (almost always
        # background) and threshold just above it. This adapts to weak
        # contrast (pale maple ~ b*148 on a desk ~ b*130) where a fixed Otsu
        # floor would clip most of the wood.
        ring = max(2, int(round(min(b_channel.shape) * 0.06)))
        border = np.concatenate([
            b_channel[:ring].ravel(), b_channel[-ring:].ravel(),
            b_channel[:, :ring].ravel(), b_channel[:, -ring:].ravel(),
        ])
        bg_b = float(np.median(border))
        thresh = max(
            float(self.config.lab_b_min),
            min(float(otsu_t), bg_b + float(self.config.lab_b_bg_margin)),
        )
        yellow_mask = (b_channel >= thresh).astype(np.uint8) * 255
        # Drop bluish specular highlights on the desk that survive Otsu.
        warm_mask = cv2.inRange(a_channel, self.config.lab_a_lower, 255)
        sat_mask = cv2.inRange(hsv[:, :, 1], self.config.min_saturation, 255)
        mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_or(warm_mask, sat_mask))

        ck = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.close_kernel, self.config.close_kernel)
        )
        ok = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.open_kernel, self.config.open_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ok)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)

        mask, shape_info = _largest_elongated_component(
            mask, self.config.min_blob_pixels, self.config.min_aspect_ratio
        )
        # The track piece is a solid bar: the two rail grooves are surface
        # channels, not holes through the silhouette. Filling the outer
        # contour yields one solid blob whose medial axis is a single clean
        # spine, instead of a noisy double-rail ladder with many junctions.
        mask = _fill_outer_contour(mask)
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        meta: dict[str, Any] = {"coverage_ratio": round(coverage, 4)}
        meta.update(shape_info)
        return mask, meta

    def _empty_result(
        self, frame_input: FrameInput, metadata: dict[str, Any], reason: str
    ) -> DetectionResult:
        metadata["rejected_reason"] = reason
        empty = np.zeros(frame_input.frame.shape[:2], dtype=np.uint8)
        return DetectionResult(
            method=self.method_name,
            centerline=[],
            heading_rad=None,
            lateral_offset_px=None,
            confidence=0.0,
            valid=False,
            debug_frame=overlay_detection(
                frame_input.frame, empty, [], self.overlay_color
            ),
            metadata=metadata,
        )

    def _path_confidence(
        self,
        mask: np.ndarray,
        skeleton: np.ndarray,
        centerline: list[tuple[float, float]],
    ) -> float:
        if not centerline:
            return 0.0
        mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        skel_len = float(np.count_nonzero(skeleton))
        diag = math.hypot(*skeleton.shape)
        span = min(1.0, skel_len / max(diag, 1.0))
        point_ratio = min(1.0, len(centerline) / float(max(self.config.min_path_points, 1)))
        confidence = min(1.0, 0.45 * span + 0.35 * point_ratio + 3.0 * mask_ratio)
        return round(confidence, 3)


# --- mask helpers ----------------------------------------------------------

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
    best_id = max(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
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
    if len(biggest) >= 5:
        (_, _), (w, h), _ = cv2.minAreaRect(biggest)
        if w >= 1 and h >= 1:
            aspect = float(max(w, h) / min(w, h))
            info["aspect_ratio"] = round(aspect, 2)
            if aspect < min_aspect_ratio:
                info["rejected_reason"] = "not_elongated"
                return np.zeros_like(mask), info
    return component, info


def _fill_outer_contour(mask: np.ndarray) -> np.ndarray:
    if not np.count_nonzero(mask):
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _zhang_suen_thinning(mask: np.ndarray, max_iterations: int) -> np.ndarray:
    img = (mask > 0).astype(np.uint8)
    for _ in range(max(1, max_iterations)):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1, mode="constant")
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = np.zeros(img.shape, dtype=np.uint8)
            for k in range(8):
                transitions += ((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8)
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            remove = (
                (img == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & (c1 == 0)
                & (c2 == 0)
            )
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return (img * 255).astype(np.uint8)


# --- skeleton graph --------------------------------------------------------

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    p = np.pad((skeleton > 0).astype(np.uint8), 1, mode="constant")
    total = np.zeros(skeleton.shape, dtype=np.uint8)
    for dy, dx in _OFFSETS:
        total += p[1 + dy:1 + dy + skeleton.shape[0], 1 + dx:1 + dx + skeleton.shape[1]]
    return total * (skeleton > 0)


def _skeleton_nodes(skeleton: np.ndarray) -> tuple[list[Pixel], list[Pixel]]:
    degree = _neighbor_count(skeleton)
    ends = [(int(y), int(x)) for y, x in zip(*np.where(degree == 1))]
    junctions = [(int(y), int(x)) for y, x in zip(*np.where(degree >= 3))]
    return ends, junctions


def _bfs_farthest(
    skeleton: np.ndarray, start: Pixel
) -> tuple[Pixel, dict[Pixel, Pixel | None]]:
    visited: dict[Pixel, Pixel | None] = {start: None}
    queue: deque[Pixel] = deque([start])
    farthest = start
    height, width = skeleton.shape
    while queue:
        y, x = queue.popleft()
        farthest = (y, x)
        for dy, dx in _OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and skeleton[ny, nx] and (ny, nx) not in visited:
                visited[(ny, nx)] = (y, x)
                queue.append((ny, nx))
    return farthest, visited


def _longest_geodesic(skeleton: np.ndarray) -> list[Pixel]:
    points = np.argwhere(skeleton > 0)
    if len(points) < 2:
        return []
    seed = (int(points[0][0]), int(points[0][1]))
    end_a, _ = _bfs_farthest(skeleton, seed)
    end_b, parents = _bfs_farthest(skeleton, end_a)
    path: list[Pixel] = []
    node: Pixel | None = end_b
    while node is not None:
        path.append(node)
        node = parents.get(node)
    path.reverse()
    return path


def _order_bottom_first(path: list[Pixel]) -> list[Pixel]:
    if path and path[0][0] < path[-1][0]:
        return list(reversed(path))
    return path


# --- path geometry ---------------------------------------------------------

def _smooth_polyline(
    points: list[tuple[float, float]], window: int
) -> list[tuple[float, float]]:
    if len(points) < 5 or window < 3:
        return points
    win = window if window % 2 == 1 else window + 1
    coords = np.array(points, dtype=np.float32)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    out = coords.copy()
    for axis in (0, 1):
        padded = np.pad(coords[:, axis], (pad, pad), mode="edge")
        out[:, axis] = np.convolve(padded, kernel, mode="valid")
    return [(float(x), float(y)) for x, y in out]


def _resample_equal_arc(
    pts: np.ndarray, seg_len: np.ndarray, arc_length: float, nodes: int
) -> np.ndarray:
    if len(pts) < 3 or arc_length <= 0:
        return pts
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    targets = np.linspace(0.0, arc_length, max(3, nodes))
    xs = np.interp(targets, cumulative, pts[:, 0])
    ys = np.interp(targets, cumulative, pts[:, 1])
    return np.column_stack([xs, ys])


def _describe_path(
    centerline: list[tuple[float, float]], straight_turn_deg: float
) -> dict[str, Any]:
    pts = np.array(centerline, dtype=np.float64)
    deltas = pts[1:] - pts[:-1]
    seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
    arc_length = float(seg_len.sum())

    # Classify the *macro* shape from how the path deviates from its straight
    # start->end chord, measured on an equal-arc coarse polyline. This is
    # immune to skeleton/grain jitter (high-frequency, low-amplitude) because
    # it looks at gross excursion, not summed per-pixel heading deltas.
    coarse = _resample_equal_arc(pts, seg_len, arc_length, nodes=18)
    if len(coarse) >= 3:
        cdeltas = coarse[1:] - coarse[:-1]
        headings = np.unwrap(np.arctan2(cdeltas[:, 1], cdeltas[:, 0]))
        turn_rad = float(headings[-1] - headings[0])
        total_abs_curv = float(np.abs(np.diff(headings)).sum()) if len(headings) >= 2 else 0.0
        chord = coarse[-1] - coarse[0]
        chord_len = float(np.hypot(*chord))
        if chord_len > 1.0:
            unit = chord / chord_len
            rel = coarse - coarse[0]
            # signed perpendicular distance of each node from the chord
            dev = rel[:, 0] * unit[1] - rel[:, 1] * unit[0]
            dev_max, dev_min = float(dev.max()), float(dev.min())
        else:
            dev_max = dev_min = 0.0
        straightness = chord_len / max(arc_length, 1.0)
    else:
        turn_rad = total_abs_curv = dev_max = dev_min = 0.0
        straightness = 1.0
    turn_deg = math.degrees(turn_rad)

    # Classification scheme the handheld-photo data robustly supports. Fine
    # snake-vs-gentle-curve separation is *not* reliable from these oblique
    # shots (a plain curved piece and a real snake have overlapping coarse
    # turning), so we expose the raw measures in metadata and only assert
    # categories with clean separation.
    excursion = max(dev_max, -dev_min)
    span_ratio = excursion / max(arc_length, 1.0)
    bow_thr = 0.05 * arc_length
    if span_ratio < 0.035 and straightness > 0.9:
        shape = "straight"
    elif straightness < 0.85:
        # Path doubles back: multi-piece layout / loop / full circuit.
        shape = "compound"
    elif dev_max > bow_thr and -dev_min > bow_thr:
        shape = "snake"
    elif (dev_max + dev_min) < 0:
        shape = "curved_left"
    else:
        shape = "curved_right"
    return {
        "path_shape": shape,
        "path_length_px": round(arc_length, 2),
        "net_turn_deg": round(turn_deg, 2),
        "total_curvature_deg": round(math.degrees(total_abs_curv), 2),
        "start_px": [round(float(pts[0][0]), 2), round(float(pts[0][1]), 2)],
        "end_px": [round(float(pts[-1][0]), 2), round(float(pts[-1][1]), 2)],
        "point_count": len(centerline),
    }


def _mask_to_original(
    mask: np.ndarray, original_width: int, roi_height: int
) -> np.ndarray:
    if mask.shape == (roi_height, original_width):
        return mask.astype(np.uint8)
    resized = cv2.resize(
        mask, (original_width, roi_height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(np.uint8)
