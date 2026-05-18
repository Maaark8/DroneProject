from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from track_detection.types import DetectionResult, FrameInput


@dataclass(slots=True)
class DroneLightConfig:
    working_width: int = 320
    blur_kernel: int = 5
    min_light_pixels: int = 8
    max_light_area_ratio: float = 0.08
    max_light_diameter_ratio: float = 0.16
    max_bbox_aspect_ratio: float = 1.8
    min_circularity: float = 0.4
    min_brightness: int = 205
    brightness_percentile: float = 99.2
    min_saturation: int = 80
    prefer_red_light: bool = True
    preferred_hue_margin: int = 16
    hotspot_percentile: float = 95.0
    hotspot_delta: int = 18
    detect_white_light: bool = False
    white_min_brightness: int = 252
    white_max_saturation: int = 55
    open_kernel: int = 3
    close_kernel: int = 5
    min_confidence: float = 0.28
    assumed_fps: float = 30.0


class DroneLightDetector:
    method_name = "drone_light"
    overlay_color = (255, 80, 80)

    def __init__(self, config: DroneLightConfig | None = None) -> None:
        self.config = config or DroneLightConfig()
        self._last_position_px: tuple[float, float] | None = None
        self._last_timestamp_s: float | None = None
        self._last_frame_id: int | None = None

    def detect(self, frame_input: FrameInput) -> DetectionResult:
        frame = frame_input.frame
        working, scale_x, scale_y = _resize_for_detection(frame, self.config.working_width)
        mask = self._light_mask(working)
        candidate = _best_light_candidate(mask, working, self.config)
        full_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        metadata: dict[str, Any] = {
            "detection_type": "drone_light",
            "schema_version": "drone_control_observation.v1",
            "frame_id": frame_input.frame_id,
            "timestamp_s": frame_input.timestamp_s,
            "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "mask_pixels": int(np.count_nonzero(mask)),
        }

        if candidate is None:
            self._remember(None, frame_input)
            metadata["rejected_reason"] = "no_light_candidate"
            return DetectionResult(
                method=self.method_name,
                centerline=[],
                heading_rad=None,
                lateral_offset_px=None,
                confidence=0.0,
                valid=False,
                debug_frame=_overlay_drone_detection(frame, full_mask, metadata, self.overlay_color),
                metadata=metadata,
            )

        cx = candidate["center"][0] * scale_x
        cy = candidate["center"][1] * scale_y
        radius_px = candidate["radius"] * ((scale_x + scale_y) / 2.0)
        x, y, w, h = candidate["bbox"]
        bbox_xywh = [
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        ]
        offset_x = cx - (frame.shape[1] / 2.0)
        offset_y = cy - (frame.shape[0] / 2.0)
        position_px = (float(cx), float(cy))
        velocity_px_s = self._velocity(position_px, frame_input)
        self._remember(position_px, frame_input)

        metadata.update(
            {
                "position_px": {"x": round(float(cx), 2), "y": round(float(cy), 2)},
                "offset_px": {"x": round(float(offset_x), 2), "y": round(float(offset_y), 2)},
                "offset_norm": {
                    "x": round(_clip(offset_x / (frame.shape[1] / 2.0), -1.0, 1.0), 4),
                    "y": round(_clip(offset_y / (frame.shape[0] / 2.0), -1.0, 1.0), 4),
                },
                "bbox_xywh": bbox_xywh,
                "radius_px": round(float(radius_px), 2),
                "area_px": int(round(candidate["area"] * scale_x * scale_y)),
                "color_bgr": candidate["color_bgr"],
                "color_hsv": candidate["color_hsv"],
                "color_name": _hue_name(candidate["color_hsv"][0], candidate["color_hsv"][1]),
                "velocity_px_s": velocity_px_s,
                "speed_px_s": None
                if velocity_px_s is None
                else round(float(np.hypot(velocity_px_s["x"], velocity_px_s["y"])), 2),
                "component_score": round(float(candidate["score"]), 3),
                "component_circularity": round(float(candidate["circularity"]), 3),
                "preferred_hue_score": round(float(candidate["preferred_hue_score"]), 3),
                "hotspot_score": round(float(candidate["hotspot_score"]), 3),
                "peak_brightness": int(candidate["peak_brightness"]),
            }
        )

        confidence = round(float(candidate["score"]), 3)
        valid = confidence >= self.config.min_confidence
        return DetectionResult(
            method=self.method_name,
            centerline=[(float(cx), float(cy))],
            heading_rad=None,
            lateral_offset_px=float(offset_x),
            confidence=confidence,
            valid=valid,
            debug_frame=_overlay_drone_detection(frame, full_mask, metadata, self.overlay_color),
            metadata=metadata,
        )

    def _light_mask(self, frame: np.ndarray) -> np.ndarray:
        blur_kernel = self.config.blur_kernel if self.config.blur_kernel % 2 == 1 else self.config.blur_kernel + 1
        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        dynamic_floor = max(
            self.config.min_brightness,
            int(np.percentile(value, self.config.brightness_percentile)) - 8,
        )
        saturated_light = cv2.inRange(
            hsv,
            (0, self.config.min_saturation, dynamic_floor),
            (179, 255, 255),
        )
        mask = saturated_light
        if self.config.detect_white_light:
            white_light = cv2.inRange(
                hsv,
                (0, 0, self.config.white_min_brightness),
                (179, self.config.white_max_saturation, 255),
            )
            mask = cv2.bitwise_or(mask, white_light)

        ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.open_kernel, self.config.open_kernel))
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.close_kernel, self.config.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ok)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)
        return mask

    def _velocity(self, position_px: tuple[float, float], frame_input: FrameInput) -> dict[str, float] | None:
        if self._last_position_px is None:
            return None
        dt = _elapsed_seconds(
            previous_timestamp_s=self._last_timestamp_s,
            current_timestamp_s=frame_input.timestamp_s,
            previous_frame_id=self._last_frame_id,
            current_frame_id=frame_input.frame_id,
            assumed_fps=self.config.assumed_fps,
        )
        if dt is None or dt <= 0:
            return None
        vx = (position_px[0] - self._last_position_px[0]) / dt
        vy = (position_px[1] - self._last_position_px[1]) / dt
        return {"x": round(float(vx), 2), "y": round(float(vy), 2)}

    def _remember(self, position_px: tuple[float, float] | None, frame_input: FrameInput) -> None:
        self._last_position_px = position_px
        self._last_timestamp_s = frame_input.timestamp_s
        self._last_frame_id = frame_input.frame_id


def _resize_for_detection(frame: np.ndarray, working_width: int) -> tuple[np.ndarray, float, float]:
    height, width = frame.shape[:2]
    target_width = min(max(1, working_width), width)
    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    if target_width == width and target_height == height:
        return frame, 1.0, 1.0
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, width / float(target_width), height / float(target_height)


def _best_light_candidate(mask: np.ndarray, frame: np.ndarray, config: DroneLightConfig) -> dict[str, Any] | None:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_area = int(mask.size * config.max_light_area_ratio)
    best: dict[str, Any] | None = None

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < config.min_light_pixels or area > max_area:
            continue
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        max_diameter = min(mask.shape[:2]) * config.max_light_diameter_ratio
        if max(width, height) > max_diameter:
            continue
        aspect_ratio = max(width, height) / float(max(min(width, height), 1))
        if aspect_ratio > config.max_bbox_aspect_ratio:
            continue

        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        component_mask[labels == component_id] = 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0 else min(1.0, (4.0 * np.pi * contour_area) / (perimeter * perimeter))
        if circularity < config.min_circularity:
            continue
        (_, _), radius = cv2.minEnclosingCircle(contour)

        ys, xs = np.nonzero(component_mask)
        mean_hsv = np.round(hsv[ys, xs].mean(axis=0)).astype(int)
        mean_bgr = np.round(frame[ys, xs].mean(axis=0)).astype(int)
        brightness_values = hsv[ys, xs, 2].astype(np.float32)
        brightness_score = float(mean_hsv[2]) / 255.0
        saturation_score = float(mean_hsv[1]) / 255.0
        area_score = min(1.0, area / float(max(config.min_light_pixels * 8, 1)))
        preferred_hue_score = _preferred_hue_score(
            hue=int(mean_hsv[0]),
            saturation=int(mean_hsv[1]),
            config=config,
        )
        peak_brightness = float(np.percentile(brightness_values, config.hotspot_percentile))
        hotspot_score = _clip01((peak_brightness - float(mean_hsv[2]) - float(config.hotspot_delta)) / 55.0)
        score = (
            (0.28 * brightness_score)
            + (0.12 * saturation_score)
            + (0.18 * circularity)
            + (0.12 * area_score)
            + (0.18 * preferred_hue_score)
            + (0.12 * hotspot_score)
        )

        candidate = {
            "center": (float(centroids[component_id][0]), float(centroids[component_id][1])),
            "bbox": [
                int(stats[component_id, cv2.CC_STAT_LEFT]),
                int(stats[component_id, cv2.CC_STAT_TOP]),
                width,
                height,
            ],
            "area": area,
            "radius": float(radius),
            "color_hsv": [int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])],
            "color_bgr": [int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])],
            "circularity": float(circularity),
            "score": float(score),
            "preferred_hue_score": float(preferred_hue_score),
            "hotspot_score": float(hotspot_score),
            "peak_brightness": int(round(peak_brightness)),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def _overlay_drone_detection(
    frame: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = frame.copy()
    mask_bool = mask > 0
    if np.any(mask_bool):
        tint = np.zeros_like(frame)
        tint[:, :] = color
        overlay[mask_bool] = cv2.addWeighted(overlay, 0.35, tint, 0.65, 0)[mask_bool]

    if "position_px" not in metadata:
        return overlay

    cx = int(round(metadata["position_px"]["x"]))
    cy = int(round(metadata["position_px"]["y"]))
    radius = max(4, int(round(metadata.get("radius_px", 4))))
    x, y, w, h = metadata["bbox_xywh"]
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    cv2.circle(overlay, (cx, cy), radius, (0, 255, 255), 2)
    cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    label = f"drone {metadata.get('confidence', metadata.get('component_score', 0.0)):.2f}"
    cv2.putText(overlay, label, (max(0, x), max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def _elapsed_seconds(
    previous_timestamp_s: float | None,
    current_timestamp_s: float | None,
    previous_frame_id: int | None,
    current_frame_id: int,
    assumed_fps: float,
) -> float | None:
    if previous_timestamp_s is not None and current_timestamp_s is not None:
        return current_timestamp_s - previous_timestamp_s
    if previous_frame_id is not None and current_frame_id > previous_frame_id and assumed_fps > 0:
        return (current_frame_id - previous_frame_id) / assumed_fps
    return None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _preferred_hue_score(hue: int, saturation: int, config: DroneLightConfig) -> float:
    if not config.prefer_red_light:
        return 1.0
    if saturation < max(config.min_saturation // 2, 1):
        return 0.0
    distance = min(abs(int(hue) - 0), abs(int(hue) - 179))
    score = 1.0 - (distance / float(max(config.preferred_hue_margin, 1)))
    return _clip01(score)


def _hue_name(hue: int, saturation: int) -> str:
    if saturation < 35:
        return "white"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    if hue < 155:
        return "purple"
    return "pink"
