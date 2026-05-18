from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from track_detection.types import DetectionResult, FrameInput


@dataclass(slots=True)
class ArucoMarkerConfig:
    dictionary_name: str = "DICT_4X4_50"
    marker_id: int = 7
    working_width: int = 640
    min_perimeter_px: float = 40.0
    min_confidence: float = 0.35
    assumed_fps: float = 30.0
    detect_inverted_marker: bool = False


class ArucoMarkerDetector:
    method_name = "drone_marker"
    overlay_color = (40, 220, 40)

    def __init__(self, config: ArucoMarkerConfig | None = None) -> None:
        self.config = config or ArucoMarkerConfig()
        if not hasattr(cv2, "aruco"):
            raise ImportError("OpenCV ArUco module is unavailable. Install an OpenCV build with cv2.aruco support.")
        dictionary_id = getattr(cv2.aruco, self.config.dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {self.config.dictionary_name}")
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        params = cv2.aruco.DetectorParameters()
        params.detectInvertedMarker = bool(self.config.detect_inverted_marker)
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
        self._last_position_px: tuple[float, float] | None = None
        self._last_timestamp_s: float | None = None
        self._last_frame_id: int | None = None

    def detect(self, frame_input: FrameInput) -> DetectionResult:
        frame = frame_input.frame
        working, scale_x, scale_y = _resize_for_detection(frame, self.config.working_width)
        corners, ids, rejected = self._detector.detectMarkers(working)

        metadata: dict[str, Any] = {
            "detection_type": "drone_marker",
            "schema_version": "drone_control_observation.v1",
            "frame_id": frame_input.frame_id,
            "timestamp_s": frame_input.timestamp_s,
            "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "marker_dictionary": self.config.dictionary_name,
            "target_marker_id": self.config.marker_id,
            "rejected_candidates": 0 if rejected is None else len(rejected),
        }

        candidate = _select_marker_candidate(
            corners=corners,
            ids=ids,
            scale_x=scale_x,
            scale_y=scale_y,
            config=self.config,
        )
        if candidate is None:
            self._remember(None, frame_input)
            metadata["rejected_reason"] = "marker_not_found"
            return DetectionResult(
                method=self.method_name,
                centerline=[],
                heading_rad=None,
                lateral_offset_px=None,
                confidence=0.0,
                valid=False,
                debug_frame=_overlay_marker_detection(frame, metadata, self.overlay_color),
                metadata=metadata,
            )

        center = candidate["center"]
        offset_x = center[0] - (frame.shape[1] / 2.0)
        offset_y = center[1] - (frame.shape[0] / 2.0)
        velocity_px_s = self._velocity(center, frame_input)
        self._remember(center, frame_input)

        metadata.update(
            {
                "position_px": {"x": round(float(center[0]), 2), "y": round(float(center[1]), 2)},
                "offset_px": {"x": round(float(offset_x), 2), "y": round(float(offset_y), 2)},
                "offset_norm": {
                    "x": round(_clip(offset_x / (frame.shape[1] / 2.0), -1.0, 1.0), 4),
                    "y": round(_clip(offset_y / (frame.shape[0] / 2.0), -1.0, 1.0), 4),
                },
                "bbox_xywh": candidate["bbox_xywh"],
                "marker_id": candidate["marker_id"],
                "marker_corners": candidate["corners"],
                "heading_rad": round(float(candidate["heading_rad"]), 4),
                "marker_size_px": round(float(candidate["marker_size_px"]), 2),
                "velocity_px_s": velocity_px_s,
                "speed_px_s": None
                if velocity_px_s is None
                else round(float(np.hypot(velocity_px_s["x"], velocity_px_s["y"])), 2),
            }
        )

        confidence = round(float(candidate["confidence"]), 3)
        metadata["confidence"] = confidence
        return DetectionResult(
            method=self.method_name,
            centerline=[(float(center[0]), float(center[1]))],
            heading_rad=float(candidate["heading_rad"]),
            lateral_offset_px=float(offset_x),
            confidence=confidence,
            valid=confidence >= self.config.min_confidence,
            debug_frame=_overlay_marker_detection(frame, metadata, self.overlay_color),
            metadata=metadata,
        )

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


def generate_aruco_marker_image(
    output_path: Path,
    dictionary_name: str = "DICT_4X4_50",
    marker_id: int = 7,
    side_pixels: int = 400,
    margin_pixels: int = 80,
) -> Path:
    if not hasattr(cv2, "aruco"):
        raise ImportError("OpenCV ArUco module is unavailable. Install an OpenCV build with cv2.aruco support.")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), int(side_pixels))
    margin = max(int(margin_pixels), 0)
    full = np.full((marker.shape[0] + (margin * 2), marker.shape[1] + (margin * 2)), 255, dtype=np.uint8)
    full[margin : margin + marker.shape[0], margin : margin + marker.shape[1]] = marker
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), full)
    return output_path


def _select_marker_candidate(
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    scale_x: float,
    scale_y: float,
    config: ArucoMarkerConfig,
) -> dict[str, Any] | None:
    if ids is None or len(corners) == 0:
        return None

    best: dict[str, Any] | None = None
    target_id = int(config.marker_id)

    for marker_corners, marker_id_value in zip(corners, ids.flatten()):
        pts = np.array(marker_corners, dtype=np.float32).reshape(4, 2)
        scaled = np.column_stack((pts[:, 0] * scale_x, pts[:, 1] * scale_y)).astype(np.float32)
        side_lengths = [float(np.hypot(*(scaled[(idx + 1) % 4] - scaled[idx]))) for idx in range(4)]
        perimeter = float(sum(side_lengths))
        if perimeter < config.min_perimeter_px:
            continue
        center = tuple(np.mean(scaled, axis=0))
        heading_vector = scaled[1] - scaled[0]
        heading_rad = float(np.arctan2(heading_vector[1], heading_vector[0]))
        x_min = int(np.floor(np.min(scaled[:, 0])))
        y_min = int(np.floor(np.min(scaled[:, 1])))
        x_max = int(np.ceil(np.max(scaled[:, 0])))
        y_max = int(np.ceil(np.max(scaled[:, 1])))
        bbox = [x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)]
        id_match = int(marker_id_value) == target_id
        if not id_match:
            continue
        perimeter_score = _clip01(perimeter / 240.0)
        confidence = 0.8 + (0.2 * perimeter_score)
        candidate = {
            "marker_id": int(marker_id_value),
            "center": (float(center[0]), float(center[1])),
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in scaled],
            "bbox_xywh": bbox,
            "heading_rad": heading_rad,
            "marker_size_px": float(np.mean(side_lengths)),
            "confidence": _clip01(confidence),
            "perimeter": perimeter,
        }
        if best is None:
            best = candidate
            continue
        best_key = (best["confidence"], best["perimeter"])
        candidate_key = (candidate["confidence"], candidate["perimeter"])
        if candidate_key > best_key:
            best = candidate

    return best


def _overlay_marker_detection(frame: np.ndarray, metadata: dict[str, Any], color: tuple[int, int, int]) -> np.ndarray:
    overlay = frame.copy()
    corners = metadata.get("marker_corners")
    if corners:
        polygon = np.array([[int(round(x)), int(round(y))] for x, y in corners], dtype=np.int32)
        cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)
        cv2.line(overlay, tuple(polygon[0]), tuple(polygon[1]), (0, 255, 255), 2)
    if "position_px" in metadata:
        cx = int(round(metadata["position_px"]["x"]))
        cy = int(round(metadata["position_px"]["y"]))
        cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        x, y, w, h = metadata["bbox_xywh"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"aruco id={metadata.get('marker_id')} score={metadata.get('confidence', 0.0):.2f}"
        cv2.putText(overlay, label, (max(0, x), max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return overlay


def _resize_for_detection(frame: np.ndarray, working_width: int) -> tuple[np.ndarray, float, float]:
    height, width = frame.shape[:2]
    target_width = min(max(1, int(working_width)), width)
    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    if target_width == width and target_height == height:
        return frame, 1.0, 1.0
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, width / float(target_width), height / float(target_height)


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
