from __future__ import annotations

from typing import Any

from .types import DetectionResult


SCHEMA_VERSION = "drone_control_observation.v1"


def to_control_observation(result: DetectionResult) -> dict[str, Any]:
    metadata = result.metadata
    frame_size = metadata.get("frame_size")
    if frame_size is None:
        frame_size = _frame_size_from_debug(result)

    observation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_method": result.method,
        "valid": bool(result.valid),
        "confidence": float(result.confidence),
        "frame_id": metadata.get("frame_id"),
        "timestamp_s": metadata.get("timestamp_s"),
        "frame_size": frame_size,
        "target": None,
    }

    if metadata.get("detection_type") in {"drone_light", "drone_marker"}:
        observation["target"] = {
            "kind": "drone",
            "position_px": metadata.get("position_px"),
            "offset_px": metadata.get("offset_px"),
            "offset_norm": metadata.get("offset_norm"),
            "velocity_px_s": metadata.get("velocity_px_s"),
            "speed_px_s": metadata.get("speed_px_s"),
            "bbox_xywh": metadata.get("bbox_xywh"),
            "radius_px": metadata.get("radius_px"),
            "color_name": metadata.get("color_name"),
            "heading_rad": metadata.get("heading_rad"),
            "marker_id": metadata.get("marker_id"),
        }
        return observation

    if result.centerline:
        frame_width = None if frame_size is None else frame_size.get("width")
        target_point = result.centerline[-1]
        offset_norm_x = None
        if result.lateral_offset_px is not None and frame_width:
            offset_norm_x = _clip(result.lateral_offset_px / (frame_width / 2.0), -1.0, 1.0)
        observation["target"] = {
            "kind": "track_centerline",
            "position_px": {"x": round(float(target_point[0]), 2), "y": round(float(target_point[1]), 2)},
            "offset_px": {"x": result.lateral_offset_px, "y": None},
            "offset_norm": {"x": None if offset_norm_x is None else round(offset_norm_x, 4), "y": None},
            "heading_rad": result.heading_rad,
            "centerline": result.centerline,
        }
    return observation


def _frame_size_from_debug(result: DetectionResult) -> dict[str, int] | None:
    if result.debug_frame is None:
        return None
    return {"width": int(result.debug_frame.shape[1]), "height": int(result.debug_frame.shape[0])}


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
