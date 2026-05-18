from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .mission import MissionPath, resample_polyline
from .types import Point

WAYPOINT_SCHEMA_VERSION = "codrone_waypoint_mission.v1"


@dataclass(slots=True)
class Waypoint:
    x_m: float
    y_m: float
    z_m: float
    speed_m_s: float
    heading_deg: int = 0
    rotational_velocity_dps: int = 0
    hold_s: float = 0.0
    label: str | None = None

    def to_dict(self) -> dict:
        return {
            "x_m": round(float(self.x_m), 4),
            "y_m": round(float(self.y_m), 4),
            "z_m": round(float(self.z_m), 4),
            "speed_m_s": round(float(self.speed_m_s), 3),
            "heading_deg": int(self.heading_deg),
            "rotational_velocity_dps": int(self.rotational_velocity_dps),
            "hold_s": round(float(self.hold_s), 3),
            "label": self.label,
        }


@dataclass(slots=True)
class WaypointMission:
    waypoints: list[Waypoint]
    image_points: list[Point]
    frame_size: dict[str, int] | None
    source: str | None
    meters_per_pixel: float
    drone_start_px: Point
    drone_forward_rad: float
    target_height_m: float
    schema_version: str = WAYPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "frame_size": self.frame_size,
            "meters_per_pixel": round(float(self.meters_per_pixel), 6),
            "drone_start_px": {"x": round(float(self.drone_start_px[0]), 2), "y": round(float(self.drone_start_px[1]), 2)},
            "drone_forward_rad": round(float(self.drone_forward_rad), 6),
            "target_height_m": round(float(self.target_height_m), 3),
            "image_points": [{"x": round(float(x), 2), "y": round(float(y), 2)} for x, y in self.image_points],
            "waypoints": [waypoint.to_dict() for waypoint in self.waypoints],
        }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def meters_per_pixel_from_reference(reference_points: tuple[Point, Point], reference_distance_cm: float) -> float:
    distance_px = float(np.hypot(reference_points[1][0] - reference_points[0][0], reference_points[1][1] - reference_points[0][1]))
    if distance_px <= 0:
        raise ValueError("Reference points must be distinct.")
    distance_m = max(float(reference_distance_cm), 0.1) / 100.0
    return distance_m / distance_px


def waypoint_mission_from_path(
    mission: MissionPath,
    drone_start_px: Point,
    drone_forward_rad: float,
    meters_per_pixel: float,
    target_height_m: float = 0.8,
    waypoint_spacing_px: float = 120.0,
    speed_m_s: float = 0.8,
    hold_s: float = 0.0,
) -> WaypointMission:
    if len(mission.points) < 2:
        raise ValueError("Mission path requires at least two points.")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive.")

    sampled_points = resample_polyline(mission.points, spacing_px=max(float(waypoint_spacing_px), 1.0))
    if len(sampled_points) < 2:
        raise ValueError("Waypoint mission requires at least two sampled points.")

    forward = np.array([math.cos(drone_forward_rad), math.sin(drone_forward_rad)], dtype=np.float32)
    left = np.array([forward[1], -forward[0]], dtype=np.float32)
    origin = np.array(drone_start_px, dtype=np.float32)

    clamped_speed = min(max(float(speed_m_s), 0.5), 2.0)
    waypoints: list[Waypoint] = []
    for index, point in enumerate(sampled_points):
        delta_px = np.array(point, dtype=np.float32) - origin
        x_m = float(np.dot(delta_px, forward) * meters_per_pixel)
        y_m = float(np.dot(delta_px, left) * meters_per_pixel)
        waypoints.append(
            Waypoint(
                x_m=x_m,
                y_m=y_m,
                z_m=float(target_height_m),
                speed_m_s=clamped_speed,
                hold_s=float(hold_s),
                label=f"wp_{index:02d}",
            )
        )

    return WaypointMission(
        waypoints=waypoints,
        frame_size=mission.frame_size,
        source=mission.source,
        image_points=[(float(point[0]), float(point[1])) for point in sampled_points],
        meters_per_pixel=float(meters_per_pixel),
        drone_start_px=(float(drone_start_px[0]), float(drone_start_px[1])),
        drone_forward_rad=float(drone_forward_rad),
        target_height_m=float(target_height_m),
    )


def marker_forward_heading_rad(marker_heading_rad: float | None) -> float | None:
    if marker_heading_rad is None:
        return None
    return _wrap_angle(float(marker_heading_rad) - (math.pi / 2.0))


def path_start_heading_rad(path: list[Point]) -> float:
    if len(path) < 2:
        raise ValueError("Path must contain at least two points.")
    start = path[0]
    end = path[1]
    return math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))


def overlay_waypoint_plan(
    frame: np.ndarray,
    mission: MissionPath,
    waypoint_mission: WaypointMission,
) -> np.ndarray:
    overlay = frame.copy()
    if mission.points:
        polyline = np.array([[int(round(x)), int(round(y))] for x, y in mission.points], dtype=np.int32)
        cv2.polylines(overlay, [polyline], isClosed=False, color=(255, 220, 0), thickness=2)

    start_x = int(round(waypoint_mission.drone_start_px[0]))
    start_y = int(round(waypoint_mission.drone_start_px[1]))
    cv2.drawMarker(overlay, (start_x, start_y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

    heading = waypoint_mission.drone_forward_rad
    arrow_end = (
        int(round(start_x + (40.0 * math.cos(heading)))),
        int(round(start_y + (40.0 * math.sin(heading)))),
    )
    cv2.arrowedLine(overlay, (start_x, start_y), arrow_end, (0, 180, 255), 2, tipLength=0.2)

    for index, point_px in enumerate(waypoint_mission.image_points):
        px = int(round(point_px[0]))
        py = int(round(point_px[1]))
        cv2.circle(overlay, (px, py), 6, (40, 220, 40), -1)
        cv2.putText(
            overlay,
            str(index),
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )

    info_lines = [
        f"meters_per_pixel={waypoint_mission.meters_per_pixel:.5f}",
        f"target_height_m={waypoint_mission.target_height_m:.2f}",
        f"waypoints={len(waypoint_mission.waypoints)}",
    ]
    for index, line in enumerate(info_lines):
        cv2.putText(
            overlay,
            line,
            (12, 28 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
