from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mission import MissionPath
from .types import Point


@dataclass(slots=True)
class DroneCommand:
    roll: int = 0
    pitch: int = 0
    yaw: int = 0
    throttle: int = 0
    hover: bool = False
    land: bool = False
    reason: str = "track_follow"

    def as_dict(self) -> dict[str, Any]:
        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "throttle": self.throttle,
            "hover": self.hover,
            "land": self.land,
            "reason": self.reason,
        }


@dataclass(slots=True)
class TrackFollowerConfig:
    lookahead_px: float = 70.0
    completion_radius_px: float = 30.0
    confidence_threshold: float = 0.45
    lost_frame_limit: int = 30
    land_on_vision_loss: bool = False
    lateral_deadband_norm: float = 0.04
    longitudinal_deadband_norm: float = 0.05
    height_target_cm: float = 80.0
    height_deadband_cm: float = 4.0
    roll_sign: int = 1
    pitch_sign: int = 1
    kp_roll: float = 30.0
    kp_pitch: float = 28.0
    kd_roll: float = 12.0
    kd_pitch: float = 10.0
    kp_height: float = 0.8
    max_roll_power: int = 20
    max_pitch_power: int = 20
    max_yaw_power: int = 0
    max_throttle_power: int = 15
    land_on_completion: bool = False


@dataclass(slots=True)
class ControllerOutput:
    command: DroneCommand
    lookahead_point: Point | None
    nearest_point: Point | None
    error_norm: dict[str, float] | None
    velocity_norm: dict[str, float] | None
    path_progress: float | None
    path_remaining_px: float | None
    finished: bool = False
    reason: str = "track_follow"

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.as_dict(),
            "lookahead_point": _point_to_dict(self.lookahead_point),
            "nearest_point": _point_to_dict(self.nearest_point),
            "error_norm": self.error_norm,
            "velocity_norm": self.velocity_norm,
            "path_progress": self.path_progress,
            "path_remaining_px": self.path_remaining_px,
            "finished": self.finished,
            "reason": self.reason,
        }


@dataclass(slots=True)
class TrackFollowerController:
    mission: MissionPath
    config: TrackFollowerConfig = field(default_factory=TrackFollowerConfig)
    _lost_frames: int = 0

    def update(self, observation: dict[str, Any], height_cm: float | None = None) -> ControllerOutput:
        target = observation.get("target") or {}
        if not observation.get("valid") or observation.get("confidence", 0.0) < self.config.confidence_threshold:
            return self._handle_missing_target(reason="low_confidence_or_invalid")
        if target.get("kind") != "drone":
            return self._handle_missing_target(reason="unexpected_target_kind")

        position = target.get("position_px")
        if position is None or position.get("x") is None or position.get("y") is None:
            return self._handle_missing_target(reason="missing_drone_position")

        self._lost_frames = 0
        drone_point = (float(position["x"]), float(position["y"]))
        nearest_point, progress_px = nearest_point_on_path(drone_point, self.mission.points)
        lookahead_progress = min(progress_px + self.config.lookahead_px, self.mission.path_length_px)
        lookahead_point = point_at_distance(self.mission.points, lookahead_progress)
        remaining_px = max(0.0, self.mission.path_length_px - progress_px)

        frame_size = observation.get("frame_size") or self.mission.frame_size
        if frame_size is None:
            raise ValueError("Track follower requires frame_size in observation or mission path.")
        frame_width = max(float(frame_size["width"]), 1.0)
        frame_height = max(float(frame_size["height"]), 1.0)

        error_x = (lookahead_point[0] - drone_point[0]) / (frame_width / 2.0)
        error_y = (lookahead_point[1] - drone_point[1]) / (frame_height / 2.0)
        error_x = _apply_deadband(error_x, self.config.lateral_deadband_norm)
        error_y = _apply_deadband(error_y, self.config.longitudinal_deadband_norm)

        velocity_norm = self._velocity_norm(target, frame_width, frame_height)
        roll_power = self.config.roll_sign * (
            (self.config.kp_roll * error_x) - (self.config.kd_roll * velocity_norm["x"])
        )
        pitch_power = self.config.pitch_sign * (
            (self.config.kp_pitch * error_y) - (self.config.kd_pitch * velocity_norm["y"])
        )
        throttle_power = self._throttle_power(height_cm)

        finished = remaining_px <= self.config.completion_radius_px
        if finished and self.config.land_on_completion:
            command = DroneCommand(land=True, reason="path_complete")
        elif finished:
            command = DroneCommand(hover=True, throttle=throttle_power, reason="path_complete")
        else:
            command = DroneCommand(
                roll=_clamp_int(roll_power, self.config.max_roll_power),
                pitch=_clamp_int(pitch_power, self.config.max_pitch_power),
                yaw=0,
                throttle=throttle_power,
                reason="track_follow",
            )

        return ControllerOutput(
            command=command,
            lookahead_point=lookahead_point,
            nearest_point=nearest_point,
            error_norm={"x": round(float(error_x), 4), "y": round(float(error_y), 4)},
            velocity_norm=velocity_norm,
            path_progress=round(float(progress_px), 2),
            path_remaining_px=round(float(remaining_px), 2),
            finished=finished,
            reason=command.reason,
        )

    def _handle_missing_target(self, reason: str) -> ControllerOutput:
        self._lost_frames += 1
        land = self.config.land_on_vision_loss and self._lost_frames > self.config.lost_frame_limit
        command = DroneCommand(
            hover=not land,
            land=land,
            reason="vision_lost_land" if land else "vision_lost_hover",
        )
        return ControllerOutput(
            command=command,
            lookahead_point=None,
            nearest_point=None,
            error_norm=None,
            velocity_norm=None,
            path_progress=None,
            path_remaining_px=None,
            finished=False,
            reason=reason,
        )

    def _velocity_norm(self, target: dict[str, Any], frame_width: float, frame_height: float) -> dict[str, float]:
        velocity_px_s = target.get("velocity_px_s") or {}
        vx = float(velocity_px_s.get("x") or 0.0) / (frame_width / 2.0)
        vy = float(velocity_px_s.get("y") or 0.0) / (frame_height / 2.0)
        return {"x": round(vx, 4), "y": round(vy, 4)}

    def _throttle_power(self, height_cm: float | None) -> int:
        if height_cm is None or height_cm <= 0 or height_cm >= 900:
            return 0
        error_cm = self.config.height_target_cm - float(height_cm)
        if abs(error_cm) < self.config.height_deadband_cm:
            return 0
        return _clamp_int(self.config.kp_height * error_cm, self.config.max_throttle_power)


def nearest_point_on_path(point: Point, path: list[Point]) -> tuple[Point, float]:
    if len(path) < 2:
        raise ValueError("Path must contain at least two points.")

    px, py = point
    best_point = path[0]
    best_progress = 0.0
    best_distance = float("inf")
    progress = 0.0

    for start, end in zip(path, path[1:]):
        projection, ratio = _project_point_to_segment((px, py), start, end)
        distance = float(np.hypot(projection[0] - px, projection[1] - py))
        segment_length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if distance < best_distance:
            best_distance = distance
            best_point = projection
            best_progress = progress + (segment_length * ratio)
        progress += segment_length
    return best_point, best_progress


def point_at_distance(path: list[Point], distance_px: float) -> Point:
    if len(path) < 2:
        raise ValueError("Path must contain at least two points.")

    remaining = max(float(distance_px), 0.0)
    for start, end in zip(path, path[1:]):
        segment_length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if segment_length <= 0:
            continue
        if remaining <= segment_length:
            ratio = remaining / segment_length
            return (
                float(start[0] + ((end[0] - start[0]) * ratio)),
                float(start[1] + ((end[1] - start[1]) * ratio)),
            )
        remaining -= segment_length
    return path[-1]


def _project_point_to_segment(point: Point, start: Point, end: Point) -> tuple[Point, float]:
    sx, sy = start
    ex, ey = end
    px, py = point
    dx = ex - sx
    dy = ey - sy
    denom = (dx * dx) + (dy * dy)
    if denom <= 0:
        return start, 0.0
    ratio = (((px - sx) * dx) + ((py - sy) * dy)) / denom
    ratio = min(max(float(ratio), 0.0), 1.0)
    projection = (float(sx + (dx * ratio)), float(sy + (dy * ratio)))
    return projection, ratio


def _point_to_dict(point: Point | None) -> dict[str, float] | None:
    if point is None:
        return None
    return {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}


def _apply_deadband(value: float, threshold: float) -> float:
    if abs(value) < threshold:
        return 0.0
    return float(value)


def _clamp_int(value: float, limit: int) -> int:
    bounded = max(min(float(value), float(limit)), -float(limit))
    return int(round(bounded))
