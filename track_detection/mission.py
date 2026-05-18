from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import DetectionResult, Point

MISSION_SCHEMA_VERSION = "track_mission_path.v1"


@dataclass(slots=True)
class MissionPath:
    points: list[Point]
    frame_size: dict[str, int] | None
    source_method: str
    source: str | None = None
    schema_version: str = MISSION_SCHEMA_VERSION
    sample_spacing_px: float | None = None
    meters_per_pixel: float | None = None
    start_point_px: Point | None = None

    @property
    def path_length_px(self) -> float:
        return _polyline_length(self.points)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_method": self.source_method,
            "source": self.source,
            "frame_size": self.frame_size,
            "sample_spacing_px": self.sample_spacing_px,
            "meters_per_pixel": self.meters_per_pixel,
            "start_point_px": None
            if self.start_point_px is None
            else {"x": round(float(self.start_point_px[0]), 2), "y": round(float(self.start_point_px[1]), 2)},
            "path_length_px": round(self.path_length_px, 2),
            "points": [{"x": round(float(x), 2), "y": round(float(y), 2)} for x, y in self.points],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MissionPath":
        points = [(float(point["x"]), float(point["y"])) for point in payload["points"]]
        start_point_payload = payload.get("start_point_px")
        start_point_px = None
        if start_point_payload is not None:
            start_point_px = (float(start_point_payload["x"]), float(start_point_payload["y"]))
        return cls(
            points=points,
            frame_size=payload.get("frame_size"),
            source_method=payload.get("source_method", "unknown"),
            source=payload.get("source"),
            schema_version=payload.get("schema_version", MISSION_SCHEMA_VERSION),
            sample_spacing_px=payload.get("sample_spacing_px"),
            meters_per_pixel=payload.get("meters_per_pixel"),
            start_point_px=start_point_px,
        )

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "MissionPath":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def mission_path_from_result(
    result: DetectionResult,
    frame_size: dict[str, int] | None = None,
    source: str | None = None,
    reverse: bool = False,
    sample_spacing_px: float = 12.0,
) -> MissionPath:
    if not result.centerline:
        raise ValueError(f"Detector {result.method!r} did not produce a centerline.")

    points = list(result.centerline)
    if reverse:
        points.reverse()
    sampled = resample_polyline(points, spacing_px=sample_spacing_px)
    if len(sampled) < 2:
        raise ValueError("Mission path requires at least two sampled points.")
    return MissionPath(
        points=sampled,
        frame_size=frame_size or result.metadata.get("frame_size"),
        source_method=result.method,
        source=source,
        sample_spacing_px=sample_spacing_px,
    )


def resample_polyline(points: list[Point], spacing_px: float) -> list[Point]:
    if len(points) < 2:
        return list(points)
    spacing = max(float(spacing_px), 1.0)
    pts = np.array(points, dtype=np.float32)
    deltas = pts[1:] - pts[:-1]
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 0:
        return [tuple(map(float, points[0]))]

    targets = np.arange(0.0, total_length, spacing, dtype=np.float32)
    if not np.isclose(targets[-1] if len(targets) > 0 else -1.0, total_length):
        targets = np.append(targets, total_length)

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sampled: list[Point] = []

    for target in targets:
        segment_index = min(int(np.searchsorted(cumulative, target, side="right")) - 1, len(segment_lengths) - 1)
        segment_index = max(segment_index, 0)
        start_distance = float(cumulative[segment_index])
        length = float(segment_lengths[segment_index])
        ratio = 0.0 if length <= 0 else (float(target) - start_distance) / length
        point = pts[segment_index] + (pts[segment_index + 1] - pts[segment_index]) * ratio
        sampled.append((float(point[0]), float(point[1])))
    return sampled


def _polyline_length(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    deltas = pts[1:] - pts[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())
