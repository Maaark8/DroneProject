from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .live_capture import normalize_capture_source, open_live_capture
from .mission import MissionPath, resample_polyline
from .types import Point
from .waypoints import meters_per_pixel_from_reference


@dataclass(slots=True)
class _ClickSession:
    window_name: str
    prompt_lines: list[str]
    min_points: int
    max_points: int | None = None
    points: list[Point] = field(default_factory=list)

    def run(self, frame: np.ndarray) -> list[Point]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                canvas = self._overlay(frame)
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 10, 32):
                    if len(self.points) >= self.min_points:
                        return list(self.points)
                elif key == ord("u"):
                    if self.points:
                        self.points.pop()
                elif key == ord("c"):
                    self.points.clear()
                elif key in (27, ord("q")):
                    raise ValueError("Manual calibration cancelled.")
        finally:
            cv2.destroyWindow(self.window_name)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.max_points is not None and len(self.points) >= self.max_points:
            self.points[-1] = (float(x), float(y))
            return
        self.points.append((float(x), float(y)))

    def _overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        for index, point in enumerate(self.points):
            px = int(round(point[0]))
            py = int(round(point[1]))
            cv2.circle(overlay, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(
                overlay,
                str(index + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(self.points) >= 2:
            polyline = np.array([[int(round(x)), int(round(y))] for x, y in self.points], dtype=np.int32)
            cv2.polylines(overlay, [polyline], isClosed=False, color=(40, 220, 40), thickness=2)

        status = [*self.prompt_lines, "Left click: add point", "Enter/Space: confirm", "U: undo  C: clear  Q/Esc: cancel"]
        for index, line in enumerate(status):
            cv2.putText(
                overlay,
                line,
                (12, 28 + (index * 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return overlay


def capture_calibration_frame(
    input_path: Path | None = None,
    camera_index: int = 0,
    source: str | int | None = None,
) -> tuple[np.ndarray, str]:
    if input_path is not None:
        if input_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            frame = cv2.imread(str(input_path))
            if frame is None:
                raise ValueError(f"Unable to read image: {input_path}")
            return frame, str(input_path)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open calibration input: {input_path}")
        try:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to read the first frame from {input_path}")
            return frame, str(input_path)
        finally:
            capture.release()

    capture_source = normalize_capture_source(source if source is not None else camera_index)
    capture = open_live_capture(capture_source)
    if not capture.isOpened():
        raise ValueError(f"Unable to open live video source: {capture_source!r}")

    window_name = "manual-calibration-capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Unable to read a live frame for manual calibration.")
            overlay = frame.copy()
            instructions = [
                "Manual calibration capture",
                "Space/Enter: freeze current frame",
                "Q/Esc: cancel",
            ]
            for index, line in enumerate(instructions):
                cv2.putText(
                    overlay,
                    line,
                    (12, 28 + (index * 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):
                return frame.copy(), f"camera:{capture_source}"
            if key in (27, ord("q")):
                raise ValueError("Manual calibration cancelled.")
    finally:
        capture.release()
        cv2.destroyWindow(window_name)


def manual_mission_from_frame(
    frame: np.ndarray,
    source: str,
    reference_distance_cm: float,
    sample_spacing_px: float = 12.0,
) -> tuple[MissionPath, np.ndarray]:
    reference_points = _ClickSession(
        window_name="manual-calibration-reference",
        prompt_lines=[
            "Click two points with known spacing.",
            "Landing pad centers work well here.",
        ],
        min_points=2,
        max_points=2,
    ).run(frame)

    path_points = _ClickSession(
        window_name="manual-calibration-path",
        prompt_lines=[
            "Click the track centerline in order.",
            "Start at the drone end and walk toward the finish.",
        ],
        min_points=2,
    ).run(frame)

    start_points = _ClickSession(
        window_name="manual-calibration-start",
        prompt_lines=[
            "Click the drone start position.",
            "Use the center point where the drone should take off from.",
        ],
        min_points=1,
        max_points=1,
    ).run(frame)
    start_point = start_points[0]

    meters_per_pixel = meters_per_pixel_from_reference(
        (reference_points[0], reference_points[1]),
        reference_distance_cm=reference_distance_cm,
    )
    sampled_points = resample_polyline(path_points, spacing_px=max(float(sample_spacing_px), 1.0))
    mission = MissionPath(
        points=sampled_points,
        frame_size={"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        source_method="manual_click",
        source=source,
        sample_spacing_px=float(sample_spacing_px),
        meters_per_pixel=float(meters_per_pixel),
        start_point_px=(float(start_point[0]), float(start_point[1])),
    )
    overlay = _overlay_manual_mission(frame, reference_points, mission)
    return mission, overlay


def manual_start_point_from_frame(frame: np.ndarray) -> tuple[Point, np.ndarray]:
    start_points = _ClickSession(
        window_name="manual-start-point",
        prompt_lines=[
            "Click the drone center at the start position.",
            "Place the drone aligned with the track start direction.",
        ],
        min_points=1,
        max_points=1,
    ).run(frame)
    start_point = start_points[0]
    overlay = frame.copy()
    cv2.drawMarker(
        overlay,
        (int(round(start_point[0])), int(round(start_point[1]))),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        24,
        2,
    )
    return start_point, overlay


def _overlay_manual_mission(frame: np.ndarray, reference_points: list[Point], mission: MissionPath) -> np.ndarray:
    overlay = frame.copy()
    for point in reference_points:
        cv2.circle(overlay, (int(round(point[0])), int(round(point[1]))), 7, (0, 0, 255), -1)
    cv2.line(
        overlay,
        (int(round(reference_points[0][0])), int(round(reference_points[0][1]))),
        (int(round(reference_points[1][0])), int(round(reference_points[1][1]))),
        (0, 0, 255),
        2,
    )
    if mission.points:
        polyline = np.array([[int(round(x)), int(round(y))] for x, y in mission.points], dtype=np.int32)
        cv2.polylines(overlay, [polyline], isClosed=False, color=(40, 220, 40), thickness=2)
    if mission.start_point_px is not None:
        cv2.drawMarker(
            overlay,
            (int(round(mission.start_point_px[0])), int(round(mission.start_point_px[1]))),
            (0, 140, 255),
            cv2.MARKER_CROSS,
            24,
            2,
        )
    cv2.putText(
        overlay,
        f"meters_per_pixel={mission.meters_per_pixel:.5f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay
