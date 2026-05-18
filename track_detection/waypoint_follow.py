from __future__ import annotations

import json
from pathlib import Path

import cv2

from detectors.factory import create_detector

from .codrone_adapter import CoDroneEDUAdapter, FlightAdapter, NullFlightAdapter
from .follow import DRONE_METHODS, _drone_point, _prefer_result, _should_reverse_path
from .io import ensure_directory
from .live_capture import normalize_capture_source, open_live_capture
from .mission import MissionPath
from .types import DetectionResult, FrameInput
from .manual_calibration import capture_calibration_frame, manual_start_point_from_frame
from .waypoints import (
    WaypointMission,
    marker_forward_heading_rad,
    overlay_waypoint_plan,
    path_start_heading_rad,
    waypoint_mission_from_path,
)


def follow_waypoints_live(
    mission_path: Path,
    camera_index: int = 0,
    source: str | int | None = None,
    output_dir: Path | None = None,
    calibration_frames: int = 45,
    drone_method: str = "drone_marker",
    target_height_m: float = 0.8,
    waypoint_spacing_px: float = 120.0,
    speed_m_s: float = 0.8,
    meters_per_pixel: float | None = None,
    position_tolerance_m: float = 0.15,
    waypoint_timeout_s: float = 3.0,
    reverse_path: bool = False,
    auto_orient: bool = True,
    dry_run: bool = False,
) -> WaypointMission:
    mission = MissionPath.load(mission_path)
    scale = float(meters_per_pixel if meters_per_pixel is not None else mission.meters_per_pixel or 0.0)
    if scale <= 0:
        raise ValueError("Waypoint flight requires a positive meters-per-pixel scale. Use manual-mission or --meters-per-pixel.")
    if drone_method not in DRONE_METHODS:
        raise ValueError(f"Unknown drone detector method: {drone_method!r}.")

    capture_source = normalize_capture_source(source if source is not None else camera_index)
    capture = open_live_capture(capture_source)
    if not capture.isOpened():
        raise ValueError(
            f"Unable to open live video source: {capture_source!r}. "
            "Use --source with a camera index, video path, or IP camera URL."
        )

    try:
        drone_result, frame = _calibrate_drone_pose(
            capture=capture,
            capture_source=capture_source,
            drone_method=drone_method,
            calibration_frames=calibration_frames,
        )
    finally:
        capture.release()

    drone_position = _drone_point(drone_result)
    if drone_position is None:
        raise ValueError("Drone calibration did not produce a valid start position.")
    forward_heading = marker_forward_heading_rad(drone_result.metadata.get("heading_rad"))
    if forward_heading is None:
        raise ValueError(
            "Waypoint flight requires drone heading from the ArUco marker. Use --drone-method drone_marker."
        )

    if not reverse_path and auto_orient:
        reverse_path = _should_reverse_path(mission.points, drone_position)
    if reverse_path:
        mission = MissionPath(
            points=list(reversed(mission.points)),
            frame_size=mission.frame_size,
            source_method=mission.source_method,
            source=mission.source,
            schema_version=mission.schema_version,
            sample_spacing_px=mission.sample_spacing_px,
            meters_per_pixel=scale,
        )

    waypoint_mission = waypoint_mission_from_path(
        mission=mission,
        drone_start_px=drone_position,
        drone_forward_rad=forward_heading,
        meters_per_pixel=scale,
        target_height_m=target_height_m,
        waypoint_spacing_px=waypoint_spacing_px,
        speed_m_s=speed_m_s,
    )

    if output_dir is not None:
        ensure_directory(output_dir)
        if drone_result.debug_frame is not None:
            cv2.imwrite(str(output_dir / "waypoint_drone_debug.png"), drone_result.debug_frame)
        cv2.imwrite(str(output_dir / "waypoint_calibration_frame.png"), frame)
        cv2.imwrite(str(output_dir / "waypoint_plan_overlay.png"), overlay_waypoint_plan(frame, mission, waypoint_mission))
        waypoint_mission.save(output_dir / "waypoint_mission.json")

    _execute_waypoint_mission(
        waypoint_mission=waypoint_mission,
        output_dir=output_dir,
        dry_run=dry_run,
        position_tolerance_m=position_tolerance_m,
        waypoint_timeout_s=waypoint_timeout_s,
    )
    return waypoint_mission


def follow_waypoints_manual_start(
    mission_path: Path,
    output_dir: Path | None = None,
    input_path: Path | None = None,
    camera_index: int = 0,
    source: str | int | None = None,
    target_height_m: float = 0.8,
    waypoint_spacing_px: float = 120.0,
    speed_m_s: float = 0.8,
    meters_per_pixel: float | None = None,
    position_tolerance_m: float = 0.15,
    waypoint_timeout_s: float = 3.0,
    reverse_path: bool = False,
    start_x: float | None = None,
    start_y: float | None = None,
    dry_run: bool = False,
) -> WaypointMission:
    mission = MissionPath.load(mission_path)
    scale = float(meters_per_pixel if meters_per_pixel is not None else mission.meters_per_pixel or 0.0)
    if scale <= 0:
        raise ValueError("Waypoint flight requires a positive meters-per-pixel scale. Use manual-mission or --meters-per-pixel.")

    if reverse_path:
        mission = MissionPath(
            points=list(reversed(mission.points)),
            frame_size=mission.frame_size,
            source_method=mission.source_method,
            source=mission.source,
            schema_version=mission.schema_version,
            sample_spacing_px=mission.sample_spacing_px,
            meters_per_pixel=scale,
            start_point_px=mission.start_point_px,
        )

    frame = None
    source_label = mission.source or "mission"
    stored_start = mission.start_point_px
    needs_manual_start = start_x is None or start_y is None
    if needs_manual_start and stored_start is not None:
        start_x = float(stored_start[0])
        start_y = float(stored_start[1])
        needs_manual_start = False

    start_point_source = "mission" if stored_start is not None and not needs_manual_start else "manual"
    should_capture_frame = needs_manual_start or input_path is not None or source is not None
    if should_capture_frame:
        frame, source_label = capture_calibration_frame(
            input_path=input_path,
            camera_index=camera_index,
            source=source,
        )

    if needs_manual_start:
        if frame is None:
            raise ValueError(
                "Manual start requires a start point from the mission, --start-x/--start-y, or a capture source for clicking."
            )
        start_point, start_overlay = manual_start_point_from_frame(frame)
    else:
        start_point = (float(start_x), float(start_y))
        start_overlay = None if frame is None else frame.copy()
        if stored_start is None:
            start_point_source = "cli"
        if start_overlay is not None:
            cv2.drawMarker(
                start_overlay,
                (int(round(start_point[0])), int(round(start_point[1]))),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )

    forward_heading = path_start_heading_rad(mission.points)
    waypoint_mission = waypoint_mission_from_path(
        mission=mission,
        drone_start_px=start_point,
        drone_forward_rad=forward_heading,
        meters_per_pixel=scale,
        target_height_m=target_height_m,
        waypoint_spacing_px=waypoint_spacing_px,
        speed_m_s=speed_m_s,
    )

    if output_dir is not None:
        ensure_directory(output_dir)
        if frame is not None:
            cv2.imwrite(str(output_dir / "manual_start_frame.png"), frame)
            if start_overlay is not None:
                cv2.imwrite(str(output_dir / "manual_start_overlay.png"), start_overlay)
            cv2.imwrite(str(output_dir / "manual_start_plan_overlay.png"), overlay_waypoint_plan(frame, mission, waypoint_mission))
        waypoint_mission.save(output_dir / "waypoint_mission.json")
        summary = {
            "source": source_label,
            "start_point_px": {"x": round(float(start_point[0]), 2), "y": round(float(start_point[1]), 2)},
            "forward_heading_rad": round(float(forward_heading), 6),
            "meters_per_pixel": round(float(scale), 6),
            "start_point_source": start_point_source,
        }
        (output_dir / "manual_start_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _execute_waypoint_mission(
        waypoint_mission=waypoint_mission,
        output_dir=output_dir,
        dry_run=dry_run,
        position_tolerance_m=position_tolerance_m,
        waypoint_timeout_s=waypoint_timeout_s,
    )
    return waypoint_mission


def _calibrate_drone_pose(
    capture,
    capture_source: str | int,
    drone_method: str,
    calibration_frames: int,
) -> tuple[DetectionResult, object]:
    detector = create_detector(drone_method)
    best: DetectionResult | None = None
    best_frame = None
    frame_id = 0

    while frame_id < max(int(calibration_frames), 1):
        ok, frame = capture.read()
        if not ok:
            break
        result = detector.detect(FrameInput(frame=frame, frame_id=frame_id))
        result.metadata["source"] = f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source)
        if _prefer_result(result, best):
            best = result
            best_frame = frame.copy()
        frame_id += 1

    if best is None or not best.valid or _drone_point(best) is None:
        raise ValueError(f"No valid drone pose found in the first {calibration_frames} live frame(s).")
    return best, best_frame


def _execute_waypoint_mission(
    waypoint_mission: WaypointMission,
    output_dir: Path | None,
    dry_run: bool,
    position_tolerance_m: float,
    waypoint_timeout_s: float,
) -> None:
    adapter: FlightAdapter = NullFlightAdapter() if dry_run else CoDroneEDUAdapter()
    log_handle = None
    if output_dir is not None:
        log_handle = (output_dir / "waypoint_execution.jsonl").open("w", encoding="utf-8")

    try:
        adapter.connect()
        adapter.takeoff()
        for index, waypoint in enumerate(waypoint_mission.waypoints):
            status = adapter.fly_waypoint(waypoint, tolerance_m=position_tolerance_m, timeout_s=waypoint_timeout_s)
            if log_handle is not None:
                payload = {
                    "index": index,
                    "waypoint": waypoint.to_dict(),
                    "status": status,
                }
                log_handle.write(json.dumps(payload) + "\n")
                log_handle.flush()
    finally:
        try:
            adapter.land()
        finally:
            adapter.close()
            if log_handle is not None:
                log_handle.close()
