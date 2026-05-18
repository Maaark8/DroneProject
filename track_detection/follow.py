from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from detectors.factory import create_detector

from .codrone_adapter import CoDroneEDUAdapter, FlightAdapter, NullFlightAdapter
from .control_output import to_control_observation
from .controller import ControllerOutput, TrackFollowerConfig, TrackFollowerController
from .io import ensure_directory, iter_image_files, result_to_payload
from .live_capture import normalize_capture_source, open_live_capture
from .mission import MissionPath, mission_path_from_result
from .types import DetectionResult, FrameInput, Point

TRACK_METHODS = tuple(method for method in ("threshold_morph", "edge_geometry", "wood_path", "segmentation"))
DRONE_METHODS = tuple(method for method in ("drone_light", "drone_marker"))


def export_mission_path(
    method: str,
    input_path: Path,
    output_path: Path,
    reverse_path: bool = False,
    max_frames: int = 120,
    sample_spacing_px: float = 12.0,
) -> MissionPath:
    if method not in TRACK_METHODS:
        raise ValueError(f"Mission export requires a track detector, got {method!r}.")

    detector = create_detector(method)
    best = _best_detection_result(detector, input_path, max_frames=max_frames)
    frame_size = _frame_size(best)
    mission = mission_path_from_result(
        best,
        frame_size=frame_size,
        source=str(input_path),
        reverse=reverse_path,
        sample_spacing_px=sample_spacing_px,
    )
    mission.save(output_path)
    return mission


def follow_track_live(
    mission_path: Path,
    camera_index: int = 0,
    source: str | int | None = None,
    output_dir: Path | None = None,
    display: bool = True,
    max_frames: int | None = None,
    command_rate_hz: float = 10.0,
    controller_config: TrackFollowerConfig | None = None,
    dry_run: bool = False,
    drone_method: str = "drone_light",
) -> None:
    mission = MissionPath.load(mission_path)
    if drone_method not in DRONE_METHODS:
        raise ValueError(f"Unknown drone detector method: {drone_method!r}")
    capture_source = normalize_capture_source(source if source is not None else camera_index)
    capture = open_live_capture(capture_source)
    if not capture.isOpened():
        raise ValueError(
            f"Unable to open live video source: {capture_source!r}. "
            "Use --source with a camera index, video path, or IP camera URL."
        )

    try:
        _run_follow_loop(
            capture=capture,
            capture_source=capture_source,
            mission=mission,
            output_dir=output_dir,
            display=display,
            max_frames=max_frames,
            command_rate_hz=command_rate_hz,
            controller_config=controller_config,
            dry_run=dry_run,
            initial_frame_id=0,
            window_name="follow-track",
            log_filename="follow_log.jsonl",
            drone_method=drone_method,
        )
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def auto_follow_track(
    method: str,
    camera_index: int = 0,
    source: str | int | None = None,
    output_dir: Path | None = None,
    display: bool = True,
    max_frames: int | None = None,
    command_rate_hz: float = 10.0,
    controller_config: TrackFollowerConfig | None = None,
    dry_run: bool = False,
    calibration_frames: int = 30,
    sample_spacing_px: float = 12.0,
    reverse_path: bool = False,
    auto_orient: bool = True,
    drone_method: str = "drone_light",
) -> None:
    if method not in TRACK_METHODS:
        raise ValueError(f"Auto follow requires a track detector, got {method!r}.")
    if drone_method not in DRONE_METHODS:
        raise ValueError(f"Unknown drone detector method: {drone_method!r}.")

    track_detector = create_detector(method)
    drone_detector = create_detector(drone_method)
    capture_source = normalize_capture_source(source if source is not None else camera_index)
    capture = open_live_capture(capture_source)
    if not capture.isOpened():
        raise ValueError(
            f"Unable to open live video source: {capture_source!r}. "
            "Use --source with a camera index, video path, or IP camera URL."
        )

    try:
        mission, initial_frame_id = _calibrate_mission_from_capture(
            capture=capture,
            capture_source=capture_source,
            track_detector=track_detector,
            drone_detector=drone_detector,
            method=method,
            calibration_frames=calibration_frames,
            sample_spacing_px=sample_spacing_px,
            reverse_path=reverse_path,
            auto_orient=auto_orient,
            output_dir=output_dir,
        )
        _run_follow_loop(
            capture=capture,
            capture_source=capture_source,
            mission=mission,
            output_dir=output_dir,
            display=display,
            max_frames=max_frames,
            command_rate_hz=command_rate_hz,
            controller_config=controller_config,
            dry_run=dry_run,
            initial_frame_id=initial_frame_id,
            window_name="auto-follow-track",
            log_filename="follow_log.jsonl",
            drone_method=drone_method,
        )
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def bridge_track_follow(
    method: str,
    camera_index: int = 0,
    source: str | int | None = None,
    output_dir: Path | None = None,
    display: bool = True,
    max_frames: int | None = None,
    command_rate_hz: float = 10.0,
    controller_config: TrackFollowerConfig | None = None,
    dry_run: bool = False,
    sample_spacing_px: float = 12.0,
    reverse_path: bool = False,
    auto_orient: bool = True,
    drone_method: str = "drone_light",
    redetect_every: int = 5,
    warmup_frames: int = 60,
    rise_fraction: float = 0.25,
    coast_frames: int = 8,
) -> None:
    """Bridge live track detection straight to drone control.

    Unlike ``auto_follow_track`` (which calibrates one fixed mission then flies
    it), this re-runs the track detector every ``redetect_every`` frames and
    swaps the controller's path live, so the drone follows the track as it is
    currently seen rather than a snapshot.
    """
    if method not in TRACK_METHODS:
        raise ValueError(f"Bridge follow requires a track detector, got {method!r}.")
    if drone_method not in DRONE_METHODS:
        raise ValueError(f"Unknown drone detector method: {drone_method!r}.")

    track_detector = create_detector(method)
    drone_detector = create_detector(drone_method)
    capture_source = normalize_capture_source(source if source is not None else camera_index)
    capture = open_live_capture(capture_source)
    if not capture.isOpened():
        raise ValueError(
            f"Unable to open live video source: {capture_source!r}. "
            "Use --source with a camera index, video path, or IP camera URL."
        )

    try:
        _run_bridge_loop(
            capture=capture,
            capture_source=capture_source,
            track_detector=track_detector,
            drone_detector=drone_detector,
            output_dir=output_dir,
            display=display,
            max_frames=max_frames,
            command_rate_hz=command_rate_hz,
            controller_config=controller_config,
            dry_run=dry_run,
            sample_spacing_px=sample_spacing_px,
            reverse_path=reverse_path,
            auto_orient=auto_orient,
            redetect_every=max(int(redetect_every), 1),
            warmup_frames=max(int(warmup_frames), 1),
            rise_fraction=min(max(float(rise_fraction), 0.05), 1.0),
            coast_frames=max(int(coast_frames), 0),
        )
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def _build_oriented_mission(
    track_result: DetectionResult,
    capture_source: str | int,
    reverse_path: bool,
    auto_orient: bool,
    drone_point: Point | None,
    sample_spacing_px: float,
    fallback_frame_size: dict[str, int] | None,
) -> MissionPath:
    should_reverse = reverse_path
    if not reverse_path and auto_orient and drone_point is not None:
        should_reverse = _should_reverse_path(track_result.centerline, drone_point)
    return mission_path_from_result(
        track_result,
        frame_size=_frame_size(track_result) or fallback_frame_size,
        source=f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source),
        reverse=should_reverse,
        sample_spacing_px=sample_spacing_px,
    )


def _synthetic_drone_observation(point: Point, frame_size: dict[str, int] | None) -> dict:
    """Build a controller observation from a remembered drone position.

    Used to coast through brief marker dropouts so following does not stall on
    a single missed detection.
    """
    return {
        "schema_version": "drone_control_observation.v1",
        "source_method": "coast",
        "valid": True,
        "confidence": 1.0,
        "frame_size": frame_size,
        "target": {
            "kind": "drone",
            "position_px": {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)},
            "velocity_px_s": {"x": 0.0, "y": 0.0},
        },
    }


def _run_bridge_loop(
    capture,
    capture_source: str | int,
    track_detector,
    drone_detector,
    output_dir: Path | None,
    display: bool,
    max_frames: int | None,
    command_rate_hz: float,
    controller_config: TrackFollowerConfig | None,
    dry_run: bool,
    sample_spacing_px: float,
    reverse_path: bool,
    auto_orient: bool,
    redetect_every: int,
    warmup_frames: int,
    rise_fraction: float,
    coast_frames: int,
) -> None:
    adapter: FlightAdapter = NullFlightAdapter() if dry_run else CoDroneEDUAdapter()
    command_duration_s = 1.0 / max(float(command_rate_hz), 1.0)
    started = time.perf_counter()
    result_handle = None
    if output_dir is not None:
        ensure_directory(output_dir)
        result_handle = (output_dir / "bridge_log.jsonl").open("w", encoding="utf-8")

    source_label = f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source)
    last_drone_point: Point | None = None
    frames_since_drone = 0

    try:
        adapter.connect()

        # Warm up: read frames until the track detector yields a usable path.
        mission: MissionPath | None = None
        frame_id = 0
        while frame_id < warmup_frames:
            ok, frame = capture.read()
            if not ok:
                break
            track_result = track_detector.detect(FrameInput(frame=frame, frame_id=frame_id))
            track_result.metadata["source"] = source_label
            if track_result.valid and track_result.centerline:
                drone_result = drone_detector.detect(FrameInput(frame=frame, frame_id=frame_id))
                last_drone_point = _drone_point(drone_result) or last_drone_point
                mission = _build_oriented_mission(
                    track_result,
                    capture_source,
                    reverse_path,
                    auto_orient,
                    last_drone_point,
                    sample_spacing_px,
                    None,
                )
                break
            frame_id += 1

        if mission is None:
            raise ValueError(
                f"No valid track detected in the first {warmup_frames} live frame(s); "
                "aim the camera at the track before bridging."
            )

        config = controller_config or TrackFollowerConfig()
        config.height_target_cm = round(float(config.height_target_cm) * rise_fraction, 1)
        controller = TrackFollowerController(mission=mission, config=config)
        adapter.takeoff()

        controlled_frames = 0
        while True:
            loop_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s = time.perf_counter() - started

            track_result = None
            if frame_id % redetect_every == 0:
                track_result = track_detector.detect(
                    FrameInput(frame=frame, frame_id=frame_id, timestamp_s=timestamp_s)
                )
                track_result.metadata["source"] = source_label
                if track_result.valid and track_result.centerline:
                    try:
                        controller.mission = _build_oriented_mission(
                            track_result,
                            capture_source,
                            reverse_path,
                            auto_orient,
                            last_drone_point,
                            sample_spacing_px,
                            controller.mission.frame_size,
                        )
                    except ValueError:
                        pass  # keep the last good path for this frame

            drone_result = drone_detector.detect(
                FrameInput(frame=frame, frame_id=frame_id, timestamp_s=timestamp_s)
            )
            drone_result.metadata["source"] = source_label
            drone_point = _drone_point(drone_result)
            if drone_point is not None:
                last_drone_point = drone_point
                frames_since_drone = 0
                observation = to_control_observation(drone_result)
            else:
                frames_since_drone += 1
                if last_drone_point is not None and frames_since_drone <= coast_frames:
                    observation = _synthetic_drone_observation(
                        last_drone_point, drone_result.metadata.get("frame_size")
                    )
                else:
                    observation = to_control_observation(drone_result)

            height_cm = adapter.get_height_cm()
            control = controller.update(observation, height_cm=height_cm)
            adapter.send_command(control.command, duration_s=command_duration_s)

            if result_handle is not None:
                payload = result_to_payload(drone_result)
                payload["track_follow"] = control.as_dict()
                payload["track_follow"]["height_cm"] = None if height_cm is None else round(float(height_cm), 2)
                payload["track_redetected"] = track_result is not None and track_result.valid
                payload["mission_length_px"] = round(float(controller.mission.path_length_px), 2)
                payload["coasting"] = drone_point is None and 0 < frames_since_drone <= coast_frames
                result_handle.write(json.dumps(payload) + "\n")
                result_handle.flush()

            # The debug PNG is heavy; only write it on re-detect frames to cut I/O lag.
            want_disk = output_dir is not None and (frame_id % redetect_every == 0)
            if display or want_disk:
                if track_result is not None and track_result.debug_frame is not None:
                    base_frame = track_result.debug_frame
                elif drone_result.debug_frame is not None:
                    base_frame = drone_result.debug_frame
                else:
                    base_frame = frame
                debug_frame = _overlay_follow_debug(base_frame, controller.mission, control, height_cm=height_cm)
                if display:
                    cv2.imshow("bridge-follow", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                if want_disk:
                    cv2.imwrite(str(output_dir / "latest_bridge_debug.png"), debug_frame)

            frame_id += 1
            controlled_frames += 1
            if control.command.land or (max_frames is not None and controlled_frames >= max_frames):
                break

            elapsed = time.perf_counter() - loop_started
            if dry_run and elapsed < command_duration_s:
                time.sleep(command_duration_s - elapsed)
    finally:
        try:
            adapter.land()
        finally:
            adapter.close()
            if result_handle is not None:
                result_handle.close()


def _best_detection_result(detector, input_path: Path, max_frames: int) -> DetectionResult:
    if input_path.is_dir():
        best: DetectionResult | None = None
        for frame_id, image_path in enumerate(iter_image_files(input_path)):
            frame = cv2.imread(str(image_path))
            result = detector.detect(FrameInput(frame=frame, frame_id=frame_id))
            result.metadata["source"] = str(image_path)
            if best is None or result.confidence > best.confidence:
                best = result
        if best is None:
            raise ValueError(f"No images found in {input_path}.")
        if not best.centerline:
            raise ValueError(f"No valid track centerline detected in {input_path}.")
        return best

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open mission source: {input_path}")

    best: DetectionResult | None = None
    try:
        frame_id = 0
        while frame_id < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.detect(FrameInput(frame=frame, frame_id=frame_id))
            result.metadata["source"] = str(input_path)
            if best is None or result.confidence > best.confidence:
                best = result
            frame_id += 1
    finally:
        capture.release()

    if best is None or not best.centerline:
        raise ValueError(f"No valid track centerline found in the first {max_frames} frame(s) of {input_path}.")
    return best


def _frame_size(result: DetectionResult) -> dict[str, int] | None:
    frame_size = result.metadata.get("frame_size")
    if frame_size is not None:
        return frame_size
    if result.debug_frame is None:
        return None
    return {"width": int(result.debug_frame.shape[1]), "height": int(result.debug_frame.shape[0])}


def _run_follow_loop(
    capture,
    capture_source: str | int,
    mission: MissionPath,
    output_dir: Path | None,
    display: bool,
    max_frames: int | None,
    command_rate_hz: float,
    controller_config: TrackFollowerConfig | None,
    dry_run: bool,
    initial_frame_id: int,
    window_name: str,
    log_filename: str,
    drone_method: str,
) -> None:
    controller = TrackFollowerController(mission=mission, config=controller_config or TrackFollowerConfig())
    detector = create_detector(drone_method)
    adapter: FlightAdapter = NullFlightAdapter() if dry_run else CoDroneEDUAdapter()
    result_handle = None
    command_duration_s = 1.0 / max(float(command_rate_hz), 1.0)
    started = time.perf_counter()

    if output_dir is not None:
        ensure_directory(output_dir)
        result_handle = (output_dir / log_filename).open("w", encoding="utf-8")

    try:
        adapter.connect()
        adapter.takeoff()
        frame_id = initial_frame_id
        controlled_frames = 0
        while True:
            loop_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_s = time.perf_counter() - started
            result = detector.detect(FrameInput(frame=frame, frame_id=frame_id, timestamp_s=timestamp_s))
            result.metadata["source"] = f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source)
            observation = to_control_observation(result)
            height_cm = adapter.get_height_cm()
            control = controller.update(observation, height_cm=height_cm)
            adapter.send_command(control.command, duration_s=command_duration_s)

            if result_handle is not None:
                payload = result_to_payload(result)
                payload["track_follow"] = control.as_dict()
                payload["track_follow"]["height_cm"] = None if height_cm is None else round(float(height_cm), 2)
                result_handle.write(json.dumps(payload) + "\n")
                result_handle.flush()

            if display or output_dir is not None:
                debug_frame = _overlay_follow_debug(
                    result.debug_frame if result.debug_frame is not None else frame,
                    mission,
                    control,
                    height_cm=height_cm,
                )
                if display:
                    cv2.imshow(window_name, debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                if output_dir is not None:
                    cv2.imwrite(str(output_dir / "latest_follow_debug.png"), debug_frame)

            frame_id += 1
            controlled_frames += 1
            if control.command.land or (max_frames is not None and controlled_frames >= max_frames):
                break

            elapsed = time.perf_counter() - loop_started
            if dry_run and elapsed < command_duration_s:
                time.sleep(command_duration_s - elapsed)
    finally:
        try:
            adapter.land()
        finally:
            adapter.close()
            if result_handle is not None:
                result_handle.close()


def _calibrate_mission_from_capture(
    capture,
    capture_source: str | int,
    track_detector,
    drone_detector,
    method: str,
    calibration_frames: int,
    sample_spacing_px: float,
    reverse_path: bool,
    auto_orient: bool,
    output_dir: Path | None,
) -> tuple[MissionPath, int]:
    best_track: DetectionResult | None = None
    best_track_frame: np.ndarray | None = None
    best_drone: DetectionResult | None = None
    best_drone_frame: np.ndarray | None = None
    frame_id = 0

    while frame_id < max(int(calibration_frames), 1):
        ok, frame = capture.read()
        if not ok:
            break

        track_result = track_detector.detect(FrameInput(frame=frame, frame_id=frame_id))
        track_result.metadata["source"] = f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source)
        if _prefer_result(track_result, best_track):
            best_track = track_result
            best_track_frame = frame.copy()

        drone_result = drone_detector.detect(FrameInput(frame=frame, frame_id=frame_id))
        drone_result.metadata["source"] = f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source)
        if _prefer_result(drone_result, best_drone):
            best_drone = drone_result
            best_drone_frame = frame.copy()

        frame_id += 1

    if best_track is None or not best_track.centerline:
        raise ValueError(f"No valid track centerline found in the first {calibration_frames} live frame(s).")

    should_reverse = reverse_path
    if not reverse_path and auto_orient and best_drone is not None and best_drone.valid:
        drone_position = _drone_point(best_drone)
        if drone_position is not None:
            should_reverse = _should_reverse_path(best_track.centerline, drone_position)

    mission = mission_path_from_result(
        best_track,
        frame_size=_frame_size(best_track),
        source=f"camera:{capture_source}" if isinstance(capture_source, int) else str(capture_source),
        reverse=should_reverse,
        sample_spacing_px=sample_spacing_px,
    )

    if output_dir is not None:
        ensure_directory(output_dir)
        mission.save(output_dir / "mission_path.json")
        if best_track_frame is not None:
            cv2.imwrite(str(output_dir / "calibration_frame.png"), best_track_frame)
        if best_track.debug_frame is not None:
            cv2.imwrite(str(output_dir / "calibration_track_debug.png"), best_track.debug_frame)
        if best_drone is not None and best_drone.debug_frame is not None:
            cv2.imwrite(str(output_dir / "calibration_drone_debug.png"), best_drone.debug_frame)
        calibration_meta = {
            "track_method": method,
            "track_confidence": round(float(best_track.confidence), 4),
            "track_frame_id": best_track.metadata.get("frame_id"),
            "drone_confidence": None if best_drone is None else round(float(best_drone.confidence), 4),
            "drone_frame_id": None if best_drone is None else best_drone.metadata.get("frame_id"),
            "path_reversed": should_reverse,
            "auto_orient_used": bool(auto_orient and best_drone is not None and best_drone.valid and not reverse_path),
        }
        (output_dir / "calibration_summary.json").write_text(json.dumps(calibration_meta, indent=2) + "\n", encoding="utf-8")

    return mission, frame_id


def _overlay_follow_debug(
    frame: np.ndarray,
    mission: MissionPath,
    control: ControllerOutput,
    height_cm: float | None,
) -> np.ndarray:
    overlay = frame.copy()
    if mission.points:
        polyline = np.array([[int(round(x)), int(round(y))] for x, y in mission.points], dtype=np.int32)
        cv2.polylines(overlay, [polyline], isClosed=False, color=(255, 220, 0), thickness=2)

    if control.nearest_point is not None:
        cv2.circle(
            overlay,
            (int(round(control.nearest_point[0])), int(round(control.nearest_point[1]))),
            4,
            (0, 255, 255),
            -1,
        )
    if control.lookahead_point is not None:
        cv2.circle(
            overlay,
            (int(round(control.lookahead_point[0])), int(round(control.lookahead_point[1]))),
            7,
            (0, 140, 255),
            -1,
        )

    command = control.command
    lines = [
        f"cmd r={command.roll} p={command.pitch} y={command.yaw} t={command.throttle}",
        f"reason={command.reason} remaining={control.path_remaining_px}",
    ]
    if control.error_norm is not None:
        lines.append(f"error x={control.error_norm['x']:.3f} y={control.error_norm['y']:.3f}")
    if height_cm is not None:
        lines.append(f"height_cm={height_cm:.1f}")

    for index, text in enumerate(lines):
        cv2.putText(
            overlay,
            text,
            (12, 28 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _prefer_result(candidate: DetectionResult, current: DetectionResult | None) -> bool:
    if not candidate.valid or not candidate.centerline:
        return False
    if current is None:
        return True
    return (candidate.confidence, len(candidate.centerline)) > (current.confidence, len(current.centerline))


def _drone_point(result: DetectionResult) -> Point | None:
    position = result.metadata.get("position_px")
    if position is None:
        return None
    x = position.get("x")
    y = position.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def _should_reverse_path(path: list[Point], drone_position: Point) -> bool:
    if not path:
        return False
    start_distance = float(np.hypot(path[0][0] - drone_position[0], path[0][1] - drone_position[1]))
    end_distance = float(np.hypot(path[-1][0] - drone_position[0], path[-1][1] - drone_position[1]))
    return end_distance < start_distance
