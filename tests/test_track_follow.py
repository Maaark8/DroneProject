from __future__ import annotations

import cv2
import numpy as np

from detectors.threshold_morph.detector import ThresholdMorphDetector
from track_detection.codrone_adapter import NullFlightAdapter
from track_detection.controller import TrackFollowerConfig, TrackFollowerController
from track_detection.follow import _should_reverse_path
from track_detection.mission import MissionPath, mission_path_from_result
from track_detection.types import FrameInput
from track_detection.waypoint_follow import follow_waypoints_manual_start
from track_detection.waypoints import (
    Waypoint,
    marker_forward_heading_rad,
    meters_per_pixel_from_reference,
    path_start_heading_rad,
    waypoint_mission_from_path,
)


def make_synthetic_track_frame(x_shift: int = 0) -> np.ndarray:
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    points = np.array(
        [
            [320 + x_shift, 470],
            [325 + x_shift, 380],
            [310 + x_shift, 290],
            [335 + x_shift, 200],
            [320 + x_shift, 80],
        ],
        dtype=np.int32,
    )
    cv2.polylines(frame, [points], isClosed=False, color=(70, 110, 170), thickness=90)
    cv2.polylines(frame, [points], isClosed=False, color=(40, 60, 90), thickness=14)
    return frame


def test_mission_path_from_detector_result_round_trips(tmp_path) -> None:
    detector = ThresholdMorphDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_track_frame()))

    mission = mission_path_from_result(
        result,
        frame_size={"width": 640, "height": 480},
        source="synthetic.png",
        sample_spacing_px=16.0,
    )
    assert mission.path_length_px > 100
    assert len(mission.points) >= 8

    output_path = tmp_path / "mission_path.json"
    mission.save(output_path)
    loaded = MissionPath.load(output_path)
    assert loaded.source_method == result.method
    assert loaded.frame_size == {"width": 640, "height": 480}
    assert loaded.points[0][1] < loaded.points[-1][1]


def test_controller_generates_roll_pitch_and_throttle_commands() -> None:
    mission = MissionPath(
        points=[(420.0, 40.0), (420.0, 220.0), (420.0, 440.0)],
        frame_size={"width": 640, "height": 480},
        source_method="threshold_morph",
    )
    controller = TrackFollowerController(mission, TrackFollowerConfig(height_target_cm=80.0))
    observation = {
        "valid": True,
        "confidence": 0.8,
        "frame_size": {"width": 640, "height": 480},
        "target": {
            "kind": "drone",
            "position_px": {"x": 320.0, "y": 180.0},
            "velocity_px_s": {"x": 0.0, "y": 0.0},
        },
    }

    control = controller.update(observation, height_cm=60.0)
    assert control.command.roll > 0
    assert control.command.pitch > 0
    assert control.command.throttle > 0
    assert control.lookahead_point is not None
    assert control.path_progress is not None


def test_controller_hovers_indefinitely_by_default_after_vision_loss() -> None:
    mission = MissionPath(
        points=[(320.0, 40.0), (320.0, 440.0)],
        frame_size={"width": 640, "height": 480},
        source_method="threshold_morph",
    )
    controller = TrackFollowerController(mission, TrackFollowerConfig(lost_frame_limit=2))
    observation = {"valid": False, "confidence": 0.0, "target": None}

    first = controller.update(observation)
    second = controller.update(observation)
    third = controller.update(observation)

    assert first.command.hover and not first.command.land
    assert second.command.hover and not second.command.land
    assert third.command.hover and not third.command.land


def test_controller_can_land_after_vision_loss_when_enabled() -> None:
    mission = MissionPath(
        points=[(320.0, 40.0), (320.0, 440.0)],
        frame_size={"width": 640, "height": 480},
        source_method="threshold_morph",
    )
    controller = TrackFollowerController(
        mission,
        TrackFollowerConfig(lost_frame_limit=2, land_on_vision_loss=True),
    )
    observation = {"valid": False, "confidence": 0.0, "target": None}

    controller.update(observation)
    controller.update(observation)
    third = controller.update(observation)

    assert third.command.land


def test_auto_orient_reverses_when_drone_is_closer_to_path_end() -> None:
    path = [(100.0, 50.0), (100.0, 200.0), (100.0, 350.0)]

    assert _should_reverse_path(path, (98.0, 340.0))
    assert not _should_reverse_path(path, (102.0, 60.0))


def test_mission_path_scale_round_trips(tmp_path) -> None:
    mission = MissionPath(
        points=[(10.0, 10.0), (20.0, 30.0)],
        frame_size={"width": 100, "height": 100},
        source_method="manual_click",
        meters_per_pixel=0.004,
        start_point_px=(12.0, 14.0),
    )
    path = tmp_path / "scaled_mission.json"
    mission.save(path)
    loaded = MissionPath.load(path)
    assert loaded.meters_per_pixel == 0.004
    assert loaded.start_point_px == (12.0, 14.0)


def test_waypoint_transform_uses_marker_forward_heading() -> None:
    mission = MissionPath(
        points=[(100.0, 100.0), (100.0, 200.0)],
        frame_size={"width": 400, "height": 400},
        source_method="manual_click",
        meters_per_pixel=0.01,
    )
    forward_heading = marker_forward_heading_rad(np.pi / 2.0)
    assert forward_heading == 0.0

    waypoint_mission = waypoint_mission_from_path(
        mission=mission,
        drone_start_px=(100.0, 100.0),
        drone_forward_rad=forward_heading,
        meters_per_pixel=0.01,
        target_height_m=0.8,
        waypoint_spacing_px=100.0,
        speed_m_s=0.25,
    )
    assert len(waypoint_mission.waypoints) == 2
    first, second = waypoint_mission.waypoints
    assert first.x_m == 0.0
    assert first.y_m == 0.0
    assert second.x_m == 0.0
    assert second.y_m < -0.9


def test_reference_distance_converts_to_scale() -> None:
    scale = meters_per_pixel_from_reference(((0.0, 0.0), (0.0, 100.0)), reference_distance_cm=50.0)
    assert scale == 0.005


def test_null_adapter_logs_waypoints() -> None:
    adapter = NullFlightAdapter()
    adapter.connect()
    adapter.takeoff()
    status = adapter.fly_waypoint(Waypoint(x_m=0.2, y_m=-0.1, z_m=0.8, speed_m_s=0.25), tolerance_m=0.08, timeout_s=6.0)
    assert status["reached"]
    assert adapter.command_history[-1]["waypoint"]["z_m"] == 0.8


def test_path_start_heading_uses_first_segment() -> None:
    heading = path_start_heading_rad([(100.0, 100.0), (100.0, 200.0), (150.0, 250.0)])
    assert np.isclose(heading, np.pi / 2.0)


def test_follow_waypoints_manual_start_uses_numeric_start(tmp_path) -> None:
    mission = MissionPath(
        points=[(100.0, 100.0), (100.0, 200.0), (100.0, 300.0)],
        frame_size={"width": 400, "height": 400},
        source_method="manual_click",
        meters_per_pixel=0.01,
    )
    mission_path = tmp_path / "mission.json"
    mission.save(mission_path)

    waypoint_mission = follow_waypoints_manual_start(
        mission_path=mission_path,
        start_x=100.0,
        start_y=100.0,
        dry_run=True,
    )
    assert len(waypoint_mission.waypoints) >= 2
    assert waypoint_mission.drone_start_px == (100.0, 100.0)


def test_follow_waypoints_manual_start_uses_mission_start_point(tmp_path) -> None:
    mission = MissionPath(
        points=[(100.0, 100.0), (100.0, 200.0), (100.0, 300.0)],
        frame_size={"width": 400, "height": 400},
        source_method="manual_click",
        meters_per_pixel=0.01,
        start_point_px=(108.0, 112.0),
    )
    mission_path = tmp_path / "mission_with_start.json"
    mission.save(mission_path)

    waypoint_mission = follow_waypoints_manual_start(
        mission_path=mission_path,
        dry_run=True,
    )
    assert waypoint_mission.drone_start_px == (108.0, 112.0)
