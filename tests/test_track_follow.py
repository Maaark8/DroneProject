from __future__ import annotations

import cv2
import numpy as np

from detectors.threshold_morph.detector import ThresholdMorphDetector
from track_detection.controller import TrackFollowerConfig, TrackFollowerController
from track_detection.mission import MissionPath, mission_path_from_result
from track_detection.types import FrameInput


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


def test_controller_hovers_then_lands_after_vision_loss() -> None:
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
    assert third.command.land
