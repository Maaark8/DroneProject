from __future__ import annotations

import cv2
import numpy as np
import pytest

from detectors.edge_geometry.detector import EdgeGeometryDetector
from detectors.drone_light.detector import DroneLightDetector
from detectors.segmentation.detector import SegmentationDetector
from detectors.threshold_morph.detector import ThresholdMorphDetector
from track_detection.control_output import to_control_observation
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


def make_synthetic_drone_frame(center: tuple[int, int] = (420, 180), color=(0, 0, 255)) -> np.ndarray:
    frame = np.full((480, 640, 3), (45, 48, 50), dtype=np.uint8)
    cv2.circle(frame, center, 24, (18, 18, 18), -1)
    cv2.circle(frame, center, 16, color, -1)
    cv2.circle(frame, center, 6, (255, 255, 255), -1)
    return frame


@pytest.mark.parametrize(
    ("detector", "expected_sign"),
    [
        (ThresholdMorphDetector(), 0),
        (EdgeGeometryDetector(), 0),
    ],
)
def test_centered_track_is_detected(detector, expected_sign: int) -> None:
    result = detector.detect(FrameInput(frame=make_synthetic_track_frame()))
    assert result.valid
    assert len(result.centerline) >= detector.config.min_centerline_points
    assert result.confidence > 0.1
    assert result.debug_frame is not None
    assert result.lateral_offset_px is not None
    assert abs(result.lateral_offset_px) < 40


def test_shifted_track_produces_positive_offset() -> None:
    detector = ThresholdMorphDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_track_frame(x_shift=90)))
    assert result.valid
    assert result.lateral_offset_px is not None
    assert result.lateral_offset_px > 40


def test_drone_light_detector_finds_colored_top_light() -> None:
    detector = DroneLightDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_drone_frame(), frame_id=7, timestamp_s=1.5))

    assert result.valid
    assert result.confidence > 0.4
    assert result.debug_frame is not None
    assert result.centerline
    assert abs(result.centerline[0][0] - 420) < 8
    assert abs(result.centerline[0][1] - 180) < 8
    assert result.metadata["offset_norm"]["x"] > 0
    assert result.metadata["offset_norm"]["y"] < 0
    assert result.metadata["color_name"] == "red"


def test_drone_light_detector_reports_velocity_for_live_control() -> None:
    detector = DroneLightDetector()
    detector.detect(FrameInput(frame=make_synthetic_drone_frame(center=(300, 240)), frame_id=0, timestamp_s=0.0))
    result = detector.detect(FrameInput(frame=make_synthetic_drone_frame(center=(330, 225)), frame_id=1, timestamp_s=0.5))

    assert result.metadata["velocity_px_s"]["x"] == pytest.approx(60.0, abs=2.0)
    assert result.metadata["velocity_px_s"]["y"] == pytest.approx(-30.0, abs=2.0)


def test_drone_detection_control_observation_schema() -> None:
    detector = DroneLightDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_drone_frame(center=(320, 240)), frame_id=3))
    observation = to_control_observation(result)

    assert observation["schema_version"] == "drone_control_observation.v1"
    assert observation["source_method"] == "drone_light"
    assert observation["target"]["kind"] == "drone"
    assert observation["target"]["offset_norm"]["x"] == pytest.approx(0.0, abs=0.01)
    assert observation["target"]["offset_norm"]["y"] == pytest.approx(0.0, abs=0.01)


def test_segmentation_detector_requires_torch() -> None:
    with pytest.raises(ImportError):
        SegmentationDetector()
