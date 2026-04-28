from __future__ import annotations

import cv2
import numpy as np
import pytest

from detectors.edge_geometry.detector import EdgeGeometryDetector
from detectors.segmentation.detector import SegmentationDetector
from detectors.threshold_morph.detector import ThresholdMorphDetector
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


def test_segmentation_detector_requires_torch() -> None:
    with pytest.raises(ImportError):
        SegmentationDetector()
