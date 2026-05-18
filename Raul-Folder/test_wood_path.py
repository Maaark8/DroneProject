"""Self-contained regression tests for the wood_path tracer.

Run from the repo root:  python -m pytest Raul-Folder/test_wood_path.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from track_detection.types import FrameInput  # noqa: E402

from wood_path_detector import WoodPathDetector  # noqa: E402


def make_synthetic_s_curve_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    # Gentle S (realistic track curvature, high straightness) that still
    # swings to both sides of its chord.
    ys = np.linspace(40, 460, 48)
    xs = 320 + 45 * np.sin((ys - 40) / 420.0 * 2.0 * np.pi)
    points = np.array([[int(x), int(y)] for x, y in zip(xs, ys)], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=False, color=(70, 110, 170), thickness=42)
    return frame


def make_synthetic_straight_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    cv2.line(frame, (320, 460), (320, 60), (70, 110, 170), 46)
    return frame


def test_wood_path_traces_s_curve_in_order() -> None:
    detector = WoodPathDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_s_curve_frame()))

    assert result.valid
    assert result.confidence > 0.1
    assert len(result.centerline) >= detector.config.min_path_points
    # Ordered bottom-of-image first, and the trace spans the curve vertically.
    assert result.centerline[0][1] > result.centerline[-1][1]
    ys = [y for _, y in result.centerline]
    assert max(ys) - min(ys) > 250
    # An S-curve swings to both sides, so the route is a snake, not straight.
    assert result.metadata["path_shape"] == "snake"
    assert result.metadata["path_length_px"] > 300
    assert result.metadata["total_curvature_deg"] > 60


def test_wood_path_classifies_straight_track() -> None:
    detector = WoodPathDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_straight_frame()))

    assert result.valid
    assert result.metadata["path_shape"] == "straight"
    assert abs(result.metadata["net_turn_deg"]) < 20.0


def test_wood_path_handles_blank_frame() -> None:
    detector = WoodPathDetector()
    blank = np.full((480, 640, 3), 205, dtype=np.uint8)
    result = detector.detect(FrameInput(frame=blank))

    assert not result.valid
    assert result.centerline == []
    assert result.confidence == 0.0
    assert result.debug_frame is not None
