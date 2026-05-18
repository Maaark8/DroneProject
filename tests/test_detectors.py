from __future__ import annotations

import cv2
import numpy as np
import pytest
from pathlib import Path

from detectors.edge_geometry.detector import EdgeGeometryDetector
from detectors.drone_light.detector import DroneLightDetector
from detectors.drone_marker.detector import ArucoMarkerDetector, generate_aruco_marker_image
from detectors.segmentation.detector import SegmentationConfig, SegmentationDetector, _infer_checkpoint_backend
from detectors.wood_path.detector import WoodPathDetector
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


def make_synthetic_s_curve_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    ys = np.linspace(40, 460, 48)
    xs = 320 + 45 * np.sin((ys - 40) / 420.0 * 2.0 * np.pi)
    points = np.array([[int(x), int(y)] for x, y in zip(xs, ys)], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=False, color=(70, 110, 170), thickness=42)
    return frame


def test_wood_path_traces_s_curve_in_order() -> None:
    detector = WoodPathDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_s_curve_frame()))

    assert result.valid
    assert len(result.centerline) >= detector.config.min_path_points
    # Ordered bottom-of-image first; trace spans the curve vertically.
    assert result.centerline[0][1] > result.centerline[-1][1]
    ys = [y for _, y in result.centerline]
    assert max(ys) - min(ys) > 250
    assert result.metadata["path_shape"] == "snake"


def test_wood_path_classifies_straight_track() -> None:
    detector = WoodPathDetector()
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    cv2.line(frame, (320, 460), (320, 60), (70, 110, 170), 42)
    result = detector.detect(FrameInput(frame=frame))

    assert result.valid
    assert result.metadata["path_shape"] == "straight"
    assert abs(result.metadata["net_turn_deg"]) < 20.0


def test_wood_path_rejects_blank_frame() -> None:
    detector = WoodPathDetector()
    result = detector.detect(FrameInput(frame=np.full((480, 640, 3), 205, dtype=np.uint8)))

    assert not result.valid
    assert result.centerline == []
    assert result.confidence == 0.0


def make_synthetic_drone_frame(center: tuple[int, int] = (420, 180), color=(0, 0, 255)) -> np.ndarray:
    frame = np.full((480, 640, 3), (45, 48, 50), dtype=np.uint8)
    cv2.circle(frame, center, 24, (18, 18, 18), -1)
    cv2.circle(frame, center, 16, color, -1)
    cv2.circle(frame, center, 6, (255, 255, 255), -1)
    return frame


def make_synthetic_drone_frame_with_distractor() -> np.ndarray:
    frame = make_synthetic_drone_frame(center=(420, 180), color=(0, 0, 255))
    cv2.circle(frame, (210, 250), 14, (0, 255, 0), -1)
    cv2.circle(frame, (210, 250), 4, (210, 255, 210), -1)
    return frame


def make_synthetic_aruco_frame(marker_id: int = 7, top_left: tuple[int, int] = (320, 120)) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 120)
    padded = np.full((160, 160), 255, dtype=np.uint8)
    padded[20:140, 20:140] = marker
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    x, y = top_left
    frame[y : y + 160, x : x + 160, :] = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    return frame


def make_synthetic_wrong_aruco_frame() -> np.ndarray:
    return make_synthetic_aruco_frame(marker_id=37, top_left=(120, 140))


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


def test_drone_light_detector_prefers_red_hotspot_over_other_bright_blobs() -> None:
    detector = DroneLightDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_drone_frame_with_distractor()))

    assert result.valid
    assert result.metadata["color_name"] == "red"
    assert result.metadata["preferred_hue_score"] > 0.8
    assert abs(result.centerline[0][0] - 420) < 8
    assert abs(result.centerline[0][1] - 180) < 8


def test_drone_detection_control_observation_schema() -> None:
    detector = DroneLightDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_drone_frame(center=(320, 240)), frame_id=3))
    observation = to_control_observation(result)

    assert observation["schema_version"] == "drone_control_observation.v1"
    assert observation["source_method"] == "drone_light"
    assert observation["target"]["kind"] == "drone"
    assert observation["target"]["offset_norm"]["x"] == pytest.approx(0.0, abs=0.01)
    assert observation["target"]["offset_norm"]["y"] == pytest.approx(0.0, abs=0.01)


def test_aruco_marker_detector_finds_expected_marker() -> None:
    detector = ArucoMarkerDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_aruco_frame(), frame_id=5, timestamp_s=0.4))

    assert result.valid
    assert result.confidence > 0.5
    assert result.debug_frame is not None
    assert abs(result.centerline[0][0] - 399.5) < 8
    assert abs(result.centerline[0][1] - 199.5) < 8
    assert result.metadata["marker_id"] == 7
    assert result.metadata["offset_norm"]["x"] > 0
    assert result.metadata["heading_rad"] == pytest.approx(0.0, abs=0.05)


def test_aruco_marker_detection_control_observation_schema() -> None:
    detector = ArucoMarkerDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_aruco_frame(top_left=(240, 160)), frame_id=9))
    observation = to_control_observation(result)

    assert observation["source_method"] == "drone_marker"
    assert observation["target"]["kind"] == "drone"
    assert observation["target"]["marker_id"] == 7
    assert observation["target"]["heading_rad"] == pytest.approx(0.0, abs=0.05)


def test_aruco_marker_detector_rejects_wrong_marker_id() -> None:
    detector = ArucoMarkerDetector()
    result = detector.detect(FrameInput(frame=make_synthetic_wrong_aruco_frame(), frame_id=11))

    assert not result.valid
    assert result.confidence == 0.0
    assert result.centerline == []
    assert result.metadata["rejected_reason"] == "marker_not_found"


def test_generate_aruco_marker_image_writes_png(tmp_path) -> None:
    output_path = tmp_path / "aruco_id7.png"
    generate_aruco_marker_image(output_path, marker_id=7, side_pixels=200, margin_pixels=40)

    image = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert image.shape == (280, 280)
    assert int(image.min()) == 0
    assert int(image.max()) == 255


def test_segmentation_detector_requires_runtime_dependency() -> None:
    with pytest.raises(ImportError):
        SegmentationDetector()


def test_segmentation_checkpoint_backend_detection() -> None:
    assert _infer_checkpoint_backend(Path("model/exp.pt")) == "yolo"

    tmp_path = Path("tests") / "__fake_unet_checkpoint__.pt"
    try:
        tmp_path.write_bytes(b"plain-state-dict-checkpoint")
        assert _infer_checkpoint_backend(tmp_path) == "unet"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_yolo_segmentation_detector_uses_full_frame_path() -> None:
    detector = object.__new__(SegmentationDetector)
    detector.config = SegmentationConfig()
    detector.backend = "yolo"
    detector.method_name = "segmentation"
    detector.overlay_color = (255, 120, 80)

    def fake_detect_mask_yolo(frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[:, 300:340] = 255
        return mask, {"backend": "yolo", "selected_confidence": 0.9}

    detector._detect_mask_yolo = fake_detect_mask_yolo

    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    result = detector.detect(FrameInput(frame=frame, frame_id=12))

    assert result.valid
    assert result.metadata["source_backend"] == "yolo"
    assert result.metadata["frame_size"] == {"width": 640, "height": 480}
    assert abs(result.centerline[-1][0] - 319.5) < 5
