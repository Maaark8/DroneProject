import cv2
from detectors.threshold_morph.detector import ThresholdMorphDetector
from track_detection.types import FrameInput
from tests.test_detectors import make_synthetic_track_frame

frame = make_synthetic_track_frame()
detector = ThresholdMorphDetector()
result = detector.detect(FrameInput(frame=frame))

print(f"valid={result.valid}")
print(f"confidence={result.confidence}")
print(f"heading_rad={result.heading_rad}")
print(f"lateral_offset_px={result.lateral_offset_px}")
print(f"centerline points: {len(result.centerline)}")

cv2.imwrite("input.png", frame)
cv2.imwrite("debug.png", result.debug_frame)
print("Saved input.png and debug.png")