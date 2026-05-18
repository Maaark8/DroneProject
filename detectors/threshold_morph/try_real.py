import os
import cv2
from detectors.threshold_morph.detector import ThresholdMorphDetector
from track_detection.types import FrameInput

INPUT_DIR = "tracks_for_drone"
OUTPUT_DIR = "real_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
detector = ThresholdMorphDetector()

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

if not files:
    print(f"No images found in {INPUT_DIR}")

for filename in sorted(files):
    in_path = os.path.join(INPUT_DIR, filename)
    frame = cv2.imread(in_path)
    if frame is None:
        print(f"[SKIP] Could not read {filename}")
        continue

    result = detector.detect(FrameInput(frame=frame))

    base, _ = os.path.splitext(filename)
    out_path = os.path.join(OUTPUT_DIR, f"{base}_debug.png")
    cv2.imwrite(out_path, result.debug_frame)

    print(
        f"{filename:30s}  valid={str(result.valid):5s}  "
        f"conf={result.confidence:.2f}  "
        f"points={len(result.centerline):3d}  "
        f"offset={result.lateral_offset_px}"
    )

print(f"\nDone. Annotated images written to {OUTPUT_DIR}/")