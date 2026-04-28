# DroneProject

Offline track-detection pipeline for a downward-facing drone camera. The
project estimates the centerline of a wooden train track and lets you compare
multiple detection methods over recorded footage.

## Project layout

- `track_detection/` shared types, preprocessing, geometry, and CLI
- `detectors/threshold_morph/` threshold + morphology detector and plan
- `detectors/edge_geometry/` edge + contour detector and plan
- `detectors/segmentation/` learned segmentation detector and plan
- `evaluation/` output format notes for offline comparison runs
- `tests/` synthetic-frame regression tests

## Detection methods

1. `threshold_morph`: HSV/Lab masking, morphology cleanup, centerline from mask
2. `edge_geometry`: edge extraction, contour scoring, centerline from geometry
3. `segmentation`: lightweight segmentation model with optional PyTorch training

All methods return the same result shape:

- centerline points in image coordinates
- heading estimate in radians
- lateral offset in pixels
- confidence score
- validity flag
- debug overlay frame

## Quick start

Create a virtual environment and install the project:

```bash
python3 -m pip install -e .
```

Run a detector over a video or image folder:

```bash
python3 -m track_detection.cli run \
  --method threshold_morph \
  --input path/to/video.mp4 \
  --output-dir outputs/threshold_run
```

Extract frames from a recorded video:

```bash
python3 -m track_detection.cli extract-frames \
  --input path/to/video.mp4 \
  --output-dir data/frames/sample_run \
  --every-n 5
```

Train the segmentation baseline after installing PyTorch:

```bash
python3 -m track_detection.cli train-segmentation \
  --images data/frames/train \
  --masks data/masks/train \
  --checkpoint outputs/segmentation/model.pt
```

## Dependencies

Base runtime:

- `numpy`
- `opencv-python`

Optional training dependency:

- `torch`

The segmentation detector is implemented, but it requires PyTorch at runtime.

## Testing

Run the unit tests with:

```bash
python3 -m pytest
```
