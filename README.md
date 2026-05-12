# DroneProject

Offline track-detection pipeline for a downward-facing drone camera. The
project estimates the centerline of a wooden train track and lets you compare
multiple detection methods over recorded footage.

## Project layout

- `track_detection/` shared types, preprocessing, geometry, movement output, and CLI
- `detectors/threshold_morph/` threshold + morphology detector and plan
- `detectors/edge_geometry/` edge + contour detector and plan
- `detectors/segmentation/` learned segmentation detector and plan
- `detectors/drone_light/` fast top-down drone detector using the CoDrone EDU top light
- `evaluation/` output format notes for offline comparison runs
- `tests/` synthetic-frame regression tests

## Detection methods

1. `threshold_morph`: HSV/Lab masking, morphology cleanup, centerline from mask
2. `edge_geometry`: edge extraction, contour scoring, centerline from geometry
3. `segmentation`: lightweight segmentation model with optional PyTorch training
4. `drone_light`: bright colored top-light localization for overhead cameras

All methods return the same result shape and a `control_observation` JSON object:

- centerline points in image coordinates
- heading estimate in radians
- lateral offset in pixels
- confidence score
- validity flag
- debug overlay frame
- movement-ready target fields such as normalized offset, position, and velocity

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

Run the CoDrone EDU top-light detector on a live overhead camera:

```bash
python3 -m track_detection.cli live \
  --method drone_light \
  --camera-index 0 \
  --output-dir outputs/live_drone
```

For a phone or IP camera, pass the stream URL instead of a local camera index:

```bash
python3 -m track_detection.cli live \
  --method drone_light \
  --source http://PHONE_IP:8080/video \
  --output-dir outputs/live_drone
```

Use `--no-display` when running headless. The live command streams
`results.jsonl` as frames arrive and writes `debug_overlay.mp4` when an output
directory is provided.

The detector defaults to colored lights because white highlights in the room can
look like white LEDs. If you need white-light tracking, construct
`DroneLightDetector(DroneLightConfig(detect_white_light=True))` and calibrate it
against your camera view.

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
