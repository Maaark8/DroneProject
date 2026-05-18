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
- `detectors/drone_marker/` ArUco marker detector for a printed marker on top of the drone
- `evaluation/` output format notes for offline comparison runs
- `tests/` synthetic-frame regression tests

## Detection methods

1. `threshold_morph`: HSV/Lab masking, morphology cleanup, centerline from mask
2. `edge_geometry`: edge extraction, contour scoring, centerline from geometry
3. `segmentation`: lightweight segmentation model with optional PyTorch training
4. `drone_light`: bright colored top-light localization for overhead cameras
5. `drone_marker`: ArUco marker localization and heading for overhead cameras

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

If `model/exp.pt` contains an Ultralytics YOLO segmentation checkpoint, the
`segmentation` method will use it automatically:

```bash
python3 -m track_detection.cli run \
  --method segmentation \
  --input path/to/video.mp4 \
  --output-dir outputs/segmentation_run
```

Run the CoDrone EDU top-light detector on a live overhead camera:

```bash
python3 -m track_detection.cli live \
  --method drone_light \
  --camera-index 0 \
  --output-dir outputs/live_drone
```

Run the ArUco marker detector on the same overhead camera:

```bash
python3 -m track_detection.cli live \
  --method drone_marker \
  --camera-index 0 \
  --output-dir outputs/live_marker
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

Export a mission path from a static overhead view of the track:

```bash
python3 -m track_detection.cli export-path \
  --method threshold_morph \
  --input tracks_for_drone/wood_track.jpeg \
  --output outputs/missions/wood_track.json
```

Follow the saved path with a CoDrone EDU under an overhead camera:

```bash
python3 -m track_detection.cli follow-track \
  --mission outputs/missions/wood_track.json \
  --drone-method drone_marker \
  --camera-index 0 \
  --output-dir outputs/follow_wood_track
```

For wood-on-wood scenes, the more reliable workflow is to calibrate once and fly
waypoints instead of chasing the track every frame. Capture one overhead frame,
click two reference points with known spacing, click the track centerline, then
click the drone start position. Colored landing pad centers work well as the
reference pair:

```bash
python3 -m track_detection.cli manual-mission \
  --source http://PHONE_IP:8080/video \
  --output outputs/missions/manual_track.json \
  --reference-distance-cm 60
```

Then fly the saved mission as CoDrone absolute waypoints using the ArUco marker
to define the drone's start position and heading:

```bash
python3 -m track_detection.cli follow-waypoints \
  --mission outputs/missions/manual_track.json \
  --drone-method drone_marker \
  --source http://PHONE_IP:8080/video \
  --output-dir outputs/follow_waypoints \
  --dry-run
```

Mount the marker so its printed top edge points toward the drone's forward
direction. `follow-waypoints` converts image deltas into CoDrone X/Y waypoints
from that heading and uses the mission scale stored in `manual-mission`.
Waypoint speed defaults to `0.8 m/s`, which matches the CoDrone EDU absolute
position API range more closely than the lower visual-following speeds.
The current waypoint defaults are more aggressive than before: `0.8 m/s`,
`120 px` spacing, `0.15 m` arrival tolerance, and `3.0 s` timeout per waypoint,
so the drone advances sooner instead of lingering on each intermediate point.

If you do not want any live drone detection, there is also a manual-start mode.
`manual-mission` now stores the clicked start point in the mission, so the
manual-start follower can reuse it directly. Place the drone at the beginning of
the track and align it with the first segment:

```bash
python3 -m track_detection.cli follow-waypoints-manual-start \
  --mission outputs/missions/manual_track.json \
  --output-dir outputs/follow_waypoints_manual \
  --dry-run
```

You can still override the stored start with a fresh click or explicit
coordinates:

```bash
python3 -m track_detection.cli follow-waypoints-manual-start \
  --mission outputs/missions/manual_track.json \
  --start-x 412 \
  --start-y 268 \
  --dry-run
```

This mode assumes the drone is already pointing along the start of the track, so
it uses the first path segment as the drone's forward heading.

Run the full overhead-camera workflow in one command. This mode samples live
frames first, detects the track, saves `mission_path.json`, then takes off and
starts following:

```bash
python3 -m track_detection.cli auto-follow \
  --method threshold_morph \
  --drone-method drone_marker \
  --camera-index 0 \
  --output-dir outputs/auto_follow_run
```

`auto-follow` tries to orient the saved path so it starts near the drone's
detected start position. Disable that with `--no-auto-orient`, or force the
opposite direction with `--reverse-path`.

Use `--dry-run` to exercise the controller and debug overlays without pairing to
the drone. If the camera axes are mirrored relative to the drone, flip them with
`--roll-sign -1` and/or `--pitch-sign -1`.

Generate a printable ArUco marker image:

```bash
python3 -m track_detection.cli generate-marker \
  --output outputs/markers/aruco_id7.png \
  --dictionary DICT_4X4_50 \
  --marker-id 7
```

Print the marker with its white border intact and tape it flat on top of the
drone. `follow-track` and `auto-follow` accept `--drone-method drone_marker` to
use it instead of the red-light detector.

For phone or IP camera streams, the live commands now prefer low-buffer capture
and always process the newest available frame so the CLI does not accumulate
seconds of stale video when detection runs slower than the stream FPS.

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
- `opencv-contrib-python`

Optional training dependency:

- `torch`
- `ultralytics` for YOLO segmentation checkpoints such as `model/exp.pt`

The segmentation detector requires either PyTorch for the built-in U-Net path or
Ultralytics for YOLO segmentation checkpoints.
The marker detector requires the ArUco module included in `opencv-contrib-python`.
The optional `yolo` extra installs the Ultralytics runtime for YOLO segmentation
models:

```bash
python3 -m pip install -e .[yolo] --no-build-isolation
```
Track following with real hardware also requires the official CoDrone EDU Python
library (`codrone_edu`) installed separately.

## Testing

Run the unit tests with:

```bash
python3 -m pytest
```
