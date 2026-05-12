# Movement Output Plan

Movement code should consume `control_observation` from each JSONL result. The
schema is intentionally detector-neutral, so track-following, drone-localization,
and later safety detections can be merged without changing the control loop.

## Current Schema

```json
{
  "schema_version": "drone_control_observation.v1",
  "source_method": "drone_light",
  "valid": true,
  "confidence": 0.82,
  "frame_id": 42,
  "timestamp_s": 1.4,
  "frame_size": {"width": 640, "height": 480},
  "target": {
    "kind": "drone",
    "position_px": {"x": 380.0, "y": 210.0},
    "offset_px": {"x": 60.0, "y": -30.0},
    "offset_norm": {"x": 0.1875, "y": -0.125},
    "velocity_px_s": {"x": 24.0, "y": -12.0},
    "speed_px_s": 26.83,
    "bbox_xywh": [360, 190, 40, 40],
    "radius_px": 20.0,
    "color_name": "red"
  }
}
```

## Control Meaning

- `valid=false`: hold position or use the last good observation briefly.
- `confidence`: gate movement; start conservatively below about `0.5`.
- `offset_norm.x`: target is right of frame center when positive.
- `offset_norm.y`: target is lower in the image when positive.
- `velocity_px_s`: damp commands so the CoDrone EDU does not chase noise.
- `position_px` and `bbox_xywh`: useful for logging and debug overlays.

For a fixed overhead camera, the simplest controller should drive the drone until
`offset_norm` is near zero. Keep the camera-to-drone axis mapping in one adapter:
depending on how the camera image is mounted, image `+x` may map to drone roll
left/right and image `+y` may map to drone pitch forward/backward or the inverse.

## Combining Different Detections

Use the same envelope for every detector:

- `target.kind="drone"` for the top-light localization detector.
- `target.kind="track_centerline"` for track-following detectors.
- Add future `target.kind` values for obstacles, landing pads, or safety zones.

Movement code should read only `control_observation`, then switch behavior by
`target.kind`. Detector-specific diagnostics stay in the top-level `metadata`
field and should not be required for flight control.
