# Drone control — hover/follow the wood track (external overhead camera)

Setup: a fixed overhead camera films the table; a paired CoDrone EDU flies in
that view. The track is detected by `wood_path` from the camera's own frame, so
track pixels and drone pixels share one coordinate system; the controller
drives the pixel error to zero (drone hovers over / follows the track).

## 0. One-time prep
- Mount the camera so it sees the whole track + flight volume, looking
  straight down, fixed (it must not move after calibration).
- Print `Raul-Folder/outputs/drone_marker_id7.png`, cut it out, stick it
  FLAT and centered on the drone's top. ArUco (`--drone-method drone_marker`)
  is much more robust than the LED under room light.
- Charge battery; clear a padded, obstacle-free area; know how to kill power.

## 1. Dry run (NO takeoff — validates vision + controller)
    python -m track_detection.cli auto-follow --method wood_path \
      --drone-method drone_marker --camera-index 0 \
      --output-dir Raul-Folder/outputs/follow --dry-run

Watch the overlay window:
- the wood track is detected (mask + yellow centerline on the real track),
- the drone marker is boxed every frame,
- `cmd r/p` values move toward the track when you slide the drone by hand.
Logs: `Raul-Folder/outputs/follow/follow_log.jsonl`.

## 2. Real flight
Remove `--dry-run`, add safety options:

    python -m track_detection.cli auto-follow --method wood_path \
      --drone-method drone_marker --camera-index 0 \
      --height-cm 70 --command-rate 10 \
      --land-on-vision-loss --land-on-complete \
      --output-dir Raul-Folder/outputs/follow

- If the drone drives the WRONG way, add `--roll-sign -1` and/or
  `--pitch-sign -1` (camera axes vs drone body axes) and retry the dry run.
- Start low (`--height-cm` 60–80), low `--command-rate` first.
- `--land-on-vision-loss` lands if the marker is lost for
  `--vision-loss-frames` frames (default 30). Keep a hand on power.

## Alternative: pre-baked mission
If you prefer to fix the path from one overhead still first:

    python -m track_detection.cli export-path --method wood_path \
      --input <overhead_frame.jpg> --output Raul-Folder/outputs/mission.json
    python -m track_detection.cli follow-track --mission Raul-Folder/outputs/mission.json \
      --drone-method drone_marker --camera-index 0 --dry-run

NOTE: with `follow-track` the still MUST be from the same fixed camera/framing
as the live run, or track pixels won't line up with the drone. `auto-follow`
avoids this by calibrating from the live feed — prefer it.

## Tuning knobs (track_detection/controller.py TrackFollowerConfig)
`kp_roll/kp_pitch` (approach aggressiveness), `kd_*` (damping/overshoot),
`lookahead_px` (corner cutting vs smoothness), `completion_radius_px`,
`height_target_cm`, deadbands. Tune in dry run against logs before flying.
