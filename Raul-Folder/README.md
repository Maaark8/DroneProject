# Raul-Folder

A second computer-vision method (`wood_path`) for detecting the wooden tracks
and the route they take, plus how it plugs into making the CoDrone EDU hover
over the track. Primary copy lives here; a mirrored copy is integrated into
`detectors/wood_path/` and wired into the factory per AGENTS.md.

## Files
- `wood_path_detector.py` — the `WoodPathDetector` CV method.
- `run.py` — standalone CLI: `trace` (overlays + route report) and
  `export-path` (mission JSON for `follow-track`).
- `test_wood_path.py` — regression tests.
- `plan.md` — design + limitations.
- `outputs/` — generated overlays / reports / missions.

## The two-stage pipeline (how the drone hovers over the track)

The repo is a **two-camera** system:

**Stage A — see the track (offline).** A static overhead photo of the wooden
track → `WoodPathDetector` → ordered centerline polyline → resampled →
`MissionPath` JSON (path in image pixels).

**Stage B — fly the drone (live).** An *external overhead camera* watches the
room. `DroneLightDetector` / `ArucoMarkerDetector` finds the CoDrone EDU each
frame → `TrackFollowerController` projects the drone's pixel position onto the
mission path, picks a lookahead point, runs PD roll/pitch + a height P-loop →
`CoDroneEDUAdapter` sends `set_roll/pitch/yaw/throttle`. The drone "hovers
above the track" because the track and the drone are localized in the **same
pixel frame**, and the controller drives that pixel error to zero.

`wood_path` improves Stage A: it traces the *actual route* (curves, S-bends,
U-turns, switches) instead of a per-row mean that fails on bends.

## Usage (run from the repo root)

Trace all sample tracks and write overlays + a route report:

    python Raul-Folder/run.py trace --input tracks_for_drone --output-dir Raul-Folder/outputs

Export a drone mission from one overhead photo (Raul-Folder runner or the
integrated CLI — both produce the same mission):

    python Raul-Folder/run.py export-path \
        --input tracks_for_drone/track_snake.jpeg \
        --output Raul-Folder/outputs/track_snake_mission.json

    python -m track_detection.cli export-path --method wood_path \
        --input tracks_for_drone/track_snake.jpeg \
        --output Raul-Folder/outputs/track_snake_mission.json

    python -m track_detection.cli follow-track \
        --mission Raul-Folder/outputs/track_snake_mission.json \
        --camera-index 0 --output-dir Raul-Folder/outputs/follow --dry-run

Tests:

    python -m pytest Raul-Folder/test_wood_path.py -q

## Route description (`DetectionResult.metadata`)
`path_shape` ∈ {straight, curved_left, curved_right, snake, compound},
`path_length_px`, `net_turn_deg`, `total_curvature_deg`, `start_px`,
`end_px`, `endpoint_count`, `junction_count`, `junctions_px`.

Note: on oblique handheld photos, fine snake-vs-gentle-curve separation is not
reliable, so the classifier only asserts categories with clean separation and
exposes the raw measures for the consumer to judge.
