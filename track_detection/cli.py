from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detectors.drone_marker.detector import generate_aruco_marker_image
from detectors.factory import DETECTOR_METHODS
from detectors.segmentation.train import train_segmentation_model

from .controller import TrackFollowerConfig
from .follow import (
    DRONE_METHODS,
    TRACK_METHODS,
    auto_follow_track,
    bridge_track_follow,
    export_mission_path,
    follow_track_live,
)
from .manual_calibration import capture_calibration_frame, manual_mission_from_frame
from .pipeline import run_live_camera, run_on_path
from .video import extract_frames
from .waypoint_follow import follow_waypoints_live, follow_waypoints_manual_start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track detection and evaluation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a detector on a video or image folder")
    run_parser.add_argument("--method", required=True, choices=DETECTOR_METHODS)
    run_parser.add_argument("--input", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)

    live_parser = subparsers.add_parser("live", help="Run a detector on a live camera feed")
    live_parser.add_argument("--method", default="drone_light", choices=DETECTOR_METHODS)
    live_parser.add_argument("--camera-index", type=int, default=0)
    live_parser.add_argument(
        "--source",
        help="Camera index, video path, or IP camera URL. Overrides --camera-index when provided.",
    )
    live_parser.add_argument("--output-dir", type=Path)
    live_parser.add_argument("--max-frames", type=int)
    live_parser.add_argument("--no-display", action="store_true")

    extract_parser = subparsers.add_parser("extract-frames", help="Extract frames from a video")
    extract_parser.add_argument("--input", required=True, type=Path)
    extract_parser.add_argument("--output-dir", required=True, type=Path)
    extract_parser.add_argument("--every-n", type=int, default=5)

    manual_parser = subparsers.add_parser(
        "manual-mission",
        help="Capture one frame, click a scale reference and track points, then save a scaled mission path",
    )
    manual_parser.add_argument("--output", required=True, type=Path)
    manual_parser.add_argument("--reference-distance-cm", required=True, type=float)
    manual_parser.add_argument("--input", type=Path)
    manual_parser.add_argument("--camera-index", type=int, default=0)
    manual_parser.add_argument(
        "--source",
        help="Camera index, video path, or IP camera URL. Overrides --camera-index when provided.",
    )
    manual_parser.add_argument("--sample-spacing", type=float, default=12.0)

    marker_parser = subparsers.add_parser("generate-marker", help="Generate an ArUco marker image for the drone")
    marker_parser.add_argument("--output", required=True, type=Path)
    marker_parser.add_argument("--dictionary", default="DICT_4X4_50")
    marker_parser.add_argument("--marker-id", type=int, default=7)
    marker_parser.add_argument("--side-px", type=int, default=400)
    marker_parser.add_argument("--margin-px", type=int, default=80)

    export_parser = subparsers.add_parser("export-path", help="Export a mission path from a track detection result")
    export_parser.add_argument("--method", required=True, choices=TRACK_METHODS)
    export_parser.add_argument("--input", required=True, type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--reverse-path", action="store_true")
    export_parser.add_argument("--max-frames", type=int, default=120)
    export_parser.add_argument("--sample-spacing", type=float, default=12.0)

    follow_parser = subparsers.add_parser("follow-track", help="Track-following control loop for CoDrone EDU")
    follow_parser.add_argument("--mission", required=True, type=Path)
    follow_parser.add_argument("--camera-index", type=int, default=0)
    follow_parser.add_argument(
        "--source",
        help="Camera index, video path, or IP camera URL. Overrides --camera-index when provided.",
    )
    follow_parser.add_argument("--output-dir", type=Path)
    follow_parser.add_argument("--max-frames", type=int)
    follow_parser.add_argument("--no-display", action="store_true")
    follow_parser.add_argument("--command-rate", type=float, default=10.0)
    follow_parser.add_argument("--height-cm", type=float, default=80.0)
    follow_parser.add_argument("--roll-sign", type=int, choices=(-1, 1), default=1)
    follow_parser.add_argument("--pitch-sign", type=int, choices=(-1, 1), default=1)
    follow_parser.add_argument("--drone-method", choices=DRONE_METHODS, default="drone_light")
    follow_parser.add_argument("--vision-loss-frames", type=int, default=30)
    follow_parser.add_argument("--land-on-vision-loss", action="store_true")
    follow_parser.add_argument("--land-on-complete", action="store_true")
    follow_parser.add_argument("--dry-run", action="store_true")

    waypoint_parser = subparsers.add_parser(
        "follow-waypoints",
        help="Fly a scaled mission path as absolute CoDrone waypoints using the start marker pose",
    )
    waypoint_parser.add_argument("--mission", required=True, type=Path)
    waypoint_parser.add_argument("--camera-index", type=int, default=0)
    waypoint_parser.add_argument(
        "--source",
        help="Camera index, video path, or IP camera URL. Overrides --camera-index when provided.",
    )
    waypoint_parser.add_argument("--output-dir", type=Path)
    waypoint_parser.add_argument("--calibration-frames", type=int, default=45)
    waypoint_parser.add_argument("--drone-method", choices=DRONE_METHODS, default="drone_marker")
    waypoint_parser.add_argument("--height-m", type=float, default=0.8)
    waypoint_parser.add_argument("--waypoint-spacing", type=float, default=120.0)
    waypoint_parser.add_argument("--speed-m-s", type=float, default=0.8)
    waypoint_parser.add_argument("--meters-per-pixel", type=float)
    waypoint_parser.add_argument("--position-tolerance-m", type=float, default=0.15)
    waypoint_parser.add_argument("--waypoint-timeout-s", type=float, default=3.0)
    waypoint_parser.add_argument("--reverse-path", action="store_true")
    waypoint_parser.add_argument("--no-auto-orient", action="store_true")
    waypoint_parser.add_argument("--dry-run", action="store_true")

    manual_start_parser = subparsers.add_parser(
        "follow-waypoints-manual-start",
        help="Fly a scaled mission path from a manually provided start point without drone detection",
    )
    manual_start_parser.add_argument("--mission", required=True, type=Path)
    manual_start_parser.add_argument("--output-dir", type=Path)
    manual_start_parser.add_argument("--input", type=Path)
    manual_start_parser.add_argument("--camera-index", type=int, default=0)
    manual_start_parser.add_argument(
        "--source",
        help="Camera index, video path, or IP camera URL. Overrides --camera-index when provided.",
    )
    manual_start_parser.add_argument("--height-m", type=float, default=0.8)
    manual_start_parser.add_argument("--waypoint-spacing", type=float, default=120.0)
    manual_start_parser.add_argument("--speed-m-s", type=float, default=0.8)
    manual_start_parser.add_argument("--meters-per-pixel", type=float)
    manual_start_parser.add_argument("--position-tolerance-m", type=float, default=0.15)
    manual_start_parser.add_argument("--waypoint-timeout-s", type=float, default=3.0)
    manual_start_parser.add_argument("--reverse-path", action="store_true")
    manual_start_parser.add_argument("--start-x", type=float)
    manual_start_parser.add_argument("--start-y", type=float)
    manual_start_parser.add_argument("--dry-run", action="store_true")

    auto_follow_parser = subparsers.add_parser(
        "auto-follow",
        help="Detect the track from the live overhead camera, then take off and follow it",
    )
    auto_follow_parser.add_argument("--method", default="threshold_morph", choices=TRACK_METHODS)
    auto_follow_parser.add_argument("--camera-index", type=int, default=0)
    auto_follow_parser.add_argument(
        "--source",
        help="Camera index, video path, or IP camera URL. Overrides --camera-index when provided.",
    )
    auto_follow_parser.add_argument("--output-dir", type=Path)
    auto_follow_parser.add_argument("--max-frames", type=int)
    auto_follow_parser.add_argument("--no-display", action="store_true")
    auto_follow_parser.add_argument("--command-rate", type=float, default=10.0)
    auto_follow_parser.add_argument("--height-cm", type=float, default=80.0)
    auto_follow_parser.add_argument("--roll-sign", type=int, choices=(-1, 1), default=1)
    auto_follow_parser.add_argument("--pitch-sign", type=int, choices=(-1, 1), default=1)
    auto_follow_parser.add_argument("--drone-method", choices=DRONE_METHODS, default="drone_light")
    auto_follow_parser.add_argument("--vision-loss-frames", type=int, default=30)
    auto_follow_parser.add_argument("--land-on-vision-loss", action="store_true")
    auto_follow_parser.add_argument("--land-on-complete", action="store_true")
    auto_follow_parser.add_argument("--dry-run", action="store_true")
    auto_follow_parser.add_argument("--calibration-frames", type=int, default=30)
    auto_follow_parser.add_argument("--sample-spacing", type=float, default=12.0)
    auto_follow_parser.add_argument("--reverse-path", action="store_true")
    auto_follow_parser.add_argument("--no-auto-orient", action="store_true")

    bridge_parser = subparsers.add_parser(
        "bridge",
        help="Bridge live track detection straight to drone control (re-detects every frame)",
    )
    bridge_parser.add_argument("--method", default="wood_path", choices=TRACK_METHODS)
    bridge_parser.add_argument("--camera-index", type=int, default=0)
    bridge_parser.add_argument(
        "--source",
        help="Camera index, video path, or IP camera URL. Overrides --camera-index when provided.",
    )
    bridge_parser.add_argument("--output-dir", type=Path)
    bridge_parser.add_argument("--max-frames", type=int)
    bridge_parser.add_argument("--no-display", action="store_true")
    bridge_parser.add_argument("--command-rate", type=float, default=10.0)
    bridge_parser.add_argument("--height-cm", type=float, default=80.0)
    bridge_parser.add_argument("--roll-sign", type=int, choices=(-1, 1), default=1)
    bridge_parser.add_argument("--pitch-sign", type=int, choices=(-1, 1), default=1)
    bridge_parser.add_argument("--drone-method", choices=DRONE_METHODS, default="drone_light")
    bridge_parser.add_argument("--vision-loss-frames", type=int, default=30)
    bridge_parser.add_argument("--land-on-vision-loss", action="store_true")
    bridge_parser.add_argument("--land-on-complete", action="store_true")
    bridge_parser.add_argument("--dry-run", action="store_true")
    bridge_parser.add_argument("--sample-spacing", type=float, default=12.0)
    bridge_parser.add_argument("--reverse-path", action="store_true")
    bridge_parser.add_argument("--no-auto-orient", action="store_true")
    bridge_parser.add_argument(
        "--redetect-every",
        type=int,
        default=5,
        help="Re-run the (heavy) track detector every N frames; higher = less lag.",
    )
    bridge_parser.add_argument(
        "--warmup-frames",
        type=int,
        default=60,
        help="Max frames to wait for a first valid track before takeoff.",
    )
    bridge_parser.add_argument(
        "--rise-fraction",
        type=float,
        default=0.25,
        help="Climb to this fraction of --height-cm (0.25 = a quarter of the way up).",
    )
    bridge_parser.add_argument(
        "--coast-frames",
        type=int,
        default=8,
        help="Keep steering from the last known drone position for up to N frames "
        "of marker dropout before hovering.",
    )

    train_parser = subparsers.add_parser("train-segmentation", help="Train the segmentation baseline")
    train_parser.add_argument("--images", required=True, type=Path)
    train_parser.add_argument("--masks", required=True, type=Path)
    train_parser.add_argument("--checkpoint", required=True, type=Path)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_on_path(args.method, args.input, args.output_dir)
        return

    if args.command == "extract-frames":
        extract_frames(args.input, args.output_dir, every_n=max(1, args.every_n))
        return

    if args.command == "manual-mission":
        frame, source_label = capture_calibration_frame(
            input_path=args.input,
            camera_index=args.camera_index,
            source=args.source,
        )
        mission, overlay = manual_mission_from_frame(
            frame=frame,
            source=source_label,
            reference_distance_cm=args.reference_distance_cm,
            sample_spacing_px=args.sample_spacing,
        )
        mission.save(args.output)
        cv2.imwrite(str(args.output.with_name(f"{args.output.stem}_overlay.png")), overlay)
        return

    if args.command == "generate-marker":
        generate_aruco_marker_image(
            output_path=args.output,
            dictionary_name=args.dictionary,
            marker_id=args.marker_id,
            side_pixels=args.side_px,
            margin_pixels=args.margin_px,
        )
        return

    if args.command == "live":
        run_live_camera(
            method=args.method,
            camera_index=args.camera_index,
            source=args.source,
            output_dir=args.output_dir,
            display=not args.no_display,
            max_frames=args.max_frames,
        )
        return

    if args.command == "export-path":
        export_mission_path(
            method=args.method,
            input_path=args.input,
            output_path=args.output,
            reverse_path=args.reverse_path,
            max_frames=args.max_frames,
            sample_spacing_px=args.sample_spacing,
        )
        return

    if args.command == "follow-track":
        follow_track_live(
            mission_path=args.mission,
            camera_index=args.camera_index,
            source=args.source,
            output_dir=args.output_dir,
            display=not args.no_display,
            max_frames=args.max_frames,
            command_rate_hz=args.command_rate,
            controller_config=TrackFollowerConfig(
                height_target_cm=args.height_cm,
                roll_sign=args.roll_sign,
                pitch_sign=args.pitch_sign,
                lost_frame_limit=args.vision_loss_frames,
                land_on_vision_loss=args.land_on_vision_loss,
                land_on_completion=args.land_on_complete,
            ),
            dry_run=args.dry_run,
            drone_method=args.drone_method,
        )
        return

    if args.command == "auto-follow":
        auto_follow_track(
            method=args.method,
            camera_index=args.camera_index,
            source=args.source,
            output_dir=args.output_dir,
            display=not args.no_display,
            max_frames=args.max_frames,
            command_rate_hz=args.command_rate,
            controller_config=TrackFollowerConfig(
                height_target_cm=args.height_cm,
                roll_sign=args.roll_sign,
                pitch_sign=args.pitch_sign,
                lost_frame_limit=args.vision_loss_frames,
                land_on_vision_loss=args.land_on_vision_loss,
                land_on_completion=args.land_on_complete,
            ),
            dry_run=args.dry_run,
            calibration_frames=args.calibration_frames,
            sample_spacing_px=args.sample_spacing,
            reverse_path=args.reverse_path,
            auto_orient=not args.no_auto_orient,
            drone_method=args.drone_method,
        )
        return

    if args.command == "follow-waypoints":
        follow_waypoints_live(
            mission_path=args.mission,
            camera_index=args.camera_index,
            source=args.source,
            output_dir=args.output_dir,
            calibration_frames=args.calibration_frames,
            drone_method=args.drone_method,
            target_height_m=args.height_m,
            waypoint_spacing_px=args.waypoint_spacing,
            speed_m_s=args.speed_m_s,
            meters_per_pixel=args.meters_per_pixel,
            position_tolerance_m=args.position_tolerance_m,
            waypoint_timeout_s=args.waypoint_timeout_s,
            reverse_path=args.reverse_path,
            auto_orient=not args.no_auto_orient,
            dry_run=args.dry_run,
        )
        return

    if args.command == "follow-waypoints-manual-start":
        follow_waypoints_manual_start(
            mission_path=args.mission,
            output_dir=args.output_dir,
            input_path=args.input,
            camera_index=args.camera_index,
            source=args.source,
            target_height_m=args.height_m,
            waypoint_spacing_px=args.waypoint_spacing,
            speed_m_s=args.speed_m_s,
            meters_per_pixel=args.meters_per_pixel,
            position_tolerance_m=args.position_tolerance_m,
            waypoint_timeout_s=args.waypoint_timeout_s,
            reverse_path=args.reverse_path,
            start_x=args.start_x,
            start_y=args.start_y,
            dry_run=args.dry_run,
        )
        return

    if args.command == "bridge":
        bridge_track_follow(
            method=args.method,
            camera_index=args.camera_index,
            source=args.source,
            output_dir=args.output_dir,
            display=not args.no_display,
            max_frames=args.max_frames,
            command_rate_hz=args.command_rate,
            controller_config=TrackFollowerConfig(
                height_target_cm=args.height_cm,
                roll_sign=args.roll_sign,
                pitch_sign=args.pitch_sign,
                lost_frame_limit=args.vision_loss_frames,
                land_on_vision_loss=args.land_on_vision_loss,
                land_on_completion=args.land_on_complete,
            ),
            dry_run=args.dry_run,
            sample_spacing_px=args.sample_spacing,
            reverse_path=args.reverse_path,
            auto_orient=not args.no_auto_orient,
            drone_method=args.drone_method,
            redetect_every=args.redetect_every,
            warmup_frames=args.warmup_frames,
            rise_fraction=args.rise_fraction,
            coast_frames=args.coast_frames,
        )
        return

    if args.command == "train-segmentation":
        train_segmentation_model(
            images_dir=args.images,
            masks_dir=args.masks,
            checkpoint_path=args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
