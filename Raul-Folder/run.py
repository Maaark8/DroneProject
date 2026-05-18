"""Standalone runner for the wood_path track tracer.

Self-contained: it imports the project's read-only building blocks but does
NOT modify the shared factory/follow modules, so everything lives in
Raul-Folder. It can render route overlays and export a mission JSON that the
existing `python -m track_detection.cli follow-track` command consumes.

Examples
--------
Trace every sample image and write overlays + a route report:

    python Raul-Folder/run.py trace --input tracks_for_drone --output-dir Raul-Folder/outputs

Export a drone mission path from one overhead photo:

    python Raul-Folder/run.py export-path \
        --input tracks_for_drone/track_snake.jpeg \
        --output Raul-Folder/outputs/track_snake_mission.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Make the repo importable even without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from track_detection.mission import mission_path_from_result  # noqa: E402
from track_detection.types import FrameInput  # noqa: E402

from wood_path_detector import WoodPathDetector  # noqa: E402

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _iter_images(path: Path):
    if path.is_dir():
        yield from sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    else:
        yield path


def _route_summary(metadata: dict) -> dict:
    keys = (
        "path_shape", "path_length_px", "net_turn_deg", "total_curvature_deg",
        "start_px", "end_px", "point_count", "endpoint_count", "junction_count",
        "coverage_ratio", "aspect_ratio", "rejected_reason",
    )
    return {k: metadata.get(k) for k in keys}


def cmd_trace(args: argparse.Namespace) -> int:
    detector = WoodPathDetector()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for image_path in _iter_images(Path(args.input)):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"skip (unreadable): {image_path}")
            continue
        result = detector.detect(FrameInput(frame=frame))
        overlay_path = output_dir / f"{image_path.stem}_wood_path_debug.png"
        if result.debug_frame is not None:
            cv2.imwrite(str(overlay_path), result.debug_frame)
        summary = _route_summary(result.metadata)
        summary["valid"] = result.valid
        summary["confidence"] = result.confidence
        report[image_path.name] = summary
        print(
            f"{image_path.name:32s} valid={result.valid} "
            f"conf={result.confidence:<5} shape={summary['path_shape']} "
            f"len_px={summary['path_length_px']} turn={summary['net_turn_deg']}"
        )

    report_path = output_dir / "route_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {report_path}")
    return 0


def cmd_export_path(args: argparse.Namespace) -> int:
    detector = WoodPathDetector()
    frame = cv2.imread(str(args.input))
    if frame is None:
        print(f"error: cannot read {args.input}")
        return 1
    result = detector.detect(FrameInput(frame=frame))
    if not result.centerline:
        reason = result.metadata.get("rejected_reason", "no_centerline")
        print(f"error: detector produced no path ({reason})")
        return 1

    mission = mission_path_from_result(
        result,
        frame_size=result.metadata.get("frame_size"),
        source=str(args.input),
        reverse=args.reverse,
        sample_spacing_px=args.spacing,
    )
    output_path = Path(args.output)
    mission.save(output_path)
    print(
        f"wrote {output_path} "
        f"({len(mission.points)} pts, {mission.path_length_px:.1f}px, "
        f"shape={result.metadata.get('path_shape')})"
    )
    print(
        "follow it with:\n"
        f"  python -m track_detection.cli follow-track --mission {output_path} "
        "--camera-index 0 --output-dir Raul-Folder/outputs/follow --dry-run"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wood_path track tracer")
    sub = parser.add_subparsers(dest="command", required=True)

    trace = sub.add_parser("trace", help="render route overlays + report")
    trace.add_argument("--input", default="tracks_for_drone", help="image file or folder")
    trace.add_argument("--output-dir", default="Raul-Folder/outputs")
    trace.set_defaults(func=cmd_trace)

    export = sub.add_parser("export-path", help="write a follow-track mission JSON")
    export.add_argument("--input", required=True, help="overhead track photo")
    export.add_argument("--output", required=True, help="mission JSON path")
    export.add_argument("--reverse", action="store_true", help="reverse path direction")
    export.add_argument("--spacing", type=float, default=12.0, help="resample spacing px")
    export.set_defaults(func=cmd_export_path)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
