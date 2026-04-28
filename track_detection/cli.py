from __future__ import annotations

import argparse
from pathlib import Path

from detectors.segmentation.train import train_segmentation_model

from .pipeline import run_on_path
from .video import extract_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track detection and evaluation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a detector on a video or image folder")
    run_parser.add_argument("--method", required=True, choices=["threshold_morph", "edge_geometry", "segmentation"])
    run_parser.add_argument("--input", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)

    extract_parser = subparsers.add_parser("extract-frames", help="Extract frames from a video")
    extract_parser.add_argument("--input", required=True, type=Path)
    extract_parser.add_argument("--output-dir", required=True, type=Path)
    extract_parser.add_argument("--every-n", type=int, default=5)

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
