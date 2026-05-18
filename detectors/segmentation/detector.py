from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
from typing import Any

import cv2
import numpy as np

from detectors.base import BaseTrackDetector
from track_detection.geometry import centerline_from_mask, heading_and_offset, overlay_detection
from track_detection.geometry import largest_component
from track_detection.types import DetectionResult, FrameInput, PreprocessConfig

from .model import SimpleUNet, torch


@dataclass(slots=True)
class SegmentationConfig(PreprocessConfig):
    threshold: float = 0.5
    checkpoint_path: str | None = None
    device: str = "cpu"
    backend: str = "auto"
    yolo_confidence: float = 0.15
    yolo_imgsz: int | None = None
    default_yolo_checkpoint: str = "model/exp.pt"


class SegmentationDetector(BaseTrackDetector):
    method_name = "segmentation"
    overlay_color = (255, 120, 80)

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        config = config or SegmentationConfig()
        super().__init__(config)
        self.checkpoint_path = _resolve_checkpoint_path(config)
        self.backend = _resolve_backend(config.backend, self.checkpoint_path)

        if self.backend == "yolo":
            self._init_yolo()
            return

        self._init_unet()

    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self.backend == "yolo":
            return self._detect_mask_yolo(working_frame)

        return self._detect_mask_unet(working_frame)

    def detect(self, frame_input: FrameInput) -> DetectionResult:
        if self.backend != "yolo":
            return super().detect(frame_input)

        mask, metadata = self._detect_mask_yolo(frame_input.frame)
        centerline = centerline_from_mask(
            mask,
            stride=self.config.centerline_stride,
            min_points=self.config.min_centerline_points,
        )
        heading_rad, lateral_offset_px = heading_and_offset(centerline, frame_input.frame.shape[1])
        valid = len(centerline) >= self.config.min_centerline_points
        confidence = self._compute_yolo_confidence(mask, centerline, metadata)
        debug_frame = overlay_detection(frame_input.frame, mask, centerline, self.overlay_color)
        metadata["mask_pixels"] = int(np.count_nonzero(mask))
        metadata["frame_id"] = frame_input.frame_id
        metadata["frame_size"] = {"width": int(frame_input.frame.shape[1]), "height": int(frame_input.frame.shape[0])}
        metadata["source_backend"] = "yolo"

        return DetectionResult(
            method=self.method_name,
            centerline=centerline,
            heading_rad=heading_rad,
            lateral_offset_px=lateral_offset_px,
            confidence=confidence,
            valid=valid,
            debug_frame=debug_frame,
            metadata=metadata,
        )

    def _init_unet(self) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for the U-Net segmentation detector. "
                "Install the 'train' extra or provide an Ultralytics YOLO segmentation checkpoint."
            )

        self.device = torch.device(self.config.device)
        self.model = SimpleUNet().to(self.device)
        self.model.eval()

        if self.checkpoint_path:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    def _init_yolo(self) -> None:
        if self.checkpoint_path is None:
            raise ValueError("YOLO segmentation backend requires a checkpoint path.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required to use the YOLO segmentation checkpoint. "
                "Install `ultralytics` (and its PyTorch dependency) in the active environment."
            ) from exc

        self.model = YOLO(str(self.checkpoint_path))
        self.device = self.config.device

    def _detect_mask_unet(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        rgb = working_frame[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask = np.where(probs >= self.config.threshold, 255, 0).astype(np.uint8)
        mask = largest_component(mask)
        mean_probability = float(probs.mean())
        return mask, {"mean_probability": round(mean_probability, 4), "backend": "unet"}

    def _detect_mask_yolo(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        predict_kwargs = {
            "verbose": False,
            "conf": float(self.config.yolo_confidence),
            "device": self.config.device,
        }
        if self.config.yolo_imgsz is not None:
            predict_kwargs["imgsz"] = int(self.config.yolo_imgsz)

        results = self.model.predict(working_frame, **predict_kwargs)
        if not results:
            return np.zeros(working_frame.shape[:2], dtype=np.uint8), {
                "backend": "yolo",
                "instance_count": 0.0,
                "selected_confidence": 0.0,
            }

        result = results[0]
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            return np.zeros(working_frame.shape[:2], dtype=np.uint8), {
                "backend": "yolo",
                "instance_count": 0.0,
                "selected_confidence": 0.0,
            }

        masks = result.masks.data.detach().cpu().numpy()
        confidences = None
        if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
            confidences = result.boxes.conf.detach().cpu().numpy()

        selected_mask, selected_confidence = _select_yolo_mask(
            masks=masks,
            confidences=confidences,
            target_shape=working_frame.shape[:2],
            threshold=self.config.threshold,
        )
        selected_mask = largest_component(selected_mask)
        area_ratio = float(np.count_nonzero(selected_mask)) / float(selected_mask.size)
        return selected_mask, {
            "backend": "yolo",
            "instance_count": float(len(masks)),
            "selected_confidence": round(float(selected_confidence), 4),
            "area_ratio": round(area_ratio, 4),
        }

    def _compute_yolo_confidence(
        self,
        mask: np.ndarray,
        centerline: list[tuple[float, float]],
        metadata: dict[str, Any],
    ) -> float:
        detection_confidence = float(metadata.get("selected_confidence", 0.0))
        mask_ratio = float(np.count_nonzero(mask)) / float(max(mask.size, 1))
        point_ratio = min(1.0, len(centerline) / float(max(self.config.min_centerline_points, 1)))
        combined = (0.6 * detection_confidence) + (0.25 * point_ratio) + (0.15 * min(mask_ratio * 8.0, 1.0))
        return round(min(max(combined, 0.0), 1.0), 3)


def _select_yolo_mask(
    masks: np.ndarray,
    confidences: np.ndarray | None,
    target_shape: tuple[int, int],
    threshold: float,
) -> tuple[np.ndarray, float]:
    best_mask = np.zeros(target_shape, dtype=np.uint8)
    best_score = -1.0
    best_confidence = 0.0

    for index, raw_mask in enumerate(masks):
        resized = raw_mask
        if resized.shape != target_shape:
            resized = cv2.resize(resized, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        binary = np.where(resized >= float(threshold), 255, 0).astype(np.uint8)
        area = float(np.count_nonzero(binary))
        confidence = 0.0 if confidences is None or index >= len(confidences) else float(confidences[index])
        score = area * max(confidence, 0.05)
        if score <= best_score:
            continue
        best_mask = binary
        best_score = score
        best_confidence = confidence
    return best_mask, best_confidence


def _resolve_checkpoint_path(config: SegmentationConfig) -> Path | None:
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    default_path = Path(config.default_yolo_checkpoint)
    if default_path.exists():
        return default_path
    return None


def _resolve_backend(backend: str, checkpoint_path: Path | None) -> str:
    normalized = backend.strip().lower()
    if normalized in {"yolo", "unet"}:
        return normalized
    if normalized != "auto":
        raise ValueError(f"Unknown segmentation backend: {backend!r}")
    if checkpoint_path is None:
        return "unet"
    return _infer_checkpoint_backend(checkpoint_path)


def _infer_checkpoint_backend(checkpoint_path: Path) -> str:
    if not checkpoint_path.exists():
        return "unet"
    if checkpoint_path.name == "exp.pt" and checkpoint_path.parent.name == "model":
        return "yolo"
    try:
        with checkpoint_path.open("rb") as handle:
            header = handle.read(512 * 1024)
        if b"ultralytics" in header.lower() or b"yolo" in header.lower():
            return "yolo"
    except OSError:
        return "unet"
    try:
        if zipfile.is_zipfile(checkpoint_path):
            with zipfile.ZipFile(checkpoint_path) as archive:
                for name in archive.namelist():
                    lowered = name.lower()
                    if "ultralytics" in lowered or "yolo" in lowered:
                        return "yolo"
    except OSError:
        return "unet"
    return "unet"
