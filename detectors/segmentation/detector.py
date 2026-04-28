from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detectors.base import BaseTrackDetector
from track_detection.geometry import largest_component
from track_detection.types import PreprocessConfig

from .model import SimpleUNet, torch


@dataclass(slots=True)
class SegmentationConfig(PreprocessConfig):
    threshold: float = 0.5
    checkpoint_path: str | None = None
    device: str = "cpu"


class SegmentationDetector(BaseTrackDetector):
    method_name = "segmentation"
    overlay_color = (255, 120, 80)

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        config = config or SegmentationConfig()
        super().__init__(config)
        if torch is None:
            raise ImportError("PyTorch is required for the segmentation detector. Install the 'train' extra.")

        self.device = torch.device(config.device)
        self.model = SimpleUNet().to(self.device)
        self.model.eval()

        if config.checkpoint_path:
            state_dict = torch.load(Path(config.checkpoint_path), map_location=self.device)
            self.model.load_state_dict(state_dict)

    def _detect_mask(self, working_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        rgb = working_frame[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask = np.where(probs >= self.config.threshold, 255, 0).astype(np.uint8)
        mask = largest_component(mask)
        mean_probability = float(probs.mean())
        return mask, {"mean_probability": round(mean_probability, 4)}
