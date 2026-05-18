from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Point = tuple[float, float]


@dataclass(slots=True)
class FrameInput:
    frame: np.ndarray
    frame_id: int = 0
    timestamp_s: float | None = None


@dataclass(slots=True)
class PreprocessConfig:
    roi_top: float = 0.20
    roi_bottom: float = 1.00
    working_width: int = 320
    working_height: int = 240
    blur_kernel: int = 5
    centerline_stride: int = 8
    min_centerline_points: int = 8
    min_track_pixels: int = 300


@dataclass(slots=True)
class DetectionResult:
    method: str
    centerline: list[Point]
    heading_rad: float | None
    lateral_offset_px: float | None
    confidence: float
    valid: bool
    debug_frame: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
