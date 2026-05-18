from __future__ import annotations


DETECTOR_METHODS = ("threshold_morph", "edge_geometry", "segmentation", "drone_light")


def create_detector(method: str):
    if method == "threshold_morph":
        from .threshold_morph.detector import ThresholdMorphDetector

        return ThresholdMorphDetector()
    if method == "edge_geometry":
        from .edge_geometry.detector import EdgeGeometryDetector

        return EdgeGeometryDetector()
    if method == "segmentation":
        from .segmentation.detector import SegmentationDetector

        return SegmentationDetector()
    if method == "drone_light":
        from .drone_light.detector import DroneLightDetector

        return DroneLightDetector()
    raise ValueError(f"Unknown detector method: {method}")
