from __future__ import annotations

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
    raise ValueError(f"Unknown detector method: {method}")
