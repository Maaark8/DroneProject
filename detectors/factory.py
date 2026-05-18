from __future__ import annotations


DETECTOR_METHODS = ("threshold_morph", "edge_geometry", "wood_path", "segmentation", "drone_light", "drone_marker")


def create_detector(method: str):
    if method == "threshold_morph":
        from .threshold_morph.detector import ThresholdMorphDetector

        return ThresholdMorphDetector()
    if method == "edge_geometry":
        from .edge_geometry.detector import EdgeGeometryDetector

        return EdgeGeometryDetector()
    if method == "wood_path":
        from .wood_path.detector import WoodPathDetector

        return WoodPathDetector()
    if method == "segmentation":
        from .segmentation.detector import SegmentationDetector

        return SegmentationDetector()
    if method == "drone_light":
        from .drone_light.detector import DroneLightDetector

        return DroneLightDetector()
    if method == "drone_marker":
        from .drone_marker.detector import ArucoMarkerDetector

        return ArucoMarkerDetector()
    raise ValueError(f"Unknown detector method: {method}")
