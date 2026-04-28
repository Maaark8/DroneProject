# Threshold + Morphology Detector

## Goal

Segment the wooden track by color and texture cues, then recover the centerline
from the dominant cleaned mask.

## Inputs and outputs

- Input: preprocessed downward-facing RGB frame
- Output: binary track mask, centerline points, heading, lateral offset, debug
  overlay

## Algorithm

1. Convert the ROI to HSV and Lab.
2. Keep warm brown pixels with configurable HSV and Lab `a` thresholds.
3. Apply close/open morphology to repair gaps and suppress small clutter.
4. Keep the largest connected component as the track candidate.
5. Extract row-wise center points from the mask and smooth them into a centerline.

## Tuning parameters

- HSV lower/upper bounds
- Lab warm-channel bounds
- close/open kernel size
- ROI crop and working resolution

## Expected failure modes

- wood-like background colors
- extreme shadows or overexposure
- track only partially visible near the ROI boundary
