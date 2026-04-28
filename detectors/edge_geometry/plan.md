# Edge + Geometry Detector

## Goal

Recover the track as a geometric object when color thresholds are unreliable.

## Inputs and outputs

- Input: preprocessed downward-facing RGB frame
- Output: candidate track contour, centerline points, heading, lateral offset,
  debug overlay

## Algorithm

1. Convert the ROI to grayscale and normalize contrast.
2. Run Canny edge detection.
3. Dilate and close the edge map to connect the track outline.
4. Find external contours and score them by area, vertical span, and distance
   from image center.
5. Fill the best contour to create a mask.
6. Extract and smooth the centerline from the mask.

## Tuning parameters

- Canny low/high thresholds
- dilation and close kernel sizes
- minimum contour area
- ROI crop and working resolution

## Expected failure modes

- weak edges from low texture contrast
- strong background contours near the image center
- large occlusions that break the track outline
