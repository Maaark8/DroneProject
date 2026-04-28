# Learned Segmentation Detector

## Goal

Learn a frame-to-mask mapping for the wooden track, then convert the predicted
mask into a centerline for comparison against the classical methods.

## Inputs and outputs

- Input: preprocessed downward-facing RGB frame
- Output: predicted track mask, centerline points, heading, lateral offset,
  debug overlay

## Algorithm

1. Prepare paired RGB frames and binary track masks.
2. Train a lightweight U-Net style model on the labeled track masks.
3. Run sigmoid thresholding on model logits to recover a binary mask.
4. Keep the dominant connected component.
5. Extract and smooth the centerline from the mask.

## Tuning parameters

- working resolution
- mask threshold
- learning rate, batch size, epoch count
- checkpoint path and device

## Expected failure modes

- poor generalization if the training data covers only one lighting condition
- mask fragmentation on unseen backgrounds
- unstable predictions when the camera view differs from the labeled dataset
