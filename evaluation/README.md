# Evaluation Outputs

Offline runs write their artifacts into an output directory you choose at run
time.

Expected outputs:

- `results.jsonl` with one detection result per frame
- `debug_frames/` when running on image folders
- `debug_overlay.mp4` when running on videos

Each result entry records the common detector schema so you can compare methods
with your own metrics later.
