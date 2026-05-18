# wood_path — design notes

## Why a new method
`geometry.centerline_from_mask` and `threshold_morph`'s rowwise/columnwise
centerlines take the **mean x per scan row**. On the `track_snake`,
`track_curved_outwards` and U-shaped pieces a single row crosses the track
twice, so the mean collapses both arms into the gap between them. That is
exactly the "what path does the track take" case the current detectors miss.

## Pipeline
1. **Preprocess** — uniform-scale ROI (no fixed 320x240 squash, which warps
   curvature and turns tall pieces square); longest side → `working_long_side`.
2. **Wood mask** — Otsu on Lab b* (wood is yellow, desk is neutral), floored
   by a background-relative threshold from a border ring so pale wood is not
   clipped; ∩ warm/saturation to drop bluish desk specular; morphology;
   keep the largest elongated blob; **fill the outer contour** so the solid
   bar (grooves are surface channels, not silhouette holes) skeletonizes to a
   single clean spine instead of a noisy double-rail ladder.
3. **Skeletonize** — vectorized Zhang–Suen thinning → 1px medial axis
   (NumPy/OpenCV only; no scikit-image / opencv-contrib dependency).
4. **Graph nodes** — per-pixel 8-neighbour degree → endpoints (deg 1) and
   junctions (deg ≥ 3, i.e. track switches).
5. **Longest geodesic** — double-BFS diameter trick over the skeleton graph;
   spurs and short branches fall away automatically.
6. **Order + smooth** — orient bottom-of-image first (matches the other
   detectors' convention), stride-subsample, moving-average smooth.
7. **Describe route** — equal-arc coarse polyline → chord-deviation +
   straightness → `path_shape` ∈ {straight, curved_left, curved_right,
   snake, compound}, plus arc length, net turn, total curvature, endpoints,
   junction count.

## Output
Same `DetectionResult` contract as the built-in detectors, so the exported
mission JSON drops straight into `python -m track_detection.cli follow-track`.
Method id / package / CLI value: `wood_path`.

## Known limitations
- Single dominant path; a junction picks the longer geodesic branch, it does
  not enumerate every switch route.
- Otsu-on-b* + background-relative threshold is tuned for wood on a light
  desk; recalibrate `lab_b_min / lab_b_bg_margin / lab_a_lower` for other
  woods or backgrounds.
- Thinning cost grows with mask area; `max_thinning_iterations` caps it.
- Sample photos are oblique handheld shots: fine snake-vs-gentle-curve
  separation is not reliable from them, so the classifier only asserts
  categories with clean data separation (real `track_snake` reads as a strong
  single curve in this view); a true top-down camera will give cleaner shape
  labels and may need the colour thresholds re-checked under its lighting.
