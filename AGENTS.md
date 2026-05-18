# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `track_detection/` and `detectors/`. Use `track_detection/` for shared pipeline, geometry, preprocessing, I/O, CLI entrypoints, and result schemas. Put detector-specific logic in `detectors/<method>/`, following the existing packages such as `threshold_morph/`, `edge_geometry/`, `segmentation/`, and `drone_light/`. Keep regression tests in `tests/`, evaluation notes in `evaluation/`, and sample inputs or generated overlays in data folders such as `tracks_for_drone/` and `real_outputs/`.

## Build, Test, and Development Commands
Create an editable install before local work:

```bash
python3 -m pip install -e .
```

Run the full test suite with:

```bash
python3 -m pytest
```

Exercise the offline pipeline:

```bash
python3 -m track_detection.cli run --method threshold_morph --input path/to/video.mp4 --output-dir outputs/run
```

Useful CLI helpers:

```bash
python3 -m track_detection.cli live --method drone_light --camera-index 0 --output-dir outputs/live
python3 -m track_detection.cli extract-frames --input path/to/video.mp4 --output-dir data/frames/sample --every-n 5
```

Install optional training dependencies only when working on segmentation:

```bash
python3 -m pip install -e .[train]
```

## Coding Style & Naming Conventions
Target Python 3.10+ and follow the existing style: 4-space indentation, type hints on public functions, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Match the current pattern of small focused modules, `dataclass` usage for structured state, and short doc-free code unless a non-obvious step needs a brief comment. No formatter or linter is configured here, so keep changes consistent with surrounding code.

## Testing Guidelines
Add or update `pytest` coverage for detector behavior changes. Name tests `test_<behavior>` and keep them deterministic by using synthetic frames, as in `tests/test_detectors.py`. When changing output schemas or control metadata, assert both validity and field-level values.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Added drone detection` and `Tune threshold_morph for real footage...`. Follow that style, keep one logical change per commit, and mention the affected detector or pipeline area. PRs should describe the scenario tested, list commands run, and include sample overlays or output paths when detection behavior changes.
