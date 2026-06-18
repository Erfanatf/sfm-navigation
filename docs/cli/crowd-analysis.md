# Crowd Analysis

**Command:** `sfm-crowd`

**Module:** [src/sfm_navigation/cli/crowd_analysis.py](../../src/sfm_navigation/cli/crowd_analysis.py)

## Purpose

This command analyzes raw ATC crowd data, bins it by density, selects one bin, and renders a crowd animation for that bin. It is the simplest analysis command in the repository.

## Scripted Function

- `main()`

## Inputs

- `--csv`: raw ATC CSV file
- `--bin`: density bin index
- `--subsample`: frame subsampling factor

## Execution Pipeline

1. Load raw ATC data.
2. Convert measurement units.
3. Bin trajectories by crowd density.
4. Select the requested bin and longest trajectory.
5. Render the bin with [crowd_analysis/visualization.py](../../src/sfm_navigation/crowd_analysis/visualization.py).
6. Write the HTML animation and open it in a browser.

## Theory and Implementation Links

The analysis is rooted in crowd-density segmentation and trajectory replay:

- density binning logic: [crowd_analysis/binning.py](../../src/sfm_navigation/crowd_analysis/binning.py)
- raw data loading: [data/atc_loader.py](../../src/sfm_navigation/data/atc_loader.py)
- visualization: [crowd_analysis/visualization.py](../../src/sfm_navigation/crowd_analysis/visualization.py)

## Output

- `atc_crowd_animation.html`

## Best Use

Use this to inspect a raw ATC scene before running the more complex robot and calibration pipelines.
