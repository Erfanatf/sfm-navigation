# Pipeline CLI

**Command:** `sfm-pipeline`

**Module:** [src/sfm_navigation/cli/pipeline_cli.py](../../src/sfm_navigation/cli/pipeline_cli.py)

## Purpose

This command runs the full behavior profiling and calibration pipeline. It is the orchestration layer for loading raw crowd data, binning it, extracting features, clustering behavioral regimes, and calibrating Extended SFM parameters.

## Scripted Function

- `main()`

## Inputs

- `--data-source`: ATC CSV or vision recording file
- `--output-dir`: pipeline result directory
- `--log-level`: logging verbosity
- `--n-top-bins`: how many dense bins to process
- `--calibrate-regime`: regime code or auto selection
- `--stages`: subset of stages to run
- `--load-run-id`: reuse data from an existing run
- `--max-windows`: cap the number of windows processed
- `--dt-sim`: override calibration timestep

## Execution Pipeline

1. Build a [PipelineConfig](../../src/sfm_navigation/behavior_pipeline/config.py).
2. Pass the config to [PipelineRunner](../../src/sfm_navigation/behavior_pipeline/pipeline_runner.py).
3. Run the stage chain defined by the pipeline runner.
4. Emit calibration artifacts to the output directory.

## Theory and Implementation Links

The pipeline combines several scientific ideas:

- temporal binning and density selection
- window-based trajectory segmentation
- kinematic, path, safety, and social feature extraction
- clustering of behavioral regimes
- parameter optimization for SFM calibration

Implementation files:

- [behavior_pipeline/config.py](../../src/sfm_navigation/behavior_pipeline/config.py)
- [behavior_pipeline/pipeline_runner.py](../../src/sfm_navigation/behavior_pipeline/pipeline_runner.py)
- [behavior_pipeline/stages/](../../src/sfm_navigation/behavior_pipeline/stages/)

## Output

- `pipeline_results/`

## Best Use

Use this when you want to regenerate calibrated mood parameters or inspect the calibration chain itself.
