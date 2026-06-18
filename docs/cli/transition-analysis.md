# Transition Analysis

**Command:** `sfm-transition-analysis`

**Module:** [src/sfm_navigation/cli/transition_analysis_cli.py](../../src/sfm_navigation/cli/transition_analysis_cli.py)

## Purpose

This command analyzes behavior or regime transitions from a completed pipeline run. It is a follow-up analysis step rather than a simulation step.

## Scripted Function

- `main()`

## Inputs

- `--run-dir`: pipeline run directory
- `--output-dir`: analysis output directory

## Execution Pipeline

1. Parse the run directory.
2. Call [behavior_pipeline/transition_analysis.py](../../src/sfm_navigation/behavior_pipeline/transition_analysis.py).
3. Generate transition matrices and summary artifacts.

## Theory and Implementation Links

This command is based on discrete transition analysis over extracted regimes or moods. It is the natural post-processing step after clustering and calibration in the pipeline.

Relevant implementation:

- [behavior_pipeline/transition_analysis.py](../../src/sfm_navigation/behavior_pipeline/transition_analysis.py)

## Output

- transition matrices and transition-analysis reports under the chosen output directory

## Best Use

Use this immediately after a pipeline run when you want to understand how often regimes or moods change over time.
