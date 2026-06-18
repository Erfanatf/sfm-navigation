# Control Signal Comparison

**Command:** `sfm-compare-controllers-signals`

**Module:** [src/sfm_navigation/cli/compare_control_signals.py](../../src/sfm_navigation/cli/compare_control_signals.py)

## Purpose

This command overlays control and disturbance signals from multiple simulation history CSV files into a single Plotly figure. It is intended for comparing controller behavior after a simulation has already been run.

## Scripted Function

- `main()`

## Inputs

- one or more history CSV files
- `--output`: output HTML file name

## Execution Pipeline

1. Load each CSV with pandas.
2. Extract robot rows only.
3. Read the controller name from the first robot row.
4. Plot linear and angular velocity traces plus disturbance channels on four aligned subplots.
5. Save the figure as HTML and open it in the browser.

## Theory and Implementation Links

This command is not a simulator; it is a post-processing comparison tool.

- The linear and angular velocity traces reflect the controller output and the executed robot motion.
- The disturbance traces visualize injected external disturbance versus disturbance observer estimates when those columns exist.
- The controller data being compared comes from [controllers/](../../src/sfm_navigation/controllers/), especially the DWA and MPC families.

Relevant source files:

- [controllers/base_controller.py](../../src/sfm_navigation/controllers/base_controller.py)
- [controllers/dwa/](../../src/sfm_navigation/controllers/dwa/)
- [controllers/mpc/](../../src/sfm_navigation/controllers/mpc/)

## Output

- `control_comparison.html`

## What to Inspect

Use this command to compare how controllers differ in commanded velocity, final applied velocity, and disturbance rejection under similar scenarios.
