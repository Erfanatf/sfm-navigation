# Compare Control Signals

**Command:** `sfm-compare-controllers-signals`

**Module:** [src/sfm_navigation/cli/compare_control_signals.py](../../src/sfm_navigation/cli/compare_control_signals.py)

## Purpose

This command compares controller signal traces across multiple simulation runs. It is a lighter-weight companion to the animation comparison command and focuses on velocities and disturbance signals rather than geometry.

## Scripted Function

- `main()`

## Inputs

- one or more history CSV files
- `--output`: output HTML file

## Execution Pipeline

1. Read each simulation history file.
2. Extract the robot rows and controller name.
3. Plot linear velocity, angular velocity, and disturbance traces in a four-row Plotly subplot grid.
4. Export and open the HTML report.

## Theory and Implementation Links

This command is useful for examining control-loop behavior at the signal level:

- commanded velocity versus executed velocity
- linear and angular disturbance channels
- optional disturbance observer estimates

Relevant controller implementations:

- [controllers/base_controller.py](../../src/sfm_navigation/controllers/base_controller.py)
- [controllers/dwa/](../../src/sfm_navigation/controllers/dwa/)
- [controllers/mpc/](../../src/sfm_navigation/controllers/mpc/)

## Output

- `control_comparison.html`

## Best Use

Use this when you want a compact signal-level comparison across controllers without the overhead of full animation.
