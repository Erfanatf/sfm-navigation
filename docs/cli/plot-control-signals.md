# Plot Control Signals

**Command:** `sfm-plot-control`

**Module:** [src/sfm_navigation/cli/plot_control_signals.py](../../src/sfm_navigation/cli/plot_control_signals.py)

## Purpose

This command plots detailed robot control and kinematic traces from a single simulation history CSV. It is more granular than the multi-controller comparison command because it includes acceleration, jerk, and error analysis.

## Scripted Function

- `main()`

## Inputs

- `--csv`: history CSV file

## Execution Pipeline

1. Load the history CSV.
2. Filter the robot rows.
3. Read the controller name for labeling.
4. Create a 5x2 Plotly subplot grid.
5. Plot command, maneuver, final, and executed velocity traces.
6. Plot acceleration, jerk, disturbance, velocity-tracking error, and distance-to-goal error.
7. Save and open the HTML report.

## Theory and Implementation Links

This command exposes low-level control behavior:

- controller command versus maneuver-adjusted versus executed values
- acceleration and jerk as smoothness indicators
- injected disturbance and disturbance-observer estimates
- position error as a convergence measure

Relevant controller and robot-state sources:

- [controllers/base_controller.py](../../src/sfm_navigation/controllers/base_controller.py)
- [controllers/maneuvers.py](../../src/sfm_navigation/controllers/maneuvers.py)
- [simulation/engine.py](../../src/sfm_navigation/simulation/engine.py)

## Output

- `*_control_signals_plot.html`

## Best Use

Use this to inspect how a single controller behaved internally over time, especially when debugging maneuver blending or disturbance handling.
