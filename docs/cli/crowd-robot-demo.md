# Crowd Robot Demo

**Command:** `sfm-crowd-robot`

**Module:** [src/sfm_navigation/cli/crowd_robot_demo.py](../../src/sfm_navigation/cli/crowd_robot_demo.py)

## Purpose

This demo runs a robot through a real ATC crowd bin rather than a preprocessed single-user trajectory. It combines crowd analysis, filtering, controller selection, and animation in one pipeline.

## Scripted Function

- `main()`

## Inputs

- `--csv`: raw ATC crowd CSV
- `--bin`: density bin index
- `--subsample`: frame subsampling factor
- `--user-id`: shifted agent ID to use as the user
- `--safety-mode`: obstacle inclusion policy
- `--overtaking`: maneuver toggle
- `--robot-mood`: calibrated robot mood
- `--max-steps`: simulation cap
- `--dt-sim`: explicit simulation timestep
- `--frame-duration`: animation timing
- `--filter` / `--no-filter`: trajectory filtering toggle
- `--process-noise`, `--measurement-noise`: Kalman filter tuning
- `--savgol-window`, `--savgol-order`: Savitzky-Golay smoothing parameters
- `--filter-method`: KF or UKF
- `--controller`: controller family

## Execution Pipeline

1. Load raw ATC data using [data/atc_loader.py](../../src/sfm_navigation/data/atc_loader.py).
2. Convert units and bin the crowd data using [crowd_analysis/binning.py](../../src/sfm_navigation/crowd_analysis/binning.py).
3. Select one bin and identify the user trajectory.
4. Optionally filter each agent trajectory using [data/filtering.py](../../src/sfm_navigation/data/filtering.py).
5. Convert the selected user into a [UserTrajectory](../../src/sfm_navigation/agents/user.py).
6. Instantiate the chosen controller through [controllers/**init**.py](../../src/sfm_navigation/controllers/__init__.py).
7. Run a simulation loop where other crowd agents become moving obstacles.
8. Record history, compute collision analysis, generate a performance report, and export animation HTML.

## Theory and Implementation Links

### Crowd Binning

The crowd is binned by density so the demo can focus on a manageable local region. The implementation is in [crowd_analysis/binning.py](../../src/sfm_navigation/crowd_analysis/binning.py).

### Trajectory Filtering

The optional filter pipeline combines Kalman or Unscented Kalman filtering with Savitzky-Golay smoothing to reduce noise before constructing the user state. The implementation is in [data/filtering.py](../../src/sfm_navigation/data/filtering.py).

### Collision Analysis

Collision analysis is shared with [robot_demo.py](../../src/sfm_navigation/cli/robot_demo.py) through `analyze_robot_collisions()`.

## Outputs

- `*_crowd_robot_history.csv`
- `*_crowd_robot_performance.txt`
- `*_crowd_robot_animation.html`

## Best Use

Use this when you want the robot evaluated against a real crowd slice instead of an extracted user trajectory.
