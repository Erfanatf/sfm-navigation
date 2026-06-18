# Robot Demo

**Command:** `sfm-robot-demo`

**Module:** [src/sfm_navigation/cli/robot_demo.py](../../src/sfm_navigation/cli/robot_demo.py)

## Purpose

This is the most feature-rich robot-facing demo in the repository. It loads processed ATC trajectories, builds a user trajectory, spawns pedestrians and static/dynamic obstacles, and runs the selected controller against that scene while recording detailed robot and pedestrian history.

The command is the main integration point for:

- SFM-based pedestrian dynamics
- calibrated mood loading from CSV files
- DWA, MPC, MPPI, NMPC, and CBF controller families
- optional LSTM user-path prediction
- optional disturbance injection and controller-fallback logic
- collision analysis and HTML animation export

## Scripted Functions

- `_auto_register_calibrated_moods(moods_dir="data/calibrated_moods")`
- `analyze_robot_collisions(history_df, safety_margin, cooldown_s=0.5)`
- `robot_in_collision(rx, ry, rrad, agents, safety_margin)`
- `main()`

## Inputs

Important CLI arguments include:

- `--data-folder`: ATC trajectory folder
- `--controller`: controller family to instantiate
- `--robot-mood`: calibrated mood for the robot itself
- `--safety-mode`: obstacle inclusion policy
- `--overtaking`, `--repulsion`, `--parking`, `--rotation`: maneuver toggles
- `--mood-switch-rate`, `--transition-matrix`: pedestrian mood dynamics
- `--disturbance-active`: inject external disturbance on robot dynamics
- `--use-cbf-opt`, `--dcbf-fallback-mode`: DCBF safety projection mode
- `--use-lstm-predictor`: enable future user-path prediction

## Execution Pipeline

1. Load processed trajectories with [data/loader.py](../../src/sfm_navigation/data/loader.py).
2. Convert a selected trajectory into a user path through [agents/user.py](../../src/sfm_navigation/agents/user.py).
3. Auto-register calibrated moods from `data/calibrated_moods/` using [data/moods.py](../../src/sfm_navigation/data/moods.py).
4. Create an [SFMPedestrianSpawner](../../src/sfm_navigation/spawner/spawner.py) and generate safe static obstacles.
5. Optionally load a mood transition matrix and LSTM predictor.
6. Instantiate the requested controller through [controllers/**init**.py](../../src/sfm_navigation/controllers/__init__.py).
7. Run a fixed-rate control loop that updates pedestrians, computes goals, calls the controller, injects disturbance, and advances the robot.
8. Record robot, user, pedestrian, obstacle, maneuver, and disturbance history.
9. Run collision analysis and export simulation history, mood switch logs, and Plotly animation HTML.

## Theory and Implementation Links

### Social Force Model

Pedestrian behavior is represented as an SFM force balance: goal attraction, obstacle avoidance, interpersonal repulsion, and anisotropic attention. The per-mood parameters come from [data/moods.py](../../src/sfm_navigation/data/moods.py) and calibrated CSV files.

### Controller Families

- [controllers/sfm_controller.py](../../src/sfm_navigation/controllers/sfm_controller.py) for direct SFM control
- [controllers/dwa/](../../src/sfm_navigation/controllers/dwa/) for Dynamic Window Approach and velocity-obstacle variants
- [controllers/mpc/](../../src/sfm_navigation/controllers/mpc/) for NMPC, MPPI, and CBF variants

### Prediction

The optional user predictor is [prediction/lstm_predictor.py](../../src/sfm_navigation/prediction/lstm_predictor.py), which uses a pre-trained sequence model to forecast the user's future goal.

### Simulation and Visualization

- [simulation/engine.py](../../src/sfm_navigation/simulation/engine.py) for the broader simulation model used across the project
- [visualization/animation.py](../../src/sfm_navigation/visualization/animation.py) for HTML animation generation

## Outputs

- `*_simulation_history.csv`
- `*_mood_switch_log.csv`
- `*_robot_sfm_animation.html`

## Notes on Coverage

This command is the best place to study end-to-end integration because it exercises controller selection, pedestrian spawning, collision tracking, maneuver logic, disturbance handling, and calibrated mood registration in one execution path.
