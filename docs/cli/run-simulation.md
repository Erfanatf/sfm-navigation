# Run Simulation

**Command:** `sfm-run`

**Module:** [src/sfm_navigation/cli/run_simulation.py](../../src/sfm_navigation/cli/run_simulation.py)

## Purpose

This is a minimal general-purpose simulation runner. It loads configuration from JSON, reads processed trajectories, creates a user trajectory, and runs the spawner animation with default settings.

## Scripted Function

- `main()`

## Inputs

- `--config`: JSON configuration file
- `--data`: ATC data folder

## Execution Pipeline

1. Load the JSON config into [config.py](../../src/sfm_navigation/config.py).
2. Load multiple trajectories from the ATC data folder.
3. Convert the first trajectory into a user trajectory.
4. Create the pedestrian spawner and safe static obstacles.
5. Render the default spawner animation.

## Theory and Implementation Links

This command is intentionally light on features. It exists to provide a simple baseline path into the simulation stack while still using the same core modules as the richer demos.

Relevant files:

- [config.py](../../src/sfm_navigation/config.py)
- [data/loader.py](../../src/sfm_navigation/data/loader.py)
- [agents/user.py](../../src/sfm_navigation/agents/user.py)
- [spawner/spawner.py](../../src/sfm_navigation/spawner/spawner.py)
- [visualization/animation.py](../../src/sfm_navigation/visualization/animation.py)

## Output

- interactive animation shown through Plotly in the browser or notebook environment

## Best Use

Use this as a compact smoke test for the trajectory loading and spawner pipeline.
