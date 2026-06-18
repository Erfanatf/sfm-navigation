# Demo

**Command:** `sfm-demo`

**Module:** [src/sfm_navigation/cli/demo.py](../../src/sfm_navigation/cli/demo.py)

## Purpose

This is the baseline crowd simulation demo. It loads a preprocessed trajectory, spawns SFM pedestrians, places obstacles, and renders the resulting crowd animation without a robot controller in the loop.

## Scripted Function

- `_auto_register_calibrated_moods(moods_dir='data/calibrated_moods')`
- `main()`

## Execution Pipeline

1. Load processed trajectories from disk.
2. Create a user trajectory from the selected trajectory sample.
3. Auto-register calibrated moods from CSV files.
4. Configure the [SFMPedestrianSpawner](../../src/sfm_navigation/spawner/spawner.py).
5. Generate safe static obstacles.
6. Run [create_sfm_spawner_animation](../../src/sfm_navigation/visualization/animation.py) to build the interactive animation.
7. Save the HTML file and optionally open the browser.

## Theory and Implementation Links

This demo is useful for studying the crowd-generation side of the system:

- pedestrian spawning and respawn logic in [spawner/spawner.py](../../src/sfm_navigation/spawner/spawner.py)
- mood-driven SFM parameters in [data/moods.py](../../src/sfm_navigation/data/moods.py)
- trajectory-to-user conversion in [agents/user.py](../../src/sfm_navigation/agents/user.py)

## Outputs

- `sfm_animation.html`

## Best Use

Use this as the simplest way to inspect the crowd generator and calibrated pedestrian behaviors without robot interaction.
