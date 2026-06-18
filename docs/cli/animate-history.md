# Animate History

**Command:** `sfm-animate`

**Module:** [src/sfm_navigation/cli/animate_history.py](../../src/sfm_navigation/cli/animate_history.py)

## Purpose

This command replays a saved simulation history CSV without re-running the physics or controller loop. It reconstructs the user, robot, pedestrians, and obstacles from recorded states and renders them into an interactive HTML animation.

## Scripted Function

- `main()`

## Inputs

- `--csv`: history CSV file to replay
- `--frame-skip`: keep every Nth frame
- `--follow-user` / `--no-follow-user`: camera mode
- `--follow-zoom-radius`: follow-camera radius
- `--frame-duration`: animation timing
- `--draw-pedestrian-safety`: overlay pedestrian safety circles
- `--output`: output HTML file

## Execution Pipeline

1. Load the history CSV.
2. Filter user rows and rebuild a [UserTrajectory](../../src/sfm_navigation/agents/user.py) from position, yaw, and velocity columns.
3. Reconstruct static obstacles from their first recorded occurrence.
4. Iterate over unique timestamps and rebuild a frame payload for the animation helper.
5. Recompute robot arcs and motion direction traces from recorded control values when available.
6. Call [create_animation_from_frames](../../src/sfm_navigation/visualization/animation.py).
7. Save and open the rendered HTML.

## Theory and Implementation Links

This command is a visualization layer on top of logged simulation state.

- Motion decomposition into forward and orthogonal components is handled with the robot and user state conventions used throughout [agents/user.py](../../src/sfm_navigation/agents/user.py).
- The animation renderer comes from [visualization/animation.py](../../src/sfm_navigation/visualization/animation.py).

## Output

- `history_animation.html`

## Best Use

Use this when the simulation has already been run and you want to inspect trajectories, goal evolution, robot arcs, and pedestrian layout with no recomputation.
