# Compare Animation

**Command:** `sfm-compare-controllers-anim`

**Module:** [src/sfm_navigation/cli/compare_animations.py](../../src/sfm_navigation/cli/compare_animations.py)

## Purpose

This command renders multiple controller histories side by side in one synchronized animation. It is the most detailed visual comparison tool for controller behavior in the repository.

## Scripted Functions

- `ellipse_points(cx, cy, angle, a, b, n=30)`
- `cone_points(cx, cy, direction, half, length=1.2, n=20)`
- `load_history(path)`
- `build_frames(df)`
- `main()`

## Inputs

- one or more history CSV files
- `--follow`: follow-camera mode
- `--zoom`: follow radius
- `--output`: output HTML file
- `--frame-skip`: frame decimation

## Execution Pipeline

1. Load every history file and extract controller name, user trajectory state, static obstacles, and safety radius.
2. Build per-file animation frames from timestamped history records.
3. Sort controllers into consistent groups before plotting.
4. Create a grid layout sized to the number of controllers.
5. Reconstruct rich geometry for each frame: user ellipse, pedestrian ellipse, heading, gaze, attention cone, robot body, robot trail, goal, future path, safety circles, and obstacle outlines.
6. Build synchronized Plotly frames and sliders so all panels play together.
7. Export HTML and open it in the browser.

## Theory and Implementation Links

This command visualizes several theoretical layers at once:

- **pedestrian body and personal space** via ellipse geometry
- **attention and field of view** via gaze cone construction
- **robot motion prediction** via future-arc reconstruction from velocity commands
- **controller family comparison** through histories produced by [controllers/](../../src/sfm_navigation/controllers/)

Relevant source files:

- [visualization/animation.py](../../src/sfm_navigation/visualization/animation.py)
- [data/moods.py](../../src/sfm_navigation/data/moods.py)
- [controllers/dwa/](../../src/sfm_navigation/controllers/dwa/)
- [controllers/mpc/](../../src/sfm_navigation/controllers/mpc/)

## Output

- `comparison_animation.html`

## Best Use

Use this when you want synchronized visual comparison of several controllers on the same scenario, especially to inspect path shape, safety margins, and maneuver activation timing.
