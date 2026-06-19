import numpy as np
import sys
from ..data.loader import load_multiple_trajectories
from ..agents.user import create_user_trajectory_from_processed_data
from ..spawner.spawner import SFMPedestrianSpawner, generate_safe_static_obstacles
from ..config import CONFIG
from ..visualization.animation import create_sfm_spawner_animation
from ..data.moods import load_calibrated_moods
import webbrowser
import os
from ..data.moods import register_mood

def _auto_register_calibrated_moods(moods_dir='data/calibrated_moods'):
    """Scan the directory for CSV files and register each as a mood."""
    if not os.path.isdir(moods_dir):
        return
    for fname in os.listdir(moods_dir):
        if fname.endswith('.csv'):
            mood_name = os.path.splitext(fname)[0]
            csv_path = os.path.join(moods_dir, fname)
            try:
                register_mood(csv_path, mood_name)
            except Exception as e:
                print(f"Warning: Failed to register {mood_name} from {csv_path}: {e}")

def main():
    # Configuration (same as the notebook)
    TRAJECTORY_INDEX = -1
    SIMULATION_DURATION = None
    SFM_DT = None
    FRAME_SKIP = 3
    FOLLOW_USER = True
    FOLLOW_ZOOM_RADIUS = 15.0
    N_PEDESTRIANS = 25
    VICINITY_RADIUS = 8.0
    RESPAWN_DISTANCE = 20.0
    MIN_SPAWN_DISTANCE = 1.0
    MAX_SPAWN_DISTANCE = 15.0
    SPEED_SCALE_FACTOR = 1.0
    N_STATIC_OBSTACLES = 50
    MIN_OBSTACLE_RADIUS = 0.3
    MAX_OBSTACLE_RADIUS = 2.0
    OBSTACLE_SAFETY_MARGIN = 1.0
    ENABLE_LOGGING = True
    LOG_WINDOW_DURATION = 5.0

    # Path to ATC data (adjust as needed, or pass as argument)
    DRIVE_DATA_FOLDER = CONFIG.atc_csv_folder
    # For flexibility, allow command line argument
    if len(sys.argv) > 1:
        DRIVE_DATA_FOLDER = sys.argv[1]

    print("Loading trajectories...")
    try:
        all_x_true, all_dt, AVG_DT = load_multiple_trajectories(DRIVE_DATA_FOLDER, '*.csv', n=24)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    trajectory_idx = TRAJECTORY_INDEX
    if trajectory_idx >= len(all_x_true):
        print(f"WARNING: trajectory_idx {trajectory_idx} >= available {len(all_x_true)}, using 0")
        trajectory_idx = 0

    x_true_sample = all_x_true[trajectory_idx]
    dt_sample = all_dt[trajectory_idx]

    print(f"=== TRAJECTORY INFO ===")
    print(f"Using pre-processed trajectory {trajectory_idx} / {len(all_x_true)-1}")
    print(f"  - Data points: {len(x_true_sample)}")
    print(f"  - Real data timestep (dt): {dt_sample:.4f} seconds ({1/dt_sample:.1f} Hz)")
    print(f"  - Total duration available: {len(x_true_sample) * dt_sample:.1f} seconds")
    print(f"  - Position X range: [{x_true_sample[:, 0].min():.2f}, {x_true_sample[:, 0].max():.2f}] m")
    print(f"  - Position Y range: [{x_true_sample[:, 1].min():.2f}, {x_true_sample[:, 1].max():.2f}] m")

    v_forw = x_true_sample[:, 3]
    v_orth = x_true_sample[:, 4]
    total_v = np.sqrt(v_forw**2 + v_orth**2)
    print(f"  - Velocity range: [{total_v.min():.2f}, {total_v.max():.2f}] m/s")
    print(f"  - Velocity mean: {total_v.mean():.2f} m/s")

    user_traj = create_user_trajectory_from_processed_data(
        x_true_sample,
        avg_dt=dt_sample,
        env_width=CONFIG.env_width,
        env_height=CONFIG.env_height
    )

    print(f"\n=== USER TRAJECTORY ===")
    print(f"  - Duration: {user_traj.total_duration:.1f} seconds")
    print(f"  - Centered position X: [{user_traj.positions_x.min():.2f}, {user_traj.positions_x.max():.2f}] m")
    print(f"  - Centered position Y: [{user_traj.positions_y.min():.2f}, {user_traj.positions_y.max():.2f}] m")

    _auto_register_calibrated_moods()
    
    print(f"\n=== SPAWNER CONFIG ===")
    spawner = SFMPedestrianSpawner(
        user_trajectory=user_traj,
        config=CONFIG,
        n_pedestrians=N_PEDESTRIANS,
        vicinity_radius=VICINITY_RADIUS,
        respawn_distance=RESPAWN_DISTANCE,
        min_spawn_distance=MIN_SPAWN_DISTANCE,
        max_spawn_distance=MAX_SPAWN_DISTANCE,
        speed_scale_factor=SPEED_SCALE_FACTOR
    )
    
    load_calibrated_moods('src/sfm_navigation/data')

    static_obs = generate_safe_static_obstacles(
        user_trajectory=user_traj,
        n_obstacles=N_STATIC_OBSTACLES,
        min_radius=MIN_OBSTACLE_RADIUS,
        max_radius=MAX_OBSTACLE_RADIUS,
        env_width=CONFIG.env_width,
        env_height=CONFIG.env_height,
        safety_margin=OBSTACLE_SAFETY_MARGIN
    )

    print(f"\n=== STATIC OBSTACLES ===")
    print(f"Generated {len(static_obs)} obstacles")

    if SIMULATION_DURATION is None:
        sim_duration = user_traj.total_duration - 0.5
    else:
        sim_duration = min(SIMULATION_DURATION, user_traj.total_duration - 0.5)

    if SFM_DT is None:
        sfm_dt = dt_sample
    else:
        sfm_dt = SFM_DT

    print(f"\n=== SIMULATION SETTINGS ===")
    print(f"  - Simulation duration: {sim_duration:.1f} seconds")
    print(f"  - SFM timestep: {sfm_dt:.4f} seconds ({1/sfm_dt:.1f} Hz)")
    print(f"  - Frame skip: {FRAME_SKIP}")
    print(f"  - Camera mode: {'FOLLOW USER' if FOLLOW_USER else 'STATIC OVERVIEW'}")
    if FOLLOW_USER:
        print(f"  - Follow zoom radius: {FOLLOW_ZOOM_RADIUS} meters")
    print(f"  - Logging: {'ENABLED' if ENABLE_LOGGING else 'DISABLED'}")
    if ENABLE_LOGGING:
        print(f"  - Log window duration: {LOG_WINDOW_DURATION}s")

    print(f"\n=== RUNNING SIMULATION ===")
    anim_fig, sim_logger = create_sfm_spawner_animation(
        user_trajectory=user_traj,
        spawner=spawner,
        simulation_duration=sim_duration,
        dt=sfm_dt,
        static_obstacles=static_obs,
        frame_skip=FRAME_SKIP,
        follow_user=FOLLOW_USER,
        follow_zoom_radius=FOLLOW_ZOOM_RADIUS,
        enable_logging=ENABLE_LOGGING,
        log_window_duration=LOG_WINDOW_DURATION
    )

    stats = spawner.get_statistics()
    print(f"\n=== SPAWNER STATISTICS ===")
    print(f"User avg speed: {stats['user_avg_speed']:.2f} m/s")
    print(f"Pedestrian base speed: {stats['pedestrian_base_speed']:.2f} m/s")
    print(f"Total spawns: {stats['total_spawns']}")
    print(f"Total respawns: {stats['respawn_count']}")
    print(f"Active pedestrians: {stats['active_pedestrians']}")

    if sim_logger is not None:
        sim_logger.print_summary()
        window_logs = sim_logger.get_window_logs()
        print(f"\nWindow logs stored in 'window_logs' variable ({len(window_logs)} windows)")

    # Save the interactive animation to a self-contained HTML file
    html_path = "sfm_animation.html"
    anim_fig.write_html(html_path)
    print(f"\nAnimation saved to {html_path}")

    # Open it automatically in the default web browser (pop‑up window)
    try:
        webbrowser.open(html_path, new=1)   # new=1 opens in a new window
        print("Browser window should open now with the interactive animation.")
    except Exception:
        print("Could not open browser automatically. Please open the file manually.")    
    
    print("Visualization displayed. Close the browser tab to exit.")


if __name__ == '__main__':
    main()