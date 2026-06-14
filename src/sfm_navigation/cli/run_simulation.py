import argparse
from ..config import load_config_from_json, CONFIG
from ..data.loader import load_multiple_trajectories
from ..agents.user import create_user_trajectory_from_processed_data
from ..spawner.spawner import SFMPedestrianSpawner, generate_safe_static_obstacles
from ..visualization.animation import create_sfm_spawner_animation

def main():
    parser = argparse.ArgumentParser(description="Run SFM spawner simulation")
    parser.add_argument('--config', type=str, default='config/default.json', help='Path to config JSON')
    parser.add_argument('--data', type=str, required=True, help='Path to ATC data folder')
    args = parser.parse_args()

    # Load configuration
    load_config_from_json(args.config)

    # Load data
    all_x_true, all_dt, AVG_DT = load_multiple_trajectories(args.data, n=24)
    # For now use first trajectory
    x_true = all_x_true[0]
    dt = all_dt[0]
    user_traj = create_user_trajectory_from_processed_data(x_true, dt)

    # Spawner and obstacles (using defaults; later read from config)
    spawner = SFMPedestrianSpawner(user_traj, CONFIG)
    static_obs = generate_safe_static_obstacles(user_traj)

    # Animation
    fig, _ = create_sfm_spawner_animation(user_traj, spawner, static_obstacles=static_obs)
    fig.show()

if __name__ == '__main__':
    main()