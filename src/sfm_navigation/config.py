from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class SimulationConfig:
    """Global configuration for the simulation."""
    # ==================== Environment Settings ====================
    env_width: float = 100.0
    env_height: float = 100.0
    dt: float = 0.1
    max_simulation_time: float = 200.0

    # ==================== Robot Specifications ====================
    robot_radius: float = 0.18
    max_linear_vel: float = 2.5
    min_linear_vel: float = 0.0
    max_angular_vel: float = 8.0
    max_linear_accel: float = 3.0
    max_angular_accel: float = 8.0
    max_linear_jerk: float = 5.0      # m/s³
    max_angular_jerk: float = 5.0     # rad/s³
    wheel_base: float = 0.16

    # ==================== DWA Algorithm Parameters ====================
    dwa_window_time: float = 1.0         # seconds (how far acceleration can change)
    v_resolution: float = 0.1
    w_resolution: float = 0.1
    predict_time: float = 1.0             # seconds (how far to predict)
    alpha_heading: float = 0.8
    beta_distance: float = 0.3
    gamma_velocity: float = 0.4
    safety_margin: float = 1.0
    goal_tolerance: float = 0.5
    park_margin: float = 0.2   # metres inside the safety radius where robot parks

    # ==================== Pedestrian Parameters ====================
    pedestrian_radius: float = 0.3
    pedestrian_avg_speed: float = 1.0
    social_zone_scale: float = 1.2   # multiplier for comfort zone dimensions

    # ==================== Dynamic Obstacle Parameters ====================
    obstacle_radius: float = 0.25
    obstacle_max_speed: float = 0.6

    # ==================== Velocity Obstacle Parameters ====================
    vo_time_horizon: float = 3.0
    rvo_responsibility: float = 0.75

    # ==================== ORCA Parameters ====================
    orca_time_horizon: float = 3.0
    orca_time_horizon_obst: float = 2.0

    # ==================== Costmap Parameters ====================
    costmap_resolution: float = 0.1
    inflation_radius: float = 0.5

    # ==================== Visualization Parameters ====================
    trajectory_history_length: int = 100
    animation_fps: int = 10

    # ==================== Data Paths ====================
    atc_csv_path: str = "data/ATC_data/atc-20121114.csv"
    atc_csv_folder: str = "data/ATC_data/"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.max_linear_vel > self.pedestrian_avg_speed, \
            "Robot max velocity should exceed pedestrian speed for overtaking capability"
        assert self.dt > 0, "Time step must be positive"
        assert self.env_width > 0 and self.env_height > 0, "Environment dimensions must be positive"

def load_config_from_json(path: str) -> SimulationConfig:
    """Load configuration from a JSON file and update the global CONFIG."""
    with open(path, 'r') as f:
        data = json.load(f)
    for section in data:
        if isinstance(data[section], dict):
            for key, value in data[section].items():
                if hasattr(CONFIG, key):
                    setattr(CONFIG, key, value)
    return CONFIG

# Global configuration instance (can be updated after loading)
CONFIG = SimulationConfig()