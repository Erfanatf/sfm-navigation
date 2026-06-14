import numpy as np
from numba import njit
from typing import Tuple

@njit
def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range."""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


@njit
def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute euclidean distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


@njit
def point_to_line_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Compute distance from a point (px, py) to line segment (x1, y1)-(x2, y2)"""
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return euclidean_distance(px, py, x1, y1)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return euclidean_distance(px, py, proj_x, proj_y)


@njit
def simulate_trajectory(
    x: float, y: float, theta: float, v: float, omega: float, dt: float, n_step: int
) -> np.ndarray:
    trajectory = np.zeros((n_step + 1, 3))
    trajectory[0, 0] = x
    trajectory[0, 1] = y
    trajectory[0, 2] = theta

    curr_x, curr_y, curr_theta = x, y, theta

    for i in range(n_step):
        if abs(omega) < 1e-6:
            curr_x += v * np.cos(curr_theta) * dt
            curr_y += v * np.sin(curr_theta) * dt
        else:
            curr_x += (v / omega) * (
                np.sin(curr_theta + omega * dt) - np.sin(curr_theta)
            )
            curr_y += (v / omega) * (
                np.cos(curr_theta) - np.cos(curr_theta + omega * dt)
            )
            curr_theta = normalize_angle(curr_theta + omega * dt)

        trajectory[i + 1, 0] = curr_x
        trajectory[i + 1, 1] = curr_y
        trajectory[i + 1, 2] = curr_theta

    return trajectory


@njit
def check_trajectory_collision(
    trajectory: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float,
    safety_margin: float,
) -> Tuple[bool, float]:
    min_dist = np.inf
    total_clearance = robot_radius + safety_margin

    for i in range(trajectory.shape[0]):
        traj_x = trajectory[i, 0]
        traj_y = trajectory[i, 1]

        for j in range(obstacles.shape[0]):
            obs_x = obstacles[j, 0]
            obs_y = obstacles[j, 1]
            obs_r = obstacles[j, 2]

            dist = (
                euclidean_distance(traj_x, traj_y, obs_x, obs_y)
                - obs_r
                - total_clearance
            )
            min_dist = min(min_dist, dist)

            if dist < 0:
                return False, min_dist

    return True, min_dist


@njit
def compute_heading_score(
    trajectory: np.ndarray, goal_x: float, goal_y: float
) -> float:
    final_x = trajectory[-1, 0]
    final_y = trajectory[-1, 1]
    final_theta = trajectory[-1, 2]

    goal_angle = np.arctan2(goal_y - final_y, goal_x - final_x)
    angle_diff = abs(normalize_angle(goal_angle - final_theta))
    return 1.0 - angle_diff / np.pi


@njit
def compute_velocity_score(v: float, max_v: float) -> float:
    return v / max_v if max_v > 0 else 0.0


@njit
def compute_distance_score(min_dist: float, max_range: float) -> float:
    if min_dist < 0:
        return 0.0
    return min(min_dist / max_range, 1.0)


@njit
def create_velocity_samples(
    v_current: float,
    omega_current: float,
    v_min: float,
    v_max: float,
    omega_min: float,
    omega_max: float,
    v_accel: float,
    omega_accel: float,
    v_res: float,
    omega_res: float,
    dt: float,
) -> np.ndarray:
    v_low = max(v_min, v_current - v_accel * dt)
    v_high = min(v_max, v_current + v_accel * dt)
    omega_low = max(omega_min, omega_current - omega_accel * dt)
    omega_high = min(omega_max, omega_current + omega_accel * dt)

    n_v = max(1, int((v_high - v_low) / v_res) + 1)
    n_omega = max(1, int((omega_high - omega_low) / omega_res) + 1)

    samples = np.zeros((n_v * n_omega, 2))
    idx = 0

    for i in range(n_v):
        v = v_low + i * (v_high - v_low) / max(1, n_v - 1) if n_v > 1 else v_low
        for j in range(n_omega):
            omega = omega_low + j * (omega_high - omega_low) / max(1, n_omega - 1) if n_omega > 1 else omega_low
            samples[idx, 0] = v
            samples[idx, 1] = omega
            idx += 1

    return samples[:idx]