"""Common DWA utility functions (Numba‑optimised)."""

import numpy as np
from numba import njit
from typing import Tuple
from ...sfm.numba_utils import (
    normalize_angle,
    euclidean_distance,
    simulate_trajectory,
    check_trajectory_collision,
    compute_heading_score,
    compute_velocity_score,
    compute_distance_score,
)


@njit
def create_velocity_samples(
    v_current: float,
    omega_current: float,
    v_min: float,
    v_max: float,
    omega_max: float,
    v_accel: float,
    omega_accel: float,
    v_res: float,
    omega_res: float,
    dt: float,
    dwa_window_time: float,
) -> np.ndarray:
    v_low = max(v_min, v_current - v_accel * dwa_window_time)
    v_high = min(v_max, v_current + v_accel * dwa_window_time)
    omega_low = max(-omega_max, omega_current - omega_accel * dwa_window_time)
    omega_high = min(omega_max, omega_current + omega_accel * dwa_window_time)
    if abs(v_current) < 0.1:
        omega_low = -omega_max
        omega_high = omega_max
        v_low = max(v_min, -0.3)
    n_v = max(1, int((v_high - v_low) / v_res) + 1)
    n_omega = max(1, int((omega_high - omega_low) / omega_res) + 1)
    samples = np.zeros((n_v * n_omega, 2))
    idx = 0
    for i in range(n_v):
        v = v_low + i * v_res if n_v > 1 else v_low
        v = min(v, v_high)
        for j in range(n_omega):
            omega = omega_low + j * omega_res if n_omega > 1 else omega_low
            omega = min(omega, omega_high)
            samples[idx, 0] = v
            samples[idx, 1] = omega
            idx += 1
    return samples[:idx]


@njit
def velocity_to_cartesian(v: float, omega: float, theta: float) -> Tuple[float, float]:
    return v * np.cos(theta), v * np.sin(theta)


@njit
def compute_vo_cone(
    robot_x,
    robot_y,
    robot_radius,
    obs_x,
    obs_y,
    obs_radius,
    obs_vx,
    obs_vy,
    time_horizon,
):
    rel_x = obs_x - robot_x
    rel_y = obs_y - robot_y
    dist = np.sqrt(rel_x**2 + rel_y**2)
    if dist < 0.001:
        return (0.0, 0.0, -np.pi, np.pi)
    combined_radius = robot_radius + obs_radius
    if dist <= combined_radius:
        return (obs_vx, obs_vy, -np.pi, np.pi)
    # Apex of the truncated VO
    apex_x = obs_vx + rel_x / time_horizon
    apex_y = obs_vy + rel_y / time_horizon
    angle_to_obs = np.arctan2(rel_y, rel_x)
    half_angle = np.arcsin(min(1.0, combined_radius / dist))
    left_angle = angle_to_obs + half_angle
    right_angle = angle_to_obs - half_angle
    return (apex_x, apex_y, left_angle, right_angle)


@njit
def is_velocity_in_vo(vx, vy, apex_x, apex_y, left_angle, right_angle):
    rel_vx = vx - apex_x
    rel_vy = vy - apex_y
    if rel_vx == 0 and rel_vy == 0:
        return False
    vel_angle = np.arctan2(rel_vy, rel_vx)
    cone_span = normalize_angle(left_angle - right_angle)
    if cone_span >= 0:
        return (
            normalize_angle(vel_angle - right_angle) >= 0
            and normalize_angle(left_angle - vel_angle) >= 0
        )
    else:
        return (
            normalize_angle(vel_angle - right_angle) >= 0
            or normalize_angle(left_angle - vel_angle) >= 0
        )


@njit
def compute_rvo_velocity(
    robot_vx,
    robot_vy,
    obs_vx,
    obs_vy,
    apex_x,
    apex_y,
    left_angle,
    right_angle,
    responsibility=0.5,
):

    # Position offset from the original VO truncation
    offset_x = apex_x - obs_vx
    offset_y = apex_y - obs_vy
    rvo_apex_x = (1 - responsibility) * robot_vx + responsibility * obs_vx + offset_x
    rvo_apex_y = (1 - responsibility) * robot_vy + responsibility * obs_vy + offset_y
    return (rvo_apex_x, rvo_apex_y, left_angle, right_angle)


@njit
def compute_orca_halfplane(
    robot_x,
    robot_y,
    robot_radius,
    robot_vx,
    robot_vy,
    obs_x,
    obs_y,
    obs_radius,
    obs_vx,
    obs_vy,
    time_horizon,
):
    rel_pos_x = obs_x - robot_x
    rel_pos_y = obs_y - robot_y
    rel_vel_x = robot_vx - obs_vx
    rel_vel_y = robot_vy - obs_vy
    dist_sq = rel_pos_x**2 + rel_pos_y**2
    combined_radius = robot_radius + obs_radius
    combined_radius_sq = combined_radius**2
    if dist_sq < combined_radius_sq:
        dist = np.sqrt(dist_sq) if dist_sq > 0 else 0.001
        normal_x = -rel_pos_x / dist
        normal_y = -rel_pos_y / dist
        return (robot_vx, robot_vy, normal_x, normal_y)
    inv_time_horizon = 1.0 / time_horizon
    cutoff_center_x = -rel_pos_x * inv_time_horizon
    cutoff_center_y = -rel_pos_y * inv_time_horizon
    w_x = rel_vel_x - cutoff_center_x
    w_y = rel_vel_y - cutoff_center_y
    w_length_sq = w_x**2 + w_y**2
    dot_product = w_x * (-cutoff_center_x) + w_y * (-cutoff_center_y)
    if (
        dot_product < 0
        and dot_product**2 > combined_radius_sq * inv_time_horizon**2 * w_length_sq
    ):
        w_length = np.sqrt(w_length_sq) if w_length_sq > 0 else 0.001
        unit_w_x = w_x / w_length
        unit_w_y = w_y / w_length
        normal_x = unit_w_x
        normal_y = unit_w_y
        u_x = (combined_radius * inv_time_horizon - w_length) * unit_w_x
        u_y = (combined_radius * inv_time_horizon - w_length) * unit_w_y
    else:
        dist = np.sqrt(dist_sq)
        leg = np.sqrt(dist_sq - combined_radius_sq)
        if rel_pos_x * rel_vel_y - rel_pos_y * rel_vel_x > 0:
            normal_x = (rel_pos_x * leg + rel_pos_y * combined_radius) / dist_sq
            normal_y = (rel_pos_y * leg - rel_pos_x * combined_radius) / dist_sq
        else:
            normal_x = (rel_pos_x * leg - rel_pos_y * combined_radius) / dist_sq
            normal_y = (rel_pos_y * leg + rel_pos_x * combined_radius) / dist_sq
            normal_x = -normal_x
            normal_y = -normal_y
        dot_rel_vel_normal = rel_vel_x * normal_x + rel_vel_y * normal_y
        u_x = dot_rel_vel_normal * normal_x - rel_vel_x
        u_y = dot_rel_vel_normal * normal_y - rel_vel_y
    point_x = robot_vx + 0.5 * u_x
    point_y = robot_vy + 0.5 * u_y
    return (point_x, point_y, normal_x, normal_y)


@njit
def velocity_satisfies_orca(vx, vy, point_x, point_y, normal_x, normal_y):
    if normal_x == 0 and normal_y == 0:
        return True
    return (vx - point_x) * normal_x + (vy - point_y) * normal_y >= -0.001
