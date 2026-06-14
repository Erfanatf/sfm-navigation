"""DWA‑RVO controller with unified maneuver manager."""
import time
import numpy as np
from numba import njit
from typing import Tuple, Optional
from collections import deque
from ...config import SimulationConfig
from ..base_controller import BaseController
from ...sfm.numba_utils import (
    normalize_angle, simulate_trajectory, check_trajectory_collision,
    compute_heading_score, compute_velocity_score, compute_distance_score,
)
from .dwa_utils import compute_vo_cone, is_velocity_in_vo, compute_rvo_velocity
from ..maneuvers import ManeuverDOB, ManeuverManager
from ..mpc.base_mpc import ControlLPF


@njit
def _compute_rvo_penalty(robot_vx, robot_vy, robot_x, robot_y, robot_radius, obstacles, time_horizon, responsibility):
    penalty = 0.0
    for i in range(obstacles.shape[0]):
        obs_x, obs_y, obs_r, obs_vx, obs_vy = obstacles[i, 0], obstacles[i, 1], obstacles[i, 2], obstacles[i, 3], obstacles[i, 4]
        if abs(obs_vx) < 0.01 and abs(obs_vy) < 0.01:
            continue
        apex_x, apex_y, left_angle, right_angle = compute_vo_cone(
            robot_x, robot_y, robot_radius, obs_x, obs_y, obs_r, obs_vx, obs_vy, time_horizon
        )
        rvo_apex_x, rvo_apex_y, _, _ = compute_rvo_velocity(
            robot_vx, robot_vy, obs_vx, obs_vy, apex_x, apex_y, left_angle, right_angle, responsibility
        )
        if is_velocity_in_vo(robot_vx, robot_vy, rvo_apex_x, rvo_apex_y, left_angle, right_angle):
            rel_vx = robot_vx - rvo_apex_x
            rel_vy = robot_vy - rvo_apex_y
            vel_mag = np.sqrt(rel_vx**2 + rel_vy**2)
            penalty += vel_mag + 1.0
    return penalty


@njit
def _dwa_rvo_core(
    x, y, theta, v_curr, omega_curr, goal_x, goal_y, obstacles,
    v_max, omega_max, v_accel, omega_accel, v_res, omega_res, dt,
    n_predict_steps, dwa_window_time, robot_radius, safety_margin,
    time_horizon, responsibility, alpha, beta, gamma, lambda_rvo,
):
    v_low = max(0.0, v_curr - v_accel * dwa_window_time)
    v_high = min(v_max, v_curr + v_accel * dwa_window_time)
    omega_low = max(-omega_max, omega_curr - omega_accel * dwa_window_time)
    omega_high = min(omega_max, omega_curr + omega_accel * dwa_window_time)

    n_v = max(1, int((v_high - v_low) / v_res) + 1)
    n_omega = max(1, int((omega_high - omega_low) / omega_res) + 1)

    static_obs = obstacles[:, :3].copy()
    best_v = 0.0
    best_omega = 0.0
    best_score = -1e10
    safe_count = 0

    for i in range(n_v):
        v = v_low + i * v_res if n_v > 1 else v_low
        v = min(v, v_high)
        for j in range(n_omega):
            omega = omega_low + j * omega_res if n_omega > 1 else omega_low
            omega = min(omega, omega_high)

            traj = simulate_trajectory(x, y, theta, v, omega, dt, n_predict_steps)
            safe, min_dist = check_trajectory_collision(traj, static_obs, robot_radius, safety_margin)
            if not safe:
                continue
            safe_count += 1

            heading_score = compute_heading_score(traj, goal_x, goal_y)
            vel_score = compute_velocity_score(v, v_max)
            dist_score = compute_distance_score(min_dist, safety_margin * 2)

            start_dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
            end_dist = np.sqrt((traj[-1, 0] - goal_x) ** 2 + (traj[-1, 1] - goal_y) ** 2)
            progress = (start_dist - end_dist) / (start_dist + 0.01)
            progress_score = max(0.0, min(1.0, progress + 0.5))

            rvx = v * np.cos(theta)
            rvy = v * np.sin(theta)
            rvo_penalty = _compute_rvo_penalty(rvx, rvy, x, y, robot_radius, obstacles, time_horizon, responsibility)

            score = (
                alpha * heading_score
                + beta * dist_score
                + gamma * vel_score
                + 0.3 * progress_score
                - lambda_rvo * rvo_penalty
            )

            if score > best_score:
                best_score = score
                best_v = v
                best_omega = omega

    if safe_count == 0:
        best_v = 0.0
        best_omega = 0.0

    return best_v, best_omega, best_score, safe_count


class DWA_RVO(BaseController):
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.alpha = config.alpha_heading
        self.beta = config.beta_distance
        self.gamma = config.gamma_velocity
        self.lambda_rvo = 0.25
        self.time_horizon = config.vo_time_horizon
        self.responsibility = config.rvo_responsibility
        self.dwa_window_time = config.dwa_window_time
        self.n_predict_steps = int(config.predict_time / config.dt)
        self.last_compute_time = 0.0

        self.overtaking_active = False
        self.parking_active = False
        self.repulsion_active = False
        self.rotation_active = False
        self.soft_recovery_active = False

        self.LPF = ControlLPF(alpha=0.4)
        self.maneuver_LPF = ControlLPF(alpha=0.4)
        self.DOB_LPF = ControlLPF(alpha=0.4)
        self.dob = ManeuverDOB(dt=0.05, L=5.0, tau=0.2)

        self._v_base = 0.0; self._omega_base = 0.0
        self._v_man = 0.0; self._omega_man = 0.0
        self._v_final = 0.0; self._omega_final = 0.0

        self.maneuvers = ManeuverManager(config)
        self.maneuver_priority = ["rotation", "overtaking", "repulsion"]
        self._robot_pos_history = deque(maxlen=100)
        self._goal_dist_history = deque(maxlen=100)
        self._was_maneuver_active = False

    def _convert_obstacles(self, obstacles):
        if isinstance(obstacles, list):
            if len(obstacles) == 0:
                return np.zeros((0, 3)), np.zeros((0, 4))
            obs_arr = np.array(obstacles, dtype=float)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            circles = obs_arr[:, :3].copy()
            walls = np.zeros((0, 4))
            return circles, walls
        else:
            obs_arr = np.asarray(obstacles, dtype=float)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            return obs_arr[:, :3], np.zeros((0, 4))

    def _apply_maneuver(self, u_man_arr, u_cmd_flt, robot_vel, dt, t_start):
        if not self._was_maneuver_active:
            self.maneuver_LPF.u_prev = robot_vel.copy()
        u_man_arr_flt = self.maneuver_LPF.filter(u_man_arr)
        u_final = self.dob.step(u_cmd_flt, u_man_arr_flt, True, robot_vel, dt)
        u_final_flt = self.DOB_LPF.filter(u_final)
        u_final_flt[0] = np.clip(u_final_flt[0], self.config.min_linear_vel, self.config.max_linear_vel)
        u_final_flt[1] = np.clip(u_final_flt[1], -self.config.max_angular_vel, self.config.max_angular_vel)
        self._v_base, self._omega_base = u_cmd_flt
        self._v_man, self._omega_man = u_man_arr_flt
        self._v_final, self._omega_final = u_final_flt
        self.last_compute_time = time.perf_counter() - t_start
        return u_final_flt[0], u_final_flt[1]

    def compute_velocity(
        self, robot_state, goal_pos, obstacles, user=None, dt=None, sim_time=None, **kwargs
    ) -> Tuple[float, float]:
        if dt is None:
            dt = self.config.dt
        t_start = time.perf_counter()

        self.overtaking_active = False; self.parking_active = False
        self.repulsion_active = False; self.rotation_active = False
        self.soft_recovery_active = False

        circles, walls = self._convert_obstacles(obstacles)
        x, y, theta = robot_state.x, robot_state.y, robot_state.theta
        v_curr, w_curr = robot_state.v, robot_state.omega
        robot_vel = np.array([v_curr, w_curr])

        dx_goal = goal_pos[0] - x; dy_goal = goal_pos[1] - y
        heading_to_goal = np.arctan2(dy_goal, dx_goal)
        heading_err = abs(normalize_angle(heading_to_goal - theta))
        dist_goal = np.hypot(dx_goal, dy_goal)
        self._robot_pos_history.append((x, y))

        state = dict(
            x=x, y=y, theta=theta, v_curr=v_curr,
            goal_pos=goal_pos, user=user, dt=dt, sim_time=sim_time,
            heading_to_goal=heading_to_goal, dist_goal=dist_goal,
            heading_err=heading_err, robot_pos=(x, y),
            robot_pos_history=self._robot_pos_history,
            goal_dist_history=self._goal_dist_history,
        )

        for name in self.maneuver_priority:
            if self.maneuvers.check(name, state):
                v_cmd, omega_cmd, active = self.maneuvers.command(name, state)
                if name == "rotation": self.rotation_active = active
                elif name == "overtaking": self.overtaking_active = active
                elif name == "repulsion": self.repulsion_active = active
                if active:
                    self._robot_pos_history.clear()
                    self._goal_dist_history.clear()
                    self._was_maneuver_active = True
                    return self._apply_maneuver(
                        np.array([v_cmd, omega_cmd]),
                        np.zeros(2), robot_vel, dt, t_start,
                    )

        if isinstance(obstacles, list):
            if len(obstacles) == 0:
                obstacles = np.zeros((0, 5))
            else:
                obs_arr = np.array(obstacles, dtype=float)
                if obs_arr.ndim == 1:
                    obs_arr = obs_arr.reshape(1, -1)
                if obs_arr.shape[1] == 3:
                    obstacles = np.column_stack([obs_arr, np.zeros((obs_arr.shape[0], 2))])
                elif obs_arr.shape[1] == 5:
                    obstacles = obs_arr
                else:
                    raise ValueError

        n_predict_steps = int(self.config.predict_time / dt)
        best_v, best_omega, _, safe_count = _dwa_rvo_core(
            x, y, theta, v_curr, w_curr, goal_pos[0], goal_pos[1],
            obstacles,
            self.config.max_linear_vel, self.config.max_angular_vel,
            self.config.max_linear_accel, self.config.max_angular_accel,
            self.config.v_resolution, self.config.w_resolution,
            dt, n_predict_steps, self.dwa_window_time,
            self.config.robot_radius, self.config.safety_margin,
            self.time_horizon, self.responsibility,
            self.alpha, self.beta, self.gamma, self.lambda_rvo,
        )

        if best_v == 0.0 and best_omega != 0.0:
            best_omega = 0.0

        u_cmd = np.array([best_v, best_omega])
        u_cmd_flt = self.LPF.filter(u_cmd)

        if safe_count == 0:
            if user is not None and user.get("active", True):
                ux, uy, urad = user["x"], user["y"], user["radius"]
                dist_user = np.hypot(x - ux, y - uy)
                if dist_user < 4.0 * urad:
                    v_park, omega_park, park_active = self.maneuvers.parking_command(
                        sim_time, x, y, theta, v_curr, user, goal_pos, (x, y)
                    )
                    if park_active:
                        self.parking_active = True
                        return self._apply_maneuver(
                            np.array([v_park, omega_park]),
                            u_cmd_flt, robot_vel, dt, t_start,
                        )
            self.maneuver_LPF.reset()
            u_final = self.dob.step(u_cmd_flt, np.zeros(2), False, robot_vel, dt)
            u_final_flt = self.DOB_LPF.filter(u_final)
            u_final_flt[0] = np.clip(u_final_flt[0], self.config.min_linear_vel, self.config.max_linear_vel)
            u_final_flt[1] = np.clip(u_final_flt[1], -self.config.max_angular_vel, self.config.max_angular_vel)
            self._v_base, self._omega_base = u_cmd_flt
            self._v_man, self._omega_man = 0.0, 0.0
            self._v_final, self._omega_final = u_final_flt
            self.last_compute_time = time.perf_counter() - t_start
            return u_final_flt[0], u_final_flt[1]

        if self._was_maneuver_active:
            self.LPF.reset()
            self._was_maneuver_active = False

        self.maneuver_LPF.reset()
        u_final = self.dob.step(u_cmd_flt, np.zeros(2), False, robot_vel, dt)
        u_final_flt = self.DOB_LPF.filter(u_final)
        u_final_flt[0] = np.clip(u_final_flt[0], self.config.min_linear_vel, self.config.max_linear_vel)
        u_final_flt[1] = np.clip(u_final_flt[1], -self.config.max_angular_vel, self.config.max_angular_vel)
        self._v_base, self._omega_base = u_cmd_flt
        self._v_man, self._omega_man = 0.0, 0.0
        self._v_final, self._omega_final = u_final_flt
        self.last_compute_time = time.perf_counter() - t_start
        return u_final_flt[0], u_final_flt[1]