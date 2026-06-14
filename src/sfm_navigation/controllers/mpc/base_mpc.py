"""Base class for all MPC controllers with shared pre/post processing."""

import time
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from collections import deque
from ...config import SimulationConfig
from ..base_controller import BaseController
from ...sfm.numba_utils import normalize_angle
from ..maneuvers import (
    compute_circulation_acceleration,
    compute_park_command,
    ManeuverDOB,
    ManeuverManager,
)

class ControlLPF:
    """First‑order low‑pass filter for smoothing control commands."""

    def __init__(self, alpha: float = 0.3, num_controls: int = 2):
        self.alpha = alpha
        self.u_prev = np.zeros(num_controls)
        self.enabled = True

    def filter(self, u_raw: np.ndarray) -> np.ndarray:
        if not self.enabled:
            self.u_prev = np.array(u_raw)
            return u_raw
        u_filt = self.alpha * np.array(u_raw) + (1.0 - self.alpha) * self.u_prev
        self.u_prev = u_filt.copy()
        return u_filt

    def reset(self):
        self.u_prev = np.zeros_like(self.u_prev)


def cosine_path_blend(
    current_pos: np.ndarray,
    current_theta: float,
    target_path: np.ndarray,
    blend_steps: int,
) -> np.ndarray:
    """Blend position and heading from current state to target path using cosine."""
    if len(target_path) < 2 or blend_steps < 1:
        return target_path
    n = min(blend_steps, len(target_path))
    blended = target_path.copy()
    for i in range(n):
        alpha = 0.5 * (1 - np.cos(np.pi * i / n))
        blended[i, 0] = (1 - alpha) * current_pos[0] + alpha * target_path[i, 0]
        blended[i, 1] = (1 - alpha) * current_pos[1] + alpha * target_path[i, 1]
        d_angle = target_path[i, 2] - current_theta
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        blended[i, 2] = current_theta + alpha * d_angle
        while blended[i, 2] > np.pi:
            blended[i, 2] -= 2 * np.pi
        while blended[i, 2] < -np.pi:
            blended[i, 2] += 2 * np.pi
    return blended


class BaseMPCController(BaseController):
    """Abstract MPC controller – maneuvers handled by ManeuverManager."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.LPF = ControlLPF(alpha=0.45)
        self.maneuver_LPF = ControlLPF(alpha=0.45)
        self.DOB_LPF = ControlLPF(alpha=0.45)
        self.dob = ManeuverDOB(dt=0.05, L=5.0, tau=0.2)

        # Logging storage
        self._v_flt = 0.0
        self._omega_flt = 0.0
        self._v_base = 0.0
        self._omega_base = 0.0
        self._v_man = 0.0
        self._omega_man = 0.0
        self._v_final = 0.0
        self._omega_final = 0.0

        self.blend_steps = 20
        self.last_compute_time = 0.0

        # Maneuver active flags (read by animation)
        self.overtaking_active = False
        self.parking_active = False
        self.repulsion_active = False
        self.rotation_active = False
        self.soft_recovery_active = False

        # Target path cache
        self._target_path = np.zeros((0, 2))

        # Distance history for stuck detection (used by maneuver manager)
        self._goal_dist_history = deque(maxlen=100)
        self._robot_pos_history = deque(maxlen=100)  # stores (x, y) tuples

        # ---- Instantiate the maneuver manager ----
        self.maneuvers = ManeuverManager(config)

        self._was_maneuver_active = False
        self.maneuver_priority = [
            "soft_recovery",
            "rotation",
            "repulsion",
            "overtaking",
        ]

    def _convert_obstacles(self, obstacles):
        if isinstance(obstacles, list):
            if len(obstacles) == 0:
                return np.zeros((0, 4)), np.zeros((0, 4))
            obs_arr = np.array(obstacles, dtype=float)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            circles = obs_arr[:, :4].copy()
            walls = np.zeros((0, 4))
            return circles, walls
        else:
            obs_arr = np.asarray(obstacles, dtype=float)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            return obs_arr[:, :4], np.zeros((0, 4))

    def _get_target_path(
        self, robot_pos, robot_theta, goal_pos, n_points=20, goal_heading=None
    ):
        """Build a cubic Bézier curve from current robot pose to the goal."""
        P0 = robot_pos
        P3 = np.array(goal_pos)
        dist = np.linalg.norm(P3 - P0)
        if dist < 0.01:
            return np.tile(
                np.array([goal_pos[0], goal_pos[1], robot_theta]), (n_points, 1)
            )

        if goal_heading is None:
            goal_heading = np.arctan2(P3[1] - P0[1], P3[0] - P0[0])
        P1 = P0 + 0.3 * dist * np.array([np.cos(robot_theta), np.sin(robot_theta)])
        P2 = P3 - 0.2 * dist * np.array([np.cos(goal_heading), np.sin(goal_heading)])

        path = np.zeros((n_points, 3))
        for i in range(n_points):
            t = (i + 1) / n_points
            pt = (
                (1 - t) ** 3 * P0
                + 3 * (1 - t) ** 2 * t * P1
                + 3 * (1 - t) * t**2 * P2
                + t**3 * P3
            )
            dpt = (
                -3 * (1 - t) ** 2 * P0
                + 3 * (1 - t) * (1 - 3 * t) * P1
                + 3 * t * (2 - 3 * t) * P2
                + 3 * t**2 * P3
            )
            heading = np.arctan2(dpt[1], dpt[0])
            path[i] = [pt[0], pt[1], heading]
        return path

    @abstractmethod
    def _solve_mpc(
        self, robot_state, target_path, circles, walls, pedestrian_params=None
    ) -> Tuple[float, float]:
        """Core MPC solver – child class implements this."""
        pass

    def _apply_maneuver(self, u_man_arr, u_cmd_flt, robot_vel, dt, t_start):
        if not self._was_maneuver_active:
            self.maneuver_LPF.u_prev = robot_vel.copy()
        u_man_arr_flt = self.maneuver_LPF.filter(u_man_arr)
        u_final = self.dob.step(u_cmd_flt, u_man_arr_flt, True, robot_vel, dt)
        u_final_flt = self.DOB_LPF.filter(u_final)
        u_final_flt[0] = np.clip(
            u_final_flt[0], self.config.min_linear_vel, self.config.max_linear_vel
        )
        u_final_flt[1] = np.clip(
            u_final_flt[1], -self.config.max_angular_vel, self.config.max_angular_vel
        )
        self._v_final, self._omega_final = u_final_flt
        self._v_base, self._omega_base = u_cmd_flt
        self._v_man, self._omega_man = u_man_arr_flt
        self._v_final, self._omega_final = u_final_flt
        self.last_compute_time = time.perf_counter() - t_start
        return u_final_flt[0], u_final_flt[1]

    # ----------------------------------------------------------------
    #  Main control pipeline
    # ----------------------------------------------------------------
    def compute_velocity(
        self,
        robot_state,
        goal_pos,
        obstacles,
        user=None,
        dt=None,
        sim_time=None,
        **kwargs
    ):
        pedestrian_params = kwargs.get("pedestrian_params", None)
        if dt is None:
            dt = self.config.dt
        t_start = time.perf_counter()

        # Reset flags
        self.overtaking_active = False
        self.parking_active = False
        self.repulsion_active = False
        self.rotation_active = False
        self.soft_recovery_active = False

        circles, walls = self._convert_obstacles(obstacles)
        x, y, theta = robot_state.x, robot_state.y, robot_state.theta
        v_curr, w_curr = robot_state.v, robot_state.omega
        robot_vel = np.array([v_curr, w_curr])

        # ---- common state ----
        dx_goal = goal_pos[0] - x
        dy_goal = goal_pos[1] - y
        heading_to_goal = np.arctan2(dy_goal, dx_goal)
        heading_err = abs(normalize_angle(heading_to_goal - theta))
        dist_goal = np.hypot(dx_goal, dy_goal)
        self._robot_pos_history.append((x, y))

        # ---- Build state dict for the manoeuvre manager ----
        state = dict(
            x=x,
            y=y,
            theta=theta,
            v_curr=v_curr,
            goal_pos=goal_pos,
            user=user,
            dt=dt,
            sim_time=sim_time,  # pass sim_time for logging
            heading_to_goal=heading_to_goal,
            dist_goal=dist_goal,
            heading_err=heading_err,
            robot_pos=(x, y),
            robot_pos_history=self._robot_pos_history,
            goal_dist_history=self._goal_dist_history,
        )

        # ---- Iterate over priority list ----
        for name in self.maneuvers.maneuver_priority:
            if self.maneuvers.check(name, state):
                v_cmd, omega_cmd, active = self.maneuvers.command(name, state)
                # Update the corresponding active flag
                if name == "soft_recovery":
                    self.soft_recovery_active = active
                elif name == "rotation":
                    self.rotation_active = active
                elif name == "overtaking":
                    self.overtaking_active = active
                elif name == "repulsion":
                    self.repulsion_active = active

                if active:
                    # Clear histories to prevent re‑trigger
                    self._robot_pos_history.clear()
                    self._goal_dist_history.clear()
                    # Reset warm‑start if rotation
                    if name == "rotation" and hasattr(self, "_warm_start_done"):
                        self._warm_start_done = False
                        self.U = np.zeros_like(self.U)

                    self._was_maneuver_active = True
                    return self._apply_maneuver(
                        np.array([v_cmd, omega_cmd]),
                        np.zeros(2),
                        robot_vel,
                        dt,
                        t_start,
                    )

        # ---- Normal operation: build target path and run child MPC ----
        robot_pos = np.array([x, y])
        user_heading = user.get("user_heading", None) if user is not None else None
        target_path = self._get_target_path(
            robot_pos, theta, goal_pos, goal_heading=user_heading
        )
        if self.blend_steps > 0 and len(target_path) > 1:
            target_path = cosine_path_blend(
                robot_pos, theta, target_path, self.blend_steps
            )
        self._target_path = target_path

        try:
            v_raw, omega_raw = self._solve_mpc(
                np.array([x, y, theta]), target_path, circles, walls, pedestrian_params
            )
        except Exception:
            # Parking fallback only if close to user
            if user is not None and np.hypot(x - user["x"], y - user["y"]) < (
                user["radius"] + 0.15
            ):
                v_man, omega_man, active = self.maneuvers.parking_command(
                    sim_time, x, y, theta, v_curr, user, goal_pos, (x, y)
                )
                if active:
                    self.parking_active = True
                    return self._apply_maneuver(
                        np.array([v_man, omega_man]),
                        np.zeros(2),
                        robot_vel,
                        dt,
                        t_start,
                    )
            # Safe stop otherwise
            v_raw, omega_raw = 0.0, 0.0

        # Reset base LPF if returning from a maneuver
        if self._was_maneuver_active:
            self.LPF.reset()
            self._was_maneuver_active = False

        u_cmd = np.array([v_raw, omega_raw])
        u_cmd_flt = self.LPF.filter(u_cmd)

        self.maneuver_LPF.reset()
        u_final = self.dob.step(u_cmd_flt, np.zeros(2), False, robot_vel, dt)
        u_final_flt = self.DOB_LPF.filter(u_final)
        u_final_flt[0] = np.clip(
            u_final_flt[0], self.config.min_linear_vel, self.config.max_linear_vel
        )
        u_final_flt[1] = np.clip(
            u_final_flt[1], -self.config.max_angular_vel, self.config.max_angular_vel
        )
        self._v_final, self._omega_final = u_final_flt
        self._v_base, self._omega_base = u_cmd_flt
        self._v_man, self._omega_man = 0.0, 0.0
        self._v_final, self._omega_final = u_final_flt
        self.last_compute_time = time.perf_counter() - t_start

        return u_final_flt[0], u_final_flt[1]
