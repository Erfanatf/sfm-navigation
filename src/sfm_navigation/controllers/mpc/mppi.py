"""MPPI controller – Model Predictive Path Integral with DOB, LPFs, and unified maneuvers."""

import time
import numpy as np
from numba import njit
from typing import Tuple, Optional
from ...config import SimulationConfig
from .base_mpc import BaseMPCController, cosine_path_blend
from ...sfm.numba_utils import point_to_line_distance, normalize_angle
from ..maneuvers import (
    compute_circulation_acceleration,
    compute_park_command,
    ManeuverDOB,
)
from ..mpc.base_mpc import ControlLPF
from .mppi_noise import NoiseGenerator


# ----------------------------------------------------------------------
#  JIT‑compiled helpers (unchanged)
# ----------------------------------------------------------------------
@njit(cache=False)
def mppi_rollout_batch(x0, y0, theta0, U_base, noise, dt, v_max, omega_max):
    K, H, _ = noise.shape
    traj = np.zeros((K, H + 1, 3))
    for k in range(K):
        x, y, th = x0, y0, theta0
        traj[k, 0, 0] = x
        traj[k, 0, 1] = y
        traj[k, 0, 2] = th
        for t in range(H):
            v = U_base[t, 0] + noise[k, t, 0]
            omega = U_base[t, 1] + noise[k, t, 1]
            if v > v_max:
                v = v_max
            if v < -v_max:
                v = -v_max
            if omega > omega_max:
                omega = omega_max
            if omega < -omega_max:
                omega = -omega_max
            th_mid = th + omega * dt / 2.0
            x += v * np.cos(th_mid) * dt
            y += v * np.sin(th_mid) * dt
            th += omega * dt
            while th > np.pi:
                th -= 2 * np.pi
            while th < -np.pi:
                th += 2 * np.pi
            traj[k, t + 1, 0] = x
            traj[k, t + 1, 1] = y
            traj[k, t + 1, 2] = th
    return traj


@njit(cache=False)
def mppi_compute_costs(
    trajectories,
    target_path,
    circles,
    walls,
    robot_radius,
    safety_margin,
    Q_path,
    Q_progress,
    Q_terminal,
    Q_speed,
    Q_heading,
    Q_social,  # new weight
    pedestrian_params,  # (M,7) array
    social_scale,
    group_threshold=1.5,  # r_group for grouping
    group_cost_weight=2.0,
):
    K = trajectories.shape[0]
    H = trajectories.shape[1] - 1
    costs = np.zeros(K)
    goal_pos = target_path[-1, :2]  # goal position (x,y)
    # Note: goal_heading is no longer used for intermediate steps

    for k in range(K):
        c = 0.0
        for t in range(1, H + 1):
            x, y, th = (
                trajectories[k, t, 0],
                trajectories[k, t, 1],
                trajectories[k, t, 2],
            )

            # Find closest point on target path (position and heading)
            min_dsq = 1e10
            closest_idx = 0
            for i in range(target_path.shape[0]):
                dx = x - target_path[i, 0]
                dy = y - target_path[i, 1]
                dsq = dx * dx + dy * dy
                if dsq < min_dsq:
                    min_dsq = dsq
                    closest_idx = i

            # Path proximity cost (distance to the path)
            c += Q_path * min_dsq

            # Progress cost (distance to final goal)
            dxg = x - goal_pos[0]
            dyg = y - goal_pos[1]
            c += Q_progress * (dxg * dxg + dyg * dyg)

            # Heading cost: align with the *path's* heading at the closest point
            if Q_heading > 0.0:
                path_heading = target_path[closest_idx, 2]
                d_th = th - path_heading
                # Normalise to [-pi, pi)
                while d_th > np.pi:
                    d_th -= 2 * np.pi
                while d_th < -np.pi:
                    d_th += 2 * np.pi
                c += Q_heading * (d_th * d_th)

            # Obstacle penalties (unchanged)
            for j in range(circles.shape[0]):
                cx, cy, r = circles[j, 0], circles[j, 1], circles[j, 2]
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                clearance = dist - r - robot_radius - safety_margin
                if clearance < 0:
                    c += 1e4 * (-clearance)

            for j in range(walls.shape[0]):
                x1, y1 = walls[j, 0], walls[j, 1]
                x2, y2 = walls[j, 2], walls[j, 3]
                d = point_to_line_distance(x, y, x1, y1, x2, y2)
                clearance = d - robot_radius - safety_margin
                if clearance < 0:
                    c += 1e4 * (-clearance)

        # ---- Social potential cost ----
        if pedestrian_params is not None and pedestrian_params.shape[0] > 0:
            for t in range(1, H + 1):
                x = trajectories[k, t, 0]
                y = trajectories[k, t, 1]
                for p in range(pedestrian_params.shape[0]):
                    px, py, phead, B, lam, phi, gaze = pedestrian_params[p]
                    B_eff = B * social_scale   # new parameter you pass from the controller
                    dx = x - px
                    dy = y - py
                    # rotate to body frame
                    total_heading = phead + gaze
                    cos_h = np.cos(total_heading)
                    sin_h = np.sin(total_heading)
                    dx_b = cos_h * dx + sin_h * dy
                    dy_b = -sin_h * dx + cos_h * dy

                    # asymmetric sigmas
                    sig_x = B_eff * (1.0 - lam) if dx_b >= 0 else B_eff * (1.0 + lam)
                    sig_y = B_eff

                    d2 = (dx_b * dx_b) / (sig_x * sig_x) + (dy_b * dy_b) / (
                        sig_y * sig_y
                    )
                    c += Q_social * np.exp(-0.5 * d2)

                # ---------- group penalty (simple line‑segment check) ----------
                M = pedestrian_params.shape[0]
                for i in range(M):
                    for j in range(i + 1, M):
                        dist_ij = np.sqrt(
                            (pedestrian_params[i, 0] - pedestrian_params[j, 0]) ** 2
                            + (pedestrian_params[i, 1] - pedestrian_params[j, 1]) ** 2
                        )
                        if dist_ij < group_threshold:
                            # robot distance to line segment
                            d_line = point_to_line_distance(
                                x,
                                y,
                                pedestrian_params[i, 0],
                                pedestrian_params[i, 1],
                                pedestrian_params[j, 0],
                                pedestrian_params[j, 1],
                            )
                            ax, ay = pedestrian_params[i, 0], pedestrian_params[i, 1]
                            bx, by = pedestrian_params[j, 0], pedestrian_params[j, 1]
                            t_proj = ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / (
                                dist_ij * dist_ij + 1e-10
                            )
                            if 0 < t_proj < 1:
                                c += group_cost_weight * np.exp(
                                    -0.5
                                    * d_line
                                    * d_line
                                    / (group_threshold * group_threshold)
                                )

        # Speed reward
        start_x, start_y = trajectories[k, 0, 0], trajectories[k, 0, 1]
        end_x, end_y = trajectories[k, H, 0], trajectories[k, H, 1]
        displacement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        c -= Q_speed * displacement

        # Terminal position cost (unchanged)
        xf, yf = trajectories[k, H, 0], trajectories[k, H, 1]
        dxf = xf - goal_pos[0]
        dyf = yf - goal_pos[1]
        c += Q_terminal * (dxf * dxf + dyf * dyf)

        costs[k] = c
    return costs


class MPPIController(BaseMPCController):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        self.horizon = 15
        self.num_samples = 2000
        self.lam = 1.0
        self.noise_sigma_v = 0.5
        self.noise_sigma_omega = 0.2
        self.dt_mpc = 0.1
        self.Q_path = 5.0
        self.Q_progress = 30.0
        self.Q_terminal = 50.0
        self.Q_speed = 2.0
        self.Q_heading = 20.0
        self.Q_social = 5.0  # social comfort weight (tune later)
        self.social_scale = config.social_zone_scale   # take from config

        self.U = np.zeros((self.horizon, 2))
        self._warm_start_done = False

        # Override LPFs if desired
        self.LPF = ControlLPF(alpha=0.45)
        self.maneuver_LPF = ControlLPF(alpha=0.45)
        self.DOB_LPF = ControlLPF(alpha=0.45)
        self.use_halton_noise = True  # set to True to test Halton‑spline noise
        if self.use_halton_noise:
            self.noise_gen = NoiseGenerator(
                num_samples_per_step=self.num_samples,
                horizon=self.horizon,
                sigma_v=self.noise_sigma_v,
                sigma_omega=self.noise_sigma_omega,
                pool_size=20000,  # adjust based on memory / desired variety
                n_keypoints=5,
            )

    # ---------- MPC solver (child-specific) ----------
    def _solve_mpc(
        self, robot_state, target_path, circles, walls, pedestrian_params=None
    ) -> Tuple[float, float]:
        x0, y0, th0 = robot_state
        if self.use_halton_noise:
            noise = self.noise_gen.sample()
        else:
            sigma = np.array([self.noise_sigma_v, self.noise_sigma_omega])
            noise = np.random.randn(self.num_samples, self.horizon, 2) * sigma
        if not self._warm_start_done:
            self._warm_start_from_path(robot_state, target_path)
            self._warm_start_done = True

        traj = mppi_rollout_batch(
            x0,
            y0,
            th0,
            self.U,
            noise,
            self.dt_mpc,
            self.config.max_linear_vel,
            self.config.max_angular_vel,
        )
        costs = mppi_compute_costs(
            traj,
            target_path,
            circles,
            walls,
            self.config.robot_radius,
            self.config.safety_margin,
            self.Q_path,
            self.Q_progress,
            self.Q_terminal,
            self.Q_speed,
            self.Q_heading,
            self.Q_social,
            pedestrian_params,
            self.social_scale
        )  # new args

        min_cost = np.min(costs)
        exp_costs = np.exp(-1.0 / self.lam * (costs - min_cost))
        weights = exp_costs / (np.sum(exp_costs) + 1e-10)
        weighted_noise = np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)
        self.U = self.U + weighted_noise
        self.U[:, 0] = np.clip(self.U[:, 0], 0.0, self.config.max_linear_vel)
        self.U[:, 1] = np.clip(
            self.U[:, 1], -self.config.max_angular_vel, self.config.max_angular_vel
        )
        u_raw = self.U[0].copy()
        self.U = np.vstack([self.U[1:], np.zeros((1, 2))])
        return u_raw[0], u_raw[1]

    def _warm_start_from_path(self, robot_state, target_path):
        x0, y0, th0 = robot_state
        H = self.horizon
        dt = self.dt_mpc

        v_des = min(self.config.max_linear_vel * 0.5, 1.5)

        dists = np.hypot(target_path[:, 0] - x0, target_path[:, 1] - y0)
        start_idx = np.argmin(dists)

        pts = target_path[start_idx:]
        if len(pts) < 2:
            self._warm_start_U(robot_state, target_path[-1, :2])
            return

        seg_len = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        cum_len = np.concatenate(([0.0], np.cumsum(seg_len)))

        step_arc = v_des * dt
        total_arc = cum_len[-1]
        if total_arc < step_arc:
            self._warm_start_U(robot_state, target_path[-1, :2])
            return

        for t in range(H):
            target_arc = min((t + 1) * step_arc, total_arc)
            idx = np.searchsorted(cum_len, target_arc) - 1
            idx = max(0, min(idx, len(pts) - 2))
            frac = (target_arc - cum_len[idx]) / (
                cum_len[idx + 1] - cum_len[idx] + 1e-10
            )

            x_ref = pts[idx, 0] + frac * (pts[idx + 1, 0] - pts[idx, 0])
            y_ref = pts[idx, 1] + frac * (pts[idx + 1, 1] - pts[idx, 1])
            th_ref = pts[idx, 2] + frac * (pts[idx + 1, 2] - pts[idx, 2])

            d_th = normalize_angle(th_ref - th0)
            omega_des = np.clip(
                d_th / dt, -self.config.max_angular_vel, self.config.max_angular_vel
            )
            dist_to_ref = np.hypot(x_ref - x0, y_ref - y0)
            v_des_step = np.clip(dist_to_ref / dt, 0.0, self.config.max_linear_vel)

            self.U[t, 0] = v_des_step
            self.U[t, 1] = omega_des

            th0 += omega_des * dt
            x0 += v_des_step * np.cos(th0) * dt
            y0 += v_des_step * np.sin(th0) * dt

    def _warm_start_U(self, robot_state, goal_pos):
        x, y, th = robot_state
        dx = goal_pos[0] - x
        dy = goal_pos[1] - y
        dist = np.hypot(dx, dy)
        if dist < 0.01:
            return
        goal_heading = np.arctan2(dy, dx)
        heading_err = normalize_angle(goal_heading - th)

        v_guess = min(self.config.max_linear_vel, dist / (self.horizon * self.dt_mpc))
        if abs(heading_err) > np.deg2rad(45):
            v_guess = 0.0
        omega_guess = 2.0 * heading_err
        omega_guess = np.clip(
            omega_guess, -self.config.max_angular_vel, self.config.max_angular_vel
        )

        for t in range(self.horizon):
            if t < 3 and abs(heading_err) > 0.1:
                self.U[t, 0] = 0.0
            else:
                self.U[t, 0] = v_guess
            self.U[t, 1] = omega_guess
