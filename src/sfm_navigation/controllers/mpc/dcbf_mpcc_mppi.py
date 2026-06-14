"""MPCC‑MPPI controller – MPPI with contouring/lag cost + DCBF safety."""

import numpy as np
from numba import njit
from typing import Tuple

from ...config import SimulationConfig
from .dcbf_mppi import DCBFMPPIController
from .mppi import mppi_rollout_batch
from ...sfm.numba_utils import normalize_angle, point_to_line_distance

@njit
def _filter_by_distance(circles, robot_x, robot_y, radius=4.0):
    if circles.shape[0] == 0:
        return circles
    dist = np.hypot(circles[:, 0] - robot_x, circles[:, 1] - robot_y)
    return circles[dist <= radius]

@njit
def mpcc_mppi_compute_costs(
    trajectories,
    target_path,
    circles,
    walls,
    robot_radius,
    safety_margin,
    Q_contour,
    Q_lag,
    Q_terminal,
    Q_speed,
    Q_heading,
    Q_goal,
    goal_pos,
    D_scale,
    Q_social,
    pedestrian_params,  # new
    social_scale,
    group_threshold=1.5,
    group_cost=2.0,
):
    """
    Contouring/lag cost + strong goal attraction.
    Terminal cost and per‑step goal distance are NOT scaled.
    """
    K = trajectories.shape[0]
    H = trajectories.shape[1] - 1
    costs = np.zeros(K)

    for k in range(K):
        cost = 0.0
        x_start = trajectories[k, 0, 0]
        y_start = trajectories[k, 0, 1]

        for t in range(1, H + 1):
            x = trajectories[k, t, 0]
            y = trajectories[k, t, 1]
            th = trajectories[k, t, 2]

            # Closest point on path
            min_dsq = 1e10
            closest_idx = 0
            for i in range(target_path.shape[0]):
                dx = x - target_path[i, 0]
                dy = y - target_path[i, 1]
                dsq = dx * dx + dy * dy
                if dsq < min_dsq:
                    min_dsq = dsq
                    closest_idx = i

            # Tangent at closest point
            path_heading = target_path[closest_idx, 2]
            cos_h = np.cos(path_heading)
            sin_h = np.sin(path_heading)

            dx_p = x - target_path[closest_idx, 0]
            dy_p = y - target_path[closest_idx, 1]

            e_l = dx_p * cos_h + dy_p * sin_h
            e_c = -dx_p * sin_h + dy_p * cos_h

            # Contouring/lag (scaled)
            cost += Q_contour * (e_c * e_c) / (D_scale * D_scale)
            cost += Q_lag * (e_l * e_l) / (D_scale * D_scale)

            # Direct goal distance (NOT scaled) – strong pull
            dxg = goal_pos[0] - x
            dyg = goal_pos[1] - y
            cost += Q_goal * (dxg * dxg + dyg * dyg)

            # Heading alignment to goal
            desired_heading = np.arctan2(dyg, dxg + 1e-6)
            d_th = th - desired_heading
            while d_th > np.pi:
                d_th -= 2 * np.pi
            while d_th < -np.pi:
                d_th += 2 * np.pi
            cost += Q_heading * (d_th * d_th)

        # ---- Social potential ----
        if pedestrian_params is not None and pedestrian_params.shape[0] > 0:
            for p in range(pedestrian_params.shape[0]):
                px, py, phead, B, lam, phi, gaze = pedestrian_params[p]
                B_eff = B * social_scale   # new parameter you pass from the controller
                dx = x - px
                dy = y - py
                total_heading = phead + gaze
                cos_h = np.cos(total_heading)
                sin_h = np.sin(total_heading)
                dx_b = cos_h * dx + sin_h * dy
                dy_b = -sin_h * dx + cos_h * dy
                sig_x = B_eff * (1.0 - lam) if dx_b >= 0 else B_eff * (1.0 + lam)
                sig_y = B_eff

                d2 = (dx_b * dx_b) / (sig_x * sig_x) + (dy_b * dy_b) / (
                    sig_y * sig_y
                )
                cost += Q_social * np.exp(-0.5 * d2)

            # ---- Group penalty ----
            M = pedestrian_params.shape[0]
            for i in range(M):
                for j in range(i + 1, M):
                    dist_ij = np.sqrt(
                        (pedestrian_params[i, 0] - pedestrian_params[j, 0]) ** 2
                        + (pedestrian_params[i, 1] - pedestrian_params[j, 1]) ** 2
                    )
                    if dist_ij < group_threshold:
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
                            cost += group_cost * np.exp(
                                -0.5
                                * d_line
                                * d_line
                                / (group_threshold * group_threshold)
                            )

        # Terminal cost (unscaled, very strong)
        x_final = trajectories[k, H, 0]
        y_final = trajectories[k, H, 1]
        dxg = x_final - goal_pos[0]
        dyg = y_final - goal_pos[1]
        cost += Q_terminal * (dxg * dxg + dyg * dyg)

        # Speed reward
        displacement = np.sqrt((x_final - x_start) ** 2 + (y_final - y_start) ** 2)
        cost -= Q_speed * displacement

        costs[k] = cost
    return costs


class DCBFMPCCMPPIController(DCBFMPPIController):
    """MPPI with MPCC cost, adaptive scaling, and heading alignment."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        # Balanced weights
        self.Q_contour = 10.0  # slightly reduced
        self.Q_lag = 10.0
        self.Q_terminal = 1000.0  # massive terminal pull
        self.Q_speed = 40.0
        self.Q_heading = 15.0
        self.Q_goal = 200.0  # strong per‑step goal attraction
        self.Q_social = 5.0  # social comfort weight
        self.social_scale = config.social_zone_scale   # take from config

        self.lam = 5.0
        self.noise_sigma_v = 0.8
        self.noise_sigma_omega = 0.3

    def _solve_mpc(
        self, robot_state, target_path, circles, walls, pedestrian_params=None
    ) -> Tuple[float, float]:
        x0, y0, th0 = robot_state
        goal_pos = target_path[-1, :2]

        D_scale = max(np.hypot(x0 - goal_pos[0], y0 - goal_pos[1]), 1.0)

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

        costs = mpcc_mppi_compute_costs(
            traj,
            target_path,
            circles,
            walls,
            self.config.robot_radius,
            self.config.safety_margin,
            self.Q_contour,
            self.Q_lag,
            self.Q_terminal,
            self.Q_speed,
            self.Q_heading,
            self.Q_goal,
            goal_pos,
            D_scale,
            self.Q_social,
            pedestrian_params,  # new
            self.social_scale
        )

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
