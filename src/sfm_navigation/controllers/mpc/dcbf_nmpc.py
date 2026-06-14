"""DCBF‑NMPC controller – hard DCBF constraints (log‑sum‑exp), multi‑mode fallback."""

import time
import numpy as np
import casadi as ca
from typing import Tuple, Optional
from ...config import SimulationConfig
from .nmpc import NMPCController
from .dcbf_mppi import (
    cbf_safe_backup_control,
    _optimization_projection,
    _cbf_value_for_state,
    diff_drive_dynamics,
)
from ...sfm.numba_utils import normalize_angle


def _filter_by_distance(circles, robot_x, robot_y, radius=4.0):
    if circles.shape[0] == 0:
        return circles
    dist = np.hypot(circles[:, 0] - robot_x, circles[:, 1] - robot_y)
    return circles[dist <= radius]


def _is_safe(self, x, y, theta, v, omega, circles, walls):
    """Return True if (v,omega) satisfies DCBF for one step."""
    x_next, y_next, _ = diff_drive_dynamics(x, y, theta, v, omega, self.dt_mpc)
    min_dcbf = _cbf_value_for_state(
        x, y, x_next, y_next,
        circles, walls,
        self.config.robot_radius, self.config.safety_margin,
        self.gamma,
    )
    return min_dcbf >= 0


class DCBFNMPCController(NMPCController):
    """
    NMPC with Discrete CBF hard constraints and multi‑mode fallback.

    Fallback modes:
      - 'grid'        : grid search over (v, ω) (same as DCBF‑MPPI)
      - 'optimization': scipy‑based constrained optimisation
      - 'analytical'  : sequential 1D projection (fast, handles dynamic obs)
    """

    def __init__(self, config: SimulationConfig, fallback_mode: str = "grid"):
        super().__init__(config)

        # Override default weights for stronger goal attraction
        self.w_pos = 1.0
        self.w_terminal = 10.0
        self.w_heading = 3.0
        self.w_speed = 3.0
        self.w_ctrl = 0.05
        self.obstacle_weight = 1e3

        # DCBF parameters
        self.gamma = 0.3
        self.rho_slack = 1e6
        self.always_project = True

        # Fallback mode
        allowed_modes = {"grid", "optimization", "analytical"}
        if fallback_mode not in allowed_modes:
            raise ValueError(f"fallback_mode must be one of {allowed_modes}")
        self.fallback_mode = fallback_mode
        self.verbose = True
        self.beta = 10.0   # sharpness of log‑sum‑exp approximation

    # ------------------------------------------------------------------
    #  Analytical projection (unchanged)
    # ------------------------------------------------------------------
    def _analytical_projection(
        self, state, u_nominal, circles, walls, obstacle_velocities=None
    ):
        x, y, theta = state
        v = u_nominal[0]
        w = u_nominal[1]
        v_max = self.config.max_linear_vel
        omega_max = self.config.max_angular_vel
        safe_base = self.config.robot_radius + self.config.safety_margin

        obs_list = []
        for i in range(circles.shape[0]):
            pos = circles[i, :2]
            r = circles[i, 2]
            total_r = safe_base + r
            vel = (
                obstacle_velocities[i]
                if obstacle_velocities is not None
                else np.zeros(2)
            )
            obs_list.append((pos, total_r, vel))

        obs_list.sort(key=lambda o: np.linalg.norm(o[0] - state[:2]))

        for pos, r_safe, v_o in obs_list:
            delta = state[:2] - pos
            h = np.dot(delta, delta) - r_safe**2
            a = 2.0 * (delta[0] * np.cos(theta) + delta[1] * np.sin(theta))
            dynamic_term = 2.0 * np.dot(delta, v_o)
            rhs = -self.gamma * h + dynamic_term

            if a * v >= rhs - 1e-12:
                continue
            if abs(a) < 1e-8:
                continue
            v_req = rhs / a
            if a > 0:
                v_new = max(v, v_req)
            else:
                v_new = min(v, v_req)
            v = np.clip(v_new, 0.0, v_max)
        w = np.clip(w, -omega_max, omega_max)
        return np.array([v, w])

    # ------------------------------------------------------------------
    #  Unified fallback call (unchanged)
    # ------------------------------------------------------------------
    def _fallback_safe_control(
        self, x0, u_nominal, circles, walls, obstacle_velocities=None
    ):
        if self.fallback_mode == "grid":
            v_safe, omega_safe = cbf_safe_backup_control(
                x0[0], x0[1], x0[2],
                u_nominal[0], u_nominal[1],
                self.dt_mpc, circles, walls,
                self.config.robot_radius, self.config.safety_margin,
                self.gamma,
                self.config.min_linear_vel, self.config.max_linear_vel,
                self.config.max_angular_vel,
            )
            return np.array([v_safe, omega_safe])

        elif self.fallback_mode == "optimization":
            v_safe, omega_safe = _optimization_projection(
                x0[0], x0[1], x0[2],
                u_nominal[0], u_nominal[1],
                self.dt_mpc, circles, walls,
                self.config.robot_radius, self.config.safety_margin,
                self.gamma,
                self.config.min_linear_vel, self.config.max_linear_vel,
                self.config.max_angular_vel,
            )
            return np.array([v_safe, omega_safe])

        else:
            return self._analytical_projection(
                x0, u_nominal, circles, walls, obstacle_velocities
            )

    # ------------------------------------------------------------------
    #  Override solver (log‑sum‑exp DCBF)
    # ------------------------------------------------------------------
    def _solve_mpc(
        self, robot_state, target_path, circles, walls, obstacle_velocities=None
    ) -> Tuple[float, float]:
        x0 = robot_state[:3]
        H = self.horizon
        dt = self.dt_mpc
        v_max = self.config.max_linear_vel
        omega_max = self.config.max_angular_vel
        a_max = self.config.max_linear_accel
        alpha_max = self.config.max_angular_accel

        path_ref = self._resample_path(target_path)
        goal_pos = path_ref[-1]
        D = max(float(np.linalg.norm(x0[:2] - goal_pos)), 1.0)

        # ---- CasADi optimisation ----
        opti = ca.Opti()
        X = opti.variable(3, H + 1)
        U = opti.variable(2, H)
        slack = opti.variable()
        opti.subject_to(slack >= 0)
        opti.subject_to(X[:, 0] == x0)

        # Dynamics (exact unicycle)
        eps = 1e-6
        for k in range(H):
            x_k = X[:, k]
            u_k = U[:, k]
            v_k, omega_k = u_k[0], u_k[1]
            x_straight = ca.vertcat(
                x_k[0] + v_k * ca.cos(x_k[2]) * dt,
                x_k[1] + v_k * ca.sin(x_k[2]) * dt,
                x_k[2],
            )
            theta_k = x_k[2]
            theta_next = theta_k + omega_k * dt
            sin_diff = ca.sin(theta_next) - ca.sin(theta_k)
            cos_diff = ca.cos(theta_k) - ca.cos(theta_next)
            ratio = v_k / omega_k
            x_curved = ca.vertcat(
                x_k[0] + ratio * sin_diff,
                x_k[1] + ratio * cos_diff,
                theta_next,
            )
            use_curved = ca.fabs(omega_k) >= eps
            x_next = ca.if_else(use_curved, x_curved, x_straight)
            opti.subject_to(X[:, k + 1] == x_next)
            opti.subject_to(U[0, k] >= 0.0)
            opti.subject_to(U[0, k] <= v_max)
            opti.subject_to(U[1, k] >= -omega_max)
            opti.subject_to(U[1, k] <= omega_max)

        # Acceleration limits
        for k in range(H - 1):
            dv = U[0, k + 1] - U[0, k]
            domega = U[1, k + 1] - U[1, k]
            opti.subject_to(dv <= a_max * dt)
            opti.subject_to(dv >= -a_max * dt)
            opti.subject_to(domega <= alpha_max * dt)
            opti.subject_to(domega >= -alpha_max * dt)

        # ---- Filter obstacles to vicinity ----
        circles = _filter_by_distance(circles, x0[0], x0[1], radius=4.0)

        # ---- Unified DCBF via log‑sum‑exp (soft‑min) ----
        beta = self.beta
        safe_dist = self.config.robot_radius + self.config.safety_margin
        for k in range(H):
            residuals = []
            # Circular obstacles
            for i in range(circles.shape[0]):
                cx, cy, cr = circles[i]
                r_total = safe_dist + cr
                h_k = ca.sumsqr(X[:2, k] - ca.vertcat(cx, cy)) - r_total**2
                h_next = ca.sumsqr(X[:2, k + 1] - ca.vertcat(cx, cy)) - r_total**2
                r_i = h_next - (1 - self.gamma) * h_k
                residuals.append(r_i)
            # Wall obstacles
            for j in range(walls.shape[0]):
                x1, y1, x2, y2 = walls[j]
                dx_w, dy_w = x2 - x1, y2 - y1
                length_sq = dx_w**2 + dy_w**2 + 1e-12
                # current step
                px, py = X[0, k], X[1, k]
                t_num = (px - x1) * dx_w + (py - y1) * dy_w
                t_val = ca.fmin(ca.fmax(t_num / length_sq, 0.0), 1.0)
                proj_x = x1 + t_val * dx_w
                proj_y = y1 + t_val * dy_w
                dist_k = ca.sqrt((px - proj_x)**2 + (py - proj_y)**2)
                h_k_w = dist_k**2 - safe_dist**2
                # next step
                px_next, py_next = X[0, k + 1], X[1, k + 1]
                t_num_next = (px_next - x1) * dx_w + (py_next - y1) * dy_w
                t_val_next = ca.fmin(ca.fmax(t_num_next / length_sq, 0.0), 1.0)
                proj_x_next = x1 + t_val_next * dx_w
                proj_y_next = y1 + t_val_next * dy_w
                dist_next = ca.sqrt((px_next - proj_x_next)**2 + (py_next - proj_y_next)**2)
                h_next_w = dist_next**2 - safe_dist**2
                r_i = h_next_w - (1 - self.gamma) * h_k_w
                residuals.append(r_i)

            if len(residuals) == 0:
                continue
            res_stack = ca.vertcat(*residuals)       # vector of individual DCBF residuals
            lse = -(1.0 / beta) * ca.log(ca.sum1(ca.exp(-beta * res_stack)))
            opti.subject_to(lse + slack >= 0)

        # ---- Cost (unchanged) ----
        L_term = 0.5
        cost = 0.0
        for k in range(1, H + 1):
            pos_err_sq = ca.sumsqr(X[:2, k] - path_ref[k - 1])
            cost += self.w_pos * pos_err_sq / (D**2)
        terminal_err_sq = ca.sumsqr(X[:2, H] - goal_pos)
        cost += self.w_terminal * terminal_err_sq / (L_term**2)
        for k in range(1, H + 1):
            dx_g = goal_pos[0] - X[0, k]
            dy_g = goal_pos[1] - X[1, k]
            dist_g = ca.sqrt(dx_g**2 + dy_g**2) + 1e-6
            des_x, des_y = dx_g / dist_g, dy_g / dist_g
            cos_th, sin_th = ca.cos(X[2, k]), ca.sin(X[2, k])
            dot_prod = cos_th * des_x + sin_th * des_y
            cost += self.w_heading * (1.0 - dot_prod)
        for k in range(H):
            cost += self.w_ctrl * (U[0, k]**2 / v_max**2 + U[1, k]**2 / omega_max**2)
        speed_scale = 1.0
        displacement = ca.sqrt((X[0, H] - x0[0])**2 + (X[1, H] - x0[1])**2)
        cost -= self.w_speed * displacement / speed_scale
        cost += self.rho_slack * slack**2

        opti.minimize(cost)

        # Warm‑start (unchanged)
        if np.any(self.U_prev):
            U0 = np.vstack([self.U_prev[1:], self.U_prev[-1]])
            U0_flat = U0.flatten()
        else:
            U0_flat = self._warm_start_guess(x0, path_ref)
        opti.set_initial(U, U0_flat.reshape(2, H))
        opti.set_initial(slack, 0.0)

        x_guess = np.zeros((3, H + 1))
        x_guess[:, 0] = x0
        for k in range(H):
            v_k, omega_k = U0_flat[2 * k], U0_flat[2 * k + 1]
            if abs(omega_k) < 1e-6:
                x_guess[0, k + 1] = x_guess[0, k] + v_k * np.cos(x_guess[2, k]) * dt
                x_guess[1, k + 1] = x_guess[1, k] + v_k * np.sin(x_guess[2, k]) * dt
                x_guess[2, k + 1] = x_guess[2, k]
            else:
                th = x_guess[2, k]
                x_guess[0, k + 1] = x_guess[0, k] + (v_k / omega_k) * (
                    np.sin(th + omega_k * dt) - np.sin(th)
                )
                x_guess[1, k + 1] = x_guess[1, k] + (v_k / omega_k) * (
                    np.cos(th) - np.cos(th + omega_k * dt)
                )
                x_guess[2, k + 1] = th + omega_k * dt
        opti.set_initial(X, x_guess)

        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 200,
            "print_level": 0,
            "print_timing_statistics": "no",
            "print_user_options": "no",
            "acceptable_tol": 1e-4,
            "constr_viol_tol": 1e-6,
        }
        opti.solver("ipopt", p_opts, s_opts)

        t_start = time.perf_counter()
        try:
            sol = opti.solve()
            self.last_compute_time = time.perf_counter() - t_start
            U_opt = sol.value(U)
            self.U_prev = U_opt.T
            u_cmd = np.array([U_opt[0, 0], U_opt[1, 0]])

            if self.always_project:
                slack_val = float(sol.value(slack))
                if slack_val > 1e-6 or not self._is_safe(
                    x0[0], x0[1], x0[2], u_cmd[0], u_cmd[1], circles, walls
                ):
                    if self.verbose:
                        print(
                            f"  [DCBF-NMPC] post‑solve projection (slack={slack_val:.3e})"
                        )
                    u_safe = self._fallback_safe_control(
                        x0, u_cmd, circles, walls, obstacle_velocities
                    )
                    return u_safe[0], u_safe[1]
            return u_cmd[0], u_cmd[1]

        except Exception as e:
            self.last_compute_time = time.perf_counter() - t_start
            u_nominal = U0_flat[:2]
            u_safe = self._fallback_safe_control(
                x0, u_nominal, circles, walls, obstacle_velocities
            )
            if self.verbose:
                print(
                    f"  [DCBF-NMPC] solver failed ({e}), fallback={self.fallback_mode}",
                    f"nominal=({u_nominal[0]:.2f}, {u_nominal[1]:.2f})",
                    f"→ safe=({u_safe[0]:.2f}, {u_safe[1]:.2f})"
                )
            return u_safe[0], u_safe[1]