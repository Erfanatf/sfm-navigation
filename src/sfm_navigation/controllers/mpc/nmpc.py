"""Nonlinear MPC controller – CasADi + IPOPT, exact kinematics, adaptive scaling."""
import time
import numpy as np
import casadi as ca
from typing import Tuple
from ...config import SimulationConfig
from .base_mpc import BaseMPCController
from ...sfm.numba_utils import normalize_angle


class NMPCController(BaseMPCController):
    """Nonlinear MPC with explicit exact dynamics, adaptive cost scaling, accel limits."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        self.horizon = 15               # planning steps
        self.dt_mpc = 0.1               # control period (10 Hz)

        # Base weights (normalised → 1.0 = equal influence)
        self.w_pos = 1.0
        self.w_terminal = 5.0
        self.w_heading = 3.0
        self.w_speed = 1.0
        self.w_ctrl = 0.1
        self.obstacle_weight = 1e3

        self.U_prev = np.zeros((self.horizon, 2))

    # ------------------------------------------------------------------
    #  Resample path to horizon points (positions only)
    # ------------------------------------------------------------------
    def _resample_path(self, path):
        n = path.shape[0]
        if n == 0:
            return np.zeros((self.horizon, 2))
        pos = path[:, :2]
        t_orig = np.linspace(0, 1, n)
        t_new = np.linspace(0, 1, self.horizon)
        x_new = np.interp(t_new, t_orig, pos[:, 0])
        y_new = np.interp(t_new, t_orig, pos[:, 1])
        return np.column_stack((x_new, y_new))

    # ------------------------------------------------------------------
    #  Feasible constant warm‑start (satisfies acceleration limits)
    # ------------------------------------------------------------------
    def _warm_start_guess(self, x0, path_ref):
        first = path_ref[0]
        dx = first[0] - x0[0]
        dy = first[1] - x0[1]
        dist = np.hypot(dx, dy) + 1e-6
        heading_err = normalize_angle(np.arctan2(dy, dx) - x0[2])
        v_guess = np.clip(dist / self.dt_mpc, 0.0, self.config.max_linear_vel)
        omega_guess = np.clip(heading_err, -self.config.max_angular_vel, self.config.max_angular_vel)
        U = np.tile([v_guess, omega_guess], (self.horizon, 1))
        return U.flatten()

    # ------------------------------------------------------------------
    #  Core solver
    # ------------------------------------------------------------------
    def _solve_mpc(self, robot_state, target_path, circles, walls) -> Tuple[float, float]:
        x0 = robot_state[:3]
        H = self.horizon
        dt = self.dt_mpc
        v_max = self.config.max_linear_vel
        omega_max = self.config.max_angular_vel
        a_max = self.config.max_linear_accel
        alpha_max = self.config.max_angular_accel

        path_ref = self._resample_path(target_path)
        goal_pos = path_ref[-1]

        # Adaptive scaling: distance to goal, clamped at 1.0 m
        D = max(float(np.linalg.norm(x0[:2] - goal_pos)), 1.0)

        # ---- CasADi problem ----
        opti = ca.Opti()
        X = opti.variable(3, H + 1)
        U = opti.variable(2, H)

        opti.subject_to(X[:, 0] == x0)

        # ---- Exact unicycle dynamics and bounds ----
        eps = 1e-6
        for k in range(H):
            x_k = X[:, k]
            u_k = U[:, k]
            v_k = u_k[0]
            omega_k = u_k[1]

            # Straight line (omega ≈ 0)
            x_straight = ca.vertcat(
                x_k[0] + v_k * ca.cos(x_k[2]) * dt,
                x_k[1] + v_k * ca.sin(x_k[2]) * dt,
                x_k[2]
            )

            # Curved motion (omega ≠ 0)
            theta_k = x_k[2]
            theta_next = theta_k + omega_k * dt
            sin_diff = ca.sin(theta_next) - ca.sin(theta_k)
            cos_diff = ca.cos(theta_k) - ca.cos(theta_next)   # cos(θ) - cos(θ+ω dt)
            ratio = v_k / omega_k
            x_curved = ca.vertcat(
                x_k[0] + ratio * sin_diff,
                x_k[1] + ratio * cos_diff,
                theta_next
            )

            use_curved = ca.fabs(omega_k) >= eps
            x_next = ca.if_else(use_curved, x_curved, x_straight)
            opti.subject_to(X[:, k+1] == x_next)

            # Control bounds
            opti.subject_to(U[0, k] >= 0.0)
            opti.subject_to(U[0, k] <= v_max)
            opti.subject_to(U[1, k] >= -omega_max)
            opti.subject_to(U[1, k] <= omega_max)

        # Acceleration limits
        for k in range(H - 1):
            dv = U[0, k+1] - U[0, k]
            domega = U[1, k+1] - U[1, k]
            opti.subject_to(dv <= a_max * dt)
            opti.subject_to(dv >= -a_max * dt)
            opti.subject_to(domega <= alpha_max * dt)
            opti.subject_to(domega >= -alpha_max * dt)

        # ---- Cost function (all normalised) ----
        cost = 0.0

        # Position tracking
        for k in range(1, H + 1):
            pos_err_sq = ca.sumsqr(X[:2, k] - path_ref[k-1])
            cost += self.w_pos * pos_err_sq / (D**2)

        # Terminal error
        terminal_err_sq = ca.sumsqr(X[:2, H] - goal_pos)
        cost += self.w_terminal * terminal_err_sq / (D**2)

        # Heading alignment (smooth)
        for k in range(1, H + 1):
            dx_g = goal_pos[0] - X[0, k]
            dy_g = goal_pos[1] - X[1, k]
            dist_g = ca.sqrt(dx_g**2 + dy_g**2) + 1e-6
            des_x = dx_g / dist_g
            des_y = dy_g / dist_g
            cos_th = ca.cos(X[2, k])
            sin_th = ca.sin(X[2, k])
            dot_prod = cos_th * des_x + sin_th * des_y
            cost += self.w_heading * (1.0 - dot_prod)

        # Control effort
        for k in range(H):
            cost += self.w_ctrl * (U[0, k]**2 / v_max**2 + U[1, k]**2 / omega_max**2)

        # Speed reward
        max_displ = v_max * H * dt
        displacement = ca.sqrt((X[0, H] - x0[0])**2 + (X[1, H] - x0[1])**2)
        cost -= self.w_speed * displacement / max_displ

        # Soft obstacles (circles)
        for i in range(circles.shape[0]):
            cx, cy, cr = circles[i]
            safe_dist = self.config.robot_radius + self.config.safety_margin + cr
            for k in range(1, H + 1):
                d = ca.sqrt((X[0, k] - cx)**2 + (X[1, k] - cy)**2)
                penetration = safe_dist - d
                cost += self.obstacle_weight * ca.fmax(0.0, penetration)**2 / (safe_dist**2)

        # Soft obstacles (walls)
        for j in range(walls.shape[0]):
            x1, y1, x2, y2 = walls[j]
            dx_w = x2 - x1
            dy_w = y2 - y1
            length_sq = dx_w**2 + dy_w**2 + 1e-12
            safe_wall = self.config.robot_radius + self.config.safety_margin
            for k in range(1, H + 1):
                px = X[0, k]
                py = X[1, k]
                t_num = (px - x1)*dx_w + (py - y1)*dy_w
                t_val = ca.fmin(ca.fmax(t_num / length_sq, 0.0), 1.0)
                proj_x = x1 + t_val * dx_w
                proj_y = y1 + t_val * dy_w
                d_w = ca.sqrt((px - proj_x)**2 + (py - proj_y)**2)
                penetration = safe_wall - d_w
                cost += self.obstacle_weight * ca.fmax(0.0, penetration)**2 / (safe_wall**2)

        opti.minimize(cost)

        # ---- Warm‑start (constant sequence) ----
        if np.any(self.U_prev):
            U0 = np.vstack([self.U_prev[1:], self.U_prev[-1]])
            U0_flat = U0.flatten()
        else:
            U0_flat = self._warm_start_guess(x0, path_ref)
        opti.set_initial(U, U0_flat.reshape(2, H))

        # State initial guess (exact integration for consistency)
        x_guess = np.zeros((3, H + 1))
        x_guess[:, 0] = x0
        for k in range(H):
            v_k = U0_flat[2*k]
            omega_k = U0_flat[2*k+1]
            if abs(omega_k) < 1e-6:
                x_guess[0, k+1] = x_guess[0, k] + v_k * np.cos(x_guess[2, k]) * dt
                x_guess[1, k+1] = x_guess[1, k] + v_k * np.sin(x_guess[2, k]) * dt
                x_guess[2, k+1] = x_guess[2, k]
            else:
                th = x_guess[2, k]
                x_guess[0, k+1] = x_guess[0, k] + (v_k/omega_k)*(np.sin(th+omega_k*dt) - np.sin(th))
                x_guess[1, k+1] = x_guess[1, k] + (v_k/omega_k)*(np.cos(th) - np.cos(th+omega_k*dt))
                x_guess[2, k+1] = th + omega_k * dt
        opti.set_initial(X, x_guess)

        # Solver options
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 200,
            "print_level": 0,
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
            return U_opt[0, 0], U_opt[1, 0]
        except Exception as e:
            print(f"  [NMPC] solver failed: {e}, using fallback")
            self.last_compute_time = time.perf_counter() - t_start
            return U0_flat[0], U0_flat[1]