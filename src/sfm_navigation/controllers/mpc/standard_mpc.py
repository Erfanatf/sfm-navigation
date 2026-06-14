"""Standard Linear MPC controller – QP‑based, convex optimisation."""
import time
import numpy as np
from typing import Tuple
from scipy.optimize import minimize
from ...config import SimulationConfig
from .base_mpc import BaseMPCController
from ...sfm.numba_utils import normalize_angle


class StandardMPCController(BaseMPCController):
    """
    Linear MPC using successive linearisation of the differential‑drive kinematics.
    Solves a convex Quadratic Program (QP) with box constraints.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        self.horizon = 15                # planning steps
        self.dt_mpc = 0.1                # matches control loop (10 Hz)

        # Quadratic cost weights
        self.Q_pos = 20.0                # position error weight (x,y)
        self.Q_theta = 5.0               # heading error weight
        self.R_v = 0.1                   # linear velocity penalty
        self.R_omega = 0.05              # angular velocity penalty
        self.Q_progress = 30.0           # negative weight on terminal displacement

        # Warm‑start control sequence (shifts previous solution)
        self.U_prev = np.zeros((self.horizon, 2))

    # ------------------------------------------------------------------
    #  Helper: build the QP matrices for a linearised model
    # ------------------------------------------------------------------
    def _build_qp(self, x0: np.ndarray, target_path: np.ndarray,
                  v_curr: float, omega_curr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Form the quadratic program:
            min_{U}  0.5 * U^T H U + f^T U
            s.t.     lb <= U <= ub

        The cost includes a progress reward:  -Q_progress * ||p_H - p_0||^2
        """
        H = self.horizon
        dt = self.dt_mpc
        n = 3       # state dimension
        m = 2       # control dimension

        # ---- 1. Linearisation about current state ----
        theta0 = x0[2]
        v0 = v_curr
        omega0 = omega_curr

        A_c = np.array([[0, 0, -v0 * np.sin(theta0)],
                        [0, 0,  v0 * np.cos(theta0)],
                        [0, 0, 0]])
        B_c = np.array([[np.cos(theta0), 0],
                        [np.sin(theta0), 0],
                        [0,             1]])

        A = np.eye(n) + dt * A_c
        B = dt * B_c

        # ---- 2. Build prediction matrices ----
        A_x0 = np.zeros((H * n, n))
        B_U = np.zeros((H * n, H * m))

        # first step (k=0)
        A_x0[:n] = A
        B_U[:n, :m] = B

        # subsequent steps (k=1..H-1)
        A_pow = A
        for k in range(1, H):
            A_pow = A @ A_pow
            A_x0[k*n:(k+1)*n] = A_pow
            for j in range(k+1):
                B_U[k*n:(k+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, k-j) @ B

        # ---- 3. Reference trajectory ----
        if target_path.shape[0] < H:
            ref = np.tile(target_path[-1], (H, 1))
        else:
            ref = target_path[:H]
        x_ref = np.zeros(H * n)
        for k in range(H):
            x_ref[k*n] = ref[k, 0]
            x_ref[k*n+1] = ref[k, 1]
            x_ref[k*n+2] = ref[k, 2]

        # ---- 4. Cost matrices (quadratic + linear) ----
        Q_diag = np.array([self.Q_pos, self.Q_pos, self.Q_theta])
        R_diag = np.array([self.R_v, self.R_omega])

        Q_bar = np.kron(np.eye(H), np.diag(Q_diag))
        R_bar = np.kron(np.eye(H), np.diag(R_diag))

        # Hessian (quadratic part)
        H_matrix = 2 * (B_U.T @ Q_bar @ B_U + R_bar)

        # Linear part from reference tracking
        f_track = 2 * (B_U.T @ Q_bar @ A_x0 @ x0 - B_U.T @ Q_bar @ x_ref)

        # ---- 5. Progress reward (linear term) ----
        # terminal displacement vector:  p_H - p_0 = [I,0,0; 0,I,0] * (X_last - X_first)
        # Extract (x_H, y_H) from X and add -Q_progress * (||p_H - p_0||^2)
        # Linearise the quadratic progress penalty around U=0 → becomes linear term.
        # For simplicity, we directly add a linear term that rewards movement in the
        # direction of the goal:  f_progress = -Q_progress * (∂||p_H - p_0||^2 / ∂U)
        # We approximate the gradient at U=0 using finite difference (once per step).
        eps = 1e-6
        U_zero = np.zeros(H*m)
        X_zero = A_x0 @ x0 + B_U @ U_zero
        p_start = x0[:2]
        p_terminal_zero = X_zero[-n:-n+2]   # last state's x,y
        dist_sq_zero = np.sum((p_terminal_zero - p_start)**2)

        # gradient of dist_sq w.r.t. U at zero
        grad_progress = np.zeros(H*m)
        for i in range(H*m):
            ei = np.zeros(H*m)
            ei[i] = eps
            X_i = A_x0 @ x0 + B_U @ ei
            p_term_i = X_i[-n:-n+2]
            dist_sq_i = np.sum((p_term_i - p_start)**2)
            grad_progress[i] = (dist_sq_i - dist_sq_zero) / eps
        f_progress = -self.Q_progress * grad_progress

        f_vector = f_track + f_progress

        # ---- 6. Control bounds ----
        v_max = self.config.max_linear_vel
        omega_max = self.config.max_angular_vel
        lb = np.zeros(H * m)
        ub = np.zeros(H * m)
        for k in range(H):
            lb[k*m] = 0.0
            ub[k*m] = v_max
            lb[k*m+1] = -omega_max
            ub[k*m+1] = omega_max

        return H_matrix, f_vector, lb, ub

    def _solve_mpc(self, robot_state: np.ndarray, target_path: np.ndarray,
                   circles: np.ndarray, walls: np.ndarray) -> Tuple[float, float]:
        """
        Solve the convex QP and return the first control input.
        A fallback proportional controller is used if the QP gives near‑zero velocity.
        """
        x0 = robot_state[:3]
        v_curr = robot_state[3]
        omega_curr = robot_state[4]

        # Build QP matrices
        H, f, lb, ub = self._build_qp(x0, target_path, v_curr, omega_curr)

        # Initial guess: shifted previous solution
        U0 = np.vstack([self.U_prev[1:], np.zeros((1, 2))]).flatten()

        t_start = time.perf_counter()
        res = minimize(
            lambda u: 0.5 * u.T @ H @ u + f.T @ u,
            U0,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': 300, 'ftol': 1e-10}
        )
        self.last_compute_time = time.perf_counter() - t_start

        U_opt = res.x.reshape(self.horizon, 2)
        v_cmd = U_opt[0, 0]
        omega_cmd = U_opt[0, 1]

        # Fallback: if the optimiser returns negligible speed, force a simple
        # proportional controller toward the goal to never freeze.
        if v_cmd < 0.05:   # less than 5 cm/s
            goal_x, goal_y = target_path[-1, 0], target_path[-1, 1]
            dx = goal_x - x0[0]
            dy = goal_y - x0[1]
            dist = np.hypot(dx, dy) + 1e-6
            desired_heading = np.arctan2(dy, dx)
            heading_err = normalize_angle(desired_heading - x0[2])
            v_cmd = min(0.8, dist * 0.5)   # proportional gain
            omega_cmd = 3.0 * heading_err
            omega_cmd = np.clip(omega_cmd, -self.config.max_angular_vel, self.config.max_angular_vel)

        self.U_prev = U_opt
        return v_cmd, omega_cmd