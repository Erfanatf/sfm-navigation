"""DCBF‑MPPI controller – MPPI with Discrete CBF safety layer.

Supports two projection methods:
  - Fast relaxed search (default): grid evaluation with penalty, never freezes.
  - Optimization‑based (flag ``use_optimization=True``): solves a
    constrained QP to find the closest safe command with a slack variable.
"""

import time
import numpy as np
from numba import njit
from typing import Tuple, Optional

from ...config import SimulationConfig
from .mppi import MPPIController
from ...sfm.numba_utils import point_to_line_distance, normalize_angle
from .mppi_noise import NoiseGenerator

@njit
def _filter_by_distance(circles, robot_x, robot_y, radius=4.0):
    if circles.shape[0] == 0:
        return circles
    dist = np.hypot(circles[:, 0] - robot_x, circles[:, 1] - robot_y)
    return circles[dist <= radius]

def _analytical_projection_continuous(
    x,
    y,
    theta,
    v_desired,
    omega_desired,
    dt,
    circles,
    walls,
    robot_radius,
    safety_margin,
    gamma,
    v_min,
    v_max,
    omega_max,
):
    """
    Analytical projection using the *continuous‑time* CBF condition:
        L_f h + L_g h * v >= -gamma * h
    (L_f h = 0 for static obstacles).
    Works sequentially, closest obstacle first, and only adjusts v.
    """
    v = v_desired
    omega = omega_desired
    safe_base = robot_radius + safety_margin

    # Build list of (obstacle_x, obstacle_y, total_safe_radius)
    obs_list = []
    for i in range(circles.shape[0]):
        obs_list.append((circles[i, 0], circles[i, 1], safe_base + circles[i, 2]))
    # For walls, approximate as a point at the closest point on the segment
    for j in range(walls.shape[0]):
        x1, y1, x2, y2 = walls[j]
        dx_w, dy_w = x2 - x1, y2 - y1
        length_sq = dx_w**2 + dy_w**2 + 1e-12
        t = max(0.0, min(1.0, ((x - x1) * dx_w + (y - y1) * dy_w) / length_sq))
        px, py = x1 + t * dx_w, y1 + t * dy_w
        obs_list.append((px, py, safe_base))  # wall has zero physical radius

    # Sort by current distance (closest first)
    obs_list.sort(key=lambda o: np.hypot(o[0] - x, o[1] - y))

    # Pre‑compute heading unit vector
    dir_x = np.cos(theta)
    dir_y = np.sin(theta)

    for ox, oy, r_total in obs_list:
        delta_x = x - ox
        delta_y = y - oy
        h = delta_x**2 + delta_y**2 - r_total**2

        # L_g h (only v component affects position)
        a = 2.0 * (delta_x * dir_x + delta_y * dir_y)
        rhs = -gamma * h

        # Already safe? (a * v >= rhs within tolerance)
        if a * v >= rhs - 1e-12:
            continue

        # If a is extremely small, v cannot influence this obstacle
        if abs(a) < 1e-8:
            continue

        v_req = rhs / a
        if a > 0:
            # Need v >= v_req
            v = max(v, v_req)
        else:
            # a < 0 -> need v <= v_req
            v = min(v, v_req)

        # Enforce actuator limits
        v = np.clip(v, v_min, v_max)

    omega = np.clip(omega, -omega_max, omega_max)
    return v, omega


# ----------------------------------------------------------------------
#  JIT helpers
# ----------------------------------------------------------------------
@njit
def diff_drive_dynamics(x, y, theta, v, omega, dt):
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt
    return x_new, y_new, theta_new


@njit
def _cbf_value_for_state(
    x,
    y,
    x_next,
    y_next,
    obstacles_circles,
    obstacles_walls,
    robot_radius,
    safety_margin,
    gamma,
):
    min_dcbf = 1e10
    safe_dist = robot_radius + safety_margin

    for i in range(obstacles_circles.shape[0]):
        cx, cy, r = (
            obstacles_circles[i, 0],
            obstacles_circles[i, 1],
            obstacles_circles[i, 2],
        )
        d_total = safe_dist + r
        h_curr = (x - cx) ** 2 + (y - cy) ** 2 - d_total**2
        h_next = (x_next - cx) ** 2 + (y_next - cy) ** 2 - d_total**2
        dcbf = h_next - (1.0 - gamma) * h_curr
        if dcbf < min_dcbf:
            min_dcbf = dcbf

    for i in range(obstacles_walls.shape[0]):
        x1, y1 = obstacles_walls[i, 0], obstacles_walls[i, 1]
        x2, y2 = obstacles_walls[i, 2], obstacles_walls[i, 3]
        dist_curr = point_to_line_distance(x, y, x1, y1, x2, y2)
        dist_next = point_to_line_distance(x_next, y_next, x1, y1, x2, y2)
        h_curr = dist_curr**2 - safe_dist**2
        h_next = dist_next**2 - safe_dist**2
        dcbf = h_next - (1.0 - gamma) * h_curr
        if dcbf < min_dcbf:
            min_dcbf = dcbf

    return min_dcbf


# ----------------------------------------------------------------------
#  Fast relaxed search (grid)
# ----------------------------------------------------------------------
@njit
def cbf_safe_backup_control(
    x,
    y,
    theta,
    v_desired,
    omega_desired,
    dt,
    obstacles_circles,
    obstacles_walls,
    robot_radius,
    safety_margin,
    gamma,
    v_min,
    v_max,
    omega_max,
):
    penalty_weight = 500.0
    best_cost = 1e10
    best_v, best_omega = 0.0, 0.0

    v_scales = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.2, -0.4, -0.6]
    )
    omega_offsets = np.array(
        [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    )

    for v_scale in v_scales:
        v_test = v_desired * v_scale
        v_test = max(v_min, min(v_max, v_test))
        for omega_off in omega_offsets:
            omega_test = omega_desired + omega_off
            omega_test = max(-omega_max, min(omega_max, omega_test))

            x_next, y_next, _ = diff_drive_dynamics(x, y, theta, v_test, omega_test, dt)
            min_dcbf = _cbf_value_for_state(
                x,
                y,
                x_next,
                y_next,
                obstacles_circles,
                obstacles_walls,
                robot_radius,
                safety_margin,
                gamma,
            )
            violation = max(0.0, -min_dcbf)
            cost = (
                (v_test - v_desired) ** 2
                + 0.1 * (omega_test - omega_desired) ** 2
                + penalty_weight * violation**2
            )
            if cost < best_cost:
                best_cost = cost
                best_v, best_omega = v_test, omega_test

    # Explicitly check the raw desired command
    x_next, y_next, _ = diff_drive_dynamics(x, y, theta, v_desired, omega_desired, dt)
    min_dcbf = _cbf_value_for_state(
        x,
        y,
        x_next,
        y_next,
        obstacles_circles,
        obstacles_walls,
        robot_radius,
        safety_margin,
        gamma,
    )
    violation = max(0.0, -min_dcbf)
    cost = penalty_weight * violation**2
    if cost < best_cost:
        best_v, best_omega = v_desired, omega_desired

    return best_v, best_omega


# ----------------------------------------------------------------------
#  Optimization‑based projection (scipy) – used when use_optimization=True
# ----------------------------------------------------------------------
def _optimization_projection(
    x,
    y,
    theta,
    v_nom,
    omega_nom,
    dt,
    circles,
    walls,
    robot_radius,
    safety_margin,
    gamma,
    v_min,
    v_max,
    omega_max,
):
    """
    Solve the constrained optimisation problem:
        min  (v - v_nom)² + 0.1*(ω - ω_nom)² + ρ * slack²
        s.t. DCBF_i(v, ω) + slack ≥ 0  ∀ obstacles
             0 ≤ slack
             |v| ≤ v_max, |ω| ≤ omega_max
    Returns (v_safe, omega_safe).
    """
    try:
        from scipy.optimize import minimize, Bounds
    except ImportError:
        print("  [CBF] scipy not available, falling back to fast relaxed search.")
        return cbf_safe_backup_control(
            x,
            y,
            theta,
            v_nom,
            omega_nom,
            dt,
            circles,
            walls,
            robot_radius,
            safety_margin,
            gamma,
            v_min,
            v_max,
            omega_max,
        )

    # Number of obstacles
    n_circles = circles.shape[0]
    n_walls = walls.shape[0]
    n_constraints = n_circles + n_walls

    # Pre‑compute h_current for each obstacle
    h_curr = np.zeros(n_constraints)
    safe_dist = robot_radius + safety_margin
    for i in range(n_circles):
        cx, cy, r = circles[i, 0], circles[i, 1], circles[i, 2]
        d_total = safe_dist + r
        h_curr[i] = (x - cx) ** 2 + (y - cy) ** 2 - d_total**2
    for i in range(n_walls):
        x1, y1 = walls[i, 0], walls[i, 1]
        x2, y2 = walls[i, 2], walls[i, 3]
        dist_curr = point_to_line_distance(x, y, x1, y1, x2, y2)
        h_curr[n_circles + i] = dist_curr**2 - safe_dist**2

    def objective(z):
        v, omega, slack = z
        return (v - v_nom) ** 2 + 0.1 * (omega - omega_nom) ** 2 + 500.0 * slack**2

    def dcbf_constraints(z):
        v, omega, slack = z
        x_next, y_next, _ = diff_drive_dynamics(x, y, theta, v, omega, dt)
        # Compute h_next
        h_next = np.zeros(n_constraints)
        for i in range(n_circles):
            cx, cy, r = circles[i, 0], circles[i, 1], circles[i, 2]
            d_total = safe_dist + r
            h_next[i] = (x_next - cx) ** 2 + (y_next - cy) ** 2 - d_total**2
        for i in range(n_walls):
            x1, y1 = walls[i, 0], walls[i, 1]
            x2, y2 = walls[i, 2], walls[i, 3]
            dist_next = point_to_line_distance(x_next, y_next, x1, y1, x2, y2)
            h_next[n_circles + i] = dist_next**2 - safe_dist**2
        # DCBF: h_next - (1-gamma)*h_curr + slack >= 0
        return h_next - (1.0 - gamma) * h_curr + slack

    # Initial guess
    z0 = np.array([v_nom, omega_nom, 0.0])
    bounds = Bounds([v_min, -omega_max, 0.0], [v_max, omega_max, 1e6])
    constraints = {"type": "ineq", "fun": dcbf_constraints}

    try:
        result = minimize(
            objective,
            z0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 30, "ftol": 1e-4},
        )
        if result.success:
            v_safe = np.clip(result.x[0], v_min, v_max)
            omega_safe = np.clip(result.x[1], -omega_max, omega_max)
        else:
            # Fallback to grid search on optimisation failure
            return cbf_safe_backup_control(
                x,
                y,
                theta,
                v_nom,
                omega_nom,
                dt,
                circles,
                walls,
                robot_radius,
                safety_margin,
                gamma,
                v_max,
                omega_max,
            )
    except Exception:
        return cbf_safe_backup_control(
            x,
            y,
            theta,
            v_nom,
            omega_nom,
            dt,
            circles,
            walls,
            robot_radius,
            safety_margin,
            gamma,
            v_max,
            omega_max,
        )

    return v_safe, omega_safe


# ----------------------------------------------------------------------
#  DCBF‑MPPI Controller class
# ----------------------------------------------------------------------
class DCBFMPPIController(MPPIController):
    """
    MPPI with Discrete CBF safety layer, two projection modes:

    * ``use_optimization = False`` (default): fast relaxed grid search
      ensures the robot never freezes.
    * ``use_optimization = True``: solves a constrained optimisation
      problem with scipy to find the closest safe control (slower, but
      more precise).
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        # CBF parameters
        self.cbf_gamma = 0.3
        self.cbf_safety_margin = config.safety_margin

        # Projection method flag
        self.use_optimization = False  # set to True to enable scipy solver
        self.fallback_mode = (
            "grid"  # default fallback mode (can be overridden by robot_demo)
        )

        # Logging
        self._cbf_interventions = 0
        self._last_cbf_state = False

    def compute_velocity(
        self,
        robot_state,
        goal_pos,
        obstacles,
        user: Optional[dict] = None,
        dt: float = None,
        sim_time: float = None,
        **kwargs,
    ) -> Tuple[float, float]:
        # 1. Nominal command from parent (MPPI + DOB + LPFs + maneuvers)
        v_nom, omega_nom = super().compute_velocity(
            robot_state,
            goal_pos,
            obstacles,
            user=user,
            dt=dt,
            sim_time=sim_time,
            **kwargs,
        )

        if dt is None:
            dt = self.config.dt

        circles, walls = self._convert_obstacles(obstacles)
        x, y, theta = robot_state.x, robot_state.y, robot_state.theta
        circles = _filter_by_distance(circles, x, y, radius=4.0)

        # 2. Check if nominal command is already safe
        x_next_nom, y_next_nom, _ = diff_drive_dynamics(
            x, y, theta, v_nom, omega_nom, dt
        )
        min_dcbf_nom = _cbf_value_for_state(
            x,
            y,
            x_next_nom,
            y_next_nom,
            circles,
            walls,
            self.config.robot_radius,
            self.cbf_safety_margin,
            self.cbf_gamma,
        )

        if min_dcbf_nom >= 0:
            # Already safe – no projection needed
            self._v_final, self._omega_final = v_nom, omega_nom
            self._last_cbf_state = False
            return v_nom, omega_nom

        # 3. Nominal is unsafe – apply chosen projection
        if self.use_optimization:
            v_safe, omega_safe = _optimization_projection(
                x,
                y,
                theta,
                v_nom,
                omega_nom,
                dt,
                circles,
                walls,
                self.config.robot_radius,
                self.cbf_safety_margin,
                self.cbf_gamma,
                self.config.min_linear_vel,
                self.config.max_linear_vel,
                self.config.max_angular_vel,
            )
        elif self.fallback_mode == "analytical":
            v_safe, omega_safe = _analytical_projection_continuous(
                x,
                y,
                theta,
                v_nom,
                omega_nom,
                dt,
                circles,
                walls,
                self.config.robot_radius,
                self.cbf_safety_margin,
                self.cbf_gamma,
                self.config.min_linear_vel,
                self.config.max_linear_vel,
                self.config.max_angular_vel,
            )
        else:
            v_safe, omega_safe = cbf_safe_backup_control(
                x,
                y,
                theta,
                v_nom,
                omega_nom,
                dt,
                circles,
                walls,
                self.config.robot_radius,
                self.cbf_safety_margin,
                self.cbf_gamma,
                self.config.min_linear_vel,
                self.config.max_linear_vel,
                self.config.max_angular_vel,
            )

        # 4. Log real intervention
        self._cbf_interventions += 1
        method = (
            "OPT"
            if self.use_optimization
            else ("ANALYTICAL" if self.fallback_mode == "analytical" else "GRID")
        )
        print(
            f"  [CBF-{method}] Intervention #{self._cbf_interventions}: "
            f"({v_nom:.2f},{omega_nom:.2f}) → ({v_safe:.2f},{omega_safe:.2f}) "
            f"at robot=({x:.1f},{y:.1f}), min DCBF = {min_dcbf_nom:.3f}"
        )

        self._v_final, self._omega_final = v_safe, omega_safe
        return v_safe, omega_safe
