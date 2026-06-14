from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from ..config import SimulationConfig
from ..sfm.numba_utils import euclidean_distance, normalize_angle
from ..utils.derivative_kf import DerivativeEstimatorKF

@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.omega])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RobotState":
        return cls(x=arr[0], y=arr[1], theta=arr[2], v=arr[3], omega=arr[4])


class DifferentialDriveRobot:
    def __init__(
        self,
        config: SimulationConfig,
        start_x: float = 1.0,
        start_y: float = 1.0,
        start_theta: float = 0.0,
        tau: float = 0.3,
    ):
        self.config = config
        self.tau = tau                     # motor time constant (seconds)
        self.state = RobotState(x=start_x, y=start_y, theta=start_theta)
        self.trajectory_history: List[Tuple[float, float, float]] = [
            (start_x, start_y, start_theta)
        ]
        self.velocity_history: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.command_history: List[Tuple[float, float]] = []
        self.accel_history: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.jerk_history: List[Tuple[float, float]] = [(0.0, 0.0)]
        self._prev_accel = np.zeros(2)     # previous step acceleration (for jerk calc)
        self._prev_v_dot = 0.0             # (optional, kept for debugging)
        self._prev_omega_dot = 0.0
        self.goal = None
        self.goal_reached = False

        self.accel_filter = DerivativeEstimatorKF(dt=config.dt)
        self.omega_accel_filter = DerivativeEstimatorKF(dt=config.dt)

    def set_goal(self, goal_x: float, goal_y: float):
        self.goal = (goal_x, goal_y)
        self.goal_reached = False

    def update(
        self, v_cmd: float, omega_cmd: float, dt: float, d_ext: np.ndarray = None
    ) -> RobotState:
        if d_ext is None:
            d_ext = np.zeros(2)

        # ---- 1. Desired acceleration from motor model + disturbance ----
        v_dot_cmd = (v_cmd - self.state.v) / self.tau
        omega_dot_cmd = (omega_cmd - self.state.omega) / self.tau

        v_dot = v_dot_cmd + d_ext[0]
        omega_dot = omega_dot_cmd + d_ext[1]

        # ---- 2. Absolute acceleration limits ----
        max_lin_acc = self.config.max_linear_accel
        max_ang_acc = self.config.max_angular_accel
        v_dot = np.clip(v_dot, -max_lin_acc, max_lin_acc)
        omega_dot = np.clip(omega_dot, -max_ang_acc, max_ang_acc)

        # ---- 3. Integrate to tentative velocity ----
        v_new = self.state.v + v_dot * dt
        omega_new = self.state.omega + omega_dot * dt

        # ---- 4. Absolute velocity limits ----
        v_new = np.clip(v_new, self.config.min_linear_vel, self.config.max_linear_vel)
        omega_new = np.clip(omega_new, -self.config.max_angular_vel, self.config.max_angular_vel)

        # ---- 5. Velocity change limits (one‑step reachability) ----
        v_new = np.clip(v_new,
                        self.state.v - max_lin_acc * self.tau,
                        self.state.v + max_lin_acc * self.tau)
        omega_new = np.clip(omega_new,
                            self.state.omega - max_ang_acc * self.tau,
                            self.state.omega + max_ang_acc * self.tau)
        
        # 6 to 10 replacement for KF

        # Ensure Kalman filters use the actual simulation dt
        self.accel_filter.set_dt(dt)
        self.omega_accel_filter.set_dt(dt)

        # ---- Kalman‑filtered acceleration and jerk (replaces finite‑difference) ----
        v_smooth, acc_smooth, jerk_smooth = self.accel_filter.update(v_new)
        omega_smooth, omega_acc_smooth, omega_jerk_smooth = self.omega_accel_filter.update(omega_new)

        # Clamp filter outputs to physical limits
        lin_acc_max = self.config.max_linear_accel
        ang_acc_max = self.config.max_angular_accel
        lin_jerk_max = self.config.max_linear_jerk
        ang_jerk_max = self.config.max_angular_jerk

        acc_smooth = np.clip(acc_smooth, -lin_acc_max, lin_acc_max)
        omega_acc_smooth = np.clip(omega_acc_smooth, -ang_acc_max, ang_acc_max)
        jerk_smooth = np.clip(jerk_smooth, -lin_jerk_max, lin_jerk_max)
        omega_jerk_smooth = np.clip(omega_jerk_smooth, -ang_jerk_max, ang_jerk_max)

        # Store the smooth values
        self.accel_history.append((acc_smooth, omega_acc_smooth))
        self.jerk_history.append((jerk_smooth, omega_jerk_smooth))
        self._prev_accel = np.array([acc_smooth, omega_acc_smooth])   # keep for compatibility if needed elsewhere

        # ---- 11. Kinematic update ----
        if abs(omega_new) < 1e-6:
            new_x = self.state.x + v_new * np.cos(self.state.theta) * dt
            new_y = self.state.y + v_new * np.sin(self.state.theta) * dt
            new_theta = self.state.theta
        else:
            new_x = self.state.x + (v_new / omega_new) * (
                np.sin(self.state.theta + omega_new * dt) - np.sin(self.state.theta)
            )
            new_y = self.state.y + (v_new / omega_new) * (
                np.cos(self.state.theta) - np.cos(self.state.theta + omega_new * dt)
            )
            new_theta = normalize_angle(self.state.theta + omega_new * dt)

        # Update state
        self.state.x = new_x
        self.state.y = new_y
        self.state.theta = new_theta
        self.state.v = v_new
        self.state.omega = omega_new

        self.trajectory_history.append((new_x, new_y, new_theta))
        self.velocity_history.append((v_new, omega_new))
        self.command_history.append((v_cmd, omega_cmd))

        if self.goal is not None:
            dist_to_goal = euclidean_distance(new_x, new_y, self.goal[0], self.goal[1])
            if dist_to_goal < self.config.goal_tolerance:
                self.goal_reached = True

        return self.state

    def get_trajectory_array(self) -> np.ndarray:
        return np.array(self.trajectory_history)

    def reset(self, x: float = 1.0, y: float = 1.0, theta: float = 0.0):
        self.state = RobotState(x=x, y=y, theta=theta)
        self.trajectory_history = [(x, y, theta)]
        self.velocity_history = [(0.0, 0.0)]
        self.command_history = []
        self.accel_history = [(0.0, 0.0)]
        self.jerk_history = [(0.0, 0.0)]
        self._prev_accel = np.zeros(2)
        self.goal_reached = False
        self.accel_filter.reset()
        self.omega_accel_filter.reset()

    def distance_to_goal(self) -> float:
        if self.goal is None:
            return float("inf")
        return euclidean_distance(
            self.state.x, self.state.y, self.goal[0], self.goal[1]
        )
