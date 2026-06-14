import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple
from ..config import CONFIG

class UserTrajectory:
    def __init__(self, trajectory_data: np.ndarray, dt: float = 0.1, data_format: str = 'time_xy'):
        self.raw_data = trajectory_data
        self.dt = dt
        self.data_format = data_format

        if data_format == 'time_xy':
            self.times = trajectory_data[:, 0]
            self.positions_x = trajectory_data[:, 1]
            self.positions_y = trajectory_data[:, 2]
            self.times = self.times - self.times[0]
        elif data_format == 'state':
            n_points = trajectory_data.shape[0]
            self.times = np.arange(n_points) * dt
            self.positions_x = trajectory_data[:, 0]
            self.positions_y = trajectory_data[:, 1]
            self.yaw = trajectory_data[:, 2]
            if trajectory_data.shape[1] >= 5:
                self.v_forw = trajectory_data[:, 3]
                self.v_orth = trajectory_data[:, 4]
        else:
            raise ValueError(f"Unknown data_format: {data_format}")

        self.interp_x = interp1d(self.times, self.positions_x, kind='linear',
                                 bounds_error=False, fill_value=(self.positions_x[0], self.positions_x[-1]))
        self.interp_y = interp1d(self.times, self.positions_y, kind='linear',
                                 bounds_error=False, fill_value=(self.positions_y[0], self.positions_y[-1]))
        self.total_duration = self.times[-1]
        self.current_idx = 0
        self.radius = CONFIG.pedestrian_radius
        self.safety_radius = CONFIG.safety_margin
        self._compute_average_velocity()
        self.n_trajectory_points = len(self.positions_x)

    def _compute_average_velocity(self):
        if self.data_format == 'state' and hasattr(self, 'v_forw'):
            total_speeds = np.sqrt(self.v_forw**2 + self.v_orth**2)
            self.avg_velocity = float(np.mean(total_speeds[total_speeds > 0.1]))
            self.max_velocity = float(np.percentile(total_speeds, 95))
        else:
            dx = np.diff(self.positions_x)
            dy = np.diff(self.positions_y)
            speeds = np.sqrt(dx**2 + dy**2) / self.dt
            self.avg_velocity = float(np.mean(speeds[speeds > 0.1]))
            self.max_velocity = float(np.percentile(speeds, 95))
        print(f"  User velocity: avg={self.avg_velocity:.2f} m/s, max={self.max_velocity:.2f} m/s")

    def get_position_at_time(self, t: float) -> Tuple[float, float]:
        return float(self.interp_x(t)), float(self.interp_y(t))

    def get_position_at_index(self, idx: int) -> Tuple[float, float]:
        idx = max(0, min(idx, len(self.positions_x) - 1))
        return float(self.positions_x[idx]), float(self.positions_y[idx])

    def get_random_trajectory_point(self) -> Tuple[float, float, int]:
        idx = np.random.randint(0, self.n_trajectory_points)
        return float(self.positions_x[idx]), float(self.positions_y[idx]), idx

    def get_direction_at_index(self, idx: int) -> Tuple[float, float]:
        idx = max(0, min(idx, self.n_trajectory_points - 2))
        dx = self.positions_x[idx + 1] - self.positions_x[idx]
        dy = self.positions_y[idx + 1] - self.positions_y[idx]
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1e-6:
            return 1.0, 0.0
        return dx / mag, dy / mag

    def get_perpendicular_at_index(self, idx: int) -> Tuple[float, float]:
        dx, dy = self.get_direction_at_index(idx)
        return -dy, dx

    def get_direction_at_time(self, t: float, lookahead: float = 0.1) -> Tuple[float, float]:
        x1, y1 = self.get_position_at_time(t)
        x2, y2 = self.get_position_at_time(t + lookahead)
        dx = x2 - x1
        dy = y2 - y1
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1e-6:
            return 1.0, 0.0
        return dx / mag, dy / mag

    def get_perpendicular_at_time(self, t: float) -> Tuple[float, float]:
        dx, dy = self.get_direction_at_time(t)
        return -dy, dx

    def get_velocity_at_time(self, t: float, lookahead: float = 0.1) -> float:
        x1, y1 = self.get_position_at_time(t)
        x2, y2 = self.get_position_at_time(t + lookahead)
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist / lookahead

    def get_trajectory_segment(self, t_start: float, t_end: float, n_points: int = 50) -> np.ndarray:
        times = np.linspace(t_start, t_end, n_points)
        points = np.array([[self.interp_x(t), self.interp_y(t)] for t in times])
        return points

    def get_full_trajectory(self) -> np.ndarray:
        return np.column_stack([self.positions_x, self.positions_y])

    def get_state_at_time(self, t: float) -> np.ndarray:
        from scipy.interpolate import interp1d
        x, y = self.get_position_at_time(t)
        if self.data_format == 'state' and hasattr(self, 'yaw'):
            if not hasattr(self, '_interp_yaw'):
                self._interp_yaw = interp1d(self.times, self.yaw, kind='linear',
                                            bounds_error=False, fill_value=(self.yaw[0], self.yaw[-1]))
                self._interp_v_forw = interp1d(self.times, self.v_forw, kind='linear',
                                               bounds_error=False, fill_value=(self.v_forw[0], self.v_forw[-1]))
                self._interp_v_orth = interp1d(self.times, self.v_orth, kind='linear',
                                               bounds_error=False, fill_value=(self.v_orth[0], self.v_orth[-1]))
            yaw = float(self._interp_yaw(t))
            v_forw = float(self._interp_v_forw(t))
            v_orth = float(self._interp_v_orth(t))
        else:
            dx, dy = self.get_direction_at_time(t)
            yaw = np.arctan2(dy, dx)
            v_forw = self.get_velocity_at_time(t)
            v_orth = 0.0
        return np.array([x, y, yaw, v_forw, v_orth])

    def check_collision_with_point(self, x: float, y: float, radius: float,
                                    safety_margin: float = 0.5) -> bool:
        min_dist = radius + self.radius + safety_margin
        distances = np.sqrt((self.positions_x - x)**2 + (self.positions_y - y)**2)
        return np.any(distances < min_dist)


def create_user_trajectory_from_processed_data(x_true: np.ndarray, avg_dt: float,
                                                 env_width: float = 20.0,
                                                 env_height: float = 20.0) -> UserTrajectory:
    data = x_true.copy()
    pos_x = data[:, 0]
    pos_y = data[:, 1]
    pos_x_centered = pos_x - pos_x.mean() + env_width / 2
    pos_y_centered = pos_y - pos_y.mean() + env_height / 2
    data[:, 0] = pos_x_centered
    data[:, 1] = pos_y_centered
    return UserTrajectory(data, dt=avg_dt, data_format='state')


def load_user_trajectory_from_csv(csv_path: str, scale_factor: float = 0.001,
                                   env_width: float = 20.0,
                                   env_height: float = 20.0) -> UserTrajectory:
    import pandas as pd
    df = pd.read_csv(csv_path)
    times = df['time'].values
    pos_x = df['pos_x'].values * scale_factor
    pos_y = df['pos_y'].values * scale_factor
    pos_x = pos_x - pos_x.mean() + env_width / 2
    pos_y = pos_y - pos_y.mean() + env_height / 2
    trajectory_data = np.column_stack([times, pos_x, pos_y])
    return UserTrajectory(trajectory_data, data_format='time_xy')