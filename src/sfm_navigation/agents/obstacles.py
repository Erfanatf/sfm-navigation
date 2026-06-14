from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from ..sfm.numba_utils import euclidean_distance

@dataclass
class StaticObstacle:
    x: float
    y: float
    radius: float
    obstacle_id: int = 0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.radius])

    def contains_point(self, px: float, py: float) -> bool:
        return euclidean_distance(self.x, self.y, px, py) <= self.radius


class DynamicObstacle:
    def __init__(self, x: float, y: float, radius: float,
                 vx: float = 0.0, vy: float = 0.0,
                 obstacle_id: int = 0,
                 motion_type: str = 'linear'):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = vx
        self.vy = vy
        self.obstacle_id = obstacle_id
        self.motion_type = motion_type
        self.initial_x = x
        self.initial_y = y
        self.time = 0.0
        self.center_x = x
        self.center_y = y - 2.0
        self.angular_speed = 0.5
        self.trajectory_history: List[Tuple[float, float]] = [(x, y)]

    def update(self, dt: float, env_width: float, env_height: float):
        self.time += dt
        if self.motion_type == 'linear':
            new_x = self.x + self.vx * dt
            new_y = self.y + self.vy * dt
            if new_x <= self.radius or new_x >= env_width - self.radius:
                self.vx *= -1
                new_x = np.clip(new_x, self.radius, env_width - self.radius)
            if new_y <= self.radius or new_y >= env_height - self.radius:
                self.vy *= -1
                new_y = np.clip(new_y, self.radius, env_height - self.radius)
            self.x = new_x
            self.y = new_y
        elif self.motion_type == 'circular':
            angle = self.angular_speed * self.time
            radius = 2.0
            self.x = self.center_x + radius * np.cos(angle)
            self.y = self.center_y + radius * np.sin(angle)
            self.vx = -radius * self.angular_speed * np.sin(angle)
            self.vy = radius * self.angular_speed * np.cos(angle)
        elif self.motion_type == 'sinusoidal':
            self.x = self.initial_x + self.vx * self.time
            self.y = self.initial_y + 2.0 * np.sin(0.5 * self.time)
            self.vy = 2.0 * 0.5 * np.cos(0.5 * self.time)
            if self.x <= self.radius or self.x >= env_width - self.radius:
                self.vx *= -1
                self.initial_x = self.x
                self.time = 0
        self.trajectory_history.append((self.x, self.y))

    def get_velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)

    def predict_position(self, t: float) -> Tuple[float, float]:
        if self.motion_type == 'linear':
            return (self.x + self.vx * t, self.y + self.vy * t)
        elif self.motion_type == 'circular':
            angle = self.angular_speed * (self.time + t)
            radius = 2.0
            return (
                self.center_x + radius * np.cos(angle),
                self.center_y + radius * np.sin(angle)
            )
        else:
            return (self.x + self.vx * t, self.y + self.vy * t)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.radius, self.vx, self.vy])