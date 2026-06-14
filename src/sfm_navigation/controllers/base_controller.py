"""Base controller interface."""
import time

class BaseController:
    """Abstract base for all controllers."""
    def compute_velocity(self, robot_state, goal_pos, obstacles):
        raise NotImplementedError

    def get_real_time_factor(self, desired_period: float) -> float:
        if desired_period <= 0:
            return 0.0
        return self.last_compute_time / desired_period
    
