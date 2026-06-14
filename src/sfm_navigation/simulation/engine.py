from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict, Optional
import time as time_module
from ..config import SimulationConfig
from ..agents.robot import DifferentialDriveRobot
from ..agents.obstacles import StaticObstacle, DynamicObstacle
from ..agents.pedestrian import Pedestrian
from ..sfm.numba_utils import euclidean_distance
from ..data.moods import PedestrianMood

@dataclass
class SimulationResult:
    success: bool
    total_time: float
    path_length: float
    robot_trajectory: np.ndarray
    robot_velocities: np.ndarray
    obstacle_trajectories: Dict[int, np.ndarray]
    pedestrian_trajectories: Dict[int, np.ndarray]
    min_distances: List[float]
    collision_count: int
    computation_times: List[float]


class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.robot = DifferentialDriveRobot(config)
        self.static_obstacles: List[StaticObstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.pedestrians: List[Pedestrian] = []
        self.controller = None
        self.time = 0.0
        self.step_count = 0
        self.running = False
        self.min_distances: List[float] = []
        self.computation_times: List[float] = []

    def reset(self, start: Tuple[float, float] = (1.0, 1.0),
              goal: Tuple[float, float] = (18.0, 18.0),
              start_theta: float = 0.0):
        self.robot.reset(start[0], start[1], start_theta)
        self.robot.set_goal(goal[0], goal[1])
        self.time = 0.0
        self.step_count = 0
        self.min_distances = []
        self.computation_times = []
        for obs in self.dynamic_obstacles:
            obs.x = obs.initial_x
            obs.y = obs.initial_y
            obs.time = 0.0
            obs.trajectory_history = [(obs.x, obs.y)]
        for ped in self.pedestrians:
            ped.x = ped.trajectory_history[0][0]
            ped.y = ped.trajectory_history[0][1]
            ped.time = 0.0
            ped.trajectory_history = [(ped.x, ped.y)]
            ped.velocity_history = [(0.0, 0.0)]

    def add_static_obstacle(self, x: float, y: float, radius: float):
        obs = StaticObstacle(x=x, y=y, radius=radius,
                            obstacle_id=len(self.static_obstacles))
        self.static_obstacles.append(obs)

    def add_dynamic_obstacle(self, x: float, y: float, radius: float,
                            vx: float, vy: float, motion_type: str = 'linear'):
        obs = DynamicObstacle(x=x, y=y, radius=radius, vx=vx, vy=vy,
                             obstacle_id=len(self.dynamic_obstacles),
                             motion_type=motion_type)
        self.dynamic_obstacles.append(obs)

    def add_pedestrian(self, x: float, y: float,
                       mood: PedestrianMood = PedestrianMood.NORMAL,
                       goal_x: float = None, goal_y: float = None):
        ped = Pedestrian(x=x, y=y, mood=mood, goal_x=goal_x, goal_y=goal_y,
                        pedestrian_id=len(self.pedestrians))
        self.pedestrians.append(ped)

    def set_controller(self, controller):
        self.controller = controller

    def get_all_obstacles_array(self) -> np.ndarray:
        obstacles = []
        for obs in self.static_obstacles:
            obstacles.append([obs.x, obs.y, obs.radius, 0.0, 0.0])
        for obs in self.dynamic_obstacles:
            obstacles.append([obs.x, obs.y, obs.radius, obs.vx, obs.vy])
        for ped in self.pedestrians:
            obstacles.append([ped.x, ped.y, ped.radius, ped.vx, ped.vy])
        if len(obstacles) == 0:
            return np.zeros((0, 5))
        return np.array(obstacles)

    def compute_min_distance(self) -> float:
        min_dist = float('inf')
        rx, ry = self.robot.state.x, self.robot.state.y
        for obs in self.static_obstacles:
            dist = euclidean_distance(rx, ry, obs.x, obs.y) - obs.radius - self.config.robot_radius
            min_dist = min(min_dist, dist)
        for obs in self.dynamic_obstacles:
            dist = euclidean_distance(rx, ry, obs.x, obs.y) - obs.radius - self.config.robot_radius
            min_dist = min(min_dist, dist)
        for ped in self.pedestrians:
            dist = euclidean_distance(rx, ry, ped.x, ped.y) - ped.radius - self.config.robot_radius
            min_dist = min(min_dist, dist)
        return min_dist

    def step(self) -> bool:
        if self.robot.goal_reached:
            return False
        if self.time >= self.config.max_simulation_time:
            return False
        for obs in self.dynamic_obstacles:
            obs.update(self.config.dt, self.config.env_width, self.config.env_height)
        robot_pos = (self.robot.state.x, self.robot.state.y)
        for ped in self.pedestrians:
            ped.update(self.config.dt, robot_pos, self.pedestrians,
                      self.config.env_width, self.config.env_height,
                      self.static_obstacles)
        obstacles = self.get_all_obstacles_array()
        if self.controller is not None:
            start_time = time_module.perf_counter()
            v_cmd, omega_cmd = self.controller.compute_velocity(
                self.robot.state, self.robot.goal, obstacles
            )
            comp_time = time_module.perf_counter() - start_time
            self.computation_times.append(comp_time)
        else:
            v_cmd, omega_cmd = 0.0, 0.0
        self.robot.update(v_cmd, omega_cmd, self.config.dt)
        min_dist = self.compute_min_distance()
        self.min_distances.append(min_dist)
        self.time += self.config.dt
        self.step_count += 1
        return True

    def run(self, start: Tuple[float, float] = (1.0, 1.0),
            goal: Tuple[float, float] = (18.0, 18.0),
            start_theta: float = 0.0,
            verbose: bool = True) -> SimulationResult:
        self.reset(start, goal, start_theta)
        self.running = True
        if verbose:
            print(f"Starting simulation: {start} → {goal}")
        while self.running:
            if not self.step():
                self.running = False
            if verbose and self.step_count % 100 == 0:
                dist = self.robot.distance_to_goal()
                print(f"  Step {self.step_count}: distance to goal = {dist:.2f}m")
        robot_traj = self.robot.get_trajectory_array()
        path_length = 0.0
        for i in range(1, len(robot_traj)):
            path_length += euclidean_distance(
                robot_traj[i-1, 0], robot_traj[i-1, 1],
                robot_traj[i, 0], robot_traj[i, 1]
            )
        collision_count = sum(1 for d in self.min_distances if d < 0)
        obs_trajs = {}
        for obs in self.dynamic_obstacles:
            obs_trajs[obs.obstacle_id] = np.array(obs.trajectory_history)
        ped_trajs = {}
        for ped in self.pedestrians:
            ped_trajs[ped.pedestrian_id] = np.array(ped.trajectory_history)
        result = SimulationResult(
            success=self.robot.goal_reached,
            total_time=self.time,
            path_length=path_length,
            robot_trajectory=robot_traj,
            robot_velocities=np.array(self.robot.velocity_history),
            obstacle_trajectories=obs_trajs,
            pedestrian_trajectories=ped_trajs,
            min_distances=self.min_distances,
            collision_count=collision_count,
            computation_times=self.computation_times
        )
        if verbose:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"Simulation {status}: time={result.total_time:.1f}s, "
                  f"path={result.path_length:.2f}m, collisions={result.collision_count}")
        return result