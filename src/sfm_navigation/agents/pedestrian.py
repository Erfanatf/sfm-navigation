import numpy as np
from typing import List, Tuple, Optional
from ..config import CONFIG
from ..sfm.numba_utils import euclidean_distance, normalize_angle
from ..data.moods import PedestrianMood, MOOD_PARAMETERS, CUSTOM_MOODS


class Pedestrian:
    def __init__(
        self,
        x: float,
        y: float,
        mood: PedestrianMood = PedestrianMood.NORMAL,
        goal_x: float = None,
        goal_y: float = None,
        pedestrian_id: int = 0,
        base_speed: float = 1.2,
    ):
        self.x = x
        self.y = y

        # ---------- Fetch parameters for this mood ----------
        if isinstance(mood, str):  # custom mood (string name)
            params = CUSTOM_MOODS.get(mood)
            if params is None:
                raise KeyError(f"Unknown custom mood '{mood}'. Did you register it?")
        else:  # standard enum mood
            params = MOOD_PARAMETERS[mood]

        self.mood = mood  # store the mood as-is
        self.pedestrian_id = pedestrian_id
        self.radius = CONFIG.pedestrian_radius

        # ---------- Mood‑driven parameters (extended SFM) ----------
        self.speed_factor = params["speed_factor"]
        self.direction_variance = params["direction_variance"]
        self.personal_space = params["personal_space"]
        self.reactivity = params["reactivity"]

        self.base_speed = base_speed
        self.speed = base_speed * self.speed_factor
        self.vx = 0.0
        self.vy = 0.0
        self.theta = np.random.uniform(-np.pi, np.pi)

        self.goal_x = goal_x if goal_x is not None else np.random.uniform(2, 18)
        self.goal_y = goal_y if goal_y is not None else np.random.uniform(2, 18)

        self.time = 0.0
        self.juggle_phase = 0.0
        self.adversary_timer = 0.0
        self.adversary_engaged = False
        self.robot_position = None

        self.trajectory_history: List[Tuple[float, float]] = [(x, y)]
        self.velocity_history: List[Tuple[float, float]] = [(0.0, 0.0)]

        # Extended SFM parameters (calibrated or default)
        self.v0 = params.get("v0", self.base_speed * self.speed_factor)
        self.tau = params.get("tau", 0.5)
        self.A_ped = params.get("A_ped", 3.0)
        self.B_ped = params.get("B_ped", 0.5)
        self.lam_base = params.get("lam_base", 0.5)
        self.phi_fov = params.get("phi_fov", np.deg2rad(90))
        self.kappa = params.get("kappa", 0.0)
        self.k_group = params.get("k_group", 0.0)
        self.r_group = params.get("r_group", 0.5)
        self.theta_gaze = params.get("theta_gaze", 0.0)
        self.w_att = params.get("w_att", 0.0)
        self.fov_att = params.get("fov_att", np.deg2rad(30))

    def set_mood(self, new_mood):
        """Change the pedestrian's mood and reload all SFM parameters."""
        if isinstance(new_mood, str):
            # Try to find it as a standard enum first
            try:
                enum_mood = PedestrianMood[new_mood]
                params = MOOD_PARAMETERS[enum_mood]
                self.mood = enum_mood
            except KeyError:
                # Not a standard enum; must be a custom mood
                params = CUSTOM_MOODS.get(new_mood)
                if params is None:
                    raise KeyError(f"Unknown mood '{new_mood}'")
                self.mood = new_mood
        else:
            params = MOOD_PARAMETERS[new_mood]
            self.mood = new_mood

        # Reload parameters (same as before)
        self.speed_factor = params["speed_factor"]
        self.direction_variance = params["direction_variance"]
        self.personal_space = params["personal_space"]
        self.reactivity = params["reactivity"]
        self.speed = self.base_speed * self.speed_factor

        self.v0 = params.get("v0", self.base_speed * self.speed_factor)
        self.tau = params.get("tau", 0.5)
        self.A_ped = params.get("A_ped", 3.0)
        self.B_ped = params.get("B_ped", 0.5)
        self.lam_base = params.get("lam_base", 0.5)
        self.phi_fov = params.get("phi_fov", np.deg2rad(90))
        self.kappa = params.get("kappa", 0.0)
        self.k_group = params.get("k_group", 0.0)
        self.r_group = params.get("r_group", 0.5)
        self.theta_gaze = params.get("theta_gaze", 0.0)
        self.w_att = params.get("w_att", 0.0)
        self.fov_att = params.get("fov_att", np.deg2rad(30))

    def update(
        self,
        dt: float,
        robot_pos: Tuple[float, float],
        other_pedestrians: List["Pedestrian"],
        env_width: float,
        env_height: float,
        static_obstacles: List = None,
    ):
        self.time += dt
        self.robot_position = robot_pos

        if self.mood == PedestrianMood.JUGGLING:
            self._update_juggling(dt)
        elif self.mood == PedestrianMood.ADVERSARIAL:
            self._update_adversarial(dt, robot_pos, env_width, env_height)
        else:
            self._update_standard(
                dt,
                robot_pos,
                other_pedestrians,
                env_width,
                env_height,
                static_obstacles,
            )

        self.trajectory_history.append((self.x, self.y))
        self.velocity_history.append((self.vx, self.vy))

    def _update_juggling(self, dt: float):
        self.juggle_phase += dt * 3.0
        offset_x = 0.1 * np.sin(self.juggle_phase)
        offset_y = 0.05 * np.cos(self.juggle_phase * 2)
        initial_x = self.trajectory_history[0][0]
        initial_y = self.trajectory_history[0][1]
        self.x = initial_x + offset_x
        self.y = initial_y + offset_y
        self.vx = 0.1 * 3.0 * np.cos(self.juggle_phase)
        self.vy = -0.05 * 2 * 3.0 * np.sin(self.juggle_phase * 2)

    def _update_adversarial(
        self,
        dt: float,
        robot_pos: Tuple[float, float],
        env_width: float,
        env_height: float,
    ):
        dist_to_robot = euclidean_distance(self.x, self.y, robot_pos[0], robot_pos[1])
        if dist_to_robot < 5.0 and not self.adversary_engaged:
            self.adversary_engaged = True
            self.adversary_timer = 0.0
        if self.adversary_engaged:
            self.adversary_timer += dt
            if self.adversary_timer < 3.0:
                angle_to_robot = np.arctan2(
                    robot_pos[1] - self.y, robot_pos[0] - self.x
                )
                self.theta = angle_to_robot
                self.speed = self.base_speed * 1.2
                if dist_to_robot < 1.5:
                    self.speed = 0.0
            else:
                angle_to_robot = np.arctan2(
                    robot_pos[1] - self.y, robot_pos[0] - self.x
                )
                self.theta = angle_to_robot + np.pi / 2
                self.speed = self.base_speed * 0.8
                if self.adversary_timer > 5.0:
                    self.adversary_engaged = False
        else:
            self._update_goal_directed(dt, env_width, env_height)

        self.vx = self.speed * np.cos(self.theta)
        self.vy = self.speed * np.sin(self.theta)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self._handle_boundaries(env_width, env_height)

    def _update_standard(
        self,
        dt: float,
        robot_pos: Tuple[float, float],
        other_pedestrians: List["Pedestrian"],
        env_width: float,
        env_height: float,
        static_obstacles: List = None,
    ):
        # ----- Goal direction and desired velocity -----
        angle_to_goal = np.arctan2(self.goal_y - self.y, self.goal_x - self.x)
        noise = np.random.normal(0, self.direction_variance)
        target_theta = angle_to_goal + noise

        max_turn_rate = 1.5
        theta_diff = normalize_angle(target_theta - self.theta)
        theta_diff = np.clip(theta_diff, -max_turn_rate * dt, max_turn_rate * dt)
        self.theta = normalize_angle(self.theta + theta_diff)

        v_des = self.v0 * self.speed_factor
        desired_vx = v_des * np.cos(self.theta)
        desired_vy = v_des * np.sin(self.theta)

        ax = (desired_vx - self.vx) / self.tau
        ay = (desired_vy - self.vy) / self.tau

        # Curvature bias
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > 0.2:
            perp_x = -np.sin(self.theta)
            perp_y = np.cos(self.theta)
            lateral_acc = self.kappa * speed**2
            ax += lateral_acc * perp_x
            ay += lateral_acc * perp_y

        # Repulsive forces from robot (user)
        if robot_pos is not None:
            dist = np.sqrt((self.x - robot_pos[0]) ** 2 + (self.y - robot_pos[1]) ** 2)
            robot_radius = 0.3
            combined_radius = self.radius + robot_radius + CONFIG.safety_margin
            if dist < combined_radius + 2.0 and dist > 1e-6:
                dir_x = (self.x - robot_pos[0]) / dist
                dir_y = (self.y - robot_pos[1]) / dist
                angle_to_robot = np.arctan2(
                    robot_pos[1] - self.y, robot_pos[0] - self.x
                )
                phi = angle_to_robot - self.theta
                w = self.lam_base + (1 - self.lam_base) * (1 + np.cos(phi)) / 2.0
                force_mag = (
                    2.0
                    * self.A_ped
                    * np.exp((combined_radius - dist) / (self.B_ped + 2.0))
                    * w
                )
                ax += force_mag * dir_x
                ay += force_mag * dir_y

        # Repulsive forces from other pedestrians
        for other in other_pedestrians:
            if other.pedestrian_id == self.pedestrian_id:
                continue
            dist = np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
            combined_radius = self.radius + other.radius
            if dist < 1e-6:
                dist = 1e-6
            if dist < combined_radius + 2.0:
                dir_x = (self.x - other.x) / dist
                dir_y = (self.y - other.y) / dist
                angle_to_other = np.arctan2(other.y - self.y, other.x - self.x)
                phi = angle_to_other - self.theta
                if abs(phi) > self.phi_fov:
                    continue
                w = self.lam_base + (1 - self.lam_base) * (1 + np.cos(phi)) / 2.0
                gaze_dir = self.theta + self.theta_gaze
                phi_att = angle_to_other - gaze_dir
                phi_att = (phi_att + np.pi) % (2 * np.pi) - np.pi
                if abs(phi_att) < self.fov_att:
                    w *= 1 + self.w_att
                force_mag = (
                    self.A_ped * np.exp((combined_radius - dist) / self.B_ped) * w
                )
                ax += force_mag * dir_x
                ay += force_mag * dir_y

        # Static obstacles
        if static_obstacles is not None:
            for obs in static_obstacles:
                obs_x = obs.x if hasattr(obs, "x") else obs[0]
                obs_y = obs.y if hasattr(obs, "y") else obs[1]
                obs_r = obs.radius if hasattr(obs, "radius") else obs[2]
                dist = np.sqrt((self.x - obs_x) ** 2 + (self.y - obs_y) ** 2)
                combined_radius = self.radius + obs_r
                if dist < 1e-6:
                    dist = 1e-6
                if dist < combined_radius + 1.5:
                    dir_x = (self.x - obs_x) / dist
                    dir_y = (self.y - obs_y) / dist
                    angle_to_obs = np.arctan2(obs_y - self.y, obs_x - self.x)
                    phi = angle_to_obs - self.theta
                    w = self.lam_base + (1 - self.lam_base) * (1 + np.cos(phi)) / 2.0
                    force_mag = (
                        self.A_ped * np.exp((combined_radius - dist) / self.B_ped) * w
                    )
                    ax += force_mag * dir_x
                    ay += force_mag * dir_y

        # Group cohesion
        if self.k_group > 0:
            group_x, group_y = [], []
            for other in other_pedestrians:
                if other.pedestrian_id == self.pedestrian_id:
                    continue
                dist = np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
                if dist < 3.0:
                    group_x.append(other.x)
                    group_y.append(other.y)
            if group_x:
                centroid_x = np.mean(group_x)
                centroid_y = np.mean(group_y)
                d_centroid = np.sqrt(
                    (centroid_x - self.x) ** 2 + (centroid_y - self.y) ** 2
                )
                if d_centroid > 0.01:
                    force_mag_group = self.k_group * (d_centroid - self.r_group)
                    ax += force_mag_group * (centroid_x - self.x) / d_centroid
                    ay += force_mag_group * (centroid_y - self.y) / d_centroid

        # Boundary forces
        boundary_margin = 1.0
        boundary_strength = 5.0
        if self.x < boundary_margin:
            ax += boundary_strength * (boundary_margin - self.x)
        elif self.x > env_width - boundary_margin:
            ax -= boundary_strength * (self.x - (env_width - boundary_margin))
        if self.y < boundary_margin:
            ay += boundary_strength * (boundary_margin - self.y)
        elif self.y > env_height - boundary_margin:
            ay -= boundary_strength * (self.y - (env_height - boundary_margin))

        self.vx += ax * dt
        self.vy += ay * dt

        speed = np.sqrt(self.vx**2 + self.vy**2)
        max_speed = self.v0 * 2.5
        if speed > max_speed:
            self.vx = self.vx / speed * max_speed
            self.vy = self.vy / speed * max_speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        margin = self.radius + 0.2
        self.x = np.clip(self.x, margin, env_width - margin)
        self.y = np.clip(self.y, margin, env_height - margin)

        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > 0.1:
            self.theta = np.arctan2(self.vy, self.vx)

        dist_to_goal = np.sqrt(
            (self.x - self.goal_x) ** 2 + (self.y - self.goal_y) ** 2
        )
        if dist_to_goal < 0.1:
            self.goal_x = np.random.uniform(2, env_width - 2)
            self.goal_y = np.random.uniform(2, env_height - 2)

    def _update_goal_directed(self, dt: float, env_width: float, env_height: float):
        angle_to_goal = np.arctan2(self.goal_y - self.y, self.goal_x - self.x)
        self.theta = angle_to_goal
        dist_to_goal = euclidean_distance(self.x, self.y, self.goal_x, self.goal_y)
        if dist_to_goal < 1.0:
            self.goal_x = np.random.uniform(2, env_width - 2)
            self.goal_y = np.random.uniform(2, env_height - 2)

    def _handle_boundaries(self, env_width: float, env_height: float):
        margin = self.radius + 0.5
        if self.x < margin:
            self.x = margin
            self.vx = abs(self.vx)
            self.theta = np.arctan2(self.vy, self.vx)
        elif self.x > env_width - margin:
            self.x = env_width - margin
            self.vx = -abs(self.vx)
            self.theta = np.arctan2(self.vy, self.vx)
        if self.y < margin:
            self.y = margin
            self.vy = abs(self.vy)
            self.theta = np.arctan2(self.vy, self.vx)
        elif self.y > env_height - margin:
            self.y = env_height - margin
            self.vy = -abs(self.vy)
            self.theta = np.arctan2(self.vy, self.vx)

    def get_velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)

    def predict_position(self, t: float) -> Tuple[float, float]:
        return (self.x + self.vx * t, self.y + self.vy * t)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.radius, self.vx, self.vy])
