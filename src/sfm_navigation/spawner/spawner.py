import numpy as np
from typing import List, Tuple, Optional, Dict
from ..agents.obstacles import StaticObstacle
from ..agents.pedestrian import Pedestrian
from ..agents.user import UserTrajectory
from ..config import SimulationConfig, CONFIG
from ..data.moods import PedestrianMood, MOOD_PARAMETERS, CUSTOM_MOODS


def load_transition_matrix(csv_path: str):
    """
    Read a CSV with columns from,to,prob and optionally rate.
    Returns (transition_dict, rates_dict).
    rates_dict maps mood_name -> lambda (per second), or None if no rate column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Build transition matrix
    matrix = {}
    for _, row in df.iterrows():
        frm = row["from"]
        to = row["to"]
        prob = row["prob"]
        matrix.setdefault(frm, {})[to] = prob
    # Normalize each row to sum to 1 (in case of rounding)
    for frm in matrix:
        total = sum(matrix[frm].values())
        if total > 0:
            for to in matrix[frm]:
                matrix[frm][to] /= total

    # Build per‑mood rates if column exists
    rates = None
    if "rate" in df.columns:
        rates = {}
        for frm, grp in df.groupby("from"):
            rate_val = grp["rate"].iloc[0]  # same for all rows of this mood
            rates[frm] = rate_val
    return matrix, rates


def generate_safe_static_obstacles(
    user_trajectory: UserTrajectory,
    n_obstacles: int = 3,
    min_radius: float = 0.3,
    max_radius: float = 0.8,
    env_width: float = 20.0,
    env_height: float = 20.0,
    safety_margin: float = 1.5,
) -> List[StaticObstacle]:
    obstacles = []
    max_attempts_per_obstacle = 100
    margin = 1.5
    n_cols = int(np.ceil(np.sqrt(n_obstacles * 2)))
    n_rows = int(np.ceil(n_obstacles / n_cols)) + 1
    cell_width = (env_width - 2 * margin) / n_cols
    cell_height = (env_height - 2 * margin) / n_rows
    available_cells = [(i, j) for i in range(n_cols) for j in range(n_rows)]
    np.random.shuffle(available_cells)
    obs_id = 0
    cell_idx = 0
    while obs_id < n_obstacles and cell_idx < len(available_cells):
        cell_i, cell_j = available_cells[cell_idx]
        cell_idx += 1
        placed = False
        for attempt in range(max_attempts_per_obstacle // 10):
            x = (
                margin
                + cell_i * cell_width
                + np.random.uniform(0.2 * cell_width, 0.8 * cell_width)
            )
            y = (
                margin
                + cell_j * cell_height
                + np.random.uniform(0.2 * cell_height, 0.8 * cell_height)
            )
            x = np.clip(x, margin, env_width - margin)
            y = np.clip(y, margin, env_height - margin)
            radius = np.random.uniform(min_radius, max_radius)
            if user_trajectory.check_collision_with_point(x, y, radius, safety_margin):
                continue
            collision_with_existing = False
            for existing_obs in obstacles:
                dist = np.sqrt((x - existing_obs.x) ** 2 + (y - existing_obs.y) ** 2)
                if dist < (radius + existing_obs.radius + 0.5):
                    collision_with_existing = True
                    break
            if collision_with_existing:
                continue
            obstacles.append(
                StaticObstacle(x=x, y=y, radius=radius, obstacle_id=obs_id)
            )
            obs_id += 1
            placed = True
            break
    pure_random_attempts = 0
    while obs_id < n_obstacles and pure_random_attempts < max_attempts_per_obstacle * 3:
        pure_random_attempts += 1
        x = np.random.uniform(margin, env_width - margin)
        y = np.random.uniform(margin, env_height - margin)
        radius = np.random.uniform(min_radius, max_radius)
        if user_trajectory.check_collision_with_point(x, y, radius, safety_margin):
            continue
        collision_with_existing = False
        for existing_obs in obstacles:
            dist = np.sqrt((x - existing_obs.x) ** 2 + (y - existing_obs.y) ** 2)
            if dist < (radius + existing_obs.radius + 0.5):
                collision_with_existing = True
                break
        if not collision_with_existing:
            obstacles.append(
                StaticObstacle(x=x, y=y, radius=radius, obstacle_id=obs_id)
            )
            obs_id += 1
    print(
        f"Generated {len(obstacles)}/{n_obstacles} safe static obstacles (uniformly distributed)"
    )
    return obstacles


class SFMPedestrianSpawner:
    def __init__(
        self,
        user_trajectory: UserTrajectory,
        config: SimulationConfig,
        n_pedestrians: int = 5,
        vicinity_radius: float = 8.0,
        respawn_distance: float = 12.0,
        min_spawn_distance: float = 2.0,
        max_spawn_distance: float = 6.0,
        speed_scale_factor: float = 1.0,
        mood_switch_rate: float = 0.0,  # per second
        mood_switch_rates: dict = None,  # per second per mood
        mood_transition_matrix: dict = None,
    ):

        self.user_trajectory = user_trajectory
        self.config = config
        self.n_pedestrians = n_pedestrians
        self.vicinity_radius = vicinity_radius
        self.respawn_distance = respawn_distance
        self.min_spawn_distance = min_spawn_distance
        self.max_spawn_distance = max_spawn_distance
        self.user_avg_speed = user_trajectory.avg_velocity
        self.pedestrian_base_speed = self.user_avg_speed * speed_scale_factor
        print(
            f"  Pedestrian base speed: {self.pedestrian_base_speed:.2f} m/s (scaled from User's {self.user_avg_speed:.2f} m/s)"
        )
        self.pedestrians: List[Pedestrian] = []
        self.pedestrian_spawn_info: Dict[int, dict] = {}
        self.user_repulsion_strength = 5.0
        self.user_repulsion_range = 0.8
        # self.available_moods = [
        #     PedestrianMood.Brisk_Individualist,
        #     PedestrianMood.Relaxed_Ped,
        #     "Social_Walker",
        #     "Social_Walker_v2",
        # ]
        self.available_moods = [
            "Uninterrupted_speed_walker",
            "Solo_sprinter",
            "Alert_fast_walker_open_space",
            "Engaged_speed_walker",
            "Group_barging_through",
            "Focused_rusher",
            "Crowd_weaving_rusher",
            "Watchful_runner",
            "Alert_crowd_sprinter",
            "Rushing_group_dense",
            "Zoned_out_weaver",
            "Group_in_a_panic",
            "Ruthless_barger",
            "Aggressive_barger",
            "Stressed_pusher",
            "Quiet_pair",
            "Desperate_rusher",
        ]
        self.mood_switch_rate = mood_switch_rate
        self.mood_switch_rates = (
            mood_switch_rates if mood_switch_rates else {}
        )  # transition_matrix[mood_from][mood_to] = probability
        self.transition_matrix = (
            mood_transition_matrix if mood_transition_matrix else {}
        )
        self.respawn_count = 0
        self.total_spawns = 0
        self.mood_switch_log = []  # list of (time, ped_id, old_mood, new_mood)

    def _find_spawn_position_along_trajectory(
        self,
        current_time: float,
        user_pos: Tuple[float, float],
        use_vicinity: bool = True,
    ) -> Tuple[float, float, int]:
        if not use_vicinity:
            traj_x, traj_y, traj_idx = (
                self.user_trajectory.get_random_trajectory_point()
            )
            return traj_x, traj_y, traj_idx
        else:
            current_idx = int(current_time / self.user_trajectory.dt)
            current_idx = max(
                0, min(current_idx, self.user_trajectory.n_trajectory_points - 1)
            )
            indices_per_meter = 1.0 / (
                self.user_trajectory.avg_velocity * self.user_trajectory.dt + 1e-6
            )
            idx_range = int(self.vicinity_radius * indices_per_meter)
            idx_range = max(10, idx_range)
            offset = np.random.randint(-idx_range // 2, idx_range)
            traj_idx = current_idx + offset
            traj_idx = max(
                0, min(traj_idx, self.user_trajectory.n_trajectory_points - 1)
            )
            traj_x, traj_y = self.user_trajectory.get_position_at_index(traj_idx)
            return traj_x, traj_y, traj_idx

    def _check_collision_with_user(
        self, x: float, y: float, user_pos: Tuple[float, float]
    ) -> bool:
        dist = np.sqrt((x - user_pos[0]) ** 2 + (y - user_pos[1]) ** 2)
        return dist < (self.user_trajectory.radius + self.user_trajectory.safety_radius)

    def _check_collision_with_pedestrians(
        self, x: float, y: float, exclude_id: int = -1
    ) -> bool:
        for ped in self.pedestrians:
            if ped.pedestrian_id == exclude_id:
                continue
            dist = np.sqrt((x - ped.x) ** 2 + (y - ped.y) ** 2)
            if dist < (ped.radius * 2 + 0.5):
                return True
        return False

    def spawn_pedestrian(
        self, current_time: float, pedestrian_id: int, initial_spawn: bool = False
    ) -> Optional[Pedestrian]:
        user_pos = self.user_trajectory.get_position_at_time(current_time)
        max_attempts = 50
        for attempt in range(max_attempts):
            traj_x, traj_y, traj_idx = self._find_spawn_position_along_trajectory(
                current_time, user_pos, use_vicinity=not initial_spawn
            )
            perp_x, perp_y = self.user_trajectory.get_perpendicular_at_index(traj_idx)
            offset_magnitude = np.random.uniform(
                self.min_spawn_distance, self.max_spawn_distance
            )
            offset_sign = np.random.choice([-1, 1])
            spawn_x = traj_x + offset_sign * offset_magnitude * perp_x
            spawn_y = traj_y + offset_sign * offset_magnitude * perp_y
            margin = 1.0
            spawn_x = np.clip(spawn_x, margin, self.config.env_width - margin)
            spawn_y = np.clip(spawn_y, margin, self.config.env_height - margin)
            if self._check_collision_with_user(spawn_x, spawn_y, user_pos):
                continue
            if self._check_collision_with_pedestrians(spawn_x, spawn_y):
                continue
            mood = np.random.choice(self.available_moods)
            goal_traj_idx = (
                traj_idx + np.random.randint(20, 100)
            ) % self.user_trajectory.n_trajectory_points
            goal_on_traj_x, goal_on_traj_y = self.user_trajectory.get_position_at_index(
                goal_traj_idx
            )
            goal_perp_x, goal_perp_y = self.user_trajectory.get_perpendicular_at_index(
                goal_traj_idx
            )
            goal_offset = np.random.uniform(2, 5) * np.random.choice([-1, 1])
            goal_x = goal_on_traj_x + goal_offset * goal_perp_x
            goal_y = goal_on_traj_y + goal_offset * goal_perp_y
            goal_x = np.clip(goal_x, 2, self.config.env_width - 2)
            goal_y = np.clip(goal_y, 2, self.config.env_height - 2)
            ped = Pedestrian(
                x=spawn_x,
                y=spawn_y,
                mood=mood,
                goal_x=goal_x,
                goal_y=goal_y,
                pedestrian_id=pedestrian_id,
                base_speed=self.pedestrian_base_speed,
            )
            self.pedestrian_spawn_info[pedestrian_id] = {
                "spawn_time": current_time,
                "spawn_pos": (spawn_x, spawn_y),
                "trajectory_ref_idx": traj_idx,
                "respawn_count": self.pedestrian_spawn_info.get(pedestrian_id, {}).get(
                    "respawn_count", 0
                ),
                "mood": mood if isinstance(mood, str) else mood.name,
                "effective_speed": ped.speed,
            }
            self.total_spawns += 1
            return ped
        print(
            f"  Warning: Failed to spawn pedestrian {pedestrian_id} after {max_attempts} attempts"
        )
        return None

    def initialize_pedestrians(self, current_time: float = 0.0):
        self.pedestrians = []
        self.pedestrian_spawn_info = {}
        self.respawn_count = 0
        self.total_spawns = 0
        for i in range(self.n_pedestrians):
            ped = self.spawn_pedestrian(
                current_time, pedestrian_id=i, initial_spawn=True
            )
            if ped is not None:
                self.pedestrians.append(ped)
        print(f"Initialized {len(self.pedestrians)}/{self.n_pedestrians} pedestrians:")
        for ped in self.pedestrians:
            # handle both string and enum moods
            if isinstance(ped.mood, str):
                mood_params = CUSTOM_MOODS[ped.mood]
                mood_name = ped.mood
            else:
                mood_params = MOOD_PARAMETERS[ped.mood]
                mood_name = ped.mood.name
            print(
                f"  Ped {ped.pedestrian_id}: {mood_name} - speed={ped.speed:.2f} m/s "
                f"(base={ped.base_speed:.2f} × factor={mood_params['speed_factor']:.2f})"
            )

    def check_and_respawn(self, current_time: float):
        user_pos = self.user_trajectory.get_position_at_time(current_time)
        for i, ped in enumerate(self.pedestrians):
            dist_to_user = np.sqrt(
                (ped.x - user_pos[0]) ** 2 + (ped.y - user_pos[1]) ** 2
            )
            if dist_to_user > self.respawn_distance:
                old_id = ped.pedestrian_id
                new_ped = self.spawn_pedestrian(
                    current_time, pedestrian_id=old_id, initial_spawn=False
                )
                if new_ped is not None:
                    self.pedestrians[i] = new_ped
                    self.pedestrian_spawn_info[old_id]["respawn_count"] += 1
                    self.respawn_count += 1

    def update_pedestrians(
        self, dt: float, current_time: float, static_obstacles: List = None
    ):
        user_pos = self.user_trajectory.get_position_at_time(current_time)
        # ---------- Mood switching (Poisson process) ----------
        # ---------- Mood switching (Poisson process) ----------
        if self.transition_matrix:
            for ped in self.pedestrians:
                current_mood = ped.mood if isinstance(ped.mood, str) else ped.mood.name
                # Determine switch probability for this pedestrian at this step
                if self.mood_switch_rates and current_mood in self.mood_switch_rates:
                    rate = self.mood_switch_rates[current_mood]
                else:
                    rate = self.mood_switch_rate  # global fallback
                if rate > 0 and np.random.rand() < rate * dt:
                    if current_mood in self.transition_matrix:
                        targets = self.transition_matrix[current_mood]
                        moods = list(targets.keys())
                        probs = list(targets.values())
                        new_mood = np.random.choice(moods, p=probs)
                        if not isinstance(ped.mood, str):
                            if new_mood in PedestrianMood.__members__:
                                new_mood = PedestrianMood[new_mood]
                        ped.set_mood(new_mood)
                        self.mood_switch_log.append(
                            (current_time, ped.pedestrian_id, current_mood, new_mood)
                        )
                        print(
                            f"  [MOOD SWITCH] t={current_time:.2f}s Ped {ped.pedestrian_id}: {current_mood} → {new_mood}"
                        )

        for ped in self.pedestrians:
            ped.update(
                dt=dt,
                robot_pos=user_pos,
                other_pedestrians=self.pedestrians,
                env_width=self.config.env_width,
                env_height=self.config.env_height,
                static_obstacles=static_obstacles,
            )

        self.check_and_respawn(current_time)

    def get_all_pedestrians_array(self) -> np.ndarray:
        if len(self.pedestrians) == 0:
            return np.zeros((0, 5))
        return np.array([ped.to_array() for ped in self.pedestrians])

    def get_statistics(self) -> dict:
        return {
            "total_spawns": self.total_spawns,
            "respawn_count": self.respawn_count,
            "active_pedestrians": len(self.pedestrians),
            "user_avg_speed": self.user_avg_speed,
            "pedestrian_base_speed": self.pedestrian_base_speed,
            "per_pedestrian_respawns": {
                pid: info["respawn_count"]
                for pid, info in self.pedestrian_spawn_info.items()
            },
        }

    def get_mood_switch_log(self) -> list:
        return self.mood_switch_log
