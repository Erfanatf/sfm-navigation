"""Performance metrics for front‑following social robot navigation.

All tunable thresholds are exposed as class‑level constants for easy adjustment.
The user is **excluded** from collision and safety distance metrics by default.
"""

import numpy as np
import pandas as pd
from ..config import CONFIG

class ControllerPerformance:
    """Compute performance metrics from a simulation history CSV file."""

    # ── Tunable constants ──────────────────────────────────────────
    GOAL_TOLERANCE         = CONFIG.goal_tolerance      # [m] distance to goal to consider "reached"
    SAFETY_MARGIN          = CONFIG.safety_margin       # [m] used for collision detection
    PERSONAL_SPACE         = 1.5       # [m] distance for Social Safety Index (SSI)
    COMFORTABLE_DISTANCE   = 3.0       # [m] comfort zone for SII
    CRITICAL_DISTANCE      = 0.3       # [m] near‑miss threshold
    TTC_TIME_HORIZON       = 2.0       # [s] cap for Time‑to‑Collision
    STOP_SPEED_THRESH      = 0.05      # [m/s] below this robot is considered stopped
    DIRECTION_CHANGE_THRESH_DEG = 5.0  # [deg] min heading change to count as direction change
    INTRUSION_COOLDOWN      = 0.5      # [s] cooldown for counting intrusion/collision events
    # ───────────────────────────────────────────────────────────────

    def __init__(self, csv_path: str,
                 goal_tolerance: float = None,
                 safety_margin: float = None,
                 personal_space: float = None,
                 comfortable_distance: float = None,
                 critical_distance: float = None,
                 ttc_time_horizon: float = None,
                 stop_speed_thresh: float = None,
                 direction_change_thresh_deg: float = None,
                 intrusion_cooldown: float = None,
                 include_user: bool = False):          # <-- NEW
        """
        Parameters
        ----------
        include_user : bool
            If False (default), the user is excluded from collision and
            minimum‑distance metrics.
        """
        self.include_user = include_user
        self.goal_tolerance = goal_tolerance or self.GOAL_TOLERANCE
        self.safety_margin = safety_margin or self.SAFETY_MARGIN
        self.personal_space = personal_space or self.PERSONAL_SPACE
        self.comfortable_distance = comfortable_distance or self.COMFORTABLE_DISTANCE
        self.critical_distance = critical_distance or self.CRITICAL_DISTANCE
        self.ttc_time_horizon = ttc_time_horizon or self.TTC_TIME_HORIZON
        self.stop_speed_thresh = stop_speed_thresh or self.STOP_SPEED_THRESH
        self.direction_change_thresh_deg = direction_change_thresh_deg or self.DIRECTION_CHANGE_THRESH_DEG
        self.intrusion_cooldown = intrusion_cooldown or self.INTRUSION_COOLDOWN

        self.df = pd.read_csv(csv_path)
        self.robot = self.df[self.df["agent_type"] == "robot"].copy()
        self.pedestrians = self.df[self.df["agent_type"] == "pedestrian"]
        self.user = self.df[self.df["agent_type"] == "user"]
        self.obs_static = self.df[self.df["agent_type"] == "static_obstacle"]
        self.obs_dynamic = self.df[self.df["agent_type"] == "dynamic_obstacle"]

        # Pre‑compute frequently used arrays
        self.times = self.robot["time"].values
        self.dt = np.mean(np.diff(self.times)) if len(self.times) > 1 else 0.04
        self.robot_pos = self.robot[["x", "y"]].values
        self.robot_theta = self.robot["theta"].values
        self.robot_radius = self.robot["radius"].values[0]   # constant
        self.robot_vel = np.sqrt(self.robot["vx"].values**2 + self.robot["vy"].values**2)
        self.robot_speed = (self.robot["lin_speed"].values if "lin_speed" in self.robot
                            else self.robot_vel)
        self.lin_accel = self.robot["lin_accel"].values if "lin_accel" in self.robot else None
        self.lin_jerk = self.robot["lin_jerk"].values if "lin_jerk" in self.robot else None
        self.goal_x = self.robot["goal_x"].values
        self.goal_y = self.robot["goal_y"].values
        self.dist_to_goal = np.sqrt((self.robot_pos[:,0]-self.goal_x)**2 +
                                    (self.robot_pos[:,1]-self.goal_y)**2)
        self.goal_reached = self.dist_to_goal[-1] < self.goal_tolerance

        # Pedestrian data – exact timestamps (must be prepared before building collision flags)
        self._prepare_pedestrian_data()

        # ---- Collision flag: either raw from CSV or rebuilt without user ----
        if self.include_user:
            self.collision_flags = (self.robot["in_collision"].values
                                    if "in_collision" in self.robot else None)
        else:
            self.collision_flags = self._build_collision_flags_no_user()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _prepare_pedestrian_data(self):
        ped_groups = self.pedestrians.groupby("time")
        self.ped_positions = {t: g[["x","y"]].values for t, g in ped_groups}
        self.ped_velocities = {t: g[["vx","vy"]].values for t, g in ped_groups}

    def _get_pedestrians_at(self, idx: int):
        t = self.times[idx]
        return self.ped_positions.get(t, np.empty((0,2))), self.ped_velocities.get(t, np.empty((0,2)))

    def _build_collision_flags_no_user(self):
        """
        Re‑compute collision flag for each time step, using only
        pedestrians, static obstacles, and dynamic obstacles.
        """
        n = len(self.times)
        flags = np.zeros(n, dtype=bool)
        rr = self.robot_radius
        margin = self.safety_margin
        for i in range(n):
            rx, ry = self.robot_pos[i]
            # Pedestrians
            ped_pos, _ = self._get_pedestrians_at(i)
            for px, py in ped_pos:
                d = np.hypot(rx - px, ry - py) - self.pedestrians["radius"].values[0] - rr
                if d < margin:
                    flags[i] = True
                    break
            if flags[i]:
                continue
            # Static obstacles
            for _, row in self.obs_static[self.obs_static["time"] == self.times[i]].iterrows():
                d = np.hypot(rx - row["x"], ry - row["y"]) - row["radius"] - rr
                if d < margin:
                    flags[i] = True
                    break
            if flags[i]:
                continue
            # Dynamic obstacles (if any)
            for _, row in self.obs_dynamic[self.obs_dynamic["time"] == self.times[i]].iterrows():
                d = np.hypot(rx - row["x"], ry - row["y"]) - row["radius"] - rr
                if d < margin:
                    flags[i] = True
                    break
        return flags

    def _count_events(self, violation_flags: np.ndarray) -> int:
        """Count distinct events using cooldown."""
        if len(violation_flags) == 0:
            return 0
        events = 0
        last_event_time = -np.inf
        for i, flag in enumerate(violation_flags):
            if flag and (self.times[i] - last_event_time) > self.intrusion_cooldown:
                events += 1
                last_event_time = self.times[i]
        return events

    # ===================================================================
    #  1. Navigation Efficiency
    # ===================================================================
    def path_length(self) -> float:
        diffs = np.diff(self.robot_pos, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def path_efficiency(self) -> float:
        if len(self.robot_pos) < 2:
            return 1.0
        direct = float(np.linalg.norm(self.robot_pos[-1] - self.robot_pos[0]))
        pl = self.path_length()
        return direct / pl if pl > 0 else 1.0

    def time_to_goal(self) -> float:
        reached = np.where(self.dist_to_goal < self.goal_tolerance)[0]
        return float(self.times[reached[0]]) if len(reached) > 0 else float(self.times[-1])

    def average_speed(self) -> float:
        return float(np.mean(self.robot_speed))

    def max_speed(self) -> float:
        return float(np.max(self.robot_speed))

    def speed_variance(self) -> float:
        return float(np.var(self.robot_speed))

    def stop_ratio(self) -> float:
        return float(np.mean(self.robot_speed < self.stop_speed_thresh))

    # ===================================================================
    #  2. Safety & Collision
    # ===================================================================
    def min_distance_to_obstacles(self) -> float:
        """Closest distance to any obstacle or pedestrian (excludes user by default)."""
        min_dist = np.inf
        rr = self.robot_radius
        for i in range(len(self.times)):
            rx, ry = self.robot_pos[i]
            # User – only if explicitly included
            if self.include_user and i < len(self.user):
                ux = self.user["x"].values[i]; uy = self.user["y"].values[i]
                ur = self.user["radius"].values[i]
                d = np.hypot(rx-ux, ry-uy) - ur - rr
                min_dist = min(min_dist, d)
            # Pedestrians
            ped_pos, _ = self._get_pedestrians_at(i)
            for px, py in ped_pos:
                d = np.hypot(rx-px, ry-py) - self.pedestrians["radius"].values[0] - rr
                min_dist = min(min_dist, d)
            # Static obstacles
            obs_rows = self.obs_static[self.obs_static["time"] == self.times[i]]
            for _, row in obs_rows.iterrows():
                d = np.hypot(rx - row["x"], ry - row["y"]) - row["radius"] - rr
                min_dist = min(min_dist, d)
            # Dynamic obstacles
            obs_dyn_rows = self.obs_dynamic[self.obs_dynamic["time"] == self.times[i]]
            for _, row in obs_dyn_rows.iterrows():
                d = np.hypot(rx - row["x"], ry - row["y"]) - row["radius"] - rr
                min_dist = min(min_dist, d)
        return float(min_dist) if min_dist != np.inf else 10.0

    def collision_event_count(self) -> int:
        """Distinct collision events (user excluded by default)."""
        if self.collision_flags is None:
            return 0
        return self._count_events(self.collision_flags)

    def collision_step_count(self) -> int:
        """Total simulation steps where collision flag is True (user excluded by default)."""
        return int(np.sum(self.collision_flags)) if self.collision_flags is not None else 0

    def social_safety_index(self) -> float:
        """Fraction of time steps where any pedestrian is within personal_space."""
        intrusions = 0
        total = len(self.times)
        for i in range(total):
            ped_pos, _ = self._get_pedestrians_at(i)
            if len(ped_pos) == 0:
                continue
            if np.any(np.linalg.norm(ped_pos - self.robot_pos[i], axis=1) < self.personal_space):
                intrusions += 1
        return intrusions / total if total > 0 else 0.0

    def personal_intrusion_event_count(self) -> int:
        """Distinct events where a pedestrian enters personal_space (cooldown)."""
        total = len(self.times)
        violation_flags = np.zeros(total, dtype=bool)
        for i in range(total):
            ped_pos, _ = self._get_pedestrians_at(i)
            if len(ped_pos) == 0:
                continue
            if np.any(np.linalg.norm(ped_pos - self.robot_pos[i], axis=1) < self.personal_space):
                violation_flags[i] = True
        return self._count_events(violation_flags)

    def time_to_collision_min(self) -> float:
        min_ttc = np.inf
        for i in range(len(self.times)):
            rx, ry = self.robot_pos[i]
            vx = self.robot["vx"].values[i]; vy = self.robot["vy"].values[i]
            ped_pos, ped_vel = self._get_pedestrians_at(i)
            for j, (px, py) in enumerate(ped_pos):
                rel_vx = vx - ped_vel[j,0]; rel_vy = vy - ped_vel[j,1]
                rel_dist = np.hypot(rx-px, ry-py)
                rel_speed = np.hypot(rel_vx, rel_vy)
                if rel_speed > 1e-6:
                    ttc = rel_dist / rel_speed
                    if ttc < min_ttc and ttc < self.ttc_time_horizon:
                        min_ttc = ttc
        return float(min_ttc) if min_ttc != np.inf else float(self.ttc_time_horizon)

    # ===================================================================
    #  3. Social Comfort
    # ===================================================================
    def social_individual_index(self) -> float:
        total_intrusion = 0.0
        N = len(self.times)
        for i in range(N):
            ped_pos, _ = self._get_pedestrians_at(i)
            for px, py in ped_pos:
                d = np.hypot(self.robot_pos[i,0]-px, self.robot_pos[i,1]-py)
                intrusion = max(0.0, 1.0 - d / self.comfortable_distance)
                total_intrusion += intrusion
        return total_intrusion / N if N > 0 else 0.0

    def relative_motion_index(self) -> float:
        total_rmi = 0.0
        N = len(self.times)
        for i in range(1, N):
            rx, ry = self.robot_pos[i]
            vx = self.robot["vx"].values[i]; vy = self.robot["vy"].values[i]
            ped_pos, ped_vel = self._get_pedestrians_at(i)
            for j, (px, py) in enumerate(ped_pos):
                diff = np.array([px - rx, py - ry])
                d = np.linalg.norm(diff)
                if d < 0.1:
                    continue
                n_rp = diff / d
                rel_vel = np.array([vx - ped_vel[j,0], vy - ped_vel[j,1]])
                approach = np.dot(rel_vel, n_rp)
                if approach > 0:
                    total_rmi += approach / d
        return total_rmi / N if N > 0 else 0.0

    def social_grace_index(self) -> float:
        scores = []
        for i in range(len(self.times)):
            rx, ry = self.robot_pos[i]
            theta = self.robot_theta[i]
            heading = np.array([np.cos(theta), np.sin(theta)])
            ped_pos, _ = self._get_pedestrians_at(i)
            for px, py in ped_pos:
                d = np.hypot(rx-px, ry-py)
                if d < 0.5 or d > 3.0:
                    continue
                to_ped = np.array([px - rx, py - ry]) / d
                scores.append(np.dot(heading, to_ped))
        return float(np.mean(scores)) if scores else 1.0

    # ===================================================================
    #  4. Smoothness & Jerk
    # ===================================================================
    def acceleration_rms(self) -> float:
        if self.lin_accel is None:
            return 0.0
        return float(np.sqrt(np.mean(self.lin_accel**2)))

    def max_acceleration(self) -> float:
        if self.lin_accel is None:
            return 0.0
        return float(np.max(np.abs(self.lin_accel)))

    def jerk_rms(self) -> float:
        if self.lin_jerk is None:
            return 0.0
        return float(np.sqrt(np.mean(self.lin_jerk**2)))

    def jerk_mean(self) -> float:
        if self.lin_jerk is None:
            return 0.0
        return float(np.mean(np.abs(self.lin_jerk)))

    def smoothness_curvature(self) -> float:
        if len(self.robot_theta) < 3:
            return 0.0
        dtheta = np.diff(self.robot_theta)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        return float(np.sum(np.abs(dtheta)))

    def sinuosity(self) -> float:
        if len(self.robot_theta) < 3:
            return 0.0
        dtheta = np.diff(self.robot_theta)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        return float(np.std(np.abs(dtheta)))

    # ===================================================================
    #  5. Path Quality
    # ===================================================================
    def legibility(self) -> float:
        scores = []
        for i in range(len(self.times)):
            theta = self.robot_theta[i]
            dx = self.goal_x[i] - self.robot_pos[i,0]
            dy = self.goal_y[i] - self.robot_pos[i,1]
            dist = np.hypot(dx, dy)
            if dist < 1e-6:
                continue
            goal_heading = np.arctan2(dy, dx)
            delta = theta - goal_heading
            scores.append(np.cos(delta))
        return float(np.mean(scores)) if scores else 1.0

    def direction_changes(self) -> int:
        if len(self.robot_theta) < 2:
            return 0
        dtheta = np.diff(self.robot_theta)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        return int(np.sum(np.abs(dtheta) > np.deg2rad(self.direction_change_thresh_deg)))

    def mean_curvature(self) -> float:
        if len(self.robot_pos) < 3:
            return 0.0
        curvatures = []
        pts = self.robot_pos
        for i in range(1, len(pts)-1):
            x0, y0 = pts[i-1]; x1, y1 = pts[i]; x2, y2 = pts[i+1]
            dx = 0.5*(x2 - x0); dy = 0.5*(y2 - y0)
            ddx = x0 - 2*x1 + x2; ddy = y0 - 2*y1 + y2
            denom = (dx**2 + dy**2)**1.5
            if denom < 1e-6:
                continue
            curvatures.append(np.abs(dx*ddy - dy*ddx) / denom)
        return float(np.mean(curvatures)) if curvatures else 0.0

    def max_curvature(self) -> float:
        if len(self.robot_pos) < 3:
            return 0.0
        curvatures = []
        pts = self.robot_pos
        for i in range(1, len(pts)-1):
            x0, y0 = pts[i-1]; x1, y1 = pts[i]; x2, y2 = pts[i+1]
            dx = 0.5*(x2 - x0); dy = 0.5*(y2 - y0)
            ddx = x0 - 2*x1 + x2; ddy = y0 - 2*y1 + y2
            denom = (dx**2 + dy**2)**1.5
            if denom < 1e-6:
                continue
            curvatures.append(np.abs(dx*ddy - dy*ddx) / denom)
        return float(np.max(curvatures)) if curvatures else 0.0

    # ===================================================================
    #  6. Computational Performance
    # ===================================================================
    def computation_time_mean(self) -> float:
        if "comp_time" in self.robot:
            return float(self.robot["comp_time"].mean() * 1000.0)
        return 0.0

    def real_time_factor(self) -> float:
        mean_comp = self.robot["comp_time"].mean() if "comp_time" in self.robot else 0.0
        return float(self.dt / mean_comp) if mean_comp > 0 else float('inf')

    # ===================================================================
    #  Aggregate all metrics
    # ===================================================================
    def compute_all(self) -> dict:
        return {
            "Navigation Efficiency": {
                "Path Length (m)": self.path_length(),
                "Path Efficiency": self.path_efficiency(),
                "Time to Goal (s)": self.time_to_goal(),
                "Average Speed (m/s)": self.average_speed(),
                "Max Speed (m/s)": self.max_speed(),
                "Speed Variance": self.speed_variance(),
                "Stop Ratio": self.stop_ratio(),
            },
            "Safety & Collision": {
                "Collision Events": self.collision_event_count(),
                "Collision Steps": self.collision_step_count(),
                "Min Distance to Obstacles (m)": self.min_distance_to_obstacles(),
                "Social Safety Index (SSI)": self.social_safety_index(),
                "Personal Intrusion Events": self.personal_intrusion_event_count(),
                "Min Time to Collision (s)": self.time_to_collision_min(),
            },
            "Social Comfort": {
                "Social Individual Index (SII)": self.social_individual_index(),
                "Relative Motion Index (RMI)": self.relative_motion_index(),
                "Social Grace Index (SGI)": self.social_grace_index(),
            },
            "Smoothness & Jerk": {
                "Acceleration RMS (m/s²)": self.acceleration_rms(),
                "Max Acceleration (m/s²)": self.max_acceleration(),
                "Jerk RMS (m/s³)": self.jerk_rms(),
                "Jerk Mean (m/s³)": self.jerk_mean(),
                "Smoothness (rad)": self.smoothness_curvature(),
                "Sinuosity (rad)": self.sinuosity(),
            },
            "Path Quality": {
                "Legibility": self.legibility(),
                "Direction Changes": self.direction_changes(),
                "Directness": self.path_efficiency(),
                "Mean Curvature (1/m)": self.mean_curvature(),
                "Max Curvature (1/m)": self.max_curvature(),
            },
            "Computational": {
                "Avg Comp Time (ms)": self.computation_time_mean(),
                "Real‑Time Factor": self.real_time_factor(),
            },
        }

    def print_report(self):
        report = self.compute_all()
        for category, metrics in report.items():
            print(f"\n{'='*50}")
            print(f"  {category}")
            print(f"{'='*50}")
            for name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {name:<35} {value:>10.4f}")
                else:
                    print(f"  {name:<35} {value:>10}")