"""Overtaking logic shared across controllers."""

import numpy as np
from ..sfm.numba_utils import normalize_angle
from ..config import CONFIG
from ..utils.derivative_kf import DerivativeEstimatorKF


def compute_overtaking_heading(x, y, goal_pos, user):
    """
    Return a desired heading (radians) that guides the robot around the user
    when it is behind the user and within the activation distance.

    Parameters
    ----------
    x, y : float        robot position
    goal_pos : tuple     (gx, gy)
    user : dict          keys 'x','y','radius','facing','active'

    Returns
    -------
    float or None : desired heading in radians, or None if overtaking is inactive.
    """
    if user is None or not user.get("active", False):
        return None
    ux, uy, urad = user["x"], user["y"], user["radius"]
    ufacing = user.get("facing", 0.0)
    dx = x - ux
    dy = y - uy
    dist = np.hypot(dx, dy)
    activation_dist = 4.0 * urad
    # Only activate when robot is behind the user
    robot_is_behind = (dx * np.cos(ufacing) + dy * np.sin(ufacing)) < 0.0
    if not robot_is_behind or dist > activation_dist or dist < 1e-6:
        return None
    # Tangential direction around user toward goal
    r_hat = np.array([dx, dy]) / dist
    goal_vec = np.array([goal_pos[0] - ux, goal_pos[1] - uy])
    cross = r_hat[0] * goal_vec[1] - r_hat[1] * goal_vec[0]
    sign = 1.0 if cross >= 0 else -1.0
    tang = np.array([-r_hat[1], r_hat[0]]) * sign
    desired = 0.7 * tang + 0.3 * goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
    return np.arctan2(desired[1], desired[0])


def compute_circulation_acceleration(
    x, y, goal_pos, user, circ_force=20.0, goal_force=8.0
):
    ax, ay = 0.0, 0.0
    if user is None or not user.get("active", False):
        return ax, ay, False

    ux, uy, urad = user["x"], user["y"], user["radius"]
    ufacing = user.get("facing", 0.0)
    dxu = x - ux
    dyu = y - uy
    dist_user = np.hypot(dxu, dyu)
    proj = dxu * np.cos(ufacing) + dyu * np.sin(ufacing)  # positive = in front
    activation_dist = 3.0 * urad

    # ---- Engage ONLY when robot is strictly behind ----
    if proj < 0.0 and dist_user < activation_dist and dist_user > 1e-6:
        # Radial unit vector (user → robot)
        r_hat = np.array([dxu, dyu]) / dist_user

        # Tangential unit vector (rotated 90° CCW)
        tang_ccw = np.array([-r_hat[1], r_hat[0]])

        # Direction from robot to goal
        goal_vec = np.array([goal_pos[0] - x, goal_pos[1] - y])
        dist_goal = np.linalg.norm(goal_vec)
        if dist_goal > 1e-6:
            goal_dir = goal_vec / dist_goal
        else:
            goal_dir = np.array([0.0, 0.0])

        # Choose the tangential direction that moves toward the goal
        # using the cross product between r_hat and goal direction
        cross = r_hat[0] * goal_dir[1] - r_hat[1] * goal_dir[0]
        sign = 1.0 if cross >= 0 else -1.0
        tang_dir = sign * tang_ccw

        # ----- Blend: 70% tangential, 30% goal attraction -----
        desired_dir = 0.7 * tang_dir + 0.3 * goal_dir
        # Normalise the blended direction
        norm = np.linalg.norm(desired_dir)
        if norm > 1e-6:
            desired_dir /= norm

        # Apply the circulation force in the blended direction
        ax += circ_force * desired_dir[0]
        ay += circ_force * desired_dir[1]

        # Optional extra attraction directly toward the goal
        ax += goal_force * goal_dir[0]
        ay += goal_force * goal_dir[1]

        # Small radial push to stay outside safety margin
        if dist_user < urad + 1.0:
            ax += 5.0 * r_hat[0]
            ay += 5.0 * r_hat[1]

        return ax, ay, True

    return ax, ay, False


def compute_front_repulsion(x, y, theta, v_curr, user):
    """
    If the robot is in front of the user and the user is too close,
    return (v_cmd, omega_cmd) that pushes the robot forward.
    Returns None if no action needed.
    """
    if user is None:
        return None
    ux, uy, urad = user["x"], user["y"], user["radius"]
    ufacing = user.get("facing", 0.0)
    dxu = x - ux
    dyu = y - uy
    dist_user = np.hypot(dxu, dyu)
    proj = dxu * np.cos(ufacing) + dyu * np.sin(ufacing)  # positive = in front

    # If robot is in front (proj > 0) and distance is less than safety margin + some buffer
    safety_dist = urad + 0.6  # user safety radius + extra buffer
    if proj > 0 and dist_user < safety_dist:
        # Repulse away from user (direction from user to robot)
        dir_x = dxu / (dist_user + 1e-6)
        dir_y = dyu / (dist_user + 1e-6)
        # Acceleration magnitude proportional to how close
        acc_mag = 10.0 * (safety_dist - dist_user)  # linear push
        ax = acc_mag * dir_x
        ay = acc_mag * dir_y

        desired_vx = v_curr * np.cos(theta) + ax * 0.05  # assume dt ~ 0.05
        desired_vy = v_curr * np.sin(theta) + ay * 0.05
        desired_speed = np.hypot(desired_vx, desired_vy)
        desired_theta = (
            np.arctan2(desired_vy, desired_vx) if desired_speed > 0.1 else theta
        )
        v_cmd = min(desired_speed, 4.0)  # max linear vel
        omega_cmd = 2.0 * np.arctan2(
            np.sin(desired_theta - theta), np.cos(desired_theta - theta)
        )
        return v_cmd, omega_cmd
    return None


def compute_park_command(x, y, theta, v_curr, user, goal_pos, config):
    """
    Smooth parking: slide the robot along the user's safety boundary
    to a lateral parking spot, using SFM circulation + repulsion.

    The parking spot is chosen on the side that is closer to the robot
    OR that requires less turning.
    """
    ux, uy, urad = user["x"], user["y"], user["radius"]
    ufacing = user.get("facing", 0.0)
    park_dist = urad - config.park_margin
    if park_dist < 0.5:
        park_dist = 0.5

    # Two candidate spots: left and right perpendicular to user's facing
    perp = np.array([-np.sin(ufacing), np.cos(ufacing)])
    target_left = np.array([ux, uy]) + park_dist * perp
    target_right = np.array([ux, uy]) - park_dist * perp

    # Choose the spot that is closer to the robot
    vec = np.array([x, y])
    d_left = np.linalg.norm(vec - target_left)
    d_right = np.linalg.norm(vec - target_right)
    target = target_left if d_left <= d_right else target_right

    # ----- SFM force field -----
    dxu = x - ux
    dyu = y - uy
    dist_user = np.hypot(dxu, dyu)
    if dist_user < 0.01:
        dist_user = 0.01

    # 1) Strong radial repulsion from the user (keeps robot outside safety margin)
    r_hat = np.array([dxu, dyu]) / dist_user
    safety_dist = urad + 0.05  # stay just outside the safety radius
    if dist_user < safety_dist:
        penetration = safety_dist - dist_user
        rep_force = 3.0 * penetration  # linear spring
        ax = rep_force * r_hat[0]
        ay = rep_force * r_hat[1]
    else:
        ax, ay = 0.0, 0.0

    # 2) Tangential force to slide toward the chosen parking spot
    to_target = target - vec
    dist_target = np.linalg.norm(to_target)
    if dist_target > 0.05:
        to_target_dir = to_target / dist_target
        # Circulation sign: cross product of r_hat and target direction
        cross = r_hat[0] * to_target_dir[1] - r_hat[1] * to_target_dir[0]
        sign = 1.0 if cross >= 0 else -1.0
        tang_dir = np.array([-r_hat[1], r_hat[0]]) * sign
        # Tangential force proportional to distance to target
        tang_force = 3.0 * min(dist_target, 2.0)
        ax += tang_force * tang_dir[0]
        ay += tang_force * tang_dir[1]

    # 3) Weak attraction toward the target (helps finish the maneuver)
    if dist_target > 0.05:
        attr_force = 2.0 * min(dist_target, 1.0)
        ax += attr_force * to_target_dir[0]
        ay += attr_force * to_target_dir[1]

    # Convert acceleration to velocity command
    dt = 0.05  # approximate time step (will be overridden by actual dt in controller)
    desired_vx = v_curr * np.cos(theta) + ax * dt
    desired_vy = v_curr * np.sin(theta) + ay * dt
    desired_speed = np.hypot(desired_vx, desired_vy)
    desired_theta = np.arctan2(desired_vy, desired_vx) if desired_speed > 0.2 else theta

    # Clamp speed
    max_v = config.max_linear_vel
    if desired_speed > max_v:
        desired_speed = max_v
    v_cmd = desired_speed

    # Steering
    angle_diff = normalize_angle(desired_theta - theta)
    omega_cmd = 5.0 * angle_diff
    omega_cmd = np.clip(omega_cmd, -config.max_angular_vel, config.max_angular_vel)

    return v_cmd, omega_cmd


class ManeuverDOB:
    """
    Nonlinear disturbance observer for velocity‑commanded differential‑drive robots.
    The robot is modelled as   v̇ = (1/τ)(u_cmd – v) + d   with lumped disturbance d.
    The observer estimates d and cancels it.
    """

    def __init__(self, dt: float = 0.05, L: float = 8.0, tau: float = 0.1):
        self.dt = dt
        self.L = L  # observer gain
        self.tau = tau  # velocity time constant (from robot_params or config)
        self.d_hat = np.zeros(2)
        self.prev_vel = np.zeros(2)
        self.initialised = False
        self.v_filter = DerivativeEstimatorKF(dt=dt)
        self.omega_filter = DerivativeEstimatorKF(dt=dt)

    def reset(self):
        self.d_hat = np.zeros(2)
        self.prev_vel = np.zeros(2)
        self.initialised = False
        self.v_filter.reset()
        self.omega_filter.reset()

    def step(
        self,
        u_cmd: np.ndarray,
        u_man: np.ndarray,
        maneuver_active: bool,
        robot_vel: np.ndarray,
        dt: float = None,
        config=CONFIG,
    ) -> np.ndarray:
        """
        u_cmd   : velocity command from the main controller  (v, ω)
        u_man   : velocity command from the active manoeuvre
        maneuver_active : bool
        robot_vel : measured robot velocity (v, ω)
        dt : timestep
        Returns the final velocity command (v, ω) after disturbance rejection.
        """
        if dt is not None:
            self.dt = dt
        u_cmd = np.asarray(u_cmd, dtype=float)
        u_man = np.asarray(u_man, dtype=float)
        robot_vel = np.asarray(robot_vel, dtype=float)

        if not self.initialised:
            self.prev_vel = robot_vel.copy()
            self.initialised = True

        # ---- desired input ----
        if maneuver_active:
            u_des = u_man
        else:
            u_des = u_cmd

        # ---- acceleration prediction ----
        accel_pred = (1.0 / self.tau) * (u_des - robot_vel)

        # ---- measured acceleration (KF estimate) ----
        if dt is not None:
            self.v_filter.set_dt(dt)
            self.omega_filter.set_dt(dt)
        _, a_v, _ = self.v_filter.update(robot_vel[0])
        _, a_omega, _ = self.omega_filter.update(robot_vel[1])
        a_v = np.clip(a_v, -config.max_linear_accel, config.max_linear_accel)
        a_omega = np.clip(a_omega, -config.max_angular_accel, config.max_angular_accel)
        vel_dot = np.array([a_v, a_omega])

        # ---- observer update ----
        d_obs = vel_dot - accel_pred
        err = d_obs - self.d_hat
        self.d_hat += self.L * self.dt * err

        # ---- compensated command ----
        u_comp = (
            u_des - self.tau * self.d_hat
        )  # convert accel disturbance to velocity offset

        # Anti‑windup: if the compensated command exceeds limits, limit the integrator
        u_comp[0] = np.clip(u_comp[0], 0.0, config.max_linear_vel)
        u_comp[1] = np.clip(u_comp[1], -config.max_angular_vel, config.max_angular_vel)
        # If saturated, freeze the disturbance estimation for that axis
        saturated_v = (u_comp[0] == 0.0 or u_comp[0] == config.max_linear_vel)
        saturated_w = (u_comp[1] == -config.max_angular_vel or u_comp[1] == config.max_angular_vel)
        # (Only update d_hat for axes that are not saturated)
        # We can implement this by scaling the error for saturated axes to zero:
        if saturated_v:
            err[0] = 0.0
        if saturated_w:
            err[1] = 0.0
        self.d_hat += self.L * self.dt * err
        # (u_comp already clipped)
        return u_comp

    """Maneuver manager for overtaking, repulsion, parking, rotation, and recovery."""


# ===========================================================================
#  New centralized ManeuverManager for all MPC controllers
# ===========================================================================


class ManeuverManager:
    """
    Holds all maneuver parameters and provides unified methods to compute
    maneuver commands and handle state logging.

    Every method that activates a maneuver returns a 3‑tuple:
        (v_cmd, omega_cmd, active_bool)
    """

    def __init__(self, config):
        # ---- Rotation parameters ----
        self.rot_threshold = np.deg2rad(120)  # heading error to trigger
        self.rot_gain = 2.5  # proportional gain
        self.stuck_speed_thresh = 0.05  # m/s
        self.stuck_progress_thresh = 0.1  # metres
        self.goal_dist_history_maxlen = 100

        # ---- Soft recovery parameters ----
        self.overshoot_near = 0.8  # engage when goal behind & within this distance

        # ---- Overtaking parameters ----
        self.overtake_circ_force = 10.0
        self.overtake_goal_force = 10.0
        self.overtake_activation_dist = (
            2.5 * config.safety_margin
        )  # relies on user radius passed dynamically

        # ---- Repulsion parameters ----
        self.repulsion_safety_extra = 0.1  # added to user radius
        self.repulsion_acc_mag = 4.0

        # ---- Parking parameters ----
        self.parking_extra = 0.15  # added to user radius

        # ---- Logging state ----
        self._last_states = {
            "rotation": False,
            "soft_recovery": False,
            "overtaking": False,
            "repulsion": False,
            "parking": False,
        }

        # Reference to config for parking (needs park_margin)
        self.config = config
        # ------------------------------------------------------------
        #  Priority buffer – define manoeuvre order here
        # ------------------------------------------------------------
        # List of names in decreasing priority.  Parking is handled
        # separately (fall‑back after normal MPC).
        self.maneuver_priority = [
            "soft_recovery",
            "rotation",
            "overtaking",
            "repulsion",
        ]

    def check(self, name: str, state: dict) -> bool:
        """Return True if the named manoeuvre should be activated."""
        if name == "soft_recovery":
            return self.check_soft_recovery(
                state["user"], state["dist_goal"], state["heading_err"]
            )
        elif name == "rotation":
            return self.check_rotation(
                state["heading_err"], state["v_curr"], state["robot_pos_history"]
            )
        elif name == "overtaking":
            _, _, active = self.overtaking_active(
                state["x"], state["y"], state["goal_pos"], state["user"]
            )
            return active
        elif name == "repulsion":
            return self.repulsion_active(state["x"], state["y"], state["user"])
        else:
            return False

    def command(self, name: str, state: dict):
        sim_time = state["sim_time"]
        if name == "soft_recovery":
            return self.soft_recovery_command(
                sim_time,
                state["user"],
                state["heading_to_goal"],
                state["theta"],
                state["robot_pos"],
            )
        elif name == "rotation":
            return self.rotation_command(
                sim_time, state["heading_to_goal"], state["theta"], state["robot_pos"]
            )
        elif name == "overtaking":
            return self.overtaking_command(
                sim_time,
                state["x"],
                state["y"],
                state["theta"],
                state["goal_pos"],
                state["user"],
                state["v_curr"],
                state["dt"],
                state["robot_pos"],
            )
        elif name == "repulsion":
            return self.repulsion_command(
                sim_time,
                state["x"],
                state["y"],
                state["theta"],
                state["user"],
                state["v_curr"],
                state["dt"],
                state["robot_pos"],
            )
        else:
            return 0.0, 0.0, False

    # ------------------------------------------------------------------
    #  Logging helper
    # ------------------------------------------------------------------
    def _log_state(self, sim_time, name, active, robot_pos):
        if active != self._last_states[name]:
            state_str = "ENGAGED" if active else "DISENGAGED"
            print(
                f"  [{name.replace('_',' ').title()}] {state_str} at t=({sim_time:.1f}s), robot=({robot_pos[0]:.1f},{robot_pos[1]:.1f})"
            )
            self._last_states[name] = active

    def reset_states(self):
        for key in self._last_states:
            self._last_states[key] = False

    # ------------------------------------------------------------------
    #  1. Rotation maneuver
    # ------------------------------------------------------------------
    def check_rotation(self, heading_err, v_curr, robot_pos_history):
        """Return True if rotation should be activated.
        Stuck detection uses robot displacement, not goal distance."""
        if heading_err > self.rot_threshold:
            return True

        # Stuck detection: robot nearly stationary and hasn't moved recently
        if v_curr < self.stuck_speed_thresh and len(robot_pos_history) >= 15:
            p0 = np.array(robot_pos_history[0])
            p1 = np.array(robot_pos_history[-1])
            displacement = np.linalg.norm(p1 - p0)
            if displacement < self.stuck_progress_thresh:
                return True
        return False

    def rotation_command(self, sim_time, heading_to_goal, theta, robot_pos):
        """Return (v, omega, active_flag).  v is always 0."""
        omega_cmd = self.rot_gain * normalize_angle(heading_to_goal - theta)
        omega_cmd = np.clip(
            omega_cmd, -4.0, 4.0
        )  # robot max angular vel (will be clamped later)
        self._log_state(sim_time, "rotation", True, robot_pos)
        return 0.0, omega_cmd, True

    def rotation_off(self, sim_time, robot_pos):
        self._log_state(sim_time, "rotation", False, robot_pos)

    # ------------------------------------------------------------------
    #  2. Soft recovery
    # ------------------------------------------------------------------
    def check_soft_recovery(self, user, dist_goal, heading_err):
        return (
            user is not None
            and dist_goal < self.overshoot_near
            and abs(heading_err) > np.deg2rad(90)
        )

    def soft_recovery_command(self, sim_time, user, heading_to_goal, theta, robot_pos):
        u_speed = user.get("user_speed", 0.0)
        v_cmd = max(0.15, u_speed * 0.4)
        omega_cmd = 0.0 * normalize_angle(heading_to_goal - theta)
        self._log_state(sim_time, "soft_recovery", True, robot_pos)
        return v_cmd, omega_cmd, True

    def soft_recovery_off(self, sim_time, robot_pos):
        self._log_state(sim_time, "soft_recovery", False, robot_pos)

    # ------------------------------------------------------------------
    #  3. Overtaking
    # ------------------------------------------------------------------
    def overtaking_active(self, x, y, goal_pos, user):
        """Return (acceleration_x, acceleration_y, active_bool)."""
        ax, ay = 0.0, 0.0
        if user is None or not user.get("overtaking_active", False):
            return ax, ay, False

        ux, uy, urad = user["x"], user["y"], user["radius"]
        ufacing = user.get("facing", 0.0)
        dxu = x - ux
        dyu = y - uy
        dist_user = np.hypot(dxu, dyu)
        proj = dxu * np.cos(ufacing) + dyu * np.sin(ufacing)
        activation_dist = (
            self.overtake_activation_dist * urad
        )  # adjust if needed (here using user radius)
        
        front_offset = 1.0
        if proj < front_offset and dist_user < activation_dist and dist_user > 1e-6:
            r_hat = np.array([dxu, dyu]) / dist_user
            tang_ccw = np.array([-r_hat[1], r_hat[0]])

            goal_vec = np.array([goal_pos[0] - x, goal_pos[1] - y])
            dist_goal = np.linalg.norm(goal_vec)
            goal_dir = goal_vec / (dist_goal + 1e-6)

            cross = r_hat[0] * goal_dir[1] - r_hat[1] * goal_dir[0]
            sign = 1.0 if cross >= 0 else -1.0
            tang_dir = sign * tang_ccw

            desired_dir = 0.7 * tang_dir + 0.3 * goal_dir
            norm = np.linalg.norm(desired_dir)
            if norm > 1e-6:
                desired_dir /= norm

            ax += self.overtake_circ_force * desired_dir[0]
            ay += self.overtake_circ_force * desired_dir[1]
            ax += self.overtake_goal_force * goal_dir[0]
            ay += self.overtake_goal_force * goal_dir[1]

            if dist_user < urad + 1.0:
                ax += 7.0 * r_hat[0]
                ay += 7.0 * r_hat[1]

            return ax, ay, True
        return ax, ay, False

    def overtaking_command(
        self, sim_time, x, y, theta, goal_pos, user, v_curr, dt, robot_pos
    ):
        """Return (v_man, omega_man, active) or (0,0,False)."""
        ax, ay, active = self.overtaking_active(x, y, goal_pos, user)
        if active:
            desired_vx = v_curr * np.cos(theta) + ax * dt
            desired_vy = v_curr * np.sin(theta) + ay * dt
            desired_speed = np.hypot(desired_vx, desired_vy)
            desired_theta = (
                np.arctan2(desired_vy, desired_vx) if desired_speed > 0.1 else theta
            )
            max_v = (
                self.config.max_linear_vel
            )  # robot max linear vel (config.max_linear_vel would be better)
            v_man = min(desired_speed, max_v)
            omega_man = 2.0 * normalize_angle(desired_theta - theta)
            max_omega = self.config.max_angular_vel
            omega_man = np.clip(omega_man, -max_omega, max_omega)
            self._log_state(sim_time, "overtaking", True, robot_pos)
            return v_man, omega_man, True
        self._log_state(sim_time, "overtaking", False, robot_pos)
        return 0.0, 0.0, False

    # ------------------------------------------------------------------
    #  4. Front repulsion
    # ------------------------------------------------------------------
    def repulsion_active(self, x, y, user):
        if user is None:
            return False
        ux, uy, urad = user["x"], user["y"], user["radius"]
        ufacing = user.get("facing", 0.0)
        dxu = x - ux
        dyu = y - uy
        dist_user = np.hypot(dxu, dyu)
        proj = dxu * np.cos(ufacing) + dyu * np.sin(ufacing)
        safety_dist = urad + self.repulsion_safety_extra
        return proj > 0 and dist_user < safety_dist

    def repulsion_command(self, sim_time, x, y, theta, user, v_curr, dt, robot_pos):
        if not self.repulsion_active(x, y, user):
            self._log_state(sim_time, "repulsion", False, robot_pos)
            return 0.0, 0.0, False

        ux, uy, urad, uv = user["x"], user["y"], user["radius"], user["user_speed"]
        dxu = x - ux
        dyu = y - uy
        dist_user = np.hypot(dxu, dyu) + 1e-6
        safety_dist = urad + self.repulsion_safety_extra
        dir_x = dxu / dist_user
        dir_y = dyu / dist_user
        acc_mag = max(
            self.repulsion_acc_mag
            * (safety_dist - dist_user)
            / min(abs(v_curr - uv), 1.0),
            0.0,
        )
        ax = acc_mag * dir_x
        ay = acc_mag * dir_y
        desired_vx = v_curr * np.cos(theta) + ax * dt
        desired_vy = v_curr * np.sin(theta) + ay * dt
        desired_speed = np.hypot(desired_vx, desired_vy) + 0.3
        desired_theta = (
            np.arctan2(desired_vy, desired_vx) if desired_speed > 0.1 else theta
        )
        v_max = self.config.max_linear_vel
        v_man = min(desired_speed, v_max)
        omega_man = 2.0 * normalize_angle(desired_theta - theta)
        omega_max = self.config.max_angular_vel
        omega_man = np.clip(omega_man, -omega_max, omega_max)
        self._log_state(sim_time, "repulsion", True, robot_pos)
        return v_man, omega_man, True

    # ------------------------------------------------------------------
    #  5. Parking
    # ------------------------------------------------------------------
    def parking_active(self, x, y, user, final_v, final_omega):
        if user is None:
            return False
        if abs(final_v) > 0.01 or abs(final_omega) > 0.01:
            return False
        ux, uy, urad = user["x"], user["y"], user["radius"]
        dist_user = np.hypot(x - ux, y - uy)
        return dist_user < (urad + self.parking_extra)

    def parking_command(self, sim_time, x, y, theta, v_curr, user, goal_pos, robot_pos):
        # Use the existing compute_park_command (still available)
        ux, uy, urad = user["x"], user["y"], user["radius"]
        # Temporarily use the old function
        v_park, omega_park = compute_park_command(
            x, y, theta, v_curr, user, goal_pos, self.config
        )
        self._log_state(sim_time, "parking", True, robot_pos)
        return v_park, omega_park, True
