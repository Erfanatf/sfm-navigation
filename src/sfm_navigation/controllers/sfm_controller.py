"""Social Force Model controller with DOB, LPFs and integrated overtaking maneuver."""

import time
import numpy as np
from ..config import SimulationConfig
from ..sfm.numba_utils import normalize_angle
from .maneuvers import ManeuverManager, ManeuverDOB
from .mpc.base_mpc import ControlLPF  # reuse the same LPF class


class SFMController:
    def __init__(self, config: SimulationConfig, robot_params: dict):
        self.config = config
        self.robot_params = robot_params
        self.last_compute_time = 0.0  # seconds

        # ── Attributes expected by robot_demo / animation ──
        self._v_flt = 0.0
        self._omega_flt = 0.0
        self._v_base = 0.0
        self._omega_base = 0.0
        self._v_man = 0.0
        self._omega_man = 0.0
        self._v_final = 0.0
        self._omega_final = 0.0

        self._target_path = np.zeros(
            (0, 2)
        )  # not used by SFM, but required for reading

        # Maneuver active flags
        self.overtaking_active = False
        self.parking_active = False
        self.repulsion_active = False
        self.rotation_active = False
        self.soft_recovery_active = False

        # ── Maneuver manager (shared with MPC) ──
        self.maneuvers = ManeuverManager(config)
        self.maneuver_priority = [
            "soft_recovery",
            "overtaking",
            "repulsion",
        ]  # SFM only uses overtaking

        # ── DOB and LPFs (same structure as MPC controllers) ──
        self.dob = ManeuverDOB(dt=0.05, L=8.0, tau=0.3)
        self.LPF = ControlLPF(alpha=0.5)  # base command filter
        self.maneuver_LPF = ControlLPF(alpha=0.5)  # maneuver filter
        self.DOB_LPF = ControlLPF(alpha=0.5)  # final output filter

    def compute_velocity(
        self,
        robot_state,
        goal_pos,
        obstacles,
        user=None,
        dt=None,
        sim_time=None,
        **kwargs
    ):
        t_start = time.perf_counter()

        # Reset per‑step flags
        self.overtaking_active = False
        self.parking_active = False
        self.repulsion_active = False
        self.rotation_active = False
        self.soft_recovery_active = False

        rx, ry, theta = robot_state.x, robot_state.y, robot_state.theta
        v_curr = robot_state.v
        w_curr = robot_state.omega
        robot_vel = np.array([v_curr, w_curr])

        if dt is None:
            dt = self.config.dt

        # ── Build state dict for the maneuver manager ──
        state = dict(
            x=rx,
            y=ry,
            theta=theta,
            v_curr=v_curr,
            goal_pos=goal_pos,
            user=user,
            dt=dt,
            sim_time=sim_time,
            heading_to_goal=np.arctan2(goal_pos[1] - ry, goal_pos[0] - rx),
            dist_goal=np.hypot(goal_pos[0] - rx, goal_pos[1] - ry),
            heading_err=abs(
                normalize_angle(np.arctan2(goal_pos[1] - ry, goal_pos[0] - rx) - theta)
            ),
            robot_pos=(rx, ry),
            robot_pos_history=None,
            goal_dist_history=None,
        )

        # ── Check overtaking via the manager ──
        v_man_cmd = 0.0
        omega_man_cmd = 0.0
        maneuver_active = False
        for name in self.maneuver_priority:
            if self.maneuvers.check(name, state):
                v_cmd, omega_cmd, active = self.maneuvers.command(name, state)
                if active:
                    self.overtaking_active = True  # only overtaking can be active
                    v_man_cmd = v_cmd
                    omega_man_cmd = omega_cmd
                    maneuver_active = True
                break  # only the first (highest priority) maneuver is executed

        # ── Normal SFM operation (unchanged core) ──
        # (always computed; if a maneuver is active it will be used as u_man,
        #  but the base command is still needed for the DOB).
        v0 = self.robot_params["v0"]
        tau = self.robot_params["tau"]
        A = self.robot_params.get("A_ped", self.robot_params.get("A", 3.0))
        B = self.robot_params.get("B_ped", self.robot_params.get("B", 0.5))
        lam = self.robot_params.get("lam_base", self.robot_params.get("lam", 0.5))
        kappa = self.robot_params.get("kappa", 0.0)

        dx = goal_pos[0] - rx
        dy = goal_pos[1] - ry
        dist_goal = np.hypot(dx, dy)
        desired_dir = np.arctan2(dy, dx) if dist_goal > 0.1 else theta
        desired_vx = v0 * np.cos(desired_dir)
        desired_vy = v0 * np.sin(desired_dir)
        ax = (desired_vx - v_curr * np.cos(theta)) / tau
        ay = (desired_vy - v_curr * np.sin(theta)) / tau

        speed = np.hypot(v_curr, 0)
        if speed > 0.2 and abs(kappa) > 1e-6:
            perp_x = -np.sin(theta)
            perp_y = np.cos(theta)
            lateral_acc = kappa * speed**2
            ax += lateral_acc * perp_x
            ay += lateral_acc * perp_y

        # Obstacle repulsion
        for obs in obstacles:
            ox, oy, orad = obs[0], obs[1], obs[2]
            dx_obs = rx - ox
            dy_obs = ry - oy
            dist = np.hypot(dx_obs, dy_obs)

            if dist < orad + 2.0 and dist > 1e-6:
                n = np.array([dx_obs, dy_obs]) / dist
                angle_to = np.arctan2(oy - ry, ox - rx)
                phi = angle_to - theta
                w = lam + (1 - lam) * (1 + np.cos(phi)) / 2.0
                force_mag = A * np.exp((orad - dist) / B) * w
                ax += force_mag * n[0]
                ay += force_mag * n[1]
                safe_dist = orad + self.config.robot_radius
                if dist < safe_dist:
                    penetration = safe_dist - dist
                    extra_force_mag = 80.0 * penetration
                    ax += extra_force_mag * n[0]
                    ay += extra_force_mag * n[1]

        # Desired velocity from resulting acceleration
        desired_vx_new = v_curr * np.cos(theta) + ax * self.config.dt
        desired_vy_new = v_curr * np.sin(theta) + ay * self.config.dt
        desired_speed = np.hypot(desired_vx_new, desired_vy_new)
        desired_theta = np.arctan2(desired_vy_new, desired_vx_new)

        slowing_dist = 0.3
        stop_dist = self.config.goal_tolerance - 0.25
        if dist_goal < stop_dist:
            v_base_raw = 0.0
        else:
            speed_factor = min(1.0, dist_goal / slowing_dist)
            v_base_raw = desired_speed * speed_factor
            v_base_raw = min(v_base_raw, self.config.max_linear_vel)

        angle_diff = normalize_angle(desired_theta - theta)
        if abs(angle_diff) < 0.05:
            omega_base_raw = 0.0
        else:
            omega_base_raw = 2.0 * angle_diff
        omega_base_raw = np.clip(
            omega_base_raw, -self.config.max_angular_vel, self.config.max_angular_vel
        )

        # ── DOB + LPF pipeline (same as MPC) ──
        # Base command from SFM (raw)
        u_cmd = np.array([v_base_raw, omega_base_raw])
        u_cmd_flt = self.LPF.filter(u_cmd)

        if maneuver_active:
            # Route the maneuver through the DOB
            u_man_arr = np.array([v_man_cmd, omega_man_cmd])
            u_man_arr_flt = self.maneuver_LPF.filter(u_man_arr)
            u_final = self.dob.step(u_cmd_flt, u_man_arr_flt, True, robot_vel, dt)
            u_final_flt = self.DOB_LPF.filter(u_final)
            self._v_base, self._omega_base = u_cmd_flt
            self._v_man, self._omega_man = u_man_arr_flt
        else:
            # Normal operation – only external disturbance is compensated
            self.maneuver_LPF.reset()
            u_final = self.dob.step(u_cmd_flt, np.zeros(2), False, robot_vel, dt)
            u_final_flt = self.DOB_LPF.filter(u_final)
            self._v_base, self._omega_base = u_cmd_flt
            self._v_man, self._omega_man = 0.0, 0.0

        self._v_final, self._omega_final = u_final_flt
        self.last_compute_time = time.perf_counter() - t_start
        return u_final_flt[0], u_final_flt[1]

    def get_real_time_factor(self, desired_period: float) -> float:
        if desired_period <= 0:
            return 0.0
        return self.last_compute_time / desired_period
