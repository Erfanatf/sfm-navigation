"""Risk‑Aware MPPI controller (Trevisan et al. 2025) – collision probability based."""

import numpy as np
from numba import njit
from typing import Tuple
from .mppi_noise import NoiseGenerator
from .mppi import MPPIController, mppi_rollout_batch, mppi_compute_costs


@njit(cache=False)
def _check_collision_for_trajectory(
    traj,  # shape (H+1, 3)  [x, y, θ]
    circles,  # (M, 3) [cx, cy, r]
    robot_radius,
    safety_margin,
):
    """Return True if the trajectory collides with any circle."""
    safe_dist = robot_radius + safety_margin
    for t in range(traj.shape[0]):
        x, y = traj[t, 0], traj[t, 1]
        for i in range(circles.shape[0]):
            cx, cy, r = circles[i, 0], circles[i, 1], circles[i, 2]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < r + safe_dist:
                return True
    return False


class RiskAwareMPPIController(MPPIController):
    """
    Risk‑Aware MPPI following Trevisan et al. (IROS 2025).

    Uses Monte‑Carlo estimates of collision probability (CP) over the planning
    horizon, adds a linear CP penalty to the nominal MPPI cost, and rejects
    trajectories whose CP exceeds a hard threshold.
    """

    def __init__(self, config):
        super().__init__(config)

        # Collision probability estimation parameters
        self.pedestrian_uncertainty = 0.3  # std. dev. of Gaussian noise [m]
        self.n_risk_samples = 20  # number of MC samples per trajectory
        self.collision_threshold = 0.1  # max CP before trajectory is rejected
        self.w_soft = 2000.0  # linear penalty weight for CP

        # MPPI sampling parameters (inherited from MPPIController, can be kept as is)
        # lam = 1.0, noise_sigma_v = 0.5, noise_sigma_omega = 0.2

        print(
            f"Risk-Aware MPPI (paper): σ={self.pedestrian_uncertainty}m, "
            f"M={self.n_risk_samples}, CP_thr={self.collision_threshold}, "
            f"w_soft={self.w_soft}"
        )

    # ------------------------------------------------------------------
    #  Override MPC solver with collision probability
    # ------------------------------------------------------------------
    def _solve_mpc(
        self, robot_state, target_path, circles, walls
    ) -> Tuple[float, float]:
        x0, y0, th0 = robot_state

        # ---- 1. Sample control noise ----
        if self.use_halton_noise:
            noise = self.noise_gen.sample()
        else:
            sigma = np.array([self.noise_sigma_v, self.noise_sigma_omega])
            noise = np.random.randn(self.num_samples, self.horizon, 2) * sigma

        if not self._warm_start_done:
            self._warm_start_from_path(robot_state, target_path)
            self._warm_start_done = True

        # ---- 2. Rollout trajectories (identical to MPPI) ----
        traj = mppi_rollout_batch(
            x0,
            y0,
            th0,
            self.U,
            noise,
            self.dt_mpc,
            self.config.max_linear_vel,
            self.config.max_angular_vel,
        )

        K = traj.shape[0]  # number of samples

        # ---- 3. Nominal MPPI cost (unperturbed obstacles) ----
        nominal_costs = mppi_compute_costs(
            traj,
            target_path,
            circles,
            walls,
            self.config.robot_radius,
            self.config.safety_margin,
            self.Q_path,
            self.Q_progress,
            self.Q_terminal,
            self.Q_speed,
            self.Q_heading,
        )

        # ---- 4. Estimate collision probability (CP) for each trajectory ----
        cp = np.zeros(K)  # collision probability per trajectory
        safe_dist = self.config.robot_radius + self.config.safety_margin

        # Only create noise once per risk sample for all trajectories? We'll loop.
        for r in range(self.n_risk_samples):
            # Perturb circle positions
            if circles.shape[0] > 0:
                perturbed = circles.copy()
                perturbed[:, :2] += (
                    np.random.randn(circles.shape[0], 2) * self.pedestrian_uncertainty
                )
            else:
                perturbed = circles

            # For each trajectory, check collision in this perturbed world
            for k in range(K):
                # Extract trajectory (H+1 steps)
                traj_k = traj[k]
                if _check_collision_for_trajectory(
                    traj_k,
                    perturbed,
                    self.config.robot_radius,
                    self.config.safety_margin,
                ):
                    cp[k] += 1.0

        cp /= self.n_risk_samples  # in [0, 1]

        # ---- 5. Modified cost = nominal cost + soft penalty + hard rejection ----
        costs = np.empty(K)
        for k in range(K):
            if cp[k] > self.collision_threshold:
                costs[k] = 1e12  # effectively infinite
            else:
                costs[k] = nominal_costs[k] + self.w_soft * cp[k]

        # ---- 6. Importance sampling weights ----
        min_cost = np.min(costs)
        exp_costs = np.exp(-1.0 / self.lam * (costs - min_cost))
        weights = exp_costs / (np.sum(exp_costs) + 1e-10)

        # ---- 7. Control update ----
        weighted_noise = np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)
        self.U = self.U + weighted_noise
        self.U[:, 0] = np.clip(self.U[:, 0], 0.0, self.config.max_linear_vel)
        self.U[:, 1] = np.clip(
            self.U[:, 1], -self.config.max_angular_vel, self.config.max_angular_vel
        )

        u_raw = self.U[0].copy()
        self.U = np.vstack([self.U[1:], np.zeros((1, 2))])
        return u_raw[0], u_raw[1]
