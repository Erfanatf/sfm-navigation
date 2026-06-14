"""MPPI noise generator using Halton sequences + spline smoothing (pre‑generated pool)."""
import numpy as np
from scipy.special import erfinv
from scipy.interpolate import CubicSpline


def _halton_sequence(n, base):
    """Generate the first `n` Halton numbers for the given prime base."""
    seq = np.zeros(n)
    for i in range(n):
        x = 0.0
        f = 1.0 / base
        idx = i + 1
        while idx > 0:
            digit = idx % base
            x += digit * f
            idx //= base
            f /= base
        seq[i] = x
    return seq


class NoiseGenerator:
    """
    Pre‑generate a large pool of Halton‑spline noise sequences and
    return random mini‑batches at each control step.
    """

    def __init__(self, num_samples_per_step, horizon, sigma_v, sigma_omega,
                 pool_size=20000, n_keypoints=5, bases=(2, 3)):
        """
        Parameters
        ----------
        num_samples_per_step : int   (K) number of trajectories per step
        horizon : int                (H) planning horizon
        sigma_v : float              standard deviation for linear velocity
        sigma_omega : float          standard deviation for angular velocity
        pool_size : int              total number of noise sequences to generate
        n_keypoints : int            number of spline knots per sequence
        bases : tuple                Halton bases for v and ω
        """
        self.K = num_samples_per_step
        self.H = horizon
        self.sigma = np.array([sigma_v, sigma_omega])
        self.pool_size = pool_size

        # ── generate the full pool of Halton‑spline sequences ──
        total_points = pool_size * n_keypoints
        halton_v = _halton_sequence(total_points, bases[0])
        halton_w = _halton_sequence(total_points, bases[1])

        # map [0,1) → N(0,1) via inverse CDF
        eps = 1e-10
        halton_v = np.clip(halton_v, eps, 1.0 - eps)
        halton_w = np.clip(halton_w, eps, 1.0 - eps)
        normal_v = np.sqrt(2) * erfinv(2 * halton_v - 1)
        normal_w = np.sqrt(2) * erfinv(2 * halton_w - 1)

        key_t = np.linspace(0, horizon - 1, n_keypoints)

        # pre‑allocate pool (pool_size, H, 2)
        self.pool = np.zeros((pool_size, horizon, 2))

        for i in range(pool_size):
            start = i * n_keypoints
            kp_v = normal_v[start : start + n_keypoints] * sigma_v
            kp_w = normal_w[start : start + n_keypoints] * sigma_omega
            spline_v = CubicSpline(key_t, kp_v, bc_type='natural')
            spline_w = CubicSpline(key_t, kp_w, bc_type='natural')
            t_all = np.arange(horizon)
            self.pool[i, :, 0] = spline_v(t_all)
            self.pool[i, :, 1] = spline_w(t_all)

    def sample(self, num_samples=None):
        """
        Return a (num_samples, H, 2) noise tensor by randomly selecting
        sequences from the pre‑generated pool.
        """
        if num_samples is None:
            num_samples = self.K
        indices = np.random.choice(self.pool_size, size=num_samples, replace=True)
        return self.pool[indices]