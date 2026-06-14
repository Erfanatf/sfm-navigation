import numpy as np
from scipy.signal import savgol_filter

class KF_CV:
    """Kalman filter with constant velocity model."""
    def __init__(self, dt, init_pos, init_vel=None, process_noise=0.1, measurement_noise=0.5):
        self.dt = dt
        vx0 = init_vel[0] if init_vel else 0.0
        vy0 = init_vel[1] if init_vel else 0.0
        self.x = np.array([init_pos[0], init_pos[1], vx0, vy0])
        self.P = np.eye(4) * 0.1
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x.copy()


class UKF_CV:
    """Unscented Kalman filter with constant velocity model."""
    def __init__(self, dt, init_pos, init_vel=None, process_noise=0.1, measurement_noise=0.5,
                 alpha=1e-3, beta=2, kappa=0):
        self.dt = dt
        vx0 = init_vel[0] if init_vel else 0.0
        vy0 = init_vel[1] if init_vel else 0.0
        self.x = np.array([init_pos[0], init_pos[1], vx0, vy0])
        self.n = 4                          # state dimension
        self.P = np.eye(self.n) * 0.1
        self.Q = np.eye(self.n) * process_noise
        self.R = np.eye(2) * measurement_noise

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n
        self._compute_weights()

    def _compute_weights(self):
        n = self.n
        lam = self.lam
        self.Wm = np.full(2*n+1, 1.0 / (2*(n+lam)))
        self.Wc = np.full(2*n+1, 1.0 / (2*(n+lam)))
        self.Wm[0] = lam / (n+lam)
        self.Wc[0] = lam / (n+lam) + (1 - self.alpha**2 + self.beta)

    def _sigma_points(self):
        n = self.n
        lam = self.lam
        L = np.linalg.cholesky((n+lam) * self.P)
        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = self.x
        for i in range(n):
            sigmas[i+1] = self.x + L[:, i]
            sigmas[i+1+n] = self.x - L[:, i]
        return sigmas

    def predict(self):
        # Update F with current dt
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        sigmas = self._sigma_points()
        sigmas_pred = sigmas @ F.T
        self.x = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)
        diff = sigmas_pred - self.x
        self.P = diff.T @ np.diag(self.Wc) @ diff + self.Q

    def update(self, z):
        sigmas = self._sigma_points()
        z_pred = sigmas[:, :2]   # measurement is position only
        z_mean = np.sum(self.Wm[:, None] * z_pred, axis=0)

        diff_z = z_pred - z_mean
        S = diff_z.T @ np.diag(self.Wc) @ diff_z + self.R

        diff_x = sigmas - self.x
        Pxz = diff_x.T @ np.diag(self.Wc) @ diff_z

        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ (z - z_mean)
        self.P = self.P - K @ S @ K.T

    def get_state(self):
        return self.x.copy()


def filter_agent_trajectory(agent_df, process_noise=0.1, measurement_noise=0.5,
                            savgol_window=9, savgol_order=2, method='KF'):
    """
    Apply KF/UKF + Savitzky‑Golay smoothing to an agent's trajectory.

    Parameters
    ----------
    agent_df : pd.DataFrame
        Must contain columns 'timestamp_rel', 'pos_x', 'pos_y'.
    method : str
        'KF' for Kalman filter, 'UKF' for Unscented Kalman filter.
    Returns
    -------
    pd.DataFrame with filtered positions, velocities, and motion angle.
    """
    times = agent_df['timestamp_rel'].values
    pos_x = agent_df['pos_x'].values
    pos_y = agent_df['pos_y'].values
    if len(times) < 2:
        return agent_df

    dt0 = times[1] - times[0] if len(times) > 1 else 0.04
    vx0 = (pos_x[1] - pos_x[0]) / dt0 if len(times) > 1 else 0.0
    vy0 = (pos_y[1] - pos_y[0]) / dt0 if len(times) > 1 else 0.0

    # Select filter
    FilterClass = KF_CV if method.upper() == 'KF' else UKF_CV
    filter_obj = FilterClass(dt=dt0, init_pos=(pos_x[0], pos_y[0]), init_vel=(vx0, vy0),
                             process_noise=process_noise, measurement_noise=measurement_noise)

    filtered_pos_x = [pos_x[0]]
    filtered_pos_y = [pos_y[0]]
    filtered_vx = [vx0]
    filtered_vy = [vy0]

    for i in range(1, len(times)):
        dt = times[i] - times[i-1] if times[i] > times[i-1] else 0.04
        # Update the filter's dt (KF and UKF use self.dt)
        if hasattr(filter_obj, 'F'):                # KF
            filter_obj.F[0, 2] = dt
            filter_obj.F[1, 3] = dt
        else:                                       # UKF
            filter_obj.dt = dt
        filter_obj.predict()
        filter_obj.update(np.array([pos_x[i], pos_y[i]]))
        state = filter_obj.get_state()
        filtered_pos_x.append(state[0])
        filtered_pos_y.append(state[1])
        filtered_vx.append(state[2])
        filtered_vy.append(state[3])

    # Savitzky‑Golay smoothing on the filtered positions
    if len(filtered_pos_x) >= savgol_window:
        filtered_pos_x = savgol_filter(filtered_pos_x, savgol_window, savgol_order)
        filtered_pos_y = savgol_filter(filtered_pos_y, savgol_window, savgol_order)

    # Build filtered DataFrame
    df_filtered = agent_df.copy()
    df_filtered['pos_x'] = filtered_pos_x
    df_filtered['pos_y'] = filtered_pos_y
    df_filtered['velocity'] = np.sqrt(np.array(filtered_vx)**2 + np.array(filtered_vy)**2)
    df_filtered['motion_angle_rad'] = np.arctan2(np.array(filtered_vy), np.array(filtered_vx))
    return df_filtered