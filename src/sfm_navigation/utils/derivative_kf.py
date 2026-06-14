"""Kalman‑filter‑based derivative estimator with command input and decoupled noise."""
import numpy as np


class DerivativeEstimatorKF:
    def __init__(
        self,
        dt: float,
        process_noise_jerk: float = 1.0,
        process_noise_acc: float = 1.0,    # increased from 0.1
        measurement_noise: float = 0.1,
    ):
        self.dt = dt
        self.q_j = process_noise_jerk
        self.q_a = process_noise_acc
        self.R = measurement_noise
        self._init_model(dt)

    def _init_model(self, dt):
        self.dt = dt
        self.x = np.zeros(3)
        self.P = np.eye(3) * 0.1

        self.F = np.array([[1, dt, 0.5 * dt * dt],
                           [0, 1, dt],
                           [0, 0, 1]])
        # No B matrix
        self.H = np.array([[1, 0, 0]])

        G_j = np.array([[0.5 * dt * dt], [dt], [1.0]])
        G_a = np.array([[0.5 * dt * dt], [dt], [0.0]])
        # self.Q = (G_j @ G_j.T) * self.q_j * dt + (G_a @ G_a.T) * self.q_a * dt
        self.Q = (G_j @ G_j.T) * self.q_j + (G_a @ G_a.T) * self.q_a


        self.R_mat = np.eye(1) * self.R
        self.initialised = False

    def set_dt(self, new_dt: float):
        self.dt = new_dt
        self.F = np.array([[1, new_dt, 0.5 * new_dt * new_dt],
                           [0, 1, new_dt],
                           [0, 0, 1]])
        G_j = np.array([[0.5 * new_dt * new_dt], [new_dt], [1.0]])
        G_a = np.array([[0.5 * new_dt * new_dt], [new_dt], [0.0]])
        self.Q = (G_j @ G_j.T) * self.q_j * new_dt + (G_a @ G_a.T) * self.q_a * new_dt

    def update(self, measured_velocity: float):
        if not self.initialised:
            self.x[0] = measured_velocity
            self.initialised = True
            return measured_velocity, 0.0, 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        y = measured_velocity - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R_mat
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(3) - np.outer(K, self.H)) @ self.P
        return self.x[0], self.x[1], self.x[2]

    def reset(self):
        self._init_model(self.dt)