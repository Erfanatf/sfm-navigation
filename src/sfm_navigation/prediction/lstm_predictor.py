import time, numpy as np
from collections import deque
import tensorflow as tf
import pickle

def physics_integration_step(prev_state, dyn_delta, dt):
    px, py, yaw, vf, vo = prev_state
    d_yaw, d_vf, d_vo = dyn_delta
    vf_next = vf + d_vf
    vo_next = vo + d_vo
    yaw_next = yaw + d_yaw
    vx = vf_next * np.cos(yaw_next) - vo_next * np.sin(yaw_next)
    vy = vf_next * np.sin(yaw_next) + vo_next * np.cos(yaw_next)
    px_next = px + vx * dt
    py_next = py + vy * dt
    return np.array([px_next, py_next, yaw_next, vf_next, vo_next], dtype=np.float32)


class LSTMPredictor:
    def __init__(self, model_path, state_scaler_path, delta_scaler_path,
                 seq_length=30, n_steps_ahead=30, dt=0.0368):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(state_scaler_path, 'rb') as f:
            self.state_scaler = pickle.load(f)
        with open(delta_scaler_path, 'rb') as f:
            self.delta_scaler = pickle.load(f)
        self.seq_length = seq_length
        self.n_steps_ahead = n_steps_ahead
        self.dt = dt                  # ATC average dt

        self.buffer = deque(maxlen=seq_length)
        self._initialized = False

    def initialize_buffer(self, user_traj, start_time=0.0):
        """Pre‑fill the buffer with the first 30 user states from the trajectory."""
        self.buffer.clear()
        for i in range(self.seq_length):
            t = start_time + i * self.dt
            state = user_traj.get_state_at_time(t)   # [x, y, yaw, v_forw, v_orth]
            self.buffer.append(state.copy())
        self._initialized = True

    def predict(self, user_state):
        """
        user_state : np.array of shape (5,)  [x, y, yaw, v_forw, v_orth]

        Returns
        -------
        goal_x, goal_y : float   estimated position 2 s ahead
        pred_time : float        computation time [s]
        """
        t_start = time.perf_counter()

        # Always add the latest observation
        self.buffer.append(user_state.copy())

        # If buffer not full yet, fallback to constant velocity
        if len(self.buffer) < self.seq_length:
            vx = (user_state[3]*np.cos(user_state[2]) - user_state[4]*np.sin(user_state[2]))
            vy = (user_state[3]*np.sin(user_state[2]) + user_state[4]*np.cos(user_state[2]))
            goal_x = user_state[0] + vx * 2.0
            goal_y = user_state[1] + vy * 2.0
            pred_time = time.perf_counter() - t_start
            return goal_x, goal_y, pred_time

        # LSTM prediction
        seq = np.array(self.buffer, dtype=np.float32).reshape(1, self.seq_length, 5)
        seq_scaled = self.state_scaler.transform(seq.reshape(-1,5)).reshape(1, self.seq_length, 5)

        with tf.device('/cpu:0'):
            pred_deltas_scaled = self.model(seq_scaled).numpy()
        pred_deltas_scaled = pred_deltas_scaled.reshape(self.n_steps_ahead, 3)
        pred_deltas = self.delta_scaler.inverse_transform(pred_deltas_scaled)

        # Integrate future trajectory
        cur_state = user_state.copy()
        for i in range(self.n_steps_ahead):
            cur_state = physics_integration_step(cur_state, pred_deltas[i], self.dt)

        # Extrapolate from the final predicted state to 2.0 s
        t_pred = self.n_steps_ahead * self.dt       # ~1.1 s
        remaining = 2.0 - t_pred
        if remaining > 0:
            vx = cur_state[3]*np.cos(cur_state[2]) - cur_state[4]*np.sin(cur_state[2])
            vy = cur_state[3]*np.sin(cur_state[2]) + cur_state[4]*np.cos(cur_state[2])
            goal_x = cur_state[0] + vx * remaining
            goal_y = cur_state[1] + vy * remaining
        else:
            goal_x, goal_y = cur_state[0], cur_state[1]

        pred_time = time.perf_counter() - t_start
        return goal_x, goal_y, pred_time