import numpy as np
import pandas as pd
import glob
from scipy.signal import savgol_filter

SAVGOL_PARAMS = {
    'pos': {'window_length': 61, 'polyorder': 1},
    'vel': {'window_length': 61, 'polyorder': 2},
    'yaw': {'window_length': 61, 'polyorder': 2}
}

def safe_savgol_filter(data, window_length, polyorder):
    """Safely applies SG filter, ensuring window is odd and smaller than data length."""
    W = min(window_length, len(data) - 1)
    if W < 3:
        return data
    if W % 2 == 0:
        W -= 1
    if W < polyorder + 1:
        W = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2
    if W >= len(data):
        W = len(data) - 1 if (len(data) - 1) % 2 != 0 else len(data) - 2
    if W < 3:
        return data
    return savgol_filter(data, W, polyorder)


def load_and_process_trajectory(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, None
    df.dropna(inplace=True)
    df.sort_values(by='time', inplace=True)
    if df.empty:
        return None, None

    t = df['time'].values
    dt_series = np.diff(t)
    avg_dt = np.mean(dt_series)

    # --- 1. Load raw position and velocity data ---
    px_raw, py_raw = df['pos_x'].values / 1000.0, df['pos_y'].values / 1000.0
    yaw_raw, v_total_raw = df['facing_angle'].values, df['velocity'].values / 1000.0
    motion_angle_raw = df['motion_angle'].values

    relative_motion_angle = motion_angle_raw - yaw_raw
    v_forw_raw = v_total_raw * np.cos(relative_motion_angle)
    v_orth_raw = v_total_raw * np.sin(relative_motion_angle)

    # --- 2. Apply Savitzky-Golay filter ---
    px = safe_savgol_filter(px_raw, **SAVGOL_PARAMS['pos'])
    py = safe_savgol_filter(py_raw, **SAVGOL_PARAMS['pos'])
    yaw_unwrapped = np.unwrap(yaw_raw)
    yaw = safe_savgol_filter(yaw_unwrapped, **SAVGOL_PARAMS['yaw'])
    v_forw = safe_savgol_filter(v_forw_raw, **SAVGOL_PARAMS['vel'])
    v_orth = safe_savgol_filter(v_orth_raw, **SAVGOL_PARAMS['vel'])

    # --- 3. Calculate Derivatives ---
    a_forw = np.gradient(v_forw, t)
    a_orth = np.gradient(v_orth, t)

    # --- 4. Stack 9-Feature State Vector ---
    x_true = np.stack([
        px, py, yaw,          # Position (0, 1, 2)
        v_forw, v_orth,       # Velocity (3, 4)
    ], axis=1)

    return x_true, avg_dt


def load_multiple_trajectories(data_folder, file_pattern='person_*.csv', n=None):
    """Load multiple trajectory files from a folder, returning all_x_true, all_dt, AVG_DT."""
    all_file_paths = sorted(glob.glob(data_folder + file_pattern))
    if n is not None:
        all_file_paths = all_file_paths[:n]
    if len(all_file_paths) == 0:
        raise FileNotFoundError(f"No files found in {data_folder} with pattern {file_pattern}")

    all_x_true, all_dt = [], []
    for path in all_file_paths:
        x_true, avg_dt = load_and_process_trajectory(path)
        if x_true is not None:
            all_x_true.append(x_true)
            all_dt.append(avg_dt)

    if not all_x_true:
        raise ValueError('No valid trajectories found')
    AVG_DT = np.mean(all_dt)
    return all_x_true, all_dt, AVG_DT