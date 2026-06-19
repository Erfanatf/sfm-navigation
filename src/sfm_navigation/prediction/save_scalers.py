"""Save the state and delta scalers fitted on the ATC training data."""
import glob, pickle, numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def safe_savgol_filter(data, w, p):
    # minimal version – use the same function from your notebook
    from scipy.signal import savgol_filter
    W = min(w, len(data)-1)
    if W < 3: return data
    if W%2==0: W-=1
    if W<p+1: W=p+1 if (p+1)%2 else p+2
    if W>=len(data): W=len(data)-1 if (len(data)-1)%2 else len(data)-2
    if W<3: return data
    return savgol_filter(data, W, p)

def main():
    DRIVE_DATA_FOLDER = CONFIG.atc_csv_folder
    file_paths = sorted(glob.glob(DRIVE_DATA_FOLDER + '*.csv'))[10:24]
    all_x_true = []
    for path in file_paths:
        df = pd.read_csv(path).dropna().sort_values('time')
        px = safe_savgol_filter(df['pos_x']/1000.0, 61,1)
        py = safe_savgol_filter(df['pos_y']/1000.0, 61,1)
        yaw_u = np.unwrap(df['facing_angle'])
        yaw = safe_savgol_filter(yaw_u, 61,2)
        vf_raw = df['velocity']/1000.0 * np.cos(df['motion_angle']-df['facing_angle'])
        vo_raw = df['velocity']/1000.0 * np.sin(df['motion_angle']-df['facing_angle'])
        vf = safe_savgol_filter(vf_raw, 61,2)
        vo = safe_savgol_filter(vo_raw, 61,2)
        state = np.stack([px, py, yaw, vf, vo], axis=1)
        all_x_true.append(state)
    x_full = np.vstack(all_x_true)
    dyn_idx = [2,3,4]
    state_scaler = StandardScaler().fit(x_full)
    deltas = x_full[1:, dyn_idx] - x_full[:-1, dyn_idx]
    delta_scaler = StandardScaler().fit(deltas)

    # Save
    import pickle as pkl
    with open('src/sfm_navigation/prediction/state_scaler.pkl', 'wb') as f:
        pkl.dump(state_scaler, f)
    with open('src/sfm_navigation/prediction/delta_scaler.pkl', 'wb') as f:
        pkl.dump(delta_scaler, f)
    print("Scalers saved.")

if __name__ == '__main__':
    main()