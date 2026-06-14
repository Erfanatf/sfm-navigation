"""Stage 2: Feature extraction (kinematic, path, safety, social)."""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from ...behavior_pipeline.config import PipelineConfig
from ...behavior_pipeline.reporting import PipelineLogger

# --------------------------------------------------------------------------
# Phase 1.1 helpers
# --------------------------------------------------------------------------
def _kinematic_features(grp):
    n = len(grp)
    speed = grp['velocity'].values
    accel = grp['accel_mag'].values
    speed_mean = np.mean(speed)
    speed_max  = np.max(speed)
    speed_std  = np.std(speed)
    p10, p25, p50, p75, p90 = np.percentile(speed, [10,25,50,75,90])
    accel_mean = np.mean(accel)
    accel_max  = np.max(accel)
    accel_std  = np.std(accel)
    if n >= 3:
        dt_vals = np.diff(grp['timestamp'].values)
        daccel_dt = np.diff(accel) / dt_vals
        jerk_mean = np.mean(np.abs(daccel_dt))
        jerk_max  = np.max(np.abs(daccel_dt))
    else:
        jerk_mean = jerk_max = np.nan
    stop_ratio = np.mean(speed < 0.15)
    return pd.Series({
        'speed_mean': speed_mean, 'speed_max': speed_max, 'speed_std': speed_std,
        'speed_p10': p10, 'speed_p25': p25, 'speed_p50': p50, 'speed_p75': p75, 'speed_p90': p90,
        'accel_mean': accel_mean, 'accel_max': accel_max, 'accel_std': accel_std,
        'jerk_mean': jerk_mean, 'jerk_max': jerk_max,
        'stop_ratio': stop_ratio, 'n_points': n
    })

def compute_kinematic_features(df_windows):
    t0 = time.time()
    features = df_windows.groupby('window_id').apply(_kinematic_features, include_groups=False).reset_index()
    win_meta = df_windows.groupby('window_id').agg(
        agent_id=('agent_id', 'first'),
        window_start=('window_start_abs', 'first'),
        window_end=('window_end_abs', 'first')
    ).reset_index()
    features = features.merge(win_meta, on='window_id', how='left')
    print(f"Kinematic features computed in {time.time()-t0:.1f}s")
    return features

# --------------------------------------------------------------------------
# Phase 1.2 helpers
# --------------------------------------------------------------------------
def _compute_curvature(x, y):
    if len(x) < 3: return np.array([np.nan])
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    denom = (dx**2 + dy**2) ** 1.5
    curv = np.divide(num, denom, where=denom > 1e-10, out=np.full_like(num, np.nan))
    return curv

def _spectral_arc_length(x, y, fs=1.0):
    v = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    if len(v) < 2: return np.nan
    fft = np.fft.rfft(v)
    amp = np.abs(fft) / len(v)
    if amp[0] > 1e-10: amp = amp / amp[0]
    freq = np.fft.rfftfreq(len(v), d=1/fs)
    sal = -np.sum(np.sqrt(np.diff(freq)**2 + np.diff(amp)**2))
    return sal

def _higuchi_fd(x, y, kmax=10):
    n = len(x)
    if n < kmax + 2: return np.nan
    Lk = []
    for k in range(1, kmax+1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, n-1, k)
            if len(idx) < 2: continue
            L = np.sum(np.sqrt((x[idx+1]-x[idx])**2 + (y[idx+1]-y[idx])**2))
            Lm.append(L * (n-1) / (k * len(idx)))
        if Lm: Lk.append(np.mean(Lm))
    if len(Lk) < 2: return np.nan
    xfit = np.log(1/np.arange(1, len(Lk)+1))
    yfit = np.log(Lk)
    return np.polyfit(xfit, yfit, 1)[0]

def _path_features(grp):
    x, y = grp['pos_x'].values, grp['pos_y'].values
    n = len(x)
    dx, dy = np.diff(x), np.diff(y)
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths)
    displacement = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
    if path_length > 1.5 and displacement > 0.01:
        tortuosity = path_length / displacement
    else:
        tortuosity = np.nan
    if path_length > 1.5 and n >= 3:
        curv = _compute_curvature(x, y)
        curv_clean = curv[~np.isnan(curv)]
        mean_curvature = np.mean(curv_clean) if len(curv_clean) else np.nan
        max_curvature = np.max(curv_clean) if len(curv_clean) else np.nan
    else:
        mean_curvature = max_curvature = np.nan
    if n >= 3 and path_length > 1.5:
        angles = np.arctan2(dy, dx)
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2*np.pi) - np.pi
        sinuosity = np.std(angle_diffs)
        mean_step_angle = np.mean(np.abs(angle_diffs))
        step_angle_std = np.std(np.abs(angle_diffs))
    else:
        sinuosity = mean_step_angle = step_angle_std = np.nan
    dt = np.median(np.diff(grp['timestamp_rel'])) if 'timestamp_rel' in grp.columns else 0.04
    fs = 1.0 / dt if dt > 0 else 1.0
    sal = _spectral_arc_length(x, y, fs) if path_length > 1.5 and n >= 3 else np.nan
    fd = _higuchi_fd(x, y, kmax=min(10, n//3)) if path_length > 1.5 and n >= 15 else np.nan
    return pd.Series({
        'path_length': path_length, 'displacement': displacement, 'tortuosity': tortuosity,
        'mean_curvature': mean_curvature, 'max_curvature': max_curvature,
        'sinuosity': sinuosity, 'spectral_arc_length': sal, 'fractal_dim': fd,
        'mean_step_angle': mean_step_angle, 'step_angle_std': step_angle_std
    })

def compute_path_features(df_windows):
    t0 = time.time()
    features = df_windows.groupby('window_id').apply(_path_features, include_groups=False).reset_index()
    meta = df_windows.groupby('window_id').agg(agent_id=('agent_id', 'first')).reset_index()
    features = features.merge(meta, on='window_id', how='left')
    print(f"Path features computed in {time.time()-t0:.1f}s")
    return features

# --------------------------------------------------------------------------
# Phase 1.3 helpers
# --------------------------------------------------------------------------
def _compute_ttc(p1, v1, p2, v2):
    rel_pos = p2 - p1
    rel_vel = v2 - v1
    a = np.sum(rel_vel**2, axis=-1)
    b = 2 * np.sum(rel_pos * rel_vel, axis=-1)
    c = np.sum(rel_pos**2, axis=-1)
    disc = b**2 - 4*a*c
    ttc = np.full_like(a, np.inf)
    pos = (disc > 0) & (a > 1e-10)
    sqrt_disc = np.sqrt(np.where(pos, disc, 0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    for t in [t1, t2]:
        mask = (t > 0) & np.isfinite(t)
        ttc[mask] = np.minimum(ttc[mask], t[mask])
    return ttc

def compute_safety_features(df_windows, df_raw):
    global_ids = df_windows['agent_id'].unique()
    df_global = df_raw[df_raw['agent_id'].isin(global_ids)].copy()
    df_global['vx'] = df_global['velocity'] * np.cos(df_global['motion_angle_rad'])
    df_global['vy'] = df_global['velocity'] * np.sin(df_global['motion_angle_rad'])
    df_global = df_global.sort_values(['agent_id', 'timestamp'])
    df_global['dt'] = df_global.groupby('agent_id')['timestamp'].diff()
    df_global['ax'] = df_global.groupby('agent_id')['vx'].diff() / df_global['dt']
    df_global['ay'] = df_global.groupby('agent_id')['vy'].diff() / df_global['dt']
    df_global['accel_mag'] = np.sqrt(df_global['ax']**2 + df_global['ay']**2)
    df_global[['ax','ay','accel_mag']] = df_global.groupby('agent_id')[['ax','ay','accel_mag']].bfill()
    df_global = df_global.dropna(subset=['vx','vy','accel_mag'])
    agent_dfs = {ag: grp.sort_values('timestamp').reset_index(drop=True)
                 for ag, grp in df_global.groupby('agent_id')}

    window_meta = df_windows.groupby('window_id').agg(
        agent_id=('agent_id', 'first'),
        window_start_abs=('window_start_abs', 'first'),
        window_end_abs=('window_end_abs', 'first')
    ).reset_index()

    features_list = []
    for _, row in window_meta.iterrows():
        win_id, ego_id, t0, t1 = row['window_id'], row['agent_id'], row['window_start_abs'], row['window_end_abs']
        ego_df = agent_dfs[ego_id]
        ego_mask = (ego_df['timestamp'] >= t0) & (ego_df['timestamp'] <= t1)
        ego_win = ego_df[ego_mask]
        if ego_win.empty: continue
        ego_pos = ego_win[['pos_x', 'pos_y']].values
        ego_vel = ego_win[['vx', 'vy']].values
        ego_times = ego_win['timestamp'].values
        N = len(ego_times)

        min_dist, min_ttc = np.inf, np.inf
        intrusion_frames = 0
        n_neighbors = 0
        intrusions = []
        close_mask = np.zeros(N, dtype=bool)

        for other_id, other_df in agent_dfs.items():
            if other_id == ego_id: continue
            other_mask = (other_df['timestamp'] >= t0) & (other_df['timestamp'] <= t1)
            if other_mask.sum() < 2: continue
            other_win = other_df[other_mask]
            other_times = other_win['timestamp'].values
            other_x = np.interp(ego_times, other_times, other_win['pos_x'].values)
            other_y = np.interp(ego_times, other_times, other_win['pos_y'].values)
            other_vx = np.interp(ego_times, other_times, other_win['vx'].values)
            other_vy = np.interp(ego_times, other_times, other_win['vy'].values)
            other_pos = np.column_stack([other_x, other_y])
            other_vel = np.column_stack([other_vx, other_vy])

            dist = np.sqrt(np.sum((ego_pos - other_pos)**2, axis=1))
            min_dist = min(min_dist, np.min(dist))
            ttc = _compute_ttc(ego_pos, ego_vel, other_pos, other_vel)
            ttc[ttc > (t1 - t0)] = np.inf
            min_ttc = min(min_ttc, np.min(ttc))

            intrude = dist < 0.8
            intrusion_frames += intrude.sum()
            close_mask |= (dist < 1.2)
            n_neighbors += 1

        intrusion_time_frac = intrusion_frames / N if N > 0 else 0.0
        num_intrusions = 0
        if intrusion_frames > 0:
            changes = np.diff(np.concatenate([[0], intrude.astype(int), [0]]))
            num_intrusions = int(np.sum(changes == 1))

        avoidance_deviation = np.nan
        p_start, p_end = ego_pos[0], ego_pos[-1]
        line_vec = p_end - p_start
        line_len = np.linalg.norm(line_vec)
        if line_len > 0.01 and n_neighbors > 0 and np.any(close_mask):
            line_dir = line_vec / line_len
            perp = np.linalg.norm(ego_pos - p_start - np.outer(np.dot(ego_pos-p_start, line_dir), line_dir), axis=1)
            avoidance_deviation = np.max(perp[close_mask])

        features_list.append({
            'window_id': win_id,
            'n_neighbors': n_neighbors,
            'min_distance': min_dist,
            'min_TTC': min_ttc,
            'intrusion_time_frac': intrusion_time_frac,
            'num_intrusions': num_intrusions,
            'avoidance_deviation': avoidance_deviation
        })

    features = pd.DataFrame(features_list)
    features = features.merge(window_meta[['window_id', 'agent_id']], on='window_id', how='left')
    features['min_TTC'] = features['min_TTC'].replace(np.inf, np.nan)
    features['min_distance'] = features['min_distance'].replace(np.inf, np.nan)
    print(f"Safety features computed for {len(features)} windows")
    return features

# --------------------------------------------------------------------------
# Phase 1.4 helpers
# --------------------------------------------------------------------------
def _lines_intersect_forward(p1, d1, p2, d2, thresh=0.5):
    A = np.column_stack([d1, -d2])
    b = p2 - p1
    try:
        st = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return False
    s, t = st[0], st[1]
    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    dist = np.linalg.norm(closest1 - closest2)
    return (s > 0) and (t > 0) and (dist < thresh)

def compute_social_features(df_windows, df_raw):
    FOV_HALF = np.deg2rad(60)
    COS_FOV = np.cos(FOV_HALF)
    GROUP_MAX_DIST = 2.0
    MIN_PRESENCE = 0.5

    global_ids = df_windows['agent_id'].unique()
    df_global = df_raw[df_raw['agent_id'].isin(global_ids)].copy()
    df_global['vx'] = df_global['velocity'] * np.cos(df_global['motion_angle_rad'])
    df_global['vy'] = df_global['velocity'] * np.sin(df_global['motion_angle_rad'])
    df_global = df_global.sort_values(['agent_id', 'timestamp'])
    agent_data = {ag: grp.sort_values('timestamp').reset_index(drop=True)
                  for ag, grp in df_global.groupby('agent_id')}

    window_meta = df_windows.groupby('window_id').agg(
        agent_id=('agent_id', 'first'),
        window_start_abs=('window_start_abs', 'first'),
        window_end_abs=('window_end_abs', 'first')
    ).reset_index()

    features_list = []
    for _, row in window_meta.iterrows():
        win_id, ego_id, t0, t1 = row['window_id'], row['agent_id'], row['window_start_abs'], row['window_end_abs']
        ego_df = agent_data[ego_id]
        ego_mask = (ego_df['timestamp'] >= t0) & (ego_df['timestamp'] <= t1)
        ego_win = ego_df[ego_mask]
        if ego_win.empty: continue
        ego_times = ego_win['timestamp'].values
        ego_pos = ego_win[['pos_x', 'pos_y']].values
        ego_facing = ego_win['facing_angle_rad'].values
        ego_fdir = np.column_stack([np.cos(ego_facing), np.sin(ego_facing)])
        N = len(ego_times)
        fov_occ_sum = mutual_sum = ospace_sum = 0.0
        neighbor_stats = {}

        for other_id, other_df in agent_data.items():
            if other_id == ego_id: continue
            other_mask = (other_df['timestamp'] >= t0) & (other_df['timestamp'] <= t1)
            if other_mask.sum() < 2: continue
            other_win = other_df[other_mask]
            other_times = other_win['timestamp'].values
            other_x = np.interp(ego_times, other_times, other_win['pos_x'].values)
            other_y = np.interp(ego_times, other_times, other_win['pos_y'].values)
            other_facing = np.interp(ego_times, other_times, other_win['facing_angle_rad'].values)
            other_pos = np.column_stack([other_x, other_y])
            other_fdir = np.column_stack([np.cos(other_facing), np.sin(other_facing)])

            vec = other_pos - ego_pos
            dist = np.linalg.norm(vec, axis=1)
            dot_ego = np.sum(vec * ego_fdir, axis=1) / (dist + 1e-8)
            in_ego = dot_ego > COS_FOV
            fov_occ_sum += in_ego.sum()

            dot_other = np.sum(-vec * other_fdir, axis=1) / (dist + 1e-8)
            mutual = in_ego & (dot_other > COS_FOV)
            mutual_sum += mutual.sum()

            for i in range(N):
                if dist[i] > 0.1:
                    if _lines_intersect_forward(ego_pos[i], ego_fdir[i], other_pos[i], other_fdir[i]):
                        ospace_sum += 1

            # Group affiliation
            other_raw_times = set(other_win['timestamp'])
            n_present = sum(1 for t in ego_times if t in other_raw_times)
            if n_present / N < MIN_PRESENCE: continue
            indices = [i for i, t in enumerate(ego_times) if t in other_raw_times]
            mean_dist = np.mean(dist[indices])
            if len(indices) < 3: continue
            v_ego = ego_win[['vx', 'vy']].iloc[indices].values
            v_other = np.column_stack([
                np.interp(ego_times[indices], other_times, other_win['vx'].values),
                np.interp(ego_times[indices], other_times, other_win['vy'].values)
            ])
            v_ego_norm = v_ego - v_ego.mean(axis=0)
            v_other_norm = v_other - v_other.mean(axis=0)
            denom = np.std(v_ego_norm) * np.std(v_other_norm)
            corr = np.mean(v_ego_norm * v_other_norm) / denom if denom > 1e-6 else 0.0
            score = max(0, 1 - mean_dist/GROUP_MAX_DIST) * max(0, corr)
            if other_id not in neighbor_stats or score > neighbor_stats[other_id]:
                neighbor_stats[other_id] = score

        fov_occ = fov_occ_sum / N
        mutual_att = mutual_sum / N
        ospace_ratio = ospace_sum / N
        group_aff = max(neighbor_stats.values()) if neighbor_stats else 0.0

        features_list.append({
            'window_id': win_id, 'agent_id': ego_id,
            'fov_occupancy': fov_occ, 'mutual_attention_count': mutual_att,
            'o_space_ratio': ospace_ratio, 'group_affiliation': group_aff
        })

    features = pd.DataFrame(features_list)
    print(f"Social features computed for {len(features)} windows")
    return features

# --------------------------------------------------------------------------
# Main stage function
# --------------------------------------------------------------------------
def run_stage2(config: PipelineConfig, logger: PipelineLogger, df_windows: pd.DataFrame, df_raw: pd.DataFrame):
    logger.info("=== Stage 2: Feature Extraction ===")
    data_dir = os.path.join(logger.output_dir, logger.run_id, "data/stage2")
    plot_dir = os.path.join(logger.output_dir, logger.run_id, "reports/stage2/plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 2.1
    logger.info("  2.1 Kinematic features...")
    f11 = compute_kinematic_features(df_windows)
    f11.to_csv(os.path.join(data_dir, "features_kinematic.csv"), index=False)
    fig = px.scatter(f11, x='speed_mean', y='speed_std', color='stop_ratio',
                     title='Mean speed vs speed variability')
    fig.write_html(os.path.join(plot_dir, "kinematic_speed_scatter.html"))

    # 2.2
    logger.info("  2.2 Path quality features...")
    f12 = compute_path_features(df_windows)
    f12.to_csv(os.path.join(data_dir, "features_path_quality.csv"), index=False)
    valid = f12.dropna(subset=['tortuosity'])
    if not valid.empty:
        fig = px.histogram(valid, x='tortuosity', title='Tortuosity distribution')
        fig.write_html(os.path.join(plot_dir, "path_tortuosity.html"))

    # 2.3
    logger.info("  2.3 Safety/interaction features...")
    f13 = compute_safety_features(df_windows, df_raw)
    f13.to_csv(os.path.join(data_dir, "features_safety.csv"), index=False)

    # 2.4
    logger.info("  2.4 Social attention features...")
    f14 = compute_social_features(df_windows, df_raw)
    f14.to_csv(os.path.join(data_dir, "features_social_attention.csv"), index=False)

    report = "Feature summaries:\n"
    for name, df in [("kinematic", f11), ("path", f12), ("safety", f13), ("social", f14)]:
        report += logger.write_dataframe_summary(df, name)
    logger.write_stage_report("stage2", report)
    logger.info("Stage 2 complete.")
    return f11, f12, f13, f14