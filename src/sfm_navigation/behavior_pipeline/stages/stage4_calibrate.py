"""Stage 4: Extended SFM Calibration for a regime."""

import os
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d  
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from ...behavior_pipeline.config import PipelineConfig
from ...behavior_pipeline.reporting import PipelineLogger

# ------------------------------------------------------------
# Feature extraction from trajectory (as in Phase 3c)
# ------------------------------------------------------------
def _compute_curvature(x, y):
    if len(x) < 3: return np.array([np.nan])
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    denom = (dx**2 + dy**2)**1.5
    curv = np.divide(num, denom, where=denom > 1e-10, out=np.full_like(num, np.nan))
    return curv[~np.isnan(curv)]

def extract_features_from_traj(pos, times, neighbours_pos):
    N = len(pos)
    if N < 5: return None
    dt = np.diff(times)
    if len(dt) == 0: return None
    vel = np.diff(pos, axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0) / dt[:-1, None]
    acc_mag = np.linalg.norm(acc, axis=1) if len(acc) > 0 else np.array([0.])
    speed_mean = np.mean(speed)
    speed_std = np.std(speed)
    stop_ratio = np.mean(speed < 0.15)
    accel_mean = np.mean(acc_mag)
    accel_std = np.std(acc_mag)
    if len(acc_mag) >= 2:
        jerk = np.diff(acc_mag) / dt[1:-1]
        jerk_mean = np.mean(np.abs(jerk))
    else:
        jerk_mean = 0.0
    displacement = np.linalg.norm(pos[-1] - pos[0])
    path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    tortuosity = path_len / displacement if displacement > 0.1 else 1.0
    curv_vals = _compute_curvature(pos[:,0], pos[:,1])
    mean_curvature = np.mean(curv_vals) if len(curv_vals) > 0 else 0.0
    angles = np.arctan2(np.diff(pos[:,1]), np.diff(pos[:,0]))
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2*np.pi) - np.pi
    sinuosity = np.std(angle_diffs) if len(angle_diffs) > 0 else 0.0
    min_dist = np.inf
    intrusion_frames = 0
    for nb_pos in neighbours_pos:
        if nb_pos.shape[0] != N: continue
        dists = np.linalg.norm(pos - nb_pos, axis=1)
        min_dist = min(min_dist, np.min(dists))
        intrusion_frames += np.sum(dists < 0.8)
    intrusion_time_frac = intrusion_frames / N if N > 0 else 0.0
    p_start, p_end = pos[0], pos[-1]
    line_vec = p_end - p_start
    line_len = np.linalg.norm(line_vec)
    avoidance_deviation = 0.0
    if line_len > 0.1 and neighbours_pos:
        line_dir = line_vec / line_len
        perp = np.linalg.norm(pos - p_start - np.outer(np.dot(pos - p_start, line_dir), line_dir), axis=1)
        close_mask = np.zeros(N, dtype=bool)
        for nb_pos in neighbours_pos:
            dist = np.linalg.norm(pos - nb_pos, axis=1)
            close_mask |= (dist < 1.2)
        if np.any(close_mask):
            avoidance_deviation = np.max(perp[close_mask])
    fov_occupancy = 0.0
    headings = np.arctan2(vel[:,1], vel[:,0])
    headings = np.insert(headings, 0, headings[0])
    for t_idx in range(N):
        heading = headings[t_idx]
        fov_dir = np.array([np.cos(heading), np.sin(heading)])
        nb_in_fov = 0
        for nb_pos in neighbours_pos:
            if nb_pos.shape[0] != N: continue
            vec = nb_pos[t_idx] - pos[t_idx]
            dist = np.linalg.norm(vec)
            if dist < 1e-6: continue
            ang = np.arccos(np.clip(np.dot(vec, fov_dir) / dist, -1, 1))
            if ang < np.deg2rad(90):
                nb_in_fov += 1
        fov_occupancy += nb_in_fov
    fov_occupancy /= N
    return {
        'speed_mean': speed_mean, 'speed_std': speed_std, 'stop_ratio': stop_ratio,
        'accel_mean': accel_mean, 'accel_std': accel_std, 'jerk_mean': jerk_mean,
        'tortuosity': tortuosity, 'mean_curvature': mean_curvature, 'sinuosity': sinuosity,
        'min_distance': min_dist, 'intrusion_time_frac': intrusion_time_frac,
        'avoidance_deviation': avoidance_deviation, 'fov_occupancy': fov_occupancy,
        'n_neighbors': len(neighbours_pos)
    }

# ------------------------------------------------------------
# Extended SFM simulator (Phase 3c)
# ------------------------------------------------------------
def simulate_extended_sfm(params, ego_init_state, waypoints, neighbours_interp, dt, T):
    """
    neighbours_interp : list of (N_steps, 2) arrays, already interpolated onto
                        the simulation time grid np.arange(0, T+dt, dt).
    """
    (v0, tau, A, B, lam_base, phi_fov,
     kappa, k_group, r_group, theta_gaze, w_att, fov_att) = params
    x, y, vx, vy, theta = ego_init_state

    n_steps = int(T / dt) + 1
    # Pre‑allocate trajectory array for speed
    pos = np.empty((n_steps, 2))
    pos[0] = (x, y)

    current_wp_idx = 0
    n_wp = len(waypoints)

    for step in range(1, n_steps):
        # ---- waypoint goal ----
        if current_wp_idx < n_wp:
            gx, gy = waypoints[current_wp_idx]
            if np.hypot(gx - x, gy - y) < 0.3 and current_wp_idx < n_wp - 1:
                current_wp_idx += 1
                gx, gy = waypoints[current_wp_idx]
        else:
            gx, gy = waypoints[-1]
        dx_goal = gx - x
        dy_goal = gy - y
        dist_goal = np.hypot(dx_goal, dy_goal)
        desired_dir = np.arctan2(dy_goal, dx_goal) if dist_goal > 0.1 else theta
        desired_vx = v0 * np.cos(desired_dir)
        desired_vy = v0 * np.sin(desired_dir)
        ax = (desired_vx - vx) / tau
        ay = (desired_vy - vy) / tau

        # ---- curvature ----
        speed = np.hypot(vx, vy)
        if speed > 0.2:
            perp_x = -np.sin(theta)
            perp_y = np.cos(theta)
            lateral_acc = kappa * speed * speed
            ax += lateral_acc * perp_x
            ay += lateral_acc * perp_y

        # ---- neighbour repulsion (pre‑interpolated) ----
        nb_positions = [nb[step] for nb in neighbours_interp]   # each nb is (N_steps,2)
        for nx, ny in nb_positions:
            dx = x - nx
            dy = y - ny
            dist = np.hypot(dx, dy)
            if dist < 1e-6:
                dist = 1e-6
            force_dir_x = dx / dist
            force_dir_y = dy / dist
            angle_to_neigh = np.arctan2(ny - y, nx - x)
            phi_ij = angle_to_neigh - theta
            phi_ij = (phi_ij + np.pi) % (2*np.pi) - np.pi
            if np.abs(phi_ij) > phi_fov:
                continue
            w_base = lam_base + (1 - lam_base) * (1 + np.cos(phi_ij)) / 2.0
            gaze_dir = theta + theta_gaze
            phi_att = angle_to_neigh - gaze_dir
            phi_att = (phi_att + np.pi) % (2*np.pi) - np.pi
            w_eff = w_base * (1 + w_att) if np.abs(phi_att) < fov_att else w_base
            force_mag = A * np.exp(-dist / B) * w_eff
            ax += force_mag * force_dir_x
            ay += force_mag * force_dir_y

        # ---- group cohesion ----
        group_nx, group_ny = [], []
        for nx, ny in nb_positions:
            if np.hypot(x - nx, y - ny) < 2.0:
                group_nx.append(nx)
                group_ny.append(ny)
        if group_nx:
            cx = np.mean(group_nx)
            cy = np.mean(group_ny)
            d_cent = np.hypot(cx - x, cy - y)
            if d_cent > 0.01:
                force_mag_group = k_group * (d_cent - r_group)
                ax += force_mag_group * (cx - x) / d_cent
                ay += force_mag_group * (cy - y) / d_cent

        # ---- Euler integration ----
        vx += ax * dt
        vy += ay * dt
        speed = np.hypot(vx, vy)
        max_speed = v0 * 2.5
        if speed > max_speed:
            vx = vx / speed * max_speed
            vy = vy / speed * max_speed
        x += vx * dt
        y += vy * dt
        if speed > 0.2:
            theta = np.arctan2(vy, vx)
        pos[step] = (x, y)

    return pos

# ------------------------------------------------------------
# Generate waypoints from real trajectory
# ------------------------------------------------------------
def generate_waypoints(ego_times, ego_pos, num_waypoints=5):
    if len(ego_pos) < num_waypoints:
        return [ego_pos[-1]]
    indices = np.linspace(0, len(ego_pos)-1, num_waypoints, dtype=int)
    indices[-1] = len(ego_pos)-1
    return ego_pos[indices].tolist()

# ------------------------------------------------------------
# Loss function
# ------------------------------------------------------------
def regime_feature_loss_ext(params, windows_subset, real_stats, dt_sim):
    weights = {
        'speed_mean': 1.0, 'speed_std': 0.5, 'stop_ratio': 1.0,
        'accel_mean': 0.5, 'accel_std': 0.5, 'jerk_mean': 0.3,
        'tortuosity': 2.0, 'mean_curvature': 1.0, 'sinuosity': 1.0,
        'min_distance': 2.0, 'intrusion_time_frac': 2.0, 'avoidance_deviation': 1.0,
        'fov_occupancy': 0.5, 'n_neighbors': 0.0,
    }
    total_loss = 0.0; count = 0
    for w in windows_subset:
        # ---------- Pre‑interpolate neighbours for the simulator ----------
        ego_times = w['ego_times']
        T = ego_times[-1] - ego_times[0]
        n_steps = int(T / dt_sim) + 1
        sim_times = np.linspace(0, T, n_steps)

        neighbours_interp_sim = []
        for nb in w['neighbours_raw']:
            nb_t = nb['times']
            nb_p = nb['pos']
            if len(nb_t) < 2:
                # constant position if only one data point
                x_arr = np.full(n_steps, nb_p[0, 0])
                y_arr = np.full(n_steps, nb_p[0, 1])
            else:
                x_arr = np.interp(sim_times, nb_t - nb_t[0], nb_p[:, 0])
                y_arr = np.interp(sim_times, nb_t - nb_t[0], nb_p[:, 1])
            neighbours_interp_sim.append(np.column_stack([x_arr, y_arr]))

        # ---------- Simulate ----------
        dt0 = ego_times[1] - ego_times[0]
        vx0 = (w['ego_pos'][1, 0] - w['ego_pos'][0, 0]) / dt0
        vy0 = (w['ego_pos'][1, 1] - w['ego_pos'][0, 1]) / dt0
        theta0 = np.arctan2(vy0, vx0)
        init_state = (w['ego_pos'][0, 0], w['ego_pos'][0, 1], vx0, vy0, theta0)
        waypoints = generate_waypoints(ego_times, w['ego_pos'], 5)

        sim_traj = simulate_extended_sfm(params, init_state, waypoints,
                                         neighbours_interp_sim, dt_sim, T)
        if sim_traj is None or len(sim_traj) < 2:
            total_loss += 5.0
            count += 1
            continue

        # ---------- Map back to ego time grid (using np.interp) ----------
        sim_x = np.interp(ego_times - ego_times[0], sim_times, sim_traj[:, 0])
        sim_y = np.interp(ego_times - ego_times[0], sim_times, sim_traj[:, 1])
        sim_pos = np.column_stack([sim_x, sim_y])

        feats = extract_features_from_traj(sim_pos, ego_times, w['neighbours_interp'])
        if feats is None:
            total_loss += 5.0; count += 1; continue
        for key in real_stats:
            if key in weights and key in feats:
                total_loss += weights[key] * np.abs(feats[key] - real_stats[key]['mean']) / (real_stats[key]['std'] + 1e-6)
        count += 1
    return total_loss / count if count > 0 else 1e6

# ------------------------------------------------------------
# Build calibration data
# ------------------------------------------------------------
def _build_calib_data(df_windows, regime_labels, df_raw, config):
    # Prepare global agent dictionary
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

    windows_meta = df_windows.groupby('window_id').agg(
        agent_id=('agent_id', 'first'),
        window_start_abs=('window_start_abs', 'first'),
        window_end_abs=('window_end_abs', 'first')
    ).reset_index()
    windows_meta = windows_meta.merge(regime_labels[['window_id', 'regime']], on='window_id', how='inner')

    def interp_neighbour(ego_times, nb_times, nb_pos):
        if len(nb_times) < 2:
            return np.tile(nb_pos[0], (len(ego_times), 1))
        fx = interp1d(nb_times, nb_pos[:,0], kind='linear', bounds_error=False, fill_value='extrapolate')
        fy = interp1d(nb_times, nb_pos[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
        return np.column_stack([fx(ego_times), fy(ego_times)])

    calib_data = []
    for regime, group in windows_meta.groupby('regime'):
        group = group.sample(min(config.max_windows_per_regime, len(group)), random_state=42)
        for _, row in group.iterrows():
            win_id = row['window_id']; ego_id = row['agent_id']
            t0, t1 = row['window_start_abs'], row['window_end_abs']
            ego_full = agent_dfs[ego_id]
            mask = (ego_full['timestamp'] >= t0) & (ego_full['timestamp'] <= t1)
            ego_win = ego_full[mask]
            if len(ego_win) < 5: continue
            ego_pos = ego_win[['pos_x', 'pos_y']].values
            ego_times = ego_win['timestamp'].values
            if np.linalg.norm(ego_pos[-1] - ego_pos[0]) < config.min_displacement: continue
            neighbours_raw = []
            neighbours_interp = []
            for other_id, other_df in agent_dfs.items():
                if other_id == ego_id: continue
                other_mask = (other_df['timestamp'] >= t0) & (other_df['timestamp'] <= t1)
                if other_mask.sum() < 2: continue
                other_win = other_df[other_mask]
                nb_times = other_win['timestamp'].values
                nb_pos = other_win[['pos_x', 'pos_y']].values
                neighbours_raw.append({'times': nb_times, 'pos': nb_pos})
                neighbours_interp.append(interp_neighbour(ego_times, nb_times, nb_pos))
            calib_data.append({
                'window_id': win_id, 'regime': regime,
                'ego_times': ego_times, 'ego_pos': ego_pos,
                'ego_goal': ego_pos[-1],
                'neighbours_raw': neighbours_raw,
                'neighbours_interp': neighbours_interp,
            })
    return calib_data, agent_dfs

# ------------------------------------------------------------
# Main stage function
# ------------------------------------------------------------
def run_stage4(config: PipelineConfig, logger: PipelineLogger,
               df_windows: pd.DataFrame, df_raw: pd.DataFrame,
               regime_labels: pd.DataFrame,
               f11: pd.DataFrame, f12: pd.DataFrame,
               f13: pd.DataFrame, f14: pd.DataFrame):
    logger.info("=== Stage 4: SFM Calibration ===")
    # Keep only windows with path_length > 1.5 and at least one neighbour
    logger.info("Filtering to active windows (path>1.5m & n_neighbors>0)...")
    merge_feat = f11[['window_id']].merge(
        f12[['window_id', 'path_length']], on='window_id', how='inner'
    ).merge(f13[['window_id', 'n_neighbors']], on='window_id', how='inner')
    active_mask = (merge_feat['path_length'] > config.min_path_length) & (merge_feat['n_neighbors'] > 0)
    active_ids = merge_feat.loc[active_mask, 'window_id']
    df_windows_active = df_windows[df_windows['window_id'].isin(active_ids)]
    logger.info(f"Active windows: {len(df_windows_active)} (out of {len(df_windows)})")

    out_dir = os.path.join(logger.output_dir, logger.run_id, "data/stage4")
    rep_dir = os.path.join(logger.output_dir, logger.run_id, "reports/stage4")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    calib_data, agent_dfs = _build_calib_data(df_windows_active, regime_labels, df_raw, config)

    # Compute real feature statistics per regime (across all regimes present)
    regimes_present = [w['regime'] for w in calib_data]
    regime_features_real = {r: [] for r in set(regimes_present)}
    for w in calib_data:
        feats = extract_features_from_traj(w['ego_pos'], w['ego_times'], w['neighbours_interp'])
        if feats is not None:
            regime_features_real[w['regime']].append(feats)

    # Decide which regime to calibrate
    regime_counts = pd.Series([w['regime'] for w in calib_data]).value_counts()
    if config.calibrate_regime == "auto":
        if len(regime_counts) == 0:
            logger.error("No active windows available for calibration.")
            return
        target_regime = regime_counts.index[0]  # most frequent
        logger.info(f"Auto-selected regime: {target_regime} ({regime_counts[target_regime]} windows)")
    else:
        target_regime = config.calibrate_regime
        if target_regime not in regime_counts:
            logger.error(f"Requested regime {target_regime} not found in active windows.")
            return
        logger.info(f"Using specified regime: {target_regime} ({regime_counts[target_regime]} windows)")

    # Keep only windows of the target regime
    calib_data = [w for w in calib_data if w['regime'] == target_regime]
    if len(calib_data) < 5:
        logger.warning(f"Not enough windows ({len(calib_data)}) for regime {target_regime}. Skipping.")
        return

    bounds_ext = [
        (0.2, 2.0), (0.5, 2.0), (0.1, 10.0), (0.1, 1.5),
        (0.0, 1.0), (np.deg2rad(20), np.deg2rad(150)),
        (-0.3, 0.3), (0.0, 5.0), (0.1, 2.0),
        (-np.pi/3, np.pi/3), (0.0, 2.0), (np.deg2rad(10), np.deg2rad(90))
    ]

    # Process the single target regime
    logger.info(f"  Calibrating regime: {target_regime}")
    regime_dir = os.path.join(rep_dir, f"calibration_{target_regime.replace(' ','_')}")
    os.makedirs(regime_dir, exist_ok=True)

    windows = calib_data
    if len(windows) < 5:
        logger.warning(f"Not enough windows ({len(windows)}) for regime {target_regime}. Skipping.")
        return

    real_stats = {}
    feat_list = regime_features_real[target_regime]
    if feat_list:
        for key in feat_list[0].keys():
            vals = [f[key] for f in feat_list]
            real_stats[key] = {'mean': np.mean(vals), 'std': np.std(vals)}

    loss_func = partial(regime_feature_loss_ext,
                        windows_subset=windows, real_stats=real_stats,
                        dt_sim=config.dt_sim)

    iteration_log = []
    best_loss = [np.inf]   # mutable container to track best

    def callback(xk, convergence):
        loss = loss_func(xk)
        iteration_log.append({'loss': loss, **dict(zip(
            ['v0','tau','A','B','lam','phi_fov','kappa','k_group','r_group','theta_gaze','w_att','fov_att'], xk))})
        if loss < best_loss[0]:
            best_loss[0] = loss
        # Log every iteration at INFO level (visible by default)
        logger.info(
            f"    iter {len(iteration_log):3d} | loss: {loss:.4f} (best {best_loss[0]:.4f}) | "
            f"v0={xk[0]:.3f} tau={xk[1]:.3f} A={xk[2]:.2f} B={xk[3]:.2f} lam={xk[4]:.2f} "
            f"phi={np.rad2deg(xk[5]):.0f}° kappa={xk[6]:.2f} kg={xk[7]:.2f} rg={xk[8]:.2f} "
            f"tg={np.rad2deg(xk[9]):.0f}° w_att={xk[10]:.2f} fov_att={np.rad2deg(xk[11]):.0f}°"
        )

        # ================================================================
    # Stage 1 – Quick global search with Differential Evolution
    # ================================================================
    logger.info(f"    Stage 1 – Differential Evolution (maxiter={config.optimizer_maxiter}, popsize={config.optimizer_popsize})")
    start_t = time.time()
    result_de = differential_evolution(
        loss_func, bounds_ext,
        maxiter=config.optimizer_maxiter,
        popsize=config.optimizer_popsize,
        tol=config.optimizer_tol,
        polish=False,
        workers=1,             # avoid pickling overhead
        seed=42,
        callback=callback      # same callback logs every iteration
    )
    de_time = time.time() - start_t
    logger.info(f"    DE finished in {de_time:.1f}s, loss={result_de.fun:.4f}")

    # Save the best from DE as a fallback
    best_params = result_de.x
    best_loss = result_de.fun

    # ================================================================
    # Stage 2 – Local refinement with L‑BFGS‑B from the DE solution
    # ================================================================
    logger.info(f"    Stage 2 – Local refinement (L‑BFGS‑B, maxiter={config.optimizer_local_maxiter})")
    local_iter_log = []
    def local_callback(xk):
        loss = loss_func(xk)
        local_iter_log.append({'loss': loss})
        logger.info(f"      local iter {len(local_iter_log):3d} | loss: {loss:.4f}")

    start_t2 = time.time()
    result_local = minimize(
        loss_func,
        result_de.x,
        method='L-BFGS-B',
        bounds=bounds_ext,
        options={'maxiter': config.optimizer_local_maxiter, 'ftol': 1e-4},
        callback=local_callback
    )
    local_time = time.time() - start_t2
    logger.info(f"    Local refinement finished in {local_time:.1f}s, loss={result_local.fun:.4f}")

    # Use the local result if it is better, otherwise keep DE result
    if result_local.fun < best_loss:
        opt_params = result_local.x
        final_loss = result_local.fun
        logger.info("    Local refinement improved the solution.")
    else:
        opt_params = best_params
        final_loss = best_loss
        logger.info("    Local refinement did not improve; using DE solution.")

    # Append local iterations to the main log (optional)
    if local_iter_log:
        # we can merge the local iterations into the main iteration_log
        # but they don't have all parameter values; keep them separate for simplicity
        pass

    logger.info(f"    Optimization finished in {de_time + local_time:.1f}s, final loss={final_loss:.4f}")

    # Save parameters
    param_names = ['v0','tau','A_ped','B_ped','lam_base','phi_fov','kappa','k_group','r_group','theta_gaze','w_att','fov_att']
    pd.DataFrame([opt_params], columns=param_names).to_csv(
        os.path.join(out_dir, f"optimized_sfm_params_{target_regime}.csv"), index=False)

    # Save iteration log
    pd.DataFrame(iteration_log).to_csv(os.path.join(regime_dir, "optimization_log.csv"), index=False)

    # Trajectory comparison (few windows)
    n_show = min(6, len(windows))
    rows = (n_show + 2) // 3; cols = 3
    fig_traj = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Win {i+1}" for i in range(n_show)])
    for i in range(n_show):
        w = windows[i]; ego_pos = w['ego_pos']; ego_times = w['ego_times']
        fig_traj.add_trace(go.Scatter(x=ego_pos[:,0], y=ego_pos[:,1], mode='lines',
                                      line=dict(color='blue'), showlegend=False),
                           row=i//cols+1, col=i%cols+1)
        dt0 = ego_times[1] - ego_times[0]
        vx0 = (ego_pos[1,0]-ego_pos[0,0])/dt0; vy0 = (ego_pos[1,1]-ego_pos[0,1])/dt0
        theta0 = np.arctan2(vy0, vx0)
        init_state = (ego_pos[0,0], ego_pos[0,1], vx0, vy0, theta0)
        waypoints = generate_waypoints(ego_times, ego_pos, 5)
        T = ego_times[-1] - ego_times[0]

        # Pre‑interpolate neighbours for the visualisation simulation
        n_steps_viz = int(T / config.dt_sim) + 1
        sim_times_viz = np.linspace(0, T, n_steps_viz)
        neighbours_interp_viz = []
        for nb in w['neighbours_raw']:
            nb_t = nb['times']
            nb_p = nb['pos']
            if len(nb_t) < 2:
                x_arr = np.full(n_steps_viz, nb_p[0, 0])
                y_arr = np.full(n_steps_viz, nb_p[0, 1])
            else:
                x_arr = np.interp(sim_times_viz, nb_t - nb_t[0], nb_p[:, 0])
                y_arr = np.interp(sim_times_viz, nb_t - nb_t[0], nb_p[:, 1])
            neighbours_interp_viz.append(np.column_stack([x_arr, y_arr]))

        sim = simulate_extended_sfm(opt_params, init_state, waypoints,
                                    neighbours_interp_viz, config.dt_sim, T)
        if sim is not None and len(sim) > 1:
            fig_traj.add_trace(go.Scatter(x=sim[:,0], y=sim[:,1], mode='lines',
                                          line=dict(color='red'), showlegend=False),
                               row=i//cols+1, col=i%cols+1)
    fig_traj.update_layout(height=300*rows, width=300*cols,
                           title=f"Trajectory Comparison – {target_regime}")
    fig_traj.write_html(os.path.join(regime_dir, "trajectory_comparison.html"))
    logger.info(f"    Report saved to {regime_dir}")
    logger.info("Stage 4 complete.")