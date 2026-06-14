"""Stage 1: Sliding window extraction from the combined bin data."""

import os
import numpy as np
import pandas as pd
from ...behavior_pipeline.config import PipelineConfig
from ...behavior_pipeline.reporting import PipelineLogger


def extract_windows_abs(agent_df, win_len, stride, min_pts):
    ts = agent_df["timestamp"].values
    t_start = ts.min()
    t_end = ts.max()
    windows = []
    w_start = t_start
    while w_start + win_len <= t_end:
        mask = (ts >= w_start) & (ts <= w_start + win_len)
        if mask.sum() >= min_pts:
            win = agent_df[mask].copy()
            win["window_start_abs"] = w_start
            win["window_end_abs"] = w_start + win_len
            windows.append(win)
        w_start += stride
    return windows


def run_stage1(config: PipelineConfig, logger: PipelineLogger, df_all: pd.DataFrame):
    logger.info("=== Stage 1: Sliding Window Extraction ===")

    # Prepare velocity/accel (same as Phase 0.2)
    df_all["vx"] = df_all["velocity"] * np.cos(df_all["motion_angle_rad"])
    df_all["vy"] = df_all["velocity"] * np.sin(df_all["motion_angle_rad"])
    df_all = df_all.sort_values(["agent_id", "timestamp"])
    df_all["dt"] = df_all.groupby("agent_id")["timestamp"].diff()
    df_all["ax"] = df_all.groupby("agent_id")["vx"].diff() / df_all["dt"]
    df_all["ay"] = df_all.groupby("agent_id")["vy"].diff() / df_all["dt"]
    df_all["accel_mag"] = np.sqrt(df_all["ax"] ** 2 + df_all["ay"] ** 2)
    df_all[["ax", "ay", "accel_mag"]] = df_all.groupby("agent_id")[
        ["ax", "ay", "accel_mag"]
    ].bfill()
    df_all = df_all.dropna(
        subset=["pos_x", "pos_y", "velocity", "vx", "vy", "accel_mag", "dt"]
    )

    all_windows = []
    for agent_id, grp in df_all.groupby("agent_id"):
        wins = extract_windows_abs(
            grp,
            config.window_len_sec,
            config.window_stride_sec,
            config.min_points_per_window,
        )
        for j, w in enumerate(wins):
            w["window_id"] = f"{agent_id}_{j}"
        all_windows.extend(wins)

    df_windows = pd.concat(all_windows, ignore_index=True)
    logger.info(
        f"Extracted {df_windows['window_id'].nunique()} windows, {df_windows['agent_id'].nunique()} agents."
    )

    os.makedirs(
        os.path.join(logger.output_dir, logger.run_id, "data/stage1"), exist_ok=True
    )
    df_windows.to_csv(
        os.path.join(logger.output_dir, logger.run_id, "data/stage1/df_windows.csv")
    )

    # Window statistics
    stats = {
        "total_windows": df_windows["window_id"].nunique(),
        "total_agents": df_windows["agent_id"].nunique(),
        "avg_frames_per_window": df_windows.groupby("window_id").size().mean(),
    }
    report = "Window Statistics:\n" + str(stats)
    logger.write_stage_report("stage1", report)
    return df_windows
