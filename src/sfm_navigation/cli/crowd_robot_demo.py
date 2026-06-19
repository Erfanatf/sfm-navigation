import os, sys, time, webbrowser, argparse
import numpy as np
import pandas as pd
from ..data.atc_loader import load_atc_raw
from ..crowd_analysis.binning import convert_units, bin_data, select_bin_data
from ..agents.user import UserTrajectory
from ..agents.robot import DifferentialDriveRobot
from ..config import CONFIG
from ..visualization.animation import create_animation_from_frames, _mood_name
from ..data.moods import CUSTOM_MOODS
from ..cli.robot_demo import analyze_robot_collisions
from ..data.filtering import filter_agent_trajectory
from ..controllers import create_controller
# Import all DWA classes for the isinstance check
from ..controllers.dwa.basic_dwa import BasicDWA
from ..controllers.dwa.dw4do import DW4DO
from ..controllers.dwa.dwa_vo import DWA_VO
from ..controllers.dwa.dwa_rvo import DWA_RVO
from ..controllers.dwa.dwa_orca import DWA_ORCA

def main():
    parser = argparse.ArgumentParser(description="ATC crowd robot demo")
    parser.add_argument(
        "--csv",
        type=str,
        default=CONFIG.atc_csv_path,
    )
    parser.add_argument("--bin", type=int, default=4)
    parser.add_argument(
        "--subsample", type=int, default=1, help="Keep every Nth frame from raw data"
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Shifted agent ID to use as User (default: longest trajectory)",
    )
    parser.add_argument(
        "--safety-mode", type=str, default="all", choices=["none", "user-only", "all"]
    )
    parser.add_argument("--overtaking", action="store_true", default=False)
    parser.add_argument("--no-overtaking", dest="overtaking", action="store_false")
    parser.add_argument(
        "--robot-mood", type=str, default=None, help="Calibrated mood for robot"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum number of simulation steps"
    )
    parser.add_argument(
        "--dt-sim",
        type=float,
        default=None,
        help="Simulation timestep (default: median dt from data)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=50,
        help="Milliseconds per animation frame",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        default=True,
        help="Apply KF+SavGol filtering to trajectories",
    )
    parser.add_argument("--no-filter", dest="filter", action="store_false")
    parser.add_argument(
        "--process-noise",
        type=float,
        default=0.1,
        help="Process noise for Kalman filter",
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=0.5,
        help="Measurement noise for Kalman filter",
    )
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=9,
        help="Window length for SavGol smoothing (must be odd)",
    )
    parser.add_argument(
        "--savgol-order",
        type=int,
        default=2,
        help="Polynomial order for SavGol smoothing",
    )
    parser.add_argument(
        "--filter-method",
        type=str,
        default="KF",
        choices=["KF", "UKF"],
        help="Filter type: KF (Kalman) or UKF (Unscented)",
    )
    parser.add_argument('--controller', type=str, default='SFM',
                        choices=['SFM', 'DWA_BASIC', 'DWA_DW4DO', 'DWA_VO',
                                 'DWA_RVO', 'DWA_ORCA'],
                        help='Controller type')

    args = parser.parse_args()

    USE_OVERTAKING = args.overtaking
    SAFETY_MODE = args.safety_mode
    print(f"Overtaking: {'ON' if USE_OVERTAKING else 'OFF'}")
    print(f"Safety mode: {SAFETY_MODE}")

    # ---------- 1. Load and prepare crowd data ----------
    print("[1/5] Loading ATC data...")
    df_raw = load_atc_raw(args.csv)
    df = convert_units(df_raw)
    bin_stats_df = bin_data(df)
    df_bin, original_longest_id = select_bin_data(
        df, bin_stats_df, selected_bin_index=args.bin
    )

    # Build per-agent dictionary (shifted IDs)
    agents = {}
    for agent_id, grp in df_bin.groupby("agent_id"):
        agents[agent_id] = grp.sort_values("timestamp_rel")

    # Find shifted ID of the longest trajectory
    agent_counts = df_bin.groupby("agent_id").size()
    shifted_longest_id = agent_counts.idxmax()
    print(
        f"Longest trajectory shifted ID: {shifted_longest_id} (original {original_longest_id})"
    )

    # Choose User ID
    if args.user_id is not None:
        user_id = args.user_id
        if user_id not in agents:
            print(
                f"User ID {user_id} not found. Available: {sorted(agents.keys())[:20]}..."
            )
            return
    else:
        user_id = shifted_longest_id
    print(f"Using agent {user_id} as the User")

    # Time grid
    timestamps = sorted(df_bin["timestamp_rel"].unique())
    if args.subsample > 1:
        timestamps = timestamps[:: args.subsample]
    if args.max_steps:
        timestamps = timestamps[: args.max_steps]

    # Determine simulation dt
    dt_data = np.median(np.diff(timestamps)) if len(timestamps) > 1 else 0.04
    dt = args.dt_sim if args.dt_sim else dt_data
    print(f"Simulation dt: {dt:.4f}s, total frames: {len(timestamps)}")

    # ---------- 2. Filter trajectories ----------
    if args.filter:
        print("[2/5] Filtering agent trajectories (KF+SavGol)...")
        for agent_id in agents:
            agents[agent_id] = filter_agent_trajectory(
                agents[agent_id],
                process_noise=args.process_noise,
                measurement_noise=args.measurement_noise,
                savgol_window=args.savgol_window,
                savgol_order=args.savgol_order,
                method=args.filter_method,
            )
        print("Filtering complete.")
    else:
        print("[2/5] Skipping filtering (--no-filter).")

    # ---------- 3. Build UserTrajectory for animation ----------
    user_data = agents[user_id]
    user_pos = user_data[["pos_x", "pos_y"]].values
    user_yaw = user_data["facing_angle_rad"].values
    user_vel = user_data["velocity"].values
    user_motion_angle = user_data["motion_angle_rad"].values
    v_forw = user_vel * np.cos(user_motion_angle - user_yaw)
    v_orth = user_vel * np.sin(user_motion_angle - user_yaw)
    user_state_array = np.column_stack([user_pos, user_yaw, v_forw, v_orth])
    user_traj = UserTrajectory(user_state_array, dt=dt_data, data_format="state")

    # ---------- 4. Setup robot ----------
    handcrafted_defaults = {
        "v0": 2.0,
        "tau": 0.3,
        "A_ped": 12.0,
        "B_ped": 0.25,
        "lam_base": 0.3,
        "kappa": 0.0,
    }
    if args.robot_mood:
        robot_params = CUSTOM_MOODS.get(args.robot_mood, handcrafted_defaults)
        print(
            f"Robot using mood '{args.robot_mood}'"
            if args.robot_mood in CUSTOM_MOODS
            else "Mood not found, using safe defaults."
        )
    else:
        robot_params = handcrafted_defaults
        print("Using hand‑crafted safe defaults for robot.")

    controller = create_controller(args.controller, CONFIG, robot_params=robot_params)

    # Initial robot position: 2 m ahead of User's first position
    first_row = agents[user_id].iloc[0]
    ux0, uy0 = first_row["pos_x"], first_row["pos_y"]
    ufacing0 = first_row["facing_angle_rad"]
    robot_start_x = ux0 + 4.0 * np.cos(ufacing0)
    robot_start_y = uy0 + 4.0 * np.sin(ufacing0)
    robot = DifferentialDriveRobot(CONFIG, start_x=robot_start_x, start_y=robot_start_y)

    # ---------- 5. Simulation loop ----------
    other_agent_ids = [aid for aid in agents if aid != user_id]

    frames_data = []
    robot_traj_abs = []
    history = []
    prev_user_pos = (ux0, uy0)
    STATIONARY_SPEED_THRESHOLD = 0.1
    desired_period = 1.0 / 10.0   # 10 Hz control loop

    print(f"[4/5] Running simulation...")
    t_start = time.perf_counter()

    is_dwa = isinstance(controller, (BasicDWA, DW4DO, DWA_VO, DWA_RVO, DWA_ORCA))

    for step, t in enumerate(timestamps):
        user_rows = agents[user_id][agents[user_id]["timestamp_rel"] == t]
        if user_rows.empty:
            continue
        user_row = user_rows.iloc[0]
        user_x, user_y = user_row["pos_x"], user_row["pos_y"]
        user_facing = user_row["facing_angle_rad"]
        dx = user_x - prev_user_pos[0]
        dy = user_y - prev_user_pos[1]
        user_speed = np.hypot(dx, dy) / dt if dt > 0 else 0.0
        user_motion_angle = np.arctan2(dy, dx) if user_speed > 0.05 else user_facing
        prev_user_pos = (user_x, user_y)

        # Obstacle list: other agents + optionally user
        obstacles = []
        for aid in other_agent_ids:
            agent_rows = agents[aid][agents[aid]["timestamp_rel"] == t]
            if not agent_rows.empty:
                r = agent_rows.iloc[0]
                ped_theta = r["motion_angle_rad"]
                vx = r["velocity"] * np.cos(ped_theta)
                vy = r["velocity"] * np.sin(ped_theta)
                obstacles.append([r["pos_x"], r["pos_y"], 0.3, vx, vy])
        # User: only added for non-DWA controllers
        if SAFETY_MODE in ("user-only", "all"):
            if not is_dwa:
                obstacles.append([user_x, user_y, user_traj.safety_radius])

        # Goal logic
        lookahead = 2.0
        if user_speed < STATIONARY_SPEED_THRESHOLD:
            goal_dist = user_traj.safety_radius + 0.5
            goal_x = user_x + goal_dist * np.cos(user_facing)
            goal_y = user_y + goal_dist * np.sin(user_facing)
        else:
            future_time = t + lookahead
            future_rows = agents[user_id][
                agents[user_id]["timestamp_rel"] >= future_time
            ]
            if not future_rows.empty:
                goal_x = future_rows.iloc[0]["pos_x"]
                goal_y = future_rows.iloc[0]["pos_y"]
            else:
                last = agents[user_id].iloc[-1]
                dir_x = np.cos(user_facing)
                dir_y = np.sin(user_facing)
                goal_x = last["pos_x"] + dir_x * user_speed * lookahead
                goal_y = last["pos_y"] + dir_y * user_speed * lookahead

        user_info = {
            "x": user_x,
            "y": user_y,
            "radius": user_traj.safety_radius,
            "facing": user_facing,
            "active": USE_OVERTAKING,
        }

        v_cmd, omega_cmd = controller.compute_velocity(
            robot.state, (goal_x, goal_y), obstacles, user=user_info, dt=dt
        )

        # Read DOB signals (all controllers now have these attributes)
        v_base   = getattr(controller, '_v_base',   0.0)
        omega_base = getattr(controller, '_omega_base', 0.0)
        v_man    = getattr(controller, '_v_man',    0.0)
        omega_man  = getattr(controller, '_omega_man',  0.0)
        v_final  = getattr(controller, '_v_final',  v_cmd)
        omega_final = getattr(controller, '_omega_final', omega_cmd)

        overtaking_now = getattr(controller, 'overtaking_active', False)
        parking_now    = getattr(controller, 'parking_active', False)
        repulsion_now  = getattr(controller, 'repulsion_active', False)

        comp_time = controller.last_compute_time
        rt_factor = controller.get_real_time_factor(desired_period)
        robot.update(v_cmd, omega_cmd, dt)

        # Collision check
        in_coll = False
        for obs in obstacles:
            dist = np.hypot(robot.state.x - obs[0], robot.state.y - obs[1])
            if dist <= CONFIG.robot_radius + obs[2] + CONFIG.safety_margin:
                in_coll = True
                break

        # History (robot, user, pedestrians)
        history.append(
            {
                "time": t,
                "agent_type": "robot",
                "agent_id": 0,
                "x": robot.state.x,
                "y": robot.state.y,
                "vx": robot.state.v * np.cos(robot.state.theta),
                "vy": robot.state.v * np.sin(robot.state.theta),
                "theta": robot.state.theta,
                "radius": CONFIG.robot_radius,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "v_cmd": v_cmd,
                "omega_cmd": omega_cmd,
                "comp_time": comp_time,
                "rt_factor": rt_factor,
                "in_collision": in_coll,
                "controller": type(controller).__name__,
                "v_base": v_base,
                "omega_base": omega_base,
                "v_man": v_man,
                "omega_man": omega_man,
                "v_final": v_final,
                "omega_final": omega_final,
            }
        )
        history.append(
            {
                "time": t,
                "agent_type": "user",
                "agent_id": -1,
                "x": user_x,
                "y": user_y,
                "vx": user_speed * np.cos(user_motion_angle),
                "vy": user_speed * np.sin(user_motion_angle),
                "theta": user_facing,
                "radius": 0.3,
                "mood": "real_user",
            }
        )
        for aid in other_agent_ids:
            rows = agents[aid][agents[aid]["timestamp_rel"] == t]
            if not rows.empty:
                r = rows.iloc[0]
                ped_theta = r["motion_angle_rad"]
                history.append(
                    {
                        "time": t,
                        "agent_type": "pedestrian",
                        "agent_id": aid,
                        "x": r["pos_x"],
                        "y": r["pos_y"],
                        "vx": r["velocity"] * np.cos(ped_theta),
                        "vy": r["velocity"] * np.sin(ped_theta),
                        "theta": ped_theta,
                        "radius": 0.3,
                        "mood": "real_crowd",
                    }
                )

        # Frame collection (subsampled)
        if step % args.subsample == 0:
            ped_info = []
            for aid in other_agent_ids:
                rows = agents[aid][agents[aid]["timestamp_rel"] == t]
                if not rows.empty:
                    r = rows.iloc[0]
                    ped_theta = r["motion_angle_rad"]
                    ped_info.append(
                        (
                            r["pos_x"],
                            r["pos_y"],
                            r["velocity"] * np.cos(ped_theta),
                            r["velocity"] * np.sin(ped_theta),
                            "real_crowd",
                            ped_theta,
                            0.0,
                            0.0,
                            0.0,
                        )
                    )

            # Robot future arc
            arc_x, arc_y = [], []
            if abs(omega_cmd) > 1e-6:
                r_arc = v_cmd / omega_cmd
                dtheta = omega_cmd * dt * 2
                theta0 = robot.state.theta
                angles = np.linspace(0, dtheta, 10)
                arc_x = robot.state.x + r_arc * (
                    np.sin(theta0 + angles) - np.sin(theta0)
                )
                arc_y = robot.state.y - r_arc * (
                    np.cos(theta0 + angles) - np.cos(theta0)
                )
            else:
                total_dist = v_cmd * dt * 2
                for k in range(11):
                    frac = k / 10.0
                    arc_x.append(
                        robot.state.x + frac * total_dist * np.cos(robot.state.theta)
                    )
                    arc_y.append(
                        robot.state.y + frac * total_dist * np.sin(robot.state.theta)
                    )

            frames_data.append(
                {
                    "time": t,
                    "user_pos": (user_x, user_y),
                    "user_motion_angle": user_motion_angle,
                    "user_facing_angle": user_facing,
                    "pedestrians": ped_info,
                    "robot_pos": (robot.state.x, robot.state.y),
                    "robot_theta": robot.state.theta,
                    "robot_arc_x": arc_x,
                    "robot_arc_y": arc_y,
                    "goal_pos": (goal_x, goal_y),
                    "future_path_x": [user_x, goal_x],
                    "future_path_y": [user_y, goal_y],
                    "dynamic_obstacles": [],
                    "respawn_count": 0,
                    "overtaking_active": overtaking_now,
                    "parking_active": parking_now,
                    "repulsion_active": repulsion_now,
                }
            )
            robot_traj_abs.append((robot.state.x, robot.state.y))

        if step % max(1, len(timestamps) // 10) == 0:
            rt_now = rt_factor
            print(
                f"  Step {step}/{len(timestamps)}, RT factor: {rt_now:.4f} "
                f"(compute {comp_time*1000:.2f}ms)"
            )

    t_total = time.perf_counter() - t_start
    print(f"Simulation complete in {t_total:.1f}s")

    # Save history
    history_df = pd.DataFrame(history)
    history_csv = type(controller).__name__ + "_crowd_robot_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"History saved to {history_csv}")

    # ---- Robot collision analysis ----
    analyze_robot_collisions(history_df, CONFIG.safety_margin)

    # ---- Controller performance ----
    robot_df = history_df[history_df["agent_type"] == "robot"]
    if not robot_df.empty:
        comp_times = robot_df["comp_time"].values
        rt_factors = robot_df["rt_factor"].values
        perf_report = []
        perf_report.append("=== CONTROLLER PERFORMANCE ===")
        perf_report.append(f"Mean computation time: {np.mean(comp_times)*1000:.3f} ms")
        perf_report.append(f"Max computation time:   {np.max(comp_times)*1000:.3f} ms")
        perf_report.append(f"Min computation time:   {np.min(comp_times)*1000:.3f} ms")
        perf_report.append(f"Mean real-time factor (10 Hz): {np.mean(rt_factors):.4f}")
        perf_report.append(f"Total controller calls: {len(comp_times)}")
        perf_report.append("===============================")
        perf_text = "\n".join(perf_report)
        print("\n" + perf_text)

        with open(type(controller).__name__ + "_crowd_robot_performance.txt", "w") as f:
            f.write(perf_text)
        print("Performance report saved to crowd_robot_performance.txt")
    else:
        print("No robot data for performance analysis.")

    # ---- Animation ----
    robot_data_dict = {"trajectory": robot_traj_abs, "radius": CONFIG.robot_radius}
    anim_fig = create_animation_from_frames(
        frames_data=frames_data,
        user_trajectory=user_traj,
        static_obstacles=[],
        follow_user=True,
        follow_zoom_radius=15.0,
        spawner=None,
        robot_data=robot_data_dict,
        draw_pedestrian_safety=(SAFETY_MODE == "all"),
        frame_duration_ms=args.frame_duration,
        controller_name=type(controller).__name__
    )
    html_path = type(controller).__name__ + "_crowd_robot_animation.html"
    anim_fig.write_html(html_path)
    print(f"Animation saved to {html_path}")
    try:
        webbrowser.open(html_path, new=1)
    except Exception:
        pass


if __name__ == "__main__":
    main()