"""CLI to animate a saved simulation history CSV without re-running the simulation."""
import argparse
import os
import webbrowser
import numpy as np
import pandas as pd
from ..agents.user import UserTrajectory
from ..config import CONFIG
from ..visualization.animation import create_animation_from_frames

def main():
    parser = argparse.ArgumentParser(description="Animate a saved history CSV")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the history CSV file")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Keep every Nth frame (1 = all frames)")
    parser.add_argument("--follow-user", action="store_true", default=True,
                        help="Camera follows the user (default: True)")
    parser.add_argument("--no-follow-user", dest="follow_user", action="store_false")
    parser.add_argument("--follow-zoom-radius", type=float, default=15.0,
                        help="Radius around user when in follow mode")
    parser.add_argument("--frame-duration", type=int, default=50,
                        help="Milliseconds per animation frame")
    parser.add_argument("--output", type=str, default="history_animation.html",
                        help="Output HTML file name")
    parser.add_argument("--draw-pedestrian-safety", action="store_true", default=False,
                        help="Draw pedestrian safety circles")
    args = parser.parse_args()

    # ---- Load CSV ----
    history_df = pd.read_csv(args.csv)

    # ---- Extract user data ----
    user_df = history_df[history_df["agent_type"] == "user"].copy()
    if user_df.empty:
        print("No user data found in CSV. Exiting.")
        return
    user_df = user_df.sort_values("time").reset_index(drop=True)

    # Build UserTrajectory (state format: px, py, yaw, v_forw, v_orth)
    user_times = user_df["time"].values
    user_pos_x = user_df["x"].values
    user_pos_y = user_df["y"].values
    user_yaw = user_df["theta"].values
    user_vx = user_df["vx"].values
    user_vy = user_df["vy"].values
    # approximate forward/orthogonal velocities
    cos_yaw = np.cos(user_yaw)
    sin_yaw = np.sin(user_yaw)
    v_forw = user_vx * cos_yaw + user_vy * sin_yaw
    v_orth = -user_vx * sin_yaw + user_vy * cos_yaw
    state_array = np.column_stack([user_pos_x, user_pos_y, user_yaw, v_forw, v_orth])
    dt_median = np.median(np.diff(user_times)) if len(user_times) > 1 else 0.04
    user_traj = UserTrajectory(state_array, dt=dt_median, data_format="state")

    # ---- Extract static obstacles (from first occurrence of each ID) ----
    static_obs_df = history_df[history_df["agent_type"] == "static_obstacle"]
    static_obstacles = []   # we'll just store them as tuples (x, y, r) for animation
    if not static_obs_df.empty:
        for obs_id, grp in static_obs_df.groupby("agent_id"):
            row = grp.iloc[0]
            static_obstacles.append((row["x"], row["y"], row["radius"]))

    # ---- Build frames_data ----
    # Get unique timestamps
    timestamps = sorted(history_df["time"].unique())
    if args.frame_skip > 1:
        timestamps = timestamps[::args.frame_skip]

    frames_data = []
    robot_traj_abs = []   # for robot trail

    for t in timestamps:
        # all rows at this time
        rows = history_df[history_df["time"] == t]

        # user
        user_rows = rows[rows["agent_type"] == "user"]
        if user_rows.empty:
            continue
        u_row = user_rows.iloc[0]
        user_pos = (u_row["x"], u_row["y"])
        user_facing = u_row["theta"]
        # motion angle from vx, vy
        u_vx, u_vy = u_row["vx"], u_row["vy"]
        u_speed = np.hypot(u_vx, u_vy)
        user_motion_angle = np.arctan2(u_vy, u_vx) if u_speed > 0.05 else user_facing

        # pedestrians
        ped_rows = rows[rows["agent_type"] == "pedestrian"]
        peds = []
        for _, p_row in ped_rows.iterrows():
            # velocity from vx, vy
            p_vx, p_vy = p_row["vx"], p_row["vy"]
            p_speed = np.hypot(p_vx, p_vy)
            p_theta = np.arctan2(p_vy, p_vx) if p_speed > 0.05 else p_row["theta"]
            peds.append((
                p_row["x"], p_row["y"],
                p_vx, p_vy,
                p_row.get("mood", "real_crowd"),
                p_theta,            # motion angle
                0.0, 0.0, 0.0       # gaze_offset, fov_att, w_att (not available)
            ))

        # robot
        robot_rows = rows[rows["agent_type"] == "robot"]
        if robot_rows.empty:
            continue
        r_row = robot_rows.iloc[0]
        robot_pos = (r_row["x"], r_row["y"])
        robot_theta = r_row["theta"]

        # robot future arc (from v_cmd, omega_cmd if available)
        arc_x, arc_y = [], []
        if "v_cmd" in r_row and "omega_cmd" in r_row:
            v_cmd = r_row["v_cmd"]
            omega_cmd = r_row["omega_cmd"]
            arc_duration = 1.0
            n_arc_pts = 20
            if abs(omega_cmd) > 1e-6:
                r_arc = v_cmd / omega_cmd
                dtheta = omega_cmd * arc_duration
                theta0 = robot_theta
                angles = np.linspace(0, dtheta, n_arc_pts)
                arc_x = robot_pos[0] + r_arc * (np.sin(theta0 + angles) - np.sin(theta0))
                arc_y = robot_pos[1] - r_arc * (np.cos(theta0 + angles) - np.cos(theta0))
            else:
                total_dist = v_cmd * arc_duration
                for k in range(n_arc_pts + 1):
                    frac = k / n_arc_pts
                    arc_x.append(robot_pos[0] + frac * total_dist * np.cos(robot_theta))
                    arc_y.append(robot_pos[1] + frac * total_dist * np.sin(robot_theta))
        # goal position
        goal_pos = (r_row["goal_x"], r_row["goal_y"]) if "goal_x" in r_row else robot_pos

        # future path (user -> goal)
        future_path_x = [user_pos[0], goal_pos[0]]
        future_path_y = [user_pos[1], goal_pos[1]]

        # dynamic obstacles
        dyn_rows = rows[rows["agent_type"] == "dynamic_obstacle"]
        dynamic_obstacles = []
        for _, d_row in dyn_rows.iterrows():
            dynamic_obstacles.append((d_row["x"], d_row["y"], d_row["radius"], d_row["agent_id"]))

        # maneuver flags
        overtaking = r_row.get("overtaking_active", False) if "overtaking_active" in r_row else False
        parking = r_row.get("parking_active", False) if "parking_active" in r_row else False
        repulsion = r_row.get("repulsion_active", False) if "repulsion_active" in r_row else False

        frames_data.append({
            "time": t,
            "user_pos": user_pos,
            "user_motion_angle": user_motion_angle,
            "user_facing_angle": user_facing,
            "pedestrians": peds,
            "robot_pos": robot_pos,
            "robot_theta": robot_theta,
            "robot_arc_x": arc_x,
            "robot_arc_y": arc_y,
            "goal_pos": goal_pos,
            "future_path_x": future_path_x,
            "future_path_y": future_path_y,
            "dynamic_obstacles": dynamic_obstacles,
            "respawn_count": 0,
            "overtaking_active": overtaking,
            "parking_active": parking,
            "repulsion_active": repulsion,
        })
        robot_traj_abs.append(robot_pos)

    # ---- Build robot_data dict ----
    robot_data = {"trajectory": robot_traj_abs, "radius": CONFIG.robot_radius}

    # ---- Get controller name ----
    controller_name = "unknown"
    robot_rows_all = history_df[history_df["agent_type"] == "robot"]
    if not robot_rows_all.empty and "controller" in robot_rows_all.columns:
        controller_name = robot_rows_all["controller"].iloc[0]

    # ---- Create animation ----
    anim_fig = create_animation_from_frames(
        frames_data=frames_data,
        user_trajectory=user_traj,
        static_obstacles=static_obstacles,
        follow_user=args.follow_user,
        follow_zoom_radius=args.follow_zoom_radius,
        spawner=None,
        robot_data=robot_data,
        draw_pedestrian_safety=args.draw_pedestrian_safety,
        frame_duration_ms=args.frame_duration,
        controller_name=controller_name,
    )
    anim_fig.write_html(args.output)
    print(f"Animation saved to {args.output}")
    try:
        webbrowser.open(args.output, new=1)
    except Exception:
        pass

if __name__ == "__main__":
    main()