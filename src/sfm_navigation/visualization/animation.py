import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from ..constants import MOOD_COLORS
from ..agents.user import UserTrajectory
from ..spawner.spawner import SFMPedestrianSpawner
from ..logging.logger import SimulationLogger
from ..config import CONFIG
from ..data.moods import CUSTOM_MOODS


def _mood_name(mood):
    return mood if isinstance(mood, str) else mood.name


def egg_contour(cx, cy, heading, B, lam, gaze, scale=1.0, n_pts=50):
    total_heading = heading + gaze
    angles = np.linspace(0, 2*np.pi, n_pts)
    r = np.zeros(n_pts)
    for i, ang in enumerate(angles):
        da = ang - total_heading
        dx_b = np.cos(da)
        dy_b = np.sin(da)
        # front/back asymmetry
        sig_x = B * scale * (1.0 - lam) if dx_b >= 0 else B * scale * (1.0 + lam)
        sig_y = B * scale       # symmetric left/right
        cos_a = np.cos(da)
        sin_a = np.sin(da)
        r2 = 1.0 / ((cos_a / sig_x)**2 + (sin_a / sig_y)**2)
        r[i] = np.sqrt(r2)
    x_cont = cx + r * np.cos(angles)
    y_cont = cy + r * np.sin(angles)
    return x_cont, y_cont


def create_sfm_spawner_animation(
    user_trajectory: UserTrajectory,
    spawner: SFMPedestrianSpawner,
    simulation_duration: float = 500.0,
    dt: float = None,
    static_obstacles: List = None,
    frame_skip: int = 1,
    follow_user: bool = False,
    follow_zoom_radius: float = 10.0,
    enable_logging: bool = True,
    log_window_duration: float = 5.0,
    robot_data: dict = None,  # NEW
) -> Tuple[go.Figure, Optional["SimulationLogger"]]:
    # -------------------- unchanged initialisation --------------------
    if dt is None:
        dt = user_trajectory.dt
        print(f"Using trajectory dt={dt:.4f}s for SFM simulation")
    else:
        print(f"Using custom dt={dt:.4f}s for SFM simulation")
    if follow_user:
        print(f"FOLLOW MODE: User-centered view with {follow_zoom_radius}m radius")

    logger = None
    if enable_logging:
        logger = SimulationLogger(
            window_duration=log_window_duration, vicinity_radius=spawner.vicinity_radius
        )
        print(f"LOGGING ENABLED: Window duration = {log_window_duration}s")

    frames_data = []
    n_steps = int(simulation_duration / dt)
    spawner.initialize_pedestrians(current_time=0.0)
    print(f"Running simulation for {simulation_duration:.2f}s ({n_steps} steps)...")
    for step in range(n_steps):
        current_time = step * dt
        user_x, user_y = user_trajectory.get_position_at_time(current_time)
        spawner.update_pedestrians(dt, current_time, static_obstacles)
        if logger:
            logger.update(
                current_time=current_time,
                pedestrians=spawner.pedestrians,
                user_pos=(user_x, user_y),
                robot_pos=None,
                static_obstacles=static_obstacles,
            )
        if step % frame_skip == 0:
            frames_data.append(
                {
                    "time": current_time,
                    "user_pos": (user_x, user_y),
                    "pedestrians": [
                        (
                            p.x,
                            p.y,
                            p.vx,
                            p.vy,
                            _mood_name(p.mood),
                            p.theta,  # motion angle (radians)
                            p.theta_gaze,  # gaze offset (radians)
                            p.fov_att,  # half‑angle of attention cone (radians)
                            p.w_att,
                        )
                        for p in spawner.pedestrians
                    ],
                    "respawn_count": spawner.respawn_count,
                }
            )
        if step % max(1, n_steps // 5) == 0:
            print(f"  Step {step}/{n_steps}, respawns: {spawner.respawn_count}")
    if logger:
        final_user_pos = user_trajectory.get_position_at_time(simulation_duration)
        logger.finalize(simulation_duration, spawner.pedestrians, final_user_pos)
    print(
        f"Simulation complete. Respawns: {spawner.respawn_count}, Frames: {len(frames_data)}"
    )

    # -------------------- prepare figure and static elements --------------------
    fig = go.Figure()
    full_traj = user_trajectory.get_full_trajectory()
    if follow_user:
        view_range = follow_zoom_radius
        x_range = [-view_range, view_range]
        y_range = [-view_range, view_range]
        init_ux, init_uy = frames_data[0]["user_pos"]
    else:
        x_range = [0, CONFIG.env_width]
        y_range = [0, CONFIG.env_height]
        init_ux, init_uy = 0, 0

    theta_circle = np.linspace(0, 2 * np.pi, 50)  # renamed to avoid conflict
    theta_obs = np.linspace(0, 2 * np.pi, 30)
    safety_r = user_trajectory.safety_radius
    vicinity_r = spawner.vicinity_radius
    # Ellipse dimensions (in metres)
    # Major axis = shoulder width (perpendicular to motion)
    # Minor axis = body depth (aligned with motion)
    PED_A = CONFIG.pedestrian_radius - 0.1  # semi‑major (perpendicular)
    PED_B = CONFIG.pedestrian_radius  # semi‑minor (parallel to direction)
    USER_A = CONFIG.pedestrian_radius - 0.1
    USER_B = CONFIG.pedestrian_radius

    trace_idx = 0

    # User trajectory line (unchanged)
    if follow_user:
        traj_x, traj_y = full_traj[:, 0] - init_ux, full_traj[:, 1] - init_uy
    else:
        traj_x, traj_y = full_traj[:, 0], full_traj[:, 1]
    fig.add_trace(
        go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines",
            line=dict(color="lightblue", width=2, dash="dot"),
            name="Trajectory",
            showlegend=True,
        )
    )
    traj_idx = trace_idx
    trace_idx += 1

    # Obstacles (unchanged)
    obs_start_idx = trace_idx
    if static_obstacles:
        for obs in static_obstacles:
            ox = obs.x if hasattr(obs, "x") else obs[0]
            oy = obs.y if hasattr(obs, "y") else obs[1]
            orad = obs.radius if hasattr(obs, "radius") else obs[2]
            if follow_user:
                ox, oy = ox - init_ux, oy - init_uy
            fig.add_trace(
                go.Scatter(
                    x=ox + orad * np.cos(theta_obs),
                    y=oy + orad * np.sin(theta_obs),
                    fill="toself",
                    fillcolor="rgba(128,128,128,0.5)",
                    line=dict(color="gray"),
                    name="Obstacle",
                    showlegend=False,
                )
            )
            trace_idx += 1

    # Safety zone and vicinity (unchanged)
    ux_disp = 0 if follow_user else frames_data[0]["user_pos"][0]
    uy_disp = 0 if follow_user else frames_data[0]["user_pos"][1]
    fig.add_trace(
        go.Scatter(
            x=ux_disp + safety_r * np.cos(theta_circle),
            y=uy_disp + safety_r * np.sin(theta_circle),
            mode="lines",
            line=dict(color="blue", width=1, dash="dash"),
            name="Safety Zone",
        )
    )
    safety_idx = trace_idx
    trace_idx += 1
    fig.add_trace(
        go.Scatter(
            x=ux_disp + vicinity_r * np.cos(theta_circle),
            y=uy_disp + vicinity_r * np.sin(theta_circle),
            mode="lines",
            line=dict(color="green", width=1, dash="dot"),
            name="Vicinity",
            opacity=0.5,
        )
    )
    vicinity_idx = trace_idx
    trace_idx += 1

    # ----- NEW: User ellipse trace (initially empty) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            line=dict(color="blue", width=2),
            fillcolor="rgba(0,100,255,0.3)",
            name="User (body)",
        )
    )
    user_ellipse_idx = trace_idx
    trace_idx += 1

    # ----- NEW: User centre marker (small dot, preserves original "User" legend) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=8,
                color="blue",
                line=dict(width=1, color="darkblue"),
            ),
            name="User",
        )
    )
    user_marker_idx = trace_idx
    trace_idx += 1

    # ----- NEW: Pedestrian ellipses trace (initially empty) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            line=dict(color="black", width=1),
            fillcolor="rgba(200,200,200,0.3)",
            name="Pedestrian body",
        )
    )
    peds_ellipse_idx = trace_idx
    trace_idx += 1

    # ----- NEW: Pedestrian centre markers (coloured by mood) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(symbol="circle", size=8, line=dict(width=1, color="black")),
            name="Pedestrians",
        )
    )
    peds_marker_idx = trace_idx
    trace_idx += 1

    # Direction lines (heading) – unchanged
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Direction (heading)",
        )
    )
    dir_idx = trace_idx
    trace_idx += 1

    # ----- NEW: Gaze lines (cyan) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="cyan", width=2, dash="dot"),
            name="Gaze direction",
        )
    )
    gaze_idx = trace_idx
    trace_idx += 1

    # ----- NEW: Attention cones (transparent yellow) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            fillcolor="rgba(255,255,0,0.2)",
            line=dict(color="yellow", width=1),
            name="Attention cone",
        )
    )
    cone_idx = trace_idx
    trace_idx += 1

    # ----- Robot traces (if robot_data is provided) -----
    robot_body_idx = None
    robot_trail_idx = None
    robot_trail_abs_x = []
    robot_trail_abs_y = []
    if robot_data is not None:
        robot_radius = robot_data.get("radius", 0.18)
        # Robot body (circle) – initial empty
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers+lines",
                marker=dict(
                    symbol="square",
                    size=14,
                    color="red",
                    line=dict(width=2, color="darkred"),
                ),
                line=dict(color="red", width=2),
                name="Robot",
            )
        )
        robot_body_idx = trace_idx
        trace_idx += 1

        # Robot trail (line)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="red", width=1, dash="dot"),
                name="Robot path",
            )
        )
        robot_trail_idx = trace_idx
        trace_idx += 1

    # Mood legend (unchanged)
    unique_moods = set()
    for f in frames_data:
        for ped in f["pedestrians"]:
            unique_moods.add(ped[4])
    for mood in sorted(unique_moods):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=MOOD_COLORS.get(mood, "orange"),
                    line=dict(width=1, color="black"),
                ),
                name=f"  {mood}",
                legendgroup="moods",
            )
        )
        trace_idx += 1

    # ----- helper: generate ellipse points (minor axis along direction) -----
    def ellipse_points(cx, cy, angle_rad, a, b, n=30):
        """
        a = semi‑major (perpendicular to direction)
        b = semi‑minor (parallel to direction)
        angle_rad = direction of motion (heading)
        """
        t = np.linspace(0, 2 * np.pi, n)
        x_ell = a * np.cos(t)
        y_ell = b * np.sin(t)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = x_ell * cos_a - y_ell * sin_a
        y_rot = x_ell * sin_a + y_ell * cos_a
        return cx + x_rot, cy + y_rot

    # ----- helper: generate points for a cone (sector) -----
    def cone_points(cx, cy, direction_rad, half_angle_rad, length=1.2, n=20):
        """
        Returns x, y arrays for a filled polygon representing a cone.
        direction_rad = centre angle of the cone (absolute, e.g. heading + gaze offset)
        half_angle_rad = half of the opening angle
        length = radius of the cone (in metres)
        """
        start_angle = direction_rad - half_angle_rad
        end_angle = direction_rad + half_angle_rad
        angles = np.linspace(start_angle, end_angle, n)
        x_arc = cx + length * np.cos(angles)
        y_arc = cy + length * np.sin(angles)
        x_poly = np.concatenate(([cx], x_arc, [cx]))
        y_poly = np.concatenate(([cy], y_arc, [cy]))
        return x_poly, y_poly

    # ----- pre‑compute user velocity for each frame (finite difference) -----
    user_velocities = []
    for i, fd in enumerate(frames_data):
        if i < len(frames_data) - 1:
            dt_frame = frames_data[i + 1]["time"] - fd["time"]
            if dt_frame > 0:
                dx = frames_data[i + 1]["user_pos"][0] - fd["user_pos"][0]
                dy = frames_data[i + 1]["user_pos"][1] - fd["user_pos"][1]
                vx = dx / dt_frame
                vy = dy / dt_frame
            else:
                vx, vy = 0.0, 0.0
        else:
            vx, vy = user_velocities[-1] if user_velocities else (0.0, 0.0)
        user_velocities.append((vx, vy))

    # ----- build animation frames -----
    frames = []
    for fi, fd in enumerate(frames_data):
        ux, uy = fd["user_pos"]
        u_vx, u_vy = user_velocities[fi]
        u_angle = (
            np.arctan2(u_vy, u_vx) if (abs(u_vx) > 1e-6 or abs(u_vy) > 1e-6) else 0.0
        )

        frame_traces = []
        frame_indices = []

        if follow_user:
            # Trajectory (static in follow mode)
            frame_traces.append(
                go.Scatter(x=full_traj[:, 0] - ux, y=full_traj[:, 1] - uy)
            )
            frame_indices.append(traj_idx)
            # Obstacles
            if static_obstacles:
                for oi, obs in enumerate(static_obstacles):
                    ox = obs.x if hasattr(obs, "x") else obs[0]
                    oy = obs.y if hasattr(obs, "y") else obs[1]
                    orad = obs.radius if hasattr(obs, "radius") else obs[2]
                    frame_traces.append(
                        go.Scatter(
                            x=(ox - ux) + orad * np.cos(theta_obs),
                            y=(oy - uy) + orad * np.sin(theta_obs),
                        )
                    )
                    frame_indices.append(obs_start_idx + oi)
            # Safety & vicinity (centred on user)
            frame_traces.append(
                go.Scatter(
                    x=safety_r * np.cos(theta_circle), y=safety_r * np.sin(theta_circle)
                )
            )
            frame_indices.append(safety_idx)
            frame_traces.append(
                go.Scatter(
                    x=vicinity_r * np.cos(theta_circle),
                    y=vicinity_r * np.sin(theta_circle),
                )
            )
            frame_indices.append(vicinity_idx)

            # ---- User ellipse (centred at origin in follow mode) ----
            ux_ell, uy_ell = ellipse_points(0.0, 0.0, u_angle, USER_A, USER_B)
            frame_traces.append(
                go.Scatter(
                    x=ux_ell,
                    y=uy_ell,
                    mode="lines",
                    fill="toself",
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0,100,255,0.3)",
                )
            )
            frame_indices.append(user_ellipse_idx)
            # User centre marker
            frame_traces.append(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="blue",
                        line=dict(width=1, color="darkblue"),
                    ),
                )
            )
            frame_indices.append(user_marker_idx)

            # ---- Pedestrian data structures ----
            x_ell_all, y_ell_all = [], []  # ellipses
            ped_cx, ped_cy, ped_colors = [], [], []  # centre markers
            x_dir, y_dir = [], []  # heading lines
            x_gaze, y_gaze = [], []  # gaze lines
            x_cone_all, y_cone_all = [], []  # attention cones

            for ped in fd["pedestrians"]:
                # Unpack: note that we use different names to avoid shadowing the outer theta_circle
                px, py, vx, vy, mood, ped_theta, gaze_offset, fov_att, cone_scale = ped
                # ped_theta = motion angle (radians)
                # gaze_offset = p.theta_gaze (offset from heading, radians)
                # fov_att = half‑angle of attention cone (radians)

                # Heading angle for ellipse and direction line = ped_theta
                angle = ped_theta
                # Absolute gaze angle = ped_theta + gaze_offset
                gaze_angle = ped_theta + gaze_offset

                # ---- ellipse ----
                cx, cy = px - ux, py - uy
                xp, yp = ellipse_points(cx, cy, angle, PED_A, PED_B)
                x_ell_all.extend(xp)
                y_ell_all.extend(yp)
                x_ell_all.append(None)
                y_ell_all.append(None)

                # ---- centre marker ----
                ped_cx.append(cx)
                ped_cy.append(cy)
                ped_colors.append(MOOD_COLORS.get(mood, "orange"))

                # ---- heading line (direction of motion) ----
                heading_dx = np.cos(angle) * 0.5
                heading_dy = np.sin(angle) * 0.5
                sx, sy = cx, cy
                ex, ey = cx + heading_dx, cy + heading_dy
                x_dir.extend([sx, ex, None])
                y_dir.extend([sy, ey, None])

                # ---- gaze line ----
                gaze_dx = np.cos(gaze_angle) * 0.5
                gaze_dy = np.sin(gaze_angle) * 0.5
                gx, gy = cx, cy
                gex, gey = cx + gaze_dx, cy + gaze_dy
                x_gaze.extend([gx, gex, None])
                y_gaze.extend([gy, gey, None])

                # ---- attention cone ----
                if fov_att > 0:
                    cone_len = 1 + cone_scale  # length of cone (can be adjusted)
                    xc, yc = cone_points(
                        cx, cy, gaze_angle, fov_att, length=cone_len, n=15
                    )
                    x_cone_all.extend(xc)
                    y_cone_all.extend(yc)
                    x_cone_all.append(None)
                    y_cone_all.append(None)

            # ---- Robot ----
            if robot_data is not None and fi < len(robot_data["trajectory"]):
                rx_abs, ry_abs = robot_data["trajectory"][fi]
                if follow_user:
                    rx = rx_abs - ux
                    ry = ry_abs - uy
                else:
                    rx, ry = rx_abs, ry_abs

                # Robot body (circle outline)
                r_circle_x = rx + robot_data.get("radius", 0.18) * np.cos(theta_circle)
                r_circle_y = ry + robot_data.get("radius", 0.18) * np.sin(theta_circle)
                # Robot body trace (marker+circle)
                frame_traces.append(
                    go.Scatter(
                        x=r_circle_x,
                        y=r_circle_y,
                        mode="markers+lines",
                        marker=dict(
                            symbol="square",
                            size=14,
                            color="red",
                            line=dict(width=2, color="darkred"),
                        ),
                        line=dict(color="red", width=2),
                    )
                )
                frame_indices.append(robot_body_idx)

                # Robot trail (accumulate absolute, then offset)
                robot_trail_abs_x.append(rx_abs)
                robot_trail_abs_y.append(ry_abs)
                if follow_user:
                    trail_x = [x - ux for x in robot_trail_abs_x]
                    trail_y = [y - uy for y in robot_trail_abs_y]
                else:
                    trail_x = list(robot_trail_abs_x)
                    trail_y = list(robot_trail_abs_y)
                frame_traces.append(
                    go.Scatter(
                        x=trail_x,
                        y=trail_y,
                        mode="lines",
                        line=dict(color="red", width=1, dash="dot"),
                    )
                )
                frame_indices.append(robot_trail_idx)

            # Ellipses trace
            frame_traces.append(
                go.Scatter(
                    x=x_ell_all,
                    y=y_ell_all,
                    mode="lines",
                    fill="toself",
                    line=dict(color="black", width=1),
                    fillcolor="rgba(200,200,200,0.3)",
                )
            )
            frame_indices.append(peds_ellipse_idx)
            # Centre markers trace
            frame_traces.append(
                go.Scatter(
                    x=ped_cx,
                    y=ped_cy,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color=ped_colors,
                        line=dict(width=1, color="black"),
                    ),
                )
            )
            frame_indices.append(peds_marker_idx)
            # Heading lines
            frame_traces.append(
                go.Scatter(
                    x=x_dir, y=y_dir, mode="lines", line=dict(color="red", width=2)
                )
            )
            frame_indices.append(dir_idx)
            # Gaze lines
            frame_traces.append(
                go.Scatter(
                    x=x_gaze,
                    y=y_gaze,
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                )
            )
            frame_indices.append(gaze_idx)
            # Cones
            frame_traces.append(
                go.Scatter(
                    x=x_cone_all,
                    y=y_cone_all,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,255,0,0.2)",
                    line=dict(color="yellow", width=1),
                )
            )
            frame_indices.append(cone_idx)

        else:  # normal (non‑follow) mode
            # Safety & vicinity
            frame_traces.append(
                go.Scatter(
                    x=ux + safety_r * np.cos(theta_circle),
                    y=uy + safety_r * np.sin(theta_circle),
                )
            )
            frame_indices.append(safety_idx)
            frame_traces.append(
                go.Scatter(
                    x=ux + vicinity_r * np.cos(theta_circle),
                    y=uy + vicinity_r * np.sin(theta_circle),
                )
            )
            frame_indices.append(vicinity_idx)

            # ---- User ellipse ----
            ux_ell, uy_ell = ellipse_points(ux, uy, u_angle, USER_A, USER_B)
            frame_traces.append(
                go.Scatter(
                    x=ux_ell,
                    y=uy_ell,
                    mode="lines",
                    fill="toself",
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0,100,255,0.3)",
                )
            )
            frame_indices.append(user_ellipse_idx)
            # User centre marker
            frame_traces.append(
                go.Scatter(
                    x=[ux],
                    y=[uy],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="blue",
                        line=dict(width=1, color="darkblue"),
                    ),
                )
            )
            frame_indices.append(user_marker_idx)

            # ---- Pedestrian data structures ----
            x_ell_all, y_ell_all = [], []
            ped_cx, ped_cy, ped_colors = [], [], []
            x_dir, y_dir = [], []
            x_gaze, y_gaze = [], []
            x_cone_all, y_cone_all = [], []

            for ped in fd["pedestrians"]:
                px, py, vx, vy, mood, ped_theta, gaze_offset, fov_att = ped
                angle = ped_theta
                gaze_angle = ped_theta + gaze_offset

                # ellipse
                xp, yp = ellipse_points(px, py, angle, PED_A, PED_B)
                x_ell_all.extend(xp)
                y_ell_all.extend(yp)
                x_ell_all.append(None)
                y_ell_all.append(None)

                # centre marker
                ped_cx.append(px)
                ped_cy.append(py)
                ped_colors.append(MOOD_COLORS.get(mood, "orange"))

                # heading line
                heading_dx = np.cos(angle) * 0.5
                heading_dy = np.sin(angle) * 0.5
                sx, sy = px, py
                ex, ey = px + heading_dx, py + heading_dy
                x_dir.extend([sx, ex, None])
                y_dir.extend([sy, ey, None])

                # gaze line
                gaze_dx = np.cos(gaze_angle) * 0.5
                gaze_dy = np.sin(gaze_angle) * 0.5
                gx, gy = px, py
                gex, gey = px + gaze_dx, py + gaze_dy
                x_gaze.extend([gx, gex, None])
                y_gaze.extend([gy, gey, None])

                # attention cone
                if fov_att > 0:
                    cone_len = 1 + cone_scale
                    xc, yc = cone_points(
                        px, py, gaze_angle, fov_att, length=cone_len, n=15
                    )
                    x_cone_all.extend(xc)
                    y_cone_all.extend(yc)
                    x_cone_all.append(None)
                    y_cone_all.append(None)

            # Ellipses trace
            frame_traces.append(
                go.Scatter(
                    x=x_ell_all,
                    y=y_ell_all,
                    mode="lines",
                    fill="toself",
                    line=dict(color="black", width=1),
                    fillcolor="rgba(200,200,200,0.3)",
                )
            )
            frame_indices.append(peds_ellipse_idx)
            # Centre markers
            frame_traces.append(
                go.Scatter(
                    x=ped_cx,
                    y=ped_cy,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color=ped_colors,
                        line=dict(width=1, color="black"),
                    ),
                )
            )
            frame_indices.append(peds_marker_idx)
            # Heading lines
            frame_traces.append(
                go.Scatter(
                    x=x_dir, y=y_dir, mode="lines", line=dict(color="red", width=2)
                )
            )
            frame_indices.append(dir_idx)
            # Gaze lines
            frame_traces.append(
                go.Scatter(
                    x=x_gaze,
                    y=y_gaze,
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                )
            )
            frame_indices.append(gaze_idx)
            # Cones
            frame_traces.append(
                go.Scatter(
                    x=x_cone_all,
                    y=y_cone_all,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,255,0,0.2)",
                    line=dict(color="yellow", width=1),
                )
            )
            frame_indices.append(cone_idx)

        frames.append(go.Frame(data=frame_traces, traces=frame_indices, name=str(fi)))

    fig.frames = frames

    # -------------------- layout (unchanged) --------------------
    title = "SFM Spawner - FOLLOW MODE" if follow_user else "SFM Pedestrian Spawner"
    # ----- Compute frame duration from actual data -----
    if len(frames_data) > 1:
        frame_times = np.array([fd["time"] for fd in frames_data])
        median_dt = np.median(np.diff(frame_times))
        frame_duration_ms = max(20, min(1000, median_dt * 1000))  # clamp 20‑1000 ms
    else:
        frame_duration_ms = 50

    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis=dict(range=x_range, title="X (m)", constrain="domain"),
        yaxis=dict(
            range=y_range,
            title="Y (m)",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                y=0,
                x=0,
                yanchor="bottom",
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": frame_duration_ms,
                                    "redraw": False,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(prefix="Time: ", suffix=" s", visible=True),
                pad=dict(b=10, t=50),
                len=0.75,
                x=0.2,
                y=0,
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                        label=f"{frames_data[i]['time']:.1f}",
                        method="animate",
                    )
                    for i in range(0, len(frames_data), max(1, len(frames_data) // 20))
                ],
            )
        ],
        height=None,
        width=None,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        margin=dict(r=150, l=60, t=60, b=60),
    )
    return fig, logger


def create_animation_from_frames(
    frames_data: list,
    user_trajectory,
    static_obstacles=None,
    follow_user=False,
    follow_zoom_radius=10.0,
    spawner=None,
    robot_data=None,
    draw_pedestrian_safety: bool = False,
    frame_duration_ms: int = None,
    controller_name: str = "SFM",
) -> go.Figure:
    """
    Build a Plotly animation from pre‑collected frames_data.
    frames_data is a list of dicts with keys:
        'time', 'user_pos', 'pedestrians', optionally 'robot_pos'.
    robot_data : dict with 'trajectory' (list of (x,y)) and 'radius'.
    """
    full_traj = user_trajectory.get_full_trajectory()
    vicinity_r = spawner.vicinity_radius if spawner else 10.0
    safety_r = user_trajectory.safety_radius

    if follow_user:
        view_range = follow_zoom_radius
        x_range = [-view_range, view_range]
        y_range = [-view_range, view_range]
        init_ux, init_uy = frames_data[0]["user_pos"]
    else:
        x_range = [0, CONFIG.env_width]
        y_range = [0, CONFIG.env_height]
        init_ux, init_uy = 0, 0

    theta_circle = np.linspace(0, 2 * np.pi, 50)
    theta_obs = np.linspace(0, 2 * np.pi, 30)
    PED_A, PED_B = 0.25, 0.35
    USER_A, USER_B = 0.25, 0.35

    fig = go.Figure()
    trace_idx = 0

    # Trajectory line
    if follow_user:
        traj_x, traj_y = full_traj[:, 0] - init_ux, full_traj[:, 1] - init_uy
    else:
        traj_x, traj_y = full_traj[:, 0], full_traj[:, 1]
    fig.add_trace(
        go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines",
            line=dict(color="lightblue", width=2, dash="dot"),
            name="Trajectory",
        )
    )
    traj_idx = trace_idx
    trace_idx += 1

    # Static obstacles
    obs_start_idx = trace_idx
    if static_obstacles:
        for obs in static_obstacles:
            ox = obs.x if hasattr(obs, "x") else obs[0]
            oy = obs.y if hasattr(obs, "y") else obs[1]
            orad = obs.radius if hasattr(obs, "radius") else obs[2]
            if follow_user:
                ox -= init_ux
                oy -= init_uy
            fig.add_trace(
                go.Scatter(
                    x=ox + orad * np.cos(theta_obs),
                    y=oy + orad * np.sin(theta_obs),
                    fill="toself",
                    fillcolor="rgba(128,128,128,0.5)",
                    line=dict(color="gray"),
                    name="Obstacle",
                    showlegend=False,
                )
            )
            trace_idx += 1

    # Safety & vicinity circles
    ux_disp = 0 if follow_user else frames_data[0]["user_pos"][0]
    uy_disp = 0 if follow_user else frames_data[0]["user_pos"][1]
    fig.add_trace(
        go.Scatter(
            x=ux_disp + safety_r * np.cos(theta_circle),
            y=uy_disp + safety_r * np.sin(theta_circle),
            mode="lines",
            line=dict(color="blue", width=1, dash="dash"),
            name="Safety Zone",
        )
    )
    safety_idx = trace_idx
    trace_idx += 1
    fig.add_trace(
        go.Scatter(
            x=ux_disp + vicinity_r * np.cos(theta_circle),
            y=uy_disp + vicinity_r * np.sin(theta_circle),
            mode="lines",
            line=dict(color="green", width=1, dash="dot"),
            name="Vicinity",
            opacity=0.5,
        )
    )
    vicinity_idx = trace_idx
    trace_idx += 1

    # User ellipse + marker
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            line=dict(color="blue", width=2),
            fillcolor="rgba(0,100,255,0.3)",
            name="User (body)",
        )
    )
    user_ellipse_idx = trace_idx
    trace_idx += 1
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=8,
                color="blue",
                line=dict(width=1, color="darkblue"),
            ),
            name="User",
        )
    )
    user_marker_idx = trace_idx
    trace_idx += 1

    # Pedestrian ellipses + markers
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            line=dict(color="black", width=1),
            fillcolor="rgba(200,200,200,0.3)",
            name="Pedestrian body",
        )
    )
    peds_ellipse_idx = trace_idx
    trace_idx += 1

    # ----- NEW: Social comfort zone (egg contours) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            line=dict(color="purple", width=0.5),
            fillcolor="rgba(255,0,255,0.15)",
            name="Social comfort",
        )
    )
    social_idx = trace_idx
    trace_idx += 1

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(symbol="circle", size=8, line=dict(width=1, color="black")),
            name="Pedestrians",
        )
    )
    peds_marker_idx = trace_idx
    trace_idx += 1

    # Direction lines (heading) & gaze lines & cones
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=2),
            name="Direction (heading)",
        )
    )
    dir_idx = trace_idx
    trace_idx += 1
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="cyan", width=2, dash="dot"),
            name="Gaze direction",
        )
    )
    gaze_idx = trace_idx
    trace_idx += 1
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            fillcolor="rgba(255,255,0,0.2)",
            line=dict(color="yellow", width=1),
            name="Attention cone",
        )
    )
    cone_idx = trace_idx
    trace_idx += 1

    # ----- New: User motion line (red, thicker) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=3, dash="solid"),
            name="User motion",
        )
    )
    user_motion_idx = trace_idx
    trace_idx += 1

    # ----- New: User facing line (blue, dashed) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="blue", width=3, dash="dash"),
            name="User facing",
        )
    )
    user_facing_idx = trace_idx
    trace_idx += 1

    # ----- New: Future path line (green, dashed) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="green", width=2, dash="dash"),
            name="Future path",
        )
    )
    future_path_idx = trace_idx
    trace_idx += 1

    # ----- New: Goal marker (green star) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                symbol="star",
                size=12,
                color="green",
                line=dict(width=1, color="darkgreen"),
            ),
            name="Goal",
        )
    )
    goal_idx = trace_idx
    trace_idx += 1

    # ----- New: Robot arc line (orange) -----
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="orange", width=2),
            name="Robot future",
        )
    )
    robot_arc_idx = trace_idx
    trace_idx += 1

    # Robot traces (if data present)
    robot_body_idx = None
    robot_trail_idx = None
    robot_trail_abs_x = []
    robot_trail_abs_y = []
    # Robot body polygon (triangle)
    if robot_data is not None:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                fill="toself",
                line=dict(color="red", width=1),
                fillcolor="rgba(255,0,0,0.5)",
                name="Robot",
            )
        )
        robot_body_idx = trace_idx
        trace_idx += 1
        # Trail
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="red", width=1, dash="dot"),
                name="Robot path",
            )
        )
        robot_trail_idx = trace_idx
        trace_idx += 1

    # ----- Pedestrian safety circles (optional) -----
    ped_safety_idx = None
    if draw_pedestrian_safety:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="orange", width=1, dash="dot"),
                name="Ped safety margin",
            )
        )
        ped_safety_idx = trace_idx
        trace_idx += 1

    # Mood legend
    unique_moods = set()
    for f in frames_data:
        for ped in f["pedestrians"]:
            unique_moods.add(ped[4])
    for mood in sorted(unique_moods):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=MOOD_COLORS.get(mood, "orange"),
                    line=dict(width=1, color="black"),
                ),
                name=f"  {mood}",
                legendgroup="moods",
            )
        )
        trace_idx += 1

    # Robot maneuver legend
    for label, color in [
        ("Normal", "red"),
        ("Overtaking", "orange"),
        ("Repulsion", "magenta"),
        ("Parking", "green"),
        ("Soft Recovery", "cyan"),
        ("Rotation", "blue"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1, color="black")),
                name=label,
                legendgroup="maneuvers",
            )
        )
        trace_idx += 1

        # ----- NEW: Target path trace -----
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="purple", width=2, dash="dash"),
                name="Target path",
            )
        )
        target_path_idx = trace_idx
        trace_idx += 1

    # Helpers (same as original)
    def ellipse_points(cx, cy, angle_rad, a, b, n=30):
        t = np.linspace(0, 2 * np.pi, n)
        x_ell = a * np.cos(t)
        y_ell = b * np.sin(t)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = x_ell * cos_a - y_ell * sin_a
        y_rot = x_ell * sin_a + y_ell * cos_a
        return cx + x_rot, cy + y_rot

    def cone_points(cx, cy, direction_rad, half_angle_rad, length=1.2, n=20):
        start_angle = direction_rad - half_angle_rad
        end_angle = direction_rad + half_angle_rad
        angles = np.linspace(start_angle, end_angle, n)
        x_arc = cx + length * np.cos(angles)
        y_arc = cy + length * np.sin(angles)
        return np.concatenate(([cx], x_arc, [cx])), np.concatenate(([cy], y_arc, [cy]))

    # Pre‑compute user velocities (for ellipse orientation)
    user_velocities = []
    for i, fd in enumerate(frames_data):
        if i < len(frames_data) - 1:
            dt_frame = frames_data[i + 1]["time"] - fd["time"]
            if dt_frame > 0:
                dx = frames_data[i + 1]["user_pos"][0] - fd["user_pos"][0]
                dy = frames_data[i + 1]["user_pos"][1] - fd["user_pos"][1]
                vx, vy = dx / dt_frame, dy / dt_frame
            else:
                vx = vy = 0.0
        else:
            vx, vy = user_velocities[-1] if user_velocities else (0.0, 0.0)
        user_velocities.append((vx, vy))

    # Build frames
    frames = []
    for fi, fd in enumerate(frames_data):
        ux, uy = fd["user_pos"]
        u_vx, u_vy = user_velocities[fi]
        u_angle = (
            np.arctan2(u_vy, u_vx) if (abs(u_vx) > 1e-6 or abs(u_vy) > 1e-6) else 0.0
        )

        frame_traces = []
        frame_indices = []

        if follow_user:
            frame_traces.append(
                go.Scatter(x=full_traj[:, 0] - ux, y=full_traj[:, 1] - uy)
            )
            frame_indices.append(traj_idx)
            if static_obstacles:
                for oi, obs in enumerate(static_obstacles):
                    ox = obs.x if hasattr(obs, "x") else obs[0]
                    oy = obs.y if hasattr(obs, "y") else obs[1]
                    orad = obs.radius if hasattr(obs, "radius") else obs[2]
                    frame_traces.append(
                        go.Scatter(
                            x=(ox - ux) + orad * np.cos(theta_obs),
                            y=(oy - uy) + orad * np.sin(theta_obs),
                        )
                    )
                    frame_indices.append(obs_start_idx + oi)
            frame_traces.append(
                go.Scatter(
                    x=safety_r * np.cos(theta_circle), y=safety_r * np.sin(theta_circle)
                )
            )
            frame_indices.append(safety_idx)
            frame_traces.append(
                go.Scatter(
                    x=vicinity_r * np.cos(theta_circle),
                    y=vicinity_r * np.sin(theta_circle),
                )
            )
            frame_indices.append(vicinity_idx)

            # User ellipse + marker
            ux_ell, uy_ell = ellipse_points(0, 0, u_angle, USER_A, USER_B)
            frame_traces.append(
                go.Scatter(
                    x=ux_ell,
                    y=uy_ell,
                    mode="lines",
                    fill="toself",
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0,100,255,0.3)",
                )
            )
            frame_indices.append(user_ellipse_idx)
            frame_traces.append(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="blue",
                        line=dict(width=1, color="darkblue"),
                    ),
                )
            )
            frame_indices.append(user_marker_idx)

            # Pedestrians
            x_ell_all, y_ell_all = [], []
            ped_cx, ped_cy, ped_colors = [], [], []
            x_dir, y_dir = [], []
            x_gaze, y_gaze = [], []
            x_cone_all, y_cone_all = [], []
            x_social_all, y_social_all = [], []
            for ped in fd["pedestrians"]:
                px, py, vx, vy, mood, ped_theta, gaze_offset, fov_att, w_att = ped
                angle = ped_theta
                gaze_angle = ped_theta + gaze_offset
                cx, cy = px - ux, py - uy
                xp, yp = ellipse_points(cx, cy, angle, PED_A, PED_B)
                x_ell_all.extend(xp)
                y_ell_all.extend(yp)
                x_ell_all.append(None)
                y_ell_all.append(None)
                ped_cx.append(cx)
                ped_cy.append(cy)
                ped_colors.append(MOOD_COLORS.get(mood, "orange"))
                heading_dx = np.cos(angle) * 0.5
                heading_dy = np.sin(angle) * 0.5
                x_dir.extend([cx, cx + heading_dx, None])
                y_dir.extend([cy, cy + heading_dy, None])
                gaze_dx = np.cos(gaze_angle) * 0.5
                gaze_dy = np.sin(gaze_angle) * 0.5
                x_gaze.extend([cx, cx + gaze_dx, None])
                y_gaze.extend([cy, cy + gaze_dy, None])
                if fov_att > 0:
                    cone_len = 1 + w_att
                    xc, yc = cone_points(
                        cx, cy, gaze_angle, fov_att, length=cone_len, n=15
                    )
                    x_cone_all.extend(xc)
                    y_cone_all.extend(yc)
                    x_cone_all.append(None)
                    y_cone_all.append(None)
                # Social comfort contour
                mood_str = ped[4]  # mood name
                params = CUSTOM_MOODS.get(mood_str, {})
                B_val = params.get("B_ped", 0.5)
                lam_val = params.get("lam_base", 0.5)
                phi_val = params.get("phi_fov", np.deg2rad(90))
                gaze_val = params.get("theta_gaze", 0.0)
                scx, scy = egg_contour(cx, cy, angle, B_val, lam_val, gaze_val, scale=CONFIG.social_zone_scale)
                x_social_all.extend(scx)
                y_social_all.extend(scy)
                x_social_all.append(None)
                y_social_all.append(None)

            frame_traces.append(
                go.Scatter(
                    x=x_social_all,
                    y=y_social_all,
                    mode="lines",
                    fill="toself",
                    line=dict(color="purple", width=0.5),
                    fillcolor="rgba(255,0,255,0.15)",
                )
            )
            frame_indices.append(social_idx)
            frame_traces.append(
                go.Scatter(
                    x=x_ell_all,
                    y=y_ell_all,
                    mode="lines",
                    fill="toself",
                    line=dict(color="black", width=1),
                    fillcolor="rgba(200,200,200,0.3)",
                )
            )
            frame_indices.append(peds_ellipse_idx)
            frame_traces.append(
                go.Scatter(
                    x=ped_cx,
                    y=ped_cy,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color=ped_colors,
                        line=dict(width=1, color="black"),
                    ),
                )
            )
            frame_indices.append(peds_marker_idx)
            frame_traces.append(
                go.Scatter(
                    x=x_dir, y=y_dir, mode="lines", line=dict(color="red", width=2)
                )
            )
            frame_indices.append(dir_idx)
            frame_traces.append(
                go.Scatter(
                    x=x_gaze,
                    y=y_gaze,
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                )
            )
            frame_indices.append(gaze_idx)
            frame_traces.append(
                go.Scatter(
                    x=x_cone_all,
                    y=y_cone_all,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,255,0,0.2)",
                    line=dict(color="yellow", width=1),
                )
            )
            frame_indices.append(cone_idx)

            # ---- User motion line ----
            if "user_motion_angle" in fd:
                angle = fd["user_motion_angle"]
                lx = 0.0
                ly = 0.0  # user at (0,0) in follow mode
                ex = lx + 1.0 * np.cos(angle)
                ey = ly + 1.0 * np.sin(angle)
                frame_traces.append(
                    go.Scatter(
                        x=[lx, ex],
                        y=[ly, ey],
                        mode="lines",
                        line=dict(color="red", width=3, dash="solid"),
                    )
                )
                frame_indices.append(user_motion_idx)
            else:
                # keep empty
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(user_motion_idx)

            # ---- User facing line ----
            if "user_facing_angle" in fd:
                angle = fd["user_facing_angle"]
                ex = lx + 1.0 * np.cos(angle)
                ey = ly + 1.0 * np.sin(angle)
                frame_traces.append(
                    go.Scatter(
                        x=[lx, ex],
                        y=[ly, ey],
                        mode="lines",
                        line=dict(color="blue", width=3, dash="dash"),
                    )
                )
                frame_indices.append(user_facing_idx)
            else:
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(user_facing_idx)

            # ---- Future path and goal ----
            if "future_path_x" in fd and "goal_pos" in fd:
                path_x = [x - ux for x in fd["future_path_x"]]
                path_y = [y - uy for y in fd["future_path_y"]]
                frame_traces.append(
                    go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode="lines",
                        line=dict(color="green", width=2, dash="dash"),
                    )
                )
                frame_indices.append(future_path_idx)
                gx, gy = fd["goal_pos"]
                frame_traces.append(
                    go.Scatter(
                        x=[gx - ux],
                        y=[gy - uy],
                        mode="markers",
                        marker=dict(
                            symbol="star",
                            size=12,
                            color="green",
                            line=dict(width=1, color="darkgreen"),
                        ),
                    )
                )
                frame_indices.append(goal_idx)
            else:
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(future_path_idx)
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(goal_idx)

            # ---- Robot arc ----
            if "robot_arc_x" in fd and robot_data is not None:
                arc_x = [x - ux for x in fd["robot_arc_x"]]
                arc_y = [y - uy for y in fd["robot_arc_y"]]
                frame_traces.append(
                    go.Scatter(
                        x=arc_x,
                        y=arc_y,
                        mode="lines",
                        line=dict(color="orange", width=2),
                    )
                )
                frame_indices.append(robot_arc_idx)
            elif robot_data is not None:
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(robot_arc_idx)

            # ---- Robot body polygon (triangle) ----
            if robot_data is not None and fi < len(robot_data["trajectory"]):
                rx_abs, ry_abs = robot_data["trajectory"][fi]
                rx = rx_abs - ux
                ry = ry_abs - uy
                # Triangle points: nose at heading, back two points
                theta_robot = fd.get("robot_theta", 0.0)
                length = 0.5  # length of triangle
                width = 0.3
                # nose
                nx = rx + length * np.cos(theta_robot)
                ny = ry + length * np.sin(theta_robot)
                # back‑left
                bl_angle = theta_robot + np.deg2rad(135)
                blx = rx + width * np.cos(bl_angle)
                bly = ry + width * np.sin(bl_angle)
                # back‑right
                br_angle = theta_robot - np.deg2rad(135)
                brx = rx + width * np.cos(br_angle)
                bry = ry + width * np.sin(br_angle)
                poly_x = [nx, blx, brx, nx]
                poly_y = [ny, bly, bry, ny]

                overtaking = fd.get("overtaking_active", False)
                parking = fd.get("parking_active", False)
                repulsion = fd.get("repulsion_active", False)
                soft_rec = fd.get("soft_recovery_active", False)
                rotation = fd.get("rotation_active", False)

                if overtaking:
                    fill_color = "rgba(255,165,0,0.7)"
                    line_color = "orange"
                elif parking:
                    fill_color = "rgba(0,255,0,0.7)"
                    line_color = "green"
                elif repulsion:
                    fill_color = "rgba(255,0,255,0.7)"
                    line_color = "magenta"
                elif soft_rec:
                    fill_color = "rgba(0,200,255,0.7)"
                    line_color = "cyan"
                elif rotation:  # <-- new condition
                    fill_color = "rgba(0,0,255,0.7)"
                    line_color = "blue"
                else:
                    fill_color = "rgba(255,0,0,0.5)"
                    line_color = "red"

                frame_traces.append(
                    go.Scatter(
                        x=poly_x,
                        y=poly_y,
                        mode="lines",
                        fill="toself",
                        line=dict(color=line_color, width=1),
                        fillcolor=fill_color,
                    )
                )
                frame_indices.append(robot_body_idx)
                # Trail
                robot_trail_abs_x.append(rx_abs)
                robot_trail_abs_y.append(ry_abs)
                trail_x = [x - ux for x in robot_trail_abs_x]
                trail_y = [y - uy for y in robot_trail_abs_y]
                frame_traces.append(
                    go.Scatter(
                        x=trail_x,
                        y=trail_y,
                        mode="lines",
                        line=dict(color="red", width=1, dash="dot"),
                    )
                )
                frame_indices.append(robot_trail_idx)
            # ---- Pedestrian safety circles ----
            if ped_safety_idx is not None:
                if draw_pedestrian_safety:
                    safety_x, safety_y = [], []
                    for ped in fd["pedestrians"]:
                        px, py = ped[0], ped[1]
                        cx = px - ux
                        cy = py - uy
                        circle_x = cx + CONFIG.safety_margin * np.cos(theta_circle)
                        circle_y = cy + CONFIG.safety_margin * np.sin(theta_circle)
                        safety_x.extend(circle_x)
                        safety_y.extend(circle_y)
                        safety_x.append(None)
                        safety_y.append(None)
                    frame_traces.append(
                        go.Scatter(
                            x=safety_x,
                            y=safety_y,
                            mode="lines",
                            line=dict(color="orange", width=1, dash="dot"),
                        )
                    )
                else:
                    frame_traces.append(go.Scatter(x=[], y=[]))

                frame_indices.append(ped_safety_idx)

                # ---- Target path ----
                if "target_path_x" in fd and "target_path_y" in fd:
                    tpx = [x - ux for x in fd["target_path_x"]]
                    tpy = [y - uy for y in fd["target_path_y"]]
                    frame_traces.append(
                        go.Scatter(
                            x=tpx,
                            y=tpy,
                            mode="lines",
                            line=dict(color="purple", width=2, dash="dash"),
                        )
                    )
                else:
                    frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(target_path_idx)
        else:
            # Safety & vicinity
            frame_traces.append(
                go.Scatter(
                    x=ux + safety_r * np.cos(theta_circle),
                    y=uy + safety_r * np.sin(theta_circle),
                )
            )
            frame_indices.append(safety_idx)
            frame_traces.append(
                go.Scatter(
                    x=ux + vicinity_r * np.cos(theta_circle),
                    y=uy + vicinity_r * np.sin(theta_circle),
                )
            )
            frame_indices.append(vicinity_idx)

            # ---- User ellipse ----
            ux_ell, uy_ell = ellipse_points(ux, uy, u_angle, USER_A, USER_B)
            frame_traces.append(
                go.Scatter(
                    x=ux_ell,
                    y=uy_ell,
                    mode="lines",
                    fill="toself",
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0,100,255,0.3)",
                )
            )
            frame_indices.append(user_ellipse_idx)
            # User centre marker
            frame_traces.append(
                go.Scatter(
                    x=[ux],
                    y=[uy],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="blue",
                        line=dict(width=1, color="darkblue"),
                    ),
                )
            )
            frame_indices.append(user_marker_idx)

            # ---- Pedestrian data structures ----
            x_ell_all, y_ell_all = [], []
            ped_cx, ped_cy, ped_colors = [], [], []
            x_dir, y_dir = [], []
            x_gaze, y_gaze = [], []
            x_cone_all, y_cone_all = [], []
            x_social_all, y_social_all = [], []
            for ped in fd["pedestrians"]:
                # Unpack all 9 values now present in frames_data
                px, py, vx, vy, mood, ped_theta, gaze_offset, fov_att, w_att = ped
                angle = ped_theta
                gaze_angle = ped_theta + gaze_offset

                # ellipse
                xp, yp = ellipse_points(px, py, angle, PED_A, PED_B)
                x_ell_all.extend(xp)
                y_ell_all.extend(yp)
                x_ell_all.append(None)
                y_ell_all.append(None)

                # centre marker
                ped_cx.append(px)
                ped_cy.append(py)
                ped_colors.append(MOOD_COLORS.get(mood, "orange"))

                # heading line
                heading_dx = np.cos(angle) * 0.5
                heading_dy = np.sin(angle) * 0.5
                sx, sy = px, py
                ex, ey = px + heading_dx, py + heading_dy
                x_dir.extend([sx, ex, None])
                y_dir.extend([sy, ey, None])

                # gaze line
                gaze_dx = np.cos(gaze_angle) * 0.5
                gaze_dy = np.sin(gaze_angle) * 0.5
                gx, gy = px, py
                gex, gey = px + gaze_dx, py + gaze_dy
                x_gaze.extend([gx, gex, None])
                y_gaze.extend([gy, gey, None])

                # attention cone
                if fov_att > 0:
                    cone_len = 1 + w_att  # use w_att, not cone_scale
                    xc, yc = cone_points(
                        px, py, gaze_angle, fov_att, length=cone_len, n=15
                    )
                    x_cone_all.extend(xc)
                    y_cone_all.extend(yc)
                    x_cone_all.append(None)
                    y_cone_all.append(None)
                # Social comfort contour
                mood_str = ped[4]  # mood name
                params = CUSTOM_MOODS.get(mood_str, {})
                B_val = params.get("B_ped", 0.5)
                lam_val = params.get("lam_base", 0.5)
                phi_val = params.get("phi_fov", np.deg2rad(90))
                gaze_val = params.get("theta_gaze", 0.0)
                scx, scy = egg_contour(cx, cy, angle, B_val, lam_val, phi_val, gaze_val, scale=CONFIG.social_zone_scale)
                x_social_all.extend(scx)
                y_social_all.extend(scy)
                x_social_all.append(None)
                y_social_all.append(None)

            frame_traces.append(
                go.Scatter(
                    x=x_social_all,
                    y=y_social_all,
                    mode="lines",
                    fill="toself",
                    line=dict(color="purple", width=0.5),
                    fillcolor="rgba(255,0,255,0.15)",
                )
            )
            frame_indices.append(social_idx)

            # Ellipses trace
            frame_traces.append(
                go.Scatter(
                    x=x_ell_all,
                    y=y_ell_all,
                    mode="lines",
                    fill="toself",
                    line=dict(color="black", width=1),
                    fillcolor="rgba(200,200,200,0.3)",
                )
            )
            frame_indices.append(peds_ellipse_idx)
            # Centre markers
            frame_traces.append(
                go.Scatter(
                    x=ped_cx,
                    y=ped_cy,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color=ped_colors,
                        line=dict(width=1, color="black"),
                    ),
                )
            )
            frame_indices.append(peds_marker_idx)
            # Heading lines
            frame_traces.append(
                go.Scatter(
                    x=x_dir, y=y_dir, mode="lines", line=dict(color="red", width=2)
                )
            )
            frame_indices.append(dir_idx)
            # Gaze lines
            frame_traces.append(
                go.Scatter(
                    x=x_gaze,
                    y=y_gaze,
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dot"),
                )
            )
            frame_indices.append(gaze_idx)
            # Cones
            frame_traces.append(
                go.Scatter(
                    x=x_cone_all,
                    y=y_cone_all,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,255,0,0.2)",
                    line=dict(color="yellow", width=1),
                )
            )
            frame_indices.append(cone_idx)

            # ---- User motion line ----
            if "user_motion_angle" in fd:
                angle = fd["user_motion_angle"]
                lx, ly = ux, uy  # absolute user position
                ex = lx + 1.0 * np.cos(angle)
                ey = ly + 1.0 * np.sin(angle)
                frame_traces.append(
                    go.Scatter(
                        x=[lx, ex],
                        y=[ly, ey],
                        mode="lines",
                        line=dict(color="red", width=3, dash="solid"),
                    )
                )
                frame_indices.append(user_motion_idx)
            else:
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(user_motion_idx)

            # ---- User facing line ----
            if "user_facing_angle" in fd:
                angle = fd["user_facing_angle"]
                ex = (
                    lx + 1.0 * np.cos(angle)
                    if "lx" in dir()
                    else ux + 1.0 * np.cos(angle)
                )
                ey = (
                    ly + 1.0 * np.sin(angle)
                    if "ly" in dir()
                    else uy + 1.0 * np.sin(angle)
                )
                frame_traces.append(
                    go.Scatter(
                        x=[ux, ex],
                        y=[uy, ey],
                        mode="lines",
                        line=dict(color="blue", width=3, dash="dash"),
                    )
                )
                frame_indices.append(user_facing_idx)
            else:
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(user_facing_idx)

            # ---- Future path and goal ----
            if "future_path_x" in fd and "goal_pos" in fd:
                frame_traces.append(
                    go.Scatter(
                        x=fd["future_path_x"],
                        y=fd["future_path_y"],
                        mode="lines",
                        line=dict(color="green", width=2, dash="dash"),
                    )
                )
                frame_indices.append(future_path_idx)
                gx, gy = fd["goal_pos"]
                frame_traces.append(
                    go.Scatter(
                        x=[gx],
                        y=[gy],
                        mode="markers",
                        marker=dict(
                            symbol="star",
                            size=12,
                            color="green",
                            line=dict(width=1, color="darkgreen"),
                        ),
                    )
                )
                frame_indices.append(goal_idx)
            else:
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(future_path_idx)
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(goal_idx)

            # ---- Robot arc ----
            if "robot_arc_x" in fd and robot_data is not None:
                frame_traces.append(
                    go.Scatter(
                        x=fd["robot_arc_x"],
                        y=fd["robot_arc_y"],
                        mode="lines",
                        line=dict(color="orange", width=2),
                    )
                )
                frame_indices.append(robot_arc_idx)
            elif robot_data is not None:
                frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(robot_arc_idx)

            # ---- Robot body polygon ----
            if robot_data is not None and fi < len(robot_data["trajectory"]):
                rx_abs, ry_abs = robot_data["trajectory"][fi]
                theta_robot = fd.get("robot_theta", 0.0)
                length = 0.5
                width = 0.3
                nx = rx_abs + length * np.cos(theta_robot)
                ny = ry_abs + length * np.sin(theta_robot)
                bl_angle = theta_robot + np.deg2rad(135)
                blx = rx_abs + width * np.cos(bl_angle)
                bly = ry_abs + width * np.sin(bl_angle)
                br_angle = theta_robot - np.deg2rad(135)
                brx = rx_abs + width * np.cos(br_angle)
                bry = ry_abs + width * np.sin(br_angle)
                poly_x = [nx, blx, brx, nx]
                poly_y = [ny, bly, bry, ny]
                overtaking = fd.get("overtaking_active", False)
                parking = fd.get("parking_active", False)
                repulsion = fd.get("repulsion_active", False)

                if overtaking:
                    fill_color = "rgba(255,165,0,0.7)"  # orange
                    line_color = "orange"
                elif parking:
                    fill_color = "rgba(0,255,0,0.7)"  # green
                    line_color = "green"
                elif repulsion:
                    fill_color = "rgba(255,0,255,0.7)"  # magenta
                    line_color = "magenta"
                else:
                    fill_color = "rgba(255,0,0,0.5)"  # red (normal)
                    line_color = "red"
                frame_traces.append(
                    go.Scatter(
                        x=poly_x,
                        y=poly_y,
                        mode="lines",
                        fill="toself",
                        line=dict(color=line_color, width=1),
                        fillcolor=fill_color,
                    )
                )
                frame_indices.append(robot_body_idx)
                robot_trail_abs_x.append(rx_abs)
                robot_trail_abs_y.append(ry_abs)
                frame_traces.append(
                    go.Scatter(
                        x=robot_trail_abs_x,
                        y=robot_trail_abs_y,
                        mode="lines",
                        line=dict(color="red", width=1, dash="dot"),
                    )
                )
                frame_indices.append(robot_trail_idx)

            # ---- Pedestrian safety circles ----
            if ped_safety_idx is not None:
                if draw_pedestrian_safety:
                    safety_x, safety_y = [], []
                    for ped in fd["pedestrians"]:
                        px, py = ped[0], ped[1]
                        circle_x = px + CONFIG.safety_margin * np.cos(theta_circle)
                        circle_y = py + CONFIG.safety_margin * np.sin(theta_circle)
                        safety_x.extend(circle_x)
                        safety_y.extend(circle_y)
                        safety_x.append(None)
                        safety_y.append(None)
                    frame_traces.append(
                        go.Scatter(
                            x=safety_x,
                            y=safety_y,
                            mode="lines",
                            line=dict(color="orange", width=1, dash="dot"),
                        )
                    )
                else:
                    frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(ped_safety_idx)

                # ---- Target path ----
                if "target_path_x" in fd and "target_path_y" in fd:
                    frame_traces.append(
                        go.Scatter(
                            x=fd["target_path_x"],
                            y=fd["target_path_y"],
                            mode="lines",
                            line=dict(color="purple", width=2, dash="dash"),
                        )
                    )
                else:
                    frame_traces.append(go.Scatter(x=[], y=[]))
                frame_indices.append(target_path_idx)

        frames.append(go.Frame(data=frame_traces, traces=frame_indices, name=str(fi)))

    fig.frames = frames

    # ----- Compute frame duration from actual data -----
    if frame_duration_ms is not None:
        frame_dur = frame_duration_ms
    else:
        if len(frames_data) > 1:
            frame_times = np.array([fd["time"] for fd in frames_data])
            median_dt = np.median(np.diff(frame_times))
            frame_dur = max(20, min(1000, int(median_dt * 1000)))
        else:
            frame_dur = 50

    title = (
        f"Robot {controller_name} Demo" if robot_data else f"{controller_name} Spawner"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis=dict(range=x_range, title="X (m)", constrain="domain"),
        yaxis=dict(
            range=y_range,
            title="Y (m)",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                y=0,
                x=0,
                yanchor="bottom",
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_dur, "redraw": False},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(prefix="Time: ", suffix=" s", visible=True),
                pad=dict(b=10, t=50),
                len=0.75,
                x=0.2,
                y=0,
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                        label=f"{frames_data[i]['time']:.1f}s",
                        method="animate",
                    )
                    for i in range(0, len(frames_data), max(1, len(frames_data) // 20))
                ],
            )
        ],
        height=None,
        width=None,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        margin=dict(r=150, l=60, t=60, b=60),
    )
    return fig
