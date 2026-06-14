"""Compare all controllers in a grid of animations with unified playback – full feature set."""
import argparse, webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- constants ----------
MOOD_COLORS = {
    "Brisk_Individualist": "red",
    "Relaxed_Ped": "green",
    "Social_Walker": "orange",
    "Social_Walker_v2": "purple",
    "real_crowd": "gray",
}
VICINITY_R = 10.0       # vicinity radius (fixed)
PED_A, PED_B = 0.25, 0.35
USER_A, USER_B = 0.25, 0.35

# ---------- helpers ----------
def ellipse_points(cx, cy, angle, a, b, n=30):
    t = np.linspace(0, 2*np.pi, n)
    xe = a*np.cos(t); ye = b*np.sin(t)
    c, s = np.cos(angle), np.sin(angle)
    return cx + xe*c - ye*s, cy + xe*s + ye*c

def cone_points(cx, cy, direction, half, length=1.2, n=20):
    start, end = direction - half, direction + half
    ang = np.linspace(start, end, n)
    xa = cx + length*np.cos(ang); ya = cy + length*np.sin(ang)
    return np.concatenate(([cx], xa, [cx])), np.concatenate(([cy], ya, [cy]))

def load_history(path):
    df = pd.read_csv(path)
    robot = df[df["agent_type"] == "robot"]
    ctrl = robot["controller"].iloc[0] if not robot.empty else "unknown"
    # user state array (px, py, yaw, v_forw, v_orth)
    user = df[df["agent_type"] == "user"]
    safety_r = user["safety_radius"].iloc[0] if "safety_radius" in user.columns else 1.0
    yaw = user["theta"].values; vx = user["vx"].values; vy = user["vy"].values
    c_yaw = np.cos(yaw); s_yaw = np.sin(yaw)
    vf = vx*c_yaw + vy*s_yaw; vo = -vx*s_yaw + vy*c_yaw
    state = np.column_stack([user["x"], user["y"], yaw, vf, vo])
    # static obstacles
    stat = df[df["agent_type"] == "static_obstacle"]
    obs = []
    if not stat.empty:
        for _, g in stat.groupby("agent_id"):
            r = g.iloc[0]; obs.append((r["x"], r["y"], r["radius"]))
    return df, ctrl, state, obs, safety_r

def build_frames(df):
    times = sorted(df["time"].unique())
    frames = []; traj = []
    for t in times:
        rows = df[df["time"] == t]
        u = rows[rows["agent_type"] == "user"].iloc[0]
        up = (u["x"], u["y"]); uf = u["theta"]
        uvx, uvy = u["vx"], u["vy"]; uspeed = np.hypot(uvx, uvy)
        um = np.arctan2(uvy, uvx) if uspeed > 0.05 else uf

        peds = []
        for _, p in rows[rows["agent_type"] == "pedestrian"].iterrows():
            pvx, pvy = p["vx"], p["vy"]; psp = np.hypot(pvx, pvy)
            pth = np.arctan2(pvy, pvx) if psp > 0.05 else p["theta"]
            # Read new fields, fallback to 0.0 if not present
            goff = p.get("theta_gaze", 0.0)
            fov_att = p.get("fov_att", 0.0)
            w_att = p.get("w_att", 0.0)
            peds.append((p["x"], p["y"], pvx, pvy,
                         p.get("mood", "real_crowd"),
                         pth, goff, fov_att, w_att))

        r = rows[rows["agent_type"] == "robot"].iloc[0]
        rp = (r["x"], r["y"]); rth = r["theta"]
        goal = (r["goal_x"], r["goal_y"]) if "goal_x" in r else rp
        fpath_x, fpath_y = [up[0], goal[0]], [up[1], goal[1]]

        ax, ay = [], []
        if "v_cmd" in r and "omega_cmd" in r:
            vc, wc = r["v_cmd"], r["omega_cmd"]
            dur=1.0; n=20
            if abs(wc)>1e-6:
                rr = vc/wc; dth = wc*dur
                ang = np.linspace(0, dth, n)
                ax = list(rp[0] + rr*(np.sin(rth+ang)-np.sin(rth)))
                ay = list(rp[1] - rr*(np.cos(rth+ang)-np.cos(rth)))
            else:
                d = vc*dur
                for k in range(n+1):
                    frac = k/n; ax.append(rp[0]+frac*d*np.cos(rth)); ay.append(rp[1]+frac*d*np.sin(rth))

        dyn = rows[rows["agent_type"] == "dynamic_obstacle"]
        dob = [(d["x"], d["y"], d["radius"], d["agent_id"]) for _, d in dyn.iterrows()]

        frames.append(dict(time=t, user_pos=up, user_motion_angle=um, user_facing_angle=uf,
                          pedestrians=peds, robot_pos=rp, robot_theta=rth,
                          robot_arc_x=ax, robot_arc_y=ay, goal_pos=goal,
                          future_path_x=fpath_x, future_path_y=fpath_y, dynamic_obstacles=dob,
                          overtaking_active=r.get("overtaking_active", False),
                          parking_active=r.get("parking_active", False),
                          repulsion_active=r.get("repulsion_active", False)))
        traj.append(rp)
    return frames, traj

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="History CSV files to compare")
    parser.add_argument("--follow", action="store_true", default=True)
    parser.add_argument("--zoom", type=float, default=15.0)
    parser.add_argument("--output", default="comparison_animation.html")
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()

    ctrls = []
    for f in args.csvs:
        df, name, state, obs, safety_r = load_history(f)
        frm, trj = build_frames(df)
        ctrls.append((name, frm, trj, state, obs, safety_r))

    if not ctrls:
        return
    
    # ----------------------------------------------------------------
    # Sort controllers: SFM → DWA → others (MPC)
    # ----------------------------------------------------------------
    dwa_names = {"BasicDWA", "DW4DO", "DWA_VO", "DWA_RVO", "DWA_ORCA"}

    def sort_key(item):
        name = item[0]
        if name == "SFMController":
            group = 0
        elif name in dwa_names:
            group = 1
        else:
            group = 2
        return (group, name)

    ctrls.sort(key=sort_key)
    # ----------------------------------------------------------------

    n = len(ctrls)
    cols = 3 if n>=3 else n
    rows = (n+cols-1)//cols

    # Use the safety radius from the first controller (all are identical)
    user_safety_r = ctrls[0][5]

    # common environment from first controller
    user_state = ctrls[0][3]
    th_circ = np.linspace(0, 2*np.pi, 50)
    th_obs = np.linspace(0, 2*np.pi, 30)

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[c[0] for c in ctrls],
                        horizontal_spacing=0.04, vertical_spacing=0.06)

    # ---------- static traces (all features per subplot) ----------
    trace_idx = 0
    subplot_trace_map = []

    for ci, (cname, frm, trj, state, obs, _) in enumerate(ctrls):
        row = ci//cols + 1; col = ci%cols + 1
        # user trajectory
        ux, uy = state[:,0], state[:,1]
        fig.add_trace(go.Scatter(x=ux, y=uy, mode='lines', line=dict(color='lightblue', width=2, dash='dot'),
                                 name='Trajectory', showlegend=(ci==0)), row=row, col=col)
        traj_idx = trace_idx; trace_idx += 1
        # obstacles
        obs_indices = []
        for o in obs:
            ox, oy, r = o
            fig.add_trace(go.Scatter(x=ox+r*np.cos(th_obs), y=oy+r*np.sin(th_obs),
                                     fill='toself', fillcolor='rgba(128,128,128,0.5)',
                                     line=dict(color='gray'), name='Obstacle', showlegend=False), row=row, col=col)
            obs_indices.append(trace_idx); trace_idx += 1
        # safety zone (initial empty – updated per frame)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='blue', width=1, dash='dash'),
                                 name='Safety Zone', showlegend=(ci==0)), row=row, col=col)
        saf_idx = trace_idx; trace_idx += 1
        # vicinity (initial empty – updated per frame)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='green', width=1, dash='dot'),
                                 name='Vicinity', showlegend=(ci==0), opacity=0.5), row=row, col=col)
        vic_idx = trace_idx; trace_idx += 1
        # user ellipse (empty init)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='toself',
                                 line=dict(color='blue', width=2), fillcolor='rgba(0,100,255,0.3)',
                                 name='User (body)', showlegend=(ci==0)), row=row, col=col)
        ue_idx = trace_idx; trace_idx += 1
        # user marker
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers',
                                 marker=dict(symbol='circle', size=8, color='blue', line=dict(width=1, color='darkblue')),
                                 name='User', showlegend=(ci==0)), row=row, col=col)
        um_idx = trace_idx; trace_idx += 1
        # ped ellipse
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='toself',
                                 line=dict(color='black', width=1), fillcolor='rgba(200,200,200,0.3)',
                                 name='Pedestrian body', showlegend=(ci==0)), row=row, col=col)
        pe_idx = trace_idx; trace_idx += 1
        # ped marker
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers',
                                 marker=dict(symbol='circle', size=8, line=dict(width=1, color='black')),
                                 name='Pedestrians', showlegend=(ci==0)), row=row, col=col)
        pm_idx = trace_idx; trace_idx += 1
        # heading lines
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='red', width=2),
                                 name='Direction (heading)', showlegend=(ci==0)), row=row, col=col)
        hd_idx = trace_idx; trace_idx += 1
        # gaze lines
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='cyan', width=2, dash='dot'),
                                 name='Gaze direction', showlegend=(ci==0)), row=row, col=col)
        gz_idx = trace_idx; trace_idx += 1
        # cones
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='toself',
                                 fillcolor='rgba(255,255,0,0.2)', line=dict(color='yellow', width=1),
                                 name='Attention cone', showlegend=(ci==0)), row=row, col=col)
        cn_idx = trace_idx; trace_idx += 1
        # robot body
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='toself',
                                 line=dict(color='red', width=1), fillcolor='rgba(255,0,0,0.5)',
                                 name='Robot', showlegend=(ci==0)), row=row, col=col)
        rb_idx = trace_idx; trace_idx += 1
        # robot trail
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                 line=dict(color='red', width=1, dash='dot'),
                                 name='Robot path', showlegend=(ci==0)), row=row, col=col)
        rt_idx = trace_idx; trace_idx += 1
        # user motion line
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                 line=dict(color='red', width=3, dash='solid'),
                                 name='User motion', showlegend=(ci==0)), row=row, col=col)
        umot_idx = trace_idx; trace_idx += 1
        # user facing line
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                 line=dict(color='blue', width=3, dash='dash'),
                                 name='User facing', showlegend=(ci==0)), row=row, col=col)
        ufac_idx = trace_idx; trace_idx += 1
        # future path
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                 line=dict(color='green', width=2, dash='dash'),
                                 name='Future path', showlegend=(ci==0)), row=row, col=col)
        fp_idx = trace_idx; trace_idx += 1
        # goal
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers',
                                 marker=dict(symbol='star', size=12, color='green', line=dict(width=1, color='darkgreen')),
                                 name='Goal', showlegend=(ci==0)), row=row, col=col)
        gl_idx = trace_idx; trace_idx += 1
        # robot arc
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                 line=dict(color='orange', width=2),
                                 name='Robot future', showlegend=(ci==0)), row=row, col=col)
        ra_idx = trace_idx; trace_idx += 1
        # pedestrian safety circles (empty by default)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                 line=dict(color='orange', width=1, dash='dot'),
                                 name='Ped safety margin', showlegend=(ci==0)), row=row, col=col)
        ps_idx = trace_idx; trace_idx += 1

        subplot_trace_map.append(dict(
            traj=traj_idx, obs=obs_indices, saf=saf_idx, vic=vic_idx,
            ue=ue_idx, um=um_idx, pe=pe_idx, pm=pm_idx, hd=hd_idx, gz=gz_idx, cn=cn_idx,
            rb=rb_idx, rt=rt_idx, umot=umot_idx, ufac=ufac_idx,
            fp=fp_idx, gl=gl_idx, ra=ra_idx, ps=ps_idx
        ))

    # Set fixed zoom ranges for follow mode
    if args.follow:
        for ci in range(len(ctrls)):
            row = ci // cols + 1
            col = ci % cols + 1
            fig.update_xaxes(range=[-args.zoom, args.zoom], row=row, col=col)
            fig.update_yaxes(range=[-args.zoom, args.zoom], row=row, col=col, scaleanchor=f"x{ci+1}", scaleratio=1)

    # ---------- frame skipping ----------
    ref_times_all = np.array([f["time"] for f in ctrls[0][1]])
    keep_idx = np.arange(0, len(ref_times_all), args.frame_skip)
    ref_times = ref_times_all[keep_idx]

    ctrl_time_dicts = []
    for _, frm, _, _, _, _ in ctrls:
        d = {f["time"]: f for f in frm}
        ctrl_time_dicts.append(d)

    # ---------- build animation frames ----------
    anim_frames = []
    trails = [[] for _ in ctrls]

    for fi, t_ref in enumerate(ref_times):
        frame_data = []
        frame_trace_indices = []
        for ci, time_dict in enumerate(ctrl_time_dicts):
            if t_ref in time_dict:
                fd = time_dict[t_ref]
            else:
                closest = min(time_dict.keys(), key=lambda x: abs(x-t_ref))
                fd = time_dict[closest]

            ux, uy = fd["user_pos"]
            offset_x, offset_y = (ux, uy) if args.follow else (0,0)
            ua = fd["user_motion_angle"]

            # user ellipse + marker
            uxe, uye = ellipse_points(ux-offset_x, uy-offset_y, ua, USER_A, USER_B)
            frame_data.append(go.Scatter(x=uxe, y=uye, mode='lines', fill='toself',
                                         line=dict(color='blue', width=2), fillcolor='rgba(0,100,255,0.3)'))
            frame_trace_indices.append(subplot_trace_map[ci]["ue"])
            frame_data.append(go.Scatter(x=[ux-offset_x], y=[uy-offset_y], mode='markers',
                                         marker=dict(symbol='circle', size=8, color='blue', line=dict(width=1, color='darkblue'))))
            frame_trace_indices.append(subplot_trace_map[ci]["um"])

            # pedestrians
            xell, yell = [], []; ped_cx, ped_cy, ped_col = [], [], []
            xdir, ydir = [], []; xgaz, ygaz = [], []; xcon, ycon = [], []
            safety_x, safety_y = [], []
            for p in fd["pedestrians"]:
                px, py, pvx, pvy, mood, pth, goff, fov, w = p
                angle = pth; gaze = pth+goff
                cx, cy = px-offset_x, py-offset_y
                xp, yp = ellipse_points(cx, cy, angle, PED_A, PED_B)
                xell.extend(xp); yell.extend(yp); xell.append(None); yell.append(None)
                ped_cx.append(cx); ped_cy.append(cy); ped_col.append(MOOD_COLORS.get(mood, 'orange'))
                hdx = np.cos(angle)*0.5; hdy = np.sin(angle)*0.5
                xdir.extend([cx, cx+hdx, None]); ydir.extend([cy, cy+hdy, None])
                gx = np.cos(gaze)*0.5; gy = np.sin(gaze)*0.5
                xgaz.extend([cx, cx+gx, None]); ygaz.extend([cy, cy+gy, None])
                if fov>0:
                    clen = 1+w; xc, yc = cone_points(cx, cy, gaze, fov, clen)
                    xcon.extend(xc); ycon.extend(yc); xcon.append(None); ycon.append(None)
                # pedestrian safety circle (use user safety radius)
                scx = cx + user_safety_r * np.cos(th_circ)
                scy = cy + user_safety_r * np.sin(th_circ)
                safety_x.extend(scx); safety_y.extend(scy)
                safety_x.append(None); safety_y.append(None)

            # ped ellipse
            frame_data.append(go.Scatter(x=xell, y=yell, mode='lines', fill='toself',
                                         line=dict(color='black', width=1), fillcolor='rgba(200,200,200,0.3)'))
            frame_trace_indices.append(subplot_trace_map[ci]["pe"])
            # ped markers
            frame_data.append(go.Scatter(x=ped_cx, y=ped_cy, mode='markers',
                                         marker=dict(symbol='circle', size=8, color=ped_col, line=dict(width=1, color='black'))))
            frame_trace_indices.append(subplot_trace_map[ci]["pm"])
            # heading
            frame_data.append(go.Scatter(x=xdir, y=ydir, mode='lines', line=dict(color='red', width=2)))
            frame_trace_indices.append(subplot_trace_map[ci]["hd"])
            # gaze
            frame_data.append(go.Scatter(x=xgaz, y=ygaz, mode='lines', line=dict(color='cyan', width=2, dash='dot')))
            frame_trace_indices.append(subplot_trace_map[ci]["gz"])
            # cones
            frame_data.append(go.Scatter(x=xcon, y=ycon, mode='lines', fill='toself',
                                         fillcolor='rgba(255,255,0,0.2)', line=dict(color='yellow', width=1)))
            frame_trace_indices.append(subplot_trace_map[ci]["cn"])
            # ped safety circles
            frame_data.append(go.Scatter(x=safety_x, y=safety_y, mode='lines',
                                         line=dict(color='orange', width=1, dash='dot')))
            frame_trace_indices.append(subplot_trace_map[ci]["ps"])

            # robot (triangle)
            rp = fd["robot_pos"]; rth = fd["robot_theta"]
            rx, ry = rp[0]-offset_x, rp[1]-offset_y
            length, width = 0.5, 0.3
            nx = rx + length*np.cos(rth); ny = ry + length*np.sin(rth)
            bl_ang = rth + np.deg2rad(135); br_ang = rth - np.deg2rad(135)
            blx = rx + width*np.cos(bl_ang); bly = ry + width*np.sin(bl_ang)
            brx = rx + width*np.cos(br_ang); bry = ry + width*np.sin(br_ang)
            poly_x = [nx, blx, brx, nx]; poly_y = [ny, bly, bry, ny]
            if fd["overtaking_active"]:
                fc, lc = 'rgba(255,165,0,0.7)', 'orange'
            elif fd["parking_active"]:
                fc, lc = 'rgba(0,255,0,0.7)', 'green'
            elif fd["repulsion_active"]:
                fc, lc = 'rgba(255,0,255,0.7)', 'magenta'
            else:
                fc, lc = 'rgba(255,0,0,0.5)', 'red'
            frame_data.append(go.Scatter(x=poly_x, y=poly_y, mode='lines', fill='toself',
                                         line=dict(color=lc, width=1), fillcolor=fc))
            frame_trace_indices.append(subplot_trace_map[ci]["rb"])
            # trail
            trails[ci].append(rp)
            tx = [p[0]-offset_x for p in trails[ci]]; ty = [p[1]-offset_y for p in trails[ci]]
            frame_data.append(go.Scatter(x=tx, y=ty, mode='lines', line=dict(color='red', width=1, dash='dot')))
            frame_trace_indices.append(subplot_trace_map[ci]["rt"])

            # user motion line
            lx, ly = ux-offset_x, uy-offset_y   # defined outside the if/else
            if 'user_motion_angle' in fd:
                ex = lx + 1.0*np.cos(fd['user_motion_angle'])
                ey = ly + 1.0*np.sin(fd['user_motion_angle'])
                frame_data.append(go.Scatter(x=[lx, ex], y=[ly, ey], mode='lines',
                                             line=dict(color='red', width=3, dash='solid')))
            else:
                frame_data.append(go.Scatter(x=[], y=[]))
            frame_trace_indices.append(subplot_trace_map[ci]["umot"])
            # user facing line
            if 'user_facing_angle' in fd:
                ex = lx + 1.0*np.cos(fd['user_facing_angle'])
                ey = ly + 1.0*np.sin(fd['user_facing_angle'])
                frame_data.append(go.Scatter(x=[lx, ex], y=[ly, ey], mode='lines',
                                             line=dict(color='blue', width=3, dash='dash')))
            else:
                frame_data.append(go.Scatter(x=[], y=[]))
            frame_trace_indices.append(subplot_trace_map[ci]["ufac"])

            # future path
            fpx = [x-offset_x for x in fd["future_path_x"]]
            fpy = [y-offset_y for y in fd["future_path_y"]]
            frame_data.append(go.Scatter(x=fpx, y=fpy, mode='lines', line=dict(color='green', width=2, dash='dash')))
            frame_trace_indices.append(subplot_trace_map[ci]["fp"])
            # goal
            gx, gy = fd["goal_pos"]
            frame_data.append(go.Scatter(x=[gx-offset_x], y=[gy-offset_y], mode='markers',
                                         marker=dict(symbol='star', size=12, color='green', line=dict(width=1, color='darkgreen'))))
            frame_trace_indices.append(subplot_trace_map[ci]["gl"])
            # robot arc
            ax = [x-offset_x for x in fd["robot_arc_x"]]
            ay = [y-offset_y for y in fd["robot_arc_y"]]
            frame_data.append(go.Scatter(x=ax, y=ay, mode='lines', line=dict(color='orange', width=2)))
            frame_trace_indices.append(subplot_trace_map[ci]["ra"])

            # Safety & vicinity circles: centred on user position
            if args.follow:
                center_x, center_y = 0.0, 0.0
            else:
                center_x, center_y = ux, uy

            frame_data.append(go.Scatter(x=user_safety_r*np.cos(th_circ)+center_x,
                                         y=user_safety_r*np.sin(th_circ)+center_y,
                                         mode='lines', line=dict(color='blue', width=1, dash='dash')))
            frame_trace_indices.append(subplot_trace_map[ci]["saf"])
            frame_data.append(go.Scatter(x=VICINITY_R*np.cos(th_circ)+center_x,
                                         y=VICINITY_R*np.sin(th_circ)+center_y,
                                         mode='lines', line=dict(color='green', width=1, dash='dot')))
            frame_trace_indices.append(subplot_trace_map[ci]["vic"])

            # user trajectory (static, but need offset)
            frame_data.append(go.Scatter(x=user_state[:,0]-offset_x, y=user_state[:,1]-offset_y,
                                         mode='lines', line=dict(color='lightblue', width=2, dash='dot')))
            frame_trace_indices.append(subplot_trace_map[ci]["traj"])
            # obstacles (static, offset)
            for oi, obs in enumerate(ctrls[ci][4]):
                ox, oy, r = obs
                frame_data.append(go.Scatter(x=(ox-offset_x)+r*np.cos(th_obs), y=(oy-offset_y)+r*np.sin(th_obs),
                                             fill='toself', fillcolor='rgba(128,128,128,0.5)',
                                             line=dict(color='gray'), showlegend=False))
                frame_trace_indices.append(subplot_trace_map[ci]["obs"][oi])

        anim_frames.append(go.Frame(data=frame_data, traces=frame_trace_indices, name=str(fi)))

    fig.frames = anim_frames

    # ---- playback controls ----
    frame_dur = max(20, min(1000, int(np.median(np.diff(ref_times))*1000)))
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=True, y=0, x=0, yanchor="bottom", xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": frame_dur, "redraw": False}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
            ])],
        sliders=[dict(active=0, yanchor="top", xanchor="left",
            currentvalue=dict(prefix="Time: ", suffix=" s", visible=True),
            pad=dict(b=10, t=50), len=0.75, x=0.2, y=0,
            steps=[dict(args=[[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}],
                        label=f"{ref_times[i]:.1f}s", method="animate")
                   for i in range(0, len(ref_times), max(1, len(ref_times)//20))])],
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1),
        margin=dict(r=150, l=60, t=60, b=60),
    )

    fig.write_html(args.output)
    print(f"Comparison animation saved to {args.output}")
    try:
        webbrowser.open(args.output, new=1)
    except:
        pass

if __name__ == "__main__":
    main()