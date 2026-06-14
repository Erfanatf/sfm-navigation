"""Plotly animation for the robot demo."""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from ..constants import MOOD_COLORS
from ..agents.user import UserTrajectory
from ..config import CONFIG

def _mood_name(mood):
    return mood if isinstance(mood, str) else mood.name

def create_robot_demo_animation(
    user_trajectory: UserTrajectory,
    frames_data: List[dict],
    static_obstacles: List = None,
    dynamic_obstacle_ids: List[int] = None,
    follow_user: bool = True,
    follow_zoom_radius: float = 15.0,
    spawner=None  # to get vicinity radius
):
    """
    frames_data : list of dicts with keys:
        'time', 'user_pos', 'pedestrians' (list of (x,y,vx,vy,mood)),
        'robot_pos' (x,y), 'dynamic_obstacles' (list of (x,y,radius,id)),
        'respawn_count'
    Returns a Plotly Figure.
    """
    full_traj = user_trajectory.get_full_trajectory()
    vicinity_r = spawner.vicinity_radius if spawner else 10.0
    safety_r = user_trajectory.safety_radius

    if follow_user:
        view_range = follow_zoom_radius
        x_range = [-view_range, view_range]
        y_range = [-view_range, view_range]
        init_ux, init_uy = frames_data[0]['user_pos']
    else:
        x_range = [0, CONFIG.env_width]
        y_range = [0, CONFIG.env_height]
        init_ux, init_uy = 0, 0

    theta_circle = np.linspace(0, 2*np.pi, 50)
    fig = go.Figure()
    trace_idx = 0

    # Trajectory
    if follow_user:
        traj_x, traj_y = full_traj[:,0] - init_ux, full_traj[:,1] - init_uy
    else:
        traj_x, traj_y = full_traj[:,0], full_traj[:,1]
    fig.add_trace(go.Scatter(x=traj_x, y=traj_y, mode='lines',
        line=dict(color='lightblue', width=2, dash='dot'), name='Trajectory'))
    traj_idx = trace_idx; trace_idx += 1

    # Static obstacles
    obs_start_idx = trace_idx
    if static_obstacles:
        for obs in static_obstacles:
            ox = obs.x if hasattr(obs,'x') else obs[0]
            oy = obs.y if hasattr(obs,'y') else obs[1]
            orad = obs.radius if hasattr(obs,'radius') else obs[2]
            if follow_user:
                ox -= init_ux; oy -= init_uy
            fig.add_trace(go.Scatter(
                x=ox + orad*np.cos(theta_circle),
                y=oy + orad*np.sin(theta_circle),
                fill='toself', fillcolor='rgba(128,128,128,0.5)',
                line=dict(color='gray'), name='Obstacle', showlegend=False))
            trace_idx += 1

    # User marker
    ux_disp = 0 if follow_user else frames_data[0]['user_pos'][0]
    uy_disp = 0 if follow_user else frames_data[0]['user_pos'][1]
    fig.add_trace(go.Scatter(x=[ux_disp], y=[uy_disp], mode='markers',
        marker=dict(symbol='circle', size=20, color='blue', line=dict(width=2, color='darkblue')),
        name='User'))
    user_idx = trace_idx; trace_idx += 1

    # Safety/Vicinity circles
    fig.add_trace(go.Scatter(x=ux_disp + safety_r*np.cos(theta_circle),
                             y=uy_disp + safety_r*np.sin(theta_circle),
                             mode='lines', line=dict(color='blue', width=1, dash='dash'),
                             name='Safety Zone'))
    safety_idx = trace_idx; trace_idx += 1
    fig.add_trace(go.Scatter(x=ux_disp + vicinity_r*np.cos(theta_circle),
                             y=uy_disp + vicinity_r*np.sin(theta_circle),
                             mode='lines', line=dict(color='green', width=1, dash='dot'),
                             name='Vicinity', opacity=0.5))
    vicinity_idx = trace_idx; trace_idx += 1

    # Pedestrians (single trace with per‑point colors)
    first_peds = frames_data[0]['pedestrians']
    ped_x = []; ped_y = []; ped_colors = []
    for px, py, _, _, mood in first_peds:
        if follow_user:
            ped_x.append(px - init_ux)
            ped_y.append(py - init_uy)
        else:
            ped_x.append(px); ped_y.append(py)
        ped_colors.append(MOOD_COLORS.get(mood, 'orange'))
    fig.add_trace(go.Scatter(x=ped_x, y=ped_y, mode='markers',
        marker=dict(symbol='circle', size=12, color=ped_colors, line=dict(width=1, color='black')),
        name='Pedestrians'))
    peds_idx = trace_idx; trace_idx += 1

    # Dynamic obstacles (first frame)
    dyn_obs_start = trace_idx
    dyn_obs_dict = {}  # id -> trace index
    if frames_data[0]['dynamic_obstacles']:
        for do_x, do_y, do_r, do_id in frames_data[0]['dynamic_obstacles']:
            if follow_user:
                do_x -= init_ux; do_y -= init_uy
            fig.add_trace(go.Scatter(
                x=do_x + do_r*np.cos(theta_circle),
                y=do_y + do_r*np.sin(theta_circle),
                fill='toself', fillcolor='rgba(255,165,0,0.5)',
                line=dict(color='orange'), name=f'DynObs {do_id}', showlegend=False))
            dyn_obs_dict[do_id] = trace_idx
            trace_idx += 1

    # Robot (current position marker + trail)
    robot_trail_idx = trace_idx
    rpos0 = frames_data[0]['robot_pos']
    if follow_user:
        rx0, ry0 = rpos0[0]-init_ux, rpos0[1]-init_uy
    else:
        rx0, ry0 = rpos0
    fig.add_trace(go.Scatter(x=[rx0], y=[ry0], mode='markers',
        marker=dict(symbol='square', size=14, color='red', line=dict(width=2, color='darkred')),
        name='Robot'))
    robot_trail_idx = trace_idx; trace_idx += 1
    # Robot trail line (empty initially)
    fig.add_trace(go.Scatter(x=[rx0], y=[ry0], mode='lines',
        line=dict(color='red', width=1, dash='dot'), name='Robot path'))
    robot_path_idx = trace_idx; trace_idx += 1

    # --- Build frames ---
    frames = []
    robot_trail_x, robot_trail_y = [], []
    for fd in frames_data:
        ux, uy = fd['user_pos']
        frame_traces = []; frame_indices = []
        if follow_user:
            frame_traces.append(go.Scatter(x=full_traj[:,0]-ux, y=full_traj[:,1]-uy))
            frame_indices.append(traj_idx)
            if static_obstacles:
                for oi, obs in enumerate(static_obstacles):
                    ox = obs.x if hasattr(obs,'x') else obs[0]
                    oy = obs.y if hasattr(obs,'y') else obs[1]
                    orad = obs.radius if hasattr(obs,'radius') else obs[2]
                    frame_traces.append(go.Scatter(x=(ox-ux)+orad*np.cos(theta_circle),
                                                   y=(oy-uy)+orad*np.sin(theta_circle)))
                    frame_indices.append(obs_start_idx + oi)
            frame_traces.append(go.Scatter(x=[0], y=[0]))
            frame_indices.append(user_idx)
            frame_traces.append(go.Scatter(x=safety_r*np.cos(theta_circle), y=safety_r*np.sin(theta_circle)))
            frame_indices.append(safety_idx)
            frame_traces.append(go.Scatter(x=vicinity_r*np.cos(theta_circle), y=vicinity_r*np.sin(theta_circle)))
            frame_indices.append(vicinity_idx)

            px_list=[]; py_list=[]; pc_list=[]
            for px,py,_,_,mood in fd['pedestrians']:
                px_list.append(px-ux); py_list.append(py-uy)
                pc_list.append(MOOD_COLORS.get(mood,'orange'))
            frame_traces.append(go.Scatter(x=px_list, y=py_list, mode='markers',
                marker=dict(size=12, color=pc_list, line=dict(width=1,color='black'))))
            frame_indices.append(peds_idx)

            # dynamic obstacles
            for do_x, do_y, do_r, do_id in fd['dynamic_obstacles']:
                if do_id in dyn_obs_dict:
                    frame_traces.append(go.Scatter(x=(do_x-ux)+do_r*np.cos(theta_circle),
                                                   y=(do_y-uy)+do_r*np.sin(theta_circle)))
                    frame_indices.append(dyn_obs_dict[do_id])

            rx, ry = fd['robot_pos']
            robot_trail_x.append(rx-ux); robot_trail_y.append(ry-uy)
            frame_traces.append(go.Scatter(x=[rx-ux], y=[ry-uy]))
            frame_indices.append(robot_trail_idx)
            frame_traces.append(go.Scatter(x=robot_trail_x, y=robot_trail_y))
            frame_indices.append(robot_path_idx)
        else:
            # non-follow version (simplified, similar adjustments)
            frame_traces.append(go.Scatter(x=[ux], y=[uy]))
            frame_indices.append(user_idx)
            frame_traces.append(go.Scatter(x=ux+safety_r*np.cos(theta_circle), y=uy+safety_r*np.sin(theta_circle)))
            frame_indices.append(safety_idx)
            frame_traces.append(go.Scatter(x=ux+vicinity_r*np.cos(theta_circle), y=uy+vicinity_r*np.sin(theta_circle)))
            frame_indices.append(vicinity_idx)
            px_list=[]; py_list=[]; pc_list=[]
            for px,py,_,_,mood in fd['pedestrians']:
                px_list.append(px); py_list.append(py)
                pc_list.append(MOOD_COLORS.get(mood,'orange'))
            frame_traces.append(go.Scatter(x=px_list, y=py_list, mode='markers',
                marker=dict(size=12, color=pc_list, line=dict(width=1,color='black'))))
            frame_indices.append(peds_idx)
            for do_x, do_y, do_r, do_id in fd['dynamic_obstacles']:
                if do_id in dyn_obs_dict:
                    frame_traces.append(go.Scatter(x=do_x+do_r*np.cos(theta_circle), y=do_y+do_r*np.sin(theta_circle)))
                    frame_indices.append(dyn_obs_dict[do_id])
            rx, ry = fd['robot_pos']
            robot_trail_x.append(rx); robot_trail_y.append(ry)
            frame_traces.append(go.Scatter(x=[rx], y=[ry]))
            frame_indices.append(robot_trail_idx)
            frame_traces.append(go.Scatter(x=robot_trail_x, y=robot_trail_y))
            frame_indices.append(robot_path_idx)

        frames.append(go.Frame(data=frame_traces, traces=frame_indices, name=str(fd['time'])))

    fig.frames = frames

    title = "Robot SFM Demo"
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis=dict(range=x_range, title="X (m)", constrain='domain'),
        yaxis=dict(range=y_range, title="Y (m)", scaleanchor="x", scaleratio=1, constrain='domain'),
        updatemenus=[dict(type="buttons", showactive=True, y=0.95, x=0, yanchor="top", xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
            ])],
        sliders=[dict(active=0, yanchor="top", xanchor="left",
            currentvalue=dict(prefix="Time: ", suffix=" s", visible=True),
            pad=dict(b=10, t=50), len=0.75, x=0.2, y=0,
            steps=[dict(args=[[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}],
                        label=f"{frames_data[i]['time']:.1f}s", method="animate")
                   for i in range(0, len(frames_data), max(1, len(frames_data)//20))])],
        height=700, width=700,
        legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02,
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1),
        margin=dict(r=150, l=60, t=60, b=60)
    )
    return fig