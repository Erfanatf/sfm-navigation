import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

def get_marker_colors(frame_data, longest_agent, vmin, vmax):
    """Return list of colors: red for longest_agent, Plasma colors by velocity for others."""
    colors = []
    for _, row in frame_data.iterrows():
        if row['agent_id'] == longest_agent:
            colors.append('red')
        else:
            norm = (row['velocity'] - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            color = sample_colorscale('Plasma', [norm])[0]
            colors.append(color)
    return colors

def arrow_xy(x, y, angle_rad, length=2.0):
    """Return (list_x, list_y) for a single arrow (used as a line with a None break)."""
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    return [x, x + dx, None], [y, y + dy, None]

def create_crowd_animation(df_bin, longest_agent_id, frame_subsample=5, max_frames=1500,
                           min_samples=50):
    """
    Build a Plotly figure for the crowd data.

    Parameters
    ----------
    df_bin : pd.DataFrame
        Must contain: timestamp_rel, agent_id, pos_x, pos_y, velocity, motion_angle_rad, facing_angle_rad
    longest_agent_id : int
        ID of the agent with the longest trajectory (will be colored red)
    frame_subsample : int
        Only every n-th unique timestamp is kept
    max_frames : int
        Cap on number of frames
    min_samples : int
        Drop agents with fewer than this many samples in the bin
    """
    # Sort and prepare
    df = df_bin.sort_values(["timestamp_rel", "agent_id"]).reset_index(drop=True)
    timestamps = sorted(df["timestamp_rel"].unique())

    # Subsample frames if needed
    if frame_subsample > 1:
        timestamps = timestamps[::frame_subsample]
    if len(timestamps) > max_frames:
        timestamps = timestamps[:max_frames]
    df = df[df["timestamp_rel"].isin(timestamps)]

    # Filter agents with enough samples
    agent_counts = df.groupby('agent_id').size()
    valid_agents = agent_counts[agent_counts >= min_samples].index
    if len(valid_agents) < len(df['agent_id'].unique()):
        df = df[df['agent_id'].isin(valid_agents)]
    agents = sorted(df["agent_id"].unique())

    vmin = df["velocity"].min()
    vmax = df["velocity"].max()

    # Static full trajectories (light gray lines)
    line_traces = []
    for agent in agents:
        sub = df[df["agent_id"] == agent].sort_values("timestamp_rel")
        line_traces.append(
            go.Scatter(
                x=sub["pos_x"],
                y=sub["pos_y"],
                mode="lines",
                line=dict(color="lightgray", width=1.5, dash="dot"),
                opacity=0.5,
                showlegend=False,
                hoverinfo="none",
            )
        )

    # First frame data
    first_t = timestamps[0]
    first_data = df[df["timestamp_rel"] == first_t]
    first_colors = get_marker_colors(first_data, longest_agent_id, vmin, vmax)

    point_trace = go.Scatter(
        x=first_data["pos_x"],
        y=first_data["pos_y"],
        mode="markers",
        marker=dict(size=10, color=first_colors, line=dict(width=1, color="black")),
        text=first_data["agent_id"],
        customdata=np.stack(
            (first_data["velocity"], first_data["motion_angle_rad"], first_data["facing_angle_rad"]),
            axis=-1,
        ),
        hovertemplate=(
            "Agent: %{text}<br>X: %{x:.2f} m<br>Y: %{y:.2f} m<br>"
            "Vel: %{customdata[0]:.2f} m/s<br>"
            "Motion: %{customdata[1]:.2f} rad<br>"
            "Facing: %{customdata[2]:.2f} rad<extra></extra>"
        ),
        name="Pedestrians",
    )

    # Motion arrows (blue)
    mx, my = [], []
    for _, row in first_data.iterrows():
        xs, ys = arrow_xy(row["pos_x"], row["pos_y"], row["motion_angle_rad"], length=2.0)
        mx.extend(xs)
        my.extend(ys)
    motion_trace = go.Scatter(
        x=mx, y=my, mode="lines", line=dict(color="blue", width=2), name="Motion direction"
    )

    # Facing arrows (red)
    fx, fy = [], []
    for _, row in first_data.iterrows():
        xs, ys = arrow_xy(row["pos_x"], row["pos_y"], row["facing_angle_rad"], length=1.0)
        fx.extend(xs)
        fy.extend(ys)
    facing_trace = go.Scatter(
        x=fx, y=fy, mode="lines", line=dict(color="red", width=2), name="Facing direction"
    )

    fig = go.Figure(data=line_traces + [point_trace, motion_trace, facing_trace])

    # Frame construction
    point_idx = len(line_traces)
    motion_idx = point_idx + 1
    facing_idx = motion_idx + 1

    frames = []
    for t in timestamps:
        fdata = df[df["timestamp_rel"] == t]
        fcolors = get_marker_colors(fdata, longest_agent_id, vmin, vmax)
        new_point = go.Scatter(
            x=fdata["pos_x"], y=fdata["pos_y"], mode="markers",
            marker=dict(size=10, color=fcolors, line=dict(width=1, color="black")),
            text=fdata["agent_id"],
            customdata=np.stack(
                (fdata["velocity"], fdata["motion_angle_rad"], fdata["facing_angle_rad"]), axis=-1
            ),
            name="Pedestrians",
        )
        mx2, my2 = [], []
        for _, row in fdata.iterrows():
            xs, ys = arrow_xy(row["pos_x"], row["pos_y"], row["motion_angle_rad"], length=2.0)
            mx2.extend(xs); my2.extend(ys)
        new_motion = go.Scatter(x=mx2, y=my2, mode="lines",
                                line=dict(color="blue", width=2), name="Motion direction")
        fx2, fy2 = [], []
        for _, row in fdata.iterrows():
            xs, ys = arrow_xy(row["pos_x"], row["pos_y"], row["facing_angle_rad"], length=1.0)
            fx2.extend(xs); fy2.extend(ys)
        new_facing = go.Scatter(x=fx2, y=fy2, mode="lines",
                                line=dict(color="red", width=2), name="Facing direction")
        frames.append(go.Frame(
            data=[new_point, new_motion, new_facing],
            name=str(t),
            traces=[point_idx, motion_idx, facing_idx]
        ))

    fig.frames = frames

    # Layout with equal axes
    x_center = (df['pos_x'].min() + df['pos_x'].max()) / 2
    y_center = (df['pos_y'].min() + df['pos_y'].max()) / 2
    half_size = max(df['pos_x'].max() - df['pos_x'].min(),
                    df['pos_y'].max() - df['pos_y'].min()) / 2 + 2

    # Compute median dt for playback speed
    time_steps = sorted(df['timestamp_rel'].unique())
    if len(time_steps) > 1:
        dt = np.median(np.diff(time_steps))
        frame_duration_ms = dt * 1000
    else:
        frame_duration_ms = 40

    fig.update_layout(
        title='ATC – Speed color (hot=fast) with motion (blue) & facing (red)',
        width=1000, height=1000,
        xaxis=dict(title='X (m)', range=[x_center - half_size, x_center + half_size]),
        yaxis=dict(title='Y (m)', range=[y_center - half_size, y_center + half_size]),
        margin=dict(r=120, t=80, l=80, b=60),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[dict(label='Play', method='animate',
                          args=[None, {'frame': {'duration': frame_duration_ms, 'redraw': True},
                                       'fromcurrent': True, 'transition': {'duration': 0}}]),
                     dict(label='Pause', method='animate',
                          args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                         'mode': 'immediate'}])]
        )],
        sliders=[dict(
            steps=[dict(method='animate',
                        args=[[str(t)], {'frame': {'duration': frame_duration_ms, 'redraw': True},
                                         'mode': 'immediate'}],
                        label=f'{t:.1f}s') for t in timestamps],
            transition=dict(duration=0),
            currentvalue=dict(prefix='Time: ', font=dict(size=12)),
            active=0
        )]
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig