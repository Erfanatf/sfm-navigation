"""CLI to plot control signals, disturbances, acceleration, and jerk from a history CSV."""

import argparse
import webbrowser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to history CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    robot = df[df["agent_type"] == "robot"]
    controller_name = robot["controller"].values[0]

    # 5 rows, 2 columns
    fig = make_subplots(
        rows=5,
        cols=2,
        shared_xaxes=True,
        subplot_titles=(
            "Linear Velocity [m/s]",
            "Angular Velocity [rad/s]",
            "Linear Acceleration [m/s²]",
            "Angular Acceleration [rad/s²]",
            "Linear Jerk [m/s³]",
            "Angular Jerk [rad/s³]",
            "Linear Disturbance [m/s²]",
            "Angular Disturbance [rad/s²]",
            "Velocity Tracking Error [m/s, rad/s]",
            "Position Error [m]",
        ),
    )

    # ── Row 1 ──────────────────────────────────────────────────
    # (1,1) Linear velocity
    for col, name in [
        ("v_base", "v_cmd (controller)"),
        ("v_man", "v_man (maneuver)"),
        ("v_final", "v_final (executed)"),
        ("lin_speed", "v_robot"),
    ]:
        if col in robot.columns:
            fig.add_trace(
                go.Scatter(x=robot["time"], y=robot[col], mode="lines", name=name),
                row=1,
                col=1,
            )

    # (1,2) Angular velocity
    for col, name in [
        ("omega_base", "ω_cmd (controller)"),
        ("omega_man", "ω_man"),
        ("omega_final", "ω_final"),
        ("ang_speed", "ω_robot"),
    ]:
        if col in robot.columns:
            fig.add_trace(
                go.Scatter(x=robot["time"], y=robot[col], mode="lines", name=name),
                row=1,
                col=2,
            )

    # ── Row 2 ──────────────────────────────────────────────────
    # (2,1) Linear acceleration
    if "lin_accel" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"], y=robot["lin_accel"], mode="lines", name="lin_accel"
            ),
            row=2,
            col=1,
        )

    # (2,2) Angular acceleration
    if "ang_accel" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"], y=robot["ang_accel"], mode="lines", name="ang_accel"
            ),
            row=2,
            col=2,
        )

    # ── Row 3 ──────────────────────────────────────────────────
    # (3,1) Linear jerk
    if "lin_jerk" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"], y=robot["lin_jerk"], mode="lines", name="lin_jerk"
            ),
            row=3,
            col=1,
        )

    # (3,2) Angular jerk
    if "ang_jerk" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"], y=robot["ang_jerk"], mode="lines", name="ang_jerk"
            ),
            row=3,
            col=2,
        )

    # ── Row 4 ──────────────────────────────────────────────────
    # (4,1) Linear disturbance
    if "d_ext_v" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"],
                y=robot["d_ext_v"],
                mode="lines",
                name="d_ext_v (injected)",
            ),
            row=4,
            col=1,
        )
    if "d_hat_v" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"],
                y=robot["d_hat_v"],
                mode="lines",
                name="d_hat_v (DOB estimate)",
            ),
            row=4,
            col=1,
        )

    # (4,2) Angular disturbance
    if "d_ext_omega" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"],
                y=robot["d_ext_omega"],
                mode="lines",
                name="d_ext_ω (injected)",
            ),
            row=4,
            col=2,
        )
    if "d_hat_omega" in robot.columns:
        fig.add_trace(
            go.Scatter(
                x=robot["time"],
                y=robot["d_hat_omega"],
                mode="lines",
                name="d_hat_ω (DOB estimate)",
            ),
            row=4,
            col=2,
        )

    # ── Row 5 ──────────────────────────────────────────────────
    # (5,1) Velocity tracking error
    if "lin_speed" in robot.columns and "v_cmd" in robot.columns:
        err_v = robot["lin_speed"] - robot["v_cmd"]
        fig.add_trace(
            go.Scatter(
                x=robot["time"],
                y=err_v,
                mode="lines",
                name="v error (actual - desired)",
            ),
            row=5,
            col=1,
        )
    if "ang_speed" in robot.columns and "omega_cmd" in robot.columns:
        err_w = robot["ang_speed"] - robot["omega_cmd"]
        fig.add_trace(
            go.Scatter(
                x=robot["time"],
                y=err_w,
                mode="lines",
                name="ω error (actual - desired)",
            ),
            row=5,
            col=1,
        )

    # (5,2) Position error (distance to goal)
    if all(col in robot.columns for col in ["x", "y", "goal_x", "goal_y"]):
        pos_err = np.sqrt((robot["x"] - robot["goal_x"])**2 + (robot["y"] - robot["goal_y"])**2)
        fig.add_trace(
            go.Scatter(x=robot["time"], y=pos_err, mode="lines", name="distance to goal"),
            row=5, col=2,
        )
    fig.update_layout(
        title=f"Control & Kinematic Signals – {controller_name}",
        hovermode="x unified",
    )

    html_path = controller_name + "_control_signals_plot.html"
    fig.write_html(html_path)
    print(f"Plot saved to {html_path}")
    try:
        webbrowser.open(html_path, new=1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
