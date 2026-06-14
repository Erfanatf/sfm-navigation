#!/usr/bin/env python3
"""Overlaid control signals for all controllers in one figure."""
import argparse, webbrowser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="History CSV files to compare")
    parser.add_argument("--output", default="control_comparison.html")
    args = parser.parse_args()

    # Read all CSVs, extract robot rows and controller names
    all_data = []
    for f in args.csvs:
        df = pd.read_csv(f)
        robot = df[df["agent_type"] == "robot"]
        if robot.empty:
            continue
        ctrl = robot["controller"].iloc[0]
        all_data.append((ctrl, robot))

    if not all_data:
        print("No robot data found.")
        return

    # Create subplots: linear vel, angular vel, dist rejection (v), dist rejection (w)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Linear Velocity [m/s]", "Angular Velocity [rad/s]",
                                        "Linear Disturbance [m/s²]", "Angular Disturbance [rad/s²]"))

    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    for i, (ctrl, rob) in enumerate(all_data):
        col = colors[i % len(colors)]
        # linear velocity
        if "v_cmd" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["v_cmd"], mode="lines",
                                     name=f"{ctrl} v_cmd", line=dict(color=col, dash="solid")), row=1, col=1)
        if "v_final" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["v_final"], mode="lines",
                                     name=f"{ctrl} v_final", line=dict(color=col, dash="dash")), row=1, col=1)
        if "lin_speed" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["lin_speed"], mode="lines",
                                     name=f"{ctrl} v_robot", line=dict(color=col, dash="dot")), row=1, col=1)

        # angular velocity
        if "omega_cmd" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["omega_cmd"], mode="lines",
                                     name=f"{ctrl} ω_cmd", line=dict(color=col, dash="solid")), row=2, col=1)
        if "omega_final" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["omega_final"], mode="lines",
                                     name=f"{ctrl} ω_final", line=dict(color=col, dash="dash")), row=2, col=1)
        if "ang_speed" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["ang_speed"], mode="lines",
                                     name=f"{ctrl} ω_robot", line=dict(color=col, dash="dot")), row=2, col=1)

        # disturbance estimation (only if columns exist)
        if "d_ext_v" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["d_ext_v"], mode="lines",
                                     name=f"{ctrl} d_ext_v", line=dict(color=col)), row=3, col=1)
        if "d_hat_v" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["d_hat_v"], mode="lines",
                                     name=f"{ctrl} d_hat_v", line=dict(color=col, dash="dot")), row=3, col=1)

        if "d_ext_omega" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["d_ext_omega"], mode="lines",
                                     name=f"{ctrl} d_ext_ω", line=dict(color=col)), row=4, col=1)
        if "d_hat_omega" in rob.columns:
            fig.add_trace(go.Scatter(x=rob["time"], y=rob["d_hat_omega"], mode="lines",
                                     name=f"{ctrl} d_hat_ω", line=dict(color=col, dash="dot")), row=4, col=1)

    fig.update_layout(title="Control Signals – All Controllers", hovermode="x unified")
    fig.write_html(args.output)
    print(f"Comparison plot saved to {args.output}")
    try:
        webbrowser.open(args.output, new=1)
    except:
        pass

if __name__ == "__main__":
    main()