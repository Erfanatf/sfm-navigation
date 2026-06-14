"""CLI to compute and compare performance metrics from simulation history CSVs."""

import argparse
import webbrowser
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..metrics.metrics import ControllerPerformance

METRIC_DIRECTION = {
    ("Navigation Efficiency", "Path Length (m)"): "lower",
    ("Navigation Efficiency", "Path Efficiency"): "higher",
    ("Navigation Efficiency", "Time to Goal (s)"): "lower",
    ("Navigation Efficiency", "Average Speed (m/s)"): "higher",
    ("Navigation Efficiency", "Max Speed (m/s)"): "higher",
    ("Navigation Efficiency", "Speed Variance"): "lower",
    ("Navigation Efficiency", "Stop Ratio"): "lower",
    ("Safety & Collision", "Collision Events"): "lower",
    ("Safety & Collision", "Collision Steps"): "lower",
    ("Safety & Collision", "Min Distance to Obstacles (m)"): "higher",
    ("Safety & Collision", "Social Safety Index (SSI)"): "lower",
    ("Safety & Collision", "Personal Intrusion Events"): "lower",
    ("Safety & Collision", "Min Time to Collision (s)"): "higher",
    ("Social Comfort", "Social Individual Index (SII)"): "lower",
    ("Social Comfort", "Relative Motion Index (RMI)"): "lower",
    ("Social Comfort", "Social Grace Index (SGI)"): "higher",
    ("Smoothness & Jerk", "Acceleration RMS (m/s²)"): "lower",
    ("Smoothness & Jerk", "Max Acceleration (m/s²)"): "lower",
    ("Smoothness & Jerk", "Jerk RMS (m/s³)"): "lower",
    ("Smoothness & Jerk", "Jerk Mean (m/s³)"): "lower",
    ("Smoothness & Jerk", "Smoothness (rad)"): "lower",
    ("Smoothness & Jerk", "Sinuosity (rad)"): "lower",
    ("Path Quality", "Legibility"): "higher",
    ("Path Quality", "Direction Changes"): "lower",
    ("Path Quality", "Directness"): "higher",
    ("Path Quality", "Tortuosity"): "lower",
    ("Path Quality", "Mean Curvature (1/m)"): "lower",
    ("Path Quality", "Max Curvature (1/m)"): "lower",
    ("Computational", "Avg Comp Time (ms)"): "lower",
    ("Computational", "Real‑Time Factor"): "higher",
}


def main():
    parser = argparse.ArgumentParser(
        description="Compute performance metrics from simulation history CSVs"
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="One or more simulation history CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="controller_metrics",
        help="Base name for output files (without extension).",
    )
    args = parser.parse_args()

    csv_files = args.csvs
    output_base = args.output

    all_metrics = []
    mood_info = {}

    for csv_path in csv_files:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  [WARN] File not found: {csv_path}")
            continue

        print(f"Processing: {csv_path.name}")
        try:
            perf = ControllerPerformance(str(csv_path), personal_space=1.5)
            metrics = perf.compute_all()
        except Exception as e:
            print(f"  [ERROR] Failed to compute metrics for {csv_path.name}: {e}")
            continue

        row = {}
        for category, items in metrics.items():
            for name, value in items.items():
                row[(category, name)] = value

        row[("Meta", "Controller")] = csv_path.stem.replace("_simulation_history", "")
        row[("Meta", "File")] = csv_path.name
        all_metrics.append(row)

        mood_csv = csv_path.with_name(
            csv_path.stem.replace("_simulation_history", "_mood_switch_log") + ".csv"
        )
        if mood_csv.exists():
            try:
                mood_df = pd.read_csv(mood_csv)
                mood_info[row[("Meta", "Controller")]] = {
                    "Total Switches": len(mood_df)
                }
            except Exception:
                pass

    if not all_metrics:
        print("No valid CSV files processed. Exiting.")
        return

    df = pd.DataFrame(all_metrics)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.set_index(("Meta", "Controller"))
    csv_out = output_base + ".csv"
    df.to_csv(csv_out)
    print(f"\nMetrics report saved to {csv_out}")

    # Build metric list
    metric_list = []
    for cat in df.columns.get_level_values(0).unique():
        if cat == "Meta":
            continue
        for met in df[cat].columns:
            metric_list.append((cat, met))

    n_total = len(metric_list)
    n_cols = 2
    n_rows = (n_total + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{cat}: {met}" for cat, met in metric_list],
        shared_xaxes=False,
        vertical_spacing=0.04,
        horizontal_spacing=0.1,
    )

    controllers = df.index.tolist()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
    ]
    legend_added = set()

    for i, (cat, met) in enumerate(metric_list):
        row = i // n_cols + 1
        col = i % n_cols + 1
        series = df[cat][met]

        direction = METRIC_DIRECTION.get((cat, met), "higher")
        higher_better = direction == "higher"

        # Sort controllers: best on left
        sort_items = []
        for ctrl in controllers:
            val = series[ctrl] if ctrl in series.index else np.nan
            if np.isnan(val):
                key = -np.inf if higher_better else np.inf
            else:
                key = val
            sort_items.append((key, ctrl))
        sort_items.sort(key=lambda x: x[0], reverse=higher_better)
        sorted_ctrls = [item[1] for item in sort_items]

        for ctrl in sorted_ctrls:
            val = series[ctrl] if ctrl in series.index else np.nan
            show = ctrl not in legend_added
            if show:
                legend_added.add(ctrl)
            fig.add_trace(
                go.Bar(
                    x=[ctrl],
                    y=[val if not np.isnan(val) else 0],
                    name=ctrl,
                    legendgroup=ctrl,
                    marker_color=colors[controllers.index(ctrl) % len(colors)],
                    showlegend=show,
                    text=None,
                    textposition="none",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(
            tickangle=-45, row=row, col=col, title_text="", tickfont=dict(size=9)
        )
        fig.update_yaxes(title_text="", row=row, col=col)

    fig.update_layout(
        height=250 * n_rows,
        title_text="Controller Performance Metrics Comparison",
        barmode="group",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.005,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=140, t=80, b=100),
    )

    html_out = output_base + ".html"
    fig.write_html(html_out)
    print(f"Comparative plot saved to {html_out}")

    try:
        webbrowser.open(html_out, new=1)
    except Exception:
        pass

    if mood_info:
        print("\nMood Switch Summary:")
        for ctrl, info in mood_info.items():
            print(f"  {ctrl}: {info}")


if __name__ == "__main__":
    main()
