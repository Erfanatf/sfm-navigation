"""Post‑hoc analysis of walking regime transitions with visualizations."""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

def analyze_transitions(run_dir: str, output_dir: str = None):
    """
    Load regime labels from a pipeline run, compute transition statistics,
    generate plots, and save the transition matrix CSV.
    """
    if output_dir is None:
        output_dir = os.path.join(run_dir, "transition_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load necessary data ---
    regime_path = os.path.join(run_dir, "data", "stage3", "regime_labels.csv")
    if not os.path.exists(regime_path):
        raise FileNotFoundError(f"Regime labels not found at {regime_path}")

    regime_df = pd.read_csv(regime_path)

    kin_path = os.path.join(run_dir, "data", "stage2", "features_kinematic.csv")
    if not os.path.exists(kin_path):
        raise FileNotFoundError(f"Kinematic features not found at {kin_path}")

    kin_df = pd.read_csv(kin_path)

    df = regime_df.merge(kin_df[['window_id', 'window_start']], on='window_id', how='inner')
    df = df.sort_values(['agent_id', 'window_start']).reset_index(drop=True)

    # --- 2. Extract transitions ---
    transitions = []
    regime_durations = []
    persistence = []

    for agent_id, grp in df.groupby('agent_id'):
        if len(grp) < 2:
            continue
        regimes = grp['regime'].values
        start_times = grp['window_start'].values
        current_regime = regimes[0]
        current_start = start_times[0]
        count = 1
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                count += 1
            else:
                transitions.append((current_regime, regimes[i]))
                regime_durations.append((agent_id, current_regime, current_start, start_times[i-1], count))
                persistence.append(count)
                current_regime = regimes[i]
                current_start = start_times[i]
                count = 1
        regime_durations.append((agent_id, current_regime, current_start, start_times[-1], count))
        persistence.append(count)

    transitions_df = pd.DataFrame(transitions, columns=['from_regime', 'to_regime'])
    durations_df = pd.DataFrame(regime_durations,
                                columns=['agent_id', 'regime', 'start_time', 'end_time', 'num_windows'])

    # --- 3. Robustness check ---
    persistence_series = pd.Series(persistence)
    print("\n=== REGIME PERSISTENCE (consecutive windows) ===")
    print(persistence_series.describe())
    single_window_frac = (persistence_series == 1).mean()
    print(f"Fraction of regimes lasting only 1 window: {single_window_frac:.2%}")
    if single_window_frac > 0.5:
        print("⚠️  WARNING: More than half of regime segments are single‑window – clustering may be noisy.")

    # --- 4. Build transition matrix ---
    from_counts = transitions_df['from_regime'].value_counts()
    transition_counts = transitions_df.groupby(['from_regime', 'to_regime']).size().unstack(fill_value=0)
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    all_regimes = sorted(df['regime'].unique())
    for reg in all_regimes:
        if reg not in transition_matrix.index:
            transition_matrix.loc[reg, reg] = 1.0
    for reg in all_regimes:
        if reg not in transition_matrix.columns:
            transition_matrix[reg] = 0.0
    transition_matrix = transition_matrix.fillna(0.0)

    # --- 5. Poisson rates ---
    window_len = 8.0
    durations_df['duration_sec'] = durations_df['num_windows'] * window_len
    regime_mean_duration = durations_df.groupby('regime')['duration_sec'].mean()
    poisson_rates = 1.0 / regime_mean_duration

    print("\n=== POISSON RATES (switches per second) ===")
    for reg, rate in poisson_rates.items():
        print(f"  {reg}: λ={rate:.4f} (mean duration {regime_mean_duration[reg]:.1f}s)")

    # --- 6. Save CSVs ---
    csv_rows = []
    for from_reg in transition_matrix.index:
        for to_reg in transition_matrix.columns:
            prob = transition_matrix.loc[from_reg, to_reg]
            csv_rows.append({'from': from_reg, 'to': to_reg, 'prob': prob})
    matrix_csv = pd.DataFrame(csv_rows)
    matrix_csv_path = os.path.join(output_dir, "transition_matrix.csv")
    matrix_csv.to_csv(matrix_csv_path, index=False)
    print(f"\nTransition matrix saved to {matrix_csv_path}")
    # Save per‑regime Poisson rates (for per‑mood λ usage)
    poisson_df = pd.DataFrame({'regime': poisson_rates.index, 'lambda': poisson_rates.values})
    poisson_csv_path = os.path.join(output_dir, "mood_poisson_rates.csv")
    poisson_df.to_csv(poisson_csv_path, index=False)
    print(f"Mood Poisson rates saved to {poisson_csv_path}")

    persistence_series.to_csv(os.path.join(output_dir, "regime_persistence.csv"),
                              index=False, header=['num_windows'])
    durations_df.to_csv(os.path.join(output_dir, "regime_durations.csv"), index=False)

    print("\n=== TRANSITION MATRIX ===")
    print(transition_matrix.round(3).to_string())

    # --- 7. Generate plots ---
    _plot_persistence(persistence_series, output_dir)
    _plot_transition_heatmap(transition_matrix, output_dir)
    _plot_poisson_rates(poisson_rates, regime_mean_duration, output_dir)

    return transition_matrix, poisson_rates


def _plot_persistence(persistence_series, output_dir):
    fig = px.histogram(persistence_series, x=0, nbins=50,
                       title='Regime Persistence (consecutive windows)',
                       labels={'0': 'Number of consecutive windows', 'count': 'Frequency'})
    fig.add_vline(x=1, line_dash='dash', line_color='red',
                  annotation_text='Single-window regimes')
    path = os.path.join(output_dir, "persistence_histogram.html")
    fig.write_html(path)
    print(f"Persistence histogram saved to {path}")


def _plot_transition_heatmap(transition_matrix, output_dir):
    labels = transition_matrix.index.tolist()
    z = transition_matrix.values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale='Viridis',
        zmin=0, zmax=1,
        text=[[f'{val:.2f}' for val in row] for row in z],
        texttemplate='%{text}',
    ))
    fig.update_layout(title='Mood Transition Probabilities',
                      xaxis_title='To Regime',
                      yaxis_title='From Regime')
    path = os.path.join(output_dir, "transition_heatmap.html")
    fig.write_html(path)
    print(f"Transition heatmap saved to {path}")


def _plot_poisson_rates(poisson_rates, regime_mean_duration, output_dir):
    df_plot = pd.DataFrame({
        'regime': poisson_rates.index,
        'rate': poisson_rates.values,
        'mean_duration': regime_mean_duration.values
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot['regime'], y=df_plot['rate'],
                         name='Poisson rate λ (switches/s)',
                         marker_color='lightskyblue'))
    fig.add_trace(go.Scatter(x=df_plot['regime'], y=df_plot['mean_duration'],
                             name='Mean duration (s)', yaxis='y2',
                             mode='lines+markers', marker=dict(color='red')))
    fig.update_layout(
        title='Poisson Switch Rates per Regime',
        xaxis_title='Regime',
        yaxis_title='Rate (per second)',
        yaxis2=dict(title='Mean duration (s)', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99)
    )
    path = os.path.join(output_dir, "poisson_rates.html")
    fig.write_html(path)
    print(f"Poisson rates plot saved to {path}")