import pandas as pd
import numpy as np

def convert_units(df, verbose=True):
    """Convert mm -> m and mm/s -> m/s; drop original mm columns."""
    if verbose:
        print("\n[2/5] Converting units: mm → m, mm/s → m/s")
    df = df.copy()
    df['pos_x'] = df['pos_x_mm'] / 1000.0
    df['pos_y'] = df['pos_y_mm'] / 1000.0
    df['pos_z'] = df['pos_z_mm'] / 1000.0
    df['velocity'] = df['velocity_mm_s'] / 1000.0
    df = df.drop(columns=['pos_x_mm', 'pos_y_mm', 'pos_z_mm', 'velocity_mm_s'])
    if verbose:
        print(f"      New ranges: X = [{df['pos_x'].min():.2f}, {df['pos_x'].max():.2f}] m, "
              f"Y = [{df['pos_y'].min():.2f}, {df['pos_y'].max():.2f}] m")
    return df

def bin_data(df, bin_width_sec=300, verbose=True):
    """Split data into time bins, compute statistics, and sort by crowdedness."""
    if verbose:
        print("\n[3/5] Splitting into 5‑minute bins...")
    start_time = df['timestamp'].min()
    df = df.copy()
    df['time_bin'] = ((df['timestamp'] - start_time) // bin_width_sec).astype(int)

    bin_stats = []
    for bin_id, group in df.groupby('time_bin'):
        agents = group['agent_id'].unique()
        n_agents = len(agents)
        samples_per_agent = group.groupby('agent_id').size()
        median_samples = samples_per_agent.median()
        total_samples = len(group)
        time_min = group['timestamp'].min()
        time_max = group['timestamp'].max()
        duration = time_max - time_min
        bin_stats.append({
            'bin_id': bin_id,
            'time_start': time_min,
            'time_end': time_max,
            'duration_sec': duration,
            'n_agents': n_agents,
            'total_samples': total_samples,
            'median_samples_per_agent': median_samples,
            'mean_samples_per_agent': samples_per_agent.mean()
        })

    bin_stats_df = pd.DataFrame(bin_stats)
    bin_stats_df = bin_stats_df.sort_values('median_samples_per_agent', ascending=False).reset_index(drop=True)
    if verbose:
        print("\n🏆 Top 10 time bins (most active / crowded):")
        print(bin_stats_df.head(10).to_string())
    return bin_stats_df

def select_bin_data(df, bin_stats_df, selected_bin_index=4, verbose=True):
    """Extract data for a specific bin and prepare it for visualization."""
    selected_bin = bin_stats_df.iloc[selected_bin_index]
    time_start = selected_bin['time_start']
    time_end = selected_bin['time_end']
    if verbose:
        print(f"\n[4/5] Selected bin #{selected_bin_index}:")
        print(f"      Time window: {time_start:.3f} → {time_end:.3f} "
              f"(duration {selected_bin['duration_sec']:.1f}s)")
        print(f"      Agents: {selected_bin['n_agents']}, "
              f"median samples: {selected_bin['median_samples_per_agent']:.1f}")

    df_bin = df[(df['timestamp'] >= time_start) & (df['timestamp'] <= time_end)].copy()

    # Find longest agent
    agent_sample_counts = df_bin.groupby('agent_id').size()
    longest_agent_id = agent_sample_counts.idxmax()
    if verbose:
        print(f"Agent with longest trajectory in this bin: {longest_agent_id} "
              f"({agent_sample_counts.max()} samples)")

    # Shift agent IDs
    min_agent_id = df_bin['agent_id'].min()
    df_bin['agent_id'] = df_bin['agent_id'] - min_agent_id
    if verbose:
        print(f"      Agent IDs shifted by -{min_agent_id} → new range 0..{df_bin['agent_id'].max()}")

    # Relative time
    df_bin['timestamp_rel'] = df_bin['timestamp'] - time_start

    return df_bin, longest_agent_id