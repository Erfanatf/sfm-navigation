import pandas as pd

def load_atc_raw(csv_path, verbose=True):
    """Load the full ATC dataset from a single CSV file.

    Returns a DataFrame with columns:
        timestamp, agent_id, pos_x_mm, pos_y_mm, pos_z_mm,
        velocity_mm_s, motion_angle_rad, facing_angle_rad
    """
    if verbose: 
        print(f"[1/5] Loading ATC from: {csv_path}")
    df = pd.read_csv(csv_path, sep=',', header=None,
                     names=['timestamp', 'person_id', 'pos_x_mm', 'pos_y_mm', 'pos_z_mm',
                            'velocity_mm_s', 'motion_angle_rad', 'facing_angle_rad'])
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.rename(columns={'person_id': 'agent_id'})
    df = df.dropna(subset=['timestamp', 'agent_id', 'pos_x_mm', 'pos_y_mm'])
    if verbose:
        print(f"      Loaded {len(df)} rows, {df['agent_id'].nunique()} agents.")
    return df