# Behavior Calibration Pipeline

## Overview

The **Behavior Calibration Pipeline** is the core mechanism for extracting realistic pedestrian behavior from real-world crowd trajectory data and converting it into calibrated Social Force Model (SFM) parameters. It processes crowd trajectory datasets (primarily ATC data), identifies distinct behavioral regimes (moods), and optimizes SFM parameters for each regime.

**Key Points**:

- Takes raw crowd trajectory data (ATC CSV format) as input
- Produces calibrated mood parameter CSVs for the Extended SFM engine
- Outputs 20+ distinct behavior regimes that can be used in simulations
- These **calibrated moods are the current primary approach** for realistic pedestrian behavior
- The legacy `PedestrianMood` enum in `data/moods.py` is outdated; calibrated moods are the recommended approach

**Pipeline Location**: `src/sfm_navigation/behavior_pipeline/`

---

## Calibrated Moods (Current Primary Approach)

The pipeline produces calibrated mood CSV files stored in `data/calibrated_moods/`. Each file represents a distinct behavioral regime extracted and optimized from real crowd data.

### Available Calibrated Moods (20 regimes)

| Mood Name                       | Category       | Characteristics                             |
| ------------------------------- | -------------- | ------------------------------------------- |
| Aggressive_barger               | High-intensity | High speed, low personal space, assertive   |
| Alert_fast_walker_open_space    | Solo           | Fast movement, alert behavior in open areas |
| Alert_crowd_sprinter            | High-intensity | Sprinting behavior with crowd awareness     |
| Crowd_weaving_rusher            | Navigation     | Weaving through crowds while rushing        |
| Desperate_rusher                | High-intensity | Very high speed, urgent movement            |
| Engaged_speed_walker            | Social         | Fast walking with social engagement         |
| Focused_rusher                  | High-intensity | Focused high-speed movement                 |
| Group_barging_through           | Group          | Group behavior with assertive movement      |
| Group_in_a_panic                | Group          | Panic/emergency group behavior              |
| Quiet_pair                      | Social         | Calm paired walking                         |
| Rushing_group_dense             | Group          | Group rushing in dense conditions           |
| Ruthless_barger                 | High-intensity | Aggressive obstacle-pushing behavior        |
| Social_Walker_v2, Social_Walker | Social         | Socially-aware standard walking             |
| Solo_sprinter                   | Solo           | Individual high-speed movement              |
| Stressed_pusher                 | High-intensity | Stressed/annoyed pushing behavior           |
| Uninterrupted_speed_walker      | Solo           | Steady high-speed unobstructed walking      |
| Watchful_runner                 | Solo           | Running with awareness of surroundings      |
| Zoned_out_weaver                | Navigation     | Distracted weaving behavior                 |

### CSV Format

Each calibrated mood file contains a single row with 12 optimized SFM parameters:

```csv
v0,tau,A_ped,B_ped,lam_base,phi_fov,kappa,k_group,r_group,theta_gaze,w_att,fov_att
1.4157402665986087,0.7160821652823167,7.456493111992314,0.586250849112031,...
```

**Parameter Definitions** (Extended SFM):

- `v0`: Desired velocity (m/s) – maximum speed the pedestrian wants to achieve
- `tau`: Relaxation time (s) – time to reach desired velocity
- `A_ped`: Pedestrian interaction strength – magnitude of repulsive force from pedestrians
- `B_ped`: Pedestrian interaction range – range parameter for pedestrian repulsion
- `lam_base`: Base lambda – anisotropy factor for directional velocity preference
- `phi_fov`: Field of view angle (rad) – angle of the pedestrian's forward vision cone
- `kappa`: Curvature term – controls smooth path curvature following
- `k_group`: Group cohesion strength – how strongly pedestrians pull toward their group
- `r_group`: Group radius (m) – characteristic range of group interaction
- `theta_gaze`: Group gaze angle offset (rad) – directional bias within groups
- `w_att`: Attention weight – how much attention affects trajectory
- `fov_att`: Attention field of view angle (rad) – angular span of attention focus

### Integration into Simulations

Calibrated moods are loaded and registered into the simulation at runtime:

```python
from sfm_navigation.data.moods import register_mood, load_calibrated_moods

# Auto-register all moods from calibrated_moods folder
def _auto_register_calibrated_moods(moods_dir='data/calibrated_moods'):
    """Scan the directory for CSV files and register each as a mood."""
    if not os.path.isdir(moods_dir):
        return
    for fname in os.listdir(moods_dir):
        if fname.endswith('.csv'):
            mood_name = os.path.splitext(fname)[0]
            csv_path = os.path.join(moods_dir, fname)
            try:
                register_mood(csv_path, mood_name)
            except Exception as e:
                print(f"Warning: Failed to register {mood_name}: {e}")

# Usage in simulation
from sfm_navigation.agents.pedestrian import Pedestrian

# Create pedestrian with calibrated mood
ped = Pedestrian(x=10.0, y=10.0, mood='Aggressive_barger', goal_x=20.0, goal_y=20.0)
```

**Advantage**: Moods can be dynamically registered from any CSV file, allowing for custom mood definitions and iterative refinement.

---

## Pipeline Architecture

### Stage Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BEHAVIOR CALIBRATION PIPELINE                      │
└──────────────────────────────────────────────────────────────────────┘

Stage 0: Data Loading & Binning
    Input:  Raw ATC trajectory CSV (timestamp, agent_id, position, velocity)
    Output: Top N crowded temporal bins (bin_id, time_start, time_end, agent_count)
    Purpose: Extract the most crowded time periods for rich behavior capture

         │
         ▼

Stage 1: Sliding Window Extraction
    Input:  Binned trajectory data
    Output: Fixed-length time windows (8 seconds default, 2 second stride)
    Purpose: Segment continuous trajectories into discrete observation windows

    Computes: vx, vy (velocity components), ax, ay (acceleration), accel_mag (magnitude)

         │
         ▼

Stage 2: Feature Extraction (4 feature sets)
    Input:  Windowed trajectory data
    Output: Four feature DataFrames (F11, F12, F13, F14)
    Purpose: Extract behavioral descriptors across multiple domains

    F11 – Kinematic Features (8 features)
        speed_mean, speed_std, speed_p10-90, accel_mean, accel_max, jerk_mean, jerk_max, stop_ratio

    F12 – Path Quality Features (7 features)
        tortuosity, mean_curvature, max_curvature, sinuosity, spectral_arc_length, step_angles

    F13 – Safety/Interaction Features (6 features)
        min_distance, min_TTC, intrusion_time_frac, num_intrusions, avoidance_deviation, n_neighbors

    F14 – Social Attention Features (4 features)
        fov_occupancy, mutual_attention_count, o_space_ratio, group_affiliation

         │
         ▼

Stage 3: Clustering & Regime Labeling
    Input:  Merged feature sets (F11, F12, F13, F14)
    Output: Regime labels (clustering assignments per window)
    Purpose: Identify distinct behavioral clusters across 4 domains

    Method: K-means or Gaussian Mixture Models (GMM)
    Domains: Kinematic, Path Quality, Safety, Social
    Per-domain clustering → Multi-domain regime identification

         │
         ▼

Stage 4: Extended SFM Parameter Calibration
    Input:  Regime labels + trajectory data + calibrated features
    Output: Optimized SFM parameters per regime (v0, tau, A_ped, B_ped, ...)
    Purpose: Fit Extended SFM model to each behavioral regime

    Optimization: Differential Evolution (global) + L-BFGS-B (local)
    Objective: Minimize MSE between simulated and observed trajectories
    Output:   CSV files in data/calibrated_moods/
```

---

## Module Structure & Classes

### Core Orchestrator

**`pipeline_runner.py`** – `PipelineRunner`

Orchestrates the entire 5-stage pipeline with checkpoint/resume capabilities.

**Key Methods**:

- `__init__(config)`: Initialize with `PipelineConfig`
- `run()`: Execute pipeline (skips stages if data exists)
- `_stage_should_run(stage_name)`: Check if stage is in execution plan
- `_load_stage*_data()`: Resume from saved checkpoints

**Features**:

- Flexible stage selection ("all", "stage0,stage2,stage4", etc.)
- Resume from previous run IDs (`load_run_id`)
- Automatic subsampling for large datasets (`max_windows_total`)

---

### Configuration

**`config.py`** – `PipelineConfig`

Dataclass centralizing all pipeline parameters (59+ tunable settings).

**Key Parameters**:

```python
# Execution control
stages: str = "all"              # Which stages to run
load_run_id: Optional[str] = None# Resume from previous run

# Stage 0 (Binning)
atc_csv_path: str = "..."        # Path to ATC trajectory CSV
bin_width_sec: float = 300.0      # 5-minute time bins
n_top_bins: int = 10             # Use top 10 crowded bins

# Stage 1 (Windowing)
window_len_sec: float = 8.0       # 8-second observation window
window_stride_sec: float = 2.0    # 2-second stride (50% overlap)
min_points_per_window: int = 20   # Minimum observations per window

# Stage 3 (Clustering)
min_windows_for_regime: int = 10  # Minimum segments per cluster
cluster_method: str = "kmeans"    # "kmeans" or "gmm"

# Stage 4 (Optimization)
optimizer_maxiter: int = 5        # DE global iterations
optimizer_popsize: int = 10       # DE population size
optimizer_local_maxiter: int = 50 # L-BFGS-B iterations
dt_sim: float = 0.2               # Simulation timestep

# Output
output_dir: str = "pipeline_results"
log_level: str = "INFO"
```

---

### Data Provision

**`data_provider.py`** – `ATCDataSource` & `AbstractDataSource`

Provides abstraction layer for data loading.

**Classes**:

- `AbstractDataSource`: Interface for pluggable data sources
- `ATCDataSource`: Load from single ATC CSV file

**Key Methods**:

- `load_raw_data()`: Returns DataFrame with columns:
  - `timestamp, agent_id, pos_x_mm, pos_y_mm, velocity_mm_s, motion_angle_rad, facing_angle_rad`
  - Automatically converts units (mm → meters)

---

### Reporting & Logging

**`reporting.py`** – `PipelineLogger`

Unified logging with per-stage reports and summary generation.

**Key Methods**:

- `info(msg)`, `warning(msg)`, `error(msg)`: Log messages
- `write_stage_report(stage_name, content)`: Save stage-specific report
- `write_dataframe_summary(df, title)`: Generate DataFrame statistics
- `final_summary()`: Write aggregated error/success summary

**Output Structure**:

```
pipeline_results/
├── 20240614_120000/          # run_id (timestamp)
│   ├── logs/
│   │   └── pipeline.log      # Full execution log
│   ├── reports/
│   │   ├── stage0_report.txt
│   │   ├── stage1_report.txt
│   │   ├── stage2_report.txt
│   │   ├── stage3_report.txt
│   │   ├── stage4_report.txt
│   │   ├── stage3/plots/     # Clustering visualizations
│   │   └── pipeline_summary.txt
│   └── data/
│       ├── stage0/           # Binned data + bin_stats.csv
│       ├── stage1/           # df_windows.csv
│       ├── stage2/           # features_*.csv (4 files)
│       ├── stage3/           # regime_labels.csv + cluster plots
│       └── stage4/           # Calibrated SFM parameters CSVs
```

---

### Stage Implementations

#### Stage 0: Data Loading & Binning

**File**: `stages/stage0_load_bin.py`  
**Function**: `run_stage0(config, logger, data_source)`

**Purpose**: Load raw ATC data and identify the most crowded time periods.

**Process**:

1. Load raw ATC CSV (500k+ rows typical)
2. Bin time axis into 5-minute windows
3. Count agents per bin (crowd density)
4. Select top N crowded bins
5. Save individual bin CSVs + aggregate statistics

**Output**:

- `bin_stats.csv`: Columns = [bin_id, time_start, time_end, agent_count, ...]
- `bin_*.csv`: Individual binned data per selected bin

**Key Function**:

- `bin_data(df, bin_width_sec)`: Compute bin statistics
- `select_bin_data(df, bins)`: Extract selected bins

---

#### Stage 1: Sliding Window Extraction

**File**: `stages/stage1_windowing.py`  
**Function**: `run_stage1(config, logger, df_all)`

**Purpose**: Segment continuous trajectories into fixed-length observation windows with computed derivatives.

**Process**:

1. Compute velocity components: `vx = v * cos(angle)`, `vy = v * sin(angle)`
2. Compute acceleration: `ax = dvx/dt`, `ay = dvy/dt`, `accel_mag = sqrt(ax² + ay²)`
3. Sliding window extraction (8 sec windows, 2 sec stride)
4. Per-agent windowing (preserves individual trajectories)
5. Filter windows with minimum point count (≥20 observations)

**Output**:

- `df_windows.csv`: Columns = [window_id, agent_id, pos_x, pos_y, vx, vy, ax, ay, accel_mag, timestamp, ...]
- ~6000-10000 windows per typical ATC dataset (10 bins × 50 agents × 12-15 windows/agent)

**Key Function**:

- `extract_windows_abs(agent_df, win_len, stride, min_pts)`: Extract windows for single agent

---

#### Stage 2: Feature Extraction

**File**: `stages/stage2_features.py`  
**Functions**: `compute_kinematic_features()`, `compute_path_features()`, `compute_safety_features()`, `compute_social_features()`

**Purpose**: Extract 25 behavioral features across 4 domains.

**F11 – Kinematic Features** (8 features from velocity/acceleration):

```python
speed_mean, speed_max, speed_std
speed_p10, speed_p25, speed_p50, speed_p75, speed_p90
accel_mean, accel_max, accel_std
jerk_mean, jerk_max
stop_ratio  # fraction of time v < 0.15 m/s
```

**F12 – Path Quality Features** (7 features from trajectory shape):

```python
tortuosity       # path_length / displacement
mean_curvature   # curvature of trajectory
max_curvature
sinuosity        # standard deviation of heading changes
spectral_arc_length  # frequency-domain path smoothness
mean_step_angle
step_angle_std
```

**F13 – Safety/Interaction Features** (6 features from proximity):

```python
min_distance     # closest approach to any other agent
min_TTC          # minimum Time-to-Collision
intrusion_time_frac  # fraction of time in personal space
num_intrusions   # count of personal space violations
avoidance_deviation  # lateral deviation when avoiding
n_neighbors      # average neighbor count
```

**F14 – Social Attention Features** (4 features from group dynamics):

```python
fov_occupancy    # fraction of field-of-view occupied by agents
mutual_attention_count  # count of mutual gaze events
o_space_ratio    # shared personal space ratio
group_affiliation  # group membership indicator
```

**Output**:

- `features_kinematic.csv`: F11 (8 cols)
- `features_path_quality.csv`: F12 (7 cols)
- `features_safety.csv`: F13 (6 cols)
- `features_social_attention.csv`: F14 (4 cols)

---

#### Stage 3: Clustering & Regime Labeling

**File**: `stages/stage3_clustering.py`  
**Function**: `run_stage3(config, logger, f11, f12, f13, f14)`

**Purpose**: Identify distinct behavioral regimes through multi-domain clustering.

**Process**:

1. Merge 4 feature sets into unified DataFrame (~12,000 rows × 25 cols)
2. Log-transform skewed features (improves clustering)
3. Per-domain clustering:
   - **Kinematic**: 8 features → N_kinematic clusters
   - **Path**: 7 features → N_path clusters
   - **Safety**: 6 features → N_safety clusters
   - **Social**: 4 features → N_social clusters
4. Combine domain labels: `regime = f"{k_label}_{p_label}_{s_label}_{so_label}"`
   - E.g., `regime = "0_1_2_0"` = kinematic-cluster-0, path-cluster-1, safety-cluster-2, social-cluster-0
5. Select active regimes (≥10 windows per regime)

**Clustering Method**:

- Default: K-means (faster, more interpretable)
- Alternative: Gaussian Mixture Models (probabilistic)
- Optimal cluster count determined via silhouette score or elbow method

**Output**:

- `regime_labels.csv`: Columns = [window_id, agent_id, regime, kinematic_cluster, path_cluster, safety_cluster, social_cluster]
- 8-15 active regimes typical
- Visualizations: PCA plots, cluster histograms

**Key Functions**:

- `_cluster_domain(df, cols, n_clusters, method)`: Cluster single domain
- `_merge_features(f11, f12, f13, f14)`: Combine all features

---

#### Stage 4: Extended SFM Parameter Calibration

**File**: `stages/stage4_calibrate.py`  
**Function**: `run_stage4(config, logger, df_windows, df_raw, labels_df, f11, f12, f13, f14)`

**Purpose**: Optimize Extended SFM parameters for each behavioral regime.

**Process**:

1. For each active regime:
   - Filter trajectories labeled with that regime
   - Extract 5-30 representative trajectories (subsampling)
   - Extract observed features (from stage 2)
2. For each representative trajectory:
   - Simulate Extended SFM with candidate parameters
   - Extract simulated features (same 25 features)
   - Compute MSE: `loss = sum((obs_feature - sim_feature)²)`
3. Global optimization (Differential Evolution):
   - Random population, mutation/crossover
   - 5 iterations, 10 population size (fast approximation)
   - Parameter bounds: v0 ∈ [0.5, 2.5], tau ∈ [0.1, 1.0], A_ped ∈ [1, 10], etc.
4. Local refinement (L-BFGS-B):
   - 50 iterations of gradient-based optimization
   - Starts from DE solution
   - Polished parameters
5. Save optimized parameters to CSV:
   - File: `{regime_name}.csv`
   - Format: Single row, 12 columns (v0, tau, A_ped, ..., fov_att)

**Output**:

- `{regime_0_1_2_0}.csv`, `{regime_1_0_1_1}.csv`, ... (one per active regime)
- Automatically copied to `data/calibrated_moods/` with friendly names
- Example: `Aggressive_barger.csv` for regime "0_1_2_0"

**Key Functions**:

- `simulate_extended_sfm(params, ego_init_state, waypoints, neighbours_interp, dt, T)`: SFM simulation
- `extract_features_from_traj(pos, times, neighbours_pos)`: Compute 25 features from simulated trajectory
- `_objective(params, obs_features, trajectory, waypoints, dt, T)`: MSE loss function
- `run_stage4(...)`: Main calibration loop

**Extended SFM Equations** (used in simulation):

```
Acceleration model:
a_x = (v0*cos(θ_goal) - vx) / τ + obstacle_repulsion_x + social_force_x + ...
a_y = (v0*sin(θ_goal) - vy) / τ + obstacle_repulsion_y + social_force_y + ...

Where:
- v0 = desired velocity
- τ = relaxation time
- θ_goal = heading toward goal
- obstacle_repulsion ∝ A_ped * exp(-distance / B_ped)
- social_force: anisotropic based on velocity direction (lam_base, phi_fov)
- group_force: cohesion toward group members (k_group, r_group)
- attention: modification based on attention focus (w_att, fov_att)
```

---

### Post-Analysis

**`transition_analysis.py`** – `analyze_transitions()`

Analyzes mood/regime transitions from completed pipeline run.

**Purpose**: Generate transition matrices and robustness metrics.

**Outputs**:

- Transition frequency matrix (regime → regime probabilities)
- Regime persistence statistics (mean duration per regime)
- Poisson switching rates (λ = 1 / mean_duration)
- Sankey plots, heatmaps, transition counts

**Function**:

- `analyze_transitions(run_dir, output_dir)`: Load labels, compute transitions

---

## Dependencies & Integration

### Internal Dependencies

```
pipeline_runner.py (orchestrator)
    ├── config.py (configuration)
    ├── reporting.py (logging/reporting)
    ├── data_provider.py (data abstraction)
    ├── stages/
    │   ├── stage0_load_bin.py
    │   │   └── crowd_analysis/binning.py
    │   │   └── data/atc_loader.py
    │   ├── stage1_windowing.py
    │   ├── stage2_features.py
    │   │   └── scipy (signal processing, interpolation)
    │   ├── stage3_clustering.py
    │   │   └── sklearn (preprocessing, PCA, clustering, imputation)
    │   └── stage4_calibrate.py
    │       └── scipy.optimize (DE, L-BFGS-B)
    └── transition_analysis.py
        └── plotly (visualizations)
```

### External Dependencies

- **NumPy, SciPy**: Numerical computations, optimization
- **Pandas**: Data frames for trajectory/feature management
- **Scikit-learn**: Clustering (K-means, GMM), feature scaling, PCA
- **Plotly**: Interactive visualizations (PCA plots, heatmaps, transitions)

### Connection to Rest of System

```
┌──────────────────────────────────────────┐
│     Behavior Calibration Pipeline         │
│     (behavior_pipeline/)                  │
└──────────────────┬───────────────────────┘
                   │ Produces
                   ▼
        data/calibrated_moods/*.csv
                   │
                   │ Consumed by
                   ▼
    src/sfm_navigation/data/moods.py
        ├── load_calibrated_moods()
        └── register_mood()
                   │
                   │ Used by
                   ▼
    src/sfm_navigation/agents/pedestrian.py
        └── Pedestrian.__init__(mood='Aggressive_barger')
                   │
                   │ Integrated into
                   ▼
    src/sfm_navigation/simulation/engine.py
        ├── SimulationEngine.add_pedestrian()
        └── SimulationEngine.step()
                   │
                   │ Creates trajectories for
                   ▼
    src/sfm_navigation/cli/
        ├── demo.py (auto-registers calibrated moods)
        ├── robot_demo.py
        └── crowd_robot_demo.py
```

---

## Usage Guide

### Running the Pipeline

**Basic Usage** (all stages):

```bash
sfm-pipeline
```

**Custom Configuration**:

```bash
sfm-pipeline \
  --data-source /path/to/atc-data.csv \
  --output-dir ./my_results \
  --n-top-bins 5 \
  --stages stage0,stage1,stage2,stage3,stage4 \
  --max-windows 5000
```

**Resume from Checkpoint**:

```bash
# Run stages 3 and 4 only (load stage 0-2 data from previous run)
sfm-pipeline \
  --load-run-id 20240614_120000 \
  --stages stage3,stage4
```

**Programmatic Usage**:

```python
from sfm_navigation.behavior_pipeline.config import PipelineConfig
from sfm_navigation.behavior_pipeline.pipeline_runner import PipelineRunner

config = PipelineConfig()
config.atc_csv_path = "/path/to/atc.csv"
config.output_dir = "my_results"
config.n_top_bins = 10
config.stages = "all"

runner = PipelineRunner(config)
runner.run()
```

### Using Calibrated Moods in Simulation

**Automatic Registration** (in demo):

```python
from sfm_navigation.cli.demo import _auto_register_calibrated_moods

_auto_register_calibrated_moods('data/calibrated_moods')
```

**Manual Registration**:

```python
from sfm_navigation.data.moods import register_mood

register_mood('path/to/Aggressive_barger.csv', 'Aggressive_barger')
```

**Using in Pedestrian**:

```python
from sfm_navigation.agents.pedestrian import Pedestrian

# Create pedestrian with calibrated mood
ped = Pedestrian(
    x=50.0, y=50.0,
    mood='Aggressive_barger',  # Will use calibrated parameters
    goal_x=90.0, goal_y=90.0
)

# Pedestrian now uses the optimized SFM parameters from the pipeline
```

### Analyzing Results

**View Pipeline Outputs**:

```
pipeline_results/
├── 20240614_120000/              # Latest run
│   ├── logs/pipeline.log         # Full execution log
│   ├── reports/
│   │   ├── stage0_report.txt     # Binning statistics
│   │   ├── stage1_report.txt     # Windowing statistics
│   │   ├── stage2_report.txt     # Feature statistics
│   │   ├── stage3_report.txt     # Cluster counts
│   │   ├── stage3/plots/         # Clustering visualizations (PCA, etc.)
│   │   └── stage4_report.txt     # Optimization summary
│   └── data/
│       ├── stage0/bin_stats.csv  # Top crowded bins
│       ├── stage1/df_windows.csv # 8-second windows
│       ├── stage2/features_*.csv # 4 feature sets
│       ├── stage3/regime_labels.csv # Regime assignments
│       └── stage4/               # Calibrated mood CSVs
```

**Extract Specific Regime Parameters**:

```python
import pandas as pd

# Load a calibrated mood
df = pd.read_csv('pipeline_results/20240614_120000/data/stage4/Aggressive_barger.csv')
v0 = df['v0'].values[0]
tau = df['tau'].values[0]
A_ped = df['A_ped'].values[0]
# ... use in simulation
```

---

## Comparison: Legacy vs. Calibrated Moods

### Legacy Approach (PedestrianMood Enum)

**File**: `src/sfm_navigation/data/moods.py`

```python
class PedestrianMood(Enum):
    NORMAL = auto()
    DISTRACTED = auto()
    # ... 9 more hardcoded moods

MOOD_PARAMETERS = {
    PedestrianMood.NORMAL: {
        'speed_factor': 0.6,
        'direction_variance': 0.1,
        'personal_space': 1.0,
        'reactivity': 0.5,
        'tau': 0.5,
        'yielding': 0.5,
        ...
    },
    ...
}
```

**Limitations**:

- Hand-tuned parameters (not data-driven)
- Only 11 fixed moods
- Parameters not validated against real behavior
- No systematic way to add new moods
- Simplistic behavior model (not full Extended SFM)

### Calibrated Moods (Current Approach)

**Source**: Behavior Pipeline output (`data/calibrated_moods/`)

**Advantages**:

- ✅ Data-driven from 500k+ real trajectories
- ✅ 20+ distinct behavioral regimes
- ✅ Full Extended SFM parameters (12 parameters per mood)
- ✅ Optimized for real-world fidelity
- ✅ Flexible: add new regimes by running pipeline on new data
- ✅ Reproducible: fully documented optimization process
- ✅ Extendable: supports custom mood registration at runtime

**Recommendation**:

> **Use calibrated moods for all new simulations.** Legacy `PedestrianMood` enum is retained for backward compatibility but should be considered deprecated.

---

## Extending & Customizing

### Adding New Behavioral Regimes

**Option 1: Run Pipeline on New Data**

```bash
sfm-pipeline \
  --data-source /path/to/new_atc_data.csv \
  --output-dir pipeline_results \
  --n-top-bins 10
```

Result: 8-15 new regimes discovered and calibrated automatically.

**Option 2: Manual Mood Registration**

```python
# Create custom mood parameters by hand or from experiment
custom_params = {
    'v0': 1.5,
    'tau': 0.4,
    'A_ped': 5.0,
    # ... 9 more parameters
}

# Save to CSV
import pandas as pd
df = pd.DataFrame([custom_params])
df.to_csv('custom_mood.csv', index=False)

# Register in simulation
from sfm_navigation.data.moods import register_mood
register_mood('custom_mood.csv', 'my_custom_mood')

# Use it
ped = Pedestrian(x=10, y=10, mood='my_custom_mood')
```

### Tuning Pipeline Parameters

Key parameters in `src/sfm_navigation/behavior_pipeline/config.py`:

- **`n_top_bins`**: How many crowded time bins to include (default: 10)
  - Increase for more behavioral diversity
  - Decrease for faster pipeline

- **`window_len_sec`**: Observation window length (default: 8 seconds)
  - Increase for longer behavior patterns
  - Decrease for finer-grain behavior capture

- **`cluster_method`**: "kmeans" or "gmm"
  - K-means: faster, more interpretable
  - GMM: probabilistic, handles overlapping clusters

- **`optimizer_maxiter`, `optimizer_popsize`**: DE optimization intensity
  - Increase for higher-quality parameters (slower)
  - Decrease for speed

---

## Performance & Computational Cost

### Typical Pipeline Runtime

| Stage                 | Time (10 bins) | Time (100 bins) | Notes                       |
| --------------------- | -------------- | --------------- | --------------------------- |
| Stage 0 (Binning)     | 5 sec          | 30 sec          | Data I/O                    |
| Stage 1 (Windowing)   | 10 sec         | 2 min           | Per-agent processing        |
| Stage 2 (Features)    | 30 sec         | 5 min           | ~7000 windows × 25 features |
| Stage 3 (Clustering)  | 20 sec         | 1 min           | Sklearn clustering          |
| Stage 4 (Calibration) | 2-5 min        | 20-30 min       | DE + L-BFGS-B per regime    |
| **Total**             | ~3-5 min       | ~30-40 min      | Full pipeline               |

### Memory Requirements

- Input ATC CSV: ~500-1000 MB (uncompressed)
- In-memory dataframes: ~2-5 GB peak (stages 1-2)
- Output CSVs: ~10-50 MB total
- Recommendation: 16+ GB RAM for large datasets

### Checkpointing

Pipeline automatically saves stage outputs, enabling resumption:

```bash
# First run: stopped at stage 2
sfm-pipeline --stages all

# Resume from stage 3 (loads stage 0-2 from previous run)
sfm-pipeline --load-run-id 20240614_120000 --stages stage3,stage4
```

---

## Troubleshooting

### Common Issues

**Issue**: "No bin CSV files found"

- **Cause**: Stage 0 failed or run directory corrupted
- **Solution**: Re-run stage 0: `sfm-pipeline --stages stage0`

**Issue**: "Too many single-window regimes"

- **Cause**: Clustering too fine-grained
- **Solution**: Increase `min_windows_for_regime` in config (default: 10)

**Issue**: Pipeline slow at stage 4

- **Cause**: Too many windows or optimization iterations
- **Solution**: Use `--max-windows 2000` to subsample, or reduce `optimizer_maxiter` in config

**Issue**: Calibrated moods not registering

- **Cause**: File not in expected format
- **Solution**: Verify CSV has exactly 12 columns in correct order: `v0, tau, A_ped, B_ped, lam_base, phi_fov, kappa, k_group, r_group, theta_gaze, w_att, fov_att`

---

## References

### Publications & Theory

- **Social Force Model**: Helbing & Molnar (1995) – "Social force model for pedestrian dynamics"
- **Extended SFM**: Jelic et al. (2012) – Anisotropic & group dynamics extensions
- **Feature Engineering**: Behavior science research on pedestrian kinematics, path quality, safety, social dynamics

### Related Modules

- [agents/pedestrian.py](src/sfm_navigation/agents/pedestrian.py) – Pedestrian agent using calibrated moods
- [data/moods.py](src/sfm_navigation/data/moods.py) – Mood registry and parameter loading
- [simulation/engine.py](src/sfm_navigation/simulation/engine.py) – Simulation engine using pedestrians
- [cli/demo.py](src/sfm_navigation/cli/demo.py) – Demo using auto-registered calibrated moods

---

## Summary

The **Behavior Calibration Pipeline** is the core system for extracting data-driven pedestrian behavior from real-world crowds. It transforms raw ATC trajectory data into **20+ calibrated mood profiles** that enable highly realistic crowd simulation. The 5-stage pipeline balances sophistication (full Extended SFM) with usability (automatic CSV export) and is the **recommended approach** for behavior modeling in sfm-navigation.
