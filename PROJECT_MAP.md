# SFM Navigation – Project Map

## Overview

**sfm-navigation** is a realistic simulation engine for pedestrian-aware robot navigation in crowds. The system integrates multiple control algorithms (DWA, DW4DO, VO-based methods, MPC, MPPI, NMPC) with Social Force Model (SFM) pedestrian dynamics, mood-based behavior modeling, and motion prediction for end-to-end testing and comparison.

## Folder Structure & Responsibilities

---

## Calibrated Moods – Core Behavioral Foundation

The system primarily uses **calibrated pedestrian moods** extracted from real-world crowd trajectory data via the Behavior Calibration Pipeline. These represent distinct, data-driven behavioral regimes with optimized Extended SFM parameters.

**Key Points**:

- **20+ calibrated mood types**: Aggressive_barger, Social_Walker, Solo_sprinter, Group_panic, etc.
- **Data-driven origin**: Extracted from ATC crowd datasets (500k+ real trajectories)
- **Extended SFM parameters**: 12 optimized parameters per mood (v0, tau, A_ped, B_ped, lam_base, phi_fov, kappa, k_group, r_group, theta_gaze, w_att, fov_att)
- **Current primary approach**: Replaces the legacy `PedestrianMood` enum for realistic simulations
- **Location**: `data/calibrated_moods/` directory (20+ CSV files)
- **Integration**: Auto-registered in simulations via `load_calibrated_moods()` and `register_mood()`

**→ See [behavior_pipeline.md](behavior_pipeline.md) for comprehensive documentation on mood generation, pipeline stages, and usage.**

---

```
sfm-navigation/
├── src/sfm_navigation/          # Main package
│   ├── agents/                  # Agent entities (robot, pedestrians, obstacles)
│   ├── behavior_pipeline/       # Data processing and calibration pipeline
│   ├── cli/                     # Command-line interfaces and demo scripts
│   ├── controllers/             # Control algorithms (DWA, MPC variants)
│   ├── crowd_analysis/          # Crowd data analysis and transition matrices
│   ├── data/                    # Data loading, filtering, mood definitions
│   ├── logging/                 # Centralized logging
│   ├── metrics/                 # Simulation metrics computation
│   ├── prediction/              # LSTM-based motion prediction models
│   ├── sfm/                     # Social Force Model utilities (Numba-optimized)
│   ├── simulation/              # Core simulation engine
│   ├── spawner/                 # Pedestrian spawner and scenario generation
│   ├── utils/                   # General utilities (filters, math helpers)
│   ├── visualization/           # Animation and HTML visualization
│   ├── config.py                # Global simulation configuration
│   └── constants.py             # Project constants (mood colors, etc.)
├── tests/                       # Unit and integration tests (currently empty)
├── config/                      # Configuration files for pipelines
├── data/                        # Data inputs (trajectories, calibration CSVs)
├── pipeline_results/            # Outputs from behavior pipeline
├── init_test_outputs/           # Test simulation outputs
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Pinned dependencies
├── README.md                    # Project overview and setup
├── PROJECT_MAP.md               # This file - architecture and structure
└── [*.csv, *.html]              # Simulation artifacts and visualizations
```

---

## Core Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Simulation Execution Flow                     │
└─────────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ├─ CONFIG (SimulationConfig) loaded from config.py or JSON
   ├─ CONTROLLER instantiated (DWA, MPC, MPPI, SFM, etc.)
   ├─ SIMULATION ENGINE created (SimulationEngine)
   ├─ PEDESTRIANS spawned (SFMPedestrianSpawner)
   └─ OBSTACLES (static & dynamic) added

2. SIMULATION LOOP (per timestep, dt=0.1s)
   ├─ Dynamic obstacles update position
   ├─ Pedestrians update:
   │  ├─ Sense robot position
   │  ├─ Apply mood-specific behavior
   │  ├─ Compute SFM forces (calibrated parameters)
   │  └─ Update position/velocity
   ├─ Controller:
   │  ├─ Read robot state, goal, obstacle positions
   │  ├─ Compute velocity command (v, ω)
   │  └─ Apply maneuvers (overtaking, rotation, parking, etc.)
   ├─ Robot updates kinematic state (with acceleration/jerk limits)
   ├─ Collision detection & metrics collected
   └─ Repeat until goal reached or timeout

3. OUTPUT & VISUALIZATION
   ├─ SimulationResult collected (trajectories, velocities, metrics)
   ├─ HTML animations rendered (Plotly)
   ├─ Metrics exported (CSV, interactive plots)
   └─ Mood switch logs recorded (mood transitions)

┌─────────────────────────────────────────────────────────────────┐
│              Behavior Calibration Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

Stage 0: Load Binned Data
  └─ Input: ATC trajectory CSV files
     Output: Binned trajectory segments

Stage 1: Windowing
  └─ Input: Binned data
     Output: Time windows with motion features

Stage 2: Feature Extraction
  └─ Input: Windowed data, raw trajectories
     Output: Four feature sets
       - F11: Kinematic (speed, acceleration, heading)
       - F12: Path quality (curvature, consistency)
       - F13: Safety (obstacle proximity, personal space)
       - F14: Social attention (group dynamics, gaze)

Stage 3: Clustering
  └─ Input: Feature sets
     Output: Regime labels (mood classification)

Stage 4: Calibration
  └─ Input: Regime labels, trajectory data
     Output: Optimized SFM parameters per regime
     Files: phase3_optimized_sfm_params_*.csv
```

### Key Classes & Architecture

#### Simulation Core

- **`SimulationEngine`** (`simulation/engine.py`): Orchestrates the entire simulation
  - Manages robot, pedestrians, obstacles
  - Executes step-by-step simulation loop
  - Collects metrics and trajectories
  - Returns `SimulationResult` with all data

- **`RobotState`** (`agents/robot.py`): Immutable state representation
  - Fields: `x, y, theta, v, omega`
  - Methods: `to_array()`, `from_array()` for serialization

- **`DifferentialDriveRobot`** (`agents/robot.py`): Robot kinematics & dynamics
  - Implements motor model with time constant `tau`
  - Enforces acceleration/jerk limits
  - Uses Kalman filter for smooth acceleration estimation
  - Tracks trajectory, velocity, acceleration, jerk histories

#### Agent Models

- **`Pedestrian`** (`agents/pedestrian.py`): Mood-aware pedestrian agent
  - Loads mood-specific parameters from `MOOD_PARAMETERS` or CSV
  - Implements multiple update modes: standard SFM, juggling, adversarial
  - Computes SFM forces based on calibrated parameters
  - Modes: NORMAL, DISTRACTED, STRESSED, IN_RUSH, RUNNING, CURIOUS, ADVERSARIAL, JUGGLING, SLOW_WALKING, Brisk_Individualist, Relaxed_Ped

- **`StaticObstacle`** (`agents/obstacles.py`): Fixed circular obstacles
- **`DynamicObstacle`** (`agents/obstacles.py`): Moving obstacles with linear/circular motion

#### Controllers (Hierarchy)

- **`BaseController`** (interface in `controllers/base_controller.py`)
  - Abstract method: `compute_velocity(robot_state, goal_pos, obstacles) → (v, ω)`

- **DWA Family** (Dynamic Window Approach)
  - `BasicDWA`: Core DWA with maneuver support
  - `DW4DO`: DWA with dynamic obstacles
  - `DWA_VO`, `DWA_RVO`, `DWA_ORCA`: DWA + velocity obstacle variants

- **MPC Family**
  - `BaseMPCController`: Abstract base for MPC-based controllers
  - `NMPCController`: Nonlinear MPC
  - `MPPIController`: Model Predictive Path Integral (sampling-based)
  - `RiskAwareMPPIController`: MPPI with risk metrics
  - `DCBFNMPCController`, `DCBFMPPIController`, `DCBFMPCCMPPIController`: Control Barrier Function variants

- **`SFMController`**: Direct SFM-based navigation (no trajectory planning)

- **Shared Components**:
  - `ManeuverManager`: Coordinates parking, overtaking, rotation, soft recovery maneuvers
  - `ManeuverDOB`: Disturbance observer for maneuver blending
  - `ControlLPF`: First-order low-pass filter for smooth commands

#### Data & Calibration

- **`SimulationConfig`** (`config.py`): Global configuration dataclass
  - Robot specs (radius, max velocity, acceleration limits, jerk limits)
  - DWA parameters (v_res, omega_res, predict_time, safety margins)
  - Pedestrian defaults (radius, speed, social zone)
  - Costmap and visualization settings

- **`PedestrianMood`** (`data/moods.py`): Enum of mood types
  - `MOOD_PARAMETERS`: Dictionary mapping moods → calibrated SFM parameters
  - `CUSTOM_MOODS`: Runtime-registered moods from CSV
  - Functions: `load_calibrated_moods()`, `register_mood()`

- **Data Loaders** (`data/loader.py`, `data/atc_loader.py`)
  - `load_multiple_trajectories()`: Load ATC crowd trajectory datasets
  - `ATCDataSource`: Interface for ATC data

- **Behavior Pipeline** (`behavior_pipeline/`)
  - `PipelineRunner`: Orchestrates multi-stage calibration
  - Stages 0–4: Data ingestion → clustering → SFM parameter optimization
  - `PipelineConfig`: Configuration for pipeline runs

#### Visualization & Output

- **`create_sfm_spawner_animation()`** (`visualization/animation.py`): Main animation renderer
  - Accepts user trajectory, spawner, simulation parameters
  - Returns Plotly Figure for HTML export
  - Renders: robot, pedestrians, static obstacles, trajectories, coordinate frames

- **`RobotAnimation`** (`visualization/robot_animation.py`): Robot-centric animation with zoom/follow

- **`SimulationLogger`** (`behavior_pipeline/reporting.py`): Pipeline execution logging

#### Prediction (ML)

- **`LSTMPredictor`** (`prediction/lstm_predictor.py`): Trajectory prediction model
  - Pre-trained Keras models: `LSTM_2ndOrder_PINN_Final.keras`, etc.
  - Scalers: `state_scaler.pkl`, `delta_scaler.pkl`
  - Methods: Sequence-to-sequence prediction

---

## Module Index

### Root-Level Modules

| Module         | Responsibility                                                     |
| -------------- | ------------------------------------------------------------------ |
| `config.py`    | `SimulationConfig` dataclass, global CONFIG instance, JSON loading |
| `constants.py` | `MOOD_COLORS` dict for consistent visualization                    |

### `agents/` – Entity Models

| Module          | Exports                                        |
| --------------- | ---------------------------------------------- |
| `robot.py`      | `RobotState`, `DifferentialDriveRobot`         |
| `pedestrian.py` | `Pedestrian` (with mood and SFM dynamics)      |
| `obstacles.py`  | `StaticObstacle`, `DynamicObstacle`            |
| `user.py`       | `create_user_trajectory_from_processed_data()` |

### `controllers/` – Control Algorithms

| Module                                       | Exports                                             |
| -------------------------------------------- | --------------------------------------------------- |
| `base_controller.py`                         | `BaseController` (abstract interface)               |
| `sfm_controller.py`                          | `SFMController` (direct social force model control) |
| `maneuvers.py`                               | `ManeuverManager`, `ManeuverDOB`, maneuver helpers  |
| `dwa/basic_dwa.py`                           | `BasicDWA`                                          |
| `dwa/dw4do.py`                               | `DW4DO`                                             |
| `dwa/dwa_vo.py`, `dwa_rvo.py`, `dwa_orca.py` | VO-based DWA variants                               |
| `dwa/dwa_utils.py`                           | Shared utilities for DWA family                     |
| `mpc/base_mpc.py`                            | `BaseMPCController`, `ControlLPF`                   |
| `mpc/nmpc.py`                                | `NMPCController`                                    |
| `mpc/mppi.py`, `mppi_noise.py`               | `MPPIController`, noise models                      |
| `mpc/risk_aware_mppi.py`                     | `RiskAwareMPPIController`                           |
| `mpc/dcbf_*.py`                              | Control Barrier Function variants                   |
| `mpc/standard_mpc.py`                        | Basic MPC formulation                               |

### `simulation/` – Core Engine

| Module      | Exports                                |
| ----------- | -------------------------------------- |
| `engine.py` | `SimulationEngine`, `SimulationResult` |

### `behavior_pipeline/` – Calibration Pipeline

| Module                        | Responsibility                           |
| ----------------------------- | ---------------------------------------- |
| `pipeline_runner.py`          | `PipelineRunner` orchestrator            |
| `config.py`                   | `PipelineConfig` for pipeline settings   |
| `data_provider.py`            | `ATCDataSource` wrapper                  |
| `reporting.py`                | `PipelineLogger` for run tracking        |
| `transition_analysis.py`      | Mood transition matrix analysis          |
| `stages/stage0_load_bin.py`   | Bin trajectory segments                  |
| `stages/stage1_windowing.py`  | Windowing and feature extraction         |
| `stages/stage2_features.py`   | Kinematic, path, safety, social features |
| `stages/stage3_clustering.py` | Clustering and regime labeling           |
| `stages/stage4_calibrate.py`  | SFM parameter optimization               |

### `data/` – Data Loading & Calibration

| Module          | Responsibility                                                |
| --------------- | ------------------------------------------------------------- |
| `loader.py`     | `load_multiple_trajectories()`                                |
| `atc_loader.py` | ATC dataset loading                                           |
| `filtering.py`  | Trajectory filtering utilities                                |
| `moods.py`      | `PedestrianMood` enum, `MOOD_PARAMETERS`, calibration loaders |
| `*.csv`         | Calibrated SFM parameters per mood regime                     |

### `sfm/` – Social Force Model (Numba-Optimized)

| Module           | Responsibility                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `numba_utils.py` | Euclidean distance, angle normalization, trajectory simulation, collision checks, scoring functions (JIT-compiled) |

### `spawner/` – Scenario Generation

| Module       | Responsibility                                             |
| ------------ | ---------------------------------------------------------- |
| `spawner.py` | `SFMPedestrianSpawner`, `generate_safe_static_obstacles()` |

### `visualization/` – Rendering & Animation

| Module               | Responsibility                                 |
| -------------------- | ---------------------------------------------- |
| `animation.py`       | `create_sfm_spawner_animation()` main renderer |
| `robot_animation.py` | Robot-centric animation with follow/zoom       |

### `metrics/` – Performance Metrics

| Module       | Responsibility                                            |
| ------------ | --------------------------------------------------------- |
| `metrics.py` | Metrics computation (path length, time, collisions, etc.) |

### `logging/` – Logging Infrastructure

| Module      | Responsibility           |
| ----------- | ------------------------ |
| `logger.py` | Centralized logger setup |

### `prediction/` – Motion Prediction

| Module              | Responsibility                    |
| ------------------- | --------------------------------- |
| `lstm_predictor.py` | `LSTMPredictor` with Keras models |
| `save_scalers.py`   | Scaler persistence                |
| `*.keras`           | Pre-trained LSTM models           |
| `*.pkl`             | Fitted scalers (state, delta)     |

### `utils/` – General Utilities

| Module             | Responsibility                                     |
| ------------------ | -------------------------------------------------- |
| `derivative_kf.py` | `DerivativeEstimatorKF` for acceleration smoothing |

### `cli/` – Command-Line Interfaces & Demos

| Module                       | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| `demo.py`                    | Main SFM spawner demo with mood calibration |
| `run_simulation.py`          | Generic simulation runner with config       |
| `robot_demo.py`              | Single robot with user trajectory demo      |
| `crowd_robot_demo.py`        | Robot in crowd scenario                     |
| `pipeline_cli.py`            | Behavior pipeline runner                    |
| `transition_analysis_cli.py` | Mood transition analysis                    |
| `animate_history.py`         | Replay animation from saved data            |
| `compare_animations.py`      | Side-by-side controller comparison          |
| `compare_control_signals.py` | Control signal comparison plots             |
| `plot_control_signals.py`    | Individual control signal visualization     |
| `crowd_analysis.py`          | Crowd dataset analysis                      |
| `metrics_report.py`          | Generate metrics HTML report                |

---

## Entry Points & CLI Commands

All entry points are defined in `pyproject.toml` under `[project.scripts]`:

```bash
# Core simulations
sfm-demo                        # Main demo (SFM spawner + crowd with mood calibration)
sfm-run                         # Generic simulation runner with config file
sfm-crowd                       # Crowd dataset analysis
sfm-crowd-robot                 # Robot navigating in crowd scenario

# Robot demos
sfm-robot-demo                  # Single robot with user trajectory
sfm-transition-analysis         # Mood transition analysis

# Behavior pipeline
sfm-pipeline                    # Run behavior calibration pipeline (stages 0-4)

# Visualization & analysis
sfm-animate                     # Replay animation from saved trajectory data
sfm-plot-control                # Plot control signals from simulation
sfm-compare-controllers-anim    # Side-by-side controller comparison (animation)
sfm-compare-controllers-signals # Side-by-side controller comparison (signals)
sfm-metrics                     # Generate metrics HTML report
```

### Typical Usage Flow

1. **Scenario Setup**: Use `SFMPedestrianSpawner` (from `spawner.py`) + `generate_safe_static_obstacles()`
2. **Controller Init**: Instantiate controller (e.g., `BasicDWA(config)`, `MPPIController(config)`)
3. **Engine Setup**: Create `SimulationEngine(config)`, add obstacles/pedestrians, set controller
4. **Execution**: Call `engine.run()` to get `SimulationResult`
5. **Visualization**: Use `create_sfm_spawner_animation()` with result trajectories
6. **Metrics**: Extract metrics from `SimulationResult` and `SimulationEngine`

---

## Key Dependencies

### External Libraries

- **NumPy** (≥1.24): Numerical arrays and operations
- **SciPy** (≥1.10): Optimization, interpolation
- **Pandas** (≥2.0): Data frames for trajectory/parameter management
- **Numba** (≥0.57): JIT compilation for performance-critical SFM utils
- **Plotly** (≥5.15): Interactive HTML animations and charts
- **Keras/TensorFlow** (implicit): LSTM prediction models (`.keras` files)

### Internal Module Dependencies (Simplified DAG)

```
config.py, constants.py  [ROOT - GLOBAL CONFIG]
            ↓
    ┌───────┴────────┬──────────────┐
    ↓                ↓              ↓
 agents/        data/moods.py   sfm/numba_utils.py
(robot, ped,      (mood params)     (Numba helpers)
 obstacles)              ↓              ↓
    ↓                    ↓              ↓
    └────────────────→ Pedestrian ←────┘
                      (uses moods,
                       SFM utils)
            ↓
    ┌───────┴─────────────────┐
    ↓                         ↓
controllers/            simulation/engine.py
(DWA, MPC,         (orchestrates
 maneuvers)         all agents)
    ↓                    ↓
    └──→ BaseController  ↓
         (interface)     ↓
         ↓               ↓
    Used by →  SimulationEngine
                    ↓
    ┌───────────────┴──────────────┐
    ↓                              ↓
visualization/               metrics/
(animation.py)            (metrics.py)
(Plotly output)           (CSV output)
    ↓                         ↓
  HTML animations      Metrics tables

behavior_pipeline/
(independent chain:
 stages 0-4)
    ├→ data/loaders
    ├→ crowd_analysis/
    └→ Produces: MOOD_PARAMETERS CSVs

prediction/
(independent ML module
 for trajectory forecasting)
```

### Dependency Order (Initialization)

1. Load `CONFIG` (config.py)
2. Load `MOOD_PARAMETERS` (data/moods.py)
3. Initialize agents (agents/)
4. Initialize controller (controllers/)
5. Create engine (simulation/engine.py)
6. Run simulation
7. Render visualization (visualization/)
8. Export metrics (metrics/)

---

## Design Patterns

### 1. **Composition over Inheritance**

- `SimulationEngine` composes `DifferentialDriveRobot`, list of `Pedestrian`, list of obstacles
- `Pedestrian` composes mood behavior via strategy pattern (standard, juggling, adversarial modes)

### 2. **Strategy Pattern (Controllers)**

- All controllers inherit `BaseController` interface
- Each algorithm is a concrete strategy (BasicDWA, NMPCController, SFMController, etc.)
- Runtime selection: `engine.set_controller(controller_instance)`

### 3. **Configuration Object**

- `SimulationConfig` dataclass centralizes all tunable parameters
- Global `CONFIG` instance accessible from any module
- JSON loading for experiment reproducibility

### 4. **Data Transfer Objects**

- `RobotState`: Immutable state snapshot
- `SimulationResult`: Encapsulates all simulation outputs
- Facilitates clean data passing between modules

### 5. **Maneuver Manager**

- Shared by DWA and MPC families
- Pluggable maneuver set (parking, overtaking, rotation, soft recovery)
- Reduces code duplication across controller variants

### 6. **Numba JIT Compilation**

- `sfm/numba_utils.py` uses `@njit` decorator for hot-path optimization
- Enables SFM force calculations and trajectory scoring at real-time speeds
- Called by `Pedestrian.update()` and DWA scoring functions

### 7. **Pipeline Chain of Responsibility**

- `PipelineRunner` orchestrates stages 0–4
- Each stage can be skipped if data already exists
- Facilitates reuse and debugging of calibration steps

### 8. **Multi-Representation Data**

- Pedestrian behavior: Enum (`PedestrianMood`) + string (custom from CSV)
- Moods lookup: `MOOD_PARAMETERS` dict (enum) + `CUSTOM_MOODS` dict (runtime)
- Supports both standard and user-defined moods

---

## Key Algorithms & Features

### Control Algorithms

- **DWA (Dynamic Window Approach)**: Reactive local planner with velocity windows
- **Velocity Obstacles (VO/RVO/ORCA)**: Collision avoidance in velocity space
- **NMPC**: Nonlinear model predictive control with prediction horizon
- **MPPI**: Sampling-based optimal control (cross-entropy method)
- **Control Barrier Functions (DCBF)**: Safety constraints for MPC variants
- **Social Force Model**: Pedestrian dynamics via attractive/repulsive forces

### Robot Dynamics

- Differential drive kinematics with motor time constant
- Acceleration/jerk saturation
- Kalman filtering for smooth acceleration estimation

### Maneuvers

- **Rotation**: In-place rotation when stuck
- **Overtaking**: Side-step to pass slow pedestrians
- **Parking**: Approach and stop maneuver
- **Soft Recovery**: Gentle obstacle avoidance

### Pedestrian Mood Dynamics

- 11 mood types with calibrated SFM parameters
- Mood-driven speed, direction variance, personal space, reactivity
- Mood transitions (via behavior_pipeline calibration)
- Special behaviors: juggling (stationary swaying), adversarial (approaches robot)

### Metrics & Analysis

- Path length, execution time, collision count
- Minimum distance to obstacles
- Computation time per step (real-time factor)
- Mood switch transitions and durations
- Safety and comfort metrics

---

## Configuration & Customization

### Key Configuration Points

**`config.py` – Global Parameters**

- Environment (width, height, timestep, max_time)
- Robot (radius, velocity limits, acceleration limits, jerk limits)
- DWA (window time, resolution, weighting factors)
- Pedestrian defaults (radius, speed, social zone)
- Costmap and visualization settings

**`data/moods.py` – Pedestrian Behavior**

- Add new mood to `PedestrianMood` enum
- Define parameters in `MOOD_PARAMETERS` dict
- Or load from CSV via `register_mood()`, `load_calibrated_moods()`

**`behavior_pipeline/config.py` – Calibration Settings**

- Data source (ATC CSV path)
- Output directory
- Stage selection (which stages to run)
- Clustering parameters
- Optimization settings

**Controllers – Algorithm Tuning**

- Each controller has tunable parameters (e.g., DWA weights α, β, γ)
- MPC prediction horizon, MPPI sampling count
- Maneuver thresholds in `ManeuverManager`

---

## Testing & Validation

### Manual Testing

- Run individual demos: `sfm-demo`, `sfm-robot-demo`, `sfm-crowd-robot`
- Compare controllers: `sfm-compare-controllers-anim`, `sfm-compare-controllers-signals`
- Inspect metrics: `sfm-metrics`

### Automated Tests

- `tests/` folder is currently empty (can add unit/integration tests)
- Suggested test categories:
  - Agent kinematics (robot, pedestrian)
  - Controller outputs validity
  - Collision detection accuracy
  - Configuration loading/validation
  - Data pipeline stages

---

## Performance Considerations

1. **Numba JIT Compilation**: SFM utils benefit significantly; first call slower, subsequent calls fast
2. **Trajectory Prediction**: LSTM models (~8.5 MB each) loaded on-demand from disk
3. **Visualization**: Plotly animations with 1000+ frames can be memory-intensive
4. **Scaling**: Tested with ~25 pedestrians in real-time; higher counts may require optimization
5. **Maneuver Overhead**: ManeuverDOB and LPF filtering add minimal computational cost

---

## Future Enhancements

- Add test suite (unit, integration, performance benchmarks)
- Expand controller family (RRT\*, lattice planners, learning-based)
- Multi-robot scenarios with communication
- More sophisticated crowd dynamics (grouping, leader-follower)
- Hardware-in-the-loop validation with real robots
- Real-time performance profiling dashboard
- Extended mood taxonomy with continuous parameters
- Support for non-circular robot/obstacle geometries

---

## File Reference Summary

### Outputs & Artifacts

- `*_robot_sfm_animation.html` – Main simulation animation (Plotly)
- `*_simulation_history.csv` – Trajectory and state history
- `*_mood_switch_log.csv` – Pedestrian mood transitions
- `*_control_signals_plot.html` – Controller signal plots
- `controller_metrics.html`, `.csv` – Comparison metrics
- `comparison_animation.html` – Side-by-side controller comparison
- `history_animation.html` – Multi-trajectory replay
- `test_sorted_anim.html` – Test animation output

### Data Inputs

- `data/*.csv` – Calibrated SFM parameters per mood
- `data/transition_matrix_*.csv` – Mood transition probabilities
- `config/default.json` – Default simulation configuration

### Pipeline Outputs

- `pipeline_results/` – Behavior calibration results
- `init_test_outputs/` – Test simulation snapshots

---

## Quick Start Guide

### Installation

```bash
# Clone or download the repository
cd sfm-navigation

# Install in development mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### Running a Demo

```bash
# Main simulation with pedestrian spawner
sfm-demo

# Robot with user trajectory
sfm-robot-demo

# Robot in crowd
sfm-crowd-robot

# Compare controllers
sfm-compare-controllers-anim

# Run behavior calibration pipeline
sfm-pipeline
```

### Creating a Custom Simulation

```python
from sfm_navigation.config import SimulationConfig
from sfm_navigation.simulation.engine import SimulationEngine
from sfm_navigation.controllers.dwa.basic_dwa import BasicDWA
from sfm_navigation.data.moods import PedestrianMood

# Configuration
config = SimulationConfig()

# Create engine and controller
engine = SimulationEngine(config)
controller = BasicDWA(config)
engine.set_controller(controller)

# Add obstacles and pedestrians
engine.add_static_obstacle(x=5.0, y=5.0, radius=0.5)
engine.add_pedestrian(x=10.0, y=10.0, mood=PedestrianMood.NORMAL)

# Run simulation
result = engine.run(start=(1.0, 1.0), goal=(18.0, 18.0))
print(f"Success: {result.success}")
print(f"Path length: {result.path_length:.2f}m")
print(f"Collisions: {result.collision_count}")
```

### Extending the System

**Adding a New Controller**

1. Inherit from `BaseController`
2. Implement `compute_velocity(robot_state, goal_pos, obstacles)`
3. Register in controller imports/tests

**Adding a New Mood**

```python
from sfm_navigation.data.moods import register_mood
register_mood('path/to/mood_params.csv', 'my_mood_name')
```

**Custom Scenario**

1. Use `SFMPedestrianSpawner` for dynamic pedestrian generation
2. Use `generate_safe_static_obstacles()` for collision-free obstacle placement
3. Pass to `SimulationEngine` for execution

---

## Contributing

When contributing, please ensure:

- Code follows existing style and patterns
- New modules include docstrings and type hints
- Controllers inherit from `BaseController`
- Configuration changes update `SimulationConfig`
- Visualization functions return Plotly figures

---

## License

MIT License – See project root for details.
