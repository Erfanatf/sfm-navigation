# SFM Navigation

A comprehensive simulation engine for pedestrian-aware robot navigation in crowds with realistic social force model dynamics, multiple control algorithms, and behavior calibration.

## Overview

**sfm-navigation** provides a complete platform for simulating and benchmarking robot navigation in dynamic crowd environments. It integrates:

- **Multiple Control Algorithms**: DWA, MPC, MPPI, NMPC, SFM-based control, velocity obstacles (VO, RVO, ORCA)
- **Realistic Pedestrian Dynamics**: Social Force Model with calibrated parameters and mood-based behavior
- **Behavior Calibration Pipeline**: Automated SFM parameter optimization from real crowd trajectories
- **Interactive Visualization**: HTML/Plotly animations with trajectory tracking and control signal plots
- **Performance Metrics**: Collision detection, path length, execution time, safety analysis
- **Motion Prediction**: LSTM-based trajectory forecasting models

Perfect for research in crowd navigation, human-robot interaction, and autonomous vehicle control.

## Features

### Control Algorithms

- **DWA (Dynamic Window Approach)**: Reactive local planner with velocity windows
- **MPC Family**: NMPC, MPPI, Risk-aware MPPI, Control Barrier Function variants
- **Velocity Obstacles**: VO, RVO, ORCA integration with DWA
- **Direct SFM Control**: Social Force Model-based steering
- **Maneuver Support**: Overtaking, rotation, parking, soft recovery

### Pedestrian Behavior

- **Mood System**: 11 calibrated pedestrian mood types
  - Standard: NORMAL, DISTRACTED, STRESSED, IN_RUSH, RUNNING
  - Special: CURIOUS, ADVERSARIAL, JUGGLING, SLOW_WALKING
  - Calibrated: Brisk_Individualist, Relaxed_Ped
- **Dynamic Mood Transitions**: Real-world behavior switching
- **Calibrated SFM Parameters**: From ATC crowd trajectory datasets
- **Realistic Forces**: Goal-directed, obstacle avoidance, social distancing

### Robot Dynamics

- Differential drive kinematics
- Acceleration/jerk saturation
- Motor time constant modeling
- Kalman filtering for smooth acceleration

### Analysis Tools

- Behavior calibration pipeline (5 stages)
- Mood transition matrix analysis
- Controller comparison framework
- Performance metrics (collisions, path length, time)
- Real-time performance profiling

## Installation

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Pandas ≥ 2.0
- Numba ≥ 0.57
- Plotly ≥ 5.15

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/sfm-navigation.git
cd sfm-navigation

# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### Run Main Demo

    > **Note**: The system now primarily uses **20+ data-driven calibrated moods** extracted from real crowd trajectories (see [behavior_pipeline.md](behavior_pipeline.md) for details). The legacy `PedestrianMood` enum is retained for backward compatibility but should be considered deprecated.

sfm-demo

# Robot navigating with user trajectory

sfm-robot-demo

# Robot in crowd environment

sfm-crowd-robot

````

### Compare Controllers

```bash
# Side-by-side animation comparison
sfm-compare-controllers-anim

# Control signal comparison
sfm-compare-controllers-signals

# Generate metrics report
sfm-metrics
````

### Run Behavior Calibration

```bash
# Process crowd data and optimize SFM parameters
sfm-pipeline
```

## Usage Example

```python
from sfm_navigation.config import SimulationConfig
from sfm_navigation.simulation.engine import SimulationEngine
from sfm_navigation.controllers.dwa.basic_dwa import BasicDWA
from sfm_navigation.data.moods import PedestrianMood
from sfm_navigation.visualization.animation import create_sfm_spawner_animation

# Configuration
config = SimulationConfig(
    env_width=100.0,
    env_height=100.0,
    robot_radius=0.18,
    max_linear_vel=2.5
)

# Create simulation engine
engine = SimulationEngine(config)
controller = BasicDWA(config)
engine.set_controller(controller)

# Add obstacles and pedestrians
engine.add_static_obstacle(x=25.0, y=50.0, radius=1.0)
engine.add_pedestrian(
    x=50.0, y=30.0,
    mood=PedestrianMood.NORMAL,
    goal_x=50.0, goal_y=70.0
)

# Run simulation
result = engine.run(
    start=(10.0, 10.0),
    goal=(90.0, 90.0),
    verbose=True
)

# Print results
print(f"Success: {result.success}")
print(f"Path length: {result.path_length:.2f}m")
print(f"Time: {result.total_time:.1f}s")
print(f"Collisions: {result.collision_count}")
```

## Architecture

### Core Modules

**Agents**

- `robot.py`: `DifferentialDriveRobot` with realistic dynamics
- `pedestrian.py`: Mood-aware pedestrians with SFM forces
- `obstacles.py`: Static and dynamic obstacles

**Controllers**

- `base_controller.py`: Abstract controller interface
- `controllers/dwa/`: DWA and velocity obstacle variants
- `controllers/mpc/`: MPC, MPPI, NMPC implementations
- `sfm_controller.py`: Direct SFM-based control

**Simulation**

- `simulation/engine.py`: Core simulation engine (`SimulationEngine`)
- `config.py`: Global configuration (`SimulationConfig`)

**Data & Calibration**

- `data/moods.py`: Mood definitions and SFM parameters
- `behavior_pipeline/`: Multi-stage calibration pipeline
- `data/loader.py`: Trajectory data loading (ATC datasets)

**Visualization**

- `visualization/animation.py`: Interactive Plotly animations
- `metrics/metrics.py`: Performance metric computation

### Data Flow

```
Configuration → Initialization → Simulation Loop → Output
     ↓                ↓                ↓             ↓
  CONFIG         Engine           Step 1-N    Visualization
               Spawner           Per Agent    Metrics
               Controller        Dynamics
               Obstacles         Physics
               Pedestrians       Collision
```

## Configuration

Global settings in `src/sfm_navigation/config.py`:

```python
SimulationConfig(
    # Environment
    env_width=100.0,          # meters
    env_height=100.0,         # meters
    dt=0.1,                   # timestep (seconds)
    max_simulation_time=200.0,# seconds

    # Robot
    robot_radius=0.18,        # meters
    max_linear_vel=2.5,       # m/s
    max_angular_vel=8.0,      # rad/s
    max_linear_accel=3.0,     # m/s²
    max_angular_accel=8.0,    # rad/s²

    # DWA
    dwa_window_time=1.0,      # seconds
    v_resolution=0.1,
    w_resolution=0.1,
    predict_time=1.0,
    alpha_heading=0.8,
    beta_distance=0.3,
    gamma_velocity=0.4,

    # Pedestrian
    pedestrian_radius=0.3,
    pedestrian_avg_speed=1.0,
    social_zone_scale=1.2,
)
```

## Pedestrian Moods

Each mood has calibrated SFM parameters:

| Mood                | Speed Factor | Reactivity | Personal Space | Use Case            |
| ------------------- | ------------ | ---------- | -------------- | ------------------- |
| NORMAL              | 0.6          | 0.5        | 1.0            | Baseline walking    |
| DISTRACTED          | 0.25         | 0.2        | 0.5            | Phone use           |
| STRESSED            | 0.55         | 0.8        | 1.5            | Anxious behavior    |
| IN_RUSH             | 0.70         | 0.3        | 0.3            | Hurried walking     |
| RUNNING             | 0.85         | 0.7        | 1.2            | Athletic movement   |
| CURIOUS             | 0.2          | 0.9        | 1.5            | Exploring behavior  |
| ADVERSARIAL         | 0.5          | 1.0        | 0.2            | Confrontational     |
| JUGGLING            | 0.0          | 0.6        | 0.8            | Stationary behavior |
| SLOW_WALKING        | 0.25         | 0.4        | 1.2            | Elderly pace        |
| Brisk_Individualist | var          | var        | var            | Calibrated regime   |
| Relaxed_Ped         | var          | var        | var            | Calibrated regime   |

> **IMPORTANT**: The table above shows legacy `PedestrianMood` enum values. The **current recommended approach** is to use **calibrated moods** from `data/calibrated_moods/` (Aggressive_barger, Social_Walker, Solo_sprinter, Group_panic, etc.). These are data-driven from real crowd trajectories and use full Extended SFM parameters (12 parameters per mood). See [behavior_pipeline.md](behavior_pipeline.md) for complete list of 20+ calibrated moods and how to use them.

## Controllers

### DWA Family

- **BasicDWA**: Core implementation with maneuvers
- **DW4DO**: For dynamic obstacles
- **DWA_VO**, **DWA_RVO**, **DWA_ORCA**: Velocity obstacle variants

### MPC Family

- **NMPCController**: Nonlinear MPC with planning horizon
- **MPPIController**: Sampling-based cross-entropy method
- **RiskAwareMPPIController**: Risk-aware trajectory selection
- **DCBF variants**: Control Barrier Function safety constraints

### Direct Control

- **SFMController**: Direct social force model steering

## Output Formats

### HTML Animations

- `*_robot_sfm_animation.html`: Main simulation animation with trajectory tracking
- `comparison_animation.html`: Side-by-side controller comparison
- `history_animation.html`: Multi-run trajectory replay
- `*_control_signals_plot.html`: Control input time series

### CSV Data

- `*_simulation_history.csv`: Trajectory and state history
- `*_mood_switch_log.csv`: Pedestrian mood transitions
- `controller_metrics.csv`: Performance comparison table

## Advanced Usage

### Custom Pedestrian Mood

```python
from sfm_navigation.data.moods import register_mood

# Register from CSV
register_mood('path/to/params.csv', 'my_mood_name')

# Create pedestrian with custom mood
ped = Pedestrian(x=10.0, y=10.0, mood='my_mood_name')
```

### Custom Controller

```python
from sfm_navigation.controllers.base_controller import BaseController

class MyController(BaseController):
    def __init__(self, config):
        self.config = config

    def compute_velocity(self, robot_state, goal_pos, obstacles):
        # Your algorithm here
        v = 1.0
        omega = 0.1
        return v, omega
```

### Behavior Pipeline

```bash
# Run full pipeline for behavior calibration
sfm-pipeline

# Configuration: src/sfm_navigation/behavior_pipeline/config.py
# Stages:
#   0: Load binned trajectory data
#   1: Create time windows
#   2: Extract 4 feature sets
#   3: Cluster to find regimes
#   4: Optimize SFM parameters per regime
```

## Project Structure

```
src/sfm_navigation/
├── agents/              # Robot, pedestrian, obstacle models
├── controllers/         # DWA, MPC, MPPI implementations
├── simulation/          # Core simulation engine
├── behavior_pipeline/   # Calibration pipeline (stages 0-4)
├── data/               # Mood definitions, data loaders
├── visualization/      # Animation and metrics rendering
├── metrics/            # Performance metrics
├── prediction/         # LSTM prediction models
├── sfm/                # Social force model utilities
├── spawner/            # Pedestrian spawning
├── cli/                # Command-line interfaces
└── config.py           # Global configuration
```

See [PROJECT_MAP.md](PROJECT_MAP.md) for detailed architecture and module documentation.

## Performance

---

### Documentation Guides

- **[PROJECT_MAP.md](PROJECT_MAP.md)** – Complete system architecture, module index, design patterns, entry points
- **[behavior_pipeline.md](behavior_pipeline.md)** – Behavior calibration pipeline (5 stages), calibrated moods (20+ types), mood generation, Extended SFM parameters

## Performance

- **Simulation Speed**: ~2-10x real-time on CPU (depending on pedestrian count and algorithm)
- **Pedestrian Capacity**: Tested with 25-50 pedestrians in real-time
- **Numba JIT**: Automatic optimization of SFM calculations
- **Visualization**: Plotly renders 1000+ frame animations smoothly

## Contributing

Contributions welcome! Please:

1. Follow existing code style and patterns
2. Add docstrings and type hints to new code
3. Update documentation for new features
4. Inherit from `BaseController` for new controllers
5. Test with existing demo scripts

## Citation

If using this simulator in research, please cite:

```bibtex
@software{sfm_navigation2024,
  title={SFM Navigation: Robot Navigation in Crowds with Realistic Pedestrian Dynamics},
  author={Erfan Atoufi},
  year={2026},
  url={https://github.com/Erfanatf/sfm-navigation}
}
```

## License

MIT License – See LICENSE file for details

## Acknowledgments

- ATC Dataset: Used for behavior calibration
- Numba: JIT compilation for performance
- Plotly: Interactive visualization
- Social Force Model: Helbing & Molnar formulation

## Support

For issues, questions, or feature requests:

- GitHub Issues: [Create an issue](https://github.com/yourusername/sfm-navigation/issues)
- Documentation: See [PROJECT_MAP.md](PROJECT_MAP.md)
- Examples: Check `src/sfm_navigation/cli/` for usage patterns
