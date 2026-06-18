# Algorithms

This document collects the main algorithms used across the repository and connects each theory to the implementation files that deploy it.

## 1. Extended Social Force Model

**Theory**

Pedestrian motion is modeled as a force balance between goal attraction, interpersonal repulsion, obstacle avoidance, group cohesion, and optional curvature bias. A compact form is:

$$
\dot{\mathbf{v}}_i = \frac{\mathbf{v}_i^{\text{des}} - \mathbf{v}_i}{\tau_i} + \sum_j \mathbf{F}_{ij}^{\text{ped}} + \sum_k \mathbf{F}_{ik}^{\text{obs}} + \mathbf{F}_i^{\text{group}} + \mathbf{F}_i^{\text{boundary}} + \mathbf{F}_i^{\text{curve}}
$$

with desired velocity

$$
\mathbf{v}_i^{\text{des}} = v_{0,i} \, [\cos(\theta_i^{\text{goal}}), \sin(\theta_i^{\text{goal}})]
$$

and anisotropic pedestrian repulsion approximately of the form

$$
\mathbf{F}_{ij}^{\text{ped}} = A_{\text{ped}} \exp\left(-\frac{d_{ij}}{B_{\text{ped}}}\right) w(\phi_{ij}), \quad w(\phi_{ij}) = \lambda_{\text{base}} + (1-\lambda_{\text{base}}) \frac{1+\cos\phi_{ij}}{2}.
$$

Group cohesion and gaze/attention terms are additional calibrated extensions used by the calibrated mood system.

**Implementation**

- `src/sfm_navigation/agents/pedestrian.py`
- `src/sfm_navigation/data/moods.py`
- `src/sfm_navigation/cli/robot_demo.py`
- `src/sfm_navigation/cli/demo.py`
- `src/sfm_navigation/cli/crowd_robot_demo.py`

**Calibrated Parameter Set**

The calibrated moods use 12 parameters per mood:

$$
(v_0, \tau, A_{\text{ped}}, B_{\text{ped}}, \lambda_{\text{base}}, \phi_{\text{fov}}, \kappa, k_{\text{group}}, r_{\text{group}}, \theta_{\text{gaze}}, w_{\text{att}}, \text{fov}_{\text{att}})
$$

These parameters are loaded from CSV files in `data/calibrated_moods/` and registered at runtime.

**Important Notes**

- The calibrated mood path is the primary one used by the demos.
- The legacy enum-based mood system still exists for backward compatibility.

## 2. Dynamic Window Approach

**Theory**

The Dynamic Window Approach searches over admissible control inputs under velocity and acceleration limits. For linear and angular velocity:

$$
v \in [v_t - a_{\max} \Delta t, \, v_t + a_{\max} \Delta t] \cap [v_{\min}, v_{\max}]
$$

$$
\omega \in [\omega_t - \alpha_{\max} \Delta t, \, \omega_t + \alpha_{\max} \Delta t] \cap [\omega_{\min}, \omega_{\max}].
$$

Each candidate is rolled out forward in time and scored by heading alignment, obstacle clearance, and progress toward the goal:

$$
J_{\text{DWA}} = w_h J_{\text{heading}} + w_d J_{\text{distance}} + w_v J_{\text{velocity}} + w_p J_{\text{progress}}.
$$

**Variants**

- **BasicDWA** – Pure dynamic-window sampling with rollout-based scoring.

Implementation:
- `src/sfm_navigation/controllers/dwa/basic_dwa.py`
- `src/sfm_navigation/sfm/numba_utils.py`

- **DW4DO** – Extends DWA with dynamic obstacle prediction and time-to-collision scoring. If obstacle motion is approximated as constant velocity, then:

$$
\mathbf{p}_{\text{obs}}(t) = \mathbf{p}_{\text{obs},0} + \mathbf{v}_{\text{obs}} t
$$

and near-term collision risk is penalized via TTC.

Implementation:
- `src/sfm_navigation/controllers/dwa/dw4do.py`

**Velocity-Obstacle Variants**

- **VO** – The velocity obstacle for obstacle \(B\) relative to agent \(A\) is the set of velocities that lead to a collision within horizon \(t_h\):

$$
VO_{A|B}(t_h) = \{ \mathbf{v} : \exists t \in [0, t_h], \| (\mathbf{p}_B - \mathbf{p}_A) - (\mathbf{v} - \mathbf{v}_B) t \| \leq r_A + r_B \}.
$$

The robot penalizes candidate velocities that fall inside these cones.

- **RVO** – Reciprocal VO splits avoidance responsibility between agents. A simplified form shifts the cone apex toward the average relative velocity.

- **ORCA** – ORCA converts collision avoidance to half-plane constraints in velocity space. A candidate velocity is safe if it lies on the admissible side of the separating half-plane.

Implementation:
- `src/sfm_navigation/controllers/dwa/dwa_vo.py`
- `src/sfm_navigation/controllers/dwa/dwa_rvo.py`
- `src/sfm_navigation/controllers/dwa/dwa_orca.py`
- `src/sfm_navigation/controllers/dwa/dwa_utils.py`

**Shared DWA Support**

- Maneuver blending and disturbance rejection: `src/sfm_navigation/controllers/maneuvers.py`
- Trajectory scoring and distance checks: `src/sfm_navigation/sfm/numba_utils.py`

## 3. Model Predictive Control Family

**Theory**

The MPC family optimizes a finite-horizon control sequence subject to robot dynamics, obstacle constraints, and maneuver logic.

For a unicycle/differential-drive model:

$$
\dot{x} = v \cos\theta, \quad \dot{y} = v \sin\theta, \quad \dot{\theta} = \omega.
$$

Discrete rollouts are generated over a horizon \(H\) and a control sequence \(U = \{ \mathbf{u}_0, \dots, \mathbf{u}_{H-1} \}\) is selected to minimize a cost of the form

$$
J(U) = \sum_{k=0}^{H-1} \left( w_p e_p(k)^2 + w_h e_h(k)^2 + w_u \|\mathbf{u}_k\|^2 + w_s e_{\text{safe}}(k) \right) + w_T e_T^2.
$$

**Base MPC Infrastructure**

The base MPC layer provides shared path generation, control blending, and maneuver integration.

Implementation:
- `src/sfm_navigation/controllers/mpc/base_mpc.py`
- `src/sfm_navigation/controllers/maneuvers.py`

**MPPI**

Model Predictive Path Integral control is a sampling-based optimizer. For \(K\) noisy rollouts:

$$
U_k = U_0 + \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, \Sigma)
$$

Each rollout is scored and weighted by

$$
w_k = \frac{\exp(-\beta J_k)}{\sum_j \exp(-\beta J_j)}.
$$

The updated control is the weighted average of the perturbations around the nominal sequence.

Implementation:
- `src/sfm_navigation/controllers/mpc/mppi.py`

**Risk-Aware MPPI**

Risk-aware MPPI adds a risk term that increases the penalty for trajectories that spend time near constraint violations:

$$
w_k \propto \exp(-\beta J_k - \kappa R_k).
$$

Implementation:
- `src/sfm_navigation/controllers/mpc/risk_aware_mppi.py`

**NMPC**

Nonlinear MPC solves a constrained nonlinear program over state and control trajectories, typically using CasADi/SQP. The objective emphasizes path tracking, terminal accuracy, smoothness, and feasibility.

Implementation:
- `src/sfm_navigation/controllers/mpc/nmpc.py`

**DCBF Controllers**

Control Barrier Functions define a safe set

$$
\mathcal{S} = \{ \mathbf{x} : h(\mathbf{x}) \geq 0 \}
$$

with forward invariance ensured by a condition like

$$
h(\mathbf{x}_{k+1}) \geq -\gamma h(\mathbf{x}_k).
$$

For obstacle avoidance, one common barrier is

$$
h(\mathbf{x}) = \|\mathbf{p} - \mathbf{p}_o\|^2 - (r + r_o + s)^2.
$$

Implementation:
- `src/sfm_navigation/controllers/mpc/dcbf_nmpc.py`
- `src/sfm_navigation/controllers/mpc/dcbf_mppi.py`
- `src/sfm_navigation/controllers/mpc/dcbf_mpcc_mppi.py`

**DCBF Fallback Modes**

- grid search: enumerate admissible commands and choose the best safe one
- optimization: project the unsafe command onto the safe set
- analytical: closed-form/sequential projection heuristic

## 4. Maneuver Manager and Disturbance Observer

**Theory**

Maneuvers are not separate planners; they are structured corrective layers that can override or blend with the base controller.

The main maneuvers are:
- overtaking via circulation force
- front repulsion
- parking along the user boundary
- rotation when stuck
- soft recovery for local minima escape

The disturbance observer estimates mismatch between issued and effective commands:

$$
\hat{\mathbf{d}}[k] = L(\mathbf{u}_{\text{measured}}[k] - \mathbf{u}_{\text{issued}}[k]).
$$

The final command is blended as

$$
\mathbf{u}_{\text{final}} = \mathbf{u}_{\text{base}} + \mathbf{u}_{\text{maneuver}} - \hat{\mathbf{d}}.
$$

**Implementation**

- `src/sfm_navigation/controllers/maneuvers.py`
- `src/sfm_navigation/controllers/sfm_controller.py`
- `src/sfm_navigation/controllers/mpc/base_mpc.py`

## 5. Behavior Calibration Pipeline

**Theory**

The behavior pipeline converts raw crowd trajectories into calibrated behavior regimes and then fits SFM parameters to those regimes.

It uses:
- temporal binning
- fixed-length sliding windows
- multi-domain feature extraction
- clustering with PCA and silhouette selection
- parameter optimization with global plus local search

**Optimization**

The calibration stage minimizes feature mismatch between real and simulated trajectories:

$$
\mathcal{L}(\theta) = \sum_{w \in W} \sum_{f \in F} \omega_f (f_{\text{real}}(w) - f_{\text{sim}}(w; \theta))^2.
$$

Global optimization uses Differential Evolution, followed by local L-BFGS-B refinement.

**Implementation**

- `src/sfm_navigation/behavior_pipeline/config.py`
- `src/sfm_navigation/behavior_pipeline/pipeline_runner.py`
- `src/sfm_navigation/behavior_pipeline/stages/stage0_load_bin.py`
- `src/sfm_navigation/behavior_pipeline/stages/stage1_windowing.py`
- `src/sfm_navigation/behavior_pipeline/stages/stage2_features.py`
- `src/sfm_navigation/behavior_pipeline/stages/stage3_clustering.py`
- `src/sfm_navigation/behavior_pipeline/stages/stage4_calibrate.py`
- `src/sfm_navigation/behavior_pipeline/transition_analysis.py`

**Stage Formulas**

Stage 1: Kinematics

$$
v = \sqrt{v_x^2 + v_y^2}, \quad a \approx \frac{v[k] - v[k-1]}{\Delta t}.
$$

Stage 2: Curvature

For planar motion, curvature is estimated by

$$
\kappa = \frac{|x' y'' - y' x''|}{(x'^2 + y'^2)^{3/2}}.
$$

Stage 2: Time-to-Collision

For relative position \(\mathbf{p}_{\text{rel}}\) and relative velocity \(\mathbf{v}_{\text{rel}}\):

$$
a = \|\mathbf{v}_{\text{rel}}\|^2, \quad b = 2(\mathbf{p}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}}), \quad c = \|\mathbf{p}_{\text{rel}}\|^2,
$$

and TTC comes from the quadratic root when \(a>0\) and the discriminant is nonnegative.

Stage 3: Clustering

Features are log-transformed where needed, standardized, optionally reduced by PCA, and clustered with K-means or GMM.

Stage 4: Search/Optimization

Differential Evolution explores parameter space globally, then L-BFGS-B refines the best candidate locally.

## 6. Crowd Analysis, Filtering, and Replay

**Crowd Binning**

Crowd analysis splits ATC data into bins by density over time windows.

Implementation:
- `src/sfm_navigation/data/atc_loader.py`
- `src/sfm_navigation/crowd_analysis/binning.py`
- `src/sfm_navigation/crowd_analysis/visualization.py`

**Trajectory Filtering**

Trajectory denoising can combine a Kalman or Unscented Kalman filter with Savitzky-Golay smoothing.

Implementation:
- `src/sfm_navigation/data/filtering.py`

**Replay and Visualization**

Saved histories are converted back into animation frames using Plotly. This is a post-processing visualization step, not a simulation step.

Implementation:
- `src/sfm_navigation/visualization/animation.py`
- `src/sfm_navigation/cli/animate_history.py`
- `src/sfm_navigation/cli/compare_animations.py`

## 7. Utility Geometry and Scoring Functions

**Trajectory Simulation**

The repository uses exact or near-exact unicycle integration in the form

$$
x_{k+1} = x_k + \begin{cases}
v \cos\theta \, \Delta t, & |\omega| \approx 0 \\
\frac{v}{\omega} (\sin(\theta + \omega \Delta t) - \sin\theta), & \text{otherwise}
\end{cases}
$$

with the corresponding \(y\) update.

**Distance and Heading**

Pointwise distance is computed with the Euclidean norm:

$$
d(\mathbf{p}, \mathbf{q}) = \sqrt{(x_p - x_q)^2 + (y_p - y_q)^2}.
$$

Heading scores are typically normalized by angular deviation from the goal direction.

**Line and Obstacle Geometry**

Point-to-line distance and circle-collision tests are used in DWA, collision analysis, and safety metrics.

Implementation:
- `src/sfm_navigation/sfm/numba_utils.py`
- `src/sfm_navigation/cli/robot_demo.py`

## 8. Metrics and Comparative Analysis

**Theory**

Performance comparison is organized around:
- navigation efficiency
- safety and collision risk
- social comfort
- smoothness and jerk
- path quality
- computational load

These are not one algorithm but a collection of measurement models applied to simulation output.

**Implementation**

- `src/sfm_navigation/metrics/metrics.py`
- `src/sfm_navigation/cli/metrics_report.py`
- `src/sfm_navigation/cli/plot_control_signals.py`
- `src/sfm_navigation/cli/compare_control_signals.py`

## 9. Direct Robot Demo Pipeline

The robot demo combines several algorithms in one runtime path:
- user trajectory reconstruction
- calibrated mood loading
- pedestrian spawning
- optional LSTM prediction
- obstacle assembly
- controller selection
- maneuver blending
- disturbance injection
- collision analysis
- animation and logging

**Implementation:**

- `src/sfm_navigation/cli/robot_demo.py`
- `src/sfm_navigation/controllers/__init__.py`
- `src/sfm_navigation/controllers/maneuvers.py`
- `src/sfm_navigation/prediction/lstm_predictor.py`

## 10. Cross-Module Inconsistency Review

The repository is internally coherent overall, but a few inconsistencies are visible:

- `--use-cbf-opt` is defined in `robot_demo.py` but is not consumed in the runtime path.
- `create_controller()` in `controllers/__init__.py` forwards `robot_params` only for SFM; other controllers ignore that keyword entirely.
- `robot_demo.py` computes `is_mppi` but does not use it.
- The LSTM model and scaler paths in `robot_demo.py` are hardcoded relative paths.
- `robot_demo.py` exposes maneuver toggles in `user_info`, but those toggles are not obviously consumed as hard gating inputs by the maneuver manager itself.
- The doc-and-CLI names are inconsistent in places, e.g. `sfm-compare-controllers-signals` points to `compare_control_signals.py`, while `sfm-plot-control` is the detailed single-run plotter.
- Several controller classes expect extra keyword arguments in different ways, but the factory normalizes them only partially.