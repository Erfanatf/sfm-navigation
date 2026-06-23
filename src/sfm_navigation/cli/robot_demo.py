import json, socket
import os, sys, time, webbrowser, argparse
import numpy as np
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO/WARNING from TF

from ..data.loader import load_multiple_trajectories
from ..agents.user import create_user_trajectory_from_processed_data
from ..agents.robot import DifferentialDriveRobot
from ..agents.obstacles import DynamicObstacle
from ..spawner.spawner import SFMPedestrianSpawner, generate_safe_static_obstacles
from ..config import CONFIG
from ..visualization.animation import create_animation_from_frames, _mood_name
from ..logging.logger import SimulationLogger
from ..controllers.sfm_controller import SFMController
from ..data.moods import register_mood
from ..data.moods import CUSTOM_MOODS
from ..controllers import create_controller
from ..controllers.dwa.basic_dwa import BasicDWA
from ..controllers.dwa.dw4do import DW4DO
from ..controllers.dwa.dwa_vo import DWA_VO
from ..controllers.mpc.mppi import MPPIController
from ..controllers.mpc.dcbf_mppi import DCBFMPPIController

# from ..prediction.lstm_predictor import LSTMPredictor


def send_sim_state(state_dict, host="127.0.0.1", port=9998):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        msg = json.dumps(state_dict).encode()
        sock.sendto(msg, (host, port))
        sock.close()
    except Exception:
        pass


def _auto_register_calibrated_moods(moods_dir="data/calibrated_moods"):
    if not os.path.isdir(moods_dir):
        return
    for fname in os.listdir(moods_dir):
        if fname.endswith(".csv"):
            mood_name = os.path.splitext(fname)[0]
            csv_path = os.path.join(moods_dir, fname)
            try:
                register_mood(csv_path, mood_name)
            except Exception as e:
                print(f"Warning: Failed to register {mood_name}: {e}")


def analyze_robot_collisions(history_df, safety_margin, cooldown_s=0.5):
    """
    Scan the simulation history and count distinct intrusion events
    where the robot violated the safety margin of any other agent.

    Returns a dict with counts and prints a summary.
    """
    robot_rows = history_df[history_df["agent_type"] == "robot"].copy()
    if robot_rows.empty:
        print("No robot data in history.")
        return None

    # Get all other agents (unique by type+id)
    other_agents = history_df[history_df["agent_type"] != "robot"].copy()

    # Initialize collision counters
    collisions = {
        "user": 0,
        "pedestrian": 0,
        "static_obstacle": 0,
        "dynamic_obstacle": 0,
        "total": 0,
        "details": [],  # list of (time, agent_type, agent_id, mood if applicable)
    }

    last_collision_time = {}

    for _, r_row in robot_rows.iterrows():
        t = r_row["time"]
        rx, ry, rrad = r_row["x"], r_row["y"], r_row["radius"]
        # For this time, get all other agents present
        others_now = other_agents[other_agents["time"] == t]
        for _, o_row in others_now.iterrows():
            ox, oy, orad = o_row["x"], o_row["y"], o_row["radius"]
            dist = np.hypot(rx - ox, ry - oy)
            min_dist = rrad + orad + safety_margin
            if dist < min_dist:
                # Collision detected
                agent_type = o_row["agent_type"]
                agent_id = o_row["agent_id"]
                key = f"{agent_type}_{agent_id}"
                # Apply cooldown
                if (
                    key in last_collision_time
                    and (t - last_collision_time[key]) < cooldown_s
                ):
                    continue
                last_collision_time[key] = t
                collisions["total"] += 1
                if agent_type == "user":
                    collisions["user"] += 1
                elif agent_type == "pedestrian":
                    collisions["pedestrian"] += 1
                    mood = o_row.get("mood", "?")
                    collisions["details"].append((t, "pedestrian", agent_id, mood))
                elif agent_type == "static_obstacle":
                    collisions["static_obstacle"] += 1
                elif agent_type == "dynamic_obstacle":
                    collisions["dynamic_obstacle"] += 1

    # Print results
    print("\n" + "=" * 60)
    print("ROBOT COLLISION ANALYSIS")
    print("=" * 60)
    print(f"Total robot collision events: {collisions['total']}")
    print(f"  - with User:            {collisions['user']}")
    print(f"  - with pedestrians:     {collisions['pedestrian']}")
    print(f"  - with static obstacles:{collisions['static_obstacle']}")
    print(f"  - with dynamic obs:     {collisions['dynamic_obstacle']}")
    if collisions["details"]:
        print("\nPedestrian collisions:")
        for t, atype, aid, mood in collisions["details"]:
            print(f"    t={t:.2f}s, ped_id={aid}, mood={mood}")
    print("=" * 60)
    return collisions


def robot_in_collision(rx, ry, rrad, agents, safety_margin):
    """Return True if robot safety margin overlaps any agent."""
    for ax, ay, arad in agents:
        if np.hypot(rx - ax, ry - ay) <= rrad + arad + safety_margin:
            return True
    return False


def main():
    np.random.seed(42)  # Fixing random seed for fair comparison

    parser = argparse.ArgumentParser(description="Robot demo with SFM controller")
    parser.add_argument(
        "--data-folder",
        type=str,
        default=CONFIG.atc_csv_folder,
        help="Path to ATC data folder",
    )
    parser.add_argument(
        "--safety-mode",
        type=str,
        default="all",
        choices=["none", "user-only", "all"],
        help="Safety margin mode: none, user-only, all",
    )
    parser.add_argument(
        "--overtaking",
        action="store_true",
        default=True,
        help="Enable overtaking circulation force",
    )
    parser.add_argument(
        "--no-overtaking",
        dest="overtaking",
        action="store_false",
        help="Disable overtaking circulation force",
    )
    parser.add_argument(
        "--repulsion",
        action="store_true",
        default=True,
        help="Enable front repulsion maneuver",
    )
    parser.add_argument(
        "--no-repulsion",
        dest="repulsion",
        action="store_false",
        help="Disable front repulsion maneuver",
    )
    parser.add_argument(
        "--parking",
        action="store_true",
        default=True,
        help="Enable parking maneuver",
    )
    parser.add_argument(
        "--no-parking",
        dest="parking",
        action="store_false",
        help="Disable parking maneuver",
    )
    parser.add_argument(
        "--rotation",
        action="store_true",
        default=True,
        help="Enable in‑place rotation maneuver",
    )
    parser.add_argument(
        "--no-rotation",
        dest="rotation",
        action="store_false",
        help="Disable in‑place rotation maneuver",
    )
    parser.add_argument(
        "--robot-mood",
        type=str,
        default=None,
        help="Name of a calibrated mood for the robot (e.g., Social_Walker). "
        "If not given or not found, uses hand‑crafted safe defaults.",
    )
    parser.add_argument(
        "--mood-switch-rate",
        type=float,
        default=0.0,
        help="Probability per second of a pedestrian changing mood",
    )
    parser.add_argument(
        "--transition-matrix",
        type=str,
        default=None,
        help="Path to CSV with from,to,prob columns",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="SFM",
        choices=[
            "SFM",
            "DWA_BASIC",
            "DWA_DW4DO",
            "DWA_VO",
            "DWA_RVO",
            "DWA_ORCA",
            "MPPI",
            "DCBF_MPPI",
            "RISK_AWARE_MPPI",
            "STANDARD_MPC",
            "NMPC",
            "DCBF_NMPC",
            "DCBF_MPCC_MPPI",
        ],
        help="Controller type",
    )
    parser.add_argument(
        "--disturbance-active",
        dest="isdistactive",
        action="store_true",
        default=False,
        help="Whether to inject disturbance on robot dynamics",
    )
    parser.add_argument(
        "--use-cbf-opt",
        dest="cbf_opt",
        action="store_true",
        default=False,
        help="Solve OPT for projecting to closest safe command",
    )
    parser.add_argument(
        "--dcbf-fallback-mode",
        type=str,
        default="grid",
        choices=["grid", "optimization", "analytical"],
        help="Fallback safety projection method for DCBF_NMPC (grid / optimization / analytical)",
    )
    parser.add_argument(
        "--use-lstm-predictor",
        action="store_true",
        default=False,
        help="Use LSTM network to predict the user's future path",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Run as fast as possible (no real‑time sync, no UDP)",
    )

    args = parser.parse_args()

    USE_OVERTAKING = args.overtaking
    USE_REPULSION = args.repulsion
    USE_PARKING = args.parking
    USE_ROTATION = args.rotation

    print(
        f"Overtaking: {'ON' if USE_OVERTAKING else 'OFF'}, "
        f"Repulsion: {'ON' if USE_REPULSION else 'OFF'}, "
        f"Parking: {'ON' if USE_PARKING else 'OFF'}, "
        f"Rotation: {'ON' if USE_ROTATION else 'OFF'}, "
    )
    print(f"DCBF projection fallback: {args.dcbf_fallback_mode}")

    _auto_register_calibrated_moods()

    # Parameters
    TRAJECTORY_INDEX = 1
    SIMULATION_DURATION = None
    SFM_DT = None
    FRAME_SKIP = 3
    FOLLOW_USER = True
    FOLLOW_ZOOM_RADIUS = 15.0
    N_PEDESTRIANS = 15
    VICINITY_RADIUS = 10.0
    RESPAWN_DISTANCE = 15.0
    MIN_SPAWN_DISTANCE = 2.0
    MAX_SPAWN_DISTANCE = 5.0
    SPEED_SCALE_FACTOR = 1.0
    N_STATIC_OBSTACLES = 10
    MIN_OBSTACLE_RADIUS = 0.3
    MAX_OBSTACLE_RADIUS = 0.8
    OBSTACLE_SAFETY_MARGIN = 0.5
    ENABLE_LOGGING = True
    LOG_WINDOW_DURATION = 5.0
    DESIRED_CONTROL_FREQ = 12.0
    STATIONARY_SPEED_THRESHOLD = 0.1

    DRIVE_DATA_FOLDER = args.data_folder
    SAFETY_MODE = args.safety_mode

    print(f"Safety mode: {SAFETY_MODE}")

    print("Loading trajectories...")
    all_x_true, all_dt, AVG_DT = load_multiple_trajectories(DRIVE_DATA_FOLDER, n=24)
    idx = TRAJECTORY_INDEX if TRAJECTORY_INDEX < len(all_x_true) else 0
    x_true_sample = all_x_true[idx]
    dt_sample = all_dt[idx]

    user_traj = create_user_trajectory_from_processed_data(
        x_true_sample,
        avg_dt=dt_sample,
        env_width=CONFIG.env_width,
        env_height=CONFIG.env_height,
    )

    transition_matrix = None
    mood_rates = None
    if args.transition_matrix:
        from ..spawner.spawner import load_transition_matrix

        transition_matrix, mood_rates = load_transition_matrix(args.transition_matrix)
        print(f"Loaded mood transition matrix from {args.transition_matrix}")
        if mood_rates:
            print(f"Loaded per‑mood Poisson rates: {mood_rates}")

    spawner = SFMPedestrianSpawner(
        user_trajectory=user_traj,
        config=CONFIG,
        n_pedestrians=N_PEDESTRIANS,
        vicinity_radius=VICINITY_RADIUS,
        respawn_distance=RESPAWN_DISTANCE,
        min_spawn_distance=MIN_SPAWN_DISTANCE,
        max_spawn_distance=MAX_SPAWN_DISTANCE,
        speed_scale_factor=SPEED_SCALE_FACTOR,
        mood_switch_rate=args.mood_switch_rate,
        mood_transition_matrix=transition_matrix,
        mood_switch_rates=mood_rates,
    )

    static_obs = generate_safe_static_obstacles(
        user_trajectory=user_traj,
        n_obstacles=N_STATIC_OBSTACLES,
        min_radius=MIN_OBSTACLE_RADIUS,
        max_radius=MAX_OBSTACLE_RADIUS,
        env_width=CONFIG.env_width,
        env_height=CONFIG.env_height,
        safety_margin=OBSTACLE_SAFETY_MARGIN,
    )

    dynamic_obs = [
        DynamicObstacle(
            60, 50, 0.5, vx=0.4, vy=0.0, obstacle_id=0, motion_type="linear"
        ),
        DynamicObstacle(
            40, 55, 0.6, vx=0.0, vy=0.0, obstacle_id=1, motion_type="circular"
        ),
    ]

    sim_duration = (
        user_traj.total_duration - 0.5
        if SIMULATION_DURATION is None
        else min(SIMULATION_DURATION, user_traj.total_duration - 0.5)
    )

    # --- LSTM predictor (optional) ---
    if args.use_lstm_predictor:
        # lstm_predictor = LSTMPredictor(
        #     model_path="src/sfm_navigation/prediction/LSTM_2ndOrder_yaw_rate_rem.keras",
        #     state_scaler_path="src/sfm_navigation/prediction/state_scaler.pkl",
        #     delta_scaler_path="src/sfm_navigation/prediction/delta_scaler.pkl",
        # )
        # lstm_predictor.initialize_buffer(user_traj, start_time=0.0)
        # print("LSTM predictor initialised.")
        print(
            "LSTM predictor is currently disabled due to loading issues. Using constant velocity prediction instead."
        )
    else:
        lstm_predictor = None

    sfm_dt = dt_sample if SFM_DT is None else SFM_DT
    desired_period = 1.0 / DESIRED_CONTROL_FREQ

    # --- Robot SFM parameters ---
    handcrafted_defaults = {
        "v0": 2.0,  # slightly faster to overtake
        "tau": 0.3,  # quicker reaction
        "A_ped": 12.0,  # much stronger repulsion
        "B_ped": 0.5,  # short range, sharp force
        "lam_base": 0.3,  # less anisotropic (more omnidirectional awareness)
        "kappa": 0.0,
    }

    if args.robot_mood is not None:
        robot_params = CUSTOM_MOODS.get(args.robot_mood)
        if robot_params is not None:
            print(f"Robot using calibrated mood '{args.robot_mood}'")
        else:
            print(f"Warning: mood '{args.robot_mood}' not found. Using safe defaults.")
            robot_params = handcrafted_defaults
    else:
        print("No robot mood specified. Using hand‑crafted safe defaults.")
        robot_params = handcrafted_defaults

    controller = create_controller(args.controller, CONFIG, robot_params=robot_params)

    # Unified fallback mode handling
    if hasattr(controller, "fallback_mode"):
        controller.fallback_mode = args.dcbf_fallback_mode
    if hasattr(controller, "use_optimization"):
        # For DCBF_MPPI: sync the old flag with the new mode
        controller.use_optimization = args.dcbf_fallback_mode == "optimization"

    ux0, uy0 = user_traj.get_position_at_time(0)
    robot = DifferentialDriveRobot(CONFIG, start_x=ux0 + 2.0, start_y=uy0, tau=0.1)

    logger = (
        SimulationLogger(
            window_duration=LOG_WINDOW_DURATION, vicinity_radius=VICINITY_RADIUS
        )
        if ENABLE_LOGGING
        else None
    )

    n_steps = int(sim_duration / sfm_dt)
    spawner.initialize_pedestrians(current_time=0.0)
    frames_data = []
    robot_traj_abs = []
    history = []  # NEW: per‑step detailed state

    prev_user_pos = user_traj.get_position_at_time(0)

    # --- Control timing ---
    desired_period = 1.0 / DESIRED_CONTROL_FREQ  # 0.1 s for 10 Hz
    next_control_time = 0.0
    v_cmd_held = 0.0
    omega_cmd_held = 0.0

    if not args.batch:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 9999))
        sock.setblocking(False)
    else:
        sock = None

    # Variable to hold the latest ω
    external_omega = 0.0

    print(f"Running simulation for {sim_duration:.2f}s ({n_steps} steps)...")
    t_total_start = time.perf_counter()
    wall_start = time.perf_counter()
    for step in range(n_steps):
        loop_start = time.perf_counter()

        # Real-time sync: start wall clock
        t = step * sfm_dt
        user_x, user_y = user_traj.get_position_at_time(t)
        user_state = user_traj.get_state_at_time(t)
        user_facing = user_state[2]
        dx = user_x - prev_user_pos[0]
        dy = user_y - prev_user_pos[1]
        user_speed = np.hypot(dx, dy) / sfm_dt
        user_motion_angle = np.arctan2(dy, dx) if user_speed > 0.05 else user_facing
        prev_user_pos = (user_x, user_y)

        for dob in dynamic_obs:
            dob.update(sfm_dt, CONFIG.env_width, CONFIG.env_height)
        spawner.update_pedestrians(sfm_dt, t, static_obstacles=static_obs)

        # Build obstacle list (unchanged)
        is_dwa = isinstance(controller, (BasicDWA, DW4DO, DWA_VO))
        is_mppi = isinstance(controller, (MPPIController, DCBFMPPIController))
        obstacles = [[obs.x, obs.y, obs.radius] for obs in static_obs]
        for dob in dynamic_obs:
            obstacles.append([dob.x, dob.y, dob.radius])
        if SAFETY_MODE in ("user-only", "all"):
            if not (is_dwa):
                obstacles.append([user_x, user_y, user_traj.radius])
        if SAFETY_MODE == "all":
            for ped in spawner.pedestrians:
                if is_dwa:
                    obstacles.append([ped.x, ped.y, ped.radius])
                else:
                    obstacles.append(
                        [ped.x, ped.y, ped.radius + user_traj.safety_radius]
                    )

        # ---- Build pedestrian mood parameters for social potential ----
        pedestrian_params = []
        for ped in spawner.pedestrians:
            params = CUSTOM_MOODS.get(ped.mood, {})
            pedestrian_params.append(
                [
                    ped.x,
                    ped.y,
                    ped.theta,  # position & heading
                    params.get("B_ped", 0.5),
                    params.get("lam_base", 0.5),
                    params.get("phi_fov", np.deg2rad(90)),
                    params.get("theta_gaze", 0.0),
                ]
            )
        pedestrian_params = (
            np.array(pedestrian_params)
            if len(pedestrian_params) > 0
            else np.empty((0, 7))
        )

        # ── Goal logic ──────────────────────────────────────────────
        t_pred_start = time.perf_counter()

        lookahead = 2.0
        lstm_pred_time = 0.0

        # Extract raw user state for LSTM (order must match training)
        user_state_raw = np.array(
            [
                user_x,
                user_y,
                user_facing,
                user_speed * np.cos(user_motion_angle - user_facing),
                user_speed * np.sin(user_motion_angle - user_facing),
            ]
        )

        # Receive latest ω from rotation predictor (non‑blocking)
        if sock is not None:
            try:
                data, _ = sock.recvfrom(1024)
                external_omega = float(data.decode().strip())
            except BlockingIOError:
                pass
            except Exception:
                external_omega = 0.0

        if args.use_lstm_predictor and lstm_predictor is not None:
            goal_x, goal_y, lstm_pred_time = lstm_predictor.predict(user_state_raw)
        else:
            if user_speed < STATIONARY_SPEED_THRESHOLD:
                goal_dist = user_traj.safety_radius + 0.5
                goal_x = user_x + goal_dist * np.cos(user_facing)
                goal_y = user_y + goal_dist * np.sin(user_facing)
            else:
                # ----- New: use external ω if available -----
                if abs(external_omega) > 1e-6:
                    # Unicycle model: constant speed, constant turn rate
                    v = user_speed
                    theta0 = user_facing
                    omega = external_omega
                    r = v / omega
                    dtheta = omega * lookahead
                    goal_x = user_x + r * (np.sin(theta0 + dtheta) - np.sin(theta0))
                    goal_y = user_y - r * (np.cos(theta0 + dtheta) - np.cos(theta0))
                else:
                    # Fallback: original constant‑velocity straight line
                    future_time = t + lookahead
                    if future_time < user_traj.total_duration:
                        goal_x = user_x + dx * 2.0 / sfm_dt
                        goal_y = user_y + dy * 2.0 / sfm_dt
                    else:
                        last_x, last_y = user_traj.get_position_at_time(
                            user_traj.total_duration
                        )
                        last_v = user_traj.avg_velocity
                        dir_x, dir_y = user_traj.get_direction_at_time(
                            user_traj.total_duration - 0.1
                        )
                        goal_x = last_x + dir_x * last_v * lookahead
                        goal_y = last_y + dir_y * last_v * lookahead

        pred_elapsed = time.perf_counter() - t_pred_start
        # ────────────────────────────────────────────────────────────

        user_info = {
            "x": user_x,
            "y": user_y,
            "radius": user_traj.safety_radius,
            "facing": user_facing,
            "overtaking_active": USE_OVERTAKING,
            "repulsion_active": USE_REPULSION,
            "parking_active": USE_PARKING,
            "rotation_active": USE_ROTATION,
            "user_speed": user_speed,
            "user_heading": user_motion_angle,
        }

        # ── Control update at desired frequency ──────────────────
        t_ctrl_start = time.perf_counter()
        if t >= next_control_time - 1e-12:
            v_cmd, omega_cmd = controller.compute_velocity(
                robot.state,
                (goal_x, goal_y),
                obstacles,
                user=user_info,
                dt=desired_period,
                sim_time=t,
                pedestrian_params=pedestrian_params,
            )
            v_cmd_held, omega_cmd_held = v_cmd, omega_cmd
            next_control_time += desired_period
        else:
            v_cmd, omega_cmd = v_cmd_held, omega_cmd_held
        ctrl_elapsed = time.perf_counter() - t_ctrl_start
        # ──────────────────────────────────────────────────────────

        # Target path (unchanged)
        target_path = getattr(controller, "_target_path", np.zeros((0, 2)))
        if target_path.shape[1] >= 2:
            target_path_x = target_path[:, 0].tolist()
            target_path_y = target_path[:, 1].tolist()
        else:
            target_path_x, target_path_y = [], []

        # Read DOB signals (unchanged)
        v_base = getattr(controller, "_v_base", 0.0)
        omega_base = getattr(controller, "_omega_base", 0.0)
        v_man = getattr(controller, "_v_man", 0.0)
        omega_man = getattr(controller, "_omega_man", 0.0)
        v_final = getattr(controller, "_v_final", v_cmd)
        omega_final = getattr(controller, "_omega_final", omega_cmd)

        overtaking_now = getattr(controller, "overtaking_active", False)
        parking_now = getattr(controller, "parking_active", False)
        repulsion_now = getattr(controller, "repulsion_active", False)
        soft_recovery_now = getattr(controller, "soft_recovery_active", False)
        rotation_now = getattr(controller, "rotation_active", False)

        comp_time = controller.last_compute_time
        rt_factor = controller.get_real_time_factor(desired_period)

        # Disturbance (unchanged)
        if args.isdistactive:
            dist_amplitude_v = 2.0
            dist_amplitude_w = 2.0
            dist_freq = 1.0
            d_ext_v = dist_amplitude_v * np.sin(2 * np.pi * dist_freq * t)
            d_ext_w = dist_amplitude_w * np.cos(2 * np.pi * dist_freq * t)
        else:
            d_ext_v, d_ext_w = 0.0, 0.0

        robot.update(v_cmd, omega_cmd, sfm_dt, d_ext=np.array([d_ext_v, d_ext_w]))

        # Acceleration & jerk from robot (unchanged)
        lin_accel, ang_accel = robot.accel_history[-1]
        lin_jerk, ang_jerk = robot.jerk_history[-1]

        loop_elapsed = time.perf_counter() - loop_start
        elapsed_wall = time.perf_counter() - wall_start

        # Note: the history block should use v_cmd, omega_cmd (which are the held values) and all flags as usual.

        # Capture DOB estimate and external disturbance
        d_hat_v, d_hat_omega = 0.0, 0.0
        if hasattr(controller, "dob") and controller.dob is not None:
            d_hat_v, d_hat_omega = controller.dob.d_hat[0], controller.dob.d_hat[1]
        d_ext_v_hist = d_ext_v  # already defined in your loop
        d_ext_w_hist = d_ext_w

        # Gather all relevant agents for collision check (user, pedestrians, dynamic/static obstacles)
        agent_positions = []
        agent_positions.append((user_x, user_y, user_traj.radius))  # user
        for ped in spawner.pedestrians:
            agent_positions.append((ped.x, ped.y, ped.radius))
        for dob in dynamic_obs:
            agent_positions.append((dob.x, dob.y, dob.radius))
        for obs in static_obs:
            agent_positions.append((obs.x, obs.y, obs.radius))

        in_coll = robot_in_collision(
            robot.state.x,
            robot.state.y,
            CONFIG.robot_radius,
            agent_positions,
            CONFIG.safety_margin,
        )

        if logger:
            logger.update(
                current_time=t,
                pedestrians=spawner.pedestrians,
                user_pos=(user_x, user_y),
                robot_pos=(robot.state.x, robot.state.y),
                static_obstacles=static_obs,
            )

        if step % max(1, n_steps // 10) == 0:
            print(
                f"  Step {step}/{n_steps}, RT factor: {rt_factor:.4f} "
                f"(controller compute: {controller.last_compute_time*1000:.2f}ms, "
                f"prediction: {pred_elapsed*1000:.2f}ms, "
                f"total loop: {loop_elapsed*1000:.1f}ms), "
                f"Prediction Time: {lstm_pred_time*1000:.2f}ms"
            )

        # --- collect detailed history ---
        history.append(
            {
                "time": t,
                "agent_type": "user",
                "agent_id": -1,
                "x": user_x,
                "y": user_y,
                "vx": user_speed * np.cos(user_motion_angle),
                "vy": user_speed * np.sin(user_motion_angle),
                "theta": user_facing,
                "radius": user_traj.radius,
                "safety_radius": user_traj.safety_radius,
            }
        )
        history.append(
            {
                "time": t,
                "agent_type": "robot",
                "agent_id": 0,
                "x": robot.state.x,
                "y": robot.state.y,
                "vx": robot.state.v * np.cos(robot.state.theta),
                "vy": robot.state.v * np.sin(robot.state.theta),
                "theta": robot.state.theta,
                "radius": CONFIG.robot_radius,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "v_cmd": v_cmd,  # same as base
                "omega_cmd": omega_cmd,  # same as base
                "comp_time": comp_time,
                "in_collision": in_coll,
                "controller": type(controller).__name__,
                "v_base": v_base,
                "omega_base": omega_base,
                "v_man": v_man,
                "omega_man": omega_man,
                "v_final": v_final,
                "omega_final": omega_final,
                "lin_speed": robot.state.v,
                "ang_speed": robot.state.omega,
                "lin_accel": lin_accel,
                "ang_accel": ang_accel,
                "lin_jerk": lin_jerk,
                "ang_jerk": ang_jerk,
                "d_hat_v": d_hat_v,
                "d_hat_omega": d_hat_omega,
                "d_ext_v": d_ext_v_hist,
                "d_ext_omega": d_ext_w_hist,
                "overtaking_active": overtaking_now,
                "parking_active": parking_now,
                "repulsion_active": repulsion_now,
                "soft_recovery_active": soft_recovery_now,
                "rotation_active": rotation_now,
                "lstm_pred_time": lstm_pred_time,
                "sim_rt_factor": elapsed_wall / (t + 1e-9) if t > 0 else 0.0,
                "predictor_loop_time_ms": 0.0,  # placeholder; filled from predictor log if needed
            }
        )
        for ped in spawner.pedestrians:
            history.append(
                {
                    "time": t,
                    "agent_type": "pedestrian",
                    "agent_id": ped.pedestrian_id,
                    "x": ped.x,
                    "y": ped.y,
                    "vx": ped.vx,
                    "vy": ped.vy,
                    "theta": ped.theta,
                    "radius": ped.radius,
                    "mood": _mood_name(ped.mood),
                    "theta_gaze": ped.theta_gaze,
                    "fov_att": ped.fov_att,
                    "w_att": ped.w_att,
                }
            )
        for dob in dynamic_obs:
            history.append(
                {
                    "time": t,
                    "agent_type": "dynamic_obstacle",
                    "agent_id": dob.obstacle_id,
                    "x": dob.x,
                    "y": dob.y,
                    "vx": dob.vx,
                    "vy": dob.vy,
                    "theta": 0.0,
                    "radius": dob.radius,
                }
            )
        for obs in static_obs:
            history.append(
                {
                    "time": t,
                    "agent_type": "static_obstacle",
                    "agent_id": obs.obstacle_id,
                    "x": obs.x,
                    "y": obs.y,
                    "vx": 0.0,
                    "vy": 0.0,
                    "theta": 0.0,
                    "radius": obs.radius,
                }
            )

        # --- frame collection for animation ---
        if step % FRAME_SKIP == 0:
            arc_x, arc_y = [], []
            # ---- Robot future arc (look‑ahead 1.0 s) ----
            arc_duration = 1.0  # how far ahead to draw (seconds)
            n_arc_pts = 20
            if abs(omega_cmd) > 1e-6:
                r_arc = v_cmd / omega_cmd
                dtheta = omega_cmd * arc_duration
                theta0 = robot.state.theta
                angles = np.linspace(0, dtheta, n_arc_pts)
                arc_x = robot.state.x + r_arc * (
                    np.sin(theta0 + angles) - np.sin(theta0)
                )
                arc_y = robot.state.y - r_arc * (
                    np.cos(theta0 + angles) - np.cos(theta0)
                )
            else:
                # Straight line
                total_dist = v_cmd * arc_duration
                for k in range(n_arc_pts + 1):
                    frac = k / n_arc_pts
                    arc_x.append(
                        robot.state.x + frac * total_dist * np.cos(robot.state.theta)
                    )
                    arc_y.append(
                        robot.state.y + frac * total_dist * np.sin(robot.state.theta)
                    )

            # Store in frames_data
            # ... (the rest of the frame_data dict already includes 'robot_arc_x' and 'robot_arc_y')

            future_path_x = [user_x, goal_x]
            future_path_y = [user_y, goal_y]

            frames_data.append(
                {
                    "time": t,
                    "user_pos": (user_x, user_y),
                    "user_motion_angle": user_motion_angle,
                    "user_facing_angle": user_facing,
                    "pedestrians": [
                        (
                            p.x,
                            p.y,
                            p.vx,
                            p.vy,
                            _mood_name(p.mood),
                            p.theta,
                            p.theta_gaze,
                            p.fov_att,
                            p.w_att,
                        )
                        for p in spawner.pedestrians
                    ],
                    "robot_pos": (robot.state.x, robot.state.y),
                    "robot_theta": robot.state.theta,
                    "robot_arc_x": arc_x,
                    "robot_arc_y": arc_y,
                    "goal_pos": (goal_x, goal_y),
                    "future_path_x": future_path_x,
                    "future_path_y": future_path_y,
                    "dynamic_obstacles": [
                        (dob.x, dob.y, dob.radius, dob.obstacle_id)
                        for dob in dynamic_obs
                    ],
                    "respawn_count": spawner.respawn_count,
                    "overtaking_active": overtaking_now,
                    "parking_active": parking_now,
                    "repulsion_active": repulsion_now,
                    "soft_recovery_active": soft_recovery_now,
                    "rotation_active": rotation_now,
                    "target_path_x": target_path_x,
                    "target_path_y": target_path_y,
                }
            )
            robot_traj_abs.append((robot.state.x, robot.state.y))

        if step % FRAME_SKIP == 0:
            # Build state dict for live view
            # Gather nearest pedestrians (unchanged)
            ped_list = []
            for p in spawner.pedestrians:
                ped_list.append((p.x, p.y, p.radius))
            ped_list.sort(key=lambda pt: np.hypot(pt[0] - user_x, pt[1] - user_y))
            ped_list = ped_list[:10]

            # ---- Sample user ground‑truth trajectory up to current time ----
            traj_points = []
            sample_dt = sfm_dt * 5
            n_samples = int(t / sample_dt) + 1
            for i in range(n_samples):
                ts = i * sample_dt
                ux, uy = user_traj.get_position_at_time(ts)
                traj_points.append((ux, uy))

            # ---- Compute predicted user arc (same logic as goal) ----
            user_arc = []
            n_arc_pts = 20
            if user_speed >= STATIONARY_SPEED_THRESHOLD and abs(external_omega) > 1e-6:
                # Unicycle model arc
                v = user_speed
                theta0 = user_facing
                omega = external_omega
                r = v / omega
                dtheta = omega * lookahead
                angles = np.linspace(0, dtheta, n_arc_pts)
                for ang in angles:
                    ax = user_x + r * (np.sin(theta0 + ang) - np.sin(theta0))
                    ay = user_y - r * (np.cos(theta0 + ang) - np.cos(theta0))
                    user_arc.append((ax, ay))
            elif user_speed >= STATIONARY_SPEED_THRESHOLD:
                # Straight line arc
                total_dist = user_speed * lookahead
                for k in range(n_arc_pts + 1):
                    frac = k / n_arc_pts
                    ax = user_x + frac * total_dist * np.cos(user_facing)
                    ay = user_y + frac * total_dist * np.sin(user_facing)
                    user_arc.append((ax, ay))
            else:
                # Stationary – just a dot at user position
                user_arc = [(user_x, user_y)]
            # -------------------------------------------------------

            state = {
                "user": (user_x, user_y, user_facing),
                "robot": (robot.state.x, robot.state.y, robot.state.theta),
                "goal": (goal_x, goal_y),
                "arc": list(zip(arc_x, arc_y)),
                "user_arc": user_arc,
                "pedestrians": ped_list,
                "user_traj": traj_points,
                "user_speed": user_speed,
                "robot_speed": robot.state.v,
                "sim_rt_factor": elapsed_wall / (t + 1e-9) if t > 0 else 0.0,
                "loop_time_ms": loop_elapsed * 1000,
                "pred_time_ms": pred_elapsed * 1000,
                "ctrl_time_ms": ctrl_elapsed * 1000,
                }

            if not args.batch:
                send_sim_state(state)

        # ── Real‑time synchronization ────────────────────────────
        if not args.batch:
            expected_wall = t
            sleep_time = expected_wall - elapsed_wall
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif step > 0 and step % 100 == 0:
                rt_factor = elapsed_wall / (t + 1e-9)
                print(
                    f"[sync] Simulation lagging: sim {t:.2f}s, wall {elapsed_wall:.2f}s, "
                    f"RT factor {rt_factor:.3f}"
                )
        # ──────────────────────────────────────────────────────────

    if sock is not None:
        sock.close()

    t_total = time.perf_counter() - t_total_start
    print(f"Simulation complete in {t_total:.1f}s")

    # Save history CSV
    history_df = pd.DataFrame(history)
    history_csv = type(controller).__name__ + "_simulation_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"Simulation history saved to {history_csv}")

    # Analyze robot collisions
    analyze_robot_collisions(history_df, CONFIG.safety_margin)

    # Save mood switch log
    switch_log = spawner.get_mood_switch_log()
    if switch_log:
        switch_df = pd.DataFrame(
            switch_log, columns=["time", "ped_id", "from_mood", "to_mood"]
        )
        switch_csv = type(controller).__name__ + "_mood_switch_log.csv"
        switch_df.to_csv(switch_csv, index=False)
        print(f"\nMood switch log saved to {switch_csv}")
        # Print summary
        print("\nMOOD SWITCH SUMMARY")
        print("=" * 40)
        print(f"Total switches: {len(switch_df)}")
        print("Switches per from_mood:")
        print(switch_df["from_mood"].value_counts().to_string())
        print("\nSwitches per to_mood:")
        print(switch_df["to_mood"].value_counts().to_string())
        print("=" * 40)
    else:
        print("No mood switches occurred.")

    if logger:
        logger.finalize(sim_duration, spawner.pedestrians, (user_x, user_y))
        logger.print_summary()

    robot_data_dict = {"trajectory": robot_traj_abs, "radius": CONFIG.robot_radius}

    anim_fig = create_animation_from_frames(
        frames_data=frames_data,
        user_trajectory=user_traj,
        static_obstacles=static_obs,
        follow_user=FOLLOW_USER,
        follow_zoom_radius=FOLLOW_ZOOM_RADIUS,
        spawner=spawner,
        robot_data=robot_data_dict,
        draw_pedestrian_safety=(SAFETY_MODE == "all"),
        controller_name=type(controller).__name__,
    )

    html_path = type(controller).__name__ + "_robot_sfm_animation.html"
    anim_fig.write_html(html_path)
    print(f"Animation saved to {html_path}")
    try:
        webbrowser.open(html_path, new=1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
