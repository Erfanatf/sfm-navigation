from enum import Enum, auto
import numpy as np
import pandas as pd

class PedestrianMood(Enum):
    NORMAL = auto()
    DISTRACTED = auto()
    STRESSED = auto()
    IN_RUSH = auto()
    JUGGLING = auto()
    RUNNING = auto()
    CURIOUS = auto()
    ADVERSARIAL = auto()
    SLOW_WALKING = auto()
    Brisk_Individualist = auto()
    Relaxed_Ped = auto()


# Default mood parameters (used when calibrated CSV not available)
MOOD_PARAMETERS = {
    PedestrianMood.NORMAL: {
        'speed_factor': 0.6,
        'direction_variance': 0.1,
        'personal_space': 1.0,
        'reactivity': 0.5,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Regular walking pattern, predictable trajectory'
    },
    PedestrianMood.DISTRACTED: {
        'speed_factor': 0.25,
        'direction_variance': 0.4,
        'personal_space': 0.5,
        'reactivity': 0.2,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Looking at phone, slower with erratic movements'
    },
    PedestrianMood.STRESSED: {
        'speed_factor': 0.55,
        'direction_variance': 0.2,
        'personal_space': 1.5,
        'reactivity': 0.8,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Fast, tense movements, maintains larger personal space'
    },
    PedestrianMood.IN_RUSH: {
        'speed_factor': 0.70,
        'direction_variance': 0.05,
        'personal_space': 0.3,
        'reactivity': 0.3,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Very fast, determined path, minimal yielding'
    },
    PedestrianMood.JUGGLING: {
        'speed_factor': 0.0,
        'direction_variance': 0.0,
        'personal_space': 0.8,
        'reactivity': 0.6,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Standing still, shifting weight foot-to-foot'
    },
    PedestrianMood.RUNNING: {
        'speed_factor': 0.85,
        'direction_variance': 0.1,
        'personal_space': 1.2,
        'reactivity': 0.7,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Athletic running, very fast movement'
    },
    PedestrianMood.CURIOUS: {
        'speed_factor': 0.2,
        'direction_variance': 0.6,
        'personal_space': 1.5,
        'reactivity': 0.9,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Slow, looking around, may stop to observe'
    },
    PedestrianMood.ADVERSARIAL: {
        'speed_factor': 0.5,
        'direction_variance': 0.3,
        'personal_space': 0.2,
        'reactivity': 1.0,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Intentionally moves toward robot, then yields'
    },
    PedestrianMood.SLOW_WALKING: {
        'speed_factor': 0.25,
        'direction_variance': 0.15,
        'personal_space': 1.2,
        'reactivity': 0.4,
        'tau': 0.5,
        'yielding': 0.5,
        'description': 'Elderly or casual pace, very slow movement'
    }
}
# Default parameters for Brisk_Individualist (fallback if CSV not available)
MOOD_PARAMETERS[PedestrianMood.Brisk_Individualist] = {
    'v0': 1.2,
    'tau': 0.5,
    'A_ped': 3.0,
    'B_ped': 0.5,
    'lam_base': 0.5,
    'phi_fov': np.deg2rad(90),
    'kappa': 0.0,
    'k_group': 0.0,
    'r_group': 0.5,
    'theta_gaze': 0.0,
    'w_att': 0.0,
    'fov_att': np.deg2rad(30),
    'speed_factor': 0.8,
    'direction_variance': 0.1,
    'personal_space': 1.0,
    'reactivity': 0.5,
    'yielding': 0.5,
    'description': 'Default Brisk Individualist (no calibration CSV)'
}

# Default parameters for Relaxed_Ped (fallback if CSV not available)
MOOD_PARAMETERS[PedestrianMood.Relaxed_Ped] = {
    'v0': 1.0,
    'tau': 0.5,
    'A_ped': 2.5,
    'B_ped': 0.6,
    'lam_base': 0.4,
    'phi_fov': np.deg2rad(80),
    'kappa': 0.0,
    'k_group': 0.0,
    'r_group': 0.5,
    'theta_gaze': 0.0,
    'w_att': 0.0,
    'fov_att': np.deg2rad(30),
    'speed_factor': 0.6,
    'direction_variance': 0.2,
    'personal_space': 1.2,
    'reactivity': 0.4,
    'yielding': 0.5,
    'description': 'Default Relaxed Pedestrian (no calibration CSV)'
}

def load_calibrated_moods(csv_dir='.'):
    """
    Load calibrated moods from CSVs and update MOOD_PARAMETERS.
    Expects files: phase3_optimized_sfm_params_Brisk_Individualist.csv
                   phase3_optimized_sfm_params_Relaxed_Ped.csv
    in csv_dir.
    """
    import os
    
    # Brisk_Individualist
    try:
        path = os.path.join(csv_dir, 'phase3_optimized_sfm_params_Brisk_Individualist.csv')
        params_df = pd.read_csv(path, index_col=0)
        opt_vals = params_df.iloc[0].values.astype(float)
        v0, tau, A, B, lam_base, phi_fov = opt_vals[0:6]
        kappa, k_group, r_group, theta_gaze, w_att, fov_att = opt_vals[6:12]
        MOOD_PARAMETERS[PedestrianMood.Brisk_Individualist] = {
            'v0': v0,
            'tau': tau,
            'A_ped': A,
            'B_ped': B,
            'lam_base': lam_base,
            'phi_fov': phi_fov,
            'kappa': kappa,
            'k_group': k_group,
            'r_group': r_group,
            'theta_gaze': theta_gaze,
            'w_att': w_att,
            'fov_att': fov_att,
            'speed_factor': 1.0,
            'direction_variance': 0.0,
            'personal_space': 1.0,
            'reactivity': 0.5,
            'yielding': 0.5,
            'description': 'Calibrated regime 0_1_1_0 (Brisk Individualist)'
        }
    except FileNotFoundError:
        print("Warning: Brisk_Individualist CSV not found, using default parameters.")

    # Relaxed_Ped
    try:
        path = os.path.join(csv_dir, 'phase3_optimized_sfm_params_Relaxed_Ped.csv')
        params_df = pd.read_csv(path, index_col=0)
        opt_vals = params_df.iloc[0].values.astype(float)
        v0, tau, A, B, lam_base, phi_fov = opt_vals[0:6]
        kappa, k_group, r_group, theta_gaze, w_att, fov_att = opt_vals[6:12]
        MOOD_PARAMETERS[PedestrianMood.Relaxed_Ped] = {
            'v0': v0,
            'tau': tau,
            'A_ped': A,
            'B_ped': B,
            'lam_base': lam_base,
            'phi_fov': phi_fov,
            'kappa': kappa,
            'k_group': k_group,
            'r_group': r_group,
            'theta_gaze': theta_gaze,
            'w_att': w_att,
            'fov_att': fov_att,
            'speed_factor': 1.0,
            'direction_variance': 0.0,
            'personal_space': 1.0,
            'reactivity': 0.5,
            'yielding': 0.5,
            'description': 'Calibrated regime 0_1_1_0 (Relaxed Pedestrian)'
        }
    except FileNotFoundError:
        print("Warning: Relaxed_Ped CSV not found, using default parameters.")



CUSTOM_MOODS: dict = {}
def register_mood(csv_path: str, mood_name: str, verbose: bool = True):
    import pandas as pd
    import numpy as np

    params_df = pd.read_csv(csv_path)
    # If the first column is an unnamed index, drop it
    if 'Unnamed: 0' in params_df.columns:
        params_df = params_df.drop(columns=['Unnamed: 0'])

    if len(params_df.columns) < 12:
        raise ValueError(f"Expected at least 12 parameter columns, got {len(params_df.columns)}. "
                         f"Columns: {list(params_df.columns)}")

    opt_vals = params_df.iloc[0].values.astype(float)
    # In case there are extra columns, take the first 12
    opt_vals = opt_vals[:12]

    v0, tau, A, B, lam_base, phi_fov = opt_vals[0:6]
    kappa, k_group, r_group, theta_gaze, w_att, fov_att = opt_vals[6:12]

    mood_params = {
        'v0': v0,
        'tau': tau,
        'A_ped': A,
        'B_ped': B,
        'lam_base': lam_base,
        'phi_fov': phi_fov,
        'kappa': kappa,
        'k_group': k_group,
        'r_group': r_group,
        'theta_gaze': theta_gaze,
        'w_att': w_att,
        'fov_att': fov_att,
        'speed_factor': 1.0,
        'direction_variance': 0.0,
        'personal_space': 1.0,
        'reactivity': 0.5,
        'yielding': 0.5,
        'description': f'Calibrated {mood_name}'
    }
    CUSTOM_MOODS[mood_name] = mood_params

    if verbose:
        print(f"Registered custom mood '{mood_name}':")
        print(f"  v0={v0:.3f}, tau={tau:.3f}, A={A:.2f}, B={B:.2f}, lam={lam_base:.2f}, phi={np.rad2deg(phi_fov):.0f}°")
        print(f"  kappa={kappa:.2f}, kg={k_group:.2f}, rg={r_group:.2f}, tg={np.rad2deg(theta_gaze):.0f}°, w_att={w_att:.2f}, fov_att={np.rad2deg(fov_att):.0f}°")
    return mood_params