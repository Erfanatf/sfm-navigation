"""Pipeline configuration for the behavior profiling pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PipelineConfig:
    """Configuration for the pipeline stages."""

    # ---------- Execution control ----------
    stages: str = "all"   # comma-separated, e.g., "stage0,stage4" or "all"
    load_run_id: Optional[str] = None   # if set, load data from this previous run

    # ---------- Performance ----------
    max_windows_total: Optional[int] = None   # set to e.g., 2000 to limit windows
    progress_interval: int = 50               # print progress every N windows

    # ---------- Stage 0 ----------
    atc_csv_path: str = "/home/erfanatf/Documents/notebooks/content/drive/MyDrive/ATC_data/atc-20121114.csv"
    bin_width_sec: float = 300.0
    n_top_bins: int = 10

    # ---------- Stage 1 ----------
    window_len_sec: float = 8.0
    window_stride_sec: float = 2.0
    min_points_per_window: int = 20

    # ---------- Stage 2 ----------
    # (feature extraction uses the same parameters as the notebook)
    min_path_length: float = 1.0

    # ---------- Stage 3 ----------
    min_windows_for_regime: int = 10
    gmm_covariance_type: str = "full"
    # Which clustering method to use for domain labels: "kmeans" or "gmm"
    cluster_method: str = "kmeans"

    # ---------- Stage 4 ----------
    max_windows_per_regime: int = 10
    min_displacement: float = 1.0
    dt_sim: float = 0.2
    optimizer_maxiter: int = 5          # DE global search iterations (reduced)
    optimizer_popsize: int = 10         # DE population size (reduced)
    optimizer_tol: float = 0.1          # DE tolerance (relaxed)
    optimizer_local_maxiter: int = 50   # L‑BFGS‑B local refinement iterations
    
    # For now only calibrate the first regime (Brisk_Individualist) as a test
    calibrate_regime: str = "auto"

    # ---------- Output ----------
    output_dir: str = "pipeline_results"
    log_level: str = "INFO"
    verbosity: int = 1  # 0 = quiet, 1 = normal, 2 = debug