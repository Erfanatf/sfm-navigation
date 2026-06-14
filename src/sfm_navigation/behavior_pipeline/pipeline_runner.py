import os
import sys
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .reporting import PipelineLogger
from .data_provider import ATCDataSource
from .stages.stage0_load_bin import run_stage0
from .stages.stage1_windowing import run_stage1
from .stages.stage2_features import run_stage2
from .stages.stage3_clustering import run_stage3
from .stages.stage4_calibrate import run_stage4

class PipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config.output_dir, config.log_level)
        self.run_dir = os.path.join(config.output_dir, self.logger.run_id)
        self.data_dir = os.path.join(self.run_dir, "data")
        if config.load_run_id:
            self.load_data_dir = os.path.join(config.output_dir, config.load_run_id, "data")
        else:
            self.load_data_dir = self.data_dir

    def _stage_should_run(self, stage_name: str) -> bool:
        if self.config.stages == "all":
            return True
        requested = [s.strip() for s in self.config.stages.split(",")]
        return stage_name in requested

    def _load_stage0_data(self):
        """Load combined bin data from saved CSVs."""
        stage0_dir = os.path.join(self.load_data_dir, "stage0")
        dfs = []
        for fname in sorted(os.listdir(stage0_dir)):
            if fname.startswith("bin_") and fname.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(stage0_dir, fname)))
        if not dfs:
            raise FileNotFoundError(f"No bin CSV files found in {stage0_dir}")
        df_all = pd.concat(dfs, ignore_index=True)
        bin_stats = pd.read_csv(os.path.join(stage0_dir, "bin_stats.csv"))
        return df_all, bin_stats

    def _load_stage1_data(self):
        """Load df_windows."""
        path = os.path.join(self.load_data_dir, "stage1/df_windows.csv")
        return pd.read_csv(path)

    def _load_raw_data(self):
        """Reload raw ATC data."""
        data_source = ATCDataSource(self.config.atc_csv_path)
        return data_source.load_raw_data()

    def _load_stage2_features(self):
        """Load the four feature DataFrames."""
        stage2_dir = os.path.join(self.load_data_dir, "stage2")
        f11 = pd.read_csv(os.path.join(stage2_dir, "features_kinematic.csv"))
        f12 = pd.read_csv(os.path.join(stage2_dir, "features_path_quality.csv"))
        f13 = pd.read_csv(os.path.join(stage2_dir, "features_safety.csv"))
        f14 = pd.read_csv(os.path.join(stage2_dir, "features_social_attention.csv"))
        return f11, f12, f13, f14

    def _load_stage3_labels(self):
        """Load regime labels."""
        path = os.path.join(self.load_data_dir, "stage3/regime_labels.csv")
        return pd.read_csv(path)

    def run(self):
        try:
            # ---- Stage 0 ----
            if self._stage_should_run("stage0"):
                self.logger.info("Starting pipeline...")
                df_all, bin_stats = run_stage0(self.config, self.logger)
            else:
                self.logger.info("Skipping Stage 0, loading saved data...")
                df_all, bin_stats = self._load_stage0_data()

            # ---- Stage 1 ----
            if self._stage_should_run("stage1"):
                df_windows = run_stage1(self.config, self.logger, df_all)
            else:
                self.logger.info("Skipping Stage 1, loading saved windows...")
                df_windows = self._load_stage1_data()

            # Optional subsampling
            if self.config.max_windows_total and len(df_windows) > self.config.max_windows_total:
                self.logger.info(f"Subsampling windows from {len(df_windows)} to {self.config.max_windows_total}")
                window_ids = df_windows['window_id'].unique()
                sampled_ids = np.random.choice(window_ids, self.config.max_windows_total, replace=False)
                df_windows = df_windows[df_windows['window_id'].isin(sampled_ids)]

            # Raw data for stage2 and stage4
            if self._stage_should_run("stage2") or self._stage_should_run("stage4"):
                df_raw = self._load_raw_data()
            else:
                df_raw = None   # won't be used

            
            # ---- Stage 2 ----
            if self._stage_should_run("stage2"):
                f11, f12, f13, f14 = run_stage2(self.config, self.logger, df_windows, df_raw)
            else:
                self.logger.info("Skipping Stage 2, loading saved features...")
                f11, f12, f13, f14 = self._load_stage2_features()

            # ---- Stage 3 ----
            if self._stage_should_run("stage3"):
                labels_df, regime_means = run_stage3(self.config, self.logger, f11, f12, f13, f14)
            else:
                self.logger.info("Skipping Stage 3, loading saved regime labels...")
                labels_df = self._load_stage3_labels()
                regime_means = None  # not needed for stage4

            # ---- Stage 4 ----
            if self._stage_should_run("stage4"):
                run_stage4(self.config, self.logger, df_windows, df_raw, labels_df, f11, f12, f13, f14)
            else:
                self.logger.info("Skipping Stage 4.")

            self.logger.final_summary()
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.final_summary()
            raise