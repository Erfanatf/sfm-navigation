"""Stage 0: Load ATC data, split into bins, select top crowded bins."""

import os
import pandas as pd
from ...crowd_analysis.binning import bin_data, select_bin_data
from ...behavior_pipeline.config import PipelineConfig
from ...behavior_pipeline.reporting import PipelineLogger
from ...behavior_pipeline.data_provider import ATCDataSource


def run_stage0(config: PipelineConfig, logger: PipelineLogger, data_source=None):
    logger.info("=== Stage 0: Data Loading & Binning ===")
    if data_source is None:
        data_source = ATCDataSource(config.atc_csv_path)

    df = data_source.load_raw_data()
    logger.info(f"Loaded {len(df)} rows, {df['agent_id'].nunique()} agents.")

    bin_stats_df = bin_data(df, bin_width_sec=config.bin_width_sec)
    logger.info(f"Computed {len(bin_stats_df)} time bins.")

    # Save bin statistics
    os.makedirs(
        os.path.join(logger.output_dir, logger.run_id, "data/stage0"), exist_ok=True
    )
    bin_stats_df.to_csv(
        os.path.join(logger.output_dir, logger.run_id, "data/stage0/bin_stats.csv"),
        index=False,
    )

    # Report
    report = "Top 10 crowded bins:\n" + bin_stats_df.head(10).to_string() + "\n"
    logger.write_stage_report("stage0", report)

    # Select top N bins and return their dataframes
    selected_bins = bin_stats_df.head(config.n_top_bins)
    bin_dfs = []
    for idx, row in selected_bins.iterrows():
        bin_id = row["bin_id"]
        t_start = row["time_start"]
        t_end = row["time_end"]
        df_bin = df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)].copy()
        df_bin["bin_id"] = bin_id
        # Save
        df_bin.to_csv(
            os.path.join(
                logger.output_dir, logger.run_id, "data/stage0", f"bin_{bin_id}.csv"
            )
        )
        bin_dfs.append(df_bin)
        logger.info(
            f"Bin {bin_id}: {len(df_bin)} rows, {df_bin['agent_id'].nunique()} agents."
        )

    # Combine all selected bins into one DataFrame for subsequent stages
    df_all = pd.concat(bin_dfs, ignore_index=True)
    logger.info(f"Total rows across top {config.n_top_bins} bins: {len(df_all)}")
    return df_all, bin_stats_df
