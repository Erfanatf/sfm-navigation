"""CLI entry point for behavior profiling pipeline."""

import argparse
from ..behavior_pipeline.config import PipelineConfig
from ..behavior_pipeline.pipeline_runner import PipelineRunner

def main():
    parser = argparse.ArgumentParser(description="Run the behavior profiling pipeline.")
    parser.add_argument('--data-source', type=str, default='/home/erfanatf/Documents/notebooks/content/drive/MyDrive/ATC_data/atc-20121114.csv',
                        help='Path to ATC CSV file (or vision recording file)')
    parser.add_argument('--output-dir', type=str, default='pipeline_results',
                        help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity')
    parser.add_argument('--n-top-bins', type=int, default=3,
                        help='Number of crowded bins to use')
    parser.add_argument('--calibrate-regime', type=str, default="auto",
                        help='Regime code to calibrate (e.g. "1_1_1_0") or "auto" for the most frequent active regime')
    parser.add_argument('--stages', type=str, default='all',
                        help='Stages to run, e.g. "stage0,stage4" or "all"')
    parser.add_argument('--load-run-id', type=str, default=None,
                        help='Load data from this existing run ID (instead of current run)')
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Maximum number of windows to process')
    parser.add_argument('--dt-sim', type=float, default=None,
                        help='Simulation timestep for calibration (default from config)')
    args = parser.parse_args()

    config = PipelineConfig()
    config.atc_csv_path = args.data_source
    config.output_dir = args.output_dir
    config.log_level = args.log_level
    config.n_top_bins = args.n_top_bins
    config.calibrate_regime = args.calibrate_regime
    config.stages = args.stages
    config.load_run_id = args.load_run_id
    config.max_windows_total = args.max_windows
    if args.max_windows is not None:
        config.max_windows_total = args.max_windows
    if args.dt_sim is not None:
        config.dt_sim = args.dt_sim
        
    runner = PipelineRunner(config)
    runner.run()

if __name__ == '__main__':
    main()