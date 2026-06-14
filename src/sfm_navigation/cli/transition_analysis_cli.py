"""CLI for transition matrix analysis."""

import argparse
from ..behavior_pipeline.transition_analysis import analyze_transitions

def main():
    parser = argparse.ArgumentParser(description="Analyze regime transitions from a pipeline run.")
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to the pipeline run folder (e.g., pipeline_results/20260522_191420)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output (default: run_dir/transition_analysis)')
    args = parser.parse_args()
    analyze_transitions(args.run_dir, args.output_dir)

if __name__ == '__main__':
    main()