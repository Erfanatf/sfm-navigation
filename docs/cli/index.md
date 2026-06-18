# CLI Command Map

This folder documents every CLI entry point registered in [pyproject.toml](../../pyproject.toml). The entries are ordered by the current usage priority you requested, with the most-used commands first.

## Priority Order

1. [robot-demo.md](robot-demo.md)
2. [control-signal-comparison.md](control-signal-comparison.md)
3. [animate-history.md](animate-history.md)
4. [compare-animation.md](compare-animation.md)
5. [compare-control-signals.md](compare-control-signals.md)
6. [metric-reports.md](metric-reports.md)
7. [crowd-robot-demo.md](crowd-robot-demo.md)
8. [demo.md](demo.md)
9. [crowd-analysis.md](crowd-analysis.md)
10. [pipeline-cli.md](pipeline-cli.md)
11. [transition-analysis.md](transition-analysis.md)
12. [run-simulation.md](run-simulation.md)
13. [plot-control-signals.md](plot-control-signals.md)

## What Each Document Covers

Each page includes:

- the command and module it maps to
- the functions, classes, and parameters it uses
- the execution pipeline from inputs to outputs
- the scientific and control theory used by that command
- the source files that implement each stage of the pipeline

## Shared Theory References

- [controllers/](../../src/sfm_navigation/controllers/) for DWA, MPC, MPPI, NMPC, and CBF methods
- [simulation/engine.py](../../src/sfm_navigation/simulation/engine.py) for the main simulation loop
- [spawner/spawner.py](../../src/sfm_navigation/spawner/spawner.py) for pedestrian generation and obstacle placement
- [data/moods.py](../../src/sfm_navigation/data/moods.py) for mood parameters and calibrated mood loading
- [behavior_pipeline/](../../src/sfm_navigation/behavior_pipeline/) for the calibration pipeline and transition analysis
- [metrics/metrics.py](../../src/sfm_navigation/metrics/metrics.py) for performance metrics
- [crowd_analysis/](../../src/sfm_navigation/crowd_analysis/) for raw crowd-data binning and visualization
