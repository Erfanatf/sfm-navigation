# Metric Reports

**Command:** `sfm-metrics`

**Module:** [src/sfm_navigation/cli/metrics_report.py](../../src/sfm_navigation/cli/metrics_report.py)

## Purpose

This command computes a broad set of performance metrics from one or more simulation history CSV files and generates both a CSV summary and a Plotly comparison dashboard.

## Scripted Constants and Functions

- `METRIC_DIRECTION`: indicates whether higher or lower is better for each metric
- `main()`

## Inputs

- `--csvs`: one or more simulation history CSV files
- `--output`: output base name

## Execution Pipeline

1. Load each history CSV.
2. Instantiate [ControllerPerformance](../../src/sfm_navigation/metrics/metrics.py) for each run.
3. Compute all metrics by category.
4. Build a multi-index pandas table and write the summary CSV.
5. Sort controllers per metric using the directionality map.
6. Render a multi-panel Plotly bar chart dashboard.
7. Save the HTML report and open it in the browser.

## Theory and Implementation Links

The metrics span several scientific ideas:

- navigation efficiency: path length, time to goal, speed statistics
- safety: collisions, distance-to-obstacle measures, time-to-collision proxies
- social comfort: SII, RMI, SGI style measures
- smoothness: acceleration and jerk statistics
- path quality: directness, tortuosity, curvature, legibility
- computational load: compute time and real-time factor

Primary implementation:

- [metrics/metrics.py](../../src/sfm_navigation/metrics/metrics.py)

## Outputs

- `controller_metrics.csv`
- `controller_metrics.html`

## Best Use

Use this after you have a set of controller runs and want to compare them quantitatively rather than visually.
