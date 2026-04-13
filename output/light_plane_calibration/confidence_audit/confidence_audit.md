# Validation Confidence Audit

## Configured Result
- Config path: D:\laser_extraction\python_workspace\light_plane_calibration_config.json
- Left segment: [400, 1500]
- Step segment: [2700, 3500]
- Right segment: [4500, 5000]
- Measured height: 1.799493965 mm
- Absolute error: 0.506 um
- Conservative error: 1.514 um
- Edge gap: 2.016 um

## Step-Window Perturbation Audit
- Grid: configured step segment +/-100 px with 25 px spacing
- Samples: 81
- Absolute error envelope: p05=0.204 um, median=0.984 um, p95=2.437 um
- Conservative error envelope: p05=1.307 um, median=4.730 um, p95=11.154 um
- Edge-gap envelope: p05=0.660 um, median=7.361 um, p95=18.069 um
- Best conservative window in this grid: step=[2600, 3575], conservative=0.527 um, absolute=0.204 um
- Worst conservative window in this grid: step=[2800, 3600], conservative=14.551 um, absolute=2.842 um

## Extraction-Method Sensitivity
- global_centroid: absolute=262.793 um, conservative=318.293 um, edge_gap=111.000 um, plane_rmse=31.878 um
- peak_window_centroid: absolute=253.105 um, conservative=263.873 um, edge_gap=21.537 um, plane_rmse=32.093 um
- gaussian_fit: absolute=0.627 um, conservative=1.066 um, edge_gap=0.878 um, plane_rmse=36.438 um
- steger_like: absolute=0.506 um, conservative=1.514 um, edge_gap=2.016 um, plane_rmse=36.545 um

## Interpretation
- The configured result may look very strong on this single window, but step-window perturbation shows the reported value is not completely window-invariant.
- Therefore, a single sub-micron average error must not be treated as a very-high-confidence system claim on its own.
- The scientifically safer claim is that this dataset supports low-micron validation when using the audited recipe and when conservative error / edge gap stay small at the same time.
- Detailed tables: D:\laser_extraction\output\light_plane_calibration\confidence_audit\validation_segment_sensitivity.csv and D:\laser_extraction\output\light_plane_calibration\confidence_audit\extraction_method_sensitivity.csv
