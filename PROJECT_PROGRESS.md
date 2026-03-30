# Project Progress

## 2026-03-30

- Added `python_workspace/camera_calibration.py` for symmetric circle-grid camera calibration.
- Confirmed the `Campos-8` dataset can be detected as a `9 x 9` symmetric circle board with a `1.0 mm` center spacing assumption.
- Standardized calibration outputs under `output/camera_calibration`.
- Latest full calibration baseline: `26/26` valid detections, image size `5496 x 3672`, overall RMS `2.442742492 px`.
- Added `python_workspace/light_plane_calibration.py` and `python_workspace/light_plane_calibration_config.json` for light-plane fitting and step-height validation using `Cam_pos1/Cam_pos2 -> Laser1/Laser2`.
- Latest light-plane baseline: fitted plane RMSE `0.031383621 mm` (`31.384 um`) from `4000` reconstructed board points.
- Latest step validation baseline: measured `2.122163815 mm` versus nominal `1.800000000 mm`, absolute error `0.322163815 mm` (`322.164 um`).

## Current Focus

- Explain and close the current `322 um` step-height validation gap before treating the light-plane model as production-ready.
- Keep validating physical board metadata before locking the final production scale.
