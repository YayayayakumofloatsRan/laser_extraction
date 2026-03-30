# Project Progress

## 2026-03-30

- Added `python_workspace/camera_calibration.py` for symmetric circle-grid camera calibration.
- Confirmed the `Campos-8` dataset can be detected as a `9 x 9` symmetric circle board with a `1.0 mm` center spacing assumption.
- Standardized calibration outputs under `output/camera_calibration`.
- Latest full calibration baseline: `26/26` valid detections, image size `5496 x 3672`, overall RMS `2.442742492 px`.
- Added `python_workspace/light_plane_calibration.py` and `python_workspace/light_plane_calibration_config.json` for light-plane fitting and step-height validation.
- Refactored the validation stage to follow the lecture definition of world coordinates: the first paired reference image defines the world frame and height is evaluated along world `Z`.
- Current default pairing is `Cam_pos15 -> Laser1`, `Cam_pos17 -> Laser2`; the earlier `Cam_pos1/Cam_pos2` assumption caused a false large validation error.
- Switched the default stripe center extraction from the earlier local `peak_window_centroid` approximation to the lecture-aligned `global_centroid` workflow.
- Current default stripe extraction parameters are `filter_mode=median+gaussian`, `threshold_ratio=0.25`, `extraction_method=global_centroid`.
- Latest light-plane baseline: fitted plane RMSE `0.051787291 mm` (`51.787 um`) from `4000` reconstructed board points.
- Latest step validation baseline: measured `1.799946728 mm` versus nominal `1.800000000 mm`, absolute error `0.000053272 mm` (`0.053 um`).

## Current Focus

- Preserve the now-correct pairing, world-frame validation logic, and lecture-aligned gray-centroid extraction settings.
- Keep validating physical board metadata before locking the final production scale.
