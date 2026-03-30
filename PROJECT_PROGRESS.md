# Project Progress

## 2026-03-30

- Added `python_workspace/camera_calibration.py` for symmetric circle-grid camera calibration.
- Confirmed the `Campos-8` dataset can be detected as a `9 x 9` symmetric circle board with a `1.0 mm` center spacing assumption.
- Standardized calibration outputs under `output/camera_calibration`.
- Latest full calibration baseline: `26/26` valid detections, image size `5496 x 3672`, overall RMS `2.442742492 px`.
- Added `python_workspace/light_plane_calibration.py` and `python_workspace/light_plane_calibration_config.json` for light-plane fitting and step-height validation.
- Refactored the validation stage to follow the lecture definition of world coordinates: the first paired reference image defines the world frame and height is evaluated along world `Z`.
- Current default pairing is `Cam_pos15 -> Laser1`, `Cam_pos17 -> Laser2`; the earlier `Cam_pos1/Cam_pos2` assumption caused a false large validation error.
- Latest light-plane baseline: fitted plane RMSE `0.052062624 mm` (`52.063 um`) from `4000` reconstructed board points.
- Latest step validation baseline: measured `1.795916203 mm` versus nominal `1.800000000 mm`, absolute error `0.004083797 mm` (`4.084 um`).

## Current Focus

- Preserve the now-correct pairing and world-frame validation logic, and avoid regressing back to the old `322 um` error path.
- Keep validating physical board metadata before locking the final production scale.
