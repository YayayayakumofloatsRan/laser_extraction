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
- Audited the step-height metric and found the `0.053 um` averaged error is not trustworthy by itself because the left/right edge results are `1.753223 mm` and `1.846671 mm`, with a `93.448 um` consistency gap.
- Current reporting now keeps the averaged error but also exposes left-edge error, right-edge error, left/right consistency gap, and a conservative error metric so visually good cancellation cannot be mistaken for true device accuracy.

## Current Focus

- Preserve the now-correct pairing, world-frame validation logic, and lecture-aligned gray-centroid extraction settings.
- Replace single-number validation claims with consistency-aware metrics and then revisit the step-height fitting procedure against the lecture.
- Keep validating physical board metadata before locking the final production scale.
