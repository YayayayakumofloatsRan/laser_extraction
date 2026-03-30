# Project Progress

## 2026-03-30

- Added `python_workspace/camera_calibration.py` for symmetric circle-grid camera calibration.
- Confirmed the `Campos-8` dataset can be detected as a `9 x 9` symmetric circle board with a `1.0 mm` center spacing assumption.
- Standardized calibration outputs under `output/camera_calibration`.
- Latest full calibration baseline: `26/26` valid detections, image size `5496 x 3672`, overall RMS `2.442742492 px`.
- Added `python_workspace/light_plane_calibration.py` and `python_workspace/light_plane_calibration_config.json` for light-plane fitting and step-height validation.
- Refactored the validation stage to follow the lecture definition of world coordinates: the first paired reference image defines the world frame and height is evaluated along world `Z`.
- Current default pairing is `Cam_pos15 -> Laser1`, `Cam_pos17 -> Laser2`; the earlier `Cam_pos1/Cam_pos2` assumption caused a false large validation error.
- Audited the false-good `0.053 um` result and confirmed it came from `global_centroid` plus left/right cancellation, not from a credible physical improvement.
- Restored the lecture-style local gray-centroid workflow and tuned it to `filter_mode=median+gaussian`, `threshold_ratio=0.33`, `peak_window_half_height=27`, `extraction_method=peak_window_centroid`.
- Latest light-plane baseline: fitted plane RMSE stays at the same tens-of-microns level, while the step validation returns to the lecture case range.
- Latest step validation baseline: measured `1.803690833 mm` versus nominal `1.800000000 mm`, averaged error `3.691 um`, conservative error `8.490 um`, left/right consistency gap `9.599 um`.

## Current Focus

- Preserve the now-correct pairing, world-frame validation logic, and tuned local gray-centroid extraction settings.
- Keep checking whether the remaining gap between averaged error and conservative error comes from the lecture's height definition or from still-imperfect stripe center extraction.
- Keep validating physical board metadata before locking the final production scale.
