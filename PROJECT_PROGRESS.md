# Project Progress

## 2026-03-30

- Added `python_workspace/camera_calibration.py` for symmetric circle-grid camera calibration.
- Confirmed the `Campos-8` dataset can be detected as a `9 x 9` symmetric circle board with a `1.0 mm` center spacing assumption.
- Standardized calibration outputs under `output/camera_calibration`.
- Latest full calibration baseline: `26/26` valid detections, image size `5496 x 3672`, overall RMS `2.442742492 px`.

## Current Focus

- Use the saved camera intrinsics and per-view extrinsics as the input for the next light-plane calibration stage.
- Keep validating physical board metadata before locking the final production scale.
