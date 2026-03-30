# Project Progress

## 2026-03-30

- Added `python_workspace/camera_calibration.py` for symmetric circle-grid camera calibration.
- Confirmed the `Campos-8` dataset can be detected as a `9 x 9` symmetric circle board with a `1.0 mm` center spacing assumption.
- Standardized calibration outputs under `output/camera_calibration`.
- Updated the camera calibration default distortion model to fix `k2` and `k3`; current baseline is `26/26` valid detections, image size `5496 x 3672`, overall RMS `2.442910862 px`.
- Added `python_workspace/light_plane_calibration.py` and `python_workspace/light_plane_calibration_config.json` for light-plane fitting and step-height validation.
- Refactored the validation stage to follow the lecture definition of world coordinates: the first paired reference image defines the world frame and height is evaluated along world `Z`.
- Restored the experiment-recorded pairing to `Cam_pos1 -> Laser1`, `Cam_pos2 -> Laser2`.
- Audited the false-good `0.053 um` result and confirmed it came from an over-optimistic extraction/model combination plus left/right cancellation, not from a credible physical improvement.
- Verified that the user-confirmed `Cam_pos1/Cam_pos2 -> Laser1/Laser2` mapping can reach the lecture range after tightening the camera distortion model and switching stripe extraction to `steger_like`.
- The light-plane calibration script now also outputs a lecture-style 3D light-plane plot in camera coordinates.
- Added process visualizations for auditability: camera reprojection errors, camera image-point coverage, camera pose distribution, per-laser extraction pipelines, light-plane residuals, and quantity-block pipeline/profile plots.
- Current light-plane baseline under the forced `Cam_pos1/Cam_pos2` mapping: plane RMSE `0.036225838 mm` (`36.226 um`).
- Current quantity-block validation baseline from `Pic_20260320142001841.png`: measured `1.797759644 mm`, averaged error `2.240 um`, conservative error `6.570 um`, left/right consistency gap `8.659 um`.

## Current Focus

- Preserve the forced `Cam_pos1/Cam_pos2` pairing, the simplified camera distortion model, and the reused `laser_extraction.py` subpixel stripe extraction path.
- Use the new process figures to decide whether the remaining `2.240 um` / `6.570 um` gap comes from residual stripe-center bias, manual ROI choice, or the lecture's exact height definition.
- Keep validating physical board metadata before locking the final production scale.
