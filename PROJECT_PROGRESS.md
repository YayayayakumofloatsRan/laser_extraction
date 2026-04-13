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
- Audited the `Laser2` ROI. Vertical or left-shifted "more centered" boxes degraded the quantity-block result badly; the stable fix was to keep `x=1750, y=1200` and trim the width from `2000` to `1900` so the ROI stays more conservatively inside the calibration-board stripe region.
- Current light-plane baseline under the forced `Cam_pos1/Cam_pos2` mapping: plane RMSE `0.036545409 mm` (`36.545 um`).
- Re-audited the quantity-block validation segments and tightened the default `step` window from `[2500, 3600]` to `[2700, 3500]` so the line fit stays inside the flatter center region instead of bleeding into the transition shoulders.
- Current quantity-block validation baseline from `Pic_20260320142001841.png`: measured `1.799493965 mm`, averaged error `0.506 um`, conservative error `1.514 um`, left/right consistency gap `2.016 um`.

## Current Focus

- Preserve the forced `Cam_pos1/Cam_pos2` pairing, the simplified camera distortion model, and the reused `laser_extraction.py` subpixel stripe extraction path.
- Preserve the audited `Laser2 ROI=[1750, 1200, 1900, 350]`; larger left/right shifts looked visually tempting but were numerically worse.
- Preserve the re-audited validation segments `left=[400, 1500], step=[2700, 3500], right=[4500, 5000]`; the wider step window looked harmless visually but pulled the fit toward the step shoulders and inflated the conservative error.
- Use the new process figures to decide whether any remaining sub-micron / low-micron gap comes from residual stripe-center bias, manual ROI choice, or the lecture's exact height definition.
- Keep validating physical board metadata before locking the final production scale.
