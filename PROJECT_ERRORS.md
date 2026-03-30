# Project Errors And Risks

## Open Items

- The physical board scale currently assumes `1.0 mm` center spacing. If the real effective spacing differs, extrinsic translation scale must be recalibrated.
- The light-plane validation image currently needs manual ROI and segment ranges; if the acquisition setup changes, the config must be updated instead of reusing the current defaults blindly.
- Because ROI is currently manual, the new pipeline/profile plots should be checked whenever validation accuracy moves unexpectedly; the quantity-block result is sensitive to segment placement and ROI drift.
- The `Cam_pos -> Laser` pairing is a hard dependency. Using the wrong pair can still produce a visually plausible light plane while destroying the step-height validation result.
- The `global_centroid` variant produced a false-good `0.053 um` averaged result because the left and right edge errors cancelled each other. It should not be used as the default validation setting for this dataset.
- A tiny averaged step-height error can still be false comfort if the left and right edge heights disagree. Always check `edge_consistency_um` and `conservative_absolute_error_um` alongside the averaged result.
- The project owner has confirmed that `Laser1/Laser2` correspond to `Cam_pos1/Cam_pos2`. Any alternative pairing may still be useful for diagnosis, but it must not replace the default workflow.
- The previous camera distortion model was too flexible for downstream use: fixing only `k3` kept the pixel RMS similar but pushed the `Cam_pos1/Cam_pos2` quantity-block result toward `2.089 mm`. Fixing both `k2` and `k3` gives a more stable downstream metric on the confirmed mapping.

## Current Calibration Baseline

- The camera calibration stage uses OpenCV circle-grid detection with blob area constraints tuned for the `Campos-8` dataset.
- Reprojection error should be checked after every future image-set expansion instead of reusing old parameters blindly.
- If the validation error jumps back, first verify the forced `Cam_pos1/Cam_pos2 -> Laser1/Laser2` mapping, the world-frame reference image, whether camera calibration drifted away from `fix_k2 + fix_k3`, and whether `stripe_extraction_params` drifted away from `steger_like + median+gaussian + threshold_ratio=0.33`.
- Even when the averaged error stays tiny, treat `edge_consistency_um` and `conservative_absolute_error_um` as the guardrails against false-good results.
- Use the generated plots as hard checks instead of trusting scalar metrics alone:
  - `output/camera_calibration/reprojection_errors.png`
  - `output/camera_calibration/image_point_coverage.png`
  - `output/camera_calibration/camera_poses_3d.png`
  - `output/light_plane_calibration/light_plane_residuals.png`
  - `output/light_plane_calibration/*_pipeline.png`
  - `output/light_plane_calibration/Pic_20260320142001841_profile.png`
