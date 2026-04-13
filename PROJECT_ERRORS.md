# Project Errors And Risks

## Open Items

- The physical board scale currently assumes `1.0 mm` center spacing. If the real effective spacing differs, extrinsic translation scale must be recalibrated.
- The light-plane validation image currently needs manual ROI and segment ranges; if the acquisition setup changes, the config must be updated instead of reusing the current defaults blindly.
- Because ROI is currently manual, the new pipeline/profile plots should be checked whenever validation accuracy moves unexpectedly; the quantity-block result is sensitive to segment placement and ROI drift.
- The quantity-block `step` segment is also sensitive. The older `[2500, 3600]` window was wide enough to leak into the shoulder regions and inflated the left/right mismatch; the currently audited default is `left=[400, 1500], step=[2700, 3500], right=[4500, 5000]`.
- Even with the re-audited center window, a single sub-micron averaged result is not enough for a very-high-confidence system claim. The confidence audit currently shows a step-window conservative-error envelope of `p05=1.307 um`, `median=4.730 um`, `p95=11.154 um`.
- The `Laser2` ROI is especially sensitive. During audit, moving it toward a visually more centered box drove the quantity-block error into the tens to hundreds of microns. The currently audited stable box is `ROI=[1750, 1200, 1900, 350]`.
- The `Cam_pos -> Laser` pairing is a hard dependency. Using the wrong pair can still produce a visually plausible light plane while destroying the step-height validation result.
- The `global_centroid` variant produced a false-good `0.053 um` averaged result because the left and right edge errors cancelled each other. It should not be used as the default validation setting for this dataset.
- A tiny averaged step-height error can still be false comfort if the left and right edge heights disagree. Always check `edge_consistency_um` and `conservative_absolute_error_um` alongside the averaged result.
- The project owner has confirmed that `Laser1/Laser2` correspond to `Cam_pos1/Cam_pos2`. Any alternative pairing may still be useful for diagnosis, but it must not replace the default workflow.
- The previous camera distortion model was too flexible for downstream use: fixing only `k3` kept the pixel RMS similar but pushed the `Cam_pos1/Cam_pos2` quantity-block result toward `2.089 mm`. Fixing both `k2` and `k3` gives a more stable downstream metric on the confirmed mapping.

## Current Calibration Baseline

- The camera calibration stage uses OpenCV circle-grid detection with blob area constraints tuned for the `Campos-8` dataset.
- Reprojection error should be checked after every future image-set expansion instead of reusing old parameters blindly.
- If the validation error jumps back, first verify the forced `Cam_pos1/Cam_pos2 -> Laser1/Laser2` mapping, the world-frame reference image, whether camera calibration drifted away from `fix_k2 + fix_k3`, and whether `stripe_extraction_params` drifted away from `steger_like + median+gaussian + threshold_ratio=0.33`.
- For this dataset, treat the validation segment ranges as part of the calibrated recipe, not a cosmetic plotting choice; widening the `step` interval can make the averaged result look acceptable while the conservative error gets much worse.
- Use `python_workspace/validation_confidence_audit.py` before making any final accuracy claim; it is the current safeguard against overfitting the quantity-block result to one manually selected window.
- Even when the averaged error stays tiny, treat `edge_consistency_um` and `conservative_absolute_error_um` as the guardrails against false-good results.
- Use the generated plots as hard checks instead of trusting scalar metrics alone:
  - `output/camera_calibration/reprojection_errors.png`
  - `output/camera_calibration/image_point_coverage.png`
  - `output/camera_calibration/camera_poses_3d.png`
  - `output/light_plane_calibration/light_plane_residuals.png`
  - `output/light_plane_calibration/*_pipeline.png`
  - `output/light_plane_calibration/Pic_20260320142001841_profile.png`
