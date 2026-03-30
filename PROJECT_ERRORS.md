# Project Errors And Risks

## Open Items

- The physical board scale currently assumes `1.0 mm` center spacing. If the real effective spacing differs, extrinsic translation scale must be recalibrated.
- The light-plane validation image currently needs manual ROI and segment ranges; if the acquisition setup changes, the config must be updated instead of reusing the current defaults blindly.
- The `Cam_pos -> Laser` pairing is a hard dependency. Using the wrong pair can still produce a visually plausible light plane while destroying the step-height validation result.
- Local peak-window stripe extraction can look visually stable while introducing a measurable micron-level bias; do not switch away from the current lecture-aligned gray-centroid default without re-running validation.
- A tiny averaged step-height error can still be false comfort if the left and right edge heights disagree. The current run shows a `93.448 um` left/right consistency gap, so the `0.053 um` averaged error is not a credible statement of absolute system accuracy.

## Current Calibration Baseline

- The camera calibration stage uses OpenCV circle-grid detection with blob area constraints tuned for the `Campos-8` dataset.
- Reprojection error should be checked after every future image-set expansion instead of reusing old parameters blindly.
- If the validation error jumps back, first verify the `Cam_pos -> Laser` pairing, the world-frame reference image, and whether `stripe_extraction_params` drifted away from `global_centroid + median+gaussian + threshold_ratio=0.25`.
- Even when the averaged error stays tiny, treat `edge_consistency_um` and `conservative_absolute_error_um` as the guardrails against false-good results.
