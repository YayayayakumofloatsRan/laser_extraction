# Project Errors And Risks

## Open Items

- The physical board scale currently assumes `1.0 mm` center spacing. If the real effective spacing differs, extrinsic translation scale must be recalibrated.
- The light-plane validation image currently needs manual ROI and segment ranges; if the acquisition setup changes, the config must be updated instead of reusing the current defaults blindly.
- The `Cam_pos -> Laser` pairing is a hard dependency. Using the wrong pair can still produce a visually plausible light plane while destroying the step-height validation result.

## Current Calibration Baseline

- The camera calibration stage uses OpenCV circle-grid detection with blob area constraints tuned for the `Campos-8` dataset.
- Reprojection error should be checked after every future image-set expansion instead of reusing old parameters blindly.
- If the validation error jumps back from the current `4.084 um` baseline, first verify the `Cam_pos -> Laser` pairing and whether the world-frame reference image was changed.
