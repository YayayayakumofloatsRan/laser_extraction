# Project Errors And Risks

## Open Items

- The physical board scale currently assumes `1.0 mm` center spacing. If the real effective spacing differs, extrinsic translation scale must be recalibrated.
- The light-plane validation image currently needs manual ROI and segment ranges; if the acquisition setup changes, the config must be updated instead of reusing the current defaults blindly.
- Current light-plane validation reports `2.122163815 mm` against the configured nominal `1.8 mm`, so either the nominal step height, the `Cam_pos -> Laser` pairing, or the validation geometry assumption is still inconsistent.

## Current Calibration Baseline

- The camera calibration stage uses OpenCV circle-grid detection with blob area constraints tuned for the `Campos-8` dataset.
- Reprojection error should be checked after every future image-set expansion instead of reusing old parameters blindly.
- If the default validation config keeps reporting a large error versus the nominal step height, first verify the true block thickness and the `Cam_pos -> Laser` pairing before treating it as a model failure.
