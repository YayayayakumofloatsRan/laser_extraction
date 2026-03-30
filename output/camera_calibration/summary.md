# Camera Calibration Summary

## Run Summary
- Run time: 2026-03-30T20:45:56
- Images requested: 26
- Successful detections: 26
- Failed detections: 0
- Image size: 5496 x 3672
- Overall RMS reprojection error: 2.442910862 px
- Calibration flags: 193
- Overlays saved: True

## Board Parameters
- Pattern: 9 x 9 symmetric circle grid
- Spacing: 1.000000 mm
- Circle diameter metadata: 0.500000 mm
- Outer size metadata: 15.000000 mm x 15.000000 mm
- Active pattern metadata: 10.000000 mm x 10.000000 mm
- Board precision metadata: +/-0.500000 um

## Intrinsics
- fx: 22164.950482541
- fy: 22058.055358775
- cx: 2616.116964408
- cy: 1811.527440131
- Distortion: [-0.229981233,  0.         , -0.002432092,  0.000865717,  0.         ]

## Worst Reprojection Errors
- Cam_pos5.png: 2.880617223 px
- Cam_pos13.png: 2.777134352 px
- Cam_pos12.png: 2.761723516 px
- Cam_pos16.png: 2.749464796 px
- Cam_pos15.png: 2.730888745 px

## Current Focus
- Camera intrinsics and per-view extrinsics are now ready for downstream light-plane calibration.
- Keep the board spacing assumption at 1.0 mm unless the physical board specification is corrected.

## Next Handoff
- Use calibration_result.npz or calibration_result.json as the camera parameter input to the light-plane calibration stage.
- If later runs add more images, rerun this script instead of editing result files manually.

## Notes
- This stage intentionally does not solve the laser light plane.
- Per-image errors are sorted in descending order in per_image_errors.csv.
