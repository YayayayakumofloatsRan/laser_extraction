# Light Plane Calibration Summary

## Inputs
- Camera calibration: D:\laser_extraction\output\camera_calibration\calibration_result.npz
- Reference images: Cam_pos1.png, Cam_pos2.png
- Laser images: Laser1.png, Laser2.png
- Validation enabled: True

## Light Plane Equation
- A: 0.033192089
- B: -0.906818180
- C: -0.420213128
- D: 39.434702576

## Plane Fit Metrics
- Mean: 0.026348214 mm (26.348 um)
- RMSE: 0.031383621 mm (31.384 um)
- Std: 0.031383621 mm (31.384 um)
- Max: 0.095033062 mm (95.033 um)

## Laser / Reference Mapping
- Laser1.png <- Cam_pos1.png, ROI=[1700, 1250, 2000, 350], points=2000
- Laser2.png <- Cam_pos2.png, ROI=[1750, 1200, 2000, 350], points=2000

## Step Validation
- Validation image: Pic_20260320142001841.png
- ROI: [0, 1300, 5496, 650]
- Segments: {'left': (400, 1500), 'step': (2500, 3600), 'right': (4500, 5000)}
- Nominal height: 1.800000000 mm
- Measured height: 2.122163815 mm
- Absolute error: 0.322163815 mm (322.164 um)
- Left-step distance: 2.123322935 mm
- Right-step distance: 2.121004695 mm

## Notes
- Board pose for each laser image is solved from the paired reference image via circle-grid detection and solvePnP.
- Laser 3D points are obtained by intersecting undistorted camera rays with the paired board plane.
- Validation 3D points are obtained by intersecting undistorted camera rays with the fitted light plane.
