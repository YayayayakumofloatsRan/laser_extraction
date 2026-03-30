# Light Plane Calibration Summary

## Inputs
- Camera calibration: D:\laser_extraction\output\camera_calibration\calibration_result.npz
- Reference images: Cam_pos15.png, Cam_pos17.png
- Laser images: Laser1.png, Laser2.png
- World reference image: Cam_pos15.png
- Validation enabled: True

## Light Plane Equation
- A: -0.067826264
- B: 0.873736849
- C: 0.481646671
- D: -45.397666420

## Plane Fit Metrics
- Mean: 0.044555401 mm (44.555 um)
- RMSE: 0.052062624 mm (52.063 um)
- Std: 0.052062624 mm (52.063 um)
- Max: 0.149120044 mm (149.120 um)

## Laser / Reference Mapping
- Laser1.png <- Cam_pos15.png, ROI=[1700, 1250, 2000, 350], points=2000
- Laser2.png <- Cam_pos17.png, ROI=[1750, 1200, 2000, 350], points=2000

## Step Validation
- Validation image: Pic_20260320142001841.png
- ROI: [0, 1300, 5496, 650]
- Segments: {'left': (400, 1500), 'step': (2500, 3600), 'right': (4500, 5000)}
- Nominal height: 1.800000000 mm
- Measured height: 1.795916203 mm
- Absolute error: 0.004083797 mm (4.084 um)
- Left-step distance: 1.813506606 mm
- Right-step distance: 1.778325800 mm

## Notes
- Board pose for each laser image is solved from the paired reference image via circle-grid detection and solvePnP.
- Laser 3D points are obtained by intersecting undistorted camera rays with the paired board plane.
- Validation 3D points are obtained by intersecting undistorted camera rays with the fitted light plane.
- Step height is evaluated in the world frame defined by the first paired reference image, with height taken along world Z.
