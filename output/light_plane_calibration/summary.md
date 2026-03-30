# Light Plane Calibration Summary

## Inputs
- Camera calibration: D:\laser_extraction\output\camera_calibration\calibration_result.npz
- Reference images: Cam_pos15.png, Cam_pos17.png
- Laser images: Laser1.png, Laser2.png
- World reference image: Cam_pos15.png
- Stripe extraction method: peak_window_centroid
- Stripe filter mode: median+gaussian
- Stripe threshold ratio: 0.330
- Validation enabled: True

## Light Plane Equation
- A: -0.067731208
- B: 0.873623739
- C: 0.481865174
- D: -45.419123598

## Plane Fit Metrics
- Mean: 0.044782147 mm (44.782 um)
- RMSE: 0.052347544 mm (52.348 um)
- Std: 0.052347544 mm (52.348 um)
- Max: 0.146170048 mm (146.170 um)

## Laser / Reference Mapping
- Laser1.png <- Cam_pos15.png, ROI=[1700, 1250, 2000, 350], points=2000
- Laser2.png <- Cam_pos17.png, ROI=[1750, 1200, 2000, 350], points=2000

## Step Validation
- Validation image: Pic_20260320142001841.png
- ROI: [0, 1300, 5496, 650]
- Segments: {'left': (400, 1500), 'step': (2500, 3600), 'right': (4500, 5000)}
- Nominal height: 1.800000000 mm
- Measured height: 1.803690833 mm
- Absolute error: 0.003690833 mm (3.691 um)
- Left-step distance: 1.808490477 mm
- Right-step distance: 1.798891189 mm
- Left edge absolute error: 8.490 um
- Right edge absolute error: 1.109 um
- Left/right consistency gap: 9.599 um
- Conservative absolute error: 8.490 um

## Notes
- Board pose for each laser image is solved from the paired reference image via circle-grid detection and solvePnP.
- Laser 3D points are obtained by intersecting undistorted camera rays with the paired board plane.
- Validation 3D points are obtained by intersecting undistorted camera rays with the fitted light plane.
- Step height is evaluated in the world frame defined by the first paired reference image, with height taken along world Z.
- The default stripe center extraction now uses the lecture-aligned gray centroid workflow instead of the earlier local peak-window approximation.
