# Light Plane Calibration Summary

## Inputs
- Camera calibration: D:\laser_extraction\output\camera_calibration\calibration_result.npz
- Reference images: Cam_pos15.png, Cam_pos17.png
- Laser images: Laser1.png, Laser2.png
- World reference image: Cam_pos15.png
- Stripe extraction method: global_centroid
- Stripe filter mode: median+gaussian
- Stripe threshold ratio: 0.250
- Validation enabled: True

## Light Plane Equation
- A: 0.066937790
- B: -0.875879911
- C: -0.477863699
- D: 45.022496622

## Plane Fit Metrics
- Mean: 0.044310023 mm (44.310 um)
- RMSE: 0.051787291 mm (51.787 um)
- Std: 0.051787291 mm (51.787 um)
- Max: 0.140974553 mm (140.975 um)

## Laser / Reference Mapping
- Laser1.png <- Cam_pos15.png, ROI=[1700, 1250, 2000, 350], points=2000
- Laser2.png <- Cam_pos17.png, ROI=[1750, 1200, 2000, 350], points=2000

## Step Validation
- Validation image: Pic_20260320142001841.png
- ROI: [0, 1300, 5496, 650]
- Segments: {'left': (400, 1500), 'step': (2500, 3600), 'right': (4500, 5000)}
- Nominal height: 1.800000000 mm
- Measured height: 1.799946728 mm
- Absolute error: 0.000053272 mm (0.053 um)
- Left-step distance: 1.753222887 mm
- Right-step distance: 1.846670569 mm

## Notes
- Board pose for each laser image is solved from the paired reference image via circle-grid detection and solvePnP.
- Laser 3D points are obtained by intersecting undistorted camera rays with the paired board plane.
- Validation 3D points are obtained by intersecting undistorted camera rays with the fitted light plane.
- Step height is evaluated in the world frame defined by the first paired reference image, with height taken along world Z.
- The default stripe center extraction now uses the lecture-aligned gray centroid workflow instead of the earlier local peak-window approximation.
