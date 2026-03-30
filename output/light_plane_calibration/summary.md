# Light Plane Calibration Summary

## Inputs
- Camera calibration: D:\laser_extraction\output\camera_calibration\calibration_result.npz
- Reference images: Cam_pos1.png, Cam_pos2.png
- Laser images: Laser1.png, Laser2.png
- Validation image: Pic_20260320142001841.png
- World reference image: Cam_pos1.png
- 3D light-plane plot: D:\laser_extraction\output\light_plane_calibration\light_plane_3d.png
- Stripe extraction method: steger_like
- Stripe filter mode: median+gaussian
- Stripe threshold ratio: 0.330
- Validation enabled: True

## Light Plane Equation
- A: 0.032986212
- B: -0.895094583
- C: -0.444654470
- D: 42.137529211

## Plane Fit Metrics
- Mean: 0.029694470 mm (29.694 um)
- RMSE: 0.036225838 mm (36.226 um)
- Std: 0.036225838 mm (36.226 um)
- Max: 0.108681235 mm (108.681 um)

## Laser / Reference Mapping
- Laser1.png <- Cam_pos1.png, ROI=[1700, 1250, 2000, 350], points=2000
- Laser2.png <- Cam_pos2.png, ROI=[1750, 1200, 2000, 350], points=2000

## Step Validation
- Validation image: Pic_20260320142001841.png
- ROI: [0, 1300, 5496, 650]
- Segments: {'left': (400, 1500), 'step': (2500, 3600), 'right': (4500, 5000)}
- Nominal height: 1.800000000 mm
- Measured height: 1.797759644 mm
- Absolute error: 0.002240356 mm (2.240 um)
- Left-step distance: 1.802089201 mm
- Right-step distance: 1.793430088 mm
- Left edge absolute error: 2.089 um
- Right edge absolute error: 6.570 um
- Left/right consistency gap: 8.659 um
- Conservative absolute error: 6.570 um

## Notes
- Board pose for each laser image is solved from the paired reference image via circle-grid detection and solvePnP.
- Laser stripe center extraction is reused directly from python_workspace/laser_extraction.py.
- Laser 3D points are obtained by intersecting undistorted camera rays with the paired board plane.
- Validation 3D points are obtained by intersecting undistorted camera rays with the fitted light plane.
- Step height is evaluated in the world frame defined by the first paired reference image, with height taken along world Z.
- Pic_20260320142001841.png is used only for quantity-block validation and is not part of the light-plane fitting input.
- The default stripe center extraction uses the steger_like subpixel method from python_workspace/laser_extraction.py.
