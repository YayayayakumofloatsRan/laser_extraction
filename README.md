# laser_extraction

This repository currently contains two working tracks:

- `python_workspace/laser_extraction.py`: laser stripe center extraction workflow.
- `python_workspace/camera_calibration.py`: monocular camera calibration for the `Campos-8` circle-grid images.
- `python_workspace/light_plane_calibration.py`: light-plane fitting and step-height validation built on top of the camera calibration result.

## Camera Calibration

Default command:

```powershell
python D:\laser_extraction\python_workspace\camera_calibration.py
```

Default inputs and outputs:

- Images: `D:\laser_extraction\Campos-8\Campos-8`
- Results: `D:\laser_extraction\output\camera_calibration`

Generated files:

- `calibration_result.npz`
- `calibration_result.json`
- `per_image_errors.csv`
- `summary.md`
- `detections/*.png`

The current implementation now covers both camera calibration and a first-pass light-plane calibration / step-height validation workflow.

## Light Plane Calibration

Default command:

```powershell
python D:\laser_extraction\python_workspace\light_plane_calibration.py --config D:\laser_extraction\python_workspace\light_plane_calibration_config.json
```

Default outputs:

- `D:\laser_extraction\output\light_plane_calibration\light_plane_result.json`
- `D:\laser_extraction\output\light_plane_calibration\light_plane_result.npz`
- `D:\laser_extraction\output\light_plane_calibration\plane_points.csv`
- `D:\laser_extraction\output\light_plane_calibration\plane_fit_metrics.csv`
- `D:\laser_extraction\output\light_plane_calibration\validation_result.json`
- `D:\laser_extraction\output\light_plane_calibration\summary.md`

The default config assumes:

- `Cam_pos1.png -> Laser1.png`
- `Cam_pos2.png -> Laser2.png`
- `Pic_20260320142001841.png` is used for step-height validation
