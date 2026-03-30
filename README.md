# laser_extraction

This repository currently contains two working tracks:

- `python_workspace/laser_extraction.py`: laser stripe center extraction workflow.
- `python_workspace/camera_calibration.py`: monocular camera calibration for the `Campos-8` circle-grid images.

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

The current implementation solves camera intrinsics, per-image extrinsics, and reprojection errors only. Light-plane calibration is the next stage and is intentionally not included here.
