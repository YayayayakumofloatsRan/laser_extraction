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
- `reprojection_errors.png`
- `image_point_coverage.png`
- `camera_poses_3d.png`
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
- `D:\laser_extraction\output\light_plane_calibration\light_plane_3d.png`
- `D:\laser_extraction\output\light_plane_calibration\light_plane_residuals.png`
- `D:\laser_extraction\output\light_plane_calibration\Laser1_pipeline.png`
- `D:\laser_extraction\output\light_plane_calibration\Laser2_pipeline.png`
- `D:\laser_extraction\output\light_plane_calibration\Pic_20260320142001841_pipeline.png`
- `D:\laser_extraction\output\light_plane_calibration\Pic_20260320142001841_profile.png`

The default config assumes:

- `Cam_pos1.png -> Laser1.png`
- `Cam_pos2.png -> Laser2.png`
- `Pic_20260320142001841.png` is used only for step-height validation
- current camera calibration default fixes `k2` and `k3`
- default light-plane stripe extraction uses `steger_like` with `median+gaussian` filtering and `threshold_ratio=0.33`
- current stripe ROI selection is manual from `python_workspace/light_plane_calibration_config.json`, not auto-ROI
- `python_workspace/light_plane_calibration.py` reuses `python_workspace/laser_extraction.py::process_image` directly

## Validation Confidence Audit

Default command:

```powershell
python D:\laser_extraction\python_workspace\validation_confidence_audit.py --config D:\laser_extraction\python_workspace\light_plane_calibration_config.json --output-dir D:\laser_extraction\output\light_plane_calibration\confidence_audit
```

Generated files:

- `D:\laser_extraction\output\light_plane_calibration\confidence_audit\confidence_audit.md`
- `D:\laser_extraction\output\light_plane_calibration\confidence_audit\confidence_audit.json`
- `D:\laser_extraction\output\light_plane_calibration\confidence_audit\validation_segment_sensitivity.csv`
- `D:\laser_extraction\output\light_plane_calibration\confidence_audit\extraction_method_sensitivity.csv`

This audit is meant to quantify how sensitive the quantity-block result is to validation-window choices and extraction-method choices. Treat the audit as a guardrail against over-claiming a single sub-micron number from one manually selected validation window.
