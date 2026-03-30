from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from camera_calibration import BoardSpec, build_object_points, create_blob_detector
from laser_extraction import ExtractionResult, ROI, overlay_centers, process_image, read_image_unicode


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "light_plane_calibration_config.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "light_plane_calibration"


@dataclass
class ReferencePose:
    image_name: str
    image_path: str
    point_count: int
    rotation_vector: list[float]
    translation_vector: list[float]
    plane_normal: list[float]
    plane_offset: float
    overlay_path: str | None


@dataclass
class LaserPlaneInput:
    laser_image_name: str
    laser_image_path: str
    reference_image_name: str
    roi: list[int]
    point_count: int
    overlay_path: str | None
    pipeline_plot_path: str | None = None


@dataclass
class PlaneMetrics:
    mean_abs_mm: float
    rmse_mm: float
    std_mm: float
    max_abs_mm: float


@dataclass
class ValidationResult:
    image_name: str
    image_path: str
    roi: list[int]
    point_count: int
    nominal_step_height_mm: float
    measured_step_height_mm: float
    absolute_error_mm: float
    absolute_error_um: float
    left_step_distance_mm: float
    right_step_distance_mm: float
    left_absolute_error_um: float
    right_absolute_error_um: float
    edge_consistency_um: float
    conservative_absolute_error_um: float
    left_fit_coeffs: list[float]
    step_fit_coeffs: list[float]
    right_fit_coeffs: list[float]
    overlay_path: str | None
    pipeline_plot_path: str | None = None
    profile_plot_path: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Light-plane calibration and step-height validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="JSON config path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save reference, stripe, and validation overlays.",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip step-height validation stage.")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def parse_roi(values: list[int]) -> ROI:
    if len(values) != 4:
        raise ValueError(f"ROI must have 4 integers, got: {values}")
    return ROI(int(values[0]), int(values[1]), int(values[2]), int(values[3]))


def parse_segments(value: dict[str, list[int]]) -> dict[str, tuple[int, int]]:
    required = {"left", "step", "right"}
    if set(value) != required:
        raise ValueError(f"validation_segments must contain exactly {sorted(required)}")
    parsed: dict[str, tuple[int, int]] = {}
    for key in ("left", "step", "right"):
        segment = value[key]
        if len(segment) != 2:
            raise ValueError(f"Segment '{key}' must have [x_start, x_end], got: {segment}")
        parsed[key] = (int(segment[0]), int(segment[1]))
    return parsed


def parse_stripe_params(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "gray_method",
        "blur_kernel",
        "threshold_ratio",
        "auto_roi",
        "roi_padding",
        "filter_mode",
        "segment_count",
        "extraction_method",
        "background_kernel",
        "peak_window_half_height",
        "smooth_kernel_size",
        "smooth_max_deviation",
    }
    params = dict(config.get("stripe_extraction_params", {}))
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"Unsupported stripe_extraction_params keys: {unknown}")
    return params


def load_camera_calibration(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise ValueError(f"Invalid camera calibration file: {path}")
    image_names = [str(name) for name in data["image_names"]]
    rvecs = [np.asarray(rvec, dtype=np.float64).reshape(3, 1) for rvec in data["rvecs"]]
    tvecs = [np.asarray(tvec, dtype=np.float64).reshape(3, 1) for tvec in data["tvecs"]]
    pose_by_name = {name: (rvec, tvec) for name, rvec, tvec in zip(image_names, rvecs, tvecs)}
    pattern_size = tuple(int(v) for v in data["pattern_size"].tolist())
    spacing_mm = float(data["spacing_mm"][0])
    return {
        "camera_matrix": np.asarray(data["camera_matrix"], dtype=np.float64),
        "dist_coeffs": np.asarray(data["dist_coeffs"], dtype=np.float64),
        "pose_by_name": pose_by_name,
        "board_spec": BoardSpec(
            pattern_cols=pattern_size[0],
            pattern_rows=pattern_size[1],
            spacing_mm=spacing_mm,
            circle_diameter_mm=0.5,
        ),
        "image_names": image_names,
    }


def board_plane_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, float]:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    normal = rotation_matrix[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ tvec.reshape(3))
    return normal, offset


def save_reference_overlay(
    image_path: Path,
    centers: np.ndarray,
    pattern_size: tuple[int, int],
    overlay_path: Path,
) -> None:
    image = read_image_unicode(image_path, flags=cv2.IMREAD_GRAYSCALE)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(canvas, pattern_size, centers, True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode reference overlay: {overlay_path}")
    encoded.tofile(str(overlay_path))


def detect_reference_pose(
    image_path: Path,
    board_spec: BoardSpec,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    overlay_dir: Path,
    save_overlays: bool,
) -> ReferencePose:
    image = read_image_unicode(image_path, flags=cv2.IMREAD_GRAYSCALE)
    detector = create_blob_detector()
    flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
    found, centers = cv2.findCirclesGrid(image, board_spec.pattern_size, flags=flags, blobDetector=detector)
    if not found or centers is None or len(centers) != board_spec.point_count:
        raise RuntimeError(f"Reference image circle-grid detection failed: {image_path}")

    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    centers = cv2.cornerSubPix(image, centers, (5, 5), (-1, -1), term_criteria)
    object_points = build_object_points(board_spec)
    solved, rvec, tvec = cv2.solvePnP(object_points, centers, camera_matrix, dist_coeffs)
    if not solved:
        raise RuntimeError(f"solvePnP failed for reference image: {image_path}")

    plane_normal, plane_offset = board_plane_from_pose(rvec, tvec)
    overlay_path: Path | None = None
    if save_overlays:
        overlay_path = overlay_dir / f"{image_path.stem}_reference.png"
        save_reference_overlay(image_path, centers, board_spec.pattern_size, overlay_path)

    return ReferencePose(
        image_name=image_path.name,
        image_path=str(image_path),
        point_count=int(len(centers)),
        rotation_vector=[float(value) for value in rvec.ravel()],
        translation_vector=[float(value) for value in tvec.ravel()],
        plane_normal=[float(value) for value in plane_normal.tolist()],
        plane_offset=float(plane_offset),
        overlay_path=str(overlay_path) if overlay_path is not None else None,
    )


def pixels_to_camera_rays(
    uv_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    points = np.asarray(uv_points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs)
    rays = np.concatenate(
        [undistorted.reshape(-1, 2), np.ones((len(undistorted), 1), dtype=np.float32)],
        axis=1,
    )
    return rays.astype(np.float64)


def intersect_rays_with_plane(
    uv_points: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    rays = pixels_to_camera_rays(uv_points, camera_matrix, dist_coeffs)
    denominator = rays @ plane_normal.reshape(3)
    if np.any(np.isclose(denominator, 0.0)):
        raise RuntimeError("Ray-plane intersection became singular.")
    scale = -plane_offset / denominator
    return rays * scale[:, None]


def fit_plane(points_3d: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points_3d - points_3d.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ points_3d.mean(axis=0))
    return normal, offset


def compute_plane_metrics(points_3d: np.ndarray, plane_normal: np.ndarray, plane_offset: float) -> PlaneMetrics:
    residuals = points_3d @ plane_normal.reshape(3) + plane_offset
    mean_abs_mm = float(np.mean(np.abs(residuals)))
    rmse_mm = float(np.sqrt(np.mean(residuals**2)))
    std_mm = float(np.std(residuals))
    max_abs_mm = float(np.max(np.abs(residuals)))
    return PlaneMetrics(mean_abs_mm=mean_abs_mm, rmse_mm=rmse_mm, std_mm=std_mm, max_abs_mm=max_abs_mm)


def mm_to_um(value_mm: Any) -> Any:
    if isinstance(value_mm, np.ndarray):
        return value_mm.astype(np.float64) * 1000.0
    return float(value_mm * 1000.0)


def save_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_overlay(
    image_path: Path,
    centers: np.ndarray,
    roi: ROI,
    overlay_path: Path,
    segment_ranges: dict[str, tuple[int, int]] | None = None,
) -> None:
    image = read_image_unicode(image_path)
    canvas = overlay_centers(image, centers, roi)
    if segment_ranges:
        colors = {"left": (255, 255, 0), "step": (0, 255, 255), "right": (255, 0, 255)}
        for name, (x_start, x_end) in segment_ranges.items():
            color = colors[name]
            cv2.line(canvas, (int(x_start), 0), (int(x_start), canvas.shape[0] - 1), color, 2)
            cv2.line(canvas, (int(x_end), 0), (int(x_end), canvas.shape[0] - 1), color, 2)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode stripe overlay: {overlay_path}")
    encoded.tofile(str(overlay_path))


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_segment_ranges(canvas: np.ndarray, segment_ranges: dict[str, tuple[int, int]] | None) -> np.ndarray:
    if not segment_ranges:
        return canvas
    result = canvas.copy()
    colors = {"left": (255, 255, 0), "step": (0, 255, 255), "right": (255, 0, 255)}
    for name, (x_start, x_end) in segment_ranges.items():
        color = colors[name]
        cv2.line(result, (int(x_start), 0), (int(x_start), result.shape[0] - 1), color, 2)
        cv2.line(result, (int(x_end), 0), (int(x_end), result.shape[0] - 1), color, 2)
    return result


def add_profile_column_marker(image: np.ndarray, column_index: int) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    x_pos = int(np.clip(column_index, 0, canvas.shape[1] - 1))
    cv2.line(canvas, (x_pos, 0), (x_pos, canvas.shape[0] - 1), (0, 255, 0), 1)
    return canvas


def build_roi_centerline_overlay(extraction: ExtractionResult) -> np.ndarray:
    canvas = cv2.cvtColor(extraction.enhanced_roi, cv2.COLOR_GRAY2BGR)
    if len(extraction.centers) >= 2:
        local_centers = extraction.centers.copy()
        local_centers[:, 0] -= extraction.roi.x
        local_centers[:, 1] -= extraction.roi.y
        polyline = np.round(local_centers).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [polyline], False, (0, 0, 255), 1)
    return canvas


def save_extraction_pipeline_plot(
    extraction: ExtractionResult,
    output_path: Path,
    title: str,
    segment_ranges: dict[str, tuple[int, int]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_overlay = overlay_centers(extraction.original_image, extraction.centers, extraction.roi)
    full_overlay = draw_segment_ranges(full_overlay, segment_ranges)

    raw_roi_preview = add_profile_column_marker(extraction.raw_roi, extraction.profile_column_index)
    filtered_roi_preview = add_profile_column_marker(extraction.filtered_roi, extraction.profile_column_index)
    enhanced_roi_preview = build_roi_centerline_overlay(extraction)

    local_x = extraction.centers[:, 0] - extraction.roi.x if len(extraction.centers) else np.asarray([], dtype=np.float32)
    local_y = extraction.centers[:, 1] - extraction.roi.y if len(extraction.centers) else np.asarray([], dtype=np.float32)
    y_axis = np.arange(len(extraction.raw_profile), dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    axes[0].imshow(image_to_rgb(full_overlay))
    axes[0].set_title(f"Full Image / ROI\nROI={extraction.roi}")
    axes[1].imshow(image_to_rgb(raw_roi_preview))
    axes[1].set_title("Raw ROI")
    axes[2].imshow(image_to_rgb(filtered_roi_preview))
    axes[2].set_title(f"Filtered ROI\nmode={extraction.filter_mode}")
    axes[3].imshow(image_to_rgb(enhanced_roi_preview))
    axes[3].set_title("Background-Suppressed ROI + Centerline")
    axes[4].plot(local_x, local_y, color="crimson", linewidth=1.2)
    axes[4].invert_yaxis()
    axes[4].grid(True, linestyle=":", alpha=0.3)
    axes[4].set_title(f"Centerline In ROI\nmethod={extraction.extraction_method}")
    axes[4].set_xlabel("ROI x (px)")
    axes[4].set_ylabel("ROI y (px)")
    axes[5].plot(extraction.raw_profile, y_axis, color="gray", alpha=0.8, label="raw")
    axes[5].plot(extraction.filtered_profile, y_axis, color="royalblue", label="filtered")
    axes[5].plot(extraction.enhanced_profile, y_axis, color="darkorange", label="enhanced")
    axes[5].invert_yaxis()
    axes[5].grid(True, linestyle=":", alpha=0.3)
    axes[5].set_title(f"Center Column Profile\ncol={extraction.profile_column_index}")
    axes[5].set_xlabel("Intensity")
    axes[5].set_ylabel("ROI y (px)")
    axes[5].legend(loc="best")
    for axis in axes[:4]:
        axis.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_plane_mesh(points_3d: np.ndarray, plane_normal: np.ndarray, plane_offset: float, steps: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    mins = mins - 0.1 * span
    maxs = maxs + 0.1 * span
    axis = int(np.argmax(np.abs(plane_normal)))

    if axis == 2:
        xs = np.linspace(mins[0], maxs[0], steps)
        ys = np.linspace(mins[1], maxs[1], steps)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_z = (-plane_normal[0] * grid_x - plane_normal[1] * grid_y - plane_offset) / plane_normal[2]
        return grid_x, grid_y, grid_z
    if axis == 1:
        xs = np.linspace(mins[0], maxs[0], steps)
        zs = np.linspace(mins[2], maxs[2], steps)
        grid_x, grid_z = np.meshgrid(xs, zs)
        grid_y = (-plane_normal[0] * grid_x - plane_normal[2] * grid_z - plane_offset) / plane_normal[1]
        return grid_x, grid_y, grid_z

    ys = np.linspace(mins[1], maxs[1], steps)
    zs = np.linspace(mins[2], maxs[2], steps)
    grid_y, grid_z = np.meshgrid(ys, zs)
    grid_x = (-plane_normal[1] * grid_y - plane_normal[2] * grid_z - plane_offset) / plane_normal[0]
    return grid_x, grid_y, grid_z


def save_light_plane_plot(
    plane_point_groups: list[np.ndarray],
    laser_labels: list[str],
    plane_normal: np.ndarray,
    plane_offset: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_points = np.vstack(plane_point_groups)
    grid_x, grid_y, grid_z = build_plane_mesh(all_points, plane_normal, plane_offset)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for index, (points, label) in enumerate(zip(plane_point_groups, laser_labels)):
        color = colors[index % len(colors)]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, alpha=0.65, color=color, label=label)

    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.28, color="gold", edgecolor="none")
    center = all_points.mean(axis=0)
    normal_scale = float(np.max(np.ptp(all_points, axis=0)) * 0.25)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        plane_normal[0],
        plane_normal[1],
        plane_normal[2],
        length=normal_scale,
        normalize=True,
        color="black",
        linewidth=2.0,
    )
    ax.scatter([0.0], [0.0], [0.0], color="black", s=40, marker="x", label="camera origin")

    ax.set_title("Light Plane In Camera Coordinates")
    ax.set_xlabel("Xc (mm)")
    ax.set_ylabel("Yc (mm)")
    ax.set_zlabel("Zc (mm)")
    ax.legend(loc="best")
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_plane_residual_plot(
    plane_point_groups: list[np.ndarray],
    laser_labels: list[str],
    plane_normal: np.ndarray,
    plane_offset: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_residuals_um: list[np.ndarray] = []
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for index, (points, label) in enumerate(zip(plane_point_groups, laser_labels)):
        color = colors[index % len(colors)]
        residuals_um = mm_to_um(points @ plane_normal.reshape(3) + plane_offset)
        all_residuals_um.append(residuals_um)
        axes[0].plot(residuals_um, color=color, linewidth=1.0, alpha=0.85, label=label)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Plane Residuals By Laser Stripe")
    axes[0].set_xlabel("Point index")
    axes[0].set_ylabel("Signed distance (um)")
    axes[0].grid(True, linestyle=":", alpha=0.3)
    axes[0].legend(loc="best")

    concatenated = np.concatenate(all_residuals_um)
    axes[1].hist(concatenated, bins=40, color="slateblue", alpha=0.85)
    axes[1].axvline(float(np.mean(concatenated)), color="crimson", linestyle="--", linewidth=1.0, label="mean")
    axes[1].axvline(float(np.median(concatenated)), color="darkgreen", linestyle=":", linewidth=1.0, label="median")
    axes[1].set_title("Plane Residual Histogram")
    axes[1].set_xlabel("Signed distance (um)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, linestyle=":", alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def project_points_to_plane_basis(
    points_3d: np.ndarray,
    plane_normal: np.ndarray,
    line_direction: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    u_axis = line_direction / np.linalg.norm(line_direction)
    v_axis = np.cross(plane_normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    centered = points_3d - origin.reshape(1, 3)
    return np.column_stack((centered @ u_axis, centered @ v_axis))


def camera_to_world(points_3d: np.ndarray, rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    return (points_3d - translation_vector.reshape(1, 3)) @ rotation_matrix


def compute_step_height_world(
    points_3d_world: np.ndarray,
    image_points: np.ndarray,
    segments: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x_coords = image_points[:, 0]
    masks = {
        key: (x_coords >= x_start) & (x_coords <= x_end)
        for key, (x_start, x_end) in segments.items()
    }
    for key, mask in masks.items():
        if int(mask.sum()) < 100:
            raise RuntimeError(f"Validation segment '{key}' does not contain enough points.")

    base_points_xy = np.vstack([points_3d_world[masks["left"], :2], points_3d_world[masks["right"], :2]])
    xy_origin = base_points_xy.mean(axis=0)
    _, _, vh = np.linalg.svd(base_points_xy - xy_origin, full_matrices=False)
    profile_direction_xy = vh[0]
    profile_direction_xy /= np.linalg.norm(profile_direction_xy)

    projected = np.column_stack(
        (
            (points_3d_world[:, :2] - xy_origin.reshape(1, 2)) @ profile_direction_xy,
            points_3d_world[:, 2],
        )
    )
    line_fits: dict[str, np.ndarray] = {}
    for key in ("left", "step", "right"):
        line_fits[key] = np.polyfit(projected[masks[key], 0], projected[masks[key], 1], 1)

    sample_x = float(np.mean(projected[masks["step"], 0]))
    z_left = np.polyval(line_fits["left"], sample_x)
    z_step = np.polyval(line_fits["step"], sample_x)
    z_right = np.polyval(line_fits["right"], sample_x)
    left_step_distance = abs(z_step - z_left)
    right_step_distance = abs(z_step - z_right)
    measured_step_height = float(abs(z_step - 0.5 * (z_left + z_right)))
    return (
        profile_direction_xy,
        projected,
        {
            "left_fit": line_fits["left"],
            "step_fit": line_fits["step"],
            "right_fit": line_fits["right"],
            "left_step_distance_mm": np.asarray([left_step_distance], dtype=np.float64),
            "right_step_distance_mm": np.asarray([right_step_distance], dtype=np.float64),
            "measured_step_height_mm": np.asarray([measured_step_height], dtype=np.float64),
        },
    )


def save_validation_profile_plot(
    projected_profile: np.ndarray,
    image_points: np.ndarray,
    segments: dict[str, tuple[int, int]],
    fit_coeffs: dict[str, np.ndarray],
    measured_step_height_mm: float,
    nominal_step_height_mm: float,
    conservative_error_um: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_coords = image_points[:, 0]
    masks = {
        key: (x_coords >= x_start) & (x_coords <= x_end)
        for key, (x_start, x_end) in segments.items()
    }
    sample_x = float(np.mean(projected_profile[masks["step"], 0]))
    z_left = float(np.polyval(fit_coeffs["left"], sample_x))
    z_step = float(np.polyval(fit_coeffs["step"], sample_x))
    z_right = float(np.polyval(fit_coeffs["right"], sample_x))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(projected_profile[:, 0], projected_profile[:, 1], s=6, alpha=0.15, color="gray", label="all points")
    colors = {"left": "tab:blue", "step": "tab:orange", "right": "tab:green"}
    for key in ("left", "step", "right"):
        color = colors[key]
        xs = projected_profile[masks[key], 0]
        zs = projected_profile[masks[key], 1]
        ax.scatter(xs, zs, s=9, alpha=0.45, color=color, label=f"{key} segment")
        fit_x = np.linspace(float(xs.min()), float(xs.max()), 200)
        fit_z = np.polyval(fit_coeffs[key], fit_x)
        ax.plot(fit_x, fit_z, color=color, linewidth=2.2)

    ax.axvline(sample_x, color="black", linestyle="--", linewidth=1.0, label="sample position")
    ax.scatter([sample_x, sample_x, sample_x], [z_left, z_step, z_right], color="black", s=28, zorder=5)
    ax.text(
        sample_x,
        z_step,
        (
            f"measured={measured_step_height_mm:.6f} mm\n"
            f"nominal={nominal_step_height_mm:.6f} mm\n"
            f"conservative={conservative_error_um:.3f} um"
        ),
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_title("Quantity-Block World-Z Profile And Line Fits")
    ax.set_xlabel("Profile coordinate (mm)")
    ax.set_ylabel("World Z (mm)")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def run_light_plane_calibration(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_json(config_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"

    camera_calibration_path = resolve_path(config["camera_calibration_path"], config_path)
    camera_calibration = load_camera_calibration(camera_calibration_path)
    camera_matrix = camera_calibration["camera_matrix"]
    dist_coeffs = camera_calibration["dist_coeffs"]
    board_spec = camera_calibration["board_spec"]

    reference_images = [resolve_path(value, config_path) for value in config["reference_images"]]
    laser_images = [resolve_path(value, config_path) for value in config["laser_images"]]
    laser_rois = [parse_roi(values) for values in config["laser_rois"]]
    stripe_params = parse_stripe_params(config)

    if len(laser_images) != len(laser_rois):
        raise ValueError("laser_images and laser_rois must have the same length.")
    if not reference_images:
        raise ValueError("At least one reference image is required.")
    if len(reference_images) not in {1, len(laser_images)}:
        raise ValueError("reference_images must contain either one image or match laser_images length.")

    reference_results_by_path: dict[Path, ReferencePose] = {}
    for reference_image in reference_images:
        if reference_image in reference_results_by_path:
            continue
        reference_results_by_path[reference_image] = detect_reference_pose(
            image_path=reference_image,
            board_spec=board_spec,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            overlay_dir=overlay_dir,
            save_overlays=args.save_overlays,
        )

    reference_for_lasers = [reference_images[0] for _ in laser_images] if len(reference_images) == 1 else reference_images
    world_reference_path = reference_for_lasers[0]
    world_reference_pose = reference_results_by_path[world_reference_path]
    world_rvec = np.asarray(world_reference_pose.rotation_vector, dtype=np.float64).reshape(3, 1)
    world_tvec = np.asarray(world_reference_pose.translation_vector, dtype=np.float64).reshape(3, 1)
    world_rotation_matrix, _ = cv2.Rodrigues(world_rvec)

    plane_points: list[np.ndarray] = []
    plane_rows: list[list[Any]] = []
    laser_inputs: list[LaserPlaneInput] = []
    laser_pipeline_paths: dict[str, str] = {}

    for laser_image, laser_roi, reference_image in zip(laser_images, laser_rois, reference_for_lasers):
        reference_pose = reference_results_by_path[reference_image]
        plane_normal = np.asarray(reference_pose.plane_normal, dtype=np.float64)
        plane_offset = float(reference_pose.plane_offset)

        extraction = process_image(image_path=laser_image, roi=laser_roi, **stripe_params)
        if len(extraction.centers) == 0:
            raise RuntimeError(f"Laser stripe extraction returned no points: {laser_image}")
        points_3d = intersect_rays_with_plane(
            uv_points=extraction.centers,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        if len(points_3d) < 100:
            raise RuntimeError(f"Laser stripe reconstruction returned too few points: {laser_image}")

        overlay_path: Path | None = None
        if args.save_overlays:
            overlay_path = overlay_dir / f"{laser_image.stem}_laser_overlay.png"
            save_overlay(laser_image, extraction.centers, laser_roi, overlay_path)
            pipeline_plot_path = output_dir / f"{laser_image.stem}_pipeline.png"
            save_extraction_pipeline_plot(
                extraction=extraction,
                output_path=pipeline_plot_path,
                title=f"{laser_image.name} Stripe Extraction Pipeline",
            )
            laser_pipeline_paths[laser_image.name] = str(pipeline_plot_path)
        else:
            pipeline_plot_path = None

        laser_inputs.append(
            LaserPlaneInput(
                laser_image_name=laser_image.name,
                laser_image_path=str(laser_image),
                reference_image_name=reference_image.name,
                roi=[laser_roi.x, laser_roi.y, laser_roi.w, laser_roi.h],
                point_count=int(len(points_3d)),
                overlay_path=str(overlay_path) if overlay_path is not None else None,
                pipeline_plot_path=str(pipeline_plot_path) if pipeline_plot_path is not None else None,
            )
        )
        plane_points.append(points_3d)
        plane_rows.extend(
            [
                [laser_image.name, float(uv[0]), float(uv[1]), float(pt[0]), float(pt[1]), float(pt[2])]
                for uv, pt in zip(extraction.centers, points_3d)
            ]
        )

    all_plane_points = np.vstack(plane_points)
    light_plane_normal, light_plane_offset = fit_plane(all_plane_points)
    plane_metrics = compute_plane_metrics(all_plane_points, light_plane_normal, light_plane_offset)
    light_plane_plot_path = output_dir / "light_plane_3d.png"
    save_light_plane_plot(
        plane_point_groups=plane_points,
        laser_labels=[item.laser_image_name for item in laser_inputs],
        plane_normal=light_plane_normal,
        plane_offset=light_plane_offset,
        output_path=light_plane_plot_path,
    )
    light_plane_residual_plot_path = output_dir / "light_plane_residuals.png"
    save_plane_residual_plot(
        plane_point_groups=plane_points,
        laser_labels=[item.laser_image_name for item in laser_inputs],
        plane_normal=light_plane_normal,
        plane_offset=light_plane_offset,
        output_path=light_plane_residual_plot_path,
    )

    validation_payload: ValidationResult | None = None
    validation_points: np.ndarray | None = None
    validation_projection: np.ndarray | None = None
    validation_line_direction: np.ndarray | None = None
    validation_arrays: dict[str, np.ndarray] = {}
    validation_segments = parse_segments(config["validation_segments"])

    if not args.skip_validation:
        validation_image = resolve_path(config["validation_image"], config_path)
        validation_roi = parse_roi(config["validation_roi"])
        validation_extraction = process_image(image_path=validation_image, roi=validation_roi, **stripe_params)
        if len(validation_extraction.centers) == 0:
            raise RuntimeError(f"Validation stripe extraction returned no points: {validation_image}")

        validation_points = intersect_rays_with_plane(
            uv_points=validation_extraction.centers,
            plane_normal=light_plane_normal,
            plane_offset=light_plane_offset,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        validation_world_points = camera_to_world(validation_points, world_rotation_matrix, world_tvec)
        validation_line_direction, validation_projection, validation_arrays = compute_step_height_world(
            points_3d_world=validation_world_points,
            image_points=validation_extraction.centers,
            segments=validation_segments,
        )

        measured_step_height = float(validation_arrays["measured_step_height_mm"][0])
        nominal_step_height = float(config["nominal_step_height_mm"])
        absolute_error = abs(measured_step_height - nominal_step_height)
        left_step_distance = float(validation_arrays["left_step_distance_mm"][0])
        right_step_distance = float(validation_arrays["right_step_distance_mm"][0])
        left_absolute_error_um = mm_to_um(abs(left_step_distance - nominal_step_height))
        right_absolute_error_um = mm_to_um(abs(right_step_distance - nominal_step_height))
        edge_consistency_um = mm_to_um(abs(left_step_distance - right_step_distance))
        conservative_absolute_error_um = max(left_absolute_error_um, right_absolute_error_um)

        validation_overlay_path: Path | None = None
        if args.save_overlays:
            validation_overlay_path = overlay_dir / f"{validation_image.stem}_validation_overlay.png"
            save_overlay(
                validation_image,
                validation_extraction.centers,
                validation_roi,
                validation_overlay_path,
                segment_ranges=validation_segments,
            )
            validation_pipeline_plot_path = output_dir / f"{validation_image.stem}_pipeline.png"
            save_extraction_pipeline_plot(
                extraction=validation_extraction,
                output_path=validation_pipeline_plot_path,
                title=f"{validation_image.name} Quantity-Block Extraction Pipeline",
                segment_ranges=validation_segments,
            )
            validation_profile_plot_path = output_dir / f"{validation_image.stem}_profile.png"
            save_validation_profile_plot(
                projected_profile=validation_projection,
                image_points=validation_extraction.centers,
                segments=validation_segments,
                fit_coeffs={
                    "left": validation_arrays["left_fit"],
                    "step": validation_arrays["step_fit"],
                    "right": validation_arrays["right_fit"],
                },
                measured_step_height_mm=measured_step_height,
                nominal_step_height_mm=nominal_step_height,
                conservative_error_um=float(conservative_absolute_error_um),
                output_path=validation_profile_plot_path,
            )
        else:
            validation_pipeline_plot_path = None
            validation_profile_plot_path = None

        validation_payload = ValidationResult(
            image_name=validation_image.name,
            image_path=str(validation_image),
            roi=[validation_roi.x, validation_roi.y, validation_roi.w, validation_roi.h],
            point_count=int(len(validation_points)),
            nominal_step_height_mm=nominal_step_height,
            measured_step_height_mm=measured_step_height,
            absolute_error_mm=float(absolute_error),
            absolute_error_um=mm_to_um(float(absolute_error)),
            left_step_distance_mm=left_step_distance,
            right_step_distance_mm=right_step_distance,
            left_absolute_error_um=float(left_absolute_error_um),
            right_absolute_error_um=float(right_absolute_error_um),
            edge_consistency_um=float(edge_consistency_um),
            conservative_absolute_error_um=float(conservative_absolute_error_um),
            left_fit_coeffs=[float(value) for value in validation_arrays["left_fit"]],
            step_fit_coeffs=[float(value) for value in validation_arrays["step_fit"]],
            right_fit_coeffs=[float(value) for value in validation_arrays["right_fit"]],
            overlay_path=str(validation_overlay_path) if validation_overlay_path is not None else None,
            pipeline_plot_path=str(validation_pipeline_plot_path) if validation_pipeline_plot_path is not None else None,
            profile_plot_path=str(validation_profile_plot_path) if validation_profile_plot_path is not None else None,
        )

    np.savez_compressed(
        output_dir / "light_plane_result.npz",
        plane_normal=light_plane_normal,
        plane_offset=np.asarray([light_plane_offset], dtype=np.float64),
        plane_points=all_plane_points.astype(np.float64),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        plane_metrics=np.asarray(
            [plane_metrics.mean_abs_mm, plane_metrics.rmse_mm, plane_metrics.std_mm, plane_metrics.max_abs_mm],
            dtype=np.float64,
        ),
        laser_image_names=np.asarray([item.laser_image_name for item in laser_inputs]),
        reference_image_names=np.asarray([item.reference_image_name for item in laser_inputs]),
        validation_points=np.asarray([] if validation_points is None else validation_points, dtype=np.float64),
        validation_projection=np.asarray([] if validation_projection is None else validation_projection, dtype=np.float64),
        validation_line_direction=np.asarray([] if validation_line_direction is None else validation_line_direction, dtype=np.float64),
        world_reference_name=np.asarray([world_reference_pose.image_name]),
        world_reference_rvec=world_rvec.astype(np.float64),
        world_reference_tvec=world_tvec.astype(np.float64),
        validation_measured_step_mm=np.asarray(
            [] if validation_payload is None else [validation_payload.measured_step_height_mm],
            dtype=np.float64,
        ),
        validation_nominal_step_mm=np.asarray(
            [] if validation_payload is None else [validation_payload.nominal_step_height_mm],
            dtype=np.float64,
        ),
        validation_left_step_mm=np.asarray(
            [] if validation_payload is None else [validation_payload.left_step_distance_mm],
            dtype=np.float64,
        ),
        validation_right_step_mm=np.asarray(
            [] if validation_payload is None else [validation_payload.right_step_distance_mm],
            dtype=np.float64,
        ),
        validation_edge_consistency_um=np.asarray(
            [] if validation_payload is None else [validation_payload.edge_consistency_um],
            dtype=np.float64,
        ),
        validation_conservative_error_um=np.asarray(
            [] if validation_payload is None else [validation_payload.conservative_absolute_error_um],
            dtype=np.float64,
        ),
    )

    plane_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "camera_calibration_path": str(camera_calibration_path),
        "stripe_extraction_params": to_jsonable(stripe_params),
        "world_reference_image": world_reference_pose.image_name,
        "light_plane_plot_path": str(light_plane_plot_path),
        "visualizations": {
            "light_plane_plot_path": str(light_plane_plot_path),
            "light_plane_residual_plot_path": str(light_plane_residual_plot_path),
            "laser_pipeline_paths": laser_pipeline_paths,
            "validation_pipeline_path": None if validation_payload is None else validation_payload.pipeline_plot_path,
            "validation_profile_plot_path": None if validation_payload is None else validation_payload.profile_plot_path,
        },
        "reference_poses": [asdict(reference_results_by_path[path]) for path in reference_results_by_path],
        "laser_inputs": [asdict(item) for item in laser_inputs],
        "light_plane": {
            "equation": {
                "A": float(light_plane_normal[0]),
                "B": float(light_plane_normal[1]),
                "C": float(light_plane_normal[2]),
                "D": float(light_plane_offset),
            },
            "metrics_mm": asdict(plane_metrics),
            "metrics_um": {
                "mean_abs_um": mm_to_um(plane_metrics.mean_abs_mm),
                "rmse_um": mm_to_um(plane_metrics.rmse_mm),
                "std_um": mm_to_um(plane_metrics.std_mm),
                "max_abs_um": mm_to_um(plane_metrics.max_abs_mm),
            },
            "point_count": int(len(all_plane_points)),
        },
    }
    (output_dir / "light_plane_result.json").write_text(
        json.dumps(to_jsonable(plane_json), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_csv(
        output_dir / "plane_points.csv",
        ["laser_image", "pixel_x", "pixel_y", "Xc_mm", "Yc_mm", "Zc_mm"],
        plane_rows,
    )
    save_csv(
        output_dir / "plane_fit_metrics.csv",
        ["metric", "value_mm", "value_um"],
        [
            ["mean_abs", f"{plane_metrics.mean_abs_mm:.9f}", f"{mm_to_um(plane_metrics.mean_abs_mm):.3f}"],
            ["rmse", f"{plane_metrics.rmse_mm:.9f}", f"{mm_to_um(plane_metrics.rmse_mm):.3f}"],
            ["std", f"{plane_metrics.std_mm:.9f}", f"{mm_to_um(plane_metrics.std_mm):.3f}"],
            ["max_abs", f"{plane_metrics.max_abs_mm:.9f}", f"{mm_to_um(plane_metrics.max_abs_mm):.3f}"],
        ],
    )

    if validation_payload is not None:
        (output_dir / "validation_result.json").write_text(
            json.dumps(to_jsonable(asdict(validation_payload)), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary_lines = [
        "# Light Plane Calibration Summary",
        "",
        "## Inputs",
        f"- Camera calibration: {camera_calibration_path}",
        f"- Reference images: {', '.join(path.name for path in reference_results_by_path)}",
        f"- Laser images: {', '.join(item.laser_image_name for item in laser_inputs)}",
        f"- Validation image: {resolve_path(config['validation_image'], config_path).name}",
        f"- World reference image: {world_reference_pose.image_name}",
        f"- 3D light-plane plot: {light_plane_plot_path}",
        f"- Light-plane residual plot: {light_plane_residual_plot_path}",
        f"- Stripe extraction method: {stripe_params.get('extraction_method', 'global_centroid')}",
        f"- Stripe filter mode: {stripe_params.get('filter_mode', 'gaussian')}",
        f"- Stripe threshold ratio: {float(stripe_params.get('threshold_ratio', 0.3)):.3f}",
        f"- Stripe ROI mode: {'auto' if stripe_params.get('auto_roi', False) else 'manual (config JSON)'}",
        f"- Validation enabled: {validation_payload is not None}",
        "",
        "## Light Plane Equation",
        f"- A: {light_plane_normal[0]:.9f}",
        f"- B: {light_plane_normal[1]:.9f}",
        f"- C: {light_plane_normal[2]:.9f}",
        f"- D: {light_plane_offset:.9f}",
        "",
        "## Plane Fit Metrics",
        f"- Mean: {plane_metrics.mean_abs_mm:.9f} mm ({mm_to_um(plane_metrics.mean_abs_mm):.3f} um)",
        f"- RMSE: {plane_metrics.rmse_mm:.9f} mm ({mm_to_um(plane_metrics.rmse_mm):.3f} um)",
        f"- Std: {plane_metrics.std_mm:.9f} mm ({mm_to_um(plane_metrics.std_mm):.3f} um)",
        f"- Max: {plane_metrics.max_abs_mm:.9f} mm ({mm_to_um(plane_metrics.max_abs_mm):.3f} um)",
        "",
        "## Laser / Reference Mapping",
    ]
    summary_lines.extend(
        f"- {item.laser_image_name} <- {item.reference_image_name}, ROI={item.roi}, points={item.point_count}, pipeline={item.pipeline_plot_path}"
        for item in laser_inputs
    )
    if validation_payload is not None:
        summary_lines.extend(
            [
                "",
                "## Step Validation",
                f"- Validation image: {validation_payload.image_name}",
                f"- ROI: {validation_payload.roi}",
                f"- Segments: {validation_segments}",
                f"- Nominal height: {validation_payload.nominal_step_height_mm:.9f} mm",
                f"- Measured height: {validation_payload.measured_step_height_mm:.9f} mm",
                f"- Absolute error: {validation_payload.absolute_error_mm:.9f} mm ({validation_payload.absolute_error_um:.3f} um)",
                f"- Left-step distance: {validation_payload.left_step_distance_mm:.9f} mm",
                f"- Right-step distance: {validation_payload.right_step_distance_mm:.9f} mm",
                f"- Left edge absolute error: {validation_payload.left_absolute_error_um:.3f} um",
                f"- Right edge absolute error: {validation_payload.right_absolute_error_um:.3f} um",
                f"- Left/right consistency gap: {validation_payload.edge_consistency_um:.3f} um",
                f"- Conservative absolute error: {validation_payload.conservative_absolute_error_um:.3f} um",
                f"- Validation pipeline plot: {validation_payload.pipeline_plot_path}",
                f"- Validation profile plot: {validation_payload.profile_plot_path}",
            ]
        )
        if validation_payload.edge_consistency_um > 10.0:
            summary_lines.extend(
                [
                    "- Warning: the averaged step-height error is being reduced by left/right cancellation, so it is not a trustworthy claim of true system accuracy.",
                ]
            )
    summary_lines.extend(
        [
            "",
            "## Notes",
            "- Board pose for each laser image is solved from the paired reference image via circle-grid detection and solvePnP.",
            "- Laser stripe center extraction is reused directly from python_workspace/laser_extraction.py.",
            "- Laser 3D points are obtained by intersecting undistorted camera rays with the paired board plane.",
            "- Validation 3D points are obtained by intersecting undistorted camera rays with the fitted light plane.",
            "- Step height is evaluated in the world frame defined by the first paired reference image, with height taken along world Z.",
            "- Pic_20260320142001841.png is used only for quantity-block validation and is not part of the light-plane fitting input.",
            "- The default stripe center extraction uses the steger_like subpixel method from python_workspace/laser_extraction.py.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "output_dir": output_dir,
        "plane_normal": light_plane_normal,
        "plane_offset": light_plane_offset,
        "plane_metrics": plane_metrics,
        "validation": validation_payload,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_light_plane_calibration(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Output directory: {result['output_dir']}")
    print(
        "Light plane equation: "
        f"{result['plane_normal'][0]:.9f} x + {result['plane_normal'][1]:.9f} y + "
        f"{result['plane_normal'][2]:.9f} z + {result['plane_offset']:.9f} = 0"
    )
    print(
        "Plane fit RMSE: "
        f"{result['plane_metrics'].rmse_mm:.9f} mm ({mm_to_um(result['plane_metrics'].rmse_mm):.3f} um)"
    )
    if result["validation"] is not None:
        validation: ValidationResult = result["validation"]
        print(
            "Measured step height: "
            f"{validation.measured_step_height_mm:.9f} mm, "
            f"error={validation.absolute_error_mm:.9f} mm ({validation.absolute_error_um:.3f} um)"
        )
        print(
            "Validation consistency: "
            f"left_error={validation.left_absolute_error_um:.3f} um, "
            f"right_error={validation.right_absolute_error_um:.3f} um, "
            f"edge_gap={validation.edge_consistency_um:.3f} um, "
            f"conservative_error={validation.conservative_absolute_error_um:.3f} um"
        )
        if validation.edge_consistency_um > 10.0:
            print(
                "WARNING: averaged step-height error is being reduced by left/right cancellation; "
                "do not treat it as true system accuracy."
            )


if __name__ == "__main__":
    main()
