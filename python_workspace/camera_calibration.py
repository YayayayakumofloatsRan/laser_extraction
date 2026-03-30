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

from laser_extraction import read_image_unicode


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_IMAGES_DIR = REPO_ROOT / "Campos-8" / "Campos-8"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "camera_calibration"
MIN_VALID_VIEWS = 8


@dataclass(frozen=True)
class BoardSpec:
    pattern_cols: int
    pattern_rows: int
    spacing_mm: float
    circle_diameter_mm: float
    board_outer_width_mm: float = 15.0
    board_outer_height_mm: float = 15.0
    active_pattern_width_mm: float = 10.0
    active_pattern_height_mm: float = 10.0
    board_precision_um: float = 0.5

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.pattern_cols, self.pattern_rows)

    @property
    def point_count(self) -> int:
        return self.pattern_cols * self.pattern_rows


@dataclass
class DetectionRecord:
    image_name: str
    image_path: str
    found: bool
    point_count: int
    overlay_path: str | None = None
    error_message: str | None = None


@dataclass
class CalibrationView:
    image_name: str
    image_path: str
    rotation_vector: list[float]
    translation_vector: list[float]
    reprojection_rmse_px: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monocular camera calibration for symmetric circle grid images.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Directory containing calibration images.")
    parser.add_argument("--pattern-cols", type=int, default=9, help="Number of columns in the circle grid.")
    parser.add_argument("--pattern-rows", type=int, default=9, help="Number of rows in the circle grid.")
    parser.add_argument("--spacing-mm", type=float, default=1.0, help="Center-to-center spacing in millimeters.")
    parser.add_argument("--circle-diameter-mm", type=float, default=0.5, help="Circle diameter metadata in millimeters.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for calibration outputs.")
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save detection overlays to output-dir/detections.",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images to process.")
    parser.add_argument(
        "--fix-k2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fix the k2 radial distortion coefficient during calibration.",
    )
    parser.add_argument(
        "--fix-k3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fix the k3 radial distortion coefficient during calibration.",
    )
    parser.add_argument(
        "--zero-tangent-dist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force tangential distortion coefficients p1 and p2 to zero.",
    )
    return parser


def build_calibration_flags(args: argparse.Namespace) -> int:
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if args.fix_k2:
        flags |= cv2.CALIB_FIX_K2
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    return flags


def resolve_image_paths(images_dir: Path, max_images: int | None) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Image directory is not a folder: {images_dir}")

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in: {images_dir}")
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be a positive integer")
        image_paths = image_paths[:max_images]
    return image_paths


def build_object_points(board: BoardSpec) -> np.ndarray:
    object_points = np.zeros((board.point_count, 3), dtype=np.float32)
    grid = np.mgrid[0 : board.pattern_cols, 0 : board.pattern_rows].T.reshape(-1, 2)
    object_points[:, :2] = grid.astype(np.float32) * np.float32(board.spacing_mm)
    return object_points


def create_blob_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 2000
    params.maxArea = 20000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = True
    params.blobColor = 0
    return cv2.SimpleBlobDetector_create(params)


def save_detection_overlay(
    gray_image: np.ndarray,
    pattern_size: tuple[int, int],
    centers: np.ndarray | None,
    found: bool,
    overlay_path: Path,
    message: str | None = None,
) -> None:
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    if centers is not None and len(centers) > 0:
        cv2.drawChessboardCorners(canvas, pattern_size, centers, found)
    if message:
        cv2.putText(canvas, message, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode overlay image: {overlay_path}")
    encoded.tofile(str(overlay_path))


def detect_circle_grid(
    image_path: Path,
    pattern_size: tuple[int, int],
    detector: cv2.SimpleBlobDetector,
    save_overlays: bool,
    overlay_dir: Path,
) -> tuple[DetectionRecord, np.ndarray | None, np.ndarray | None, tuple[int, int] | None]:
    gray_image = read_image_unicode(image_path, flags=cv2.IMREAD_GRAYSCALE)
    image_size = (int(gray_image.shape[1]), int(gray_image.shape[0]))
    flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
    found, centers = cv2.findCirclesGrid(gray_image, pattern_size, flags=flags, blobDetector=detector)

    refined_centers = None
    overlay_path: Path | None = None
    if found:
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        refined_centers = cv2.cornerSubPix(gray_image, centers, (5, 5), (-1, -1), term_criteria)

    if save_overlays:
        overlay_path = overlay_dir / image_path.name
        message = None if found else "DETECTION FAILED"
        save_detection_overlay(gray_image, pattern_size, refined_centers if found else centers, found, overlay_path, message)

    record = DetectionRecord(
        image_name=image_path.name,
        image_path=str(image_path),
        found=bool(found),
        point_count=0 if refined_centers is None else int(len(refined_centers)),
        overlay_path=str(overlay_path) if overlay_path is not None else None,
        error_message=None if found else "Circle grid not found.",
    )
    return record, gray_image, refined_centers, image_size


def compute_reprojection_rmse(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return float(cv2.norm(image_points, projected_points, cv2.NORM_L2) / np.sqrt(len(projected_points)))


def numpy_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, list):
        return [numpy_to_python(item) for item in value]
    if isinstance(value, dict):
        return {key: numpy_to_python(item) for key, item in value.items()}
    return value


def set_3d_axes_equal(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(np.maximum(maxs - mins, 1e-6)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_reprojection_error_plot(views: list[CalibrationView], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [view.image_name for view in views]
    errors = np.asarray([view.reprojection_rmse_px for view in views], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(views))
    bars = ax.barh(positions, errors, color="steelblue", alpha=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(float(np.mean(errors)), color="crimson", linestyle="--", linewidth=1.5, label="mean")
    ax.set_xlabel("RMSE (px)")
    ax.set_title("Per-Image Reprojection Errors")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(loc="lower right")
    for index, (bar, value) in enumerate(zip(bars, errors)):
        if index < 5 or index >= len(bars) - 3:
            ax.text(
                value + 0.01,
                bar.get_y() + bar.get_height() * 0.5,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_image_point_coverage_plot(
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_points = np.vstack([points.reshape(-1, 2) for points in image_points]).astype(np.float64)
    board_centers = np.asarray([points.reshape(-1, 2).mean(axis=0) for points in image_points], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(all_points[:, 0], all_points[:, 1], s=10, alpha=0.12, color="royalblue", label="circle centers")
    ax.scatter(board_centers[:, 0], board_centers[:, 1], s=28, color="darkorange", label="per-view center")
    ax.scatter(
        [camera_matrix[0, 2]],
        [camera_matrix[1, 2]],
        s=80,
        marker="x",
        linewidths=2.0,
        color="crimson",
        label="principal point",
    )
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title("Calibration Point Coverage On Sensor")
    ax.grid(linestyle=":", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_camera_pose_plot(views: list[CalibrationView], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    origins = np.asarray([view.translation_vector for view in views], dtype=np.float64)
    errors = np.asarray([view.reprojection_rmse_px for view in views], dtype=np.float64)
    normals = []
    for view in views:
        rvec = np.asarray(view.rotation_vector, dtype=np.float64).reshape(3, 1)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        normals.append(rotation_matrix[:, 2])
    normals_array = np.asarray(normals, dtype=np.float64)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        c=errors,
        cmap="viridis",
        s=45,
        depthshade=True,
    )
    normal_length = float(max(np.max(np.ptp(origins, axis=0)), 1.0) * 0.12)
    for origin, normal in zip(origins, normals_array):
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            normal[0],
            normal[1],
            normal[2],
            length=normal_length,
            normalize=True,
            color="black",
            linewidth=0.9,
            alpha=0.8,
        )

    label_indices = list(range(min(3, len(views))))
    label_indices.extend(
        index for index, view in enumerate(views) if view.image_name in {"Cam_pos1.png", "Cam_pos2.png"}
    )
    for index in sorted(set(label_indices)):
        origin = origins[index]
        ax.text(origin[0], origin[1], origin[2], views[index].image_name, fontsize=8)

    ax.scatter([0.0], [0.0], [0.0], color="crimson", marker="x", s=90, linewidths=2.0, label="camera")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.8)
    colorbar.set_label("RMSE (px)")
    all_points = np.vstack([origins, np.zeros((1, 3), dtype=np.float64)])
    set_3d_axes_equal(ax, all_points)
    ax.set_xlabel("Xc (mm)")
    ax.set_ylabel("Yc (mm)")
    ax.set_zlabel("Zc (mm)")
    ax.set_title("Board Poses In Camera Coordinates")
    ax.legend(loc="best")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_per_image_errors(output_path: Path, views: list[CalibrationView]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "reprojection_rmse_px"])
        for view in views:
            writer.writerow([view.image_name, f"{view.reprojection_rmse_px:.9f}"])


def write_summary(
    output_path: Path,
    board: BoardSpec,
    image_size: tuple[int, int],
    detection_records: list[DetectionRecord],
    views: list[CalibrationView],
    rms: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_paths: list[Path],
    save_overlays: bool,
    calibration_flags: int,
    visualization_paths: dict[str, str],
) -> None:
    failed_images = [record.image_name for record in detection_records if not record.found]
    worst_views = views[:5]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera Calibration Summary",
        "",
        "## Run Summary",
        f"- Run time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Images requested: {len(image_paths)}",
        f"- Successful detections: {len(views)}",
        f"- Failed detections: {len(failed_images)}",
        f"- Image size: {image_size[0]} x {image_size[1]}",
        f"- Overall RMS reprojection error: {rms:.9f} px",
        f"- Calibration flags: {calibration_flags}",
        f"- Overlays saved: {save_overlays}",
        "",
        "## Visualizations",
        f"- Reprojection errors plot: {visualization_paths['reprojection_error_plot']}",
        f"- Image-point coverage plot: {visualization_paths['image_point_coverage_plot']}",
        f"- Board poses plot: {visualization_paths['camera_pose_plot']}",
        "",
        "## Board Parameters",
        f"- Pattern: {board.pattern_cols} x {board.pattern_rows} symmetric circle grid",
        f"- Spacing: {board.spacing_mm:.6f} mm",
        f"- Circle diameter metadata: {board.circle_diameter_mm:.6f} mm",
        f"- Outer size metadata: {board.board_outer_width_mm:.6f} mm x {board.board_outer_height_mm:.6f} mm",
        f"- Active pattern metadata: {board.active_pattern_width_mm:.6f} mm x {board.active_pattern_height_mm:.6f} mm",
        f"- Board precision metadata: +/-{board.board_precision_um:.6f} um",
        "",
        "## Intrinsics",
        f"- fx: {camera_matrix[0, 0]:.9f}",
        f"- fy: {camera_matrix[1, 1]:.9f}",
        f"- cx: {camera_matrix[0, 2]:.9f}",
        f"- cy: {camera_matrix[1, 2]:.9f}",
        f"- Distortion: {np.array2string(dist_coeffs.ravel(), precision=9, separator=', ')}",
        "",
        "## Worst Reprojection Errors",
    ]
    for view in worst_views:
        lines.append(f"- {view.image_name}: {view.reprojection_rmse_px:.9f} px")
    if failed_images:
        lines.extend(["", "## Failed Detections"])
        lines.extend(f"- {image_name}" for image_name in failed_images)
    lines.extend(
        [
            "",
            "## Current Focus",
            "- Camera intrinsics and per-view extrinsics are now ready for downstream light-plane calibration.",
            "- Keep the board spacing assumption at 1.0 mm unless the physical board specification is corrected.",
            "",
            "## Next Handoff",
            "- Use calibration_result.npz or calibration_result.json as the camera parameter input to the light-plane calibration stage.",
            "- If later runs add more images, rerun this script instead of editing result files manually.",
            "",
            "## Notes",
            "- This stage intentionally does not solve the laser light plane.",
            "- Per-image errors are sorted in descending order in per_image_errors.csv.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    board = BoardSpec(
        pattern_cols=args.pattern_cols,
        pattern_rows=args.pattern_rows,
        spacing_mm=args.spacing_mm,
        circle_diameter_mm=args.circle_diameter_mm,
    )
    image_paths = resolve_image_paths(args.images_dir, args.max_images)
    detector = create_blob_detector()
    calibration_flags = build_calibration_flags(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_dir = output_dir / "detections"

    object_template = build_object_points(board)
    detection_records: list[DetectionRecord] = []
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_image_paths: list[Path] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        record, _, centers, current_image_size = detect_circle_grid(
            image_path=image_path,
            pattern_size=board.pattern_size,
            detector=detector,
            save_overlays=args.save_overlays,
            overlay_dir=detection_dir,
        )
        detection_records.append(record)
        if image_size is None:
            image_size = current_image_size
        elif image_size != current_image_size:
            raise ValueError(f"Inconsistent image size: {image_path} has {current_image_size}, expected {image_size}")

        if centers is None:
            continue
        object_points.append(object_template.copy())
        image_points.append(centers.astype(np.float32))
        used_image_paths.append(image_path)

    if image_size is None:
        raise RuntimeError("No images were read.")
    if len(used_image_paths) < MIN_VALID_VIEWS:
        raise RuntimeError(
            f"Need at least {MIN_VALID_VIEWS} valid detections, but only {len(used_image_paths)} succeeded."
        )

    initial_camera_matrix = cv2.initCameraMatrix2D(object_points, image_points, image_size)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        initial_camera_matrix,
        None,
        flags=calibration_flags,
    )

    calibrated_rows: list[dict[str, Any]] = []
    for image_path, object_pts, image_pts, rvec, tvec in zip(
        used_image_paths, object_points, image_points, rvecs, tvecs
    ):
        reprojection_rmse = compute_reprojection_rmse(
            object_pts,
            image_pts,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        calibrated_rows.append(
            {
                "view": CalibrationView(
                    image_name=image_path.name,
                    image_path=str(image_path),
                    rotation_vector=[float(value) for value in rvec.ravel()],
                    translation_vector=[float(value) for value in tvec.ravel()],
                    reprojection_rmse_px=reprojection_rmse,
                ),
                "object_points": object_pts,
                "image_points": image_pts,
                "rvec": rvec,
                "tvec": tvec,
            }
        )

    calibrated_rows.sort(key=lambda item: item["view"].reprojection_rmse_px, reverse=True)
    views = [item["view"] for item in calibrated_rows]
    visualization_paths = {
        "reprojection_error_plot": str(output_dir / "reprojection_errors.png"),
        "image_point_coverage_plot": str(output_dir / "image_point_coverage.png"),
        "camera_pose_plot": str(output_dir / "camera_poses_3d.png"),
    }
    save_reprojection_error_plot(views, Path(visualization_paths["reprojection_error_plot"]))
    save_image_point_coverage_plot(
        [item["image_points"] for item in calibrated_rows],
        image_size,
        camera_matrix,
        Path(visualization_paths["image_point_coverage_plot"]),
    )
    save_camera_pose_plot(views, Path(visualization_paths["camera_pose_plot"]))

    np.savez_compressed(
        output_dir / "calibration_result.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=np.asarray([item["rvec"].ravel() for item in calibrated_rows], dtype=np.float64),
        tvecs=np.asarray([item["tvec"].ravel() for item in calibrated_rows], dtype=np.float64),
        per_image_errors=np.asarray([view.reprojection_rmse_px for view in views], dtype=np.float64),
        image_names=np.asarray([view.image_name for view in views]),
        image_paths=np.asarray([view.image_path for view in views]),
        image_size=np.asarray(image_size, dtype=np.int32),
        pattern_size=np.asarray(board.pattern_size, dtype=np.int32),
        spacing_mm=np.asarray([board.spacing_mm], dtype=np.float64),
        object_points=np.asarray([item["object_points"] for item in calibrated_rows], dtype=np.float32),
        image_points=np.asarray([item["image_points"] for item in calibrated_rows], dtype=np.float32),
        calibration_flags=np.asarray([calibration_flags], dtype=np.int32),
        rms=np.asarray([rms], dtype=np.float64),
    )

    json_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "images_dir": str(args.images_dir),
        "output_dir": str(output_dir),
        "save_overlays": bool(args.save_overlays),
        "max_images": args.max_images,
        "fix_k2": bool(args.fix_k2),
        "fix_k3": bool(args.fix_k3),
        "zero_tangent_dist": bool(args.zero_tangent_dist),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "calibration_flags": int(calibration_flags),
        "overall_rms_reprojection_error_px": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.ravel(),
        "board_spec": asdict(board),
        "visualizations": visualization_paths,
        "detection_records": [asdict(record) for record in detection_records],
        "views": [asdict(view) for view in views],
    }
    (output_dir / "calibration_result.json").write_text(
        json.dumps(numpy_to_python(json_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_per_image_errors(output_dir / "per_image_errors.csv", views)
    write_summary(
        output_path=output_dir / "summary.md",
        board=board,
        image_size=image_size,
        detection_records=detection_records,
        views=views,
        rms=float(rms),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_paths=image_paths,
        save_overlays=bool(args.save_overlays),
        calibration_flags=int(calibration_flags),
        visualization_paths=visualization_paths,
    )

    return {
        "output_dir": output_dir,
        "image_count": len(image_paths),
        "valid_view_count": len(used_image_paths),
        "image_size": image_size,
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "views": views,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_calibration(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Images processed: {result['image_count']}")
    print(f"Valid detections: {result['valid_view_count']}")
    print(f"Image size: {result['image_size'][0]} x {result['image_size'][1]}")
    print(f"Overall RMS reprojection error: {result['rms']:.9f} px")
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
