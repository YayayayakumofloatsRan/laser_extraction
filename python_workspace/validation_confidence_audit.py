from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

import light_plane_calibration as lpc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "light_plane_calibration_config.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "light_plane_calibration" / "confidence_audit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit quantity-block validation confidence via window perturbation and method sensitivity."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="JSON config path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Audit output directory.")
    parser.add_argument(
        "--step-pad",
        type=int,
        default=100,
        help="Pixels to perturb the configured step segment on both sides during sensitivity analysis.",
    )
    parser.add_argument(
        "--step-grid",
        type=int,
        default=25,
        help="Grid spacing in pixels for step-segment perturbation analysis.",
    )
    return parser


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_validation_profile(
    config: dict[str, Any],
    plane_normal: np.ndarray,
    plane_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    config_path = Path(config["_config_path"])
    camera_calibration = lpc.load_camera_calibration(
        lpc.resolve_path(config["camera_calibration_path"], config_path)
    )
    camera_matrix = camera_calibration["camera_matrix"]
    dist_coeffs = camera_calibration["dist_coeffs"]
    pose_by_name = camera_calibration["pose_by_name"]

    world_rvec, world_tvec = pose_by_name[Path(config["reference_images"][0]).name]
    world_rotation_matrix, _ = cv2.Rodrigues(world_rvec)

    stripe_params = lpc.parse_stripe_params(config)
    validation_image = lpc.resolve_path(config["validation_image"], config_path)
    validation_roi = lpc.parse_roi(config["validation_roi"])
    extraction = lpc.process_image(image_path=validation_image, roi=validation_roi, **stripe_params)
    validation_points = lpc.intersect_rays_with_plane(
        uv_points=extraction.centers,
        plane_normal=plane_normal,
        plane_offset=plane_offset,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    validation_world_points = lpc.camera_to_world(validation_points, world_rotation_matrix, world_tvec)
    return extraction.centers, validation_world_points


def compute_metrics(
    image_points: np.ndarray,
    world_points: np.ndarray,
    segments: dict[str, tuple[int, int]],
    nominal_step_height_mm: float,
) -> dict[str, float]:
    _, _, arrays = lpc.compute_step_height_world(world_points, image_points, segments)
    measured_step_height = float(arrays["measured_step_height_mm"][0])
    left_step_distance = float(arrays["left_step_distance_mm"][0])
    right_step_distance = float(arrays["right_step_distance_mm"][0])
    left_error_um = float(lpc.mm_to_um(abs(left_step_distance - nominal_step_height_mm)))
    right_error_um = float(lpc.mm_to_um(abs(right_step_distance - nominal_step_height_mm)))
    edge_gap_um = float(lpc.mm_to_um(abs(left_step_distance - right_step_distance)))
    return {
        "measured_step_height_mm": measured_step_height,
        "absolute_error_um": float(lpc.mm_to_um(abs(measured_step_height - nominal_step_height_mm))),
        "left_error_um": left_error_um,
        "right_error_um": right_error_um,
        "edge_gap_um": edge_gap_um,
        "conservative_error_um": max(left_error_um, right_error_um),
    }


def run_method_sensitivity(
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    methods = ["global_centroid", "peak_window_centroid", "gaussian_fit", "steger_like"]
    results: list[dict[str, Any]] = []
    for method in methods:
        method_config = json.loads(json.dumps(config, ensure_ascii=False))
        method_config["stripe_extraction_params"]["extraction_method"] = method
        cfg_path = output_dir / f"method_{method}.json"
        cfg_path.write_text(json.dumps(method_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        method_output_dir = output_dir / f"method_{method}"
        if method_output_dir.exists():
            shutil.rmtree(method_output_dir)
        args = SimpleNamespace(config=cfg_path, output_dir=method_output_dir, save_overlays=False, skip_validation=False)
        result = lpc.run_light_plane_calibration(args)
        validation = result["validation"]
        if validation is None:
            raise RuntimeError(f"Validation unexpectedly missing for extraction method: {method}")
        results.append(
            {
                "method": method,
                "measured_step_height_mm": float(validation.measured_step_height_mm),
                "absolute_error_um": float(validation.absolute_error_um),
                "left_error_um": float(validation.left_absolute_error_um),
                "right_error_um": float(validation.right_absolute_error_um),
                "edge_gap_um": float(validation.edge_consistency_um),
                "conservative_error_um": float(validation.conservative_absolute_error_um),
                "plane_rmse_um": float(lpc.mm_to_um(result["plane_metrics"].rmse_mm)),
            }
        )
    return results


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    config = load_config(config_path)
    config["_config_path"] = str(config_path)
    nominal = float(config["nominal_step_height_mm"])

    with tempfile.TemporaryDirectory(prefix="validation_audit_", dir=str(REPO_ROOT / "tmp")) as tmpdir:
        base_output_dir = Path(tmpdir) / "base_run"
        base_args = SimpleNamespace(
            config=config_path,
            output_dir=base_output_dir,
            save_overlays=False,
            skip_validation=False,
        )
        base_result = lpc.run_light_plane_calibration(base_args)
        validation = base_result["validation"]
        if validation is None:
            raise RuntimeError("Validation output is required for confidence audit.")

        configured_segments = lpc.parse_segments(config["validation_segments"])
        image_points, world_points = build_validation_profile(
            config=config,
            plane_normal=np.asarray(base_result["plane_normal"], dtype=np.float64),
            plane_offset=float(base_result["plane_offset"]),
        )

        segment_rows: list[dict[str, Any]] = []
        left_segment = configured_segments["left"]
        right_segment = configured_segments["right"]
        step_segment = configured_segments["step"]
        for start_delta in range(-args.step_pad, args.step_pad + 1, args.step_grid):
            for end_delta in range(-args.step_pad, args.step_pad + 1, args.step_grid):
                step_start = step_segment[0] + start_delta
                step_end = step_segment[1] + end_delta
                if step_end - step_start < 600:
                    continue
                candidate_segments = {
                    "left": left_segment,
                    "step": (step_start, step_end),
                    "right": right_segment,
                }
                metrics = compute_metrics(image_points, world_points, candidate_segments, nominal)
                segment_rows.append(
                    {
                        "left_start": left_segment[0],
                        "left_end": left_segment[1],
                        "step_start": step_start,
                        "step_end": step_end,
                        "right_start": right_segment[0],
                        "right_end": right_segment[1],
                        **metrics,
                    }
                )

        write_csv(
            output_dir / "validation_segment_sensitivity.csv",
            [
                "left_start",
                "left_end",
                "step_start",
                "step_end",
                "right_start",
                "right_end",
                "measured_step_height_mm",
                "absolute_error_um",
                "left_error_um",
                "right_error_um",
                "edge_gap_um",
                "conservative_error_um",
            ],
            [
                [
                    row["left_start"],
                    row["left_end"],
                    row["step_start"],
                    row["step_end"],
                    row["right_start"],
                    row["right_end"],
                    f"{row['measured_step_height_mm']:.9f}",
                    f"{row['absolute_error_um']:.6f}",
                    f"{row['left_error_um']:.6f}",
                    f"{row['right_error_um']:.6f}",
                    f"{row['edge_gap_um']:.6f}",
                    f"{row['conservative_error_um']:.6f}",
                ]
                for row in segment_rows
            ],
        )

        method_output_dir = Path(tmpdir) / "method_sensitivity_runs"
        method_output_dir.mkdir(parents=True, exist_ok=True)
        method_rows = run_method_sensitivity(config=config, output_dir=method_output_dir)
        write_csv(
            output_dir / "extraction_method_sensitivity.csv",
            [
                "method",
                "measured_step_height_mm",
                "absolute_error_um",
                "left_error_um",
                "right_error_um",
                "edge_gap_um",
                "conservative_error_um",
                "plane_rmse_um",
            ],
            [
                [
                    row["method"],
                    f"{row['measured_step_height_mm']:.9f}",
                    f"{row['absolute_error_um']:.6f}",
                    f"{row['left_error_um']:.6f}",
                    f"{row['right_error_um']:.6f}",
                    f"{row['edge_gap_um']:.6f}",
                    f"{row['conservative_error_um']:.6f}",
                    f"{row['plane_rmse_um']:.6f}",
                ]
                for row in method_rows
            ],
        )

        sensitivity_summary = {
            "measured_step_height_mm": quantiles([row["measured_step_height_mm"] for row in segment_rows]),
            "absolute_error_um": quantiles([row["absolute_error_um"] for row in segment_rows]),
            "conservative_error_um": quantiles([row["conservative_error_um"] for row in segment_rows]),
            "edge_gap_um": quantiles([row["edge_gap_um"] for row in segment_rows]),
        }
        best_conservative = min(segment_rows, key=lambda row: (row["conservative_error_um"], row["edge_gap_um"]))
        worst_conservative = max(segment_rows, key=lambda row: row["conservative_error_um"])

        audit_json = {
            "config_path": str(config_path),
            "configured_segments": {
                "left": list(left_segment),
                "step": list(step_segment),
                "right": list(right_segment),
            },
            "configured_result": asdict(validation),
            "segment_sensitivity_summary": sensitivity_summary,
            "best_conservative_window_in_grid": best_conservative,
            "worst_conservative_window_in_grid": worst_conservative,
            "method_sensitivity": method_rows,
            "notes": [
                "This audit perturbs only the configured step segment while keeping left/right segments fixed.",
                "The perturbation audit is intended to test robustness, not to select a new window from the same validation truth.",
                "A single quantity-block image cannot justify a very-high-confidence sub-micron system claim by itself.",
                "Use conservative_error_um and edge_gap_um as primary guardrails; absolute_error_um alone can be misleading.",
            ],
        }
        (output_dir / "confidence_audit.json").write_text(
            json.dumps(audit_json, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        lines = [
            "# Validation Confidence Audit",
            "",
            "## Configured Result",
            f"- Config path: {config_path}",
            f"- Left segment: {list(left_segment)}",
            f"- Step segment: {list(step_segment)}",
            f"- Right segment: {list(right_segment)}",
            f"- Measured height: {validation.measured_step_height_mm:.9f} mm",
            f"- Absolute error: {validation.absolute_error_um:.3f} um",
            f"- Conservative error: {validation.conservative_absolute_error_um:.3f} um",
            f"- Edge gap: {validation.edge_consistency_um:.3f} um",
            "",
            "## Step-Window Perturbation Audit",
            f"- Grid: configured step segment +/-{args.step_pad} px with {args.step_grid} px spacing",
            f"- Samples: {len(segment_rows)}",
            f"- Absolute error envelope: p05={sensitivity_summary['absolute_error_um']['p05']:.3f} um, median={sensitivity_summary['absolute_error_um']['median']:.3f} um, p95={sensitivity_summary['absolute_error_um']['p95']:.3f} um",
            f"- Conservative error envelope: p05={sensitivity_summary['conservative_error_um']['p05']:.3f} um, median={sensitivity_summary['conservative_error_um']['median']:.3f} um, p95={sensitivity_summary['conservative_error_um']['p95']:.3f} um",
            f"- Edge-gap envelope: p05={sensitivity_summary['edge_gap_um']['p05']:.3f} um, median={sensitivity_summary['edge_gap_um']['median']:.3f} um, p95={sensitivity_summary['edge_gap_um']['p95']:.3f} um",
            f"- Best conservative window in this grid: step=[{best_conservative['step_start']}, {best_conservative['step_end']}], conservative={best_conservative['conservative_error_um']:.3f} um, absolute={best_conservative['absolute_error_um']:.3f} um",
            f"- Worst conservative window in this grid: step=[{worst_conservative['step_start']}, {worst_conservative['step_end']}], conservative={worst_conservative['conservative_error_um']:.3f} um, absolute={worst_conservative['absolute_error_um']:.3f} um",
            "",
            "## Extraction-Method Sensitivity",
        ]
        for row in method_rows:
            lines.append(
                f"- {row['method']}: absolute={row['absolute_error_um']:.3f} um, conservative={row['conservative_error_um']:.3f} um, edge_gap={row['edge_gap_um']:.3f} um, plane_rmse={row['plane_rmse_um']:.3f} um"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "- The configured result may look very strong on this single window, but step-window perturbation shows the reported value is not completely window-invariant.",
                "- Therefore, a single sub-micron average error must not be treated as a very-high-confidence system claim on its own.",
                "- The scientifically safer claim is that this dataset supports low-micron validation when using the audited recipe and when conservative error / edge gap stay small at the same time.",
                f"- Detailed tables: {output_dir / 'validation_segment_sensitivity.csv'} and {output_dir / 'extraction_method_sensitivity.csv'}",
            ]
        )
        (output_dir / "confidence_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Audit output directory: {output_dir}")
    print(f"Configured absolute error: {validation.absolute_error_um:.3f} um")
    print(
        "Step-window conservative envelope: "
        f"p05={sensitivity_summary['conservative_error_um']['p05']:.3f} um, "
        f"median={sensitivity_summary['conservative_error_um']['median']:.3f} um, "
        f"p95={sensitivity_summary['conservative_error_um']['p95']:.3f} um"
    )


if __name__ == "__main__":
    main()
