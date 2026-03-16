from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_ROOT = SCRIPT_DIR / "\u4eff\u771f\u5b9e\u8df5\u56fe\u7247"

TASK1 = "\u4efb\u52a11-\u7b80\u5355\u76f4\u7ebf"
TASK2 = "\u4efb\u52a12-\u566a\u58f0\u56fe\u50cf"
TASK3 = "\u4efb\u52a13-\u590d\u6742\u6fc0\u5149\u6761\u7eb9"

TASK_IMAGE_MAP = {
    TASK1: [IMAGE_ROOT / TASK1 / "laserline.png"],
    TASK2: [
        IMAGE_ROOT / TASK2 / "data003.png",
        IMAGE_ROOT / TASK2 / "data003_noisy_sigma50.png",
        IMAGE_ROOT / TASK2 / "data003_noisy_sigma75.png",
        IMAGE_ROOT / TASK2 / "data003_noisy_sigma100.png",
    ],
    TASK3: [
        IMAGE_ROOT / TASK3 / "Pic_20260121221817667_aug0.png",
        IMAGE_ROOT / TASK3 / "Pic_20260121225030094_aug0.png",
    ],
}


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    def clipped(self, width: int, height: int) -> "ROI":
        x = int(np.clip(self.x, 0, max(width - 1, 0)))
        y = int(np.clip(self.y, 0, max(height - 1, 0)))
        max_w = max(width - x, 1)
        max_h = max(height - y, 1)
        w = int(np.clip(self.w, 1, max_w))
        h = int(np.clip(self.h, 1, max_h))
        return ROI(x=x, y=y, w=w, h=h)


@dataclass
class ExtractionResult:
    image_path: Path
    roi: ROI
    centers: np.ndarray
    raw_roi: np.ndarray
    filtered_roi: np.ndarray
    enhanced_roi: np.ndarray
    gray_raw: np.ndarray
    display_image: np.ndarray
    raw_profile: np.ndarray
    filtered_profile: np.ndarray
    enhanced_profile: np.ndarray
    profile_column_index: int
    filter_mode: str
    segment_count: int
    extraction_method: str


def read_image_unicode(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_for_display(gray_raw: np.ndarray) -> np.ndarray:
    return cv2.normalize(gray_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def make_odd(kernel_size: int) -> int:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def apply_filter(image: np.ndarray, filter_mode: str = "gaussian", kernel_size: int = 21) -> np.ndarray:
    kernel_size = make_odd(kernel_size)
    if filter_mode == "none":
        return image.copy()
    if filter_mode == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if filter_mode == "median":
        return cv2.medianBlur(image, kernel_size)
    if filter_mode == "gaussian+median":
        return cv2.medianBlur(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0), kernel_size)
    if filter_mode == "median+gaussian":
        return cv2.GaussianBlur(cv2.medianBlur(image, kernel_size), (kernel_size, kernel_size), 0)
    if filter_mode == "bilateral+gaussian":
        bilateral_size = make_odd(max(5, min(kernel_size, 15)))
        bilateral = cv2.bilateralFilter(image, bilateral_size, 75, 75)
        return cv2.GaussianBlur(bilateral, (kernel_size, kernel_size), 0)
    raise ValueError(f"Unknown filter mode: {filter_mode}")


def estimate_background(image: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    kernel_size = make_odd(kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def smooth_centerline(centers: np.ndarray, kernel_size: int = 15, max_deviation: float = 12.0) -> np.ndarray:
    if len(centers) == 0:
        return centers
    kernel_size = make_odd(max(3, kernel_size))
    y = centers[:, 1].astype(np.float32)
    pad = kernel_size // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    median_y = np.array([np.median(padded[i : i + kernel_size]) for i in range(len(y))], dtype=np.float32)
    smoothed = centers.copy()
    mask = np.abs(y - median_y) > float(max_deviation)
    smoothed[mask, 1] = median_y[mask]
    return smoothed


def quadratic_subpixel_offset(y_minus: float, y0: float, y_plus: float) -> float:
    denom = y_minus - 2.0 * y0 + y_plus
    if abs(denom) < 1e-6:
        return 0.0
    offset = 0.5 * (y_minus - y_plus) / denom
    return float(np.clip(offset, -1.0, 1.0))


def steger_like_center(column: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(column) - 1:
        return float(peak_idx)
    offset = quadratic_subpixel_offset(
        float(column[peak_idx - 1]),
        float(column[peak_idx]),
        float(column[peak_idx + 1]),
    )
    return float(peak_idx + offset)


def gaussian_fit_center(column: np.ndarray, peak_idx: int, half_window: int = 3) -> float:
    lo = max(0, peak_idx - half_window)
    hi = min(len(column), peak_idx + half_window + 1)
    y_idx = np.arange(lo, hi, dtype=np.float64)
    values = column[lo:hi].astype(np.float64)
    values = np.maximum(values, 1e-6)
    if len(y_idx) < 3:
        return float(peak_idx)
    coeffs = np.polyfit(y_idx, np.log(values), deg=2)
    a, b, _ = coeffs
    if abs(a) < 1e-9:
        return float(peak_idx)
    center = -b / (2.0 * a)
    return float(np.clip(center, lo, hi - 1))


def suggest_roi(
    gray_raw: np.ndarray,
    blur_kernel: int = 21,
    threshold_ratio: float = 0.3,
    padding: int = 20,
    filter_mode: str = "gaussian",
) -> ROI:
    filtered = apply_filter(gray_raw, filter_mode=filter_mode, kernel_size=blur_kernel)
    max_value = float(np.max(filtered))
    if max_value <= 0:
        return ROI(0, 0, gray_raw.shape[1], gray_raw.shape[0])
    binary = filtered >= max_value * float(threshold_ratio)
    points = np.column_stack(np.where(binary))
    if len(points) == 0:
        return ROI(0, 0, gray_raw.shape[1], gray_raw.shape[0])
    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)
    x = max(int(x_min) - padding, 0)
    y = max(int(y_min) - padding, 0)
    w = min(int(x_max) + padding, gray_raw.shape[1] - 1) - x + 1
    h = min(int(y_max) + padding, gray_raw.shape[0] - 1) - y + 1
    return ROI(x, y, w, h).clipped(gray_raw.shape[1], gray_raw.shape[0])


def extract_centers_from_roi(
    gray_raw: np.ndarray,
    roi: ROI,
    blur_kernel: int = 21,
    threshold_ratio: float = 0.3,
    filter_mode: str = "gaussian",
    segment_count: int = 1,
    extraction_method: str = "global_centroid",
    background_kernel: int = 51,
    peak_window_half_height: int = 22,
    smooth_kernel_size: int = 15,
    smooth_max_deviation: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    roi = roi.clipped(gray_raw.shape[1], gray_raw.shape[0])
    raw_roi = gray_raw[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w].copy()
    filtered_roi = apply_filter(raw_roi, filter_mode=filter_mode, kernel_size=blur_kernel)
    background = estimate_background(filtered_roi, kernel_size=background_kernel)
    enhanced_roi = cv2.subtract(filtered_roi, background)

    rows, cols = enhanced_roi.shape
    centers: list[list[float]] = []
    y_all = np.arange(rows, dtype=np.float32)
    segment_edges = np.linspace(0, cols, max(1, int(segment_count)) + 1, dtype=int)
    peak_window_half_height = max(2, int(peak_window_half_height))

    for start, end in zip(segment_edges[:-1], segment_edges[1:]):
        for i in range(start, end):
            col = enhanced_roi[:, i].astype(np.float32)
            peak_idx = int(np.argmax(col))

            if extraction_method in {"peak_window_centroid", "gaussian_fit", "steger_like"}:
                peak_idx = int(np.argmax(col))
                lo = max(0, peak_idx - peak_window_half_height)
                hi = min(rows, peak_idx + peak_window_half_height + 1)
                work = col[lo:hi].copy()
                y_idx = np.arange(lo, hi, dtype=np.float32)
            else:
                work = col.copy()
                y_idx = y_all

            peak = float(np.max(work))
            if peak <= 0:
                continue
            work[work < peak * float(threshold_ratio)] = 0
            total = float(np.sum(work))
            if total <= 0:
                continue

            if extraction_method == "global_centroid" or extraction_method == "peak_window_centroid":
                centroid_y = float(np.sum(y_idx * work) / total)
            elif extraction_method == "gaussian_fit":
                local_peak = int(np.argmax(work))
                centroid_y = gaussian_fit_center(work, local_peak, half_window=min(3, len(work) // 2))
                if extraction_method != "global_centroid":
                    centroid_y += float(lo)
            elif extraction_method == "steger_like":
                local_peak = int(np.argmax(work))
                centroid_y = steger_like_center(work, local_peak)
                centroid_y += float(lo)
            else:
                raise ValueError(f"Unknown extraction method: {extraction_method}")

            centers.append([float(roi.x + i), float(roi.y + centroid_y)])

    centers_array = np.asarray(centers, dtype=np.float32)
    if extraction_method in {"peak_window_centroid", "gaussian_fit", "steger_like"}:
        centers_array = smooth_centerline(
            centers_array,
            kernel_size=smooth_kernel_size,
            max_deviation=smooth_max_deviation,
        )

    profile_column_index = cols // 2
    raw_profile = raw_roi[:, profile_column_index].astype(np.float32)
    filtered_profile = filtered_roi[:, profile_column_index].astype(np.float32)
    enhanced_profile = enhanced_roi[:, profile_column_index].astype(np.float32)
    return (
        centers_array,
        raw_roi,
        filtered_roi,
        enhanced_roi,
        raw_profile,
        filtered_profile,
        enhanced_profile,
        profile_column_index,
    )


def overlay_centers(display_image: np.ndarray, centers: np.ndarray, roi: ROI | None = None) -> np.ndarray:
    result = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    if roi is not None:
        cv2.rectangle(
            result,
            (int(roi.x), int(roi.y)),
            (int(roi.x + roi.w - 1), int(roi.y + roi.h - 1)),
            (0, 255, 255),
            2,
        )
    for x, y in centers:
        cv2.circle(result, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)
    return result


def build_manual_roi_preview(display_image: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(display_image, 60, 180)
    preview[edges > 0] = (0, 255, 0)
    height, width = display_image.shape[:2]
    cv2.line(preview, (width // 2, 0), (width // 2, height - 1), (255, 255, 0), 1)
    cv2.line(preview, (0, height // 2), (width - 1, height // 2), (255, 255, 0), 1)
    return preview


def save_centers_csv(centers: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        writer.writerows(centers.tolist())
    return output_path


def load_task_images(task_name: str) -> list[Path]:
    if task_name not in TASK_IMAGE_MAP:
        raise KeyError(f"Unknown task: {task_name}")
    return [path for path in TASK_IMAGE_MAP[task_name] if path.exists()]


def process_image(
    image_path: str | Path,
    roi: ROI | None = None,
    blur_kernel: int = 21,
    threshold_ratio: float = 0.3,
    auto_roi: bool = False,
    roi_padding: int = 20,
    filter_mode: str = "gaussian",
    segment_count: int = 1,
    extraction_method: str = "global_centroid",
    background_kernel: int = 51,
    peak_window_half_height: int = 22,
    smooth_kernel_size: int = 15,
    smooth_max_deviation: float = 12.0,
) -> ExtractionResult:
    image_path = Path(image_path)
    gray_raw = ensure_grayscale(read_image_unicode(image_path))
    display_image = normalize_for_display(gray_raw)
    if auto_roi or roi is None:
        roi = suggest_roi(
            gray_raw,
            blur_kernel=blur_kernel,
            threshold_ratio=threshold_ratio,
            padding=roi_padding,
            filter_mode=filter_mode,
        )
    (
        centers,
        raw_roi,
        filtered_roi,
        enhanced_roi,
        raw_profile,
        filtered_profile,
        enhanced_profile,
        profile_column_index,
    ) = extract_centers_from_roi(
        gray_raw,
        roi=roi,
        blur_kernel=blur_kernel,
        threshold_ratio=threshold_ratio,
        filter_mode=filter_mode,
        segment_count=segment_count,
        extraction_method=extraction_method,
        background_kernel=background_kernel,
        peak_window_half_height=peak_window_half_height,
        smooth_kernel_size=smooth_kernel_size,
        smooth_max_deviation=smooth_max_deviation,
    )
    return ExtractionResult(
        image_path=image_path,
        roi=roi,
        centers=centers,
        raw_roi=raw_roi,
        filtered_roi=filtered_roi,
        enhanced_roi=enhanced_roi,
        gray_raw=gray_raw,
        display_image=display_image,
        raw_profile=raw_profile,
        filtered_profile=filtered_profile,
        enhanced_profile=enhanced_profile,
        profile_column_index=profile_column_index,
        filter_mode=filter_mode,
        segment_count=max(1, int(segment_count)),
        extraction_method=extraction_method,
    )


def plot_profile(raw_profile: np.ndarray, filtered_profile: np.ndarray, enhanced_profile: np.ndarray, column_index: int) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(raw_profile, color="gray", alpha=0.8, label="raw")
    plt.plot(filtered_profile, color="royalblue", label="filtered")
    plt.plot(enhanced_profile, color="crimson", label="background-suppressed")
    plt.title(f"Intensity Profile of ROI Column {column_index}")
    plt.xlabel("Y Coordinate")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_result_window(result: ExtractionResult) -> None:
    result_image = overlay_centers(result.display_image, result.centers, result.roi)
    cv2.namedWindow("Extraction Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Extraction Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_roi_with_opencv(display_image: np.ndarray) -> ROI:
    cv2.namedWindow("Select ROI and press ENTER", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("Select ROI and press ENTER", display_image, fromCenter=False)
    cv2.destroyWindow("Select ROI and press ENTER")
    if w <= 0 or h <= 0:
        raise ValueError("No valid ROI selected")
    return ROI(int(x), int(y), int(w), int(h))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laser centerline extraction")
    parser.add_argument("--image", type=Path, default=None, help="Input image path")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "results.csv", help="CSV output path")
    parser.add_argument("--blur-kernel", type=int, default=21, help="Blur kernel size")
    parser.add_argument("--threshold-ratio", type=float, default=0.25, help="Threshold ratio")
    parser.add_argument("--auto-roi", action="store_true", help="Use auto ROI instead of OpenCV ROI window")
    parser.add_argument("--roi-padding", type=int, default=20, help="Auto ROI padding")
    parser.add_argument("--show-plot", action="store_true", help="Show intensity profile plot")
    parser.add_argument("--show-window", action="store_true", help="Show OpenCV result window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image or load_task_images(TASK1)[0]
    gray_raw = ensure_grayscale(read_image_unicode(image_path))
    display_image = normalize_for_display(gray_raw)
    if args.auto_roi:
        roi = suggest_roi(
            gray_raw,
            blur_kernel=args.blur_kernel,
            threshold_ratio=args.threshold_ratio,
            padding=args.roi_padding,
            filter_mode="gaussian",
        )
    else:
        roi = select_roi_with_opencv(display_image)

    result = process_image(
        image_path=image_path,
        roi=roi,
        blur_kernel=args.blur_kernel,
        threshold_ratio=args.threshold_ratio,
        auto_roi=False,
        roi_padding=args.roi_padding,
        filter_mode="gaussian+median",
        extraction_method="peak_window_centroid",
    )
    output_path = save_centers_csv(result.centers, args.output)
    print(f"Done. Saved center points to: {output_path}")
    print(f"Extracted {len(result.centers)} points, ROI={result.roi}")
    if args.show_plot:
        plot_profile(result.raw_profile, result.filtered_profile, result.enhanced_profile, result.profile_column_index)
    if args.show_window:
        show_result_window(result)


if __name__ == "__main__":
    main()
