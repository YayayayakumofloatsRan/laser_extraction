from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_ROOT = SCRIPT_DIR / "仿真实践图片"

TASK_IMAGE_MAP = {
    "任务1-简单直线": [
        IMAGE_ROOT / "任务1-简单直线" / "laserline.png",
    ],
    "任务2-噪声图像": [
        IMAGE_ROOT / "任务2-噪声图像" / "data003.png",
        IMAGE_ROOT / "任务2-噪声图像" / "data003_noisy_sigma50.png",
        IMAGE_ROOT / "任务2-噪声图像" / "data003_noisy_sigma75.png",
        IMAGE_ROOT / "任务2-噪声图像" / "data003_noisy_sigma100.png",
    ],
    "任务3-复杂激光条纹": [
        IMAGE_ROOT / "任务3-复杂激光条纹" / "Pic_20260121221817667_aug0.png",
        IMAGE_ROOT / "任务3-复杂激光条纹" / "Pic_20260121225030094_aug0.png",
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
    gray_raw: np.ndarray
    display_image: np.ndarray
    raw_profile: np.ndarray
    filtered_profile: np.ndarray
    profile_column_index: int
    filter_mode: str
    segment_count: int


def read_image_unicode(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到图像文件: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"无法解码图像文件: {path}")
    return image


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"不支持的图像维度: {image.shape}")


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
        bilateral_size = max(5, min(kernel_size, 15))
        bilateral_size = make_odd(bilateral_size)
        bilateral = cv2.bilateralFilter(image, bilateral_size, 75, 75)
        return cv2.GaussianBlur(bilateral, (kernel_size, kernel_size), 0)
    raise ValueError(f"未知滤波模式: {filter_mode}")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    roi = roi.clipped(gray_raw.shape[1], gray_raw.shape[0])
    roi_gray = gray_raw[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w].copy()
    filtered_roi = apply_filter(roi_gray, filter_mode=filter_mode, kernel_size=blur_kernel)

    rows, cols = filtered_roi.shape
    centers: list[list[float]] = []
    segment_count = max(1, int(segment_count))
    segment_edges = np.linspace(0, cols, segment_count + 1, dtype=int)

    for start, end in zip(segment_edges[:-1], segment_edges[1:]):
        for i in range(start, end):
            col_data = filtered_roi[:, i].astype(np.float32)
            threshold = float(np.max(col_data)) * float(threshold_ratio)
            col_data[col_data < threshold] = 0

            sum_i = float(np.sum(col_data))
            if sum_i <= 0:
                continue

            y_indices = np.arange(rows, dtype=np.float32)
            centroid_y = float(np.sum(y_indices * col_data) / sum_i)
            centers.append([float(roi.x + i), float(roi.y + centroid_y)])

    profile_column_index = cols // 2
    raw_profile = roi_gray[:, profile_column_index].astype(np.float32)
    filtered_profile = filtered_roi[:, profile_column_index].astype(np.float32)
    return np.asarray(centers, dtype=np.float32), roi_gray, filtered_roi, raw_profile, filtered_profile, profile_column_index


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
        raise KeyError(f"未知任务: {task_name}")
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
) -> ExtractionResult:
    image_path = Path(image_path)
    img_raw = read_image_unicode(image_path)
    gray_raw = ensure_grayscale(img_raw)
    display_image = normalize_for_display(gray_raw)

    if auto_roi or roi is None:
        roi = suggest_roi(
            gray_raw,
            blur_kernel=blur_kernel,
            threshold_ratio=threshold_ratio,
            padding=roi_padding,
            filter_mode=filter_mode,
        )

    centers, raw_roi, filtered_roi, raw_profile, filtered_profile, profile_column_index = extract_centers_from_roi(
        gray_raw,
        roi=roi,
        blur_kernel=blur_kernel,
        threshold_ratio=threshold_ratio,
        filter_mode=filter_mode,
        segment_count=segment_count,
    )

    return ExtractionResult(
        image_path=image_path,
        roi=roi,
        centers=centers,
        raw_roi=raw_roi,
        filtered_roi=filtered_roi,
        gray_raw=gray_raw,
        display_image=display_image,
        raw_profile=raw_profile,
        filtered_profile=filtered_profile,
        profile_column_index=profile_column_index,
        filter_mode=filter_mode,
        segment_count=max(1, int(segment_count)),
    )


def plot_profile(raw_profile: np.ndarray, filtered_profile: np.ndarray, column_index: int) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(raw_profile, color="gray", alpha=0.8, label="raw")
    plt.plot(filtered_profile, color="blue", label="filtered")
    plt.title(f"Intensity Profile of ROI Column {column_index}")
    plt.xlabel("Y Coordinate")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_result_window(result: ExtractionResult) -> None:
    result_image = overlay_centers(result.display_image, result.centers, result.roi)
    win_name = "Extraction Result"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_roi_with_opencv(display_image: np.ndarray) -> ROI:
    win_name = "Select ROI and press ENTER"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI(win_name, display_image, fromCenter=False)
    cv2.destroyWindow(win_name)
    if w <= 0 or h <= 0:
        raise ValueError("未选择有效 ROI")
    return ROI(int(x), int(y), int(w), int(h))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="激光中心线提取脚本")
    parser.add_argument("--image", type=Path, default=None, help="待处理图像路径")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "results.csv", help="CSV 输出路径")
    parser.add_argument("--blur-kernel", type=int, default=21, help="高斯滤波核大小")
    parser.add_argument("--threshold-ratio", type=float, default=0.3, help="按列动态阈值比例")
    parser.add_argument("--auto-roi", action="store_true", help="自动估计 ROI，不打开手动框选窗口")
    parser.add_argument("--roi-padding", type=int, default=20, help="自动 ROI 外扩像素")
    parser.add_argument("--show-plot", action="store_true", help="显示中间列强度曲线")
    parser.add_argument("--show-window", action="store_true", help="显示 OpenCV 结果窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = args.image or load_task_images("任务1-简单直线")[0]
    img_raw = read_image_unicode(image_path)
    gray_raw = ensure_grayscale(img_raw)
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
        filter_mode="gaussian",
        segment_count=1,
    )

    output_path = save_centers_csv(result.centers, args.output)
    print(f"实验完成，中心点已保存至: {output_path}")
    print(f"共提取 {len(result.centers)} 个中心点，ROI={result.roi}")

    if args.show_plot:
        plot_profile(result.raw_profile, result.filtered_profile, result.profile_column_index)
    if args.show_window:
        show_result_window(result)


if __name__ == "__main__":
    main()
