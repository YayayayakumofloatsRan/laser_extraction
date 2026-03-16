from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 该文件是项目的核心算法模块。
# 主要职责：
# 1. 兼容中文路径读取图像；
# 2. 对 ROI 内条纹做滤波、去背景和中心提取；
# 3. 同时向命令行脚本和 notebook 交互界面提供统一接口。

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
    """矩形 ROI 定义，所有后续处理都围绕该区域进行。"""

    x: int
    y: int
    w: int
    h: int

    def clipped(self, width: int, height: int) -> "ROI":
        # 将用户给定或自动生成的 ROI 裁剪到图像内部，防止索引越界。
        x = int(np.clip(self.x, 0, max(width - 1, 0)))
        y = int(np.clip(self.y, 0, max(height - 1, 0)))
        max_w = max(width - x, 1)
        max_h = max(height - y, 1)
        w = int(np.clip(self.w, 1, max_w))
        h = int(np.clip(self.h, 1, max_h))
        return ROI(x=x, y=y, w=w, h=h)


@dataclass
class ExtractionResult:
    """单张图处理后的完整结果集合。

    notebook 需要同时展示原始 ROI、滤波后 ROI、去背景后 ROI、
    列向量曲线和最终中心点，因此这里统一封装所有中间量。
    """

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
    """兼容 Windows 中文路径读取图像。

    直接使用 cv2.imread 读取中文路径时，OpenCV 在部分环境下会失败，
    因此这里使用 np.fromfile + cv2.imdecode 的组合方式。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """将输入图像统一转换成灰度图。"""
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_for_display(gray_raw: np.ndarray) -> np.ndarray:
    """将灰度图归一化到 0~255，便于在 notebook 和 OpenCV 窗口中显示。"""
    return cv2.normalize(gray_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def make_odd(kernel_size: int) -> int:
    """滤波核必须为正奇数，这里统一做约束。"""
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def apply_filter(image: np.ndarray, filter_mode: str = "gaussian", kernel_size: int = 21) -> np.ndarray:
    """根据配置应用不同的去噪组合。

    说明：
    - Gaussian：常规平滑，适合压低随机噪声；
    - Median：对脉冲型异常点更稳；
    - Gaussian -> Median / Median -> Gaussian：用于比较不同顺序的影响；
    - Bilateral -> Gaussian：先保边去噪，再做整体平滑。
    """
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
    """估计 ROI 内的慢变化背景分量。

    使用大核形态学开运算得到背景，再从滤波结果中相减，
    能有效减弱任务 2 中背景噪声和宽尾部对中心位置的拖偏。
    """
    kernel_size = make_odd(kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def smooth_centerline(centers: np.ndarray, kernel_size: int = 15, max_deviation: float = 12.0) -> np.ndarray:
    """对逐列中心线结果做鲁棒平滑。

    核心思路是使用滑动中值作为局部参考线，
    对偏离过大的点进行回填，抑制噪声导致的孤立跳点。
    """
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
    """通过三点抛物线插值估计峰值的亚像素偏移。"""
    denom = y_minus - 2.0 * y0 + y_plus
    if abs(denom) < 1e-6:
        return 0.0
    offset = 0.5 * (y_minus - y_plus) / denom
    return float(np.clip(offset, -1.0, 1.0))


def steger_like_center(column: np.ndarray, peak_idx: int) -> float:
    """Steger 风格的轻量近似实现。

    严格的 Steger 法依赖 Hessian 和法向信息；
    这里保留其“局部二次曲线求脊线中心”的思想，
    用于 notebook 中和质心法做方法对比。
    """
    if peak_idx <= 0 or peak_idx >= len(column) - 1:
        return float(peak_idx)
    offset = quadratic_subpixel_offset(
        float(column[peak_idx - 1]),
        float(column[peak_idx]),
        float(column[peak_idx + 1]),
    )
    return float(peak_idx + offset)


def gaussian_fit_center(column: np.ndarray, peak_idx: int, half_window: int = 3) -> float:
    """函数拟合法：在主峰附近做高斯拟合，返回拟合中心。"""
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
    """自动估计 ROI。

    先进行基础滤波，再通过相对最大值阈值找到亮条纹的大致包围区域，
    最后加上 padding，防止 ROI 卡得过紧。
    """
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
    """在 ROI 内逐列提取条纹中心。

    支持的方法包括：
    - global_centroid：整列质心；
    - peak_window_centroid：主峰附近窗口质心；
    - gaussian_fit：主峰附近高斯拟合；
    - steger_like：局部二次插值脊线定位。

    任务 3 的“分段处理”也是在这里实现的：把 ROI 按列拆分成多个子区间，
    再分别执行相同的逐列提取逻辑。
    """
    roi = roi.clipped(gray_raw.shape[1], gray_raw.shape[0])
    # 先得到原始 ROI，再做滤波和背景抑制。
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

            # 对局部方法，只在主峰附近取小窗口；对全局质心法则保留整列。
            if extraction_method in {"peak_window_centroid", "gaussian_fit", "steger_like"}:
                peak_idx = int(np.argmax(col))
                lo = max(0, peak_idx - peak_window_half_height)
                hi = min(rows, peak_idx + peak_window_half_height + 1)
                work = col[lo:hi].copy()
                y_idx = np.arange(lo, hi, dtype=np.float32)
            else:
                work = col.copy()
                y_idx = y_all

            # 用相对阈值进一步压掉弱背景响应。
            peak = float(np.max(work))
            if peak <= 0:
                continue
            work[work < peak * float(threshold_ratio)] = 0
            total = float(np.sum(work))
            if total <= 0:
                continue

            if extraction_method == "global_centroid" or extraction_method == "peak_window_centroid":
                # 质心法：以亮度作为权重求加权平均位置。
                centroid_y = float(np.sum(y_idx * work) / total)
            elif extraction_method == "gaussian_fit":
                # 函数法：把主峰附近当作高斯分布来拟合。
                local_peak = int(np.argmax(work))
                centroid_y = gaussian_fit_center(work, local_peak, half_window=min(3, len(work) // 2))
                if extraction_method != "global_centroid":
                    centroid_y += float(lo)
            elif extraction_method == "steger_like":
                # Steger 风格法：用二次插值估计脊线中心。
                local_peak = int(np.argmax(work))
                centroid_y = steger_like_center(work, local_peak)
                centroid_y += float(lo)
            else:
                raise ValueError(f"Unknown extraction method: {extraction_method}")

            centers.append([float(roi.x + i), float(roi.y + centroid_y)])

    centers_array = np.asarray(centers, dtype=np.float32)
    # 局部峰值类方法更容易出现个别列跳点，因此统一做一次平滑。
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
    """将 ROI 和中心点叠加到显示图上。"""
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
    """构造手动 ROI 选择窗口使用的预览图。

    当前保持原图内容不变，只额外绘制完整边框，
    让用户在边缘位置选框时更容易看清图像边界。
    """
    preview = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    height, width = display_image.shape[:2]
    cv2.rectangle(preview, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)
    return preview


def save_centers_csv(centers: np.ndarray, output_path: str | Path) -> Path:
    """保存中心点坐标到 CSV。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        writer.writerows(centers.tolist())
    return output_path


def load_task_images(task_name: str) -> list[Path]:
    """返回指定任务下的全部图像路径。"""
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
    """单张图像的完整处理入口。

    该函数会串起：
    读图 -> 灰度化 -> ROI 获取 -> 滤波/去背景 -> 中心提取 -> 结果打包
    """
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
    """绘制某一列在不同阶段的能量分布曲线。"""
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
    """用 OpenCV 弹窗展示最终叠加结果。"""
    result_image = overlay_centers(result.display_image, result.centers, result.roi)
    cv2.namedWindow("Extraction Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Extraction Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_roi_with_opencv(display_image: np.ndarray) -> ROI:
    """打开 OpenCV 原生 ROI 选择窗口。

    这里会主动放大窗口并启用十字光标，
    提高手动框选时对 ROI 范围的辨识度。
    """
    window_name = "Select ROI and press ENTER"
    height, width = display_image.shape[:2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(width, 1600), min(height, 900))
    x, y, w, h = cv2.selectROI(window_name, display_image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    if w <= 0 or h <= 0:
        raise ValueError("No valid ROI selected")
    return ROI(int(x), int(y), int(w), int(h))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
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
    """命令行入口，默认处理任务 1 的第一张图。"""
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
