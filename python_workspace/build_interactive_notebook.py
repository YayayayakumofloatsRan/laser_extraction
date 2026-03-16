from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent / "laser_extraction_interactive.ipynb"


def split_lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": split_lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(text),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Experiment: Interactive Laser Centerline Extraction Notebook

            Objective:
            - Build an interactive notebook on top of the existing centroid-based workflow in `laser_extraction.py`.
            - Run Task 1, Task 2, and Task 3 independently.
            - Show step-by-step figures and short analysis for each task.

            ## Scope

            - Task 1: center extraction for a simple straight laser stripe.
            - Task 2: denoising plus center extraction for noisy laser stripes.
            - Task 3: center extraction for complex laser stripe shapes with optional segmented processing.
            """
        ),
        markdown_cell(
            """
            ## Usage

            - `ROI Mode` supports three strategies:
              - `Fixed ROI`: use a preset ROI for the selected task.
              - `Manual ROI`: adjust `x/y/w/h` with sliders or open the OpenCV ROI window.
              - `Auto ROI`: estimate ROI automatically from the bright stripe.
            - `Filter Mode` is mainly for Task 2, to compare denoising combinations.
            - `Segments` is mainly for Task 3, to process complex stripes by sub-regions.
            - `Preview Current Setup` shows step-by-step figures and a short analysis.
            - `Run Current Image` processes only the selected image.
            - `Run All Images In Task` exports CSV files for every image in the selected task.
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import importlib.util
            import sys
            from pathlib import Path

            REQUIRED_MODULES = {
                "numpy": "numpy",
                "matplotlib": "matplotlib",
                "cv2": "opencv-python",
                "ipywidgets": "ipywidgets",
                "IPython": "ipython",
            }

            missing = [package for module_name, package in REQUIRED_MODULES.items() if importlib.util.find_spec(module_name) is None]
            if missing:
                install_cmd = f"{sys.executable} -m pip install -r requirements-notebook.txt"
                raise ModuleNotFoundError(
                    "Missing dependencies: "
                    + ", ".join(missing)
                    + "\\nRun this first in the current directory:\\n"
                    + install_cmd
                )

            WORKSPACE_CANDIDATES = [
                Path.cwd(),
                Path.cwd() / "python_workspace",
                Path(r"D:\\laser_extraction\\python_workspace"),
            ]

            WORKSPACE_DIR = None
            for candidate in WORKSPACE_CANDIDATES:
                if (candidate / "laser_extraction.py").exists():
                    WORKSPACE_DIR = candidate
                    break

            if WORKSPACE_DIR is None:
                raise FileNotFoundError("Cannot find laser_extraction.py. Make sure the notebook is opened inside this project.")

            if str(WORKSPACE_DIR) not in sys.path:
                sys.path.insert(0, str(WORKSPACE_DIR))

            print(f"Workspace: {WORKSPACE_DIR}")
            print("Dependency check passed")
            """
        ),
        code_cell(
            """
            import csv
            from pathlib import Path

            import ipywidgets as widgets
            import matplotlib.pyplot as plt
            import numpy as np
            from IPython.display import Markdown, clear_output, display
            from IPython import get_ipython

            import laser_extraction as le

            ip = get_ipython()
            if ip is not None:
                ip.run_line_magic("matplotlib", "inline")

            OUTPUT_DIR = WORKSPACE_DIR / "notebook_outputs"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            TASK_OPTIONS = list(le.TASK_IMAGE_MAP.keys())
            TASK_DISPLAY_NAMES = {
                "任务1-简单直线": "Task 1 - Straight Laser Stripe",
                "任务2-噪声图像": "Task 2 - Noisy Laser Stripe",
                "任务3-复杂激光条纹": "Task 3 - Complex Laser Stripe",
            }
            FILTER_OPTIONS = [
                ("none", "none"),
                ("gaussian", "gaussian"),
                ("median", "median"),
                ("gaussian+median", "gaussian+median"),
                ("median+gaussian", "median+gaussian"),
                ("bilateral+gaussian", "bilateral+gaussian"),
            ]
            ROI_MODE_OPTIONS = [("Fixed ROI", "fixed"), ("Manual ROI", "manual"), ("Auto ROI", "auto")]
            FIXED_ROI_BY_TASK = {
                "任务1-简单直线": le.ROI(0, 1653, 5496, 183),
                "任务2-噪声图像": le.ROI(0, 2189, 5496, 341),
                "任务3-复杂激光条纹": le.ROI(0, 255, 512, 140),
            }
            IMAGE_CACHE = {}

            plt.rcParams["figure.figsize"] = (15, 5)
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman"]
            plt.rcParams["font.sans-serif"] = ["Times New Roman"]
            """
        ),
        code_cell(
            """
            def get_task_images(task_name: str) -> list[Path]:
                return le.load_task_images(task_name)


            def get_image_meta(image_path: Path) -> dict:
                image_path = Path(image_path)
                if image_path not in IMAGE_CACHE:
                    img_raw = le.read_image_unicode(image_path)
                    gray_raw = le.ensure_grayscale(img_raw)
                    IMAGE_CACHE[image_path] = {
                        "gray": gray_raw,
                        "display": le.normalize_for_display(gray_raw),
                        "height": int(gray_raw.shape[0]),
                        "width": int(gray_raw.shape[1]),
                    }
                return IMAGE_CACHE[image_path]


            def build_result_path(task_name: str, image_path: Path) -> Path:
                task_dir = OUTPUT_DIR / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                return task_dir / f"{image_path.stem}_centers.csv"


            def clip_roi_to_image(roi: le.ROI, image_path: Path) -> le.ROI:
                meta = get_image_meta(image_path)
                return roi.clipped(meta["width"], meta["height"])


            def set_roi_sliders(roi: le.ROI, image_path: Path) -> None:
                meta = get_image_meta(image_path)
                x_slider.max = max(meta["width"] - 1, 0)
                y_slider.max = max(meta["height"] - 1, 0)
                w_slider.max = max(meta["width"], 1)
                h_slider.max = max(meta["height"], 1)

                clipped = roi.clipped(meta["width"], meta["height"])
                x_slider.value = clipped.x
                y_slider.value = clipped.y
                w_slider.value = clipped.w
                h_slider.value = clipped.h


            def get_roi_for_image(image_path: Path) -> le.ROI:
                mode = roi_mode_dropdown.value
                if mode == "fixed":
                    return clip_roi_to_image(FIXED_ROI_BY_TASK[task_dropdown.value], image_path)
                if mode == "auto":
                    meta = get_image_meta(image_path)
                    return le.suggest_roi(
                        meta["gray"],
                        blur_kernel=blur_slider.value,
                        threshold_ratio=threshold_slider.value,
                        padding=padding_slider.value,
                        filter_mode=filter_dropdown.value,
                    )
                return clip_roi_to_image(
                    le.ROI(
                        x=int(x_slider.value),
                        y=int(y_slider.value),
                        w=int(w_slider.value),
                        h=int(h_slider.value),
                    ),
                    image_path,
                )


            def describe_result(task_name: str, result: le.ExtractionResult) -> str:
                coverage = len(result.centers) / max(result.roi.w, 1)
                raw_std = float(np.std(result.raw_profile))
                filtered_std = float(np.std(result.filtered_profile))
                profile_delta = raw_std - filtered_std
                lines = [
                    f"**Task**: `{TASK_DISPLAY_NAMES[task_name]}`",
                    f"**Image**: `{result.image_path.name}`",
                    f"**ROI**: `{result.roi}`",
                    f"**Filter Mode**: `{result.filter_mode}`",
                    f"**Point Count**: `{len(result.centers)}`",
                    f"**Column Coverage**: `{coverage:.3f}`",
                ]
                if task_name == "任务2-噪声图像":
                    lines.append(f"**Column Profile Std**: `raw={raw_std:.2f} -> filtered={filtered_std:.2f}`")
                    if profile_delta > 0:
                        lines.append("**Analysis**: after filtering, the column profile is smoother and the energy distribution is more concentrated, which is suitable for centroid extraction.")
                    else:
                        lines.append("**Analysis**: the current filter combination does not reduce profile fluctuation clearly; try a stronger filter chain or adjust the ROI.")
                elif task_name == "任务3-复杂激光条纹":
                    lines.append(f"**Segments**: `{result.segment_count}`")
                    if result.segment_count > 1:
                        lines.append("**Analysis**: segmented processing is enabled, which can reduce local bias for complex stripe shapes.")
                    else:
                        lines.append("**Analysis**: whole-ROI processing is used; if local shape variation is large, try increasing the segment count.")
                else:
                    lines.append("**Analysis**: the straight stripe is stable, and the centroid method usually gives a continuous centerline directly.")
                return "  \\\\n".join(lines)
            """
        ),
        code_cell(
            """
            task_dropdown = widgets.Dropdown(
                options=[(TASK_DISPLAY_NAMES[name], name) for name in TASK_OPTIONS],
                value=TASK_OPTIONS[0],
                description="Task",
                layout=widgets.Layout(width="360px"),
            )

            image_dropdown = widgets.Dropdown(
                description="Image",
                layout=widgets.Layout(width="560px"),
            )

            roi_mode_dropdown = widgets.Dropdown(
                options=ROI_MODE_OPTIONS,
                value="fixed",
                description="ROI Mode",
                layout=widgets.Layout(width="360px"),
            )

            filter_dropdown = widgets.Dropdown(
                options=FILTER_OPTIONS,
                value="gaussian",
                description="Filter",
                layout=widgets.Layout(width="360px"),
            )

            blur_slider = widgets.IntSlider(
                value=21,
                min=3,
                max=61,
                step=2,
                description="Kernel",
                continuous_update=False,
                layout=widgets.Layout(width="560px"),
            )

            threshold_slider = widgets.FloatSlider(
                value=0.30,
                min=0.05,
                max=0.90,
                step=0.01,
                description="Threshold",
                readout_format=".2f",
                continuous_update=False,
                layout=widgets.Layout(width="560px"),
            )

            padding_slider = widgets.IntSlider(
                value=20,
                min=0,
                max=200,
                step=2,
                description="Padding",
                continuous_update=False,
                layout=widgets.Layout(width="560px"),
            )

            segment_slider = widgets.IntSlider(
                value=1,
                min=1,
                max=8,
                step=1,
                description="Segments",
                continuous_update=False,
                layout=widgets.Layout(width="560px"),
            )

            x_slider = widgets.IntSlider(description="ROI x", continuous_update=False, layout=widgets.Layout(width="560px"))
            y_slider = widgets.IntSlider(description="ROI y", continuous_update=False, layout=widgets.Layout(width="560px"))
            w_slider = widgets.IntSlider(description="ROI w", min=1, continuous_update=False, layout=widgets.Layout(width="560px"))
            h_slider = widgets.IntSlider(description="ROI h", min=1, continuous_update=False, layout=widgets.Layout(width="560px"))

            apply_fixed_roi_button = widgets.Button(description="Load Fixed ROI", button_style="info")
            apply_auto_roi_button = widgets.Button(description="Estimate Auto ROI", button_style="info")
            manual_roi_button = widgets.Button(description="Open Manual ROI Window", button_style="info")
            preview_button = widgets.Button(description="Preview Current Setup", button_style="warning")
            run_image_button = widgets.Button(description="Run Current Image", button_style="success")
            run_task_button = widgets.Button(description="Run All Images In Task", button_style="success")

            preview_output = widgets.Output()
            run_output = widgets.Output()
            manual_roi_output = widgets.Output()
            """
        ),
        code_cell(
            """
            def refresh_image_dropdown(*_args) -> None:
                task_name = task_dropdown.value
                image_paths = get_task_images(task_name)
                image_dropdown.options = [(path.name, str(path)) for path in image_paths]
                if image_paths:
                    image_dropdown.value = str(image_paths[0])


            def load_fixed_roi(*_args) -> None:
                if not image_dropdown.value:
                    return
                set_roi_sliders(FIXED_ROI_BY_TASK[task_dropdown.value], Path(image_dropdown.value))


            def load_auto_roi(*_args) -> None:
                if not image_dropdown.value:
                    return
                image_path = Path(image_dropdown.value)
                roi = get_roi_for_image(image_path) if roi_mode_dropdown.value == "auto" else le.suggest_roi(
                    get_image_meta(image_path)["gray"],
                    blur_kernel=blur_slider.value,
                    threshold_ratio=threshold_slider.value,
                    padding=padding_slider.value,
                    filter_mode=filter_dropdown.value,
                )
                set_roi_sliders(roi, image_path)


            def open_manual_roi_window(*_args) -> None:
                if not image_dropdown.value:
                    return

                image_path = Path(image_dropdown.value)
                meta = get_image_meta(image_path)

                with manual_roi_output:
                    clear_output(wait=True)
                    print("Opening ROI selection window. Drag a rectangle, press Enter to confirm, or Esc to cancel.")

                roi_mode_dropdown.value = "manual"
                try:
                    roi = le.select_roi_with_opencv(meta["display"])
                    set_roi_sliders(roi, image_path)
                    with manual_roi_output:
                        clear_output(wait=True)
                        print(f"Manual ROI updated: {roi}")
                except Exception as exc:
                    with manual_roi_output:
                        clear_output(wait=True)
                        print(f"Manual ROI selection failed or was cancelled: {exc}")


            def sync_sliders_for_image(*_args) -> None:
                if not image_dropdown.value:
                    return
                image_path = Path(image_dropdown.value)
                if roi_mode_dropdown.value == "fixed":
                    set_roi_sliders(FIXED_ROI_BY_TASK[task_dropdown.value], image_path)
                elif roi_mode_dropdown.value == "auto":
                    load_auto_roi()
                else:
                    meta = get_image_meta(image_path)
                    default_roi = clip_roi_to_image(FIXED_ROI_BY_TASK[task_dropdown.value], image_path)
                    set_roi_sliders(default_roi, image_path)
                    x_slider.max = max(meta["width"] - 1, 0)
                    y_slider.max = max(meta["height"] - 1, 0)
                    w_slider.max = max(meta["width"], 1)
                    h_slider.max = max(meta["height"], 1)


            def render_preview(*_args) -> le.ExtractionResult | None:
                if not image_dropdown.value:
                    return None

                image_path = Path(image_dropdown.value)
                roi = get_roi_for_image(image_path)
                set_roi_sliders(roi, image_path)

                result = le.process_image(
                    image_path=image_path,
                    roi=roi,
                    blur_kernel=blur_slider.value,
                    threshold_ratio=threshold_slider.value,
                    auto_roi=False,
                    roi_padding=padding_slider.value,
                    filter_mode=filter_dropdown.value,
                    segment_count=segment_slider.value,
                )

                overlay = le.overlay_centers(result.display_image, result.centers, result.roi)

                with preview_output:
                    clear_output(wait=True)
                    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

                    axes[0, 0].imshow(result.display_image, cmap="gray")
                    axes[0, 0].add_patch(
                        plt.Rectangle(
                            (result.roi.x, result.roi.y),
                            result.roi.w,
                            result.roi.h,
                            fill=False,
                            edgecolor="yellow",
                            linewidth=2,
                        )
                    )
                    axes[0, 0].set_title("Step 1: Original Image + ROI")
                    axes[0, 0].axis("off")

                    axes[0, 1].imshow(result.raw_roi, cmap="gray")
                    axes[0, 1].set_title("Step 2: Raw ROI")
                    axes[0, 1].axis("off")

                    axes[0, 2].imshow(result.filtered_roi, cmap="gray")
                    axes[0, 2].set_title(f"Step 3: Filtered ROI ({result.filter_mode})")
                    axes[0, 2].axis("off")

                    axes[1, 0].plot(result.raw_profile, color="gray", label="raw")
                    axes[1, 0].plot(result.filtered_profile, color="royalblue", label="filtered")
                    axes[1, 0].set_title(f"Step 4: Column Energy Profile (col={result.profile_column_index})")
                    axes[1, 0].set_xlabel("Y")
                    axes[1, 0].set_ylabel("Intensity")
                    axes[1, 0].grid(True)
                    axes[1, 0].legend()

                    axes[1, 1].imshow(overlay[:, :, ::-1])
                    axes[1, 1].set_title(f"Step 5: Center Extraction Result, points={len(result.centers)}")
                    axes[1, 1].axis("off")

                    axes[1, 2].imshow(result.display_image, cmap="gray")
                    if result.segment_count > 1:
                        edges = np.linspace(result.roi.x, result.roi.x + result.roi.w, result.segment_count + 1)
                        for edge in edges[1:-1]:
                            axes[1, 2].axvline(edge, color="cyan", linestyle="--", linewidth=1.2)
                    axes[1, 2].add_patch(
                        plt.Rectangle(
                            (result.roi.x, result.roi.y),
                            result.roi.w,
                            result.roi.h,
                            fill=False,
                            edgecolor="red",
                            linewidth=2,
                        )
                    )
                    axes[1, 2].set_title("Step 6: ROI / Segment Layout")
                    axes[1, 2].axis("off")

                    plt.tight_layout()
                    plt.show()

                    display(Markdown(describe_result(task_dropdown.value, result)))

                return result
            """
        ),
        code_cell(
            """
            def run_single_image(*_args) -> None:
                result = render_preview()
                if result is None:
                    return

                task_name = task_dropdown.value
                output_path = build_result_path(task_name, result.image_path)
                le.save_centers_csv(result.centers, output_path)

                with run_output:
                    clear_output(wait=True)
                    display(
                        Markdown(
                            f"**当前图片处理完成**  \\\\n"
                            f"- Task: `{TASK_DISPLAY_NAMES[task_name]}`  \\\\n"
                            f"- Image: `{result.image_path.name}`  \\\\n"
                            f"- Points: `{len(result.centers)}`  \\\\n"
                            f"- CSV: `{output_path}`"
                        )
                    )


            def run_task_batch(*_args) -> None:
                task_name = task_dropdown.value
                image_paths = get_task_images(task_name)
                batch_rows = []

                with run_output:
                    clear_output(wait=True)
                    print(f"Batch processing started: {TASK_DISPLAY_NAMES[task_name]}")

                for image_path in image_paths:
                    roi = get_roi_for_image(image_path)
                    result = le.process_image(
                        image_path=image_path,
                        roi=roi,
                        blur_kernel=blur_slider.value,
                        threshold_ratio=threshold_slider.value,
                        auto_roi=False,
                        roi_padding=padding_slider.value,
                        filter_mode=filter_dropdown.value,
                        segment_count=segment_slider.value,
                    )
                    output_path = build_result_path(task_name, image_path)
                    le.save_centers_csv(result.centers, output_path)
                    batch_rows.append((image_path.name, len(result.centers), str(result.roi), result.filter_mode, result.segment_count, str(output_path)))

                summary_path = OUTPUT_DIR / task_name / "_summary.csv"
                with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["image_name", "point_count", "roi", "filter_mode", "segment_count", "csv_path"])
                    writer.writerows(batch_rows)

                with run_output:
                    clear_output(wait=True)
                    display(Markdown(f"**Batch processing finished**: `{TASK_DISPLAY_NAMES[task_name]}`"))
                    for image_name, point_count, roi_text, filter_mode, segment_count, output_path in batch_rows:
                        print(f"{image_name}: points={point_count}, roi={roi_text}, filter={filter_mode}, segments={segment_count}")
                        print(f"  -> {output_path}")
                    print(f"Summary file: {summary_path}")
            """
        ),
        code_cell(
            """
            task_dropdown.observe(refresh_image_dropdown, names="value")
            image_dropdown.observe(sync_sliders_for_image, names="value")
            roi_mode_dropdown.observe(sync_sliders_for_image, names="value")

            apply_fixed_roi_button.on_click(load_fixed_roi)
            apply_auto_roi_button.on_click(load_auto_roi)
            manual_roi_button.on_click(open_manual_roi_window)
            preview_button.on_click(render_preview)
            run_image_button.on_click(run_single_image)
            run_task_button.on_click(run_task_batch)

            refresh_image_dropdown()
            sync_sliders_for_image()

            ui = widgets.VBox(
                [
                    widgets.HBox([task_dropdown, image_dropdown]),
                    widgets.HBox([roi_mode_dropdown, filter_dropdown]),
                    blur_slider,
                    threshold_slider,
                    padding_slider,
                    segment_slider,
                    x_slider,
                    y_slider,
                    w_slider,
                    h_slider,
                    widgets.HBox([apply_fixed_roi_button, apply_auto_roi_button, manual_roi_button, preview_button, run_image_button, run_task_button]),
                    manual_roi_output,
                    preview_output,
                    run_output,
                ]
            )
            display(ui)
            render_preview()
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
