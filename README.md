
# 激光条纹中心提取项目

Powered by Codex GPT-5.4

这是一个基于 Python 与 Jupyter Notebook 的激光条纹中心提取项目，支持交互式参数调整、不同方法对比和结果导出。

本项目面向三类任务：

1. 任务 1：简单直线激光条纹中心提取
2. 任务 2：带噪声激光条纹的去噪与中心提取
3. 任务 3：复杂激光条纹轮廓的中心提取与分段处理

---

## 项目结构


python_workspace/
├─ laser_extraction.py
├─ build_interactive_notebook.py
├─ laser_extraction_interactive.ipynb
├─ requirements-notebook.txt
├─ 仿真实践图片/
│  ├─ 任务1-简单直线/
│  ├─ 任务2-噪声图像/
│  └─ 任务3-复杂激光条纹/
└─ notebook_outputs/


---

## 主要文件说明

### `laser_extraction.py`
核心算法模块。

主要负责：
- 图像读取（兼容中文路径）
- 灰度化处理
- ROI 处理
- 去噪与背景抑制
- 中心提取
- 结果可视化
- CSV 导出

### `build_interactive_notebook.py`
Notebook 生成脚本。

运行该脚本后，会重新生成交互式 notebook：

```bash
python python_workspace/build_interactive_notebook.py
```

### `laser_extraction_interactive.ipynb`
交互式 notebook 主界面。

支持：
- 任务切换
- ROI 模式选择
- 灰度化方式选择
- 滤波方式选择
- 中心提取方法选择
- 复杂条纹分段处理
- 分步结果展示
- CSV 导出

---

## 支持的任务

### 任务 1：简单直线激光条纹
适用于规则、连续、噪声较少的直线条纹图像。

推荐设置：
- ROI Mode：`Fixed ROI` 或 `Manual ROI`
- Filter：`Gaussian`
- Method：`Global Centroid` 或 `Peak-Window Centroid`

### 任务 2：带噪声激光条纹
适用于含明显背景噪声或强噪声干扰的图像。

推荐设置：
- Filter：`Gaussian -> Median`
- Method：`Peak-Window Centroid`
- BG Kernel：`51`
- Peak Half Window：`22`

也可以在 notebook 中对比其他方法：
- `Global Centroid`
- `Gaussian Fit`
- `Steger-Like Ridge`

### 任务 3：复杂激光条纹
适用于复杂曲线、局部变化明显、存在弯折或轮廓不规则的条纹图像。

推荐设置：
- Method：`Peak-Window Centroid`
- Segments：`4` 或按实际情况调整

---

## 支持的中心提取方法

当前 notebook 中支持以下方法：

- `Global Centroid`
- `Peak-Window Centroid`
- `Gaussian Fit`
- `Steger-Like Ridge`

这些方法可用于同一张图像上的对比实验。

---

## 支持的灰度化方式

项目中提供以下灰度化策略：

- `OpenCV Gray`
- `Luminance`
- `Red Channel`
- `Green Channel`
- `Blue Channel`
- `Max Channel`

可用于观察不同灰度化方式对提取结果的影响。

---

## 安装依赖

建议使用 Python 3.12 或兼容环境。

安装依赖：

```bash
pip install -r python_workspace/requirements-notebook.txt
```

---

## 运行方式

### 方式一：直接使用交互式 Notebook
打开：

```text
python_workspace/laser_extraction_interactive.ipynb
```

然后按顺序运行各单元格。

### 方式二：重新生成 Notebook
如果修改了 notebook 结构或界面逻辑，可以重新生成：

```bash
python python_workspace/build_interactive_notebook.py
```

---

## 输出结果

提取得到的中心点坐标会保存到：

```text
python_workspace/notebook_outputs/
```

每个 CSV 文件包含两列：

- `X`
- `Y`

---

## 项目特性

- 支持中文路径图像读取
- 支持多种灰度化方式
- 支持多种滤波方式
- 支持多种中心提取方法
- 支持固定、手动、自动三种 ROI 模式
- 支持复杂条纹的分段处理
- 支持分步可视化展示
- 支持结果导出为 CSV

---

## 使用说明

- `laser_extraction.py` 是算法核心
- `build_interactive_notebook.py` 用于生成 notebook
- `laser_extraction_interactive.ipynb` 是最终交互界面
- 如果修改了 notebook 生成逻辑，需要重新运行生成脚本
- 如果只修改算法，一般只需要修改 `laser_extraction.py`

---

## 注意事项

- `仿真实践图片` 目录中的原始图像是项目运行所必须的数据
- 手动 ROI 选择依赖 OpenCV 弹窗
- 如果 notebook 在多次运行后出现控件重复响应，建议重启 kernel 后重新运行
- 结果展示主要用于实验分析，实际参数应结合具体图像进行调整

---

## 项目用途

本项目适用于：
- 激光条纹中心提取实验
- 多种去噪方法对比
- 多种中心提取算法对比
- 基于 notebook 的交互式实验分析与展示

---
