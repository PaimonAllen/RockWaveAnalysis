# RockWaveAnalysis

> Utilities & Jupyter notebooks for **rock crack / shape analysis** — including shape classification, crack presence detection, crack length/size categorization, and crack *shape* classification. This repo is organized by tasks and includes reusable feature-extraction code.

## Repository structure

```
RockWaveAnalysis/
├─ 1.shapeClassification/                # 分类任务：岩石形状/结构（notebooks）
├─ 2.isCrackClassification/              # 分类任务：是否存在裂纹（notebooks）
├─ 3.crackLengthAndsizeClassification/   # 分类任务：裂纹长度与尺寸（notebooks）
├─ 4.crackShapeClassification/           # 分类任务：裂纹形态（notebooks）
├─ featureExtract/
│  └─ featuresExtractFunction/           # 特征工程与复用函数
├─ test.ipynb                            # 示例/临时实验（notebook）
├─ test.py                               # 简单脚本（Python）
└─ LICENSE                               # MIT
```

> 上述目录名与文件来自仓库公开页面；具体 notebook / 脚本内容以实际文件为准。

## What this repo provides

- 📓 **Task‑oriented notebooks**：把问题拆成 4 个独立分类任务，便于快速试验与对比。  
- 🧩 **可复用的特征工程**：`featureExtract/featuresExtractFunction/` 中集中常用特征提取逻辑，避免重复造轮子。  
- 🧪 **轻量脚手架**：`test.ipynb` 与 `test.py` 用于快速验证思路或函数。  

## Getting started

### 1) 准备环境

建议使用 Python 3.10+ 与一个独立虚拟环境（conda/venv 均可）。多数经典图像/机器学习实验可用到：

```bash
# 任选其一：conda 或 venv
conda create -n rockwave python=3.10 -y && conda activate rockwave

# 必需/常用（请按你的 notebook 实际需要增删）
pip install numpy pandas scikit-learn matplotlib scikit-image opencv-python tqdm jupyter
```

> 如果某些 notebook 依赖其它库（如 `seaborn`, `xgboost`, `lightgbm`, `pytorch`, `tensorflow` 等），请根据 notebook 顶部的 `import` 自行补齐。

### 2) 获取数据

本仓库不自带数据集。根据各任务的 notebook 顶部说明把**你的数据路径**改成本地路径，或在运行前用环境变量/配置项传入。一个常见的做法是放到项目根目录的 `data/` 下（例如 `data/raw`, `data/processed`），并在 notebook 顶部设置：

```python
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
```

### 3) 运行 notebook

以“是否存在裂纹”任务为例：

1. 打开 `2.isCrackClassification/` 下对应的 `.ipynb`。  
2. 在第一、二个单元格调整 **数据路径** 与 **超参数**。  
3. 依次执行单元格，观察**特征提取**、**训练/验证**与**评估**输出。  
4. 结果（如混淆矩阵/ROC/PR、关键指标）可在最后若干单元格查看或另存为图片。

> 其余三个任务（形状分类、长度/尺寸分类、裂纹形态分类）用法一致。



## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

**Citation / Acknowledgement**  
If you use this repository in academic work, please cite or acknowledge 

```
@article{tian2024machine,
  title = {A Machine Learning-Assisted Nondestructive Testing Method Based on Time-Domain Wave Signals},
  author = {Tian, Zhuoran and Li, Jianchun and Li, Xing and Wang, Zhijie and Zhou, Xiaozhou and Sang, Yang and Zou, Chunjiang},
  year = {2024},
  journal = {International Journal of Rock Mechanics and Mining Sciences},
  volume = {177},
  pages = {105731},
  doi = {10.1016/j.ijrmms.2024.105731}
}

```

