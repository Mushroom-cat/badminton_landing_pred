# Badminton Landing Point Prediction

基于人体姿态序列的羽毛球落点预测系统。输入运动员的 3D 姿态关键点时序数据，预测羽毛球的落点坐标 (XYZ)、飞行时间和飞行方向。

## 项目结构

```
badminton_pred-main/
├── train/                       # 训练脚本
│   ├── train_landpoint_pred.py  # 主训练入口
│   ├── train_multi_offset.py    # 多 offset 对比实验
│   └── eval_landpoint_pred.py   # 仅测试评估（不训练）
├── infer/                       # 推理脚本
│   └── infer_landpoint_pred.py   # ONNX 推理
├── analysis/                    # 分析与可视化
│   ├── visual_csv.py            # 结果可视化脚本
│   └── visual_csv.ipynb         # 交互式可视化 Notebook
├── util/                        # 基础模块
│   ├── dataset.py               # 数据集加载、解析、Dataset 定义
│   ├── model.py                 # 模型定义（Transformer、LSTM 等）
│   ├── trainer.py               # 训练循环、验证、测试导出
│   └── merge_ball.py            # 数据预处理：合并多源坐标文件
├── datasets/                    # 数据集存放目录
├── models/                      # 训练好的模型权重
├── results/                     # 测试结果 CSV
├── visualization/               # 可视化输出图表
├── logs/                        # 训练日志
└── 1.sh                         # 批量实验脚本
```

## 环境准备

首先创建以下输出目录（如果不存在）：

```bash
mkdir models results visualization logs
```

将数据集放入 `datasets/` 目录下，例如：

```
datasets/
├── data_1217_ball_ext5/         # 21 点姿态数据（不含羽毛球坐标）
├── 20260106_pose_infer/         # 22 点姿态数据（含羽毛球坐标）
├── 20260202_all/                # 合并后的完整数据
└── ...
```

## 可直接运行的脚本

> 所有脚本均从 **项目根目录** 运行。

---

### `train/train_landpoint_pred.py` — 主训练脚本

从零开始训练落点预测模型。完整流程：加载数据 → 80/20 划分训练/测试集 → 训练模型 → 测试并保存结果 CSV → 生成可视化图表。

```bash
python train/train_landpoint_pred.py
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_folder` | `datasets/data_1217_ball_ext5` | 数据集路径 |
| `--points_num` | `21` | 关键点数量（21=不含羽毛球，22=含羽毛球） |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `2e-5` | 学习率 |
| `--batch_size` | `32` | 批大小 |
| `--min_offset_len` | `5` | 训练时滑动窗口最小偏移量 |
| `--max_offset_len` | `25` | 训练时滑动窗口最大偏移量 |
| `--temp_test_offset` | `5` | 测试时固定偏移量（<0 则随机） |
| `--aug_method` | `None` | 数据增强方式（可选：平移、旋转、缩放、噪声） |

输出：
- 最佳模型权重 → `models/`
- 测试结果 CSV → `results/`
- 可视化图表 → `visualization/`
- 训练日志 → `logs/`

---

### `train/train_multi_offset.py` — 多 Offset 对比实验

依次以不同的 `temp_test_offset`（0, 1, 2, 3, 4）运行完整的训练+测试流程，最终汇总绘制「预测误差 vs 不确定性」散点图，用于分析不同预测时机对模型效果的影响。

```bash
python train/train_multi_offset.py
```

输出：
- 每个 offset 各自的训练日志、模型权重、结果 CSV
- 汇总散点图 → `visualization/Error-std_aleatoric_Correlation_multi_df`

---

### `train/eval_landpoint_pred.py` — 仅测试评估（不训练）

加载已训练好的模型权重，直接在指定数据集上进行测试评估。

```bash
python train/eval_landpoint_pred.py
```

注意：需要手动修改脚本中 `trainer.best_model_path` 指向要评估的模型权重文件。

---

### `infer/infer_landpoint_pred.py` — ONNX 推理

支持「击球后」和「击球前」两种推理模式的 ONNX 推理脚本：

- **after 模式**：使用完整 66 维特征（含羽毛球坐标），取最后 50 帧推理
- **before 模式**：使用 63 维特征（不含羽毛球坐标），可通过 `skip_n` 参数跳过末尾若干帧，模拟击球前的预测场景

```bash
python infer/infer_landpoint_pred.py
```

脚本运行后会分别执行 after 和 before 两种模式的批量评估，并输出各自的平均欧氏距离误差。注意两个模式分别对应不同的模型参数形状，因为击球前模型不包含球位置，击球后模型包含球位置。

---

### `analysis/visual_csv.py` — 结果可视化分析

对模型的测试结果 CSV 进行全面的可视化分析。直接运行时，会读取指定的结果 CSV 文件并生成以下图表：

```bash
python analysis/visual_csv.py
```

可视化内容：
- 落点预测散点图（按误差分组：高于/低于均值）
- 误差分布图（落点 XY 误差散点、欧氏距离直方图、时间误差直方图、方向误差直方图）
- 不确定性估计散点图（预测误差 vs 预测不确定性）

此外还提供以下可选分析函数（需在脚本中取消注释）：
- `visual_samples_distribution()`：击球位置/高度/速度的样本分布
- `visual_shot_categories()`：按击球类型（杀球、吊球、搓放等）分类的误差分析

---

### `util/merge_ball.py` — 数据合并工具

将两个数据源按帧 ID 进行合并：把 folder2 中每帧的 3 维坐标（如羽毛球坐标）追加到 folder1 每行的末尾，用于构建 22 点（21 点姿态 + 1 点羽毛球）的完整数据集。

```bash
python util/merge_ball.py
```

注意：运行前需修改脚本末尾的 `dir1`、`dir2`、`out_dir` 路径为实际的文件夹位置。

---

### `analysis/visual_csv.ipynb` — 交互式可视化 Notebook

提供与 `visual_csv.py` 类似的可视化功能，以 Jupyter Notebook 形式呈现，适合交互式探索和调参。在 Jupyter 环境中打开即可使用。

---

## 滑动窗口机制说明

训练和测试时使用滑动窗口从序列末尾截取子序列：

- `min_offset_len` / `max_offset_len`：训练时随机偏移的范围
- `temp_test_offset`：测试时使用的固定偏移量（若 < 0 则测试时也随机）

**击球前模型**：设置 `min_offset_len=5, max_offset_len=25`（跳过末尾击球后的帧）

**击球后模型**：设置 `min_offset_len=0, max_offset_len=4`（使用接近末尾的帧）
