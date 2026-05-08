# Badminton Landing Point Prediction

基于人体姿态序列的羽毛球落点预测系统。包含从原始图像采集到模型训练的完整pipeline：双目相机图像 → 2D 关键点检测 → 3D 坐标重建 → 落点预测模型训练与推理。

## 项目结构

```
badminton-main/
│
├── run.py                              # 完整pipeline入口（数据采集 + 落点预测训练）
├── cvt_2d_3d.py                        # 2D → 3D 坐标转换（双目三角化）
│
├── cfg/                                # 标定参数 & 预训练权重
│   ├── 20250809_intrinsic.yml          #   相机内参
│   ├── 20250809_extrinsic.yml          #   相机外参
│   ├── ball_det.pt                     #   YOLO 球检测权重
│   └── bat_pose.pt                     #   YOLO-Pose 球拍关键点权重
│
├── util/                               # 公共工具模块
│   ├── cam_calib.py                    #   双目相机标定 & 2D→3D 三角化
│   ├── dataset.py                      #   数据集加载、解析、Dataset 定义
│   ├── model.py                        #   模型定义（Transformer、LSTM 等）
│   ├── trainer.py                      #   训练循环、验证、MC-Dropout 测试
│   └── merge_ball.py                   #   数据预处理：合并多源坐标文件
│
├── datasets/                           # 数据集目录
│   ├── Infered_3D_to_Sample.py         #   合并 3D 球拍+球坐标为训练样本
│   ├── scene1/                         #   场景1 数据
│   ├── scene2/                         #   场景2 数据
│   └── scene1+2/                       #   场景1+2 合并数据
│
├── train/                              # 训练脚本
│   ├── train_landpoint_pred.py         #   主训练入口
│   ├── train_multi_offset.py           #   多 offset 对比实验
│   └── eval_landpoint_pred.py          #   仅测试评估（不训练）
│
├── infer/                              # 推理脚本
│   └── infer_landpoint_pred.py         #   ONNX 推理
│
├── analysis/                           # 分析与可视化
│   ├── visual_csv.py                   #   结果可视化脚本
│   └── visual_csv.ipynb                #   交互式可视化 Notebook
│
├── Labeled_2D_Points_Ball/             # 球检测标注数据 & 训练脚本
│   ├── ball_det_train.py               #   YOLO 球检测训练
│   ├── Image/                          #   训练图像
│   └── Image_val/                      #   验证图像
│
├── Labeled_2D_Points_Racket/           # 球拍关键点标注数据 & 训练脚本
│   ├── racket_pose_train.py            #   YOLO-Pose 球拍训练
│   ├── select_training.py              #   标注数据质检工具
│   └── 20250809_143033/                #   标注数据集
│
├── Infered_2D_Points_Ball/             # 球 2D 推理结果
│   ├── ball_infer.py                   #   YOLO 球检测推理
│   └── 20250809_143033/                #   推理输出 (JSON)
│
├── Infered_2D_Points_Racket/           # 球拍 2D 推理结果
│   ├── racket_infer.py                 #   YOLO-Pose 球拍推理
│   └── 20250809_143033/                #   推理输出 (JSON)
│
├── Infered_3D_Points_Ball/             # 球 3D 坐标（由 cvt_2d_3d.py 生成）
├── Infered_3D_Points_Racket/           # 球拍 3D 坐标（由 cvt_2d_3d.py 生成）
├── Labeled_3D_Points_Ball/             # 球 3D 标注数据
├── Labeled_3D_Points_Racket/           # 球拍 3D 标注数据
│
├── models/                             # 训练好的模型权重（.pt / .onnx）
├── results/                            # 测试结果 CSV
├── visualization/                      # 可视化输出图表
└── logs/                               # 训练日志
```

## 环境准备

首先创建以下输出目录（如果不存在）：

```bash
mkdir models results visualization logs
```

将数据集放入 `datasets/` 目录下，例如：

```
datasets/
├── scene1/                          # 场景1 样本
├── scene2/                          # 场景2 样本
├── scene1+2/                        # 合并场景
└── ...
```

每个样本为一个 `.txt` 文件，每行格式为 `帧号:关键点坐标`，最后一行为落点坐标。

## 完整pipeline：`run.py`

`run.py` 将数据采集和落点预测训练整合为一个完整pipeline，支持通过命令行参数灵活控制。

### 基本用法

```bash
# 执行完整pipeline（数据采集 + 训练）
python run.py

# 跳过数据采集，直接训练
python run.py --skip_data_pipeline

# 仅执行数据采集，不训练
python run.py --skip_train

# 指定数据集和训练参数
python run.py --skip_data_pipeline --data_folder datasets/scene1+2 --epochs 200 --lr 1e-5
```

### pipeline执行流程

**阶段一：数据采集pipeline**（可通过 `--skip_data_pipeline` 跳过）

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | `Infered_2D_Points_Racket/racket_infer.py` | YOLO-Pose 球拍关键点推理 |
| 2 | `Infered_2D_Points_Ball/ball_infer.py` | YOLO 球检测推理 |
| 3 | `cvt_2d_3d.py` | 双目 2D → 3D 坐标转换 |
| 4 | `datasets/Infered_3D_to_Sample.py` | 合并 3D 数据为训练样本 |

**阶段二：落点预测训练pipeline**（可通过 `--skip_train` 跳过）

| 步骤 | 函数 | 说明 |
|------|------|------|
| 1 | `step1_init_model` | 初始化 ImprovedTransformerModel |
| 2 | `step2_load_data` | 加载样本并 80/20 划分训练/测试集 |
| 3 | `step3_build_datasets` | 构建 BadmintonDataset（含归一化统计） |
| 4 | `step4_train` | 训练模型 |
| 5 | `step5_test_and_save` | MC-Dropout 测试并保存结果 CSV |
| 6 | `step6_visualize` | 生成可视化图表 |

### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip_data_pipeline` | `False` | 跳过数据采集pipeline |
| `--skip_train` | `False` | 跳过落点预测训练 |
| `--data_folder` | `datasets/scene1+2` | 数据集路径 |
| `--points_num` | `21` | 关键点数量（21=不含球，22=含球） |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `2e-5` | 学习率 |
| `--batch_size` | `32` | 批大小 |
| `--min_len` | `10` | 最小序列长度 |
| `--max_len` | `50` | 最大序列长度 |
| `--min_offset_len` | `5` | 训练时滑动窗口最小偏移量 |
| `--max_offset_len` | `25` | 训练时滑动窗口最大偏移量 |
| `--temp_test_offset` | `5` | 测试时固定偏移量（<0 则随机） |
| `--num_subsamples` | `5` | 子采样数量 |
| `--delta` | `1.0` | Huber loss delta（XYZ 损失） |
| `--lambda_time` | `0.1` | 时间预测损失权重 |
| `--lambda_direction` | `0.1` | 方向预测损失权重 |
| `--aug_method` | `None` | 数据增强方式（可选：平移、旋转、缩放、噪声） |
| `--model_dir` | `./models` | 模型权重保存路径 |
| `--results_dir` | `./results` | 测试结果保存路径 |

### 输出

- 最佳模型权重 → `models/`
- 测试结果 CSV → `results/`
- 可视化图表 → `visualization/`
- 训练日志 → `logs/`

---

## 其他可独立运行的脚本

> 所有脚本均从 **项目根目录** 运行。

### `train/train_landpoint_pred.py` — 独立训练脚本

与 `run.py` 的训练阶段功能相同，可独立运行。

```bash
python train/train_landpoint_pred.py --data_folder datasets/scene1+2
```

### `train/train_multi_offset.py` — 多 Offset 对比实验

依次以不同的 `temp_test_offset`（0, 1, 2, 3, 4）运行完整的训练+测试流程，最终汇总绘制「预测误差 vs 不确定性」散点图。

```bash
python train/train_multi_offset.py
```

### `train/eval_landpoint_pred.py` — 仅测试评估

加载已训练好的模型权重，直接在指定数据集上进行测试评估。需要手动修改脚本中 `trainer.best_model_path` 指向要评估的模型权重文件。

```bash
python train/eval_landpoint_pred.py
```

### `infer/infer_landpoint_pred.py` — ONNX 推理

支持「击球后」和「击球前」两种推理模式：

- **after 模式**：使用完整 66 维特征（含球坐标），取最后 50 帧推理
- **before 模式**：使用 63 维特征（不含球坐标），可通过 `skip_n` 跳过末尾若干帧

```bash
python infer/infer_landpoint_pred.py
```

### `analysis/visual_csv.py` — 结果可视化分析

对测试结果 CSV 进行全面的可视化分析：落点预测散点图、误差分布图、不确定性估计散点图等。

```bash
python analysis/visual_csv.py
```

### `util/merge_ball.py` — 数据合并工具

将两个数据源按帧 ID 合并，把羽毛球 3D 坐标追加到姿态数据的每行末尾。运行前需修改脚本末尾的路径配置。

```bash
python util/merge_ball.py
```

### 检测模型训练脚本

各检测模型的训练脚本已放入对应的标注数据目录中：

```bash
# 训练球检测模型
python Labeled_2D_Points_Ball/ball_det_train.py

# 训练球拍关键点检测模型
python Labeled_2D_Points_Racket/racket_pose_train.py
```

---

## 滑动窗口机制说明

训练和测试时使用滑动窗口从序列末尾截取子序列：

- `min_offset_len` / `max_offset_len`：训练时随机偏移的范围
- `temp_test_offset`：测试时使用的固定偏移量（若 < 0 则测试时也随机）

**击球前模型**：设置 `min_offset_len=5, max_offset_len=25`（跳过末尾击球后的帧）

**击球后模型**：设置 `min_offset_len=0, max_offset_len=4`（使用接近末尾的帧）
