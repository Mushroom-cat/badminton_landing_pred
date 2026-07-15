# 球轨迹 + 球拍运动击球后模型

## 保留关系

本版本是新增模型，不覆盖前一版球轨迹模型：

- 球轨迹版本：`models/after_motion_trend.pt`；
- 球轨迹 + 球拍版本：`models/after_ball_racket_trend.pt`；
- 原始 Transformer：`models/after.pt`。

三维对比脚本会同时加载三者。默认 after 结果使用球+球拍版本，紫色空心点是球轨迹版本，橙色点是原始 Transformer。

## 球拍坐标如何使用

没有把 4 个球拍点直接展平后交给网络。每个样本从与最后 5–6 个有效球点对应的球拍帧中构造 43 个显式特征：

- 球拍中心位置、全段速度、最近速度和加速度；
- `P2-P1` 短轴方向和长度；
- `P4-P3` 长轴方向和长度；
- 长短轴叉积得到的拍面法向；
- 长轴、短轴方向随时间的变化率；
- 球相对球拍中心的位置和相对速度；
- 球速度及球拍-球相对速度在短轴、长轴和拍面法向上的投影；
- 当前及观察期间最小的球拍-球距离。

球拍几何会先过滤：长短轴退化、长度异常、坐标非有限或最后一帧球拍无效的样本不会进入训练。

## 融合方式

模型由两个分支组成：

1. 球分支使用前一版的 19 个球运动特征，独立产生基础落点；
2. 球拍分支编码 43 个球拍运动和球拍-球相对特征；
3. 门控网络分别控制球拍对前向距离和横向距离的修正量；
4. 最终结果为“球分支基础结果 + 门控球拍残差”。

训练时保留球分支辅助损失，并约束球拍修正幅度，因此球拍信息用于修正而不会覆盖球轨迹主趋势。最终仍在球速度坐标系中预测前向/横向距离。

## 训练与验证

```powershell
python train/train_after_ball_racket_trend.py `
  --data-dir datasets/scene1+2 `
  --model-out models/after_ball_racket_trend.pt `
  --report-dir results/after_ball_racket_trend `
  --device cuda
```

数据审计和划分：

- 原始文件 3857 个；
- 球轨迹质量过滤后 3761 个；
- 再过滤球拍几何异常后 3735 个；
- 训练 2975 个样本、304 个 group；
- 验证 760 个样本、78 个 group；
- 训练和验证 group 交集为 0；
- 镜像和 5/6 点长度扩增后训练实例 11866 个。

最佳权重位于第 54 轮，训练到第 94 轮后早停。相同 760 个验证样本上的公平比较：

| 模型 | XY 平均误差 | 中位误差 | P90 |
|---|---:|---:|---:|
| 球轨迹模型 | 61.95 cm | 50.26 cm | 120.69 cm |
| 球轨迹 + 球拍模型 | 57.04 cm | 44.99 cm | 113.94 cm |

五个外部样本相对落点拟合结果的平均差值：

| 模型 | 平均差值 |
|---|---:|
| 原始 Transformer | 321.53 cm |
| 球轨迹模型 | 83.85 cm |
| 球轨迹 + 球拍模型 | 68.76 cm |

球拍消融中，倒序球拍轨迹使落点平均移动 69.54 cm，把球拍固定为最后一帧使落点平均移动 63.15 cm，说明新模型确实使用了球拍的运动与姿态变化，而不仅使用球坐标。

五个外部样本没有真实落点，以上外部差值只表示与落点拟合算法的一致性。

## 复算命令和输出

原 before 对比命令无需修改；存在新权重时会自动优先加载球+球拍版本。

```powershell
python infer/validate_after_ball_racket_trend.py
python infer/validate_after_models_scene1_2.py
```

主要输出：

- `results/after_ball_racket_trend/training_report.json`；
- `results/after_ball_racket_trend/validation_predictions.csv`；
- `results/after_ball_racket_trend/same_split_model_comparison.json`；
- `results/after_ball_racket_trend/five_sample_validation.json`；
- `infer/duida_pose_fit_prediction_comparison.csv`；
- `infer/duida_pose_fit_3d/*.html`。
