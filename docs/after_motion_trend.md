# 击球后球运动趋势模型

## 为什么重做 after 模型

旧 `after.pt` 虽然输入 66 维数据，但网络只把球拍和球的原始坐标送入 Transformer，输出端直接读取最后一个 token。五个外部测试样本的时序消融显示：将有效球点倒序后，旧模型落点平均只移动约 3.59 cm，因此它主要利用绝对位置，没有可靠利用运动顺序。

新模型位于 `util/after_motion.py`，训练入口是 `train/train_after_motion_trend.py`。它只从最后 5–6 个有效球点提取以下显式运动量：

- 最后球坐标；
- 全段线性速度和最近三点速度；
- 二次拟合加速度；
- 首尾位移、二维路径长度和直线度；
- 有效点数和帧跨度。

网络不直接预测世界坐标 `(x, y)`，而是预测：

1. 沿观测球速度方向的正向距离；
2. 垂直于球速度方向的横向距离。

沿轨迹距离使用 `log1p` 参数化并保持为正，所以落点方向在结构上依赖球的运动方向。倒序球轨迹会反转预测方向，不再能退化为与顺序无关的位置查表。

## scene1+2 数据清洗和划分

本次扫描了 `datasets/scene1+2` 的 3857 个文件：

- 接受 3761 个样本，过滤 96 个样本；
- 85 个样本的标签方向与最后 6 个球点的运动方向夹角超过 45°；
- 6 个落点标签越界；
- 2 个样本少于 5 个有效球点；
- 3 个样本的最后球点离窗口尾部过远。

文件名中 `---` 之前的采集段作为 group。同一 group 只能出现在训练集或验证集中的一个，避免同一视频相邻击球泄漏。本次固定随机种子 42，得到：

- 训练集：2993 个原始样本、304 个 group；
- 验证集：768 个原始样本、78 个 group；
- group 交集：0。

训练集使用 5 点和 6 点两个时间长度，并增加 Y 轴镜像样本，以匹配外部测试样本只有 5–6 个击球后球点的情况。

## 训练

在项目根目录运行：

```powershell
python train/train_after_motion_trend.py `
  --data-dir datasets/scene1+2 `
  --model-out models/after_motion_trend.pt `
  --report-dir results/after_motion_trend `
  --device cuda
```

当前权重在第 71 轮取得最佳验证结果：XY 平均误差 61.86 cm，中位数 50.31 cm，P90 为 120.23 cm。

## 五个外部样本验证

该球轨迹版本继续保存在 `models/after_motion_trend.pt`，不会被球+球拍版本覆盖。当前对比脚本会优先加载更新的 `after_ball_racket_trend.pt`；如果要单独使用本版本，可显式增加 `--after-pose-model ../models/after_motion_trend.pt`：

```powershell
cd infer
python compare_duida_pose_fit_predictions.py `
  --pose-model ../models/before.pt `
  --fit-model ../models/c_parameter_mlp_fall.pt `
  --pose-mode before `
  --pose-dir ../datasets/2026-7-10test/4-poseball `
  --fit-dir ../datasets/2026-7-10test/3-fit `
  --pose-skip-n 5
```

单独复算新旧 after 模型的趋势敏感性：

```powershell
python infer/validate_after_motion_trend.py
```

输出：

- `models/after_motion_trend.pt`：新权重；
- `results/after_motion_trend/training_report.json`：数据划分和验证指标；
- `results/after_motion_trend/validation_predictions.csv`：768 个验证样本预测；
- `results/after_motion_trend/five_sample_validation.json`：五样本新旧模型对比；
- `infer/duida_pose_fit_prediction_comparison.csv`：before、拟合、新 after、旧 after 的逐样本坐标；
- `infer/duida_pose_fit_3d/*.html`：五个可交互三维图。

五个外部文件没有人工测量的真实落点，因此其中的“误差”是相对落点拟合结果的差值，只能作为一致性指标，不能当作真实泛化误差。
