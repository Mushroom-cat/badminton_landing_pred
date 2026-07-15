# 击球前球拍趋势与人体姿态模型

## 模型与保留关系

本版本参照最新 after 模型的设计原则重做 before 模型，但没有覆盖旧权重：

- 新模型：`models/before_pose_racket_trend.pt`
- 旧 Transformer：`models/before.pt`
- 新训练脚本：`train/train_before_pose_racket_trend.py`
- 特征与推理实现：`util/before_pose_racket.py`

使用比较脚本时，将 `--pose-model` 指向新权重即可加载新模型。三维 HTML 只绘制本次实际加载的 before、after 预测落点和拟合落点，不再绘制旧模型参考落点；CSV 中仍保留必要的数值对比字段。

## 为什么调整

旧 before Transformer 直接接收 50 帧、每帧 63 维原始坐标，没有显式建立球拍挥动方向、速度、拍面方向与落点之间的关系。对 `datasets/scene1+2` 的击球前窗口审计发现，最近三帧球拍中心运动方向与真实落点方向高度相关：

- 中位夹角：8.26°
- 平均夹角：14.57°
- P90：26.62°
- 落点位于挥拍前方的样本比例：97.34%

因此，新模型不再直接回归全局 XY，而是在最近球拍挥动坐标系中预测“前向距离 + 横向偏移”。

## 输入特征与融合方式

模型从击球前窗口末尾最多 12 个有效球拍帧提取 41 个球拍特征，包括：

- 球拍中心位置、全段速度、最近速度、加速度和位移；
- 水平路径长度与直线度；
- 球拍长轴、短轴、拍面法向和轴长度；
- 长轴、短轴、拍面方向的角速度；
- 全段速度和最近速度大小。

人体分支从肩、肘、腕、髋 8 个关键点提取 70 个相对生物力学特征，包括：

- 相对骨盆、按人体尺度归一化的位置；
- 相对速度；
- 肩轴、髋轴、躯干轴、躯干法向；
- 肩宽、髋宽、躯干长度与人体尺度；
- 骨盆平移速度和肩相对骨盆速度。

融合结构与新版 after 的“主趋势 + 门控修正”一致：

1. 球拍分支独立预测基础落点；
2. 人体分支编码身体姿态和动力链信息；
3. 门控网络分别控制人体信息对前向距离和横向距离的修正强度；
4. 最终输出为“球拍基础预测 + 门控人体残差”。

训练目标为相对最后球拍中心、沿最近挥拍方向的 `[log(1 + 前向距离), 横向距离]`。这样球拍运动趋势直接决定预测坐标基，时间顺序不会被网络忽略。

## 训练划分和记录

训练命令：

```powershell
python train/train_before_pose_racket_trend.py `
  --data-dir datasets/scene1+2 `
  --model-out models/before_pose_racket_trend.pt `
  --report-dir results/before_pose_racket_trend `
  --device cuda
```

数据按文件名前缀对应的采集 group 划分，避免同一段采集同时进入训练和验证：

- 原始文件：3857
- 通过质量过滤：3705
- 训练：2950 个样本、304 个 group
- 验证：755 个样本、78 个 group
- 训练/验证 group 交集：0
- 镜像与不同击球前偏移增强后：17286 个训练实例
- 最佳权重：第 68 轮
- 早停：第 108 轮

每一轮记录保存在 `results/before_pose_racket_trend/training_history.csv`，共有 108 行 epoch 记录。总体配置和最终指标在 `training_report.json`，逐验证样本预测在 `validation_predictions.csv`。

## 验证结果

在 755 个 group-held-out 验证样本上：

| 模型/分支 | XY 平均误差 | 中位误差 | P90 |
|---|---:|---:|---:|
| 新模型球拍基础分支 | 112.47 cm | 97.03 cm | 212.21 cm |
| 新模型球拍 + 人体门控 | 100.13 cm | 84.08 cm | 193.81 cm |
| 旧 before Transformer（参考） | 192.50 cm | 172.66 cm | 381.26 cm |

新模型相对旧模型平均误差降低 92.37 cm，并在 537/755 个样本上更好。需要注意：此验证集对新模型是严格 group-held-out；旧 checkpoint 原始训练划分未知，因此旧模型数字属于同输入集合上的参考，不等价于重新训练后的完全无泄漏公平对照。

5 个外部测试样本没有实测落点标签，只能以落点拟合结果作为代理参照：

| 模型 | 对拟合落点平均差值 | 更优样本数 |
|---|---:|---:|
| 新 before | 118.59 cm | 4/5（相对旧模型） |
| 旧 before | 235.74 cm | 1/5 |

消融测试显示，倒序球拍轨迹使预测平均移动 786.62 cm；保持拍面几何但将球拍平移速度降到 10% 时，预测平均移动 301.43 cm。倒序人体序列使预测平均移动 55.07 cm，冻结人体动作使预测平均移动 45.11 cm。这说明模型确实利用了球拍运动趋势与人体姿态变化，而不是只记最后一帧原始坐标。

## 推理与复算

5 个样本的完整比较命令：

```powershell
python infer/compare_duida_pose_fit_predictions.py `
  --pose-model models/before_pose_racket_trend.pt `
  --fit-model models/c_parameter_mlp_fall.pt `
  --pose-mode before `
  --pose-dir datasets/2026-7-10test/4-poseball `
  --fit-dir datasets/2026-7-10test/3-fit `
  --pose-skip-n 5
```

验证复算：

```powershell
python infer/validate_before_models_scene1_2.py
python infer/validate_before_pose_racket_trend.py
```

主要输出：

- `results/before_pose_racket_trend/training_history.csv`
- `results/before_pose_racket_trend/training_report.json`
- `results/before_pose_racket_trend/validation_predictions.csv`
- `results/before_pose_racket_trend/same_split_model_comparison.json`
- `results/before_pose_racket_trend/five_sample_validation.json`
- `infer/duida_pose_fit_prediction_comparison.csv`
- `infer/duida_pose_fit_3d/*.html`
