# 滑动窗口落点预测实验与 PNG/GIF 可视化分析

## 1. 实验目标

本实验在 `landing_fitting/trajectory.txt` 这一条轨迹上，使用论文 MV-BMR 第 3.2.2 节的三条轨迹公式进行落点预测，并将“全量拟合”改为“滑动窗口拟合”。

核心目的：

- 用固定长度窗口动态模拟实时预测过程。
- 观察预测落点随轨迹后续点进入窗口后的变化。
- 通过 PNG 折线图分析预测误差收敛趋势。
- 通过 GIF 在羽毛球场 XY 平面上演示预测点、当前窗口和 ground truth 的空间关系。

当前默认实验配置：

```text
fps = 300
landing_z = 0
window_size = 40
```

真实落点标签为 `trajectory.txt` 最后一行：

```text
ground truth = (162.839000, -68.789000, 1.567910)
```

预测时使用 `z=0` 作为落地高度，因此主要评价指标是 XY 平面误差。

## 2. 当前实现方式

实现文件：

```text
landing_fitting/fit_trajectory.py
```

脚本保留了原来的全量拟合模式，同时新增了滑动窗口模式。

### 2.1 全量拟合模式

运行：

```bash
python landing_fitting/fit_trajectory.py
```

逻辑：

- 读取 `trajectory.txt`。
- 最后一行作为真实落点。
- 前 111 个点全部用于拟合。
- 使用论文公式拟合 `x(t)`、`y(t)`、`z(t)`。
- 求解 `z(t)=0` 的时间，再得到预测落点。

当前结果：

```text
observed_points: 111
predicted_xyz: (252.256147, -29.993502, 0.000000)
landing_time_ms: 675.461599
xy_error: 97.470595
xyz_error: 97.483204
```

该结果误差较大，说明全量点拟合被前段轨迹和全参数自由拟合拉偏。

### 2.2 滑动窗口模式

运行 40 帧窗口：

```bash
python landing_fitting/fit_trajectory.py --sliding --window-size 40
```

运行 20 帧窗口：

```bash
python landing_fitting/fit_trajectory.py --sliding --window-size 20 \
  --csv-output landing_fitting/sliding_window_predictions_w20.csv \
  --plot-output landing_fitting/sliding_window_predictions_w20.png \
  --gif-output landing_fitting/sliding_window_predictions_w20.gif
```

滑动窗口逻辑：

1. 从观测轨迹中取连续 `window_size` 个点。
2. 对窗口内点拟合论文三条公式：

   ```text
   x(t) = a1 * exp(b1 * t) + c1
   y(t) = a2 * exp(-0.002 * t) + b2 * t + c2
   z(t) = a3 * exp(b3 * t) - 0.79 * t + c3
   ```

3. 对每个窗口求解 `z(t)=0`。
4. 计算预测落点与真实落点的 `xy_error` 和 `xyz_error`。
5. 将每个窗口结果保存到 CSV。
6. 绘制 PNG 折线图。
7. 生成 GIF 动画。

早期窗口可能失败，原因是轨迹还处于较早阶段，拟合出的 `z(t)` 无法稳定求到 `z=0` 的根。失败窗口会记录在 CSV 中，但不会进入 PNG/GIF 的成功预测曲线。

## 3. 实验结果

### 3.1 40 帧滑动窗口

输出文件：

```text
landing_fitting/sliding_window_predictions.csv
landing_fitting/sliding_window_predictions.png
landing_fitting/sliding_window_predictions.gif
```

结果：

```text
observed_points: 111
window_size: 40
windows_total: 72
windows_success: 47
windows_failed: 25

last_window_predicted_xyz: (174.086196, -66.657514, 0.000000)
last_window_xy_error: 11.447386

best_window: start=67, end=106, end_time_ms=353.333333
best_predicted_xyz: (169.070640, -63.863869, 0.000000)
best_xy_error: 7.942937
```

结论：

- 40 帧窗口比全量拟合明显更准确。
- 最后窗口误差约 `11.45 cm`，已经接近可用精度。
- 最佳窗口误差约 `7.94 cm`。
- 预测曲线整体较稳定，适合作为当前样本的主要窗口设置。

### 3.2 20 帧滑动窗口

输出文件：

```text
landing_fitting/sliding_window_predictions_w20.csv
landing_fitting/sliding_window_predictions_w20.png
landing_fitting/sliding_window_predictions_w20.gif
```

结果：

```text
observed_points: 111
window_size: 20
windows_total: 92
windows_success: 69
windows_failed: 23

last_window_predicted_xyz: (189.122165, -71.012923, 0.000000)
last_window_xy_error: 26.377084

best_window: start=69, end=88, end_time_ms=293.333333
best_predicted_xyz: (163.830365, -66.306566, 0.000000)
best_xy_error: 2.673066
```

结论：

- 20 帧窗口的最佳窗口误差非常低，约 `2.67 cm`。
- 但 20 帧窗口整体波动明显大于 40 帧。
- 最后窗口误差约 `26.38 cm`，差于 40 帧最后窗口。
- 20 帧更适合分析“最早何时可能预测准确”，但不如 40 帧稳定。

### 3.3 结果对比

| 方法 | 窗口 | 最后窗口误差 | 最佳窗口误差 | 稳定性 |
|---|---:|---:|---:|---|
| 全量拟合 | 111 点 | 97.470595 cm | 不适用 | 差 |
| 滑动窗口 | 40 点 | 11.447386 cm | 7.942937 cm | 较好 |
| 滑动窗口 | 20 点 | 26.377084 cm | 2.673066 cm | 波动较大 |

综合当前单样本结果，`window_size=40` 更适合作为默认实验设置。

## 4. PNG 折线图分析

40 帧 PNG：

```text
landing_fitting/sliding_window_predictions.png
```

20 帧 PNG：

```text
landing_fitting/sliding_window_predictions_w20.png
```

PNG 图包含上下两个子图：

1. 上图：`xy_error` 随窗口结束时间变化。
2. 下图：预测的 `pred_x`、`pred_y` 与真实 `label_x`、`label_y` 对比。

### 4.1 40 帧 PNG 现象

40 帧图中，早期成功窗口的误差仍较高，随后逐渐下降。后期窗口中，预测点逐渐靠近 ground truth：

- `pred_x` 从较大的值逐步靠近 `label_x=162.839`。
- `pred_y` 也逐步靠近 `label_y=-68.789`。
- 后期误差稳定在十几厘米量级。

这说明使用后段窗口可以减少前段轨迹对拟合参数的拉偏。

### 4.2 20 帧 PNG 现象

20 帧图中，曲线波动更明显：

- 某些窗口预测非常准。
- 也有一些窗口出现明显尖峰。
- `pred_x`、`pred_y` 的变化比 40 帧窗口更剧烈。

这说明 20 帧窗口提供的信息量偏少，非线性拟合容易受局部噪声或参数病态影响。

## 5. GIF 动画分析

40 帧 GIF：

```text
landing_fitting/sliding_window_predictions.gif
```

20 帧 GIF：

```text
landing_fitting/sliding_window_predictions_w20.gif
```

GIF 中的图例：

- 灰色线和灰色点：完整观测轨迹的 XY 投影。
- 蓝色线和蓝色点：当前滑动窗口。
- 橙色线：历史预测落点轨迹。
- 红点：当前窗口预测落点。
- 黑色星号：ground truth。

### 5.1 羽毛球场坐标背景

GIF 中增加了羽毛球场划线，坐标系如下：

```text
原点 (0,0): 左侧短边中点
长边方向: X 轴，范围 0 到 1340
短边方向: Y 轴，范围 -335 到 335
```

绘制内容：

- 外框：`1340 x 670`
- 网线：`x = 670`
- 前发球线：`x = 472` 和 `x = 868`
- 双打长发球线：`x = 76` 和 `x = 1264`
- 中线：`y = 0`
- 单打边线：按标准宽度比例 `5.18 / 6.10` 缩放后绘制

这样可以直接观察预测落点和真实落点在球场内的位置关系。

### 5.2 40 帧 GIF 现象

40 帧 GIF 中可以看到：

- 早期预测点离 ground truth 较远。
- 随着窗口移动到轨迹后段，红色预测点逐渐靠近黑色 ground truth。
- 后期预测点在 ground truth 附近波动，最终窗口误差约 `11.45 cm`。

这与 PNG 中的误差下降趋势一致。

### 5.3 20 帧 GIF 现象

20 帧 GIF 中可以看到：

- 预测点移动更快，局部跳动更明显。
- 某些中间窗口可以非常接近 ground truth。
- 但后期并不一定保持最佳状态，最后窗口误差反而大于 40 帧。

这说明 20 帧窗口的预测更敏感，不适合作为稳定默认值。

## 6. 关于“预测点不沿窗口直线”的原因

GIF 中一个容易困惑的现象是：当前窗口的 XY 轨迹看起来近似直线，但红色预测点并不总是沿这条直线延伸。

原因是当前模型不是直接在 XY 平面拟合直线，而是分别拟合三个时间函数：

```text
x(t), y(t), z(t)
```

预测流程是：

1. 先根据 `z(t)=0` 求未来落地时间。
2. 再把这个时间代回 `x(t)` 和 `y(t)`。
3. 得到预测落点。

因此，预测点是时间维度上的非线性外推结果，不是 XY 平面上的直线延长点。

另外，当前窗口内只有短时间片段。例如 40 帧窗口长度约：

```text
40 / 300 * 1000 = 133.333 ms
```

但最后窗口从窗口末端到落地还需要继续外推约 `330 ms`。外推时间明显长于窗口自身持续时间，因此 `x(t)`、`y(t)` 或 `z(t)` 参数的细小偏差都会被放大。

这也是论文中引入 MLP 预测 `c1/c2/c3` 的原因之一：通过数据驱动参数给轨迹公式增加先验约束，降低纯非线性全参数拟合的不稳定性。

## 7. 当前结论

1. 当前单样本上，全量 111 点拟合误差较大，不推荐作为主要策略。
2. 40 帧滑动窗口明显改善误差，并且后期预测较稳定。
3. 20 帧滑动窗口有机会达到更低最佳误差，但整体波动较大。
4. PNG 更适合观察误差收敛趋势。
5. GIF 更适合观察预测点在球场坐标中的空间变化。
6. 对当前样本，推荐默认使用 `window_size=40`。
7. 如果要进一步接近论文，应补充 MLP 预测 `c1/c2/c3`，而不是完全依赖 `curve_fit` 全参数拟合。
