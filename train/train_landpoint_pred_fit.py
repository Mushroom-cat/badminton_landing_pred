"""
基于「拟合落点」的关键点落点预测训练脚本。

流程：
1. 以 ``datasets/20260418_label_fit_fall`` 为输入，用 ``landing_fitting/fit_trajectory.py``
   的滑动窗口拟合（取最后一个成功窗口的预测）算出每个 txt 对应的落点位置。
2. 在 ``datasets/scene3`` 中找到同名 txt（文件名最后一个 ``_`` 对应 scene3 的 ``---``），
   把该文件最后一行的落点坐标替换为拟合出的落点，写到新数据集 ``datasets/scene3+fit``。
   不修改任何原始 txt。
3. 用新数据集 ``datasets/scene3+fit`` 走与 ``train/train_landpoint_pred.py`` 相同的
   关键点落点预测训练流程。

用法：
    # 默认：先构建 scene3+fit 数据集，再训练
    python train/train_landpoint_pred_fit.py

    # 数据集已生成时，跳过构建直接训练
    python train/train_landpoint_pred_fit.py --skip_build

    # 可调拟合参数（滑动窗口大小、帧率、落点 z 等）与所有原训练超参
    python train/train_landpoint_pred_fit.py --fit_window_size 40 --epochs 100 --batch_size 32

常用参数：
    --fit_fall_dir     fit_trajectory 输入数据集目录（默认 datasets/20260418_label_fit_fall）
    --scene3_dir       原始 scene3 关键点数据集目录（默认 datasets/scene3）
    --scene3_fit_dir   生成的 scene3+fit 数据集目录（默认 datasets/scene3+fit）
    --fit_window_size  滑动窗口大小，取最后一个成功窗口的预测（默认 40）
    --fit_fps          拟合时间轴帧率（默认 300）
    --fit_landing_z    求解落点时间的目标 z 值（默认 0）
    --skip_build       跳过数据集构建，直接训练已有的 scene3+fit
    --epochs/--batch_size/--lr ...  其余训练超参与 train/train_landpoint_pred.py 一致
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib.util
import random
from datetime import datetime
import logging

import numpy as np
import torch

from util.dataset import BadmintonDataset, parse_dataset_fps_values, split_samples_by_dataset
from util.model import *
from util.trainer import Trainer
from analysis.visual_csv import visual_df


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FIT_FALL_DIR = os.path.join(PROJECT_ROOT, "datasets", "20260418_label_fit_fall")
DEFAULT_SCENE3_DIR = os.path.join(PROJECT_ROOT, "datasets", "scene3")
DEFAULT_SCENE3_FIT_DIR = os.path.join(PROJECT_ROOT, "datasets", "scene3+fit")
SCENE3_FIT_DATASET_NAME = "scene3+fit"


def _load_fit_module():
    """以文件路径方式加载 landing_fitting/fit_trajectory.py（该目录不是包）。"""
    fit_path = os.path.join(PROJECT_ROOT, "landing_fitting", "fit_trajectory.py")
    spec = importlib.util.spec_from_file_location("fit_trajectory_module", fit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene3_name_from_fit_name(fit_stem):
    """把 fit_fall 文件名（最后一个下划线分隔）映射到 scene3 文件名（--- 分隔）。

    例如 ``tuiqiu_round01_027940`` -> ``tuiqiu_round01---027940``。
    若没有下划线则原样返回。
    """
    if "_" not in fit_stem:
        return fit_stem
    head, _, tail = fit_stem.rpartition("_")
    return f"{head}---{tail}"


def _predict_landing_last_window(fit_mod, txt_path, fps, landing_z, window_size, logger):
    """用滑动窗口的「最后一个成功窗口」预测落点；失败时回退到整段拟合。

    返回 (predicted_xyz: np.ndarray(3,), method: str) 或在彻底失败时 (None, reason)。
    """
    from pathlib import Path

    observed, label = fit_mod.load_trajectory(Path(txt_path))
    n_obs = len(observed)
    effective_ws = min(window_size, n_obs)
    if effective_ws < 4:
        return None, f"observed points {n_obs} < 4"

    # 从最后一个窗口往前尝试，取第一个能成功拟合的窗口（即「最后一个成功窗口」）。
    for start in range(n_obs - effective_ws, -1, -1):
        window = observed[start:start + effective_ws]
        try:
            result = fit_mod.predict_window(window, label, fps, landing_z)
            return np.asarray(result["predicted"], dtype=np.float64), f"last_window[{start}:{start + effective_ws}]"
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    # 所有滑动窗口都失败 → 回退整段拟合。
    try:
        result = fit_mod.run(Path(txt_path), fps, landing_z)
        return np.asarray(result["predicted"], dtype=np.float64), "fallback_full_fit"
    except Exception as exc:  # noqa: BLE001
        return None, f"all windows failed ({last_exc}); full fit failed ({exc})"


def _replace_last_label_line(scene3_text, predicted_xyz):
    """复制 scene3 文件内容，把最后一行（落点标签）的坐标替换为拟合落点。

    最后一行格式为 ``frame_id:x,y,z``，仅替换 ``:`` 之后的坐标，保留 frame_id 与其余所有行。
    """
    lines = scene3_text.splitlines()
    # 找到最后一行非空行（落点标签行）
    last_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_idx = i
            break
    if last_idx is None:
        raise ValueError("scene3 文件没有有效内容")

    label_line = lines[last_idx].strip()
    if ":" not in label_line:
        raise ValueError(f"落点标签行格式异常: {label_line!r}")
    frame_id = label_line.split(":", 1)[0].strip()

    px, py, pz = (float(v) for v in predicted_xyz[:3])
    lines[last_idx] = f"{frame_id}:{px:.6f},{py:.6f},{pz:.6f}"
    return "\n".join(lines) + "\n"


def build_scene3_fit_dataset(fit_fall_dir, scene3_dir, output_dir,
                             fps, landing_z, window_size, logger, overwrite=True):
    """构建 scene3+fit 数据集，返回成功生成的样本数量。"""
    fit_mod = _load_fit_module()
    os.makedirs(output_dir, exist_ok=True)

    fit_files = sorted(f for f in os.listdir(fit_fall_dir) if f.endswith(".txt"))
    logger.info(f"在 {fit_fall_dir} 找到 {len(fit_files)} 个待拟合 txt")

    n_ok, n_skip = 0, 0
    for fit_name in fit_files:
        fit_path = os.path.join(fit_fall_dir, fit_name)
        stem = os.path.splitext(fit_name)[0]
        scene3_name = _scene3_name_from_fit_name(stem) + ".txt"
        scene3_path = os.path.join(scene3_dir, scene3_name)
        out_path = os.path.join(output_dir, scene3_name)

        if not os.path.isfile(scene3_path):
            logger.warning(f"[跳过] {fit_name}: 在 scene3 中找不到同名文件 {scene3_name}")
            n_skip += 1
            continue

        predicted, method = _predict_landing_last_window(
            fit_mod, fit_path, fps, landing_z, window_size, logger
        )
        if predicted is None:
            logger.warning(f"[跳过] {fit_name}: 拟合落点失败 ({method})")
            n_skip += 1
            continue

        if (not overwrite) and os.path.isfile(out_path):
            logger.info(f"[已存在] {scene3_name} 跳过写入")
            n_ok += 1
            continue

        with open(scene3_path, "r", encoding="utf-8") as f:
            scene3_text = f.read()
        try:
            new_text = _replace_last_label_line(scene3_text, predicted)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[跳过] {fit_name}: 替换落点标签失败 ({exc})")
            n_skip += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        logger.info(
            f"[生成] {scene3_name}  落点=({predicted[0]:.2f}, {predicted[1]:.2f}, {predicted[2]:.2f})  方法={method}"
        )
        n_ok += 1

    logger.info(f"scene3+fit 数据集构建完成: 成功 {n_ok}, 跳过 {n_skip} -> {output_dir}")
    return n_ok


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(timestamp, log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger


def main():
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    # ---- 数据集构建相关参数 ----
    parser.add_argument('--fit_fall_dir', type=str, default=DEFAULT_FIT_FALL_DIR,
                        help="fit_trajectory 输入数据集目录")
    parser.add_argument('--scene3_dir', type=str, default=DEFAULT_SCENE3_DIR,
                        help="原始 scene3 关键点数据集目录")
    parser.add_argument('--scene3_fit_dir', type=str, default=DEFAULT_SCENE3_FIT_DIR,
                        help="生成的 scene3+fit 数据集目录")
    parser.add_argument('--fit_fps', type=float, default=300.0,
                        help="拟合时间轴使用的帧率")
    parser.add_argument('--fit_landing_z', type=float, default=0.0,
                        help="求解落点时间的目标 z 值")
    parser.add_argument('--fit_window_size', type=int, default=40,
                        help="滑动窗口大小（取最后一个成功窗口的预测）")
    parser.add_argument('--skip_build', action='store_true',
                        help="若设置则跳过数据集构建，直接训练已有的 scene3+fit")

    # ---- 训练相关参数（与 train_landpoint_pred.py 对齐）----
    parser.add_argument('--data_folder', type=str, default=None,
                        help="训练数据集路径；默认使用生成的 scene3+fit 目录")
    parser.add_argument('--data_folders', type=str, nargs='+', default=None,
                        help="多个训练数据集路径；若提供则忽略 --data_folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--points_num", type=int, default=22)
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--min_offset_len", type=int, default=0)
    parser.add_argument("--max_offset_len", type=int, default=4)
    parser.add_argument("--temp_test_offset", type=int, default=0)
    parser.add_argument("--num_subsamples", type=int, default=5)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--lambda_time", type=float, default=0.1)
    parser.add_argument("--lambda_direction", type=float, default=0.1)
    parser.add_argument("--aug_method", type=str, default='None')
    parser.add_argument("--use_time_pos_encoding", action="store_true")
    parser.add_argument("--time_label_unit", choices=["frames", "seconds"], default="frames")
    parser.add_argument("--reference_fps", type=float, default=300.0)
    parser.add_argument("--hit_index", type=int, default=100)
    parser.add_argument("--dataset_fps", type=str, nargs="*", default=None,
                        help="Dataset fps mapping, e.g. scene3+fit=300")
    args = parser.parse_args()

    # Init Logger
    logger = setup_logger(start_time, log_dir="./logs")
    logger.time = start_time

    # ====== 1. 构建 scene3+fit 数据集 ======
    set_seed()
    if not args.skip_build:
        logger.info("========== 🛠 构建 scene3+fit 数据集 ==========")
        n_ok = build_scene3_fit_dataset(
            fit_fall_dir=args.fit_fall_dir,
            scene3_dir=args.scene3_dir,
            output_dir=args.scene3_fit_dir,
            fps=args.fit_fps,
            landing_z=args.fit_landing_z,
            window_size=args.fit_window_size,
            logger=logger,
        )
        if n_ok < 2:
            raise RuntimeError(
                f"scene3+fit 仅成功生成 {n_ok} 个样本，至少需要 2 个才能划分训练/测试集"
            )
    else:
        logger.info("========== ⏭ 跳过数据集构建，直接使用已有 scene3+fit ==========")

    # scene3 数据集帧率为 300，确保 scene3+fit 也能解析到 fps
    dataset_fps = parse_dataset_fps_values(args.dataset_fps)
    dataset_fps.setdefault(SCENE3_FIT_DATASET_NAME, 300.0)

    # 默认训练数据集指向生成的 scene3+fit
    if args.data_folder is None:
        args.data_folder = args.scene3_fit_dir

    # ====== 2. 定义模型 ======
    set_seed()
    if args.use_time_pos_encoding:
        model = ImprovedTransformerTimePEModel(
            seq_len=args.max_len,
            num_points=args.points_num,
            reference_fps=args.reference_fps,
        )
    else:
        model = ImprovedTransformerModel(seq_len=args.max_len, num_points=args.points_num)

    logger.info("========== Model 信息 ==========")
    logger.info(model)

    logger.info("========== Config 信息 ==========")
    for arg in vars(args):
        logger.info(f"--{arg}: {getattr(args, arg)}")
    logger.info(f"--resolved_dataset_fps: {dataset_fps}")

    # ====== 3. 加载数据 ======
    set_seed()
    logger.info("========== 📂 Loading samples ==========")
    data_folders = args.data_folders if args.data_folders else [args.data_folder]
    logger.info(f"使用数据集目录: {data_folders}")

    train_samples, test_samples, dataset_stats = split_samples_by_dataset(
        data_folders, point_num=args.points_num, train_ratio=0.8, dataset_fps=dataset_fps
    )
    for stat in dataset_stats:
        logger.info(
            f"数据集 {stat['dataset']} ({stat['folder']}): "
            f"总样本 {stat['total']}, 训练 {stat['train']}, 测试 {stat['test']}"
        )
    logger.info(f"合并后训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}")

    train_dataset = BadmintonDataset(
        train_samples,
        mode="train",
        min_len=args.min_len,
        max_len=args.max_len,
        min_offset_len=args.min_offset_len,
        max_offset_len=args.max_offset_len,
        temp_test_offset=args.temp_test_offset,
        num_subsamples=args.num_subsamples,
        aug_method=args.aug_method,
        use_time_pos_encoding=args.use_time_pos_encoding,
        time_label_unit=args.time_label_unit,
        reference_fps=args.reference_fps,
        hit_index=args.hit_index,
    )
    feat_mean, feat_std, label_mean, label_std = train_dataset.get_norm_stats()
    test_dataset = BadmintonDataset(
        test_samples,
        mode="test",
        max_len=args.max_len,
        min_offset_len=args.min_offset_len,
        max_offset_len=args.max_offset_len,
        temp_test_offset=args.temp_test_offset,
        feature_mean=feat_mean,
        feature_std=feat_std,
        label_mean=label_mean,
        label_std=label_std,
        use_time_pos_encoding=args.use_time_pos_encoding,
        time_label_unit=args.time_label_unit,
        reference_fps=args.reference_fps,
        hit_index=args.hit_index,
    )

    logger.info("========== Training Data 统计信息 ==========")
    logger.info(f"训练样本数量: {len(train_dataset)}")
    logger.info(f"测试样本数量: {len(test_dataset)}")
    logger.info(f"特征 mean: {feat_mean.shape}, 示例前5个维度: {feat_mean[0, :5]}")
    logger.info(f"特征 std : {feat_std.shape}, 示例前5个维度: {feat_std[0, :5]}")
    logger.info(f"标签 mean: {label_mean.shape}, 值: {label_mean[0]}")
    logger.info(f"标签 std : {label_std.shape}, 值: {label_std[0]}")

    # ====== 4. 定义 Trainer ======
    trainer = Trainer(
        args=args,
        logger=logger,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.model_dir,
    )

    # ====== 5. 训练 ======
    trainer.train(num_epochs=args.epochs)

    # ====== 6. 测试并保存结果 ======
    res_df = trainer.test_and_save(save_dir=args.results_dir)

    # ====== 7. 可视化 ======
    set_seed()
    visual_df(model.name, start_time, res_df)
    logger.info("📄 Saved visualization")


if __name__ == "__main__":
    main()
