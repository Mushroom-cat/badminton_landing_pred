import subprocess
import sys
import os
import argparse
import random
import logging
from datetime import datetime

import numpy as np
import torch

from util.dataset import BadmintonDataset, split_samples_by_dataset
from util.model import ImprovedTransformerModel
from util.trainer import Trainer
from analysis.visual_csv import visual_df


# ========================= 工具函数 =========================

def set_seed(seed=42):
    """设置全局随机种子，确保可复现性"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(timestamp, log_dir="./logs"):
    """配置日志记录器，同时输出到文件和控制台"""
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


def run_python_script(python_path, script_path, args_list):
    """
    执行单个Python脚本
    :param python_path: Python解释器路径
    :param script_path: 要执行的py文件路径
    :param args_list: 参数列表（如 ["--input", "data.txt"]）
    :return: (是否成功, 退出码)
    """
    if not os.path.exists(script_path):
        print(f"[错误] 脚本文件不存在：{script_path}")
        return False, -1

    cmd = [python_path, script_path] + args_list
    print(f"执行命令：{' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            encoding="utf-8",
            shell=False
        )
        success = result.returncode == 0
        if success:
            print(f"[成功] {script_path} 执行完成\n")
        else:
            print(f"[失败] {script_path} 执行出错，退出码：{result.returncode}\n")
        return success, result.returncode
    except Exception as e:
        print(f"[异常] {script_path} 执行抛出异常：{str(e)}\n")
        return False, -2


# ========================= 数据采集 =========================

def run_data_pipeline(python_path):
    """
    执行数据采集：
      1. 球拍关键点推理 (racket_infer.py)
      2. 球检测推理 (ball_infer.py)
      3. 2D → 3D 坐标转换 (cvt_2d_3d.py)
      4. 合并 3D 数据为训练样本 (Infered_3D_to_Sample.py)
    """
    SCRIPTS_TO_RUN = [
        # ("./Labeled_2D_Points_Racket/select_training.py", []),
        ("./Infered_2D_Points_Racket/racket_infer.py", []),
        ("./Infered_2D_Points_Ball/ball_infer.py", []),
        ("cvt_2d_3d.py", []),
        ("./datasets/Infered_3D_to_Sample.py", []),
    ]

    print("=" * 50)
    print(f"[数据采集] 开始执行 {len(SCRIPTS_TO_RUN)} 个脚本...")
    print("=" * 50 + "\n")

    all_success = True
    for idx, (script_path, args) in enumerate(SCRIPTS_TO_RUN, 1):
        print(f"[第 {idx} 步] 开始执行：{script_path}")
        print("-" * 30)
        success, exit_code = run_python_script(python_path, script_path, args)
        if not success:
            all_success = False

    return all_success


# ========================= 落点预测训练 =========================

def parse_train_args():
    """解析落点预测训练参数"""
    parser = argparse.ArgumentParser(description="羽毛球落点预测完整")

    # 控制
    parser.add_argument("--skip_data_pipeline", action="store_true", default=False,
                        help="跳过数据采集，直接进行训练")
    parser.add_argument("--skip_train", action="store_true", default=False,
                        help="跳过训练，仅执行数据采集")

    # 数据参数
    parser.add_argument('--data_folder', type=str, default='datasets/scene1+2',
                        help="训练数据集路径；未设置 --data_folders 时使用")
    parser.add_argument('--data_folders', type=str, nargs='+', default=None,
                        help="多个训练数据集路径；若提供则忽略 --data_folder")
    parser.add_argument("--points_num", type=int, default=22,
                        help="关键点数量")
    parser.add_argument("--min_len", type=int, default=10,
                        help="最小序列长度")
    parser.add_argument("--max_len", type=int, default=50,
                        help="最大序列长度")
    parser.add_argument("--min_offset_len", type=int, default=5)
    parser.add_argument("--max_offset_len", type=int, default=25)
    parser.add_argument("--temp_test_offset", type=int, default=5)
    parser.add_argument("--num_subsamples", type=int, default=5)
    parser.add_argument("--aug_method", type=str, default='None',
                        help="数据增强方法: None, 平移, 旋转, 缩放, 噪声")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Huber loss delta (xyz)")
    parser.add_argument("--lambda_time", type=float, default=0.1,
                        help="时间预测损失权重")
    parser.add_argument("--lambda_direction", type=float, default=0.1,
                        help="方向预测损失权重")

    # 输出路径
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--results_dir", type=str, default="./results")

    return parser.parse_args()


def step1_init_model(args, logger):
    """步骤1: 初始化模型"""
    logger.info("=" * 40)
    logger.info("[步骤1] 初始化模型")
    logger.info("=" * 40)

    set_seed()
    model = ImprovedTransformerModel(seq_len=args.max_len, num_points=args.points_num)

    logger.info(f"模型: {model.name}")
    logger.info(model)

    logger.info("========== Config 信息 ==========")
    for arg in vars(args):
        logger.info(f"--{arg}: {getattr(args, arg)}")

    return model


def step2_load_data(args, logger):
    """步骤2: 加载并划分数据集"""
    logger.info("=" * 40)
    logger.info("[步骤2] 加载数据集")
    logger.info("=" * 40)

    set_seed()
    data_folders = args.data_folders if args.data_folders else [args.data_folder]
    logger.info(f"使用数据集目录: {data_folders}")

    train_samples, test_samples, dataset_stats = split_samples_by_dataset(
        data_folders, point_num=args.points_num, train_ratio=0.8
    )
    for stat in dataset_stats:
        logger.info(
            f"数据集 {stat['dataset']} ({stat['folder']}): "
            f"总样本 {stat['total']}, 训练 {stat['train']}, 测试 {stat['test']}"
        )
    logger.info(f"合并后训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}")

    return train_samples, test_samples


def step3_build_datasets(args, train_samples, test_samples, logger):
    """步骤3: 构建训练/测试 Dataset"""
    logger.info("=" * 40)
    logger.info("[步骤3] 构建 Dataset")
    logger.info("=" * 40)

    train_dataset = BadmintonDataset(
        train_samples, mode="train",
        min_len=args.min_len, max_len=args.max_len,
        min_offset_len=args.min_offset_len, max_offset_len=args.max_offset_len,
        temp_test_offset=args.temp_test_offset, num_subsamples=args.num_subsamples,
        aug_method=args.aug_method
    )
    feat_mean, feat_std, label_mean, label_std = train_dataset.get_norm_stats()

    test_dataset = BadmintonDataset(
        test_samples, mode="test",
        max_len=args.max_len,
        min_offset_len=args.min_offset_len, max_offset_len=args.max_offset_len,
        temp_test_offset=args.temp_test_offset,
        feature_mean=feat_mean, feature_std=feat_std,
        label_mean=label_mean, label_std=label_std
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"特征 mean shape: {feat_mean.shape}, 前5维: {feat_mean[0, :5]}")
    logger.info(f"特征 std  shape: {feat_std.shape},  前5维: {feat_std[0, :5]}")
    logger.info(f"标签 mean: {label_mean[0]}")
    logger.info(f"标签 std : {label_std[0]}")

    return train_dataset, test_dataset


def step4_train(args, model, train_dataset, test_dataset, logger):
    """步骤4: 训练模型"""
    logger.info("=" * 40)
    logger.info("[步骤4] 开始训练")
    logger.info("=" * 40)

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

    trainer.train(num_epochs=args.epochs)
    return trainer


def step5_test_and_save(trainer, args, logger):
    """步骤5: 测试模型并保存预测结果"""
    logger.info("=" * 40)
    logger.info("[步骤5] 测试并保存结果")
    logger.info("=" * 40)

    res_df = trainer.test_and_save(save_dir=args.results_dir)
    return res_df


def step6_visualize(model, start_time, res_df, logger):
    """步骤6: 可视化预测结果"""
    logger.info("=" * 40)
    logger.info("[步骤6] 可视化")
    logger.info("=" * 40)

    set_seed()
    visual_df(model.name, start_time, res_df)
    logger.info("可视化结果已保存")


# ========================= 主流程 =========================

def main():
    args = parse_train_args()

    PYTHON_PATH = (
        "E:\\Pointnet_Pointnet2_pytorch-master\\"
        "Pointnet_Pointnet2_pytorch-master\\.venv\\Scripts\\python.exe"
    )

    # ---- 阶段一：数据采集 ----
    if not args.skip_data_pipeline:
        pipeline_ok = run_data_pipeline(PYTHON_PATH)
        if not pipeline_ok:
            print("[警告] 数据采集部分脚本执行失败，请检查！")
    else:
        print("[跳过] 数据采集")

    # ---- 阶段二：落点预测训练 ----
    if not args.skip_train:
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = setup_logger(start_time, log_dir="./logs")
        logger.time = start_time

        print("\n" + "=" * 50)
        print("开始落点预测训练...")
        print("=" * 50 + "\n")

        # 步骤 1: 初始化模型
        model = step1_init_model(args, logger)

        # 步骤 2: 加载数据
        train_samples, test_samples = step2_load_data(args, logger)

        # 步骤 3: 构建 Dataset
        train_dataset, test_dataset = step3_build_datasets(
            args, train_samples, test_samples, logger
        )

        # 步骤 4: 训练
        trainer = step4_train(args, model, train_dataset, test_dataset, logger)

        # 步骤 5: 测试并保存
        res_df = step5_test_and_save(trainer, args, logger)

        # 步骤 6: 可视化
        step6_visualize(model, start_time, res_df, logger)

        print("\n" + "=" * 50)
        print("落点预测训练执行完毕！")
        print("=" * 50)
    else:
        print("[跳过] 落点预测训练")

    # ---- 最终汇总 ----
    print("\n" + "=" * 50)
    print("所有流程执行完毕！")
    print("=" * 50)


if __name__ == "__main__":
    main()
