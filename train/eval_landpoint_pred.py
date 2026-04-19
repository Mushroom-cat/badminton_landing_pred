import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
from util.dataset import load_all_samples, BadmintonDataset, resampling
from util.model import *
from util.trainer import Trainer
from analysis.visual_csv import visual_df

import logging
from datetime import datetime


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for deterministic cudnn (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(timestamp, log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,  # 记录级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),  # 写入文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logger = logging.getLogger()
    return logger

def main():
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_folder', type=str, default='/home/zhaoxuhao/badminton_xh/20250809_Seq_data_v2/20250809_Seq_data')
    parser.add_argument('--data_folder', type=str, default='datasets/20260106_pose_infer')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--hidden_dim", type=int, default=128)
    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--points_num", type=int, default=22)
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--min_offset_len", type=int, default=5)
    parser.add_argument("--max_offset_len", type=int, default=25)
    parser.add_argument("--temp_test_offset", type=int, default=25)
    parser.add_argument("--num_subsamples", type=int, default=5)
    parser.add_argument("--delta", type=float, default=1.0)  # for huber loss (xyz loss)
    parser.add_argument("--lambda_time", type=float, default=0.1)  # for huber loss (time loss)
    parser.add_argument("--lambda_direction", type=float, default=0.1)  # for huber loss (direction loss)
    parser.add_argument("--aug_method", type=str, default='None')  # 可选：None, '平移', '旋转', '缩放', '噪声'
    args = parser.parse_args()

    # 1. 加载数据
    set_seed()
    # 2. 定义模型
    # model = LSTMRegressor()
    # model = ImprovedLSTMRegressor()
    # model = SimplifiedLSTMRegressor()
    model = ImprovedTransformerModel(seq_len=args.max_len, num_points=args.points_num)

    set_seed()
    # samples = load_all_samples("datasets/data_1217_ball_ext5", args.points_num)
    samples = load_all_samples(args.data_folder, args.points_num)
    random.shuffle(samples)
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    train_dataset = BadmintonDataset(train_samples, mode="train", min_len=args.min_len, max_len=args.max_len,
                                     min_offset_len=args.min_offset_len, max_offset_len=args.max_offset_len,
                                     temp_test_offset=args.temp_test_offset, num_subsamples=args.num_subsamples,
                                     aug_method=args.aug_method)
    feat_mean, feat_std, label_mean, label_std = train_dataset.get_norm_stats()

    logger = setup_logger(start_time, log_dir="./logs")
    logger.time = start_time
    logger.info("========== 📂 Loading samples ==========")
    samples = load_all_samples(args.data_folder, args.points_num)
    random.shuffle(samples)
    logger.info(f"一共加载到 {len(samples)} 个样本")

    # test_samples = samples[split_idx:]
    test_samples = samples[:]

    # train_dataset = BadmintonDataset(train_samples, mode="train", min_len=args.min_len, max_len=args.max_len,
    #                                  min_offset_len=args.min_offset_len, max_offset_len=args.max_offset_len,
    #                                  temp_test_offset=args.temp_test_offset, num_subsamples=args.num_subsamples,
    #                                  aug_method=args.aug_method)
    # feat_mean, feat_std, label_mean, label_std = train_dataset.get_norm_stats()
    test_dataset = BadmintonDataset(test_samples, mode="test", max_len=args.max_len, min_offset_len=args.min_offset_len,
                                    max_offset_len=args.max_offset_len, temp_test_offset=args.temp_test_offset,
                                    feature_mean=feat_mean, feature_std=feat_std,
                                    label_mean=label_mean, label_std=label_std)
    # 4. 打印
    logger.info("========== Training Data 统计信息 ==========")
    logger.info(f"样本数量: {len(train_dataset)}")
    logger.info(f"特征 mean: {feat_mean.shape}, 示例前5个维度: {feat_mean[0, :5]}")
    logger.info(f"特征 std : {feat_std.shape}, 示例前5个维度: {feat_std[0, :5]}")

    logger.info(f"标签 mean: {label_mean.shape}, 值: {label_mean[0]}")
    logger.info(f"标签 std : {label_std.shape}, 值: {label_std[0]}")

    # 3. 定义 Trainer
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

    # 5. 测试并保存结果
    trainer.best_model_path = r'./models/ImprovedTransformerModel_20260419_202118.pt'
    res_df = trainer.test_and_save(save_dir=args.results_dir)

    set_seed()
    visual_df(model.name, start_time, res_df)
    logger.info(f"📄 Saved visualization")

if __name__ == "__main__":
    main()
