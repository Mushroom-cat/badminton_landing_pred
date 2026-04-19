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
import matplotlib.pyplot as plt


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


def main(outside_temp_test_offset):
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_folder', type=str, default='/home/zhaoxuhao/badminton_xh/20250809_Seq_data_v2/20250809_Seq_data')
    parser.add_argument('--data_folder', type=str, default='datasets/data_1217_ball_ext5')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--hidden_dim", type=int, default=128)
    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--min_offset_len", type=int, default=0)
    parser.add_argument("--max_offset_len", type=int, default=4)
    parser.add_argument("--temp_test_offset", type=int, default=-1)
    parser.add_argument("--num_subsamples", type=int, default=5)
    parser.add_argument("--delta", type=float, default=1.0)  # for huber loss (xyz loss)
    parser.add_argument("--lambda_time", type=float, default=0.1)  # for huber loss (time loss)
    parser.add_argument("--lambda_direction", type=float, default=0.1)  # for huber loss (direction loss)
    parser.add_argument("--aug_method", type=str, default='None')  # 可选：None, '平移', '旋转', '缩放', '噪声'
    args = parser.parse_args()
    args.temp_test_offset = outside_temp_test_offset

    # 定义模型
    set_seed()
    # model = RNNRegressor()
    # model = LSTMRegressor()
    # model = ImprovedLSTMRegressor()
    # model = SimplifiedLSTMRegressor()
    # model = TransformerModel()
    model = ImprovedTransformerModel(seq_len=args.max_len)

    # Init Logger
    logger = setup_logger(start_time, log_dir="./logs")
    logger.time = start_time

    logger.info("========== Model 信息 ==========")
    logger.info(model)

    logger.info("========== Config 信息 ==========")
    for arg in vars(args):
        logger.info(f"--{arg}: {getattr(args, arg)}")

    # 1. 加载数据
    set_seed()
    logger.info("========== 📂 Loading samples ==========")
    samples = load_all_samples(args.data_folder)
    random.shuffle(samples)
    logger.info(f"一共加载到 {len(samples)} 个样本")

    # 划分 train/test
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    train_dataset = BadmintonDataset(train_samples, mode="train", min_len=args.min_len, max_len=args.max_len,
                                     min_offset_len=args.min_offset_len, max_offset_len=args.max_offset_len,
                                     temp_test_offset=args.temp_test_offset, num_subsamples=args.num_subsamples,
                                     aug_method=args.aug_method)
    feat_mean, feat_std, label_mean, label_std = train_dataset.get_norm_stats()
    test_dataset = BadmintonDataset(test_samples, mode="test", max_len=args.max_len, min_offset_len=args.min_offset_len,
                                    max_offset_len=args.max_offset_len, temp_test_offset=args.temp_test_offset,
                                    feature_mean=feat_mean, feature_std=feat_std,
                                    label_mean=label_mean, label_std=label_std)
    # 4. 打印
    logger.info("========== Training Data 统计信息 ==========")
    logger.info(f"训练样本数量: {len(train_dataset)}")
    logger.info(f"测试样本数量: {len(test_dataset)}")
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

    # 4. 训练
    trainer.train(num_epochs=args.epochs)

    # 5. 测试并保存结果
    res_df = trainer.test_and_save(save_dir=args.results_dir)

    # 6. 可视化
    set_seed()
    visual_df(model.name, start_time, res_df)
    logger.info(f"📄 Saved visualization")

    return res_df


def visual_uncertainty_list(df_list, threshold=75, labels=None):
    """
    df_list : list of pandas.DataFrame
    threshold : 不确定性阈值
    labels : list of str, 每个 df 对应的图例名称
    """

    print("\n======== 可视化不确定性估计 ========")

    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(df_list))]

    plt.figure(figsize=(10, 6))
    global_max_std = 0.0

    for idx, (df, label) in enumerate(zip(df_list, labels)):
        df = df.copy()

        # 计算不确定性（以 aleatoric 为例）
        df['std_euclidean'] = np.sqrt(
            df['std_x'] ** 2 +
            df['std_y'] ** 2 +
            df['std_z'] ** 2
        )

        # 更新全局最大值
        current_max = df['std_euclidean'].max()
        if current_max > global_max_std:
            global_max_std = current_max

        # 绘制散点
        plt.scatter(
            df['err_euclidean'],
            df['std_euclidean'],
            alpha=0.6,
            s=20,
            label=label
        )

        # ================== 统计分析（每个 df 单独） ==================
        df_less_1m = df[df['err_euclidean'] <= 100]
        df_more_1m = df[df['err_euclidean'] > 100]

        less_1m = len(df_less_1m[df_less_1m['std_euclidean'] < threshold])
        more_1m = len(df_more_1m[df_more_1m['std_euclidean'] > threshold])

        print(f"\n[{label}]")
        print(f"在设定不确定性的截断阈值为 {threshold} 的情况下：")
        if len(df_less_1m) == 0:
            print("无 <1m 样本")
        else:
            print(f"保留的 <1m 样本占比 {less_1m / len(df_less_1m):.4f}")
        if len(df_more_1m) == 0:
            print("无 >1m 样本")
        else:
            print(f"丢弃的 >1m 样本占比 {more_1m / len(df_more_1m):.4f}")

        df_less_threshold = df[df['std_euclidean'] <= threshold]
        df_more_threshold = df[df['std_euclidean'] > threshold]

        less_threshold_true = len(df_less_threshold[df_less_threshold['err_euclidean'] <= 100])
        more_threshold_true = len(df_more_threshold[df_more_threshold['err_euclidean'] > 100])

        if len(df_less_threshold) == 0:
            print("无保留的样本")
        else:
            print(f"保留的样本中正确保留占比 {less_threshold_true / len(df_less_threshold):.4f}")
        if len(df_more_threshold) == 0:
            print("无丢弃的样本")
        else:
            print(f"丢弃的样本中正确丢弃占比 {more_threshold_true / len(df_more_threshold):.4f}")

    # ================== 图像整体设置 ==================
    plt.title('Predictive Error vs. Predictive Uncertainty')
    plt.xlabel('Predictive Error')
    plt.ylabel('Predictive Uncertainty')
    plt.ylim(0, global_max_std * 1.07)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('visualization/Error-std_aleatoric_Correlation_multi_df', bbox_inches='tight')

    print("\n成功保存多数据集预测误差-不确定性的散点图")
    print("\n================================\n")

if __name__ == "__main__":
    df_list = list()
    offset_list = [0,1,2,3,4]
    for ii, outside_temp_test_offset in enumerate(offset_list):
        next_df = main(outside_temp_test_offset)
        df_list.append(next_df)

    visual_uncertainty_list(df_list, threshold=70, labels=[5,4,3,2,1])