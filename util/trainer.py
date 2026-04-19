# trainer.py
import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
from util.dataset import collate_fn_dynamic
from util.model import EndToEndModel


def nll_loss(pred_mu, pred_log_var, target, lambda_nll=1.0):
    # 钳位 log_var 以避免数值不稳定 (可选，但推荐)
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)

    # 方差 sigma^2
    pred_var = torch.exp(pred_log_var)

    # 损失项 1: log(sigma^2)
    loss_term_1 = lambda_nll * pred_log_var

    # 损失项 2: (y - mu)^2 / sigma^2 (带权重的 MSE)
    loss_term_2 = torch.square(target - pred_mu) / pred_var

    # 整体损失
    loss = 0.5 * (loss_term_1 + loss_term_2)

    return loss.mean()


def nll_loss(pred_mu, pred_log_var, target, lambda_nll=1.0):
    # 钳位 log_var 以避免数值不稳定 (可选，但推荐)
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)

    # 方差 sigma^2
    pred_var = torch.exp(pred_log_var)

    # 损失项 1: log(sigma^2)
    loss_term_1 = lambda_nll * pred_log_var

    # 损失项 2: (y - mu)^2 / sigma^2 (带权重的 MSE)
    loss_term_2 = torch.square(target - pred_mu) / pred_var

    # 整体损失
    loss = 0.5 * (loss_term_1 + loss_term_2)

    return loss.mean()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, args, logger, model, train_dataset, test_dataset, device=None, batch_size=32, lr=1e-3, seed=42,
                 save_dir="models", pretrained_path=None, finetune=False):
        set_seed(seed)
        self.logger = logger
        self.args = args
        self.lambda_time = args.lambda_time
        self.lambda_direction = args.lambda_direction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       collate_fn=lambda batch: collate_fn_dynamic(batch, max_len=args.max_len),
                                       num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: collate_fn_dynamic(batch, max_len=args.max_len),
                                      num_workers=0)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        if finetune:
            backbone_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if not ("mlp" in name):
                    # print('backbone', name)
                    backbone_params.append(param)
                else:
                    # print('head', name)
                    head_params.append(param)
            self.optimizer = torch.optim.Adam([
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": head_params, "lr": lr},
            ], weight_decay=1e-5)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=1e-5
            )

        self.criterion = torch.nn.MSELoss()
        self.criterion_cos = torch.nn.CosineEmbeddingLoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.regression_criterion_xyz = nll_loss

        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, f"{self.model.name}_{self.logger.time}.pt")
        os.makedirs(save_dir, exist_ok=True)

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)
            self.logger.info(f"Loaded pretrained model from {pretrained_path}")

    def cal_xyz_loss(self, pred_xyz, labels_xyz):
        # 1. 计算元素级的 error（向量/张量）
        error = pred_xyz - labels_xyz  # 形状与 y_pred 相同，如 [batch_size, dim]

        # 2. 元素级判断：|error| ≤ delta → 掩码为 True，否则为 False
        # abs(error) 是元素级绝对值，self.delta 会广播到与 error 同形状
        mask = torch.abs(error) <= self.args.delta

        # 3. 元素级计算分段损失
        # 掩码为 True 的位置：用 MSE 项；False 的位置：用 MAE 项
        loss = torch.where(
            mask,
            0.5 * torch.square(error),  # MSE 分支（元素级平方）
            self.args.delta * (torch.abs(error) - 0.5 * self.args.delta)  # MAE 分支
        )
        return loss.mean()
        # return self.criterion(pred_xyz, labels_xyz)

    def train(self, num_epochs=50):
        best_epoch, best_loss, best_xyz_err, best_time_err = 0, float("inf"), float("inf"), float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            # xyz_alpha = (5 - epoch*0.5//10) if epoch <= 30 else 2
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [b.to(self.device) for b in
                                                                                   batch[:-1]]

                self.optimizer.zero_grad()
                pred_xyz, pred_xyz_var, pred_time, pred_direction = self.model(seqs, masks)

                # loss_xyz = self.cal_xyz_loss(pred_xyz, labels_xyz)
                loss_xyz = self.regression_criterion_xyz(pred_xyz, pred_xyz_var, labels_xyz)
                loss_time = self.criterion(pred_time, labels_time)
                # loss_time = self.l1_criterion(pred_time, labels_time)
                targets = torch.ones(labels_direction.size(0), device=self.device)
                loss_direction = self.criterion_cos(pred_direction, labels_direction, targets)
                loss = loss_xyz + self.lambda_time * loss_time + self.lambda_direction * loss_direction

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(self.train_loader)
            res_str = f"Epoch {epoch + 1}: [Train] Loss = {avg_loss:.4f}|"

            # validation
            avg_xyz_loss, avg_time_loss, avg_dir_loss, avg_xyz_err, avg_xy_err, avg_time_err, avg_direction_err = self.evaluate()
            res_str += f"\t[Valid] XYZ Loss = {avg_xyz_loss:.4f}| Time Loss = {avg_time_loss:.4f}| Dir Loss = {avg_dir_loss:.4f}|"
            res_str += f"\tXYZ Err(L2 Distance): {avg_xyz_err:.4f}|\tXY Err: {avg_xy_err:.4f}|\tTime Err: {avg_time_err:.4f}|\tDirection Err: {avg_direction_err:.4f}"
            self.logger.info(res_str)

            # save best model
            # if avg_xyz_loss + self.lambda_time * avg_time_loss + self.lambda_direction * avg_dir_loss < best_loss:
            if avg_xyz_loss < best_loss:
                # if avg_xyz_err < best_xyz_err:
                best_epoch = epoch
                best_loss = avg_xyz_loss
                best_xyz_err = avg_xyz_err
                best_time_err = avg_time_err
                best_direction_err = avg_direction_err
                torch.save(self.model.state_dict(), self.best_model_path)

                # save model as onnx
                self.model.to('cpu')
                feat_dim = self.train_loader.dataset.feature_mean.shape[-1]
                max_len = self.train_loader.dataset.max_len
                dummy_seq = torch.randn(1, max_len, feat_dim)
                dummy_mask = torch.ones(1, max_len, dtype=torch.bool)

                model_with_norm = EndToEndModel(self.model, torch.tensor(self.train_loader.dataset.feature_mean),
                                                torch.tensor(self.train_loader.dataset.feature_std),
                                                torch.tensor(self.train_loader.dataset.label_mean),
                                                torch.tensor(self.train_loader.dataset.label_std))
                model_with_norm.eval()

                torch.onnx.export(
                    model_with_norm,
                    (dummy_seq, dummy_mask),
                    self.best_model_path.replace('.pt', '.onnx'),
                    input_names=['seq', 'mask'],
                    output_names=['xyz', 'var', 'time', 'direction'],
                    dynamic_axes={"seq": {0: "batch_size", 1: "seq_len", 2: "feature_dim"},
                                  "mask": {0: "batch_size", 1: "seq_len"}}
                )
                self.model.to(self.device)

                self.logger.info(f"✅ Saved best model to {self.best_model_path}")

        self.logger.info(
            f"Training finished. Best epoch: {best_epoch}|\tBest [Valid] loss: {best_loss:.4f}|\tXYZ Err: {best_xyz_err:.4f}|\tTime Err: {best_time_err:.4f}|\tDir Err: {best_direction_err:.2f}")

    def evaluate(self):
        self.model.eval()
        xyz_loss, time_loss, dir_loss = 0.0, 0.0, 0.0
        preds, labels = [], []
        dir_preds, dir_labels = [], []
        label_mean = self.test_loader.dataset.label_mean
        label_std = self.test_loader.dataset.label_std

        with torch.no_grad():
            for batch in self.test_loader:
                seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [b.to(self.device) for b in
                                                                                   batch[:-1]]
                pred_xyz, pred_xyz_var, pred_time, pred_direction = self.model(seqs, masks)

                # loss_xyz = self.criterion(pred_xyz, labels_xyz)
                loss_xyz = self.regression_criterion_xyz(pred_xyz, pred_xyz_var, labels_xyz)
                loss_time = self.criterion(pred_time, labels_time)
                targets = torch.ones(labels_direction.size(0), device=self.device)
                loss_dir = self.criterion_cos(pred_direction, labels_direction, targets)

                xyz_loss += loss_xyz.item()
                time_loss += loss_time.item()
                dir_loss += loss_dir.item()

                preds.append(torch.cat([pred_xyz, pred_time.unsqueeze(1)], dim=-1).cpu().numpy())
                labels.append(torch.cat([labels_xyz, labels_time.unsqueeze(1)], dim=-1).cpu().numpy())
                dir_preds.append(pred_direction.cpu().numpy())
                dir_labels.append(labels_direction.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        dir_preds = np.concatenate(dir_preds, axis=0)
        dir_labels = np.concatenate(dir_labels, axis=0)
        # ===== 反归一化 =====
        preds = preds * label_std + label_mean
        labels = labels * label_std + label_mean
        xyz_preds, time_preds = preds[:, :3], preds[:, -1]
        xyz_labels, time_labels = labels[:, :3], labels[:, -1]

        avg_xyz_loss = xyz_loss / len(self.test_loader)
        avg_time_loss = time_loss / len(self.test_loader)
        avg_dir_loss = dir_loss / len(self.test_loader)
        avg_xyz_dist = np.mean(np.linalg.norm(xyz_preds - xyz_labels, axis=1))
        avg_xy_dist = np.mean(np.linalg.norm(xyz_preds[:, :2] - xyz_labels[:, :2], axis=1))
        avg_time_err = np.mean(np.abs(time_preds - time_labels))

        # === 方向误差（角度，单位：度） ===
        cos_sim = np.sum(dir_preds * dir_labels, axis=1) / (
                np.linalg.norm(dir_preds, axis=1) * np.linalg.norm(dir_labels, axis=1)
        )
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 避免数值超出 [-1,1]
        avg_direction_err = np.mean(np.degrees(np.arccos(cos_sim)))

        return avg_xyz_loss, avg_time_loss, avg_dir_loss, avg_xyz_dist, avg_xy_dist, avg_time_err, avg_direction_err

    # def test_and_save(self, save_dir="./results"):
    #     self.model.eval()
    #     preds, labels, filenames = [], [], []
    #
    #     label_mean = self.test_loader.dataset.label_mean
    #     label_std = self.test_loader.dataset.label_std
    #     with torch.no_grad():
    #         for batch in self.test_loader:
    #             seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [b.to(self.device) for b in
    #                                                                                batch[:-1]]
    #             fn = batch[-1]
    #             best_param = torch.load(self.best_model_path)
    #             self.model.load_state_dict(best_param)
    #             pred_xyz, pred_time, pred_direction = self.model(seqs, masks)
    #
    #             preds.append(torch.cat(
    #                 [pred_xyz, pred_time.unsqueeze(1), pred_direction], dim=-1
    #             ).cpu().numpy())
    #             labels.append(torch.cat(
    #                 [labels_xyz, labels_time.unsqueeze(1), labels_direction], dim=-1
    #             ).cpu().numpy())
    #             filenames.append(fn)
    #
    #     preds = np.concatenate(preds, axis=0)
    #     labels = np.concatenate(labels, axis=0)
    #     filenames = [fn for fnames in filenames for fn in fnames]
    #     # ===== 反归一化 (只对 xyz,time 进行，direction 是单位向量不需要归一化) =====
    #     preds[:, :4] = preds[:, :4] * label_std + label_mean
    #     labels[:, :4] = labels[:, :4] * label_std + label_mean
    #
    #     df = pd.DataFrame({
    #         "pred_x": preds[:, 0],
    #         "pred_y": preds[:, 1],
    #         "pred_z": preds[:, 2],
    #         "pred_time": preds[:, 3],
    #         "pred_dir_x": preds[:, 4],
    #         "pred_dir_y": preds[:, 5],
    #         "label_x": labels[:, 0],
    #         "label_y": labels[:, 1],
    #         "label_z": labels[:, 2],
    #         "label_time": labels[:, 3],
    #         "label_dir_x": labels[:, 4],
    #         "label_dir_y": labels[:, 5],
    #         "file_name": filenames,
    #     })
    #
    #     # 拼接文件名
    #     os.makedirs(save_dir, exist_ok=True)
    #     filename = f"{self.logger.time}.csv"
    #     csv_file = os.path.join(save_dir, filename)
    #     df.to_csv(csv_file, index=False)
    #     self.logger.info(f"📄 Saved test results to {csv_file}")
    #     return df

    def test_and_save(self, save_dir="./results"):
        # 加载最佳模型参数
        best_param = torch.load(self.best_model_path)
        self.model.load_state_dict(best_param)
        self.model.to(self.device)

        # === 修改点 1：切换到 eval 模式 (关闭 Dropout，进行确定性推理) ===
        self.model.eval()

        # 用于存储预测结果
        preds_xyz_list = []
        preds_log_var_list = []  # 存储预测的 log_var
        preds_time_list = []
        preds_dir_list = []

        labels_list, filenames_list = [], []

        # 获取归一化参数 (保持原有逻辑)
        label_mean = self.test_loader.dataset.label_mean
        label_std = self.test_loader.dataset.label_std
        std_xyz = label_std[0][:3]
        std_xyz_sq = std_xyz ** 2  # 用于方差反归一化

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing (Single Pass)"):
                seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [
                    b.to(self.device) for b in batch[:-1]
                ]

                # 运行模型获取预测
                output = self.model(seqs, masks)
                pred_xyz_mu, pred_xyz_log_var, pred_time, pred_direction = output

                # 收集结果
                preds_xyz_list.append(pred_xyz_mu.cpu().numpy())
                preds_log_var_list.append(pred_xyz_log_var.cpu().numpy())
                preds_time_list.append(pred_time.cpu().numpy())
                preds_dir_list.append(pred_direction.cpu().numpy())

                # 收集标签和文件名
                fn = batch[-1]
                labels = torch.cat(
                    [labels_xyz, labels_time.unsqueeze(1), labels_direction], dim=-1
                ).cpu().numpy()
                labels_list.append(labels)
                filenames_list.append(fn)

        # 堆叠结果
        preds_xyz = np.concatenate(preds_xyz_list, axis=0)
        preds_log_var = np.concatenate(preds_log_var_list, axis=0)
        preds_time = np.concatenate(preds_time_list, axis=0)
        preds_dir = np.concatenate(preds_dir_list, axis=0)

        labels = np.concatenate(labels_list, axis=0)
        filenames = [fn for fnames in filenames_list for fn in fnames]

        # 拼接 XYZT 均值
        preds_xyzt = np.concatenate(
            [preds_xyz, preds_time[:, np.newaxis]], axis=1
        )

        # ==========================================================
        # 1. 不确定性计算 (仅计算 Aleatoric)
        # Var = exp(log_var)
        # ==========================================================
        var_xyz_norm = np.exp(preds_log_var)

        # ==========================================================
        # 2. 反归一化 (保持原有逻辑不变)
        # ==========================================================
        # 均值反归一化 (前 4 列)
        preds_xyzt_denorm = preds_xyzt * label_std + label_mean
        labels_denorm = labels[:, :4] * label_std + label_mean

        # 方差反归一化: Var_denorm = Var_norm * std_label^2
        var_xyz_denorm = var_xyz_norm * std_xyz_sq

        # 转化为标准差 (Std = sqrt(Var))
        std_xyz_denorm = np.sqrt(var_xyz_denorm)

        # ==========================================================
        # 3. 保存到 DataFrame
        # ==========================================================

        df = pd.DataFrame({
            "pred_x": preds_xyzt_denorm[:, 0],
            "pred_y": preds_xyzt_denorm[:, 1],
            "pred_z": preds_xyzt_denorm[:, 2],
            "pred_time": preds_xyzt_denorm[:, 3],
            "pred_dir_x": preds_dir[:, 0],
            "pred_dir_y": preds_dir[:, 1],

            # 🎯 Uncertainty (这里只剩下模型直接预测的数据不确定性)
            "std_x": std_xyz_denorm[:, 0],
            "std_y": std_xyz_denorm[:, 1],
            "std_z": std_xyz_denorm[:, 2],

            "label_x": labels_denorm[:, 0],
            "label_y": labels_denorm[:, 1],
            "label_z": labels_denorm[:, 2],
            "label_time": labels_denorm[:, 3],
            "label_dir_x": labels[:, 4],
            "label_dir_y": labels[:, 5],
            "file_name": filenames,
        })

        # 拼接文件名
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{self.logger.time}.csv"
        csv_file = os.path.join(save_dir, filename)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"📄 Saved test results to {csv_file}")

        return df
