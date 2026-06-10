## model.py

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EndToEndModel(nn.Module):
    def __init__(self, original_model, feature_mean, feature_std, label_mean, label_std):
        super(EndToEndModel, self).__init__()
        self.backbone = original_model

        # 【关键点】使用 register_buffer 将均值和方差注册为模型的一部分
        # 这样它们会作为常量 (Constant) 存入 ONNX，而不是作为输入节点
        self.register_buffer('feature_mean', feature_mean)
        self.register_buffer('feature_std', feature_std)
        self.register_buffer('label_mean', label_mean)
        self.register_buffer('label_std', label_std)

    def forward(self, sequences, masks, time_pos=None):
        # 1. 模型原始推理 (输出是归一化的)
        # print('shape', sequences.shape, self.feature_mean.shape, self.feature_std.shape, self.label_mean.shape, self.label_std.shape)
        normalized_sequence = (sequences - self.feature_mean) / self.feature_std
        if getattr(self.backbone, "use_time_pos_encoding", False):
            normalized_output = self.backbone(normalized_sequence, masks, time_pos)
        else:
            normalized_output = self.backbone(normalized_sequence, masks)
        # print('shape2', [i.shape for i in normalized_output])

        # 2. 模型内进行反归一化
        # 公式: 真实值 = 归一化值 * 方差 + 均值
        real_output_xyz = normalized_output[0] * self.label_std[:,:3] + self.label_mean[:,:3]
        real_output_var = normalized_output[1] * self.label_std[:,:3]
        real_output_time = normalized_output[2] * self.label_std[:,3] + self.label_mean[:,3]

        return real_output_xyz, real_output_var, real_output_time, normalized_output[3]



class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, bidirectional=True, drop=0.5):
        super().__init__()
        self.name = 'LSTMRegressor'
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0  # LSTM层间dropout，仅当层数>1时有效
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc_xyz = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop),  # 全连接层后添加dropout
            nn.Linear(64, 3)
        )
        self.fc_time = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop),  # 全连接层后添加dropout
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths, mask=None):
        # x: (batch, max_len, 63)
        # lengths: (batch,)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)

        # h: (num_layers * num_directions, batch, hidden_dim)
        feat = torch.cat([h[-2], h[-1]], dim=1)  # (batch, 128*2)

        pred_xyz = self.fc_xyz(feat)         # (batch, 3)
        pred_time = self.fc_time(feat).squeeze(1)  # (batch,)

        return pred_xyz, pred_time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Conv1DResidual(nn.Module):
    """1D卷积残差块，用于提取姿态内部特征"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接适配层
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        # x: (batch, seq_len, num_points, 3) -> 经过permute后为(batch, seq_len, 3, num_points)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class AttentionLayer(nn.Module):
    """注意力层，用于突出重要帧"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, x, lengths):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        
        # 计算注意力权重
        attn_weights = self.attention(x).squeeze(2)  # (batch, seq_len)
        
        # 对填充部分进行掩码
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        attn_weights.masked_fill_(mask, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # (batch, 1, seq_len)
        
        # 应用注意力
        weighted_sum = torch.bmm(attn_weights, x).squeeze(1)  # (batch, hidden_dim)
        return weighted_sum, attn_weights

class ImprovedLSTMRegressor(nn.Module):
    def __init__(self, num_points=21, input_dim=3, conv_dims=[32, 64], 
                 hidden_dim=64, num_layers=2, bidirectional=True, drop=0.5):
        super().__init__()
        self.name = 'ImprovedLSTMRegressor'
        self.num_points = num_points
        self.bidirectional = bidirectional
        
        # 1D卷积特征提取部分
        conv_layers = []
        in_channels = input_dim  # 每个点有3个坐标(x,y,z)
        for out_channels in conv_dims:
            conv_layers.append(Conv1DResidual(in_channels, out_channels))
            in_channels = out_channels
        
        self.conv_extractor = nn.Sequential(*conv_layers)
        self.conv_out_dim = conv_dims[-1]
        self.point_feature_dim = self.conv_out_dim * num_points  # 每个姿态的总特征维度
        
        # LSTM时序特征提取
        self.lstm = nn.LSTM(
            input_size=self.point_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0
        )
        
        # 注意力层
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = AttentionLayer(lstm_out_dim)
        
        # 输出预测头
        self.fc_xyz = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.fc_time = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths, mask=None):
        # x: (batch, max_len, num_points*3) -> 需要先转换为(batch, max_len, num_points, 3)
        batch_size, max_len, _ = x.size()
        
        # 重塑输入以分离每个点的坐标
        x = x.view(batch_size, max_len, self.num_points, 3)  # (batch, max_len, num_points, 3)
        
        # 调整维度用于1D卷积: (batch, max_len, 3, num_points)
        x = x.permute(0, 1, 3, 2)
        
        # 提取每个姿态的空间特征
        conv_feat = []
        for t in range(max_len):
            # 对每个时间步的姿态应用卷积提取器
            frame_feat = self.conv_extractor(x[:, t, :, :])  # (batch, conv_out_dim, num_points)
            frame_feat = frame_feat.flatten(1)  # (batch, conv_out_dim * num_points)
            conv_feat.append(frame_feat)
        
        # 堆叠成序列: (batch, max_len, point_feature_dim)
        conv_seq = torch.stack(conv_feat, dim=1)
        
        # LSTM处理时序特征
        packed = pack_padded_sequence(conv_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # (batch, max_len, lstm_out_dim)
        
        # 应用注意力机制
        attn_feat, attn_weights = self.attention(lstm_out, lengths)  # (batch, lstm_out_dim)
        
        # 预测输出
        pred_xyz = self.fc_xyz(attn_feat)         # (batch, 3)
        pred_time = self.fc_time(attn_feat).squeeze(1)  # (batch,)
        
        return pred_xyz, pred_time

class SimplifiedLSTMRegressor(nn.Module):
    def __init__(self, num_points=21, input_dim=3, conv_dims=[32, 64],  # 减少卷积维度
                 hidden_dim=256, num_layers=5, bidirectional=False,  # 简化LSTM
                 drop=0.3):  # 降低dropout率
        super().__init__()
        self.name = 'SimplifiedLSTMRegressor'
        self.num_points = num_points
        self.bidirectional = bidirectional
        
        # 简化卷积特征提取：减少层数和维度，移除残差连接
        conv_layers = []
        in_channels = input_dim
        for out_channels in conv_dims:
            # 使用更大的步长减少计算量，移除 BatchNorm
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            # conv_layers.append(nn.BatchNorm1d(out_channels))  # 新增 BatchNorm
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.conv_extractor = nn.Sequential(*conv_layers)
        self.conv_out_dim = conv_dims[-1]
        self.point_feature_dim = self.conv_out_dim * num_points
        
        # 简化LSTM：减少层数，改为单向
        self.lstm = nn.LSTM(
            input_size=self.point_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0
        )
        
        # 移除注意力机制，直接使用LSTM最后一个时间步的输出
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        
        # 简化全连接层：减少层数和维度
        self.fc_xyz = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 3)
        )
        
        self.fc_time = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, 1)
        )

    # def forward(self, x, lengths, mask=None):
    #     # x: (batch, max_len, num_points*3)
    #     batch_size, max_len, _ = x.size()
        
    #     # 重塑输入
    #     x = x.view(batch_size, max_len, self.num_points, 3)  # (batch, max_len, num_points, 3)
    #     x = x.permute(0, 1, 3, 2)  # (batch, max_len, 3, num_points)
        
    #     # 卷积特征提取：批量处理所有时间步（替代循环）
    #     # 调整维度以便并行处理所有帧：(batch*max_len, 3, num_points)
    #     x_reshaped = x.reshape(-1, 3, self.num_points)
    #     conv_feat = self.conv_extractor(x_reshaped)  # (batch*max_len, conv_out_dim, num_points)
    #     conv_feat = conv_feat.flatten(1)  # (batch*max_len, conv_out_dim*num_points)
        
    #     # 恢复时序结构：(batch, max_len, point_feature_dim)
    #     conv_seq = conv_feat.view(batch_size, max_len, -1)
        
    #     # LSTM处理
    #     packed = pack_padded_sequence(conv_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
    #     _, (h_n, _) = self.lstm(packed)  # 只保留最后一个时间步的隐状态
        
    #     # 取最后一层的输出
    #     if self.bidirectional:
    #         feat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # 双向时拼接正反方向
    #     else:
    #         feat = h_n[-1]  # 单向时直接取最后一层
        
    #     # 预测输出
    #     pred_xyz = self.fc_xyz(feat)
    #     pred_time = self.fc_time(feat).squeeze(1)
        
    #     return pred_xyz, pred_time
    
    def forward(self, x, mask):
        # x: (batch, max_len, num_points*3)
        batch_size, max_len, _ = x.size()
        lengths = torch.sum(mask, dim=1)
        
        # 1. 卷积特征提取（保持不变）
        x = x.view(batch_size, max_len, self.num_points, 3)  # (batch, max_len, num_points, 3)
        x = x.permute(0, 1, 3, 2)  # (batch, max_len, 3, num_points)
        x_reshaped = x.reshape(-1, 3, self.num_points)  # (batch*max_len, 3, num_points)
        conv_feat = self.conv_extractor(x_reshaped)  # (batch*max_len, conv_out_dim, num_points)
        conv_feat = conv_feat.flatten(1)  # (batch*max_len, conv_out_dim*num_points)
        conv_seq = conv_feat.view(batch_size, max_len, -1)  # (batch, max_len, point_feature_dim)
        
        # 2. 应用mask：将填充位置的特征置为0（关键步骤1）
        # mask形状: (batch, max_len) -> 扩展为 (batch, max_len, 1) 与特征维度匹配
        conv_seq = conv_seq * mask.unsqueeze(-1).float()
        
        # 3. LSTM直接处理整个填充后的序列（不使用pack_padded_sequence）
        lstm_out, (h_n, c_n) = self.lstm(conv_seq)  # lstm_out: (batch, max_len, hidden_dim*direction)
        
        # 4. 提取每个序列的真实最后一个有效时间步的输出（关键步骤2）
        # 生成索引: 每个样本的最后有效位置 = lengths - 1
        last_indices = (lengths - 1).unsqueeze(1).unsqueeze(1)  # (batch, 1, 1)
        # 扩展索引维度以匹配lstm_out: (batch, 1, hidden_dim*direction)
        last_indices = last_indices.expand(-1, -1, lstm_out.size(-1))
        # 按索引取最后一个有效时间步的输出
        final_feat = torch.gather(lstm_out, dim=1, index=last_indices).squeeze(1)  # (batch, hidden_dim*direction)
        
        # 5. 预测输出
        pred_xyz = self.fc_xyz(final_feat)
        pred_time = self.fc_time(final_feat).squeeze(1)
        
        return pred_xyz, pred_time


class RNNRegressor(nn.Module):
    def __init__(self, num_points=21, input_dim=3, conv_dims=[32, 64],
                 hidden_dim=128, num_layers=5, bidirectional=False,
                 drop=0.3):
        super().__init__()
        self.name = 'RNNRegressor'
        self.num_points = num_points
        self.bidirectional = bidirectional

        # 简化卷积特征提取（保持不变）
        conv_layers = []
        in_channels = input_dim
        for out_channels in conv_dims:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_extractor = nn.Sequential(*conv_layers)
        self.conv_out_dim = conv_dims[-1]
        self.point_feature_dim = self.conv_out_dim * num_points

        # 核心更改: 从 nn.LSTM 替换为 nn.RNN
        self.rnn = nn.RNN(
            input_size=self.point_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0
        )

        # 全连接层（保持不变）
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc_xyz = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 3)
        )
        self.fc_time = nn.Sequential(
            nn.Linear(rnn_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        batch_size, max_len, _ = x.size()
        lengths = torch.sum(mask, dim=1)

        # 1. 卷积特征提取（保持不变）
        x = x.view(batch_size, max_len, self.num_points, 3)
        x = x.permute(0, 1, 3, 2)
        x_reshaped = x.reshape(-1, 3, self.num_points)
        conv_feat = self.conv_extractor(x_reshaped)
        conv_feat = conv_feat.flatten(1)
        conv_seq = conv_feat.view(batch_size, max_len, -1)

        # 2. 应用mask：将填充位置的特征置为0（保持不变）
        conv_seq = conv_seq * mask.unsqueeze(-1).float()

        # 3. RNN直接处理整个填充后的序列
        # 注意: RNN的输出格式与LSTM略有不同，它不返回c_n
        # RNN的输出只有: output, h_n
        rnn_out, h_n = self.rnn(conv_seq) # 核心更改：移除c_n

        # 4. 提取每个序列的真实最后一个有效时间步的输出（保持不变）
        last_indices = (lengths - 1).unsqueeze(1).unsqueeze(1)
        last_indices = last_indices.expand(-1, -1, rnn_out.size(-1))
        final_feat = torch.gather(rnn_out, dim=1, index=last_indices).squeeze(1)

        # 5. 预测输出（保持不变）
        pred_xyz = self.fc_xyz(final_feat)
        pred_time = self.fc_time(final_feat).squeeze(1)

        return pred_xyz, pred_time


class TransformerModel(nn.Module):
    def __init__(self, seq_len=50, num_points=21, d_model=256, nhead=8, num_layers=6, point_dim=3):
        super().__init__()
        self.name = 'TransformerModel'
        self.num_points = num_points
        self.point_dim = point_dim
        self.special_indices = [i for i in range(17*point_dim, 21*point_dim)]  # 球拍四点坐标
        self.base_input_dim = num_points * point_dim  # 原始关键点维度

        # 新增特征维度：1 (速度) + 3 (法向量) = 4
        self.extra_dim = 8  # 从4改成8
        self.input_dim = self.base_input_dim + self.extra_dim
        self.special_input_dim = len(self.special_indices) + self.extra_dim

        self.d_model = d_model

        # 主分支输入映射
        self.input_fc_main = nn.Linear(self.input_dim, d_model)
        # 特殊点分支输入映射
        self.input_fc_special = nn.Linear(self.special_input_dim, d_model)

        # Transformer 编码器
        encoder_layer_main = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_main = nn.TransformerEncoder(encoder_layer_main, num_layers=num_layers)

        encoder_layer_special = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_special = nn.TransformerEncoder(encoder_layer_special, num_layers=num_layers)

        # 融合后的 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, point_dim)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.direction_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Tanh()
        )

    def _compute_extra_features(self, x):
        """
        输入: x (B, T, 63)，即21点*3维
        输出: (B, T-2, 8)，包含：
            - 球拍速度(1维)
            - 加速度(1维)
            - 位移方向(3维)
            - 法向量(3维)
        """
        B, T, _ = x.shape
        x_points = x.view(B, T, self.num_points, self.point_dim)  # (B, T, 21, 3)

        # 球拍四点：17,18,19,20
        racket_pts = x_points[:, :, 17:21, :]  # (B, T, 4, 3)
        racket_center = racket_pts.mean(dim=2)  # (B, T, 3)

        # === 速度 ===
        velocity_vec = racket_center[:, 1:, :] - racket_center[:, :-1, :]  # (B, T-1, 3)
        velocity = torch.norm(velocity_vec, dim=-1, keepdim=True)  # (B, T-1, 1)

        # === 加速度 ===
        acc_vec = velocity_vec[:, 1:, :] - velocity_vec[:, :-1, :]  # (B, T-2, 3)
        acc_norm = torch.norm(acc_vec, dim=-1, keepdim=True)  # (B, T-2, 1)

        # === 位移方向（单位化速度向量） ===
        vel_dir = velocity_vec[:, 1:, :]  # 对齐到 T-2
        vel_dir_norm = torch.norm(vel_dir, dim=-1, keepdim=True) + 1e-8
        vel_dir_unit = vel_dir / vel_dir_norm  # (B, T-2, 3)

        # === 球拍法向量 ===
        short_axis = x_points[:, :, 19, :] - x_points[:, :, 17, :]  # (B, T, 3)
        long_axis = x_points[:, :, 20, :] - x_points[:, :, 18, :]  # (B, T, 3)
        normal = torch.cross(short_axis, long_axis, dim=-1)
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
        normal = normal[:, 2:, :]  # 对齐到 T-2

        # === 速度大小 ===
        velocity_mag = velocity[:, 1:, :]  # (B, T-2, 1)

        # 拼接: (B, T-2, 8)
        extra_feat = torch.cat([velocity_mag, acc_norm, vel_dir_unit, normal], dim=-1)
        return extra_feat

    def forward(self, x, mask):
        # x: (B, T, 63)，mask: (B, T)
        B, T, _ = x.shape

        # === 特征工程 ===
        extra_feat = self._compute_extra_features(x)  # (B, T-2, 8)
        x = x[:, 2:, :]  # 去掉前两帧 (B, T-2, 63)
        mask = mask[:, 2:]  # (B, T-2)

        # 拼接新特征
        x_aug = torch.cat([x, extra_feat], dim=-1)  # (B, T-2, 71)

        # --- 主分支 ---
        x_main_proj = self.input_fc_main(x_aug)
        out_main = self.transformer_main(x_main_proj, src_key_padding_mask=~mask)
        final_feat_main = out_main[:, -1, :]

        # --- 特殊点分支 ---
        special_x = torch.cat([x[:, :, self.special_indices], extra_feat], dim=-1)
        special_proj = self.input_fc_special(special_x)
        out_special = self.transformer_special(special_proj, src_key_padding_mask=~mask)
        final_feat_special = out_special[:, -1, :]

        # --- 融合 ---
        fused = torch.cat([final_feat_main, final_feat_special], dim=1)
        output = self.fusion_mlp(fused)
        output_time = self.time_mlp(fused).squeeze(1)
        output_direction = self.direction_mlp(fused)

        return output, output_time, output_direction


class ImprovedTransformerModel(nn.Module):
    def __init__(self, seq_len=100, num_points=21, d_model=1024, nhead=4, num_layers=4, point_dim=3,
                 use_time_pos_encoding=False, reference_fps=300.0):
        super().__init__()
        self.name = 'ImprovedTransformerModel'
        self.num_points = num_points
        self.special_indices = [i for i in range(17 * point_dim, num_points * point_dim)]
        self.input_dim = num_points * point_dim
        self.special_input_dim = len(self.special_indices)
        self.d_model = d_model
        self.use_time_pos_encoding = use_time_pos_encoding
        self.reference_fps = float(reference_fps)

        # self.num_special_points = self.special_input_dim // point_dim
        # self.pe_dim_per_point = 2
        # self.point_pe_embedding = nn.Parameter(
        #     torch.randn(1, 1, self.num_special_points * self.pe_dim_per_point)
        # )

        # fixed_point_pe = self._create_fixed_point_pe()  # 调用新的生成函数
        # # 注册为 buffer，它不可训练，但会随模型 state_dict 一起保存和加载
        # self.register_buffer('fixed_point_pe', fixed_point_pe)

        if self.use_time_pos_encoding:
            self.pos_encoder = TimePositionalEncoding(d_model, dropout=0.1, reference_fps=self.reference_fps)
        else:
            self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)

        # self.special_input_dim = self.special_input_dim + self.num_special_points * self.pe_dim_per_point  # 12 + 16 = 28

        # 特殊点的输入映射
        self.input_fc_special = nn.Sequential(
            nn.Linear(self.special_input_dim, d_model),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(d_model // 2, d_model),
        )

        # 特殊点Transformer编码器
        encoder_layer_special = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_special = nn.TransformerEncoder(encoder_layer_special, num_layers=num_layers)

        self.xyz_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, point_dim),
        )

        self.xyz_var_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, point_dim),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.direction_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )

    def _create_fixed_point_pe(self):
        # 这里的逻辑是为每个特殊点 (self.num_special_points) 生成一个固定的编码
        # 假设 special_indices 对应 4 个点 (17, 18, 19, 20)，即 num_special_points = 4
        # 每个点的 PE 维度是 2 (pe_dim_per_point = 2)

        # 1. 创建位置索引 (0, 1, 2, 3)
        position_indices = torch.arange(self.num_special_points, dtype=torch.float)

        # 2. 生成 sin/cos 编码 (类似 Transformer 的 PE)
        # 示例：使用维度 0 和 1
        div_term = torch.exp(
            torch.arange(0, self.pe_dim_per_point, 2).float() * (-np.log(10000.0) / self.pe_dim_per_point))

        # 初始化 [num_special_points, pe_dim_per_point]
        pe = torch.zeros(self.num_special_points, self.pe_dim_per_point)
        pe[:, 0] = torch.sin(position_indices * div_term)
        pe[:, 1] = torch.cos(position_indices * div_term)

        # 展平并调整形状: [1, 1, num_special_points * pe_dim_per_point]
        pe = pe.view(1, 1, -1)
        return pe

    def forward(self, x, mask, time_pos=None):
        B, T, _ = x.shape

        special_x = x[:, :, self.special_indices]  # (B, T, )

        # point_pe_broadcast = self.point_pe_embedding.repeat(B, T, 1)
        # point_pe_broadcast = self.fixed_point_pe.repeat(B, T, 1)  # B, T, D_pe
        # 
        # special_x = torch.cat([special_x, point_pe_broadcast], dim=-1)

        special_proj = self.input_fc_special(special_x)  # (B, T, d_model)

        # fixed frame-index PE or real-time PE
        if self.use_time_pos_encoding:
            if time_pos is None:
                raise ValueError("time_pos is required when use_time_pos_encoding=True")
            special_proj = self.pos_encoder(special_proj, time_pos)
        else:
            special_proj = self.pos_encoder(special_proj)

        # learnable position encoding
        # position_ids = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)  # (1, T)
        # pos_embeds = self.pos_embedding(position_ids)
        # special_proj = special_proj + pos_embeds
        # special_proj = self.pos_dropout(special_proj)

        out_special = self.transformer_special(special_proj, src_key_padding_mask=~mask)
        final = out_special[:, -1, :]  # (B, d_model)

        output = self.xyz_mlp(final)  # (B, 2)
        output_var = self.xyz_var_mlp(final)
        output_time = self.time_mlp(final).squeeze(1)  # (B,)
        output_direction = self.direction_mlp(final)

        return output, output_var, output_time, output_direction


class ImprovedTransformerTimePEModel(ImprovedTransformerModel):
    def __init__(self, seq_len=100, num_points=21, d_model=1024, nhead=4, num_layers=4, point_dim=3,
                 reference_fps=300.0):
        super().__init__(
            seq_len=seq_len,
            num_points=num_points,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            point_dim=point_dim,
            use_time_pos_encoding=True,
            reference_fps=reference_fps,
        )
        self.name = 'ImprovedTransformerTimePEModel'


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, reference_fps: float = 300.0):
        super().__init__()
        self.reference_fps = float(reference_fps)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, time_pos):
        if time_pos.dim() != 2:
            raise ValueError(f"time_pos must have shape (batch, seq_len), got {tuple(time_pos.shape)}")
        if time_pos.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"time_pos shape {tuple(time_pos.shape)} does not match sequence shape {tuple(x.shape[:2])}"
            )

        positions = time_pos.to(device=x.device, dtype=x.dtype).unsqueeze(-1) * self.reference_fps
        div_term = self.div_term.to(device=x.device, dtype=x.dtype)
        pe = torch.zeros_like(x)
        pe[..., 0::2] = torch.sin(positions * div_term)
        pe[..., 1::2] = torch.cos(positions * div_term)
        return self.dropout(x + pe)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer，它不是可训练参数，但会随模型保存和加载
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor, 形状 (batch_size, seq_len, d_model)
        """
        # pe 的形状是 (max_len, 1, d_model)
        # x 的形状是 (B, T, d_model)，T <= max_len
        x = x + self.pe[:x.size(1)].squeeze(1)
        return self.dropout(x)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from util.dataset import load_all_samples, BadmintonDataset, collate_fn_dynamic

    def test_model(data_folder='/home/zhaoxuhao/badminton_xh/20250809_Seq_data', batch_size=8):
        """
        测试模型 forward 是否正常
        """
        # 1. 加载数据
        samples = load_all_samples(data_folder)
        print(f"Loaded {len(samples)} samples.")

        dataset = BadmintonDataset(samples, min_len=40, max_len=50)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_dynamic)

        # 2. 构建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMRegressor(input_dim=63, hidden_dim=128, num_layers=2, bidirectional=True)
        model.to(device)
        model.eval()

        # 3. 测试一个 batch
        with torch.no_grad():
            for batch in dataloader:
                seqs, lengths, masks, xyzs, times = batch
                seqs = seqs.to(device)
                lengths = lengths.to(device)

                pred_xyz, pred_time = model(seqs, lengths)
                print("pred_xyz:", pred_xyz.shape, pred_xyz)   # (B, 3)
                print("pred_time:", pred_time.shape, pred_time) # (B,)
                break

    test_model()
