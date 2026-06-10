## dataset.py

import os
import random
import numpy as np
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


DEFAULT_DATASET_FPS = {
    "20251217_scene1": 245.0,
    "20260202_scene2": 160.0,
    "20260418_scene3": 300.0,
}


def _dataset_name_from_folder(folder: str) -> str:
    return os.path.basename(os.path.normpath(folder))


def parse_dataset_fps_values(values=None) -> Dict[str, float]:
    fps_map = dict(DEFAULT_DATASET_FPS)
    if not values:
        return fps_map

    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid dataset fps value: {item}. Expected name=fps")
        name, fps = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid dataset fps value: {item}. Dataset name is empty")
        fps_value = float(fps)
        if fps_value <= 0:
            raise ValueError(f"Invalid fps for {name}: {fps_value}. FPS must be positive")
        fps_map[name] = fps_value
    return fps_map


def resolve_dataset_fps(dataset_name: str, dataset_fps=None, required=False):
    fps_map = DEFAULT_DATASET_FPS if dataset_fps is None else dataset_fps
    fps = fps_map.get(dataset_name)
    if fps is None and required:
        raise ValueError(
            f"FPS is not configured for dataset '{dataset_name}'. "
            "Pass --dataset_fps dataset_name=fps."
        )
    return fps


def parse_sample_file(file_path: str, point_num=22, dataset_name=None, fps=None) -> Dict:
    """
    解析单个样本文件，保留全部帧信息（不裁切）
    返回结构：{
      'frames': np.array (N, point_num * 3),
      'frame_ids': np.array (N,),
      'drop_frame': int,
      'label_xyz': np.array (3,)
    }
    """
    expected_dim = point_num * 3
    display_name = os.path.basename(file_path)
    if dataset_name:
        display_name = f"{dataset_name}/{display_name}"

    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 2:
        raise ValueError(f"文件行数太少: {file_path}")

    # 最后一行是落点
    last_line = lines[-1]
    drop_frame_str, drop_xyz_str = last_line.split(":")
    drop_frame = int(drop_frame_str)
    drop_xyz = np.array(list(map(float, drop_xyz_str.split(","))), dtype=np.float32)
    if drop_xyz[0] < -10 or drop_xyz[0] > 680 or drop_xyz[1] < -320 or drop_xyz[1] > 320 or abs(drop_xyz[2]) > 10:
        print(file_path, drop_xyz)
        return None
    # print(drop_frame, drop_xyz)
    drop_xyz[2] = 0.0

    frame_ids = []
    frames = []
    warned_extra_coords = False
    for ln in lines[:-1]:
        fid_str, coords_str = ln.split(":")
        fid = int(fid_str)
        coords = np.array(list(map(float, coords_str.split(","))), dtype=np.float32)
        if len(coords) < expected_dim:
            raise ValueError(
                f"{file_path} frame {fid}: expected {expected_dim} coordinates "
                f"({point_num} points), got {len(coords)}"
            )
        if len(coords) > expected_dim:
            if not warned_extra_coords:
                print(
                    f"Warning: {file_path} has extra coordinates; first seen at frame {fid}. "
                    f"Expected {expected_dim} coordinates ({point_num} points), got {len(coords)}. "
                    "Extra coordinates will be truncated."
                )
                warned_extra_coords = True
            coords = coords[:expected_dim]

        # ================== 新增：球拍几何合法性检查 (NumPy版) ==================
        # 提取末 4 个关键点 (4, 3)
        # 假设最后12个数是球拍的4个点
        pts = coords[17*3:21*3].reshape(4, 3)
        P1, P2, P3, P4 = pts[0], pts[1], pts[2], pts[3]

        # 规则 1：四面体体积
        # matrix shape: (3, 3)
        mat = np.stack([P2 - P1, P3 - P1, P4 - P1])
        v = np.abs(np.linalg.det(mat)) / 6.0
        bad_volume = (v > 2000)

        # 规则 2：长轴 / 短轴长度
        short_axis = np.linalg.norm(P2 - P1)
        long_axis = np.linalg.norm(P4 - P3)
        bad_axis = (short_axis > 100) or (long_axis > 100)

        # 若满足任意规则 → 将该帧数据全置为 0
        if bad_volume or bad_axis:
            coords[:] = np.nan
        # ======================================================================

        frame_ids.append(fid)
        frames.append(coords)

    return {
        'file_name': display_name,
        'source_dataset': dataset_name or _dataset_name_from_folder(os.path.dirname(file_path)),
        'fps': fps,
        'frames': np.stack(frames, axis=0),
        'frame_ids': np.array(frame_ids, dtype=np.int32),
        'drop_frame': drop_frame,
        'label_xyz': drop_xyz
    }


def load_all_samples(folder: str, point_num=22, suffix='.txt', dataset_name=None, dataset_fps=None) -> List[Dict]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"数据集目录不存在: {folder}")

    dataset_name = dataset_name or _dataset_name_from_folder(folder)
    fps = resolve_dataset_fps(dataset_name, dataset_fps, required=False)
    samples = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(suffix):
            continue
        path = os.path.join(folder, fn)
        s = parse_sample_file(path, point_num, dataset_name=dataset_name, fps=fps)
        if s:
            samples.append(s)
    return samples


def load_samples_from_folders(folders: List[str], point_num=22, suffix='.txt', dataset_fps=None) -> List[Dict]:
    samples = []
    for folder in folders:
        samples.extend(load_all_samples(folder, point_num=point_num, suffix=suffix, dataset_fps=dataset_fps))
    return samples


def split_samples_by_dataset(folders: List[str],
                             point_num=22,
                             train_ratio=0.8,
                             suffix='.txt',
                             dataset_fps=None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_samples = []
    test_samples = []
    stats = []

    for folder in folders:
        dataset_name = _dataset_name_from_folder(folder)
        samples = load_all_samples(
            folder,
            point_num=point_num,
            suffix=suffix,
            dataset_name=dataset_name,
            dataset_fps=dataset_fps,
        )
        if len(samples) < 2:
            raise ValueError(f"数据集 {folder} 至少需要 2 个样本才能按数据集分层划分训练/测试集")

        random.shuffle(samples)
        split_idx = int(train_ratio * len(samples))
        split_idx = max(1, min(len(samples) - 1, split_idx))

        train_part = samples[:split_idx]
        test_part = samples[split_idx:]
        train_samples.extend(train_part)
        test_samples.extend(test_part)
        stats.append({
            "dataset": dataset_name,
            "folder": folder,
            "total": len(samples),
            "train": len(train_part),
            "test": len(test_part),
        })

    random.shuffle(train_samples)
    random.shuffle(test_samples)
    return train_samples, test_samples, stats

def collate_fn_dynamic(batch, max_len = None):
    """
    batch: list of tuples (seq, length, label_xyz, label_time)
    seq: (seq_len_i, feature_dim)
    length: int
    label_xyz: (3,)
    label_time: float

    返回：
        seqs_padded: (B, max_len, feature_dim)
        lengths: (B,)
        xyzs: (B,3)
        times: (B,)
        mask: (B, max_len)
    """
    seqs, lengths, xyzs, times, dirs, filename = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    feature_dim = seqs[0].shape[1]
    if max_len == None:
        max_len = max(lengths)

    seqs_padded = []
    masks = []

    for seq, l, fn in zip(seqs, lengths, filename):
        pad_len = max_len - l
        if pad_len > 0:
            pad = torch.zeros(pad_len, feature_dim, dtype=seq.dtype)
            seq_padded = torch.cat([pad, seq], dim=0)
        else:
            seq_padded = seq
        mask = torch.ones(max_len, dtype=torch.bool)  # 先全设为True（默认填充）
        mask[:pad_len] = False  # 将前pad_l个位置设为False

        is_bad_frame = torch.isnan(seq_padded).any(dim=1)  # (max_len,)
        if is_bad_frame.any():
            mask[is_bad_frame] = False
            seq_padded = torch.nan_to_num(seq_padded, nan=0.0)

        seqs_padded.append(seq_padded)
        masks.append(mask)

    seqs_padded = torch.stack(seqs_padded, dim=0)  # (B, max_len, feature_dim)
    masks = torch.stack(masks, dim=0)              # (B, max_len)
    xyzs = torch.stack(xyzs, dim=0)                # (B,3)
    times = torch.stack(times, dim=0)              # (B,)
    dirs = torch.stack(dirs, dim=0)                # (B, 2)

    return seqs_padded, lengths, masks, xyzs, times, dirs, filename


def collate_fn_dynamic(batch, max_len=None):
    has_time_pos = len(batch[0]) == 7
    if has_time_pos:
        seqs, lengths, xyzs, times, dirs, time_positions, filename = zip(*batch)
    else:
        seqs, lengths, xyzs, times, dirs, filename = zip(*batch)
        time_positions = None

    lengths = torch.tensor(lengths, dtype=torch.long)
    feature_dim = seqs[0].shape[1]
    if max_len is None:
        max_len = max(lengths)

    seqs_padded = []
    masks = []
    time_positions_padded = []

    for idx, (seq, l, fn) in enumerate(zip(seqs, lengths, filename)):
        pad_len = max_len - l
        if pad_len > 0:
            pad = torch.zeros(pad_len, feature_dim, dtype=seq.dtype)
            seq_padded = torch.cat([pad, seq], dim=0)
        else:
            seq_padded = seq

        if has_time_pos:
            time_pos = time_positions[idx]
            if pad_len > 0:
                time_pad = torch.zeros(pad_len, dtype=time_pos.dtype)
                time_pos_padded = torch.cat([time_pad, time_pos], dim=0)
            else:
                time_pos_padded = time_pos

        mask = torch.ones(max_len, dtype=torch.bool)
        mask[:pad_len] = False

        is_bad_frame = torch.isnan(seq_padded).any(dim=1)
        if is_bad_frame.any():
            mask[is_bad_frame] = False
            seq_padded = torch.nan_to_num(seq_padded, nan=0.0)

        seqs_padded.append(seq_padded)
        masks.append(mask)
        if has_time_pos:
            time_positions_padded.append(time_pos_padded)

    seqs_padded = torch.stack(seqs_padded, dim=0)
    masks = torch.stack(masks, dim=0)
    xyzs = torch.stack(xyzs, dim=0)
    times = torch.stack(times, dim=0)
    dirs = torch.stack(dirs, dim=0)

    if has_time_pos:
        time_positions_padded = torch.stack(time_positions_padded, dim=0)
        return seqs_padded, lengths, masks, xyzs, times, dirs, time_positions_padded, filename

    return seqs_padded, lengths, masks, xyzs, times, dirs, filename

def resampling(samples: List[Dict],
                num_subsamples: int = 5,
                min_len: int = 10,
                max_len: int = 50) -> Tuple[List[Dict], List[Dict]]:
    """
    将原始样本扩展成多个子样本。

    Args:
        samples: 原始样本，每个 sample 包含 frames, frame_ids, drop_frame, label_xyz
        num_subsamples: 每条样本扩展的子样本数
        min_len, max_len: 每个子样本的帧长度范围

    Returns:
        train_samples, test_samples
    """
    expanded_samples = []
    for s in samples:
        total_len = len(s["frames"])
        for k in range(num_subsamples):
            seq_len = np.random.randint(min_len, max_len + 1)
            if seq_len > total_len:
                continue
            # 随机从结尾往前取一段
            end_idx = np.random.randint(seq_len - 1, total_len)
            start_idx = end_idx - seq_len + 1

            sub_sample = {
                "file_name": s.get("file_name", ""),
                "source_dataset": s.get("source_dataset", ""),
                "fps": s.get("fps"),
                "frames": s["frames"][start_idx:end_idx+1],
                "frame_ids": s["frame_ids"][start_idx:end_idx+1],
                "drop_frame": s["drop_frame"],
                "label_xyz": s["label_xyz"]
            }
            expanded_samples.append(sub_sample)

    # 打乱并划分
    random.shuffle(expanded_samples)
    return expanded_samples

def resampling_v2(samples: List[Dict],
                num_subsamples: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """
    将原始样本扩展成多个子样本。

    Args:
        samples: 原始样本，每个 sample 包含 frames, frame_ids, drop_frame, label_xyz
        num_subsamples: 每条样本扩展的子样本数
        min_len, max_len: 每个子样本的帧长度范围

    Returns:
        train_samples, test_samples
    """
    expanded_samples = []
    for s in samples:
        total_len = len(s["frames"])
        for k in range(num_subsamples):
            sub_sample = {
                'file_name': s["file_name"],
                "source_dataset": s.get("source_dataset", ""),
                "fps": s.get("fps"),
                "frames": s["frames"],
                "frame_ids": s["frame_ids"],
                "drop_frame": s["drop_frame"],
                "label_xyz": s["label_xyz"]
            }
            expanded_samples.append(sub_sample)

    # 打乱并划分
    random.shuffle(expanded_samples)
    return expanded_samples

def down_sampling(samples: List[Dict],
                min_len: int = 10,
                max_len: int = 50) -> Tuple[List[Dict], List[Dict]]:
    """
    将原始样本进行抽帧。

    Args:
        samples: 原始样本，每个 sample 包含 frames, frame_ids, drop_frame, label_xyz
        min_len, max_len: 每个子样本的帧长度范围

    Returns:
        train_samples, test_samples
    """
    expanded_samples = []
    total_len = len(samples[0]["frames"])
    for s in samples:
        seq_len = np.random.randint(min_len, max_len + 1)
        start_idx = total_len - seq_len + 1

        sub_sample = {
            "file_name": s.get("file_name", ""),
            "source_dataset": s.get("source_dataset", ""),
            "fps": s.get("fps"),
            "frames": s["frames"][start_idx:total_len],
            "frame_ids": s["frame_ids"][start_idx:total_len],
            "drop_frame": s["drop_frame"],
            "label_xyz": s["label_xyz"]
        }
        expanded_samples.append(sub_sample)

    # 打乱并划分
    random.shuffle(expanded_samples)
    return expanded_samples

class BadmintonDataset(Dataset):
    def __init__(self, samples: List[Dict], min_len: int = 10, max_len: int = 50, min_offset_len=0, max_offset_len=20, temp_test_offset=-1,
                 mode: str = "train",
                 num_subsamples: int = 5, # not use
                 feature_mean=None, feature_std=None,
                 label_mean=None, label_std=None,
                 aug_method=None,
                 use_time_pos_encoding=False,
                 time_label_unit="frames",
                 reference_fps=300.0,
                 hit_index=100):
        super().__init__()
        assert len(samples) > 0
        self.samples = samples
        self.min_len = min_len
        self.max_len = max_len
        self.min_offset_len = min_offset_len
        self.max_offset_len = max_offset_len
        self.temp_test_offset = temp_test_offset
        self.mode = mode
        self.aug_method = aug_method
        self.use_time_pos_encoding = use_time_pos_encoding
        self.time_label_unit = time_label_unit
        self.reference_fps = float(reference_fps)
        self.hit_index = hit_index

        if self.time_label_unit not in ("frames", "seconds"):
            raise ValueError(f"time_label_unit must be 'frames' or 'seconds', got {self.time_label_unit}")
        if self.use_time_pos_encoding or self.time_label_unit == "seconds":
            for sample in self.samples:
                if sample.get("fps") is None:
                    raise ValueError(
                        f"Missing fps for sample {sample.get('file_name', '')}. "
                        "Configure dataset_fps before using time positional encoding or seconds labels."
                    )


        if mode == "train":
            # 统计归一化参数（仅初始化时统计一次）
            all_features = np.concatenate([s["frames"] for s in samples], axis=0)
            all_labels = np.stack([np.concatenate([s["label_xyz"], [self._label_time_value(s, s["frame_ids"][-1])]])
                                    for s in samples], axis=0)
            self.feature_mean = np.nanmean(all_features, axis=0, keepdims=True)
            self.feature_std = np.nanstd(all_features, axis=0, keepdims=True) + 1e-6
            self.label_mean = np.nanmean(all_labels, axis=0, keepdims=True)
            self.label_std = np.nanstd(all_labels, axis=0, keepdims=True) + 1e-6
            self.noise_std_x = self.label_std[0][0]/5  # 176.7 * 0.1（可后续调为1/8或1/5倍）
            self.noise_std_y = self.label_std[0][1]/5  # 181.86 * 0.1

            # resampling_v2, 每个样本copy几份
            self.samples = resampling_v2(self.samples, num_subsamples)
        else:
            # 验证/测试：使用训练集的统计量
            self.feature_mean = feature_mean
            self.feature_std = feature_std
            self.label_mean = label_mean
            self.label_std = label_std

    def _sample_fps(self, sample):
        fps = sample.get("fps")
        if fps is None:
            raise ValueError(
                f"Missing fps for sample {sample.get('file_name', '')}. "
                "Configure dataset_fps before using time positional encoding or seconds labels."
            )
        return float(fps)

    def _label_time_value(self, sample, end_frame_id):
        frame_delta = float(sample["drop_frame"] - end_frame_id)
        if self.time_label_unit == "seconds":
            return frame_delta / self._sample_fps(sample)
        return frame_delta

    def _time_positions(self, sample, frame_ids):
        if self.hit_index < 1 or self.hit_index > len(sample["frame_ids"]):
            raise ValueError(
                f"hit_index={self.hit_index} is outside sample length {len(sample['frame_ids'])} "
                f"for {sample.get('file_name', '')}"
            )
        hit_frame_id = float(sample["frame_ids"][self.hit_index - 1])
        return (frame_ids.astype(np.float32) - hit_frame_id) / self._sample_fps(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        file_name = s["file_name"]
        total_frames = s["frames"]
        total_frame_ids = s["frame_ids"]
        drop_frame = s["drop_frame"]

        # 动态生成子样本（仅训练时随机截取，测试时取固定长度）
        if self.mode == "train":
            total_len = len(total_frames)
            seq_len = np.random.randint(self.min_len, self.max_len + 1)
            if seq_len > total_len:
                seq_len = total_len  # 防止长度超过原始序列
            offset = random.randint(self.min_offset_len, self.max_offset_len)
            end_idx = total_len - 1 - offset
            start_idx = end_idx - seq_len + 1
        else:
            # 测试时：从结尾取固定长度（或保持原始逻辑）
            total_len = len(total_frames)
            seq_len = min(total_len, self.max_len)
            if self.temp_test_offset < 0:
                offset = random.randint(self.min_offset_len, self.max_offset_len)
            else:
                offset = self.temp_test_offset
            end_idx = total_len - 1 - offset
            start_idx = end_idx - seq_len + 1
            # end_idx = total_len - 1
            # start_idx = end_idx - seq_len + 1

        # 截取子样本
        seq = torch.from_numpy(total_frames[start_idx:end_idx+1]).float()
        frame_ids = total_frame_ids[start_idx:end_idx+1]
        length = seq.shape[0]
        label_xyz_raw = torch.tensor(s["label_xyz"], dtype=torch.float32)  # 原始XY轴标签（物理空间）
        label_time_raw = torch.tensor(self._label_time_value(s, frame_ids[-1]), dtype=torch.float32)  # 时间标签（暂不加噪声）
        if self.use_time_pos_encoding:
            time_pos = torch.from_numpy(self._time_positions(s, frame_ids)).float()

        # 方向标签
        direction_vec = label_xyz_raw[:2] - torch.tensor([300.0, 0.0])
        norm = torch.norm(direction_vec)
        if norm > 1e-6:
            direction_unit = direction_vec / norm
        else:
            direction_unit = torch.zeros_like(direction_vec)

        # 数据增强
        seq = seq.view(length, -1, 3)
        if self.mode == 'train' and random.random() > 0.5:
            if self.aug_method == '平移':
                # 在归一化后的数据上进行平移
                dx = random.uniform(-20, 20)
                dy = random.uniform(-20, 20)
                translation = torch.tensor([dx, dy, 0.0], dtype=torch.float32)
                # 对整个序列和落点进行平移
                seq = seq + translation
                if label_xyz_raw is not None:
                    label_xyz_raw = label_xyz_raw + translation
            elif self.aug_method == '旋转':
                # 旋转逻辑
                angle = random.uniform(-5, 5) * np.pi / 180.0
                cos_a, sin_a = np.cos(angle), np.sin(angle)

                # 使用一个嵌套函数来处理旋转
                def rotate_points(points):
                    x_old, y_old = points[..., 0], points[..., 1]
                    x_new = cos_a * x_old - sin_a * y_old
                    y_new = sin_a * x_old + cos_a * y_old
                    rotated = torch.stack([x_new, y_new, points[..., 2]], dim=-1)
                    return rotated

                seq = rotate_points(seq)
                if label_xyz_raw is not None:
                    label_xyz_raw = rotate_points(label_xyz_raw)
            elif self.aug_method == '缩放':
                scale = random.uniform(0.98, 1.02)
                seq = seq * scale
                # 落点也需要同步缩放
                if label_xyz_raw is not None:
                    label_xyz_raw = label_xyz_raw * scale
            elif self.aug_method == '噪声':
                noise = torch.randn_like(seq) * 10
                seq = seq + noise
        seq = seq.view(length, -1)

        # 归一化
        seq = (seq - torch.from_numpy(self.feature_mean).float()) / torch.from_numpy(self.feature_std).float()
        # 训练集专属：XY轴标签加噪声（核心步骤）
        # if self.mode == "train":
        #     # 2.1 确保噪声在原始物理空间添加（先反归一化？不——这里label_xyz_raw是原始空间，无需反归一化）
        #     # 生成高斯噪声（与标签同设备、同 dtype）
        #     noise_x = torch.normal(mean=0.0, std=self.noise_std_x, size=[], dtype=label_xyz_raw.dtype, device=label_xyz_raw.device)
        #     noise_y = torch.normal(mean=0.0, std=self.noise_std_y, size=[], dtype=label_xyz_raw.dtype, device=label_xyz_raw.device)
        #     # 给XY轴标签加噪声（Z轴若无需增强可跳过）
        #     label_xyz_raw[0] += noise_x  # X轴加噪声
        #     label_xyz_raw[1] += noise_y  # Y轴加噪声

        label_all = torch.cat([label_xyz_raw, label_time_raw.unsqueeze(0)], dim=0)
        label_all = (label_all - torch.from_numpy(self.label_mean).squeeze(0).float()) / \
                    torch.from_numpy(self.label_std).squeeze(0).float()

        label_xyz_norm = label_all[:3]
        label_time_norm = label_all[3]

        if self.use_time_pos_encoding:
            return seq, torch.tensor(length, dtype=torch.long), label_xyz_norm, label_time_norm, direction_unit, time_pos, file_name

        return seq, torch.tensor(length, dtype=torch.long), label_xyz_norm, label_time_norm, direction_unit, file_name

    def get_norm_stats(self):
        return self.feature_mean, self.feature_std, self.label_mean, self.label_std


if __name__ == "__main__":
    # path = '/home/zhaoxuhao/badminton_xh/20250809_Seq_data/20250809_150058---008377.txt'
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_folder', type=str, default='/home/zhaoxuhao/badminton_xh/20250809_Seq_data')
    parser.add_argument('--data_folder', type=str, default='/home/zhaoxuhao/badminton_xh/20250809_Seq_data_v2/20250809_Seq_data')
    args = parser.parse_args()
    # 加载所有样本
    samples = load_all_samples(args.data_folder)
    print(f"一共加载到 {len(samples)} 个样本") # print(samples)
    # 构建 Dataset
    dataset = BadmintonDataset(samples, mode='train', min_len=10, max_len=50)
    feature_mean, feature_std, label_mean, label_std = dataset.get_norm_stats()
    test_dataset = BadmintonDataset(samples, mode='test', min_len=10, max_len=50,
                               feature_mean=feature_mean, feature_std=feature_std,
                               label_mean=label_mean, label_std=label_std)

    # 4. 打印
    print("========== 数据集统计信息 ==========")
    print(f"特征 mean: {feature_mean.shape}, 示例前5个维度: {feature_mean[0, :5]}")
    print(f"特征 std : {feature_std.shape}, 示例前5个维度: {feature_std[0, :5]}")

    print(f"标签 mean: {label_mean.shape}, 值: {label_mean[0]}")
    print(f"标签 std : {label_std.shape}, 值: {label_std[0]}")

    # 随机取几个看看
    for i in range(5):
        xyz_seq, seq_len, label_xyz, label_t = dataset[i]
        print(f"样本 {i}:")
        print(f" 输入序列 shape: {xyz_seq.shape}") # (L, 63), L ∈ [40, 50]
        print(f" seq len: {seq_len}")
        print(f" 落点 label xyz: {label_xyz.numpy()}")
        print(f" 飞行时间 label: {label_t.item()}")

    print("\ntest data")
    # 随机取几个看看
    for i in range(5):
        xyz_seq, seq_len, label_xyz, label_t = test_dataset[i]
        print(f"样本 {i}:")
        print(f" 输入序列 shape: {xyz_seq.shape}") # (L, 63), L ∈ [40, 50]
        print(f" seq len: {seq_len}")
        print(f" 落点 label xyz: {label_xyz.numpy()}")
        print(f" 飞行时间 label: {label_t.item()}")
