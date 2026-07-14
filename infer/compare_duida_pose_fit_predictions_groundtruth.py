#!/usr/bin/env python3
"""
对比两套落点算法在 duida 数据上的预测差异。

姿态侧:
- 复用 infer/infer_landpoint_pred.py 的取窗与 ONNX 推理思路
- 默认使用 before 模式，因为 20260418_duida_3d_pose 每行只有 63 维姿态，
  不包含 after 模式需要的 66 维（含球坐标）输入

球轨迹侧:
- 复用 landing_fitting/trajectory_mlp.py 中的推理逻辑
- 默认使用 MLP + 剩余参数拟合的 hybrid 方案
- 也支持切到纯 curve-fit 方案做对照

当前脚本只提供接口，不会自动执行。模型参数准备好后再手动运行。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LANDING_FITTING_DIR = ROOT / "landing_fitting"
if str(LANDING_FITTING_DIR) not in sys.path:
    sys.path.insert(0, str(LANDING_FITTING_DIR))

from trajectory_mlp import (  # noqa: E402
    CParameterPredictor,
    evaluate_position,
    fit_all_parameters,
    predict_hybrid_parameters,
    solve_landing_time,
)
from util.model import ImprovedTransformerModel, ImprovedTransformerTimePEModel  # noqa: E402


DEFAULT_POSE_DIR = ROOT / "datasets" / "20260418_duida_3d_pose"
DEFAULT_FIT_DIR = ROOT / "datasets" / "20260418_duida_fit_fall"
DEFAULT_OUTPUT_CSV = ROOT / "infer" / "duida_pose_fit_prediction_comparison.csv"
DEFAULT_OUTPUT_JSON = ROOT / "infer" / "duida_pose_fit_prediction_summary.json"


@dataclass
class PoseSequence:
    path: Path
    frame_ids: np.ndarray
    features: np.ndarray
    groundtruth: np.ndarray


@dataclass
class TrajectorySequence:
    path: Path
    frame_ids: np.ndarray
    points: np.ndarray
    groundtruth: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 duida 姿态落点模型与球轨迹落点模型的预测差异"
    )
    parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=None,
        help="姿态落点.pt模型路径。duida 3d pose 默认应配 before 模型。",
    )
    parser.add_argument(
        "--pose-norm-stats",
        type=Path,
        default=None,
        help="姿态模型配套的 _norm_stats.npz；不传时会按模型文件名自动推断。",
    )
    parser.add_argument(
        "--fit-model",
        type=Path,
        default=None,
        help="landing_fitting 的 MLP checkpoint 路径；仅 fit-mode=mlp 时需要。",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("mlp", "curve-fit"),
        default="mlp",
        help="球轨迹落点算法。默认使用 landing_fitting 的 MLP hybrid 推理。",
    )
    parser.add_argument(
        "--pose-mode",
        choices=("before", "after"),
        default="before",
        help="姿态落点算法模式。duida 3d pose 默认应使用 before。",
    )
    parser.add_argument(
        "--pose-window-size",
        type=int,
        default=50,
        help="姿态模型单次推理使用的帧数，保持与 infer_landpoint_pred.py 一致。",
    )
    parser.add_argument(
        "--pose-skip-n",
        type=int,
        default=0,
        help="仅 before 模式使用，表示跳过末尾多少帧后再取 50 帧。",
    )
    parser.add_argument("--fit-fps", type=float, default=300.0)
    parser.add_argument("--landing-z", type=float, default=0.0)
    parser.add_argument(
        "--fit-device",
        type=str,
        default="cpu",
        help="MLP checkpoint 的推理设备，例如 cpu 或 cuda。",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查文件配对与参数，不加载模型、不执行推理。",
    )
    return parser.parse_args()


def normalize_sample_id_from_pose(stem: str) -> str:
    if "---" in stem:
        head, tail = stem.rsplit("---", 1)
        return f"{head}_{tail}"
    return stem


def normalize_sample_id_from_fit(stem: str) -> str:
    return stem


def list_txt_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"路径不是目录: {folder}")
    files = sorted(path for path in folder.iterdir() if path.suffix == ".txt")
    if not files:
        raise ValueError(f"目录中没有 txt 文件: {folder}")
    return files


def collect_pairs(pose_dir: Path, fit_dir: Path) -> list[tuple[str, Path, Path]]:
    pose_map = {
        normalize_sample_id_from_pose(path.stem): path
        for path in list_txt_files(pose_dir)
    }
    fit_map = {
        normalize_sample_id_from_fit(path.stem): path
        for path in list_txt_files(fit_dir)
    }

    common_ids = sorted(set(pose_map) & set(fit_map))
    if not common_ids:
        raise ValueError("两套数据没有可配对的同名样本")

    missing_pose = sorted(set(fit_map) - set(pose_map))
    missing_fit = sorted(set(pose_map) - set(fit_map))
    if missing_pose:
        print(f"跳过 {len(missing_pose)} 个没有对应 pose 文件的 fit 样本", file=sys.stderr)
    if missing_fit:
        print(f"跳过 {len(missing_fit)} 个没有对应 fit 文件的 pose 样本", file=sys.stderr)

    return [(sample_id, pose_map[sample_id], fit_map[sample_id]) for sample_id in common_ids]


def parse_numeric_line(raw_line: str, expected_min_dims: int, path: Path, line_no: int) -> tuple[int | None, np.ndarray]:
    text = raw_line.strip()
    if not text:
        raise ValueError(f"{path}: 第 {line_no} 行为空")

    frame_id = None
    payload = text
    if ":" in text:
        frame_text, payload = text.split(":", 1)
        try:
            frame_id = int(frame_text.strip())
        except ValueError as exc:
            raise ValueError(f"{path}: 第 {line_no} 行 frame id 非法: {frame_text!r}") from exc

    parts = [part.strip() for part in payload.split(",")]
    if len(parts) < expected_min_dims:
        raise ValueError(
            f"{path}: 第 {line_no} 行维度不足，期望至少 {expected_min_dims}，实际 {len(parts)}"
        )
    try:
        values = np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"{path}: 第 {line_no} 行包含非法数值") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: 第 {line_no} 行包含非有限数值")
    return frame_id, values


def split_data_and_groundtruth_lines(path: Path) -> tuple[list[tuple[int, str]], tuple[int, str]]:
    lines = [
        (line_no, raw_line)
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if raw_line.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"{path}: 至少需要 1 行数据和 1 行 groundtruth")
    return lines[:-1], lines[-1]


def load_pose_sequence(path: Path) -> PoseSequence:
    frame_ids: list[int] = []
    rows: list[np.ndarray] = []
    data_lines, groundtruth_line = split_data_and_groundtruth_lines(path)

    for line_no, raw_line in data_lines:
        try:
            frame_id, values = parse_numeric_line(raw_line, expected_min_dims=63, path=path, line_no=line_no)
        except ValueError as exc:
            if "维度不足" in str(exc):
                print(f"跳过坏帧: {exc}")
                continue
            raise
        rows.append(values)
        frame_ids.append(frame_id if frame_id is not None else len(frame_ids))

    if not rows:
        raise ValueError(f"{path}: 没有可用姿态帧")

    features = np.stack(rows, axis=0)
    ids = np.asarray(frame_ids, dtype=np.int64)
    if np.any(np.diff(ids) <= 0):
        raise ValueError(f"{path}: frame id 必须严格递增")

    gt_line_no, gt_raw_line = groundtruth_line
    _gt_frame_id, groundtruth = parse_numeric_line(
        gt_raw_line, expected_min_dims=3, path=path, line_no=gt_line_no
    )
    return PoseSequence(path=path, frame_ids=ids, features=features, groundtruth=groundtruth[:3].astype(np.float64))


def load_trajectory_sequence(path: Path) -> TrajectorySequence:
    frame_ids: list[int] = []
    rows: list[np.ndarray] = []
    data_lines, groundtruth_line = split_data_and_groundtruth_lines(path)

    for line_no, raw_line in data_lines:
        frame_id, values = parse_numeric_line(raw_line, expected_min_dims=3, path=path, line_no=line_no)
        rows.append(values[:3].astype(np.float64))
        frame_ids.append(frame_id if frame_id is not None else len(frame_ids))

    if len(rows) < 3:
        raise ValueError(f"{path}: 至少需要 3 个轨迹点")

    points = np.stack(rows, axis=0)
    ids = np.asarray(frame_ids, dtype=np.int64)
    if np.any(np.diff(ids) <= 0):
        raise ValueError(f"{path}: frame id 必须严格递增")

    gt_line_no, gt_raw_line = groundtruth_line
    _gt_frame_id, groundtruth = parse_numeric_line(
        gt_raw_line, expected_min_dims=3, path=path, line_no=gt_line_no
    )
    return TrajectorySequence(path=path, frame_ids=ids, points=points, groundtruth=groundtruth[:3].astype(np.float64))


def select_pose_window(
    sequence: PoseSequence,
    mode: str,
    window_size: int,
    skip_n: int,
) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("pose-window-size 必须为正数")
    if skip_n < 0:
        raise ValueError("pose-skip-n 不能为负数")

    total_frames = len(sequence.features)
    if total_frames < window_size + skip_n:
        raise ValueError(
            f"{sequence.path}: 帧数不足，至少需要 {window_size + skip_n} 帧，实际 {total_frames}"
        )

    end = total_frames - skip_n if skip_n > 0 else total_frames
    start = end - window_size
    selected = sequence.features[start:end]

    if mode == "before":
        if selected.shape[1] < 63:
            raise ValueError(f"{sequence.path}: before 模式需要至少 63 维输入")
        selected = selected[:, :63]
    elif mode == "after":
        if selected.shape[1] < 66:
            raise ValueError(
                f"{sequence.path}: after 模式需要 66 维输入，但当前只有 {selected.shape[1]} 维"
            )
        selected = selected[:, :66]
    else:
        raise ValueError(f"未知 pose 模式: {mode}")

    return selected.astype(np.float32, copy=False)


class PoseLandpointPredictor:
    def __init__(self, model_path: Path, norm_stats_path: Path | None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PyTorch，请在项目训练环境中运行，例如 conda activate torchcu3"
            ) from exc

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_stats_path = self._resolve_norm_stats_path(model_path, norm_stats_path)
        stats = np.load(self.norm_stats_path)
        self.feature_mean = np.asarray(stats["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(stats["feature_std"], dtype=np.float32)
        self.label_mean = np.asarray(stats["label_mean"], dtype=np.float32)
        self.label_std = np.asarray(stats["label_std"], dtype=np.float32)
        self.use_time_pos_encoding = bool(
            np.asarray(stats.get("use_time_pos_encoding", False)).item()
        )
        self.reference_fps = float(np.asarray(stats.get("reference_fps", 300.0)).item())
        self.input_dim = int(self.feature_mean.shape[-1])
        if self.input_dim % 3 != 0:
            raise ValueError(
                f"{self.norm_stats_path}: feature_mean 最后一维不是 3 的倍数，实际 {self.input_dim}"
            )
        self.num_points = self.input_dim // 3

        if np.any(self.feature_std <= 0) or np.any(self.label_std <= 0):
            raise ValueError(f"{self.norm_stats_path}: norm stats 中存在非正标准差")

        if self.use_time_pos_encoding:
            self.model = ImprovedTransformerTimePEModel(
                seq_len=50,
                num_points=self.num_points,
                reference_fps=self.reference_fps,
            )
        else:
            self.model = ImprovedTransformerModel(seq_len=50, num_points=self.num_points)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_norm_stats_path(model_path: Path, explicit_path: Path | None) -> Path:
        if explicit_path is not None:
            if not explicit_path.is_file():
                raise FileNotFoundError(f"姿态 norm stats 不存在: {explicit_path}")
            return explicit_path

        inferred = model_path.with_name(f"{model_path.stem}_norm_stats.npz")
        if not inferred.is_file():
            raise FileNotFoundError(
                f"未找到姿态模型配套 norm stats: {inferred}。请显式传入 --pose-norm-stats"
            )
        return inferred

    def predict(self, pose_window: np.ndarray) -> np.ndarray:
        if pose_window.ndim != 2:
            raise ValueError(f"姿态输入必须是二维数组，实际 shape={pose_window.shape}")
        if pose_window.shape[1] != self.input_dim:
            raise ValueError(
                f"姿态输入维度与模型不匹配: 输入 {pose_window.shape[1]} 维, 模型期望 {self.input_dim} 维"
            )

        normalized = (pose_window - self.feature_mean) / self.feature_std
        seq = self.torch.as_tensor(
            normalized[np.newaxis, :, :],
            dtype=self.torch.float32,
            device=self.device,
        )
        mask = self.torch.ones((1, pose_window.shape[0]), dtype=self.torch.bool, device=self.device)

        with self.torch.no_grad():
            if self.use_time_pos_encoding:
                raise ValueError(
                    "当前姿态 .pt 推理脚本暂不支持 use_time_pos_encoding=True 的模型；"
                    "需要先明确 hit_index/time_pos 的推理口径。"
                )
            pred_xyz, _pred_var, _pred_time, _pred_direction = self.model(seq, mask)

        pred_xyz = pred_xyz.squeeze(0).detach().cpu().numpy().astype(np.float32)
        real_xyz = pred_xyz * self.label_std[0, :3] + self.label_mean[0, :3]
        if real_xyz.shape[0] < 3:
            raise ValueError(f"姿态模型输出维度不足 3: shape={real_xyz.shape}")
        if not np.all(np.isfinite(real_xyz[:3])):
            raise ValueError(f"姿态模型输出存在非有限值: {real_xyz[:3]}")
        return real_xyz[:3].astype(np.float64)


class FitLandingPredictor:
    def __init__(self, fit_mode: str, model_path: Path | None, device: str):
        self.fit_mode = fit_mode
        self.predictor = None
        if fit_mode == "mlp":
            if model_path is None:
                raise ValueError("fit-mode=mlp 时必须提供 --fit-model")
            if not model_path.is_file():
                raise FileNotFoundError(f"fit MLP checkpoint 不存在: {model_path}")
            self.predictor = CParameterPredictor(model_path, device=device)

    def predict(self, sequence: TrajectorySequence, fps: float, landing_z: float) -> dict:
        if self.fit_mode == "mlp":
            assert self.predictor is not None
            params, c_values, features = predict_hybrid_parameters(
                sequence.points,
                sequence.frame_ids,
                fps,
                self.predictor,
                self.predictor.state_window_size,
            )
            state_window_size = self.predictor.state_window_size
        elif self.fit_mode == "curve-fit":
            params = fit_all_parameters(sequence.points, sequence.frame_ids, fps)
            c_values = np.asarray(
                [params["x"][2], params["y"][2], params["z"][2]],
                dtype=np.float64,
            )
            features = None
            state_window_size = None
        else:
            raise ValueError(f"未知 fit-mode: {self.fit_mode}")

        landing_time_ms = float(solve_landing_time(params, landing_z))
        prediction = evaluate_position(landing_time_ms, params).astype(np.float64)
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"球轨迹预测结果存在非有限值: {prediction}")

        return {
            "predicted_xyz": prediction,
            "landing_time_ms": landing_time_ms,
            "c_values": np.asarray(c_values, dtype=np.float64),
            "state_window_size": state_window_size,
            "state_features": None if features is None else np.asarray(features, dtype=np.float64),
        }


def vector_to_row(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_x": float(values[0]),
        f"{prefix}_y": float(values[1]),
        f"{prefix}_z": float(values[2]),
    }


def compute_summary(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    xy_diffs = np.asarray([row["delta_xy"] for row in rows], dtype=np.float64)
    xyz_diffs = np.asarray([row["delta_xyz"] for row in rows], dtype=np.float64)
    pose_error_xy = np.asarray([row["pose_error_xy"] for row in rows], dtype=np.float64)
    pose_error_xyz = np.asarray([row["pose_error_xyz"] for row in rows], dtype=np.float64)
    fit_error_xy = np.asarray([row["fit_error_xy"] for row in rows], dtype=np.float64)
    fit_error_xyz = np.asarray([row["fit_error_xyz"] for row in rows], dtype=np.float64)
    gt_delta_xy = np.asarray([row["groundtruth_delta_xy"] for row in rows], dtype=np.float64)
    gt_delta_xyz = np.asarray([row["groundtruth_delta_xyz"] for row in rows], dtype=np.float64)

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "sample_count": len(rows),
        "delta_xy": stats(xy_diffs),
        "delta_xyz": stats(xyz_diffs),
        "pose_error_xy": stats(pose_error_xy),
        "pose_error_xyz": stats(pose_error_xyz),
        "fit_error_xy": stats(fit_error_xy),
        "fit_error_xyz": stats(fit_error_xyz),
        "groundtruth_delta_xy": stats(gt_delta_xy),
        "groundtruth_delta_xyz": stats(gt_delta_xyz),
    }


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "sample_id",
        "pose_file",
        "fit_file",
        "pose_mode",
        "pose_skip_n",
        "pose_window_size",
        "pose_frames_total",
        "fit_mode",
        "fit_frames_total",
        "fit_landing_time_ms",
        "fit_state_window_size",
        "groundtruth_x",
        "groundtruth_y",
        "groundtruth_z",
        "fit_groundtruth_x",
        "fit_groundtruth_y",
        "fit_groundtruth_z",
        "pose_pred_x",
        "pose_pred_y",
        "pose_pred_z",
        "fit_pred_x",
        "fit_pred_y",
        "fit_pred_z",
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_xy",
        "delta_xyz",
        "pose_error_x",
        "pose_error_y",
        "pose_error_z",
        "pose_error_xy",
        "pose_error_xyz",
        "fit_error_x",
        "fit_error_y",
        "fit_error_z",
        "fit_error_xy",
        "fit_error_xyz",
        "groundtruth_delta_xy",
        "groundtruth_delta_xyz",
        "fit_c1",
        "fit_c2",
        "fit_c3",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, payload: dict) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def validate_args(args: argparse.Namespace) -> None:
    if args.pose_mode == "before" and args.pose_skip_n < 0:
        raise ValueError("--pose-skip-n 不能为负数")
    if args.fit_fps <= 0:
        raise ValueError("--fit-fps 必须为正数")
    if args.pose_window_size <= 0:
        raise ValueError("--pose-window-size 必须为正数")
    if not args.dry_run and args.pose_model is None:
        raise ValueError("请提供姿态 .pt 模型路径 --pose-model")
    if not args.dry_run and args.fit_mode == "mlp" and args.fit_model is None:
        raise ValueError("fit-mode=mlp 时请提供 --fit-model")


def run_comparison(args: argparse.Namespace) -> dict:
    pairs = collect_pairs(args.pose_dir, args.fit_dir)

    if args.dry_run:
        return {
            "sample_count": len(pairs),
            "sample_ids": [sample_id for sample_id, _, _ in pairs],
            "pose_mode": args.pose_mode,
            "fit_mode": args.fit_mode,
        }

    pose_predictor = PoseLandpointPredictor(args.pose_model, args.pose_norm_stats)
    fit_predictor = FitLandingPredictor(args.fit_mode, args.fit_model, args.fit_device)

    rows: list[dict] = []
    for sample_id, pose_path, fit_path in pairs:
        pose_sequence = load_pose_sequence(pose_path)
        fit_sequence = load_trajectory_sequence(fit_path)

        pose_window = select_pose_window(
            pose_sequence,
            mode=args.pose_mode,
            window_size=args.pose_window_size,
            skip_n=args.pose_skip_n,
        )
        pose_pred = pose_predictor.predict(pose_window)
        fit_result = fit_predictor.predict(
            fit_sequence,
            fps=args.fit_fps,
            landing_z=args.landing_z,
        )
        fit_pred = fit_result["predicted_xyz"]

        groundtruth = pose_sequence.groundtruth
        fit_groundtruth = fit_sequence.groundtruth
        delta = pose_pred - fit_pred
        pose_error = pose_pred - groundtruth
        fit_error = fit_pred - groundtruth
        groundtruth_delta = groundtruth - fit_groundtruth
        delta_xy = float(np.linalg.norm(delta[:2]))
        delta_xyz = float(np.linalg.norm(delta))
        pose_error_xy = float(np.linalg.norm(pose_error[:2]))
        pose_error_xyz = float(np.linalg.norm(pose_error))
        fit_error_xy = float(np.linalg.norm(fit_error[:2]))
        fit_error_xyz = float(np.linalg.norm(fit_error))
        groundtruth_delta_xy = float(np.linalg.norm(groundtruth_delta[:2]))
        groundtruth_delta_xyz = float(np.linalg.norm(groundtruth_delta))
        if not all(
            math.isfinite(value)
            for value in (
                delta_xy,
                delta_xyz,
                pose_error_xy,
                pose_error_xyz,
                fit_error_xy,
                fit_error_xyz,
                groundtruth_delta_xy,
                groundtruth_delta_xyz,
            )
        ):
            raise ValueError(f"{sample_id}: 差值存在非有限数值")

        row = {
            "sample_id": sample_id,
            "pose_file": pose_path.name,
            "fit_file": fit_path.name,
            "pose_mode": args.pose_mode,
            "pose_skip_n": args.pose_skip_n,
            "pose_window_size": args.pose_window_size,
            "pose_frames_total": len(pose_sequence.features),
            "fit_mode": args.fit_mode,
            "fit_frames_total": len(fit_sequence.points),
            "fit_landing_time_ms": float(fit_result["landing_time_ms"]),
            "fit_state_window_size": fit_result["state_window_size"],
            "groundtruth_x": float(groundtruth[0]),
            "groundtruth_y": float(groundtruth[1]),
            "groundtruth_z": float(groundtruth[2]),
            "fit_groundtruth_x": float(fit_groundtruth[0]),
            "fit_groundtruth_y": float(fit_groundtruth[1]),
            "fit_groundtruth_z": float(fit_groundtruth[2]),
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "delta_z": float(delta[2]),
            "delta_xy": delta_xy,
            "delta_xyz": delta_xyz,
            "pose_error_x": float(pose_error[0]),
            "pose_error_y": float(pose_error[1]),
            "pose_error_z": float(pose_error[2]),
            "pose_error_xy": pose_error_xy,
            "pose_error_xyz": pose_error_xyz,
            "fit_error_x": float(fit_error[0]),
            "fit_error_y": float(fit_error[1]),
            "fit_error_z": float(fit_error[2]),
            "fit_error_xy": fit_error_xy,
            "fit_error_xyz": fit_error_xyz,
            "groundtruth_delta_xy": groundtruth_delta_xy,
            "groundtruth_delta_xyz": groundtruth_delta_xyz,
            "fit_c1": float(fit_result["c_values"][0]),
            "fit_c2": float(fit_result["c_values"][1]),
            "fit_c3": float(fit_result["c_values"][2]),
        }
        row.update(vector_to_row("pose_pred", pose_pred))
        row.update(vector_to_row("fit_pred", fit_pred))
        rows.append(row)

    summary = compute_summary(rows)
    payload = {
        "config": {
            "pose_dir": str(args.pose_dir),
            "fit_dir": str(args.fit_dir),
            "pose_model": str(args.pose_model),
            "pose_norm_stats": None if args.pose_norm_stats is None else str(args.pose_norm_stats),
            "fit_model": None if args.fit_model is None else str(args.fit_model),
            "pose_mode": args.pose_mode,
            "pose_skip_n": args.pose_skip_n,
            "pose_window_size": args.pose_window_size,
            "fit_mode": args.fit_mode,
            "fit_fps": args.fit_fps,
            "landing_z": args.landing_z,
            "fit_device": args.fit_device,
        },
        "summary": summary,
        "rows": rows,
    }
    write_csv(args.output_csv, rows)
    write_summary_json(args.output_json, payload)
    return payload


def main() -> None:
    args = parse_args()
    validate_args(args)
    result = run_comparison(args)

    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = result["summary"]
    print(f"已完成 {summary['sample_count']} 个样本的两套算法落点对比。")
    print(f"delta_xy  mean={summary['delta_xy']['mean']:.6f}, max={summary['delta_xy']['max']:.6f}")
    print(f"delta_xyz mean={summary['delta_xyz']['mean']:.6f}, max={summary['delta_xyz']['max']:.6f}")
    print(
        "pose_error_xy "
        f"mean={summary['pose_error_xy']['mean']:.6f}, max={summary['pose_error_xy']['max']:.6f}"
    )
    print(
        "fit_error_xy  "
        f"mean={summary['fit_error_xy']['mean']:.6f}, max={summary['fit_error_xy']['max']:.6f}"
    )
    print(
        "pose_error_xyz "
        f"mean={summary['pose_error_xyz']['mean']:.6f}, max={summary['pose_error_xyz']['max']:.6f}"
    )
    print(
        "fit_error_xyz  "
        f"mean={summary['fit_error_xyz']['mean']:.6f}, max={summary['fit_error_xyz']['max']:.6f}"
    )
    print(f"CSV 输出: {args.output_csv}")
    print(f"JSON 输出: {args.output_json}")


if __name__ == "__main__":
    main()
