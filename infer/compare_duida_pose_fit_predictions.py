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
from util.after_motion import AfterMotionPredictor, is_after_motion_checkpoint  # noqa: E402
from util.after_ball_racket import (  # noqa: E402
    AfterBallRacketPredictor,
    is_after_ball_racket_checkpoint,
)
from util.before_pose_racket import (  # noqa: E402
    BeforePoseRacketPredictor,
    is_before_pose_racket_checkpoint,
)


DEFAULT_POSE_DIR = ROOT / "datasets" / "20260418_duida_3d_pose"
DEFAULT_FIT_DIR = ROOT / "datasets" / "20260418_duida_fit_fall"
DEFAULT_OUTPUT_CSV = ROOT / "infer" / "duida_pose_fit_prediction_comparison.csv"
DEFAULT_OUTPUT_JSON = ROOT / "infer" / "duida_pose_fit_prediction_summary.json"
DEFAULT_VISUALIZATION_DIR = ROOT / "infer" / "duida_pose_fit_3d"

COURT_LENGTH_CM = 1340.0
COURT_WIDTH_CM = 670.0
COURT_HALF_WIDTH_CM = COURT_WIDTH_CM / 2.0
COURT_NET_X_CM = COURT_LENGTH_CM / 2.0
COURT_NET_HEIGHT_CM = 155.0


@dataclass
class PoseSequence:
    path: Path
    frame_ids: np.ndarray
    features: np.ndarray


@dataclass
class TrajectorySequence:
    path: Path
    frame_ids: np.ndarray
    points: np.ndarray


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
        "--before-pose-model",
        type=Path,
        default=None,
        help=(
            "三维可视化使用的 before 模型。不传时，若 --pose-mode=before 则复用 "
            "--pose-model，否则自动查找同目录 before.pt。"
        ),
    )
    parser.add_argument(
        "--before-pose-norm-stats",
        type=Path,
        default=None,
        help="before 模型配套 norm stats；不传时按模型文件名推断。",
    )
    parser.add_argument(
        "--after-pose-model",
        type=Path,
        default=None,
        help=(
            "三维可视化使用的 after 模型。不传时，若 --pose-mode=after 则复用 "
            "--pose-model，否则自动查找同目录 after.pt。"
        ),
    )
    parser.add_argument(
        "--after-pose-norm-stats",
        type=Path,
        default=None,
        help="after 模型配套 norm stats；不传时按模型文件名推断。",
    )
    parser.add_argument(
        "--ball-only-after-pose-model",
        type=Path,
        default=ROOT / "models" / "after_motion_trend.pt",
        help="Optional previous ball-only motion model shown as a separate 3-D reference point.",
    )
    parser.add_argument(
        "--legacy-after-pose-model",
        type=Path,
        default=ROOT / "models" / "after.pt",
        help="Optional legacy after model shown as a separate reference point in 3-D output.",
    )
    parser.add_argument(
        "--legacy-after-pose-norm-stats",
        type=Path,
        default=None,
        help="Norm stats paired with --legacy-after-pose-model.",
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
        "--visualization-dir",
        type=Path,
        default=DEFAULT_VISUALIZATION_DIR,
        help=(
            "每个样本的交互式三维 HTML 输出目录。默认: "
            f"{DEFAULT_VISUALIZATION_DIR}"
        ),
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="关闭 before/after/拟合轨迹的三维可视化输出。",
    )
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
        raise ValueError(f"以下 fit 样本没有对应 pose 文件: {missing_pose}")
    if missing_fit:
        raise ValueError(f"以下 pose 样本没有对应 fit 文件: {missing_fit}")

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


def load_pose_sequence(path: Path) -> PoseSequence:
    frame_ids: list[int] = []
    rows: list[np.ndarray] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
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
    return PoseSequence(path=path, frame_ids=ids, features=features)


def load_trajectory_sequence(path: Path) -> TrajectorySequence:
    frame_ids: list[int] = []
    rows: list[np.ndarray] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        frame_id, values = parse_numeric_line(raw_line, expected_min_dims=3, path=path, line_no=line_no)
        rows.append(values[:3].astype(np.float64))
        frame_ids.append(frame_id if frame_id is not None else len(frame_ids))

    if len(rows) < 3:
        raise ValueError(f"{path}: 至少需要 3 个轨迹点")

    points = np.stack(rows, axis=0)
    ids = np.asarray(frame_ids, dtype=np.int64)
    if np.any(np.diff(ids) <= 0):
        raise ValueError(f"{path}: frame id 必须严格递增")
    return TrajectorySequence(path=path, frame_ids=ids, points=points)


def pose_window_bounds(
    sequence: PoseSequence,
    window_size: int,
    skip_n: int,
) -> tuple[int, int]:
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
    return start, end


def select_pose_window(
    sequence: PoseSequence,
    mode: str,
    window_size: int,
    skip_n: int,
) -> np.ndarray:
    start, end = pose_window_bounds(sequence, window_size, skip_n)
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
    def __init__(self, model_path: Path, norm_stats_path: Path | None, state_dict=None):
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

        if state_dict is None:
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

    def predict(
        self,
        pose_window: np.ndarray,
        frame_ids: np.ndarray | None = None,
    ) -> np.ndarray:
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


def load_pose_predictor(model_path: Path, norm_stats_path: Path | None):
    """Load a legacy Transformer or the trajectory-order-sensitive after model."""

    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    if is_after_ball_racket_checkpoint(checkpoint):
        return AfterBallRacketPredictor(model_path)
    if is_after_motion_checkpoint(checkpoint):
        return AfterMotionPredictor(model_path)
    if is_before_pose_racket_checkpoint(checkpoint):
        return BeforePoseRacketPredictor(model_path)
    return PoseLandpointPredictor(model_path, norm_stats_path, state_dict=checkpoint)


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
            "params": params,
        }


def resolve_pose_model_for_mode(
    args: argparse.Namespace,
    mode: str,
) -> tuple[Path, Path | None]:
    if mode == "before":
        explicit_model = args.before_pose_model
        explicit_stats = args.before_pose_norm_stats
    elif mode == "after":
        explicit_model = args.after_pose_model
        explicit_stats = args.after_pose_norm_stats
    else:
        raise ValueError(f"未知 pose 模式: {mode}")

    if explicit_model is not None:
        model_path = explicit_model
    elif args.pose_mode == mode:
        model_path = args.pose_model
    else:
        if mode == "after":
            ball_racket_model = args.pose_model.with_name(
                "after_ball_racket_trend.pt"
            )
            ball_only_model = args.pose_model.with_name("after_motion_trend.pt")
            if ball_racket_model.is_file():
                model_path = ball_racket_model
            elif ball_only_model.is_file():
                model_path = ball_only_model
            else:
                model_path = args.pose_model.with_name("after.pt")
        else:
            pose_racket_model = args.pose_model.with_name(
                "before_pose_racket_trend.pt"
            )
            if pose_racket_model.is_file():
                model_path = pose_racket_model
            else:
                model_path = args.pose_model.with_name("before.pt")

    stats_path = explicit_stats
    if stats_path is None and model_path == args.pose_model:
        stats_path = args.pose_norm_stats

    if not model_path.is_file():
        raise FileNotFoundError(
            f"三维可视化需要 {mode} 模型，但未找到: {model_path}。"
            f"请通过 --{mode}-pose-model 显式指定。"
        )
    return model_path, stats_path


def _racket_center(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2 or window.shape[1] < 63:
        raise ValueError(f"球拍轨迹需要至少 63 维姿态输入，实际 shape={window.shape}")
    racket_points = window[:, 17 * 3 : 21 * 3].reshape(-1, 4, 3)
    return np.mean(racket_points, axis=1)


def _after_ball_trajectory(
    window: np.ndarray,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if window.ndim != 2 or window.shape[1] < 66:
        raise ValueError(f"击球后球轨迹需要 66 维输入，实际 shape={window.shape}")
    ball = window[:, 21 * 3 : 22 * 3]
    valid = np.all(np.isfinite(ball), axis=1) & ~np.all(
        np.isclose(ball, 0.0, atol=1e-9), axis=1
    )
    return ball[valid], frame_ids[valid]


def _fit_curve_points(fit_result: dict, point_count: int = 240) -> np.ndarray:
    params = fit_result["params"]
    start_time = float(params["times_ms"][0])
    landing_time = float(fit_result["landing_time_ms"])
    if landing_time < start_time:
        raise ValueError(
            f"拟合落地时间 {landing_time:.6f} 早于轨迹起始时间 {start_time:.6f}"
        )
    times = np.linspace(start_time, landing_time, point_count, dtype=np.float64)
    return evaluate_position(times, params).T


def write_sample_3d_visualization(
    output_path: Path,
    sample_id: str,
    fit_sequence: TrajectorySequence,
    fit_result: dict,
    before_window: np.ndarray,
    before_frame_ids: np.ndarray,
    after_window: np.ndarray,
    after_frame_ids: np.ndarray,
    before_pred: np.ndarray,
    after_pred: np.ndarray,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "生成交互式三维轨迹图需要 plotly，请先安装: pip install plotly"
        ) from exc

    fit_curve = _fit_curve_points(fit_result)
    fit_pred = np.asarray(fit_result["predicted_xyz"], dtype=np.float64)
    before_racket = _racket_center(before_window)
    after_racket = _racket_center(after_window)
    after_ball, after_ball_frame_ids = _after_ball_trajectory(
        after_window, after_frame_ids
    )

    court_x = [
        0.0,
        COURT_LENGTH_CM,
        COURT_LENGTH_CM,
        0.0,
        0.0,
        None,
        COURT_NET_X_CM,
        COURT_NET_X_CM,
    ]
    court_y = [
        -COURT_HALF_WIDTH_CM,
        -COURT_HALF_WIDTH_CM,
        COURT_HALF_WIDTH_CM,
        COURT_HALF_WIDTH_CM,
        -COURT_HALF_WIDTH_CM,
        None,
        -COURT_HALF_WIDTH_CM,
        COURT_HALF_WIDTH_CM,
    ]
    court_z = [0.0] * len(court_x)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=court_x,
            y=court_y,
            z=court_z,
            mode="lines",
            line={"color": "#7a7a7a", "width": 4},
            name="Court boundary / net line",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[COURT_NET_X_CM, COURT_NET_X_CM, None, COURT_NET_X_CM, COURT_NET_X_CM],
            y=[-COURT_HALF_WIDTH_CM, -COURT_HALF_WIDTH_CM, None, COURT_HALF_WIDTH_CM, COURT_HALF_WIDTH_CM],
            z=[0.0, COURT_NET_HEIGHT_CM, None, 0.0, COURT_NET_HEIGHT_CM],
            mode="lines",
            line={"color": "#7a7a7a", "width": 3},
            name="Net posts",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=fit_sequence.points[:, 0],
            y=fit_sequence.points[:, 1],
            z=fit_sequence.points[:, 2],
            mode="markers",
            marker={"size": 3, "color": "#6b7280", "opacity": 0.65},
            name="Fit input ball observations",
            hovertemplate="fit observation<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=fit_curve[:, 0],
            y=fit_curve[:, 1],
            z=fit_curve[:, 2],
            mode="lines",
            line={"color": "#d946ef", "width": 7},
            name="Fitted / extrapolated ball trajectory",
            hovertemplate="fitted trajectory<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=before_racket[:, 0],
            y=before_racket[:, 1],
            z=before_racket[:, 2],
            mode="lines+markers",
            line={"color": "#2563eb", "width": 5},
            marker={"size": 2, "color": "#2563eb"},
            customdata=before_frame_ids,
            name="Racket-center trajectory (before window)",
            hovertemplate="before racket center<br>frame=%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=after_racket[:, 0],
            y=after_racket[:, 1],
            z=after_racket[:, 2],
            mode="lines+markers",
            line={"color": "#16a34a", "width": 5, "dash": "dash"},
            marker={"size": 2, "color": "#16a34a"},
            customdata=after_frame_ids,
            name="Racket-center trajectory (after window)",
            hovertemplate="after racket center<br>frame=%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    if len(after_ball):
        fig.add_trace(
            go.Scatter3d(
                x=after_ball[:, 0],
                y=after_ball[:, 1],
                z=after_ball[:, 2],
                mode="lines+markers",
                line={"color": "#f59e0b", "width": 8},
                marker={"size": 5, "color": "#f59e0b", "symbol": "circle"},
                customdata=after_ball_frame_ids,
                name="Ball trajectory used by after model",
                hovertemplate="after-model ball<br>frame=%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
            )
        )

    prediction_specs = (
        (fit_pred, "Fit predicted landing", "#dc2626", "x"),
        (before_pred, "Before-model predicted landing", "#2563eb", "diamond"),
        (after_pred, "Current loaded after predicted landing", "#16a34a", "square"),
    )
    for point, name, color, symbol in prediction_specs:
        fig.add_trace(
            go.Scatter3d(
                x=[float(point[0])],
                y=[float(point[1])],
                z=[float(point[2])],
                mode="markers+text",
                marker={
                    "size": 9,
                    "color": color,
                    "symbol": symbol,
                    "line": {"color": "#ffffff", "width": 1},
                },
                text=[name],
                textposition="top center",
                name=name,
                hovertemplate=(
                    f"{name}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>"
                ),
            )
        )

    numeric_sets = [
        fit_sequence.points,
        fit_curve,
        before_racket,
        after_racket,
        np.vstack([fit_pred, before_pred, after_pred]),
    ]
    if len(after_ball):
        numeric_sets.append(after_ball)
    all_points = np.vstack(numeric_sets)
    x_min = min(-30.0, float(np.min(all_points[:, 0])) - 30.0)
    x_max = max(COURT_LENGTH_CM + 30.0, float(np.max(all_points[:, 0])) + 30.0)
    y_min = min(-COURT_HALF_WIDTH_CM - 30.0, float(np.min(all_points[:, 1])) - 30.0)
    y_max = max(COURT_HALF_WIDTH_CM + 30.0, float(np.max(all_points[:, 1])) + 30.0)
    z_min = min(-10.0, float(np.min(all_points[:, 2])) - 10.0)
    z_max = max(COURT_NET_HEIGHT_CM + 30.0, float(np.max(all_points[:, 2])) + 30.0)

    fig.update_layout(
        title={"text": f"{sample_id}: 3D landing prediction comparison", "x": 0.5},
        template="plotly_white",
        scene={
            "xaxis": {"title": "Court X (cm)", "range": [x_min, x_max]},
            "yaxis": {"title": "Court Y (cm)", "range": [y_min, y_max]},
            "zaxis": {"title": "Height Z (cm)", "range": [z_min, z_max]},
            "aspectmode": "manual",
            "aspectratio": {"x": 2.0, "y": 1.0, "z": 0.8},
            "camera": {"eye": {"x": 1.55, "y": -1.7, "z": 1.15}},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 0, "r": 0, "b": 0, "t": 105},
        width=1280,
        height=850,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs="directory",
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )


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

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary = {
        "sample_count": len(rows),
        "delta_xy": stats(xy_diffs),
        "delta_xyz": stats(xyz_diffs),
    }
    after_values = [row.get("after_delta_xy_to_fit") for row in rows]
    after_values = [value for value in after_values if value != "" and value is not None]
    if after_values:
        summary["after_delta_xy_to_fit"] = stats(
            np.asarray(after_values, dtype=np.float64)
        )
    legacy_values = [row.get("legacy_after_delta_xy_to_fit") for row in rows]
    legacy_values = [value for value in legacy_values if value != "" and value is not None]
    if legacy_values:
        summary["legacy_after_delta_xy_to_fit"] = stats(
            np.asarray(legacy_values, dtype=np.float64)
        )
    ball_only_values = [row.get("ball_only_after_delta_xy_to_fit") for row in rows]
    ball_only_values = [
        value for value in ball_only_values if value != "" and value is not None
    ]
    if ball_only_values:
        summary["ball_only_after_delta_xy_to_fit"] = stats(
            np.asarray(ball_only_values, dtype=np.float64)
        )
    return summary


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
        "pose_pred_x",
        "pose_pred_y",
        "pose_pred_z",
        "before_pred_x",
        "before_pred_y",
        "before_pred_z",
        "after_pred_x",
        "after_pred_y",
        "after_pred_z",
        "ball_only_after_pred_x",
        "ball_only_after_pred_y",
        "ball_only_after_pred_z",
        "legacy_after_pred_x",
        "legacy_after_pred_y",
        "legacy_after_pred_z",
        "after_delta_xy_to_fit",
        "ball_only_after_delta_xy_to_fit",
        "legacy_after_delta_xy_to_fit",
        "fit_pred_x",
        "fit_pred_y",
        "fit_pred_z",
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_xy",
        "delta_xyz",
        "fit_c1",
        "fit_c2",
        "fit_c3",
        "visualization_file",
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

    pose_predictor = load_pose_predictor(args.pose_model, args.pose_norm_stats)
    fit_predictor = FitLandingPredictor(args.fit_mode, args.fit_model, args.fit_device)

    before_predictor = None
    after_predictor = None
    before_model_path = None
    after_model_path = None
    ball_only_after_predictor = None
    ball_only_after_model_path = None
    legacy_after_predictor = None
    legacy_after_model_path = None
    if not args.no_visualization:
        before_model_path, before_stats_path = resolve_pose_model_for_mode(args, "before")
        after_model_path, after_stats_path = resolve_pose_model_for_mode(args, "after")

        if before_model_path == args.pose_model and before_stats_path == args.pose_norm_stats:
            before_predictor = pose_predictor
        else:
            before_predictor = load_pose_predictor(before_model_path, before_stats_path)

        if after_model_path == args.pose_model and after_stats_path == args.pose_norm_stats:
            after_predictor = pose_predictor
        else:
            after_predictor = load_pose_predictor(after_model_path, after_stats_path)

        if (
            args.ball_only_after_pose_model is not None
            and args.ball_only_after_pose_model.is_file()
            and args.ball_only_after_pose_model.resolve() != after_model_path.resolve()
        ):
            ball_only_after_model_path = args.ball_only_after_pose_model
            ball_only_after_predictor = load_pose_predictor(
                ball_only_after_model_path,
                None,
            )

        if (
            args.legacy_after_pose_model is not None
            and args.legacy_after_pose_model.is_file()
            and args.legacy_after_pose_model.resolve() != after_model_path.resolve()
        ):
            legacy_after_model_path = args.legacy_after_pose_model
            legacy_after_predictor = load_pose_predictor(
                legacy_after_model_path,
                args.legacy_after_pose_norm_stats,
            )

    rows: list[dict] = []
    visualization_files: list[str] = []
    for sample_id, pose_path, fit_path in pairs:
        pose_sequence = load_pose_sequence(pose_path)
        fit_sequence = load_trajectory_sequence(fit_path)

        pose_start, pose_end = pose_window_bounds(
            pose_sequence,
            args.pose_window_size,
            args.pose_skip_n,
        )
        pose_window = select_pose_window(
            pose_sequence,
            mode=args.pose_mode,
            window_size=args.pose_window_size,
            skip_n=args.pose_skip_n,
        )
        pose_pred = pose_predictor.predict(
            pose_window,
            pose_sequence.frame_ids[pose_start:pose_end],
        )
        fit_result = fit_predictor.predict(
            fit_sequence,
            fps=args.fit_fps,
            landing_z=args.landing_z,
        )
        fit_pred = fit_result["predicted_xyz"]

        before_pred = None
        after_pred = None
        ball_only_after_pred = None
        legacy_after_pred = None
        visualization_path = None
        if not args.no_visualization:
            assert before_predictor is not None
            assert after_predictor is not None

            before_start, before_end = pose_window_bounds(
                pose_sequence,
                args.pose_window_size,
                args.pose_skip_n,
            )
            after_start, after_end = pose_window_bounds(
                pose_sequence,
                args.pose_window_size,
                0,
            )
            before_window = select_pose_window(
                pose_sequence,
                mode="before",
                window_size=args.pose_window_size,
                skip_n=args.pose_skip_n,
            )
            after_window = select_pose_window(
                pose_sequence,
                mode="after",
                window_size=args.pose_window_size,
                skip_n=0,
            )
            before_pred = before_predictor.predict(
                before_window,
                pose_sequence.frame_ids[before_start:before_end],
            )
            after_pred = after_predictor.predict(
                after_window,
                pose_sequence.frame_ids[after_start:after_end],
            )
            if ball_only_after_predictor is not None:
                ball_only_after_pred = ball_only_after_predictor.predict(
                    after_window,
                    pose_sequence.frame_ids[after_start:after_end],
                )
            if legacy_after_predictor is not None:
                legacy_after_pred = legacy_after_predictor.predict(
                    after_window,
                    pose_sequence.frame_ids[after_start:after_end],
                )
            visualization_path = (
                args.visualization_dir / f"{sample_id}_3d.html"
            ).resolve()
            write_sample_3d_visualization(
                visualization_path,
                sample_id,
                fit_sequence,
                fit_result,
                before_window,
                pose_sequence.frame_ids[before_start:before_end],
                after_window,
                pose_sequence.frame_ids[after_start:after_end],
                before_pred,
                after_pred,
            )
            visualization_files.append(str(visualization_path))

        delta = pose_pred - fit_pred
        delta_xy = float(np.linalg.norm(delta[:2]))
        delta_xyz = float(np.linalg.norm(delta))
        if not math.isfinite(delta_xy) or not math.isfinite(delta_xyz):
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
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "delta_z": float(delta[2]),
            "delta_xy": delta_xy,
            "delta_xyz": delta_xyz,
            "fit_c1": float(fit_result["c_values"][0]),
            "fit_c2": float(fit_result["c_values"][1]),
            "fit_c3": float(fit_result["c_values"][2]),
            "visualization_file": "" if visualization_path is None else str(visualization_path),
        }
        row.update(vector_to_row("pose_pred", pose_pred))
        row.update(vector_to_row("fit_pred", fit_pred))
        if before_pred is not None:
            row.update(vector_to_row("before_pred", before_pred))
        else:
            row.update({"before_pred_x": "", "before_pred_y": "", "before_pred_z": ""})
        if after_pred is not None:
            row.update(vector_to_row("after_pred", after_pred))
            row["after_delta_xy_to_fit"] = float(
                np.linalg.norm(after_pred[:2] - fit_pred[:2])
            )
        else:
            row.update({"after_pred_x": "", "after_pred_y": "", "after_pred_z": ""})
            row["after_delta_xy_to_fit"] = ""
        if ball_only_after_pred is not None:
            row.update(vector_to_row("ball_only_after_pred", ball_only_after_pred))
            row["ball_only_after_delta_xy_to_fit"] = float(
                np.linalg.norm(ball_only_after_pred[:2] - fit_pred[:2])
            )
        else:
            row.update(
                {
                    "ball_only_after_pred_x": "",
                    "ball_only_after_pred_y": "",
                    "ball_only_after_pred_z": "",
                    "ball_only_after_delta_xy_to_fit": "",
                }
            )
        if legacy_after_pred is not None:
            row.update(vector_to_row("legacy_after_pred", legacy_after_pred))
            row["legacy_after_delta_xy_to_fit"] = float(
                np.linalg.norm(legacy_after_pred[:2] - fit_pred[:2])
            )
        else:
            row.update(
                {
                    "legacy_after_pred_x": "",
                    "legacy_after_pred_y": "",
                    "legacy_after_pred_z": "",
                    "legacy_after_delta_xy_to_fit": "",
                }
            )
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
            "before_pose_model": None if before_model_path is None else str(before_model_path),
            "after_pose_model": None if after_model_path is None else str(after_model_path),
            "ball_only_after_pose_model": None if ball_only_after_model_path is None else str(ball_only_after_model_path),
            "legacy_after_pose_model": None if legacy_after_model_path is None else str(legacy_after_model_path),
            "visualization_dir": None if args.no_visualization else str(args.visualization_dir),
        },
        "summary": summary,
        "rows": rows,
        "visualizations": visualization_files,
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
    print(f"CSV 输出: {args.output_csv}")
    print(f"JSON 输出: {args.output_json}")
    if not args.no_visualization:
        print(f"三维可视化输出: {args.visualization_dir}")


if __name__ == "__main__":
    main()
