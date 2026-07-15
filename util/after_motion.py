"""Ball-motion-aware landing predictor used by the after-hit model.

The legacy after-hit Transformer receives ball coordinates, but it can learn an
almost order-insensitive mapping from absolute position to landing position.
This module makes the trajectory direction part of the output geometry: the
network predicts a forward distance and a lateral distance in a coordinate
frame whose x-axis is the observed ball velocity.  Reversing the trajectory
therefore necessarily changes the predicted landing direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT = "after_motion_trend_v1"
BALL_SLICE = slice(21 * 3, 22 * 3)
DEFAULT_MAX_BALL_POINTS = 6

FEATURE_NAMES = (
    "last_ball_x",
    "last_ball_y",
    "last_ball_z",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
    "recent_velocity_x",
    "recent_velocity_y",
    "recent_velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "observed_displacement_x",
    "observed_displacement_y",
    "observed_displacement_z",
    "path_length_xy",
    "straightness_xy",
    "valid_point_count",
    "frame_span",
)


@dataclass
class BallMotionFeatures:
    features: np.ndarray
    last_ball: np.ndarray
    direction_xy: np.ndarray
    perpendicular_xy: np.ndarray
    selected_indices: np.ndarray
    selected_frame_ids: np.ndarray
    selected_ball: np.ndarray


def _polyfit_axis(times: np.ndarray, values: np.ndarray, degree: int) -> np.ndarray:
    degree = min(int(degree), len(times) - 1)
    if degree < 1:
        raise ValueError("At least two points are required to fit ball motion")
    return np.polyfit(times, values, degree)


def extract_ball_motion_features(
    frames: np.ndarray,
    frame_ids: Optional[np.ndarray] = None,
    max_ball_points: int = DEFAULT_MAX_BALL_POINTS,
    min_ball_points: int = 4,
) -> BallMotionFeatures:
    """Extract explicit position, velocity and acceleration from recent ball points."""

    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < BALL_SLICE.stop:
        raise ValueError(
            "after motion model expects a 2-D array with at least 66 columns, "
            f"got shape={frames.shape}"
        )
    if frame_ids is None:
        frame_ids = np.arange(len(frames), dtype=np.float64)
    else:
        frame_ids = np.asarray(frame_ids, dtype=np.float64)
        if frame_ids.ndim != 1 or len(frame_ids) != len(frames):
            raise ValueError(
                f"frame_ids must have shape ({len(frames)},), got {frame_ids.shape}"
            )

    ball = frames[:, BALL_SLICE]
    valid = np.all(np.isfinite(ball), axis=1) & ~np.all(
        np.isclose(ball, 0.0, atol=1e-9), axis=1
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < min_ball_points:
        raise ValueError(
            f"after motion model requires at least {min_ball_points} valid ball points, "
            f"got {len(valid_indices)}"
        )
    selected_indices = valid_indices[-max_ball_points:]
    selected_ball = ball[selected_indices]
    selected_frame_ids = frame_ids[selected_indices]
    if np.any(np.diff(selected_frame_ids) <= 0):
        raise ValueError("selected ball frame ids must be strictly increasing")

    times = selected_frame_ids - selected_frame_ids[-1]
    linear_velocity = np.asarray(
        [_polyfit_axis(times, selected_ball[:, axis], 1)[0] for axis in range(3)],
        dtype=np.float64,
    )
    recent_count = min(3, len(selected_ball))
    recent_times = times[-recent_count:]
    recent_ball = selected_ball[-recent_count:]
    recent_velocity = np.asarray(
        [_polyfit_axis(recent_times, recent_ball[:, axis], 1)[0] for axis in range(3)],
        dtype=np.float64,
    )

    acceleration = np.zeros(3, dtype=np.float64)
    if len(selected_ball) >= 3:
        for axis in range(3):
            quadratic = _polyfit_axis(times, selected_ball[:, axis], 2)
            acceleration[axis] = 2.0 * quadratic[0]

    velocity_xy_norm = float(np.linalg.norm(linear_velocity[:2]))
    if velocity_xy_norm < 1e-6:
        raise ValueError("observed ball horizontal speed is too small")
    direction_xy = linear_velocity[:2] / velocity_xy_norm
    perpendicular_xy = np.asarray([-direction_xy[1], direction_xy[0]])

    displacement = selected_ball[-1] - selected_ball[0]
    segment_xy = np.diff(selected_ball[:, :2], axis=0)
    path_length_xy = float(np.linalg.norm(segment_xy, axis=1).sum())
    chord_xy = float(np.linalg.norm(displacement[:2]))
    straightness_xy = chord_xy / max(path_length_xy, 1e-6)
    frame_span = float(selected_frame_ids[-1] - selected_frame_ids[0])

    features = np.concatenate(
        [
            selected_ball[-1],
            linear_velocity,
            recent_velocity,
            acceleration,
            displacement,
            np.asarray(
                [
                    path_length_xy,
                    straightness_xy,
                    float(len(selected_ball)),
                    frame_span,
                ]
            ),
        ]
    ).astype(np.float32)
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(f"feature size mismatch: {len(features)} != {len(FEATURE_NAMES)}")
    if not np.all(np.isfinite(features)):
        raise ValueError("ball motion features contain non-finite values")

    return BallMotionFeatures(
        features=features,
        last_ball=selected_ball[-1].astype(np.float64),
        direction_xy=direction_xy.astype(np.float64),
        perpendicular_xy=perpendicular_xy.astype(np.float64),
        selected_indices=selected_indices.astype(np.int64),
        selected_frame_ids=selected_frame_ids.astype(np.float64),
        selected_ball=selected_ball.astype(np.float64),
    )


class AfterMotionTrendModel(nn.Module):
    """Small MLP predicting motion-aligned forward/lateral landing offsets."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.08,
    ):
        super().__init__()
        layers = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def motion_target(
    landing_xy: np.ndarray,
    motion: BallMotionFeatures,
) -> np.ndarray:
    """Return [log1p(forward_cm), lateral_cm] in the observed motion frame."""

    delta = np.asarray(landing_xy, dtype=np.float64) - motion.last_ball[:2]
    forward = float(np.dot(delta, motion.direction_xy))
    lateral = float(np.dot(delta, motion.perpendicular_xy))
    if forward <= 0:
        raise ValueError(f"landing is not forward along the observed trajectory: {forward:.3f} cm")
    return np.asarray([np.log1p(forward), lateral], dtype=np.float32)


def reconstruct_landing_xy(
    target: np.ndarray,
    motion: BallMotionFeatures,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    log_forward = float(np.clip(target[0], 0.0, np.log1p(2500.0)))
    forward = float(np.expm1(log_forward))
    lateral = float(np.clip(target[1], -1500.0, 1500.0))
    return (
        motion.last_ball[:2]
        + forward * motion.direction_xy
        + lateral * motion.perpendicular_xy
    )


def is_after_motion_checkpoint(checkpoint: Any) -> bool:
    return isinstance(checkpoint, dict) and checkpoint.get("format") == CHECKPOINT_FORMAT


class AfterMotionPredictor:
    """Inference wrapper matching the compare script's predictor interface."""

    def __init__(self, model_path: Path, device: Optional[str] = None):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"after motion model does not exist: {model_path}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        if not is_after_motion_checkpoint(checkpoint):
            raise ValueError(f"not an {CHECKPOINT_FORMAT} checkpoint: {model_path}")

        config: Dict[str, Any] = checkpoint["model_config"]
        self.max_ball_points = int(checkpoint.get("max_ball_points", DEFAULT_MAX_BALL_POINTS))
        def as_numpy(value: Any) -> np.ndarray:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        self.feature_mean = as_numpy(checkpoint["feature_mean"])
        self.feature_std = as_numpy(checkpoint["feature_std"])
        self.target_mean = as_numpy(checkpoint["target_mean"])
        self.target_std = as_numpy(checkpoint["target_std"])
        self.model = AfterMotionTrendModel(**config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.model_path = model_path
        self.checkpoint = checkpoint

    def predict_with_details(
        self,
        pose_window: np.ndarray,
        frame_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        motion = extract_ball_motion_features(
            pose_window,
            frame_ids=frame_ids,
            max_ball_points=self.max_ball_points,
        )
        normalized = (motion.features - self.feature_mean) / self.feature_std
        tensor = torch.as_tensor(
            normalized[None, :], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            predicted_normalized = self.model(tensor)[0].cpu().numpy()
        predicted_target = predicted_normalized * self.target_std + self.target_mean
        landing_xy = reconstruct_landing_xy(predicted_target, motion)
        landing_xyz = np.asarray([landing_xy[0], landing_xy[1], 0.0], dtype=np.float64)
        return {
            "landing_xyz": landing_xyz,
            "motion": motion,
            "target": predicted_target.astype(np.float64),
            "normalized_target": predicted_normalized.astype(np.float64),
        }

    def predict(
        self,
        pose_window: np.ndarray,
        frame_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_with_details(pose_window, frame_ids)["landing_xyz"]
