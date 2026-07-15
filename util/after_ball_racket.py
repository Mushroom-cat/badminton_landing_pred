"""Motion-aware after-hit model that fuses explicit ball and racket dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from util.after_motion import (
    DEFAULT_MAX_BALL_POINTS,
    FEATURE_NAMES as BALL_FEATURE_NAMES,
    BallMotionFeatures,
    extract_ball_motion_features,
    reconstruct_landing_xy,
)


CHECKPOINT_FORMAT = "after_ball_racket_trend_v2"
RACKET_SLICE = slice(17 * 3, 21 * 3)

RACKET_FEATURE_NAMES = (
    "racket_center_x",
    "racket_center_y",
    "racket_center_z",
    "racket_velocity_x",
    "racket_velocity_y",
    "racket_velocity_z",
    "racket_recent_velocity_x",
    "racket_recent_velocity_y",
    "racket_recent_velocity_z",
    "racket_acceleration_x",
    "racket_acceleration_y",
    "racket_acceleration_z",
    "racket_short_axis_x",
    "racket_short_axis_y",
    "racket_short_axis_z",
    "racket_long_axis_x",
    "racket_long_axis_y",
    "racket_long_axis_z",
    "racket_face_normal_x",
    "racket_face_normal_y",
    "racket_face_normal_z",
    "racket_short_axis_length",
    "racket_long_axis_length",
    "racket_short_axis_angular_velocity_x",
    "racket_short_axis_angular_velocity_y",
    "racket_short_axis_angular_velocity_z",
    "racket_long_axis_angular_velocity_x",
    "racket_long_axis_angular_velocity_y",
    "racket_long_axis_angular_velocity_z",
    "ball_relative_to_racket_x",
    "ball_relative_to_racket_y",
    "ball_relative_to_racket_z",
    "ball_racket_relative_velocity_x",
    "ball_racket_relative_velocity_y",
    "ball_racket_relative_velocity_z",
    "ball_velocity_on_short_axis",
    "ball_velocity_on_long_axis",
    "ball_velocity_on_face_normal",
    "relative_velocity_on_short_axis",
    "relative_velocity_on_long_axis",
    "relative_velocity_on_face_normal",
    "last_ball_racket_distance",
    "minimum_observed_ball_racket_distance",
)


@dataclass
class BallRacketMotionFeatures:
    ball_motion: BallMotionFeatures
    ball_features: np.ndarray
    racket_features: np.ndarray
    features: np.ndarray
    valid_racket_count: int
    last_racket_center: np.ndarray


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError(f"{name} contains a degenerate vector")
    return values / norms


def _normalized_mean(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.mean(values[-min(3, len(values)) :], axis=0)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError(f"{name} has an unstable mean direction")
    return vector / norm


def _linear_velocity(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.polyfit(times, values[:, axis], 1)[0] for axis in range(3)],
        dtype=np.float64,
    )


def _quadratic_acceleration(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    acceleration = np.zeros(3, dtype=np.float64)
    if len(times) >= 3:
        for axis in range(3):
            coefficients = np.polyfit(times, values[:, axis], 2)
            acceleration[axis] = 2.0 * coefficients[0]
    return acceleration


def extract_ball_racket_motion_features(
    frames: np.ndarray,
    frame_ids: Optional[np.ndarray] = None,
    max_ball_points: int = DEFAULT_MAX_BALL_POINTS,
    min_ball_points: int = 4,
    min_racket_points: int = 3,
) -> BallRacketMotionFeatures:
    """Build smoothed racket geometry/dynamics and ball-racket interaction features."""

    frames = np.asarray(frames, dtype=np.float64)
    ball_motion = extract_ball_motion_features(
        frames,
        frame_ids=frame_ids,
        max_ball_points=max_ball_points,
        min_ball_points=min_ball_points,
    )
    rackets = frames[ball_motion.selected_indices, RACKET_SLICE].reshape(-1, 4, 3)
    finite = np.all(np.isfinite(rackets), axis=(1, 2))
    short_vectors = rackets[:, 1] - rackets[:, 0]
    long_vectors = rackets[:, 3] - rackets[:, 2]
    short_lengths = np.linalg.norm(short_vectors, axis=1)
    long_lengths = np.linalg.norm(long_vectors, axis=1)
    valid = (
        finite
        & (short_lengths >= 0.5)
        & (short_lengths <= 120.0)
        & (long_lengths >= 0.5)
        & (long_lengths <= 150.0)
    )
    if int(np.sum(valid)) < min_racket_points:
        raise ValueError(
            f"ball+racket model requires at least {min_racket_points} valid racket frames "
            f"among selected ball points, got {int(np.sum(valid))}"
        )
    if not bool(valid[-1]):
        raise ValueError("the last selected ball point has no valid racket geometry")

    rackets = rackets[valid]
    times = ball_motion.selected_frame_ids[valid]
    times = times - times[-1]
    aligned_ball = ball_motion.selected_ball[valid]
    short_vectors = short_vectors[valid]
    long_vectors = long_vectors[valid]
    short_lengths = short_lengths[valid]
    long_lengths = long_lengths[valid]
    centers = np.mean(rackets, axis=1)

    short_units = _normalized_rows(short_vectors, "racket short axis")
    long_units = _normalized_rows(long_vectors, "racket long axis")
    normals = _normalized_rows(
        np.cross(short_units, long_units), "racket face normal"
    )
    last_short_unit = _normalized_mean(short_units, "racket short axis")
    last_long_unit = _normalized_mean(long_units, "racket long axis")
    last_normal = _normalized_mean(normals, "racket face normal")

    center_velocity = _linear_velocity(times, centers)
    recent_count = min(3, len(times))
    recent_velocity = _linear_velocity(
        times[-recent_count:], centers[-recent_count:]
    )
    center_acceleration = _quadratic_acceleration(times, centers)
    short_angular_velocity = _linear_velocity(times, short_units)
    long_angular_velocity = _linear_velocity(times, long_units)

    ball_velocity = np.asarray(ball_motion.features[3:6], dtype=np.float64)
    relative_position = ball_motion.last_ball - centers[-1]
    relative_velocity = ball_velocity - center_velocity
    orientation_basis = np.stack(
        [last_short_unit, last_long_unit, last_normal], axis=0
    )
    ball_velocity_projection = orientation_basis @ ball_velocity
    relative_velocity_projection = orientation_basis @ relative_velocity
    aligned_distances = np.linalg.norm(aligned_ball - centers, axis=1)

    racket_features = np.concatenate(
        [
            centers[-1],
            center_velocity,
            recent_velocity,
            center_acceleration,
            last_short_unit,
            last_long_unit,
            last_normal,
            np.asarray([short_lengths[-1], long_lengths[-1]]),
            short_angular_velocity,
            long_angular_velocity,
            relative_position,
            relative_velocity,
            ball_velocity_projection,
            relative_velocity_projection,
            np.asarray([aligned_distances[-1], np.min(aligned_distances)]),
        ]
    ).astype(np.float32)
    if len(racket_features) != len(RACKET_FEATURE_NAMES):
        raise AssertionError(
            f"racket feature mismatch: {len(racket_features)} != {len(RACKET_FEATURE_NAMES)}"
        )
    if not np.all(np.isfinite(racket_features)):
        raise ValueError("ball+racket motion features contain non-finite values")
    combined = np.concatenate([ball_motion.features, racket_features]).astype(
        np.float32
    )
    return BallRacketMotionFeatures(
        ball_motion=ball_motion,
        ball_features=ball_motion.features,
        racket_features=racket_features,
        features=combined,
        valid_racket_count=int(np.sum(valid)),
        last_racket_center=centers[-1].astype(np.float64),
    )


class AfterBallRacketTrendModel(nn.Module):
    """Ball base prediction plus a gated, explicit-racket residual correction."""

    def __init__(
        self,
        ball_input_dim: int = len(BALL_FEATURE_NAMES),
        racket_input_dim: int = len(RACKET_FEATURE_NAMES),
        hidden_dim: int = 128,
        dropout: float = 0.08,
    ):
        super().__init__()

        def encoder(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

        self.ball_input_dim = int(ball_input_dim)
        self.racket_input_dim = int(racket_input_dim)
        self.ball_encoder = encoder(self.ball_input_dim)
        self.racket_encoder = encoder(self.racket_input_dim)
        self.ball_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        fusion_dim = hidden_dim * 2
        self.racket_gate = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),
        )
        self.racket_correction = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward_components(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        ball = features[:, : self.ball_input_dim]
        racket = features[
            :, self.ball_input_dim : self.ball_input_dim + self.racket_input_dim
        ]
        ball_embedding = self.ball_encoder(ball)
        racket_embedding = self.racket_encoder(racket)
        fused = torch.cat([ball_embedding, racket_embedding], dim=1)
        ball_prediction = self.ball_head(ball_embedding)
        gate = self.racket_gate(fused)
        correction = self.racket_correction(fused)
        prediction = ball_prediction + gate * correction
        return {
            "prediction": prediction,
            "ball_prediction": ball_prediction,
            "racket_gate": gate,
            "racket_correction": correction,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_components(features)["prediction"]


def is_after_ball_racket_checkpoint(checkpoint: Any) -> bool:
    return isinstance(checkpoint, dict) and checkpoint.get("format") == CHECKPOINT_FORMAT


class AfterBallRacketPredictor:
    def __init__(self, model_path: Path, device: Optional[str] = None):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ball+racket after model does not exist: {model_path}")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        if not is_after_ball_racket_checkpoint(checkpoint):
            raise ValueError(f"not an {CHECKPOINT_FORMAT} checkpoint: {model_path}")

        def as_numpy(value: Any) -> np.ndarray:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        self.max_ball_points = int(
            checkpoint.get("max_ball_points", DEFAULT_MAX_BALL_POINTS)
        )
        self.feature_mean = as_numpy(checkpoint["feature_mean"])
        self.feature_std = as_numpy(checkpoint["feature_std"])
        self.target_mean = as_numpy(checkpoint["target_mean"])
        self.target_std = as_numpy(checkpoint["target_std"])
        self.model = AfterBallRacketTrendModel(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint = checkpoint
        self.model_path = model_path

    def predict_with_details(
        self,
        pose_window: np.ndarray,
        frame_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        motion = extract_ball_racket_motion_features(
            pose_window,
            frame_ids=frame_ids,
            max_ball_points=self.max_ball_points,
        )
        normalized = (motion.features - self.feature_mean) / self.feature_std
        tensor = torch.as_tensor(
            normalized[None, :], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            components = self.model.forward_components(tensor)
        normalized_target = components["prediction"][0].cpu().numpy()
        target = normalized_target * self.target_std + self.target_mean
        landing_xy = reconstruct_landing_xy(target, motion.ball_motion)
        return {
            "landing_xyz": np.asarray(
                [landing_xy[0], landing_xy[1], 0.0], dtype=np.float64
            ),
            "motion": motion,
            "target": target.astype(np.float64),
            "normalized_target": normalized_target.astype(np.float64),
            "ball_prediction": components["ball_prediction"][0]
            .cpu()
            .numpy()
            .astype(np.float64),
            "racket_gate": components["racket_gate"][0]
            .cpu()
            .numpy()
            .astype(np.float64),
            "racket_correction": components["racket_correction"][0]
            .cpu()
            .numpy()
            .astype(np.float64),
        }

    def predict(
        self,
        pose_window: np.ndarray,
        frame_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.predict_with_details(pose_window, frame_ids)["landing_xyz"]

