"""Explicit racket-swing and body-biomechanics model for before-hit prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT = "before_pose_racket_trend_v1"
RACKET_SLICE = slice(17 * 3, 21 * 3)
DEFAULT_MAX_RACKET_POINTS = 12

RACKET_FEATURE_NAMES = (
    "last_racket_center_x",
    "last_racket_center_y",
    "last_racket_center_z",
    "racket_velocity_x",
    "racket_velocity_y",
    "racket_velocity_z",
    "racket_recent_velocity_x",
    "racket_recent_velocity_y",
    "racket_recent_velocity_z",
    "racket_acceleration_x",
    "racket_acceleration_y",
    "racket_acceleration_z",
    "racket_observed_displacement_x",
    "racket_observed_displacement_y",
    "racket_observed_displacement_z",
    "racket_path_length_xy",
    "racket_path_straightness_xy",
    "valid_racket_point_count",
    "racket_frame_span",
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
    "racket_face_angular_velocity_x",
    "racket_face_angular_velocity_y",
    "racket_face_angular_velocity_z",
    "racket_speed",
    "racket_recent_speed",
)

BODY_POINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
BODY_POINT_INDICES = np.asarray([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
BODY_FEATURE_NAMES = tuple(
    [f"{name}_relative_{axis}" for name in BODY_POINT_NAMES for axis in "xyz"]
    + [
        f"{name}_relative_velocity_{axis}"
        for name in BODY_POINT_NAMES
        for axis in "xyz"
    ]
    + [f"shoulder_axis_{axis}" for axis in "xyz"]
    + [f"hip_axis_{axis}" for axis in "xyz"]
    + [f"torso_axis_{axis}" for axis in "xyz"]
    + [f"torso_normal_{axis}" for axis in "xyz"]
    + [
        "shoulder_width_normalized",
        "hip_width_normalized",
        "torso_length_normalized",
        "body_scale",
    ]
    + [f"pelvis_velocity_{axis}" for axis in "xyz"]
    + [f"shoulder_relative_velocity_{axis}" for axis in "xyz"]
)


@dataclass
class BeforePoseRacketFeatures:
    racket_features: np.ndarray
    body_features: np.ndarray
    features: np.ndarray
    last_racket_center: np.ndarray
    direction_xy: np.ndarray
    perpendicular_xy: np.ndarray
    selected_indices: np.ndarray
    selected_frame_ids: np.ndarray
    selected_rackets: np.ndarray


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


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError(f"{name} contains a degenerate vector")
    return values / norms


def _normalized_mean(values: np.ndarray, name: str) -> np.ndarray:
    value = np.mean(values[-min(3, len(values)) :], axis=0)
    norm = float(np.linalg.norm(value))
    if norm < 1e-6:
        raise ValueError(f"{name} has an unstable mean direction")
    return value / norm


def extract_before_pose_racket_features(
    frames: np.ndarray,
    frame_ids: Optional[np.ndarray] = None,
    max_racket_points: int = DEFAULT_MAX_RACKET_POINTS,
    min_racket_points: int = 6,
) -> BeforePoseRacketFeatures:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < 63:
        raise ValueError(
            f"before pose+racket model expects shape (frames, >=63), got {frames.shape}"
        )
    if frame_ids is None:
        frame_ids = np.arange(len(frames), dtype=np.float64)
    else:
        frame_ids = np.asarray(frame_ids, dtype=np.float64)
        if frame_ids.shape != (len(frames),):
            raise ValueError("frame_ids shape does not match frames")

    rackets = frames[:, RACKET_SLICE].reshape(-1, 4, 3)
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
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < min_racket_points:
        raise ValueError(
            f"before model requires at least {min_racket_points} valid racket frames, "
            f"got {len(valid_indices)}"
        )
    if int(valid_indices[-1]) < len(frames) - 3:
        raise ValueError("last valid racket frame is too far from the before-window tail")
    selected_indices = valid_indices[-max_racket_points:]
    selected_rackets = rackets[selected_indices]
    selected_frame_ids = frame_ids[selected_indices]
    times = selected_frame_ids - selected_frame_ids[-1]
    centers = np.mean(selected_rackets, axis=1)

    short_vectors = short_vectors[selected_indices]
    long_vectors = long_vectors[selected_indices]
    short_lengths = short_lengths[selected_indices]
    long_lengths = long_lengths[selected_indices]
    short_units = _normalized_rows(short_vectors, "racket short axis")
    long_units = _normalized_rows(long_vectors, "racket long axis")
    normals = _normalized_rows(
        np.cross(short_units, long_units), "racket face normal"
    )
    last_short = _normalized_mean(short_units, "racket short axis")
    last_long = _normalized_mean(long_units, "racket long axis")
    last_normal = _normalized_mean(normals, "racket face normal")

    velocity = _linear_velocity(times, centers)
    recent_count = min(3, len(times))
    recent_velocity = _linear_velocity(
        times[-recent_count:], centers[-recent_count:]
    )
    acceleration = _quadratic_acceleration(times, centers)
    direction_norm = float(np.linalg.norm(recent_velocity[:2]))
    if direction_norm < 1e-6:
        raise ValueError("recent racket horizontal speed is too small")
    direction_xy = recent_velocity[:2] / direction_norm
    perpendicular_xy = np.asarray([-direction_xy[1], direction_xy[0]])
    displacement = centers[-1] - centers[0]
    path_length = float(np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1).sum())
    straightness = float(np.linalg.norm(displacement[:2])) / max(path_length, 1e-6)
    short_angular_velocity = _linear_velocity(times, short_units)
    long_angular_velocity = _linear_velocity(times, long_units)
    normal_angular_velocity = _linear_velocity(times, normals)

    racket_features = np.concatenate(
        [
            centers[-1],
            velocity,
            recent_velocity,
            acceleration,
            displacement,
            np.asarray(
                [
                    path_length,
                    straightness,
                    float(len(selected_indices)),
                    float(selected_frame_ids[-1] - selected_frame_ids[0]),
                ]
            ),
            last_short,
            last_long,
            last_normal,
            np.asarray([short_lengths[-1], long_lengths[-1]]),
            short_angular_velocity,
            long_angular_velocity,
            normal_angular_velocity,
            np.asarray(
                [np.linalg.norm(velocity), np.linalg.norm(recent_velocity)]
            ),
        ]
    ).astype(np.float32)

    body = frames[selected_indices, : 17 * 3].reshape(-1, 17, 3)
    required_body = body[:, BODY_POINT_INDICES]
    body_valid = np.all(np.isfinite(required_body), axis=(1, 2)) & ~np.all(
        np.isclose(required_body, 0.0, atol=1e-9), axis=(1, 2)
    )
    if int(np.sum(body_valid)) < min_racket_points:
        raise ValueError(
            f"before model requires at least {min_racket_points} valid body frames, "
            f"got {int(np.sum(body_valid))}"
        )
    body = body[body_valid]
    body_times = times[body_valid]
    points = body[:, BODY_POINT_INDICES]
    pelvis = np.mean(body[:, [11, 12]], axis=1)
    shoulder_center = np.mean(body[:, [5, 6]], axis=1)
    shoulder_vectors = body[:, 6] - body[:, 5]
    hip_vectors = body[:, 12] - body[:, 11]
    torso_vectors = shoulder_center - pelvis
    shoulder_units = _normalized_rows(shoulder_vectors, "shoulder axis")
    hip_units = _normalized_rows(hip_vectors, "hip axis")
    torso_units = _normalized_rows(torso_vectors, "torso axis")
    torso_normals = _normalized_rows(
        np.cross(shoulder_units, torso_units), "torso normal"
    )
    shoulder_width = np.linalg.norm(shoulder_vectors[-1])
    hip_width = np.linalg.norm(hip_vectors[-1])
    torso_length = np.linalg.norm(torso_vectors[-1])
    body_scale = float(np.mean([shoulder_width, hip_width, torso_length]))
    if body_scale < 1e-6:
        raise ValueError("body scale is degenerate")
    pelvis_velocity = _linear_velocity(body_times, pelvis)
    shoulder_velocity = _linear_velocity(body_times, shoulder_center)
    point_velocities = np.stack(
        [
            _linear_velocity(body_times, points[:, point_index])
            for point_index in range(points.shape[1])
        ],
        axis=0,
    )
    relative_positions = (points[-1] - pelvis[-1]) / body_scale
    relative_velocities = (point_velocities - pelvis_velocity) / body_scale
    body_features = np.concatenate(
        [
            relative_positions.reshape(-1),
            relative_velocities.reshape(-1),
            _normalized_mean(shoulder_units, "shoulder axis"),
            _normalized_mean(hip_units, "hip axis"),
            _normalized_mean(torso_units, "torso axis"),
            _normalized_mean(torso_normals, "torso normal"),
            np.asarray(
                [
                    shoulder_width / body_scale,
                    hip_width / body_scale,
                    torso_length / body_scale,
                    body_scale,
                ]
            ),
            pelvis_velocity,
            (shoulder_velocity - pelvis_velocity) / body_scale,
        ]
    ).astype(np.float32)
    if len(racket_features) != len(RACKET_FEATURE_NAMES):
        raise AssertionError(
            f"racket feature mismatch: {len(racket_features)} != {len(RACKET_FEATURE_NAMES)}"
        )
    if len(body_features) != len(BODY_FEATURE_NAMES):
        raise AssertionError(
            f"body feature mismatch: {len(body_features)} != {len(BODY_FEATURE_NAMES)}"
        )
    features = np.concatenate([racket_features, body_features]).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("before pose+racket features contain non-finite values")
    return BeforePoseRacketFeatures(
        racket_features=racket_features,
        body_features=body_features,
        features=features,
        last_racket_center=centers[-1].astype(np.float64),
        direction_xy=direction_xy.astype(np.float64),
        perpendicular_xy=perpendicular_xy.astype(np.float64),
        selected_indices=selected_indices.astype(np.int64),
        selected_frame_ids=selected_frame_ids.astype(np.float64),
        selected_rackets=selected_rackets.astype(np.float64),
    )


def before_motion_target(
    landing_xy: np.ndarray, motion: BeforePoseRacketFeatures
) -> np.ndarray:
    delta = np.asarray(landing_xy, dtype=np.float64) - motion.last_racket_center[:2]
    forward = float(np.dot(delta, motion.direction_xy))
    lateral = float(np.dot(delta, motion.perpendicular_xy))
    if forward <= 0:
        raise ValueError(
            f"landing is not forward along the recent racket swing: {forward:.3f} cm"
        )
    return np.asarray([np.log1p(forward), lateral], dtype=np.float32)


def reconstruct_before_landing_xy(
    target: np.ndarray, motion: BeforePoseRacketFeatures
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    forward = float(np.expm1(np.clip(target[0], 0.0, np.log1p(2500.0))))
    lateral = float(np.clip(target[1], -1500.0, 1500.0))
    return (
        motion.last_racket_center[:2]
        + forward * motion.direction_xy
        + lateral * motion.perpendicular_xy
    )


class BeforePoseRacketTrendModel(nn.Module):
    """Racket-swing base prediction plus a gated body-pose correction."""

    def __init__(
        self,
        racket_input_dim: int = len(RACKET_FEATURE_NAMES),
        body_input_dim: int = len(BODY_FEATURE_NAMES),
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

        self.racket_input_dim = int(racket_input_dim)
        self.body_input_dim = int(body_input_dim)
        self.racket_encoder = encoder(self.racket_input_dim)
        self.body_encoder = encoder(self.body_input_dim)
        self.racket_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        fusion_dim = hidden_dim * 2
        self.body_gate = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),
        )
        self.body_correction = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward_components(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        racket = features[:, : self.racket_input_dim]
        body = features[
            :, self.racket_input_dim : self.racket_input_dim + self.body_input_dim
        ]
        racket_embedding = self.racket_encoder(racket)
        body_embedding = self.body_encoder(body)
        fused = torch.cat([racket_embedding, body_embedding], dim=1)
        racket_prediction = self.racket_head(racket_embedding)
        gate = self.body_gate(fused)
        correction = self.body_correction(fused)
        prediction = racket_prediction + gate * correction
        return {
            "prediction": prediction,
            "racket_prediction": racket_prediction,
            "body_gate": gate,
            "body_correction": correction,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_components(features)["prediction"]


def is_before_pose_racket_checkpoint(checkpoint: Any) -> bool:
    return isinstance(checkpoint, dict) and checkpoint.get("format") == CHECKPOINT_FORMAT


class BeforePoseRacketPredictor:
    def __init__(self, model_path: Path, device: Optional[str] = None):
        model_path = Path(model_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        if not is_before_pose_racket_checkpoint(checkpoint):
            raise ValueError(f"not a {CHECKPOINT_FORMAT} checkpoint: {model_path}")

        def as_numpy(value: Any) -> np.ndarray:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        self.max_racket_points = int(
            checkpoint.get("max_racket_points", DEFAULT_MAX_RACKET_POINTS)
        )
        self.feature_mean = as_numpy(checkpoint["feature_mean"])
        self.feature_std = as_numpy(checkpoint["feature_std"])
        self.target_mean = as_numpy(checkpoint["target_mean"])
        self.target_std = as_numpy(checkpoint["target_std"])
        self.model = BeforePoseRacketTrendModel(**checkpoint["model_config"])
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
        motion = extract_before_pose_racket_features(
            pose_window,
            frame_ids=frame_ids,
            max_racket_points=self.max_racket_points,
        )
        normalized = (motion.features - self.feature_mean) / self.feature_std
        tensor = torch.as_tensor(
            normalized[None, :], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            components = self.model.forward_components(tensor)
        normalized_target = components["prediction"][0].cpu().numpy()
        target = normalized_target * self.target_std + self.target_mean
        landing_xy = reconstruct_before_landing_xy(target, motion)
        return {
            "landing_xyz": np.asarray(
                [landing_xy[0], landing_xy[1], 0.0], dtype=np.float64
            ),
            "motion": motion,
            "target": target.astype(np.float64),
            "racket_prediction": components["racket_prediction"][0]
            .cpu()
            .numpy()
            .astype(np.float64),
            "body_gate": components["body_gate"][0]
            .cpu()
            .numpy()
            .astype(np.float64),
            "body_correction": components["body_correction"][0]
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

