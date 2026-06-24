from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, brentq, curve_fit


FEATURE_NAMES = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
)
TARGET_NAMES = ("c1", "c2", "c3")
DEFAULT_STATE_WINDOW_SIZE = 20
DEFAULT_MLP_HIDDEN_DIMS = (128, 128, 64)
MAX_FUNCTION_EVALS = 50_000
MAX_EXP_ARG = 700.0
NEGATIVE_BOUNDS = (-0.1, -1e-8)


@dataclass
class TrajectorySample:
    frame_ids: np.ndarray
    points: np.ndarray
    landing_frame: int | None
    landing_xyz: np.ndarray
    path: Path | None = None
    has_explicit_frame_ids: bool = True
    landing_frame_estimated: bool = False


def safe_exp(value):
    return np.exp(np.clip(value, -MAX_EXP_ARG, MAX_EXP_ARG))


def x_model(t_ms, a1, b1, c1):
    return a1 * safe_exp(b1 * t_ms) + c1


def y_model(t_ms, a2, b2, c2):
    return a2 * safe_exp(-0.002 * t_ms) + b2 * t_ms + c2


def z_model(t_ms, a3, b3, c3):
    return a3 * safe_exp(b3 * t_ms) - 0.79 * t_ms + c3


def _parse_line(raw_line, line_number):
    text = raw_line.strip()
    if not text:
        return None

    frame_id = None
    if ":" in text:
        frame_text, text = text.split(":", 1)
        try:
            frame_id = int(frame_text.strip())
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: invalid frame id {frame_text!r}") from exc

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: expected three comma-separated coordinates, got {len(parts)}"
        )
    try:
        point = np.asarray([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid coordinate values") from exc
    if not np.all(np.isfinite(point)):
        raise ValueError(f"Line {line_number}: coordinates must be finite")
    return frame_id, point


def load_trajectory_file(path, require_landing_frame=False, skip_zero_observations=False):
    path = Path(path)
    parsed = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        item = _parse_line(raw_line, line_number)
        if item is not None:
            parsed.append(item)

    if len(parsed) < 4:
        raise ValueError(f"{path}: need at least three observations and one landing label")

    observation_items = parsed[:-1]
    label_frame, landing_xyz = parsed[-1]
    has_observation_frames = any(frame_id is not None for frame_id, _ in observation_items)
    has_missing_observation_frames = any(frame_id is None for frame_id, _ in observation_items)
    if has_observation_frames and has_missing_observation_frames:
        raise ValueError(f"{path}: observation frame ids must be present on every line or none")

    if has_observation_frames:
        frame_ids = np.asarray([frame_id for frame_id, _ in observation_items], dtype=np.int64)
    else:
        frame_ids = np.arange(len(observation_items), dtype=np.int64)

    points = np.stack([point for _, point in observation_items], axis=0)
    if skip_zero_observations:
        valid = ~np.all(np.isclose(points, 0.0, atol=1e-12), axis=1)
        frame_ids = frame_ids[valid]
        points = points[valid]

    if len(points) < 3:
        raise ValueError(f"{path}: fewer than three valid observations remain")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError(f"{path}: observation frame ids must be strictly increasing")
    if require_landing_frame and not has_observation_frames:
        raise ValueError(f"{path}: every training observation must include frame_id:x,y,z")
    if require_landing_frame and label_frame is None:
        raise ValueError(f"{path}: the landing label must include landing_frame:x,y,z")
    if label_frame is not None and label_frame <= int(frame_ids[-1]):
        raise ValueError(
            f"{path}: landing frame {label_frame} must be later than last observation "
            f"frame {int(frame_ids[-1])}"
        )

    return TrajectorySample(
        frame_ids=frame_ids,
        points=points,
        landing_frame=label_frame,
        landing_xyz=landing_xyz,
        path=path,
        has_explicit_frame_ids=has_observation_frames,
    )


def estimate_landing_frame(sample, slope_window=50):
    if sample.landing_frame is not None:
        return sample
    if slope_window < 2:
        raise ValueError("slope_window must be at least 2")

    count = min(int(slope_window), len(sample.points))
    recent_frames = sample.frame_ids[-count:].astype(np.float64)
    recent_z = sample.points[-count:, 2]
    slope_per_frame = float(np.polyfit(recent_frames, recent_z, deg=1)[0])
    if not np.isfinite(slope_per_frame) or slope_per_frame >= -1e-6:
        raise ValueError(
            "Cannot estimate landing frame because the recent z trajectory is not descending"
        )

    remaining_height = float(sample.points[-1, 2] - sample.landing_xyz[2])
    if remaining_height <= 0.0:
        horizon_frames = 1
    else:
        horizon_frames = max(1, int(round(remaining_height / -slope_per_frame)))
    return replace(
        sample,
        landing_frame=int(sample.frame_ids[-1]) + horizon_frames,
        landing_frame_estimated=True,
    )


def frame_times_ms(frame_ids, fps, origin_frame=None):
    if fps <= 0:
        raise ValueError("FPS must be positive")
    frame_ids = np.asarray(frame_ids, dtype=np.float64)
    origin = float(frame_ids[0] if origin_frame is None else origin_frame)
    return (frame_ids - origin) / float(fps) * 1000.0


def extract_state_features(points, frame_ids, fps, window_size=DEFAULT_STATE_WINDOW_SIZE):
    points = np.asarray(points, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if len(points) != len(frame_ids):
        raise ValueError("points and frame_ids must have equal length")
    if len(points) < window_size:
        raise ValueError(f"Need at least {window_size} points, got {len(points)}")

    recent_points = points[-window_size:]
    recent_frames = frame_ids[-window_size:]
    if np.any(np.diff(recent_frames) <= 0):
        raise ValueError("Feature frame ids must be strictly increasing")

    # Seconds keep the polynomial system well-conditioned; StandardScaler handles magnitudes.
    times_s = (recent_frames.astype(np.float64) - float(recent_frames[-1])) / float(fps)
    features = []
    for axis in range(3):
        quadratic, linear, intercept = np.polyfit(times_s, recent_points[:, axis], deg=2)
        features.extend((intercept, linear, 2.0 * quadratic))

    # Reorder per derivative group: xyz, velocity xyz, acceleration xyz.
    per_axis = np.asarray(features, dtype=np.float64).reshape(3, 3)
    ordered = np.concatenate([per_axis[:, 0], per_axis[:, 1], per_axis[:, 2]])
    if not np.all(np.isfinite(ordered)):
        raise ValueError("Extracted state features contain non-finite values")
    return ordered


def _bounded_curve_fit(model, times_ms, values, initial, lower, upper):
    with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", OptimizeWarning)
        params, _ = curve_fit(
            model,
            times_ms,
            values,
            p0=np.clip(initial, lower, upper),
            bounds=(lower, upper),
            maxfev=MAX_FUNCTION_EVALS,
        )
    if not np.all(np.isfinite(params)):
        raise ValueError("Fitted parameters contain non-finite values")
    return params


def fit_complete_trajectory_labels(sample, fps):
    if sample.landing_frame is None:
        raise ValueError("A landing frame is required to fit c labels")

    origin = int(sample.frame_ids[0])
    observed_times = frame_times_ms(sample.frame_ids, fps, origin)
    landing_time = frame_times_ms([sample.landing_frame], fps, origin)[0]
    if landing_time <= observed_times[-1]:
        raise ValueError("Landing time must be later than the final observation")

    times_ms = np.concatenate([observed_times, [landing_time]])
    values = np.vstack([sample.points, sample.landing_xyz])
    x_values, y_values, z_values = values.T

    x_initial = (
        float(x_values[0] - x_values[-1]),
        -0.002,
        float(x_values[-1]),
    )
    z_initial = (
        float(z_values[0] - z_values[-1] + 0.79 * landing_time),
        -0.002,
        float(z_values[0]),
    )
    x_params = _bounded_curve_fit(
        x_model,
        times_ms,
        x_values,
        x_initial,
        np.asarray([-1e6, NEGATIVE_BOUNDS[0], -1e5]),
        np.asarray([1e6, NEGATIVE_BOUNDS[1], 1e5]),
    )
    y_design = np.column_stack([safe_exp(-0.002 * times_ms), times_ms, np.ones_like(times_ms)])
    y_params, *_ = np.linalg.lstsq(y_design, y_values, rcond=None)
    z_params = _bounded_curve_fit(
        z_model,
        times_ms,
        z_values,
        z_initial,
        np.asarray([-1e6, NEGATIVE_BOUNDS[0], -1e5]),
        np.asarray([1e6, NEGATIVE_BOUNDS[1], 1e5]),
    )

    all_params = {
        "times_ms": observed_times,
        "x": x_params,
        "y": y_params,
        "z": z_params,
    }
    c_values = np.asarray([x_params[2], y_params[2], z_params[2]], dtype=np.float64)
    if not np.all(np.isfinite(c_values)):
        raise ValueError("c labels contain non-finite values")
    return c_values, all_params


def fit_all_parameters(points, frame_ids, fps):
    points = np.asarray(points, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if len(points) != len(frame_ids):
        raise ValueError("points and frame_ids must have equal length")
    if len(points) < 3:
        raise ValueError("Need at least three points to fit trajectory parameters")

    times_ms = frame_times_ms(frame_ids, fps)
    x_values, y_values, z_values = points.T
    last_t = times_ms[-1]
    x_params = _bounded_curve_fit(
        x_model,
        times_ms,
        x_values,
        (float(x_values[0] - x_values[-1]), -0.002, float(x_values[-1])),
        np.asarray([-1e6, NEGATIVE_BOUNDS[0], -1e5]),
        np.asarray([1e6, NEGATIVE_BOUNDS[1], 1e5]),
    )
    y_design = np.column_stack(
        [safe_exp(-0.002 * times_ms), times_ms, np.ones_like(times_ms)]
    )
    y_params, *_ = np.linalg.lstsq(y_design, y_values, rcond=None)
    z_params = _bounded_curve_fit(
        z_model,
        times_ms,
        z_values,
        (
            float(z_values[0] - z_values[-1] + 0.79 * last_t),
            -0.002,
            float(z_values[0]),
        ),
        np.asarray([-1e6, NEGATIVE_BOUNDS[0], -1e5]),
        np.asarray([1e6, NEGATIVE_BOUNDS[1], 1e5]),
    )
    return {
        "times_ms": times_ms,
        "x": x_params,
        "y": y_params,
        "z": z_params,
    }


def fit_remaining_parameters(points, frame_ids, fps, c_values):
    points = np.asarray(points, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    c1, c2, c3 = np.asarray(c_values, dtype=np.float64)
    times_ms = frame_times_ms(frame_ids, fps)
    x_values, y_values, z_values = points.T

    def x_fixed_c(t_ms, a1, b1):
        return x_model(t_ms, a1, b1, c1)

    def z_fixed_c(t_ms, a3, b3):
        return z_model(t_ms, a3, b3, c3)

    x_params_ab = _bounded_curve_fit(
        x_fixed_c,
        times_ms,
        x_values,
        (float(x_values[0] - c1), -0.002),
        np.asarray([-1e6, NEGATIVE_BOUNDS[0]]),
        np.asarray([1e6, NEGATIVE_BOUNDS[1]]),
    )
    y_design = np.column_stack([safe_exp(-0.002 * times_ms), times_ms])
    y_params_ab, *_ = np.linalg.lstsq(y_design, y_values - c2, rcond=None)
    z_params_ab = _bounded_curve_fit(
        z_fixed_c,
        times_ms,
        z_values,
        (float(z_values[0] - c3), -0.002),
        np.asarray([-1e6, NEGATIVE_BOUNDS[0]]),
        np.asarray([1e6, NEGATIVE_BOUNDS[1]]),
    )

    return {
        "times_ms": times_ms,
        "x": np.asarray([x_params_ab[0], x_params_ab[1], c1]),
        "y": np.asarray([y_params_ab[0], y_params_ab[1], c2]),
        "z": np.asarray([z_params_ab[0], z_params_ab[1], c3]),
    }


def evaluate_position(time_ms, params):
    return np.asarray(
        [
            x_model(time_ms, *params["x"]),
            y_model(time_ms, *params["y"]),
            z_model(time_ms, *params["z"]),
        ],
        dtype=np.float64,
    )


def solve_landing_time(params, landing_z=0.0):
    start = float(params["times_ms"][-1])

    def height_delta(time_ms):
        return float(z_model(time_ms, *params["z"]) - landing_z)

    start_delta = height_delta(start)
    if np.isclose(start_delta, 0.0, atol=1e-9):
        return start

    step = max(1000.0, start if start > 0 else 1000.0)
    end = start + step
    for _ in range(20):
        end_delta = height_delta(end)
        if np.isfinite(end_delta) and np.sign(start_delta) != np.sign(end_delta):
            return brentq(height_delta, start, end)
        step *= 2.0
        end = start + step
    raise ValueError(f"Could not bracket z={landing_z} after t={start:.3f} ms")


def build_c_parameter_mlp(
    input_dim=9,
    output_dim=3,
    dropout=0.1,
    hidden_dims=DEFAULT_MLP_HIDDEN_DIMS,
):
    import torch.nn as nn

    hidden_dims = tuple(int(value) for value in hidden_dims)
    if not hidden_dims or any(value <= 0 for value in hidden_dims):
        raise ValueError("hidden_dims must contain positive layer widths")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    layers = []
    previous_dim = input_dim
    for index, hidden_dim in enumerate(hidden_dims):
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
        if index < len(hidden_dims) - 1:
            layers.append(nn.Dropout(dropout))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class CParameterPredictor:
    def __init__(self, checkpoint_path, device="cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for --parameter-mode mlp. "
                "Use the project's PyTorch interpreter, for example: py -3.11."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float64)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float64)
        self.target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float64)
        self.target_std = np.asarray(checkpoint["target_std"], dtype=np.float64)
        self.feature_names = tuple(checkpoint["feature_names"])
        self.state_window_size = int(checkpoint["state_window_size"])
        self.training_fps = float(checkpoint["fps"])
        self.splits = checkpoint.get("splits", {})
        self.training_data_format = checkpoint.get("training_data_format")
        self.hidden_dims = tuple(
            int(value)
            for value in checkpoint.get("hidden_dims", DEFAULT_MLP_HIDDEN_DIMS)
        )
        if self.feature_names != FEATURE_NAMES:
            raise ValueError(
                f"Checkpoint feature order {self.feature_names!r} does not match "
                f"expected {FEATURE_NAMES!r}"
            )
        if np.any(self.feature_std <= 0.0) or np.any(self.target_std <= 0.0):
            raise ValueError("Checkpoint normalization standard deviations must be positive")

        self.model = build_c_parameter_mlp(
            input_dim=len(self.feature_names),
            output_dim=len(TARGET_NAMES),
            dropout=float(checkpoint.get("dropout", 0.1)),
            hidden_dims=self.hidden_dims,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.shape != self.feature_mean.shape:
            raise ValueError(
                f"Expected feature shape {self.feature_mean.shape}, got {features.shape}"
            )
        normalized = (features - self.feature_mean) / self.feature_std
        tensor = self.torch.as_tensor(normalized, dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            prediction = self.model(tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
        c_values = prediction * self.target_std + self.target_mean
        if not np.all(np.isfinite(c_values)):
            raise ValueError("MLP predicted non-finite c parameters")
        return c_values.astype(np.float64)


def predict_hybrid_parameters(
    points,
    frame_ids,
    fps,
    predictor,
    state_window_size=DEFAULT_STATE_WINDOW_SIZE,
):
    if state_window_size != predictor.state_window_size:
        raise ValueError(
            f"Checkpoint expects state_window_size={predictor.state_window_size}, "
            f"got {state_window_size}"
        )
    features = extract_state_features(points, frame_ids, fps, state_window_size)
    c_values = predictor.predict(features)
    params = fit_remaining_parameters(points, frame_ids, fps, c_values)
    return params, c_values, features
