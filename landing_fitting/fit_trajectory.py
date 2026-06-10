import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, brentq, curve_fit


DEFAULT_INPUT = Path(__file__).with_name("trajectory.txt")
DEFAULT_FPS = 300.0
DEFAULT_LANDING_Z = 0.0
DEFAULT_WINDOW_SIZE = 40
DEFAULT_PLOT_OUTPUT = Path(__file__).with_name("sliding_window_predictions.png")
DEFAULT_CSV_OUTPUT = Path(__file__).with_name("sliding_window_predictions.csv")
DEFAULT_GIF_OUTPUT = Path(__file__).with_name("sliding_window_predictions.gif")
DEFAULT_GIF_FPS = 6
MAX_FUNCTION_EVALS = 50_000
MAX_EXP_ARG = 700.0

COURT_LENGTH = 1340.0
COURT_WIDTH = 670.0
COURT_HALF_WIDTH = COURT_WIDTH / 2.0
COURT_NET_X = COURT_LENGTH / 2.0
COURT_SHORT_SERVICE_DISTANCE = 198.0
COURT_DOUBLES_LONG_SERVICE_DISTANCE = 76.0
COURT_SINGLES_HALF_WIDTH = COURT_HALF_WIDTH * (518.0 / 610.0)


def safe_exp(value):
    return np.exp(np.clip(value, -MAX_EXP_ARG, MAX_EXP_ARG))


def x_model(t, a1, b1, c1):
    return a1 * safe_exp(b1 * t) + c1


def y_model(t, a2, b2, c2):
    return a2 * safe_exp(-0.002 * t) + b2 * t + c2


def z_model(t, a3, b3, c3):
    return a3 * safe_exp(b3 * t) - 0.79 * t + c3


def parse_point_line(raw_line, line_number):
    text = raw_line.strip()
    if not text:
        return None

    if ":" in text:
        text = text.split(":", 1)[1]

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Line {line_number}: expected three comma-separated values, got {len(parts)}"
        )

    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid numeric value: {raw_line!r}") from exc


def load_trajectory(path):
    points = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        point = parse_point_line(raw_line, line_number)
        if point is not None:
            points.append(point)

    if len(points) < 4:
        raise ValueError("Need at least three observed points plus one landing label")

    data = np.asarray(points, dtype=np.float64)
    observed = data[:-1]
    label = data[-1]
    return observed, label


def build_time_axis(num_points, fps):
    if fps <= 0:
        raise ValueError("FPS must be positive")
    return np.arange(num_points, dtype=np.float64) / fps * 1000.0


def fit_axis(model, times_ms, values, initial_params):
    with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", OptimizeWarning)
        params, _ = curve_fit(
            model,
            times_ms,
            values,
            p0=initial_params,
            maxfev=MAX_FUNCTION_EVALS,
        )
    if not np.all(np.isfinite(params)):
        raise ValueError(f"Non-finite fitted parameters: {params}")
    return params


def fit_trajectory(observed, fps):
    times_ms = build_time_axis(len(observed), fps)
    x_values, y_values, z_values = observed.T
    last_t = times_ms[-1]

    x_initial = (x_values[0] - x_values[-1], -0.002, x_values[-1])
    y_slope = (y_values[-1] - y_values[0]) / last_t if last_t else 0.0
    y_initial = (y_values[0] - y_values[-1], y_slope, y_values[-1])
    z_initial = (z_values[0] - z_values[-1] + 0.79 * last_t, -0.002, z_values[0])

    return {
        "times_ms": times_ms,
        "x": fit_axis(x_model, times_ms, x_values, x_initial),
        "y": fit_axis(y_model, times_ms, y_values, y_initial),
        "z": fit_axis(z_model, times_ms, z_values, z_initial),
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


def solve_landing_time(params, landing_z):
    times_ms = params["times_ms"]
    start = times_ms[-1]

    def height_delta(time_ms):
        return z_model(time_ms, *params["z"]) - landing_z

    start_delta = height_delta(start)
    if np.isclose(start_delta, 0.0, atol=1e-9):
        return start

    step = max(1000.0, start if start > 0 else 1000.0)
    end = start + step
    end_delta = height_delta(end)

    for _ in range(20):
        if np.sign(start_delta) != np.sign(end_delta):
            return brentq(height_delta, start, end)
        step *= 2.0
        end = start + step
        end_delta = height_delta(end)

    raise ValueError(
        f"Could not bracket z={landing_z} after t={start:.3f} ms; "
        f"z(start)-target={start_delta:.6g}, z(end)-target={end_delta:.6g}"
    )


def assert_finite_result(result):
    for key, value in result.items():
        array = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Result field {key!r} contains non-finite values: {value}")


def format_vector(values):
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


def run(input_path, fps, landing_z):
    observed, label = load_trajectory(input_path)
    params = fit_trajectory(observed, fps)
    landing_time_ms = solve_landing_time(params, landing_z)
    predicted = evaluate_position(landing_time_ms, params)

    xy_error = float(np.linalg.norm(predicted[:2] - label[:2]))
    xyz_error = float(np.linalg.norm(predicted - label))

    result = {
        "observed_points": len(observed),
        "label": label,
        "predicted": predicted,
        "landing_time_ms": landing_time_ms,
        "xy_error": xy_error,
        "xyz_error": xyz_error,
        "x_params": params["x"],
        "y_params": params["y"],
        "z_params": params["z"],
    }
    assert_finite_result(result)
    return result


def predict_window(observed_window, label, fps, landing_z):
    params = fit_trajectory(observed_window, fps)
    landing_time_ms = solve_landing_time(params, landing_z)
    predicted = evaluate_position(landing_time_ms, params)

    result = {
        "landing_time_ms": float(landing_time_ms),
        "predicted": predicted,
        "xy_error": float(np.linalg.norm(predicted[:2] - label[:2])),
        "xyz_error": float(np.linalg.norm(predicted - label)),
        "x_params": params["x"],
        "y_params": params["y"],
        "z_params": params["z"],
    }
    assert_finite_result(result)
    return result


def run_sliding_windows(input_path, fps, landing_z, window_size):
    observed, label = load_trajectory(input_path)
    if window_size < 4:
        raise ValueError("Window size must be at least 4")
    if len(observed) < window_size:
        raise ValueError(
            f"Window size {window_size} is larger than observed point count {len(observed)}"
        )

    rows = []
    for start_index in range(0, len(observed) - window_size + 1):
        end_index = start_index + window_size - 1
        window = observed[start_index : start_index + window_size]
        row = {
            "start_index": start_index,
            "end_index": end_index,
            "window_end_time_ms": end_index / fps * 1000.0,
            "status": "ok",
        }
        try:
            result = predict_window(window, label, fps, landing_z)
            predicted = result["predicted"]
            row.update(
                {
                    "landing_time_from_window_ms": result["landing_time_ms"],
                    "pred_x": predicted[0],
                    "pred_y": predicted[1],
                    "pred_z": predicted[2],
                    "xy_error": result["xy_error"],
                    "xyz_error": result["xyz_error"],
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": f"failed: {exc}",
                    "landing_time_from_window_ms": np.nan,
                    "pred_x": np.nan,
                    "pred_y": np.nan,
                    "pred_z": np.nan,
                    "xy_error": np.nan,
                    "xyz_error": np.nan,
                }
            )
        rows.append(row)

    return {
        "observed_points": len(observed),
        "observed": observed,
        "label": label,
        "window_size": window_size,
        "rows": rows,
    }


def write_sliding_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "start_index",
        "end_index",
        "window_end_time_ms",
        "landing_time_from_window_ms",
        "pred_x",
        "pred_y",
        "pred_z",
        "xy_error",
        "xyz_error",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_sliding_result(sliding_result, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in sliding_result["rows"] if row["status"] == "ok"]
    if not rows:
        raise ValueError("No successful sliding-window predictions to plot")

    times = np.asarray([row["window_end_time_ms"] for row in rows], dtype=np.float64)
    pred_x = np.asarray([row["pred_x"] for row in rows], dtype=np.float64)
    pred_y = np.asarray([row["pred_y"] for row in rows], dtype=np.float64)
    xy_error = np.asarray([row["xy_error"] for row in rows], dtype=np.float64)
    label = sliding_result["label"]

    best_index = int(np.nanargmin(xy_error))
    best_time = times[best_index]
    best_error = xy_error[best_index]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(times, xy_error, color="#1f77b4", linewidth=2, label="xy_error")
    axes[0].scatter([best_time], [best_error], color="#d62728", zorder=3, label="best")
    axes[0].set_ylabel("XY error (cm)")
    axes[0].set_title(
        f"Sliding window landing prediction, window={sliding_result['window_size']} frames"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(times, pred_x, color="#2ca02c", linewidth=2, label="pred_x")
    axes[1].plot(times, pred_y, color="#ff7f0e", linewidth=2, label="pred_y")
    axes[1].axhline(label[0], color="#2ca02c", linestyle="--", alpha=0.6, label="label_x")
    axes[1].axhline(label[1], color="#ff7f0e", linestyle="--", alpha=0.6, label="label_y")
    axes[1].set_xlabel("Window end time (ms)")
    axes[1].set_ylabel("Landing coordinate (cm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_badminton_court(ax):
    from matplotlib.patches import Rectangle

    line_color = "#1b5e20"
    fill_color = "#dff2df"
    outer = Rectangle(
        (0.0, -COURT_HALF_WIDTH),
        COURT_LENGTH,
        COURT_WIDTH,
        facecolor=fill_color,
        edgecolor=line_color,
        linewidth=2.0,
        alpha=0.35,
        zorder=0,
    )
    ax.add_patch(outer)

    # Net and service lines. The coordinate origin is the midpoint of the left short side.
    ax.axvline(COURT_NET_X, color=line_color, linewidth=2.0, zorder=1)
    ax.axvline(
        COURT_NET_X - COURT_SHORT_SERVICE_DISTANCE,
        color=line_color,
        linewidth=1.4,
        zorder=1,
    )
    ax.axvline(
        COURT_NET_X + COURT_SHORT_SERVICE_DISTANCE,
        color=line_color,
        linewidth=1.4,
        zorder=1,
    )
    ax.axvline(
        COURT_DOUBLES_LONG_SERVICE_DISTANCE,
        color=line_color,
        linewidth=1.2,
        zorder=1,
    )
    ax.axvline(
        COURT_LENGTH - COURT_DOUBLES_LONG_SERVICE_DISTANCE,
        color=line_color,
        linewidth=1.2,
        zorder=1,
    )

    # Singles sidelines scaled from the standard 5.18m / 6.10m width ratio.
    ax.axhline(COURT_SINGLES_HALF_WIDTH, color=line_color, linewidth=1.2, zorder=1)
    ax.axhline(-COURT_SINGLES_HALF_WIDTH, color=line_color, linewidth=1.2, zorder=1)

    # Center service lines on both halves.
    ax.plot(
        [0.0, COURT_NET_X - COURT_SHORT_SERVICE_DISTANCE],
        [0.0, 0.0],
        color=line_color,
        linewidth=1.2,
        zorder=1,
    )
    ax.plot(
        [COURT_NET_X + COURT_SHORT_SERVICE_DISTANCE, COURT_LENGTH],
        [0.0, 0.0],
        color=line_color,
        linewidth=1.2,
        zorder=1,
    )

    ax.scatter([0.0], [0.0], color=line_color, s=32, zorder=2)
    ax.text(12.0, 12.0, "origin (0,0)", color=line_color, fontsize=9, zorder=2)
    ax.text(COURT_NET_X + 8.0, COURT_HALF_WIDTH - 34.0, "net", color=line_color, fontsize=9, zorder=2)


def make_sliding_gif(sliding_result, output_path, fps):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in sliding_result["rows"] if row["status"] == "ok"]
    if not rows:
        raise ValueError("No successful sliding-window predictions to animate")

    observed = sliding_result["observed"]
    label = sliding_result["label"]
    window_size = sliding_result["window_size"]
    pred_xy = np.asarray([[row["pred_x"], row["pred_y"]] for row in rows], dtype=np.float64)

    x_values = np.concatenate([observed[:, 0], pred_xy[:, 0], [label[0], 0.0, COURT_LENGTH]])
    y_values = np.concatenate(
        [observed[:, 1], pred_xy[:, 1], [label[1], -COURT_HALF_WIDTH, COURT_HALF_WIDTH]]
    )
    x_pad = max(20.0, 0.08 * (np.nanmax(x_values) - np.nanmin(x_values)))
    y_pad = max(20.0, 0.08 * (np.nanmax(y_values) - np.nanmin(y_values)))
    x_lim = (np.nanmin(x_values) - x_pad, np.nanmax(x_values) + x_pad)
    y_lim = (np.nanmin(y_values) - y_pad, np.nanmax(y_values) + y_pad)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_index, row in enumerate(rows):
        start = int(row["start_index"])
        end = int(row["end_index"])
        window = observed[start : end + 1]
        history = pred_xy[: frame_index + 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        draw_badminton_court(ax)
        ax.plot(observed[:, 0], observed[:, 1], color="#bdbdbd", linewidth=1.5, label="observed path")
        ax.scatter(observed[:, 0], observed[:, 1], color="#d9d9d9", s=14)
        ax.plot(window[:, 0], window[:, 1], color="#1f77b4", linewidth=2.5, label="current window")
        ax.scatter(window[:, 0], window[:, 1], color="#1f77b4", s=20)
        ax.plot(history[:, 0], history[:, 1], color="#ff7f0e", linewidth=2, label="predicted history")
        ax.scatter(history[:-1, 0], history[:-1, 1], color="#ffbb78", s=24)
        ax.scatter(history[-1, 0], history[-1, 1], color="#d62728", s=90, label="current prediction")
        ax.scatter(label[0], label[1], color="#111111", marker="*", s=180, label="ground truth")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title("Sliding-window landing prediction on XY plane")
        ax.text(
            0.02,
            0.98,
            (
                f"window: {start}-{end} ({window_size} frames)\n"
                f"end time: {row['window_end_time_ms']:.1f} ms\n"
                f"pred: ({row['pred_x']:.1f}, {row['pred_y']:.1f})\n"
                f"GT: ({label[0]:.1f}, {label[1]:.1f})\n"
                f"xy error: {row['xy_error']:.2f} cm"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88},
        )
        ax.legend(loc="lower right")

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps)


def print_result(result, fps, landing_z):
    print("Badminton trajectory fitting")
    print(f"fps: {fps:.6f}")
    print(f"landing_z: {landing_z:.6f}")
    print(f"observed_points: {result['observed_points']}")
    print(f"label_xyz: {format_vector(result['label'])}")
    print(f"predicted_xyz: {format_vector(result['predicted'])}")
    print(f"x_params(a1,b1,c1): {format_vector(result['x_params'])}")
    print(f"y_params(a2,b2,c2): {format_vector(result['y_params'])}")
    print(f"z_params(a3,b3,c3): {format_vector(result['z_params'])}")
    print(f"landing_time_ms: {result['landing_time_ms']:.6f}")
    print(f"xy_error: {result['xy_error']:.6f}")
    print(f"xyz_error: {result['xyz_error']:.6f}")


def print_sliding_result(result, fps, landing_z, csv_output, plot_output, gif_output):
    ok_rows = [row for row in result["rows"] if row["status"] == "ok"]
    failed_rows = [row for row in result["rows"] if row["status"] != "ok"]
    best_row = min(ok_rows, key=lambda row: row["xy_error"]) if ok_rows else None
    last_ok_row = ok_rows[-1] if ok_rows else None

    print("Badminton sliding-window trajectory fitting")
    print(f"fps: {fps:.6f}")
    print(f"landing_z: {landing_z:.6f}")
    print(f"observed_points: {result['observed_points']}")
    print(f"window_size: {result['window_size']}")
    print(f"windows_total: {len(result['rows'])}")
    print(f"windows_success: {len(ok_rows)}")
    print(f"windows_failed: {len(failed_rows)}")
    print(f"label_xyz: {format_vector(result['label'])}")
    if last_ok_row:
        print(
            "last_window_predicted_xyz: "
            f"{format_vector([last_ok_row['pred_x'], last_ok_row['pred_y'], last_ok_row['pred_z']])}"
        )
        print(f"last_window_xy_error: {last_ok_row['xy_error']:.6f}")
    if best_row:
        print(
            "best_window: "
            f"start={best_row['start_index']}, end={best_row['end_index']}, "
            f"end_time_ms={best_row['window_end_time_ms']:.6f}"
        )
        print(
            "best_predicted_xyz: "
            f"{format_vector([best_row['pred_x'], best_row['pred_y'], best_row['pred_z']])}"
        )
        print(f"best_xy_error: {best_row['xy_error']:.6f}")
    print(f"csv_output: {csv_output}")
    print(f"plot_output: {plot_output}")
    print(f"gif_output: {gif_output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit badminton trajectory using the formulas from MV-BMR Section 3.2.2."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Trajectory file path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Frame rate used to build the time axis. Default: {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--landing-z",
        type=float,
        default=DEFAULT_LANDING_Z,
        help=f"Target z value for landing-time solving. Default: {DEFAULT_LANDING_Z}",
    )
    parser.add_argument(
        "--sliding",
        action="store_true",
        help="Run dynamic sliding-window landing prediction instead of one full-sequence fit.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Observed points per sliding window. Default: {DEFAULT_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help=f"Sliding-window plot output path. Default: {DEFAULT_PLOT_OUTPUT}",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"Sliding-window CSV output path. Default: {DEFAULT_CSV_OUTPUT}",
    )
    parser.add_argument(
        "--gif-output",
        type=Path,
        default=DEFAULT_GIF_OUTPUT,
        help=f"Sliding-window GIF output path. Default: {DEFAULT_GIF_OUTPUT}",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=DEFAULT_GIF_FPS,
        help=f"Frames per second for the sliding-window GIF. Default: {DEFAULT_GIF_FPS}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sliding:
        result = run_sliding_windows(args.input, args.fps, args.landing_z, args.window_size)
        write_sliding_csv(result["rows"], args.csv_output)
        plot_sliding_result(result, args.plot_output)
        make_sliding_gif(result, args.gif_output, args.gif_fps)
        print_sliding_result(
            result,
            args.fps,
            args.landing_z,
            args.csv_output,
            args.plot_output,
            args.gif_output,
        )
    else:
        result = run(args.input, args.fps, args.landing_z)
        print_result(result, args.fps, args.landing_z)


if __name__ == "__main__":
    main()
