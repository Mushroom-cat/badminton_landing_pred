import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

try:
    from trajectory_mlp import (
        CParameterPredictor,
        evaluate_position,
        fit_all_parameters,
        load_trajectory_file,
        predict_hybrid_parameters,
        solve_landing_time,
    )
except ModuleNotFoundError:
    from .trajectory_mlp import (
        CParameterPredictor,
        evaluate_position,
        fit_all_parameters,
        load_trajectory_file,
        predict_hybrid_parameters,
        solve_landing_time,
    )


ROOT = Path(__file__).parents[1]
DEFAULT_DATA_DIR = Path(__file__).with_name("20260418_label_fit_fall")
DEFAULT_MODEL = ROOT / "models" / "c_parameter_mlp_fall.pt"
DEFAULT_CSV_OUTPUT = Path(__file__).with_name(
    "20260418_sliding_visibility_comparison.csv"
)
DEFAULT_PLOT_OUTPUT = Path(__file__).with_name(
    "20260418_sliding_visibility_comparison.png"
)
CSV_FIELDS = (
    "file",
    "trajectory_group",
    "shot_type",
    "total_observed_frames",
    "visible_frames",
    "hidden_frames",
    "mlp_pred_x",
    "mlp_pred_y",
    "mlp_pred_z",
    "mlp_landing_time_ms",
    "mlp_xy_error",
    "mlp_status",
    "direct_pred_x",
    "direct_pred_y",
    "direct_pred_z",
    "direct_landing_time_ms",
    "direct_xy_error",
    "direct_status",
)


def shot_type(path):
    return path.stem.split("_round", 1)[0]


def trajectory_group(path):
    stem = path.stem
    base_stem, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit() and path.with_name(base_stem + path.suffix).exists():
        return base_stem
    return stem


def load_test_samples(data_dir, predictor):
    test_names = predictor.splits.get("test")
    if not test_names:
        raise ValueError("The MLP checkpoint does not contain a test split")
    if len(test_names) != len(set(test_names)):
        raise ValueError("The checkpoint test split contains duplicate filenames")

    samples = []
    for name in test_names:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Test trajectory not found: {path}")
        samples.append(load_trajectory_file(path))
    return samples


def _empty_prediction(prefix):
    return {
        f"{prefix}_pred_x": "",
        f"{prefix}_pred_y": "",
        f"{prefix}_pred_z": "",
        f"{prefix}_landing_time_ms": "",
        f"{prefix}_xy_error": "",
    }


def _finite_prediction(predicted, landing_time_ms, xy_error):
    values = np.concatenate(
        [
            np.asarray(predicted, dtype=np.float64),
            np.asarray([landing_time_ms, xy_error], dtype=np.float64),
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Prediction contains non-finite values")


def predict_mlp(sample, visible_frames, fps, landing_z, predictor):
    points = sample.points[:visible_frames]
    frame_ids = sample.frame_ids[:visible_frames]
    params, _, _ = predict_hybrid_parameters(
        points,
        frame_ids,
        fps,
        predictor,
        predictor.state_window_size,
    )
    landing_time_ms = solve_landing_time(params, landing_z)
    predicted = evaluate_position(landing_time_ms, params)
    xy_error = float(np.linalg.norm(predicted[:2] - sample.landing_xyz[:2]))
    _finite_prediction(predicted, landing_time_ms, xy_error)
    return predicted, float(landing_time_ms), xy_error


def predict_direct(sample, visible_frames, fps, landing_z):
    points = sample.points[:visible_frames]
    frame_ids = sample.frame_ids[:visible_frames]
    params = fit_all_parameters(points, frame_ids, fps)
    landing_time_ms = solve_landing_time(params, landing_z)
    predicted = evaluate_position(landing_time_ms, params)
    xy_error = float(np.linalg.norm(predicted[:2] - sample.landing_xyz[:2]))
    _finite_prediction(predicted, landing_time_ms, xy_error)
    return predicted, float(landing_time_ms), xy_error


def run_experiment(samples, predictor, fps, landing_z, min_visible_frames):
    if min_visible_frames < predictor.state_window_size:
        raise ValueError(
            f"min_visible_frames must be at least {predictor.state_window_size} "
            "for this checkpoint"
        )

    max_visible_frames = min(len(sample.points) for sample in samples)
    if max_visible_frames < min_visible_frames:
        raise ValueError(
            f"Only {max_visible_frames} common frames are available, fewer than "
            f"the requested minimum {min_visible_frames}"
        )

    rows = []
    for visible_frames in range(max_visible_frames, min_visible_frames - 1, -1):
        for sample in samples:
            row = {
                "file": sample.path.name,
                "trajectory_group": trajectory_group(sample.path),
                "shot_type": shot_type(sample.path),
                "total_observed_frames": len(sample.points),
                "visible_frames": visible_frames,
                "hidden_frames": len(sample.points) - visible_frames,
                "mlp_status": "ok",
                "direct_status": "ok",
            }
            row.update(_empty_prediction("mlp"))
            row.update(_empty_prediction("direct"))

            try:
                predicted, landing_time_ms, xy_error = predict_mlp(
                    sample,
                    visible_frames,
                    fps,
                    landing_z,
                    predictor,
                )
                row.update(
                    {
                        "mlp_pred_x": predicted[0],
                        "mlp_pred_y": predicted[1],
                        "mlp_pred_z": predicted[2],
                        "mlp_landing_time_ms": landing_time_ms,
                        "mlp_xy_error": xy_error,
                    }
                )
            except Exception as exc:
                row["mlp_status"] = f"failed: {exc}"

            try:
                predicted, landing_time_ms, xy_error = predict_direct(
                    sample,
                    visible_frames,
                    fps,
                    landing_z,
                )
                row.update(
                    {
                        "direct_pred_x": predicted[0],
                        "direct_pred_y": predicted[1],
                        "direct_pred_z": predicted[2],
                        "direct_landing_time_ms": landing_time_ms,
                        "direct_xy_error": xy_error,
                    }
                )
            except Exception as exc:
                row["direct_status"] = f"failed: {exc}"
            rows.append(row)

    return rows, max_visible_frames


def summarize_rows(rows):
    visible_values = sorted(
        {int(row["visible_frames"]) for row in rows},
        reverse=True,
    )
    summaries = []
    for visible_frames in visible_values:
        summary = {"visible_frames": visible_frames}
        matching = [
            row for row in rows if int(row["visible_frames"]) == visible_frames
        ]
        grouped = {}
        for row in matching:
            grouped.setdefault(row["trajectory_group"], []).append(row)
        for prefix in ("mlp", "direct"):
            group_values = []
            for group_rows in grouped.values():
                successful = [
                    float(row[f"{prefix}_xy_error"])
                    for row in group_rows
                    if row[f"{prefix}_status"] == "ok"
                ]
                if successful:
                    group_values.append(float(np.mean(successful)))
            values = np.asarray(group_values, dtype=np.float64)
            if not len(values):
                raise ValueError(
                    f"No successful {prefix} predictions at {visible_frames} frames"
                )
            summary.update(
                {
                    f"{prefix}_mean": float(np.mean(values)),
                    f"{prefix}_median": float(np.median(values)),
                    f"{prefix}_q25": float(np.quantile(values, 0.25)),
                    f"{prefix}_q75": float(np.quantile(values, 0.75)),
                    f"{prefix}_p90": float(np.quantile(values, 0.90)),
                    f"{prefix}_max": float(np.max(values)),
                    f"{prefix}_success": int(len(values)),
                    f"{prefix}_failed": int(len(grouped) - len(values)),
                }
            )
        summaries.append(summary)
    return summaries


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(summaries, output_path, trajectory_count):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["visible_frames"] for row in summaries], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "mlp": {"color": "#087e8b", "label": "MLP-assisted fitting"},
        "direct": {
            "color": "#d1495b",
            "label": "Direct fitting, matched constraints (no MLP)",
        },
    }
    for prefix, style in styles.items():
        mean = np.asarray([row[f"{prefix}_mean"] for row in summaries])
        q25 = np.asarray([row[f"{prefix}_q25"] for row in summaries])
        q75 = np.asarray([row[f"{prefix}_q75"] for row in summaries])
        ax.fill_between(x, q25, q75, color=style["color"], alpha=0.16)
        ax.plot(
            x,
            mean,
            color=style["color"],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=style["label"],
        )

        offsets = ((8, 8), (8, -18)) if prefix == "mlp" else ((8, 8), (8, 8))
        for index, offset in zip((0, -1), offsets):
            ax.annotate(
                f"{mean[index]:.2f} cm",
                (x[index], mean[index]),
                xytext=offset,
                textcoords="offset points",
                color=style["color"],
                fontsize=9,
                fontweight="semibold",
            )

    ax.set_xlim(float(np.max(x)), float(np.min(x)))
    max_frame = int(np.max(x))
    min_frame = int(np.min(x))
    tick_values = [max_frame]
    tick_values.extend(
        value
        for value in range(max_frame - 1, min_frame - 1, -1)
        if value % 5 == 0 and max_frame - value >= 5
    )
    if min_frame not in tick_values:
        tick_values.append(min_frame)
    ax.set_xticks(tick_values)
    ax.set_xlabel("Visible frames (decreasing)")
    ax.set_ylabel("Mean XY landing error (cm)")
    ax.set_title(
        f"Landing prediction as visible trajectory shrinks ({trajectory_count} test trajectories)"
    )
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_summary(
    rows,
    summaries,
    csv_output,
    plot_output,
    trajectory_file_count,
    trajectory_group_count,
):
    first = summaries[0]
    last = summaries[-1]
    print("Sliding visibility comparison")
    print(f"test_trajectory_files: {trajectory_file_count}")
    print(f"independent_trajectory_groups: {trajectory_group_count}")
    print(
        f"visible_frames: {first['visible_frames']} -> {last['visible_frames']}"
    )
    print(f"trajectory_window_rows: {len(rows)}")
    for summary in (first, last):
        visible = summary["visible_frames"]
        print(
            f"frames={visible}: mlp_mean={summary['mlp_mean']:.6f}, "
            f"direct_mean={summary['direct_mean']:.6f}, "
            f"mlp_success={summary['mlp_success']}, "
            f"direct_success={summary['direct_success']}"
        )
    print(f"csv_output: {csv_output}")
    print(f"plot_output: {plot_output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare MLP-assisted and direct landing prediction while visible "
            "trajectory frames shrink."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mlp-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=300.0)
    parser.add_argument("--landing-z", type=float, default=0.0)
    parser.add_argument("--min-visible-frames", type=int, default=20)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.min_visible_frames < 3:
        raise ValueError("--min-visible-frames must be at least 3")

    device = args.device
    if device == "auto":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required; run with py -3.11") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = CParameterPredictor(args.mlp_model, device=device)
    if not np.isclose(args.fps, predictor.training_fps):
        warnings.warn(
            f"Checkpoint FPS is {predictor.training_fps:g}, but experiment FPS "
            f"is {args.fps:g}",
            RuntimeWarning,
        )
    samples = load_test_samples(args.data_dir, predictor)
    trajectory_group_count = len(
        {trajectory_group(sample.path) for sample in samples}
    )
    rows, _ = run_experiment(
        samples,
        predictor,
        args.fps,
        args.landing_z,
        args.min_visible_frames,
    )
    summaries = summarize_rows(rows)
    write_csv(rows, args.csv_output)
    plot_comparison(summaries, args.plot_output, trajectory_group_count)
    print_summary(
        rows,
        summaries,
        args.csv_output,
        args.plot_output,
        len(samples),
        trajectory_group_count,
    )


if __name__ == "__main__":
    main()
