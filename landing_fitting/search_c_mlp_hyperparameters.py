import argparse
import csv
import gc
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from sliding_visibility_experiment import (
        plot_comparison,
        run_experiment,
        summarize_rows,
        write_csv as write_sliding_csv,
    )
    from train_c_mlp import (
        build_prefix_dataset,
        evaluate_landing_predictions,
        load_labeled_trajectories,
        normalize,
        save_checkpoint,
        train_model,
        trajectory_group_key,
        write_test_report,
    )
    from trajectory_mlp import CParameterPredictor, build_c_parameter_mlp
except ModuleNotFoundError:
    from .sliding_visibility_experiment import (
        plot_comparison,
        run_experiment,
        summarize_rows,
        write_csv as write_sliding_csv,
    )
    from .train_c_mlp import (
        build_prefix_dataset,
        evaluate_landing_predictions,
        load_labeled_trajectories,
        normalize,
        save_checkpoint,
        train_model,
        trajectory_group_key,
        write_test_report,
    )
    from .trajectory_mlp import CParameterPredictor, build_c_parameter_mlp


ROOT = Path(__file__).parents[1]
HERE = Path(__file__).parent
DEFAULT_DATA_DIR = HERE / "20260418_label_fit_fall"
DEFAULT_REFERENCE_MODEL = ROOT / "models" / "c_parameter_mlp_fall_w30.pt"
DEFAULT_OUTPUT_MODEL = ROOT / "models" / "c_parameter_mlp_fall_w30_tuned.pt"
DEFAULT_RESULTS = HERE / "20260418_mlp_hyperparameter_search.csv"
DEFAULT_SEARCH_PLOT = HERE / "20260418_mlp_hyperparameter_search.png"
DEFAULT_TEST_REPORT = HERE / "20260418_mlp_w30_tuned_test_results.csv"
DEFAULT_SLIDING_CSV = HERE / "20260418_sliding_visibility_comparison_w30_tuned.csv"
DEFAULT_SLIDING_PLOT = HERE / "20260418_sliding_visibility_comparison_w30_tuned.png"

HIDDEN_LAYOUTS = (
    (64, 64, 32),
    (128, 64, 32),
    (128, 128, 64),
    (256, 128, 64),
)
BASELINE_CONFIG = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "dropout": 0.1,
    "batch_size": 64,
    "hidden_dims": (128, 128, 64),
}
RESULT_FIELDS = (
    "phase",
    "config_id",
    "seed",
    "learning_rate",
    "weight_decay",
    "dropout",
    "batch_size",
    "hidden_dims",
    "best_epoch",
    "epochs_ran",
    "best_val_loss",
    "val_success_groups",
    "val_total_groups",
    "val_mean_xy_cm",
    "val_median_xy_cm",
    "val_group_std_cm",
    "seed_mean_xy_cm",
    "seed_std_xy_cm",
    "duration_seconds",
    "selected_top5",
    "selected_final",
)


def config_signature(config):
    return (
        round(float(config["lr"]), 14),
        round(float(config["weight_decay"]), 14),
        round(float(config["dropout"]), 12),
        int(config["batch_size"]),
        tuple(int(value) for value in config["hidden_dims"]),
    )


def generate_search_configs(trial_count, seed=42):
    if trial_count < 1:
        raise ValueError("trial_count must be positive")
    rng = np.random.default_rng(seed)
    configs = [dict(BASELINE_CONFIG)]
    signatures = {config_signature(configs[0])}
    while len(configs) < trial_count:
        config = {
            "lr": float(10 ** rng.uniform(math.log10(1e-4), math.log10(3e-3))),
            "weight_decay": float(
                10 ** rng.uniform(math.log10(1e-7), math.log10(1e-3))
            ),
            "dropout": float(rng.uniform(0.0, 0.3)),
            "batch_size": int(rng.choice((32, 64, 128))),
            "hidden_dims": tuple(HIDDEN_LAYOUTS[int(rng.integers(len(HIDDEN_LAYOUTS)))]),
        }
        signature = config_signature(config)
        if signature not in signatures:
            signatures.add(signature)
            configs.append(config)
    return configs


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_path_map(data_dir):
    paths = sorted(data_dir.rglob("*.txt"))
    path_map = {}
    for path in paths:
        if path.name in path_map:
            raise ValueError(f"Duplicate trajectory filename: {path.name}")
        path_map[path.name] = path
    return path_map


def load_named_records(path_map, names, fps, state_window_size, data_format):
    missing = [name for name in names if name not in path_map]
    if missing:
        raise ValueError(f"Missing trajectory files: {missing}")
    records, failures = load_labeled_trajectories(
        [path_map[name] for name in names],
        fps,
        state_window_size,
        data_format,
        None,
    )
    if failures:
        details = "; ".join(f"{path.name}: {reason}" for path, reason in failures)
        raise ValueError(f"Could not load reference split: {details}")
    record_map = {record[0].path.name: record for record in records}
    return [record_map[name] for name in names]


def prepare_prefix_data(train_records, val_records, fps, state_window_size, stride):
    train_x, train_y, _ = build_prefix_dataset(
        train_records, fps, state_window_size, stride
    )
    val_x, val_y, _ = build_prefix_dataset(
        val_records, fps, state_window_size, stride
    )
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    target_mean = train_y.mean(axis=0)
    target_std = np.maximum(train_y.std(axis=0), 1e-6)
    return {
        "train_x": normalize(train_x, feature_mean, feature_std).astype(np.float32),
        "train_y": normalize(train_y, target_mean, target_std).astype(np.float32),
        "val_x": normalize(val_x, feature_mean, feature_std).astype(np.float32),
        "val_y": normalize(val_y, target_mean, target_std).astype(np.float32),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def summarize_group_errors(rows, records):
    group_by_file = {
        sample.path.name: trajectory_group_key(record)
        for record in records
        for sample in [record[0]]
    }
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[group_by_file[row["file"]]].append(row)

    group_errors = []
    for group_rows in grouped_rows.values():
        if all(row["status"] == "ok" for row in group_rows):
            group_errors.append(
                float(np.mean([float(row["xy_error"]) for row in group_rows]))
            )
    values = np.asarray(group_errors, dtype=np.float64)
    return {
        "success_groups": len(values),
        "total_groups": len(grouped_rows),
        "mean": float(np.mean(values)) if len(values) else float("inf"),
        "median": float(np.median(values)) if len(values) else float("inf"),
        "std": float(np.std(values)) if len(values) else float("inf"),
    }


def trial_rank(row):
    failed_groups = int(row["val_total_groups"]) - int(row["val_success_groups"])
    mean_error = float(row["val_mean_xy_cm"])
    return failed_groups, mean_error


def confirmation_rank(summary):
    return (
        int(summary["failed_seed_runs"]),
        float(summary["seed_mean_xy_cm"]),
        float(summary["seed_std_xy_cm"]),
    )


def run_config(
    config_id,
    config,
    seed,
    prepared,
    val_records,
    args,
    device,
    phase,
):
    started = time.perf_counter()
    training = train_model(
        prepared["train_x"],
        prepared["train_y"],
        prepared["val_x"],
        prepared["val_y"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        epochs=args.epochs,
        patience=args.patience,
        seed=seed,
        device=device,
    )
    eval_args = SimpleNamespace(
        fps=args.fps,
        state_window_size=args.state_window_size,
        landing_z=args.landing_z,
    )
    prediction_rows = evaluate_landing_predictions(
        training["model"],
        val_records,
        prepared["feature_mean"],
        prepared["feature_std"],
        prepared["target_mean"],
        prepared["target_std"],
        eval_args,
        device,
        visible_points=args.visible_points,
    )
    summary = summarize_group_errors(prediction_rows, val_records)
    row = {
        "phase": phase,
        "config_id": config_id,
        "seed": seed,
        "learning_rate": config["lr"],
        "weight_decay": config["weight_decay"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "hidden_dims": ",".join(str(value) for value in config["hidden_dims"]),
        "best_epoch": training["best_epoch"],
        "epochs_ran": training["epochs_ran"],
        "best_val_loss": training["best_val_loss"],
        "val_success_groups": summary["success_groups"],
        "val_total_groups": summary["total_groups"],
        "val_mean_xy_cm": summary["mean"],
        "val_median_xy_cm": summary["median"],
        "val_group_std_cm": summary["std"],
        "seed_mean_xy_cm": "",
        "seed_std_xy_cm": "",
        "duration_seconds": time.perf_counter() - started,
        "selected_top5": False,
        "selected_final": False,
    }
    return row, training


def summarize_confirmations(config_id, config, rows):
    complete = [
        row
        for row in rows
        if int(row["val_success_groups"]) == int(row["val_total_groups"])
    ]
    values = np.asarray(
        [float(row["val_mean_xy_cm"]) for row in complete], dtype=np.float64
    )
    return {
        "config_id": config_id,
        "config": config,
        "failed_seed_runs": len(rows) - len(complete),
        "seed_mean_xy_cm": float(np.mean(values)) if len(values) else float("inf"),
        "seed_std_xy_cm": float(np.std(values)) if len(values) else float("inf"),
    }


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_search(screening_rows, confirmation_summaries, baseline_error, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    successful = [
        row
        for row in screening_rows
        if int(row["val_success_groups"]) == int(row["val_total_groups"])
    ]
    axes[0].scatter(
        [int(row["config_id"]) for row in successful],
        [float(row["val_mean_xy_cm"]) for row in successful],
        color="#087f8c",
        s=32,
    )
    axes[0].axhline(baseline_error, color="#d1495b", linestyle="--", label="Baseline")
    axes[0].set_title("36-config screening (seed 42)")
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Validation mean XY error (cm)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    ordered = sorted(confirmation_summaries, key=confirmation_rank)
    labels = [f"C{item['config_id']}" for item in ordered]
    means = [item["seed_mean_xy_cm"] for item in ordered]
    stds = [item["seed_std_xy_cm"] for item in ordered]
    colors = ["#087f8c" if index == 0 else "#6c757d" for index in range(len(ordered))]
    axes[1].bar(labels, means, yerr=stds, color=colors, capsize=4)
    axes[1].set_title("Top-5 confirmation (seeds 42/43/44)")
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel("Validation mean XY error (cm)")
    axes[1].grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_seed_list(value):
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one repeat seed is required")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search MLP hyperparameters using only train and validation data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--search-plot", type=Path, default=DEFAULT_SEARCH_PLOT)
    parser.add_argument("--test-report", type=Path, default=DEFAULT_TEST_REPORT)
    parser.add_argument("--sliding-csv", type=Path, default=DEFAULT_SLIDING_CSV)
    parser.add_argument("--sliding-plot", type=Path, default=DEFAULT_SLIDING_PLOT)
    parser.add_argument("--fps", type=float, default=300.0)
    parser.add_argument("--landing-z", type=float, default=0.0)
    parser.add_argument("--state-window-size", type=int, default=30)
    parser.add_argument("--visible-points", type=int, default=30)
    parser.add_argument("--prefix-stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--trials", type=int, default=36)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--repeat-seeds", type=parse_seed_list, default=(42, 43, 44))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.state_window_size != 30 or args.visible_points != 30:
        raise ValueError("This experiment requires a fixed 30-point state and objective")
    if args.trials < args.top_k or args.top_k < 1:
        raise ValueError("--trials must be at least --top-k and --top-k must be positive")
    if 42 not in args.repeat_seeds:
        raise ValueError("--repeat-seeds must include 42 for the final comparison model")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    reference = load_checkpoint(args.reference_model)
    if int(reference["state_window_size"]) != args.state_window_size:
        raise ValueError("Reference checkpoint does not use a 30-point state window")
    if not np.isclose(float(reference["fps"]), args.fps):
        raise ValueError("Reference checkpoint FPS does not match --fps")
    reference_splits = reference.get("splits", {})
    required_splits = ("train", "validation", "test")
    if any(not reference_splits.get(name) for name in required_splits):
        raise ValueError("Reference checkpoint must contain train/validation/test splits")
    split_sets = [set(reference_splits[name]) for name in required_splits]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Reference checkpoint splits overlap")

    path_map = build_path_map(args.data_dir)
    train_records = load_named_records(
        path_map,
        reference_splits["train"],
        args.fps,
        args.state_window_size,
        "fall-segment",
    )
    val_records = load_named_records(
        path_map,
        reference_splits["validation"],
        args.fps,
        args.state_window_size,
        "fall-segment",
    )
    prepared = prepare_prefix_data(
        train_records,
        val_records,
        args.fps,
        args.state_window_size,
        args.prefix_stride,
    )
    print(
        f"Fixed reference split: train={len(train_records)}, "
        f"validation={len(val_records)}, test={len(reference_splits['test'])} (not loaded)"
    )
    print(f"Using device: {device}")

    configs = generate_search_configs(args.trials, args.search_seed)
    screening_rows = []
    for index, config in enumerate(configs, 1):
        row, training = run_config(
            index,
            config,
            args.search_seed,
            prepared,
            val_records,
            args,
            device,
            "screening",
        )
        screening_rows.append(row)
        print(
            f"screen {index:02d}/{args.trials}: val={float(row['val_mean_xy_cm']):.4f} "
            f"cm success={row['val_success_groups']}/{row['val_total_groups']}"
        )
        del training
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    ranked_screening = sorted(screening_rows, key=trial_rank)
    top_ids = {int(row["config_id"]) for row in ranked_screening[: args.top_k]}
    for row in screening_rows:
        row["selected_top5"] = int(row["config_id"]) in top_ids

    confirmation_rows = []
    confirmation_states = {}
    confirmation_summaries = []
    for config_id in sorted(top_ids):
        config = configs[config_id - 1]
        config_rows = []
        for seed in args.repeat_seeds:
            row, training = run_config(
                config_id,
                config,
                seed,
                prepared,
                val_records,
                args,
                device,
                "confirmation",
            )
            row["selected_top5"] = True
            config_rows.append(row)
            confirmation_rows.append(row)
            if seed == 42:
                confirmation_states[config_id] = training["model_state"]
            print(
                f"confirm C{config_id} seed={seed}: "
                f"val={float(row['val_mean_xy_cm']):.4f} cm"
            )
            del training
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        summary = summarize_confirmations(config_id, config, config_rows)
        confirmation_summaries.append(summary)
        for row in config_rows:
            row["seed_mean_xy_cm"] = summary["seed_mean_xy_cm"]
            row["seed_std_xy_cm"] = summary["seed_std_xy_cm"]

    confirmation_summaries.sort(key=confirmation_rank)
    winner = confirmation_summaries[0]
    if winner["failed_seed_runs"]:
        raise RuntimeError("No top configuration succeeded for every repeat seed")
    winner_id = winner["config_id"]
    winner_config = winner["config"]
    for row in screening_rows + confirmation_rows:
        row["selected_final"] = int(row["config_id"]) == winner_id

    all_rows = screening_rows + confirmation_rows
    write_results(args.results, all_rows)
    plot_search(
        screening_rows,
        confirmation_summaries,
        float(screening_rows[0]["val_mean_xy_cm"]),
        args.search_plot,
    )

    # The development test split is loaded only after the winning configuration is fixed.
    test_records = load_named_records(
        path_map,
        reference_splits["test"],
        args.fps,
        args.state_window_size,
        "fall-segment",
    )
    final_state = confirmation_states.get(winner_id)
    if final_state is None:
        raise RuntimeError("Winning configuration has no seed-42 model state")
    final_model = build_c_parameter_mlp(
        dropout=winner_config["dropout"],
        hidden_dims=winner_config["hidden_dims"],
    ).to(device)
    final_model.load_state_dict(final_state)
    final_model.eval()
    final_args = SimpleNamespace(
        fps=args.fps,
        landing_z=args.landing_z,
        state_window_size=args.state_window_size,
        prefix_stride=args.prefix_stride,
        dropout=winner_config["dropout"],
        hidden_dims=winner_config["hidden_dims"],
        batch_size=winner_config["batch_size"],
        lr=winner_config["lr"],
        weight_decay=winner_config["weight_decay"],
        epochs=args.epochs,
        patience=args.patience,
        seed=42,
        data_format="fall-segment",
    )
    seed42_row = next(
        row
        for row in confirmation_rows
        if int(row["config_id"]) == winner_id and int(row["seed"]) == 42
    )
    save_checkpoint(
        args.output_model,
        final_state,
        prepared["feature_mean"],
        prepared["feature_std"],
        prepared["target_mean"],
        prepared["target_std"],
        final_args,
        {"train": train_records, "validation": val_records, "test": test_records},
        training_metadata={
            "best_epoch": int(seed42_row["best_epoch"]),
            "epochs_ran": int(seed42_row["epochs_ran"]),
            "best_val_loss": float(seed42_row["best_val_loss"]),
        },
        extra_metadata={
            "splits": reference_splits,
            "search": {
                "objective": "validation_group_mean_xy_error_at_30_visible_points",
                "trial_count": args.trials,
                "top_k": args.top_k,
                "repeat_seeds": list(args.repeat_seeds),
                "winning_config_id": winner_id,
                "winning_seed_mean_xy_cm": winner["seed_mean_xy_cm"],
                "winning_seed_std_xy_cm": winner["seed_std_xy_cm"],
            },
        },
    )

    test_rows = evaluate_landing_predictions(
        final_model,
        test_records,
        prepared["feature_mean"],
        prepared["feature_std"],
        prepared["target_mean"],
        prepared["target_std"],
        final_args,
        device,
        visible_points=args.visible_points,
    )
    write_test_report(args.test_report, test_rows)
    test_summary = summarize_group_errors(test_rows, test_records)

    predictor = CParameterPredictor(args.output_model, device=device)
    sliding_rows, _ = run_experiment(
        [sample for sample, _ in test_records],
        predictor,
        args.fps,
        args.landing_z,
        args.visible_points,
    )
    sliding_summaries = summarize_rows(sliding_rows)
    write_sliding_csv(sliding_rows, args.sliding_csv)
    trajectory_groups = len({trajectory_group_key(record) for record in test_records})
    plot_comparison(sliding_summaries, args.sliding_plot, trajectory_groups)

    print("Hyperparameter search complete")
    print(f"winner: C{winner_id} {winner_config}")
    print(
        f"validation_3seed_xy_cm: mean={winner['seed_mean_xy_cm']:.6f}, "
        f"std={winner['seed_std_xy_cm']:.6f}"
    )
    print(
        f"development_test_30point_xy_cm: mean={test_summary['mean']:.6f}, "
        f"success={test_summary['success_groups']}/{test_summary['total_groups']}"
    )
    print(f"model: {args.output_model}")
    print(f"results: {args.results}")
    print(f"search_plot: {args.search_plot}")
    print(f"sliding_csv: {args.sliding_csv}")
    print(f"sliding_plot: {args.sliding_plot}")


if __name__ == "__main__":
    main()
