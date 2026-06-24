import argparse
import csv
import copy
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from trajectory_mlp import (
        DEFAULT_MLP_HIDDEN_DIMS,
        FEATURE_NAMES,
        TARGET_NAMES,
        build_c_parameter_mlp,
        estimate_landing_frame,
        evaluate_position,
        extract_state_features,
        fit_complete_trajectory_labels,
        fit_remaining_parameters,
        load_trajectory_file,
        safe_exp,
        solve_landing_time,
        x_model,
        y_model,
        z_model,
    )
except ModuleNotFoundError:
    from .trajectory_mlp import (
        DEFAULT_MLP_HIDDEN_DIMS,
        FEATURE_NAMES,
        TARGET_NAMES,
        build_c_parameter_mlp,
        estimate_landing_frame,
        evaluate_position,
        extract_state_features,
        fit_complete_trajectory_labels,
        fit_remaining_parameters,
        load_trajectory_file,
        safe_exp,
        solve_landing_time,
        x_model,
        y_model,
        z_model,
    )


DEFAULT_OUTPUT = Path(__file__).parents[1] / "models" / "c_parameter_mlp.pt"
DEFAULT_SYNTHETIC_DIR = Path(__file__).with_name("synthetic_debug_data")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_hidden_dims(value):
    if isinstance(value, (tuple, list)):
        hidden_dims = tuple(int(item) for item in value)
    else:
        try:
            hidden_dims = tuple(int(item.strip()) for item in str(value).split(","))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "hidden dimensions must be comma-separated integers"
            ) from exc
    if not hidden_dims or any(item <= 0 for item in hidden_dims):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive")
    return hidden_dims


def format_number(value):
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def generate_synthetic_dataset(output_dir, count, fps, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    generated_paths = []

    for sample_index in range(count):
        landing_frame = int(rng.integers(180, 301))
        max_observed = min(100, landing_frame - 35)
        observation_count = int(rng.integers(45, max_observed + 1))
        frame_ids = np.arange(observation_count, dtype=np.int64)
        times_ms = frame_ids / fps * 1000.0
        landing_time_ms = landing_frame / fps * 1000.0

        c1 = float(rng.uniform(80.0, 350.0))
        b1 = float(rng.uniform(-0.006, -0.0012))
        a1 = float(rng.uniform(450.0, 1000.0))

        c2 = float(rng.uniform(-180.0, 180.0))
        a2 = float(rng.uniform(-220.0, 220.0))
        b2 = float(rng.uniform(-0.12, 0.12))

        b3 = float(rng.uniform(-0.0045, -0.0005))
        a3 = float(rng.uniform(100.0, 500.0))
        c3 = float(0.79 * landing_time_ms - a3 * safe_exp(b3 * landing_time_ms))

        points = np.column_stack(
            [
                x_model(times_ms, a1, b1, c1),
                y_model(times_ms, a2, b2, c2),
                z_model(times_ms, a3, b3, c3),
            ]
        )
        points += rng.normal(0.0, 0.35, size=points.shape)
        landing_xyz = np.asarray(
            [
                x_model(landing_time_ms, a1, b1, c1),
                y_model(landing_time_ms, a2, b2, c2),
                0.0,
            ],
            dtype=np.float64,
        )

        output_path = output_dir / f"synthetic_{sample_index:04d}.txt"
        generated_paths.append(output_path)
        with output_path.open("w", encoding="utf-8") as handle:
            for frame_id, point in zip(frame_ids, points):
                coords = ",".join(format_number(value) for value in point)
                handle.write(f"{frame_id}:{coords}\n")
            landing_coords = ",".join(format_number(value) for value in landing_xyz)
            handle.write(f"{landing_frame}:{landing_coords}\n")

    print(f"Generated {count} synthetic trajectories in {output_dir}")
    return generated_paths


def trajectory_group_key(record):
    sample, _ = record
    path = sample.path
    if path is None:
        return str(id(sample))
    match = re.match(r"^(.+)_\d+$", path.stem)
    if match and path.with_name(match.group(1) + path.suffix).exists():
        return match.group(1)
    return path.stem


def split_records(records, seed):
    groups = {}
    for record in records:
        groups.setdefault(trajectory_group_key(record), []).append(record)
    grouped_records = list(groups.values())
    random.Random(seed).shuffle(grouped_records)
    if len(grouped_records) < 10:
        raise ValueError("Need at least 10 valid trajectory files for train/val/test splitting")

    train_end = max(1, int(len(grouped_records) * 0.8))
    val_end = max(train_end + 1, int(len(grouped_records) * 0.9))
    val_end = min(val_end, len(grouped_records) - 1)

    def flatten(group_slice):
        return [record for group in group_slice for record in group]

    return (
        flatten(grouped_records[:train_end]),
        flatten(grouped_records[train_end:val_end]),
        flatten(grouped_records[val_end:]),
    )


def load_labeled_trajectories(
    paths,
    fps,
    min_points,
    data_format,
    max_abs_c,
):
    records = []
    failures = []
    for path in paths:
        try:
            sample = load_trajectory_file(
                path,
                require_landing_frame=data_format == "timestamped",
                skip_zero_observations=True,
            )
            if data_format == "fall-segment":
                sample = estimate_landing_frame(sample)
            if len(sample.points) < min_points:
                raise ValueError(
                    f"need at least {min_points} valid observations, got {len(sample.points)}"
                )
            c_values, _ = fit_complete_trajectory_labels(sample, fps)
            if max_abs_c is not None and np.max(np.abs(c_values)) > max_abs_c:
                raise ValueError(
                    f"degenerate c label exceeds --max-abs-c={max_abs_c:g}: "
                    f"{c_values.tolist()}"
                )
            records.append((sample, c_values))
        except Exception as exc:
            failures.append((path, str(exc)))
    return records, failures


def filter_degenerate_labels(records, max_abs_c):
    if max_abs_c is None:
        return list(records), []
    valid = []
    rejected = []
    for record in records:
        sample, c_values = record
        if np.max(np.abs(c_values)) > max_abs_c:
            rejected.append(
                (
                    sample.path,
                    f"degenerate c label exceeds --max-abs-c={max_abs_c:g}: "
                    f"{c_values.tolist()}",
                )
            )
        else:
            valid.append(record)
    return valid, rejected


def build_prefix_dataset(records, fps, state_window_size, prefix_stride):
    features = []
    targets = []
    owners = []

    for sample, c_values in records:
        endpoints = list(range(state_window_size, len(sample.points) + 1, prefix_stride))
        if not endpoints:
            raise ValueError(
                f"{sample.path}: need at least {state_window_size} valid observations"
            )
        if endpoints[-1] != len(sample.points):
            endpoints.append(len(sample.points))
        for endpoint in endpoints:
            state = extract_state_features(
                sample.points[:endpoint],
                sample.frame_ids[:endpoint],
                fps,
                state_window_size,
            )
            features.append(state)
            targets.append(c_values)
            owners.append(sample.path.name if sample.path else "")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        owners,
    )


def normalize(values, mean, std):
    return (values - mean) / std


def make_loader(features, targets, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(targets).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def average_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            loss = criterion(model(features), targets)
            total += float(loss.item()) * len(features)
            count += len(features)
    return total / max(count, 1)


def train_model(
    train_x,
    train_y,
    val_x,
    val_y,
    *,
    hidden_dims=DEFAULT_MLP_HIDDEN_DIMS,
    dropout=0.1,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=300,
    patience=30,
    seed=42,
    device="cpu",
    verbose=False,
):
    set_seed(seed)
    device = torch.device(device)
    model = build_c_parameter_mlp(
        dropout=dropout,
        hidden_dims=hidden_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.SmoothL1Loss()
    train_loader = make_loader(train_x, train_y, batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    final_epoch = 0
    final_train_loss = float("nan")

    for epoch in range(1, epochs + 1):
        final_epoch = epoch
        model.train()
        train_total = 0.0
        train_count = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(features)
            train_count += len(features)

        final_train_loss = train_total / max(train_count, 1)
        val_loss = average_loss(model, val_loader, criterion, device)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"epoch={epoch:03d} train_loss={final_train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in best_state.items()
    }
    return {
        "model": model,
        "model_state": cpu_state,
        "best_epoch": best_epoch,
        "epochs_ran": final_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
    }


def evaluate_parameter_mae(model, features, targets, target_mean, target_std, device):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(features).float().to(device)).cpu().numpy()
    predictions = predictions * target_std + target_mean
    targets = targets * target_std + target_mean
    return np.mean(np.abs(predictions - targets), axis=0)


def shot_type(path):
    return path.stem.split("_round", 1)[0] if path else "unknown"


def evaluate_landing_predictions(
    model,
    records,
    feature_mean,
    feature_std,
    target_mean,
    target_std,
    args,
    device,
    visible_points=None,
):
    rows = []
    model.eval()
    for sample, target_c in records:
        point_count = len(sample.points) if visible_points is None else visible_points
        row = {
            "file": sample.path.name if sample.path else "",
            "shot_type": shot_type(sample.path),
            "observed_points": point_count,
            "landing_frame": sample.landing_frame,
            "landing_frame_estimated": sample.landing_frame_estimated,
            "status": "ok",
        }
        try:
            if len(sample.points) < point_count:
                raise ValueError(
                    f"need {point_count} visible points, got {len(sample.points)}"
                )
            points = sample.points[:point_count]
            frame_ids = sample.frame_ids[:point_count]
            features = extract_state_features(
                points,
                frame_ids,
                args.fps,
                args.state_window_size,
            )
            normalized = normalize(features, feature_mean, feature_std)
            with torch.no_grad():
                pred_normalized = (
                    model(torch.from_numpy(normalized).float().unsqueeze(0).to(device))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            c_values = pred_normalized * target_std + target_mean
            params = fit_remaining_parameters(
                points,
                frame_ids,
                args.fps,
                c_values,
            )
            landing_time = solve_landing_time(params, args.landing_z)
            predicted = evaluate_position(landing_time, params)
            row.update(
                {
                    "target_c1": target_c[0],
                    "target_c2": target_c[1],
                    "target_c3": target_c[2],
                    "pred_c1": c_values[0],
                    "pred_c2": c_values[1],
                    "pred_c3": c_values[2],
                    "pred_x": predicted[0],
                    "pred_y": predicted[1],
                    "pred_z": predicted[2],
                    "label_x": sample.landing_xyz[0],
                    "label_y": sample.landing_xyz[1],
                    "label_z": sample.landing_xyz[2],
                    "predicted_landing_time_ms": landing_time,
                    "xy_error": float(
                        np.linalg.norm(predicted[:2] - sample.landing_xyz[:2])
                    ),
                    "xyz_error": float(np.linalg.norm(predicted - sample.landing_xyz)),
                }
            )
        except Exception as exc:
            row["status"] = f"failed: {exc}"
        rows.append(row)
    return rows


def write_test_report(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "shot_type",
        "observed_points",
        "landing_frame",
        "landing_frame_estimated",
        "target_c1",
        "target_c2",
        "target_c3",
        "pred_c1",
        "pred_c2",
        "pred_c3",
        "pred_x",
        "pred_y",
        "pred_z",
        "label_x",
        "label_y",
        "label_z",
        "predicted_landing_time_ms",
        "xy_error",
        "xyz_error",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path,
    model_state,
    feature_mean,
    feature_std,
    target_mean,
    target_std,
    args,
    split_records_map,
    training_metadata=None,
    extra_metadata=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    training_metadata = training_metadata or {}
    checkpoint = {
        "model_state_dict": model_state,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "state_window_size": args.state_window_size,
        "fps": args.fps,
        "dropout": args.dropout,
        "hidden_dims": list(args.hidden_dims),
        "seed": args.seed,
        "hyperparameters": {
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "prefix_stride": args.prefix_stride,
            "seed": args.seed,
        },
        "best_epoch": training_metadata.get("best_epoch"),
        "epochs_ran": training_metadata.get("epochs_ran"),
        "best_val_loss": training_metadata.get("best_val_loss"),
        "training_data_format": args.data_format,
        "splits": {
            split_name: [record[0].path.name for record in records]
            for split_name, records in split_records_map.items()
        },
        "formula": {
            "x": "a1*exp(b1*t)+c1",
            "y": "a2*exp(-0.002*t)+b2*t+c2",
            "z": "a3*exp(b3*t)-0.79*t+c3",
            "time_unit": "milliseconds",
            "coordinate_unit": "centimeters",
        },
    }
    if extra_metadata:
        checkpoint.update(extra_metadata)
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP to predict c1/c2/c3.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-format",
        choices=["timestamped", "fall-segment"],
        default="timestamped",
        help=(
            "timestamped requires frame_id:x,y,z and landing_frame:x,y,z; "
            "fall-segment accepts x,y,z lines and estimates the landing frame from z"
        ),
    )
    parser.add_argument(
        "--test-report",
        type=Path,
        default=None,
        help="Optional per-trajectory test prediction CSV",
    )
    parser.add_argument(
        "--max-abs-c",
        type=float,
        default=20_000.0,
        help="Reject degenerate fitted c labels beyond this magnitude; use 0 to disable",
    )
    parser.add_argument("--fps", type=float, default=300.0)
    parser.add_argument("--landing-z", type=float, default=0.0)
    parser.add_argument("--state-window-size", type=int, default=20)
    parser.add_argument("--prefix-stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=DEFAULT_MLP_HIDDEN_DIMS,
        help="Comma-separated hidden layer widths, for example 128,128,64",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--synthetic-debug", action="store_true")
    parser.add_argument("--synthetic-data-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--synthetic-count", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.state_window_size < 3:
        raise ValueError("--state-window-size must be at least 3")
    if args.prefix_stride <= 0:
        raise ValueError("--prefix-stride must be positive")
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--batch-size, --epochs, and --patience must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("--lr must be positive and --weight-decay must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.synthetic_count < 10:
        raise ValueError("--synthetic-count must be at least 10")
    set_seed(args.seed)

    if args.synthetic_debug:
        all_paths = generate_synthetic_dataset(
            args.synthetic_data_dir,
            args.synthetic_count,
            args.fps,
            args.seed,
        )
        data_dir = args.synthetic_data_dir
        args.data_format = "timestamped"
    elif args.data_dir is not None:
        data_dir = args.data_dir
    else:
        raise ValueError("Pass --data-dir or use --synthetic-debug")

    if not args.synthetic_debug:
        all_paths = sorted(data_dir.rglob("*.txt"))
    records, failures = load_labeled_trajectories(
        all_paths,
        args.fps,
        args.state_window_size,
        args.data_format,
        None,
    )
    if failures:
        print(f"Skipped {len(failures)} invalid trajectories")
        for path, reason in failures[:10]:
            print(f"  {path}: {reason}")
    train_records, val_records, test_records = split_records(records, args.seed)
    max_abs_c = args.max_abs_c if args.max_abs_c > 0 else None
    train_records, rejected_train = filter_degenerate_labels(train_records, max_abs_c)
    val_records, rejected_val = filter_degenerate_labels(val_records, max_abs_c)
    rejected = rejected_train + rejected_val
    if rejected:
        print(f"Skipped {len(rejected)} degenerate train/validation trajectories")
        for path, reason in rejected:
            print(f"  {path}: {reason}")
    if not train_records or not val_records or not test_records:
        raise ValueError("Train, validation, and test splits must not be empty")

    train_x, train_y, _ = build_prefix_dataset(
        train_records, args.fps, args.state_window_size, args.prefix_stride
    )
    val_x, val_y, _ = build_prefix_dataset(
        val_records, args.fps, args.state_window_size, args.prefix_stride
    )
    test_x, test_y, _ = build_prefix_dataset(
        test_records, args.fps, args.state_window_size, args.prefix_stride
    )

    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    target_mean = train_y.mean(axis=0)
    target_std = np.maximum(train_y.std(axis=0), 1e-6)
    train_x = normalize(train_x, feature_mean, feature_std).astype(np.float32)
    val_x = normalize(val_x, feature_mean, feature_std).astype(np.float32)
    test_x = normalize(test_x, feature_mean, feature_std).astype(np.float32)
    train_y = normalize(train_y, target_mean, target_std).astype(np.float32)
    val_y = normalize(val_y, target_mean, target_std).astype(np.float32)
    test_y = normalize(test_y, target_mean, target_std).astype(np.float32)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(
        f"Trajectories train/val/test: "
        f"{len(train_records)}/{len(val_records)}/{len(test_records)}"
    )
    estimated_count = sum(
        record[0].landing_frame_estimated
        for record in train_records + val_records + test_records
    )
    print(f"Training data format: {args.data_format}")
    print(f"Estimated landing frames: {estimated_count}/{len(records)}")
    print(f"Prefix samples train/val/test: {len(train_x)}/{len(val_x)}/{len(test_x)}")
    print(f"Using device: {device}")
    training = train_model(
        train_x,
        train_y,
        val_x,
        val_y,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=device,
        verbose=True,
    )
    model = training["model"]
    save_checkpoint(
        args.output,
        training["model_state"],
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        args,
        {
            "train": train_records,
            "validation": val_records,
            "test": test_records,
            "excluded_train_validation": [
                (sample, c_values)
                for sample, c_values in records
                if sample.path in {path for path, _ in rejected}
            ],
        },
        training_metadata=training,
    )

    parameter_mae = evaluate_parameter_mae(
        model,
        test_x,
        test_y,
        target_mean,
        target_std,
        device,
    )
    prediction_rows = evaluate_landing_predictions(
        model,
        test_records,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        args,
        device,
    )
    landing_errors = np.asarray(
        [row["xy_error"] for row in prediction_rows if row["status"] == "ok"],
        dtype=np.float64,
    )
    print(
        "test_c_mae: "
        + ", ".join(f"{name}={value:.6f}" for name, value in zip(TARGET_NAMES, parameter_mae))
    )
    if len(landing_errors):
        print(
            f"test_landing_xy_error_cm: mean={landing_errors.mean():.6f}, "
            f"median={np.median(landing_errors):.6f}, "
            f"p90={np.percentile(landing_errors, 90):.6f}, "
            f"max={landing_errors.max():.6f}, count={len(landing_errors)}"
        )
    else:
        print("test_landing_xy_error_cm: no successful landing predictions")
    if args.test_report is not None:
        write_test_report(args.test_report, prediction_rows)
        print(f"Saved test report: {args.test_report}")
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
