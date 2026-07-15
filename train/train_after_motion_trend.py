#!/usr/bin/env python3
"""Train a trajectory-order-sensitive after-hit landing model on scene1+2."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.after_motion import (  # noqa: E402
    CHECKPOINT_FORMAT,
    DEFAULT_MAX_BALL_POINTS,
    FEATURE_NAMES,
    AfterMotionTrendModel,
    extract_ball_motion_features,
    motion_target,
    reconstruct_landing_xy,
)


@dataclass
class RawSample:
    path: Path
    group: str
    family: str
    frame_ids: np.ndarray
    frames: np.ndarray
    landing_xyz: np.ndarray


@dataclass
class Example:
    sample: RawSample
    feature: np.ndarray
    target: np.ndarray
    motion: object
    mirrored: bool
    ball_point_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ball-motion-aligned after-hit landing predictor"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "scene1+2")
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "after_motion_trend.pt")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results" / "after_motion_trend")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ball-points", type=int, default=DEFAULT_MAX_BALL_POINTS)
    parser.add_argument("--max-trend-angle", type=float, default=45.0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_family(group: str) -> str:
    if group and group[0].isdigit():
        return "dated_session"
    match = re.match(r"^(round\d+_data)_session\d+$", group)
    if match:
        return match.group(1)
    match = re.match(r"^(.+?)_round\d+$", group)
    if match:
        return match.group(1)
    return group


def load_sample(path: Path) -> RawSample:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("too few lines")

    _, landing_text = lines[-1].split(":", 1)
    landing_xyz = np.asarray([float(value) for value in landing_text.split(",")], dtype=np.float64)
    if len(landing_xyz) < 3 or not np.all(np.isfinite(landing_xyz[:3])):
        raise ValueError("invalid landing label")
    landing_xyz = landing_xyz[:3]
    if (
        landing_xyz[0] < -10
        or landing_xyz[0] > 680
        or landing_xyz[1] < -320
        or landing_xyz[1] > 320
        or abs(landing_xyz[2]) > 10
    ):
        raise ValueError("landing label outside court range")
    landing_xyz[2] = 0.0

    frame_ids: List[int] = []
    frames: List[np.ndarray] = []
    for line in lines[:-1]:
        frame_text, values_text = line.split(":", 1)
        values = np.asarray([float(value) for value in values_text.split(",")], dtype=np.float64)
        if len(values) < 66:
            raise ValueError("frame has fewer than 66 coordinates")
        frame_ids.append(int(frame_text))
        frames.append(values[:66])
    frame_ids_array = np.asarray(frame_ids, dtype=np.int64)
    if np.any(np.diff(frame_ids_array) <= 0):
        raise ValueError("frame ids are not increasing")
    group = path.stem.rsplit("---", 1)[0]
    return RawSample(
        path=path,
        group=group,
        family=group_family(group),
        frame_ids=frame_ids_array,
        frames=np.asarray(frames, dtype=np.float64),
        landing_xyz=landing_xyz,
    )


def quality_check(sample: RawSample, max_ball_points: int, max_trend_angle: float) -> Tuple[bool, str]:
    ball = sample.frames[:, 63:66]
    valid_ball_count = int(
        np.sum(
            np.all(np.isfinite(ball), axis=1)
            & ~np.all(np.isclose(ball, 0.0, atol=1e-9), axis=1)
        )
    )
    if valid_ball_count < 5:
        return False, "fewer than 5 valid ball points"
    try:
        motion = extract_ball_motion_features(
            sample.frames,
            sample.frame_ids,
            max_ball_points=max_ball_points,
        )
    except ValueError as exc:
        return False, str(exc)
    if int(motion.selected_indices[-1]) < len(sample.frames) - 5:
        return False, "last ball point is not near the observation-window tail"
    delta = sample.landing_xyz[:2] - motion.last_ball[:2]
    distance = float(np.linalg.norm(delta))
    if not 1.0 <= distance <= 2000.0:
        return False, "landing displacement outside robust range"
    cosine = float(np.dot(delta, motion.direction_xy) / max(distance, 1e-9))
    angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    if angle > max_trend_angle:
        return False, f"landing/trajectory angle exceeds {max_trend_angle:g} degrees"
    return True, "ok"


def load_and_filter_samples(
    data_dir: Path,
    max_ball_points: int,
    max_trend_angle: float,
) -> Tuple[List[RawSample], Counter]:
    samples: List[RawSample] = []
    rejected: Counter = Counter()
    for path in sorted(data_dir.glob("*.txt")):
        try:
            sample = load_sample(path)
        except (ValueError, OSError) as exc:
            rejected[str(exc)] += 1
            continue
        accepted, reason = quality_check(sample, max_ball_points, max_trend_angle)
        if accepted:
            samples.append(sample)
        else:
            rejected[reason] += 1
    if not samples:
        raise ValueError(f"no usable samples in {data_dir}")
    return samples, rejected


def split_by_group(
    samples: Sequence[RawSample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[RawSample], List[RawSample], List[str], List[str]]:
    family_groups: Dict[str, List[str]] = defaultdict(list)
    group_samples: Dict[str, List[RawSample]] = defaultdict(list)
    for sample in samples:
        group_samples[sample.group].append(sample)
    for group, members in group_samples.items():
        family_groups[members[0].family].append(group)

    rng = random.Random(seed)
    train_groups = set()
    val_groups = set()
    singleton_families = []
    for family in sorted(family_groups):
        groups = sorted(family_groups[family])
        rng.shuffle(groups)
        if len(groups) == 1:
            singleton_families.extend(groups)
            continue
        val_count = max(1, int(round(len(groups) * val_ratio)))
        val_count = min(val_count, len(groups) - 1)
        val_groups.update(groups[:val_count])
        train_groups.update(groups[val_count:])

    # Single-group shot families cannot be split without leakage. Keep them in
    # training; the held-out validation remains group-disjoint.
    train_groups.update(singleton_families)
    train = [sample for sample in samples if sample.group in train_groups]
    val = [sample for sample in samples if sample.group in val_groups]
    if not train or not val:
        raise ValueError("group split produced an empty train or validation set")
    if train_groups & val_groups:
        raise AssertionError("group leakage between train and validation")
    return train, val, sorted(train_groups), sorted(val_groups)


def mirror_sample(sample: RawSample) -> RawSample:
    frames = sample.frames.copy()
    frames[:, 1::3] *= -1.0
    label = sample.landing_xyz.copy()
    label[1] *= -1.0
    return RawSample(
        path=sample.path,
        group=sample.group,
        family=sample.family,
        frame_ids=sample.frame_ids,
        frames=frames,
        landing_xyz=label,
    )


def build_example(sample: RawSample, point_count: int, mirrored: bool) -> Example:
    source = mirror_sample(sample) if mirrored else sample
    motion = extract_ball_motion_features(
        source.frames,
        source.frame_ids,
        max_ball_points=point_count,
    )
    target = motion_target(source.landing_xyz[:2], motion)
    return Example(
        sample=source,
        feature=motion.features,
        target=target,
        motion=motion,
        mirrored=mirrored,
        ball_point_count=point_count,
    )


def build_examples(
    samples: Sequence[RawSample],
    max_ball_points: int,
    train: bool,
) -> List[Example]:
    examples: List[Example] = []
    for sample in samples:
        ball = sample.frames[:, 63:66]
        valid_count = int(
            np.sum(
                np.all(np.isfinite(ball), axis=1)
                & ~np.all(np.isclose(ball, 0.0, atol=1e-9), axis=1)
            )
        )
        largest = min(max_ball_points, valid_count)
        point_counts = range(5, largest + 1) if train else [largest]
        for point_count in point_counts:
            try:
                examples.append(build_example(sample, point_count, mirrored=False))
                if train:
                    examples.append(build_example(sample, point_count, mirrored=True))
            except ValueError:
                # Very short noisy tails can point backwards even when the
                # six-point regression is valid. They are unsuitable for the
                # forward-distance parameterization and are omitted.
                continue
    return examples


def arrays_from_examples(examples: Sequence[Example]) -> Tuple[np.ndarray, np.ndarray]:
    features = np.stack([example.feature for example in examples]).astype(np.float32)
    targets = np.stack([example.target for example in examples]).astype(np.float32)
    return features, targets


def reconstruction_metrics(predicted_targets: np.ndarray, examples: Sequence[Example]) -> Dict[str, float]:
    errors = []
    x_errors = []
    y_errors = []
    trend_angles = []
    for target, example in zip(predicted_targets, examples):
        prediction = reconstruct_landing_xy(target, example.motion)
        truth = example.sample.landing_xyz[:2]
        delta = prediction - truth
        errors.append(float(np.linalg.norm(delta)))
        x_errors.append(abs(float(delta[0])))
        y_errors.append(abs(float(delta[1])))
        predicted_displacement = prediction - example.motion.last_ball[:2]
        cosine = float(
            np.dot(predicted_displacement, example.motion.direction_xy)
            / max(np.linalg.norm(predicted_displacement), 1e-9)
        )
        trend_angles.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
    values = np.asarray(errors)
    return {
        "count": int(len(values)),
        "xy_mean_cm": float(np.mean(values)),
        "xy_median_cm": float(np.median(values)),
        "xy_p90_cm": float(np.percentile(values, 90)),
        "xy_max_cm": float(np.max(values)),
        "x_mae_cm": float(np.mean(x_errors)),
        "y_mae_cm": float(np.mean(y_errors)),
        "predicted_trend_angle_median_deg": float(np.median(trend_angles)),
        "predicted_trend_angle_p90_deg": float(np.percentile(trend_angles, 90)),
    }


def predict_targets(
    model: AfterMotionTrendModel,
    normalized_features: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(normalized_features), batch_size):
            batch = torch.as_tensor(
                normalized_features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(model(batch).cpu().numpy())
    normalized = np.concatenate(chunks, axis=0)
    return normalized * target_std + target_mean


def write_prediction_csv(
    path: Path,
    examples: Sequence[Example],
    predicted_targets: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "group",
        "ball_points",
        "observed_direction_x",
        "observed_direction_y",
        "true_x",
        "true_y",
        "pred_x",
        "pred_y",
        "error_xy_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for example, target in zip(examples, predicted_targets):
            prediction = reconstruct_landing_xy(target, example.motion)
            truth = example.sample.landing_xyz[:2]
            writer.writerow(
                {
                    "file": example.sample.path.name,
                    "group": example.sample.group,
                    "ball_points": example.ball_point_count,
                    "observed_direction_x": float(example.motion.direction_xy[0]),
                    "observed_direction_y": float(example.motion.direction_xy[1]),
                    "true_x": float(truth[0]),
                    "true_y": float(truth[1]),
                    "pred_x": float(prediction[0]),
                    "pred_y": float(prediction[1]),
                    "error_xy_cm": float(np.linalg.norm(prediction - truth)),
                }
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val-ratio must be between 0 and 1")
    if args.max_ball_points < 4:
        raise ValueError("max-ball-points must be at least 4")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    samples, rejected = load_and_filter_samples(
        args.data_dir,
        max_ball_points=args.max_ball_points,
        max_trend_angle=args.max_trend_angle,
    )
    train_samples, val_samples, train_groups, val_groups = split_by_group(
        samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_examples = build_examples(train_samples, args.max_ball_points, train=True)
    val_examples = build_examples(val_samples, args.max_ball_points, train=False)
    train_features, train_targets = arrays_from_examples(train_examples)
    val_features, val_targets = arrays_from_examples(val_examples)

    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1e-5)
    target_mean = train_targets.mean(axis=0)
    target_std = np.maximum(train_targets.std(axis=0), 1e-5)
    train_x = (train_features - feature_mean) / feature_std
    train_y = (train_targets - target_mean) / target_std
    val_x = (val_features - feature_mean) / feature_std
    val_y = (val_targets - target_mean) / target_std

    dataset = TensorDataset(
        torch.from_numpy(train_x.astype(np.float32)),
        torch.from_numpy(train_y.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_config = {
        "input_dim": len(FEATURE_NAMES),
        "hidden_dims": [256, 256, 128],
        "dropout": 0.08,
    }
    model = AfterMotionTrendModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-5
    )
    smooth_l1 = torch.nn.SmoothL1Loss(beta=0.5)

    val_x_tensor = torch.from_numpy(val_x.astype(np.float32)).to(device)
    val_y_tensor = torch.from_numpy(val_y.astype(np.float32)).to(device)
    target_mean_tensor = torch.from_numpy(target_mean.astype(np.float32)).to(device)
    target_std_tensor = torch.from_numpy(target_std.astype(np.float32)).to(device)

    print(
        json.dumps(
            {
                "device": str(device),
                "raw_files": len(list(args.data_dir.glob("*.txt"))),
                "accepted_samples": len(samples),
                "rejected_samples": int(sum(rejected.values())),
                "rejected_reasons": dict(rejected),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "train_augmented_examples": len(train_examples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            regression_loss = smooth_l1(prediction, target)

            prediction_raw = prediction * target_std_tensor + target_mean_tensor
            target_raw = target * target_std_tensor + target_mean_tensor
            prediction_vector = torch.stack(
                [
                    torch.expm1(torch.clamp(prediction_raw[:, 0], 0.0, math.log1p(2500.0))),
                    prediction_raw[:, 1],
                ],
                dim=1,
            )
            target_vector = torch.stack(
                [torch.expm1(target_raw[:, 0]), target_raw[:, 1]], dim=1
            )
            direction_loss = 1.0 - torch.nn.functional.cosine_similarity(
                prediction_vector, target_vector, dim=1
            ).mean()
            loss = regression_loss + 0.10 * direction_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(features)
            train_count += len(features)

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x_tensor)
            val_loss = float(smooth_l1(val_prediction, val_y_tensor).item())
        predicted_targets = (
            val_prediction.cpu().numpy() * target_std + target_mean
        )
        metrics = reconstruction_metrics(predicted_targets, val_examples)
        metric = metrics["xy_mean_cm"]
        scheduler.step(metric)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train={row['train_loss']:.5f} "
                f"val={val_loss:.5f} val_xy_mean={metric:.2f}cm "
                f"median={metrics['xy_median_cm']:.2f}cm lr={row['lr']:.2e}"
            )

        if metric < best_metric - 1e-4:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    predicted_targets = predict_targets(
        model,
        val_x,
        target_mean,
        target_std,
        device,
        args.batch_size,
    )
    final_metrics = reconstruction_metrics(predicted_targets, val_examples)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {name: value.detach().cpu() for name, value in best_state.items()}
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "model_state": cpu_state,
        "model_config": model_config,
        "max_ball_points": args.max_ball_points,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": torch.from_numpy(feature_mean.astype(np.float32)),
        "feature_std": torch.from_numpy(feature_std.astype(np.float32)),
        "target_mean": torch.from_numpy(target_mean.astype(np.float32)),
        "target_std": torch.from_numpy(target_std.astype(np.float32)),
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "split": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "train_groups": train_groups,
            "val_groups": val_groups,
        },
        "data_filter": {
            "max_trend_angle_deg": args.max_trend_angle,
            "rejected_reasons": dict(rejected),
        },
    }
    torch.save(checkpoint, args.model_out)

    history_path = args.report_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    write_prediction_csv(
        args.report_dir / "validation_predictions.csv",
        val_examples,
        predicted_targets,
    )
    report = {
        "checkpoint": str(args.model_out.resolve()),
        "dataset": str(args.data_dir.resolve()),
        "raw_file_count": len(list(args.data_dir.glob("*.txt"))),
        "accepted_sample_count": len(samples),
        "rejected_sample_count": int(sum(rejected.values())),
        "rejected_reasons": dict(rejected),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(val_samples),
        "train_group_count": len(train_groups),
        "validation_group_count": len(val_groups),
        "group_overlap": sorted(set(train_groups) & set(val_groups)),
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "feature_names": list(FEATURE_NAMES),
    }
    (args.report_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
