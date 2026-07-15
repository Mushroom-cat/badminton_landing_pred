#!/usr/bin/env python3
"""Train the explicit racket-swing + gated body-pose before-hit model."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_after_motion_trend import (  # noqa: E402
    RawSample,
    load_sample,
    mirror_sample,
    set_seed,
    split_by_group,
)
from util.before_pose_racket import (  # noqa: E402
    BODY_FEATURE_NAMES,
    CHECKPOINT_FORMAT,
    RACKET_FEATURE_NAMES,
    BeforePoseRacketTrendModel,
    before_motion_target,
    extract_before_pose_racket_features,
    reconstruct_before_landing_xy,
)


@dataclass
class Example:
    sample: RawSample
    feature: np.ndarray
    target: np.ndarray
    motion: object
    skip_n: int
    racket_point_count: int
    mirrored: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "scene1+2")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "models" / "before_pose_racket_trend.pt",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "results" / "before_pose_racket_trend",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-n", type=int, default=5)
    parser.add_argument("--max-racket-points", type=int, default=12)
    parser.add_argument("--max-trend-angle", type=float, default=60.0)
    parser.add_argument("--racket-aux-weight", type=float, default=0.25)
    parser.add_argument("--body-correction-weight", type=float, default=0.02)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def select_before_window(sample: RawSample, skip_n: int):
    end = len(sample.frames) - skip_n
    if end < 6:
        raise ValueError("before window is too short")
    return sample.frames[:end, :63], sample.frame_ids[:end]


def load_and_filter_samples(args) -> Tuple[List[RawSample], Counter]:
    accepted = []
    rejected: Counter = Counter()
    for path in sorted(args.data_dir.glob("*.txt")):
        try:
            sample = load_sample(path)
            frames, frame_ids = select_before_window(sample, args.skip_n)
            motion = extract_before_pose_racket_features(
                frames,
                frame_ids,
                max_racket_points=args.max_racket_points,
            )
            delta = sample.landing_xyz[:2] - motion.last_racket_center[:2]
            angle = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(delta, motion.direction_xy)
                            / max(np.linalg.norm(delta), 1e-9),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            if angle > args.max_trend_angle:
                raise ValueError(
                    f"landing/racket-swing angle exceeds {args.max_trend_angle:g} degrees"
                )
            before_motion_target(sample.landing_xyz[:2], motion)
        except (ValueError, OSError) as exc:
            rejected[str(exc)] += 1
            continue
        accepted.append(sample)
    if not accepted:
        raise ValueError("no valid before samples")
    return accepted, rejected


def build_example(
    sample: RawSample,
    skip_n: int,
    racket_point_count: int,
    mirrored: bool,
) -> Example:
    source = mirror_sample(sample) if mirrored else sample
    frames, frame_ids = select_before_window(source, skip_n)
    motion = extract_before_pose_racket_features(
        frames,
        frame_ids,
        max_racket_points=racket_point_count,
    )
    target = before_motion_target(source.landing_xyz[:2], motion)
    return Example(
        sample=source,
        feature=motion.features,
        target=target,
        motion=motion,
        skip_n=skip_n,
        racket_point_count=racket_point_count,
        mirrored=mirrored,
    )


def build_examples(
    samples: Sequence[RawSample],
    default_skip: int,
    max_racket_points: int,
    train: bool,
) -> Tuple[List[Example], Counter]:
    examples = []
    rejected: Counter = Counter()
    if train:
        configurations = [
            (max(default_skip - 1, 0), max(8, max_racket_points - 2)),
            (default_skip, max_racket_points),
            (default_skip + 1, max(8, max_racket_points - 2)),
        ]
    else:
        configurations = [(default_skip, max_racket_points)]
    for sample in samples:
        for skip_n, racket_points in configurations:
            try:
                examples.append(
                    build_example(sample, skip_n, racket_points, mirrored=False)
                )
                if train:
                    examples.append(
                        build_example(sample, skip_n, racket_points, mirrored=True)
                    )
            except ValueError as exc:
                rejected[str(exc)] += 1
    if not examples:
        raise ValueError("no before training examples were produced")
    return examples, rejected


def arrays(examples):
    return (
        np.stack([example.feature for example in examples]).astype(np.float32),
        np.stack([example.target for example in examples]).astype(np.float32),
    )


def predict_components(
    model,
    normalized_features,
    target_mean,
    target_std,
    device,
    batch_size,
):
    result = {
        "prediction": [],
        "racket_prediction": [],
        "body_gate": [],
        "body_correction": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(normalized_features), batch_size):
            tensor = torch.as_tensor(
                normalized_features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            components = model.forward_components(tensor)
            for key in result:
                result[key].append(components[key].cpu().numpy())
    for key in result:
        result[key] = np.concatenate(result[key], axis=0)
    result["prediction_raw"] = result["prediction"] * target_std + target_mean
    result["racket_prediction_raw"] = (
        result["racket_prediction"] * target_std + target_mean
    )
    return result


def metrics(targets, examples):
    errors = []
    x_errors = []
    y_errors = []
    trend_angles = []
    for target, example in zip(targets, examples):
        prediction = reconstruct_before_landing_xy(target, example.motion)
        truth = example.sample.landing_xyz[:2]
        difference = prediction - truth
        errors.append(float(np.linalg.norm(difference)))
        x_errors.append(abs(float(difference[0])))
        y_errors.append(abs(float(difference[1])))
        displacement = prediction - example.motion.last_racket_center[:2]
        cosine = float(
            np.dot(displacement, example.motion.direction_xy)
            / max(np.linalg.norm(displacement), 1e-9)
        )
        trend_angles.append(
            math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        )
    errors = np.asarray(errors)
    return {
        "count": len(errors),
        "xy_mean_cm": float(np.mean(errors)),
        "xy_median_cm": float(np.median(errors)),
        "xy_p90_cm": float(np.percentile(errors, 90)),
        "xy_max_cm": float(np.max(errors)),
        "x_mae_cm": float(np.mean(x_errors)),
        "y_mae_cm": float(np.mean(y_errors)),
        "predicted_swing_angle_median_deg": float(np.median(trend_angles)),
        "predicted_swing_angle_p90_deg": float(np.percentile(trend_angles, 90)),
    }


def write_predictions(path: Path, examples, components):
    fields = [
        "file",
        "group",
        "true_x",
        "true_y",
        "racket_base_x",
        "racket_base_y",
        "pred_x",
        "pred_y",
        "racket_base_error_xy_cm",
        "pred_error_xy_cm",
        "body_gate_forward",
        "body_gate_lateral",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, example in enumerate(examples):
            prediction = reconstruct_before_landing_xy(
                components["prediction_raw"][index], example.motion
            )
            racket_base = reconstruct_before_landing_xy(
                components["racket_prediction_raw"][index], example.motion
            )
            truth = example.sample.landing_xyz[:2]
            writer.writerow(
                {
                    "file": example.sample.path.name,
                    "group": example.sample.group,
                    "true_x": float(truth[0]),
                    "true_y": float(truth[1]),
                    "racket_base_x": float(racket_base[0]),
                    "racket_base_y": float(racket_base[1]),
                    "pred_x": float(prediction[0]),
                    "pred_y": float(prediction[1]),
                    "racket_base_error_xy_cm": float(
                        np.linalg.norm(racket_base - truth)
                    ),
                    "pred_error_xy_cm": float(np.linalg.norm(prediction - truth)),
                    "body_gate_forward": float(components["body_gate"][index, 0]),
                    "body_gate_lateral": float(components["body_gate"][index, 1]),
                }
            )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    samples, rejected = load_and_filter_samples(args)
    train_samples, val_samples, train_groups, val_groups = split_by_group(
        samples, args.val_ratio, args.seed
    )
    train_examples, train_example_rejected = build_examples(
        train_samples, args.skip_n, args.max_racket_points, train=True
    )
    val_examples, val_example_rejected = build_examples(
        val_samples, args.skip_n, args.max_racket_points, train=False
    )
    train_features, train_targets = arrays(train_examples)
    val_features, val_targets = arrays(val_examples)
    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1e-5)
    target_mean = train_targets.mean(axis=0)
    target_std = np.maximum(train_targets.std(axis=0), 1e-5)
    train_x = (train_features - feature_mean) / feature_std
    train_y = (train_targets - target_mean) / target_std
    val_x = (val_features - feature_mean) / feature_std
    val_y = (val_targets - target_mean) / target_std

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x.astype(np.float32)),
            torch.from_numpy(train_y.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model_config = {
        "racket_input_dim": len(RACKET_FEATURE_NAMES),
        "body_input_dim": len(BODY_FEATURE_NAMES),
        "hidden_dim": 128,
        "dropout": 0.08,
    }
    model = BeforePoseRacketTrendModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-5
    )
    smooth_l1 = torch.nn.SmoothL1Loss(beta=0.5)
    target_mean_tensor = torch.from_numpy(target_mean.astype(np.float32)).to(device)
    target_std_tensor = torch.from_numpy(target_std.astype(np.float32)).to(device)

    audit = {
        "device": str(device),
        "raw_files": len(list(args.data_dir.glob("*.txt"))),
        "accepted_samples": len(samples),
        "rejected_samples": int(sum(rejected.values())),
        "rejected_reasons": dict(rejected),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "train_groups": len(train_groups),
        "validation_groups": len(val_groups),
        "train_augmented_examples": len(train_examples),
        "train_example_rejected": int(sum(train_example_rejected.values())),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            components = model.forward_components(features)
            prediction = components["prediction"]
            racket_prediction = components["racket_prediction"]
            prediction_raw = prediction * target_std_tensor + target_mean_tensor
            target_raw = target * target_std_tensor + target_mean_tensor
            prediction_vector = torch.stack(
                [
                    torch.expm1(
                        torch.clamp(
                            prediction_raw[:, 0], 0.0, math.log1p(2500.0)
                        )
                    ),
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
            gated_correction = components["body_gate"] * components["body_correction"]
            loss = (
                smooth_l1(prediction, target)
                + args.racket_aux_weight * smooth_l1(racket_prediction, target)
                + 0.10 * direction_loss
                + args.body_correction_weight * torch.mean(gated_correction**2)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(features)
            count += len(features)

        components = predict_components(
            model, val_x, target_mean, target_std, device, args.batch_size
        )
        final_metrics = metrics(components["prediction_raw"], val_examples)
        racket_metrics = metrics(components["racket_prediction_raw"], val_examples)
        metric = final_metrics["xy_mean_cm"]
        scheduler.step(metric)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "val_xy_mean_cm": metric,
            "val_xy_median_cm": final_metrics["xy_median_cm"],
            "racket_base_xy_mean_cm": racket_metrics["xy_mean_cm"],
            "body_gate_forward_mean": float(np.mean(components["body_gate"][:, 0])),
            "body_gate_lateral_mean": float(np.mean(components["body_gate"][:, 1])),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train={row['train_loss']:.5f} "
                f"val_xy={metric:.2f}cm median={final_metrics['xy_median_cm']:.2f}cm "
                f"racket_base={racket_metrics['xy_mean_cm']:.2f}cm "
                f"gate=({row['body_gate_forward_mean']:.3f},"
                f"{row['body_gate_lateral_mean']:.3f})"
            )
        if metric < best_metric - 1e-4:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    components = predict_components(
        model, val_x, target_mean, target_std, device, args.batch_size
    )
    final_metrics = metrics(components["prediction_raw"], val_examples)
    racket_metrics = metrics(components["racket_prediction_raw"], val_examples)
    gate_summary = {
        "forward_mean": float(np.mean(components["body_gate"][:, 0])),
        "forward_std": float(np.std(components["body_gate"][:, 0])),
        "lateral_mean": float(np.mean(components["body_gate"][:, 1])),
        "lateral_std": float(np.std(components["body_gate"][:, 1])),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "model_state": {
            name: value.detach().cpu() for name, value in best_state.items()
        },
        "model_config": model_config,
        "max_racket_points": args.max_racket_points,
        "skip_n": args.skip_n,
        "racket_feature_names": list(RACKET_FEATURE_NAMES),
        "body_feature_names": list(BODY_FEATURE_NAMES),
        "feature_mean": torch.from_numpy(feature_mean.astype(np.float32)),
        "feature_std": torch.from_numpy(feature_std.astype(np.float32)),
        "target_mean": torch.from_numpy(target_mean.astype(np.float32)),
        "target_std": torch.from_numpy(target_std.astype(np.float32)),
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "racket_base_validation_metrics": racket_metrics,
        "body_gate_summary": gate_summary,
        "split": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "train_groups": train_groups,
            "validation_groups": val_groups,
        },
        "data_filter": {
            "rejected": dict(rejected),
            "train_example_rejected": dict(train_example_rejected),
            "validation_example_rejected": dict(val_example_rejected),
        },
    }
    torch.save(checkpoint, args.model_out)
    with (args.report_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    write_predictions(
        args.report_dir / "validation_predictions.csv", val_examples, components
    )
    report = {
        "checkpoint": str(args.model_out.resolve()),
        **audit,
        "group_overlap": sorted(set(train_groups) & set(val_groups)),
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "racket_base_validation_metrics": racket_metrics,
        "body_gate_summary": gate_summary,
        "racket_feature_count": len(RACKET_FEATURE_NAMES),
        "body_feature_count": len(BODY_FEATURE_NAMES),
    }
    (args.report_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
