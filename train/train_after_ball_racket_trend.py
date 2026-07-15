#!/usr/bin/env python3
"""Train the explicit ball-motion + gated racket-motion after-hit model."""

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
    load_and_filter_samples,
    mirror_sample,
    reconstruction_metrics,
    set_seed,
    split_by_group,
)
from util.after_motion import motion_target, reconstruct_landing_xy  # noqa: E402
from util.after_ball_racket import (  # noqa: E402
    BALL_FEATURE_NAMES,
    CHECKPOINT_FORMAT,
    RACKET_FEATURE_NAMES,
    AfterBallRacketTrendModel,
    extract_ball_racket_motion_features,
)


@dataclass
class Example:
    sample: RawSample
    feature: np.ndarray
    target: np.ndarray
    motion: object
    ball_racket_motion: object
    mirrored: bool
    ball_point_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ball-motion base model with a gated racket-motion correction"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "scene1+2")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "models" / "after_ball_racket_trend.pt",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "results" / "after_ball_racket_trend",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ball-points", type=int, default=6)
    parser.add_argument("--max-trend-angle", type=float, default=45.0)
    parser.add_argument("--ball-aux-weight", type=float, default=0.25)
    parser.add_argument("--racket-correction-weight", type=float, default=0.02)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def filter_racket_samples(
    samples: Sequence[RawSample], max_ball_points: int
) -> Tuple[List[RawSample], Counter]:
    accepted = []
    rejected: Counter = Counter()
    for sample in samples:
        try:
            extract_ball_racket_motion_features(
                sample.frames,
                sample.frame_ids,
                max_ball_points=max_ball_points,
                min_ball_points=5,
            )
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        accepted.append(sample)
    return accepted, rejected


def build_example(sample: RawSample, point_count: int, mirrored: bool) -> Example:
    source = mirror_sample(sample) if mirrored else sample
    ball_racket_motion = extract_ball_racket_motion_features(
        source.frames,
        source.frame_ids,
        max_ball_points=point_count,
        min_ball_points=5,
    )
    target = motion_target(
        source.landing_xyz[:2], ball_racket_motion.ball_motion
    )
    return Example(
        sample=source,
        feature=ball_racket_motion.features,
        target=target,
        motion=ball_racket_motion.ball_motion,
        ball_racket_motion=ball_racket_motion,
        mirrored=mirrored,
        ball_point_count=point_count,
    )


def build_examples(
    samples: Sequence[RawSample], max_ball_points: int, train: bool
) -> Tuple[List[Example], Counter]:
    examples: List[Example] = []
    rejected: Counter = Counter()
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
                examples.append(build_example(sample, point_count, False))
                if train:
                    examples.append(build_example(sample, point_count, True))
            except ValueError as exc:
                rejected[str(exc)] += 1
    if not examples:
        raise ValueError("no ball+racket training examples were produced")
    return examples, rejected


def example_arrays(examples: Sequence[Example]) -> Tuple[np.ndarray, np.ndarray]:
    features = np.stack([example.feature for example in examples]).astype(np.float32)
    targets = np.stack([example.target for example in examples]).astype(np.float32)
    return features, targets


def predict_components(
    model: AfterBallRacketTrendModel,
    normalized_features: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    batch_size: int,
):
    collected = {
        "prediction": [],
        "ball_prediction": [],
        "racket_gate": [],
        "racket_correction": [],
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
            for key in collected:
                collected[key].append(components[key].cpu().numpy())
    for key in collected:
        collected[key] = np.concatenate(collected[key], axis=0)
    collected["prediction_raw"] = (
        collected["prediction"] * target_std + target_mean
    )
    collected["ball_prediction_raw"] = (
        collected["ball_prediction"] * target_std + target_mean
    )
    return collected


def write_validation_predictions(
    path: Path,
    examples: Sequence[Example],
    components,
) -> None:
    fields = [
        "file",
        "group",
        "ball_points",
        "valid_racket_frames",
        "true_x",
        "true_y",
        "ball_base_x",
        "ball_base_y",
        "pred_x",
        "pred_y",
        "ball_base_error_xy_cm",
        "pred_error_xy_cm",
        "racket_gate_forward",
        "racket_gate_lateral",
        "racket_correction_forward_normalized",
        "racket_correction_lateral_normalized",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, example in enumerate(examples):
            prediction = reconstruct_landing_xy(
                components["prediction_raw"][index], example.motion
            )
            ball_base = reconstruct_landing_xy(
                components["ball_prediction_raw"][index], example.motion
            )
            truth = example.sample.landing_xyz[:2]
            writer.writerow(
                {
                    "file": example.sample.path.name,
                    "group": example.sample.group,
                    "ball_points": example.ball_point_count,
                    "valid_racket_frames": example.ball_racket_motion.valid_racket_count,
                    "true_x": float(truth[0]),
                    "true_y": float(truth[1]),
                    "ball_base_x": float(ball_base[0]),
                    "ball_base_y": float(ball_base[1]),
                    "pred_x": float(prediction[0]),
                    "pred_y": float(prediction[1]),
                    "ball_base_error_xy_cm": float(np.linalg.norm(ball_base - truth)),
                    "pred_error_xy_cm": float(np.linalg.norm(prediction - truth)),
                    "racket_gate_forward": float(components["racket_gate"][index, 0]),
                    "racket_gate_lateral": float(components["racket_gate"][index, 1]),
                    "racket_correction_forward_normalized": float(
                        components["racket_correction"][index, 0]
                    ),
                    "racket_correction_lateral_normalized": float(
                        components["racket_correction"][index, 1]
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    base_samples, base_rejected = load_and_filter_samples(
        args.data_dir,
        max_ball_points=args.max_ball_points,
        max_trend_angle=args.max_trend_angle,
    )
    samples, racket_rejected = filter_racket_samples(
        base_samples, args.max_ball_points
    )
    train_samples, val_samples, train_groups, val_groups = split_by_group(
        samples, args.val_ratio, args.seed
    )
    train_examples, train_example_rejected = build_examples(
        train_samples, args.max_ball_points, train=True
    )
    val_examples, val_example_rejected = build_examples(
        val_samples, args.max_ball_points, train=False
    )
    train_features, train_targets = example_arrays(train_examples)
    val_features, val_targets = example_arrays(val_examples)

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
        "ball_input_dim": len(BALL_FEATURE_NAMES),
        "racket_input_dim": len(RACKET_FEATURE_NAMES),
        "hidden_dim": 128,
        "dropout": 0.08,
    }
    model = AfterBallRacketTrendModel(**model_config).to(device)
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

    audit = {
        "device": str(device),
        "raw_files": len(list(args.data_dir.glob("*.txt"))),
        "ball_quality_accepted": len(base_samples),
        "ball_racket_quality_accepted": len(samples),
        "racket_rejected": int(sum(racket_rejected.values())),
        "racket_rejected_reasons": dict(racket_rejected),
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
        sample_count = 0
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            components = model.forward_components(features)
            prediction = components["prediction"]
            ball_prediction = components["ball_prediction"]
            regression_loss = smooth_l1(prediction, target)
            ball_aux_loss = smooth_l1(ball_prediction, target)

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
            gated_correction = (
                components["racket_gate"] * components["racket_correction"]
            )
            correction_loss = torch.mean(gated_correction**2)
            loss = (
                regression_loss
                + args.ball_aux_weight * ball_aux_loss
                + 0.10 * direction_loss
                + args.racket_correction_weight * correction_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(features)
            sample_count += len(features)

        components = predict_components(
            model, val_x, target_mean, target_std, device, args.batch_size
        )
        metrics = reconstruction_metrics(
            components["prediction_raw"], val_examples
        )
        ball_metrics = reconstruction_metrics(
            components["ball_prediction_raw"], val_examples
        )
        with torch.no_grad():
            val_prediction = model(val_x_tensor)
            val_loss = float(smooth_l1(val_prediction, val_y_tensor).item())
        metric = metrics["xy_mean_cm"]
        scheduler.step(metric)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(sample_count, 1),
            "val_loss": val_loss,
            "val_xy_mean_cm": metric,
            "val_xy_median_cm": metrics["xy_median_cm"],
            "ball_base_xy_mean_cm": ball_metrics["xy_mean_cm"],
            "gate_forward_mean": float(np.mean(components["racket_gate"][:, 0])),
            "gate_lateral_mean": float(np.mean(components["racket_gate"][:, 1])),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train={row['train_loss']:.5f} "
                f"val_xy={metric:.2f}cm median={metrics['xy_median_cm']:.2f}cm "
                f"ball_base={ball_metrics['xy_mean_cm']:.2f}cm "
                f"gate=({row['gate_forward_mean']:.3f},{row['gate_lateral_mean']:.3f})"
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
    final_metrics = reconstruction_metrics(
        components["prediction_raw"], val_examples
    )
    ball_base_metrics = reconstruction_metrics(
        components["ball_prediction_raw"], val_examples
    )
    gate_summary = {
        "forward_mean": float(np.mean(components["racket_gate"][:, 0])),
        "forward_std": float(np.std(components["racket_gate"][:, 0])),
        "lateral_mean": float(np.mean(components["racket_gate"][:, 1])),
        "lateral_std": float(np.std(components["racket_gate"][:, 1])),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "model_state": {
            name: value.detach().cpu() for name, value in best_state.items()
        },
        "model_config": model_config,
        "max_ball_points": args.max_ball_points,
        "ball_feature_names": list(BALL_FEATURE_NAMES),
        "racket_feature_names": list(RACKET_FEATURE_NAMES),
        "feature_mean": torch.from_numpy(feature_mean.astype(np.float32)),
        "feature_std": torch.from_numpy(feature_std.astype(np.float32)),
        "target_mean": torch.from_numpy(target_mean.astype(np.float32)),
        "target_std": torch.from_numpy(target_std.astype(np.float32)),
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "ball_base_validation_metrics": ball_base_metrics,
        "racket_gate_summary": gate_summary,
        "split": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "train_groups": train_groups,
            "validation_groups": val_groups,
        },
        "data_filter": {
            "base_rejected": dict(base_rejected),
            "racket_rejected": dict(racket_rejected),
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
    write_validation_predictions(
        args.report_dir / "validation_predictions.csv",
        val_examples,
        components,
    )
    report = {
        "checkpoint": str(args.model_out.resolve()),
        **audit,
        "group_overlap": sorted(set(train_groups) & set(val_groups)),
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "ball_base_validation_metrics": ball_base_metrics,
        "racket_gate_summary": gate_summary,
        "ball_feature_count": len(BALL_FEATURE_NAMES),
        "racket_feature_count": len(RACKET_FEATURE_NAMES),
    }
    (args.report_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
