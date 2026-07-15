#!/usr/bin/env python3
"""Compare the new and legacy before models on one group-held-out split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer.compare_duida_pose_fit_predictions import load_pose_predictor  # noqa: E402
from train.train_after_motion_trend import load_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "scene1+2")
    parser.add_argument(
        "--validation-index",
        type=Path,
        default=ROOT
        / "results"
        / "before_pose_racket_trend"
        / "validation_predictions.csv",
    )
    parser.add_argument(
        "--new-model",
        type=Path,
        default=ROOT / "models" / "before_pose_racket_trend.pt",
    )
    parser.add_argument(
        "--legacy-model", type=Path, default=ROOT / "models" / "before.pt"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "results"
        / "before_pose_racket_trend"
        / "same_split_model_comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT
        / "results"
        / "before_pose_racket_trend"
        / "same_split_model_comparison.csv",
    )
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--skip-n", type=int, default=5)
    return parser.parse_args()


def metric(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    with args.validation_index.open(encoding="utf-8", newline="") as file:
        index_rows = list(csv.DictReader(file))
    file_names = list(dict.fromkeys(row["file"] for row in index_rows))

    new_predictor = load_pose_predictor(args.new_model, None)
    legacy_predictor = load_pose_predictor(args.legacy_model, None)
    rows = []
    for file_name in file_names:
        sample = load_sample(args.data_dir / file_name)
        end = len(sample.frames) - args.skip_n
        start = end - args.window_size
        if start < 0:
            raise ValueError(f"{file_name}: fewer than window_size + skip_n frames")
        frames = sample.frames[start:end, :63].astype(np.float32, copy=False)
        frame_ids = sample.frame_ids[start:end]
        new_xyz = new_predictor.predict(frames, frame_ids)
        legacy_xyz = legacy_predictor.predict(frames, frame_ids)
        truth = sample.landing_xyz[:2]
        new_error = float(np.linalg.norm(new_xyz[:2] - truth))
        legacy_error = float(np.linalg.norm(legacy_xyz[:2] - truth))
        rows.append(
            {
                "file": file_name,
                "group": sample.group,
                "true_x": float(truth[0]),
                "true_y": float(truth[1]),
                "new_x": float(new_xyz[0]),
                "new_y": float(new_xyz[1]),
                "legacy_x": float(legacy_xyz[0]),
                "legacy_y": float(legacy_xyz[1]),
                "new_error_xy_cm": new_error,
                "legacy_error_xy_cm": legacy_error,
                "improvement_cm": legacy_error - new_error,
            }
        )

    summary = {
        "sample_count": len(rows),
        "group_count": len(set(row["group"] for row in rows)),
        "new_error_xy_cm": metric([row["new_error_xy_cm"] for row in rows]),
        "legacy_error_xy_cm": metric(
            [row["legacy_error_xy_cm"] for row in rows]
        ),
        "new_better_count": int(sum(row["improvement_cm"] > 0 for row in rows)),
        "mean_improvement_cm": float(
            np.mean([row["improvement_cm"] for row in rows])
        ),
        "warning": (
            "This split is group-held-out for the new model. The legacy checkpoint's "
            "original training split is unknown, so its number is a reference rather "
            "than a leakage-free retraining comparison."
        ),
    }
    payload = {
        "config": {
            "data_dir": str(args.data_dir.resolve()),
            "validation_index": str(args.validation_index.resolve()),
            "new_model": str(args.new_model.resolve()),
            "legacy_model": str(args.legacy_model.resolve()),
            "window_size": args.window_size,
            "skip_n": args.skip_n,
        },
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
