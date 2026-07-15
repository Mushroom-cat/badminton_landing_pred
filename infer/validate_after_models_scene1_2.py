#!/usr/bin/env python3
"""Evaluate ball-only and ball+racket models on the same group-held-out split."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "scene1+2")
    parser.add_argument(
        "--validation-index",
        type=Path,
        default=ROOT
        / "results"
        / "after_ball_racket_trend"
        / "validation_predictions.csv",
    )
    parser.add_argument(
        "--ball-racket-model",
        type=Path,
        default=ROOT / "models" / "after_ball_racket_trend.pt",
    )
    parser.add_argument(
        "--ball-only-model",
        type=Path,
        default=ROOT / "models" / "after_motion_trend.pt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "results"
        / "after_ball_racket_trend"
        / "same_split_model_comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT
        / "results"
        / "after_ball_racket_trend"
        / "same_split_model_comparison.csv",
    )
    return parser.parse_args()


def metric(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main():
    args = parse_args()
    with args.validation_index.open(encoding="utf-8", newline="") as file:
        index_rows = list(csv.DictReader(file))
    file_names = []
    seen = set()
    for row in index_rows:
        if row["file"] not in seen:
            seen.add(row["file"])
            file_names.append(row["file"])

    ball_racket = load_pose_predictor(args.ball_racket_model, None)
    ball_only = load_pose_predictor(args.ball_only_model, None)
    rows = []
    for file_name in file_names:
        sample = load_sample(args.data_dir / file_name)
        ball_racket_xyz = ball_racket.predict(sample.frames, sample.frame_ids)
        ball_only_xyz = ball_only.predict(sample.frames, sample.frame_ids)
        truth = sample.landing_xyz[:2]
        ball_racket_error = float(np.linalg.norm(ball_racket_xyz[:2] - truth))
        ball_only_error = float(np.linalg.norm(ball_only_xyz[:2] - truth))
        rows.append(
            {
                "file": file_name,
                "group": sample.group,
                "true_x": float(truth[0]),
                "true_y": float(truth[1]),
                "ball_racket_x": float(ball_racket_xyz[0]),
                "ball_racket_y": float(ball_racket_xyz[1]),
                "ball_only_x": float(ball_only_xyz[0]),
                "ball_only_y": float(ball_only_xyz[1]),
                "ball_racket_error_xy_cm": ball_racket_error,
                "ball_only_error_xy_cm": ball_only_error,
                "improvement_cm": ball_only_error - ball_racket_error,
            }
        )

    summary = {
        "sample_count": len(rows),
        "group_count": len(set(row["group"] for row in rows)),
        "ball_racket_error_xy_cm": metric(
            [row["ball_racket_error_xy_cm"] for row in rows]
        ),
        "ball_only_error_xy_cm": metric(
            [row["ball_only_error_xy_cm"] for row in rows]
        ),
        "ball_racket_better_count": int(sum(row["improvement_cm"] > 0 for row in rows)),
    }
    payload = {
        "config": {
            "data_dir": str(args.data_dir.resolve()),
            "validation_index": str(args.validation_index.resolve()),
            "ball_racket_model": str(args.ball_racket_model.resolve()),
            "ball_only_model": str(args.ball_only_model.resolve()),
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
