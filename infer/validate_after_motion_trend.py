#!/usr/bin/env python3
"""Validate the motion-aware after model on the five external duida samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer.compare_duida_pose_fit_predictions import (  # noqa: E402
    load_pose_predictor,
    load_pose_sequence,
    pose_window_bounds,
    select_pose_window,
)
from util.after_motion import extract_ball_motion_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=ROOT / "datasets" / "2026-7-10test" / "4-poseball",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=ROOT / "infer" / "duida_pose_fit_prediction_comparison.csv",
    )
    parser.add_argument(
        "--new-model",
        type=Path,
        default=ROOT / "models" / "after_motion_trend.pt",
    )
    parser.add_argument(
        "--legacy-model",
        type=Path,
        default=ROOT / "models" / "after.pt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "results" / "after_motion_trend" / "five_sample_validation.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "after_motion_trend" / "five_sample_validation.csv",
    )
    parser.add_argument("--window-size", type=int, default=50)
    return parser.parse_args()


def angle_to_motion(point_xy: np.ndarray, last_xy: np.ndarray, direction: np.ndarray) -> float:
    displacement = point_xy - last_xy
    norm = float(np.linalg.norm(displacement))
    if norm < 1e-9:
        return 0.0
    cosine = float(np.dot(displacement, direction) / norm)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    with args.comparison_csv.open(encoding="utf-8", newline="") as file:
        comparison = {row["sample_id"]: row for row in csv.DictReader(file)}
    new_predictor = load_pose_predictor(args.new_model, None)
    legacy_predictor = load_pose_predictor(args.legacy_model, None)

    rows = []
    for path in sorted(args.pose_dir.glob("*.txt")):
        sample_id = path.stem.replace("---", "_")
        if sample_id not in comparison:
            raise ValueError(f"{sample_id} is missing from {args.comparison_csv}")
        reference = comparison[sample_id]
        fit_xy = np.asarray(
            [float(reference["fit_pred_x"]), float(reference["fit_pred_y"])],
            dtype=np.float64,
        )
        sequence = load_pose_sequence(path)
        start, end = pose_window_bounds(sequence, args.window_size, 0)
        frame_ids = sequence.frame_ids[start:end]
        window = select_pose_window(sequence, "after", args.window_size, 0)
        motion = extract_ball_motion_features(window, frame_ids)

        new_xyz = new_predictor.predict(window, frame_ids)
        legacy_xyz = legacy_predictor.predict(window, frame_ids)
        reversed_window = window.copy()
        reversed_window[motion.selected_indices, 63:66] = motion.selected_ball[::-1]
        reversed_new_xyz = new_predictor.predict(reversed_window, frame_ids)
        reversed_legacy_xyz = legacy_predictor.predict(reversed_window, frame_ids)

        new_error = float(np.linalg.norm(new_xyz[:2] - fit_xy))
        legacy_error = float(np.linalg.norm(legacy_xyz[:2] - fit_xy))
        row = {
            "sample_id": sample_id,
            "ball_point_count": int(len(motion.selected_ball)),
            "motion_direction_x": float(motion.direction_xy[0]),
            "motion_direction_y": float(motion.direction_xy[1]),
            "fit_x": float(fit_xy[0]),
            "fit_y": float(fit_xy[1]),
            "new_x": float(new_xyz[0]),
            "new_y": float(new_xyz[1]),
            "legacy_x": float(legacy_xyz[0]),
            "legacy_y": float(legacy_xyz[1]),
            "new_error_to_fit_cm": new_error,
            "legacy_error_to_fit_cm": legacy_error,
            "improvement_cm": legacy_error - new_error,
            "new_angle_to_motion_deg": angle_to_motion(
                new_xyz[:2], motion.last_ball[:2], motion.direction_xy
            ),
            "legacy_angle_to_motion_deg": angle_to_motion(
                legacy_xyz[:2], motion.last_ball[:2], motion.direction_xy
            ),
            "fit_angle_to_motion_deg": angle_to_motion(
                fit_xy, motion.last_ball[:2], motion.direction_xy
            ),
            "new_reverse_shift_cm": float(
                np.linalg.norm(reversed_new_xyz[:2] - new_xyz[:2])
            ),
            "legacy_reverse_shift_cm": float(
                np.linalg.norm(reversed_legacy_xyz[:2] - legacy_xyz[:2])
            ),
        }
        rows.append(row)

    summary = {
        "sample_count": len(rows),
        "reference": "landing-fitting prediction (proxy; the five files have no measured ground-truth landing)",
        "new_error_to_fit_cm": stats([row["new_error_to_fit_cm"] for row in rows]),
        "legacy_error_to_fit_cm": stats([row["legacy_error_to_fit_cm"] for row in rows]),
        "improved_sample_count": int(sum(row["improvement_cm"] > 0 for row in rows)),
        "new_angle_to_motion_deg": stats([row["new_angle_to_motion_deg"] for row in rows]),
        "legacy_angle_to_motion_deg": stats([row["legacy_angle_to_motion_deg"] for row in rows]),
        "fit_angle_to_motion_deg": stats([row["fit_angle_to_motion_deg"] for row in rows]),
        "new_reverse_shift_cm": stats([row["new_reverse_shift_cm"] for row in rows]),
        "legacy_reverse_shift_cm": stats([row["legacy_reverse_shift_cm"] for row in rows]),
    }
    payload = {
        "config": {
            "pose_dir": str(args.pose_dir.resolve()),
            "comparison_csv": str(args.comparison_csv.resolve()),
            "new_model": str(args.new_model.resolve()),
            "legacy_model": str(args.legacy_model.resolve()),
            "window_size": args.window_size,
        },
        "summary": summary,
        "rows": rows,
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
