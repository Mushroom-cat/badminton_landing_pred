#!/usr/bin/env python3
"""Compare legacy, ball-only and ball+racket after models on five samples."""

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
from util.after_ball_racket import (  # noqa: E402
    RACKET_SLICE,
    extract_ball_racket_motion_features,
)


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
        "--legacy-model", type=Path, default=ROOT / "models" / "after.pt"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "results"
        / "after_ball_racket_trend"
        / "five_sample_validation.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT
        / "results"
        / "after_ball_racket_trend"
        / "five_sample_validation.csv",
    )
    parser.add_argument("--window-size", type=int, default=50)
    return parser.parse_args()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def angle_to_motion(point_xy, last_xy, direction):
    displacement = point_xy - last_xy
    norm = float(np.linalg.norm(displacement))
    cosine = float(np.dot(displacement, direction) / max(norm, 1e-9))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def main() -> None:
    args = parse_args()
    with args.comparison_csv.open(encoding="utf-8", newline="") as file:
        comparison = {row["sample_id"]: row for row in csv.DictReader(file)}
    ball_racket_predictor = load_pose_predictor(args.ball_racket_model, None)
    ball_only_predictor = load_pose_predictor(args.ball_only_model, None)
    legacy_predictor = load_pose_predictor(args.legacy_model, None)

    rows = []
    for path in sorted(args.pose_dir.glob("*.txt")):
        sample_id = path.stem.replace("---", "_")
        reference = comparison[sample_id]
        fit_xy = np.asarray(
            [float(reference["fit_pred_x"]), float(reference["fit_pred_y"])]
        )
        sequence = load_pose_sequence(path)
        start, end = pose_window_bounds(sequence, args.window_size, 0)
        frame_ids = sequence.frame_ids[start:end]
        window = select_pose_window(sequence, "after", args.window_size, 0)
        motion = extract_ball_racket_motion_features(window, frame_ids)

        ball_racket_details = ball_racket_predictor.predict_with_details(
            window, frame_ids
        )
        ball_racket_xyz = ball_racket_details["landing_xyz"]
        ball_only_xyz = ball_only_predictor.predict(window, frame_ids)
        legacy_xyz = legacy_predictor.predict(window, frame_ids)

        reversed_ball = window.copy()
        reversed_ball[motion.ball_motion.selected_indices, 63:66] = (
            motion.ball_motion.selected_ball[::-1]
        )
        reversed_ball_xyz = ball_racket_predictor.predict(
            reversed_ball, frame_ids
        )

        racket_indices = motion.ball_motion.selected_indices
        selected_rackets = window[racket_indices, RACKET_SLICE].copy()
        reversed_racket = window.copy()
        reversed_racket[racket_indices, RACKET_SLICE] = selected_rackets[::-1]
        reversed_racket_xyz = ball_racket_predictor.predict(
            reversed_racket, frame_ids
        )
        frozen_racket = window.copy()
        frozen_racket[racket_indices, RACKET_SLICE] = selected_rackets[-1]
        frozen_racket_xyz = ball_racket_predictor.predict(
            frozen_racket, frame_ids
        )

        def error(point):
            return float(np.linalg.norm(point[:2] - fit_xy))

        row = {
            "sample_id": sample_id,
            "ball_point_count": int(len(motion.ball_motion.selected_ball)),
            "valid_racket_frame_count": motion.valid_racket_count,
            "fit_x": float(fit_xy[0]),
            "fit_y": float(fit_xy[1]),
            "ball_racket_x": float(ball_racket_xyz[0]),
            "ball_racket_y": float(ball_racket_xyz[1]),
            "ball_only_x": float(ball_only_xyz[0]),
            "ball_only_y": float(ball_only_xyz[1]),
            "legacy_x": float(legacy_xyz[0]),
            "legacy_y": float(legacy_xyz[1]),
            "ball_racket_error_to_fit_cm": error(ball_racket_xyz),
            "ball_only_error_to_fit_cm": error(ball_only_xyz),
            "legacy_error_to_fit_cm": error(legacy_xyz),
            "ball_racket_angle_to_motion_deg": angle_to_motion(
                ball_racket_xyz[:2],
                motion.ball_motion.last_ball[:2],
                motion.ball_motion.direction_xy,
            ),
            "fit_angle_to_motion_deg": angle_to_motion(
                fit_xy,
                motion.ball_motion.last_ball[:2],
                motion.ball_motion.direction_xy,
            ),
            "ball_reverse_shift_cm": float(
                np.linalg.norm(reversed_ball_xyz[:2] - ball_racket_xyz[:2])
            ),
            "racket_reverse_shift_cm": float(
                np.linalg.norm(reversed_racket_xyz[:2] - ball_racket_xyz[:2])
            ),
            "racket_frozen_shift_cm": float(
                np.linalg.norm(frozen_racket_xyz[:2] - ball_racket_xyz[:2])
            ),
            "racket_gate_forward": float(ball_racket_details["racket_gate"][0]),
            "racket_gate_lateral": float(ball_racket_details["racket_gate"][1]),
        }
        rows.append(row)

    summary = {
        "sample_count": len(rows),
        "reference": "landing-fitting prediction (proxy; no measured ground-truth landing in these five files)",
        "ball_racket_error_to_fit_cm": stats(
            [row["ball_racket_error_to_fit_cm"] for row in rows]
        ),
        "ball_only_error_to_fit_cm": stats(
            [row["ball_only_error_to_fit_cm"] for row in rows]
        ),
        "legacy_error_to_fit_cm": stats(
            [row["legacy_error_to_fit_cm"] for row in rows]
        ),
        "ball_racket_better_than_ball_only_count": int(
            sum(
                row["ball_racket_error_to_fit_cm"]
                < row["ball_only_error_to_fit_cm"]
                for row in rows
            )
        ),
        "ball_reverse_shift_cm": stats(
            [row["ball_reverse_shift_cm"] for row in rows]
        ),
        "racket_reverse_shift_cm": stats(
            [row["racket_reverse_shift_cm"] for row in rows]
        ),
        "racket_frozen_shift_cm": stats(
            [row["racket_frozen_shift_cm"] for row in rows]
        ),
        "racket_gate_forward": stats(
            [row["racket_gate_forward"] for row in rows]
        ),
        "racket_gate_lateral": stats(
            [row["racket_gate_lateral"] for row in rows]
        ),
    }
    payload = {
        "config": {
            "pose_dir": str(args.pose_dir.resolve()),
            "ball_racket_model": str(args.ball_racket_model.resolve()),
            "ball_only_model": str(args.ball_only_model.resolve()),
            "legacy_model": str(args.legacy_model.resolve()),
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
