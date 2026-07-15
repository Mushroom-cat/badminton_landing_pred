#!/usr/bin/env python3
"""Validate the new before model and its temporal feature use on five samples."""

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

from infer.compare_duida_pose_fit_predictions import (  # noqa: E402
    load_pose_predictor,
    load_pose_sequence,
    pose_window_bounds,
    select_pose_window,
)
from util.before_pose_racket import RACKET_SLICE  # noqa: E402


BODY_SLICE = slice(0, 17 * 3)


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
        / "five_sample_validation.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT
        / "results"
        / "before_pose_racket_trend"
        / "five_sample_validation.csv",
    )
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--skip-n", type=int, default=5)
    return parser.parse_args()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    with args.comparison_csv.open(encoding="utf-8", newline="") as file:
        comparison = {row["sample_id"]: row for row in csv.DictReader(file)}

    predictor = load_pose_predictor(args.new_model, None)
    legacy = load_pose_predictor(args.legacy_model, None)
    rows = []
    for path in sorted(args.pose_dir.glob("*.txt")):
        sample_id = path.stem.replace("---", "_")
        reference = comparison[sample_id]
        fit_xy = np.asarray(
            [float(reference["fit_pred_x"]), float(reference["fit_pred_y"])],
            dtype=np.float64,
        )
        sequence = load_pose_sequence(path)
        start, end = pose_window_bounds(sequence, args.window_size, args.skip_n)
        frame_ids = sequence.frame_ids[start:end]
        window = select_pose_window(
            sequence, "before", args.window_size, args.skip_n
        )
        details = predictor.predict_with_details(window, frame_ids)
        prediction = details["landing_xyz"]
        legacy_prediction = legacy.predict(window, frame_ids)
        selected = details["motion"].selected_indices

        reversed_racket = window.copy()
        reversed_racket[selected, RACKET_SLICE] = reversed_racket[
            selected, RACKET_SLICE
        ][::-1]
        slowed_racket = window.copy()
        selected_rackets = slowed_racket[selected, RACKET_SLICE].reshape(-1, 4, 3)
        racket_centers = np.mean(selected_rackets, axis=1, keepdims=True)
        slowed_centers = racket_centers[-1:] + 0.1 * (
            racket_centers - racket_centers[-1:]
        )
        slowed_rackets = selected_rackets - racket_centers + slowed_centers
        slowed_racket[selected, RACKET_SLICE] = slowed_rackets.reshape(-1, 12)
        reversed_body = window.copy()
        reversed_body[selected, BODY_SLICE] = reversed_body[
            selected, BODY_SLICE
        ][::-1]
        frozen_body = window.copy()
        frozen_body[selected, BODY_SLICE] = frozen_body[selected[-1], BODY_SLICE]

        reversed_racket_prediction = predictor.predict(reversed_racket, frame_ids)
        slowed_racket_prediction = predictor.predict(slowed_racket, frame_ids)
        reversed_body_prediction = predictor.predict(reversed_body, frame_ids)
        frozen_body_prediction = predictor.predict(frozen_body, frame_ids)

        def error_to_fit(point):
            return float(np.linalg.norm(point[:2] - fit_xy))

        def shift(point):
            return float(np.linalg.norm(point[:2] - prediction[:2]))

        rows.append(
            {
                "sample_id": sample_id,
                "fit_x": float(fit_xy[0]),
                "fit_y": float(fit_xy[1]),
                "new_x": float(prediction[0]),
                "new_y": float(prediction[1]),
                "legacy_x": float(legacy_prediction[0]),
                "legacy_y": float(legacy_prediction[1]),
                "new_error_to_fit_cm": error_to_fit(prediction),
                "legacy_error_to_fit_cm": error_to_fit(legacy_prediction),
                "racket_reverse_shift_cm": shift(reversed_racket_prediction),
                "racket_slowed_to_10pct_shift_cm": shift(slowed_racket_prediction),
                "body_reverse_shift_cm": shift(reversed_body_prediction),
                "body_frozen_shift_cm": shift(frozen_body_prediction),
                "body_gate_forward": float(details["body_gate"][0]),
                "body_gate_lateral": float(details["body_gate"][1]),
                "valid_racket_frame_count": int(len(selected)),
            }
        )

    summary = {
        "sample_count": len(rows),
        "reference": (
            "landing-fitting prediction (proxy only; the five pose files do not "
            "contain measured ground-truth landings)"
        ),
        "new_error_to_fit_cm": stats(
            [row["new_error_to_fit_cm"] for row in rows]
        ),
        "legacy_error_to_fit_cm": stats(
            [row["legacy_error_to_fit_cm"] for row in rows]
        ),
        "new_better_than_legacy_count": int(
            sum(
                row["new_error_to_fit_cm"] < row["legacy_error_to_fit_cm"]
                for row in rows
            )
        ),
        "racket_reverse_shift_cm": stats(
            [row["racket_reverse_shift_cm"] for row in rows]
        ),
        "racket_slowed_to_10pct_shift_cm": stats(
            [row["racket_slowed_to_10pct_shift_cm"] for row in rows]
        ),
        "body_reverse_shift_cm": stats(
            [row["body_reverse_shift_cm"] for row in rows]
        ),
        "body_frozen_shift_cm": stats(
            [row["body_frozen_shift_cm"] for row in rows]
        ),
        "body_gate_forward": stats(
            [row["body_gate_forward"] for row in rows]
        ),
        "body_gate_lateral": stats(
            [row["body_gate_lateral"] for row in rows]
        ),
    }
    payload = {
        "config": {
            "pose_dir": str(args.pose_dir.resolve()),
            "new_model": str(args.new_model.resolve()),
            "legacy_model": str(args.legacy_model.resolve()),
            "window_size": args.window_size,
            "skip_n": args.skip_n,
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
