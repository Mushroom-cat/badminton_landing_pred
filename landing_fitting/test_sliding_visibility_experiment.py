import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import landing_fitting.sliding_visibility_experiment as sve
from landing_fitting.sliding_visibility_experiment import (
    load_test_samples,
    run_experiment,
    summarize_rows,
    visible_segment,
)


class SlidingVisibilitySummaryTests(unittest.TestCase):
    def test_summaries_are_descending_and_exclude_failed_predictions(self):
        rows = [
            {
                "trajectory_group": "trajectory_a",
                "visible_frames": 20,
                "mlp_xy_error": 4.0,
                "mlp_status": "ok",
                "direct_xy_error": 8.0,
                "direct_status": "ok",
            },
            {
                "trajectory_group": "trajectory_a",
                "visible_frames": 21,
                "mlp_xy_error": 2.0,
                "mlp_status": "ok",
                "direct_xy_error": 6.0,
                "direct_status": "ok",
            },
            {
                "trajectory_group": "trajectory_b",
                "visible_frames": 21,
                "mlp_xy_error": 4.0,
                "mlp_status": "ok",
                "direct_xy_error": "",
                "direct_status": "failed: fit error",
            },
        ]

        summaries = summarize_rows(rows)

        self.assertEqual([row["visible_frames"] for row in summaries], [21, 20])
        self.assertEqual(summaries[0]["mlp_mean"], 3.0)
        self.assertEqual(summaries[0]["direct_mean"], 6.0)
        self.assertEqual(summaries[0]["direct_success"], 1)
        self.assertEqual(summaries[0]["direct_failed"], 1)


class MissingTestFileTests(unittest.TestCase):
    def test_missing_checkpoint_test_files_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            present = data_dir / "present.txt"
            present.write_text(
                "\n".join(
                    [
                        "0:0,0,3",
                        "1:1,0,2",
                        "2:2,0,1",
                        "3:3,0,0",
                    ]
                ),
                encoding="utf-8",
            )
            predictor = SimpleNamespace(splits={"test": ["present.txt", "missing.txt"]})

            with self.assertRaises(FileNotFoundError):
                load_test_samples(data_dir, predictor)

            samples = load_test_samples(data_dir, predictor, allow_missing=True)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].path.name, "present.txt")


class VisibleSegmentTests(unittest.TestCase):
    def test_prefix_and_suffix_select_matching_points_and_frame_ids(self):
        sample = SimpleNamespace(
            points=np.asarray(
                [
                    [0.0, 0.0, 4.0],
                    [1.0, 0.0, 3.0],
                    [2.0, 0.0, 2.0],
                    [3.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
            frame_ids=np.asarray([10, 11, 12, 13, 14]),
        )

        points, frame_ids, start_index, end_index = visible_segment(
            sample,
            3,
            "prefix",
        )
        np.testing.assert_allclose(points[:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(frame_ids, [10, 11, 12])
        self.assertEqual(start_index, 0)
        self.assertEqual(end_index, 2)

        points, frame_ids, start_index, end_index = visible_segment(
            sample,
            3,
            "suffix",
        )
        np.testing.assert_allclose(points[:, 0], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(frame_ids, [12, 13, 14])
        self.assertEqual(start_index, 2)
        self.assertEqual(end_index, 4)


class SlidingVisibilityRowTests(unittest.TestCase):
    def test_suffix_rows_record_alignment_and_visible_indices(self):
        sample = SimpleNamespace(
            path=Path("sample.txt"),
            points=np.asarray(
                [
                    [0.0, 0.0, 4.0],
                    [1.0, 0.0, 3.0],
                    [2.0, 0.0, 2.0],
                    [3.0, 0.0, 1.0],
                ]
            ),
            frame_ids=np.asarray([0, 1, 2, 3]),
            landing_xyz=np.asarray([3.0, 0.0, 0.0]),
        )
        predictor = SimpleNamespace(state_window_size=3)
        prediction = (np.asarray([3.0, 0.0, 0.0]), 100.0, 0.0)

        with patch.object(sve, "predict_mlp", return_value=prediction), patch.object(
            sve,
            "predict_direct",
            return_value=prediction,
        ):
            rows, max_visible = run_experiment(
                [sample],
                predictor,
                fps=300.0,
                landing_z=0.0,
                min_visible_frames=3,
                alignment="suffix",
            )

        self.assertEqual(max_visible, 4)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["alignment"], "suffix")
        self.assertEqual(rows[0]["visible_frames"], 4)
        self.assertEqual(rows[0]["visible_start_index"], 0)
        self.assertEqual(rows[0]["visible_end_index"], 3)
        self.assertEqual(rows[1]["alignment"], "suffix")
        self.assertEqual(rows[1]["visible_frames"], 3)
        self.assertEqual(rows[1]["visible_start_index"], 1)
        self.assertEqual(rows[1]["visible_end_index"], 3)


if __name__ == "__main__":
    unittest.main()
