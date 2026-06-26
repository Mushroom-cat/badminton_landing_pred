import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from landing_fitting.fit_trajectory import main, run_batch, run_prefix_predictions


class BatchPredictionTests(unittest.TestCase):
    def test_mlp_batch_uses_predictor_for_every_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "a.txt").write_text("unused", encoding="utf-8")
            (input_dir / "b.txt").write_text("unused", encoding="utf-8")
            predictor = object()
            predictions = [
                {"xy_error": 1.0, "xyz_error": 2.0},
                {"xy_error": 3.0, "xyz_error": 4.0},
            ]

            with patch(
                "landing_fitting.fit_trajectory.run_mlp",
                side_effect=predictions,
            ) as run_mlp_mock, redirect_stdout(io.StringIO()):
                result = run_batch(
                    input_dir,
                    300.0,
                    0.0,
                    parameter_mode="mlp",
                    predictor=predictor,
                    state_window_size=30,
                )

            self.assertEqual(run_mlp_mock.call_count, 2)
            self.assertEqual(result["average_xy_error"], 2.0)
            self.assertEqual(result["average_xyz_error"], 3.0)
            for call in run_mlp_mock.call_args_list:
                self.assertIs(call.args[3], predictor)
                self.assertEqual(call.args[4], 30)

    def test_sliding_batch_uses_visible_prefix_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "sample.txt").write_text("unused", encoding="utf-8")
            sliding_result = {
                "rows": [
                    {"status": "ok", "xy_error": 4.0, "xyz_error": 5.0},
                    {"status": "ok", "xy_error": 2.0, "xyz_error": 3.0},
                ]
            }

            with patch(
                "landing_fitting.fit_trajectory.run_prefix_predictions",
                return_value=sliding_result,
            ) as sliding_mock, redirect_stdout(io.StringIO()):
                result = run_batch(
                    input_dir,
                    300.0,
                    0.0,
                    sliding=True,
                    parameter_mode="curve-fit",
                )

            sliding_mock.assert_called_once()
            self.assertEqual(result["average_best_xy_error"], 2.0)
            self.assertEqual(result["average_last_xy_error"], 2.0)

    def test_curve_fit_sliding_uses_growing_visible_prefixes(self):
        observed = np.asarray(
            [
                [0.0, 0.0, 100.0],
                [1.0, 0.0, 90.0],
                [2.0, 0.0, 80.0],
                [3.0, 0.0, 70.0],
                [4.0, 0.0, 60.0],
            ],
            dtype=np.float64,
        )
        label = np.asarray([5.0, 0.0, 0.0], dtype=np.float64)
        seen_lengths = []

        def fake_predict(prefix, _label, _fps, _landing_z):
            seen_lengths.append(len(prefix))
            value = float(len(prefix))
            return {
                "landing_time_ms": value,
                "predicted": np.asarray([value, 0.0, 0.0], dtype=np.float64),
                "xy_error": value,
                "xyz_error": value,
                "x_params": np.asarray([0.0, 0.0, value], dtype=np.float64),
                "y_params": np.asarray([0.0, 0.0, value], dtype=np.float64),
                "z_params": np.asarray([0.0, 0.0, value], dtype=np.float64),
            }

        with patch(
            "landing_fitting.fit_trajectory.load_trajectory",
            return_value=(observed, label),
        ), patch(
            "landing_fitting.fit_trajectory.predict_visible_prefix",
            side_effect=fake_predict,
        ):
            result = run_prefix_predictions(Path("unused.txt"), 300.0, 0.0)

        self.assertEqual(seen_lengths, [4, 5])
        self.assertEqual([row["visible_points"] for row in result["rows"]], [4, 5])
        self.assertEqual(result["rows"][0]["prefix_end_index"], 3)


class InputDispatchTests(unittest.TestCase):
    def make_args(self, input_path):
        return SimpleNamespace(
            parameter_mode="curve-fit",
            input=input_path,
            fps=300.0,
            landing_z=0.0,
            sliding=False,
            state_window_size=20,
        )

    def test_main_dispatches_directory_to_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_args(Path(temp_dir))
            with patch(
                "landing_fitting.fit_trajectory.parse_args", return_value=args
            ), patch("landing_fitting.fit_trajectory.run_batch") as batch_mock:
                main()

            batch_mock.assert_called_once()

    def test_main_preserves_single_file_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "sample.txt"
            input_path.write_text("unused", encoding="utf-8")
            args = self.make_args(input_path)
            result = {"parameter_mode": "curve-fit"}
            with patch(
                "landing_fitting.fit_trajectory.parse_args", return_value=args
            ), patch(
                "landing_fitting.fit_trajectory.run", return_value=result
            ) as run_mock, patch(
                "landing_fitting.fit_trajectory.print_result"
            ) as print_mock, patch(
                "landing_fitting.fit_trajectory.run_batch"
            ) as batch_mock:
                main()

            run_mock.assert_called_once_with(input_path, 300.0, 0.0)
            print_mock.assert_called_once_with(result, 300.0, 0.0)
            batch_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
