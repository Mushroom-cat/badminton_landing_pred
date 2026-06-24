import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from landing_fitting.trajectory_mlp import (
    CParameterPredictor,
    FEATURE_NAMES,
    TARGET_NAMES,
    TrajectorySample,
    build_c_parameter_mlp,
    estimate_landing_frame,
    extract_state_features,
    fit_all_parameters,
    fit_complete_trajectory_labels,
    load_trajectory_file,
    safe_exp,
    x_model,
    y_model,
    z_model,
)


class StateFeatureTests(unittest.TestCase):
    def test_quadratic_state_features(self):
        fps = 10.0
        frame_ids = np.arange(20, dtype=np.int64)
        times = (frame_ids - frame_ids[-1]) / fps
        points = np.column_stack(
            [
                1.0 + 2.0 * times + 3.0 * times**2,
                -4.0 + 5.0 * times - 2.0 * times**2,
                8.0 - 3.0 * times + 0.5 * times**2,
            ]
        )

        features = extract_state_features(points, frame_ids, fps, window_size=20)

        expected = np.asarray(
            [1.0, -4.0, 8.0, 2.0, 5.0, -3.0, 6.0, -4.0, 1.0]
        )
        np.testing.assert_allclose(features, expected, atol=1e-9)


class LabelFitTests(unittest.TestCase):
    def test_all_parameter_fit_reproduces_noise_free_formula_points(self):
        fps = 300.0
        frame_ids = np.arange(40, dtype=np.int64)
        times_ms = frame_ids / fps * 1000.0
        points = np.column_stack(
            [
                x_model(times_ms, 500.0, -0.003, 120.0),
                y_model(times_ms, -80.0, 0.04, 30.0),
                z_model(times_ms, -400.0, -0.001, 650.0),
            ]
        )

        params = fit_all_parameters(points, frame_ids, fps)
        fitted = np.column_stack(
            [
                x_model(times_ms, *params["x"]),
                y_model(times_ms, *params["y"]),
                z_model(times_ms, *params["z"]),
            ]
        )

        np.testing.assert_allclose(fitted, points, atol=1e-4)

    def test_complete_formula_fit_recovers_c_parameters(self):
        fps = 300.0
        frame_ids = np.arange(70, dtype=np.int64)
        landing_frame = 180
        times_ms = frame_ids / fps * 1000.0
        landing_time_ms = landing_frame / fps * 1000.0

        x_params = (760.0, -0.003, 125.0)
        y_params = (-130.0, 0.08, -45.0)
        z_a = 260.0
        z_b = -0.0015
        z_c = 0.79 * landing_time_ms - z_a * safe_exp(z_b * landing_time_ms)
        z_params = (z_a, z_b, z_c)
        points = np.column_stack(
            [
                x_model(times_ms, *x_params),
                y_model(times_ms, *y_params),
                z_model(times_ms, *z_params),
            ]
        )
        landing_xyz = np.asarray(
            [
                x_model(landing_time_ms, *x_params),
                y_model(landing_time_ms, *y_params),
                0.0,
            ]
        )
        sample = TrajectorySample(
            frame_ids=frame_ids,
            points=points,
            landing_frame=landing_frame,
            landing_xyz=landing_xyz,
        )

        c_values, _ = fit_complete_trajectory_labels(sample, fps)

        np.testing.assert_allclose(
            c_values,
            [x_params[2], y_params[2], z_params[2]],
            atol=1e-3,
        )


class ParserTests(unittest.TestCase):
    def test_landing_frame_can_be_estimated_for_descending_segment(self):
        frame_ids = np.arange(20, dtype=np.int64)
        points = np.column_stack(
            [
                np.arange(20, dtype=np.float64),
                np.zeros(20),
                100.0 - 2.0 * frame_ids,
            ]
        )
        sample = TrajectorySample(
            frame_ids=frame_ids,
            points=points,
            landing_frame=None,
            landing_xyz=np.asarray([50.0, 0.0, 0.0]),
            has_explicit_frame_ids=False,
        )

        estimated = estimate_landing_frame(sample)

        self.assertEqual(estimated.landing_frame, 50)
        self.assertTrue(estimated.landing_frame_estimated)

    def test_mixed_observation_frame_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixed.txt"
            path.write_text(
                "0:1,2,3\n"
                "2,3,4\n"
                "2:3,4,5\n"
                "5:6,7,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frame ids"):
                load_trajectory_file(path, require_landing_frame=True)

    def test_landing_frame_must_follow_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "early_landing.txt"
            path.write_text(
                "0:1,2,3\n"
                "1:2,3,4\n"
                "2:3,4,5\n"
                "2:6,7,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be later"):
                load_trajectory_file(path, require_landing_frame=True)


class CheckpointCompatibilityTests(unittest.TestCase):
    def write_checkpoint(self, path, hidden_dims, include_hidden_dims):
        model = build_c_parameter_mlp(hidden_dims=hidden_dims, dropout=0.1)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "feature_mean": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            "feature_std": np.ones(len(FEATURE_NAMES), dtype=np.float32),
            "target_mean": np.zeros(len(TARGET_NAMES), dtype=np.float32),
            "target_std": np.ones(len(TARGET_NAMES), dtype=np.float32),
            "feature_names": FEATURE_NAMES,
            "state_window_size": 20,
            "fps": 300.0,
            "dropout": 0.1,
        }
        if include_hidden_dims:
            checkpoint["hidden_dims"] = list(hidden_dims)
        torch.save(checkpoint, path)

    def test_old_checkpoint_uses_default_hidden_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old.pt"
            self.write_checkpoint(path, (128, 128, 64), include_hidden_dims=False)

            predictor = CParameterPredictor(path)

            self.assertEqual(predictor.hidden_dims, (128, 128, 64))
            self.assertTrue(np.all(np.isfinite(predictor.predict(np.zeros(9)))))

    def test_configurable_hidden_layers_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "new.pt"
            self.write_checkpoint(path, (64, 64, 32), include_hidden_dims=True)

            predictor = CParameterPredictor(path)

            self.assertEqual(predictor.hidden_dims, (64, 64, 32))
            self.assertTrue(np.all(np.isfinite(predictor.predict(np.zeros(9)))))


if __name__ == "__main__":
    unittest.main()
