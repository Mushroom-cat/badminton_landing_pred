import unittest

import numpy as np

from landing_fitting.search_c_mlp_hyperparameters import (
    BASELINE_CONFIG,
    config_signature,
    confirmation_rank,
    generate_search_configs,
    summarize_confirmations,
    trial_rank,
)
from landing_fitting.train_c_mlp import train_model


class SearchConfigurationTests(unittest.TestCase):
    def test_search_configs_are_unique_and_include_baseline(self):
        configs = generate_search_configs(36, seed=42)

        self.assertEqual(len(configs), 36)
        self.assertEqual(configs[0], BASELINE_CONFIG)
        self.assertEqual(len({config_signature(config) for config in configs}), 36)

    def test_trial_ranking_prioritizes_complete_predictions(self):
        complete = {
            "val_success_groups": 8,
            "val_total_groups": 8,
            "val_mean_xy_cm": 12.0,
        }
        incomplete = {
            "val_success_groups": 7,
            "val_total_groups": 8,
            "val_mean_xy_cm": 1.0,
        }

        self.assertLess(trial_rank(complete), trial_rank(incomplete))

    def test_confirmation_uses_seed_mean_and_standard_deviation(self):
        rows = [
            {
                "val_success_groups": 8,
                "val_total_groups": 8,
                "val_mean_xy_cm": value,
            }
            for value in (10.0, 12.0, 14.0)
        ]

        summary = summarize_confirmations(3, BASELINE_CONFIG, rows)

        self.assertEqual(summary["failed_seed_runs"], 0)
        self.assertEqual(summary["seed_mean_xy_cm"], 12.0)
        self.assertAlmostEqual(summary["seed_std_xy_cm"], np.std([10.0, 12.0, 14.0]))
        self.assertEqual(confirmation_rank(summary), (0, 12.0, summary["seed_std_xy_cm"]))


class SearchTrainingSmokeTests(unittest.TestCase):
    def test_short_training_returns_finite_metadata(self):
        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(32, 9)).astype(np.float32)
        train_y = rng.normal(size=(32, 3)).astype(np.float32)
        val_x = rng.normal(size=(12, 9)).astype(np.float32)
        val_y = rng.normal(size=(12, 3)).astype(np.float32)

        result = train_model(
            train_x,
            train_y,
            val_x,
            val_y,
            hidden_dims=(64, 64, 32),
            dropout=0.05,
            batch_size=16,
            lr=1e-3,
            weight_decay=1e-5,
            epochs=2,
            patience=1,
            seed=42,
            device="cpu",
        )

        self.assertGreaterEqual(result["best_epoch"], 1)
        self.assertTrue(np.isfinite(result["best_val_loss"]))
        self.assertTrue(result["model_state"])


if __name__ == "__main__":
    unittest.main()
