import unittest

from landing_fitting.sliding_visibility_experiment import summarize_rows


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


if __name__ == "__main__":
    unittest.main()
