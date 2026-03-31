from __future__ import annotations

from datetime import datetime
import unittest

import pandas as pd

from nusri_project.config.schemas import TrainingConfig
from nusri_project.training.lgbm_workflow import _build_reweighter
from nusri_project.training.time_decay_reweighter import ExpHalflifeReweighter


class ExpHalflifeReweighterTests(unittest.TestCase):
    def test_exp_halflife_reweighter_is_monotonic_and_mean_one(self) -> None:
        frame = pd.DataFrame(
            {"label": [0, 1, 0]},
            index=pd.MultiIndex.from_arrays(
                [
                    pd.to_datetime(
                        [
                            "2024-07-31 23:00:00",
                            "2024-10-31 23:00:00",
                            "2025-01-31 23:00:00",
                        ]
                    ),
                    ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
                ]
            ),
        )

        weights = ExpHalflifeReweighter(
            reference_time="2025-01-31 23:00:00",
            half_life_months=6,
        ).reweight(frame)

        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_build_reweighter_returns_none_for_uniform_weighting(self) -> None:
        training = TrainingConfig(
            run_mode="rolling",
            training_window_months=24,
            rolling_step_months=1,
            sample_weight_mode="uniform",
        )

        reweighter = _build_reweighter(training, datetime(2025, 1, 31, 23, 0, 0))

        self.assertIsNone(reweighter)

    def test_build_reweighter_rejects_unknown_mode(self) -> None:
        training = TrainingConfig(
            run_mode="rolling",
            training_window_months=24,
            rolling_step_months=1,
            sample_weight_mode="mystery_mode",
        )

        with self.assertRaises(ValueError):
            _build_reweighter(training, datetime(2025, 1, 31, 23, 0, 0))


if __name__ == "__main__":
    unittest.main()
