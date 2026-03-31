from __future__ import annotations

import unittest

from nusri_project.strategy.continuous_position_mapping import compute_target_weight_from_score_signal


class ContinuousPositionMappingTests(unittest.TestCase):
    def test_compute_target_weight_from_score_signal_uses_gate_and_power_curve(self) -> None:
        target = compute_target_weight_from_score_signal(
            pred_score=0.60,
            current_weight=0.0,
            max_position=0.25,
            open_score=0.40,
            close_score=0.20,
            size_floor_score=0.40,
            size_full_score=0.80,
            curve_gamma=1.0,
            min_holding_bars=0,
            holding_bars=0,
            cooldown_bars=0,
            bars_since_trade=99,
            drawdown=0.0,
            drawdown_de_risk_threshold=0.08,
            de_risk_position=0.0,
        )

        self.assertAlmostEqual(target, 0.125)

    def test_compute_target_weight_from_score_signal_applies_nonlinear_curve_gamma(self) -> None:
        target = compute_target_weight_from_score_signal(
            pred_score=0.60,
            current_weight=0.0,
            max_position=0.25,
            open_score=0.40,
            close_score=0.20,
            size_floor_score=0.40,
            size_full_score=0.80,
            curve_gamma=2.0,
            min_holding_bars=0,
            holding_bars=0,
            cooldown_bars=0,
            bars_since_trade=99,
            drawdown=0.0,
            drawdown_de_risk_threshold=0.08,
            de_risk_position=0.0,
        )

        self.assertAlmostEqual(target, 0.0625)

    def test_compute_target_weight_from_score_signal_stays_flat_between_open_gate_and_sizing_floor(self) -> None:
        target = compute_target_weight_from_score_signal(
            pred_score=0.50,
            current_weight=0.0,
            max_position=0.25,
            open_score=0.60,
            close_score=0.20,
            size_floor_score=0.40,
            size_full_score=0.80,
            curve_gamma=1.0,
            min_holding_bars=0,
            holding_bars=0,
            cooldown_bars=0,
            bars_since_trade=99,
            drawdown=0.0,
            drawdown_de_risk_threshold=0.08,
            de_risk_position=0.0,
        )

        self.assertEqual(target, 0.0)

    def test_compute_target_weight_from_score_signal_closes_when_score_hits_close_threshold(self) -> None:
        target = compute_target_weight_from_score_signal(
            pred_score=0.20,
            current_weight=0.25,
            max_position=0.25,
            open_score=0.40,
            close_score=0.20,
            size_floor_score=0.40,
            size_full_score=0.80,
            curve_gamma=1.5,
            min_holding_bars=0,
            holding_bars=0,
            cooldown_bars=0,
            bars_since_trade=99,
            drawdown=0.0,
            drawdown_de_risk_threshold=0.08,
            de_risk_position=0.0,
        )

        self.assertEqual(target, 0.0)


if __name__ == "__main__":
    unittest.main()
