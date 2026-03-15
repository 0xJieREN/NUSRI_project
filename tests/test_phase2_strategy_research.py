from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from nusri_project.strategy.phase2_strategy_research import (
    build_scan_profile,
    build_parameter_grid,
    find_prediction_files,
    rank_scan_results,
    select_top_feasible_candidates,
)


class Phase2StrategyResearchTests(unittest.TestCase):
    def test_find_prediction_files_sorts_yearly_prediction_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            for name in ("pred_202403.pkl", "pred_202401.pkl", "pred_202402.pkl", "pred_202501.pkl"):
                path = root / ("a" if "03" in name else "b") / name
                path.write_text("x")

            paths = find_prediction_files(root, year=2024)

        self.assertEqual([path.name for path in paths], ["pred_202401.pkl", "pred_202402.pkl", "pred_202403.pkl"])

    def test_build_parameter_grid_filters_invalid_combinations(self) -> None:
        grid = build_parameter_grid(
            entry_thresholds=[0.001, 0.002],
            exit_thresholds=[0.0, 0.003],
            full_position_thresholds=[0.0015, 0.004],
            min_holding_hours_list=[12],
            cooldown_hours_list=[6],
            drawdown_thresholds=[0.08],
            de_risk_positions=[0.5],
        )

        self.assertEqual(
            grid,
            [
                {
                    "entry_threshold": 0.001,
                    "exit_threshold": 0.0,
                    "full_position_threshold": 0.0015,
                    "max_position": 1.0,
                    "min_holding_hours": 12,
                    "cooldown_hours": 6,
                    "drawdown_de_risk_threshold": 0.08,
                    "de_risk_position": 0.5,
                },
                {
                    "entry_threshold": 0.001,
                    "exit_threshold": 0.0,
                    "full_position_threshold": 0.004,
                    "max_position": 1.0,
                    "min_holding_hours": 12,
                    "cooldown_hours": 6,
                    "drawdown_de_risk_threshold": 0.08,
                    "de_risk_position": 0.5,
                },
                {
                    "entry_threshold": 0.002,
                    "exit_threshold": 0.0,
                    "full_position_threshold": 0.004,
                    "max_position": 1.0,
                    "min_holding_hours": 12,
                    "cooldown_hours": 6,
                    "drawdown_de_risk_threshold": 0.08,
                    "de_risk_position": 0.5,
                },
            ],
        )

    def test_rank_scan_results_prioritizes_feasible_candidates_then_sharpe(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "a",
                    "annualized_return": 0.05,
                    "max_drawdown": 0.08,
                    "sharpe": 0.4,
                    "calmar": 0.5,
                },
                {
                    "candidate_id": "b",
                    "annualized_return": 0.03,
                    "max_drawdown": 0.06,
                    "sharpe": 1.2,
                    "calmar": 0.9,
                },
                {
                    "candidate_id": "c",
                    "annualized_return": 0.06,
                    "max_drawdown": 0.07,
                    "sharpe": 0.8,
                    "calmar": 0.7,
                },
            ]
        )

        ranked = rank_scan_results(frame, min_annualized_return=0.04, max_drawdown=0.10)

        self.assertEqual(list(ranked["candidate_id"]), ["c", "a", "b"])
        self.assertEqual(list(ranked["meets_constraints"]), [True, True, False])

    def test_build_scan_profile_conservative_includes_lower_max_position(self) -> None:
        grid = build_scan_profile("conservative")

        self.assertTrue(len(grid) > 0)
        self.assertTrue(all(candidate["max_position"] <= 0.5 for candidate in grid))
        self.assertTrue(all(candidate["drawdown_de_risk_threshold"] <= 0.05 for candidate in grid))
        self.assertTrue(all(candidate["entry_threshold"] >= 0.001 for candidate in grid))

    def test_build_scan_profile_conservative_fast_stays_small(self) -> None:
        grid = build_scan_profile("conservative_fast")

        self.assertTrue(0 < len(grid) <= 32)
        self.assertTrue(all(candidate["max_position"] <= 0.35 for candidate in grid))

    def test_build_scan_profile_label72_trade_tuning_uses_targeted_ranges(self) -> None:
        grid = build_scan_profile("label72_trade_tuning")

        self.assertTrue(0 < len(grid) <= 512)
        self.assertTrue(all(candidate["entry_threshold"] in {0.0015, 0.003, 0.005, 0.008} for candidate in grid))
        self.assertTrue(all(candidate["max_position"] in {0.10, 0.15, 0.20, 0.25, 0.35, 0.50} for candidate in grid))
        self.assertTrue(all(candidate["min_holding_hours"] in {48, 72, 96} for candidate in grid))

    def test_build_scan_profile_label72_trade_tuning_fast_stays_small(self) -> None:
        grid = build_scan_profile("label72_trade_tuning_fast")

        self.assertTrue(0 < len(grid) <= 64)
        self.assertTrue(all(candidate["entry_threshold"] in {0.0015, 0.003, 0.005} for candidate in grid))
        self.assertTrue(all(candidate["max_position"] in {0.15, 0.25, 0.35} for candidate in grid))

    def test_select_top_feasible_candidates_prefers_sharpe_then_return(self) -> None:
        frame = pd.DataFrame(
            [
                {"candidate_id": "a", "annualized_return": 0.05, "max_drawdown": 0.08, "sharpe": 1.0, "calmar": 0.7},
                {"candidate_id": "b", "annualized_return": 0.07, "max_drawdown": 0.09, "sharpe": 1.2, "calmar": 0.8},
                {"candidate_id": "c", "annualized_return": 0.06, "max_drawdown": 0.11, "sharpe": 2.0, "calmar": 1.0},
                {"candidate_id": "d", "annualized_return": 0.09, "max_drawdown": 0.07, "sharpe": 1.2, "calmar": 0.9},
            ]
        )

        selected = select_top_feasible_candidates(frame, limit=2)

        self.assertEqual(list(selected["candidate_id"]), ["d", "b"])


if __name__ == "__main__":
    unittest.main()
