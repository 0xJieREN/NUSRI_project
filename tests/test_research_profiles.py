from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nusri_project.training.label_factory import (
    build_label_config,
    build_label_mode_config,
    get_backtest_target_expr,
    get_cost_aware_binary_label_expr,
    get_label_expr,
    get_prediction_output_column,
)
from nusri_project.training.model_factory import get_model_loss
from nusri_project.strategy.cost_aware_label_round1 import (
    build_cost_aware_round1_matrix,
    build_cost_aware_round1_modes,
    find_cost_aware_prediction_files,
)
from nusri_project.strategy.research_profiles import (
    build_probability_trading_shells,
    build_return_trading_shells,
    find_horizon_prediction_files,
)


class ResearchProfilesTests(unittest.TestCase):
    def test_get_label_expr_builds_expected_forward_return_expression(self) -> None:
        self.assertEqual(get_label_expr(24), "Ref($close, -24) / $close - 1")
        self.assertEqual(get_label_expr(48), "Ref($close, -48) / $close - 1")
        self.assertEqual(get_label_expr(72), "Ref($close, -72) / $close - 1")

    def test_build_label_config_uses_named_label_column(self) -> None:
        label_exprs, label_names = build_label_config(48)

        self.assertEqual(label_exprs, ["Ref($close, -48) / $close - 1"])
        self.assertEqual(label_names, ["label_48h"])

    def test_return_trading_shells_define_balanced_and_conservative_profiles(self) -> None:
        shells = build_return_trading_shells()

        self.assertEqual(set(shells.keys()), {"balanced", "conservative"})
        self.assertEqual(shells["balanced"]["max_position"], 0.25)
        self.assertEqual(shells["conservative"]["max_position"], 0.15)

    def test_probability_shells_define_balanced_and_conservative_profiles(self) -> None:
        shells = build_probability_trading_shells()

        self.assertEqual(set(shells.keys()), {"balanced", "conservative"})
        self.assertEqual(shells["balanced"]["signal_kind"], "probability")
        self.assertEqual(shells["balanced"]["enter_prob_threshold"], 0.65)
        self.assertEqual(shells["balanced"]["full_prob_threshold"], 0.80)

    def test_build_prediction_artifact_name_inputs_still_match_label_helpers(self) -> None:
        self.assertEqual(get_cost_aware_binary_label_expr(72, 0.005), "If(Gt(Ref($close, -72) / $close - 1, 0.005), 1, 0)")
        label_exprs, label_names = build_label_mode_config(
            label_mode="classification_72h_costaware",
            label_horizon_hours=72,
            positive_threshold=0.005,
        )
        self.assertEqual(label_exprs, ["If(Gt(Ref($close, -72) / $close - 1, 0.005), 1, 0)"])
        self.assertEqual(label_names, ["label_cls_72h_costaware"])
        self.assertEqual(get_backtest_target_expr(72), "Ref($close, -72) / $close - 1")
        self.assertEqual(get_model_loss("regression_72h"), "mse")
        self.assertEqual(get_model_loss("classification_72h_costaware"), "binary")
        self.assertEqual(get_prediction_output_column("regression_72h"), "pred_return")
        self.assertEqual(get_prediction_output_column("classification_72h_costaware"), "pred_prob")

    def test_find_horizon_prediction_files_filters_by_horizon_and_year(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "pred_24h_202401.pkl",
                "pred_24h_202402.pkl",
                "pred_48h_202401.pkl",
                "pred_24h_202501.pkl",
            ):
                (root / name).write_text("x")

            files = find_horizon_prediction_files(root, label_horizon_hours=24, year=2024)

        self.assertEqual([path.name for path in files], ["pred_24h_202401.pkl", "pred_24h_202402.pkl"])

    def test_find_horizon_prediction_files_falls_back_to_parent_when_horizon_dir_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "pred_72h_202401.pkl",
                "pred_72h_202402.pkl",
                "pred_72h_202501.pkl",
            ):
                (root / name).write_text("x")

            files = find_horizon_prediction_files(root / "72h", label_horizon_hours=72, year=2024)

        self.assertEqual([path.name for path in files], ["pred_72h_202401.pkl", "pred_72h_202402.pkl"])

    def test_cost_aware_round1_modes_and_matrix(self) -> None:
        self.assertEqual(build_cost_aware_round1_modes(), ["regression_72h", "classification_72h_costaware"])
        matrix = build_cost_aware_round1_matrix()
        self.assertEqual(len(matrix), 4)
        self.assertEqual(matrix[0], {"label_mode": "regression_72h", "shell_name": "balanced"})
        self.assertEqual(matrix[-1], {"label_mode": "classification_72h_costaware", "shell_name": "conservative"})

    def test_find_cost_aware_prediction_files_filters_by_mode_and_year(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "pred_classification_72h_costaware_72h_202401.pkl",
                "pred_classification_72h_costaware_72h_202402.pkl",
                "pred_regression_72h_72h_202401.pkl",
                "pred_classification_72h_costaware_72h_202501.pkl",
            ):
                (root / name).write_text("x")

            files = find_cost_aware_prediction_files(
                root,
                label_mode="classification_72h_costaware",
                label_horizon_hours=72,
                year=2024,
            )

        self.assertEqual(
            [path.name for path in files],
            [
                "pred_classification_72h_costaware_72h_202401.pkl",
                "pred_classification_72h_costaware_72h_202402.pkl",
            ],
        )


if __name__ == "__main__":
    unittest.main()
