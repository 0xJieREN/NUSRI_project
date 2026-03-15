from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nusri_project.training.lgbm_workflow import (
    build_label_config,
    build_label_mode_config,
    build_prediction_artifact_name,
    get_backtest_target_expr,
    get_cost_aware_binary_label_expr,
    get_label_expr,
    get_model_loss,
    transform_prediction_score,
)
from nusri_project.strategy.label_optimization_round1 import (
    build_round1_horizons,
    build_round1_run_plan,
    build_round1_trading_shells,
    build_round1_matrix,
    find_horizon_prediction_files,
)
from nusri_project.strategy.cost_aware_label_round1 import (
    build_cost_aware_round1_matrix,
    build_cost_aware_round1_modes,
    find_cost_aware_prediction_files,
)


class LabelOptimizationRound1Tests(unittest.TestCase):
    def test_get_label_expr_builds_expected_forward_return_expression(self) -> None:
        self.assertEqual(get_label_expr(24), "Ref($close, -24) / $close - 1")
        self.assertEqual(get_label_expr(48), "Ref($close, -48) / $close - 1")
        self.assertEqual(get_label_expr(72), "Ref($close, -72) / $close - 1")

    def test_build_label_config_uses_named_label_column(self) -> None:
        label_exprs, label_names = build_label_config(48)

        self.assertEqual(label_exprs, ["Ref($close, -48) / $close - 1"])
        self.assertEqual(label_names, ["label_48h"])

    def test_round1_horizons_are_fixed_to_three_regression_candidates(self) -> None:
        self.assertEqual(build_round1_horizons(), [24, 48, 72])

    def test_round1_trading_shells_define_balanced_and_conservative_profiles(self) -> None:
        shells = build_round1_trading_shells()

        self.assertEqual(set(shells.keys()), {"balanced", "conservative"})
        self.assertEqual(shells["balanced"]["max_position"], 0.25)
        self.assertEqual(shells["conservative"]["max_position"], 0.15)

    def test_round1_matrix_crosses_horizons_and_shells(self) -> None:
        matrix = build_round1_matrix()

        self.assertEqual(len(matrix), 6)
        self.assertEqual(
            matrix[0],
            {"label_horizon_hours": 24, "shell_name": "balanced"},
        )
        self.assertEqual(
            matrix[-1],
            {"label_horizon_hours": 72, "shell_name": "conservative"},
        )

    def test_build_prediction_artifact_name_encodes_horizon_and_month(self) -> None:
        self.assertEqual(build_prediction_artifact_name(24, "2024-03"), "pred_24h_202403.pkl")
        self.assertEqual(build_prediction_artifact_name(72, "2025-12"), "pred_72h_202512.pkl")

    def test_cost_aware_binary_label_expr_uses_72h_and_threshold(self) -> None:
        expr = get_cost_aware_binary_label_expr(label_horizon_hours=72, positive_threshold=0.005)

        self.assertEqual(expr, "If(Gt(Ref($close, -72) / $close - 1, 0.005), 1, 0)")

    def test_build_label_mode_config_for_costaware_binary(self) -> None:
        label_exprs, label_names = build_label_mode_config(
            label_mode="classification_72h_costaware",
            label_horizon_hours=72,
            positive_threshold=0.005,
        )

        self.assertEqual(label_exprs, ["If(Gt(Ref($close, -72) / $close - 1, 0.005), 1, 0)"])
        self.assertEqual(label_names, ["label_cls_72h_costaware"])

    def test_get_model_loss_switches_between_regression_and_binary(self) -> None:
        self.assertEqual(get_model_loss("regression_72h"), "mse")
        self.assertEqual(get_model_loss("classification_72h_costaware"), "binary")

    def test_get_backtest_target_expr_always_returns_continuous_future_return(self) -> None:
        self.assertEqual(get_backtest_target_expr(72), "Ref($close, -72) / $close - 1")

    def test_transform_prediction_score_centers_binary_probability_to_threshold_scale(self) -> None:
        self.assertAlmostEqual(
            transform_prediction_score(
                raw_prediction=0.8,
                label_mode="classification_72h_costaware",
                positive_threshold=0.005,
            ),
            0.003,
            places=9,
        )
        self.assertAlmostEqual(
            transform_prediction_score(
                raw_prediction=0.2,
                label_mode="classification_72h_costaware",
                positive_threshold=0.005,
            ),
            -0.003,
            places=9,
        )

    def test_build_prediction_artifact_name_can_encode_label_mode(self) -> None:
        self.assertEqual(
            build_prediction_artifact_name(72, "2024-03", label_mode="classification_72h_costaware"),
            "pred_classification_72h_costaware_72h_202403.pkl",
        )

    def test_build_round1_run_plan_creates_horizon_specific_prediction_dirs(self) -> None:
        plan = build_round1_run_plan(Path("reports/label_round1"))

        self.assertEqual(len(plan), 6)
        self.assertEqual(plan[0]["label_horizon_hours"], 24)
        self.assertEqual(plan[0]["shell_name"], "balanced")
        self.assertEqual(plan[0]["prediction_dir"], Path("reports/label_round1/predictions/24h"))
        self.assertEqual(plan[-1]["prediction_dir"], Path("reports/label_round1/predictions/72h"))

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

    def test_cost_aware_round1_modes_are_fixed_to_regression_and_binary(self) -> None:
        self.assertEqual(build_cost_aware_round1_modes(), ["regression_72h", "classification_72h_costaware"])

    def test_cost_aware_round1_matrix_crosses_modes_and_shells(self) -> None:
        matrix = build_cost_aware_round1_matrix()

        self.assertEqual(len(matrix), 4)
        self.assertEqual(
            matrix[0],
            {"label_mode": "regression_72h", "shell_name": "balanced"},
        )
        self.assertEqual(
            matrix[-1],
            {"label_mode": "classification_72h_costaware", "shell_name": "conservative"},
        )

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
