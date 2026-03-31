from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from nusri_project.config.schemas import (
    DataConfig,
    FactorConfig,
    FusedExperimentRuntimeConfig,
    FusionProfileConfig,
    LabelConfig,
    ModelConfig,
    SignalComponentRuntimeConfig,
    TrainingConfig,
)
from nusri_project.training.fused_signal_workflow import (
    fuse_component_predictions,
    run_fused_signal_workflow,
)
from nusri_project.training.signal_transform import apply_component_transform


class FusedSignalWorkflowTests(unittest.TestCase):
    @staticmethod
    def _index(*timestamps: str) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(list(timestamps)),
                ["BTCUSDT"] * len(timestamps),
            ]
        )

    def _runtime(self, *, cache_component_predictions: bool = True) -> FusedExperimentRuntimeConfig:
        return FusedExperimentRuntimeConfig(
            experiment_name="regression_fused_main",
            data=DataConfig(
                start_time="2019-09-10 08:00:00",
                end_time="2025-12-31 23:00:00",
                freq="60min",
                provider_uri="./qlib_data/my_crypto_data",
                instrument="BTCUSDT",
                fields=("ohlcv",),
                deal_price="close",
                initial_cash=100000.0,
                fee_rate=0.001,
                min_cost=0.0,
            ),
            fusion=FusionProfileConfig(
                name="regression_fused_main",
                components=("reg_24h", "cls_72h"),
                weights=(0.4, 0.6),
                component_transform="robust_norm_clip",
                transform_fit_scope="train_only",
                output_column="pred_score",
                cache_component_predictions=cache_component_predictions,
            ),
            components=(
                SignalComponentRuntimeConfig(
                    name="reg_24h",
                    factor=FactorConfig(feature_set="top23"),
                    label=LabelConfig(kind="regression", horizon_hours=24),
                    model=ModelConfig(model_type="lightgbm", objective="mse"),
                    training=TrainingConfig(
                        run_mode="rolling",
                        training_window_months=24,
                        rolling_step_months=1,
                    ),
                ),
                SignalComponentRuntimeConfig(
                    name="cls_72h",
                    factor=FactorConfig(feature_set="top23"),
                    label=LabelConfig(
                        kind="classification_costaware",
                        horizon_hours=72,
                        round_trip_cost=0.002,
                        safety_margin=0.004,
                        positive_threshold=0.006,
                    ),
                    model=ModelConfig(model_type="lightgbm", objective="binary"),
                    training=TrainingConfig(
                        run_mode="rolling",
                        training_window_months=24,
                        rolling_step_months=1,
                    ),
                ),
            ),
        )

    def test_fuse_component_predictions_emits_pred_score_and_real_return(self) -> None:
        index = self._index("2025-01-31 23:00:00", "2025-02-28 23:00:00")
        component_frames = {
            "first": pd.DataFrame(
                {"component_score": [0.2, -0.4], "real_return": [0.05, -0.01]},
                index=index,
            ),
            "second": pd.DataFrame(
                {"component_score": [0.4, 0.2], "real_return": [0.05, -0.01]},
                index=index,
            ),
        }

        fused = fuse_component_predictions(
            component_frames,
            weights=(0.4, 0.6),
            output_column="pred_score",
        )

        self.assertIn("pred_score", fused.columns)
        self.assertIn("real_return", fused.columns)
        self.assertAlmostEqual(fused.iloc[0]["pred_score"], 0.32)
        self.assertAlmostEqual(fused.iloc[1]["pred_score"], -0.04)

    def test_fuse_component_predictions_uses_longest_horizon_target_independent_of_order(self) -> None:
        index = self._index("2025-01-31 23:00:00", "2025-02-28 23:00:00")
        short_frame = pd.DataFrame(
            {"component_score": [0.2, -0.4], "real_return": [0.01, 0.02]},
            index=index,
        )
        long_frame = pd.DataFrame(
            {"component_score": [0.4, 0.2], "real_return": [0.11, 0.12]},
            index=index,
        )

        fused_ab = fuse_component_predictions(
            {"short": short_frame, "long": long_frame},
            weights=(0.4, 0.6),
            output_column="pred_score",
            component_horizons={"short": 24, "long": 72},
        )
        fused_ba = fuse_component_predictions(
            {"long": long_frame, "short": short_frame},
            weights=(0.6, 0.4),
            output_column="pred_score",
            component_horizons={"short": 24, "long": 72},
        )

        pd.testing.assert_series_equal(fused_ab["real_return"], long_frame["real_return"])
        pd.testing.assert_series_equal(fused_ba["real_return"], long_frame["real_return"])

    def test_fuse_component_predictions_rejects_zero_sum_weights(self) -> None:
        index = self._index("2025-01-31 23:00:00")
        component_frames = {
            "first": pd.DataFrame(
                {"component_score": [0.2], "real_return": [0.05]},
                index=index,
            ),
            "second": pd.DataFrame(
                {"component_score": [0.4], "real_return": [0.05]},
                index=index,
            ),
        }

        with self.assertRaises(ValueError):
            fuse_component_predictions(
                component_frames,
                weights=(0.4, -0.4),
                output_column="pred_score",
            )

    def test_run_fused_signal_workflow_transforms_components_fuses_and_caches_artifacts(self) -> None:
        train_reg = pd.DataFrame(
            {"pred_return": [0.0, 1.0, 2.0, 3.0], "real_return": [0.01, 0.02, 0.03, 0.04]},
            index=self._index(
                "2023-09-30 23:00:00",
                "2023-10-31 23:00:00",
                "2023-11-30 23:00:00",
                "2023-12-31 23:00:00",
            ),
        )
        test_reg = pd.DataFrame(
            {"pred_return": [3.0, 4.0], "real_return": [0.08, 0.12]},
            index=self._index("2024-01-31 23:00:00", "2024-02-29 23:00:00"),
        )
        train_cls = pd.DataFrame(
            {"pred_prob": [0.2, 0.4, 0.6, 0.8], "real_return": [0.01, 0.02, 0.03, 0.04]},
            index=self._index(
                "2023-09-30 23:00:00",
                "2023-10-31 23:00:00",
                "2023-11-30 23:00:00",
                "2023-12-31 23:00:00",
            ),
        )
        test_cls = pd.DataFrame(
            {"pred_prob": [0.7, 0.9], "real_return": [0.08, 0.12]},
            index=self._index("2024-01-31 23:00:00", "2024-02-29 23:00:00"),
        )
        expected_reg = apply_component_transform(
            test_reg["pred_return"],
            transform="robust_norm_clip",
            params={"median": 1.5, "scale": 1.4826, "clip_value": 3.0},
            clip_value=3.0,
        )
        expected_cls = apply_component_transform(
            test_cls["pred_prob"],
            transform="robust_norm_clip",
            params={"median": 0.5, "scale": 0.29652, "clip_value": 3.0},
            clip_value=3.0,
        )

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                patch(
                    "nusri_project.training.fused_signal_workflow.load_fused_runtime_config",
                    return_value=self._runtime(cache_component_predictions=True),
                ),
                patch("nusri_project.training.fused_signal_workflow.init_qlib"),
                patch(
                    "nusri_project.training.fused_signal_workflow.pd.date_range",
                    return_value=[pd.Timestamp("2024-01-01 00:00:00")],
                ),
                patch(
                    "nusri_project.training.fused_signal_workflow._build_component_predictions_for_month",
                    side_effect=[(train_reg, test_reg), (train_cls, test_cls)],
                ) as build_predictions,
            ):
                yearly_results = run_fused_signal_workflow(
                    "config.toml",
                    experiment_name="regression_fused_main",
                    prediction_output_dir=output_dir,
                )
            self.assertEqual(build_predictions.call_count, 2)
            fused = yearly_results["2024"][0]
            self.assertIn("pred_score", fused.columns)
            self.assertIn("real_return", fused.columns)

            expected_score = (expected_reg * 0.4 + expected_cls * 0.6) / 1.0
            pd.testing.assert_series_equal(
                fused["pred_score"],
                expected_score.rename("pred_score"),
                check_names=True,
            )
            pd.testing.assert_series_equal(
                fused["real_return"],
                test_reg["real_return"],
                check_names=True,
            )

            cached_component_paths = sorted(output_dir.glob("component_*.pkl"))
            self.assertEqual(len(cached_component_paths), 2)
            fused_paths = sorted(output_dir.glob("pred_fused_*.pkl"))
            self.assertEqual(len(fused_paths), 1)
            fused_artifact = pd.read_pickle(fused_paths[0])
            self.assertEqual(list(fused_artifact.columns), ["pred_score", "real_return"])

    def test_run_fused_signal_workflow_uses_shared_rolling_step_for_test_window(self) -> None:
        runtime = self._runtime(cache_component_predictions=False)
        runtime = FusedExperimentRuntimeConfig(
            experiment_name=runtime.experiment_name,
            data=runtime.data,
            fusion=runtime.fusion,
            components=tuple(
                SignalComponentRuntimeConfig(
                    name=component.name,
                    factor=component.factor,
                    label=component.label,
                    model=component.model,
                    training=TrainingConfig(
                        run_mode="rolling",
                        training_window_months=component.training.training_window_months,
                        rolling_step_months=3,
                        sample_weight_mode=component.training.sample_weight_mode,
                        half_life_months=component.training.half_life_months,
                    ),
                )
                for component in runtime.components
            ),
        )
        train_frame = pd.DataFrame(
            {"pred_return": [0.0, 1.0, 2.0, 3.0], "real_return": [0.01, 0.02, 0.03, 0.04]},
            index=self._index(
                "2023-09-30 23:00:00",
                "2023-10-31 23:00:00",
                "2023-11-30 23:00:00",
                "2023-12-31 23:00:00",
            ),
        )
        test_reg = pd.DataFrame(
            {"pred_return": [3.0], "real_return": [0.08]},
            index=self._index("2024-03-31 23:00:00"),
        )
        test_cls = pd.DataFrame(
            {"pred_prob": [0.7], "real_return": [0.08]},
            index=self._index("2024-03-31 23:00:00"),
        )

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                patch(
                    "nusri_project.training.fused_signal_workflow.load_fused_runtime_config",
                    return_value=runtime,
                ),
                patch("nusri_project.training.fused_signal_workflow.init_qlib"),
                patch(
                    "nusri_project.training.fused_signal_workflow.pd.date_range",
                    return_value=[pd.Timestamp("2024-01-01 00:00:00")],
                ),
                patch(
                    "nusri_project.training.fused_signal_workflow._build_component_predictions_for_month",
                    side_effect=[(train_frame, test_reg), (train_frame.rename(columns={"pred_return": "pred_prob"}), test_cls)],
                ) as build_predictions,
            ):
                run_fused_signal_workflow(
                    "config.toml",
                    experiment_name="regression_fused_main",
                    prediction_output_dir=output_dir,
                )

        first_call = build_predictions.call_args_list[0]
        self.assertEqual(first_call.kwargs["month_start"], pd.Timestamp("2024-01-01 00:00:00"))
        self.assertEqual(first_call.kwargs["month_end"], pd.Timestamp("2024-03-31 23:00:00"))

    def test_run_fused_signal_workflow_rejects_zero_sum_weights_before_writing_artifacts(self) -> None:
        runtime = self._runtime(cache_component_predictions=True)
        runtime = FusedExperimentRuntimeConfig(
            experiment_name=runtime.experiment_name,
            data=runtime.data,
            fusion=FusionProfileConfig(
                name=runtime.fusion.name,
                components=runtime.fusion.components,
                weights=(0.4, -0.4),
                component_transform=runtime.fusion.component_transform,
                transform_fit_scope=runtime.fusion.transform_fit_scope,
                output_column=runtime.fusion.output_column,
                cache_component_predictions=runtime.fusion.cache_component_predictions,
            ),
            components=runtime.components,
        )
        train_reg = pd.DataFrame(
            {"pred_return": [0.0, 1.0, 2.0, 3.0], "real_return": [0.01, 0.02, 0.03, 0.04]},
            index=self._index(
                "2023-09-30 23:00:00",
                "2023-10-31 23:00:00",
                "2023-11-30 23:00:00",
                "2023-12-31 23:00:00",
            ),
        )
        test_reg = pd.DataFrame(
            {"pred_return": [3.0], "real_return": [0.08]},
            index=self._index("2024-01-31 23:00:00"),
        )
        train_cls = pd.DataFrame(
            {"pred_prob": [0.2, 0.4, 0.6, 0.8], "real_return": [0.01, 0.02, 0.03, 0.04]},
            index=self._index(
                "2023-09-30 23:00:00",
                "2023-10-31 23:00:00",
                "2023-11-30 23:00:00",
                "2023-12-31 23:00:00",
            ),
        )
        test_cls = pd.DataFrame(
            {"pred_prob": [0.7], "real_return": [0.08]},
            index=self._index("2024-01-31 23:00:00"),
        )

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                patch(
                    "nusri_project.training.fused_signal_workflow.load_fused_runtime_config",
                    return_value=runtime,
                ),
                patch("nusri_project.training.fused_signal_workflow.init_qlib"),
                patch(
                    "nusri_project.training.fused_signal_workflow.pd.date_range",
                    return_value=[pd.Timestamp("2024-01-01 00:00:00")],
                ),
                patch(
                    "nusri_project.training.fused_signal_workflow._build_component_predictions_for_month",
                    side_effect=[(train_reg, test_reg), (train_cls, test_cls)],
                ),
            ):
                with self.assertRaises(ValueError):
                    run_fused_signal_workflow(
                        "config.toml",
                        experiment_name="regression_fused_main",
                        prediction_output_dir=output_dir,
                    )
                self.assertEqual(list(output_dir.glob("*.pkl")), [])


if __name__ == "__main__":
    unittest.main()
