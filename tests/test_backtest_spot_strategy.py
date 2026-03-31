from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap
import unittest
from unittest.mock import patch

import pandas as pd
from qlib.backtest.position import Position

from nusri_project.strategy.backtest_spot_strategy import (
    _expand_prediction_globs,
    align_backtest_window,
    build_backtest_components,
    load_prediction_frames,
    main,
    normalize_prediction_frame,
    prepare_signal_frame,
    summarize_report,
)
from nusri_project.strategy.probability_signal_strategy import compute_target_weight_from_probability_signal
from nusri_project.strategy.qlib_spot_strategy import QlibSingleAssetOrderGen
from nusri_project.strategy.return_signal_strategy import compute_target_weight_from_return_signal
from nusri_project.strategy.strategy_config import SpotStrategyConfig


CONFIG_TEXT = """
[defaults]
experiment_profile = "score_main"

[data.btc_1h_full]
start_time = "2024-01-01 00:00:00"
end_time = "2024-01-31 23:00:00"
freq = "60min"
provider_uri = "./qlib_data/my_crypto_data"
instrument = "BTCUSDT"
fields = ["ohlcv"]
deal_price = "close"
initial_cash = 100000.0
fee_rate = 0.001
min_cost = 0.0

[factors.top23]
feature_set = "top23"

[labels.regression_72h]
kind = "regression"
horizon_hours = 72

[models.lgbm_regression_default]
model_type = "lightgbm"
objective = "mse"

[training.single_full]
run_mode = "single"
training_window = "all"

[trading.score_weighted]
signal_kind = "score"
open_score = 0.40
close_score = 0.20
size_floor_score = 0.40
size_full_score = 0.80
curve_gamma = 2.0
max_position = 0.25
min_holding_hours = 24
cooldown_hours = 12
drawdown_de_risk_threshold = 0.02
de_risk_position = 0.0

[experiments.score_main]
data_profile = "btc_1h_full"
factor_profile = "top23"
label_profile = "regression_72h"
model_profile = "lgbm_regression_default"
training_profile = "single_full"
trade_profile = "score_weighted"
"""


class SpotBacktestTests(unittest.TestCase):
    def test_prepare_signal_frame_builds_instrument_datetime_multiindex(self) -> None:
        frame = pd.DataFrame(
            {
                "pred_return": [0.02, 0.01],
                "real_return": [0.03, -0.01],
            },
            index=pd.to_datetime(["2024-01-01 02:00:00", "2024-01-01 01:00:00"]),
        )

        normalized = normalize_prediction_frame(frame)
        signal = prepare_signal_frame(normalized, instrument="BTCUSDT")

        self.assertEqual(signal.index.names, ["instrument", "datetime"])
        self.assertEqual(list(signal.columns), ["score"])
        self.assertEqual(list(signal.index.get_level_values("instrument").unique()), ["BTCUSDT"])
        self.assertEqual(
            list(signal.index.get_level_values("datetime")),
            [pd.Timestamp("2024-01-01 01:00:00"), pd.Timestamp("2024-01-01 02:00:00")],
        )

    def test_prepare_signal_frame_can_use_probability_signal_column(self) -> None:
        frame = pd.DataFrame(
            {
                "pred_prob": [0.81, 0.67],
                "real_return": [0.03, -0.01],
            },
            index=pd.to_datetime(["2024-01-01 02:00:00", "2024-01-01 01:00:00"]),
        )

        normalized = normalize_prediction_frame(frame, signal_column="pred_prob")
        signal = prepare_signal_frame(normalized, instrument="BTCUSDT", signal_column="pred_prob")

        self.assertEqual(list(signal.columns), ["score"])
        self.assertAlmostEqual(float(signal.iloc[0, 0]), 0.67)
        self.assertAlmostEqual(float(signal.iloc[1, 0]), 0.81)

    def test_prepare_signal_frame_can_use_score_signal_column(self) -> None:
        frame = pd.DataFrame(
            {
                "pred_score": [0.81, 0.67],
                "real_return": [0.03, -0.01],
            },
            index=pd.to_datetime(["2024-01-01 02:00:00", "2024-01-01 01:00:00"]),
        )

        normalized = normalize_prediction_frame(frame, signal_column="pred_score")
        signal = prepare_signal_frame(normalized, instrument="BTCUSDT", signal_column="pred_score")

        self.assertEqual(list(signal.columns), ["score"])
        self.assertAlmostEqual(float(signal.iloc[0, 0]), 0.67)
        self.assertAlmostEqual(float(signal.iloc[1, 0]), 0.81)

    def test_compute_target_weight_from_return_signal_applies_thresholds_and_guards(self) -> None:
        self.assertEqual(
            compute_target_weight_from_return_signal(
                pred_return=0.03,
                current_weight=0.0,
                max_position=1.0,
                entry_threshold=0.01,
                exit_threshold=0.0,
                full_position_threshold=0.02,
                min_holding_bars=2,
                holding_bars=0,
                cooldown_bars=2,
                bars_since_trade=99,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.5,
            ),
            1.0,
        )

        self.assertEqual(
            compute_target_weight_from_return_signal(
                pred_return=-0.01,
                current_weight=1.0,
                max_position=1.0,
                entry_threshold=0.01,
                exit_threshold=0.0,
                full_position_threshold=0.02,
                min_holding_bars=2,
                holding_bars=1,
                cooldown_bars=2,
                bars_since_trade=99,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.5,
            ),
            1.0,
        )

        self.assertEqual(
            compute_target_weight_from_return_signal(
                pred_return=0.03,
                current_weight=0.0,
                max_position=1.0,
                entry_threshold=0.01,
                exit_threshold=0.0,
                full_position_threshold=0.02,
                min_holding_bars=0,
                holding_bars=0,
                cooldown_bars=2,
                bars_since_trade=1,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.5,
            ),
            0.0,
        )

    def test_compute_target_weight_from_return_signal_caps_exposure_in_drawdown(self) -> None:
        target = compute_target_weight_from_return_signal(
            pred_return=0.03,
            current_weight=0.0,
            max_position=1.0,
            entry_threshold=0.01,
            exit_threshold=0.0,
            full_position_threshold=0.02,
            min_holding_bars=0,
            holding_bars=0,
            cooldown_bars=0,
            bars_since_trade=99,
            drawdown=0.12,
            drawdown_de_risk_threshold=0.1,
            de_risk_position=0.5,
        )

        self.assertEqual(target, 0.5)

    def test_compute_target_weight_from_probability_signal_uses_probability_thresholds(self) -> None:
        self.assertEqual(
            compute_target_weight_from_probability_signal(
                pred_prob=0.85,
                current_weight=0.0,
                max_position=0.25,
                enter_prob_threshold=0.65,
                exit_prob_threshold=0.50,
                full_prob_threshold=0.80,
                min_holding_bars=0,
                holding_bars=0,
                cooldown_bars=0,
                bars_since_trade=99,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.0,
            ),
            0.25,
        )
        self.assertEqual(
            compute_target_weight_from_probability_signal(
                pred_prob=0.70,
                current_weight=0.0,
                max_position=0.25,
                enter_prob_threshold=0.65,
                exit_prob_threshold=0.50,
                full_prob_threshold=0.80,
                min_holding_bars=0,
                holding_bars=0,
                cooldown_bars=0,
                bars_since_trade=99,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.0,
            ),
            0.125,
        )
        self.assertEqual(
            compute_target_weight_from_probability_signal(
                pred_prob=0.49,
                current_weight=0.25,
                max_position=0.25,
                enter_prob_threshold=0.65,
                exit_prob_threshold=0.50,
                full_prob_threshold=0.80,
                min_holding_bars=0,
                holding_bars=0,
                cooldown_bars=0,
                bars_since_trade=99,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.0,
            ),
            0.0,
        )

    def test_build_backtest_components_uses_qlib_native_executor(self) -> None:
        signal = pd.DataFrame(
            {"score": [0.01]},
            index=pd.MultiIndex.from_tuples(
                [("BTCUSDT", pd.Timestamp("2024-01-01 00:00:00"))],
                names=["instrument", "datetime"],
            ),
        )
        config = SpotStrategyConfig(
            provider_uri="/tmp/provider",
            instrument="BTCUSDT",
            start_time="2024-01-01 00:00:00",
            end_time="2024-01-31 23:00:00",
            initial_cash=100_000.0,
            fee_rate=0.001,
            min_cost=0.0,
            deal_price="close",
            freq="60min",
        )

        strategy_config, executor_config, backtest_config = build_backtest_components(signal, config)

        self.assertEqual(strategy_config["class"], "QlibReturnLongFlatStrategy")
        self.assertEqual(strategy_config["module_path"], "nusri_project.strategy.return_signal_strategy")
        self.assertIs(strategy_config["kwargs"]["signal"], signal)
        self.assertEqual(executor_config["class"], "SimulatorExecutor")
        self.assertEqual(executor_config["module_path"], "qlib.backtest.executor")
        self.assertEqual(executor_config["kwargs"]["time_per_step"], "60min")
        self.assertEqual(backtest_config["exchange_kwargs"]["open_cost"], 0.001)
        self.assertEqual(backtest_config["exchange_kwargs"]["close_cost"], 0.001)
        self.assertTrue(isinstance(backtest_config["benchmark"], pd.Series))
        self.assertTrue((backtest_config["benchmark"] == 0.0).all())

    def test_build_backtest_components_uses_probability_strategy_when_requested(self) -> None:
        signal = pd.DataFrame(
            {"score": [0.81]},
            index=pd.MultiIndex.from_tuples(
                [("BTCUSDT", pd.Timestamp("2024-01-01 00:00:00"))],
                names=["instrument", "datetime"],
            ),
        )
        config = SpotStrategyConfig(
            provider_uri="/tmp/provider",
            instrument="BTCUSDT",
            start_time="2024-01-01 00:00:00",
            end_time="2024-01-31 23:00:00",
            initial_cash=100_000.0,
            fee_rate=0.001,
            min_cost=0.0,
            deal_price="close",
            freq="60min",
            signal_kind="probability",
            enter_prob_threshold=0.65,
            exit_prob_threshold=0.50,
            full_prob_threshold=0.80,
        )

        strategy_config, _, _ = build_backtest_components(signal, config)

        self.assertEqual(strategy_config["class"], "QlibProbabilityLongFlatStrategy")
        self.assertEqual(strategy_config["module_path"], "nusri_project.strategy.probability_signal_strategy")
        self.assertEqual(strategy_config["kwargs"]["enter_prob_threshold"], 0.65)
        self.assertEqual(strategy_config["kwargs"]["full_prob_threshold"], 0.80)

    def test_build_backtest_components_uses_score_signal_column(self) -> None:
        signal = pd.DataFrame(
            {"score": [0.60]},
            index=pd.MultiIndex.from_tuples(
                [("BTCUSDT", pd.Timestamp("2024-01-01 00:00:00"))],
                names=["instrument", "datetime"],
            ),
        )
        config = SpotStrategyConfig(
            signal_kind="score",
            open_score=0.4,
            close_score=0.2,
            size_floor_score=0.4,
            size_full_score=0.8,
            curve_gamma=1.5,
            max_position=0.25,
        )

        strategy_config, _, _ = build_backtest_components(signal, config)

        self.assertEqual(strategy_config["class"], "QlibScoreLongFlatStrategy")
        self.assertEqual(strategy_config["module_path"], "nusri_project.strategy.score_signal_strategy")
        self.assertEqual(strategy_config["kwargs"]["open_score"], 0.4)
        self.assertEqual(strategy_config["kwargs"]["curve_gamma"], 1.5)

    def test_main_with_config_uses_pred_score_signal_column_for_score_strategy(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(textwrap.dedent(CONFIG_TEXT).strip() + "\n")

            pred_path = Path(tmp) / "pred_score.pkl"
            pd.DataFrame(
                {"pred_score": [0.60]},
                index=pd.to_datetime(["2024-01-01 00:00:00"]),
            ).to_pickle(pred_path)

            output_dir = Path(tmp) / "output"
            report = pd.DataFrame(
                {
                    "return": [0.01],
                    "cost": [0.001],
                    "turnover": [0.0],
                    "value": [25_000.0],
                    "account": [100_000.0],
                },
                index=pd.to_datetime(["2024-01-01 01:00:00"]),
            )
            captured: dict[str, object] = {}

            def fake_run_qlib_backtest(signal, config):
                captured["signal"] = signal.copy()
                captured["config"] = config
                return report, {"BTCUSDT": 0.25}, None

            with patch("nusri_project.strategy.backtest_spot_strategy.run_qlib_backtest", side_effect=fake_run_qlib_backtest):
                with patch(
                    "sys.argv",
                    [
                        "backtest_spot_strategy",
                        "--pred-glob",
                        str(pred_path),
                        "--config",
                        str(config_path),
                        "--experiment-profile",
                        "score_main",
                        "--output-dir",
                        str(output_dir),
                    ],
                ):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["config"].signal_kind, "score")
        self.assertEqual(captured["config"].curve_gamma, 2.0)
        self.assertAlmostEqual(float(captured["signal"].iloc[0, 0]), 0.60)

    def test_summarize_report_uses_net_returns_and_custom_scaler(self) -> None:
        report = pd.DataFrame(
            {
                "return": [0.01, -0.005, 0.002],
                "cost": [0.001, 0.0, 0.001],
                "bench": [0.0, 0.0, 0.0],
            },
            index=pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                ]
            ),
        )

        summary = summarize_report(report, annualization_periods=24 * 365)

        self.assertIn("annualized_return", summary)
        self.assertIn("annualized_volatility", summary)
        self.assertIn("sharpe", summary)
        self.assertIn("calmar", summary)
        self.assertIn("max_drawdown", summary)
        self.assertIn("exposure_ratio", summary)
        self.assertIn("avg_holding_hours", summary)
        self.assertAlmostEqual(summary["total_return"], 0.004958954999999765, places=9)

    def test_expand_prediction_globs_accepts_absolute_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "pred_202401.pkl"
            pd.DataFrame(
                {"pred_return": [0.01]},
                index=pd.to_datetime(["2024-01-01 00:00:00"]),
            ).to_pickle(pred_path)

            resolved = _expand_prediction_globs([str(pred_path)])

        self.assertEqual(resolved, [pred_path.resolve()])

    def test_load_prediction_frames_accepts_probability_signal_column(self) -> None:
        with TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "pred_202401.pkl"
            pd.DataFrame(
                {"pred_prob": [0.81]},
                index=pd.to_datetime(["2024-01-01 00:00:00"]),
            ).to_pickle(pred_path)

            combined = load_prediction_frames([pred_path], signal_column="pred_prob")

        self.assertIn("pred_prob", combined.columns)

    def test_load_prediction_frames_accepts_score_signal_column(self) -> None:
        with TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "pred_202401.pkl"
            pd.DataFrame(
                {"pred_score": [0.81]},
                index=pd.to_datetime(["2024-01-01 00:00:00"]),
            ).to_pickle(pred_path)

            combined = load_prediction_frames([pred_path], signal_column="pred_score")

        self.assertIn("pred_score", combined.columns)

    def test_align_backtest_window_moves_end_before_last_signal_bar(self) -> None:
        signal = pd.DataFrame(
            {"score": [0.01, 0.02, 0.03]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("BTCUSDT", pd.Timestamp("2025-12-31 21:00:00")),
                    ("BTCUSDT", pd.Timestamp("2025-12-31 22:00:00")),
                    ("BTCUSDT", pd.Timestamp("2025-12-31 23:00:00")),
                ],
                names=["instrument", "datetime"],
            ),
        )

        start_time, end_time = align_backtest_window(
            signal,
            start_time="2025-01-01 00:00:00",
            end_time="2025-12-31 23:00:00",
        )

        self.assertEqual(start_time, "2025-01-01 00:00:00")
        self.assertEqual(end_time, "2025-12-31 22:00:00")

    def test_single_asset_order_generator_preserves_fractional_target_weight(self) -> None:
        class StubExchange:
            open_cost = 0.001
            close_cost = 0.001

            def is_stock_tradable(self, stock_id, start_time, end_time):
                return True

            def get_deal_price(self, stock_id, start_time, end_time, direction):
                return 10.0

            def get_factor(self, stock_id, start_time, end_time):
                return 1.0

            def generate_order_for_target_amount_position(self, target_position, current_position, start_time, end_time):
                self.target_position = target_position
                return target_position

        exchange = StubExchange()
        current = Position(cash=100_000.0, position_dict={})
        generator = QlibSingleAssetOrderGen()

        target = generator.generate_order_list_from_target_weight_position(
            current=current,
            trade_exchange=exchange,
            target_weight_position={"BTCUSDT": 0.25},
            risk_degree=1.0,
            pred_start_time=pd.Timestamp("2024-01-01 00:00:00"),
            pred_end_time=pd.Timestamp("2024-01-01 00:59:00"),
            trade_start_time=pd.Timestamp("2024-01-01 01:00:00"),
            trade_end_time=pd.Timestamp("2024-01-01 01:59:00"),
        )

        self.assertAlmostEqual(exchange.target_position["BTCUSDT"], 2497.502497502498, places=9)
        self.assertAlmostEqual(target["BTCUSDT"], 2497.502497502498, places=9)

    def test_single_asset_order_generator_allows_fractional_crypto_amounts(self) -> None:
        class StubExchange:
            open_cost = 0.001
            close_cost = 0.001

            def is_stock_tradable(self, stock_id, start_time, end_time):
                return True

            def get_deal_price(self, stock_id, start_time, end_time, direction):
                return 40_000.0

            def generate_order_for_target_amount_position(self, target_position, current_position, start_time, end_time):
                self.target_position = target_position
                return target_position

        exchange = StubExchange()
        current = Position(cash=100_000.0, position_dict={})
        generator = QlibSingleAssetOrderGen()

        target = generator.generate_order_list_from_target_weight_position(
            current=current,
            trade_exchange=exchange,
            target_weight_position={"BTCUSDT": 0.15},
            risk_degree=1.0,
            pred_start_time=pd.Timestamp("2024-01-01 00:00:00"),
            pred_end_time=pd.Timestamp("2024-01-01 00:59:00"),
            trade_start_time=pd.Timestamp("2024-01-01 01:00:00"),
            trade_end_time=pd.Timestamp("2024-01-01 01:59:00"),
        )

        self.assertAlmostEqual(exchange.target_position["BTCUSDT"], 0.374625, places=6)
        self.assertAlmostEqual(target["BTCUSDT"], 0.374625, places=6)


if __name__ == "__main__":
    unittest.main()
