from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from backtest_spot_strategy import (
    _expand_prediction_globs,
    build_backtest_components,
    normalize_prediction_frame,
    prepare_signal_frame,
    summarize_report,
)
from qlib_spot_strategy import compute_target_weight
from strategy_config import SpotStrategyConfig


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

    def test_compute_target_weight_applies_thresholds_and_guards(self) -> None:
        self.assertEqual(
            compute_target_weight(
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
            compute_target_weight(
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
            compute_target_weight(
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

    def test_compute_target_weight_caps_exposure_in_drawdown(self) -> None:
        target = compute_target_weight(
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

        self.assertEqual(strategy_config["class"], "QlibLongFlatStrategy")
        self.assertEqual(strategy_config["module_path"], "qlib_spot_strategy")
        self.assertIs(strategy_config["kwargs"]["signal"], signal)
        self.assertEqual(executor_config["class"], "SimulatorExecutor")
        self.assertEqual(executor_config["module_path"], "qlib.backtest.executor")
        self.assertEqual(executor_config["kwargs"]["time_per_step"], "60min")
        self.assertEqual(backtest_config["exchange_kwargs"]["open_cost"], 0.001)
        self.assertEqual(backtest_config["exchange_kwargs"]["close_cost"], 0.001)
        self.assertTrue(isinstance(backtest_config["benchmark"], pd.Series))
        self.assertTrue((backtest_config["benchmark"] == 0.0).all())

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


if __name__ == "__main__":
    unittest.main()
