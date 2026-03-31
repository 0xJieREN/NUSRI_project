from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap
import unittest

from nusri_project.config.runtime_config import load_runtime_config
from nusri_project.strategy.strategy_config import build_spot_strategy_config_from_runtime
from scripts.analysis.run_72h_trade_tuning import parse_args as parse_72h_args
from scripts.analysis.run_cost_aware_label_round1 import parse_args as parse_cost_aware_args
from scripts.analysis.run_phase2_baseline import parse_args as parse_phase2_args
from nusri_project.strategy.backtest_spot_strategy import parse_args as parse_backtest_args


CONFIG_TEXT = """
[defaults]
experiment_profile = "cost_aware_main"

[data.btc_1h_full]
start_time = "2019-09-10 08:00:00"
end_time = "2025-12-31 23:00:00"
freq = "60min"
provider_uri = "./qlib_data/my_crypto_data"
instrument = "BTCUSDT"
fields = ["ohlcv", "amount", "taker_buy_base_volume", "taker_buy_quote_volume", "funding_rate"]
deal_price = "close"
initial_cash = 100000.0
fee_rate = 0.001
min_cost = 0.0

[factors.top23]
feature_set = "top23"

[labels.regression_72h]
kind = "regression"
horizon_hours = 72

[labels.classification_72h_costaware]
kind = "classification_costaware"
horizon_hours = 72
round_trip_cost = 0.002
safety_margin = 0.003
positive_threshold = 0.005

[models.lgbm_binary_default]
model_type = "lightgbm"
objective = "binary"

[models.lgbm_regression_default]
model_type = "lightgbm"
objective = "mse"

[training.rolling_2y_monthly]
run_mode = "rolling"
training_window = "2y"
rolling_step_months = 1

[trading.prob_conservative]
signal_kind = "probability"
enter_prob_threshold = 0.65
exit_prob_threshold = 0.50
full_prob_threshold = 0.80
max_position = 0.15
min_holding_hours = 48
cooldown_hours = 12
drawdown_de_risk_threshold = 0.02
de_risk_position = 0.0

[trading.return_conservative]
signal_kind = "return"
entry_threshold = 0.0015
exit_threshold = 0.0
full_position_threshold = 0.003
max_position = 0.15
min_holding_hours = 48
cooldown_hours = 12
drawdown_de_risk_threshold = 0.02
de_risk_position = 0.0

[trading.score_weighted]
signal_kind = "score"
open_score = 0.40
close_score = 0.20
size_floor_score = 0.40
size_full_score = 0.80
curve_gamma = 1.5
max_position = 0.20
min_holding_hours = 24
cooldown_hours = 6
drawdown_de_risk_threshold = 0.02
de_risk_position = 0.0

[experiments.cost_aware_main]
data_profile = "btc_1h_full"
factor_profile = "top23"
label_profile = "classification_72h_costaware"
model_profile = "lgbm_binary_default"
training_profile = "rolling_2y_monthly"
trade_profile = "prob_conservative"

[experiments.regression_main]
data_profile = "btc_1h_full"
factor_profile = "top23"
label_profile = "regression_72h"
model_profile = "lgbm_regression_default"
training_profile = "rolling_2y_monthly"
trade_profile = "return_conservative"

[experiments.score_main]
data_profile = "btc_1h_full"
factor_profile = "top23"
label_profile = "regression_72h"
model_profile = "lgbm_regression_default"
training_profile = "rolling_2y_monthly"
trade_profile = "score_weighted"
"""


class AnalysisEntrypointsConfigTests(unittest.TestCase):
    def _write_config(self) -> Path:
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.toml"
        path.write_text(textwrap.dedent(CONFIG_TEXT).strip() + "\n")
        return path

    def test_phase2_parse_args_accepts_config_profile(self) -> None:
        args = parse_phase2_args(
            [
                "--mlruns-root",
                "mlruns",
                "--config",
                "config.toml",
                "--experiment-profile",
                "regression_main",
            ]
        )

        self.assertEqual(args.config, "config.toml")
        self.assertEqual(args.experiment_profile, "regression_main")
        self.assertIsNone(args.provider_uri)

    def test_72h_tuning_parse_args_accepts_config_profile(self) -> None:
        args = parse_72h_args(
            [
                "--predictions-root",
                "reports/preds",
                "--config",
                "config.toml",
                "--experiment-profile",
                "regression_main",
            ]
        )

        self.assertEqual(args.config, "config.toml")
        self.assertEqual(args.experiment_profile, "regression_main")
        self.assertIsNone(args.provider_uri)

    def test_cost_aware_parse_args_accepts_config_profile(self) -> None:
        args = parse_cost_aware_args(
            [
                "--predictions-root",
                "reports/preds",
                "--config",
                "config.toml",
                "--experiment-profile",
                "cost_aware_main",
            ]
        )

        self.assertEqual(args.config, "config.toml")
        self.assertEqual(args.experiment_profile, "cost_aware_main")
        self.assertIsNone(args.provider_uri)

    def test_backtest_parse_args_accepts_config_profile(self) -> None:
        args = parse_backtest_args(
            [
                "--pred-glob",
                "pred_*.pkl",
                "--config",
                "config.toml",
                "--experiment-profile",
                "cost_aware_main",
            ]
        )

        self.assertEqual(args.config, "config.toml")
        self.assertEqual(args.experiment_profile, "cost_aware_main")
        self.assertIsNone(args.provider_uri)

    def test_build_spot_strategy_config_from_runtime_uses_probability_trade_profile(self) -> None:
        config_path = self._write_config()
        runtime = load_runtime_config(config_path, experiment_name="cost_aware_main")

        config = build_spot_strategy_config_from_runtime(
            runtime,
            start_time="2025-01-01 00:00:00",
            end_time="2025-12-31 23:00:00",
        )

        self.assertEqual(config.signal_kind, "probability")
        self.assertEqual(config.enter_prob_threshold, 0.65)
        self.assertEqual(config.full_prob_threshold, 0.80)
        self.assertEqual(config.max_position, 0.15)

    def test_build_spot_strategy_config_from_runtime_uses_score_trade_profile(self) -> None:
        config_path = self._write_config()
        runtime = load_runtime_config(config_path, experiment_name="score_main")

        config = build_spot_strategy_config_from_runtime(
            runtime,
            start_time="2025-01-01 00:00:00",
            end_time="2025-12-31 23:00:00",
        )

        self.assertEqual(config.signal_kind, "score")
        self.assertEqual(config.open_score, 0.40)
        self.assertEqual(config.close_score, 0.20)
        self.assertEqual(config.size_floor_score, 0.40)
        self.assertEqual(config.size_full_score, 0.80)
        self.assertEqual(config.curve_gamma, 1.5)
        self.assertEqual(config.max_position, 0.20)

    def test_phase2_default_output_dir_is_short(self) -> None:
        args = parse_phase2_args(["--mlruns-root", "mlruns"])
        self.assertIsNone(args.output_dir)

    def test_72h_tuning_default_output_dir_is_short(self) -> None:
        args = parse_72h_args(["--predictions-root", "reports/preds"])
        self.assertIsNone(args.output_dir)

    def test_cost_aware_default_output_root_is_short(self) -> None:
        args = parse_cost_aware_args(["--predictions-root", "reports/preds"])
        self.assertIsNone(args.output_root)


if __name__ == "__main__":
    unittest.main()
