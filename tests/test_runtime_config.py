from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap
import unittest

from nusri_project.config.runtime_config import load_runtime_config


CONFIG_TEXT = """
[defaults]
data_profile = "btc_1h_full"
factor_profile = "top23"
label_profile = "classification_72h_costaware"
model_profile = "lgbm_binary_default"
training_profile = "rolling_2y_monthly"
trade_profile = "prob_conservative"
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
safety_margin = 0.004
positive_threshold = 0.006

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

[training.single_full]
run_mode = "single"
training_window = "all"

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

[trading.return_balanced]
signal_kind = "return"
entry_threshold = 0.0015
exit_threshold = 0.0
full_position_threshold = 0.003
max_position = 0.25
min_holding_hours = 48
cooldown_hours = 12
drawdown_de_risk_threshold = 0.02
de_risk_position = 0.10

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
training_profile = "single_full"
trade_profile = "return_balanced"
"""


class RuntimeConfigTests(unittest.TestCase):
    def _write_config(self, text: str = CONFIG_TEXT) -> Path:
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.toml"
        path.write_text(textwrap.dedent(text).strip() + "\n")
        return path

    def test_load_runtime_config_uses_default_experiment_profile(self) -> None:
        config_path = self._write_config()

        runtime = load_runtime_config(config_path)

        self.assertEqual(runtime.experiment_name, "cost_aware_main")
        self.assertEqual(runtime.data.instrument, "BTCUSDT")
        self.assertEqual(runtime.factors.feature_set, "top23")
        self.assertEqual(runtime.label.kind, "classification_costaware")
        self.assertEqual(runtime.model.objective, "binary")
        self.assertEqual(runtime.trade.signal_kind, "probability")

    def test_load_runtime_config_can_select_explicit_experiment_profile(self) -> None:
        config_path = self._write_config()

        runtime = load_runtime_config(config_path, experiment_name="regression_main")

        self.assertEqual(runtime.experiment_name, "regression_main")
        self.assertEqual(runtime.label.kind, "regression")
        self.assertEqual(runtime.model.objective, "mse")
        self.assertEqual(runtime.training.run_mode, "single")
        self.assertEqual(runtime.trade.signal_kind, "return")

    def test_classification_label_config_preserves_threshold_components(self) -> None:
        config_path = self._write_config()

        runtime = load_runtime_config(config_path)

        self.assertEqual(runtime.label.horizon_hours, 72)
        self.assertAlmostEqual(runtime.label.round_trip_cost or 0.0, 0.002)
        self.assertAlmostEqual(runtime.label.safety_margin or 0.0, 0.004)
        self.assertAlmostEqual(runtime.label.positive_threshold or 0.0, 0.006)

    def test_probability_trade_config_parses_probability_thresholds(self) -> None:
        config_path = self._write_config()

        runtime = load_runtime_config(config_path)

        self.assertAlmostEqual(runtime.trade.enter_prob_threshold or 0.0, 0.65)
        self.assertAlmostEqual(runtime.trade.exit_prob_threshold or 0.0, 0.50)
        self.assertAlmostEqual(runtime.trade.full_prob_threshold or 0.0, 0.80)
        self.assertAlmostEqual(runtime.trade.max_position, 0.15)

    def test_invalid_probability_trade_threshold_order_raises_value_error(self) -> None:
        config_path = self._write_config(
            CONFIG_TEXT.replace("full_prob_threshold = 0.80", "full_prob_threshold = 0.60")
        )

        with self.assertRaises(ValueError):
            load_runtime_config(config_path)

    def test_invalid_costaware_threshold_components_raise_value_error(self) -> None:
        config_path = self._write_config(
            CONFIG_TEXT.replace("positive_threshold = 0.006", "positive_threshold = 0.005")
        )

        with self.assertRaises(ValueError):
            load_runtime_config(config_path)


if __name__ == "__main__":
    unittest.main()
