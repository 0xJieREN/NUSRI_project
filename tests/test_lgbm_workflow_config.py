from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap
import unittest

import pandas as pd

from nusri_project.config.schemas import TrainingConfig
from nusri_project.training.label_factory import build_label_mode_config
from nusri_project.training.lgbm_workflow import (
    _build_reweighter,
    _resolve_train_start,
    build_conf_from_runtime,
    load_training_runtime_bundle,
)


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
safety_margin = 0.004
positive_threshold = 0.006

[labels.classification_24h_costaware]
kind = "classification_costaware"
horizon_hours = 24
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

[training.rolling_18m_halflife_6m]
run_mode = "rolling"
training_window_months = 18
rolling_step_months = 1
sample_weight_mode = "exp_halflife"
half_life_months = 6

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
trade_profile = "return_conservative"

[experiments.cost_aware_24h_main]
data_profile = "btc_1h_full"
factor_profile = "top23"
label_profile = "classification_24h_costaware"
model_profile = "lgbm_binary_default"
training_profile = "rolling_18m_halflife_6m"
trade_profile = "prob_conservative"

"""


class LgbmWorkflowConfigTests(unittest.TestCase):
    def _write_config(self) -> Path:
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.toml"
        path.write_text(textwrap.dedent(CONFIG_TEXT).strip() + "\n")
        return path

    def test_load_training_runtime_bundle_for_classification_profile(self) -> None:
        config_path = self._write_config()

        bundle = load_training_runtime_bundle(config_path, experiment_name="cost_aware_main")

        self.assertEqual(bundle.feature_set, "top23")
        self.assertEqual(bundle.label_mode, "classification_72h_costaware")
        self.assertEqual(bundle.label_horizon_hours, 72)
        self.assertAlmostEqual(bundle.positive_threshold, 0.006)
        self.assertEqual(bundle.run_mode, "rolling")
        self.assertEqual(bundle.provider_uri, "./qlib_data/my_crypto_data")

    def test_build_conf_from_runtime_for_classification_uses_binary_label(self) -> None:
        config_path = self._write_config()
        bundle = load_training_runtime_bundle(config_path, experiment_name="cost_aware_main")

        workflow_conf = build_conf_from_runtime(bundle.runtime)

        label_exprs, label_names = workflow_conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["label"]
        self.assertEqual(label_exprs, ["If(Gt(Ref($close, -72) / $close - 1, 0.006), 1, 0)"])
        self.assertEqual(label_names, ["label_cls_72h_costaware"])
        self.assertEqual(workflow_conf["task"]["model"]["kwargs"]["loss"], "binary")

    def test_build_conf_from_runtime_for_regression_uses_mse_label(self) -> None:
        config_path = self._write_config()
        bundle = load_training_runtime_bundle(config_path, experiment_name="regression_main")

        workflow_conf = build_conf_from_runtime(bundle.runtime)

        label_exprs, label_names = workflow_conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["label"]
        self.assertEqual(label_exprs, ["Ref($close, -72) / $close - 1"])
        self.assertEqual(label_names, ["label_72h"])
        self.assertEqual(workflow_conf["task"]["model"]["kwargs"]["loss"], "mse")

    def test_build_label_mode_config_supports_24h_costaware(self) -> None:
        exprs, names = build_label_mode_config(
            label_mode="classification_24h_costaware",
            label_horizon_hours=24,
            positive_threshold=0.006,
        )

        self.assertEqual(exprs, ["If(Gt(Ref($close, -24) / $close - 1, 0.006), 1, 0)"])
        self.assertEqual(names, ["label_cls_24h_costaware"])

    def test_build_conf_from_runtime_for_24h_costaware_uses_binary_label(self) -> None:
        config_path = self._write_config()
        bundle = load_training_runtime_bundle(config_path, experiment_name="cost_aware_24h_main")

        workflow_conf = build_conf_from_runtime(bundle.runtime)

        label_exprs, label_names = workflow_conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["label"]
        self.assertEqual(label_exprs, ["If(Gt(Ref($close, -24) / $close - 1, 0.006), 1, 0)"])
        self.assertEqual(label_names, ["label_cls_24h_costaware"])
        self.assertEqual(workflow_conf["task"]["model"]["kwargs"]["loss"], "binary")

    def test_resolve_train_start_uses_runtime_training_window_months(self) -> None:
        training = TrainingConfig(
            run_mode="rolling",
            training_window_months=18,
            rolling_step_months=1,
        )
        month_start = pd.Timestamp("2025-03-01 00:00:00")
        data_start_ts = pd.Timestamp("2024-01-15 00:00:00")

        train_start = _resolve_train_start(month_start, training, data_start_ts)

        self.assertEqual(train_start, pd.Timestamp("2024-01-15 00:00:00"))

    def test_build_reweighter_returns_exp_halflife_instance(self) -> None:
        training = TrainingConfig(
            run_mode="rolling",
            training_window_months=18,
            rolling_step_months=1,
            sample_weight_mode="exp_halflife",
            half_life_months=6,
        )

        reweighter = _build_reweighter(training, datetime(2025, 1, 31, 23, 0, 0))

        self.assertIsNotNone(reweighter)
        self.assertEqual(reweighter.reference_time, pd.Timestamp("2025-01-31 23:00:00"))
        self.assertAlmostEqual(reweighter.half_life_hours, 6 * 30 * 24)

if __name__ == "__main__":
    unittest.main()
