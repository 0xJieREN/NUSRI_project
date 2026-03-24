import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from nusri_project.config.alpha261_config import get_alpha261_config, get_top23_config
from nusri_project.config.runtime_config import load_runtime_config
from nusri_project.config.schemas import ExperimentRuntimeConfig
from nusri_project.training.label_factory import (
    build_label_config,
    build_label_mode_config,
    get_backtest_target_expr,
    get_cost_aware_binary_label_expr,
    get_label_expr,
    get_label_mode_from_config,
    get_prediction_output_column,
)
from nusri_project.training.model_factory import (
    build_lgb_model_config,
    build_model_config_from_runtime,
    get_model_loss,
)

PROVIDER_URI = "./qlib_data/my_crypto_data"
DEFAULT_FEATURE_SET = "top23"
DEFAULT_RUN_MODE = "rolling"
DEFAULT_LABEL_HORIZON_HOURS = 8
DEFAULT_LABEL_MODE = "regression_72h"
DEFAULT_COST_AWARE_THRESHOLD = 0.005

market = "all"
benchmark = "BTCUSDT"

DATA_START_TIME = "2019-09-10 08:00:00"
DATA_END_TIME = "2025-12-31 23:00:00"

# Rolling training config (monthly retrain)
ROLLING_START = "2024-01-01 00:00:00"
ROLLING_END = "2025-12-31 23:00:00"
ROLLING_TRAIN_YEARS = 2


@dataclass(frozen=True)
class TrainingRuntimeBundle:
    runtime: ExperimentRuntimeConfig
    feature_set: str
    label_mode: str
    label_horizon_hours: int
    positive_threshold: float
    run_mode: str
    provider_uri: str


def init_qlib(provider_uri: str = PROVIDER_URI) -> None:
    qlib.init(provider_uri=provider_uri, region=REG_CN)


def get_feature_config(feature_set: str):
    if feature_set == "alpha261":
        return get_alpha261_config()
    if feature_set == "top23":
        return get_top23_config()
    raise ValueError(f"Unknown FEATURE_SET: {feature_set}")


def build_prediction_artifact_name(
    label_horizon_hours: int,
    month_label: str,
    *,
    label_mode: str | None = None,
) -> str:
    month_key = month_label.replace("-", "")
    if label_mode is None:
        return f"pred_{label_horizon_hours}h_{month_key}.pkl"
    return f"pred_{label_mode}_{label_horizon_hours}h_{month_key}.pkl"


def load_training_runtime_bundle(
    config_path: str | Path,
    *,
    experiment_name: str | None = None,
) -> TrainingRuntimeBundle:
    runtime = load_runtime_config(config_path, experiment_name=experiment_name)
    label_mode = get_label_mode_from_config(runtime.label)
    return TrainingRuntimeBundle(
        runtime=runtime,
        feature_set=runtime.factors.feature_set,
        label_mode=label_mode,
        label_horizon_hours=runtime.label.horizon_hours,
        positive_threshold=float(runtime.label.positive_threshold or DEFAULT_COST_AWARE_THRESHOLD),
        run_mode=runtime.training.run_mode,
        provider_uri=runtime.data.provider_uri,
    )


def build_conf_from_runtime(runtime: ExperimentRuntimeConfig) -> dict:
    label_mode = get_label_mode_from_config(runtime.label)
    positive_threshold = float(runtime.label.positive_threshold or DEFAULT_COST_AWARE_THRESHOLD)
    feature_config = get_feature_config(runtime.factors.feature_set)
    label_config = build_label_mode_config(
        label_mode=label_mode,
        label_horizon_hours=runtime.label.horizon_hours,
        positive_threshold=positive_threshold,
    )
    model_config = build_model_config_from_runtime(runtime.model)
    workflow_conf = build_conf(
        feature_set=runtime.factors.feature_set,
        label_horizon_hours=runtime.label.horizon_hours,
        label_mode=label_mode,
        positive_threshold=positive_threshold,
    )
    workflow_conf["task"]["model"] = model_config
    workflow_conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["feature"] = feature_config
    workflow_conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["label"] = label_config
    workflow_conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["freq"] = runtime.data.freq
    return workflow_conf


def build_conf(
    feature_set: str = DEFAULT_FEATURE_SET,
    label_horizon_hours: int = DEFAULT_LABEL_HORIZON_HOURS,
    label_mode: str = DEFAULT_LABEL_MODE,
    positive_threshold: float = DEFAULT_COST_AWARE_THRESHOLD,
) -> dict:
    feature_config = get_feature_config(feature_set)
    label_config = build_label_mode_config(
        label_mode=label_mode,
        label_horizon_hours=label_horizon_hours,
        positive_threshold=positive_threshold,
    )
    model_loss = get_model_loss(label_mode)
    return {
        "task": {
            "model": build_lgb_model_config(model_loss),
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "DataHandlerLP",
                        "module_path": "qlib.data.dataset.handler",
                        "kwargs": {
                            "start_time": DATA_START_TIME,
                            "end_time": DATA_END_TIME,
                            "instruments": market,
                            "data_loader": {
                                "class": "QlibDataLoader",
                                "module_path": "qlib.data.dataset.loader",
                                "kwargs": {
                                    "config": {
                                    "feature": feature_config,
                                        "label": label_config,
                                    },
                                    "freq": "60min",
                                },
                            },
                            "infer_processors": [
                                {
                                    "class": "RobustZScoreNorm",
                                    "kwargs": {
                                        "fields_group": "feature",
                                        "clip_outlier": True,
                                        "fit_start_time": DATA_START_TIME,
                                        "fit_end_time": "2023-12-31 19:00:00",
                                    },
                                },
                                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                            ],
                            "learn_processors": [
                                {"class": "DropnaLabel"},
                            ],
                        },
                    },
                    "segments": {
                        "train": ("2019-09-10 08:00:00", "2023-12-31 19:00:00"),
                        "valid": ("2024-01-01 00:00:00", "2024-12-31 19:00:00"),
                        "test": ("2025-01-01 00:00:00", "2025-12-31 23:00:00"),
                    },
                },
            },
        },
    }


conf = build_conf()


def _build_dataset_kwargs(
    conf: dict,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict:
    dataset_kwargs = deepcopy(conf["task"]["dataset"]["kwargs"])
    handler_kwargs = dataset_kwargs["handler"]["kwargs"]

    handler_kwargs["start_time"] = train_start
    handler_kwargs["end_time"] = test_end

    for proc in handler_kwargs.get("infer_processors", []):
        if proc.get("class") == "RobustZScoreNorm":
            proc["kwargs"]["fit_start_time"] = train_start
            proc["kwargs"]["fit_end_time"] = train_end

    dataset_kwargs["segments"] = {
        "train": (train_start, train_end),
        "test": (test_start, test_end),
    }
    return dataset_kwargs


def _load_actual_return(dataset_conf: dict, segment: str, label_horizon_hours: int) -> pd.DataFrame:
    temp_dataset_conf = deepcopy(dataset_conf)
    label_config = ([get_backtest_target_expr(label_horizon_hours)], ["real_return"])
    temp_dataset_conf["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["label"] = label_config
    temp_dataset = init_instance_by_config(temp_dataset_conf)
    actual_return = temp_dataset.prepare(segment, col_set="label")
    actual_return.columns = ["real_return"]
    return actual_return


def _make_predictions(
    dataset,
    model,
    segment: str,
    *,
    dataset_conf: dict,
    label_horizon_hours: int,
    label_mode: str,
    positive_threshold: float,
) -> pd.DataFrame:
    prediction_column = get_prediction_output_column(label_mode)
    pred = model.predict(dataset, segment=segment).to_frame(prediction_column)

    actual_return = _load_actual_return(dataset_conf, segment, label_horizon_hours)
    combined = pred[[prediction_column]].join(actual_return, how="inner").dropna()
    return combined


def _print_summary(combined: pd.DataFrame, label_name: str, head_n: int = 10) -> None:
    print(f"\n=== {label_name} 预测结果 ===")

    if combined.empty:
        print(
            f"\n{label_name} 数据为空：请检查 segments 时间范围是否超出数据范围，或 label horizon 导致尾部被丢弃。"
        )
        return

    prediction_column = next(column for column in combined.columns if column != "real_return")
    prediction_frame = combined[[prediction_column, "real_return"]]
    ic = prediction_frame.corr().iloc[0, 1]
    rank_ic = prediction_frame.corr(method="spearman").iloc[0, 1]
    direction_signal = combined[prediction_column]
    if prediction_column == "pred_prob":
        direction_signal = direction_signal - 0.5
    same_sign = direction_signal * combined["real_return"] > 0
    accuracy = same_sign.sum() / len(same_sign)

    dt_index = combined.index.get_level_values(0)
    month = pd.Series(dt_index).dt.to_period("M")

    def _month_ic(df):
        if len(df) < 5:
            return pd.NA
        return df.corr().iloc[0, 1]

    monthly_ic = combined.groupby(month).apply(_month_ic).dropna()

    print("\n" + "=" * 30)
    print(f"{label_name} IC       : {ic:.4f}")
    print(f"{label_name} RankIC   : {rank_ic:.4f}")
    print(f"{label_name} 方向准确率 (Acc): {accuracy:.2%}")
    if len(monthly_ic) > 0:
        ic_mean = monthly_ic.mean()
        ic_std = monthly_ic.std()
        n_m = len(monthly_ic)
        icir = ic_mean / (ic_std + 1e-12)
        t_stat = ic_mean / ((ic_std + 1e-12) / (n_m**0.5))
        print(f"{label_name} 月度IC均值/标准差: {ic_mean:.4f} / {ic_std:.4f}")
        print(f"{label_name} 月度ICIR/t-stat(n={n_m}): {icir:.2f} / {t_stat:.2f}")
    print("=" * 30)

    print(f"\n前{head_n}条预测:")
    print(combined.head(head_n))


def run_single(
    workflow_conf: dict | None = None,
    *,
    label_horizon_hours: int = DEFAULT_LABEL_HORIZON_HOURS,
    label_mode: str = DEFAULT_LABEL_MODE,
    positive_threshold: float = DEFAULT_COST_AWARE_THRESHOLD,
):
    workflow_conf = conf if workflow_conf is None else workflow_conf
    with R.start(experiment_name="btc_raw_return_lgb"):
        print("正在训练模型...")

        model = init_instance_by_config(workflow_conf["task"]["model"])
        dataset = init_instance_by_config(workflow_conf["task"]["dataset"])
        model.fit(dataset)

        recorder = R.get_recorder()
        valid_pred = _make_predictions(
            dataset,
            model,
            "valid",
            dataset_conf=workflow_conf["task"]["dataset"],
            label_horizon_hours=label_horizon_hours,
            label_mode=label_mode,
            positive_threshold=positive_threshold,
        )
        _print_summary(valid_pred, "验证集", head_n=10)
        recorder.save_objects(**{"pred_valid.pkl": valid_pred})

        test_pred = _make_predictions(
            dataset,
            model,
            "test",
            dataset_conf=workflow_conf["task"]["dataset"],
            label_horizon_hours=label_horizon_hours,
            label_mode=label_mode,
            positive_threshold=positive_threshold,
        )
        _print_summary(test_pred, "测试集", head_n=10)
        recorder.save_objects(**{"pred_test.pkl": test_pred})


def run_rolling_monthly(
    workflow_conf: dict | None = None,
    *,
    label_horizon_hours: int = DEFAULT_LABEL_HORIZON_HOURS,
    label_mode: str = DEFAULT_LABEL_MODE,
    positive_threshold: float = DEFAULT_COST_AWARE_THRESHOLD,
    prediction_output_dir: str | None = None,
):
    workflow_conf = conf if workflow_conf is None else workflow_conf
    start_ts = pd.Timestamp(ROLLING_START)
    end_ts = pd.Timestamp(ROLLING_END)
    data_start_ts = pd.Timestamp(DATA_START_TIME)

    yearly_results: dict[str, list[pd.DataFrame]] = {}
    pred_output_path = None if prediction_output_dir is None else Path(prediction_output_dir)
    if pred_output_path is not None:
        pred_output_path.mkdir(parents=True, exist_ok=True)

    for month_start in pd.date_range(start=start_ts, end=end_ts, freq="MS"):
        next_month_start = month_start + pd.DateOffset(months=1)
        month_end = next_month_start - pd.Timedelta(hours=1)
        if month_end > end_ts:
            month_end = end_ts

        train_end = month_start - pd.Timedelta(hours=1)
        train_start = month_start - pd.DateOffset(years=ROLLING_TRAIN_YEARS)
        if train_start < data_start_ts:
            train_start = data_start_ts

        month_start_dt = cast(datetime, month_start.to_pydatetime())
        month_end_dt = cast(datetime, month_end.to_pydatetime())
        train_start_dt = cast(datetime, train_start.to_pydatetime())
        train_end_dt = cast(datetime, train_end.to_pydatetime())

        dataset_kwargs = _build_dataset_kwargs(
            workflow_conf,
            train_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            train_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            month_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            month_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        )

        dataset_conf = deepcopy(workflow_conf["task"]["dataset"])
        dataset_conf["kwargs"] = dataset_kwargs

        label_name = month_start_dt.strftime("%Y-%m")
        year_key = month_start_dt.strftime("%Y")
        with R.start(experiment_name="btc_raw_return_lgb_rolling"):
            model = init_instance_by_config(workflow_conf["task"]["model"])
            dataset = init_instance_by_config(dataset_conf)
            model.fit(dataset)

            test_pred = _make_predictions(
                dataset,
                model,
                "test",
                dataset_conf=dataset_conf,
                label_horizon_hours=label_horizon_hours,
                label_mode=label_mode,
                positive_threshold=positive_threshold,
            )
            yearly_results.setdefault(year_key, []).append(test_pred)
            if pred_output_path is not None:
                output_name = build_prediction_artifact_name(
                    label_horizon_hours,
                    label_name,
                    label_mode=label_mode,
                )
                test_pred.to_pickle(pred_output_path / output_name)

    for year_key in sorted(yearly_results.keys()):
        yearly_pred = pd.concat(yearly_results[year_key]).sort_index()
        _print_summary(yearly_pred, f"{year_key}年", head_n=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BTCUSDT LightGBM workflow.")
    parser.add_argument("--feature-set", choices=("alpha261", "top23"), default=DEFAULT_FEATURE_SET)
    parser.add_argument("--run-mode", choices=("single", "rolling"), default=DEFAULT_RUN_MODE)
    parser.add_argument("--provider-uri", default=PROVIDER_URI)
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment-profile", default=None)
    parser.add_argument("--label-horizon-hours", type=int, default=DEFAULT_LABEL_HORIZON_HOURS)
    parser.add_argument(
        "--label-mode",
        choices=("regression_72h", "classification_72h_costaware"),
        default=DEFAULT_LABEL_MODE,
    )
    parser.add_argument("--cost-aware-threshold", type=float, default=DEFAULT_COST_AWARE_THRESHOLD)
    parser.add_argument("--prediction-output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_set = args.feature_set
    label_horizon_hours = args.label_horizon_hours
    label_mode = args.label_mode
    positive_threshold = args.cost_aware_threshold
    run_mode = args.run_mode
    provider_uri = args.provider_uri

    if args.config is not None:
        bundle = load_training_runtime_bundle(args.config, experiment_name=args.experiment_profile)
        workflow_conf = build_conf_from_runtime(bundle.runtime)
        feature_set = bundle.feature_set
        label_horizon_hours = bundle.label_horizon_hours
        label_mode = bundle.label_mode
        positive_threshold = bundle.positive_threshold
        run_mode = bundle.run_mode
        provider_uri = bundle.provider_uri
    else:
        workflow_conf = build_conf(
            feature_set=feature_set,
            label_horizon_hours=label_horizon_hours,
            label_mode=label_mode,
            positive_threshold=positive_threshold,
        )

    init_qlib(provider_uri=provider_uri)
    if run_mode == "single":
        run_single(
            workflow_conf,
            label_horizon_hours=label_horizon_hours,
            label_mode=label_mode,
            positive_threshold=positive_threshold,
        )
    elif run_mode == "rolling":
        run_rolling_monthly(
            workflow_conf,
            label_horizon_hours=label_horizon_hours,
            label_mode=label_mode,
            positive_threshold=positive_threshold,
            prediction_output_dir=args.prediction_output_dir,
        )
    else:
        raise ValueError(f"Unknown RUN_MODE: {run_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
