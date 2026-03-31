from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
from qlib.utils import init_instance_by_config
from qlib.workflow import R

from nusri_project.config.runtime_config import load_fused_runtime_config
from nusri_project.config.schemas import DataConfig, FusedExperimentRuntimeConfig, SignalComponentRuntimeConfig
from nusri_project.training.label_factory import get_label_mode_from_config, get_prediction_output_column
from nusri_project.training.lgbm_workflow import (
    DEFAULT_COST_AWARE_THRESHOLD,
    ROLLING_END,
    ROLLING_START,
    _build_dataset_kwargs,
    _build_reweighter,
    _make_predictions,
    _resolve_rolling_frequency,
    _resolve_train_start,
    build_conf,
    init_qlib,
)
from nusri_project.training.model_factory import build_model_config_from_runtime
from nusri_project.training.signal_transform import apply_component_transform, fit_component_transform


def fit_component_prediction_transform(
    train_frame: pd.DataFrame,
    *,
    prediction_column: str,
    transform: str,
    clip_value: float = 3.0,
) -> dict[str, float]:
    return fit_component_transform(
        train_frame[prediction_column],
        transform=transform,
        clip_value=clip_value,
    )


def apply_component_prediction_transform(
    test_frame: pd.DataFrame,
    *,
    prediction_column: str,
    transform: str,
    params: dict[str, float],
    clip_value: float = 3.0,
) -> pd.DataFrame:
    transformed = test_frame.copy()
    transformed["component_score"] = apply_component_transform(
        transformed[prediction_column],
        transform=transform,
        params=params,
        clip_value=clip_value,
    )
    return transformed


def fuse_component_predictions(
    component_frames: dict[str, pd.DataFrame],
    *,
    weights: tuple[float, ...],
    output_column: str = "pred_score",
    component_horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    if not component_frames:
        raise ValueError("component_frames must not be empty")
    if len(weights) != len(component_frames):
        raise ValueError("weights must match component frame count")
    weight_sum = sum(weights)
    if abs(weight_sum) <= 1e-12:
        raise ValueError("fusion weights must not sum to zero")

    score_map = {
        name: frame["component_score"]
        for name, frame in component_frames.items()
    }
    aligned_scores = pd.concat(score_map, axis=1).dropna()
    weight_series = pd.Series(weights, index=aligned_scores.columns, dtype=float)
    fused_score = aligned_scores.mul(weight_series, axis=1).sum(axis=1) / weight_sum

    target_component_name = _resolve_target_component_name(
        component_frames,
        aligned_scores.index,
        component_horizons=component_horizons,
    )
    aligned_return = component_frames[target_component_name].loc[aligned_scores.index, "real_return"]
    return pd.DataFrame(
        {
            output_column: fused_score,
            "real_return": aligned_return,
        }
    )


def build_component_prediction_artifact_name(component_name: str, month_label: str) -> str:
    return f"component_{component_name}_{month_label.replace('-', '')}.pkl"


def build_fused_prediction_artifact_name(month_label: str) -> str:
    return f"pred_fused_{month_label.replace('-', '')}.pkl"


def _resolve_target_component_name(
    component_frames: dict[str, pd.DataFrame],
    aligned_index: pd.Index,
    *,
    component_horizons: dict[str, int] | None,
) -> str:
    component_names = tuple(component_frames.keys())
    if component_horizons is not None:
        missing = [name for name in component_names if name not in component_horizons]
        if missing:
            raise ValueError(f"missing component horizons for: {', '.join(missing)}")
        longest_horizon = max(component_horizons[name] for name in component_names)
        target_components = [
            name for name in component_names if component_horizons[name] == longest_horizon
        ]
        if len(target_components) != 1:
            raise ValueError("fused target component must have a unique longest label horizon")
        return target_components[0]

    reference_name = component_names[0]
    reference_return = component_frames[reference_name].loc[aligned_index, "real_return"]
    for name in component_names[1:]:
        candidate_return = component_frames[name].loc[aligned_index, "real_return"]
        if not reference_return.equals(candidate_return):
            raise ValueError(
                "component real_return columns differ; provide component_horizons to select the fused target"
            )
    return reference_name


def _build_component_workflow_conf(
    data: DataConfig,
    component: SignalComponentRuntimeConfig,
) -> dict:
    label_mode = get_label_mode_from_config(component.label)
    positive_threshold = float(component.label.positive_threshold or DEFAULT_COST_AWARE_THRESHOLD)
    workflow_conf = build_conf(
        feature_set=component.factor.feature_set,
        label_horizon_hours=component.label.horizon_hours,
        label_mode=label_mode,
        positive_threshold=positive_threshold,
    )
    workflow_conf["task"]["model"] = build_model_config_from_runtime(component.model)

    dataset_kwargs = workflow_conf["task"]["dataset"]["kwargs"]
    handler_kwargs = dataset_kwargs["handler"]["kwargs"]
    handler_kwargs["start_time"] = data.start_time
    handler_kwargs["end_time"] = data.end_time
    handler_kwargs["instruments"] = data.instrument
    handler_kwargs["data_loader"]["kwargs"]["freq"] = data.freq
    return workflow_conf


def _build_component_predictions_for_month(
    component: SignalComponentRuntimeConfig,
    *,
    data: DataConfig,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if component.training.run_mode != "rolling":
        raise ValueError("fused signal workflow only supports rolling component training")

    data_start_ts = pd.Timestamp(data.start_time)
    train_start = _resolve_train_start(month_start, component.training, data_start_ts)
    train_end = month_start - pd.Timedelta(hours=1)

    workflow_conf = _build_component_workflow_conf(data, component)
    dataset_kwargs = _build_dataset_kwargs(
        workflow_conf,
        cast(datetime, train_start.to_pydatetime()).strftime("%Y-%m-%d %H:%M:%S"),
        cast(datetime, train_end.to_pydatetime()).strftime("%Y-%m-%d %H:%M:%S"),
        cast(datetime, month_start.to_pydatetime()).strftime("%Y-%m-%d %H:%M:%S"),
        cast(datetime, month_end.to_pydatetime()).strftime("%Y-%m-%d %H:%M:%S"),
    )
    dataset_conf = deepcopy(workflow_conf["task"]["dataset"])
    dataset_conf["kwargs"] = dataset_kwargs

    label_mode = get_label_mode_from_config(component.label)
    positive_threshold = float(component.label.positive_threshold or DEFAULT_COST_AWARE_THRESHOLD)

    with R.start(experiment_name=f"fused_signal_{component.name}"):
        model = init_instance_by_config(workflow_conf["task"]["model"])
        dataset = init_instance_by_config(dataset_conf)
        reweighter = _build_reweighter(
            component.training,
            cast(datetime, train_end.to_pydatetime()),
        )
        model.fit(dataset, reweighter=reweighter)

        train_pred = _make_predictions(
            dataset,
            model,
            "train",
            dataset_conf=dataset_conf,
            label_horizon_hours=component.label.horizon_hours,
            label_mode=label_mode,
            positive_threshold=positive_threshold,
        )
        test_pred = _make_predictions(
            dataset,
            model,
            "test",
            dataset_conf=dataset_conf,
            label_horizon_hours=component.label.horizon_hours,
            label_mode=label_mode,
            positive_threshold=positive_threshold,
        )
    return train_pred, test_pred


def _resolve_fusion_frequency(runtime: FusedExperimentRuntimeConfig) -> str:
    frequencies = {
        _resolve_rolling_frequency(component.training)
        for component in runtime.components
    }
    if len(frequencies) != 1:
        raise ValueError("all fused components must share the same rolling_step_months")
    return next(iter(frequencies))


def _resolve_shared_rolling_step_months(runtime: FusedExperimentRuntimeConfig) -> int:
    step_months = {
        component.training.rolling_step_months
        for component in runtime.components
    }
    if len(step_months) != 1:
        raise ValueError("all fused components must share the same rolling_step_months")
    resolved_step = next(iter(step_months))
    if resolved_step is None or resolved_step <= 0:
        raise ValueError("fused components require a positive rolling_step_months")
    return resolved_step


def _resolve_prediction_output_dir(
    prediction_output_dir: str | Path | None,
    runtime: FusedExperimentRuntimeConfig,
) -> Path:
    if prediction_output_dir is not None:
        return Path(prediction_output_dir)
    return Path("reports") / "fused-signal-preds" / runtime.experiment_name


def run_fused_rolling_workflow(
    runtime: FusedExperimentRuntimeConfig,
    *,
    prediction_output_dir: str | Path | None = None,
) -> dict[str, list[pd.DataFrame]]:
    if abs(sum(runtime.fusion.weights)) <= 1e-12:
        raise ValueError("fusion weights must not sum to zero")

    output_dir = _resolve_prediction_output_dir(prediction_output_dir, runtime)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = max(pd.Timestamp(ROLLING_START), pd.Timestamp(runtime.data.start_time))
    end_ts = min(pd.Timestamp(ROLLING_END), pd.Timestamp(runtime.data.end_time))
    rolling_freq = _resolve_fusion_frequency(runtime)
    rolling_step_months = _resolve_shared_rolling_step_months(runtime)
    component_horizons = {
        component.name: component.label.horizon_hours
        for component in runtime.components
    }

    yearly_results: dict[str, list[pd.DataFrame]] = {}
    for month_start in pd.date_range(start=start_ts, end=end_ts, freq=rolling_freq):
        next_month_start = month_start + pd.DateOffset(months=rolling_step_months)
        month_end = min(next_month_start - pd.Timedelta(hours=1), end_ts)
        month_label = month_start.strftime("%Y-%m")
        year_key = month_start.strftime("%Y")

        component_frames: dict[str, pd.DataFrame] = {}
        for component in runtime.components:
            train_pred, test_pred = _build_component_predictions_for_month(
                component,
                data=runtime.data,
                month_start=month_start,
                month_end=month_end,
            )
            prediction_column = get_prediction_output_column(get_label_mode_from_config(component.label))
            transform_params = fit_component_prediction_transform(
                train_pred,
                prediction_column=prediction_column,
                transform=runtime.fusion.component_transform,
            )
            transformed_test = apply_component_prediction_transform(
                test_pred,
                prediction_column=prediction_column,
                transform=runtime.fusion.component_transform,
                params=transform_params,
            )
            component_frames[component.name] = transformed_test

            if runtime.fusion.cache_component_predictions:
                transformed_test.to_pickle(
                    output_dir / build_component_prediction_artifact_name(component.name, month_label)
                )

        fused = fuse_component_predictions(
            component_frames,
            weights=runtime.fusion.weights,
            output_column=runtime.fusion.output_column,
            component_horizons=component_horizons,
        )
        fused.to_pickle(output_dir / build_fused_prediction_artifact_name(month_label))
        yearly_results.setdefault(year_key, []).append(fused)

    return yearly_results


def run_fused_signal_workflow(
    config_path: str | Path,
    *,
    experiment_name: str | None = None,
    prediction_output_dir: str | Path | None = None,
) -> dict[str, list[pd.DataFrame]]:
    runtime = load_fused_runtime_config(config_path, experiment_name=experiment_name)
    init_qlib(provider_uri=runtime.data.provider_uri)
    return run_fused_rolling_workflow(
        runtime,
        prediction_output_dir=prediction_output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fused signal workflow.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-profile", default=None)
    parser.add_argument("--prediction-output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_fused_signal_workflow(
        args.config,
        experiment_name=args.experiment_profile,
        prediction_output_dir=args.prediction_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
