from __future__ import annotations

from pathlib import Path
import tomllib
import warnings

from nusri_project.config.schemas import (
    DataConfig,
    ExperimentRuntimeConfig,
    FactorConfig,
    FusedExperimentRuntimeConfig,
    FusionProfileConfig,
    LabelConfig,
    ModelConfig,
    SignalComponentConfig,
    SignalComponentRuntimeConfig,
    TradeConfig,
    TrainingConfig,
)


def _read_toml(path: Path) -> dict:
    with path.open("rb") as file_obj:
        return tomllib.load(file_obj)


def _require_named_section(config: dict, section: str, name: str) -> dict:
    section_map = config.get(section, {})
    if name not in section_map:
        raise ValueError(f"missing profile [{section}.{name}]")
    raw = section_map[name]
    if not isinstance(raw, dict):
        raise ValueError(f"profile [{section}.{name}] must be a table")
    return raw


def _validate_label_config(label: LabelConfig) -> None:
    if label.kind not in {"regression", "classification_costaware"}:
        raise ValueError(f"unsupported label kind: {label.kind}")
    if label.horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if label.kind == "classification_costaware":
        if label.round_trip_cost is None or label.safety_margin is None or label.positive_threshold is None:
            raise ValueError("classification_costaware label requires round_trip_cost, safety_margin, and positive_threshold")
        expected = label.round_trip_cost + label.safety_margin
        if abs(label.positive_threshold - expected) > 1e-12:
            raise ValueError("positive_threshold must equal round_trip_cost + safety_margin")


def _validate_training_config(training: TrainingConfig) -> None:
    if training.run_mode not in {"rolling", "single"}:
        raise ValueError(f"unsupported training run_mode: {training.run_mode}")
    if training.training_window is not None and training.training_window not in {"all", "2y"}:
        raise ValueError(f"unsupported training_window: {training.training_window}")
    if training.training_window_months is not None and training.training_window_months <= 0:
        raise ValueError("training_window_months must be positive")
    if training.run_mode == "rolling" and not training.rolling_step_months:
        raise ValueError("rolling run_mode requires rolling_step_months")
    if training.sample_weight_mode not in {"uniform", "exp_halflife"}:
        raise ValueError(f"unsupported sample_weight_mode: {training.sample_weight_mode}")
    if training.sample_weight_mode == "exp_halflife":
        if training.half_life_months is None or training.half_life_months <= 0:
            raise ValueError("exp_halflife sample weighting requires positive half_life_months")
    elif training.half_life_months is not None:
        raise ValueError("half_life_months is only supported with exp_halflife sample weighting")


def _validate_trade_config(trade: TradeConfig) -> None:
    if not 0 <= trade.de_risk_position <= trade.max_position <= 1:
        raise ValueError("trade config must satisfy 0 <= de_risk_position <= max_position <= 1")
    if not 0 <= trade.drawdown_de_risk_threshold <= 1:
        raise ValueError("drawdown_de_risk_threshold must be in [0, 1]")
    if trade.signal_kind == "return":
        if (
            trade.entry_threshold is None
            or trade.exit_threshold is None
            or trade.full_position_threshold is None
        ):
            raise ValueError("return signal config requires entry/exit/full_position thresholds")
        if not trade.full_position_threshold >= trade.entry_threshold >= trade.exit_threshold:
            raise ValueError("return thresholds must satisfy full_position_threshold >= entry_threshold >= exit_threshold")
    elif trade.signal_kind == "probability":
        if (
            trade.enter_prob_threshold is None
            or trade.exit_prob_threshold is None
            or trade.full_prob_threshold is None
        ):
            raise ValueError("probability signal config requires enter/exit/full probability thresholds")
        for value in (
            trade.enter_prob_threshold,
            trade.exit_prob_threshold,
            trade.full_prob_threshold,
        ):
            if not 0 <= value <= 1:
                raise ValueError("probability thresholds must be in [0, 1]")
        if not trade.full_prob_threshold >= trade.enter_prob_threshold >= trade.exit_prob_threshold:
            raise ValueError("probability thresholds must satisfy full_prob_threshold >= enter_prob_threshold >= exit_prob_threshold")
    else:
        raise ValueError(f"unsupported trade signal_kind: {trade.signal_kind}")
    if abs(trade.de_risk_position - trade.max_position) <= 1e-12:
        warnings.warn("de_risk_position equals max_position; de-risking will not reduce exposure", stacklevel=2)


def _build_data_config(raw: dict) -> DataConfig:
    return DataConfig(
        start_time=str(raw["start_time"]),
        end_time=str(raw["end_time"]),
        freq=str(raw["freq"]),
        provider_uri=str(raw["provider_uri"]),
        instrument=str(raw["instrument"]),
        fields=tuple(str(item) for item in raw["fields"]),
        deal_price=str(raw.get("deal_price", "close")),
        initial_cash=float(raw.get("initial_cash", 100_000.0)),
        fee_rate=float(raw.get("fee_rate", 0.001)),
        min_cost=float(raw.get("min_cost", 0.0)),
    )


def _build_factor_config(raw: dict) -> FactorConfig:
    return FactorConfig(feature_set=str(raw["feature_set"]))


def _build_label_config(raw: dict) -> LabelConfig:
    label = LabelConfig(
        kind=str(raw["kind"]),
        horizon_hours=int(raw["horizon_hours"]),
        round_trip_cost=float(raw["round_trip_cost"]) if "round_trip_cost" in raw else None,
        safety_margin=float(raw["safety_margin"]) if "safety_margin" in raw else None,
        positive_threshold=float(raw["positive_threshold"]) if "positive_threshold" in raw else None,
    )
    _validate_label_config(label)
    return label


def _build_model_config(raw: dict) -> ModelConfig:
    hyperparameters = raw.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        raise ValueError("model hyperparameters must be a table")
    return ModelConfig(
        model_type=str(raw["model_type"]),
        objective=str(raw["objective"]),
        hyperparameters=dict(hyperparameters),
    )


def _legacy_training_window_to_months(raw_window: str | None) -> int | None:
    if raw_window == "all":
        return None
    if raw_window == "2y":
        return 24
    raise ValueError(f"unsupported training_window: {raw_window}")


def _build_training_config(raw: dict) -> TrainingConfig:
    raw_training_window = str(raw["training_window"]) if "training_window" in raw else None
    legacy_training_window_months = (
        _legacy_training_window_to_months(raw_training_window)
        if raw_training_window is not None
        else None
    )
    explicit_training_window_months = (
        int(raw["training_window_months"])
        if "training_window_months" in raw
        else None
    )
    if raw_training_window is None and explicit_training_window_months is None:
        raise ValueError("training config requires training_window or training_window_months")
    if raw_training_window == "all" and explicit_training_window_months is not None:
        raise ValueError("training_window='all' cannot be combined with training_window_months")
    if (
        explicit_training_window_months is not None
        and legacy_training_window_months is not None
        and explicit_training_window_months != legacy_training_window_months
    ):
        raise ValueError("training_window and training_window_months must resolve to the same duration")
    training = TrainingConfig(
        run_mode=str(raw["run_mode"]),
        training_window=raw_training_window,
        training_window_months=(
            explicit_training_window_months
            if explicit_training_window_months is not None
            else legacy_training_window_months
        ),
        rolling_step_months=int(raw["rolling_step_months"]) if "rolling_step_months" in raw else None,
        sample_weight_mode=str(raw.get("sample_weight_mode", "uniform")),
        half_life_months=float(raw["half_life_months"]) if "half_life_months" in raw else None,
    )
    _validate_training_config(training)
    return training


def _build_signal_component_config(name: str, raw: dict) -> SignalComponentConfig:
    factor_profile = raw.get("factor_profile")
    return SignalComponentConfig(
        name=name,
        factor_profile=str(factor_profile) if factor_profile is not None else None,
        label_profile=str(raw["label_profile"]),
        model_profile=str(raw["model_profile"]),
        training_profile=str(raw["training_profile"]),
    )


def _build_fusion_profile_config(name: str, raw: dict) -> FusionProfileConfig:
    components = tuple(str(component) for component in raw["components"])
    weights = tuple(float(weight) for weight in raw["weights"])
    fusion = FusionProfileConfig(
        name=name,
        components=components,
        weights=weights,
        component_transform=str(raw.get("component_transform", "robust_norm_clip")),
        transform_fit_scope=str(raw.get("transform_fit_scope", "train_only")),
        output_column=str(raw.get("output_column", "pred_score")),
        cache_component_predictions=bool(raw.get("cache_component_predictions", False)),
    )
    if len(fusion.weights) != len(fusion.components):
        raise ValueError("fusion profile weights must match component count")
    return fusion


def _build_trade_config(raw: dict) -> TradeConfig:
    trade = TradeConfig(
        signal_kind=str(raw["signal_kind"]),
        max_position=float(raw["max_position"]),
        min_holding_hours=int(raw["min_holding_hours"]),
        cooldown_hours=int(raw["cooldown_hours"]),
        drawdown_de_risk_threshold=float(raw["drawdown_de_risk_threshold"]),
        de_risk_position=float(raw["de_risk_position"]),
        entry_threshold=float(raw["entry_threshold"]) if "entry_threshold" in raw else None,
        exit_threshold=float(raw["exit_threshold"]) if "exit_threshold" in raw else None,
        full_position_threshold=float(raw["full_position_threshold"]) if "full_position_threshold" in raw else None,
        enter_prob_threshold=float(raw["enter_prob_threshold"]) if "enter_prob_threshold" in raw else None,
        exit_prob_threshold=float(raw["exit_prob_threshold"]) if "exit_prob_threshold" in raw else None,
        full_prob_threshold=float(raw["full_prob_threshold"]) if "full_prob_threshold" in raw else None,
    )
    _validate_trade_config(trade)
    return trade


def _resolve_experiment_profile(config: dict, experiment_name: str | None = None) -> tuple[str, dict, dict]:
    defaults = config.get("defaults", {})
    selected_experiment = experiment_name or defaults.get("experiment_profile")
    if not selected_experiment:
        raise ValueError("no experiment profile was provided and defaults.experiment_profile is missing")
    experiment_raw = _require_named_section(config, "experiments", selected_experiment)
    return str(selected_experiment), defaults, experiment_raw


def _resolve_profile_name(
    experiment_raw: dict,
    defaults: dict,
    *,
    field_name: str,
) -> str:
    resolved = experiment_raw.get(field_name, defaults.get(field_name))
    if resolved in {None, ""}:
        raise ValueError(f"experiment profile resolution failed; missing {field_name}")
    return str(resolved)


def load_runtime_config(config_path: str | Path, experiment_name: str | None = None) -> ExperimentRuntimeConfig:
    path = Path(config_path)
    config = _read_toml(path)
    selected_experiment, defaults, experiment_raw = _resolve_experiment_profile(config, experiment_name=experiment_name)

    data_profile = _resolve_profile_name(experiment_raw, defaults, field_name="data_profile")
    factor_profile = _resolve_profile_name(experiment_raw, defaults, field_name="factor_profile")
    label_profile = _resolve_profile_name(experiment_raw, defaults, field_name="label_profile")
    model_profile = _resolve_profile_name(experiment_raw, defaults, field_name="model_profile")
    training_profile = _resolve_profile_name(experiment_raw, defaults, field_name="training_profile")
    trade_profile = _resolve_profile_name(experiment_raw, defaults, field_name="trade_profile")

    data = _build_data_config(_require_named_section(config, "data", data_profile))
    factors = _build_factor_config(_require_named_section(config, "factors", factor_profile))
    label = _build_label_config(_require_named_section(config, "labels", label_profile))
    model = _build_model_config(_require_named_section(config, "models", model_profile))
    training = _build_training_config(_require_named_section(config, "training", training_profile))
    trade = _build_trade_config(_require_named_section(config, "trading", trade_profile))

    return ExperimentRuntimeConfig(
        experiment_name=selected_experiment,
        data=data,
        factors=factors,
        label=label,
        model=model,
        training=training,
        trade=trade,
    )


def load_fused_runtime_config(
    config_path: str | Path,
    experiment_name: str | None = None,
) -> FusedExperimentRuntimeConfig:
    path = Path(config_path)
    config = _read_toml(path)
    selected_experiment, defaults, experiment_raw = _resolve_experiment_profile(config, experiment_name=experiment_name)

    data_profile = _resolve_profile_name(experiment_raw, defaults, field_name="data_profile")
    fusion_profile = _resolve_profile_name(experiment_raw, defaults, field_name="fusion_profile")
    experiment_factor_profile = experiment_raw.get("factor_profile", defaults.get("factor_profile"))
    resolved_experiment_factor_profile = (
        str(experiment_factor_profile)
        if experiment_factor_profile not in {None, ""}
        else None
    )

    data = _build_data_config(_require_named_section(config, "data", data_profile))
    fusion = _build_fusion_profile_config(
        fusion_profile,
        _require_named_section(config, "fusion_profiles", fusion_profile),
    )

    components: list[SignalComponentRuntimeConfig] = []
    for component_name in fusion.components:
        component_config = _build_signal_component_config(
            component_name,
            _require_named_section(config, "signal_components", component_name),
        )
        factor_profile = component_config.factor_profile or resolved_experiment_factor_profile
        if factor_profile is None:
            raise ValueError(
                f"component {component_name} requires factor_profile or an experiment/default fallback"
            )
        components.append(
            SignalComponentRuntimeConfig(
                name=component_config.name,
                factor=_build_factor_config(_require_named_section(config, "factors", factor_profile)),
                label=_build_label_config(_require_named_section(config, "labels", component_config.label_profile)),
                model=_build_model_config(_require_named_section(config, "models", component_config.model_profile)),
                training=_build_training_config(
                    _require_named_section(config, "training", component_config.training_profile)
                ),
            )
        )

    return FusedExperimentRuntimeConfig(
        experiment_name=selected_experiment,
        data=data,
        fusion=fusion,
        components=tuple(components),
    )
