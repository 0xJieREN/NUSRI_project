from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    start_time: str
    end_time: str
    freq: str
    provider_uri: str
    instrument: str
    fields: tuple[str, ...]
    deal_price: str
    initial_cash: float
    fee_rate: float
    min_cost: float


@dataclass(frozen=True)
class FactorConfig:
    feature_set: str


@dataclass(frozen=True)
class LabelConfig:
    kind: str
    horizon_hours: int
    round_trip_cost: float | None = None
    safety_margin: float | None = None
    positive_threshold: float | None = None


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    objective: str
    hyperparameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    run_mode: str
    training_window: str | None = None
    training_window_months: int | None = None
    rolling_step_months: int | None = None
    sample_weight_mode: str = "uniform"
    half_life_months: float | None = None


@dataclass(frozen=True)
class SignalComponentConfig:
    name: str
    factor_profile: str | None
    label_profile: str
    model_profile: str
    training_profile: str


@dataclass(frozen=True)
class FusionProfileConfig:
    name: str
    components: tuple[str, ...]
    weights: tuple[float, ...]
    component_transform: str = "robust_norm_clip"
    transform_fit_scope: str = "train_only"
    output_column: str = "pred_score"
    cache_component_predictions: bool = False


@dataclass(frozen=True)
class TradeConfig:
    signal_kind: str
    max_position: float
    min_holding_hours: int
    cooldown_hours: int
    drawdown_de_risk_threshold: float
    de_risk_position: float
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    full_position_threshold: float | None = None
    enter_prob_threshold: float | None = None
    exit_prob_threshold: float | None = None
    full_prob_threshold: float | None = None


@dataclass(frozen=True)
class ExperimentRuntimeConfig:
    experiment_name: str
    data: DataConfig
    factors: FactorConfig
    label: LabelConfig
    model: ModelConfig
    training: TrainingConfig
    trade: TradeConfig


@dataclass(frozen=True)
class SignalComponentRuntimeConfig:
    name: str
    factor: FactorConfig
    label: LabelConfig
    model: ModelConfig
    training: TrainingConfig


@dataclass(frozen=True)
class FusedExperimentRuntimeConfig:
    experiment_name: str
    data: DataConfig
    fusion: FusionProfileConfig
    components: tuple[SignalComponentRuntimeConfig, ...]
