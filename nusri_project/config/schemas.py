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
    training_window: str
    rolling_step_months: int | None = None


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
