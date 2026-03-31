from __future__ import annotations

from dataclasses import dataclass
import warnings

from nusri_project.config.schemas import ExperimentRuntimeConfig


@dataclass(frozen=True)
class SpotStrategyConfig:
    provider_uri: str = "./qlib_data/my_crypto_data"
    instrument: str = "BTCUSDT"
    start_time: str = "2024-01-01 00:00:00"
    end_time: str = "2024-12-31 23:00:00"
    freq: str = "60min"
    initial_cash: float = 100_000.0
    fee_rate: float = 0.001
    min_cost: float = 0.0
    deal_price: str = "close"
    signal_kind: str = "return"
    entry_threshold: float = 0.0025
    exit_threshold: float = 0.0005
    full_position_threshold: float = 0.005
    enter_prob_threshold: float | None = None
    exit_prob_threshold: float | None = None
    full_prob_threshold: float | None = None
    open_score: float | None = None
    close_score: float | None = None
    size_floor_score: float | None = None
    size_full_score: float | None = None
    curve_gamma: float = 1.0
    min_holding_hours: int = 24
    cooldown_hours: int = 12
    max_position: float = 1.0
    drawdown_de_risk_threshold: float = 0.08
    de_risk_position: float = 0.5
    annualization_periods: int = 24 * 365

    def validate(self) -> None:
        if not self.provider_uri:
            raise ValueError("provider_uri must not be empty")
        if not self.instrument:
            raise ValueError("instrument must not be empty")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate must be in [0, 1)")
        if self.min_cost < 0:
            raise ValueError("min_cost must be non-negative")
        if not 0 <= self.max_position <= 1:
            raise ValueError("max_position must be in [0, 1]")
        if self.signal_kind == "return":
            if self.full_position_threshold < self.entry_threshold:
                raise ValueError("full_position_threshold must be >= entry_threshold")
            if self.entry_threshold < self.exit_threshold:
                raise ValueError("entry_threshold must be >= exit_threshold")
        elif self.signal_kind == "probability":
            if self.enter_prob_threshold is None or self.exit_prob_threshold is None or self.full_prob_threshold is None:
                raise ValueError("probability signal config requires enter_prob_threshold, exit_prob_threshold, and full_prob_threshold")
            if not 0 <= self.exit_prob_threshold <= self.enter_prob_threshold <= self.full_prob_threshold <= 1:
                raise ValueError("probability thresholds must satisfy 0 <= exit_prob_threshold <= enter_prob_threshold <= full_prob_threshold <= 1")
        elif self.signal_kind == "score":
            if (
                self.open_score is None
                or self.close_score is None
                or self.size_floor_score is None
                or self.size_full_score is None
            ):
                raise ValueError("score signal config requires open_score, close_score, size_floor_score, and size_full_score")
            if self.open_score < self.close_score:
                raise ValueError("score thresholds must satisfy open_score >= close_score")
            if self.open_score < self.size_floor_score:
                raise ValueError("score thresholds must satisfy open_score >= size_floor_score")
            if self.close_score > self.size_floor_score:
                raise ValueError("score thresholds must satisfy close_score <= size_floor_score")
            if self.size_full_score <= self.size_floor_score:
                raise ValueError("score thresholds must satisfy size_full_score > size_floor_score")
            if self.curve_gamma <= 0:
                raise ValueError("curve_gamma must be positive")
        else:
            raise ValueError("signal_kind must be one of 'return', 'probability', or 'score'")
        if self.min_holding_hours < 0:
            raise ValueError("min_holding_hours must be non-negative")
        if self.cooldown_hours < 0:
            raise ValueError("cooldown_hours must be non-negative")
        if not 0 <= self.drawdown_de_risk_threshold <= 1:
            raise ValueError("drawdown_de_risk_threshold must be in [0, 1]")
        if not 0 <= self.de_risk_position <= self.max_position:
            raise ValueError("de_risk_position must be in [0, max_position]")
        if abs(self.de_risk_position - self.max_position) <= 1e-12:
            warnings.warn("de_risk_position equals max_position; de-risking will not reduce exposure", stacklevel=2)


def build_spot_strategy_config_from_runtime(
    runtime: ExperimentRuntimeConfig,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> SpotStrategyConfig:
    trade = runtime.trade
    config = SpotStrategyConfig(
        provider_uri=runtime.data.provider_uri,
        instrument=runtime.data.instrument,
        start_time=start_time or runtime.data.start_time,
        end_time=end_time or runtime.data.end_time,
        freq=runtime.data.freq,
        initial_cash=runtime.data.initial_cash,
        fee_rate=runtime.data.fee_rate,
        min_cost=runtime.data.min_cost,
        deal_price=runtime.data.deal_price,
        signal_kind=trade.signal_kind,
        entry_threshold=float(trade.entry_threshold or 0.0),
        exit_threshold=float(trade.exit_threshold or 0.0),
        full_position_threshold=float(trade.full_position_threshold or 0.0),
        enter_prob_threshold=trade.enter_prob_threshold,
        exit_prob_threshold=trade.exit_prob_threshold,
        full_prob_threshold=trade.full_prob_threshold,
        open_score=getattr(trade, "open_score", None),
        close_score=getattr(trade, "close_score", None),
        size_floor_score=getattr(trade, "size_floor_score", None),
        size_full_score=getattr(trade, "size_full_score", None),
        curve_gamma=float(trade.curve_gamma) if getattr(trade, "curve_gamma", None) is not None else 1.0,
        min_holding_hours=trade.min_holding_hours,
        cooldown_hours=trade.cooldown_hours,
        max_position=trade.max_position,
        drawdown_de_risk_threshold=trade.drawdown_de_risk_threshold,
        de_risk_position=trade.de_risk_position,
    )
    config.validate()
    return config
