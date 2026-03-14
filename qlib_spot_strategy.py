from __future__ import annotations

import math
from typing import Any

from qlib.contrib.strategy.order_generator import OrderGenWInteract
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.utils.time import Freq


def hours_to_bars(hours: int, time_per_step: str) -> int:
    if hours <= 0:
        return 0

    count, base = Freq.parse(time_per_step)
    hours_per_bar = {
        Freq.NORM_FREQ_MINUTE: count / 60.0,
        Freq.NORM_FREQ_DAY: count * 24.0,
        Freq.NORM_FREQ_WEEK: count * 24.0 * 7.0,
        Freq.NORM_FREQ_MONTH: count * 24.0 * 30.0,
    }[base]
    return int(math.ceil(hours / hours_per_bar))


def compute_target_weight(
    *,
    pred_return: float,
    current_weight: float,
    max_position: float,
    entry_threshold: float,
    exit_threshold: float,
    full_position_threshold: float,
    min_holding_bars: int,
    holding_bars: int,
    cooldown_bars: int,
    bars_since_trade: float,
    drawdown: float,
    drawdown_de_risk_threshold: float,
    de_risk_position: float,
) -> float:
    if pred_return >= full_position_threshold:
        target_weight = max_position
    elif pred_return >= entry_threshold:
        target_weight = min(max_position, 0.5)
    elif pred_return <= exit_threshold:
        target_weight = 0.0
    else:
        target_weight = current_weight

    if drawdown >= drawdown_de_risk_threshold:
        target_weight = min(target_weight, de_risk_position)

    if target_weight < current_weight and holding_bars < min_holding_bars:
        return current_weight

    if target_weight > current_weight and bars_since_trade < cooldown_bars:
        return current_weight

    return target_weight


class QlibLongFlatStrategy(WeightStrategyBase):
    def __init__(
        self,
        *,
        instrument: str,
        time_per_step: str = "60min",
        entry_threshold: float = 0.0025,
        exit_threshold: float = 0.0005,
        full_position_threshold: float = 0.005,
        min_holding_hours: int = 24,
        cooldown_hours: int = 12,
        max_position: float = 1.0,
        drawdown_de_risk_threshold: float = 0.08,
        de_risk_position: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(order_generator_cls_or_obj=OrderGenWInteract(), **kwargs)
        self.instrument = instrument
        self.time_per_step = time_per_step
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.full_position_threshold = full_position_threshold
        self.min_holding_bars = hours_to_bars(min_holding_hours, time_per_step)
        self.cooldown_bars = hours_to_bars(cooldown_hours, time_per_step)
        self.max_position = max_position
        self.drawdown_de_risk_threshold = drawdown_de_risk_threshold
        self.de_risk_position = de_risk_position
        self.high_watermark: float | None = None
        self.last_trade_step: int | None = None

    def _extract_pred_return(self, score) -> float | None:
        if score is None:
            return None
        if hasattr(score, "columns"):
            if self.instrument not in score.index:
                return None
            row = score.loc[self.instrument]
            if hasattr(row, "iloc"):
                return float(row.iloc[0])
            return float(row)
        if self.instrument not in score.index:
            return None
        return float(score.loc[self.instrument])

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        pred_return = self._extract_pred_return(score)
        if pred_return is None:
            return current.get_stock_weight_dict()

        equity = float(current.calculate_value())
        if self.high_watermark is None:
            self.high_watermark = equity
        else:
            self.high_watermark = max(self.high_watermark, equity)

        drawdown = 0.0 if self.high_watermark <= 0 else 1.0 - (equity / self.high_watermark)
        current_weight = current.get_stock_weight_dict().get(self.instrument, 0.0)
        holding_bars = 0
        if self.instrument in current.get_stock_list():
            holding_bars = int(current.get_stock_count(self.instrument, bar=self.trade_calendar.get_freq()))

        trade_step = self.trade_calendar.get_trade_step()
        bars_since_trade = float("inf") if self.last_trade_step is None else trade_step - self.last_trade_step

        target_weight = compute_target_weight(
            pred_return=pred_return,
            current_weight=current_weight,
            max_position=self.max_position,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            full_position_threshold=self.full_position_threshold,
            min_holding_bars=self.min_holding_bars,
            holding_bars=holding_bars,
            cooldown_bars=self.cooldown_bars,
            bars_since_trade=bars_since_trade,
            drawdown=drawdown,
            drawdown_de_risk_threshold=self.drawdown_de_risk_threshold,
            de_risk_position=self.de_risk_position,
        )

        if abs(target_weight - current_weight) > 1e-12:
            self.last_trade_step = trade_step

        if target_weight <= 0:
            return {}
        return {self.instrument: target_weight}
