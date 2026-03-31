from __future__ import annotations

from nusri_project.strategy.return_signal_strategy import _apply_trade_guards


def compute_target_weight_from_score_signal(
    *,
    pred_score: float,
    current_weight: float,
    max_position: float,
    open_score: float,
    close_score: float,
    size_floor_score: float,
    size_full_score: float,
    curve_gamma: float,
    min_holding_bars: int,
    holding_bars: int,
    cooldown_bars: int,
    bars_since_trade: float,
    drawdown: float,
    drawdown_de_risk_threshold: float,
    de_risk_position: float,
) -> float:
    if current_weight <= 0 and pred_score < open_score:
        target_weight = 0.0
    elif current_weight > 0 and pred_score <= close_score:
        target_weight = 0.0
    else:
        normalized_score = (pred_score - size_floor_score) / (size_full_score - size_floor_score)
        normalized_score = min(max(normalized_score, 0.0), 1.0)
        target_weight = max_position * (normalized_score**curve_gamma)

    if drawdown >= drawdown_de_risk_threshold:
        target_weight = min(target_weight, de_risk_position)

    return _apply_trade_guards(
        target_weight=target_weight,
        current_weight=current_weight,
        min_holding_bars=min_holding_bars,
        holding_bars=holding_bars,
        cooldown_bars=cooldown_bars,
        bars_since_trade=bars_since_trade,
    )
