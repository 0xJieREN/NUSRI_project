from __future__ import annotations

from nusri_project.config.schemas import LabelConfig


def get_label_expr(label_horizon_hours: int) -> str:
    return f"Ref($close, -{label_horizon_hours}) / $close - 1"


def build_label_config(label_horizon_hours: int) -> tuple[list[str], list[str]]:
    return [get_label_expr(label_horizon_hours)], [f"label_{label_horizon_hours}h"]


def get_cost_aware_binary_label_expr(label_horizon_hours: int, positive_threshold: float) -> str:
    return f"If(Gt({get_label_expr(label_horizon_hours)}, {positive_threshold}), 1, 0)"


def get_backtest_target_expr(label_horizon_hours: int) -> str:
    return get_label_expr(label_horizon_hours)


def build_label_mode_config(
    *,
    label_mode: str,
    label_horizon_hours: int,
    positive_threshold: float,
) -> tuple[list[str], list[str]]:
    if label_mode.startswith("regression_"):
        return build_label_config(label_horizon_hours)
    if label_mode.startswith("classification_") and label_mode.endswith("_costaware"):
        return [get_cost_aware_binary_label_expr(label_horizon_hours, positive_threshold)], [
            f"label_cls_{label_horizon_hours}h_costaware"
        ]
    raise ValueError(f"Unknown label_mode: {label_mode}")


def get_prediction_output_column(label_mode: str) -> str:
    if label_mode.startswith("regression_"):
        return "pred_return"
    if label_mode.startswith("classification_") and label_mode.endswith("_costaware"):
        return "pred_prob"
    raise ValueError(f"Unknown label_mode: {label_mode}")


def get_label_mode_from_config(label: LabelConfig) -> str:
    if label.kind == "regression":
        return f"regression_{label.horizon_hours}h"
    if label.kind == "classification_costaware":
        return f"classification_{label.horizon_hours}h_costaware"
    raise ValueError(f"Unsupported label kind: {label.kind}")
