from __future__ import annotations

import pandas as pd


def fit_component_transform(
    values: pd.Series,
    *,
    transform: str = "robust_norm_clip",
    clip_value: float = 3.0,
) -> dict[str, float]:
    if transform != "robust_norm_clip":
        raise ValueError(f"unsupported transform: {transform}")
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    if values.empty:
        raise ValueError("cannot fit transform on empty values")

    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = mad * 1.4826 if mad > 0 else float(values.std(ddof=0))
    if scale <= 0:
        scale = 1.0
    return {
        "median": median,
        "scale": scale,
        "clip_value": clip_value,
    }


def apply_component_transform(
    values: pd.Series,
    *,
    transform: str = "robust_norm_clip",
    params: dict[str, float],
    clip_value: float = 3.0,
) -> pd.Series:
    if transform != "robust_norm_clip":
        raise ValueError(f"unsupported transform: {transform}")
    scale = float(params["scale"])
    if scale <= 0:
        raise ValueError("transform scale must be positive")

    resolved_clip_value = float(params.get("clip_value", clip_value))
    normalized = (values - float(params["median"])) / scale
    clipped = normalized.clip(-resolved_clip_value, resolved_clip_value)
    return clipped / resolved_clip_value
