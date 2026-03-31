from __future__ import annotations

from nusri_project.config.schemas import ModelConfig


DEFAULT_LGBM_KWARGS = {
    "colsample_bytree": 0.6,
    "subsample": 0.7,
    "learning_rate": 0.005,
    "lambda_l1": 1.5,
    "lambda_l2": 5.0,
    "max_depth": 5,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "bagging_freq": 5,
    "num_threads": 8,
}


def get_model_loss(label_mode: str) -> str:
    if label_mode.startswith("regression_"):
        return "mse"
    if label_mode.startswith("classification_") and label_mode.endswith("_costaware"):
        return "binary"
    raise ValueError(f"Unknown label_mode: {label_mode}")


def build_lgb_model_config(loss: str, hyperparameters: dict[str, object] | None = None) -> dict:
    kwargs = dict(DEFAULT_LGBM_KWARGS)
    if hyperparameters is not None:
        kwargs.update(hyperparameters)
    kwargs["loss"] = loss
    return {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": kwargs,
    }


def build_model_config_from_runtime(model: ModelConfig) -> dict:
    if model.model_type != "lightgbm":
        raise ValueError(f"Unsupported model_type: {model.model_type}")
    return build_lgb_model_config(model.objective, model.hyperparameters)
