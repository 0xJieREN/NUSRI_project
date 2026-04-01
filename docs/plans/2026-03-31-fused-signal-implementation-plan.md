# Fused Signal Implementation Plan

> **Status Update (2026-04-01):** This implementation plan has already been executed. The current recommended stage is `regression_fused_aggressive_v3_best`; see `docs/research/2026-04-01-regression-fused-best-stage-summary.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two reusable fused signal experiments, `regression_fused_main` and `costaware_fused_main`, with generic component/fusion interfaces, QLib-native half-life weighting, and a continuous `pred_score` execution path.

**Architecture:** Keep single-component training in the existing LightGBM workflow, add typed config/runtime support for signal components and fusion profiles, and introduce a fused workflow that trains components monthly, transforms their out-of-sample predictions, and emits one standardized `pred_score` artifact per fusion profile. Add a dedicated score-based strategy path so continuous position sizing consumes the same `pred_score` interface for both fused branches.

**Tech Stack:** Python 3.12, `uv`, QLib, LightGBM, TOML via stdlib `tomllib`, `unittest`, pandas, NumPy

---

## File Map

- `config.toml`
  - Add `signal_components`, `fusion_profiles`, new rolling training profiles, and score-based trade profiles.
- `nusri_project/config/schemas.py`
  - Extend `TrainingConfig`; add `SignalComponentConfig`, `FusionProfileConfig`, `SignalComponentRuntime`, and `FusedExperimentRuntimeConfig`.
- `nusri_project/config/runtime_config.py`
  - Parse and validate new sections; resolve an experiment into fully typed fused runtime data.
- `nusri_project/training/label_factory.py`
  - Generalize cost-aware labels beyond `72h`.
- `nusri_project/training/model_factory.py`
  - Accept generic `classification_*_costaware` label modes.
- `nusri_project/training/lgbm_workflow.py`
  - Become the reusable single-component trainer with runtime-driven training windows and optional reweighter injection.
- `nusri_project/training/time_decay_reweighter.py`
  - Implement QLib `Reweighter` for exponential half-life sample weighting.
- `nusri_project/training/signal_transform.py`
  - Implement `robust_norm_clip` and other transform hooks.
- `nusri_project/training/fused_signal_workflow.py`
  - Orchestrate component training, alignment, transform fitting, fusion, and artifact output.
- `scripts/training/fused_signal_workflow.py`
  - Thin CLI wrapper for the fused workflow.
- `nusri_project/strategy/continuous_position_mapping.py`
  - Implement `gated_power_curve`.
- `nusri_project/strategy/score_signal_strategy.py`
  - New QLib strategy class that consumes `pred_score`.
- `nusri_project/strategy/strategy_config.py`
  - Add `signal_kind="score"` and its parameters.
- `nusri_project/strategy/backtest_spot_strategy.py`
  - Route `pred_score` to the new strategy path.
- `tests/test_runtime_config.py`
  - Add fused runtime config coverage.
- `tests/test_lgbm_workflow_config.py`
  - Add generic cost-aware horizon and training-profile propagation coverage.
- `tests/test_time_decay_reweighter.py`
  - New reweighter unit tests.
- `tests/test_signal_transform.py`
  - New transform unit tests.
- `tests/test_fused_signal_workflow.py`
  - New fused orchestration unit tests.
- `tests/test_continuous_position_mapping.py`
  - New mapping unit tests.
- `tests/test_backtest_spot_strategy.py`
  - Extend backtest path coverage for `pred_score`.

### Task 1: Add Typed Runtime Support for Components, Fusion Profiles, and Training Weighting

**Files:**
- Modify: `config.toml`
- Modify: `nusri_project/config/schemas.py`
- Modify: `nusri_project/config/runtime_config.py`
- Modify: `tests/test_runtime_config.py`

- [ ] **Step 1: Write the failing tests for fused runtime config resolution**

Add tests like:

```python
def test_load_fused_runtime_config_resolves_component_specific_factor_profile(self) -> None:
    runtime = load_fused_runtime_config(config_path, experiment_name="regression_fused_main")

    self.assertEqual(runtime.experiment_name, "regression_fused_main")
    self.assertEqual(runtime.fusion.name, "regression_fused_main")
    self.assertEqual(tuple(component.name for component in runtime.components), ("reg_24h", "reg_72h"))
    self.assertEqual(runtime.components[0].factor.feature_set, "top23")
    self.assertEqual(runtime.components[1].label.horizon_hours, 72)
    self.assertEqual(runtime.components[0].training.sample_weight_mode, "uniform")
    self.assertEqual(runtime.components[1].training.training_window_months, 24)

def test_load_fused_runtime_config_rejects_weight_count_mismatch(self) -> None:
    bad_path = self._write_config(
        CONFIG_TEXT.replace('weights = [0.4, 0.6]', 'weights = [1.0]')
    )
    with self.assertRaises(ValueError):
        load_fused_runtime_config(bad_path, experiment_name="regression_fused_main")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_runtime_config -v
```

Expected:
- FAIL because `load_fused_runtime_config` and the new schema fields do not exist yet.

- [ ] **Step 3: Implement the config schema and runtime loader**

Add config examples to `config.toml`:

```toml
[training.rolling_24m_uniform]
run_mode = "rolling"
training_window_months = 24
rolling_step_months = 1
sample_weight_mode = "uniform"

[training.rolling_24m_halflife_6m]
run_mode = "rolling"
training_window_months = 24
rolling_step_months = 1
sample_weight_mode = "exp_halflife"
half_life_months = 6

[signal_components.reg_24h]
factor_profile = "top23"
label_profile = "regression_24h"
model_profile = "lgbm_regression_default"
training_profile = "rolling_24m_uniform"

[fusion_profiles.regression_fused_main]
components = ["reg_24h", "reg_72h"]
weights = [0.4, 0.6]
component_transform = "robust_norm_clip"
transform_fit_scope = "train_only"
output_column = "pred_score"
cache_component_predictions = false
```

Add schema types:

```python
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
```

Extend `TrainingConfig`:

```python
@dataclass(frozen=True)
class TrainingConfig:
    run_mode: str
    training_window: str | None = None
    training_window_months: int | None = None
    rolling_step_months: int | None = None
    sample_weight_mode: str = "uniform"
    half_life_months: float | None = None
```

Add loader helpers in `runtime_config.py`:

```python
def _legacy_training_window_to_months(raw_window: str | None) -> int | None:
    if raw_window == "all":
        return None
    if raw_window == "2y":
        return 24
    raise ValueError(f"unsupported training_window: {raw_window}")

def load_fused_runtime_config(config_path: str | Path, experiment_name: str | None = None) -> FusedExperimentRuntimeConfig:
    path = Path(config_path)
    config = _read_toml(path)
    defaults = config.get("defaults", {})
    selected_experiment = experiment_name or defaults.get("experiment_profile")
    experiment_raw = _require_named_section(config, "experiments", selected_experiment)
    fusion_name = str(experiment_raw["fusion_profile"])
    fusion = _build_fusion_profile_config(
        fusion_name,
        _require_named_section(config, "fusion_profiles", fusion_name),
    )
    components = tuple(
        _resolve_signal_component_runtime(
            config,
            component_name,
            default_factor_profile=experiment_raw.get("factor_profile", defaults.get("factor_profile")),
        )
        for component_name in fusion.components
    )
    data = _build_data_config(_require_named_section(config, "data", str(experiment_raw.get("data_profile", defaults.get("data_profile")))))
    trade = _build_trade_config(_require_named_section(config, "trading", str(experiment_raw.get("trade_profile", defaults.get("trade_profile")))))
    return FusedExperimentRuntimeConfig(
        experiment_name=selected_experiment,
        data=data,
        trade=trade,
        fusion=fusion,
        components=components,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_runtime_config -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add config.toml nusri_project/config/schemas.py nusri_project/config/runtime_config.py tests/test_runtime_config.py
git commit -m "feat: add fused runtime config support"
```

### Task 2: Generalize Component Training and Add the QLib Half-Life Reweighter

**Files:**
- Modify: `nusri_project/training/label_factory.py`
- Modify: `nusri_project/training/model_factory.py`
- Modify: `nusri_project/training/lgbm_workflow.py`
- Create: `nusri_project/training/time_decay_reweighter.py`
- Modify: `tests/test_lgbm_workflow_config.py`
- Create: `tests/test_time_decay_reweighter.py`

- [ ] **Step 1: Write the failing tests for generic cost-aware horizons and training weighting**

Add tests like:

```python
def test_build_label_mode_config_supports_24h_costaware(self) -> None:
    exprs, names = build_label_mode_config(
        label_mode="classification_24h_costaware",
        label_horizon_hours=24,
        positive_threshold=0.006,
    )
    self.assertEqual(exprs, ["If(Gt(Ref($close, -24) / $close - 1, 0.006), 1, 0)"])
    self.assertEqual(names, ["label_cls_24h_costaware"])

def test_exp_halflife_reweighter_is_monotonic_and_mean_one(self) -> None:
    weights = ExpHalflifeReweighter(reference_time="2025-01-31 23:00:00", half_life_months=6).reweight(frame)
    self.assertGreater(weights[-1], weights[0])
    self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_lgbm_workflow_config tests.test_time_decay_reweighter -v
```

Expected:
- FAIL because only `classification_72h_costaware` is supported and the reweighter file does not exist.

- [ ] **Step 3: Implement generic label modes, runtime-driven rolling windows, and the reweighter**

Generalize `label_factory.py`:

```python
def build_label_mode_config(*, label_mode: str, label_horizon_hours: int, positive_threshold: float) -> tuple[list[str], list[str]]:
    if label_mode.startswith("regression_"):
        return build_label_config(label_horizon_hours)
    if label_mode.startswith("classification_") and label_mode.endswith("_costaware"):
        return [get_cost_aware_binary_label_expr(label_horizon_hours, positive_threshold)], [f"label_cls_{label_horizon_hours}h_costaware"]
    raise ValueError(f"Unknown label_mode: {label_mode}")
```

Generalize `model_factory.py`:

```python
def get_model_loss(label_mode: str) -> str:
    if label_mode.startswith("regression_"):
        return "mse"
    if label_mode.startswith("classification_") and label_mode.endswith("_costaware"):
        return "binary"
    raise ValueError(f"Unknown label_mode: {label_mode}")
```

Create the reweighter:

```python
class ExpHalflifeReweighter(Reweighter):
    def __init__(self, *, reference_time: str, half_life_months: float) -> None:
        self.reference_time = pd.Timestamp(reference_time)
        self.half_life_hours = half_life_months * 30 * 24

    def reweight(self, data: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(data.index.get_level_values(0))
        age_hours = (self.reference_time - timestamps).total_seconds() / 3600.0
        weights = np.exp(-np.log(2) * age_hours / self.half_life_hours)
        return weights / weights.mean()
```

Update `lgbm_workflow.py` so rolling windows use runtime values:

```python
def _resolve_train_start(month_start: pd.Timestamp, training: TrainingConfig, data_start_ts: pd.Timestamp) -> pd.Timestamp:
    if training.training_window_months is None:
        return data_start_ts
    train_start = month_start - pd.DateOffset(months=training.training_window_months)
    return max(train_start, data_start_ts)

def _build_reweighter(training: TrainingConfig, train_end_dt: datetime) -> Reweighter | None:
    if training.sample_weight_mode == "uniform":
        return None
    if training.sample_weight_mode == "exp_halflife":
        return ExpHalflifeReweighter(reference_time=train_end_dt.strftime("%Y-%m-%d %H:%M:%S"), half_life_months=float(training.half_life_months))
    raise ValueError(f"unsupported sample_weight_mode: {training.sample_weight_mode}")
```

Use it at fit time:

```python
reweighter = _build_reweighter(training_config, train_end_dt)
model.fit(dataset, reweighter=reweighter)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_lgbm_workflow_config tests.test_time_decay_reweighter -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add nusri_project/training/label_factory.py nusri_project/training/model_factory.py nusri_project/training/lgbm_workflow.py nusri_project/training/time_decay_reweighter.py tests/test_lgbm_workflow_config.py tests/test_time_decay_reweighter.py
git commit -m "feat: generalize component training and add time decay weights"
```

### Task 3: Add Signal Transforms and the Fused Workflow Orchestrator

**Files:**
- Create: `nusri_project/training/signal_transform.py`
- Create: `nusri_project/training/fused_signal_workflow.py`
- Create: `scripts/training/fused_signal_workflow.py`
- Create: `tests/test_signal_transform.py`
- Create: `tests/test_fused_signal_workflow.py`

- [ ] **Step 1: Write the failing tests for transform fitting and fusion orchestration**

Add tests like:

```python
def test_robust_norm_clip_uses_train_only_statistics(self) -> None:
    train = pd.Series([0.0, 1.0, 2.0, 3.0])
    test = pd.Series([3.0, 4.0])
    params = fit_component_transform(train, transform="robust_norm_clip", clip_value=3.0)
    result = apply_component_transform(test, transform="robust_norm_clip", params=params, clip_value=3.0)
    self.assertTrue((result <= 1.0).all())
    self.assertTrue((result >= -1.0).all())

def test_fuse_component_predictions_emits_pred_score_and_real_return(self) -> None:
    fused = fuse_component_predictions(component_frames, weights=(0.4, 0.6), output_column="pred_score")
    self.assertIn("pred_score", fused.columns)
    self.assertIn("real_return", fused.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_signal_transform tests.test_fused_signal_workflow -v
```

Expected:
- FAIL because the transform module and fused workflow do not exist yet.

- [ ] **Step 3: Implement transform fitting and fused orchestration**

Create `signal_transform.py`:

```python
def fit_component_transform(values: pd.Series, *, transform: str, clip_value: float = 3.0) -> dict[str, float]:
    if transform != "robust_norm_clip":
        raise ValueError(f"unsupported transform: {transform}")
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = mad * 1.4826 if mad > 0 else float(values.std(ddof=0) or 1.0)
    return {"median": median, "scale": scale, "clip_value": clip_value}

def apply_component_transform(values: pd.Series, *, transform: str, params: dict[str, float], clip_value: float = 3.0) -> pd.Series:
    normalized = (values - params["median"]) / params["scale"]
    clipped = normalized.clip(-clip_value, clip_value)
    return clipped / clip_value
```

Create core fusion helper:

```python
def fuse_component_predictions(component_frames: dict[str, pd.DataFrame], *, weights: tuple[float, ...], output_column: str = "pred_score") -> pd.DataFrame:
    score_map = {name: frame["component_score"] for name, frame in component_frames.items()}
    aligned_scores = pd.concat(score_map, axis=1).dropna()
    aligned_return = next(iter(component_frames.values())).loc[aligned_scores.index, "real_return"]
    weight_series = pd.Series(weights, index=aligned_scores.columns, dtype=float)
    fused_score = aligned_scores.mul(weight_series, axis=1).sum(axis=1) / weight_series.sum()
    return pd.DataFrame({output_column: fused_score, "real_return": aligned_return})
```

Add fused workflow skeleton:

```python
def run_fused_rolling_monthly(runtime: FusedExperimentRuntimeConfig, *, prediction_output_dir: str, debug_component_output_dir: str | None = None) -> None:
    output_dir = Path(prediction_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for month_label, component_frames in build_monthly_component_frames(runtime):
        transformed_frames: dict[str, pd.DataFrame] = {}
        for component_name, frame in component_frames.items():
            params = fit_component_transform(
                frame["component_score_train"],
                transform=runtime.fusion.component_transform,
            )
            transformed_frames[component_name] = pd.DataFrame(
                {
                    "component_score": apply_component_transform(
                        frame["component_score_test"],
                        transform=runtime.fusion.component_transform,
                        params=params,
                    ),
                    "real_return": frame["real_return_test"],
                }
            )
        fused = fuse_component_predictions(
            transformed_frames,
            weights=runtime.fusion.weights,
            output_column=runtime.fusion.output_column,
        )
        fused.to_pickle(output_dir / f"pred_{runtime.fusion.name}_{month_label}.pkl")
```

Add CLI wrapper:

```python
from nusri_project.training.fused_signal_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_signal_transform tests.test_fused_signal_workflow -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add nusri_project/training/signal_transform.py nusri_project/training/fused_signal_workflow.py scripts/training/fused_signal_workflow.py tests/test_signal_transform.py tests/test_fused_signal_workflow.py
git commit -m "feat: add fused signal workflow orchestration"
```

### Task 4: Add Continuous Score Mapping and the `pred_score` Strategy Path

**Files:**
- Create: `nusri_project/strategy/continuous_position_mapping.py`
- Create: `nusri_project/strategy/score_signal_strategy.py`
- Modify: `nusri_project/strategy/strategy_config.py`
- Modify: `nusri_project/strategy/backtest_spot_strategy.py`
- Create: `tests/test_continuous_position_mapping.py`
- Modify: `tests/test_backtest_spot_strategy.py`

- [ ] **Step 1: Write the failing tests for `gated_power_curve` and `pred_score` backtesting**

Add tests like:

```python
def test_compute_target_weight_from_score_signal_uses_gate_and_power_curve(self) -> None:
    target = compute_target_weight_from_score_signal(
        pred_score=0.60,
        current_weight=0.0,
        max_position=0.25,
        open_score=0.40,
        close_score=0.20,
        size_floor_score=0.40,
        size_full_score=0.80,
        curve_gamma=1.0,
        min_holding_bars=0,
        holding_bars=0,
        cooldown_bars=0,
        bars_since_trade=99,
        drawdown=0.0,
        drawdown_de_risk_threshold=0.08,
        de_risk_position=0.0,
    )
    self.assertAlmostEqual(target, 0.125)

def test_build_backtest_components_uses_score_signal_column(self) -> None:
    config = SpotStrategyConfig(signal_kind="score", open_score=0.4, close_score=0.2, size_floor_score=0.4, size_full_score=0.8, curve_gamma=1.5, max_position=0.25)
    strategy_config, _, _ = build_backtest_components(signal, config)
    self.assertEqual(strategy_config["class"], "QlibScoreLongFlatStrategy")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_continuous_position_mapping tests.test_backtest_spot_strategy -v
```

Expected:
- FAIL because the score-based strategy path does not exist.

- [ ] **Step 3: Implement the score mapping and strategy path**

Create mapping helper:

```python
def compute_target_weight_from_score_signal(*, pred_score: float, current_weight: float, max_position: float, open_score: float, close_score: float, size_floor_score: float, size_full_score: float, curve_gamma: float, min_holding_bars: int, holding_bars: int, cooldown_bars: int, bars_since_trade: float, drawdown: float, drawdown_de_risk_threshold: float, de_risk_position: float) -> float:
    if current_weight <= 0 and pred_score < open_score:
        target_weight = 0.0
    elif current_weight > 0 and pred_score <= close_score:
        target_weight = 0.0
    else:
        u = (pred_score - size_floor_score) / (size_full_score - size_floor_score)
        u = min(max(u, 0.0), 1.0)
        target_weight = max_position * (u ** curve_gamma)
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
```

Add strategy config fields:

```python
open_score: float | None = None
close_score: float | None = None
size_floor_score: float | None = None
size_full_score: float | None = None
curve_gamma: float | None = None
```

Route backtest runner to `pred_score`:

```python
signal_column = {
    "probability": "pred_prob",
    "return": "pred_return",
    "score": "pred_score",
}[config.signal_kind]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_continuous_position_mapping tests.test_backtest_spot_strategy -v
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add nusri_project/strategy/continuous_position_mapping.py nusri_project/strategy/score_signal_strategy.py nusri_project/strategy/strategy_config.py nusri_project/strategy/backtest_spot_strategy.py tests/test_continuous_position_mapping.py tests/test_backtest_spot_strategy.py
git commit -m "feat: add continuous score execution path"
```

### Task 5: Wire the Two Fused Mainline Profiles and Run Final Verification

**Files:**
- Modify: `config.toml`
- Modify: `tests/test_runtime_config.py`
- Modify: `tests/test_fused_signal_workflow.py`
- Modify: `tests/test_backtest_spot_strategy.py`

- [ ] **Step 1: Write the failing tests for the two supported fused mainlines**

Add tests like:

```python
def test_regression_fused_main_resolves_only_two_regression_components(self) -> None:
    runtime = load_fused_runtime_config(config_path, experiment_name="regression_fused_main")
    self.assertEqual(tuple(component.name for component in runtime.components), ("reg_24h", "reg_72h"))
    self.assertEqual(runtime.fusion.output_column, "pred_score")

def test_costaware_fused_main_resolves_only_two_costaware_components(self) -> None:
    runtime = load_fused_runtime_config(config_path, experiment_name="costaware_fused_main")
    self.assertEqual(tuple(component.name for component in runtime.components), ("cls_24h_costaware", "cls_72h_costaware"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_runtime_config tests.test_fused_signal_workflow tests.test_backtest_spot_strategy -v
```

Expected:
- FAIL because the final fused experiments and score trade profile are not wired completely.

- [ ] **Step 3: Finish the config wiring and CLI defaults**

Make `config.toml` contain the final exposed profiles:

```toml
[trading.score_conservative]
signal_kind = "score"
open_score = 0.40
close_score = 0.20
size_floor_score = 0.40
size_full_score = 0.85
curve_gamma = 1.5
max_position = 0.25
min_holding_hours = 48
cooldown_hours = 12
drawdown_de_risk_threshold = 0.02
de_risk_position = 0.10

[experiments.regression_fused_main]
data_profile = "btc_1h_full"
fusion_profile = "regression_fused_main"
trade_profile = "score_conservative"

[experiments.costaware_fused_main]
data_profile = "btc_1h_full"
fusion_profile = "costaware_fused_main"
trade_profile = "score_conservative"
```

- [ ] **Step 4: Run final verification**

Run:

```bash
uv run python -m unittest tests.test_runtime_config tests.test_lgbm_workflow_config tests.test_time_decay_reweighter tests.test_signal_transform tests.test_fused_signal_workflow tests.test_continuous_position_mapping tests.test_backtest_spot_strategy -v
```

Expected:
- PASS

Optional smoke commands after unit tests pass:

```bash
uv run python -m scripts.training.fused_signal_workflow --config config.toml --experiment-profile regression_fused_main --prediction-output-dir /tmp/regression-fused-smoke
uv run python -m scripts.training.fused_signal_workflow --config config.toml --experiment-profile costaware_fused_main --prediction-output-dir /tmp/costaware-fused-smoke
```

Expected:
- Monthly `pred_score` artifacts appear under each output directory without tracebacks.

- [ ] **Step 5: Commit**

```bash
git add config.toml tests/test_runtime_config.py tests/test_fused_signal_workflow.py tests/test_backtest_spot_strategy.py
git commit -m "feat: wire fused signal mainline profiles"
```

## Self-Review Checklist

- Spec coverage:
  - Two fused public branches: covered by Tasks 1, 3, and 5.
  - Generic component/fusion interfaces: covered by Task 1.
  - Half-life weighting via QLib reweighter: covered by Task 2.
  - Continuous `pred_score` execution path: covered by Task 4.
  - Validation and ranking flow: covered by Task 5.
- Placeholder scan:
  - No placeholder markers or deferred notes remain.
- Type consistency:
  - Public prediction column stays `pred_score`.
  - New trade signal kind stays `score`.
  - Training weighting names stay `uniform` and `exp_halflife`.
