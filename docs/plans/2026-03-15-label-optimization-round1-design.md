# Label Optimization Round 1 Design

> **Status Update (2026-04-01):** This design is preserved as historical label-optimization context. The current recommended stage is `regression_fused_aggressive_v3_best`; see `docs/research/2026-04-01-regression-fused-best-stage-summary.md`.

**Goal**

Run the first controlled label-optimization round by comparing `24h`, `48h`, and `72h` forward-return regression labels under the already validated low-drawdown spot trading framework.

**Problem Statement**

The current repository has a working spot `long/flat` trading layer and can already produce feasible low-drawdown candidates on the 2024 development set. However, those candidates do not carry enough return into the 2025 final backtest set. At this point, the main bottleneck is no longer execution or risk controls; it is signal quality under the current label definition.

**Scope**

- Keep the trading layer fixed
- Keep the current model family fixed: QLib `LGBModel`
- Keep the current feature family fixed for round 1
- Compare only regression labels:
  - `24h forward return`
  - `48h forward return`
  - `72h forward return`
- Use the existing phase 2 backtest shell to evaluate strategy outcomes

**Why This Is The Right Next Step**

The current strategy framework has already proven that:

1. the execution path is valid,
2. the risk controls are expressible in QLib,
3. parameter semantics are now credible after fixing single-asset sizing.

That means the next useful experiment is to change the prediction target while holding the execution shell steady. This is the cleanest way to test whether longer-horizon labels improve tradable signal density and 2025 robustness.

**Design Principles**

- Change one major variable at a time
- Do not enlarge the search space unnecessarily in round 1
- Preserve comparability with all existing 2024 / 2025 strategy results
- Reuse current rolling workflow and saved prediction artifact shape

**Architecture**

Round 1 will add a configurable label horizon to the training workflow and make all new experiment outputs carry the label-horizon identity. The phase 2 evaluation code will then run a narrow comparison against a small set of already validated trading-parameter shells instead of running a full combinational search immediately.

**Round 1 Experiment Matrix**

Label candidates:

- `24h`
- `48h`
- `72h`

Trading shells to attach to each label:

- Balanced shell:
  - `entry_threshold=0.0015`
  - `exit_threshold=0.0`
  - `full_position_threshold=0.003`
  - `max_position=0.25`
  - `min_holding_hours=48`
  - `cooldown_hours=12`
  - `drawdown_de_risk_threshold=0.02`
  - `de_risk_position=0.25`
- Conservative shell:
  - `entry_threshold=0.0015`
  - `exit_threshold=0.0`
  - `full_position_threshold=0.003`
  - `max_position=0.15`
  - `min_holding_hours=48`
  - `cooldown_hours=12`
  - `drawdown_de_risk_threshold=0.02`
  - `de_risk_position=0.25`

This keeps round 1 small enough to interpret while still comparing two different exposure styles.

**Expected Code Changes**

- Training workflow:
  - add label horizon configurability
  - expose horizon in CLI
  - encode horizon in experiment naming and/or output paths
- Analysis workflow:
  - add a small helper to run the same evaluation flow for multiple label horizons
  - keep report shape and ranking logic unchanged where possible

**Success Criteria**

Round 1 is successful if at least one of the new horizons does one of the following:

1. improves 2024 strategy quality under the fixed trading shell without violating drawdown limits, or
2. improves 2025 performance relative to the current baseline candidates.

**Failure Criteria**

Round 1 is not worth extending if all three horizons:

- underperform the current label on 2024, or
- show the same 2025 collapse pattern while adding no risk-adjusted advantage.

**Out Of Scope**

- Cost-aware labels
- Classification labels
- New factors
- New models
- Large hyperparameter search
- Trading-layer redesign
