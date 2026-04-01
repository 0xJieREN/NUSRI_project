# Cost-Aware Label Round 1 Design

> **Status Update (2026-04-01):** This design is preserved as historical label-optimization context. The current recommended stage is `regression_fused_aggressive_v3_best`; see `docs/research/2026-04-01-regression-fused-best-stage-summary.md`.

**Goal**

Evaluate whether a `72h` cost-aware binary label can outperform the current `72h` regression label by reducing low-quality trades in the 2025 final backtest.

**Problem Statement**

The first label-optimization round established that `72h` regression labels are the best of the tested regression horizons, but even with the strongest `72h` trading shell, the 2025 final backtest still remains negative. This suggests that the main failure mode is not horizon choice alone; it is likely the model still emits too many trades whose expected edge does not survive costs.

**Fixed Inputs**

- Feature family remains unchanged
- Trading shell family remains unchanged
- Label horizon remains fixed at `72h`
- Spot backtest implementation remains unchanged

**New Label Definition**

This round introduces a cost-aware binary label:

- horizon: `72h`
- round-trip fee estimate: `0.2%`
- safety margin: `0.3%`
- positive threshold: `0.5%`

Definition:

- class `1`: future `72h` return `> 0.5%`
- class `0`: future `72h` return `<= 0.5%`

**Comparison Scope**

The round compares exactly two label modes under the same `72h` setup:

1. `72h` regression
2. `72h` cost-aware binary classification

Each label mode will be tested against two fixed trading shells:

- balanced shell
- conservative shell

This yields four directly comparable experiments.

**Architecture**

The training workflow will support a label-mode abstraction rather than a single label expression. The regression mode will continue to use `LGBModel(loss="mse")`, while the cost-aware mode will use `LGBModel(loss="binary")`. Prediction artifact shape will remain compatible with the existing strategy evaluation path, but score semantics will differ:

- regression mode score: expected forward return proxy
- classification mode score: probability-like positive-class score

**Success Criteria**

This round is successful if the cost-aware binary label does one of the following:

1. improves 2025 annualized return relative to `72h` regression under the same shell, or
2. materially reduces 2025 loss while preserving acceptable drawdown, or
3. improves 2024 Sharpe without increasing drawdown in a way that breaks your constraints.

**Failure Criteria**

Stop extending this direction if:

- classification underperforms regression on both shells in both 2024 and 2025, or
- classification becomes too sparse and effectively stops trading, or
- classification marginally improves return but destroys risk-adjusted quality.

**Out Of Scope**

- Multi-threshold classification
- Cost-aware continuous regression
- New factor engineering
- Model family changes
- Trading-shell redesign
