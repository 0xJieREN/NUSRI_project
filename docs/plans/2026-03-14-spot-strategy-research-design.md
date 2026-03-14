# Spot Strategy Research Design

**Goal**

Build a repeatable research and backtesting flow for a BTCUSDT spot `long/flat` strategy that uses model predictions as timing signals and evaluates success by net strategy performance rather than prediction correlation alone.

**Scope**

- Development period: `2024-01-01 00:00:00` to `2024-12-31 23:00:00`
- Final backtest period: `2025-01-01 00:00:00` to `2025-12-31 23:00:00`
- Trading mode: Binance spot `long/flat` only
- Capital base: `100000 USD`
- Fee model: `0.1%` per side
- Primary objective: maximize risk-adjusted return
- Hard constraints:
  - annualized return must exceed `4%`
  - max drawdown must remain below `10%`

**Why The Current Setup Is Not Enough**

The current repository can train and score a predictive model, but it does not yet turn those predictions into a realistic cash-managed spot strategy. The reported `IC`, `RankIC`, and directional accuracy are useful diagnostics, but they are not sufficient under a `0.2%` round-trip cost model. A weak signal can still look statistically non-random while being economically non-tradable.

**Research Protocol**

All forward work should follow these rules:

1. Use `2024` only for strategy development, threshold selection, and model comparison.
2. Treat `2025` as the final locked backtest set. Do not repeatedly tune against it.
3. Compare experiments by net strategy metrics first, predictive metrics second.
4. Keep the first strategy layer intentionally simple:
   - position states limited to `0%`, `50%`, `100%`
   - default state is cash
   - thresholds determine entry, add, reduce, and exit
   - minimum holding time and cooldown reduce churn
   - drawdown de-risking caps exposure when equity deteriorates
5. Prefer fewer moving parts over more complex logic until a robust baseline exists.

**Phase 0 Deliverables**

- A fixed evaluation protocol for all future experiments
- A standard metric set:
  - annualized return
  - annualized volatility
  - Sharpe ratio
  - Calmar ratio
  - max drawdown
  - turnover
  - average holding hours
  - exposure ratio
- A baseline strategy configuration that can be reused across experiments

**Phase 1 Deliverables**

- A Qlib-first spot backtest layer that reuses official backtest interfaces wherever they fit the strategy shape
- A three-state `long/flat` execution model with fees, hysteresis, and cash management, expressed through Qlib-native strategy and executor hooks when possible
- Persisted outputs for:
  - equity curve
  - monthly returns
  - summary metrics
- Tests covering:
  - prediction frame normalization
  - fee-aware position transitions
  - minimum holding and cooldown behavior
  - summary metric calculation

**Initial Strategy Assumptions**

- Predictions are interpreted as expected forward return for the matching horizon.
- The strategy rebalances only when thresholds imply a state change.
- Trading cost is charged on absolute position change.
- Position changes happen before the next realized forward return is applied.
- Drawdown protection is simple and deterministic in the first version.

**Out Of Scope For Phase 0/1**

- Short selling
- Leverage
- Complex stop-loss trees
- Portfolio optimization across multiple assets
- Deep hyperparameter search
- New factor engineering
- New labels

**Success Criteria For Phase 1**

Phase 1 is successful if the repository can run a fee-aware spot backtest through Qlib-native machinery where feasible and emit a consistent summary table that can be used to decide whether the current signal stack is worth further development.
