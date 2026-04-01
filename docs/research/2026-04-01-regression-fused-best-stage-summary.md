# 2026-04-01 Regression Fused 当前最佳阶段总结

## 当前推荐阶段

截至 `2026-04-01`，仓库当前最优的端到端阶段不再是旧的 `cost_aware_main`，而是：

- 训练信号：`regression_fused_main`
- 交易执行：`regression_fused_aggressive_v3_best`

对应配置栈为：

- 数据：`btc_1h_full`
- 因子：`top23`
- 融合组件：`reg_24h + reg_72h`
- 融合 profile：`regression_fused_main`
- 模型：`lgbm_regression_default`
- 训练：`rolling_24m_halflife_6m`
- 交易：`score_regression_aggressive_v3_best`
- 对外实验：`regression_fused_aggressive_v3_best`

交易层参数为：

- `open_score = 0.45`
- `close_score = -0.05`
- `size_floor_score = 0.45`
- `size_full_score = 0.60`
- `curve_gamma = 1.0`
- `max_position = 0.75`
- `min_holding_hours = 48`
- `cooldown_hours = 12`
- `drawdown_de_risk_threshold = 0.02`
- `de_risk_position = 0.0`

## 当前最佳结果

这组参数来自已完成的更激进 score 扫描，并通过同一套 QLib 回测路径做了 `2024` 回看验证。

| year | annualized return | sharpe | max drawdown | turnover | exposure | avg holding hours |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | `3.97%` | `0.78` | `3.02%` | `6.05` | `2.11%` | `66.88` |
| 2025 | `17.29%` | `1.97` | `2.92%` | `22.09` | `4.54%` | `65.32` |

当前判断：

- 在已完成的扫描里，这组是 `2025 max_drawdown <= 10%` 约束下的最优候选。
- `2024` 仍保持正收益，说明它不是只靠单年偶然拟合。
- `max_position = 1.0` 这轮没有打赢 `0.75`，所以当前不建议继续无脑抬满仓上限。
- 更有效的放松来自：
  - 把 `close_score` 压到 `-0.05`
  - 把 `size_full_score` 压到 `0.60`
  - 保持 `open_score = size_floor_score = 0.45`

## 推荐使用方式

### 1. 生成当前主信号预测

```bash
uv run python -m scripts.training.fused_signal_workflow \
  --config config.toml \
  --experiment-profile regression_fused_main \
  --prediction-output-dir reports/fused-signal-preds/regression_fused_main
```

### 2. 回测当前最佳阶段

```bash
uv run python -m scripts.analysis.backtest_spot_strategy \
  --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2025*.pkl" \
  --config config.toml \
  --experiment-profile regression_fused_aggressive_v3_best \
  --start-time "2025-01-01 00:00:00" \
  --end-time "2025-12-31 23:00:00"
```

### 3. 做开发集回看

```bash
uv run python -m scripts.analysis.backtest_spot_strategy \
  --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2024*.pkl" \
  --config config.toml \
  --experiment-profile regression_fused_aggressive_v3_best \
  --start-time "2024-01-01 00:00:00" \
  --end-time "2024-12-31 23:00:00"
```

## 文档状态说明

下面这些文档仍保留，但它们记录的是旧阶段或中间设计，不应继续当作当前主线结论：

- `docs/research/2026-03-24-cost-aware-mainline-comparison.md`
- `docs/research/2026-03-15-研究收敛备忘录.md`
- `docs/plans/2026-03-24-cost-aware-optimization-experiment-plan.md`
- `docs/plans/2026-03-31-fused-signal-design.md`
- `docs/plans/2026-03-31-fused-signal-implementation-plan.md`

这些文档现在只保留历史背景、设计轨迹和实现记录的价值。
