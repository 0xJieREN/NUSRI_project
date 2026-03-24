# Cost-Aware Optimization Experiment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 `2025` 最大回撤显著低于 `10%` 的前提下，把 `cost_aware_main` 从当前“小幅正收益但过于保守”的状态推进到更高的风险调整后收益水平。

**Architecture:** 当前主线固定为 `classification_72h_costaware + probability trading shell`。后续实验按“先交易壳、再标签阈值、再特征、最后训练窗”的顺序推进，每一轮只改变一层变量，并保留 `regression_72h_main` 作为对照基线。所有实验统一通过 `config.toml`、配置驱动脚本和 `reports/html/` 报告页进行记录与比较。

**Tech Stack:** Python 3.12, uv, QLib, LightGBM, config-driven experiment profiles, local HTML reporting.

---

## Current Baseline

当前建议固定的 4 组基线结果如下，用于后续所有实验对照：

| profile | year | annualized return | sharpe | max drawdown | turnover | exposure |
|---|---:|---:|---:|---:|---:|---:|
| `regression_72h_main` | 2024 | `9.60%` | `1.61` | `6.65%` | `27.26` | `8.55%` |
| `regression_72h_main` | 2025 | `-0.60%` | `-0.25` | `2.44%` | `5.26` | `1.79%` |
| `cost_aware_main` | 2024 | `-1.24%` | `-0.90` | `2.07%` | `3.29` | `0.92%` |
| `cost_aware_main` | 2025 | `0.30%` | `0.15` | `1.82%` | `8.12` | `2.45%` |

当前判断：
- `regression` 仍然更擅长样本内拟合，但 `2025` 泛化不足。
- `cost_aware` 已经在 `2025` 转为正收益，但资金利用率和暴露水平过低。
- 后续主线只围绕 `cost_aware_main` 优化，`regression` 只保留为比较基线。

## Files and Outputs

**Read / adjust:**
- `config.toml`
- `nusri_project/config/runtime_config.py`
- `nusri_project/strategy/probability_signal_strategy.py`
- `nusri_project/strategy/phase2_strategy_research.py`
- `nusri_project/strategy/research_profiles.py`
- `scripts/analysis/backtest_spot_strategy.py`
- `scripts/analysis/run_cost_aware_label_round1.py`
- `scripts/analysis/generate_html_reports.py`

**Primary outputs:**
- `reports/baseline-reg-2024`
- `reports/baseline-reg-2025`
- `reports/baseline-cost-2024`
- `reports/baseline-cost-2025`
- `reports/prob-tune-2024`
- `reports/prob-tune-2025`
- `reports/cost-thr-2024`
- `reports/cost-thr-2025`
- `reports/feature-cls-2024`
- `reports/feature-cls-2025`
- `reports/trainwin-2024`
- `reports/trainwin-2025`
- `reports/html/`

## Shared Evaluation Rules

- 所有实验优先比较 `2025`，`2024` 只作为开发集和稳定性参考。
- 统一看这 6 个指标：
  - `annualized_return`
  - `sharpe`
  - `max_drawdown`
  - `turnover`
  - `exposure_ratio`
  - `avg_holding_hours`
- 默认通过标准：
  - `2025` 年化收益高于当前基线 `0.30%`
  - `2025` `max_drawdown` 不高于 `10%`
  - `2025` Sharpe 不低于当前基线 `0.15`
- 若某轮实验没有明显提升，则停止在该方向继续细扫，进入下一层变量。

### Task 1: Freeze Baseline Results

**Files:**
- Read: `config.toml`
- Read: `reports/costaware-2024/summary.json`
- Read: `reports/costaware-2025/summary.json`
- Read: `reports/regression-2024/summary.json`
- Read: `reports/regression-2025/summary.json`
- Modify: `scripts/analysis/generate_html_reports.py` only if the default experiment set changes

- [ ] **Step 1: Confirm the four baseline reports still exist**

Run:
```bash
uv run python -m scripts.analysis.generate_html_reports \
  --reports-root reports \
  --output-root reports/html \
  --experiments regression-2024 regression-2025 costaware-2024 costaware-2025
```

Expected: `reports/html/index.html` contains exactly the four baseline experiments.

- [ ] **Step 2: Snapshot baseline metrics into a note or table**

Record the metrics from each `summary.json` so later experiments can be compared directly.

- [ ] **Step 3: Do not modify any baseline profile in `config.toml`**

This task ends only when the current baseline is frozen and reproducible.

### Task 2: Probability Shell Tuning

**Files:**
- Modify: `config.toml`
- Modify: `nusri_project/strategy/phase2_strategy_research.py` only if a probability scan helper is needed
- Read: `nusri_project/strategy/probability_signal_strategy.py`

**Purpose:**
先解决当前 `cost_aware_main` 太保守的问题，优先提升仓位利用率和净收益密度。

- [ ] **Step 1: Add dedicated probability scan profiles to `config.toml`**

Recommended first-pass search space:

```toml
[scan_profiles.prob_trade_tuning_fast]
kind = "probability_grid"
enter_prob_thresholds = [0.58, 0.62, 0.65, 0.68]
exit_prob_thresholds = [0.45, 0.50, 0.55]
full_prob_thresholds = [0.72, 0.78, 0.84]
max_positions = [0.15, 0.20, 0.25, 0.35]
min_holding_hours_list = [24, 48, 72]
cooldown_hours_list = [12]
drawdown_thresholds = [0.02, 0.04]
de_risk_positions = [0.0, 0.10, 0.20]
```

- [ ] **Step 2: Run the fast probability scan on 2024**

Run:
```bash
uv run python -m scripts.analysis.run_72h_trade_tuning \
  --predictions-root reports/costaware-preds \
  --config config.toml \
  --experiment-profile cost_aware_main \
  --year 2024 \
  --scan-profile prob_trade_tuning_fast \
  --output-root reports/prob-tune-2024 \
  --update-html
```

Expected: a ranked 2024 scan report under `reports/prob-tune-2024`.

- [ ] **Step 3: Select top 3 candidates by 2024 constraints**

Selection rule:
- `annualized_return > 4%`
- `max_drawdown < 10%`
- `sharpe` ranked highest among feasible candidates

- [ ] **Step 4: Re-run the selected candidates on 2025**

Expected output: `reports/prob-tune-2025`.

- [ ] **Step 5: Compare against baseline**

Pass if at least one candidate beats baseline `costaware-2025` on:
- higher annualized return than `0.30%`
- `max_drawdown` still materially below `10%`
- no severe turnover explosion

### Task 3: Cost-Aware Threshold Tuning

**Files:**
- Modify: `config.toml`
- Read: `nusri_project/training/label_factory.py`
- Read: `nusri_project/training/lgbm_workflow.py`

**Purpose:**
在概率壳较合理后，再调整“什么样的未来收益才算值得交易”。

- [ ] **Step 1: Add three label profiles**

Add:
- `classification_72h_costaware_loose` with `positive_threshold = 0.004`
- `classification_72h_costaware_base` with `positive_threshold = 0.005`
- `classification_72h_costaware_strict` with `positive_threshold = 0.006`

- [ ] **Step 2: Create matching experiment profiles**

Each should reuse:
- same data profile
- same factor profile
- same model profile
- same training profile
- same best trade profile from Task 2

- [ ] **Step 3: Train and backtest the three variants**

Recommended outputs:
- `reports/cost-thr-2024`
- `reports/cost-thr-2025`

- [ ] **Step 4: Select threshold by 2025 net result**

Pass if a threshold improves 2025 annualized return without causing:
- sharp MDD jump
- turnover surge that obviously destroys net returns

### Task 4: Classification Feature Shrink

**Files:**
- Modify: `config.toml`
- Read: `nusri_project/config/alpha261_config.py`
- Read: feature importance outputs only if needed for candidate set design

**Purpose:**
验证 `top23` 是否对分类任务过重，争取用更简单、更稳的特征集合获得更好的泛化。

- [ ] **Step 1: Define factor profiles for classification comparison**

Recommended profiles:
- `top23`
- `top15`
- `top10`
- one lighter order-flow / funding focused subset if the formulas already exist

- [ ] **Step 2: Keep label and probability trade profile fixed**

This experiment should change only the factor profile.

- [ ] **Step 3: Run 2024 then 2025 comparison**

Recommended outputs:
- `reports/feature-cls-2024`
- `reports/feature-cls-2025`

- [ ] **Step 4: Prefer the smallest set that preserves or improves 2025**

Pass if a smaller set:
- matches or beats baseline 2025 annualized return
- keeps MDD controlled
- simplifies the model

### Task 5: Training Window and Recency Weighting

**Files:**
- Modify: `config.toml`
- Modify: `nusri_project/training/lgbm_workflow.py` only if recent-sample weighting is not yet exposed
- Read: `nusri_project/training/model_factory.py`

**Purpose:**
如果前 3 轮已经得到更可信的分类信号和交易壳，再让训练窗口更贴近 `2025` regime。

- [ ] **Step 1: Add training profiles**

Recommended profiles:
- `rolling_1y_monthly`
- `rolling_18m_monthly`
- `rolling_2y_monthly`

- [ ] **Step 2: Optionally add recent-sample weighting**

Only do this if 1y / 18m / 2y still all不理想 but the rest of the stack looks stable.

- [ ] **Step 3: Reuse the best setup from Tasks 2-4**

Only training profile changes in this task.

- [ ] **Step 4: Evaluate 2025 first, then confirm 2024 stability**

Pass if:
- 2025 annualized return improves again
- Sharpe remains acceptable
- MDD does not jump into the danger zone

## Stop-Loss Rules

- If Task 2 fails to improve `costaware-2025`, do **not** over-invest in more trade-shell micro-tuning.
- If Task 3 also fails, the next priority becomes feature and training-window changes.
- If Task 4 shows no difference between `top23` and smaller subsets, prefer the smaller, simpler subset.
- If Task 5 still cannot materially improve `2025`, re-open the label definition itself instead of endlessly tuning execution.

## Recommended Execution Order

1. Freeze current baseline
2. Tune probability shell
3. Tune `positive_threshold`
4. Shrink classification features
5. Adjust training window / recent weighting

## Suggested Commit Rhythm

- `feat: add probability trade tuning profiles`
- `feat: compare cost-aware thresholds`
- `feat: add classification feature comparison profiles`
- `feat: compare training window profiles`
- `docs: summarize cost-aware optimization experiment results`

