# 配置驱动研究仓使用说明

## 当前状态

仓库已经开始从“脚本内置默认值驱动”迁移到“`config.toml` 单一真源驱动”。

当前推荐使用方式是：

- 训练、回测、扫描、标签比较都优先通过 `config.toml`
- 通过 `--config` + `--experiment-profile` 选择实验组合
- 回归信号、分类信号和 fused score 信号已经在交易层语义上明确分流

当前最佳阶段总结：

- `docs/research/2026-04-01-regression-fused-best-stage-summary.md`

## 配置文件

主配置文件：

- `config.toml`

其中按 profile 分层：

- `[data.*]`
- `[factors.*]`
- `[labels.*]`
- `[models.*]`
- `[training.*]`
- `[trading.*]`
- `[experiments.*]`

### 当前推荐阶段

当前默认推荐阶段建议理解为：

- 数据：`btc_1h_full`
- 因子：`top23`
- 融合组件：`reg_24h + reg_72h`
- 融合 profile：`regression_fused_main`
- 模型：`lgbm_regression_default`
- 训练：`rolling_24m_halflife_6m`
- 交易：`score_regression_aggressive_v3_best`
- 实验：`regression_fused_aggressive_v3_best`

## 标签与交易层语义

### 回归模式

- 预测列：`pred_return`
- 交易阈值：
  - `entry_threshold`
  - `exit_threshold`
  - `full_position_threshold`

### 分类模式

- 预测列：`pred_prob`
- 含义：
  - `P(未来 horizon 收益 > positive_threshold)`
- 交易阈值：
  - `enter_prob_threshold`
  - `exit_prob_threshold`
  - `full_prob_threshold`

### 重要变化

过去分类模式会把概率映射成伪收益，再复用回归交易阈值。  
现在这条执行路径已经废弃：

- 分类信号直接走 `pred_prob`
- 分类交易层直接比较概率阈值

### score / fused 模式

- 预测列：`pred_score`
- 含义：
  - 来自同类 horizon 组件融合后的连续分数
- 交易阈值：
  - `open_score`
  - `close_score`
  - `size_floor_score`
  - `size_full_score`
  - `curve_gamma`

## 常用命令

### 训练当前 fused signal

```bash
uv run python -m scripts.training.fused_signal_workflow --config config.toml --experiment-profile regression_fused_main --prediction-output-dir reports/fused-signal-preds/regression_fused_main
```

### 现货回测当前最佳阶段（2025）

```bash
uv run python -m scripts.analysis.backtest_spot_strategy \
  --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2025*.pkl" \
  --config config.toml \
  --experiment-profile regression_fused_aggressive_v3_best \
  --start-time "2025-01-01 00:00:00" \
  --end-time "2025-12-31 23:00:00"
```

### 现货回测当前最佳阶段（2024 回看）

```bash
uv run python -m scripts.analysis.backtest_spot_strategy \
  --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2024*.pkl" \
  --config config.toml \
  --experiment-profile regression_fused_aggressive_v3_best \
  --start-time "2024-01-01 00:00:00" \
  --end-time "2024-12-31 23:00:00"
```

### 历史研究工具

```bash
uv run python -m scripts.analysis.run_cost_aware_label_round1 \
  --predictions-root reports/costaware-preds \
  --config config.toml \
  --experiment-profile cost_aware_main \
  --year 2025 \
  --update-html

uv run python -m scripts.analysis.run_72h_trade_tuning \
  --predictions-root reports/costaware-preds \
  --config config.toml \
  --experiment-profile regression_72h_main \
  --year 2024 \
  --update-html
```

### phase2 baseline / scan

```bash
uv run python -m scripts.analysis.run_phase2_baseline \
  --mlruns-root mlruns \
  --config config.toml \
  --experiment-profile regression_72h_main \
  --year 2024 \
  --scan \
  --update-html
```

## 兼容层说明

以下内容仍然保留，但不再是新的配置真源：

- `nusri_project/strategy/strategy_config.py` 中的字段默认值
- 各脚本中为兼容保留的旧 CLI 参数

它们的角色是：

- 保持旧命令能继续运行
- 为尚未完全迁移的实验提供过渡入口

不要把它们当作当前研究默认值。

## 后续建议

当前剩余的收尾重点是：

- 优先围绕 `regression_fused_aggressive_v3_best` 做更窄的执行层微调
- 如果继续扩信号层，优先在 fused regression 分支内做更细的组件 / 权重研究
- 把更多历史文档降级为“阶段记录”，避免再把旧 cost-aware 主线当作当前默认结论
