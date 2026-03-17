# 配置驱动研究仓使用说明

## 当前状态

仓库已经开始从“脚本内置默认值驱动”迁移到“`config.toml` 单一真源驱动”。

当前推荐使用方式是：

- 训练、回测、扫描、标签比较都优先通过 `config.toml`
- 通过 `--config` + `--experiment-profile` 选择实验组合
- 回归信号与分类信号已经在交易层语义上明确分流

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

### 当前推荐主线

当前默认主线建议理解为：

- 数据：`btc_1h_full`
- 因子：`top23`
- 标签：`classification_72h_costaware`
- 模型：`lgbm_binary_default`
- 训练：`rolling_2y_monthly`
- 交易：`prob_conservative`
- 实验：`cost_aware_main`

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

## 常用命令

### 训练

```bash
uv run python -m scripts.training.lgbm_workflow --config config.toml --experiment-profile cost_aware_main
```

### 现货回测

```bash
uv run python -m scripts.analysis.backtest_spot_strategy \
  --pred-glob "reports/costaware-preds/pred_classification_72h_costaware_72h_2025*.pkl" \
  --config config.toml \
  --experiment-profile cost_aware_main
```

### cost-aware round1 对比

```bash
uv run python -m scripts.analysis.run_cost_aware_label_round1 \
  --predictions-root reports/costaware-preds \
  --config config.toml \
  --experiment-profile cost_aware_main \
  --year 2025 \
  --update-html
```

### 72h 交易层调参

```bash
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

- 继续减少脚本内置研究默认值
- 让更多扫描 profile 迁入 `config.toml`
- 将 README 和代理说明完全改成配置驱动表述
