# 配置驱动研究仓重构设计

> **状态更新（2026-04-01）**：本设计文档保留为配置驱动重构阶段记录。当前仓库推荐阶段已更新为 `regression_fused_aggressive_v3_best`，详见 `docs/research/2026-04-01-regression-fused-best-stage-summary.md`。

## 目标

将当前以脚本和分散默认值驱动的研究仓，重构为以单一配置真源驱动的研究仓。重构后：

- 所有研究变量统一收敛到一个 `config.toml`
- 回归标签和分类标签拥有清晰、互不混淆的交易语义
- 训练、回测、扫描、报告脚本统一从配置文件读取实验组合
- 后续新增配置、废弃旧配置、扩展新实验时，不再依赖复制脚本和散落常量

## 当前问题

### 1. 配置来源分散

当前配置同时散落在：

- `nusri_project/training/lgbm_workflow.py`
- `nusri_project/strategy/strategy_config.py`
- `nusri_project/strategy/label_optimization_round1.py`
- `nusri_project/strategy/phase2_strategy_research.py`
- 多个 `scripts/analysis/*.py`

这会导致：

- 默认值难以追踪
- 实验复现依赖记忆和脚本上下文
- 同一配置语义在不同文件中重复定义

### 2. 分类信号和回归信号语义混淆

当前分类标签使用 LightGBM `binary` 目标，但预测输出会通过 `transform_prediction_score` 被映射为伪收益，然后再走回归交易阈值：

- 回归信号本应比较预测收益率
- 分类信号本应比较事件发生概率

这种混用带来：

- 阈值语义不自然
- 解释成本高
- 后续调参很容易误用

### 3. 实验脚本不是配置驱动

当前实验脚本多数内置 profile、参数网格和默认值。这样的问题是：

- 新实验常常要复制脚本或改脚本常量
- 参数扫描范围无法统一管理
- 旧 profile 难以清理

## 设计原则

### 1. 单一真源

所有研究变量统一由一个顶层配置文件管理：

- `config.toml`

代码中的 dataclass 和运行时配置对象只作为配置解析结果，不再承载独立默认值。

### 2. 分层配置

配置按研究层次划分：

- 数据层
- 因子层
- 标签层
- 模型层
- 训练层
- 交易层
- 实验组合层

实验组合层只负责引用其他 profile，不重复定义底层参数。

### 3. 回归与分类彻底分流

回归模式和分类模式共享回测骨架，但不共享交易信号语义。

#### 回归模式

- 输出字段：`pred_return`
- 交易阈值：
  - `entry_threshold`
  - `exit_threshold`
  - `full_position_threshold`

#### 分类模式

- 输出字段：`pred_prob`
- 含义：`P(未来 horizon 收益 > positive_threshold)`
- 交易阈值：
  - `enter_prob_threshold`
  - `exit_prob_threshold`
  - `full_prob_threshold`

`transform_prediction_score` 作为过渡逻辑，直接废弃，不做兼容保留。

### 4. QLib-first

保留现有 QLib-first 思路：

- 官方 `backtest(...)`
- `SimulatorExecutor`
- `Exchange`
- `WeightStrategyBase`
- `PortAnaRecord` 保持可接入

重构重点放在：

- 配置中心
- 信号层与交易层语义分流
- 实验编排统一

## 新配置结构

顶层主文件：

- `config.toml`

内部按 profile 组织：

- `[defaults]`
- `[data.<name>]`
- `[factors.<name>]`
- `[labels.<name>]`
- `[models.<name>]`
- `[training.<name>]`
- `[trading.<name>]`
- `[experiments.<name>]`

### 示例结构

```toml
[defaults]
data_profile = "btc_1h_full"
factor_profile = "top23"
label_profile = "classification_72h_costaware"
model_profile = "lgbm_binary_default"
training_profile = "rolling_2y_monthly"
trade_profile = "prob_conservative"
experiment_profile = "cost_aware_main"

[data.btc_1h_full]
start_time = "2019-09-10 08:00:00"
end_time = "2025-12-31 23:00:00"
freq = "60min"
provider_uri = "./qlib_data/my_crypto_data"
instrument = "BTCUSDT"
fields = ["ohlcv", "amount", "taker_buy_base_volume", "taker_buy_quote_volume", "funding_rate"]

[labels.classification_72h_costaware]
kind = "classification_costaware"
horizon_hours = 72
round_trip_cost = 0.002
safety_margin = 0.003
positive_threshold = 0.005

[trading.prob_conservative]
signal_kind = "probability"
enter_prob_threshold = 0.65
exit_prob_threshold = 0.50
full_prob_threshold = 0.80
max_position = 0.15
min_holding_hours = 48
cooldown_hours = 12
drawdown_de_risk_threshold = 0.02
de_risk_position = 0.0
```

## 配置项收敛结果

### 数据层

保留：

- `start_time`
- `end_time`
- `freq`
- `provider_uri`
- `instrument`
- `fields`
- `deal_price`
- `initial_cash`
- `fee_rate`
- `min_cost`

### 因子层

保留：

- `alpha158`
- `alpha261`
- `top23`

配置仅表达“选哪套因子库”，不在 TOML 中内联因子公式。

### 标签层

保留：

- `kind`
  - `regression`
  - `classification_costaware`
- `horizon_hours`
  - `8`
  - `24`
  - `48`
  - `72`

分类专属：

- `round_trip_cost`
- `safety_margin`
- `positive_threshold`

### 模型层

保留：

- `model_type`
- `objective`
- `hyperparameters`

### 训练层

保留：

- `run_mode`
  - `rolling`
  - `single`
- `training_window`
  - `all`
  - `2y`
- `rolling_step_months`

### 交易层

#### 回归交易壳

- `entry_threshold`
- `exit_threshold`
- `full_position_threshold`
- `max_position`
- `min_holding_hours`
- `cooldown_hours`
- `drawdown_de_risk_threshold`
- `de_risk_position`

#### 分类交易壳

- `enter_prob_threshold`
- `exit_prob_threshold`
- `full_prob_threshold`
- `max_position`
- `min_holding_hours`
- `cooldown_hours`
- `drawdown_de_risk_threshold`
- `de_risk_position`

## 校验规则

### 回归模式

- `full_position_threshold >= entry_threshold >= exit_threshold`

### 分类模式

- `full_prob_threshold >= enter_prob_threshold >= exit_prob_threshold`
- 所有概率阈值必须在 `[0, 1]`

### 风控约束

- `0 <= de_risk_position <= max_position <= 1`
- 若 `de_risk_position == max_position`，应报 warning

### 标签阈值约束

- `positive_threshold == round_trip_cost + safety_margin`

## 新代码结构

新增和调整以下单元：

- `config.toml`
- `nusri_project/config/schemas.py`
- `nusri_project/config/runtime_config.py`
- `nusri_project/training/label_factory.py`
- `nusri_project/training/model_factory.py`
- `nusri_project/strategy/return_signal_strategy.py`
- `nusri_project/strategy/probability_signal_strategy.py`
- `nusri_project/strategy/backtest_runner.py`

现有文件保留但弱化职责：

- `nusri_project/training/lgbm_workflow.py`
- `nusri_project/strategy/strategy_config.py`
- `nusri_project/strategy/qlib_spot_strategy.py`
- `nusri_project/strategy/phase2_strategy_research.py`

## 分阶段实施

### 阶段 1：配置中心落地

- 新增 `config.toml`
- 新增 schema 和 loader
- 让训练和回测脚本接受统一配置
- 不改变当前主线策略行为

### 阶段 2：回归/分类交易逻辑分流

- 删除 `transform_prediction_score`
- 分类输出 `pred_prob`
- 新增概率交易策略
- 回归交易继续保留 `pred_return`

### 阶段 3：实验脚本迁移到配置驱动

- 标签优化 round1
- cost-aware round1
- 72h trade tuning
- phase2 baseline

统一从 `config.toml` 读取 profile 和参数网格。

### 阶段 4：清理旧逻辑

- 清理脚本内嵌 profile
- 降级 `strategy_config.py` 默认值角色
- 更新 README/AGENTS/CLAUDE 文档

## 第一轮落地范围

为了避免一次性改炸，第一轮重构只强制跑通：

- `1h` 数据主线
- `top23`
- `72h regression`
- `72h classification_costaware`
- `LightGBM`
- `rolling_2y_monthly`
- 回归交易壳
- 概率交易壳

其他 profile 可先留在配置结构中，但不要求第一轮全部验证通过。

## 当前推荐阶段

当前默认推荐阶段应理解为：

- 数据：`btc_1h_full`
- 因子：`top23`
- 融合 profile：`regression_fused_main`
- 模型：`lgbm_regression_default`
- 训练：`rolling_24m_halflife_6m`
- 交易：`score_regression_aggressive_v3_best`
- 实验：`regression_fused_aggressive_v3_best`

这能保证重构后的默认世界观与当前最佳阶段一致。
