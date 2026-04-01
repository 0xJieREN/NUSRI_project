# 2026-03-31 Fused Signal Design

> **状态更新（2026-04-01）**：本设计已经完成实现，且当前最佳阶段已经落在 `regression_fused_aggressive_v3_best`。本文件主要保留设计动机和接口边界，当前结果请参考 `docs/research/2026-04-01-regression-fused-best-stage-summary.md`。

## Goal

在 `2025 max_drawdown <= 10%` 的约束下，尽量提高策略收益。当前优先级不再受旧主线约束，允许围绕 `2025` 直接优化，但仍保留基本稳定性护栏，避免只对单年偶然拟合。

本设计只定义信号研究与执行接口，不在这一阶段实现跨大类信号融合。最终对外只保留两套融合信号：

- `regression_fused_main`
- `costaware_fused_main`

两套融合信号都统一输出连续列 `pred_score`，供后续连续仓位映射和回测复用。

## Why This Design

当前证据显示问题首先出在信号层，而不是简单的交易壳参数过于保守：

- 更激进的 probability shell 扫描没有自然把 `2024` 做正。
- 当前分类分数在 `2024/2025` 上没有稳定的正向排序能力。
- 单独提高 `max_position` 或降低 `enter_prob_threshold` 不能解决根因。

因此本轮工作先做两件事：

1. 把同类 horizon 信号融合成更稳定的连续分数。
2. 给每个基础信号接入基于半衰期的时间衰减训练权重。

## Non-Goals

- 本轮不实现 `regression_fused_main` 和 `costaware_fused_main` 之间的跨类融合。
- 本轮不引入黑盒 meta-model 或 stacking。
- 本轮不把仓位映射参数和融合权重一起做大规模联合搜索。
- 本轮不修改现有 QLib 回测骨架的总体结构。

## Public Interface

### Outward-facing experiments

对外只暴露两套融合实验：

- `regression_fused_main`
- `costaware_fused_main`

两者的 prediction artifacts 都统一包含：

- `pred_score`
- `real_return`

### Internal reusable layers

内部接口分三层：

1. `signal_components`
2. `fusion_profiles`
3. `experiments`

这样后续如果增加 `48h`、更换 `factor_profile`、更换模型或增加新的标签定义，不需要重写主流程。

## Config Model

### signal_components

`signal_components` 是最小可训练单元。每个 component 允许独立指定：

- `factor_profile`
- `label_profile`
- `model_profile`
- `training_profile`

第一版预期支持的 component：

- `reg_24h`
- `reg_72h`
- `cls_24h_costaware`
- `cls_72h_costaware`

### fusion_profiles

`fusion_profiles` 定义如何把多个 component 融成一条连续信号。

每个 fusion profile 至少包含：

- `components`
- `weights`
- `component_transform`
- `transform_fit_scope`
- `output_column`
- `cache_component_predictions`

第一版对外只定义：

- `regression_fused_main`
- `costaware_fused_main`

默认约束：

- `len(weights) == len(components)`
- `output_column = "pred_score"`
- `transform_fit_scope = "train_only"`

### experiments

`experiments` 改为引用 `fusion_profile`，而不是直接引用单个 `label_profile`。

每个 experiment 至少包含：

- `data_profile`
- `fusion_profile`
- `trade_profile`

如果 experiment 级别提供默认 `factor_profile`，则 component 未显式声明时可以回退到该默认值。component 级配置优先级更高。

## Training Interface

### training_profile fields

现有 `training_profile` 需要扩展成真正可驱动 rolling 训练和样本加权的接口。建议统一支持：

- `run_mode = "rolling" | "single"`
- `training_window_months`
- `rolling_step_months`
- `sample_weight_mode = "uniform" | "exp_halflife"`
- `half_life_months`

兼容策略：

- 现有 `training_window = "2y" | "all"` 仍可解析
- 内部统一归一成上述新字段

### Half-life weighting

时间加权不手写旁路 `sample_weight` 管道，而是复用 QLib `LGBModel.fit(..., reweighter=...)` 接口。

新增一个本地 reweighter，实现逻辑：

- 输入：训练样本的 datetime index
- 参考点：当前 rolling train window 的末端
- 权重公式：`exp(-ln(2) * age / half_life)`
- 权重归一化到均值约为 `1.0`

第一版默认半衰期候选从 `6` 个月开始。

## Workflow Architecture

### Single-component trainer

保留 `lgbm_workflow.py` 作为单 component 训练器，负责：

- 构造 component runtime
- 按 label / model / factor / training profile 训练单个模型
- 输出该 component 的 test predictions

该文件需要从“单标签主流程”泛化为“通用 component trainer”，但不负责多 component 融合 orchestration。

### Fused workflow

新增一个上层 fused workflow，职责是：

1. 读取 experiment
2. 解析 fusion profile
3. 展开 component 列表
4. 调用单 component trainer
5. 对齐多 component predictions
6. 做 component transform
7. 按权重融合
8. 输出 fused `pred_score`

这样可以避免把训练、融合、落盘都塞进同一个入口文件。

## Rolling Data Flow

每个月的 rolling 执行顺序固定如下：

1. 解析 `experiment -> fusion_profile -> components`
2. 对每个 component：
   - 解析自己的 `factor_profile / label_profile / model_profile / training_profile`
   - 计算 train/test 时间窗
   - 若 `sample_weight_mode = "exp_halflife"`，构造 QLib reweighter
   - 调用 `model.fit(dataset, reweighter=...)`
   - 生成当月 component test predictions
3. 按 `datetime + instrument` 对齐该月所有 component predictions
4. 对每个 component prediction 执行同一 fusion profile 指定的变换
5. 按权重生成 fused `pred_score`
6. 落 fused artifact

## Component Transform Rules

不允许直接用原始 `24h/72h` 输出做加权平均。默认变换接口支持：

- `raw`
- `rank_pct_centered`
- `robust_norm_clip`

第一版默认使用 `robust_norm_clip`。

### robust_norm_clip

对每个 component：

1. 用 train window 内该 component 的 prediction 分布拟合稳健位置和尺度
2. 对当月 test prediction 做标准化
3. clip 到固定范围，例如 `[-3, 3]`
4. 再缩放到 `[-1, 1]`

纪律要求：

- 变换参数只能从 train window prediction 中拟合
- 不允许用 test month 自身分布反推参数

### Output contract

融合后最终统一输出：

- `pred_score in [-1, 1]`

这保证后续连续仓位映射只消费一种标准化后的连续信号接口。

## Continuous Position Mapping

仓位映射采用可解释、可调、带 hysteresis 的连续 long/flat 规则，而不是当前的硬阶梯。

推荐映射：`gated_power_curve`

### Gate rules

- 空仓时，只有 `pred_score >= open_score` 才允许开仓
- 持仓时，若 `pred_score <= close_score` 则强制平仓
- `open_score > close_score`，形成 hysteresis

### Continuous sizing

若允许持仓，则：

- `u = clip((pred_score - size_floor_score) / (size_full_score - size_floor_score), 0, 1)`
- `target_weight = max_position * u^gamma`

其中：

- `size_floor_score` 以上开始给仓位
- `size_full_score` 以上给满仓位上限
- `gamma` 控制曲线保守或激进程度

默认建议从 `gamma = 1.5` 起步。

### Existing guards kept

连续映射之后，仍保留现有风控逻辑：

- `min_holding_hours`
- `cooldown_hours`
- `drawdown_de_risk_threshold`
- `de_risk_position`

顺序为：

1. 先计算连续 `target_weight`
2. 再做 drawdown cap
3. 最后过 holding / cooldown guards

## Outputs

默认只落 fused predictions，component predictions 只在 debug 或 cache 模式下保留。

建议目录结构：

- `reports/fused-preds/<fusion_profile>/pred_<fusion_profile>_<YYYYMM>.pkl`
- `reports/fused-preds-debug/<fusion_profile>/<component>/pred_<component>_<YYYYMM>.pkl`

默认 fused artifact 至少包含：

- `pred_score`
- `real_return`

## Validation Plan

实验顺序固定为四轮，避免变量同时漂移：

1. 打通两套 fused signal，不接新仓位映射
2. 在 fused signal 上比较 `uniform` vs `exp_halflife`
3. 在更可信的 fused signal 上接连续仓位映射
4. 最后才做小范围联合细调

### Ranking rules

最终排序规则：

1. 先过滤：`2025 max_drawdown <= 10%`
2. 主排序：`2025 annualized_return`
3. 次排序：
   - `2025 sharpe`
   - `2025 calmar`
   - `turnover` 不爆炸
   - `2024` 不显著恶化

### Signal diagnostics

在接仓位映射前，先做 fused signal 诊断：

- 年度 IC
- 年度 RankIC
- 分位收益
- 月度稳定性
- `2024 / 2025` 对比

若 fused signal 本身排序能力没有明显改善，则不进入连续仓位映射细扫。

## File Scope

本设计预计涉及的实现文件：

- `config.toml`
- `nusri_project/config/schemas.py`
- `nusri_project/config/runtime_config.py`
- `nusri_project/training/label_factory.py`
- `nusri_project/training/lgbm_workflow.py`
- `nusri_project/training/time_decay_reweighter.py`
- `nusri_project/training/fused_signal_workflow.py`
- `nusri_project/training/signal_transform.py`
- `nusri_project/strategy/continuous_position_mapping.py`
- 对应测试文件

## Risks and Mitigations

### Risk: Too many degrees of freedom

Mitigation:

- 先固定两套 fused branches
- 先不做跨大类融合
- 实验按四轮顺序推进

### Risk: Leakage in component transform

Mitigation:

- 只允许用 train window 的 prediction 分布拟合变换参数

### Risk: Half-life weighting improves fit but hurts stability

Mitigation:

- 必须保留 `uniform` 对照
- 先比较信号诊断，再比较回测结果

### Risk: Continuous sizing over-amplifies noisy scores

Mitigation:

- 先验证 fused signal 排序能力
- 第一版使用保守的 `gamma > 1`
- 保留现有 holding / cooldown / drawdown guards

## Decision Summary

本轮设计的核心决策如下：

- 对外只保留 `regression_fused_main` 和 `costaware_fused_main`
- 对外统一 prediction column 为 `pred_score`
- 内部通过 `signal_components + fusion_profiles + experiments` 提供通用扩展接口
- 半衰期权重通过 QLib `reweighter` 接入
- 连续仓位映射采用 `gated_power_curve`
- 当前阶段不做跨大类融合
