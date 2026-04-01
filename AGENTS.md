# AGENTS.md — nusri-project（智能编码代理指南）

本仓库是 Python 3.12 的加密货币价格预测项目：
- QLib：数据处理 / 特征工程 / 数据集抽象
- LightGBM：通过 QLib `LGBModel` 训练
- 当前研究工作流已经收敛到 `config.toml` 驱动的 fused signal + score execution 路径

当前推荐的最佳阶段配置：
- 因子：`top23`
- 融合组件：`reg_24h + reg_72h`
- 融合 profile：`regression_fused_main`
- 模型：`LightGBM regression`
- 训练窗口：`rolling_24m_halflife_6m`
- 交易壳：`score_regression_aggressive_v3_best`
- 对外实验：`regression_fused_aggressive_v3_best`

最新研究结论文档：
- `docs/research/2026-04-01-regression-fused-best-stage-summary.md`

规则来源检查结果：
- 未发现 Cursor 规则：`.cursor/rules/`、`.cursorrules`
- 未发现 Copilot 规则：`.github/copilot-instructions.md`

---

## 1) 环境与依赖

- Python：`3.12`（见 `.python-version`）
- 依赖：`pyproject.toml`
- 锁文件：`uv.lock`

推荐使用 `uv`：
- 安装/同步：`uv sync`
- 运行脚本：`uv run python <script>.py [args...]`

如果没有 `uv`：可用 `venv + pip`，但避免在无锁环境下随意升级依赖。

---

## 2) 常用运行命令（脚本驱动）

本仓库无“build”步骤（不打包构建），主要是运行脚本。

### 2.1 获取 Binance 小时级数据（1h）

- `uv run python -m scripts.data.request_1h`

输出示例：`BTCUSDT_1h_binance_data.csv`

要点：
- 同时抓取期货衍生数据（如 `funding_rate`）并写入 CSV
- 网络请求包含 timeout + 有界重试；某些 400 会减少重试

### 2.2 原始 CSV → QLib source CSV

- `uv run python -m scripts.data.clean_data --input data/raw/BTCUSDT_1h_binance_data.csv --output qlib_source_data/BTCUSDT.csv`

要点：
- 默认分隔符 `;`（可 `--sep` 修改）
- 补齐 `symbol` 列
- `date` 规范到秒级 `%Y-%m-%d %H:%M:%S`

### 2.3 QLib source CSV → QLib binary 数据

- `uv run python -m scripts.data.dump_bin dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min`

要点：
- `scripts/data/dump_bin.py` 使用 `fire` CLI
- 输出目录包含 `calendars/ features/ instruments/ ...`

### 2.4 训练当前 fused signal（配置驱动主流程）

- `uv run python -m scripts.training.fused_signal_workflow --config config.toml --experiment-profile regression_fused_main --prediction-output-dir reports/fused-signal-preds/regression_fused_main`

当前最佳阶段的实际含义：
- `top23 + regression_fused_main + rolling_24m_halflife_6m + score_regression_aggressive_v3_best`

兼容旧参数模式（仅保留兼容，不作为研究主入口）：

- `uv run python -m scripts.training.lgbm_workflow`

要点：
- 新主线优先从 `config.toml` 读取 profile
- 旧参数模式仍保留，但不再是研究默认值真源

### 2.5 导出特征重要性

- `uv run python -m scripts.analysis.dump_lgbm_feature_importance --importance-type gain --out reports/feature_importance/lgbm_feature_importance.csv --top 20`

### 2.6 运行配置驱动回测 / 扫描

- `uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2025*.pkl" --config config.toml --experiment-profile regression_fused_aggressive_v3_best --start-time "2025-01-01 00:00:00" --end-time "2025-12-31 23:00:00"`
- `uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "reports/fused-signal-preds/regression_fused_main/pred_fused_2024*.pkl" --config config.toml --experiment-profile regression_fused_aggressive_v3_best --start-time "2024-01-01 00:00:00" --end-time "2024-12-31 23:00:00"`
- `uv run python -m scripts.analysis.run_phase2_baseline --mlruns-root ./mlruns --config config.toml --experiment-profile regression_72h_main --year 2024 --scan`
- `uv run python -m scripts.analysis.run_72h_trade_tuning --predictions-root reports/costaware-prob-preds --config config.toml --experiment-profile cost_aware_main --year 2024`
- `uv run python -m scripts.analysis.run_cost_aware_label_round1 --predictions-root reports/costaware-preds --config config.toml --experiment-profile cost_aware_main --year 2025`

当前最重要的结果文件：
- `docs/research/2026-04-01-regression-fused-best-stage-summary.md`
- `reports/score-scan-regression-fused-2025-v5-aggressive-focus/scan_2025.csv`
- `reports/score-scan-regression-fused-2025-v5-aggressive-focus/top8_2024_validation.csv`

---

## 3) 测试（当前与推荐）

### 3.1 当前仓库状态

- 未配置 pytest（未发现 `pytest.ini` / `conftest.py` / `pyproject` pytest 配置）
- 当前测试主入口是标准库 `unittest`
- 回测、配置、训练、fused signal 和 score execution 相关测试已经覆盖当前主路径

### 3.2 现有 smoke test（配置 / 训练 / fused / 回测）

- `uv run python -m unittest tests.test_runtime_config tests.test_lgbm_workflow_config tests.test_time_decay_reweighter tests.test_signal_transform tests.test_fused_signal_workflow tests.test_continuous_position_mapping tests.test_backtest_spot_strategy tests.test_phase2_strategy_research tests.test_analysis_entrypoints_config tests.test_research_profiles -v`

用途：验证当前最佳阶段的配置驱动训练、半衰期加权、fused workflow、连续仓位映射、扫描器和回测层。

### 3.3 若未来引入 pytest（建议的统一命令）

- 跑全部：`uv run pytest -q`
- 跑单文件：`uv run pytest -q path/to/test_file.py`
- 跑单用例（关键字）：`uv run pytest -q path/to/test_file.py -k test_name_substring`
- 跑单节点：`uv run pytest -q path/to/test_file.py::TestClass::test_method`

---

## 4) Lint / 格式化 / 类型检查（当前与建议）

当前状态：
- `pyproject.toml` 仅有运行依赖（无 ruff/black/isort/mypy 等配置）
- 环境里也不保证已安装上述工具（不要擅自引入）

如果维护者决定引入，建议：
- Ruff：`uv run ruff check .` / `uv run ruff check . --fix` / `uv run ruff format .`
- Mypy：`uv run mypy .`

---

## 5) 代码风格与约定（修改时请严格遵守）

总原则：最小改动、保持文件内风格一致，避免顺手重构/全量格式化。

### 5.1 Imports

- 三段式导入（段间空行）：标准库 / 第三方 / 本地模块
- 避免 `import *`
- 可执行入口尽量放在 `if __name__ == "__main__":` 下

### 5.2 格式化

- 遵循 PEP 8；行宽建议 88–100
- 保持单文件内引号风格一致

### 5.3 类型标注

- 对可复用函数与复杂逻辑加类型标注
- 3.12+ 可用 `list[str]` 等内建泛型；但若文件已使用 `typing.List/Dict`，优先保持一致

### 5.4 命名

- 函数/变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 文件名：`snake_case.py`
- 金融字段优先可读：`funding_rate`, `taker_buy_quote_volume` 等

### 5.5 错误处理与日志

- 网络请求必须有 timeout；重试要有界并带退避；已知不可恢复错误可减少重试（参考 `scripts/data/request_1h.py`）
- 不要静默吞异常；返回空结果时调用方必须可处理
- 抛异常优先具体类型：`ValueError` / `FileNotFoundError` / `RuntimeError`
- 脚本中可用 `print`；若修改 `scripts/data/dump_bin.py`，保持 `loguru.logger` 风格

### 5.6 路径与 I/O

- 推荐 `pathlib.Path`
- 输出目录要确保存在：`mkdir(parents=True, exist_ok=True)`
- 不要提交大文件/生成物（见 `.gitignore`）：`qlib_data/`、`mlruns/`、大 CSV、模型产物

---

## 6) QLib / 数据约定

- 默认 QLib 数据目录：`./qlib_data/my_crypto_data`
- 高频时间格式：`%Y-%m-%d %H:%M:%S`
- 研究配置真源：`./config.toml`
- `nusri_project/config/alpha261_config.py` 因子名必须唯一（重复会 `raise ValueError("duplicate factor name")`）
- 分类标签输出列使用 `pred_prob`；回归标签输出列使用 `pred_return`
- fused / score 主路径输出列使用 `pred_score`
- 当前最佳阶段信号定义：
  - 组件：`regression_24h` 与 `regression_72h`
  - 融合：`regression_fused_main`
  - 权重训练：`rolling_24m_halflife_6m`
- 分类交易层比较概率阈值：`enter_prob_threshold / exit_prob_threshold / full_prob_threshold`
- 回归交易层比较收益阈值：`entry_threshold / exit_threshold / full_position_threshold`
- score 交易层比较连续阈值：`open_score / close_score / size_floor_score / size_full_score / curve_gamma`
- 当前最佳阶段训练窗口为 `rolling_24m_halflife_6m`
- `top15/top10`、`18m/1y`、`0.004/0.005` 以及旧 `cost_aware_main` probability 主线仅作为已完成的对比实验结论保留在历史文档中，不再作为长期主线配置
- 涉及策略回测、执行器、交易成本、组合分析时，先检查 `Qlib` 官方现成能力是否已覆盖，例如 `qlib.backtest.backtest`、`qlib.contrib.evaluate.backtest_daily`、`qlib.workflow.record_temp.PortAnaRecord`
- 如果 `Qlib` 已有合适能力，优先通过配置、封装和对接现有接口实现；不要先手写一套平行回测框架
- 只有在 `Qlib` 现成接口无法准确表达当前需求时，才允许补充自定义实现；并在代码或文档中明确说明缺口

---

## 7) 协作守则（对代理很重要）

- 只做用户要求的改动；不要擅自新增依赖/工具链（pytest/ruff 等）
- 修改训练/数据脚本前，先确认路径与时间段配置不误伤现有数据
- 优先用 `uv run ...` 在锁定环境中执行命令
