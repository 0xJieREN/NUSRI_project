# AGENTS.md — NUSRI_project（智能编码代理指南）

本仓库是 Python 3.12 的加密货币价格预测项目：
- QLib：数据处理 / 特征工程 / 数据集抽象
- LightGBM：通过 QLib `LGBModel` 训练
- 以脚本驱动为主（顶层脚本较多，非标准包结构）

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

- `uv run python request_1h.py`

输出示例：`BTCUSDT_1h_binance_data.csv`

要点：
- 同时抓取期货衍生数据（如 `funding_rate`）并写入 CSV
- 网络请求包含 timeout + 有界重试；某些 400 会减少重试

### 2.2 原始 CSV → QLib source CSV

- `uv run python clean_data.py --input BTCUSDT_1h_binance_data.csv --output qlib_source_data/BTCUSDT.csv`

要点：
- 默认分隔符 `;`（可 `--sep` 修改）
- 补齐 `symbol` 列
- `date` 规范到秒级 `%Y-%m-%d %H:%M:%S`

### 2.3 QLib source CSV → QLib binary 数据

- `uv run python dump_bin.py dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min`

要点：
- `dump_bin.py` 使用 `fire` CLI
- 输出目录包含 `calendars/ features/ instruments/ ...`

### 2.4 训练 LightGBM（主流程）

- `uv run python LGBM_workflow.py`

要点：
- `provider_uri = ./qlib_data/my_crypto_data`
- 训练/验证/测试区间通过 `segments` 配置

### 2.5 导出特征重要性

- `uv run python dump_lgbm_feature_importance.py --importance-type gain --out lgbm_feature_importance.csv --top 20`

---

## 3) 测试（当前与推荐）

### 3.1 当前仓库状态

- 未配置 pytest（未发现 `pytest.ini` / `conftest.py` / `pyproject` pytest 配置）
- `.gitignore` 忽略 `test/`，但其中有 `test/test_qlib.py`（更像 smoke test）

### 3.2 现有 smoke test（QLib 读数据）

- `uv run python test/test_qlib.py`

用途：验证 `qlib_data/my_crypto_data` 是否能被 QLib 正常加载。

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

- 网络请求必须有 timeout；重试要有界并带退避；已知不可恢复错误可减少重试（参考 `request_1h.py`）
- 不要静默吞异常；返回空结果时调用方必须可处理
- 抛异常优先具体类型：`ValueError` / `FileNotFoundError` / `RuntimeError`
- 脚本中可用 `print`；若修改 `dump_bin.py`，保持 `loguru.logger` 风格

### 5.6 路径与 I/O

- 推荐 `pathlib.Path`
- 输出目录要确保存在：`mkdir(parents=True, exist_ok=True)`
- 不要提交大文件/生成物（见 `.gitignore`）：`qlib_data/`、`mlruns/`、大 CSV、模型产物

---

## 6) QLib / 数据约定

- 默认 QLib 数据目录：`./qlib_data/my_crypto_data`
- 高频时间格式：`%Y-%m-%d %H:%M:%S`
- `alpha261_config.py` 因子名必须唯一（重复会 `raise ValueError("duplicate factor name")`）
- 涉及策略回测、执行器、交易成本、组合分析时，先检查 `Qlib` 官方现成能力是否已覆盖，例如 `qlib.backtest.backtest`、`qlib.contrib.evaluate.backtest_daily`、`qlib.workflow.record_temp.PortAnaRecord`
- 如果 `Qlib` 已有合适能力，优先通过配置、封装和对接现有接口实现；不要先手写一套平行回测框架
- 只有在 `Qlib` 现成接口无法准确表达当前需求时，才允许补充自定义实现；并在代码或文档中明确说明缺口

---

## 7) 协作守则（对代理很重要）

- 只做用户要求的改动；不要擅自新增依赖/工具链（pytest/ruff 等）
- 修改训练/数据脚本前，先确认路径与时间段配置不误伤现有数据
- 优先用 `uv run ...` 在锁定环境中执行命令
