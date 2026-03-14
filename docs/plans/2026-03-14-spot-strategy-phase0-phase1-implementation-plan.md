# Spot Strategy Phase 0/1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fixed research protocol and a Qlib-first fee-aware BTCUSDT spot `long/flat` backtest layer for saved model prediction artifacts.

**Architecture:** Keep the implementation script-driven and small. Prefer Qlib's official backtest, strategy, executor, and record abstractions. Only add thin local wrappers for project-specific configuration, signal adaptation, and summary output.

**Tech Stack:** Python 3.12, pandas, pathlib, json, unittest

---

### Task 1: Save the research protocol

**Files:**
- Create: `docs/plans/2026-03-14-spot-strategy-research-design.md`
- Create: `docs/plans/2026-03-14-spot-strategy-phase0-phase1-implementation-plan.md`

**Step 1: Write the design and implementation plan**

Capture the development/final split, metric protocol, strategy scope, and the minimum backtester requirements.

**Step 2: Review for scope discipline**

Confirm the documents keep phase 0/1 narrow and do not introduce label or factor work yet.

**Step 3: Commit**

```bash
git add docs/plans/2026-03-14-spot-strategy-research-design.md docs/plans/2026-03-14-spot-strategy-phase0-phase1-implementation-plan.md
git commit -m "docs: define spot strategy research protocol"
```

### Task 2: Add failing tests for the Qlib-first backtest wrapper

**Files:**
- Create: `tests/test_backtest_spot_strategy.py`

**Step 1: Write failing tests**

Cover:
- prediction artifact normalization and signal adaptation
- Qlib-facing configuration assembly
- project-level metric extraction and summary output
- only the custom logic that is not already covered by Qlib internals

**Step 2: Run tests to verify failure**

Run:

```bash
/Users/jared/src/nusri-project/NUSRI_project/.venv/bin/python -m unittest tests.test_backtest_spot_strategy -v
```

Expected: import failure because the new wrapper module does not exist yet.

### Task 3: Implement the Qlib-first backtest wrapper

**Files:**
- Create: `strategy_config.py`
- Create: `backtest_spot_strategy.py`

**Step 1: Write minimal implementation**

Implement:
- a dataclass for default strategy settings
- prediction frame loading/normalization helpers
- Qlib strategy/executor/backtest config assembly
- project-specific summary and export helpers
- a CLI that reads prediction files and writes artifacts to an output directory

**Step 2: Run tests to verify they pass**

Run:

```bash
/Users/jared/src/nusri-project/NUSRI_project/.venv/bin/python -m unittest tests.test_backtest_spot_strategy -v
```

Expected: all tests pass.

**Step 3: Run a smoke CLI check**

Run:

```bash
/Users/jared/src/nusri-project/NUSRI_project/.venv/bin/python backtest_spot_strategy.py --help
```

Expected: CLI help prints without error.

### Task 4: Verify and commit phase 1

**Files:**
- Verify: `strategy_config.py`
- Verify: `backtest_spot_strategy.py`
- Verify: `tests/test_backtest_spot_strategy.py`

**Step 1: Re-run verification**

Run:

```bash
/Users/jared/src/nusri-project/NUSRI_project/.venv/bin/python -m unittest tests.test_backtest_spot_strategy -v
/Users/jared/src/nusri-project/NUSRI_project/.venv/bin/python backtest_spot_strategy.py --help
```

**Step 2: Commit**

```bash
git add strategy_config.py backtest_spot_strategy.py tests/test_backtest_spot_strategy.py
git commit -m "feat: add minimal spot backtest layer"
```
