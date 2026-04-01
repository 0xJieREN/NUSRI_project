# Spot Strategy Phase 2 Implementation Plan

> **Status Update (2026-04-01):** This plan is kept as historical implementation context. The current recommended stage is `regression_fused_aggressive_v3_best`; see `docs/research/2026-04-01-regression-fused-best-stage-summary.md`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the 2024 baseline strategy evaluation and parameter scan flow on top of the Qlib-first spot backtest layer.

**Architecture:** Reuse the Qlib-first backtest wrapper from phase 1. Add one small research module for prediction file discovery, parameter-grid construction, constraint-aware ranking, and result export. Add one CLI entry that runs a baseline backtest and an optional parameter scan over 2024 prediction artifacts.

**Tech Stack:** Python 3.12, pandas, pathlib, itertools, json, csv, unittest

---

### Task 1: Save the phase 2 plan

**Files:**
- Create: `docs/plans/2026-03-14-spot-strategy-phase2-implementation-plan.md`

**Step 1: Write the implementation plan**

Document the baseline run flow, parameter scan flow, expected artifacts, and verification commands.

**Step 2: Commit**

```bash
git add docs/plans/2026-03-14-spot-strategy-phase2-implementation-plan.md
git commit -m "docs: add phase2 strategy research plan"
```

### Task 2: Add failing tests for phase 2 research helpers

**Files:**
- Create: `tests/test_phase2_strategy_research.py`

**Step 1: Write failing tests**

Cover:
- discovering and sorting `pred_YYYYMM.pkl` files for a target year
- building only valid parameter combinations
- ranking scan results by feasibility first, then by risk-adjusted metrics

**Step 2: Run test to verify failure**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_phase2_strategy_research -v
```

Expected: import failure because the new phase 2 helper module does not exist yet.

### Task 3: Implement phase 2 research helpers

**Files:**
- Create: `nusri_project/strategy/phase2_strategy_research.py`
- Create: `scripts/analysis/run_phase2_baseline.py`

**Step 1: Write minimal implementation**

Implement:
- prediction artifact discovery for one year
- parameter-grid generation with invalid combinations removed
- constraint-aware ranking for strategy scan results
- a baseline runner that exports one baseline report and one ranked scan table

**Step 2: Run tests to verify they pass**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_phase2_strategy_research -v
```

Expected: all tests pass.

### Task 4: Run a real 2024 baseline and small scan

**Files:**
- Verify: `scripts/analysis/run_phase2_baseline.py`

**Step 1: Run the real baseline**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.analysis.run_phase2_baseline --mlruns-root /Users/jared/src/nusri-project/mlruns --provider-uri /Users/jared/src/nusri-project/qlib_data/my_crypto_data --year 2024 --output-dir reports/phase2_2024
```

Expected: baseline artifacts and scan CSV files are generated.

### Task 5: Verify and commit phase 2 foundation

**Files:**
- Verify: `nusri_project/strategy/phase2_strategy_research.py`
- Verify: `scripts/analysis/run_phase2_baseline.py`
- Verify: `tests/test_phase2_strategy_research.py`

**Step 1: Re-run verification**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_phase2_strategy_research -v
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.analysis.run_phase2_baseline --help
```

**Step 2: Commit**

```bash
git add nusri_project/strategy/phase2_strategy_research.py scripts/analysis/run_phase2_baseline.py tests/test_phase2_strategy_research.py
git commit -m "feat: add phase2 strategy baseline scan"
```
