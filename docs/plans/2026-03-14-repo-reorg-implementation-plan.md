# Repository Reorganization Implementation Plan

> **Status Update (2026-04-01):** This plan is kept as historical implementation context. The current recommended stage is `regression_fused_aggressive_v3_best`; see `docs/research/2026-04-01-regression-fused-best-stage-summary.md`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the repository into clearer functional areas, archive legacy artifacts, and update documentation so the new command surface is consistent.

**Architecture:** Keep the repository script-driven, but separate runnable entrypoints from reusable modules. Place data/training/analysis entrypoints under `scripts/`, reusable logic under `nusri_project/`, archive legacy files under `archive/`, and keep repository-level docs and config at the root.

**Tech Stack:** Python 3.12, pathlib, unittest, Git file moves

---

### Task 1: Create the target structure and archive legacy files

**Files:**
- Create: `scripts/`
- Create: `nusri_project/`
- Create: `archive/`
- Move: current root scripts and reusable modules
- Archive: legacy generated CSVs and old `test/` directory

**Step 1: Create directories**

Create:
- `scripts/data`
- `scripts/training`
- `scripts/analysis`
- `nusri_project/config`
- `nusri_project/training`
- `nusri_project/strategy`
- `archive/artifacts`
- `archive/legacy_test`

**Step 2: Move active files**

Move the active code into the new script/module split.

**Step 3: Archive agreed legacy files**

Archive:
- `BTCUSDT_1h_binance_data.csv`
- `lgbm_feature_importance.csv`
- `lgbm_feature_importance_split.csv`
- old `test/` directory

### Task 2: Update imports and runnable entrypoints

**Files:**
- Modify: moved training/strategy modules
- Create: thin `python -m ...` wrappers in `scripts/`
- Test: `tests/test_backtest_spot_strategy.py`
- Test: `tests/test_phase2_strategy_research.py`

**Step 1: Write failing tests or update regression expectations**

Adjust tests so they import the new package paths and verify the new Qlib module path strings.

**Step 2: Run tests to verify failure**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_backtest_spot_strategy tests.test_phase2_strategy_research -v
```

Expected: failures caused by old imports/module paths.

**Step 3: Implement the new import structure**

Add package `__init__.py` files, move code, and make wrappers call package `main()` functions.

**Step 4: Run tests to verify they pass**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_backtest_spot_strategy tests.test_phase2_strategy_research -v
```

Expected: all tests pass.

### Task 3: Update repository documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`

**Step 1: Replace old root-level commands**

Update commands to the new `python -m scripts...` entrypoints.

**Step 2: Reflect the new structure**

Document:
- active code locations
- archive locations
- updated workflow

### Task 4: Verify the reorganized repository

**Files:**
- Verify: all moved scripts and docs

**Step 1: Run command smoke checks**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.analysis.backtest_spot_strategy --help
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.analysis.run_phase2_baseline --help
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.training.lgbm_workflow --help
```

**Step 2: Confirm no stray temp artifacts remain**

Check for:
- `__pycache__`
- legacy `reports/`
- obsolete root files that should now live under `scripts/`, `nusri_project/`, or `archive/`

### Task 5: Commit the reorganization

**Files:**
- Commit all structural and documentation changes

**Step 1: Commit**

```bash
git add .
git commit -m "refactor: reorganize repository structure"
```
