# Label Optimization Round 1 Implementation Plan

> **Status Update (2026-04-01):** This plan is kept as historical implementation context. The current recommended stage is `regression_fused_aggressive_v3_best`; see `docs/research/2026-04-01-regression-fused-best-stage-summary.md`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable `24h / 48h / 72h` regression labels and evaluate them under fixed spot trading shells.

**Architecture:** Reuse the current QLib-first training and strategy stack. Make the label horizon configurable in the training workflow, keep prediction artifact shape stable, and add a focused comparison runner that evaluates a small set of label horizons against fixed trading shells.

**Tech Stack:** Python 3.12, QLib, pandas, argparse, unittest

---

### Task 1: Save the label-optimization design and plan

**Files:**
- Create: `docs/plans/2026-03-15-label-optimization-round1-design.md`
- Create: `docs/plans/2026-03-15-label-optimization-round1-implementation-plan.md`

**Step 1: Write the design and implementation plan**

Capture the fixed boundaries, label candidates, trading shells, success criteria, and verification path.

**Step 2: Commit**

```bash
git add docs/plans/2026-03-15-label-optimization-round1-design.md docs/plans/2026-03-15-label-optimization-round1-implementation-plan.md
git commit -m "docs: define label optimization round1"
```

### Task 2: Add failing tests for label horizon configuration

**Files:**
- Modify: `tests/test_backtest_spot_strategy.py`
- Create: `tests/test_label_optimization_round1.py`

**Step 1: Write failing tests**

Cover:
- building the correct QLib label expression from a horizon in hours
- preserving experiment naming / metadata for different horizons
- generating the intended comparison matrix for `24h / 48h / 72h`

**Step 2: Run tests to verify failure**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: import failure or missing function failure because the new label helper does not exist yet.

### Task 3: Implement configurable label horizons in training

**Files:**
- Modify: `nusri_project/training/lgbm_workflow.py`
- Modify: `scripts/training/lgbm_workflow.py`

**Step 1: Write minimal implementation**

Implement:
- a helper that maps horizon hours to the correct label expression
- CLI support for `--label-horizon-hours`
- a way to build workflow config for `24 / 48 / 72`

**Step 2: Run tests to verify they pass**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: all tests pass.

### Task 4: Add the round 1 comparison runner

**Files:**
- Create: `nusri_project/strategy/label_optimization_round1.py`
- Create: `scripts/analysis/run_label_optimization_round1.py`
- Test: `tests/test_label_optimization_round1.py`

**Step 1: Write failing tests for the comparison runner**

Cover:
- generating the `24h / 48h / 72h` comparison matrix
- attaching fixed trading shells
- ranking results in a compact table

**Step 2: Run test to verify failure**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: failure due to missing comparison helpers.

**Step 3: Write minimal implementation**

Implement:
- horizon definitions
- fixed trading-shell definitions
- summary table generation for label comparison

**Step 4: Re-run tests**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: all tests pass.

### Task 5: Verify entrypoints and commit round 1 foundation

**Files:**
- Verify: `nusri_project/training/lgbm_workflow.py`
- Verify: `nusri_project/strategy/label_optimization_round1.py`
- Verify: `scripts/analysis/run_label_optimization_round1.py`

**Step 1: Run smoke checks**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.training.lgbm_workflow --help
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.analysis.run_label_optimization_round1 --help
```

**Step 2: Run the relevant tests**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 tests.test_backtest_spot_strategy -v
```

**Step 3: Commit**

```bash
git add nusri_project/training/lgbm_workflow.py scripts/training/lgbm_workflow.py nusri_project/strategy/label_optimization_round1.py scripts/analysis/run_label_optimization_round1.py tests/test_label_optimization_round1.py
git commit -m "feat: add label optimization round1"
```
