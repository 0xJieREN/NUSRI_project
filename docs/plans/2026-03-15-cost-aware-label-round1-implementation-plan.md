# Cost-Aware Label Round 1 Implementation Plan

> **Status Update (2026-04-01):** This plan is kept as historical implementation context. The current recommended stage is `regression_fused_aggressive_v3_best`; see `docs/research/2026-04-01-regression-fused-best-stage-summary.md`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `72h` cost-aware binary label mode and compare it against the current `72h` regression label under fixed trading shells.

**Architecture:** Extend the training workflow to support explicit label modes instead of only horizon-based regression labels. Reuse `qlib.contrib.model.gbdt.LGBModel` in both `mse` and `binary` modes, keep monthly prediction artifact output compatible with the current backtest pipeline, and add a comparison runner that evaluates regression vs classification under the same shells.

**Tech Stack:** Python 3.12, QLib, LightGBM, pandas, argparse, unittest

---

### Task 1: Save the design and plan

**Files:**
- Create: `docs/plans/2026-03-15-cost-aware-label-round1-design.md`
- Create: `docs/plans/2026-03-15-cost-aware-label-round1-implementation-plan.md`

**Step 1: Write the design and implementation plan**

Capture the label definition, scope, comparison matrix, and success criteria.

**Step 2: Commit**

```bash
git add docs/plans/2026-03-15-cost-aware-label-round1-design.md docs/plans/2026-03-15-cost-aware-label-round1-implementation-plan.md
git commit -m "docs: define cost-aware label round1"
```

### Task 2: Add failing tests for label modes

**Files:**
- Modify: `tests/test_label_optimization_round1.py`

**Step 1: Write failing tests**

Cover:
- building a `72h` cost-aware binary expression with threshold `0.005`
- selecting `mse` vs `binary` model loss from label mode
- encoding label mode in prediction artifact naming

**Step 2: Run tests to verify failure**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: failure caused by missing label-mode helpers.

### Task 3: Implement label-mode support in training

**Files:**
- Modify: `nusri_project/training/lgbm_workflow.py`
- Modify: `scripts/training/lgbm_workflow.py`

**Step 1: Write minimal implementation**

Implement:
- label-mode constants
- cost-aware binary label expression helper
- model-loss selection helper
- CLI support for label mode
- prediction artifact naming that includes label mode

**Step 2: Re-run tests**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: all tests pass.

### Task 4: Add the comparison runner

**Files:**
- Create: `nusri_project/strategy/cost_aware_label_round1.py`
- Create: `scripts/analysis/run_cost_aware_label_round1.py`
- Modify: `tests/test_label_optimization_round1.py`

**Step 1: Write failing tests**

Cover:
- fixed comparison matrix with regression/classification × balanced/conservative
- horizon-specific prediction discovery for both label modes
- compact result-table generation

**Step 2: Run tests to verify failure**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: failures for missing comparison helpers.

**Step 3: Write minimal implementation**

Implement the comparison matrix and evaluation runner using existing backtest helpers.

**Step 4: Re-run tests**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 -v
```

Expected: all tests pass.

### Task 5: Verify entrypoints and commit the round 1 implementation

**Files:**
- Verify: `nusri_project/training/lgbm_workflow.py`
- Verify: `nusri_project/strategy/cost_aware_label_round1.py`
- Verify: `scripts/analysis/run_cost_aware_label_round1.py`

**Step 1: Run smoke checks**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.training.lgbm_workflow --help
/Users/jared/src/nusri-project/.venv/bin/python -m scripts.analysis.run_cost_aware_label_round1 --help
```

**Step 2: Run tests**

Run:

```bash
/Users/jared/src/nusri-project/.venv/bin/python -m unittest tests.test_label_optimization_round1 tests.test_backtest_spot_strategy -v
```

**Step 3: Commit**

```bash
git add nusri_project/training/lgbm_workflow.py scripts/training/lgbm_workflow.py nusri_project/strategy/cost_aware_label_round1.py scripts/analysis/run_cost_aware_label_round1.py tests/test_label_optimization_round1.py
git commit -m "feat: add cost-aware label round1"
```
