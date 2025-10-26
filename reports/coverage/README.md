# Coverage Reports

This directory tracks Step-1 solver coverage on ARC training datasets, split by ARC-1 (original) and ARC-2 (2024 expansion).

## Quick Summary

### Step-2 (Current) — Phase 7

| Dataset | Tasks | Solved | Coverage | vs Step-1 |
|---------|-------|--------|----------|-----------|
| **ARC-1** | 391 | **212** | **54.2%** | +48.3% (9× boost) |
| **ARC-2** | 609 | **339** | **55.7%** | +53.9% (31× boost) |
| **Combined** | 1000 | **551** | **55.1%** | +51.7% (16× boost) |

### Step-1 Baseline — Phase 3

| Dataset | Tasks | Solved | Coverage | Status |
|---------|-------|--------|----------|--------|
| **ARC-1** | 391 | 23 | 5.9% | ✅ Baseline |
| **ARC-2** | 609 | 11 | 1.8% | ✅ Baseline |
| **Combined** | 1000 | 34 | 3.4% | ✅ Baseline |

**Last Updated**: 2025-10-26
**Solver Phase**: Phase 7 (Step-2: P + Φ/GLUE Compositional Solver)

## Reports

### Phase 7 (Step-2: P + Φ/GLUE)
- [**arc1_phase7_step2.md**](arc1_phase7_step2.md) - ARC-1 Step-2 coverage (54.2%, 212/391 solved)
- [**arc2_phase7_step2.md**](arc2_phase7_step2.md) - ARC-2 Step-2 coverage (55.7%, 339/609 solved)

### Phase 3 (Step-1: Global Solver Baseline)
- [**arc1_baseline.md**](arc1_baseline.md) - ARC-1 Step-1 coverage (5.9%, 23/391 solved)
- [**arc2_baseline.md**](arc2_baseline.md) - ARC-2 Step-1 coverage (1.8%, 11/609 solved)

## Key Insights (Phase 7 — Step-2)

### 1. Step-2 Compositional Power is MASSIVE

**Coverage Boost**:
```
ARC-1: 5.9% → 54.2% (9× improvement, +189 tasks)
ARC-2: 1.8% → 55.7% (31× improvement, +328 tasks)
Combined: 3.4% → 55.1% (16× improvement, +517 tasks)
```

**Status**: ✅ **Massively exceeds targets** (target: 20-30%, achieved: 55%)

### 2. Identity + Φ Dominates

**Identity as Global Transform**:
- ARC-1: 191/212 solved (90%)
- ARC-2: 328/339 solved (97%)
- **Combined: 519/551 solved (94%)**

**Interpretation**: Most ARC tasks don't need complex global transforms—they need smart **local pixel reasoning** via Φ features + class-based actions.

### 3. ARC-2 Responds Better to Compositional Reasoning

**Step-1**: ARC-2 is 3× harder (1.8% vs 5.9%)
**Step-2**: ARC-2 and ARC-1 are **equally solvable** (~55% both)

**Why**: ARC-2 was designed to avoid simple global patterns, but local Φ reasoning works equally well on both datasets.

### 2. Different Family Distributions

**ARC-1 top families**:
- Isometry: 7 tasks (30% of solved)
- ParityTile: 5 tasks (22% of solved)
- ColorMap: 2 tasks (9% of solved)

**ARC-2 top families**:
- BlockDown, BlockPermutation, NPSDown: 2 tasks each
- No dominant family (more spread out)
- Notably: 0 Isometry, 0 ColorMap hits

**Interpretation**:
- ARC-1 has more basic geometric patterns
- ARC-2 focuses on structural/block-based reasoning
- Step-1 limitations more evident on ARC-2

### 4. Coverage Progress vs Targets

| Phase | Solver | Target | ARC-1 Actual | ARC-2 Actual | Status |
|-------|--------|--------|--------------|--------------|--------|
| Phase 3 | Step-1 only | 5-10% | 5.9% ✅ | 1.8% ✅ | Baseline |
| **Phase 7** | **+ Step-2 (P+Φ/GLUE)** | **20-30%** | **54.2%** ✅✅ | **55.7%** ✅✅ | **Done** |
| Phase 9 | + Step-3 (LUT) | 65-75% | TBD | TBD | Next |

**Phase 7 massively exceeds expectations** (2× better than target!)

## How to Measure Coverage

### On Full Dataset
```bash
python tests/measure_coverage.py data/arc-agi_training_challenges.json
```

### On ARC-1 Only
```bash
python tests/measure_coverage.py data/arc1_training.json
```

### On ARC-2 Only
```bash
python tests/measure_coverage.py data/arc2_training.json
```

## Dataset Split Details

The training dataset is split using `scripts/split_arc_dataset.py`:

- **ARC-1**: Task IDs from original ARC-AGI-1 repository (2019)
  - Source: https://github.com/fchollet/ARC-AGI/tree/master/data/training
  - Expected: 400 tasks
  - Present in competition dataset: 391 tasks (9 tasks missing)

- **ARC-2**: All other task IDs (2024 expansion)
  - Source: ARC-AGI-2 new tasks
  - Expected: 600 tasks
  - Present in competition dataset: 609 tasks

### Generate Split Datasets
```bash
python scripts/split_arc_dataset.py
```

Outputs:
- `data/arc1_training.json` (391 tasks)
- `data/arc2_training.json` (609 tasks)

## Tracking Progress

As we implement Step-2 (Phase 7) and Step-3 (Phase 9), we should:

1. **Re-run coverage measurements** after each major phase
2. **Update baseline reports** with new coverage percentages
3. **Create timestamped snapshots** to track improvement over time
4. **Compare ARC-1 vs ARC-2 growth** to understand solver strengths

### Recommended Naming Convention

When creating new coverage reports:
```
arc1_phaseN_YYYY-MM-DD.md
arc2_phaseN_YYYY-MM-DD.md
```

Example:
```
arc1_phase7_2025-11-01.md  (after Step-2 implementation)
arc1_phase9_2025-12-01.md  (after Step-3 implementation)
```

## Next Milestones

| Phase | Goal | ARC-1 Target | ARC-1 Actual | ARC-2 Target | ARC-2 Actual |
|-------|------|--------------|--------------|--------------|--------------|
| ~~Phase 7~~ | ~~Step-2 (P+Φ/GLUE)~~ | ~~20-25%~~ | ✅ **54.2%** | ~~8-12%~~ | ✅ **55.7%** |
| **Phase 9** | **Step-3 (LUT)** | **65-75%** | **TBD** | **65-75%** | **TBD** |
| Optimization | Tune families + edge cases | 75-80% | TBD | 70-75% | TBD |

## Notes

### Missing ARC-1 Tasks

9 tasks from the original ARC-1 set are not present in the competition dataset:
- These may have been moved to evaluation/test sets
- Or removed for quality/difficulty balancing
- This is expected and does not affect our analysis

### Why Track Separately?

Splitting ARC-1 and ARC-2 allows us to:
1. **Understand solver characteristics**: Which types of tasks do we handle well?
2. **Set realistic expectations**: ARC-2 is inherently harder
3. **Prioritize development**: Focus on areas with biggest impact
4. **Benchmark progress**: Compare improvement rates on both datasets

### Coverage Tool

The coverage measurement tool is located at `tests/measure_coverage.py` (P3-04).

**Features**:
- ✅ Deterministic (sorted task IDs and families)
- ✅ Per-family breakdown
- ✅ Robust error handling (per-task try/catch)
- ✅ Fixed-format output for easy parsing
- ✅ Completeness invariant: total = pass + unsat

## Related Documentation

- [Phase 3 Implementation Plan](../../docs/phasewise_implementation_plan.md)
- [P3-04 Context Pack](../../docs/context-packs/P3-04.md) - Coverage meter spec
- [Fundamental Decisions](../../docs/anchors/fundamental_decisions.md) - Pure math approach
