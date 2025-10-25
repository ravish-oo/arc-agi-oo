# ARC-1 Training Coverage Baseline

**Date**: 2025-10-25
**Solver**: Step-1 Solver (Global families only)
**Dataset**: ARC-1 Training Tasks (Original 400, 391 present in competition dataset)

## Overall Coverage

```
========================================
Step 1 Coverage: 23/391 (5.9%)
========================================
```

**Summary**:
- **Total tasks**: 391
- **Solved (PASS)**: 23 (5.9%)
- **Unsolved (UNSAT)**: 368 (94.1%)

## Breakdown by Family

| Family | Solved Tasks | Percentage |
|--------|--------------|------------|
| Isometry | 7 | 1.8% |
| ParityTile | 5 | 1.3% |
| ColorMap | 2 | 0.5% |
| MirrorComplete | 2 | 0.5% |
| NPSDown | 2 | 0.5% |
| PixelReplicate | 2 | 0.5% |
| BlockPermutation | 2 | 0.5% |
| BlockDown | 1 | 0.3% |

## Analysis

### Key Observations

1. **Isometry leads coverage**: 7 tasks solved (30% of solved tasks)
   - Identity, rotations, flips, and reflections
   - Most fundamental transformation family

2. **ParityTile second**: 5 tasks solved (22% of solved tasks)
   - Checkerboard and tiling patterns
   - Shows some geometric regularity

3. **Low overall coverage expected**: Step-1 is intentionally simple
   - Only global exact transformations
   - No compositional mode (Step 2)
   - No local fallback (Step 3)

### Family Coverage Distribution

```
Isometry:        ███████ 7
ParityTile:      █████ 5
ColorMap:        ██ 2
MirrorComplete:  ██ 2
NPSDown:         ██ 2
PixelReplicate:  ██ 2
BlockPermutation:██ 2
BlockDown:       █ 1
```

## Expectations

The 5.9% coverage on ARC-1 is **within expected range** for Step-1 solver:

- **Target baseline**: 25-30% after all 3 steps
- **Step-1 only**: ~5-10% (global exact only)
- **Step-2 addition**: +15-20% (compositional P+Φ/GLUE)
- **Step-3 addition**: +5-10% (LUT local fallback)

ARC-1 tasks are generally more amenable to simple transformations than ARC-2.

## Next Steps

1. **Phase 7**: Implement Step-2 solver (P+Φ/GLUE compositional mode)
   - Expected to add ~60-80 more solved tasks
   - Target: 25-30% total coverage

2. **Phase 9**: Implement Step-3 solver (LUT local fallback)
   - Expected to add ~20-40 more solved tasks
   - Target: 30-40% total coverage

3. **Optimization**: Per-family tuning and edge case handling
   - Additional 5-10% improvement possible

## Reproduce

```bash
# Run coverage measurement
python tests/measure_coverage.py data/arc1_training.json
```

## Dataset Details

- **Source**: Original ARC-AGI-1 (2019)
- **File**: `data/arc1_training.json`
- **Tasks**: 391 (9 tasks from original 400 not present in competition dataset)
- **Task IDs**: Fetched from https://github.com/fchollet/ARC-AGI/tree/master/data/training
