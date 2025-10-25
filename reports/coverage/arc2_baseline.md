# ARC-2 Training Coverage Baseline

**Date**: 2025-10-25
**Solver**: Step-1 Solver (Global families only)
**Dataset**: ARC-2 Training Tasks (New 600 tasks added in 2024)

## Overall Coverage

```
========================================
Step 1 Coverage: 11/609 (1.8%)
========================================
```

**Summary**:
- **Total tasks**: 609
- **Solved (PASS)**: 11 (1.8%)
- **Unsolved (UNSAT)**: 598 (98.2%)

## Breakdown by Family

| Family | Solved Tasks | Percentage |
|--------|--------------|------------|
| BlockDown | 2 | 0.3% |
| BlockPermutation | 2 | 0.3% |
| NPSDown | 2 | 0.3% |
| BlockSubstitution | 1 | 0.2% |
| CopyMoveAllComponents | 1 | 0.2% |
| NPSUp | 1 | 0.2% |
| ParityTile | 1 | 0.2% |
| PixelReplicate | 1 | 0.2% |

## Analysis

### Key Observations

1. **Much lower coverage than ARC-1**: 1.8% vs 5.9%
   - ARC-2 intentionally designed to be harder
   - Fewer tasks solvable by simple global transformations

2. **No Isometry or ColorMap hits**:
   - ARC-1: Isometry solved 7 tasks, ColorMap solved 2
   - ARC-2: These families solved 0 tasks
   - Suggests ARC-2 avoids simple identity/rotation/color-swap patterns

3. **Different family distribution**:
   - More structural/block-based families (BlockDown, BlockPermutation, NPSDown)
   - Less basic geometric transformations
   - Indicates more complex spatial reasoning required

4. **Broader family spread**:
   - 8 different families vs ARC-1's 8
   - But each family solves only 1-2 tasks
   - No dominant pattern family

### Family Coverage Distribution

```
BlockDown:             ██ 2
BlockPermutation:      ██ 2
NPSDown:               ██ 2
BlockSubstitution:     █ 1
CopyMoveAllComponents: █ 1
NPSUp:                 █ 1
ParityTile:            █ 1
PixelReplicate:        █ 1
```

## Comparison with ARC-1

| Metric | ARC-1 | ARC-2 | Ratio |
|--------|-------|-------|-------|
| Total tasks | 391 | 609 | 1.6× |
| Solved tasks | 23 | 11 | 0.5× |
| Coverage % | 5.9% | 1.8% | 0.3× |
| Top family hits | 7 (Isometry) | 2 (3-way tie) | 0.3× |

**Key insight**: ARC-2 is **~3× harder** for Step-1 solver than ARC-1.

## Expectations

The 1.8% coverage on ARC-2 is **lower than expected** but reasonable:

- **Original target**: 25-30% after all 3 steps on ARC-1 (400 tasks)
- **ARC-2 adjustment**: Expected to be significantly harder
- **Step-1 only**: 1.8% suggests Step-2 and Step-3 are critical

### Projected Coverage Growth

| Phase | ARC-1 Target | ARC-2 Target | Rationale |
|-------|--------------|--------------|-----------|
| Step-1 (current) | 5.9% | 1.8% | Global exact only |
| + Step-2 (P+Φ/GLUE) | 20-25% | 8-12% | Compositional mode |
| + Step-3 (LUT) | 30-40% | 15-20% | Local fallback |

ARC-2 will likely require more sophisticated composition and local reasoning.

## Next Steps

1. **Phase 7**: Implement Step-2 solver (P+Φ/GLUE compositional mode)
   - Expected to add ~40-60 more solved tasks on ARC-2
   - Critical for ARC-2 as global-only is insufficient

2. **Phase 9**: Implement Step-3 solver (LUT local fallback)
   - Expected to add ~30-50 more solved tasks on ARC-2
   - May be more important for ARC-2 than ARC-1

3. **ARC-2 specific optimization**:
   - Study UNSAT tasks to identify patterns
   - May need additional Φ functions beyond Phase 4
   - Consider expanding P family set if justified

## Reproduce

```bash
# Run coverage measurement
python tests/measure_coverage.py data/arc2_training.json
```

## Dataset Details

- **Source**: ARC-AGI-2 (2024 expansion)
- **File**: `data/arc2_training.json`
- **Tasks**: 609 (additional tasks beyond original 400)
- **Task IDs**: All IDs not present in original ARC-AGI-1 training set
- **Difficulty**: Intentionally harder than ARC-1, requires more advanced reasoning
