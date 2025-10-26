# ARC-1 Coverage Report — Phase 7 Step-2

**Date**: 2025-10-26
**Solver**: Step-2 (P + Φ/GLUE Compositional Solver)
**Dataset**: ARC-1 Training (391 tasks)

---

## Summary

| Metric | Step-1 | Step-2 | Delta |
|--------|--------|--------|-------|
| **Solved** | 23 | **212** | +189 |
| **Coverage** | 5.9% | **54.2%** | +48.3% |
| **UNSAT** | 368 | 179 | -189 |

**Coverage Boost**: **9× improvement** (5.9% → 54.2%)

---

## Step-2 Breakdown by P

| P (Global Transform) | Tasks Solved | % of Total |
|---------------------|--------------|------------|
| **Identity** | 191 | 48.8% |
| ParityTile | 5 | 1.3% |
| Isometry | 5 | 1.3% |
| BlockPermutation | 2 | 0.5% |
| ColorMap | 2 | 0.5% |
| MirrorComplete | 2 | 0.5% |
| NPSDown | 2 | 0.5% |
| PixelReplicate | 2 | 0.5% |
| BlockDown | 1 | 0.3% |
| **UNSAT** | **179** | **45.8%** |

---

## Key Insights

### 1. Identity + Φ Dominates
- **191/212 tasks (90%)** solved with Identity as global transform
- This means most tasks don't need complex global transformations
- Local Φ reasoning (pixel features + class-based actions) is sufficient

### 2. Massive Improvement Over Step-1
- Step-1: Only 23 tasks (5.9%)
- Step-2: 212 tasks (54.2%)
- **9× coverage boost** from compositional reasoning

### 3. Remaining UNSAT (179 tasks, 45.8%)
Likely require:
- Step-3 (LUT) for lookup-table patterns
- Additional families not yet implemented
- Novel approaches beyond current methods

---

## Comparison with Step-1

**Step-1 by Family**:
- Isometry: 7 tasks
- ParityTile: 5 tasks
- BlockPermutation: 2 tasks
- ColorMap: 2 tasks
- MirrorComplete: 2 tasks
- NPSDown: 2 tasks
- PixelReplicate: 2 tasks
- BlockDown: 1 task
- UNSAT: 368 tasks

**Step-2 P Distribution**:
- Most families maintain or improve coverage
- Identity emerges as dominant pattern (191 new tasks)
- UNSAT reduced from 368 → 179 (51% reduction)

---

## Next Steps

**Target for Step-3 (LUT)**:
- Expected coverage: 65-75% (additional 10-20% improvement)
- Focus: Lookup-table based patterns
- Remaining UNSAT: ~25-35%

**Status**: ✅ **Step-2 exceeds targets** (target: 20-25%, achieved: 54.2%)
