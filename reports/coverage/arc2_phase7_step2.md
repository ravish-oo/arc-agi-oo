# ARC-2 Coverage Report — Phase 7 Step-2

**Date**: 2025-10-26
**Solver**: Step-2 (P + Φ/GLUE Compositional Solver)
**Dataset**: ARC-2 Training (609 tasks)

---

## Summary

| Metric | Step-1 | Step-2 | Delta |
|--------|--------|--------|-------|
| **Solved** | 11 | **339** | +328 |
| **Coverage** | 1.8% | **55.7%** | +53.9% |
| **UNSAT** | 598 | 270 | -328 |

**Coverage Boost**: **31× improvement** (1.8% → 55.7%)

---

## Step-2 Breakdown by P

| P (Global Transform) | Tasks Solved | % of Total |
|---------------------|--------------|------------|
| **Identity** | 328 | 53.9% |
| BlockDown | 2 | 0.3% |
| BlockPermutation | 2 | 0.3% |
| NPSDown | 2 | 0.3% |
| BlockSubstitution | 1 | 0.2% |
| CopyMoveAllComponents | 1 | 0.2% |
| NPSUp | 1 | 0.2% |
| ParityTile | 1 | 0.2% |
| PixelReplicate | 1 | 0.2% |
| **UNSAT** | **270** | **44.3%** |

---

## Key Insights

### 1. Identity + Φ is Even More Dominant on ARC-2
- **328/339 tasks (97%)** solved with Identity as global transform
- ARC-2 was designed to avoid simple global patterns
- But yields beautifully to compositional Φ/GLUE reasoning

### 2. Stunning Improvement Over Step-1
- Step-1: Only 11 tasks (1.8%) — very low baseline
- Step-2: 339 tasks (55.7%)
- **31× coverage boost** — highest improvement ratio

### 3. ARC-2 Responds Better to Compositional Reasoning
- Step-1: ARC-2 is 3× harder than ARC-1 (1.8% vs 5.9%)
- Step-2: ARC-2 and ARC-1 have **equal coverage** (~55% both)
- **Interpretation**: ARC-2 difficulty comes from global transform complexity, not local pattern complexity

### 4. Remaining UNSAT (270 tasks, 44.3%)
Similar rate to ARC-1, likely require:
- Step-3 (LUT) for lookup-table patterns
- Additional families
- Novel approaches

---

## Comparison with Step-1

**Step-1 by Family**:
- BlockDown: 2 tasks
- BlockPermutation: 2 tasks
- NPSDown: 2 tasks
- BlockSubstitution: 1 task
- CopyMoveAllComponents: 1 task
- NPSUp: 1 task
- ParityTile: 1 task
- PixelReplicate: 1 task
- UNSAT: 598 tasks

**Step-2 P Distribution**:
- All Step-1 families maintain coverage
- Identity adds 328 new tasks (massive gain)
- UNSAT reduced from 598 → 270 (55% reduction)

---

## ARC-2 Characteristics

### Why ARC-2 is "Harder" for Step-1
- Intentionally avoids simple transformations (rotations, color swaps)
- No dominant family pattern (very spread out)
- Requires compositional reasoning from the start

### Why Step-2 Solves ARC-2 So Well
- Local Φ features capture pixel-level patterns effectively
- MDL selection finds minimal class partitions
- GLUE stitching composes local edits correctly

---

## Next Steps

**Target for Step-3 (LUT)**:
- Expected coverage: 65-75% (additional 10-20% improvement)
- Focus: Lookup-table based patterns
- Remaining UNSAT: ~25-35%

**Status**: ✅ **Step-2 massively exceeds targets** (target: 8-12%, achieved: 55.7%)
