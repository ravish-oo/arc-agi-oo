# ARC AGI Solver — Repository Architecture

**Purpose:** Record folder structure, module organization, and what's implemented.
**Scope:** WHERE things are and WHAT they do (not HOW they work).
**Last Updated:** 2025-10-24

---

## Repository Structure

```
arc-agi-oo/
├── data/                          # ARC AGI dataset (Kaggle)
│   ├── training/                  # 400 training tasks
│   ├── evaluation/                # 400 evaluation tasks
│   └── test/                      # Hidden test set
│
├── docs/                          # Documentation
│   ├── anchors/                   # Source-of-truth specifications
│   │   ├── primary-anchor.md      # Mathematical foundation (Π/FY/GLUE)
│   │   ├── spec.md                # Locked algebra (16 families, Φ, A)
│   │   ├── implementation_plan.md # Pseudocode guide
│   │   └── fundamental_decisions.md # Resolved ambiguities
│   │
│   ├── context-packs/             # Phase-specific implementation briefs
│   │   ├── P1-01.md              # Core Grid Utilities
│   │   ├── P1-02.md              # D8 Isometries Registry
│   │   ├── P1-03.md              # Π Canonical Grid
│   │   ├── P1-04.md              # OFA Patch Recolor + D8 Patch Canonical
│   │   ├── P1-05.md              # 8-Connected Components by Color
│   │   └── P1-06.md              # NPS Boundary Bands
│   │
│   ├── architecture.md            # This file (repository structure map)
│   ├── context_index.md           # Navigation map (what/where, not how)
│   ├── phasewise_implementation_plan.md # 12-phase roadmap with checkboxes
│   ├── kaggle_submission_readiness.md # Submission requirements
│   └── arc-agi-kaggle-docs.md     # Competition rules
│
├── src/                           # Implementation modules
│   ├── utils.py                   # [P1-01] Core grid utilities
│   ├── canonicalization.py        # [P1-02,03,04] D8 isometries, Π, OFA
│   └── components.py              # [P1-05,06] Components, bbox, NPS bands
│
├── tests/                         # Test suites
│   ├── test_utils.py              # [P1-01] 45 tests - grid primitives
│   ├── test_isometries.py         # [P1-02] 38 tests - D8 group properties
│   ├── test_canonicalization.py   # [P1-03] 27 tests - Π idempotence
│   ├── test_canonical_patch.py    # [P1-04] 33 tests - OFA locality
│   ├── test_components.py         # [P1-05] 33 tests - 8-connectivity
│   └── test_nps_boundaries.py     # [P1-06] 42 tests - NPS partitioning
│
├── reviews/                       # Mathematical correctness reviews
│   ├── math_invariants_review_P1-01.md
│   ├── math_invariants_review_P1-02.md
│   ├── math_invariants_review_P1-03.md
│   ├── math_invariants_review_P1-04.md
│   ├── math_invariants_review_P1-05.md
│   └── math_invariants_review_P1-06.md
│
└── README.md                      # Project overview
```

---

## Implemented Modules (Phase 1 Foundation)

### src/utils.py (268 lines) — P1-01
**Purpose:** Core grid utilities and D8 primitive transformations
**Public API:**
- `dims(g)` → (rows, cols) with rectangularity validation
- `copy_grid(g)` → Deep copy without row aliasing
- `deep_eq(a, b)` → Bit-for-bit equality check
- `transpose(g)` → Main diagonal reflection (H×W → W×H)
- `rot90/rot180/rot270(g)` → Clockwise rotations
- `flip_h(g)` → Horizontal flip (left↔right)
- `flip_v(g)` → Vertical flip (top↔bottom)

**Invariants:**
- Pure functions (no mutations)
- Deterministic (same input → same output)
- D8 group identities verified (rot90⁴=id, transpose²=id, etc.)

**Tests:** 45 tests in `tests/test_utils.py` (100% pass)

---

### src/canonicalization.py (395 lines) — P1-02, P1-03, P1-04
**Purpose:** D8 isometries, grid canonicalization (Π), OFA patch normalization

#### P1-02: D8 Isometries Registry (lines 1-146)
**Public API:**
- `ISOMETRIES` → Fixed list of 8 isometry names
- `all_isometries()` → ["id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "flip_anti"]
- `apply_isometry(g, name)` → Apply named D8 transform
- `flip_anti(g)` → Anti-diagonal reflection (H×W → W×H)

**Invariants:**
- Fixed deterministic ordering (never changes)
- Complete D8 group coverage
- Purity + idempotence properties

**Tests:** 38 tests in `tests/test_isometries.py` (100% pass)

#### P1-03: Π Canonical Grid (lines 148-243)
**Public API:**
- `canonical_key(g)` → (rows, cols, row_major_values) for lexicographic comparison
- `canonical_grid(g)` → D8 transform with minimal canonical_key (Π operator)

**Invariants:**
- **Π² = Π** (idempotence): `canonical_grid(canonical_grid(g)) == canonical_grid(g)`
- Minimality: Result has lexicographically minimal key over all D8 transforms
- Deterministic tie-breaking (earliest σ in all_isometries() order)

**Tests:** 27 tests in `tests/test_canonicalization.py` (100% pass)

#### P1-04: OFA Patch Normalization (lines 246-395)
**Public API:**
- `ofa_normalize_patch_colors(p)` → Remap colors by order-of-first-appearance (row-major)
- `canonical_patch_key(p)` → (rows, cols, ofa_values_flat) for comparison
- `canonical_d8_patch(p)` → D8-minimal patch via OFA keys (patch-level Π)

**Invariants:**
- **OFA Locality:** Uses only patch-local color ordering (never global palette)
- **Π² = Π on patches:** `canonical_d8_patch(canonical_d8_patch(p)) == canonical_d8_patch(p)`
- Compact palette: Output uses {0..k-1} for k distinct colors

**Tests:** 33 tests in `tests/test_canonical_patch.py` (100% pass)

---

### src/components.py (360 lines) — P1-05, P1-06
**Purpose:** 8-connected components, bounding boxes, NPS boundary detection

#### P1-05: 8-Connected Components (lines 1-202)
**Public API:**
- `NEIGHBORS_8` → 8 neighbor offsets (including diagonals)
- `bbox(cells)` → (r0, c0, r1, c1) inclusive bounding box
- `components_by_color(g)` → Extract 8-connected components grouped by color

**Component Structure:**
```python
{
    "id": int,              # 0-based ID within color (deterministic)
    "color": int,           # Pixel value (0..9)
    "cells": list[(r,c)],   # Sorted row-major ascending
    "bbox": (r0,c0,r1,c1)   # Inclusive bounds
}
```

**Invariants:**
- **8-connectivity:** max(|Δr|, |Δc|) == 1 (includes diagonals, unlike 4-connected)
- **Deterministic IDs:** Sorted by (-size, bbox_lex, first_cell_lex)
- Cells sorted row-major: (r,c) lex ascending
- Purity: No mutations, no aliasing

**Tests:** 33 tests in `tests/test_components.py` (100% pass)

#### P1-06: NPS Boundary Bands (lines 205-359)
**Public API:**
- `boundaries_by_any_change(g, axis)` → Detect change boundaries along rows or columns
- `bands_from_boundaries(n, boundaries)` → Convert boundary indices to inclusive (start, end) bands

**Semantics:**
- `axis="row"`: Boundary at i if rows i and i+1 differ in ANY column
- `axis="col"`: Boundary at j if columns j and j+1 differ in ANY row
- Bands partition [0..n-1] exactly (no gaps, no overlaps)

**Invariants:**
- **Input-only (Φ.3):** Never depends on target Y
- **Exact equality:** Uses `!=` comparison (no heuristics)
- **Partition correctness:** Full coverage, disjoint, contiguous bands
- Purity + determinism

**Tests:** 42 tests in `tests/test_nps_boundaries.py` (100% pass)

---

## Test Coverage Summary

| Module | Test File | Tests | Status |
|--------|-----------|-------|--------|
| P1-01: utils.py | test_utils.py | 45 | ✓ 100% |
| P1-02: isometries | test_isometries.py | 38 | ✓ 100% |
| P1-03: canonical grid | test_canonicalization.py | 27 | ✓ 100% |
| P1-04: OFA patches | test_canonical_patch.py | 33 | ✓ 100% |
| P1-05: components | test_components.py | 33 | ✓ 100% |
| P1-06: NPS bands | test_nps_boundaries.py | 42 | ✓ 100% |
| **Total** | **6 files** | **218** | **✓ 100%** |

**Test Execution:**
```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_<module>.py -v    # Run specific module
```

---

## Code Statistics

| Category | Lines | Complexity |
|----------|-------|------------|
| `src/utils.py` | 268 | LOW |
| `src/canonicalization.py` | 395 | MEDIUM |
| `src/components.py` | 360 | MEDIUM |
| **Phase 1 Total** | **1,023** | **LOW-MEDIUM** |

**Expected Final:** ~7,000-8,000 lines (across 12 phases)

---

## Documentation

### Anchor Documents (Source of Truth)
- `docs/anchors/primary-anchor.md` — Mathematical foundation (Π/FY/GLUE principles)
- `docs/anchors/spec.md` — Locked algebra (16 families, Φ features, action set)
- `docs/anchors/implementation_plan.md` — Pseudocode guide
- `docs/anchors/fundamental_decisions.md` — Resolved ambiguities

### Context Packs (Implementation Briefs)
- `docs/context-packs/P1-01.md` — Core Grid Utilities
- `docs/context-packs/P1-02.md` — D8 Isometries Registry
- `docs/context-packs/P1-03.md` — Π Canonical Grid
- `docs/context-packs/P1-04.md` — OFA Patch Recolor + D8 Patch Canonical
- `docs/context-packs/P1-05.md` — 8-Connected Components by Color
- `docs/context-packs/P1-06.md` — NPS Boundary Bands

Each context pack provides:
- Scope (in/out, LOC budget)
- API contracts (signatures + semantics)
- Mini fixtures (copy-paste test data)
- Tests to add (assertions + edge cases)
- Acceptance gates (review criteria)
- Risks & don'ts (implementation pitfalls)

### Review Reports (Mathematical Verification)
- `reviews/math_invariants_review_P1-01.md` through `P1-06.md`
- Each confirms: mathematical properties verified, no toy patterns, PASS/FAIL verdict

---

## Phase 1 Completion Checklist

- [x] **P1-01:** Core Grid Utilities — 45 tests ✓
- [x] **P1-02:** D8 Isometries Registry — 38 tests ✓
- [x] **P1-03:** Π Canonical Grid — 27 tests ✓
- [x] **P1-04:** OFA Patch Recolor + D8 Patch Canonical — 33 tests ✓
- [x] **P1-05:** 8-Connected Components by Color — 33 tests ✓
- [x] **P1-06:** NPS Boundary Bands — 42 tests ✓
- [x] **All 218 tests passing**
- [x] **6 mathematical correctness reviews (all PASS)**
- [x] **No stubs, no TODOs, no prototype code**

**Phase 1 Status:** ✅ **FULLY COMPLETE** — Ready for Phase 2

---

## Next Phase

**Phase 2:** 16 Global Families (P)
**Duration:** ~2 days
**LOC Estimate:** ~2,500 lines
**Complexity:** MEDIUM (16 independent solver modules)
**Dependencies:** Phase 1 complete ✓
**Expected Coverage:** N/A (scaffolding for Step 1)

See `docs/phasewise_implementation_plan.md` lines 203-332 for detailed Phase 2 specifications.

---

**Maintainers:** Update this file when:
- New modules are added to `src/`
- New test files are created
- Phase milestones are reached
- Directory structure changes
