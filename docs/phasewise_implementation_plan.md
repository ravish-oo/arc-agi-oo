# Phasewise Implementation Plan

## Purpose of This Document

This document provides a **phase-by-phase implementation roadmap** for building the ARC AGI pure math solver. It ensures:
- **No stubs or toy code** - every piece is complete and tested before moving forward
- **Clear dependencies** - inside-out order ensures each phase builds on working components
- **Measurable progress** - coverage tracking after each vertical slice
- **Implementation fidelity** - checkboxes prevent forgotten work

**Last updated:** 2025-10-24

---

## How to Use This Document

1. **Work through phases sequentially** - Don't skip ahead
2. **Complete all checkboxes in a phase** before moving to next
3. **Test after each phase** - Verify deliverables
4. **Measure coverage** - Track progress on training data
5. **Update checkboxes** - Mark tasks complete as you go
6. **Add sub-tasks** - Break down phases further if needed

**Golden Rule:** Never write code you can't test immediately.

---

## The Golden Rules (Read Before Starting)

### 1. NO STUBS, EVER

❌ **WRONG:**
```python
def solve_step2(task):
    # TODO: implement Φ partition
    pass
```

✅ **RIGHT:**
```python
# Don't write solve_step2() until Φ partition is fully implemented
# Work on signature_builders.py first (Phase 4)
```

### 2. Inside-Out Dependency Order

Only implement what you can **test independently**:
- Phase 1 (utils) has no dependencies → implement first
- Phase 2 (families) depends on Phase 1 → implement second
- Phase 3 (Step 1 solver) depends on Phase 2 → implement third
- etc.

### 3. Test Everything Immediately

After implementing a module:
```bash
# Unit test
pytest tests/test_<module>.py

# Integration test (if applicable)
python <module>.py --test data/sample_task.json
```

### 4. Measure Coverage After Each Vertical Slice

After completing Step 1 (Phase 3):
```bash
python measure_coverage.py data/arc-agi_training_challenges.json
# Expected: ~25-30% baseline coverage
```

After completing Step 2 (Phase 7):
```bash
python measure_coverage.py data/arc-agi_training_challenges.json
# Expected: ~60-70% coverage (increase of +40%)
```

### 5. Only Advance When Phase is 100% Complete

- All checkboxes ticked
- All tests passing
- Coverage measured (for solver phases)
- No TODOs, no placeholders, no "come back later"

---

## Implementation Phases

### Phase 1: Foundation (No Dependencies) ✅ COMPLETE

**Duration:** 2 days
**Deliverable:** Base utilities, canonicalization, component detection
**Dependencies:** None
**Anchor References:** primary-anchor.md §1, spec.md lines 8-13

#### Modules to Implement:

**`src/utils.py`**
- [ ] `dims(g)` - get grid dimensions
- [ ] `copy_grid(g)` - deep copy of grid
- [ ] `deep_eq(a, b)` - equality check for grids
- [ ] `transpose(g)` - matrix transpose
- [ ] `rot90(g)` - 90° rotation
- [ ] `rot180(g)` - 180° rotation
- [ ] `rot270(g)` - 270° rotation
- [ ] `flip_h(g)` - horizontal flip
- [ ] `flip_v(g)` - vertical flip

**`src/canonicalization.py`**
- [ ] `ISOMETRIES` - list of D8 transformations
- [ ] `canonical_grid(g)` - lexicographically minimal D8 variant
- [ ] `ofa_normalize_patch_colors(p)` - order-of-first-appearance color remapping
- [ ] `canonical_d8_patch(p)` - D8+OFA canonical patch
- [ ] Test: Π² = Π (idempotence)

**`src/components.py`**
- [ ] `components_by_color(g)` - 8-connected component detection
- [ ] `bbox(cells)` - bounding box of component
- [ ] `boundaries_by_any_change(g, axis)` - NPS band detection
- [ ] Test: Components stable, correct 8-connectivity

#### Testing Criteria:
```bash
pytest tests/test_utils.py          # All pass
pytest tests/test_canonicalization.py  # Π² = Π verified
pytest tests/test_components.py     # Components correct
```

#### Notes:
- **Π idempotence is critical** - test thoroughly
- Use primary-anchor.md §1 (lines 37-42) for Π specification
- 8-connected (not 4-connected) per fundamental_decisions.md Decision 10

#### Implementation WOs
Full Phase-1 WO lineup (titles only), each ≤500-LOC and dependency-free beyond prior WOs:

* **P1-01 — Core Grid Utilities (dims, copy, deep_eq, transpose, rot90/180/270, flip_h/v)** ✅ COMPLETE
* **P1-02 — D8 Isometries Registry (ISOMETRIES) and shape-safety contracts** ✅ COMPLETE
* **P1-03 — Π: Lexicographic Canonical Grid over D8 (canonical_grid) + Π² tests** ✅ COMPLETE
* **P1-04 — OFA Patch Recolor + D8 Patch Canonical (ofa_normalize_patch_colors, canonical_d8_patch) + Π² on patches** ✅ COMPLETE
* **P1-05 — 8-Connected Components by Color + bbox (deterministic IDs, tie-breaks)** ✅ COMPLETE
* **P1-06 — NPS Boundary Bands (boundaries_by_any_change) with row/col suites** ✅ COMPLETE


---

### Phase 2: Global Families (16 Independent Modules)

**Duration:** 2 days
**Deliverable:** All 16 global P families, each complete and tested
**Dependencies:** Phase 1
**Anchor References:** spec.md lines 19-27, implementation_plan.md lines 127-638

#### Suggested Implementation Order:

**Simple families first (test fundamentals):**
1. [ ] `src/families/isometry.py` - D8 transformations
2. [ ] `src/families/color_map.py` - Color palette remapping
3. [ ] `src/families/iso_color_map.py` - Isometry + color map

**Scaling families:**
4. [ ] `src/families/pixel_replicate.py` - Uniform upsampling (kH × kW)
5. [ ] `src/families/block_down.py` - Uniform downsampling (center/majority/min/max/first_nonzero)

**Non-uniform scaling:**
6. [ ] `src/families/nps_down.py` - Non-uniform partition downsampling
7. [ ] `src/families/nps_up.py` - Non-uniform partition upsampling

**Tiling and permutations:**
8. [ ] `src/families/parity_tile.py` - Tile with h/v/hv parity flips
9. [ ] `src/families/block_permutation.py` - Tile permutation
10. [ ] `src/families/block_substitution.py` - Per-color k×k glyph expansion

**Row/column transforms:**
11. [ ] `src/families/row_permutation.py` - Row reordering
12. [ ] `src/families/col_permutation.py` - Column reordering
13. [ ] `src/families/sort_rows.py` - Lexicographic row sort
14. [ ] `src/families/sort_cols.py` - Lexicographic column sort

**Symmetry and objects:**
15. [ ] `src/families/mirror_complete.py` - H/V/Diag mirror completion
16. [ ] `src/families/copy_move.py` - Component translation

#### Each family must implement:

```python
class FamilyName:
    def __init__(self):
        self.name = "FamilyName"
        # ... learned parameters

    def fit(self, train_pairs):
        """
        Learn parameters on first pair, verify on all pairs.
        Returns True if FY satisfied (bit-for-bit equality on all trains).
        """
        # FULL IMPLEMENTATION (no stubs)
        ...

    def apply(self, X):
        """Apply learned transform to input X."""
        # FULL IMPLEMENTATION (no stubs)
        ...
```

#### Testing Criteria:

For EACH family:
```bash
# Unit test with synthetic data
pytest tests/test_families/test_<family>.py

# Integration test on real training tasks
python -m src.families.<family> --test data/sample_tasks/<family>_example.json
```

Create a test runner:
```bash
pytest tests/test_families/  # All 16 families pass
```

#### Notes:
- **Each family is independent** - implement in any order
- See implementation_plan.md lines 127-638 for pseudocode
- Test FY acceptance: `all(deep_eq(apply(X), Y) for X,Y in trains)`
- Use spec.md lines 19-27 for family definitions

#### Implementation WOs
Here’s the Phase-2 WO lineup (titles only), mapped 1:1 to the 16 global families:

* **P2-01 — Isometry (D8/transpose)** ✅ COMPLETE 
* **P2-02 — ColorMap (per-color LUT)** ✅ COMPLETE 
* **P2-03 — IsoColorMap (isometry + per-color LUT)** ✅ COMPLETE 
* **P2-04 — PixelReplicate (uniform upsampling kH×kW)** ✅ COMPLETE 
* **P2-05 — BlockDown (center/majority/min/max/first_nonzero)** ✅ COMPLETE 
* **P2-06 — NPSDown (band aggregation over change boundaries)**
* **P2-07 — NPSUp (band replication maps learned from train)**
* **P2-08 — ParityTile (tile with h/v/hv parity flips)**
* **P2-09 — BlockPermutation (tile reorder)**
* **P2-10 — BlockSubstitution (per-color k×k glyph expansion)**
* **P2-11 — RowPermutation (row reordering)**
* **P2-12 — ColPermutation (column reordering)**
* **P2-13 — SortRowsLex (lexicographic row sort)**
* **P2-14 — SortColsLex (lexicographic column sort)**
* **P2-15 — MirrorComplete (H/V/Diag symmetry completion)**
* **P2-16 — CopyMoveAllComponents ((Δr,Δc) per color)**

These 16 are the locked global P menu (spec; families list), and Phase-2 implements each as an independent, fully tested module before Step-1 uses them.  

---

### Phase 3: Step 1 Solver (First Vertical Slice)

**Duration:** 1-2 days
**Deliverable:** Working Step 1 solver with baseline coverage
**Dependencies:** Phase 1, Phase 2
**Anchor References:** primary-anchor.md lines 106-109, spec.md line 64

#### Modules to Implement:

**`src/solver_step1.py`**
- [ ] `solve_step1(task)` - main entry point
- [ ] Loop over all 16 families
- [ ] For each family: call `fit()`, verify FY on all trains
- [ ] Return first passing family's predictions
- [ ] Generate receipts: `{"mode": "global", "solver": family.name}`

**`src/receipts.py`**
- [ ] `generate_receipt_global(family, task)` - PASS receipt for Step 1
- [ ] `generate_receipt_unsat(reason)` - UNSAT receipt

**`tests/measure_coverage.py`** (critical for tracking)
- [ ] Load all training tasks
- [ ] Run solver_step1 on each
- [ ] Count PASS vs UNSAT
- [ ] Breakdown by family (which solved what)
- [ ] Output: coverage %, family stats

#### Testing Criteria:

```bash
# Single task test
python src/solver_step1.py --task data/arc-agi_training_challenges.json --task-id <id>

# Full training set
python tests/measure_coverage.py data/arc-agi_training_challenges.json

# Expected output:
# ========================================
# Step 1 Coverage: 112/400 (28%)
# ========================================
# Isometry: 34 tasks
# ColorMap: 28 tasks
# PixelReplicate: 19 tasks
# ...
# UNSAT: 288 tasks
```

#### Deliverable Checklist:
- [ ] All 400 training tasks run without crashing
- [ ] Coverage measured and logged
- [ ] Receipts generated for all tasks
- [ ] PASS tasks verified manually (sample 10-20)
- [ ] UNSAT tasks logged for Phase 4-7 analysis

#### Notes:
- **This is first working vertical slice** - celebrate!
- Baseline coverage expected: **25-30%**
- Step 1 uses **first-pass** (not MDL) per fundamental_decisions.md
- See primary-anchor.md lines 106-109 for Step 1 specification

---

### Phase 4: Signature Builders (Φ Features)

**Duration:** 2 days
**Deliverable:** Complete Φ signature system
**Dependencies:** Phase 1
**Anchor References:** primary-anchor.md lines 63-82, spec.md lines 32-40

#### Modules to Implement:

**`src/signature_builders.py`**

**Index predicates:**
- [ ] `parity_mask(g)` - (r+c) mod 2 ∈ {0,1}
- [ ] `rowmod_mask(g, k)` - r mod k for k ∈ {2,3}
- [ ] `colmod_mask(g, k)` - c mod k for k ∈ {2,3}

**NPS bands:**
- [ ] `row_band_masks(g)` - from `boundaries_by_any_change(g, axis=0)`
- [ ] `col_band_masks(g)` - from `boundaries_by_any_change(g, axis=1)`

**Local content:**
- [ ] `is_color_mask(g, c)` - pixels with color c
- [ ] `touching_color_mask(g, c)` - 4-neighbor dilation one step

**Component structure:**
- [ ] `component_id_table(g)` - assign IDs to 8-connected components
  - Tie-breaking: -size → bbox lex (per spec)

**Patch keys:**
- [ ] `patchkey_table(g, r)` - for r ∈ {2,3,4} (5×5, 7×7, 9×9)
  - Extract patch around each pixel
  - Apply OFA normalization
  - Apply D8 canonicalization
  - Store canonical key per pixel

**Main function:**
- [ ] `phi_signature_tables(X)` - returns dict of all features for grid X

#### Testing Criteria:

**Φ.3 Stability test (critical):**
```python
# Test: Changing Y does not affect Φ
X = [[1,2],[3,4]]
Y1 = [[5,6],[7,8]]
Y2 = [[9,0],[1,2]]

phi1 = phi_signature_tables(X)
phi2 = phi_signature_tables(X)  # Same X

assert phi1 == phi2  # Must be identical
```

**Finiteness test:**
```python
# For bounded grids, signature space is finite
# Verify: number of distinct signatures < some bound
```

**Disjointness test (Φ.2):**
```python
# Two distinct signatures → disjoint pixel sets
```

```bash
pytest tests/test_signatures.py  # All stability/finiteness/disjointness tests pass
```

#### Notes:
- **Φ uses ONLY input features** (never Y) - see fundamental_decisions.md Decision 8
- OFA is local (patch-level only) - see fundamental_decisions.md Decision 7
- primary-anchor.md lines 63-82 for full Φ specification
- This phase is **independent of solver** - pure feature engineering

---

### Phase 5: Action Inference

**Duration:** 1 day
**Deliverable:** Complete action inference system
**Dependencies:** Phase 4
**Anchor References:** primary-anchor.md lines 83-96, spec.md lines 43-49

#### Modules to Implement:

**`src/action_inference.py`**

**Action set A:**
- [ ] `apply_set_color(Xp, coords, color)` - set all coords to color
- [ ] `apply_mirror_h(Xp, coords)` - horizontal mirror at coords
- [ ] `apply_mirror_v(Xp, coords)` - vertical mirror at coords
- [ ] `apply_keep_nonzero(Xp, coords)` - keep non-zero, else 0
- [ ] `apply_identity(Xp, coords)` - no change

**Inference logic:**
- [ ] `infer_action_for_class(items, class_coords)` - main function
  - Try each action in order: set_color, mirror_h, mirror_v, keep_nonzero, identity
  - For set_color: only try if all target pixels in class have same color
  - Accept first action that matches ALL class pixels across ALL trains
  - Return None if no action matches (UNSAT for this class)

**Testing helper:**
- [ ] `verify_action_on_class(action, items, class_coords)` - check if action satisfies FY

#### Testing Criteria:

**Synthetic tests:**
```python
# Test: set_color inference
items = [
    (Xp=[[0,0]], Y=[[1,1]], feats=..., residual=...),
    (Xp=[[0,0]], Y=[[1,1]], feats=..., residual=...)
]
class_coords = [(0,0,0), (0,0,1), (1,0,0), (1,0,1)]  # All pixels

action = infer_action_for_class(items, class_coords)
assert action == ("set_color", 1)
```

**Real-world test:**
- Create small tasks where answer is known
- Verify correct action inferred

```bash
pytest tests/test_actions.py  # All action inference tests pass
```

#### Notes:
- Actions are **deterministic** - no randomness
- See spec.md lines 43-49 for action definitions
- primary-anchor.md lines 193-196 for inference logic
- Constructive actions (draw_line, draw_box) will be added in Phase 12 if needed

---

### Phase 6: GLUE Stitching

**Duration:** 1 day
**Deliverable:** Class partitioning and stitching system
**Dependencies:** Phase 4, Phase 5
**Anchor References:** primary-anchor.md lines 105-117, fundamental_decisions.md Decision 11

#### Modules to Implement:

**`src/glue.py`**

**Residual computation:**
- [ ] `compute_residual(Xp, Y)` - pixelwise difference (None if equal, else Y[r][c])

**Class building:**
- [ ] `build_phi_partition(tr_pairs_afterP)` - main function
  - Input: list of (P(X), Y) pairs
  - Compute residual for each pair
  - Build Φ signatures for each P(X)
  - Group residual pixels by signature
  - Return: (items, classes)
    - items: [(Xp, Y, feats, residual), ...]
    - classes: {class_id: [(train_idx, r, c), ...]}

**Stitching:**
- [ ] `stitch_from_classes(items, classes, actions_by_cid)` - apply actions per class
  - For each train: start with copy of Xp
  - For each class: apply action to class pixels
  - Return list of stitched outputs

**Verification:**
- [ ] `verify_stitched_equality(items, classes, actions_by_cid)` - FY check
  - Stitch outputs
  - Compare to target Y for each train
  - Return True only if ALL trains match exactly

#### Testing Criteria:

**GLUE exactness test (Theorem T3):**
```python
# Disjoint classes → stitched = one-shot
# Create synthetic task with known disjoint classes
# Verify stitching produces exact result
```

**Seam detection test:**
```python
# If classes overlap → should detect and reject
# (Though Φ guarantees disjoint by construction)
```

```bash
pytest tests/test_glue.py  # Stitching exactness verified
```

#### Notes:
- **Disjoint classes are guaranteed by Φ** (Φ.2 property)
- See primary-anchor.md lines 132-133 for GLUE theorem
- fundamental_decisions.md Decision 1 has full stitching example

---

### Phase 7: Step 2 Solver (Second Vertical Slice)

**Duration:** 2 days
**Deliverable:** Working Step 2 solver with MDL selection
**Dependencies:** Phase 1, Phase 2, Phase 4, Phase 5, Phase 6
**Anchor References:** primary-anchor.md lines 111-127, fundamental_decisions.md Decision 11

#### Modules to Implement:

**`src/mdl_selection.py`**
- [ ] `compute_mdl(P, classes, actions)` - returns (num_classes, num_action_types, p_index, hash)
- [ ] `hash_candidate(P, classes)` - stable hash for tie-breaking
- [ ] Test: deterministic ordering (same input → same hash)

**`src/solver_step2.py`**
- [ ] `solve_step2(task)` - main entry point
- [ ] **Shape safety check**: skip P if dims(P(X)) ≠ dims(Y) for any train
- [ ] Loop over ALL P ∈ {Identity} ∪ GLOBAL_MENU:
  - Apply P to all trains
  - Build Φ partition
  - Infer actions for each class
  - If any class fails → skip this P
  - Verify GLUE/FY
  - If pass → add (mdl_cost, P, classes, actions) to candidates
- [ ] After loop: if candidates non-empty, pick best by MDL
- [ ] Apply chosen candidate to test inputs
- [ ] Generate receipts: `{"mode": "phi_partition", "mdl_candidates": [...], "chosen_candidate": {...}}`

**Update `src/receipts.py`:**
- [ ] `generate_receipt_phi(P, classes, actions, mdl_candidates)` - full MDL receipts

**Update `tests/measure_coverage.py`:**
- [ ] Add Step 2 measurement
- [ ] Show coverage delta from Step 1

#### Testing Criteria:

```bash
# Single task test
python src/solver_step2.py --task-id <id>

# Full training set
python tests/measure_coverage.py data/arc-agi_training_challenges.json --steps 1,2

# Expected output:
# ========================================
# Step 1 Coverage: 112/400 (28%)
# Step 2 Coverage: 268/400 (67%)  [+156 tasks, +39%]
# ========================================
```

**MDL tie-breaking test:**
```python
# Create task where multiple P pass
# Verify: best by MDL is selected
# Verify: deterministic (run twice, same result)
```

#### Deliverable Checklist:
- [ ] All 400 tasks run without crashing
- [ ] Coverage increased significantly (+30-40% expected)
- [ ] MDL candidates logged in receipts
- [ ] Shape safety prevents crashes
- [ ] Determinism verified (same task → same solution)

#### Notes:
- **This is the MAJOR coverage boost** - compositional power unlocks
- See fundamental_decisions.md Decision 11 for MDL specification
- primary-anchor.md lines 111-127 for complete Step 2 algorithm
- Step 2 uses **best-pass** (MDL), not first-pass like Step 1

---

### Phase 8: LUT (Local Lookup Tables)

**Duration:** 1 day
**Deliverable:** LUTPatch for r=2,3,4
**Dependencies:** Phase 1
**Anchor References:** implementation_plan.md lines 640-677

#### Modules to Implement:

**`src/lut.py`**

**`class LUTPatch`:**
- [ ] `__init__(self, r)` - radius r ∈ {2,3,4}
- [ ] `_extract_patch_key(X, i, j)` - extract (2r+1)×(2r+1) patch around (i,j)
  - Apply OFA normalization
  - Apply D8 canonicalization
  - Return canonical key
- [ ] `fit(self, train_pairs)` - build LUT
  - For each train pair (X,Y):
    - For each pixel (i,j):
      - key = _extract_patch_key(X, i, j)
      - value = Y[i][j]
      - If key already in LUT and LUT[key] ≠ value → CONFLICT → return False
      - Else: LUT[key] = value
  - Verify: all train pairs can be reproduced
- [ ] `apply(self, X)` - apply LUT to input
  - For each pixel: lookup patch key in LUT
  - If key not found → return None (UNSAT)
  - Else: output[i][j] = LUT[key]

#### Testing Criteria:

**Conflict detection test:**
```python
# Create task where same patch key maps to different outputs
# Verify: LUT.fit() returns False
```

**Reproduction test:**
```python
# Fit LUT on training pairs
# Verify: LUT.apply(X) == Y for all trains
```

```bash
pytest tests/test_lut.py  # All LUT tests pass
```

#### Notes:
- LUT is **whole-grid** approach (not per-class)
- See implementation_plan.md lines 640-677 for pseudocode
- OFA+D8 canonicalization ensures same local pattern → same key
- This is Phase 3's **fallback** when structure is hard to characterize

---

### Phase 9: Step 3 Solver (Third Vertical Slice)

**Duration:** 1 day
**Deliverable:** Complete Step 3 with final coverage measurement
**Dependencies:** Phase 8
**Anchor References:** primary-anchor.md lines 119-120

#### Modules to Implement:

**`src/solver_step3.py`**
- [ ] `solve_step3(task)` - main entry point
- [ ] Try LUTPatch for r=2, 3, 4 in order
- [ ] For each r:
  - lut = LUTPatch(r)
  - if lut.fit(train_pairs):
    - Apply to test inputs
    - Generate receipts: `{"mode": "lut", "solver": f"LUTPatch(r={r})"}`
    - Return predictions
- [ ] Return None if no r works (UNSAT)

**Update `tests/measure_coverage.py`:**
- [ ] Add Step 3 measurement
- [ ] Show final coverage across all 3 steps

#### Testing Criteria:

```bash
# Full training set with all 3 steps
python tests/measure_coverage.py data/arc-agi_training_challenges.json --steps 1,2,3

# Expected output:
# ========================================
# Step 1 Coverage: 112/400 (28%)
# Step 2 Coverage: 268/400 (67%)  [+156]
# Step 3 Coverage: 324/400 (81%)  [+56]
# ========================================
# Breakdown by step:
# - Step 1 (Global P): 112 tasks
# - Step 2 (P+Φ/GLUE): 156 tasks
# - Step 3 (LUT): 56 tasks
# - UNSAT: 76 tasks
```

#### Deliverable Checklist:
- [ ] Final coverage measured
- [ ] LUT contribution quantified
- [ ] UNSAT tasks analyzed (prepare for Phase 12)

#### Notes:
- Step 3 is **fallback** for tasks with local patterns but no global structure
- Expected contribution: +10-20% coverage
- See primary-anchor.md lines 119-120 for Step 3 specification

---

### Phase 10: Main Solver (Orchestration)

**Duration:** 1 day
**Deliverable:** Unified solver calling all 3 steps
**Dependencies:** Phase 3, Phase 7, Phase 9
**Anchor References:** primary-anchor.md lines 145-181 (pseudocode)

#### Modules to Implement:

**`src/solver.py`**

**`solve_task(task_id, task)` - main entry point:**
- [ ] Extract train pairs and test inputs from task
- [ ] Apply Π (canonicalization) to all inputs
- [ ] **Step 1**: Try global P families
  - If PASS → return predictions + receipts
- [ ] **Step 2**: Try P + Φ/GLUE with MDL
  - If PASS → return predictions + receipts
- [ ] **Step 3**: Try LUT fallback
  - If PASS → return predictions + receipts
- [ ] **Else**: Return UNSAT witness
  - Generate receipt: `{"mode": "unsat", "witness": {...}}`
  - Include: first failing class, reason, signature summary

**Update `src/receipts.py`:**
- [ ] `generate_receipt_unsat(witness)` - full UNSAT receipt with class details

#### Testing Criteria:

```bash
# Single task end-to-end
python src/solver.py --task-id <id>

# Full training set
python src/solver.py --all data/arc-agi_training_challenges.json --output results/
# Output:
# - results/predictions.json (all PASS predictions)
# - results/receipts.json (all task receipts)
# - results/summary.csv (task_id, mode, solver, coverage)
```

**Determinism test:**
```python
# Run solver twice on same task
# Verify: identical predictions and receipts
```

#### Deliverable Checklist:
- [ ] All 400 tasks processed end-to-end
- [ ] Receipts generated for every task (PASS and UNSAT)
- [ ] Determinism verified
- [ ] No crashes, all errors handled gracefully

#### Notes:
- This is **complete solver** - all 3 steps integrated
- See primary-anchor.md lines 145-181 for full pseudocode
- Receipts are critical - every decision must be auditable
- UNSAT witnesses guide Phase 12

---

### Phase 11: Submission Wrapper

**Duration:** 1 day
**Deliverable:** Kaggle-ready submission generator
**Dependencies:** Phase 10
**Anchor References:** submission_readiness.md

#### Modules to Implement:

**`src/submission_generator.py`**
- [ ] `generate_kaggle_submission(test_challenges_path, output_path)` - main function
- [ ] For each task:
  - Run solver
  - Extract top 2 attempts:
    - If PASS and multiple MDL candidates → use top 2
    - Else → duplicate best solution
  - If UNSAT → best-effort guess (Identity transform)
  - Format as `{"attempt_1": grid, "attempt_2": grid}`
- [ ] Write submission.json
- [ ] Validate format

**`src/best_effort.py`**
- [ ] `best_effort_guess(task, receipts)` - UNSAT fallback
  - Strategy: Identity (copy input)
  - Future: use closest P+Φ candidate from receipts

**`tests/test_submission_format.py`**
- [ ] Validate submission.json structure
- [ ] Check all task_ids present
- [ ] Check attempt_1 and attempt_2 exist
- [ ] Check grid format (list of lists, integers 0-9)

**`tests/test_local_scoring.py`**
- [ ] Score submission against solutions
- [ ] Compute Kaggle metric (EITHER attempt matches → 1, else 0)
- [ ] Breakdown by task

#### Testing Criteria:

```bash
# Generate submission for training set
python src/submission_generator.py data/arc-agi_training_challenges.json --output train_submission.json

# Validate format
python tests/test_submission_format.py train_submission.json data/arc-agi_training_challenges.json
# Output: ✓ Submission format is valid!

# Score locally
python tests/test_local_scoring.py train_submission.json data/arc-agi_training_solutions.json
# Output:
# Overall Score: 0.8125 (81.25%)
# Correct: 325 / 400
# Incorrect: 75 / 400
```

#### Deliverable Checklist:
- [ ] Submission format validated
- [ ] Local scoring matches expected coverage
- [ ] Two-attempt strategy working (top 2 MDL candidates)
- [ ] UNSAT best-effort implemented

#### Notes:
- See submission_readiness.md for complete specification
- This is **thin wrapper** - solver does all the work
- Two-attempt strategy leverages MDL infrastructure (Phase 7)
- Ready for Kaggle notebook integration

---

### Phase 12: UNSAT Analysis & Completion

**Duration:** Variable (1-3 days depending on findings)
**Deliverable:** 95%+ coverage on training set
**Dependencies:** Phase 10, Phase 11
**Anchor References:** primary-anchor.md lines 223-229, spec.md lines 46-47

#### Process:

**Analyze UNSAT tasks:**
- [ ] Load all UNSAT receipts from Phase 10
- [ ] Group by UNSAT reason:
  - no_action_matches_class_on_trains
  - lut_key_conflict
  - glue_mismatch
  - dimension_mismatch
- [ ] Examine sample tasks from each group
- [ ] Identify missing algebra

**Common missing pieces (if needed):**

**Constructive actions:**
- [ ] `draw_line(anchors)` - Bresenham line between component extrema
  - Implementation: ~50 lines
  - Plug into action inference (Phase 5)
- [ ] `draw_box(bbox)` - Rectangle border
  - Implementation: ~30 lines
  - Plug into action inference (Phase 5)

**Additional families (if discovered):**
- [ ] Identify pattern from UNSAT tasks
- [ ] Implement new family following Phase 2 template
- [ ] Add to GLOBAL_MENU
- [ ] Re-run solver

**Iterate:**
- [ ] Add missing piece
- [ ] Re-run on all 400 training tasks
- [ ] Measure coverage increase
- [ ] Repeat until 95%+ coverage

#### Testing Criteria:

```bash
# After adding missing pieces
python tests/measure_coverage.py data/arc-agi_training_challenges.json

# Target:
# ========================================
# Final Coverage: 382/400 (95.5%)
# ========================================
```

#### Deliverable Checklist:
- [ ] 95%+ coverage on training set
- [ ] All additions documented (which tasks needed what)
- [ ] Receipts updated to reflect new families/actions
- [ ] No regression (tasks that passed before still pass)

#### Notes:
- See spec.md lines 46-47 for pre-approved constructive actions
- Each addition must be **finite, deterministic, receipts-only**
- Document every addition for reproducibility
- This is **empirical completion** - guided by data, not guessing

---

## Testing Workflow Summary

### After Each Phase:

```bash
# 1. Unit tests
pytest tests/test_<module>.py

# 2. Integration test (if solver phase)
python src/solver_step<N>.py --task-id <sample>

# 3. Coverage measurement (if solver phase)
python tests/measure_coverage.py data/arc-agi_training_challenges.json --steps <N>
```

### Final Validation (After Phase 11):

```bash
# 1. Generate submission for evaluation set
python src/submission_generator.py data/arc-agi_evaluation_challenges.json --output eval_submission.json

# 2. Score on evaluation set
python tests/test_local_scoring.py eval_submission.json data/arc-agi_evaluation_solutions.json

# Expected: ~80-90% (similar to training coverage)

# 3. Validate determinism
python tests/test_determinism.py data/arc-agi_evaluation_challenges.json
# Run solver twice, verify identical outputs
```

---

## Coverage Tracking

### Expected Coverage Progression:

| Phase | Step | Expected Coverage | Cumulative | Notes |
|-------|------|------------------|------------|-------|
| 3 | Step 1 (Global P) | 25-30% | 28% | Baseline - pure global transforms |
| 7 | Step 2 (P+Φ) | +35-45% | 67% | Compositional power unlocks |
| 9 | Step 3 (LUT) | +10-20% | 81% | Local pattern fallback |
| 12 | + Missing algebra | +10-15% | 95%+ | Empirical completion |

### Tracking Template:

```
# coverage_log.md

## Phase 3 (Step 1 Complete)
- Date: 2025-10-XX
- Coverage: 112/400 (28%)
- Top families:
  - Isometry: 34 tasks
  - ColorMap: 28 tasks
  - PixelReplicate: 19 tasks
  - NPSDown: 12 tasks
  - ...

## Phase 7 (Step 2 Complete)
- Date: 2025-10-XX
- Coverage: 268/400 (67%)
- Delta: +156 tasks (+39%)
- Top P+Φ combos:
  - ColorMap + Φ: 48 tasks
  - Isometry + Φ: 39 tasks
  - Identity + Φ: 34 tasks
  - ...

## Phase 9 (Step 3 Complete)
- Date: 2025-10-XX
- Coverage: 324/400 (81%)
- Delta: +56 tasks (+14%)
- LUT breakdown:
  - r=2: 21 tasks
  - r=3: 24 tasks
  - r=4: 11 tasks

## Phase 12 (Final)
- Date: 2025-10-XX
- Coverage: 382/400 (95.5%)
- Additions:
  - draw_line action: +32 tasks
  - draw_box action: +18 tasks
  - Custom family X: +8 tasks
```

---

## Weekly Timeline (Suggested)

### Week 1: Foundation + Step 1
- **Mon-Tue:** Phase 1-2 (utils, families)
- **Wed-Thu:** Phase 3 (Step 1 solver)
- **Fri:** Coverage analysis, plan Week 2

**Deliverable:** Baseline coverage (~28%)

### Week 2: Step 2 (Compositional Power)
- **Mon-Tue:** Phase 4-6 (Φ, actions, GLUE)
- **Wed-Thu:** Phase 7 (Step 2 solver)
- **Fri:** Coverage analysis

**Deliverable:** Major coverage boost (~67%)

### Week 3: Completion
- **Mon:** Phase 8-9 (LUT, Step 3)
- **Tue:** Phase 10 (Main solver)
- **Wed:** Phase 11 (Submission wrapper)
- **Thu-Fri:** Phase 12 (UNSAT analysis, completion)

**Deliverable:** 95%+ coverage, Kaggle-ready

---

## Checklist: Overall Progress

### Foundation
- [ ] Phase 1: Utils, Π, components (COMPLETE)
- [ ] Phase 2: 16 global families (COMPLETE)

### Solver Vertical Slices
- [ ] Phase 3: Step 1 (Global P) - **Baseline coverage**
- [ ] Phase 7: Step 2 (P+Φ) - **Major boost**
- [ ] Phase 9: Step 3 (LUT) - **Final coverage**

### Support Systems
- [ ] Phase 4: Φ signatures
- [ ] Phase 5: Action inference
- [ ] Phase 6: GLUE stitching
- [ ] Phase 8: LUT

### Integration
- [ ] Phase 10: Main solver (all 3 steps)
- [ ] Phase 11: Submission wrapper
- [ ] Phase 12: UNSAT analysis + completion

### Validation
- [ ] All unit tests passing
- [ ] Coverage > 95% on training set
- [ ] Submission format validated
- [ ] Evaluation set tested
- [ ] Determinism verified

---

## Notes for Future Self

### When You Get Stuck:

1. **Check dependencies:** Did you complete prerequisite phases?
2. **Re-read anchors:** primary-anchor.md for algorithm, spec.md for components
3. **Check decisions:** fundamental_decisions.md for resolved ambiguities
4. **Test independently:** Isolate the failing module, test with synthetic data
5. **Measure coverage:** Run measure_coverage.py to see if other parts are working

### When Coverage is Low:

1. **Don't panic** - 28% after Step 1 is expected
2. **Analyze receipts** - Which families are working? Which aren't?
3. **Test families individually** - Verify each family on its expected tasks
4. **Check FY strictness** - Are we rejecting near-matches? (Good! That's by design)
5. **Wait for Step 2** - Compositional power is where the magic happens

### When UNSAT Count is High:

1. **After Step 1:** 70-75% UNSAT is normal
2. **After Step 2:** 30-35% UNSAT is expected
3. **After Step 3:** 15-20% UNSAT means we need Phase 12
4. **Examine witnesses** - They tell you exactly what's missing
5. **Add incrementally** - One missing piece at a time, measure impact

---

**Implementation begins with Phase 1. Good luck!** 🚀
