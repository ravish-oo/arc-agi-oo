# Context Index — ARC AGI Pure Math Solver

## Purpose of This Document

This is a **repository navigation map** (WHAT and WHERE, not HOW).

**Use this to:**
- Navigate to the right anchor document or code location
- Understand the hierarchy of truth when conflicts arise
- Find resolved decisions and ambiguities
- Quickly orient yourself or future agents to the repo structure

**How to edit:**
- Add new sections when new anchors/modules are created
- Update "Resolved Decisions" when ambiguities are clarified
- Keep entries minimal: 1-2 sentences per item, linking to the actual doc
- DO NOT duplicate content from anchors; point to them instead

---

## Source of Truth Hierarchy

When conflicts or ambiguities arise, resolve in this order:

1. **`docs/anchors/primary-anchor.md`** — Mathematical foundation, Π/FY/GLUE principles, theorems, algorithm structure
2. **`docs/anchors/spec.md`** — Locked algebra components (16 global families, Φ features, action set)
3. **`docs/anchors/implementation_plan.md`** — Pseudocode guide (algorithmic intent, NOT literal runnable code)

---

## Anchor Documents

### `docs/anchors/primary-anchor.md`
**What:** Complete mathematical specification of the solver
**Key sections:**
- §0: Universe-functioning principles (Π/FY/GLUE)
- §1: Canonicalization (D8, OFA)
- §2: Finite algebra (P global families, Φ signature, A actions)
- §3: Decision procedure (3-step algorithm)
- §4: Theorems with proof sketches
- §5: Pseudocode (exact algorithm flow)
- §6-7: Code contracts, receipts
- §10: FAQ for context drift

**Note:** Completeness is ITERATIVE — run on training corpus, collect UNSAT witnesses, add finite constructors as needed. See primary-anchor:251.

---

### `docs/anchors/spec.md`
**What:** Final locked specification of the algebra
**Key sections:**
- Three rules (Π/FY/GLUE) — fixed forever
- P: 16 global exact families (lines 19-27)
- Φ: Canonical partition features (lines 32-40)
- A: Deterministic actions (lines 43-49)
- Decision procedure (lines 53-65)
- Why this covers the full 1000 (lines 69-76)

**Note:** "LOCKED" means the STRUCTURE is final (Π/FY/GLUE, 3-step algorithm, finite algebra approach). The 16 families are EMPIRICAL claims validated by running on training data. See spec:75-76.

---

### `docs/anchors/implementation_plan.md`
**What:** Skeletal pseudocode guide showing algorithmic intent
**Key sections:**
- Lines 42-89: Utilities & Π (canonicalization)
- Lines 90-126: Components, bands, NPS helpers
- Lines 127-638: Global menu P (16 solver classes)
- Lines 640-677: LUTPatch (local fallback)
- Lines 678-858: Φ partition & class actions
- Lines 859-983: Driver (solve_task function)
- Lines 985-1018: Main script

**Note:** This is PSEUDOCODE. Syntax errors (e.g., `_init_` vs `__init__`) are expected. Structure and logic are what matter. See note added at top of file.

---

### `docs/anchors/fundamental_decisions.md`
**What:** Resolved ambiguities and confusions that naturally arise from reading the three anchor docs
**Purpose:** Eliminate context drift by capturing the reasoning journey from confusion to clarity
**When to read:** AFTER reading primary-anchor.md, spec.md, and implementation_plan.md

**Key decisions resolved:**
1. Loop over ALL P in Step 2 (correctness requirement + collect all passing + pick best by MDL)
2. Pseudocode is skeletal (structure matters, not literal syntax)
3. "LOCKED" means structure is final (algebra completeness is empirical)
4. Algebra completeness is the ONLY real unknown (everything else is specified)
5. Source of truth hierarchy (primary-anchor → spec → implementation_plan)
6. Step order matters (Global → P+Φ → LUT, not what pseudocode shows)
7. OFA is local only (patch-based signatures, not global transforms)
8. Φ uses input features only (Φ.3 stability - never peek at target Y)
9. Constructive actions (add when UNSAT witnesses show need)
10. Component connectivity (8-connected per spec, may need 4-connected variant)
11. MDL cost and best-candidate selection (when multiple P pass in Step 2, pick simplest by 4-level tie-breaking)

**Format:** Each decision shows: confusion → question → reasoning → resolution → implementation impact

**Use this:** When you're confused after reading anchors, find the relevant decision and follow the step-by-step reasoning to arrive at clarity.

---

### `docs/phasewise_implementation_plan.md`
**What:** Complete implementation roadmap with 12 sequential phases, checkboxes, and testing criteria
**Purpose:** Ensure NO STUBS, inside-out dependency order, and measurable progress at every step
**When to read:** BEFORE starting implementation AND to track progress during development

**Golden Rules (lines 28-84):**
1. NO STUBS, EVER — Never write code you can't test immediately
2. Inside-Out Dependency Order — Only implement what you can test independently
3. Test Everything Immediately — pytest after each module
4. Measure Coverage After Each Vertical Slice — Track Step 1, Step 2, Step 3 coverage
5. Only Advance When Phase is 100% Complete — All checkboxes ticked, all tests passing

**The 12 Phases (lines 87-856):**

| Phase | Focus | Duration | Dependencies | Expected Coverage |
|-------|-------|----------|--------------|-------------------|
| 1 | Foundation (utils, Π, components) | 2 days | None | N/A ✅ |
| 2 | 16 Global Families (P) | 2 days | Phase 1 | N/A ✅ |
| 3 | Step 1 Solver (Global P) | 1-2 days | Phase 1, 2 | **28% baseline** |
| 4 | Φ Signatures (features) | 2 days | Phase 1 | N/A |
| 5 | Action Inference (A) | 1 day | Phase 4 | N/A |
| 6 | GLUE Stitching | 1 day | Phase 4, 5 | N/A |
| 7 | Step 2 Solver (P+Φ+MDL) | 2 days | Phase 1-6 | **67% (+39%)** |
| 8 | LUT (Local Lookup Tables) | 1 day | Phase 1 | N/A |
| 9 | Step 3 Solver (LUT fallback) | 1 day | Phase 8 | **81% (+14%)** |
| 10 | Main Solver (orchestration) | 1 day | Phase 3,7,9 | N/A |
| 11 | Submission Wrapper | 1 day | Phase 10 | N/A |
| 12 | UNSAT Analysis & Completion | 1-3 days | Phase 10, 11 | **95%+ target** |

**Critical Complexity Zones (Zero Tolerance for Bugs):**
- **Phase 1**: Π idempotence (Π² = Π), OFA normalization (order-of-FIRST-appearance)
- **Phase 4**: Φ.3 stability (NEVER use Y features, only X)
- **Phase 5**: FY exactness (bit-for-bit equality, no approximation)
- **Phase 6**: GLUE disjointness (classes must partition perfectly)
- **Phase 7**: MDL selection (4-level tie-breaking), loop over ALL P (not first-pass), shape safety

**Code Size Estimate:** ~7000-8000 lines total
- Phase 1: ~500 lines (LOW complexity)
- Phase 2: ~2500 lines (MEDIUM - 16 independent modules)
- Phase 4: ~800 lines (HIGH - 10+ feature types, index bookkeeping)
- Phase 5-6: ~900 lines (MEDIUM-HIGH - tuple bookkeeping)
- Phase 7: ~800 lines (HIGHEST - 7 nested concerns)
- Phase 8-11: ~1500 lines (MEDIUM)

**Testing Workflow (lines 859-888):**
```bash
# After each phase
pytest tests/test_<module>.py              # Unit tests
python src/solver_step<N>.py --task-id <id>  # Integration test (solver phases)
python tests/measure_coverage.py            # Coverage measurement (vertical slices)
```

**Coverage Tracking (lines 890-945):**
- Step 1 (Phase 3): 25-30% baseline (pure global transforms)
- Step 2 (Phase 7): +35-45% boost (compositional power unlocks)
- Step 3 (Phase 9): +10-20% (local pattern fallback)
- Final (Phase 12): +10-15% (empirical completion via UNSAT analysis)

**Use this plan to:**
- Know what to implement next (sequential phases)
- Track progress with checkboxes
- Verify testing criteria before advancing
- Estimate remaining work
- Create context packs for specialized implementer agents
- Review scope before agent assignment

---

## Resolved Decisions & Ambiguities

**→ For full details, see `docs/anchors/fundamental_decisions.md`**

The sections below provide quick summaries. For complete reasoning and implementation impact, consult fundamental_decisions.md.

### 1. Step 2 Algorithm: Loop Over All P
**Question:** Should Step 2 (Φ-partition mode) try only Identity or loop over all P?
**Answer:** MUST loop over all P ∈ {Identity} ∪ GLOBAL_MENU per primary-anchor:106
**Reason:** Compositional power requires trying "global transform + local class-based fixes"
**Status:** [To be confirmed by user regarding implementation priority]

### 2. Algorithm Step Order
**Canonical order (per primary-anchor:100-116):**
1. Global P exact (try each P alone)
2. P + Φ/GLUE (try each P with local class-based corrections)
3. Whole-grid LUT fallback

### 3. Constructive Actions (draw_line, draw_box)
**Status:** Listed in spec.md as "locked" but not in implementation_plan pseudocode
**Decision:** Implement when UNSAT witnesses show need; trivial ~50 line addition to action set
**Location:** Will be added to actions in Φ partition logic when needed

### 4. Algebra Exhaustiveness (16 Global Families)
**Claim:** The 16 families in spec.md:28 cover all ARC transformations
**Reality:** Empirical claim requiring validation
**Approach:** Run on all 400 training tasks → analyze UNSAT witnesses → add missing finite constructors → iterate to closure
**Expectation:** May discover additional families; each addition is finite and receipts-guided

### 5. Component Connectivity
**Current:** 8-connected only (implementation_plan:100-107)
**Potential gap:** Some tasks might need 4-connected
**Status:** Will discover via UNSAT witnesses if needed

---

## Repository Structure

### `/data/`
ARC AGI dataset downloaded from Kaggle
- `arc-agi_training_challenges.json` — Full 1000 training tasks
- `arc1_training.json` — ARC-1 subset (391 original tasks from 2019)
- `arc2_training.json` — ARC-2 subset (609 new tasks from 2024)
- Evaluation and test sets

### `/docs/`
All documentation and anchor files

### `/docs/anchors/`
Source-of-truth specifications
- `primary-anchor.md` — Mathematical foundation
- `spec.md` — Locked algebra components
- `implementation_plan.md` — Pseudocode guide
- `fundamental_decisions.md` — Resolved confusions with step-by-step reasoning
- `for-human-understanding.md` — Plain English summary (read this first for overview)

### `/docs/` (implementation guides)
- `context_index.md` (this file) — Repository navigation map
- `phasewise_implementation_plan.md` — 12-phase implementation roadmap with checkboxes
- `architecture.md` — Folder structure and module details
- `kaggle_submission_readiness.md` — Kaggle submission requirements
- `arc-agi-kaggle-docs.md` — Competition rules

### `/docs/context-packs/`
- Phase-specific implementation briefs
  - **Phase 1**: P1-01 through P1-06 (foundation modules) ✅
  - **Phase 2**: P2-01 through P2-16 (16 global families) ✅
  - **Phase 3**: P3-01 through P3-04 (Step-1 solver, receipts, CLI, coverage meter) ✅

### `/reports/`
Coverage tracking and metrics
- `/reports/coverage/` — Step-1 solver coverage baselines (ARC-1 vs ARC-2 breakdown)

### `/scripts/`
Utility scripts
- `split_arc_dataset.py` — Split training dataset into ARC-1 (391) and ARC-2 (609) subsets

### `/src/`
**Phase 1 Foundation** (for APIs and usage, see architecture.md):
- `utils.py` — Grid primitives, D8 transformations (dims, transpose, rotations, flips)
- `canonicalization.py` — D8 isometries, Π canonical operator, OFA patch normalization
- `components.py` — 8-connected components, bounding boxes, NPS boundary detection

**Phase 2 Global Families** (`/src/families/`, 16 modules):
- All 16 global P families implemented (Isometry, ColorMap, IsoColorMap, PixelReplicate, BlockDown, NPSDown, NPSUp, ParityTile, BlockPermutation, BlockSubstitution, RowPermutation, ColPermutation, SortRowsLex, SortColsLex, MirrorComplete, CopyMoveAllComponents)
- See spec.md lines 19-27 for family definitions

**Phase 3 Step-1 Solver**:
- `solver_step1.py` — Step-1 solver (global P only, first-pass selection) with CLI
- `receipts.py` — Proof-of-work receipts (PASS/UNSAT, task_meta, stable hash)

### `/tests/`
- **Phase 1**: 218 tests (utils, canonicalization, components) ✅
- **Phase 2**: 581 tests (16 global families, comprehensive coverage) ✅
- **Phase 3**: 55 tests (receipts, CLI, coverage meter) ✅
- `measure_coverage.py` — Coverage measurement tool (deterministic, sorted output, per-family breakdown)

### `/reviews/`
- **Phase 1**: P1-01 through P1-06 mathematical correctness verification (all PASS) ✅
- **Phase 2**: P2-01 through P2-16 math-reviewer verification (all PASS) ✅
- **Phase 3**: P3-02 (receipts), P3-04 (coverage meter) algorithm-guardian verification (all PASS) ✅

---

## Quick Navigation Cheat Sheet

| I need to... | Go to... |
|-------------|----------|
| Understand Π/FY/GLUE principles | primary-anchor.md §0 |
| See the 3-step algorithm | primary-anchor.md §3 |
| Check what's in global menu P | spec.md lines 19-27 |
| Understand Φ signature features | spec.md lines 32-40 or primary-anchor.md §2.2 |
| See action set A | spec.md lines 43-49 |
| Look at pseudocode structure | implementation_plan.md |
| Resolve confusion after reading anchors | fundamental_decisions.md |
| Understand "should I loop all P in Step 2?" | fundamental_decisions.md Decision 1 |
| Know what's locked vs empirical | fundamental_decisions.md Decision 3 |
| Find out what's really blocking us | fundamental_decisions.md Decision 4 |
| **Know what to implement next** | **phasewise_implementation_plan.md** |
| **See implementation phases & checkboxes** | **phasewise_implementation_plan.md lines 87-856** |
| **Understand the Golden Rules (NO STUBS)** | **phasewise_implementation_plan.md lines 28-84** |
| **Check expected coverage progression** | **phasewise_implementation_plan.md lines 890-945** |
| **Find critical complexity zones** | **This file → phasewise_implementation_plan.md section** |
| **Estimate code size for a phase** | **This file → phasewise_implementation_plan.md section** |
| **Create context pack for agent** | **phasewise_implementation_plan.md (specific phase)** |
| **See folder structure & Phase 1 details** | **architecture.md** |
| Understand Kaggle submission requirements | submission_readiness.md |
| Know how to format submission.json | submission_readiness.md "Output Format" |
| Handle 2-attempt requirement | submission_readiness.md "Two-Attempt Challenge" |
| Deal with UNSAT tasks in submission | submission_readiness.md "UNSAT Handling" |
| Resolve conflicts/ambiguities | This file → "Resolved Decisions" or fundamental_decisions.md |
| Check repo structure | This file → "Repository Structure" section or architecture.md |
| **Check Step-1 coverage baselines** | **reports/coverage/** (ARC-1: 5.9%, ARC-2: 1.8%) |
| **Measure coverage on dataset** | **tests/measure_coverage.py** |

---

**Last updated:** 2025-10-25 (Phase 3 complete, Phase 4 ready)
**Maintainers:** Update this file when adding new anchors, modules, or resolving ambiguities.
