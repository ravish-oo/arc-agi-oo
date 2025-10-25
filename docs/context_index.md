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
- Training challenges JSON
- Evaluation set
- Test set

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
- `kaggle_submission_readiness.md.md` — Kaggle submission requirements and wrapper strategy
- `arc-agi-kaggle-docs.md` — Competition rules and dataset description

---

## Future Sections (To Be Added)

### `/src/` (when created)
Implementation modules:
- Canonicalization (Π)
- Global families (P)
- Signature builders (Φ)
- Action inference (A)
- Solver driver
- Receipt generation

### `/tests/` (when created)
Validation tests per primary-anchor:232-238:
- Π idempotence tests
- Family fit tests
- Φ stability tests
- GLUE stitching tests
- LUT conflict tests
- Determinism tests

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
| Understand Kaggle submission requirements | submission_readiness.md |
| Know how to format submission.json | submission_readiness.md "Output Format" |
| Handle 2-attempt requirement | submission_readiness.md "Two-Attempt Challenge" |
| Deal with UNSAT tasks in submission | submission_readiness.md "UNSAT Handling" |
| Resolve conflicts/ambiguities | This file → "Resolved Decisions" or fundamental_decisions.md |
| Check repo structure | This file → "Repository Structure" section |

---

**Last updated:** 2025-10-24
**Maintainers:** Update this file when adding new anchors, modules, or resolving ambiguities.
