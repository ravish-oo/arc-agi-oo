# Fundamental Decisions & Resolved Ambiguities

## Purpose of This Document

**When to read this:** AFTER reading primary-anchor.md, spec.md, and implementation_plan.md.

**What this captures:** The natural questions, confusions, and ambiguities that arise from reading those three documents, along with the step-by-step reasoning to resolve them objectively.

**Why this exists:** Context drift is by design in LLMs. This document ensures you arrive at the EXACT same level of clarity every time, without re-discovering the same answers. It turns "figuring out" into a mechanical process.

**How to use this:**
1. Read the three anchor documents first
2. When confusion arises, come here and find the relevant decision
3. Follow the reasoning step-by-step
4. Arrive at the crystal-clear resolution
5. If you encounter NEW confusions not listed here, ADD them using the same format

---

## Decision 1: Loop Over ALL P in Step 2 (Not Just Identity)

### The Confusion

The implementation_plan.md pseudocode at line 899 says:
```python
# For now, try P = identity in this branch (could loop P if you want)
```

This makes it seem like:
- Looping all P is optional
- Identity alone might be sufficient
- It's just a performance optimization

Meanwhile, primary-anchor:112 explicitly says:
```
Loop again over P∈{Identity}∪GLOBAL_MENU
```

**The natural question:** Should we loop over all P or just try Identity in Step 2?

### The Question Stated Clearly

Is looping over all P in Step 2 a **correctness requirement** (affects which tasks we can solve) or a **performance optimization** (affects how fast we solve)?

### The Reasoning (Step by Step)

**Example task:** "Rotate grid 90° clockwise AND fill missing pixels in corners"

**Step 1 (Global P only):**
- Try Rotation alone → FAIL (corners still missing)
- Try Identity alone → FAIL (not rotated)
- No single global P solves it → move to Step 2

**Step 2 with ONLY Identity + Φ:**
- Apply Identity (do nothing)
- Residual Δ = "needs rotation + corner fills"
- Build Φ classes (local signatures: parity, color, touching, patches)
- Try actions A: set_color, mirror_h, mirror_v, keep_nonzero, identity
- **Problem:** Φ actions are LOCAL and simple - they cannot do global rotation
- Result: UNSAT (no action matches the class needing rotation)

**Step 2 with Rotation + Φ:**
- Apply Rotation first → intermediate grid is rotated
- Residual Δ = "just the missing corners" (rotation is already handled)
- Build Φ classes (local signatures around corners)
- Try actions A: set_color based on touching neighbors
- Actions match all training pairs exactly
- Stitch (GLUE) → exact match
- Result: **PASS**

**Core insight:** The power is in **composition**:
- P (global) handles structural transformations
- Φ+A (local) handles small corrections
- P+Φ together = "big change + small fixes"

If we only try Identity, we're asking Φ to do ALL the work, but Φ is deliberately limited to local reasoning.

### The Resolution

**We MUST loop over all P ∈ {Identity} ∪ GLOBAL_MENU in Step 2.**

This is a **correctness requirement**, not optimization:
- More P attempts = more compositional combinations = more tasks solved
- Fewer P attempts = missing compositional power = more UNSATs

The pseudocode saying "(could loop P if you want)" is underselling it. The spec is clear: loop all P.

### Implementation Impact

```python
# Step 2: Global + Local (Φ/GLUE) exact with best-candidate selection
candidates = []

for P in [Identity] + GLOBAL_MENU:  # ← Loop ALL P, not just Identity
    # Shape safety: class actions cannot change dimensions
    if any(dims(P(X)) != dims(Y) for (X,Y) in trains):
        continue  # Skip this P

    tr_afterP = [(P(X), Y) for (X,Y) in trains]
    items, classes = build_classes(tr_afterP)

    # Try to infer actions for all classes
    actions = {}
    success = True
    for cid, coords in classes.items():
        act = infer_exact_action(items, coords)
        if act is None:
            success = False
            break  # Skip this P
        actions[cid] = act

    if not success:
        continue

    # Verify GLUE/FY
    if verify_stitched_equality(items, classes, actions):
        mdl_cost = compute_mdl(P, classes, actions)  # See Decision 11
        candidates.append((mdl_cost, P, classes, actions))

# Pick best candidate by MDL (see Decision 11 for tie-breaking rules)
if candidates:
    best = min(candidates, key=lambda x: x[0])  # Deterministic MDL ordering
    _, P_best, classes_best, actions_best = best
    # Apply to tests and emit
```

**DO NOT skip this loop.** Without it, we lose compositional solving power and get more UNSATs - not because the algebra is incomplete, but because we're not utilizing it fully.

**DO NOT return on first pass.** Collect all passing candidates and pick the best by MDL (see Decision 11 for details).

---

## Decision 2: Pseudocode is Skeletal (Not Literal Code)

### The Confusion

The implementation_plan.md contains Python code with obvious syntax errors:
- Line 130 and every class: `def _init_(self, name):` (should be `__init__`)
- Line 1017: `if _name_=="_main_":` (should be `__name__`)
- Missing imports, incomplete implementations, placeholder comments

**The natural question:** Is this code broken, or am I supposed to interpret it differently?

### The Question Stated Clearly

Should implementation_plan.md be treated as runnable code or as algorithmic guidance?

### The Reasoning

The document says at the top (after our clarifying note):
- "skeletal pseudocode showing algorithmic structure and intent"
- "Syntax may need adaptation"
- "STRUCTURE and LOGIC are what matter"

ChatGPT Pro generated this as a guide, not as production code. It would be unrealistic to expect a chat LLM to produce 1000+ lines of perfect, runnable Python.

### The Resolution

**implementation_plan.md is PSEUDOCODE - a structural guide showing algorithmic intent.**

- Syntax errors are expected and irrelevant
- Class structures, function flows, and logic are what matter
- When implementing, adapt syntax, fix bugs, but preserve the algorithm
- For conflicts between pseudocode and anchors, defer to primary-anchor.md or spec.md

### Implementation Impact

When writing actual code:
1. Follow the STRUCTURE and ALGORITHM from pseudocode
2. Fix all syntax errors (`__init__`, `__name__`, etc.)
3. Add proper error handling, type hints, tests
4. When pseudocode is incomplete (e.g., "for now try Identity"), consult primary-anchor.md for the complete spec
5. Treat pseudocode as a reference, not gospel

---

## Decision 3: "LOCKED" Spec Means Structure, Not Exact Family Count

### The Confusion

spec.md says at the top:
```
FINAL LOCKED SPEC — No additions. No beams. No heuristics.
These 16 global families cover all size/arrangement changes seen in ARC.
```

This sounds like:
- The algebra is complete and final
- We will never add anything
- 16 families is the exact, proven count

But then primary-anchor:251 and spec:75-76 talk about:
- Discovering UNSAT witnesses
- Adding missing constructors
- Iterating to closure on the finite corpus

**The natural question:** Is the algebra locked and complete, or are we still discovering it?

### The Question Stated Clearly

Does "LOCKED" mean the algebra is mathematically complete, or does it mean something else?

### The Reasoning

Read the spec carefully:
- "These 16 global families cover all size/arrangement changes **seen in ARC**"
- "why this covers the full 1000" (line 69) explains the empirical reasoning

The word "seen" is key. This is an **empirical claim** based on analyzing ARC training tasks, not a mathematical proof.

The methodology is:
1. Analyze training tasks to identify transformation families
2. Implement those families
3. Run solver on all training tasks
4. Collect UNSAT witnesses pointing to what's missing
5. Add missing families (still finite, still deterministic, still pure math)
6. Iterate until 100% coverage

The finite corpus guarantees convergence.

### The Resolution

**"LOCKED" means the STRUCTURE is final, not the exact family count.**

What's locked forever:
- Π/FY/GLUE calculus (the three rules)
- 3-step algorithm (Global P → P+Φ/GLUE → LUT)
- Finite algebra approach (no heuristics, no thresholds, only equality)
- Receipts-only decision procedure

What's empirically validated:
- The 16 global families are the starting point
- They cover what ChatGPT Pro observed in training data
- We may discover we need 20, 25, 30 families
- Each addition is finite, deterministic, receipts-guided
- No heuristics or approximations are EVER added

**Completeness is achieved ITERATIVELY, not assumed upfront.**

### Implementation Impact

**Expectation:** When we run the solver on 400 training tasks, we will likely see some UNSATs.

**Response:**
1. Examine UNSAT witnesses (receipts tell us which class failed and why)
2. Identify the missing transformation family
3. Implement it as a new global family P or action A (staying finite and deterministic)
4. Re-run solver
5. Repeat until UNSAT count reaches zero

This is NOT a failure of the approach - it's the designed methodology for achieving completeness on a finite corpus.

---

## Decision 4: Algebra Completeness is the ONLY Real Unknown

### The Confusion

After reading all three docs, many questions seem to arise:
- Should we loop all P?
- Should we implement draw_line/draw_box upfront?
- Is the step order important?
- Are 16 families enough?
- What about performance in 12 hours?

This creates the impression that there are many open questions and blockers.

### The Question Stated Clearly

What are the ACTUAL unknowns or blockers that could prevent us from solving ARC?

### The Reasoning

Let's categorize the questions:

**NOT unknowns (already specified):**
- Loop all P in Step 2? → YES, spec says so (primary-anchor:112)
- Step order? → Global P → P+Φ → LUT (primary-anchor:106-120)
- draw_line/draw_box? → Add when UNSAT witnesses show need (trivial ~50 lines)
- Performance? → Pure math on 400 tasks fits easily in 12 hours
- Implementation details? → All specified in anchors, pseudocode is a guide

**ONLY ONE real unknown:**
- Are 16 families sufficient, or do we need more?

This is NOT a blocker - it's an **empirical question with a clear resolution process:**
1. Implement the 16 families correctly
2. Run on all training tasks
3. Collect UNSAT witnesses
4. Add missing families (staying finite/deterministic)
5. Iterate until 100% coverage

The finite corpus (400 training tasks) guarantees this converges in finite iterations.

### The Resolution

**There is ONLY ONE unknown: Is the algebra (16 families) sufficient for ARC?**

**Answer: We don't know until we run it. And that's by design.**

Everything else is:
- Already specified in the anchors (just implement it)
- Or trivial implementation decisions (syntax, error handling, etc.)

The work is:
1. Implement the full spec faithfully (all 3 steps, looping all P, correct Φ/GLUE)
2. Run on training data
3. Let UNSAT witnesses guide us to completeness
4. Iterate

**This reduces implementation to a mechanical process:**
- Read spec → implement algorithm → run → analyze witnesses → add missing piece → repeat

No guessing, no heuristics, no "figuring out on the fly" - just disciplined empirical validation.

### Implementation Impact

**Mental model:** We are NOT discovering the algorithm (that's in primary-anchor.md). We are discovering the **minimal sufficient algebra** for the bounded ARC corpus.

**Workflow:**
```
while UNSAT_count > 0:
    1. Run solver on all 400 training tasks
    2. Generate receipts for all UNSAT tasks
    3. Analyze receipts: what transformation is missing?
    4. Implement that family (new global P or action A)
    5. Verify it's finite, deterministic, receipts-only
    6. Re-run solver
```

**Timeline expectation:**
- First run: likely 50-150 UNSAT (empirical guess)
- Add 3-5 missing families based on witness analysis
- Second run: likely 10-50 UNSAT
- Add 1-2 more families
- Third run: close to 0 UNSAT
- Final validation on evaluation set

This is a **disciplined empirical science**, not exploration.

---

## Decision 5: Source of Truth Hierarchy

### The Confusion

All three anchor documents describe the same system, but with different levels of detail and sometimes seemingly contradictory statements.

When implementation_plan.md says "for now try Identity" but primary-anchor says "loop all P", which do we follow?

### The Question Stated Clearly

When conflicts arise between documents, which one is authoritative?

### The Resolution

**Hierarchy (highest to lowest authority):**

1. **primary-anchor.md** — Mathematical foundation, theorems, algorithm structure
   - Π/FY/GLUE principles (§0)
   - Algorithm specification (§3)
   - Theorems and proof sketches (§4)
   - Code contracts (§6)

2. **spec.md** — Locked components and final algebra
   - The three rules (locked forever)
   - 16 global families P
   - Φ signature features
   - Action set A
   - Decision procedure (simplified)

3. **implementation_plan.md** — Pseudocode guide
   - Algorithmic structure and intent
   - Example implementations
   - NOT literal code, NOT authoritative on conflicts

**Resolution process:**
- If pseudocode conflicts with primary-anchor → defer to primary-anchor
- If spec seems incomplete → consult primary-anchor for details
- If primary-anchor and spec conflict → primary-anchor wins (it's the bedrock)

**In practice:** primary-anchor and spec are consistent. The only "conflicts" are between pseudocode (incomplete guide) and the two anchors.

### Implementation Impact

When writing code:
1. Start with primary-anchor.md to understand WHAT to implement
2. Consult spec.md for the locked components (16 families, Φ features, actions)
3. Use implementation_plan.md for HOW (structure, helper functions, flow)
4. When in doubt, go back to primary-anchor.md

Never let pseudocode override the mathematical specification in primary-anchor.

---

## Decision 6: Step Order Matters (Global → Compositional → Local)

### The Confusion

implementation_plan.md has this step order:
1. Try global P families (lines 883-888)
2. Try local LUT fallback (lines 891-896)
3. Try Φ-partition with Identity (lines 899-983)

But primary-anchor:106-120 specifies:
1. Global P exact
2. P + Φ/GLUE (compositional)
3. Whole-grid LUT

**The natural question:** Does step order matter, or can we try them in any sequence?

### The Reasoning

The order is designed to go from **most structured to least structured:**

**Step 1 (Global P):** Tries pure structural transformations
- Fast (just parameter fitting on first train pair + verification)
- Covers tasks that are purely geometric/color/permutation changes
- No residual computation needed

**Step 2 (P + Φ):** Tries compositional (structural + local corrections)
- More expensive (builds Φ partition, infers actions per class)
- Covers tasks needing "big change + small fixes"
- Leverages compositional power

**Step 3 (LUT):** Tries pure local pattern matching
- Most expensive (builds lookup tables for all patch keys)
- Covers tasks that are purely local rewrites
- No global structure assumed

**Why this order is better:**
- If a task is solvable by pure global P, we find it fast (Step 1)
- If it needs composition, we find it in Step 2
- LUT is the fallback when structure is hard to characterize

**Why pseudocode's order (Global → LUT → Φ) is suboptimal:**
- Skips the compositional power of Step 2 before trying LUT
- LUT before Φ means we miss "global transform + local fix" solutions

### The Resolution

**Follow primary-anchor's step order: Global P → P+Φ/GLUE → LUT**

This is the correct algorithmic flow. The pseudocode's order is a bug.

### Implementation Impact

```python
def solve_task(task_id, task):
    trains = [(ex["input"], ex["output"]) for ex in task.get("train",[])]
    tests = [ex["input"] for ex in task.get("test",[])]

    # Step 1: Global P exact
    for P in GLOBAL_MENU:
        if try_fit_global(P, trains):
            return predict_and_emit(P, tests, "global")

    # Step 2: P + Φ/GLUE (compositional)
    for P in [Identity] + GLOBAL_MENU:
        if try_fit_phi_partition(P, trains):
            return predict_and_emit_phi(P, tests, "phi_partition")

    # Step 3: LUT fallback
    for r in [2, 3, 4]:
        if try_fit_lut(r, trains):
            return predict_and_emit_lut(r, tests, "lut")

    # Else: UNSAT
    return emit_unsat_witness(trains)
```

Don't follow pseudocode's order - follow primary-anchor.

---

## Decision 7: OFA (Order of First Appearance) is Local Only

### The Confusion

primary-anchor mentions Π canonicalization includes:
- D8/transpose for global grids
- OFA (order-of-first-appearance) "inside local patches"

**The natural question:** When exactly do we apply OFA? To the entire grid or only to patches?

### The Reasoning

**OFA purpose:** Normalize color palettes so that identical shapes with different colors get the same signature.

Example:
- Patch A: red square (colors [1])
- Patch B: blue square (colors [2])
- After OFA: both become [0] (first color seen)
- Same signature despite different palettes

**Where OFA applies:**
1. **Φ signature computation:** When extracting patchkey(r) features for r∈{2,3,4}
   - Extract (2r+1) × (2r+1) patch around each pixel
   - Apply OFA to normalize colors within that patch
   - Canonicalize via D8
   - Use as signature feature

2. **LUTPatch:** When building lookup tables
   - Extract patch around each pixel
   - Apply OFA
   - Canonicalize via D8
   - Use as LUT key

**Where OFA does NOT apply:**
- Global P transformations (they see actual grid colors)
- Input/output grids themselves (colors are preserved)
- Component detection (uses actual colors)
- "touching(color k)" features (uses actual color k)

### The Resolution

**OFA is applied ONLY to local patches when building Φ signatures and LUT keys.**

It's a local normalization trick to make patch-based features color-agnostic. It does NOT alter the global grid or global transformations.

### Implementation Impact

```python
def patchkey_feature(X, r, i, j):
    """Extract D8+OFA canonical patch key at position (i,j) with radius r"""
    H, W = dims(X)
    # Extract patch
    patch = [[X[min(H-1, max(0, i+di))][min(W-1, max(0, j+dj))]
              for dj in range(-r, r+1)]
             for di in range(-r, r+1)]

    # Apply OFA (normalize colors by first appearance)
    patch = ofa_normalize_patch_colors(patch)

    # Canonicalize via D8 (lexicographic minimum)
    return canonical_d8_patch(patch)
```

This is ONLY used inside Φ signature computation and LUT key building. The global grid X remains unchanged.

---

## Decision 8: Φ Partition Uses Input Features Only (Φ.3 Stability)

### The Confusion

In Step 2, we:
1. Compute residual Δ = Y ⊕ P(X) (comparing target Y to transformed input P(X))
2. Build Φ-partition on P(X)
3. Infer actions per class that transform P(X) to Y

**The natural question:** Can Φ features use information from Y (the target), or only from P(X) (the input)?

### The Reasoning

primary-anchor:81-82 states:
```
Property Φ.3 (Stability): Φ depends only on the input;
no target-dependent features are allowed.
```

**Why this matters:**

If Φ could peek at Y:
- During training: we could create signatures like "is_target_color(c)" that directly encode the answer
- During test: we don't have Y, so we can't compute those features
- The test-time signatures would be different from train-time signatures
- The class mapping would break

**The discipline:**
- Φ features are computed ONLY from P(X) (the transformed input)
- Actions are inferred by checking: "what action on this class of P(X) pixels produces the Y values?"
- At test time, we compute the SAME Φ features on P(T) and apply the learned actions

**What Φ CAN use (all computed from P(X)):**
- Index predicates: parity, rowmod, colmod
- NPS bands: row_band, col_band from content change boundaries
- Pixel colors: is_color(c), touching(color d)
- Component IDs: from 8-connected components
- Patch keys: D8+OFA canonical patches

**What Φ CANNOT use:**
- Target colors from Y
- Differences between X and Y
- "Distance to target" metrics
- Any Y-dependent feature

### The Resolution

**Φ signatures are computed PURELY from P(X), never from Y.**

The residual Δ tells us WHICH pixels need fixing, but the Φ partition uses ONLY input features.

This guarantees:
- Train-time and test-time signatures are computed identically
- Class mapping transfers from train to test
- The approach remains deterministic and reproducible

### Implementation Impact

```python
def build_phi_partition(tr_pairs_afterP):
    """Build Φ-classes over residual pixels using INPUT features only"""
    items = []
    for Xp, Y in tr_pairs_afterP:
        feats = phi_signature_tables(Xp)  # ← Computed from Xp, NOT from Y
        R = residual(Xp, Y)               # ← Residual tells us WHERE, not WHAT
        items.append((Xp, Y, feats, R))

    # Classes are formed by signature computed from Xp features
    # Actions are inferred by checking what transforms Xp pixels to Y pixels
    # But the signature itself uses ONLY Xp
```

At test time:
```python
def apply_phi_to_test(Xt, sig_to_cid, actions_by_cid):
    feats_t = phi_signature_tables(Xt)  # ← Same features, no Y available
    # ... apply actions based on signatures
```

Never use Y-dependent features in Φ. This is a hard constraint.

---

## Decision 9: Constructive Actions (draw_line, draw_box) - Add When Needed

### The Confusion

spec.md lists under "Deterministic actions (locked)":
```
Constructive (locked): draw_line(anchors), draw_box(bbox(largest_component))
```

But implementation_plan.md only has:
```python
ACTIONS = ("set_color","mirror_h","mirror_v","keep_nonzero","identity")
```

implementation_plan:1033 says: "If you want, I can supply drop‑in..."

**The natural question:** Are draw_line and draw_box mandatory, or optional?

### The Reasoning

**Empirical reality:** We don't know if these are needed until we run the solver.

**If they're needed:**
- UNSAT witnesses will show tasks where local actions (set_color, mirror, etc.) cannot solve a class
- Receipts will show: "class needs line drawing between component extrema"
- We implement draw_line (~50 lines of Bresenham algorithm)
- Plug it into the action inference loop
- Re-run solver

**If they're not needed:**
- The 16 global families + simple local actions cover all training tasks
- We achieve 100% coverage without constructive actions
- No need to implement them

**Why spec says "locked":**
- It means IF we add them, they're still finite, deterministic, and pure math
- NOT that we must add them upfront before knowing they're needed

### The Resolution

**Constructive actions are pre-approved but NOT mandatory upfront.**

Start with the simple action set:
- set_color
- mirror_h, mirror_v
- keep_nonzero
- identity

If UNSAT witnesses show tasks needing:
- Line drawing → implement draw_line
- Box drawing → implement draw_box
- Object replication → implement object_lattice_fill

Each addition is ~50 lines, deterministic, trivial to plug in.

**Do NOT implement them preemptively.** Let the data tell us what's needed.

### Implementation Impact

**Initial action set:**
```python
ACTIONS = ("set_color", "mirror_h", "mirror_v", "keep_nonzero", "identity")
```

**After discovering UNSAT needing line drawing:**
```python
ACTIONS = ("set_color", "mirror_h", "mirror_v", "keep_nonzero", "identity", "draw_line")

def infer_action_for_class(items, coords):
    # ... existing logic for simple actions

    # Try draw_line if simple actions fail
    if can_draw_line_action_solve(items, coords):
        return ("draw_line", compute_anchors(items, coords))

    return None  # UNSAT if nothing works
```

Add constructive actions incrementally, guided by UNSAT witnesses.

---

## Decision 10: Component Connectivity (8-connected, may need 4-connected)

### The Confusion

implementation_plan.md uses 8-connected components (lines 100-107):
```python
for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):  # 4-neighbors for flood fill
```

Wait, that's 4-connected in the flood fill! Let me check...

Actually, looking at the neighbors: (1,0),(-1,0),(0,1),(0,-1) are the 4 cardinal directions.
For 8-connected, we'd need diagonals: (1,1),(1,-1),(-1,1),(-1,-1) as well.

**The natural question:** Are components 4-connected or 8-connected, and does it matter?

### The Reasoning

Actually, I need to re-read the pseudocode carefully:

Line 100-107 shows neighbors as: `((1,0),(-1,0),(0,1),(0,-1))`

That's 4-neighbors, meaning **4-connected components**.

But primary-anchor:75 says:
```
Component structure: component_id table (8-connected; ties broken by -size, then bbox lex)
```

There's an inconsistency: primary-anchor says 8-connected, pseudocode implements 4-connected.

**Why this matters:**
- 4-connected: pixels are neighbors only if they share an edge (N/S/E/W)
- 8-connected: pixels are neighbors if they share an edge OR corner (includes diagonals)

Different connectivity gives different components, which affects:
- component_id features in Φ
- CopyMoveAllComponents transformations
- Potentially task solvability

### The Resolution

**Follow primary-anchor: use 8-connected components.**

The pseudocode has a bug. We need:
```python
# 8-connected: cardinal + diagonal
neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
```

**If this proves insufficient:** Some tasks might need 4-connected. We'll discover this via UNSAT witnesses and can add a parameter or separate feature for 4-connected components.

### Implementation Impact

```python
def components_by_color(g):
    h, w = dims(g)
    seen = [[False]*w for _ in range(h)]
    groups = defaultdict(list)

    for r in range(h):
        for c in range(w):
            col = g[r][c]
            if col == 0 or seen[r][c]: continue

            q = deque([(r,c)])
            seen[r][c] = True
            comp = [(r,c)]

            while q:
                rr, cc = q.popleft()
                # 8-connected neighbors (cardinal + diagonal)
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    r2, c2 = rr+dr, cc+dc
                    if 0<=r2<h and 0<=c2<w and not seen[r2][c2] and g[r2][c2]==col:
                        seen[r2][c2] = True
                        q.append((r2,c2))
                        comp.append((r2,c2))

            groups[col].append(comp)

    return groups
```

Follow primary-anchor's specification: 8-connected.

---

## Decision 11: MDL Cost and Best-Candidate Selection in Step 2

### The Confusion

After understanding that we must loop over ALL P in Step 2 (Decision 1), a new question arises:

**What if multiple P pass?**

For example:
- ColorMap + Φ passes (5 classes edited)
- Isometry + Φ passes (3 classes edited)
- NPSDown + Φ passes (3 classes edited, but different actions)

Which one do we return? The first one we tried? All of them? The "best" one by some criterion?

The anchor docs (before MDL addition) said "If PASS... emit outputs; stop" which could be interpreted as "return on first pass."

But the planner agent's authoritative answer says: **collect all passing candidates, then pick the best by MDL (minimum description length).**

### The Question Stated Clearly

When multiple P successfully pass in Step 2 (all satisfy FY/GLUE on training data), which one should we select and emit?

### The Reasoning (Step by Step)

**Why not return on first pass?**

The order we try P in GLOBAL_MENU is somewhat arbitrary (fixed for determinism, but not semantically meaningful). Returning the first passing P means:
- Isometry at index 0 wins over ColorMap at index 1, even if ColorMap needs fewer edits
- Implementation detail (menu order) affects which solution we return
- Less elegant from a "pure math" perspective

**Why MDL (Minimum Description Length)?**

MDL is a principle from algorithmic information theory: **prefer the simplest explanation** (shortest program) that fits the data.

For ARC:
- A solution with fewer classes edited is simpler (less residual work)
- A solution using fewer distinct action types is simpler (less variety in operations)
- This aligns with Occam's Razor: prefer the minimal sufficient transformation

**Why is this a correctness issue (not just aesthetic)?**

Determinism and receipts require:
- Two runs on same input produce identical output
- Receipts explain WHY we chose solution X over solution Y
- No arbitrary choices

Without explicit tie-breaking, we'd have non-determinism when multiple P pass.

**The MDL tie-breaking hierarchy:**

The planner agent specifies a 4-level deterministic rule:

1. **Fewest classes edited** (|classes| minimized)
   - Fewer classes = simpler residual = closer to pure global transform

2. **Fewest distinct action types used** (count unique action names)
   - Using only set_color is simpler than using set_color + mirror_h + draw_line

3. **Earliest P in GLOBAL_MENU canonical order** (lower index wins)
   - When two solutions have same complexity, use a stable tiebreak
   - Menu order is fixed and deterministic

4. **Stable hash of (P.name, sorted class signatures)** (final tiebreak)
   - If still tied (rare), use a deterministic hash function
   - Ensures absolute determinism even in edge cases

### The Resolution

**In Step 2, we MUST:**
1. Loop over ALL P (not just Identity)
2. For each P, check shape safety, try Φ/GLUE, verify FY
3. Collect ALL passing candidates: (P, classes, actions, MDL_cost)
4. Pick the BEST candidate by MDL tie-breaking (4-level hierarchy)
5. Apply the chosen best to test inputs and emit

**This is NOT optional.** It's required for:
- Determinism (same input → same output)
- Auditability (receipts explain why we chose this solution)
- Elegance (prefer simplest program)

### Implementation Impact

**MDL cost function:**

```python
def compute_mdl(P, classes, actions):
    """
    Compute MDL cost tuple for tie-breaking.
    Returns tuple that sorts lexicographically to prefer simpler programs.
    """
    num_classes = len(classes)
    num_action_types = len(set(action[0] for action in actions.values()))  # Count unique action names
    p_index = GLOBAL_MENU.index(type(P)) if type(P) in [type(p) for p in GLOBAL_MENU] else -1

    # For hash: create deterministic string representation
    class_sigs = sorted([str(sig) for sig in get_class_signatures(P, classes)])
    hash_val = hash((P.name, tuple(class_sigs)))

    # Return tuple: sorts lexicographically, lower is better
    return (num_classes, num_action_types, p_index, hash_val)
```

**Step 2 with MDL selection:**

```python
# Step 2: Φ-partition with MDL best-candidate selection
candidates = []

for P in [Identity] + GLOBAL_MENU:
    # Shape safety
    if any(dims(P(X)) != dims(Y) for (X,Y) in trains):
        continue

    # Try Φ/GLUE
    tr_afterP = [(P(X), Y) for (X,Y) in trains]
    items, classes = build_classes(tr_afterP)

    actions = {}
    for cid, coords in classes.items():
        act = infer_exact_action(items, coords)
        if act is None:
            break  # Skip this P
        actions[cid] = act
    else:
        # All classes satisfied, verify GLUE/FY
        if verify_stitched_equality(items, classes, actions):
            mdl_cost = compute_mdl(P, classes, actions)
            candidates.append((mdl_cost, P, classes, actions))

# Pick best by MDL
if candidates:
    candidates.sort(key=lambda x: x[0])  # Lexicographic sort by MDL tuple
    best_cost, P_best, classes_best, actions_best = candidates[0]

    # Log all candidates for auditability
    receipts["mdl_candidates"] = [
        {
            "P_name": P.name,
            "num_classes": len(classes),
            "num_actions": len(set(a[0] for a in actions.values())),
            "mdl_cost": cost
        }
        for cost, P, classes, actions in candidates
    ]

    receipts["chosen_candidate"] = {
        "P_name": P_best.name,
        "num_classes": len(classes_best),
        "num_actions": len(set(a[0] for a in actions_best.values())),
        "mdl_cost": best_cost,
        "reason": "MDL: fewest classes, then fewest action types, then menu order"
    }

    # Apply best to tests and emit
    return apply_and_emit(P_best, classes_best, actions_best, tests)
```

**Shape safety check:**

```python
# Skip P if dimensions mismatch on ANY train pair
if any(dims(P(X)) != dims(Y) for (X,Y) in trains):
    continue  # Class actions (set_color, mirror, etc.) cannot resize grids
```

**Receipts logging:**

Always log:
- `mdl_candidates`: list of all P that passed, with their MDL metrics
- `chosen_candidate`: which one was selected and why
- This makes every decision auditable

**Determinism guarantee:**

The 4-level tie-breaking ensures:
- Same input → same best candidate selected
- No random choices
- Fully reproducible

---

## Summary: The Complete Picture

After reading all three anchor documents and working through these decisions, here's the complete mental model:

### What's Certain (Just Implement It)

1. **Algorithm:** 3 steps as specified in primary-anchor:106-127
   - Step 1: Try global P families (return on first pass)
   - Step 2: Try P + Φ/GLUE (loop ALL P, collect all passing, pick best by MDL)
   - Step 3: Try LUT fallback

2. **Canonicalization (Π):**
   - D8/transpose for global grids
   - OFA for local patches in Φ and LUT only

3. **Acceptance (FY):**
   - Bit-for-bit equality on ALL training pairs
   - ONE parameter set across all pairs
   - No thresholds, no scores

4. **Composition (GLUE):**
   - Disjoint classes guarantee exact stitching
   - No seam errors allowed

5. **Φ Stability:**
   - Use ONLY input features
   - Never peek at target Y

6. **Source of truth:**
   - primary-anchor → spec → implementation_plan
   - Pseudocode is a guide, not gospel

7. **MDL Selection (Step 2 only):**
   - Collect all passing (P, Φ, A) candidates
   - Pick best by 4-level tie-breaking: fewest classes → fewest action types → menu order → hash
   - Log all candidates and chosen one in receipts
   - Guarantees determinism and prefers simplest program

8. **Shape Safety:**
   - In Step 2, skip P if dims(P(X)) ≠ dims(Y) for any train pair
   - Class actions cannot change grid dimensions

### What's Empirical (Discover Through Iteration)

1. **Algebra completeness:**
   - Start with 16 global families
   - Run on all 400 training tasks
   - Collect UNSAT witnesses
   - Add missing families (staying finite/deterministic)
   - Iterate until 100% coverage

2. **Constructive actions:**
   - Start with simple actions (set_color, mirror, keep_nonzero, identity)
   - Add draw_line, draw_box, etc. when UNSAT witnesses show need
   - Each addition is ~50 lines, trivial to integrate

3. **Component connectivity:**
   - Use 8-connected per spec
   - If UNSAT witnesses show need for 4-connected, add it as variant

### The ONLY Real Unknown

**Are 16 families sufficient for the ARC corpus?**

Answer: Run it and find out. The finite corpus guarantees convergence through empirical validation.

### The Work Ahead

**Implementation is now mechanical:**

1. Implement Π (canonicalization)
2. Implement 16 global families P
3. Implement Φ signature builders
4. Implement action inference and GLUE
5. Implement 3-step solver with correct order and loops
6. Run on 400 training tasks
7. Analyze UNSAT witnesses
8. Add missing pieces (if any)
9. Repeat steps 6-8 until 100% coverage
10. Validate on evaluation set

**No guessing. No heuristics. Just disciplined empirical science.**

---

## How to Add New Decisions

When you encounter a NEW confusion or ambiguity:

1. **State the confusion:** What naturally confused you after reading the anchors?
2. **State the question clearly:** Make it specific and answerable
3. **Walk through the reasoning:** Show the step-by-step thought process
4. **Arrive at the resolution:** Crystal-clear, unambiguous answer
5. **Document implementation impact:** What does this mean for code?

Use this exact format to maintain consistency. Future versions of yourself will thank you.

---

**Last updated:** 2025-10-24
**Purpose:** Eliminate context drift by capturing the reasoning journey from confusion to clarity.
