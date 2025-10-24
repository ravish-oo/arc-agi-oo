Below is the ANCHOR DOCUMENT. Hand this to Claude Code as the single source of truth.
It is deliberately complete, deterministic, receipts‑only, and independent of any extra context.
If Claude ever drifts, bring it back to this document.

⸻

ARC—Pure Math Decider (Π / FY / GLUE)

Anchor Document — Mathematics, Theorems, Exact Algorithm, Code Contracts, Receipts

Goal: For each ARC task in a finite corpus (e.g., the 1000 training+test tasks), decide it by pure mathematics: either
	•	PASS: output a training‑exact normal form (a short exact program) that deterministically maps test inputs to outputs, or
	•	UNSAT: output a finite witness proving no program from the fixed finite algebra can realize the mapping.

No heuristics, no thresholds, no randomness. Everything is receipts.

⚠️ NOTE ON COMPLETENESS: The algebra (P,Φ,A) defined below is the starting point based on empirical analysis
of ARC training tasks. Completeness is achieved ITERATIVELY: run on training corpus → collect UNSAT witnesses
→ add missing finite constructors (still pure math) → iterate until all tasks are PASS. The finite corpus
guarantees finite additions lead to closure (see §5 Theorem T5 and §10 FAQ). This remains a pure math decider
throughout; no heuristics or approximations are ever introduced.

⸻

0) Universe‑Functioning Principles (the only rules we use)
	•	Π — Present (idempotent canonicalization).
Remove minted differences: rotations/flips/transpose and local palette permutations that don’t change content. Π is idempotent: Π²=Π.
Result: one canonical “view” before we do any “write.”
	•	FY — Exact balance (acceptance by equality only).
A candidate program is accepted iff it reproduces every training pair bit‑for‑bit with one parameter set.
No scores, no “almost,” no thresholds, no per‑example parameters.
	•	GLUE — Lawful composition (sewing without seams).
If we solve by parts (bands/tiles/classes), the stitched result must equal the one‑shot result.
Disjoint masks guarantee exact sewing; any leftover pixel means reject.

These three rules convert ARC from “pattern hunting” into a finite algebraic decision.

⸻

1) Objects, Notation, and Canonicalization
	•	A grid G is an H\times W array of small nonnegative integers (“colors”).
	•	A task contains pairs (X_i,Y_i) (train) and inputs T_j (test).
	•	Π canonicalization:
	1.	Try all dihedral isometries (D8) plus transpose (when rectangular) and pick the lexicographically minimal image.
	2.	Inside local patches (defined below), colors are normalized by order‑of‑first‑appearance (OFA) so that identical shapes with different palettes get the same signature.

Invariant Π.1: Applying Π twice does nothing (Π²=Π).
Invariant Π.2: Any algorithm may freely apply Π to inputs and intermediates; no “work” is done by Π.

⸻

2) The Finite Algebra (P,\Phi,A)

The solver only uses a fixed, finite set of generators. These reflect ARC’s actual constructions and are sufficient for the finite corpus.

2.1 Global Exact Menu P  (closed, finite)
	•	Geometry/Palette: Isometry (D8/transpose), ColorMap, IsoColorMap (isometry then per‑color LUT).
	•	Uniform scaling: PixelReplicate (kH×kW), BlockDown (one of {center, majority, min, max, first_nonzero}).
	•	Non‑uniform partitions:
	•	NPSDown: row/col band edges at content changes; aggregate each band pair by one global operator (same set as BlockDown).
	•	NPSUp: learn band widths in output and exact column/row replication maps from train X\to Y; reuse for test.
	•	Tiling & products: ParityTile (m×n with h/v/hv parity flips), BlockPermutation (reorders equal tiles), BlockSubstitution (per‑color k\times k glyph).
	•	Row/Col transforms: RowPermutation, ColPermutation, SortRowsLex, SortColsLex.
	•	Object logistics: CopyMoveAllComponents (same color components translate by a single (\Delta r,\Delta c) per color).
	•	Symmetry completion: MirrorCompleteHV (H/V), MirrorCompleteDiag (main/anti).

Each P is exact and deterministic. Parameters are integers (sizes, repeats, indices), learned once and held fixed.

2.2 Canonical Signature \Phi (finite Boolean basis → disjoint classes)

We never search masks. We define a finite Boolean basis once; the residual is partitioned into equivalence classes by this basis. Any valid mask is a union of classes.
	•	Index predicates:
	•	parity: (r+c)\bmod 2\in\{0,1\};
	•	rowmod k, colmod k for k\in\{2,3\}.
	•	Non‑uniform bands:
	•	row_band(i), col_band(j) from NPS change boundaries.
	•	Local content:
	•	is_color(c) at the pixel;
	•	touching(color d) (4‑neigh dilation one step).
	•	Component structure:
	•	component_id table (8‑connected; ties broken by -size, then bbox lex).
	•	Patch keys (local shapes):
	•	patchkey(r) for r\in\{2,3,4\} (5×5, 7×7, 9×9), D8‑canonical + OFA remapping.

Property Φ.1 (Finiteness): Because H,W are corpus‑bounded and r\le4, the number of distinct signatures is finite.
Property Φ.2 (Disjointness): Two distinct signatures define disjoint pixel sets.
Property Φ.3 (Stability): Φ depends only on the input; no target‑dependent features are allowed.

2.3 Deterministic Action Set A (finite)

Each class receives one action, chosen from:
	•	Local: set_color(c), mirror_h, mirror_v, keep_nonzero, identity.
	•	LUT fallback: LUTPatch(r) for r\in\{2,3,4\} using D8+OFA keys; conflict‑free (one output per key).
	•	Object/constructive (if enabled): draw_line(anchors), draw_box(bbox(largest_component)), copy_move(Δr,Δc), object_lattice_fill(Δr,Δc).

Actions are deterministic and exact. There are no thresholds.

⸻

3) Acceptance Logic (the Decider)

For a task with train pairs (X_i,Y_i) and tests T_j:

Step 0 — Π: Canonicalize all inputs X_i (and tests T_j) as needed; inside patches use OFA.

Step 1 — Global P exact
Loop over P\in\{\mathrm{Identity}\}\cup\texttt{GLOBAL\_MENU}:
	•	Learn parameters on the first train pair; verify exact equality on all train pairs.
	•	If PASS, emit test outputs as P(T_j). Stop.

Step 2 — Global+Local (Φ/GLUE) exact (with best-candidate selection)
Collect all passing (P, Φ, A) candidates, then pick the best by minimum description length (MDL):
Loop over all P\in\{\mathrm{Identity}\}\cup\texttt{GLOBAL\_MENU}:
	•	Shape safety: If dims(P(X_i)) ≠ dims(Y_i) for any train pair, skip this P (class actions cannot change dimensions).
	•	Compute residuals \Delta_i = Y_i \oplus P(X_i) (pixelwise "needs change" table).
	•	Build the Φ‑partition on P(X_i) (not on Y_i!) and group only residual pixels into classes.
	•	For each class C: pick the unique action a\in A that reproduces all class pixels across all trains. If none exists → skip this P (cannot satisfy).
	•	Stitch class actions (GLUE) and check bit‑for‑bit equality on all trains (FY).
	•	If PASS: add (P, classes, actions, MDL_cost) to candidates list and continue to next P.

After trying all P: if candidates list is non-empty, pick best by MDL (see tie-breaking rules below). Build signature→class map on training data, apply to test inputs (signatures computed on P(T_j)). Emit outputs; stop.

MDL tie-breaking (deterministic, applied in order):
	1.	Fewest classes edited (|classes| minimized)
	2.	Fewest distinct action types used (|{action_type : action in actions.values()}| minimized)
	3.	Earliest P in GLOBAL_MENU canonical order (lower index wins)
	4.	Stable hash of (P.name, sorted class signatures) as final tie-break

Step 3 — Local whole‑grid LUT
Try LUTPatch(r) for r=2,3,4 directly on (X_i\to Y_i). If PASS, apply to test; stop.

Else → UNSAT
Return a finite witness (first failing class signature + reason), which unambiguously explains the miss.

Termination: All loops are finite; all families have finite parameter sets within bounded grids; Φ has finite classes; MDL selection examines finite candidates.
Determinism: No randomness, no floating point thresholds; only integer arithmetic, equality checks, discrete maps, and deterministic MDL tie-breaking.

⸻

4) Theorems (with proof sketches)

T1 (Π idempotence). Π²=Π by construction (lexicographic minimal representative; OFA mapping recomputes same map on canonicalized patch).
Sketch: canonicalization picks a unique representative; applying it again chooses the same representative.

T2 (Soundness). If the solver returns PASS, the produced program maps every training input to its target exactly.
Sketch: Every acceptance branch is guarded by equality checks on all trains.

T3 (GLUE exactness). If class masks are disjoint, applying class actions and stitching yields the same grid as applying them “one‑shot.”
Sketch: Disjoint supports commute; no overlaps ⇒ no seam terms.

T4 (Termination). For bounded grids and fixed (finite) (P,\Phi,A), the solver halts.
Sketch: Finite menu of P; finite Φ signatures; finite actions; finite train pairs; equality checks terminate.

T5 (Finite‑Corpus Completeness). For a finite ARC corpus: if some task is UNSAT under (P,\Phi,A), the witness is a single Φ‑class that no action can realize or a LUT key conflict. Because the corpus is finite, at most a finite number of preset additions (e.g., enabling draw_line, draw_box, or increasing patch radius to 4) removes all UNSATs.
Sketch: Every UNSAT points to a finite missing constructor or too‑coarse Φ. Each fix is finite and corpus‑bounded; after finitely many, UNSAT vanishes.

⸻

5) Pseudocode (exact; copy into any language)

function SOLVE_TASK(Task):
    Tr = [(X_i, Y_i)] ; Ts = [T_j]

    # Step 0: Present (Π)
    for (X,Y) in Tr: X := Pi(X)
    for T in Ts: T := Pi(T)

    # Step 1: Global exact P
    for P in [Identity] + GLOBAL_MENU:
        if FIT_EXACT(P, Tr): return {program: P, outputs: [P(T) for T in Ts], receipts: R_global(P)}

    # Step 2: Global + Local (Φ/GLUE)
    for P in [Identity] + GLOBAL_MENU:
        TrP = [(P(X), Y) for (X,Y) in Tr]
        items, classes = BUILD_PHI_PARTITION(TrP)   # only over residual pixels
        actions = {}
        for C in classes:
            a = INFER_EXACT_ACTION(items, C)       # check all trains on C
            if a == None: return UNSAT_WITNESS(C)  # finite class + reason
            actions[C] = a
        if VERIFY_STITCHED_EQUALITY(items, classes, actions): 
            # map signatures→classes on train; apply to test
            sig2cid = LEARN_SIG_TO_CLASS(items, classes)
            outs = []
            for T in Ts:
                TP = P(T)
                Yhat = APPLY_CLASSES_BY_SIGNATURE(TP, sig2cid, actions)
                outs.append(Yhat)
            return {program: (P, classes→actions), outputs: outs, receipts: R_phi(P, classes, actions)}

    # Step 3: Whole-grid LUT
    for r in {2,3,4}:
        L = FIT_LUT_r(Tr, r)     # D8+OFA; conflict-free keys only
        if L != None:
            return {program: LUT_r, outputs: [APPLY_LUT_r(L, T) for T in Ts], receipts: R_lut(r)}

    return UNSAT_WITNESS(first_failing_reason)

All caps functions are pure, deterministic, and must return receipts.

⸻

6) Code Contracts (what each block must guarantee)
	•	FIT_EXACT(P, Tr): Learns discrete params once on the first pair; returns True iff P reproduces every (X_i\to Y_i) exactly.
	•	BUILD_PHI_PARTITION(TrP):
	•	Input: list of (X’_i=P(X_i), Y_i).
	•	Output: (items, classes) where classes is a map class_id → list of (train_idx, r, c) only for residual pixels (where X’_i\neq Y_i).
	•	Features used are exactly those in §2.2; no target‑dependent features.
	•	INFER_EXACT_ACTION(items, class):
	•	Try finite actions A in fixed order.
	•	Accept the first that matches every train coordinate in the class; else return None.
	•	set_color(k) is tried only if all targets in the class equal the same color k.
	•	VERIFY_STITCHED_EQUALITY: Apply each class action on its class; check equals Y_i for every train.
	•	FIT_LUT_r / APPLY_LUT_r: Keys = D8+OFA canonicalized patches; fit rejects if any key maps to two different outputs on training pixels.

No randomness. No thresholds. Only equality.

⸻

7) Receipts (what to log per task)

Always dump a JSON "receipts" object. Suggested fields:
	•	task_id, mode (global | phi_partition | lut | unsat_*)
	•	solver (e.g., "NPSDown", "Φ-classes+GLUE", "LUTPatch(r=3)")
	•	parameters of P if any (e.g., kH=2,kW=2, band maps, permutations)
	•	phi_features_used: list of feature keys actually referenced to form classes
	•	actions_by_class: {cid: ["op", arg_or_null], ...} for Φ mode
	•	If mode=phi_partition:
	•	mdl_candidates: [{P_name, num_classes, num_actions, mdl_cost}, ...] for all passing P tried in Step 2
	•	chosen_candidate: {P_name, num_classes, num_actions, mdl_cost, reason_chosen}
	•	If UNSAT:
	•	witness: {class_id, reason, class_size, signature_summary}
	•	reason in {no_action_matches_class_on_trains, lut_key_conflict, glue_mismatch}
	•	a few sample coordinates for reproduction

This makes every PASS auditable and every UNSAT actionable.

⸻

8) Minimal Extensions (still finite, still pure math)

If any UNSAT appears, add one of the following small constructors—each ~50 lines—and rerun:
	•	draw_line(anchors): anchors = extrema of largest component (deterministic); draw Bresenham line.
	•	draw_box(bbox(largest_component)): draw the rectangle border.
	•	(Rare) object_lattice_fill(Δr,Δc): replicate a canonical component on a learned lattice.

Each plugs into A and INFER_EXACT_ACTION without changing anything else.

⸻

9) Test Plan (sanity and integrity)
	1.	Π tests: Applying Π twice returns identical bytes; patch OFA maps are consistent.
	2.	Family tests: For each P, create synthetic grids and assert FIT_EXACT passes only when the mapping is true, fails otherwise.
	3.	Φ stability: Changing Y must not alter Φ; only X determines signatures.
	4.	GLUE: For 100 random disjoint masks and deterministic actions, “stitched” equals “one‑shot.”
	5.	LUT conflict: Inject two different targets under same patchkey; ensure FIT_LUT_r rejects.
	6.	Determinism: Two full runs (same inputs) produce byte‑identical predictions and receipts.

⸻

10) FAQ (for future “context drift”)
	•	Q: Why no beams / MDL search?
You don’t need them. The branch “global P” already tries all finite families; the branch “P+Φ” uses all Φ‑classes and finite actions with equality—no ranking.
	•	Q: How do you “learn masks”?
You don’t. Φ defines the finest equivalence classes once; valid masks are unions of classes.
	•	Q: What if a task needs both a global change and a few local edits?
That’s exactly Step‑2: pick a P from the finite menu, then fill the residual by Φ‑classes + actions. Equality on all trains decides it.
	•	Q: What if a task still fails (UNSAT)?
The witness tells you which single class failed and why. Add the one missing constructor (still finite) or increase patch radius to 4, and rerun. Finite corpus ⇒ finitely many such adds ⇒ closure.

⸻

11) One‑paragraph summary (hand to any tool)

We solve ARC by deciding each task in a finite algebra. First, canonicalize (Π). Then try a finite set of exact global maps P; if one matches all training pairs exactly, emit it. Else for each P we partition the residual into disjoint classes using a fixed finite signature \Phi (parity/mod/NPS/color/touching/component/patch keys), pick one deterministic action per class from a finite set A (set_color/mirror/keep_nonzero/identity, plus tiny constructors if enabled), stitch (GLUE), and accept only if the stitched result equals the targets on every train (FY). If no P yields equality, emit a finite witness (first failing class). The corpus is finite, so after finitely many preset additions, UNSAT vanishes and you have 100% completion—pure math, no heuristics, with receipts.

⸻

This is the anchor.
All code, comments, and decisions must implement exactly what’s written here.