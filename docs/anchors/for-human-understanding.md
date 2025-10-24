Core Philosophy

  This is a pure mathematical decision procedure - not ML, not search heuristics. Each ARC task gets either:
  - PASS: An exact program that deterministically solves it
  - UNSAT: A finite proof that no program in the current algebra can solve it

  The Three Laws (Π / FY / GLUE)

  Π (Present - Canonicalization):
  - Remove "meaningless" variations (rotations, flips, transpose, local color permutations)
  - Idempotent: applying twice does nothing (Π² = Π)
  - Creates ONE canonical view before doing any work

  FY (Exact Balance - No Approximations):
  - Accept ONLY if the program reproduces EVERY training pair bit-for-bit with one fixed parameter set
  - No scores, no "close enough", no per-example tweaking
  - Pure equality checking

  GLUE (Lawful Composition - Part-based solving):
  - If solving by parts (different pixel classes), stitching them must equal solving all-at-once
  - Disjoint masks guarantee this - no pixel belongs to multiple classes
  - Any overlap or leftover = reject

  The Finite Algebra

  P (Global Transformations) - finite menu:
  - Geometry: rotations, flips, color mappings
  - Scaling: pixel replication, block downsampling
  - Non-uniform partitions (NPS): row/col bands based on content changes
  - Tiling, permutations, object movements, symmetry completion

  Φ (Signature for Classification) - finite Boolean basis:
  - Index predicates: parity, row/col modulo
  - Non-uniform bands from NPS
  - Local content: is_color, touching(color)
  - Component structure: connected components
  - Patch keys: local 5×5, 7×7, 9×9 shapes (canonicalized)

  Key: Φ uses ONLY input features, never target-dependent

  A (Actions) - finite deterministic set:
  - Local: set_color, mirror_h/v, keep_nonzero, identity
  - LUT: patch-based lookup tables
  - Optional constructive: draw_line, draw_box, object_lattice_fill

  The Three-Step Algorithm

  Step 1 - Global Only:
  Try each P in the menu. Learn parameters on first training pair, verify exact equality on ALL pairs. If match → emit solution.

  Step 2 - Global + Local (P + Φ/GLUE):
  - Apply P to get intermediate grids
  - Compute residual Δ = Y ⊕ P(X) (what still needs fixing)
  - Partition residual pixels into Φ-classes (based on P(X) features, not Y!)
  - For each class: find ONE action that fixes all pixels in that class across ALL training pairs
  - Stitch classes together (GLUE), verify exact equality
  - If match → emit solution

  Step 3 - Whole-grid LUT:
  Try patch-based lookup tables (radius 2,3,4) directly on X→Y. If match → emit solution.

  Else → UNSAT: Return which class failed and why.

  Key Insights I See

  1. Finite and Deterministic: Everything terminates, no randomness, no floating point
  2. Receipts-Only: Every decision is auditable with exact reasons
  3. Compositional: Can combine global transform P with local class-based edits
  4. No Search Over Masks: Φ defines the finest partition once; valid masks are unions of classes
  5. Finite Corpus Completeness: For a finite ARC dataset, adding finitely many missing constructors eventually eliminates all UNSATs