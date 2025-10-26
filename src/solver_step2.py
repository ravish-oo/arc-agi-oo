"""
Step-2 Solver: Candidate Builder for Global Transform P + Φ/GLUE

Provides candidate building for Step-2 solving (P + Φ/GLUE mode).

Functions:
    build_candidate_for_P: Evaluate ONE P on ONE task, return candidate dict or None
    solve_step2: Main Step-2 entry point (enumerate ALL P, MDL select, apply to tests)
    _enumerate_P_registry: Helper to build P menu (Identity + 16 families)
    _apply_candidate_to_tests: Apply chosen candidate to test inputs via Φ matching
"""

from src.utils import dims, copy_grid
from src.glue import build_phi_partition, verify_stitched_equality, _build_signature
from src.action_inference import infer_action_for_class
from src.mdl_selection import compute_mdl
from src.signature_builders import phi_signature_tables, Schema
from src.solver_step1 import GLOBAL_FAMILIES
from src.canvas import IdentityColor
from src.task_color_canon import TaskColorCanon
from collections import defaultdict


def build_candidate_for_P(
    task: dict,
    p_desc: dict,
    apply_p_fn: "callable[[list[list[int]]], list[list[int]]]",
    schema: str = 'S3'
) -> dict | None:
    """
    Evaluate ONE global transform P with ONE schema on ONE task for Step-2 solving.

    Schema Lattice (MDL Minimality):
    - S0: Base features only (no patchkeys) - COARSEST, minimal discrimination
    - S1: S0 + patchkey_r2 (5×5 local context)
    - S2: S0 + patchkey_r3 (7×7 local context)
    - S3: S0 + patchkey_r4 (9×9 local context) - FINEST, maximal discrimination

    Args:
        task: {"train": [(X, Y), ...], "test": [...], ...} (extra keys ignored)
        p_desc: {"name": str, "index": int} describing the global family
        apply_p_fn: Pure transform function (X: grid) -> Xp: grid
        schema: Schema level ('S0', 'S1', 'S2', 'S3') - controls patchkey inclusion

    Returns:
        Canonical candidate dict if all checks pass:
        {
            "P": p_desc,
            "schema": str,  # Schema level used ('S0', 'S1', 'S2', 'S3')
            "classes": {cid: [(train_idx, row, col), ...], ...},
            "actions": {cid: (action_name, param_or_None), ...},
            "mdl": {
                "num_classes": int,
                "num_action_types": int,
                "p_index": int,
                "schema_cost": int,  # Schema cost (0=S0, 1=S1, 2=S2, 3=S3)
                "hash": str
            }
        }

        Returns None if:
        - Shape safety fails (any dims(P(X)) != dims(Y))
        - Empty train set
        - Any class has no admissible action
        - GLUE/FY verification fails

    Raises:
        ValueError: If p_desc missing fields or invalid
        Propagates exceptions from dependencies

    Invariants:
        - Φ.3: Φ features computed ONLY from Xp (never from Y)
        - Shape safety: Hard gate (any mismatch → None)
        - Deterministic: Same inputs → same candidate (incl. MDL hash)
        - GLUE: All reads from frozen Xp, writes to fresh outputs

    Edge Cases:
        - No residuals after P (classes == {}) → valid, mdl with 0 classes
        - Empty train set → return None
        - Any class without admissible action → return None
    """
    # Validate p_desc
    if "name" not in p_desc or "index" not in p_desc:
        raise ValueError("p_desc must contain 'name' and 'index' fields")
    if not isinstance(p_desc["index"], int) or p_desc["index"] < -1:
        raise ValueError("p_desc['index'] must be >= -1 (Identity uses -1, GLOBAL_MENU uses 0+)")

    # Extract train pairs
    train_pairs = task.get("train", [])
    if not train_pairs:
        return None  # Insufficient evidence

    # Step 1: Shape safety check
    # Apply P to all train inputs and check dims match targets
    tr_pairs_afterP = []
    for X, Y in train_pairs:
        try:
            Xp = apply_p_fn(X)
        except Exception:
            # P application failed (e.g., invalid grid)
            return None

        # Check dimensions match
        if dims(Xp) != dims(Y):
            return None  # Shape mismatch → reject this P

        tr_pairs_afterP.append((Xp, Y))

    # Step 2: Build Φ partition on residual pixels
    # (Φ features computed ONLY from Xp, Y used only for residual filtering)
    # Schema determines which patchkeys are included in signatures
    try:
        items, classes = build_phi_partition(tr_pairs_afterP, schema)
    except Exception:
        # Partition build failed (e.g., ragged grids)
        return None

    # Step 3: Infer action for each class (deterministic ascending order)
    actions_by_cid = {}
    for cid in sorted(classes.keys()):
        coords = classes[cid]

        try:
            action = infer_action_for_class(items, coords)
        except Exception:
            # Action inference failed
            return None

        if action is None:
            # No admissible action for this class
            return None

        actions_by_cid[cid] = action

    # Step 4: Verify GLUE/FY (stitched outputs == targets exactly)
    try:
        stitched_ok = verify_stitched_equality(items, classes, actions_by_cid)
    except Exception:
        # GLUE verification failed
        return None

    if not stitched_ok:
        # Stitched outputs don't match targets
        return None

    # Step 5: Compute MDL for candidate ranking
    try:
        mdl_tuple = compute_mdl(p_desc, classes, actions_by_cid)
    except Exception:
        # MDL computation failed
        return None

    num_classes, num_action_types, p_index, stable_hash = mdl_tuple

    # Compute schema cost (0=S0, 1=S1, 2=S2, 3=S3)
    # S0 is cheapest (fewest features), S3 is costliest (most features)
    schema_cost_map = {'S0': 0, 'S1': 1, 'S2': 2, 'S3': 3}
    schema_cost = schema_cost_map.get(schema, 3)  # Default to S3 cost if invalid

    # Step 6: Return canonical candidate dict
    return {
        "P": p_desc,
        "schema": schema,  # Store schema for test-time use
        "classes": classes,
        "actions": actions_by_cid,
        "mdl": {
            "num_classes": num_classes,
            "num_action_types": num_action_types,
            "p_index": p_index,
            "schema_cost": schema_cost,  # Schema cost for MDL ranking
            "hash": stable_hash
        }
    }


def _enumerate_P_registry() -> list[dict]:
    """
    Return deterministic list of ALL P ∈ {Identity} ∪ GLOBAL_MENU.

    Returns:
        List of dicts, each with:
        {
            "name": str,                    # e.g., "Identity", "Isometry", ...
            "index": int,                   # -1 for Identity, 0..15 for families
            "family_class": class or None   # None for Identity, class for families
        }

        Ordering: Identity first (index=-1), then 16 families (index=0..15)
        in GLOBAL_FAMILIES order from solver_step1.py

    Edge cases:
        - Always returns exactly 17 items (1 Identity + 16 families)
        - Deterministic: same every call

    Invariants:
        - index=-1 for Identity only
        - index 0..15 for families in spec order
        - No duplicates
    """
    # Identity first (index=-1)
    registry = [
        {
            "name": "Identity",
            "index": -1,
            "family_class": None
        }
    ]

    # Add 16 families from GLOBAL_FAMILIES (indices 0..15)
    for i, family_class in enumerate(GLOBAL_FAMILIES):
        # Instantiate to get name
        instance = family_class()
        registry.append({
            "name": instance.name,
            "index": i,
            "family_class": family_class
        })

    return registry


def _apply_candidate_to_tests(
    task: dict,
    chosen: dict
) -> list[list[list[int]]]:
    """
    Apply chosen (P, classes, actions) to test inputs via Φ signature matching.

    Algorithm:
        1. Extract test_inputs from task["test"]
        2. Build sig_to_cid map from training data
        3. For each test input X_test:
            a. Apply P: Xp_test = chosen["P"]["apply_fn"](X_test)
            b. Build Φ signatures: feats_test = phi_signature_tables(Xp_test)
            c. For each pixel (r,c) in Xp_test:
                - Build 13-field signature
                - Lookup signature in sig_to_cid map
                - If found: queue (r,c) for that class's action
            d. Stitch: apply actions per class to Xp_test
            e. Return stitched output
        4. Return list of outputs for all tests

    Args:
        task: Task dict with "test": [{"input": grid}, ...]
        chosen: Candidate dict from build_candidate_for_P with added "apply_fn":
            {
                "P": {"name", "index", "apply_fn"},
                "classes": {cid: [(train_idx, r, c), ...], ...},
                "actions": {cid: (action_name, param), ...},
                "mdl": {...}
            }

    Returns:
        List of output grids, one per test input

    Edge cases:
        - Empty test inputs: return []
        - Signature not found in train map: leave pixel unchanged
        - Classes == {}: return [P(X_test) for X_test in tests]

    Invariants:
        - Φ.3: signatures computed on Xp_test (never on unknown test outputs)
        - GLUE: all reads from frozen Xp_test, writes to fresh outputs
        - Same signature → same class → same action
    """
    # Extract test inputs
    test_inputs = [test_entry["input"] for test_entry in task.get("test", [])]
    if not test_inputs:
        return []

    # Get apply_fn, canvas, and schema from chosen candidate
    apply_p_fn = chosen["P"]["apply_fn"]
    canvas = chosen["canvas"]  # Canvas used during training
    schema = chosen["schema"]  # Use same schema as training
    classes = chosen["classes"]
    actions_by_cid = chosen["actions"]

    # Edge case: no classes (Identity with 0 edits)
    if not classes:
        # Apply canvas then P to test inputs
        results = []
        for X_test in test_inputs:
            X_canvas, _ = canvas.apply(X_test)
            results.append(apply_p_fn(X_canvas))
        return results

    # Build sig_to_cid map from training data
    # Re-apply canvas + P to training to get items and rebuild signatures with same schema
    train_pairs = [(pair["input"], pair["output"]) for pair in task["train"]]
    # Apply canvas first, then P (same order as training)
    tr_pairs_canvas = [(canvas.apply(X)[0], Y) for X, Y in train_pairs]
    tr_pairs_afterP = [(apply_p_fn(X_c), Y) for X_c, Y in tr_pairs_canvas]
    items, _ = build_phi_partition(tr_pairs_afterP, schema)  # Use stored schema

    # Build signature → class_id map
    sig_to_cid = {}
    for class_id, coords in classes.items():
        # Pick first coord from this class to get signature
        if coords:
            train_idx, r, c = coords[0]
            sig = _build_signature(items[train_idx]["feats"], r, c, items[train_idx]["Xp"], schema)
            sig_to_cid[sig] = class_id

    # Apply to each test input
    predictions = []
    for X_test in test_inputs:
        # Apply canvas first, then P (same order as training)
        X_test_canvas, _ = canvas.apply(X_test)
        Xp_test = apply_p_fn(X_test_canvas)

        # Build Φ signatures for test with same schema as training
        feats_test = phi_signature_tables(Xp_test, schema)

        # Group pixels by class_id via signature matching
        pixels_by_cid = defaultdict(list)
        R, C = dims(Xp_test)
        for r in range(R):
            for c in range(C):
                # Build signature for this test pixel with same schema
                sig = _build_signature(feats_test, r, c, Xp_test, schema)

                # Lookup in training signature map
                if sig in sig_to_cid:
                    class_id = sig_to_cid[sig]
                    pixels_by_cid[class_id].append((r, c))

        # Stitch: apply actions to Xp_test (frozen-base reads)
        Out = copy_grid(Xp_test)
        for class_id in sorted(pixels_by_cid.keys()):
            action_name, param = actions_by_cid[class_id]
            coords = pixels_by_cid[class_id]

            # Apply action (same logic as stitch_from_classes)
            if action_name == "set_color":
                for r, c in coords:
                    Out[r][c] = param
            elif action_name == "mirror_h":
                for r, c in coords:
                    Out[r][c] = Xp_test[R - 1 - r][c]  # Read from frozen Xp_test
            elif action_name == "mirror_v":
                for r, c in coords:
                    Out[r][c] = Xp_test[r][C - 1 - c]  # Read from frozen Xp_test
            elif action_name == "keep_nonzero":
                for r, c in coords:
                    Out[r][c] = Xp_test[r][c] if Xp_test[r][c] != 0 else 0
            elif action_name == "identity":
                for r, c in coords:
                    Out[r][c] = Xp_test[r][c]

        predictions.append(Out)

    return predictions


def solve_step2(task: dict) -> dict:
    """
    Step-2 solver: Enumerate ALL P, build candidates, pick MDL-best, apply to tests.

    Algorithm (per primary-anchor lines 111-121):
        1. Extract train_pairs and test_inputs from task
        2. Enumerate P_registry = _enumerate_P_registry()
        3. Initialize candidates = []
        4. For each P in P_registry:
            a. Fit family (if applicable) or use Identity
            b. Call build_candidate_for_P(task, P_desc, apply_fn)
            c. If candidate is not None: append to candidates list
            d. Continue to next P (NO early return)
        5. If candidates is empty: return UNSAT
        6. Sort candidates by MDL tuple: (num_classes, num_action_types, p_index, hash)
        7. Pick best = candidates[0] (argmin by lex order)
        8. Apply best to tests: predictions = _apply_candidate_to_tests(task, best)
        9. Return {"status": "PASS", "predictions": predictions, "candidate": best}

    Args:
        task: Task dict with "train" and "test" keys

    Returns:
        Dict with:
        - status: "PASS" or "UNSAT"
        - predictions: list of output grids (if PASS)
        - candidate: chosen candidate dict (if PASS)
        - candidates_tried: list of all passing candidates (for receipts)

    Edge cases:
        - Empty train: return UNSAT
        - No passing candidates: return UNSAT
        - Single passing candidate: choose it (MDL tie-breaking not needed)
        - Multiple candidates with same MDL tuple: hash tie-breaking

    Critical constraints:
        - Enumerate ALL P (no early return when first passes)
        - Collect ALL passing candidates
        - MDL selection: argmin over 4-tuple (num_classes, num_action_types, p_index, hash)
        - Determinism: same task → same candidate chosen
    """
    # Extract train pairs
    train_data = task.get("train", [])
    if not train_data:
        return {
            "status": "UNSAT",
            "reason": "empty_train",
            "candidates_tried": []
        }

    # Stage G0: Fit canvases on raw training inputs
    # Two canvas options: IdentityColor (no-op) and TaskColorCanon (WL-based)
    train_inputs_raw = [pair["input"] for pair in train_data]

    canvas_identity = IdentityColor()
    canvas_identity.fit(train_inputs_raw)  # no-op for Identity

    canvas_taskcanon = TaskColorCanon()
    canvas_taskcanon.fit(train_inputs_raw)

    # Enumerate all P
    P_registry = _enumerate_P_registry()

    # Helper: Feature complexity vector for MDL minimality
    def feature_complexity_vector(schema):
        """
        Return tuple ranking schema by feature complexity (coarse-to-fine).

        MDL principle: Prefer simplest schema that satisfies FY/GLUE constraints.
        Ordering: colorless < colorful, fewer patchkeys < more patchkeys.

        Returns:
            Tuple (use_is_color, patchkey_count) for lexicographic comparison.
            Examples:
                Schema()                              → (0, 0) - COARSEST
                Schema(use_patch_r2=True)             → (0, 1)
                Schema(use_patch_r3=True)             → (0, 1)
                Schema(use_patch_r4=True)             → (0, 1)
                Schema(use_is_color=True)             → (1, 0)
                Schema(use_is_color=True, use_patch_r2=True) → (1, 1) - FINEST
        """
        # Color features: either palette-specific (use_is_color) or canonical (use_canon_color_id)
        color_cost = 1 if (schema.use_is_color or schema.use_canon_color_id) else 0
        patchkey_count = sum([schema.use_patch_r2, schema.use_patch_r3, schema.use_patch_r4])
        return (color_cost, patchkey_count)

    # Schema lattice for MDL minimality (coarse-to-fine, colorless-first)
    # Ordering: prefer canonical color schemas first (cross-palette generalization)
    # Within each group: prefer fewer patchkeys (coarser → finer)
    SCHEMAS = [
        # Canonical color variants (best for palette-permutation tasks)
        Schema(use_canon_color_id=True),                                    # S0_canonical: 7-tuple (spatial + canon_color)
        Schema(use_canon_color_id=True, use_patch_r2=True),                 # S0_canonical + r2
        Schema(use_canon_color_id=True, use_patch_r3=True),                 # S0_canonical + r3
        Schema(use_canon_color_id=True, use_patch_r4=True),                 # S0_canonical + r4
        # Palette-specific color variants (for color-dependent tasks)
        Schema(use_is_color=True),                                          # S0_colorful: 7-tuple (+ is_color)
        Schema(use_is_color=True, use_patch_r2=True),                       # S1_colorful
        Schema(use_is_color=True, use_patch_r3=True),                       # S2_colorful
        Schema(use_is_color=True, use_patch_r4=True),                       # S3_colorful (FINEST)
    ]

    # Collect all passing candidates (all C × P × schemas)
    candidates = []

    for schema in SCHEMAS:
        # Stage C: Determine compatible canvas for this schema
        # Canon schemas use TaskColorCanon, colorful schemas use IdentityColor
        if schema.use_canon_color_id:
            canvas = canvas_taskcanon
        elif schema.use_is_color:
            canvas = canvas_identity
        else:
            # Colorless schema - use Identity (no color transformation needed)
            canvas = canvas_identity

        # Apply canvas to training inputs (NOT outputs!)
        train_pairs_canvas = []
        for pair in train_data:
            X_raw = pair["input"]
            Y_raw = pair["output"]  # Keep output in original colors

            X_canvas, aux_data = canvas.apply(X_raw)
            train_pairs_canvas.append((X_canvas, Y_raw))

        for p_entry in P_registry:
            p_desc = {
                "name": p_entry["name"],
                "index": p_entry["index"]
            }

            # Get apply function (fit family if needed)
            if p_entry["family_class"] is None:
                # Identity: use lambda
                apply_p_fn = lambda X: copy_grid(X)
            else:
                # Family: instantiate and fit
                family_class = p_entry["family_class"]
                instance = family_class()

                # Fit on canvas-transformed training data
                # Convert to dict format for fit()
                train_data_canvas = [{"input": X_c, "output": Y} for X_c, Y in train_pairs_canvas]
                fit_success = instance.fit(train_data_canvas)
                if not fit_success:
                    # Fit failed, skip this P
                    continue

                # Use instance.apply as the transform function
                apply_p_fn = instance.apply

            # Pass canvas-transformed task to build_candidate_for_P
            task_for_candidate = {**task, "train": train_pairs_canvas}

            # Try to build candidate for this (P, schema) combination
            try:
                candidate = build_candidate_for_P(task_for_candidate, p_desc, apply_p_fn, schema)
            except Exception:
                # Candidate build failed, skip this (P, schema)
                continue

            if candidate is not None:
                # Add apply_fn and canvas to candidate for test application
                candidate["P"]["apply_fn"] = apply_p_fn
                candidate["canvas"] = canvas  # Store canvas for test-time application
                candidates.append(candidate)

    # If no candidates pass, return UNSAT
    if not candidates:
        return {
            "status": "UNSAT",
            "reason": "no_passing_candidates",
            "candidates_tried": []
        }

    # Sort candidates by MDL tuple (lexicographic order)
    # MDL tuple with schema lattice + feature complexity:
    # (feature_complexity, num_classes, num_action_types, p_index, stable_hash)
    # Prefer simpler schemas: colorless < colorful, fewer patchkeys < more patchkeys
    def mdl_key(c):
        mdl = c["mdl"]
        schema = c["schema"]
        return (
            feature_complexity_vector(schema),  # Primary: prefer coarser schemas (colorless first)
            mdl["num_classes"],                 # Secondary: fewer classes better
            mdl["num_action_types"],            # Tertiary: fewer action types better
            mdl["p_index"],                     # Quaternary: earlier P in registry
            mdl["hash"]                         # Quinary: deterministic tie-breaking
        )

    candidates.sort(key=mdl_key)

    # Pick best candidate (first after sort)
    best = candidates[0]

    # Apply best candidate to test inputs
    predictions = _apply_candidate_to_tests(task, best)

    # Return result
    return {
        "status": "PASS",
        "predictions": predictions,
        "candidate": best,
        "candidates_tried": candidates
    }


def _cli_main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for Step-2 solver on single task.

    Algorithm:
        1. Parse args: --task <path>, --task-id <id>, [--print-receipt]
        2. Load dataset JSON from <path>
        3. Extract task with id == <id>
        4. Validate task has "train" and "test" keys
        5. Call solve_step2(task)
        6. Build output dict with task_id and predictions
        7. Optionally include receipt if --print-receipt
        8. Print single JSON line to stdout (deterministic)
        9. Return 0

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None)
              Format: ["--task", "path.json", "--task-id", "abc123", "--print-receipt"]

    Returns:
        Exit code:
        - 0: Success (PASS or UNSAT with valid output)
        - 2: File not found or JSON parse error
        - 3: Task ID not found in dataset
        - 4: Malformed task (missing "train" or "test")

    Output (to stdout):
        Single JSON line:
        {"task_id": "abc123", "predictions": [[[1,2],[3,4]]], "receipt": {...}}

        If --print-receipt not set, omit "receipt" key.
        If UNSAT, omit "predictions" key.

        JSON format: sort_keys=True, separators=(',',':')

    Edge cases:
        - Empty train set: solve_step2 returns UNSAT → exit 0 with no predictions
        - Empty test set: predictions = [] → exit 0
        - Invalid grid (ragged rows): solve_step2 handles → exit 0 with UNSAT

    Invariants:
        - Deterministic: same inputs → same outputs (sorted keys)
        - Pure I/O: read file, print once, exit
        - No side effects: no logs, no timestamps, no env access

    Purity:
        - Read-only on task
        - Stdout writes only
    """
    import argparse
    import json
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run Step-2 solver on a single ARC task",
        add_help=True
    )
    parser.add_argument("--task", required=True, help="Path to task dataset JSON")
    parser.add_argument("--task-id", required=True, help="Task ID to solve")
    parser.add_argument("--print-receipt", action="store_true",
                        help="Include receipt in output")

    try:
        if argv is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code else 0

    task_path = args.task
    task_id = args.task_id
    print_receipt = args.print_receipt

    # Load dataset
    try:
        with open(task_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(json.dumps({"error": "file_not_found"}), file=sys.stderr)
        return 2
    except json.JSONDecodeError:
        print(json.dumps({"error": "json_parse_error"}), file=sys.stderr)
        return 2

    # Extract task by ID
    if task_id not in dataset:
        print(json.dumps({"error": "task_id_not_found"}), file=sys.stderr)
        return 3

    task = dataset[task_id]

    # Validate task structure
    if "train" not in task or "test" not in task:
        print(json.dumps({"error": "malformed_task"}), file=sys.stderr)
        return 4

    # Add task ID to task dict for receipts
    task["id"] = task_id

    # Call solve_step2
    try:
        result = solve_step2(task)
    except Exception as e:
        # Solver exception → treat as UNSAT (defensive)
        result = {"status": "UNSAT", "reason": "solver_exception"}

    # Build output dict
    output = {"task_id": task_id}

    # Add predictions if PASS
    if result["status"] == "PASS" and "predictions" in result:
        output["predictions"] = result["predictions"]

    # Add receipt if requested
    if print_receipt:
        # Build receipt from result
        if result["status"] == "PASS":
            # PASS receipt: Φ-mode with chosen candidate
            from src.receipts import generate_receipt_phi

            candidate = result["candidate"]
            candidates_tried = result.get("candidates_tried", [])

            receipt = generate_receipt_phi(
                candidate["P"],
                candidate["classes"],
                candidate["actions"],
                candidates_tried
            )
            output["receipt"] = receipt
        else:
            # UNSAT receipt
            from src.receipts import generate_receipt_unsat

            reason = result.get("reason", "no_passing_candidates")
            task_meta = {
                "task_id": task_id,
                "train_n": len(task.get("train", [])),
                "test_n": len(task.get("test", []))
            }
            receipt = generate_receipt_unsat(reason, task_meta)
            output["receipt"] = receipt

    # Print output as single JSON line (deterministic)
    print(json.dumps(output, sort_keys=True, separators=(',', ':')))

    return 0


if __name__ == "__main__":
    import sys
    exit_code = _cli_main()
    sys.exit(exit_code)
