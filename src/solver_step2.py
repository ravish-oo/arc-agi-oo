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
from src.signature_builders import phi_signature_tables
from src.solver_step1 import GLOBAL_FAMILIES
from collections import defaultdict


def build_candidate_for_P(
    task: dict,
    p_desc: dict,
    apply_p_fn: "callable[[list[list[int]]], list[list[int]]]"
) -> dict | None:
    """
    Evaluate ONE global transform P on ONE task for Step-2 solving.

    Args:
        task: {"train": [(X, Y), ...], "test": [...], ...} (extra keys ignored)
        p_desc: {"name": str, "index": int} describing the global family
        apply_p_fn: Pure transform function (X: grid) -> Xp: grid

    Returns:
        Canonical candidate dict if all checks pass:
        {
            "P": p_desc,
            "classes": {cid: [(train_idx, row, col), ...], ...},
            "actions": {cid: (action_name, param_or_None), ...},
            "mdl": {
                "num_classes": int,
                "num_action_types": int,
                "p_index": int,
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
    try:
        items, classes = build_phi_partition(tr_pairs_afterP)
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

    # Step 6: Return canonical candidate dict
    return {
        "P": p_desc,
        "classes": classes,
        "actions": actions_by_cid,
        "mdl": {
            "num_classes": num_classes,
            "num_action_types": num_action_types,
            "p_index": p_index,
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

    # Get apply_fn from chosen candidate
    apply_p_fn = chosen["P"]["apply_fn"]
    classes = chosen["classes"]
    actions_by_cid = chosen["actions"]

    # Edge case: no classes (Identity with 0 edits)
    if not classes:
        return [apply_p_fn(X_test) for X_test in test_inputs]

    # Build sig_to_cid map from training data
    # Re-apply P to training to get items and rebuild signatures
    train_pairs = [(pair["input"], pair["output"]) for pair in task["train"]]
    tr_pairs_afterP = [(apply_p_fn(X), Y) for X, Y in train_pairs]
    items, _ = build_phi_partition(tr_pairs_afterP)

    # Build signature → class_id map
    sig_to_cid = {}
    for class_id, coords in classes.items():
        # Pick first coord from this class to get signature
        if coords:
            train_idx, r, c = coords[0]
            sig = _build_signature(items[train_idx]["feats"], r, c, items[train_idx]["Xp"])
            sig_to_cid[sig] = class_id

    # Apply to each test input
    predictions = []
    for X_test in test_inputs:
        # Apply P
        Xp_test = apply_p_fn(X_test)

        # Build Φ signatures for test
        feats_test = phi_signature_tables(Xp_test)

        # Group pixels by class_id via signature matching
        pixels_by_cid = defaultdict(list)
        R, C = dims(Xp_test)
        for r in range(R):
            for c in range(C):
                # Build signature for this test pixel
                sig = _build_signature(feats_test, r, c, Xp_test)

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

    # Enumerate all P
    P_registry = _enumerate_P_registry()

    # Collect all passing candidates
    candidates = []

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

            # Fit on training data (expects list of dicts with "input" and "output")
            fit_success = instance.fit(train_data)
            if not fit_success:
                # Fit failed, skip this P
                continue

            # Use instance.apply as the transform function
            apply_p_fn = instance.apply

        # Convert train to expected format for build_candidate_for_P
        train_pairs = [(pair["input"], pair["output"]) for pair in train_data]
        task_for_candidate = {**task, "train": train_pairs}

        # Try to build candidate for this P
        try:
            candidate = build_candidate_for_P(task_for_candidate, p_desc, apply_p_fn)
        except Exception:
            # Candidate build failed, skip this P
            continue

        if candidate is not None:
            # Add apply_fn to candidate for test application
            candidate["P"]["apply_fn"] = apply_p_fn
            candidates.append(candidate)

    # If no candidates pass, return UNSAT
    if not candidates:
        return {
            "status": "UNSAT",
            "reason": "no_passing_candidates",
            "candidates_tried": []
        }

    # Sort candidates by MDL tuple (lexicographic order)
    # MDL tuple: (num_classes, num_action_types, p_index, stable_hash)
    def mdl_key(c):
        mdl = c["mdl"]
        return (mdl["num_classes"], mdl["num_action_types"], mdl["p_index"], mdl["hash"])

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
