"""
Step-2 Solver: Candidate Builder for Global Transform P + Φ/GLUE

Provides candidate building for Step-2 solving (P + Φ/GLUE mode).

Functions:
    build_candidate_for_P: Evaluate ONE P on ONE task, return candidate dict or None
"""

from src.utils import dims
from src.glue import build_phi_partition, verify_stitched_equality
from src.action_inference import infer_action_for_class
from src.mdl_selection import compute_mdl


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
