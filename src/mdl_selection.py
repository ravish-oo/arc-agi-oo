"""
MDL Primitives for Step-2 Best-Candidate Selection

Provides deterministic MDL scoring and stable hashing for Step-2 candidates.
MDL tuple order (lexicographic): (num_classes, num_action_types, p_index, stable_hash).

Functions:
    hash_candidate: Produce stable SHA-256 hex digest for tie-breaking
    compute_mdl: Return 4-tuple MDL cost for candidate selection
"""

import hashlib
import json


def hash_candidate(
    P_desc: dict,
    classes: dict[int, list[tuple[int, int, int]]],
    actions_by_cid: dict[int, tuple[str, object | None]]
) -> str:
    """
    Produce stable SHA-256 hex digest by canonicalizing and hashing candidate.

    Args:
        P_desc: {"name": str, "index": int} describing the global family
        classes: {cid: [(train_idx, row, col), ...], ...}
        actions_by_cid: {cid: (action_name, param_or_None), ...}

    Returns:
        Hex digest string (64 characters)

    Raises:
        ValueError: If P_desc missing fields or index<0
        ValueError: If coords or cids malformed
        TypeError: If param not JSON-serializable (only None or int allowed)

    Invariants:
        - Deterministic under identical inputs
        - Independent of dict insertion order
        - Pure (no mutation)

    Canonicalization:
        - Sort classes by cid
        - Sort coords within each class by (train_idx, row, col)
        - Sort actions by cid
        - JSON encode with sort_keys=True, separators=(',',':')
        - SHA-256 hash → hexdigest
    """
    # Validate P_desc
    if "name" not in P_desc or "index" not in P_desc:
        raise ValueError("P_desc must contain 'name' and 'index' fields")
    if not isinstance(P_desc["index"], int) or P_desc["index"] < 0:
        raise ValueError("P_desc['index'] must be non-negative integer")

    # Build canonical payload
    payload = {
        "P": {
            "name": P_desc["name"],
            "index": P_desc["index"]
        },
        "classes": [],
        "actions": []
    }

    # Sort classes by cid, coords by (i,r,c)
    for cid in sorted(classes.keys()):
        coords = classes[cid]
        # Validate coords
        for coord in coords:
            if not isinstance(coord, tuple) or len(coord) != 3:
                raise ValueError(f"Invalid coord format: {coord}")
            if not all(isinstance(x, int) for x in coord):
                raise ValueError(f"Coord elements must be integers: {coord}")

        # Sort coords by (train_idx, row, col)
        sorted_coords = sorted(coords, key=lambda c: (c[0], c[1], c[2]))
        payload["classes"].append([cid, [list(c) for c in sorted_coords]])

    # Sort actions by cid
    for cid in sorted(actions_by_cid.keys()):
        action_name, param = actions_by_cid[cid]

        # Validate param is JSON-serializable (only None or int)
        if param is not None and not isinstance(param, int):
            raise TypeError(f"Action param must be None or int, got {type(param)}")

        payload["actions"].append([cid, [action_name, param]])

    # Canonical JSON encoding
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))

    # SHA-256 hash
    hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_obj.hexdigest()


def compute_mdl(
    P_desc: dict,
    classes: dict[int, list[tuple[int, int, int]]],
    actions_by_cid: dict[int, tuple[str, object | None]]
) -> tuple[int, int, int, str]:
    """
    Compute MDL cost tuple for Step-2 candidate selection.

    Args:
        P_desc: {"name": str, "index": int} describing the global family
        classes: {cid: [(train_idx, row, col), ...], ...}
        actions_by_cid: {cid: (action_name, param_or_None), ...}

    Returns:
        (num_classes, num_action_types, p_index, stable_hash) where:
            - num_classes: number of Φ-classes edited
            - num_action_types: count of unique action names (not params)
            - p_index: position in {Identity} ∪ GLOBAL_MENU
            - stable_hash: hex digest from hash_candidate

    Raises:
        Propagates exceptions from hash_candidate

    Invariants:
        - Deterministic 4-tuple under identical inputs
        - No hidden fields, no randomness

    Edge Cases:
        - Empty classes → (0, num_action_types, p_index, hash)
        - Multiple actions with same name but different params → count as 1 type
    """
    # Number of classes
    num_classes = len(classes)

    # Number of unique action types (count unique action names, not params)
    action_names = set(action_name for action_name, _ in actions_by_cid.values())
    num_action_types = len(action_names)

    # P index
    p_index = P_desc["index"]

    # Stable hash
    stable_hash = hash_candidate(P_desc, classes, actions_by_cid)

    return (num_classes, num_action_types, p_index, stable_hash)
