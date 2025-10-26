"""
Receipts Module for ARC AGI Solver.

Generates proof-of-work receipts for PASS and UNSAT results.
Every solver step must emit a receipt containing:
- mode: "global" | "phi_partition" | "unsat"
- solver: family name (if PASS in Step-1)
- chosen_candidate: P + MDL + summary (if PASS in Step-2)
- task: lightweight task metadata
- params: learned parameters (if available and JSON-serializable)
- reason: why UNSAT (if UNSAT) or MDL selection explanation

Receipts are deterministic, JSON-serializable, and hashable.

Critical constraints:
- FY: Receipts are proof-of-work, not guesses
- Determinism: Same input → same receipt → same hash
- Completeness: Every PASS/UNSAT gets a receipt
- Serializable: All fields must be JSON-compatible
- Stable hashing: Canonical JSON + SHA-256 for verification

task_meta structure:
    {
        "task_id": str | int,    # Unique task identifier
        "train_n": int,          # Number of training pairs
        "test_n": int            # Number of test inputs
    }

Functions:
    generate_receipt_global: Step-1 PASS receipt
    generate_receipt_unsat: UNSAT receipt
    generate_receipt_phi: Step-2 PASS receipt (Φ-mode)
    receipt_stable_hash: Compute stable SHA-256 hash
"""

import json
import hashlib
from src.mdl_selection import compute_mdl


def generate_receipt_global(family, task_meta: dict) -> dict:
    """
    Generate PASS receipt for Step-1 global family success.

    Args:
        family: Fitted family instance with .name attribute (str)
                May have .params attribute (dict or None)
        task_meta: Lightweight task metadata dict with keys:
            - "task_id": str or int (unique task identifier)
            - "train_n": int (number of training pairs)
            - "test_n": int (number of test inputs)

    Returns:
        dict with keys:
            - "mode": "global" (always)
            - "solver": family.name (str)
            - "task": task_meta (dict, exactly as provided)
            - "params": dict (ONLY if family.params is JSON-serializable dict; omit otherwise)

    Edge cases:
        - family.params is None → omit "params" key entirely
        - family.params is not a dict → omit "params" key
        - family.params has non-JSON-serializable values → omit "params" key
        - Missing task_meta keys → function may error (caller responsibility)

    Invariants:
        - Deterministic: same (family, task_meta) → same receipt dict
        - JSON-serializable: all values are JSON-compatible types
        - Pure: read-only on inputs, no mutations
        - No timestamps, no randomness, no environment dependencies

    Purity:
        - Read-only on inputs
    """
    receipt = {
        "mode": "global",
        "solver": family.name,
        "task": task_meta
    }

    # Include params if available and JSON-serializable
    if hasattr(family, 'params') and family.params is not None:
        # Check if params is a dict and JSON-serializable
        if isinstance(family.params, dict):
            try:
                # Test JSON serialization
                json.dumps(family.params)
                receipt["params"] = family.params
            except (TypeError, ValueError):
                # Not JSON-serializable → omit params
                pass
        elif hasattr(family.params, '__dict__'):
            # params is an object with __dict__
            try:
                params_dict = vars(family.params)
                # Filter out None values and private attributes
                params_dict = {k: v for k, v in params_dict.items()
                             if not k.startswith('_') and v is not None}
                # Test JSON serialization
                json.dumps(params_dict)
                receipt["params"] = params_dict
            except (TypeError, ValueError):
                # Not JSON-serializable → omit params
                pass

    return receipt


def generate_receipt_unsat(reason: str, task_meta: dict) -> dict:
    """
    Generate UNSAT receipt when no global family satisfies FY.

    Args:
        reason: Short, deterministic string explaining UNSAT
                Examples: "no_global_P_satisfied_FY", "no_train_pairs", "malformed_input"
        task_meta: Lightweight task metadata dict (same structure as above)

    Returns:
        dict with keys:
            - "mode": "unsat" (always)
            - "reason": reason (str, exactly as provided)
            - "task": task_meta (dict, exactly as provided)

    Edge cases:
        - Empty reason string → accepted (but should be avoided by caller)
        - Reason with non-deterministic content (e.g., timestamps) → caller responsibility to avoid

    Invariants:
        - Deterministic: same (reason, task_meta) → same receipt
        - JSON-serializable
        - Pure: read-only on inputs

    Purity:
        - Read-only on inputs
    """
    return {
        "mode": "unsat",
        "reason": reason,
        "task": task_meta
    }


def receipt_stable_hash(receipt: dict) -> str:
    """
    Compute stable, deterministic hash of a receipt dict.

    Args:
        receipt: Receipt dict (from generate_receipt_global or generate_receipt_unsat)

    Returns:
        str: SHA-256 hex digest (64 hex characters)

    Algorithm:
        1. Serialize receipt to JSON with sorted keys and no whitespace
           (use json.dumps with sort_keys=True, separators=(',',':'))
        2. Encode as UTF-8 bytes
        3. Compute SHA-256 digest
        4. Return hex digest as string

    Edge cases:
        - Nested dicts/lists → handled by sort_keys=True (recursive)
        - None values → JSON null (handled)
        - Unicode strings → UTF-8 encoding (deterministic)

    Invariants:
        - Same receipt dict → same hash (bit-for-bit)
        - Cross-process stability: hash on machine A == hash on machine B
        - Cross-run stability: hash at time T1 == hash at time T2
        - Canonical form: no whitespace, sorted keys, deterministic separators

    Purity:
        - Read-only on receipt
        - No side effects
    """
    # Serialize to canonical JSON (sorted keys, no whitespace)
    canonical_json = json.dumps(receipt, sort_keys=True, separators=(',', ':'))

    # Encode to UTF-8 bytes
    json_bytes = canonical_json.encode('utf-8')

    # Compute SHA-256 hash
    hash_digest = hashlib.sha256(json_bytes).hexdigest()

    return hash_digest


def generate_receipt_phi(
    P_desc: dict,
    classes: dict[int, list[tuple[int, int, int]]],
    actions_by_cid: dict[int, tuple[str, object | None]],
    mdl_candidates: list[dict]
) -> dict:
    """
    Generate Φ-mode receipt for Step-2 candidate selection.

    Args:
        P_desc: {"name": str, "index": int} - chosen P description
        classes: {cid: [(train_idx, r, c), ...]} - chosen classes
        actions_by_cid: {cid: (action_name, param_or_None)} - chosen actions
        mdl_candidates: list of candidate dicts from solve_step2, each with:
            {
                "P": {"name": str, "index": int},
                "mdl": {"num_classes": int, "num_action_types": int, "p_index": int, "hash": str},
                ...
            }

    Returns:
        dict with structure:
        {
            "mode": "phi_partition",
            "chosen_candidate": {
                "P": {"name": str, "index": int},
                "mdl": {"num_classes": int, "num_action_types": int, "p_index": int, "hash": str},
                "summary": {
                    "num_classes": int,
                    "num_action_types": int,
                    "actions_histogram": {"action_name": count, ...}
                },
                "reason": "MDL: fewer classes ▸ fewer action types ▸ lower P index ▸ stable hash"
            },
            "mdl_candidates": [
                {"P": {...}, "mdl": {...}, "summary": {...}},
                ...
            ]
        }

    Edge cases:
        - Empty classes → num_classes=0, empty actions_histogram
        - Single candidate → validates correctly
        - Identity (index=-1) → accepted as valid

    Invariants:
        - Deterministic: same inputs → same receipt
        - JSON-serializable: all fields are JSON-compatible
        - Canonical ordering: mdl_candidates sorted by MDL tuple
        - Validates chosen candidate is in mdl_candidates list

    Raises:
        ValueError: If P_desc missing "name" or "index" fields
        ValueError: If P_desc["index"] < -1
        ValueError: If mdl_candidates is empty
        ValueError: If chosen candidate not found in mdl_candidates

    Purity:
        - Read-only on all inputs (no mutation)
        - No timestamps, no randomness, no environment dependencies
    """
    # Validate P_desc
    if "name" not in P_desc or "index" not in P_desc:
        raise ValueError("P_desc must contain 'name' and 'index' fields")
    if not isinstance(P_desc["index"], int) or P_desc["index"] < -1:
        raise ValueError("P_desc['index'] must be >= -1 (Identity uses -1, families use 0+)")

    # Validate mdl_candidates not empty
    if not mdl_candidates:
        raise ValueError("mdl_candidates must not be empty")

    # Compute MDL for chosen candidate
    mdl_tuple = compute_mdl(P_desc, classes, actions_by_cid)
    num_classes, num_action_types, p_index, stable_hash = mdl_tuple

    # Build actions histogram (count action names, not params)
    actions_histogram = {}
    for action_name, _ in actions_by_cid.values():
        actions_histogram[action_name] = actions_histogram.get(action_name, 0) + 1

    # Sort histogram by action name for determinism
    actions_histogram = dict(sorted(actions_histogram.items()))

    # Build chosen_candidate dict
    chosen_candidate = {
        "P": {
            "name": P_desc["name"],
            "index": P_desc["index"]
        },
        "mdl": {
            "num_classes": num_classes,
            "num_action_types": num_action_types,
            "p_index": p_index,
            "hash": stable_hash
        },
        "summary": {
            "num_classes": num_classes,
            "num_action_types": num_action_types,
            "actions_histogram": actions_histogram
        },
        "reason": "MDL: fewer classes ▸ fewer action types ▸ lower P index ▸ stable hash"
    }

    # Sort mdl_candidates by MDL tuple (lexicographic order)
    def mdl_key(cand):
        mdl = cand["mdl"]
        return (mdl["num_classes"], mdl["num_action_types"], mdl["p_index"], mdl["hash"])

    sorted_candidates = sorted(mdl_candidates, key=mdl_key)

    # Validate chosen candidate is in sorted list
    chosen_mdl_tuple = (num_classes, num_action_types, p_index, stable_hash)
    found = False
    for cand in sorted_candidates:
        cand_mdl = cand["mdl"]
        cand_tuple = (cand_mdl["num_classes"], cand_mdl["num_action_types"],
                      cand_mdl["p_index"], cand_mdl["hash"])
        if cand_tuple == chosen_mdl_tuple:
            found = True
            break

    if not found:
        raise ValueError(f"Chosen candidate with MDL {chosen_mdl_tuple} not found in mdl_candidates")

    # Build candidate entries for receipt (compact: P + mdl + summary only)
    receipt_candidates = []
    for cand in sorted_candidates:
        # Build actions histogram for this candidate (if it has actions)
        if "actions" in cand:
            cand_histogram = {}
            for action_name, _ in cand["actions"].values():
                cand_histogram[action_name] = cand_histogram.get(action_name, 0) + 1
            cand_histogram = dict(sorted(cand_histogram.items()))
        else:
            cand_histogram = {}

        receipt_candidates.append({
            "P": {
                "name": cand["P"]["name"],
                "index": cand["P"]["index"]
            },
            "mdl": {
                "num_classes": cand["mdl"]["num_classes"],
                "num_action_types": cand["mdl"]["num_action_types"],
                "p_index": cand["mdl"]["p_index"],
                "hash": cand["mdl"]["hash"]
            },
            "summary": {
                "num_classes": cand["mdl"]["num_classes"],
                "num_action_types": cand["mdl"]["num_action_types"],
                "actions_histogram": cand_histogram
            }
        })

    # Build and return receipt
    return {
        "mode": "phi_partition",
        "chosen_candidate": chosen_candidate,
        "mdl_candidates": receipt_candidates
    }
