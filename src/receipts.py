"""
Receipts Module for ARC AGI Solver.

Generates proof-of-work receipts for PASS and UNSAT results.
Every solver step must emit a receipt containing:
- mode: "global" | "unsat"
- solver: family name (if PASS)
- task: lightweight task metadata
- params: learned parameters (if available and JSON-serializable)
- reason: why UNSAT (if UNSAT)

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
"""

import json
import hashlib


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
