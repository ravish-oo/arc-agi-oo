"""
Receipts Module for ARC AGI Solver.

Generates proof-of-work receipts for PASS and UNSAT results.
Every solver step must emit a receipt containing:
- mode: "global" | "composed" | "local" | "unsat"
- step: 1 | 2 | 3
- solver: family name (if PASS)
- params: learned parameters (if available)
- reason: why UNSAT (if UNSAT)

Receipts are deterministic and serializable (JSON-compatible).

Critical constraints:
- FY: Receipts are proof-of-work, not guesses
- Determinism: Same input → same receipt
- Completeness: Every PASS/UNSAT gets a receipt
- Serializable: All fields must be JSON-compatible
"""


def generate_receipt_global(family_instance, task: dict) -> dict:
    """
    Generate PASS receipt for Step 1 (global P family success).

    Args:
        family_instance: Fitted family instance with .name and .params attributes
        task: Original task dict with "train" and "test" keys

    Returns:
        dict with keys:
            - "mode": "global"
            - "solver": family_instance.name (str)
            - "step": 1
            - "num_trains": len(task["train"])
            - "num_tests": len(task["test"])
            - "params": family_instance.params.__dict__ (if available)

    Edge cases:
        - Missing params attribute: Use {}
        - params not serializable: Convert to str representation
        - Missing train/test keys: Use 0

    Determinism:
        - Same (family_instance, task) → same receipt

    Purity:
        - Read-only on inputs
    """
    receipt = {
        "mode": "global",
        "solver": family_instance.name,
        "step": 1,
        "num_trains": len(task.get("train", [])),
        "num_tests": len(task.get("test", []))
    }

    # Include params if available
    if hasattr(family_instance, 'params') and family_instance.params is not None:
        try:
            # Try to get __dict__ from params object
            if hasattr(family_instance.params, '__dict__'):
                params_dict = vars(family_instance.params)
                # Filter out None values and private attributes
                params_dict = {k: v for k, v in params_dict.items()
                             if not k.startswith('_') and v is not None}
                receipt["params"] = params_dict
            else:
                # params is already a dict or other type
                receipt["params"] = family_instance.params
        except Exception:
            # Fallback: convert to string representation
            receipt["params"] = str(family_instance.params)
    else:
        receipt["params"] = {}

    return receipt


def generate_receipt_unsat(reason: str) -> dict:
    """
    Generate UNSAT receipt for Step 1 (no family matched).

    Args:
        reason: Human-readable string explaining why UNSAT
                Examples: "no_family_matched", "no_train_pairs", "malformed_task"

    Returns:
        dict with keys:
            - "mode": "unsat"
            - "step": 1
            - "reason": reason (str)
            - "witness": None (Step 1 has no class witnesses; that's Step 2)

    Determinism:
        - Same reason → same receipt

    Purity:
        - Read-only on inputs
    """
    return {
        "mode": "unsat",
        "step": 1,
        "reason": reason,
        "witness": None  # Step 1 has no class witnesses (that's Step 2)
    }
