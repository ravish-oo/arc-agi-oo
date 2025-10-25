"""
Step-1 Solver for ARC AGI Pure Math Solver.

Step 1 of the 3-step algorithm:
    1. Try 16 global P families in fixed order (this module)
    2. Try P + Φ/GLUE compositional mode (Step 2, Phase 7)
    3. Try LUT local fallback (Step 3, Phase 9)

This module implements Step 1: Global exact transformations only.

Algorithm:
    - Loop through 16 global families in deterministic order
    - For each family: fit(trains) → verify FY → apply(tests)
    - Selection strategy: FIRST-PASS (not MDL; that's Step 2)
    - Return first family that passes FY on ALL trains

Critical constraints:
    - FY exactness: deep_eq() on ALL training pairs (bit-for-bit)
    - First-pass: Return FIRST family that succeeds (no MDL)
    - Determinism: Fixed iteration order, same task → same result
    - Receipts: Every PASS/UNSAT gets proof-of-work dict
    - Purity: No mutations to task or grids

Expected coverage: 25-30% baseline on ARC training set.
"""

from src.utils import deep_eq

# Import all 16 Phase-2 families in deterministic order
from src.families.isometry import IsometryFamily
from src.families.color_map import ColorMapFamily
from src.families.iso_color_map import IsoColorMapFamily
from src.families.pixel_replicate import PixelReplicateFamily
from src.families.block_down import BlockDownFamily
from src.families.nps_down import NPSDownFamily
from src.families.nps_up import NPSUpFamily
from src.families.parity_tile import ParityTileFamily
from src.families.block_permutation import BlockPermutationFamily
from src.families.block_substitution import BlockSubstitutionFamily
from src.families.row_permutation import RowPermutationFamily
from src.families.col_permutation import ColPermutationFamily
from src.families.sort_rows import SortRowsLexFamily
from src.families.sort_cols import SortColsLexFamily
from src.families.mirror_complete import MirrorCompleteFamily
from src.families.copy_move import CopyMoveAllComponentsFamily

from src.receipts import generate_receipt_global, generate_receipt_unsat


# Fixed order (matches spec.md lines 26-33)
GLOBAL_FAMILIES = [
    IsometryFamily,
    ColorMapFamily,
    IsoColorMapFamily,
    PixelReplicateFamily,
    BlockDownFamily,
    NPSDownFamily,
    NPSUpFamily,
    ParityTileFamily,
    BlockPermutationFamily,
    BlockSubstitutionFamily,
    RowPermutationFamily,
    ColPermutationFamily,
    SortRowsLexFamily,
    SortColsLexFamily,
    MirrorCompleteFamily,
    CopyMoveAllComponentsFamily
]


def _try_family(family_class, train_pairs: list[dict]) -> tuple[object | None, bool]:
    """
    Helper: Instantiate family, call fit(), verify FY on all trains.

    Algorithm:
        1. Instantiate family: instance = family_class()
        2. Call instance.fit(train_pairs)
        3. If fit() returns False: return (None, False)
        4. Verify FY: For each pair in train_pairs:
            - Check deep_eq(instance.apply(X), Y)
            - If ANY mismatch: return (None, False)
        5. If all verified: return (instance, True)

    Args:
        family_class: Class object (e.g., IsometryFamily)
        train_pairs: list of {"input": grid, "output": grid}

    Returns:
        tuple of (family_instance | None, success: bool)
        - (instance, True) if FY satisfied on ALL trains
        - (None, False) otherwise

    Semantics:
        - Catches exceptions from fit() or apply() → (None, False)
        - Verification uses deep_eq() from utils.py
        - FY strictness: Single pixel diff → fail

    Edge cases:
        - fit() raises exception: catch and return (None, False)
        - apply() raises exception: catch and return (None, False)
        - Empty train_pairs: return (None, False)

    Determinism:
        - Same (family_class, train_pairs) → same result

    Purity:
        - Read-only on train_pairs
    """
    # Edge case: empty train pairs
    if not train_pairs:
        return (None, False)

    try:
        # Instantiate family
        instance = family_class()

        # Call fit()
        fit_result = instance.fit(train_pairs)
        if not fit_result:
            return (None, False)

        # Verify FY on ALL trains (critical: fit() may be imperfect)
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Apply family to input
            Y_pred = instance.apply(X)

            # Check bit-for-bit equality
            if not deep_eq(Y_pred, Y):
                return (None, False)

        # All trains verified → success
        return (instance, True)

    except Exception:
        # Any error during fit/apply → reject family
        # This includes: RuntimeError, KeyError, IndexError, etc.
        return (None, False)


def solve_step1(task: dict) -> dict:
    """
    Step-1 solver: Try 16 global P families in fixed order (first-pass).

    Algorithm:
        1. Extract train_pairs from task["train"] as list of dicts
        2. Extract test_inputs from task["test"] as list of dicts
        3. For each family in GLOBAL_FAMILIES (fixed order):
            a. Call _try_family(family_class, train_pairs)
            b. If family returns (instance, True):
                - Apply instance.apply(X) to all test inputs
                - Generate predictions list
                - Return PASS result with receipt
        4. If NO family succeeds, return UNSAT result with receipt

    Args:
        task: dict with keys "train" (list of {"input": grid, "output": grid})
                           and "test" (list of {"input": grid})
                           Optionally "id" for task identifier

    Returns:
        dict with keys:
            - "status": "PASS" | "UNSAT"
            - "predictions": list[grid] (if PASS, else None)
            - "receipt": dict (see generate_receipt_global/unsat)

    Semantics:
        - FIRST-PASS: Returns first family that passes FY (not best by MDL)
        - FY exactness: Family must match ALL trains bit-for-bit
        - Determinism: Same task → same result (fixed family order)
        - No mutations: task dict is never modified
        - Empty trains/tests: Return UNSAT with reason

    Edge cases:
        - Empty train: Return UNSAT ("no_train_pairs")
        - Empty test: Return PASS with empty predictions list
        - Malformed task (missing "train"/"test"): Return UNSAT

    Determinism/Purity:
        - No randomness; iteration order is deterministic
        - No mutations to task or grids
        - Same inputs → same outputs
    """
    # Edge case: malformed task
    if "train" not in task or "test" not in task:
        # Build minimal task_meta for malformed task
        task_meta = {
            "task_id": task.get("id", "unknown"),
            "train_n": len(task.get("train", [])),
            "test_n": len(task.get("test", []))
        }
        return {
            "status": "UNSAT",
            "predictions": None,
            "receipt": generate_receipt_unsat("malformed_task", task_meta)
        }

    train_pairs = task["train"]
    test_examples = task["test"]

    # Build task_meta for receipts
    task_meta = {
        "task_id": task.get("id", "unknown"),
        "train_n": len(train_pairs),
        "test_n": len(test_examples)
    }

    # Edge case: empty train
    if not train_pairs:
        return {
            "status": "UNSAT",
            "predictions": None,
            "receipt": generate_receipt_unsat("no_train_pairs", task_meta)
        }

    # Extract test inputs
    test_inputs = [ex["input"] for ex in test_examples]

    # Try each family in fixed order (first-pass)
    for family_class in GLOBAL_FAMILIES:
        instance, success = _try_family(family_class, train_pairs)

        if success:
            # Apply to all test inputs
            predictions = []
            try:
                for X in test_inputs:
                    Y_pred = instance.apply(X)
                    predictions.append(Y_pred)

                # Success: return PASS result
                return {
                    "status": "PASS",
                    "predictions": predictions,
                    "receipt": generate_receipt_global(instance, task_meta)
                }

            except Exception:
                # If apply() fails on test (shouldn't happen after FY verification)
                # Skip this family and try next one
                continue

    # No family matched
    return {
        "status": "UNSAT",
        "predictions": None,
        "receipt": generate_receipt_unsat("no_family_matched", task_meta)
    }


def _cli_main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for single-task execution.

    Algorithm:
        1. Parse args: --task <path>, --task-id <id>, [--print-receipt]
        2. Load JSON from <path> (exit 2 on file/parse error)
        3. Extract task[<id>] (exit 3 if missing)
        4. Validate task has "train" and "test" keys (exit 4 if malformed)
        5. Call solve_step1(task) with task["id"] = <id>
        6. Print JSON to stdout:
           - Keys: "task_id", "predictions", ["receipt"] (if --print-receipt)
           - Sorted keys, no trailing whitespace
        7. Return 0 on success

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None)
              Example: ["--task", "data.json", "--task-id", "abc123", "--print-receipt"]

    Returns:
        Exit code:
            0: Success (task solved, JSON printed)
            2: File not found or JSON parse error
            3: Task ID not found in dataset
            4: Malformed task (missing "train" or "test")

    Output format (stdout):
        Without --print-receipt:
        {"predictions":[[...]],"task_id":"abc123"}

        With --print-receipt:
        {"predictions":[[...]],"receipt":{...},"task_id":"abc123"}

    Edge cases:
        - Missing --task or --task-id: print usage to stderr, exit 2
        - Empty task dataset {}: exit 3 (no tasks)
        - Task with empty train/test: solve_step1 returns UNSAT, predictions=None
        - Predictions=None: omit "predictions" key (or include as null)

    Semantics:
        - Deterministic: same inputs → same outputs (sorted keys)
        - Pure I/O: read file, print once, exit
        - No side effects: no logs, no timestamps, no env access
        - No mutations: task dict is read-only

    Purity:
        - Read-only on argv and loaded data
        - Stdout/stderr writes only (no file writes)
    """
    import argparse
    import json
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run Step-1 solver on single ARC task",
        add_help=True
    )
    parser.add_argument("--task", required=True, help="Path to JSON dataset")
    parser.add_argument("--task-id", required=True, help="Task ID to solve")
    parser.add_argument("--print-receipt", action="store_true",
                       help="Include receipt in output")

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse calls sys.exit on error or --help
        return e.code if e.code else 0

    task_path = args.task
    task_id = args.task_id
    print_receipt = args.print_receipt

    # Load JSON dataset
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

    # Add ID to task for receipts
    task["id"] = task_id

    # Validate task structure (minimal check - solve_step1 does full validation)
    if "train" not in task or "test" not in task:
        print(json.dumps({"error": "malformed_task"}), file=sys.stderr)
        return 4

    # Call solve_step1
    result = solve_step1(task)

    # Build output dict
    output = {"task_id": task_id}

    # Include predictions (even if None)
    if result["predictions"] is not None:
        output["predictions"] = result["predictions"]

    # Include receipt if requested
    if print_receipt:
        output["receipt"] = result["receipt"]

    # Print with sorted keys, compact format (no trailing whitespace)
    print(json.dumps(output, sort_keys=True, separators=(',', ':')))

    return 0


if __name__ == "__main__":
    import sys
    exit_code = _cli_main()
    sys.exit(exit_code)
