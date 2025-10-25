"""
Coverage Meter for Step-1 Solver — Phase 3 work order P3-04.

Measures Step-1 solver coverage on ARC training dataset.
Runs solve_step1() on all tasks, aggregates results by family.

Expected baseline coverage: 25-30% (100-120 tasks out of 400).

Usage:
    python tests/measure_coverage.py data/arc-agi_training_challenges.json

Output format (deterministic):
    ========================================
    Step 1 Coverage: 112/400 (28.0%)
    ========================================
    BlockDown: 8
    ColorMap: 25
    Isometry: 34
    PixelReplicate: 19
    UNSAT: 288
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver_step1 import solve_step1


def measure_coverage(dataset_path: str) -> dict:
    """
    Load dataset, run solve_step1 on ALL tasks, aggregate results.

    Algorithm:
        1. Load JSON from dataset_path (raise on file/parse error)
        2. Extract all task IDs, sort lexicographically (deterministic order)
        3. For each task:
           - Call solve_step1(task) with task["id"] = task_id
           - Inspect receipt["mode"]:
             - If "global": increment pass, record receipt["solver"] family
             - Else: increment unsat
        4. Build summary dict with sorted keys
        5. Return summary

    Args:
        dataset_path: Path to JSON dataset file

    Returns:
        {
            "total": int,              # Total tasks processed
            "pass": int,               # Tasks with mode="global"
            "unsat": int,              # Tasks with mode="unsat"
            "by_family": {             # Pass count per family (sorted keys)
                "Isometry": int,
                "ColorMap": int,
                ...
            }
        }

    Edge cases:
        - Empty dataset {} → total=0, pass=0, unsat=0, by_family={}
        - Missing "train"/"test" in task → counted as UNSAT
        - solve_step1 raises exception → caught, counted as UNSAT
        - File not found or bad JSON → raise exception (caller handles)

    Invariants:
        - Deterministic: sorted task IDs, sorted family names
        - Purity: No mutations to dataset or tasks
        - Robustness: Catches solve_step1 exceptions per task
        - Completeness: total == pass + unsat (always)

    Purity:
        - Read-only on dataset
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Initialize counters
    total = 0
    pass_count = 0
    unsat_count = 0
    by_family = {}

    # Sort task IDs for deterministic iteration
    task_ids = sorted(dataset.keys())

    # Process each task
    for task_id in task_ids:
        total += 1
        task = dataset[task_id]

        # Add task ID for receipts
        task["id"] = task_id

        try:
            # Call solve_step1
            result = solve_step1(task)

            # Inspect receipt
            receipt = result["receipt"]

            if receipt["mode"] == "global":
                # PASS: increment pass count and family count
                pass_count += 1
                solver = receipt["solver"]
                by_family[solver] = by_family.get(solver, 0) + 1
            else:
                # UNSAT or other mode
                unsat_count += 1

        except Exception:
            # Any exception during solve_step1 → count as UNSAT
            unsat_count += 1

    # Build summary dict with sorted family keys
    summary = {
        "total": total,
        "pass": pass_count,
        "unsat": unsat_count,
        "by_family": dict(sorted(by_family.items()))
    }

    return summary


def _cli_main(argv: list[str] | None = None) -> int:
    """
    CLI wrapper: parse args, call measure_coverage, print report.

    Algorithm:
        1. Parse single positional arg: <dataset_path>
        2. Call measure_coverage(dataset_path)
        3. Print fixed-format report to stdout:
           - Header separator (40 '=' chars)
           - Summary line: "Step 1 Coverage: {pass}/{total} ({percent:.1f}%)"
           - Header separator (40 '=' chars)
           - Per-family breakdown (sorted, one per line: "{family}: {count}")
           - UNSAT line: "UNSAT: {unsat}"
        4. Return 0 on success

    Args:
        argv: Command-line args (defaults to sys.argv[1:])
              Example: ["data/arc-agi_training_challenges.json"]

    Returns:
        Exit code:
            0: Success (report printed)
            2: File not found or JSON parse error

    Output format (stdout, deterministic):
        ========================================
        Step 1 Coverage: 112/400 (28.0%)
        ========================================
        BlockDown: 8
        ColorMap: 25
        CopyMoveAllComponents: 5
        Isometry: 34
        PixelReplicate: 19
        SortRowsLex: 3
        UNSAT: 288

    Edge cases:
        - Missing arg → print usage to stderr, exit 2
        - Empty dataset → "Step 1 Coverage: 0/0 (0.0%)"
        - All UNSAT → only UNSAT line printed (no family lines)

    Invariants:
        - No timestamps, no debug logs, no environment info
        - Sorted families (lexicographic)
        - Deterministic: same dataset → byte-identical stdout
        - No randomness, no logging beyond report

    Purity:
        - Read-only on argv and dataset
        - Stdout writes only
    """
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Measure Step-1 solver coverage on ARC dataset",
        add_help=True
    )
    parser.add_argument("dataset_path", help="Path to JSON dataset")

    try:
        if argv is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code else 0

    dataset_path = args.dataset_path

    # Call measure_coverage
    try:
        summary = measure_coverage(dataset_path)
    except FileNotFoundError:
        print(json.dumps({"error": "file_not_found"}), file=sys.stderr)
        return 2
    except json.JSONDecodeError:
        print(json.dumps({"error": "json_parse_error"}), file=sys.stderr)
        return 2

    # Print report
    total = summary["total"]
    pass_count = summary["pass"]
    unsat_count = summary["unsat"]

    # Calculate percentage
    if total > 0:
        percent = (pass_count / total) * 100.0
    else:
        percent = 0.0

    # Header
    print("=" * 40)
    print(f"Step 1 Coverage: {pass_count}/{total} ({percent:.1f}%)")
    print("=" * 40)

    # Per-family breakdown (sorted)
    for family, count in summary["by_family"].items():
        print(f"{family}: {count}")

    # UNSAT line
    print(f"UNSAT: {unsat_count}")

    return 0


if __name__ == "__main__":
    exit_code = _cli_main()
    sys.exit(exit_code)
