"""
Coverage Meter for Step-1 and Step-2 Solvers — Phase 7 work order P7-05.

Measures solver coverage on ARC training dataset.
Runs solve_step1() and/or solve_step2() on all tasks, aggregates results.

Expected coverage:
- Step-1 baseline: 25-30% (100-120 tasks out of 400)
- Step-2 target: ~67% (268 tasks out of 400)

Usage:
    python tests/measure_coverage.py data/arc-agi_training_challenges.json
    python tests/measure_coverage.py data/arc-agi_training_challenges.json --steps 1
    python tests/measure_coverage.py data/arc-agi_training_challenges.json --steps 1,2

Output format (deterministic, when both steps run):
    ========================================
    Step 1 Coverage: 112/400 (28.0%)
    Step 2 Coverage: 268/400 (67.0%)  [+156 tasks, +39.0%]
    ========================================
    Step 1 by family:
    ColorMap: 25
    Isometry: 34
    UNSAT: 288
    ---
    Step 2 by P:
    ColorMap: 48
    Identity: 34
    UNSAT: 132
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver_step1 import solve_step1
from src.solver_step2 import solve_step2


def measure_coverage(dataset_path: str, steps: list[int] = [1, 2]) -> dict:
    """
    Load dataset, run specified solver steps on ALL tasks, aggregate results.

    Algorithm:
        1. Load JSON from dataset_path (raise on file/parse error)
        2. Extract all task IDs, sort lexicographically (deterministic order)
        3. For each task:
           - If 1 in steps: call solve_step1(task), aggregate by family
           - If 2 in steps: call solve_step2(task), aggregate by P name
        4. If both steps run: compute delta (Step-2 PASS - Step-1 PASS)
        5. Build summary dict with sorted keys
        6. Return summary

    Args:
        dataset_path: Path to JSON dataset file
        steps: List of step numbers to run (default: [1, 2])
               Examples: [1], [2], [1, 2]

    Returns:
        {
            "total": int,                    # Total tasks processed
            "step1": {                       # Step-1 results (if 1 in steps)
                "pass": int,
                "unsat": int,
                "by_family": {str: int}      # Sorted family names → count
            } | None,
            "step2": {                       # Step-2 results (if 2 in steps)
                "pass": int,
                "unsat": int,
                "by_p": {str: int}           # Sorted P names → count
            } | None,
            "delta": {                       # Delta (Step-2 - Step-1)
                "abs": int,                  # pass2 - pass1
                "pct": float                 # delta_abs / total * 100
            } | None                         # None if not both steps
        }

    Edge cases:
        - Empty dataset {} → total=0, all counts=0
        - steps=[2] only → step1=None, delta=None
        - Missing "train"/"test" → counted as UNSAT
        - Solver raises exception → caught, counted as UNSAT
        - File not found or bad JSON → raise exception (caller handles)

    Invariants:
        - Deterministic: sorted task IDs, sorted names
        - Purity: No mutations to dataset or tasks
        - Robustness: Catches solver exceptions per task
        - Completeness: total == pass + unsat (always, per step)

    Purity:
        - Read-only on dataset
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Initialize counters
    total = 0

    # Step-1 counters
    step1_pass = 0
    step1_unsat = 0
    by_family = {}

    # Step-2 counters
    step2_pass = 0
    step2_unsat = 0
    by_p = {}

    # Sort task IDs for deterministic iteration
    task_ids = sorted(dataset.keys())

    # Process each task
    for task_id in task_ids:
        total += 1
        task = dataset[task_id]

        # Add task ID for receipts
        task["id"] = task_id

        # Run Step-1 if requested
        if 1 in steps:
            try:
                result = solve_step1(task)
                receipt = result["receipt"]

                if receipt["mode"] == "global":
                    step1_pass += 1
                    solver = receipt["solver"]
                    by_family[solver] = by_family.get(solver, 0) + 1
                else:
                    step1_unsat += 1

            except Exception:
                step1_unsat += 1

        # Run Step-2 if requested
        if 2 in steps:
            try:
                result = solve_step2(task)

                if result["status"] == "PASS":
                    step2_pass += 1
                    # Extract P name from chosen candidate
                    p_name = result["candidate"]["P"]["name"]
                    by_p[p_name] = by_p.get(p_name, 0) + 1
                else:
                    step2_unsat += 1

            except Exception:
                step2_unsat += 1

    # Build summary dict
    summary = {"total": total}

    # Add Step-1 results if run
    if 1 in steps:
        summary["step1"] = {
            "pass": step1_pass,
            "unsat": step1_unsat,
            "by_family": dict(sorted(by_family.items()))
        }
    else:
        summary["step1"] = None

    # Add Step-2 results if run
    if 2 in steps:
        summary["step2"] = {
            "pass": step2_pass,
            "unsat": step2_unsat,
            "by_p": dict(sorted(by_p.items()))
        }
    else:
        summary["step2"] = None

    # Compute delta if both steps run
    if 1 in steps and 2 in steps:
        delta_abs = step2_pass - step1_pass
        if total > 0:
            delta_pct = (delta_abs / total) * 100.0
        else:
            delta_pct = 0.0

        summary["delta"] = {
            "abs": delta_abs,
            "pct": delta_pct
        }
    else:
        summary["delta"] = None

    return summary


def _cli_main(argv: list[str] | None = None) -> int:
    """
    CLI wrapper: parse args, call measure_coverage, print report.

    Algorithm:
        1. Parse args: <dataset_path> and optional --steps flag
        2. Parse --steps value (e.g., "1" or "1,2") into list of ints
        3. Call measure_coverage(dataset_path, steps)
        4. Print formatted report based on which steps ran
        5. Return 0 on success

    Args:
        argv: Command-line args (defaults to sys.argv[1:])
              Examples:
              - ["data/arc-agi_training_challenges.json"]
              - ["data/arc-agi_training_challenges.json", "--steps", "1"]
              - ["data/arc-agi_training_challenges.json", "--steps", "1,2"]

    CLI flags:
        dataset_path: Positional, required
        --steps: Comma-separated step numbers (default: "1,2")
                 Examples: "1", "2", "1,2"

    Returns:
        Exit code:
            0: Success (report printed)
            2: File not found, JSON parse error, or bad --steps value

    Output format (deterministic, when both steps run):
        ========================================
        Step 1 Coverage: 112/400 (28.0%)
        Step 2 Coverage: 268/400 (67.0%)  [+156 tasks, +39.0%]
        ========================================
        Step 1 by family:
        ColorMap: 25
        Isometry: 34
        UNSAT: 288
        ---
        Step 2 by P:
        ColorMap: 48
        Identity: 34
        UNSAT: 132

    Edge cases:
        - Missing arg → print usage to stderr, exit 2
        - Empty dataset → "Coverage: 0/0 (0.0%)"
        - --steps 1 only → omit Step-2 section and delta

    Invariants:
        - No timestamps, no debug logs, no environment info
        - Sorted families/P names (lexicographic)
        - Deterministic: same dataset → byte-identical stdout

    Purity:
        - Read-only on argv and dataset
        - Stdout writes only
    """
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Measure solver coverage on ARC dataset",
        add_help=True
    )
    parser.add_argument("dataset_path", help="Path to JSON dataset")
    parser.add_argument("--steps", default="1,2",
                        help="Comma-separated step numbers (default: '1,2')")

    try:
        if argv is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code else 0

    dataset_path = args.dataset_path

    # Parse steps
    try:
        steps = [int(s.strip()) for s in args.steps.split(',')]
        # Validate: only 1 and 2 are allowed
        if not all(s in [1, 2] for s in steps):
            raise ValueError("Steps must be 1 or 2")
    except ValueError:
        print(json.dumps({"error": "invalid_steps_value"}), file=sys.stderr)
        return 2

    # Call measure_coverage
    try:
        summary = measure_coverage(dataset_path, steps)
    except FileNotFoundError:
        print(json.dumps({"error": "file_not_found"}), file=sys.stderr)
        return 2
    except json.JSONDecodeError:
        print(json.dumps({"error": "json_parse_error"}), file=sys.stderr)
        return 2

    # Print report
    total = summary["total"]

    # Header
    print("=" * 40)

    # Step-1 summary line (if run)
    if summary["step1"] is not None:
        step1 = summary["step1"]
        pass1 = step1["pass"]
        pct1 = (pass1 / total * 100.0) if total > 0 else 0.0
        print(f"Step 1 Coverage: {pass1}/{total} ({pct1:.1f}%)")

    # Step-2 summary line with delta (if run)
    if summary["step2"] is not None:
        step2 = summary["step2"]
        pass2 = step2["pass"]
        pct2 = (pass2 / total * 100.0) if total > 0 else 0.0

        # Add delta if both steps ran
        if summary["delta"] is not None:
            delta = summary["delta"]
            delta_abs = delta["abs"]
            delta_pct = delta["pct"]
            sign = "+" if delta_abs >= 0 else ""
            print(f"Step 2 Coverage: {pass2}/{total} ({pct2:.1f}%)  [{sign}{delta_abs} tasks, {sign}{delta_pct:.1f}%]")
        else:
            print(f"Step 2 Coverage: {pass2}/{total} ({pct2:.1f}%)")

    print("=" * 40)

    # Step-1 breakdown (if run)
    if summary["step1"] is not None:
        step1 = summary["step1"]
        print("Step 1 by family:")
        for family, count in step1["by_family"].items():
            print(f"{family}: {count}")
        print(f"UNSAT: {step1['unsat']}")

    # Separator between steps (if both ran)
    if summary["step1"] is not None and summary["step2"] is not None:
        print("---")

    # Step-2 breakdown (if run)
    if summary["step2"] is not None:
        step2 = summary["step2"]
        print("Step 2 by P:")
        for p_name, count in step2["by_p"].items():
            print(f"{p_name}: {count}")
        print(f"UNSAT: {step2['unsat']}")

    return 0


if __name__ == "__main__":
    exit_code = _cli_main()
    sys.exit(exit_code)
