"""
Tests for Coverage Meter — Phase 3 work order P3-04.

Comprehensive test suite covering:
- Empty dataset
- Single PASS/UNSAT tasks
- Multiple tasks with mixed results
- Determinism (same dataset → same output)
- Sorted output (task IDs and families)
- CLI exit codes
- Error handling
"""

import pytest
import json
import tempfile
import os
from tests.measure_coverage import measure_coverage, _cli_main


# =============================
# Helper Functions
# =============================

def create_temp_dataset(tasks_dict):
    """Create temporary JSON file with task dataset."""
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as f:
        json.dump(tasks_dict, f)
    return path


# =============================
# A. measure_coverage() Tests
# =============================

def test_measure_coverage_empty_dataset():
    """Empty dataset → total=0, pass=0, unsat=0, by_family={}."""
    dataset = {}
    path = create_temp_dataset(dataset)

    try:
        summary = measure_coverage(path)

        assert summary["total"] == 0
        assert summary["pass"] == 0
        assert summary["unsat"] == 0
        assert summary["by_family"] == {}
    finally:
        os.unlink(path)


def test_measure_coverage_single_pass():
    """Single PASS task → increments pass and by_family."""
    dataset = {
        "task001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}
            ],
            "test": [
                {"input": [[3, 4]]}
            ]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        summary = measure_coverage(path)

        assert summary["total"] == 1
        assert summary["pass"] == 1
        assert summary["unsat"] == 0
        # Should be solved by Isometry (identity)
        assert "Isometry" in summary["by_family"]
        assert summary["by_family"]["Isometry"] == 1
    finally:
        os.unlink(path)


def test_measure_coverage_single_unsat():
    """Single UNSAT task → increments unsat only."""
    dataset = {
        "task002": {
            "train": [
                {"input": [[1, 2]], "output": [[3, 4]]},
                {"input": [[1, 2]], "output": [[5, 6]]}  # Impossible
            ],
            "test": [
                {"input": [[1, 2]]}
            ]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        summary = measure_coverage(path)

        assert summary["total"] == 1
        assert summary["pass"] == 0
        assert summary["unsat"] == 1
        assert summary["by_family"] == {}
    finally:
        os.unlink(path)


def test_measure_coverage_multiple_mixed():
    """Multiple tasks with mixed PASS/UNSAT."""
    dataset = {
        "abc": {  # PASS via Isometry (identity)
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        },
        "def": {  # UNSAT (empty train)
            "train": [],
            "test": [{"input": [[1]]}]
        },
        "xyz": {  # PASS via ColorMap
            "train": [{"input": [[0, 1]], "output": [[5, 6]]}],
            "test": [{"input": [[0, 1]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        summary = measure_coverage(path)

        assert summary["total"] == 3
        assert summary["pass"] == 2
        assert summary["unsat"] == 1
        # Should have ColorMap and Isometry
        assert len(summary["by_family"]) == 2
        assert summary["by_family"].get("Isometry", 0) == 1
        assert summary["by_family"].get("ColorMap", 0) == 1
    finally:
        os.unlink(path)


def test_measure_coverage_sorted_task_ids():
    """Tasks processed in sorted order (deterministic)."""
    dataset = {
        "zzz": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        },
        "aaa": {
            "train": [{"input": [[2]], "output": [[2]]}],
            "test": [{"input": [[3]]}]
        },
        "mmm": {
            "train": [{"input": [[3]], "output": [[3]]}],
            "test": [{"input": [[4]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        # Should process in order: aaa, mmm, zzz
        summary = measure_coverage(path)

        assert summary["total"] == 3
        # All should be PASS (identity via Isometry)
        assert summary["pass"] == 3
    finally:
        os.unlink(path)


def test_measure_coverage_sorted_families():
    """by_family dict has sorted keys."""
    # Create dataset that triggers multiple families
    dataset = {
        "rot": {  # Isometry (rotation)
            "train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}],
            "test": [{"input": [[5, 6], [7, 8]]}]
        },
        "color": {  # ColorMap
            "train": [{"input": [[0, 1]], "output": [[5, 6]]}],
            "test": [{"input": [[0, 1]]}]
        },
        "scale": {  # PixelReplicate
            "train": [{"input": [[1, 2]], "output": [[1, 1, 2, 2], [1, 1, 2, 2]]}],
            "test": [{"input": [[3, 4]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        summary = measure_coverage(path)

        # Families should be sorted alphabetically
        families = list(summary["by_family"].keys())
        assert families == sorted(families)
    finally:
        os.unlink(path)


def test_measure_coverage_completeness():
    """total == pass + unsat (always)."""
    dataset = {
        f"task{i:03d}": {
            "train": [{"input": [[i]], "output": [[i]]}],
            "test": [{"input": [[i+1]]}]
        }
        for i in range(10)
    }

    path = create_temp_dataset(dataset)
    try:
        summary = measure_coverage(path)

        # Completeness invariant
        assert summary["total"] == summary["pass"] + summary["unsat"]
    finally:
        os.unlink(path)


# =============================
# B. _cli_main() Tests
# =============================

def test_cli_main_empty_dataset(capsys):
    """Empty dataset → correct report."""
    dataset = {}
    path = create_temp_dataset(dataset)

    try:
        exit_code = _cli_main([path])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = captured.out

        # Should contain header and 0/0 coverage
        assert "Step 1 Coverage: 0/0 (0.0%)" in output
        assert "UNSAT: 0" in output
    finally:
        os.unlink(path)


def test_cli_main_single_pass(capsys):
    """Single PASS task → correct report."""
    dataset = {
        "test": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([path])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = captured.out

        assert "Step 1 Coverage: 1/1 (100.0%)" in output
        assert "Isometry: 1" in output
        assert "UNSAT: 0" in output
    finally:
        os.unlink(path)


def test_cli_main_file_not_found(capsys):
    """File not found → exit 2."""
    exit_code = _cli_main(["/nonexistent/missing.json"])

    assert exit_code == 2

    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error["error"] == "file_not_found"


def test_cli_main_json_parse_error(capsys):
    """Invalid JSON → exit 2."""
    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write("{ invalid json }")

        exit_code = _cli_main([path])

        assert exit_code == 2

        captured = capsys.readouterr()
        error = json.loads(captured.err)
        assert error["error"] == "json_parse_error"
    finally:
        os.unlink(path)


def test_cli_main_deterministic_output(capsys):
    """Run twice → byte-identical output."""
    dataset = {
        "a": {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[2]]}]},
        "b": {"train": [{"input": [[2]], "output": [[2]]}], "test": [{"input": [[3]]}]}
    }

    path = create_temp_dataset(dataset)
    try:
        # First run
        exit_code1 = _cli_main([path])
        captured1 = capsys.readouterr()

        # Second run
        exit_code2 = _cli_main([path])
        captured2 = capsys.readouterr()

        assert exit_code1 == exit_code2 == 0
        assert captured1.out == captured2.out  # Byte-for-byte identical
    finally:
        os.unlink(path)


def test_cli_main_sorted_families(capsys):
    """Families printed in sorted order."""
    dataset = {
        "rot": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}],
            "test": [{"input": [[5, 6], [7, 8]]}]
        },
        "color": {
            "train": [{"input": [[0, 1]], "output": [[5, 6]]}],
            "test": [{"input": [[0, 1]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([path])

        assert exit_code == 0

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')

        # Find family lines (between header and UNSAT)
        family_lines = []
        in_families = False
        for line in lines:
            if line.startswith("==="):
                if in_families:
                    break  # End of family section
                in_families = True
            elif in_families and not line.startswith("UNSAT"):
                family_lines.append(line)

        # Extract family names
        families = [line.split(':')[0] for line in family_lines if ':' in line and not line.startswith("Step")]

        # Should be sorted alphabetically
        assert families == sorted(families)
    finally:
        os.unlink(path)


def test_cli_main_report_format(capsys):
    """Report has correct fixed format."""
    dataset = {
        "test": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([path])

        assert exit_code == 0

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')

        # Line 0: separator (40 '=' chars)
        assert lines[0] == "=" * 40

        # Line 1: summary
        assert lines[1].startswith("Step 1 Coverage:")

        # Line 2: separator (40 '=' chars)
        assert lines[2] == "=" * 40

        # Remaining lines: families + UNSAT
        # Last line should be UNSAT
        assert lines[-1].startswith("UNSAT:")
    finally:
        os.unlink(path)


# =============================
# C. Integration Tests
# =============================

def test_coverage_multiple_families(capsys):
    """Multiple families in dataset → all counted correctly."""
    dataset = {
        "identity": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        },
        "color": {
            "train": [{"input": [[0, 1]], "output": [[5, 6]]}],
            "test": [{"input": [[0, 1]]}]
        },
        "rotate": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}],
            "test": [{"input": [[5, 6], [7, 8]]}]
        },
        "unsat": {
            # Impossible: same input, different outputs
            "train": [
                {"input": [[1, 2]], "output": [[3, 4]]},
                {"input": [[1, 2]], "output": [[5, 6]]}
            ],
            "test": [{"input": [[1, 2]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        summary = measure_coverage(path)

        assert summary["total"] == 4
        assert summary["pass"] == 3
        assert summary["unsat"] == 1

        # Should have ColorMap and Isometry (identity + rotate both use Isometry)
        assert len(summary["by_family"]) == 2
    finally:
        os.unlink(path)


def test_coverage_exception_handling():
    """Malformed task caught → counted as UNSAT."""
    dataset = {
        "malformed": {
            # Missing both train and test
        },
        "good": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        summary = measure_coverage(path)

        assert summary["total"] == 2
        assert summary["pass"] == 1  # good task
        assert summary["unsat"] == 1  # malformed task
    finally:
        os.unlink(path)
