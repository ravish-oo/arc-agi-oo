"""
Test suite for P7-05: Coverage Meter Update

Test categories:
- CLI flag tests (3 tests)
- Format validation (2 tests)
- Determinism (2 tests)
- Edge cases (2 tests)
- Aggregation (1 test)

Total: 10 tests
"""

import pytest
import json
import tempfile
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.measure_coverage import measure_coverage, _cli_main


# ============================================================================
# CATEGORY 1: CLI FLAG TESTS (3 tests)
# ============================================================================

def test_cli_default_runs_both_steps(capsys):
    """CLI default runs both steps (--steps 1,2)."""
    # Create minimal temp dataset
    dataset = {
        "test_001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}  # Identity
            ],
            "test": [{"input": [[3, 4]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        # Run without --steps flag (default should be 1,2)
        exit_code = _cli_main([temp_path])

        assert exit_code == 0

        # Capture output
        captured = capsys.readouterr()
        output = captured.out

        # Should contain both Step 1 and Step 2 coverage lines
        assert "Step 1 Coverage:" in output
        assert "Step 2 Coverage:" in output
        assert "---" in output  # Separator between steps
        assert "Step 1 by family:" in output
        assert "Step 2 by P:" in output

    finally:
        import os
        os.unlink(temp_path)


def test_cli_steps_one_only(capsys):
    """--steps 1 runs Step-1 only."""
    dataset = {
        "test_001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}
            ],
            "test": [{"input": [[3, 4]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        exit_code = _cli_main([temp_path, "--steps", "1"])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = captured.out

        # Should contain only Step 1
        assert "Step 1 Coverage:" in output
        assert "Step 2 Coverage:" not in output
        assert "---" not in output
        assert "Step 2 by P:" not in output

    finally:
        import os
        os.unlink(temp_path)


def test_cli_steps_two_only(capsys):
    """--steps 2 runs Step-2 only."""
    dataset = {
        "test_001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}
            ],
            "test": [{"input": [[3, 4]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        exit_code = _cli_main([temp_path, "--steps", "2"])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = captured.out

        # Should contain only Step 2
        assert "Step 1 Coverage:" not in output
        assert "Step 2 Coverage:" in output
        assert "Step 1 by family:" not in output
        assert "Step 2 by P:" in output

    finally:
        import os
        os.unlink(temp_path)


# ============================================================================
# CATEGORY 2: FORMAT VALIDATION (2 tests)
# ============================================================================

def test_format_delta_line_positive(capsys):
    """Delta line format: [+N tasks, +X.X%] when Step-2 > Step-1."""
    # Create dataset where Step-2 solves more than Step-1
    dataset = {
        "test_001": {
            "train": [
                {"input": [[0, 0]], "output": [[1, 2]]}  # Step-1 UNSAT, Step-2 PASS
            ],
            "test": [{"input": [[0, 0]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        exit_code = _cli_main([temp_path, "--steps", "1,2"])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = captured.out

        # Should have positive delta with + sign
        assert "[+" in output
        assert "tasks" in output
        assert "%]" in output

    finally:
        import os
        os.unlink(temp_path)


def test_format_two_step_report_structure(capsys):
    """Two-step report has correct structure and separator."""
    dataset = {
        "test_001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}
            ],
            "test": [{"input": [[3, 4]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        exit_code = _cli_main([temp_path])

        assert exit_code == 0

        captured = capsys.readouterr()
        lines = captured.out.split('\n')

        # Verify structure: header, summaries, separator, breakdowns
        assert "=" * 40 in captured.out
        assert "---" in captured.out
        assert "Step 1 by family:" in captured.out
        assert "Step 2 by P:" in captured.out

    finally:
        import os
        os.unlink(temp_path)


# ============================================================================
# CATEGORY 3: DETERMINISM (2 tests)
# ============================================================================

def test_determinism_identical_runs():
    """Same dataset → same summary dict."""
    dataset = {
        "test_001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}
            ],
            "test": [{"input": [[3, 4]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        summary1 = measure_coverage(temp_path, [1, 2])
        summary2 = measure_coverage(temp_path, [1, 2])

        assert summary1 == summary2
        assert summary1["total"] == summary2["total"]
        if summary1["delta"] is not None:
            assert summary1["delta"]["abs"] == summary2["delta"]["abs"]

    finally:
        import os
        os.unlink(temp_path)


def test_determinism_stable_output(capsys):
    """Repeated runs yield byte-identical output."""
    dataset = {
        "test_001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}
            ],
            "test": [{"input": [[3, 4]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        # Run 1
        _cli_main([temp_path])
        output1 = capsys.readouterr().out

        # Run 2
        _cli_main([temp_path])
        output2 = capsys.readouterr().out

        assert output1 == output2

    finally:
        import os
        os.unlink(temp_path)


# ============================================================================
# CATEGORY 4: EDGE CASES (2 tests)
# ============================================================================

def test_empty_dataset():
    """Empty dataset → all zeros, valid structure."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({}, f)
        temp_path = f.name

    try:
        summary = measure_coverage(temp_path, [1, 2])

        assert summary["total"] == 0
        assert summary["step1"]["pass"] == 0
        assert summary["step1"]["unsat"] == 0
        assert summary["step2"]["pass"] == 0
        assert summary["step2"]["unsat"] == 0
        assert summary["delta"]["abs"] == 0
        assert summary["delta"]["pct"] == 0.0

    finally:
        import os
        os.unlink(temp_path)


def test_bad_steps_value():
    """Bad --steps value → exit 2."""
    dataset = {
        "test_001": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[1]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        exit_code = _cli_main([temp_path, "--steps", "abc"])

        assert exit_code == 2

    finally:
        import os
        os.unlink(temp_path)


# ============================================================================
# CATEGORY 5: AGGREGATION (1 test)
# ============================================================================

def test_step2_aggregation_by_p():
    """Step-2 aggregates by P name correctly."""
    # Create dataset where different P's solve different tasks
    # Note: This is a simplified test; real aggregation depends on actual solver behavior
    dataset = {
        "test_001": {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]}  # Identity
            ],
            "test": [{"input": [[3, 4]]}]
        },
        "test_002": {
            "train": [
                {"input": [[0, 0]], "output": [[1, 2]]}  # Identity + Φ
            ],
            "test": [{"input": [[0, 0]]}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset, f)
        temp_path = f.name

    try:
        summary = measure_coverage(temp_path, [2])  # Step-2 only

        assert summary["step2"] is not None
        assert isinstance(summary["step2"]["by_p"], dict)
        # Keys should be sorted alphabetically
        if summary["step2"]["by_p"]:
            keys = list(summary["step2"]["by_p"].keys())
            assert keys == sorted(keys)

    finally:
        import os
        os.unlink(temp_path)
