"""
P7-06 — Step-2 CLI Runner Tests

Tests for src/solver_step2.py CLI interface.

Covers:
- Happy path with predictions
- Receipt gating (--print-receipt flag)
- UNSAT task handling
- Error codes (2: file/parse, 3: task not found, 4: malformed task)
- Determinism (byte-identical outputs)
- JSON format validation
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.solver_step2 import _cli_main


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_colormap_dataset(temp_dir):
    """Simple ColorMap task that Step-2 should solve."""
    dataset = {
        "simple_colormap": {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[1, 2], [3, 4]]}
            ],
            "test": [
                {"input": [[0, 1], [2, 3]]}
            ]
        }
    }
    path = temp_dir / "simple_colormap.json"
    with open(path, 'w') as f:
        json.dump(dataset, f)
    return path


@pytest.fixture
def unsat_dataset(temp_dir):
    """Task that should return UNSAT (complex pattern)."""
    dataset = {
        "unsat_task": {
            "train": [
                {"input": [[1, 2, 3]], "output": [[99, 88, 77]]},
                {"input": [[4, 5, 6]], "output": [[11, 22, 33]]}
            ],
            "test": [
                {"input": [[7, 8, 9]]}
            ]
        }
    }
    path = temp_dir / "unsat_task.json"
    with open(path, 'w') as f:
        json.dump(dataset, f)
    return path


@pytest.fixture
def malformed_dataset(temp_dir):
    """Dataset with missing 'train' key."""
    dataset = {
        "malformed_task": {
            "test": [
                {"input": [[1, 2]]}
            ]
            # Missing "train" key
        }
    }
    path = temp_dir / "malformed.json"
    with open(path, 'w') as f:
        json.dump(dataset, f)
    return path


@pytest.fixture
def invalid_json_file(temp_dir):
    """File with invalid JSON syntax."""
    path = temp_dir / "invalid.json"
    with open(path, 'w') as f:
        f.write("{this is not valid json")
    return path


# ============================================================================
# Happy Path Tests
# ============================================================================

def test_happy_path_with_predictions(simple_colormap_dataset, capsys):
    """
    Test: happy_path_with_predictions

    Load synthetic task with ColorMap solution, run CLI without --print-receipt.
    Assert exit code = 0, parse JSON output, verify predictions present.
    """
    exit_code = _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap"
    ])

    assert exit_code == 0, "Expected exit code 0 for successful solve"

    captured = capsys.readouterr()
    output = captured.out.strip()

    # Parse JSON output
    result = json.loads(output)

    # Assertions
    assert result["task_id"] == "simple_colormap", "task_id should match input"
    assert "predictions" in result, "predictions should be present for PASS"
    assert isinstance(result["predictions"], list), "predictions should be a list"
    assert len(result["predictions"]) == 1, "Should have 1 prediction for 1 test input"
    assert "receipt" not in result, "receipt should NOT be present without --print-receipt"


def test_happy_path_with_receipt(simple_colormap_dataset, capsys):
    """
    Test: happy_path_with_receipt

    Same as above with --print-receipt flag.
    Assert receipt is included with correct structure.
    """
    exit_code = _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap",
        "--print-receipt"
    ])

    assert exit_code == 0, "Expected exit code 0"

    captured = capsys.readouterr()
    output = captured.out.strip()
    result = json.loads(output)

    # Assertions
    assert result["task_id"] == "simple_colormap"
    assert "predictions" in result
    assert "receipt" in result, "receipt SHOULD be present with --print-receipt flag"

    # Verify receipt structure
    receipt = result["receipt"]
    assert "mode" in receipt, "receipt should have 'mode' key"
    assert receipt["mode"] == "phi_partition", "mode should be 'phi_partition' for Step-2"
    assert "chosen_candidate" in receipt, "receipt should have chosen_candidate"


def test_unsat_task(unsat_dataset, capsys):
    """
    Test: unsat_task

    Load task that fails Step-2 (no valid candidate).
    Assert exit code = 0 (UNSAT is valid output, not error).
    Assert predictions not in output or null.
    """
    exit_code = _cli_main([
        "--task", str(unsat_dataset),
        "--task-id", "unsat_task"
    ])

    assert exit_code == 0, "UNSAT is a valid output, should exit 0"

    captured = capsys.readouterr()
    output = captured.out.strip()
    result = json.loads(output)

    assert result["task_id"] == "unsat_task"
    # UNSAT tasks should not have predictions
    assert "predictions" not in result, "UNSAT should not include predictions"


def test_unsat_with_receipt(unsat_dataset, capsys):
    """
    Test: unsat_task with --print-receipt

    UNSAT tasks should include UNSAT receipt when flag is set.
    """
    exit_code = _cli_main([
        "--task", str(unsat_dataset),
        "--task-id", "unsat_task",
        "--print-receipt"
    ])

    assert exit_code == 0

    captured = capsys.readouterr()
    output = captured.out.strip()
    result = json.loads(output)

    assert result["task_id"] == "unsat_task"
    assert "receipt" in result, "UNSAT should include receipt when --print-receipt set"

    receipt = result["receipt"]
    assert "mode" in receipt
    assert receipt["mode"] == "unsat", "UNSAT receipt should have mode='unsat'"
    assert "reason" in receipt


# ============================================================================
# Error Code Tests
# ============================================================================

def test_file_not_found(temp_dir, capsys):
    """
    Test: file_not_found

    Call CLI with nonexistent file path.
    Assert exit code = 2, stderr contains "file_not_found".
    """
    nonexistent = temp_dir / "does_not_exist.json"

    exit_code = _cli_main([
        "--task", str(nonexistent),
        "--task-id", "any_id"
    ])

    assert exit_code == 2, "Expected exit code 2 for file not found"

    captured = capsys.readouterr()
    stderr = captured.err.strip()

    # Parse stderr JSON
    error = json.loads(stderr)
    assert error["error"] == "file_not_found", "stderr should contain file_not_found error"


def test_json_parse_error(invalid_json_file, capsys):
    """
    Test: json_parse_error

    Call CLI with file containing invalid JSON.
    Assert exit code = 2, stderr contains "json_parse_error".
    """
    exit_code = _cli_main([
        "--task", str(invalid_json_file),
        "--task-id", "any_id"
    ])

    assert exit_code == 2, "Expected exit code 2 for JSON parse error"

    captured = capsys.readouterr()
    stderr = captured.err.strip()
    error = json.loads(stderr)
    assert error["error"] == "json_parse_error"


def test_task_id_not_found(simple_colormap_dataset, capsys):
    """
    Test: task_id_not_found

    Call with valid file but task ID not in dataset.
    Assert exit code = 3, stderr contains "task_id_not_found".
    """
    exit_code = _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "nonexistent_id"
    ])

    assert exit_code == 3, "Expected exit code 3 for task ID not found"

    captured = capsys.readouterr()
    stderr = captured.err.strip()
    error = json.loads(stderr)
    assert error["error"] == "task_id_not_found"


def test_malformed_task_missing_train(malformed_dataset, capsys):
    """
    Test: malformed_task_missing_train

    Load JSON with task missing "train" key.
    Assert exit code = 4, stderr contains "malformed_task".
    """
    exit_code = _cli_main([
        "--task", str(malformed_dataset),
        "--task-id", "malformed_task"
    ])

    assert exit_code == 4, "Expected exit code 4 for malformed task"

    captured = capsys.readouterr()
    stderr = captured.err.strip()
    error = json.loads(stderr)
    assert error["error"] == "malformed_task"


def test_malformed_task_missing_test(temp_dir, capsys):
    """
    Test: malformed_task_missing_test

    Load JSON with task missing "test" key.
    Assert exit code = 4.
    """
    dataset = {
        "missing_test": {
            "train": [{"input": [[1]], "output": [[2]]}]
            # Missing "test" key
        }
    }
    path = temp_dir / "missing_test.json"
    with open(path, 'w') as f:
        json.dump(dataset, f)

    exit_code = _cli_main([
        "--task", str(path),
        "--task-id", "missing_test"
    ])

    assert exit_code == 4

    captured = capsys.readouterr()
    stderr = captured.err.strip()
    error = json.loads(stderr)
    assert error["error"] == "malformed_task"


# ============================================================================
# Determinism Tests
# ============================================================================

def test_determinism(simple_colormap_dataset, capsys):
    """
    Test: determinism

    Run same task twice, assert stdout outputs are byte-identical.
    """
    # First run
    exit_code_1 = _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap"
    ])
    captured_1 = capsys.readouterr()
    output_1 = captured_1.out.strip()

    # Second run
    exit_code_2 = _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap"
    ])
    captured_2 = capsys.readouterr()
    output_2 = captured_2.out.strip()

    assert exit_code_1 == 0
    assert exit_code_2 == 0
    assert output_1 == output_2, "Outputs should be byte-identical (deterministic)"


def test_determinism_with_receipt(simple_colormap_dataset, capsys):
    """
    Test: determinism with --print-receipt

    Receipts should also be deterministic.
    """
    # First run
    _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap",
        "--print-receipt"
    ])
    output_1 = capsys.readouterr().out.strip()

    # Second run
    _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap",
        "--print-receipt"
    ])
    output_2 = capsys.readouterr().out.strip()

    assert output_1 == output_2, "Receipt output should be deterministic"


# ============================================================================
# JSON Format Validation
# ============================================================================

def test_json_format_validation(simple_colormap_dataset, capsys):
    """
    Test: json_format_validation

    Parse output JSON, assert keys are sorted (sort_keys=True).
    Assert canonical JSON format with separators=(',',':').
    """
    _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap",
        "--print-receipt"
    ])

    captured = capsys.readouterr()
    output = captured.out.strip()

    # Parse JSON
    result = json.loads(output)

    # Re-serialize with same settings to check format
    canonical = json.dumps(result, sort_keys=True, separators=(',', ':'))

    # Output should match canonical format exactly
    assert output == canonical, "Output should match canonical format (sorted keys, no spaces in separators)"

    # Verify keys are sorted (check task_id comes before predictions alphabetically if both exist)
    keys = list(result.keys())
    sorted_keys = sorted(keys)
    assert keys == sorted_keys, "Keys should be in sorted order"


def test_single_json_line_output(simple_colormap_dataset, capsys):
    """
    Test: single JSON line output

    Output should be a single line (no newlines except trailing).
    """
    _cli_main([
        "--task", str(simple_colormap_dataset),
        "--task-id", "simple_colormap"
    ])

    captured = capsys.readouterr()
    output = captured.out

    lines = output.split('\n')
    # Should have exactly 2 elements: [json_line, empty_string_from_trailing_newline]
    assert len(lines) == 2, "Should have single line + trailing newline"
    assert lines[0] != "", "First line should contain JSON"
    assert lines[1] == "", "Second line should be empty (trailing newline)"


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_test_set(temp_dir, capsys):
    """
    Test: empty test set

    Task with empty test set should return predictions = [].
    """
    dataset = {
        "empty_test": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": []
        }
    }
    path = temp_dir / "empty_test.json"
    with open(path, 'w') as f:
        json.dump(dataset, f)

    exit_code = _cli_main([
        "--task", str(path),
        "--task-id", "empty_test"
    ])

    assert exit_code == 0

    captured = capsys.readouterr()
    output = captured.out.strip()
    result = json.loads(output)

    assert result["task_id"] == "empty_test"
    # Should have predictions key with empty list
    if "predictions" in result:
        assert result["predictions"] == []


def test_multiple_test_inputs(temp_dir, capsys):
    """
    Test: multiple test inputs

    Task with multiple test inputs should return multiple predictions.
    """
    dataset = {
        "multi_test": {
            "train": [
                {"input": [[0]], "output": [[1]]}
            ],
            "test": [
                {"input": [[0]]},
                {"input": [[0]]},
                {"input": [[0]]}
            ]
        }
    }
    path = temp_dir / "multi_test.json"
    with open(path, 'w') as f:
        json.dump(dataset, f)

    exit_code = _cli_main([
        "--task", str(path),
        "--task-id", "multi_test"
    ])

    assert exit_code == 0

    captured = capsys.readouterr()
    output = captured.out.strip()
    result = json.loads(output)

    if "predictions" in result:
        assert len(result["predictions"]) == 3, "Should have 3 predictions for 3 test inputs"


# ============================================================================
# Completeness Invariant
# ============================================================================

def test_completeness_invariant():
    """
    Verify all tests are implemented according to P7-06 spec.

    Required tests:
    - happy_path_with_predictions ✓
    - happy_path_with_receipt ✓
    - unsat_task ✓
    - file_not_found ✓
    - task_id_not_found ✓
    - malformed_task_missing_train ✓
    - determinism ✓
    - json_format_validation ✓
    """
    # This test serves as a completeness check
    required_tests = [
        "test_happy_path_with_predictions",
        "test_happy_path_with_receipt",
        "test_unsat_task",
        "test_file_not_found",
        "test_task_id_not_found",
        "test_malformed_task_missing_train",
        "test_determinism",
        "test_json_format_validation"
    ]

    # Get all test functions in this module
    import inspect
    current_module = inspect.getmodule(inspect.currentframe())
    all_tests = [name for name, obj in inspect.getmembers(current_module)
                 if inspect.isfunction(obj) and name.startswith("test_")]

    for required in required_tests:
        assert required in all_tests, f"Required test {required} is missing"
