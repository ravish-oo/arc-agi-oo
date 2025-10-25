"""
Tests for Step-1 CLI — Phase 3 work order P3-03.

Comprehensive test suite covering:
- Argument parsing (missing args, help)
- File I/O errors (missing file, bad JSON)
- Task ID errors (ID not found)
- Success cases (with/without receipt)
- Determinism (same input → same output)
- JSON formatting (sorted keys, no whitespace)
"""

import pytest
import json
import tempfile
import os
from src.solver_step1 import _cli_main


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
# A. Argument Parsing Tests
# =============================

def test_cli_missing_task_arg(capsys):
    """Missing --task argument → exit 2."""
    exit_code = _cli_main(["--task-id", "abc123"])

    assert exit_code == 2


def test_cli_missing_task_id_arg(capsys):
    """Missing --task-id argument → exit 2."""
    exit_code = _cli_main(["--task", "data.json"])

    assert exit_code == 2


def test_cli_help():
    """--help flag → exit 0."""
    exit_code = _cli_main(["--help"])

    assert exit_code == 0


# =============================
# B. File I/O Error Tests
# =============================

def test_cli_file_not_found(capsys):
    """File not found → exit 2."""
    exit_code = _cli_main([
        "--task", "/nonexistent/missing.json",
        "--task-id", "abc123"
    ])

    assert exit_code == 2
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error["error"] == "file_not_found"


def test_cli_json_parse_error(capsys):
    """Invalid JSON → exit 2."""
    # Create temp file with invalid JSON
    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write("{ invalid json }")

        exit_code = _cli_main([
            "--task", path,
            "--task-id", "abc123"
        ])

        assert exit_code == 2
        captured = capsys.readouterr()
        error = json.loads(captured.err)
        assert error["error"] == "json_parse_error"
    finally:
        os.unlink(path)


# =============================
# C. Task ID Error Tests
# =============================

def test_cli_task_id_not_found(capsys):
    """Task ID not in dataset → exit 3."""
    dataset = {
        "existing_task": {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test": [{"input": [[3, 4]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "nonexistent_id"
        ])

        assert exit_code == 3
        captured = capsys.readouterr()
        error = json.loads(captured.err)
        assert error["error"] == "task_id_not_found"
    finally:
        os.unlink(path)


# =============================
# D. Malformed Task Tests
# =============================

def test_cli_malformed_task_missing_train(capsys):
    """Task missing 'train' key → exit 4."""
    dataset = {
        "malformed": {
            "test": [{"input": [[1, 2]]}]
            # Missing "train"
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "malformed"
        ])

        assert exit_code == 4
        captured = capsys.readouterr()
        error = json.loads(captured.err)
        assert error["error"] == "malformed_task"
    finally:
        os.unlink(path)


def test_cli_malformed_task_missing_test(capsys):
    """Task missing 'test' key → exit 4."""
    dataset = {
        "malformed": {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}]
            # Missing "test"
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "malformed"
        ])

        assert exit_code == 4
        captured = capsys.readouterr()
        error = json.loads(captured.err)
        assert error["error"] == "malformed_task"
    finally:
        os.unlink(path)


# =============================
# E. Success Cases
# =============================

def test_cli_single_task_success(capsys):
    """Valid task, solved successfully → exit 0."""
    dataset = {
        "identity_task": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]}
            ],
            "test": [
                {"input": [[9, 0], [1, 2]]}
            ]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "identity_task"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "task_id" in output
        assert output["task_id"] == "identity_task"
        assert "predictions" in output
        assert len(output["predictions"]) == 1
        assert output["predictions"][0] == [[9, 0], [1, 2]]
        assert "receipt" not in output  # No --print-receipt
    finally:
        os.unlink(path)


def test_cli_with_receipt(capsys):
    """With --print-receipt flag → receipt included."""
    dataset = {
        "test_task": {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test": [{"input": [[3, 4]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "test_task",
            "--print-receipt"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "task_id" in output
        assert "predictions" in output
        assert "receipt" in output  # --print-receipt enabled
        assert output["receipt"]["mode"] in ["global", "unsat"]
    finally:
        os.unlink(path)


def test_cli_without_receipt(capsys):
    """Without --print-receipt flag → receipt omitted."""
    dataset = {
        "test_task": {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test": [{"input": [[3, 4]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "test_task"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "task_id" in output
        assert "predictions" in output
        assert "receipt" not in output  # No --print-receipt
    finally:
        os.unlink(path)


def test_cli_unsat_task(capsys):
    """Task that cannot be solved → UNSAT, predictions omitted."""
    dataset = {
        "impossible": {
            "train": [
                {"input": [[1, 2]], "output": [[3, 4]]},
                {"input": [[1, 2]], "output": [[5, 6]]}  # Same input, different output
            ],
            "test": [{"input": [[1, 2]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "impossible"
        ])

        assert exit_code == 0  # Still success (UNSAT is a valid result)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "task_id" in output
        # predictions should be omitted when None (or included as null - we omit)
        assert "predictions" not in output
    finally:
        os.unlink(path)


# =============================
# F. Determinism Tests
# =============================

def test_cli_deterministic_output(capsys):
    """Run twice → identical outputs."""
    dataset = {
        "det_task": {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test": [{"input": [[3, 4]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        # First run
        exit_code1 = _cli_main([
            "--task", path,
            "--task-id", "det_task"
        ])
        captured1 = capsys.readouterr()

        # Second run
        exit_code2 = _cli_main([
            "--task", path,
            "--task-id", "det_task"
        ])
        captured2 = capsys.readouterr()

        assert exit_code1 == exit_code2 == 0
        assert captured1.out == captured2.out  # Byte-for-byte identical
    finally:
        os.unlink(path)


# =============================
# G. JSON Formatting Tests
# =============================

def test_cli_sorted_keys(capsys):
    """Output JSON has sorted keys."""
    dataset = {
        "format_task": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "format_task",
            "--print-receipt"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output_str = captured.out.strip()

        # Parse and verify
        output = json.loads(output_str)

        # Re-serialize with sorted keys
        expected = json.dumps(output, sort_keys=True, separators=(',', ':'))

        assert output_str == expected
    finally:
        os.unlink(path)


def test_cli_no_whitespace(capsys):
    """Output JSON has no extra whitespace (compact format)."""
    dataset = {
        "ws_task": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "ws_task"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output_str = captured.out.strip()

        # Should not contain ': ' or ', ' (compact separators)
        assert ': ' not in output_str
        # Note: ',' will appear in arrays/objects but without space after
        # The separators=(',', ':') ensures no spaces
    finally:
        os.unlink(path)


def test_cli_valid_json(capsys):
    """Output is valid JSON."""
    dataset = {
        "json_task": {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test": [{"input": [[3, 4]]}]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "json_task"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output_str = captured.out.strip()

        # Should parse without error
        output = json.loads(output_str)
        assert isinstance(output, dict)
    finally:
        os.unlink(path)


# =============================
# H. Integration Tests
# =============================

def test_cli_rotation_task(capsys):
    """Real rotation task → correct predictions."""
    dataset = {
        "rot90": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
                {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]}
            ],
            "test": [
                {"input": [[9, 0], [1, 2]]}
            ]
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "rot90",
            "--print-receipt"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["task_id"] == "rot90"
        assert "predictions" in output
        assert output["predictions"] == [[[1, 9], [2, 0]]]
        assert output["receipt"]["mode"] == "global"
        assert output["receipt"]["solver"] == "Isometry"
    finally:
        os.unlink(path)


def test_cli_empty_test(capsys):
    """Task with empty test → empty predictions."""
    dataset = {
        "empty_test": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": []
        }
    }

    path = create_temp_dataset(dataset)
    try:
        exit_code = _cli_main([
            "--task", path,
            "--task-id", "empty_test"
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["task_id"] == "empty_test"
        assert "predictions" in output
        assert output["predictions"] == []
    finally:
        os.unlink(path)
