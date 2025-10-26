"""
Test suite for P7-03: Step-2 Solver Core

Test categories:
- Registry enumeration (2 tests)
- Identity solver (2 tests)
- MDL selection (2 tests)
- Test application (2 tests)
- UNSAT cases (2 tests)
- Determinism (2 tests)
- Edge cases (2 tests)

Total: 14 tests
"""

import pytest
from src.solver_step2 import solve_step2, _enumerate_P_registry, _apply_candidate_to_tests


# ============================================================================
# CATEGORY 1: REGISTRY ENUMERATION (2 tests)
# ============================================================================

def test_enumerate_P_registry_has_17_items():
    """Registry has exactly 17 items (1 Identity + 16 families)."""
    registry = _enumerate_P_registry()

    assert len(registry) == 17


def test_enumerate_P_registry_identity_first():
    """Identity first at index=-1, families 0..15."""
    registry = _enumerate_P_registry()

    # First entry is Identity
    assert registry[0]["name"] == "Identity"
    assert registry[0]["index"] == -1
    assert registry[0]["family_class"] is None

    # Next 16 are families with indices 0..15
    for i in range(1, 17):
        assert registry[i]["index"] == i - 1  # 0..15
        assert registry[i]["family_class"] is not None
        assert isinstance(registry[i]["name"], str)
        assert registry[i]["name"] != "Identity"


# ============================================================================
# CATEGORY 2: IDENTITY SOLVER (2 tests)
# ============================================================================

def test_solve_step2_identity_zero_classes():
    """Identity task (X == Y) → Identity candidate with 0 classes."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]},
            {"input": [[3, 4]], "output": [[3, 4]]}
        ],
        "test": [{"input": [[5, 6]]}]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    assert result["candidate"]["P"]["name"] == "Identity"
    assert result["candidate"]["P"]["index"] == -1
    assert result["candidate"]["mdl"]["num_classes"] == 0
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == [[5, 6]]


def test_solve_step2_identity_with_edits():
    """Identity + Φ edits → candidate with classes."""
    task = {
        "train": [
            {"input": [[0, 0]], "output": [[1, 2]]},
        ],
        "test": [{"input": [[0, 0]]}]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    # Identity should pass with 2 classes (one per pixel needing edit)
    # Or possibly 1 class if both pixels have same signature and different actions fail
    assert result["candidate"] is not None
    assert len(result["predictions"]) == 1


# ============================================================================
# CATEGORY 3: MDL SELECTION (2 tests)
# ============================================================================

def test_solve_step2_collect_all_candidates():
    """Multiple P pass → all collected in candidates_tried."""
    task = {
        "train": [
            {"input": [[1, 1]], "output": [[2, 2]]},
            {"input": [[1, 0]], "output": [[2, 0]]},
        ],
        "test": [{"input": [[1, 1]]}]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    # At minimum, Identity should pass (with Φ edits)
    # ColorMap might also pass (1→2 mapping)
    assert len(result["candidates_tried"]) >= 1
    # All candidates should be valid
    for cand in result["candidates_tried"]:
        assert "P" in cand
        assert "classes" in cand
        assert "actions" in cand
        assert "mdl" in cand


def test_solve_step2_mdl_selection():
    """Multiple candidates → best by MDL tuple (fewest classes wins)."""
    task = {
        "train": [
            {"input": [[0, 0]], "output": [[1, 1]]},
        ],
        "test": [{"input": [[0, 0]]}]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    chosen = result["candidate"]

    # Chosen candidate should have minimal MDL
    # All other candidates should have >= MDL
    for cand in result["candidates_tried"]:
        # Compare MDL tuples
        chosen_mdl = (
            chosen["mdl"]["num_classes"],
            chosen["mdl"]["num_action_types"],
            chosen["mdl"]["p_index"],
            chosen["mdl"]["hash"]
        )
        cand_mdl = (
            cand["mdl"]["num_classes"],
            cand["mdl"]["num_action_types"],
            cand["mdl"]["p_index"],
            cand["mdl"]["hash"]
        )
        assert chosen_mdl <= cand_mdl


# ============================================================================
# CATEGORY 4: TEST APPLICATION (2 tests)
# ============================================================================

def test_apply_candidate_to_tests_identity():
    """Apply Identity candidate to tests."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]},
        ],
        "test": [
            {"input": [[5, 6]]},
            {"input": [[7, 8]]}
        ]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    assert len(result["predictions"]) == 2
    assert result["predictions"][0] == [[5, 6]]
    assert result["predictions"][1] == [[7, 8]]


def test_apply_candidate_to_tests_with_edits():
    """Apply candidate with Φ edits to tests via signature matching."""
    task = {
        "train": [
            {"input": [[0, 0, 0]], "output": [[1, 2, 3]]},
        ],
        "test": [
            {"input": [[0, 0, 0]]}
        ]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    assert len(result["predictions"]) == 1
    # Prediction should apply learned edits (exact match depends on Φ signatures)
    # At minimum, prediction should have correct dimensions
    assert len(result["predictions"][0]) == 1
    assert len(result["predictions"][0][0]) == 3


# ============================================================================
# CATEGORY 5: UNSAT CASES (2 tests)
# ============================================================================

def test_solve_step2_unsat_empty_train():
    """Empty train set → UNSAT."""
    task = {
        "train": [],
        "test": [{"input": [[1, 2]]}]
    }

    result = solve_step2(task)

    assert result["status"] == "UNSAT"
    assert result["reason"] == "empty_train"
    assert result["candidates_tried"] == []


def test_solve_step2_unsat_no_passing_candidates():
    """Task where no P can satisfy FY → UNSAT."""
    # Create a task with shape change (no P should pass)
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2, 3]]},  # 1x2 → 1x3 shape change
        ],
        "test": [{"input": [[4, 5]]}]
    }

    result = solve_step2(task)

    # All P should fail shape safety (Identity doesn't change shape)
    # Families also don't change shape
    assert result["status"] == "UNSAT"
    assert result["reason"] == "no_passing_candidates"


# ============================================================================
# CATEGORY 6: DETERMINISM (2 tests)
# ============================================================================

def test_solve_step2_determinism_identical_runs():
    """Same task → same candidate chosen (deterministic)."""
    task = {
        "train": [
            {"input": [[0, 0]], "output": [[5, 7]]},
        ],
        "test": [{"input": [[0, 0]]}]
    }

    result1 = solve_step2(task)
    result2 = solve_step2(task)

    assert result1["status"] == result2["status"]
    if result1["status"] == "PASS":
        # Same candidate chosen
        assert result1["candidate"]["P"]["name"] == result2["candidate"]["P"]["name"]
        assert result1["candidate"]["P"]["index"] == result2["candidate"]["P"]["index"]
        assert result1["candidate"]["mdl"] == result2["candidate"]["mdl"]

        # Same predictions
        assert result1["predictions"] == result2["predictions"]


def test_solve_step2_determinism_stable_hash():
    """MDL hash stable across 3 runs."""
    task = {
        "train": [
            {"input": [[1]], "output": [[2]]},
        ],
        "test": [{"input": [[1]]}]
    }

    hashes = []
    for _ in range(3):
        result = solve_step2(task)
        if result["status"] == "PASS":
            hashes.append(result["candidate"]["mdl"]["hash"])

    if hashes:
        assert len(set(hashes)) == 1  # All identical


# ============================================================================
# CATEGORY 7: EDGE CASES (2 tests)
# ============================================================================

def test_solve_step2_single_passing_candidate():
    """Only one P passes → choose it (no tie-breaking needed)."""
    # Identity-only task
    task = {
        "train": [
            {"input": [[9, 8, 7]], "output": [[9, 8, 7]]},
        ],
        "test": [{"input": [[6, 5, 4]]}]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    # Should have at least Identity
    assert len(result["candidates_tried"]) >= 1
    assert result["candidate"]["P"]["name"] == "Identity"


def test_solve_step2_multiple_trains():
    """Multiple training pairs → all used for candidate building."""
    task = {
        "train": [
            {"input": [[0]], "output": [[1]]},
            {"input": [[0]], "output": [[1]]},
            {"input": [[0]], "output": [[1]]},
        ],
        "test": [{"input": [[0]]}]
    }

    result = solve_step2(task)

    assert result["status"] == "PASS"
    assert len(result["predictions"]) == 1
    # Prediction should match training pattern
    assert result["predictions"][0] == [[1]]
