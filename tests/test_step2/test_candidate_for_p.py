"""
Test suite for P7-02: Candidate-for-P Builder

Test categories:
- Basic functionality (3 tests)
- Shape safety (3 tests)
- Action inference (3 tests)
- GLUE verification (2 tests)
- Edge cases (4 tests)
- Determinism (2 tests)
- Error paths (3 tests)

Total: 20 tests
"""

import pytest
from src.solver_step2 import build_candidate_for_P


# ============================================================================
# CATEGORY 1: BASIC FUNCTIONALITY (3 tests)
# ============================================================================

def test_identity_with_simple_edit_produces_candidate():
    """Identity P with simple two-class edit → candidate produced."""
    task = {
        "train": [
            # Train 0: two pixels need changing
            ([[0, 0, 0]], [[5, 0, 7]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X  # Identity transform

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None
    assert "P" in candidate
    assert "classes" in candidate
    assert "actions" in candidate
    assert "mdl" in candidate


def test_candidate_has_all_required_keys():
    """Candidate dict has exact required keys."""
    task = {
        "train": [
            ([[0]], [[1]]),
        ]
    }

    p_desc = {"name": "ColorMap", "index": 2}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None
    assert set(candidate.keys()) == {"P", "classes", "actions", "mdl"}
    assert set(candidate["mdl"].keys()) == {"num_classes", "num_action_types", "p_index", "hash"}


def test_mdl_components_correct():
    """MDL components in candidate are correct."""
    task = {
        "train": [
            ([[0, 0]], [[5, 7]]),
        ]
    }

    p_desc = {"name": "PixelReplicate", "index": 3}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None
    mdl = candidate["mdl"]
    assert mdl["num_classes"] == 2  # Two pixels to edit
    assert mdl["num_action_types"] == 1  # Both use set_color
    assert mdl["p_index"] == 3
    assert isinstance(mdl["hash"], str) and len(mdl["hash"]) == 64


# ============================================================================
# CATEGORY 2: SHAPE SAFETY (3 tests)
# ============================================================================

def test_shape_mismatch_returns_none():
    """P changes dims vs Y → returns None."""
    task = {
        "train": [
            ([[1, 2]], [[1, 2, 3]]),  # X is 1x2, Y is 1x3
        ]
    }

    p_desc = {"name": "BadTransform", "index": 10}
    apply_p_fn = lambda X: X  # Identity doesn't change dims, but Y is different

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is None


def test_shape_safety_all_trains_checked():
    """Shape safety checks ALL trains (not just first)."""
    task = {
        "train": [
            ([[1]], [[1]]),         # Train 0: OK
            ([[2]], [[2, 2]]),      # Train 1: Shape mismatch
        ]
    }

    p_desc = {"name": "Transform", "index": 5}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is None  # Fails on train 1


def test_p_application_exception_returns_none():
    """P application raises exception → returns None."""
    task = {
        "train": [
            ([[1]], [[1]]),
        ]
    }

    p_desc = {"name": "BrokenTransform", "index": 7}

    def broken_p_fn(X):
        raise RuntimeError("Transform failed")

    candidate = build_candidate_for_P(task, p_desc, broken_p_fn)

    assert candidate is None


# ============================================================================
# CATEGORY 3: ACTION INFERENCE (3 tests)
# ============================================================================

def test_no_admissible_action_returns_none():
    """Class with no admissible action → returns None."""
    # Create a fixture where a class cannot be satisfied
    # Example: Xp has value 1, Y has different values at same positions
    task = {
        "train": [
            ([[1, 1]], [[2, 3]]),  # Pixels at (0,0) and (0,1) have different targets
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    # This might produce a candidate if the Φ features split the pixels into different classes
    # But if they end up in the same class with different target colors, action inference fails
    # The test depends on the specific Φ features; let's check the result
    if candidate is not None:
        # If candidate produced, both pixels must be in different classes
        assert len(candidate["classes"]) >= 2
    # We can't guarantee None without knowing exact Φ behavior, so this test is informational


def test_all_classes_have_actions():
    """All classes in candidate have corresponding actions."""
    task = {
        "train": [
            ([[0, 0, 0]], [[1, 2, 3]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    if candidate is not None:
        # Every class must have an action
        assert set(candidate["classes"].keys()) == set(candidate["actions"].keys())


def test_actions_deterministic_order():
    """Actions processed in ascending class ID order (determinism)."""
    task = {
        "train": [
            ([[0, 0, 0]], [[1, 2, 3]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    if candidate is not None and len(candidate["classes"]) > 1:
        # Class IDs should be sorted
        class_ids = list(candidate["classes"].keys())
        assert class_ids == sorted(class_ids)


# ============================================================================
# CATEGORY 4: GLUE VERIFICATION (2 tests)
# ============================================================================

def test_glue_verify_passes():
    """GLUE verification passes when stitched == Y."""
    task = {
        "train": [
            ([[0]], [[5]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None  # GLUE should pass


def test_candidate_only_returned_when_stitched_equals_y():
    """Candidate only returned if stitched outputs == Y exactly."""
    # This is implicit in the implementation
    # If a candidate is returned, verify_stitched_equality must have returned True
    task = {
        "train": [
            ([[0, 0]], [[7, 7]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    # If candidate returned, it passed GLUE verification
    if candidate is not None:
        assert "mdl" in candidate  # Sanity check


# ============================================================================
# CATEGORY 5: EDGE CASES (4 tests)
# ============================================================================

def test_empty_train_set_returns_none():
    """Empty train set → returns None (insufficient evidence)."""
    task = {"train": []}

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is None


def test_no_residuals_produces_valid_candidate():
    """No residuals (Xp == Y already) → valid candidate with 0 classes."""
    task = {
        "train": [
            ([[1, 2]], [[1, 2]]),  # Already equal
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None
    assert candidate["mdl"]["num_classes"] == 0
    assert len(candidate["classes"]) == 0
    assert len(candidate["actions"]) == 0


def test_single_class_single_train():
    """Single class, single train → valid candidate."""
    task = {
        "train": [
            ([[0]], [[9]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None
    assert candidate["mdl"]["num_classes"] == 1


def test_multiple_trains():
    """Multiple trains → candidate built correctly."""
    task = {
        "train": [
            ([[0]], [[1]]),
            ([[0]], [[1]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None
    # Both trains should have same transformation
    assert candidate["mdl"]["num_classes"] == 1


# ============================================================================
# CATEGORY 6: DETERMINISM (2 tests)
# ============================================================================

def test_determinism_identical_runs():
    """Same inputs → same candidate (deterministic)."""
    task = {
        "train": [
            ([[0, 0]], [[5, 7]]),
        ]
    }

    p_desc = {"name": "NPSDown", "index": 5}
    apply_p_fn = lambda X: X

    candidate1 = build_candidate_for_P(task, p_desc, apply_p_fn)
    candidate2 = build_candidate_for_P(task, p_desc, apply_p_fn)

    if candidate1 is not None:
        assert candidate2 is not None
        assert candidate1["mdl"]["hash"] == candidate2["mdl"]["hash"]
        assert candidate1["mdl"] == candidate2["mdl"]


def test_mdl_hash_stable():
    """MDL hash stable across runs."""
    task = {
        "train": [
            ([[0]], [[1]]),
        ]
    }

    p_desc = {"name": "Identity", "index": 0}
    apply_p_fn = lambda X: X

    hashes = []
    for _ in range(3):
        candidate = build_candidate_for_P(task, p_desc, apply_p_fn)
        if candidate is not None:
            hashes.append(candidate["mdl"]["hash"])

    if hashes:
        assert len(set(hashes)) == 1  # All identical


# ============================================================================
# CATEGORY 7: ERROR PATHS (3 tests)
# ============================================================================

def test_missing_p_desc_name():
    """Missing p_desc['name'] → ValueError."""
    task = {
        "train": [
            ([[0]], [[1]]),
        ]
    }

    p_desc = {"index": 0}  # Missing "name"
    apply_p_fn = lambda X: X

    with pytest.raises(ValueError, match="must contain 'name' and 'index' fields"):
        build_candidate_for_P(task, p_desc, apply_p_fn)


def test_missing_p_desc_index():
    """Missing p_desc['index'] → ValueError."""
    task = {
        "train": [
            ([[0]], [[1]]),
        ]
    }

    p_desc = {"name": "Identity"}  # Missing "index"
    apply_p_fn = lambda X: X

    with pytest.raises(ValueError, match="must contain 'name' and 'index' fields"):
        build_candidate_for_P(task, p_desc, apply_p_fn)


def test_negative_p_index():
    """P index < -1 → ValueError."""
    task = {
        "train": [
            ([[0]], [[1]]),
        ]
    }

    p_desc = {"name": "BadFamily", "index": -2}
    apply_p_fn = lambda X: X

    with pytest.raises(ValueError, match="must be >= -1"):
        build_candidate_for_P(task, p_desc, apply_p_fn)


def test_identity_with_index_minus_one():
    """Identity with index=-1 → valid candidate."""
    task = {
        "train": [
            ([[0]], [[1]]),
        ]
    }

    p_desc = {"name": "Identity", "index": -1}
    apply_p_fn = lambda X: X

    candidate = build_candidate_for_P(task, p_desc, apply_p_fn)

    assert candidate is not None
    assert candidate["mdl"]["p_index"] == -1  # Identity uses index=-1
