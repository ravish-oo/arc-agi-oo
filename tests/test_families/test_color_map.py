"""
Test suite for ColorMapFamily (P2-02).

Tests cover:
- Basic fit/apply workflow
- Unified mapping requirement (critical: ONE mapping for ALL pairs)
- Conflict detection (same pair and across pairs)
- FY exactness (bit-for-bit equality)
- Unseen color handling (KeyError)
- Edge cases (empty grids, single pairs)
- Purity (no mutations)
- Determinism
"""

import pytest
from src.families.color_map import ColorMapFamily
from src.utils import deep_eq, copy_grid


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def family():
    """Fresh ColorMapFamily instance for each test."""
    return ColorMapFamily()


# Test grids
g_empty = []

g_single = [[5]]

g_identity = [[1, 2],
              [3, 4]]

g_simple = [[1, 1, 2],
            [2, 3, 3]]


# ============================================================================
# Basic fit/apply
# ============================================================================

def test_fit_identity_mapping(family):
    """fit() with identical input/output should learn identity mapping."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {1: 1, 2: 2, 3: 3, 4: 4}


def test_fit_simple_remap(family):
    """fit() with simple remapping should learn correct mapping."""
    train = [
        {"input": [[1, 2], [3, 1]], "output": [[2, 3], [1, 2]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {1: 2, 2: 3, 3: 1}


def test_fit_multiple_pairs_same_mapping(family):
    """fit() with multiple pairs using same mapping should succeed."""
    train = [
        {"input": [[1, 2], [3, 1]], "output": [[2, 3], [1, 2]]},  # {1:2, 2:3, 3:1}
        {"input": [[2, 3], [1, 2]], "output": [[3, 1], [2, 3]]}   # same mapping
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {1: 2, 2: 3, 3: 1}


def test_apply_after_fit(family):
    """apply() after successful fit() should return correct remapped grid."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}  # {1:3, 2:4}
    ]
    family.fit(train)
    X = [[2, 1], [1, 2]]
    result = family.apply(X)
    expected = [[4, 3], [3, 4]]  # Apply mapping {1:3, 2:4}
    assert deep_eq(result, expected)


def test_apply_before_fit_raises(family):
    """apply() before fit() should raise RuntimeError."""
    X = [[1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="Must call fit"):
        family.apply(X)


# ============================================================================
# Unified mapping requirement (CRITICAL)
# ============================================================================

def test_unified_mapping_conflict_same_pair(family):
    """fit() with conflict in same pair (1→3 and 1→4) must return False."""
    train = [
        {"input": [[1, 1], [2, 2]], "output": [[3, 4], [5, 5]]}  # 1→3 at (0,0), 1→4 at (0,1)
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.mapping is None


def test_unified_mapping_conflict_across_pairs(family):
    """fit() with conflict across pairs (1→3 in pair0, 1→5 in pair1) must return False."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]},  # {1:3, 2:4}
        {"input": [[1, 2]], "output": [[5, 4]]}   # {1:5, 2:4} - conflict on color 1
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.mapping is None


def test_unified_mapping_unseen_color_in_later_pair(family):
    """fit() must reject if later pair has color not seen in first pair."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]},  # {1:3, 2:4}
        {"input": [[1, 5]], "output": [[3, 6]]}   # color 5 not in first pair
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.mapping is None


def test_unified_mapping_all_pairs_must_verify():
    """fit() must verify ALL pairs, not just first."""
    # First two pairs match {1:2, 2:3}, third pair conflicts
    train = [
        {"input": [[1, 2]], "output": [[2, 3]]},  # {1:2, 2:3}
        {"input": [[2, 1]], "output": [[3, 2]]},  # same mapping
        {"input": [[1, 2]], "output": [[9, 3]]}   # {1:9, 2:3} - conflict on color 1
    ]
    family = ColorMapFamily()
    result = family.fit(train)
    assert result is False  # Must fail because of third pair
    assert family.params.mapping is None


# ============================================================================
# FY exactness
# ============================================================================

def test_fy_single_pixel_difference_rejects(family):
    """Single pixel difference in verification → reject mapping."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},  # {1:5, 2:6, 3:7, 4:8}
        {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 9]]}   # Almost same, but (1,1) is 9 not 8
    ]
    result = family.fit(train)
    assert result is False  # 4→8 in pair0, 4→9 in pair1
    assert family.params.mapping is None


def test_fy_dimension_mismatch_rejects(family):
    """fit() must reject if input/output dimensions don't match."""
    train = [
        {"input": [[1, 2]], "output": [[3], [4]]}  # 1x2 → 2x1 mismatch
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.mapping is None


def test_fy_exact_equality_required():
    """After successful fit, applying mapping to train inputs must reproduce outputs exactly."""
    train = [
        {"input": [[1, 2, 3]], "output": [[4, 5, 6]]},  # {1:4, 2:5, 3:6}
        {"input": [[3, 2, 1]], "output": [[6, 5, 4]]}   # same mapping
    ]
    family = ColorMapFamily()
    result = family.fit(train)
    assert result is True

    # Verify FY exactness by applying to all train inputs
    for pair in train:
        X = pair["input"]
        Y = pair["output"]
        Y_predicted = family.apply(X)
        assert deep_eq(Y_predicted, Y)


# ============================================================================
# Unseen color handling
# ============================================================================

def test_apply_unseen_color_raises_keyerror(family):
    """apply() with unseen color should raise KeyError."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}  # {1:3, 2:4}
    ]
    family.fit(train)

    X = [[1, 5]]  # 5 not in mapping
    with pytest.raises(KeyError):
        family.apply(X)


def test_apply_keyerror_identifies_color():
    """KeyError should identify which color is unseen (implicit in Python KeyError)."""
    family = ColorMapFamily()
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}  # {1:3, 2:4}
    ]
    family.fit(train)

    X = [[7]]  # 7 not in mapping
    try:
        family.apply(X)
        assert False, "Expected KeyError"
    except KeyError as e:
        # Python KeyError includes the key in the error message
        assert "7" in str(e) or 7 in e.args


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_train_pairs_returns_false(family):
    """fit([]) with empty train_pairs should return False."""
    result = family.fit([])
    assert result is False
    assert family.params.mapping is None


def test_empty_grids(family):
    """fit() with empty grids ([] → []) should succeed with empty mapping."""
    train = [
        {"input": [], "output": []}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {}


def test_empty_grids_mismatch(family):
    """fit() with [] → [[1]] should fail (dimension mismatch)."""
    train = [
        {"input": [], "output": [[1]]}
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.mapping is None


def test_apply_empty_grid(family):
    """apply([]) with valid mapping should return []."""
    train = [
        {"input": [[1]], "output": [[2]]}  # {1:2}
    ]
    family.fit(train)
    result = family.apply([])
    assert result == []


def test_single_pixel_grid(family):
    """fit() with 1x1 grids should work correctly."""
    train = [
        {"input": [[5]], "output": [[7]]}  # {5:7}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {5: 7}


def test_all_pixels_same_color(family):
    """fit() with all pixels same color should work."""
    train = [
        {"input": [[3, 3], [3, 3]], "output": [[9, 9], [9, 9]]}  # {3:9}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {3: 9}


# ============================================================================
# Purity
# ============================================================================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() must not mutate train_pairs input."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}
    ]
    # Deep copy to compare later
    train_copy = [
        {"input": copy_grid(train[0]["input"]),
         "output": copy_grid(train[0]["output"])}
    ]

    family.fit(train)

    # Verify train_pairs unchanged
    assert deep_eq(train[0]["input"], train_copy[0]["input"])
    assert deep_eq(train[0]["output"], train_copy[0]["output"])


def test_apply_does_not_mutate_input(family):
    """apply() must not mutate input grid X."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}  # {1:3, 2:4}
    ]
    family.fit(train)

    X = [[2, 1], [1, 2]]
    X_copy = copy_grid(X)

    result = family.apply(X)

    # Verify X unchanged
    assert deep_eq(X, X_copy)
    # Verify result is correct
    assert deep_eq(result, [[4, 3], [3, 4]])


def test_apply_no_row_aliasing(family):
    """apply() output must not share row references with input."""
    train = [
        {"input": [[1]], "output": [[2]]}  # {1:2}
    ]
    family.fit(train)

    X = [[1, 1]]
    result = family.apply(X)

    # Mutate result
    result[0][0] = 999

    # Verify X unchanged (no aliasing)
    assert X[0][0] == 1


# ============================================================================
# Determinism
# ============================================================================

def test_deterministic_repeated_fit(family):
    """Running fit() twice on same train_pairs yields identical mapping."""
    train = [
        {"input": [[1, 2, 3]], "output": [[4, 5, 6]]}  # {1:4, 2:5, 3:6}
    ]

    result1 = family.fit(train)
    mapping1 = family.params.mapping.copy()

    # Reset and re-fit
    family.params.mapping = None
    result2 = family.fit(train)
    mapping2 = family.params.mapping

    assert result1 is True
    assert result2 is True
    assert mapping1 == mapping2


def test_deterministic_apply(family):
    """apply() with same input yields identical output."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}  # {1:3, 2:4}
    ]
    family.fit(train)

    X = [[2, 1], [1, 2]]
    result1 = family.apply(X)
    result2 = family.apply(X)

    assert deep_eq(result1, result2)


# ============================================================================
# Integration tests
# ============================================================================

def test_complex_mapping_multiple_pairs():
    """Full workflow with complex mapping across multiple pairs."""
    # Note: Mapping is learned from FIRST pair only.
    # Later pairs can only use colors seen in first pair.
    # Mapping: {0:1, 1:2, 2:3, 3:4, 4:5}
    train = [
        {"input": [[0, 1, 2, 3, 4]], "output": [[1, 2, 3, 4, 5]]},  # Learn {0:1, 1:2, 2:3, 3:4, 4:5}
        {"input": [[3, 4, 0]], "output": [[4, 5, 1]]},              # Verify subset of mapping
        {"input": [[1, 2, 3, 4]], "output": [[2, 3, 4, 5]]}         # Verify subset of mapping
    ]
    family = ColorMapFamily()

    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

    # Apply on new input
    X = [[4, 3, 2, 1, 0]]
    result = family.apply(X)
    expected = [[5, 4, 3, 2, 1]]
    assert deep_eq(result, expected)


def test_mapping_learned_only_from_first_pair():
    """Mapping is learned from first pair; later pairs only verify."""
    train = [
        {"input": [[1, 2]], "output": [[5, 6]]},  # Learn {1:5, 2:6}
        {"input": [[1, 2]], "output": [[5, 6]]},  # Verify
        {"input": [[2, 1]], "output": [[6, 5]]}   # Verify
    ]
    family = ColorMapFamily()
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {1: 5, 2: 6}


def test_reverse_mapping():
    """Test with reverse color mapping."""
    # {1:4, 2:3, 3:2, 4:1} - reverse mapping
    train = [
        {"input": [[1, 2, 3, 4]], "output": [[4, 3, 2, 1]]}
    ]
    family = ColorMapFamily()
    result = family.fit(train)
    assert result is True
    assert family.params.mapping == {1: 4, 2: 3, 3: 2, 4: 1}

    # Apply and verify
    X = [[4, 3, 2, 1]]
    result = family.apply(X)
    expected = [[1, 2, 3, 4]]
    assert deep_eq(result, expected)
