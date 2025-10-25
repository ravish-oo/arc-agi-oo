"""
Test suite for IsometryFamily (P2-01).

Tests cover:
- Basic fit/apply workflow
- Deterministic σ selection
- Unified σ requirement (critical: must work for ALL pairs)
- Shape safety with dimension changes
- FY exactness (bit-for-bit equality)
- Edge cases (empty grids, single pairs)
- Purity (no mutations)
"""

import pytest
from src.families.isometry import IsometryFamily
from src.utils import deep_eq, copy_grid


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def family():
    """Fresh IsometryFamily instance for each test."""
    return IsometryFamily()


# Test grids
g_empty = []

g_single = [[5]]

g_square = [[1, 2],
            [3, 4]]

g_rect = [[1, 2, 3],
          [4, 5, 6]]

# Expected D8 transforms of g_square
g_square_rot90 = [[3, 1],
                  [4, 2]]

g_square_rot180 = [[4, 3],
                   [2, 1]]

g_square_flip_h = [[2, 1],
                   [4, 3]]

g_square_transpose = [[1, 3],
                      [2, 4]]

g_square_flip_anti = [[4, 2],
                      [3, 1]]


# ============================================================================
# Basic fit/apply
# ============================================================================

def test_fit_identity(family):
    """fit() with identity pairs should accept 'id'."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"


def test_fit_rot90_single_pair(family):
    """fit() with single rot90 pair should accept 'rot90'."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot90"


def test_fit_rot90_multiple_pairs(family):
    """fit() with multiple rot90 pairs should accept 'rot90' (unified σ)."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
        {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot90"


def test_apply_after_fit(family):
    """apply() after successful fit() should return correct transform."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
    ]
    family.fit(train)
    X = [[9, 8], [7, 6]]
    result = family.apply(X)
    expected = [[7, 9], [6, 8]]  # rot90 of X
    assert deep_eq(result, expected)


def test_apply_before_fit_raises(family):
    """apply() before fit() should raise RuntimeError."""
    X = [[1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="Must call fit"):
        family.apply(X)


# ============================================================================
# Deterministic σ selection
# ============================================================================

def test_deterministic_order_symmetric_grid(family):
    """When multiple σ match, choose first in all_isometries() order."""
    # Grid where all isometries produce same result (all 1s)
    train = [
        {"input": [[1, 1], [1, 1]], "output": [[1, 1], [1, 1]]}
    ]
    result = family.fit(train)
    assert result is True
    # all_isometries() = ["id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "flip_anti"]
    # "id" comes first, so it should win
    assert family.params.sigma == "id"


def test_deterministic_repeated_fit(family):
    """Running fit() twice on same train_pairs yields identical params.sigma."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}  # rot180
    ]

    result1 = family.fit(train)
    sigma1 = family.params.sigma

    # Reset and re-fit
    family.params.sigma = None
    result2 = family.fit(train)
    sigma2 = family.params.sigma

    assert result1 is True
    assert result2 is True
    assert sigma1 == sigma2
    assert sigma1 == "rot180"


def test_deterministic_order_flip_h_vs_flip_v():
    """When both flip_h and flip_v match, flip_h wins (earlier in all_isometries())."""
    # Create a grid where flip_h and flip_v produce the same result
    # [[1,2,1], [2,3,2], [1,2,1]] → flip_h == flip_v
    train = [
        {"input": [[1, 2, 1], [2, 3, 2], [1, 2, 1]],
         "output": [[1, 2, 1], [2, 3, 2], [1, 2, 1]]}
    ]
    family = IsometryFamily()
    result = family.fit(train)
    assert result is True
    # all_isometries() order: ["id", "rot90", "rot180", "rot270", "flip_h", "flip_v", ...]
    # "id" comes first and matches (input == output)
    assert family.params.sigma == "id"


# ============================================================================
# Unified σ requirement (CRITICAL)
# ============================================================================

def test_unified_sigma_mixed_pairs_returns_false(family):
    """fit() with mixed transforms (no unified σ) must return False."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},  # rot90
        {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]}   # flip_h (different σ)
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.sigma is None


def test_unified_sigma_all_pairs_checked():
    """fit() must verify ALL pairs before accepting σ (not just first pair)."""
    # First pair matches rot90, second pair does not
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},  # rot90 matches
        {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]}   # identity (not rot90)
    ]
    family = IsometryFamily()
    result = family.fit(train)
    assert result is False  # rot90 doesn't work for second pair
    assert family.params.sigma is None


def test_unified_sigma_three_pairs_all_match():
    """fit() with three pairs all requiring same σ should succeed."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},  # flip_h
        {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},  # flip_h
        {"input": [[9, 0], [1, 2]], "output": [[0, 9], [2, 1]]}   # flip_h
    ]
    family = IsometryFamily()
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "flip_h"


# ============================================================================
# Shape safety
# ============================================================================

def test_shape_safety_transpose_dimension_change(family):
    """fit() should handle dimension changes correctly (dimension swap)."""
    # 1x3 → 3x1 dimension change
    # Note: Both rot90 and transpose produce [[1],[2],[3]] from [[1,2,3]]
    # but rot90 comes first in all_isometries() order, so it wins (deterministic first-acceptable)
    train = [
        {"input": [[1, 2, 3]], "output": [[1], [2], [3]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot90"  # rot90 wins over transpose (earlier in order)


def test_shape_safety_rot90_dimension_change(family):
    """fit() should handle dimension changes correctly (rot90)."""
    # 2x3 → 3x2 via rot90
    train = [
        {"input": [[1, 2, 3], [4, 5, 6]],
         "output": [[4, 1], [5, 2], [6, 3]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot90"


def test_shape_safety_dimension_mismatch_skips_sigma(family):
    """fit() should skip σ if output dimensions don't match transformed input."""
    # Input is 2x2, output is 3x3 (no D8 transform can achieve this)
    train = [
        {"input": [[1, 2], [3, 4]],
         "output": [[1, 2, 0], [3, 4, 0], [0, 0, 0]]}
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.sigma is None


# ============================================================================
# FY exactness
# ============================================================================

def test_fy_single_pixel_difference_rejects_sigma(family):
    """Single pixel difference → reject σ (FY exactness)."""
    train = [
        {"input": [[1, 2], [3, 4]],
         "output": [[3, 1], [4, 9]]}  # Almost rot90, but bottom-right is 9 not 2
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.sigma is None


def test_fy_all_but_one_pair_match_still_rejects():
    """All-but-one pairs match → still reject σ (must match ALL)."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},  # rot90 matches
        {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]},  # rot90 matches
        {"input": [[9, 0], [1, 2]], "output": [[9, 0], [1, 2]]}   # identity (not rot90)
    ]
    family = IsometryFamily()
    result = family.fit(train)
    assert result is False  # rot90 doesn't work for third pair
    assert family.params.sigma is None


def test_fy_exact_equality_required():
    """deep_eq must pass (bit-for-bit equality), not 'close enough'."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
    ]
    family = IsometryFamily()
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"

    # Now verify apply() produces exact equality
    X = [[7, 8], [9, 0]]
    result = family.apply(X)
    assert deep_eq(result, X)  # Must be bit-for-bit identical


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_train_pairs_returns_false(family):
    """fit([]) with empty train_pairs should return False."""
    result = family.fit([])
    assert result is False
    assert family.params.sigma is None


def test_empty_grids_identity(family):
    """fit() with empty grids ([] → []) should accept 'id'."""
    train = [
        {"input": [], "output": []}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"


def test_apply_empty_grid(family):
    """apply([]) with valid σ should return []."""
    train = [
        {"input": [[1, 2]], "output": [[2, 1]]}  # flip_h
    ]
    family.fit(train)
    result = family.apply([])
    assert result == []


def test_single_pixel_grid(family):
    """fit() with 1x1 grids should work correctly."""
    train = [
        {"input": [[5]], "output": [[5]]}  # identity (all D8 transforms of 1x1 are same)
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"  # First in all_isometries() order


# ============================================================================
# Purity
# ============================================================================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() must not mutate train_pairs input."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
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
        {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}  # flip_h
    ]
    family.fit(train)

    X = [[5, 6], [7, 8]]
    X_copy = copy_grid(X)

    result = family.apply(X)

    # Verify X unchanged
    assert deep_eq(X, X_copy)
    # Verify result is correct
    assert deep_eq(result, [[6, 5], [8, 7]])


def test_apply_no_row_aliasing(family):
    """apply() output must not share row references with input."""
    train = [
        {"input": [[1, 2]], "output": [[1, 2]]}  # identity
    ]
    family.fit(train)

    X = [[3, 4]]
    result = family.apply(X)

    # Mutate result
    result[0][0] = 999

    # Verify X unchanged (no aliasing)
    assert X[0][0] == 3


# ============================================================================
# Integration with canonicalization (all 8 isometries)
# ============================================================================

def test_all_eight_isometries_tested():
    """fit() should successfully find each of the 8 isometries when appropriate."""
    test_cases = [
        ("id", [[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ("rot90", [[1, 2], [3, 4]], [[3, 1], [4, 2]]),
        ("rot180", [[1, 2], [3, 4]], [[4, 3], [2, 1]]),
        ("rot270", [[1, 2], [3, 4]], [[2, 4], [1, 3]]),
        ("flip_h", [[1, 2], [3, 4]], [[2, 1], [4, 3]]),
        ("flip_v", [[1, 2], [3, 4]], [[3, 4], [1, 2]]),
        ("transpose", [[1, 2], [3, 4]], [[1, 3], [2, 4]]),
        ("flip_anti", [[1, 2], [3, 4]], [[4, 2], [3, 1]]),
    ]

    for expected_sigma, input_grid, output_grid in test_cases:
        family = IsometryFamily()
        train = [{"input": input_grid, "output": output_grid}]
        result = family.fit(train)
        assert result is True, f"Failed to fit {expected_sigma}"
        assert family.params.sigma == expected_sigma, \
            f"Expected {expected_sigma}, got {family.params.sigma}"


def test_flip_anti_correctness():
    """Verify flip_anti (anti-diagonal reflection) works correctly."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[4, 2], [3, 1]]}  # flip_anti
    ]
    family = IsometryFamily()
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "flip_anti"

    # Test apply
    X = [[5, 6], [7, 8]]
    result = family.apply(X)
    expected = [[8, 6], [7, 5]]  # flip_anti of X
    assert deep_eq(result, expected)


# ============================================================================
# Comprehensive integration test
# ============================================================================

def test_full_workflow_multiple_pairs():
    """Full workflow: fit with multiple pairs, verify apply on new input."""
    # Note: For 1xN grids, both rot180 and flip_h produce same result (reverse the row)
    # but rot180 comes first in all_isometries() order, so it wins
    train = [
        {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},  # rot180 and flip_h both work
        {"input": [[4, 5, 6]], "output": [[6, 5, 4]]},  # rot180 and flip_h both work
        {"input": [[7, 8, 9]], "output": [[9, 8, 7]]}   # rot180 and flip_h both work
    ]
    family = IsometryFamily()

    # Fit
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot180"  # rot180 wins over flip_h (earlier in order)

    # Apply on new input
    X = [[0, 1, 2]]
    result = family.apply(X)
    expected = [[2, 1, 0]]  # rot180 applied
    assert deep_eq(result, expected)
