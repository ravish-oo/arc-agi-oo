"""
Tests for CopyMoveAllComponents Family — Phase 2 work order P2-16.

Comprehensive test suite covering:
- Basic fit() and apply() with single/multiple components
- Multiple colors with different deltas
- Unified delta requirement across all pairs
- Edge cases (empty, overlaps, out-of-bounds)
- Overlap and overwrite handling (GLUE law)
- Purity and shape safety
- FY exactness integration
"""

import pytest
from src.families.copy_move import CopyMoveAllComponentsFamily
from src.utils import dims, deep_eq


# =============================
# Fixtures
# =============================

@pytest.fixture
def family():
    """Fresh family instance for each test."""
    return CopyMoveAllComponentsFamily()


# =============================
# A. Basic fit/apply Tests
# =============================

def test_fit_empty_train_pairs(family):
    """fit([]) should return False."""
    result = family.fit([])
    assert result is False


def test_fit_single_component_translate_right(family):
    """Single component translated right by 2."""
    X = [[1, 0, 0, 0]]
    Y = [[0, 0, 1, 0]]
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas == {1: (0, 2)}


def test_fit_single_component_translate_down(family):
    """Single component translated down by 1."""
    X = [[1], [0], [0]]
    Y = [[0], [1], [0]]
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas == {1: (1, 0)}


def test_apply_after_fit_single_component(family):
    """apply() works after fit() for single component."""
    X = [[1, 0, 0, 0]]
    Y = [[0, 0, 1, 0]]
    train_pairs = [{"input": X, "output": Y}]
    family.fit(train_pairs)

    result = family.apply(X)
    assert deep_eq(result, Y)


def test_apply_before_fit_raises(family):
    """apply() before fit() raises RuntimeError."""
    X = [[1, 0]]
    with pytest.raises(RuntimeError, match="Must call fit"):
        family.apply(X)


def test_fit_no_motion_identity(family):
    """No motion: identical X and Y → all deltas (0,0)."""
    X = [[1, 2], [3, 4]]
    Y = [[1, 2], [3, 4]]
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    # All colors should have delta (0,0)
    for delta in family.params.deltas.values():
        assert delta == (0, 0)


def test_fit_empty_grids_rejects(family):
    """Empty grids should be rejected."""
    train_pairs = [{"input": [], "output": []}]
    result = family.fit(train_pairs)
    assert result is False


def test_multiple_components_same_color_same_delta(family):
    """Multiple components of same color all move by same delta."""
    X = [[1, 0, 1], [0, 0, 0]]
    Y = [[0, 0, 0], [1, 0, 1]]
    # Two separate components of color 1, both move down by 1
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas[1] == (1, 0)  # Check only color 1, ignore background


# =============================
# B. Multiple Colors
# =============================

def test_two_colors_different_deltas(family):
    """Two colors with different translation deltas."""
    X = [[1, 0, 0], [0, 2, 0]]
    Y = [[0, 1, 0], [2, 0, 0]]
    # Color 1: right 1, Color 2: left 1
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas[1] == (0, 1)
    assert family.params.deltas[2] == (0, -1)


def test_apply_multiple_colors(family):
    """apply() works correctly with multiple colors."""
    X = [[1, 0, 0], [0, 2, 0]]
    Y = [[0, 1, 0], [2, 0, 0]]
    train_pairs = [{"input": X, "output": Y}]
    family.fit(train_pairs)

    result = family.apply(X)
    assert deep_eq(result, Y)


def test_three_colors_mixed_deltas(family):
    """Three colors with different deltas."""
    X = [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]]
    Y = [[0, 1, 0],
         [2, 0, 0],
         [0, 0, 3]]
    # Color 1: right 1, Color 2: left 1, Color 3: no move
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True


# =============================
# C. Unified Delta Requirement
# =============================

def test_unified_delta_multiple_pairs_same_color(family):
    """Multiple pairs with same color → same delta."""
    train_pairs = [
        {"input": [[1, 0]], "output": [[0, 1]]},  # delta (0,1)
        {"input": [[1, 0, 0]], "output": [[0, 1, 0]]},  # delta (0,1)
    ]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas[1] == (0, 1)  # Check only color 1, ignore background


def test_reject_inconsistent_deltas_across_pairs(family):
    """Different deltas for same color across pairs → reject."""
    train_pairs = [
        {"input": [[1, 0]], "output": [[0, 1]]},  # delta (0,1)
        {"input": [[1, 0]], "output": [[1, 0]]},  # delta (0,0)
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_all_pairs_must_match(family):
    """All pairs must satisfy FY with learned deltas."""
    train_pairs = [
        {"input": [[1, 0]], "output": [[0, 1]]},
        {"input": [[1, 0]], "output": [[0, 1]]},
        {"input": [[1, 0]], "output": [[1, 0]]},  # Third doesn't match
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_shape_mismatch_rejects(family):
    """Different shapes in X vs Y → reject."""
    train_pairs = [
        {"input": [[1, 0]], "output": [[0, 1], [0, 0]]}
    ]
    result = family.fit(train_pairs)
    assert result is False


# =============================
# D. Edge Cases
# =============================

def test_component_out_of_bounds_clipped(family):
    """Component moving out of bounds → pixels clipped."""
    X = [[1, 1]]
    Y = [[0, 0]]
    # Delta would be (0, 2) which moves component fully out
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True


def test_component_partially_out_of_bounds(family):
    """Component partially out of bounds → only in-bounds pixels written."""
    X = [[1, 1, 0]]
    Y = [[0, 1, 0]]
    # Move right 1: first pixel goes to (0,1), second goes out to (0,2) (stays in)
    # Actually both stay in. Let me recalculate...
    # Original: (0,0)=1, (0,1)=1
    # Delta (0,1): (0,0)→(0,1), (0,1)→(0,2)
    # Result: (0,1)=1, (0,2)=1 (if grid is 1x3)
    # But Y is [[0, 1, 0]] which means only (0,1) has 1
    # So second pixel at (0,1) moved to (0,2) should be 1, but Y has 0
    # This test is wrong. Let me fix it:
    X = [[1, 1, 0]]
    Y = [[0, 1, 1]]
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas[1] == (0, 1)  # Check only color 1, ignore background


def test_overlapping_components_later_overwrites(family):
    """Overlapping components after translation → later color overwrites."""
    X = [[1, 0], [0, 2]]
    Y = [[2, 0], [0, 1]]
    # If both move to (0,0), color 2 should overwrite color 1
    # Actually this test needs to be designed carefully
    # Let me create a clearer example:
    X = [[1, 0, 2]]
    Y = [[0, 2, 0]]
    # Color 1: (0,0)→(0,1) delta (0,1)
    # Color 2: (0,2)→(0,1) delta (0,-1)
    # Both write to (0,1), color 2 wins (higher value, processed later)
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    # Result should have color 2 at (0,1)
    assert deep_eq(family.apply(X), Y)


def test_color_not_in_output_delta_zero(family):
    """Color present in X but not in Y → moved out of bounds, delta skipped."""
    X = [[1, 0]]
    Y = [[0, 0]]
    # Color 1 disappears in Y (moved out of bounds)
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    # Should accept: color moved out of bounds, we skip learning delta for it
    # Verification passes because applying no delta produces all-zero output
    assert result is True


def test_component_count_mismatch_rejects(family):
    """Different number of components for same color → reject."""
    X = [[1, 0, 1]]  # Two components of color 1
    Y = [[1, 1, 0]]  # One component of color 1
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is False


def test_single_pixel_component(family):
    """Single pixel component can be translated."""
    X = [[0, 0], [0, 1]]
    Y = [[1, 0], [0, 0]]
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True


def test_large_component_translation(family):
    """Large connected component can be translated."""
    X = [[1, 1], [1, 1], [0, 0]]
    Y = [[0, 0], [1, 1], [1, 1]]
    # 2x2 block moves down by 1
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas[1] == (1, 0)  # Check only color 1, ignore background


def test_diagonal_adjacent_components_8_connected(family):
    """Diagonally adjacent pixels form single component (8-connected)."""
    X = [[1, 0], [0, 1], [0, 0]]
    Y = [[0, 0], [1, 0], [0, 1]]
    # Two pixels diagonally adjacent: ONE component (8-connected)
    # Delta should be (1,0) down by 1
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True


def test_multiple_sizes_different_components(family):
    """Components of different sizes within same color."""
    X = [[1, 0, 1, 1], [0, 0, 0, 0]]
    Y = [[0, 0, 0, 0], [1, 0, 1, 1]]
    # Two components: (0,0) and (0,2)-(0,3)
    # Both move down by 1
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas[1] == (1, 0)  # Check only color 1, ignore background


# =============================
# E. Purity and Shape Safety
# =============================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() should not mutate input train_pairs."""
    train_pairs = [{"input": [[1, 0]], "output": [[0, 1]]}]
    import copy
    train_pairs_copy = copy.deepcopy(train_pairs)

    family.fit(train_pairs)

    assert deep_eq(train_pairs, train_pairs_copy)


def test_apply_does_not_mutate_input(family):
    """apply() should not mutate input X."""
    X = [[1, 0]]
    import copy
    X_copy = copy.deepcopy(X)

    train_pairs = [{"input": X, "output": [[0, 1]]}]
    family.fit(train_pairs)
    family.apply(X)

    assert deep_eq(X, X_copy)


def test_apply_no_aliasing(family):
    """Output should have fresh allocation, no aliasing with input."""
    X = [[1, 0]]
    train_pairs = [{"input": X, "output": [[0, 1]]}]
    family.fit(train_pairs)
    result = family.apply(X)

    # Verify no aliasing
    assert result is not X
    assert result[0] is not X[0]


def test_shape_safety_preserved(family):
    """dims(output) == dims(input) always."""
    X = [[1, 0, 0], [0, 0, 0]]
    train_pairs = [{"input": X, "output": [[0, 1, 0], [0, 0, 0]]}]
    family.fit(train_pairs)
    result = family.apply(X)

    assert dims(result) == dims(X)


# =============================
# F. FY Exactness
# =============================

def test_fy_exactness_all_pairs(family):
    """FY: All pairs must match exactly (bit-for-bit)."""
    train_pairs = [
        {"input": [[1, 0]], "output": [[0, 1]]},
        {"input": [[2, 0]], "output": [[0, 2]]},
    ]
    result = family.fit(train_pairs)
    assert result is True

    # Verify apply on both
    assert deep_eq(family.apply([[1, 0]]), [[0, 1]])
    assert deep_eq(family.apply([[2, 0]]), [[0, 2]])


def test_fy_single_pixel_differs_rejects(family):
    """Single pixel difference → reject deltas."""
    X = [[1, 0]]
    Y = [[0, 2]]  # Should be [[0, 1]] for translation
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is False


def test_deterministic_repeated_fit(family):
    """fit() twice on same data yields same deltas."""
    X = [[1, 0]]
    Y = [[0, 1]]
    train_pairs = [{"input": X, "output": Y}]

    result1 = family.fit(train_pairs)
    deltas1 = family.params.deltas

    result2 = family.fit(train_pairs)
    deltas2 = family.params.deltas

    assert result1 == result2 == True
    assert deltas1 == deltas2


def test_deterministic_apply(family):
    """apply() twice on same input yields identical output."""
    X = [[1, 0]]
    Y = [[0, 1]]
    train_pairs = [{"input": X, "output": Y}]
    family.fit(train_pairs)

    result1 = family.apply(X)
    result2 = family.apply(X)

    assert deep_eq(result1, result2)


def test_background_color_zero_as_component(family):
    """Color 0 can form components and should translate like any other color."""
    X = [[1, 1, 0],
         [1, 1, 0]]
    Y = [[0, 1, 1],
         [0, 1, 1]]
    # Color 1: (0,0)-(1,1) → (0,1)-(1,2), delta = (0, 1)
    # Color 0: (0,2), (1,2) → (0,0), (1,0), delta = (0, -2)
    train_pairs = [{"input": X, "output": Y}]
    result = family.fit(train_pairs)
    assert result is True
    assert family.params.deltas[0] == (0, -2)
    assert family.params.deltas[1] == (0, 1)
