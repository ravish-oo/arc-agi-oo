"""
Comprehensive tests for src/canonicalization.py D8 isometry registry.

Tests cover:
- Enumeration (8 names, correct set, deterministic order)
- Group identities (involutions and cyclic properties)
- Cross-identities (commutativity properties)
- Purity (no mutation, no aliasing)
- Edge cases (empty, 1×1, 1×N, N×1, rectangular, ragged)
- Exceptions (unknown name, ragged input)
- Determinism
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.canonicalization import ISOMETRIES, all_isometries, apply_isometry
from src.utils import deep_eq, copy_grid


# ============================================================================
# Test Fixtures
# ============================================================================

# Empty
g_empty = []

# Single cell
g_1x1 = [[5]]

# Horizontal
g_1x4 = [[1, 2, 3, 4]]

# Vertical
g_4x1 = [[1], [2], [3], [4]]

# Square 2×2
g_2x2 = [[1, 2],
         [3, 4]]

# Rectangular 2×3
g_2x3 = [[1, 2, 3],
         [4, 5, 6]]

# Ragged (invalid)
g_ragged = [[1, 2], [3]]


# ============================================================================
# Enumeration Tests
# ============================================================================

def test_all_isometries_returns_8_names():
    """Should return exactly 8 isometry names."""
    assert len(all_isometries()) == 8


def test_all_isometries_has_correct_set():
    """Should contain all 8 D8 isometry names."""
    expected = {"id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "flip_anti"}
    assert set(all_isometries()) == expected


def test_all_isometries_deterministic():
    """Calling twice should return identical list (same order)."""
    first = all_isometries()
    second = all_isometries()
    assert first == second


def test_all_isometries_snapshot():
    """Order must be FIXED (snapshot test)."""
    expected = ["id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "flip_anti"]
    assert all_isometries() == expected


def test_ISOMETRIES_constant_is_list():
    """ISOMETRIES should be a list."""
    assert isinstance(ISOMETRIES, list)


def test_ISOMETRIES_matches_all_isometries():
    """ISOMETRIES constant and all_isometries() should match."""
    assert ISOMETRIES == all_isometries()


# ============================================================================
# Group Identity Tests
# ============================================================================

def test_rot90_fourth_power_is_identity():
    """rot90^4 = identity (4x 90° = 360°)"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result = g
        for _ in range(4):
            result = apply_isometry(result, "rot90")
        assert deep_eq(result, g), f"rot90^4 != id for {g}"


def test_rot180_squared_is_identity():
    """rot180^2 = identity (2x 180° = 360°)"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result = apply_isometry(apply_isometry(g, "rot180"), "rot180")
        assert deep_eq(result, g), f"rot180^2 != id for {g}"


def test_rot270_fourth_power_is_identity():
    """rot270^4 = identity"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result = g
        for _ in range(4):
            result = apply_isometry(result, "rot270")
        assert deep_eq(result, g), f"rot270^4 != id for {g}"


def test_flip_h_squared_is_identity():
    """flip_h^2 = identity"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result = apply_isometry(apply_isometry(g, "flip_h"), "flip_h")
        assert deep_eq(result, g), f"flip_h^2 != id for {g}"


def test_flip_v_squared_is_identity():
    """flip_v^2 = identity"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result = apply_isometry(apply_isometry(g, "flip_v"), "flip_v")
        assert deep_eq(result, g), f"flip_v^2 != id for {g}"


def test_transpose_squared_is_identity():
    """transpose^2 = identity"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result = apply_isometry(apply_isometry(g, "transpose"), "transpose")
        assert deep_eq(result, g), f"transpose^2 != id for {g}"


def test_flip_anti_squared_is_identity():
    """flip_anti^2 = identity (anti-diagonal reflection)"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result = apply_isometry(apply_isometry(g, "flip_anti"), "flip_anti")
        assert deep_eq(result, g), f"flip_anti^2 != id for {g}"


# ============================================================================
# Cross-Identity Tests
# ============================================================================

def test_transpose_flip_h_equals_flip_v_transpose():
    """transpose ∘ flip_h = flip_v ∘ transpose"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        # transpose(flip_h(g))
        result1 = apply_isometry(apply_isometry(g, "flip_h"), "transpose")
        # flip_v(transpose(g))
        result2 = apply_isometry(apply_isometry(g, "transpose"), "flip_v")
        assert deep_eq(result1, result2), f"transpose∘flip_h != flip_v∘transpose for {g}"


def test_transpose_flip_v_equals_flip_h_transpose():
    """transpose ∘ flip_v = flip_h ∘ transpose"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        # transpose(flip_v(g))
        result1 = apply_isometry(apply_isometry(g, "flip_v"), "transpose")
        # flip_h(transpose(g))
        result2 = apply_isometry(apply_isometry(g, "transpose"), "flip_h")
        assert deep_eq(result1, result2), f"transpose∘flip_v != flip_h∘transpose for {g}"


def test_rot90_equals_transpose_flip_v():
    """rot90 = transpose ∘ flip_v (structural identity)"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result1 = apply_isometry(g, "rot90")
        result2 = apply_isometry(apply_isometry(g, "flip_v"), "transpose")
        assert deep_eq(result1, result2), f"rot90 != transpose∘flip_v for {g}"


def test_rot270_equals_flip_v_transpose():
    """rot270 = flip_v ∘ transpose (structural identity)"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result1 = apply_isometry(g, "rot270")
        result2 = apply_isometry(apply_isometry(g, "transpose"), "flip_v")
        assert deep_eq(result1, result2), f"rot270 != flip_v∘transpose for {g}"


def test_flip_anti_equals_transpose_rot180():
    """flip_anti = transpose ∘ rot180 (equivalence)"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result1 = apply_isometry(g, "flip_anti")
        result2 = apply_isometry(apply_isometry(g, "rot180"), "transpose")
        assert deep_eq(result1, result2), f"flip_anti != transpose∘rot180 for {g}"


def test_flip_anti_equals_rot180_transpose():
    """flip_anti = rot180 ∘ transpose (equivalence)"""
    for g in [g_empty, g_1x1, g_1x4, g_4x1, g_2x2, g_2x3]:
        result1 = apply_isometry(g, "flip_anti")
        result2 = apply_isometry(apply_isometry(g, "transpose"), "rot180")
        assert deep_eq(result1, result2), f"flip_anti != rot180∘transpose for {g}"


# ============================================================================
# Purity Tests
# ============================================================================

def test_apply_isometry_does_not_mutate_input():
    """Input grid should remain unchanged after apply_isometry."""
    g_orig = [[1, 2, 3], [4, 5, 6]]
    g_snapshot = copy_grid(g_orig)

    # Apply all isometries
    for name in all_isometries():
        _ = apply_isometry(g_orig, name)

    # Original unchanged
    assert deep_eq(g_orig, g_snapshot)


def test_apply_isometry_no_row_aliasing():
    """Output rows should be distinct objects from input rows."""
    g_orig = [[1, 2], [3, 4]]

    # Test id (copy) - most likely to alias
    result = apply_isometry(g_orig, "id")

    # Modify result
    result[0][0] = 99

    # Original unchanged
    assert g_orig[0][0] == 1


def test_apply_isometry_creates_new_grid():
    """Each call should create a new grid object."""
    g = [[1, 2], [3, 4]]

    # Apply same transform twice
    result1 = apply_isometry(g, "rot90")
    result2 = apply_isometry(g, "rot90")

    # Results are equal but not the same object
    assert deep_eq(result1, result2)
    assert result1 is not result2


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_grid():
    """All transforms on [] should return []."""
    for name in all_isometries():
        result = apply_isometry(g_empty, name)
        assert result == []


def test_1x1_grid():
    """All transforms on [[v]] should return [[v]]."""
    for name in all_isometries():
        result = apply_isometry(g_1x1, name)
        assert deep_eq(result, g_1x1)


def test_1xN_grid_dimension_swaps():
    """1×N grid: rot90, rot270, transpose, flip_anti should swap to N×1."""
    # These transforms swap dimensions
    for name in ["rot90", "rot270", "transpose", "flip_anti"]:
        result = apply_isometry(g_1x4, name)
        assert len(result) == 4  # Now 4 rows
        assert len(result[0]) == 1  # Each row has 1 element

    # These preserve dimensions
    for name in ["id", "rot180", "flip_h", "flip_v"]:
        result = apply_isometry(g_1x4, name)
        assert len(result) == 1  # Still 1 row
        assert len(result[0]) == 4  # Each row has 4 elements


def test_Nx1_grid_dimension_swaps():
    """N×1 grid: rot90, rot270, transpose, flip_anti should swap to 1×N."""
    # These transforms swap dimensions
    for name in ["rot90", "rot270", "transpose", "flip_anti"]:
        result = apply_isometry(g_4x1, name)
        assert len(result) == 1  # Now 1 row
        assert len(result[0]) == 4  # Each row has 4 elements

    # These preserve dimensions
    for name in ["id", "rot180", "flip_h", "flip_v"]:
        result = apply_isometry(g_4x1, name)
        assert len(result) == 4  # Still 4 rows
        assert len(result[0]) == 1  # Each row has 1 element


def test_rectangular_2x3_dimension_swaps():
    """2×3 grid: rot90, rot270, transpose, flip_anti should swap to 3×2."""
    # These transforms swap dimensions
    for name in ["rot90", "rot270", "transpose", "flip_anti"]:
        result = apply_isometry(g_2x3, name)
        assert len(result) == 3  # Now 3 rows
        assert len(result[0]) == 2  # Each row has 2 elements

    # These preserve dimensions
    for name in ["id", "rot180", "flip_h", "flip_v"]:
        result = apply_isometry(g_2x3, name)
        assert len(result) == 2  # Still 2 rows
        assert len(result[0]) == 3  # Each row has 3 elements


# ============================================================================
# Specific Transformation Tests (from context pack fixtures)
# ============================================================================

def test_1x4_rot90():
    """Verify rot90 on 1×4 grid."""
    # Correct output: left element goes to top after 90° clockwise rotation
    expected = [[1], [2], [3], [4]]
    assert deep_eq(apply_isometry(g_1x4, "rot90"), expected)


def test_1x4_flip_anti():
    """Verify flip_anti on 1×4 grid."""
    expected = [[4], [3], [2], [1]]
    assert deep_eq(apply_isometry(g_1x4, "flip_anti"), expected)


def test_2x2_rot90():
    """Verify rot90 on 2×2 square."""
    expected = [[3, 1], [4, 2]]
    assert deep_eq(apply_isometry(g_2x2, "rot90"), expected)


def test_2x2_flip_anti():
    """Verify flip_anti on 2×2 square."""
    expected = [[4, 2], [3, 1]]
    assert deep_eq(apply_isometry(g_2x2, "flip_anti"), expected)


def test_2x3_rot90():
    """Verify rot90 on 2×3 rectangular."""
    expected = [[4, 1], [5, 2], [6, 3]]
    assert deep_eq(apply_isometry(g_2x3, "rot90"), expected)


def test_2x3_flip_anti():
    """Verify flip_anti on 2×3 rectangular."""
    expected = [[6, 3], [5, 2], [4, 1]]
    assert deep_eq(apply_isometry(g_2x3, "flip_anti"), expected)


# ============================================================================
# Exception Tests
# ============================================================================

def test_ragged_input_raises_ValueError():
    """Ragged input should raise ValueError for all transforms."""
    for name in all_isometries():
        with pytest.raises(ValueError, match="rectangular"):
            apply_isometry(g_ragged, name)


def test_unknown_name_raises_KeyError():
    """Unknown isometry name should raise KeyError."""
    g = [[1, 2], [3, 4]]
    with pytest.raises(KeyError, match="Unknown isometry"):
        apply_isometry(g, "unknown")


def test_unknown_name_includes_valid_names():
    """KeyError message should include valid names."""
    g = [[1, 2], [3, 4]]
    try:
        apply_isometry(g, "rotate_90")
    except KeyError as e:
        assert "all_isometries" in str(e) or "id" in str(e)


# ============================================================================
# Determinism Tests
# ============================================================================

def test_same_input_same_output():
    """Applying same transform twice should yield identical results."""
    g = [[1, 2, 3], [4, 5, 6]]

    for name in all_isometries():
        result1 = apply_isometry(g, name)
        result2 = apply_isometry(g, name)
        assert deep_eq(result1, result2), f"Non-deterministic for {name}"


def test_all_isometries_deterministic_order_stable():
    """Calling all_isometries() multiple times should return same order."""
    results = [all_isometries() for _ in range(10)]

    # All should be identical
    for result in results:
        assert result == results[0]
