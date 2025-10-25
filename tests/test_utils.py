"""
Comprehensive tests for src/utils.py grid utilities.

Tests cover:
- Basic functionality
- Deep copy no-aliasing
- Deep equality
- D8 group properties (idempotence)
- Composition identities
- Edge cases
- Determinism
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.utils import (
    dims, copy_grid, deep_eq, transpose,
    rot90, rot180, rot270, flip_h, flip_v
)


# ============================================================================
# Test Fixtures
# ============================================================================

# Empty grid
g0 = []

# Single pixel
g1 = [[5]]

# Rectangular 2x3
g2 = [[1, 2, 3],
      [4, 5, 6]]

# Rectangular 3x2 (transpose of g2)
g3 = [[1, 4],
      [2, 5],
      [3, 6]]

# Square 3x3
g4 = [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]]

# Wide 1xN
g_wide = [[1, 2, 3, 4, 5]]

# Tall Nx1
g_tall = [[1], [2], [3], [4], [5]]

# Ragged (invalid)
g_ragged = [[1, 2], [3]]


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_dims_empty():
    assert dims(g0) == (0, 0)


def test_dims_single():
    assert dims(g1) == (1, 1)


def test_dims_rectangular_2x3():
    assert dims(g2) == (2, 3)


def test_dims_rectangular_3x2():
    assert dims(g3) == (3, 2)


def test_dims_square():
    assert dims(g4) == (3, 3)


def test_dims_wide():
    assert dims(g_wide) == (1, 5)


def test_dims_tall():
    assert dims(g_tall) == (5, 1)


def test_dims_ragged_raises():
    with pytest.raises(ValueError, match="rectangular"):
        dims(g_ragged)


# ============================================================================
# Deep Copy No-Aliasing Tests
# ============================================================================

def test_copy_grid_equality():
    """Copied grid should be equal to original."""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        assert deep_eq(copy_grid(g), g)


def test_copy_grid_no_aliasing():
    """Modifying copy should not affect original."""
    g_orig = [[1, 2], [3, 4]]
    g_copy = copy_grid(g_orig)

    # Modify copy
    g_copy[0][0] = 99

    # Original unchanged
    assert g_orig[0][0] == 1
    assert g_copy[0][0] == 99


def test_copy_grid_no_outer_aliasing():
    """Outer list should also be a new object."""
    g_orig = [[1, 2], [3, 4]]
    g_copy = copy_grid(g_orig)

    # Append to copy
    g_copy.append([5, 6])

    # Original unchanged
    assert len(g_orig) == 2
    assert len(g_copy) == 3


# ============================================================================
# Deep Equality Tests
# ============================================================================

def test_deep_eq_empty():
    assert deep_eq([], [])


def test_deep_eq_single():
    assert deep_eq([[1]], [[1]])


def test_deep_eq_identical():
    assert deep_eq(g2, g2)
    assert deep_eq(g4, g4)


def test_deep_eq_value_mismatch():
    assert not deep_eq([[1, 2]], [[1, 3]])


def test_deep_eq_shape_mismatch_cols():
    assert not deep_eq([[1, 2]], [[1, 2, 3]])


def test_deep_eq_shape_mismatch_rows():
    assert not deep_eq([[1, 2]], [[1, 2], [3, 4]])


def test_deep_eq_reflexive():
    """Every grid equals itself."""
    for g in [g0, g1, g2, g3, g4]:
        assert deep_eq(g, g)


def test_deep_eq_symmetric():
    """If a == b, then b == a."""
    g_a = [[1, 2], [3, 4]]
    g_b = [[1, 2], [3, 4]]
    assert deep_eq(g_a, g_b)
    assert deep_eq(g_b, g_a)


# ============================================================================
# D8 Group Properties (Idempotence)
# ============================================================================

def test_rot90_idempotence():
    """rot90^4 = identity"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        result = rot90(rot90(rot90(rot90(g))))
        assert deep_eq(result, g), f"rot90^4 failed for grid {g}"


def test_rot180_idempotence():
    """rot180^2 = identity"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        result = rot180(rot180(g))
        assert deep_eq(result, g), f"rot180^2 failed for grid {g}"


def test_transpose_idempotence():
    """transpose^2 = identity"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        result = transpose(transpose(g))
        assert deep_eq(result, g), f"transpose^2 failed for grid {g}"


def test_flip_h_idempotence():
    """flip_h^2 = identity"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        result = flip_h(flip_h(g))
        assert deep_eq(result, g), f"flip_h^2 failed for grid {g}"


def test_flip_v_idempotence():
    """flip_v^2 = identity"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        result = flip_v(flip_v(g))
        assert deep_eq(result, g), f"flip_v^2 failed for grid {g}"


# ============================================================================
# Composition Identity Tests
# ============================================================================

def test_rot180_equals_rot90_twice():
    """rot180(g) == rot90(rot90(g))"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        assert deep_eq(rot180(g), rot90(rot90(g)))


def test_rot270_equals_rot90_rot180():
    """rot270(g) == rot90(rot180(g))"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        assert deep_eq(rot270(g), rot90(rot180(g)))


def test_rot270_equals_rot180_rot90():
    """rot270(g) == rot180(rot90(g))"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        assert deep_eq(rot270(g), rot180(rot90(g)))


def test_rot90_rot270_identity():
    """rot90(rot270(g)) == g and rot270(rot90(g)) == g"""
    for g in [g0, g1, g2, g3, g4, g_wide, g_tall]:
        assert deep_eq(rot90(rot270(g)), g)
        assert deep_eq(rot270(rot90(g)), g)


def test_transpose_flip_commutation_square():
    """transpose(flip_h(g)) == flip_v(transpose(g)) for square grids"""
    for g in [g0, g1, g4]:  # Only square grids
        assert deep_eq(transpose(flip_h(g)), flip_v(transpose(g)))


# ============================================================================
# Specific Transformation Tests
# ============================================================================

def test_transpose_2x3():
    """Specific test for transpose of 2x3 grid."""
    expected = [[1, 4], [2, 5], [3, 6]]
    assert deep_eq(transpose(g2), expected)


def test_rot90_2x3():
    """Specific test for rot90 of 2x3 grid."""
    expected = [[4, 1], [5, 2], [6, 3]]
    assert deep_eq(rot90(g2), expected)


def test_rot180_2x3():
    """Specific test for rot180 of 2x3 grid."""
    expected = [[6, 5, 4], [3, 2, 1]]
    assert deep_eq(rot180(g2), expected)


def test_flip_h_2x3():
    """Specific test for flip_h of 2x3 grid."""
    expected = [[3, 2, 1], [6, 5, 4]]
    assert deep_eq(flip_h(g2), expected)


def test_flip_v_2x3():
    """Specific test for flip_v of 2x3 grid."""
    expected = [[4, 5, 6], [1, 2, 3]]
    assert deep_eq(flip_v(g2), expected)


# ============================================================================
# Edge Cases
# ============================================================================

def test_all_transforms_empty():
    """All transforms on [] return []."""
    assert transpose(g0) == []
    assert rot90(g0) == []
    assert rot180(g0) == []
    assert rot270(g0) == []
    assert flip_h(g0) == []
    assert flip_v(g0) == []


def test_all_transforms_single_pixel():
    """All transforms on [[x]] return [[x]]."""
    assert deep_eq(transpose(g1), g1)
    assert deep_eq(rot90(g1), g1)
    assert deep_eq(rot180(g1), g1)
    assert deep_eq(rot270(g1), g1)
    assert deep_eq(flip_h(g1), g1)
    assert deep_eq(flip_v(g1), g1)


def test_wide_grid_1xN():
    """Wide grid 1xN transforms correctly."""
    assert dims(g_wide) == (1, 5)
    assert dims(transpose(g_wide)) == (5, 1)
    assert dims(rot90(g_wide)) == (5, 1)


def test_tall_grid_Nx1():
    """Tall grid Nx1 transforms correctly."""
    assert dims(g_tall) == (5, 1)
    assert dims(transpose(g_tall)) == (1, 5)
    assert dims(rot90(g_tall)) == (1, 5)


def test_ragged_transpose_raises():
    with pytest.raises(ValueError):
        transpose(g_ragged)


def test_ragged_rot90_raises():
    with pytest.raises(ValueError):
        rot90(g_ragged)


def test_ragged_flip_h_raises():
    with pytest.raises(ValueError):
        flip_h(g_ragged)


def test_ragged_flip_v_raises():
    with pytest.raises(ValueError):
        flip_v(g_ragged)


# ============================================================================
# Determinism Tests
# ============================================================================

def test_determinism_all_ops():
    """Running operations twice should yield identical results."""
    g = [[1, 2, 3], [4, 5, 6]]

    # Run each operation twice
    assert dims(g) == dims(g)  # dims returns tuple
    assert deep_eq(copy_grid(g), copy_grid(g))
    assert deep_eq(transpose(g), transpose(g))
    assert deep_eq(rot90(g), rot90(g))
    assert deep_eq(rot180(g), rot180(g))
    assert deep_eq(rot270(g), rot270(g))
    assert deep_eq(flip_h(g), flip_h(g))
    assert deep_eq(flip_v(g), flip_v(g))


def test_determinism_hash_stability():
    """Hash of repr should be stable (spot check)."""
    g = [[1, 2], [3, 4]]

    # Run operations and check hash stability
    h1 = hash(repr(rot90(g)))
    h2 = hash(repr(rot90(g)))
    assert h1 == h2


# ============================================================================
# Purity Tests (No Mutation)
# ============================================================================

def test_purity_no_mutation():
    """No operation should mutate the input grid."""
    g_orig = [[1, 2, 3], [4, 5, 6]]
    g_snapshot = copy_grid(g_orig)

    # Run all operations
    _ = dims(g_orig)
    _ = copy_grid(g_orig)
    _ = transpose(g_orig)
    _ = rot90(g_orig)
    _ = rot180(g_orig)
    _ = rot270(g_orig)
    _ = flip_h(g_orig)
    _ = flip_v(g_orig)

    # Original unchanged
    assert deep_eq(g_orig, g_snapshot)
