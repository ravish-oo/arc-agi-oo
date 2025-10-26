"""
Test suite for P6-01: compute_residual(Xp, Y)

22 tests across 5 categories:
- Correctness: basic behavior, edge cases
- Validation: error handling for invalid inputs
- Purity: no mutation, no aliasing
- Determinism: stable across repeated calls
- Adversarial: edge patterns, boundary values
"""

import pytest
from src.glue import compute_residual


# ============================================================================
# A. CORRECTNESS TESTS (9 tests)
# ============================================================================

def test_residual_empty_grid():
    """Empty grid returns empty residual."""
    R = compute_residual([], [])
    assert R == []


def test_residual_1x1_equal():
    """Single pixel, equal: residual is None."""
    R = compute_residual([[5]], [[5]])
    assert R == [[None]]


def test_residual_1x1_different():
    """Single pixel, different: residual is target value."""
    R = compute_residual([[3]], [[7]])
    assert R == [[7]]


def test_residual_all_equal_2x3():
    """All pixels match: residual is all None."""
    Xp = [[1, 2, 3], [4, 5, 6]]
    Y = [[1, 2, 3], [4, 5, 6]]
    R = compute_residual(Xp, Y)
    assert R == [[None, None, None], [None, None, None]]


def test_residual_all_different_2x3():
    """No pixels match: residual equals target."""
    Xp = [[0, 0, 0], [0, 0, 0]]
    Y = [[1, 2, 3], [4, 5, 6]]
    R = compute_residual(Xp, Y)
    assert R == [[1, 2, 3], [4, 5, 6]]


def test_residual_mixed_middle_column():
    """Only middle column differs: residual marks it."""
    Xp = [[1, 0, 3], [4, 0, 6]]
    Y = [[1, 2, 3], [4, 5, 6]]
    R = compute_residual(Xp, Y)
    expected = [[None, 2, None], [None, 5, None]]
    assert R == expected


def test_residual_mixed_diagonal():
    """Diagonal differs: residual marks diagonal only."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Y = [[9, 2, 3], [4, 9, 6], [7, 8, 9]]
    R = compute_residual(Xp, Y)
    expected = [[9, None, None], [None, 9, None], [None, None, None]]
    assert R == expected


def test_residual_large_sparse():
    """10x10 grid with one row different."""
    # Build grids: all rows identical except row 5
    Xp = [[i for i in range(10)] for _ in range(10)]
    Y = [[i for i in range(10)] for r in range(10)]
    # Change row 5 in Y
    Y[5] = [(i + 1) % 10 for i in range(10)]

    R = compute_residual(Xp, Y)

    # Verify row 5 has residual [1,2,3,4,5,6,7,8,9,0]
    assert R[5] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    # Verify all other rows are all None
    for r in range(10):
        if r != 5:
            assert R[r] == [None] * 10


def test_residual_boundary_values():
    """Values 0 and 9 at boundaries correctly handled."""
    Xp = [[0, 9, 0, 9]]
    Y = [[0, 9, 9, 0]]
    R = compute_residual(Xp, Y)
    expected = [[None, None, 9, 0]]
    assert R == expected


# ============================================================================
# B. VALIDATION TESTS (4 tests)
# ============================================================================

def test_residual_ragged_Xp_raises():
    """Ragged Xp raises ValueError with clear message."""
    Xp = [[1, 2], [3]]  # ragged!
    Y = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="ragged"):
        compute_residual(Xp, Y)


def test_residual_ragged_Y_raises():
    """Ragged Y raises ValueError with clear message."""
    Xp = [[1, 2], [3, 4]]
    Y = [[1, 2], [3]]  # ragged!
    with pytest.raises(ValueError, match="ragged"):
        compute_residual(Xp, Y)


def test_residual_shape_mismatch_raises():
    """Shape mismatch raises ValueError with both shapes."""
    Xp = [[1, 2, 3]]  # 1x3
    Y = [[1, 2], [3, 4]]  # 2x2
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_residual(Xp, Y)


def test_residual_empty_vs_nonempty_raises():
    """Empty vs non-empty grids raise shape mismatch."""
    # Empty Xp, non-empty Y
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_residual([], [[1]])

    # Non-empty Xp, empty Y
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_residual([[1]], [])


# ============================================================================
# C. PURITY TESTS (3 tests)
# ============================================================================

def test_residual_does_not_mutate_Xp():
    """Xp is never mutated by compute_residual."""
    Xp_orig = [[1, 2], [3, 4]]
    Xp = [row[:] for row in Xp_orig]  # Copy for safety
    Y = [[1, 0], [0, 4]]

    R = compute_residual(Xp, Y)

    # Verify Xp unchanged
    assert Xp == Xp_orig


def test_residual_does_not_mutate_Y():
    """Y is never mutated by compute_residual."""
    Y_orig = [[1, 2], [3, 4]]
    Y = [row[:] for row in Y_orig]  # Copy for safety
    Xp = [[1, 0], [0, 4]]

    R = compute_residual(Xp, Y)

    # Verify Y unchanged
    assert Y == Y_orig


def test_residual_no_aliasing():
    """Residual R has no aliasing to Xp or Y."""
    Xp = [[1, 2], [3, 4]]
    Y = [[1, 0], [0, 4]]

    R = compute_residual(Xp, Y)

    # Mutate R
    R[0][0] = 999

    # Verify Xp and Y unchanged
    assert Xp[0][0] == 1
    assert Y[0][0] == 1


# ============================================================================
# D. DETERMINISM TESTS (2 tests)
# ============================================================================

def test_residual_deterministic_repeated_calls():
    """Same inputs always produce identical output."""
    Xp = [[1, 2, 3], [4, 5, 6]]
    Y = [[1, 0, 3], [0, 5, 0]]

    R1 = compute_residual(Xp, Y)
    R2 = compute_residual(Xp, Y)

    # Exact same result
    assert R1 == R2


def test_residual_deterministic_order_independent():
    """Order independence (vacuous but tests stability)."""
    # This test is vacuous since compute_residual is not sensitive to
    # input order, but we include it for completeness per spec
    Xp = [[1, 2], [3, 4]]
    Y = [[5, 6], [7, 8]]

    R = compute_residual(Xp, Y)

    # Expected: all different, so R == Y
    assert R == [[5, 6], [7, 8]]


# ============================================================================
# E. ADVERSARIAL / EDGE TESTS (4 tests)
# ============================================================================

def test_residual_all_zeros():
    """All zeros, equal: residual is all None."""
    Xp = [[0, 0], [0, 0]]
    Y = [[0, 0], [0, 0]]
    R = compute_residual(Xp, Y)
    assert R == [[None, None], [None, None]]


def test_residual_all_nines():
    """All nines, equal: residual is all None."""
    Xp = [[9, 9], [9, 9]]
    Y = [[9, 9], [9, 9]]
    R = compute_residual(Xp, Y)
    assert R == [[None, None], [None, None]]


def test_residual_alternating_pattern():
    """Alternating pattern: half pixels differ."""
    Xp = [[0, 1, 0, 1], [1, 0, 1, 0]]
    Y = [[1, 1, 1, 1], [1, 1, 1, 1]]

    R = compute_residual(Xp, Y)

    # Expected: pixels where Xp != Y get Y value (1)
    # Row 0: [0→1, 1→None, 0→1, 1→None]
    # Row 1: [1→None, 0→1, 1→None, 0→1]
    expected = [[1, None, 1, None], [None, 1, None, 1]]
    assert R == expected


def test_residual_single_pixel_differs():
    """Only one pixel differs in entire grid."""
    Xp = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    Y = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]

    R = compute_residual(Xp, Y)

    # Only R[1][1] should be 0, rest None
    assert R[1][1] == 0
    assert all(
        R[r][c] is None
        for r in range(3)
        for c in range(3)
        if (r, c) != (1, 1)
    )
