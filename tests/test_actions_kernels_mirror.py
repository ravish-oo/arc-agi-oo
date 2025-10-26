"""
Test suite for mirror action kernels (P5-02).

Categories:
1. Purity Tests — Xp unchanged after mirroring
2. Correctness Tests — Mirror semantics (grid-global axes)
3. GLUE Safety Tests — Two-sided coords, commutativity (CRITICAL)
4. Edge Cases — 1×1, odd/even dimensions, empty coords, duplicates
5. Validation Tests — Ragged, empty, out-of-bounds
6. Determinism Tests — Unsorted coords, repeated calls
7. Mathematical Properties — Involution, identity on center

Coverage Target: ≥35 tests
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.action_inference import apply_mirror_h, apply_mirror_v


# ============================================================================
# Category 1: Purity Tests (4 tests)
# ============================================================================


def test_mirror_h_purity():
    """apply_mirror_h does not mutate Xp."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Xp_copy = [row[:] for row in Xp]
    _ = apply_mirror_h(Xp, [(0, 0), (2, 2)])
    assert Xp == Xp_copy


def test_mirror_v_purity():
    """apply_mirror_v does not mutate Xp."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Xp_copy = [row[:] for row in Xp]
    _ = apply_mirror_v(Xp, [(0, 0), (1, 2)])
    assert Xp == Xp_copy


def test_mirror_h_returns_new_object():
    """Result is different object from Xp."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_h(Xp, [(0, 0)])
    assert result is not Xp


def test_mirror_v_returns_new_object():
    """Result is different object from Xp."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_v(Xp, [(0, 0)])
    assert result is not Xp


# ============================================================================
# Category 2: Correctness Tests (12 tests)
# ============================================================================


def test_mirror_h_3x3_single_coord():
    """Horizontal mirror in 3×3 grid."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = apply_mirror_h(Xp, [(0, 1)])
    # Out[0][1] = Xp[2][1] = 8
    assert result == [[1, 8, 3], [4, 5, 6], [7, 8, 9]]


def test_mirror_h_3x3_multiple_coords():
    """Horizontal mirror multiple coords."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    coords = [(0, 0), (0, 2)]
    result = apply_mirror_h(Xp, coords)
    # Out[0][0] = Xp[2][0] = 7
    # Out[0][2] = Xp[2][2] = 9
    assert result == [[7, 2, 9], [4, 5, 6], [7, 8, 9]]


def test_mirror_v_3x3_single_coord():
    """Vertical mirror in 3×3 grid."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = apply_mirror_v(Xp, [(1, 0)])
    # Out[1][0] = Xp[1][2] = 6
    assert result == [[1, 2, 3], [6, 5, 6], [7, 8, 9]]


def test_mirror_v_3x3_multiple_coords():
    """Vertical mirror multiple coords."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    coords = [(0, 0), (2, 2)]
    result = apply_mirror_v(Xp, coords)
    # Out[0][0] = Xp[0][2] = 3
    # Out[2][2] = Xp[2][0] = 7
    assert result == [[3, 2, 3], [4, 5, 6], [7, 8, 7]]


def test_mirror_h_even_rows():
    """Horizontal mirror with even row count."""
    Xp = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 4×2
    result = apply_mirror_h(Xp, [(0, 0)])
    # Out[0][0] = Xp[3][0] = 7
    assert result == [[7, 2], [3, 4], [5, 6], [7, 8]]


def test_mirror_h_odd_rows():
    """Horizontal mirror with odd row count."""
    Xp = [[1, 2], [3, 4], [5, 6]]  # 3×2
    result = apply_mirror_h(Xp, [(0, 0)])
    # Out[0][0] = Xp[2][0] = 5
    assert result == [[5, 2], [3, 4], [5, 6]]


def test_mirror_v_even_cols():
    """Vertical mirror with even column count."""
    Xp = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 2×4
    result = apply_mirror_v(Xp, [(0, 0)])
    # Out[0][0] = Xp[0][3] = 4
    assert result == [[4, 2, 3, 4], [5, 6, 7, 8]]


def test_mirror_v_odd_cols():
    """Vertical mirror with odd column count."""
    Xp = [[1, 2, 3], [4, 5, 6]]  # 2×3
    result = apply_mirror_v(Xp, [(0, 0)])
    # Out[0][0] = Xp[0][2] = 3
    assert result == [[3, 2, 3], [4, 5, 6]]


def test_mirror_h_center_row():
    """Mirror center row in odd-height grid."""
    Xp = [[1, 2], [3, 4], [5, 6]]  # 3×2, center row = 1
    result = apply_mirror_h(Xp, [(1, 0)])
    # Out[1][0] = Xp[3-1-1][0] = Xp[1][0] = 3 (maps to itself)
    assert result == [[1, 2], [3, 4], [5, 6]]


def test_mirror_v_center_col():
    """Mirror center column in odd-width grid."""
    Xp = [[1, 2, 3], [4, 5, 6]]  # 2×3, center col = 1
    result = apply_mirror_v(Xp, [(0, 1)])
    # Out[0][1] = Xp[0][3-1-1] = Xp[0][1] = 2 (maps to itself)
    assert result == [[1, 2, 3], [4, 5, 6]]


def test_mirror_h_bottom_to_top():
    """Mirror bottom row value to top."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_h(Xp, [(1, 1)])
    # Out[1][1] = Xp[2-1-1][1] = Xp[0][1] = 2
    assert result == [[1, 2], [3, 2]]


def test_mirror_v_right_to_left():
    """Mirror rightmost column value to left."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_v(Xp, [(1, 1)])
    # Out[1][1] = Xp[1][2-1-1] = Xp[1][0] = 3
    assert result == [[1, 2], [3, 3]]


# ============================================================================
# Category 3: GLUE Safety Tests (6 tests) — MOST CRITICAL
# ============================================================================


def test_mirror_h_two_sided_coords_GLUE_safety():
    """CRITICAL: Two-sided coords verify GLUE safety.

    Include both (r, c) and (R-1-r, c) in coords.
    If implementation reads from Out, results will be wrong.
    """
    Xp = [[1, 2], [3, 4], [5, 6]]  # 3×2
    # Include both row 0 and row 2 (which mirror to each other)
    coords = [(0, 0), (2, 0)]

    result = apply_mirror_h(Xp, coords)

    # GLUE-safe behavior:
    #   Out[0][0] = Xp[2][0] = 5  (reads from Xp)
    #   Out[2][0] = Xp[0][0] = 1  (reads from Xp)
    # Result: [[5, 2], [3, 4], [1, 6]]

    # GLUE-UNSAFE behavior (if reading from Out):
    #   Processing (0, 0) first: Out[0][0] = Out[2][0] = 5
    #   Processing (2, 0) next:  Out[2][0] = Out[0][0] = 5 (wrong!)
    # Wrong result: [[5, 2], [3, 4], [5, 6]]

    assert result == [[5, 2], [3, 4], [1, 6]], "GLUE violation detected!"


def test_mirror_v_two_sided_coords_GLUE_safety():
    """CRITICAL: Two-sided coords verify GLUE safety for vertical mirror."""
    Xp = [[1, 2, 3], [4, 5, 6]]  # 2×3
    # Include both col 0 and col 2 (which mirror to each other)
    coords = [(0, 0), (0, 2)]

    result = apply_mirror_v(Xp, coords)

    # GLUE-safe:
    #   Out[0][0] = Xp[0][2] = 3
    #   Out[0][2] = Xp[0][0] = 1
    # Result: [[3, 2, 1], [4, 5, 6]]

    assert result == [[3, 2, 1], [4, 5, 6]], "GLUE violation detected!"


def test_mirror_h_three_way_symmetry_GLUE():
    """Three rows mirroring to each other."""
    Xp = [[1], [2], [3], [4], [5]]  # 5×1
    coords = [(0, 0), (2, 0), (4, 0)]

    result = apply_mirror_h(Xp, coords)

    # Out[0][0] = Xp[4][0] = 5
    # Out[2][0] = Xp[2][0] = 3 (center, maps to itself)
    # Out[4][0] = Xp[0][0] = 1
    assert result == [[5], [2], [3], [4], [1]]


def test_mirror_v_three_way_symmetry_GLUE():
    """Three columns mirroring to each other."""
    Xp = [[1, 2, 3, 4, 5]]  # 1×5
    coords = [(0, 0), (0, 2), (0, 4)]

    result = apply_mirror_v(Xp, coords)

    # Out[0][0] = Xp[0][4] = 5
    # Out[0][2] = Xp[0][2] = 3 (center)
    # Out[0][4] = Xp[0][0] = 1
    assert result == [[5, 2, 3, 4, 1]]


def test_mirror_h_commutativity():
    """Processing order should not affect result (GLUE property)."""
    Xp = [[1, 2], [3, 4], [5, 6]]
    coords_order1 = [(0, 0), (2, 0)]
    coords_order2 = [(2, 0), (0, 0)]  # Reversed

    result1 = apply_mirror_h(Xp, coords_order1)
    result2 = apply_mirror_h(Xp, coords_order2)

    # Both should produce same result (deterministic sorting)
    assert result1 == result2


def test_mirror_v_commutativity():
    """Processing order should not affect result (GLUE property)."""
    Xp = [[1, 2, 3], [4, 5, 6]]
    coords_order1 = [(0, 0), (0, 2)]
    coords_order2 = [(0, 2), (0, 0)]  # Reversed

    result1 = apply_mirror_v(Xp, coords_order1)
    result2 = apply_mirror_v(Xp, coords_order2)

    assert result1 == result2


# ============================================================================
# Category 4: Edge Cases (6 tests)
# ============================================================================


def test_mirror_h_1x1_grid():
    """1×1 grid: cell mirrors to itself."""
    Xp = [[5]]
    result = apply_mirror_h(Xp, [(0, 0)])
    # Out[0][0] = Xp[1-1-0][0] = Xp[0][0] = 5
    assert result == [[5]]


def test_mirror_v_1x1_grid():
    """1×1 grid: cell mirrors to itself."""
    Xp = [[5]]
    result = apply_mirror_v(Xp, [(0, 0)])
    # Out[0][0] = Xp[0][1-1-0] = Xp[0][0] = 5
    assert result == [[5]]


def test_mirror_h_empty_coords():
    """Empty coords → identity."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_h(Xp, [])
    assert result == Xp
    assert result is not Xp


def test_mirror_v_empty_coords():
    """Empty coords → identity."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_v(Xp, [])
    assert result == Xp
    assert result is not Xp


def test_mirror_h_duplicate_coords():
    """Duplicate coords handled deterministically."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_h(Xp, [(0, 0), (0, 0)])
    # Last write wins (after sorting, both are same coord)
    # Out[0][0] = Xp[1][0] = 3
    assert result == [[3, 2], [3, 4]]


def test_mirror_v_duplicate_coords():
    """Duplicate coords handled deterministically."""
    Xp = [[1, 2], [3, 4]]
    result = apply_mirror_v(Xp, [(0, 0), (0, 0)])
    # Out[0][0] = Xp[0][1] = 2
    assert result == [[2, 2], [3, 4]]


# ============================================================================
# Category 5: Validation Tests (6 tests)
# ============================================================================


def test_mirror_h_ragged_grid():
    """Ragged grid raises ValueError."""
    Xp = [[1, 2], [3]]  # Ragged
    with pytest.raises(ValueError, match="rectangular"):
        apply_mirror_h(Xp, [(0, 0)])


def test_mirror_v_ragged_grid():
    """Ragged grid raises ValueError."""
    Xp = [[1, 2], [3]]
    with pytest.raises(ValueError, match="rectangular"):
        apply_mirror_v(Xp, [(0, 0)])


def test_mirror_h_empty_grid():
    """Empty grid raises ValueError."""
    Xp = []
    with pytest.raises(ValueError, match="empty"):
        apply_mirror_h(Xp, [(0, 0)])


def test_mirror_v_empty_grid():
    """Empty grid raises ValueError."""
    Xp = []
    with pytest.raises(ValueError, match="empty"):
        apply_mirror_v(Xp, [(0, 0)])


def test_mirror_h_out_of_bounds():
    """Out-of-bounds coord raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_mirror_h(Xp, [(2, 0)])  # Row 2 doesn't exist


def test_mirror_v_out_of_bounds():
    """Out-of-bounds coord raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_mirror_v(Xp, [(0, 2)])  # Col 2 doesn't exist


# ============================================================================
# Category 6: Determinism Tests (2 tests)
# ============================================================================


def test_mirror_h_determinism():
    """Repeated calls produce identical results."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    coords = [(0, 0), (1, 1), (2, 2)]
    result1 = apply_mirror_h(Xp, coords)
    result2 = apply_mirror_h(Xp, coords)
    assert result1 == result2


def test_mirror_v_determinism():
    """Repeated calls produce identical results."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    coords = [(0, 0), (1, 1)]
    result1 = apply_mirror_v(Xp, coords)
    result2 = apply_mirror_v(Xp, coords)
    assert result1 == result2


# ============================================================================
# Category 7: Mathematical Properties (3 tests)
# ============================================================================


def test_mirror_h_involution_full_grid():
    """Mirroring all coords twice returns original (involution)."""
    Xp = [[1, 2], [3, 4]]
    all_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

    once = apply_mirror_h(Xp, all_coords)
    twice = apply_mirror_h(once, all_coords)

    assert twice == Xp


def test_mirror_v_involution_full_grid():
    """Mirroring all coords twice returns original (involution)."""
    Xp = [[1, 2], [3, 4]]
    all_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

    once = apply_mirror_v(Xp, all_coords)
    twice = apply_mirror_v(once, all_coords)

    assert twice == Xp


def test_mirror_identity_on_center():
    """Center coords map to themselves."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3×3

    # Center row (r=1) maps to itself
    result_h = apply_mirror_h(Xp, [(1, 1)])
    assert result_h[1][1] == Xp[1][1]  # 5 stays 5

    # Center col (c=1) maps to itself
    result_v = apply_mirror_v(Xp, [(1, 1)])
    assert result_v[1][1] == Xp[1][1]  # 5 stays 5
