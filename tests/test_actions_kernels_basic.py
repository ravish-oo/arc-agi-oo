"""
Test suite for basic action kernels (P5-01).

Categories:
1. Purity Tests — Xp unchanged after application
2. Correctness Tests — Each function behavior
3. Edge Cases — Empty coords, duplicates, border coords, zeros
4. Validation Tests — Ragged grids, out-of-bounds, invalid color
5. Determinism Tests — Repeated calls, unsorted coords

Coverage Target: ≥40 tests
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.action_inference import apply_set_color, apply_keep_nonzero, apply_identity


# ============================================================================
# Category 1: Purity Tests
# ============================================================================


def test_set_color_purity():
    """apply_set_color does not mutate Xp."""
    Xp = [[1, 2], [3, 4]]
    Xp_copy = [[1, 2], [3, 4]]
    _ = apply_set_color(Xp, [(0, 0)], 5)
    assert Xp == Xp_copy


def test_keep_nonzero_purity():
    """apply_keep_nonzero does not mutate Xp."""
    Xp = [[1, 0, 3], [0, 5, 0]]
    Xp_copy = [[1, 0, 3], [0, 5, 0]]
    _ = apply_keep_nonzero(Xp, [(0, 0), (1, 1)])
    assert Xp == Xp_copy


def test_identity_purity():
    """apply_identity does not mutate Xp."""
    Xp = [[1, 2], [3, 4]]
    Xp_copy = [[1, 2], [3, 4]]
    _ = apply_identity(Xp, [(0, 0)])
    assert Xp == Xp_copy


# ============================================================================
# Category 2: Correctness Tests
# ============================================================================


def test_set_color_single_coord():
    """Set single coordinate."""
    Xp = [[1, 2], [3, 4]]
    result = apply_set_color(Xp, [(0, 0)], 5)
    assert result == [[5, 2], [3, 4]]


def test_set_color_multiple_coords():
    """Set multiple coordinates."""
    Xp = [[1, 2, 3], [4, 5, 6]]
    result = apply_set_color(Xp, [(0, 0), (1, 2)], 7)
    assert result == [[7, 2, 3], [4, 5, 7]]


def test_set_color_diagonal():
    """Set diagonal coordinates."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = apply_set_color(Xp, [(0, 0), (1, 1), (2, 2)], 0)
    assert result == [[0, 2, 3], [4, 0, 6], [7, 8, 0]]


def test_set_color_to_zero():
    """Set coordinates to black (0)."""
    Xp = [[1, 2], [3, 4]]
    result = apply_set_color(Xp, [(0, 1), (1, 0)], 0)
    assert result == [[1, 0], [0, 4]]


def test_set_color_to_nine():
    """Set coordinates to color 9."""
    Xp = [[1, 2], [3, 4]]
    result = apply_set_color(Xp, [(0, 0)], 9)
    assert result == [[9, 2], [3, 4]]


def test_keep_nonzero_mixed():
    """Keep nonzero, zero stays zero."""
    Xp = [[1, 0, 3], [0, 5, 0]]
    result = apply_keep_nonzero(Xp, [(0, 0), (0, 1), (1, 1)])
    # (0,0): 1 != 0 → keep 1
    # (0,1): 0 == 0 → keep 0
    # (1,1): 5 != 0 → keep 5
    assert result == [[1, 0, 3], [0, 5, 0]]


def test_keep_nonzero_all_nonzero():
    """All nonzero values preserved."""
    Xp = [[1, 2], [3, 4]]
    result = apply_keep_nonzero(Xp, [(0, 0), (1, 1)])
    assert result == [[1, 2], [3, 4]]


def test_keep_nonzero_all_zeros():
    """All zeros remain zeros."""
    Xp = [[0, 0], [0, 0]]
    result = apply_keep_nonzero(Xp, [(0, 0), (1, 1)])
    assert result == [[0, 0], [0, 0]]


def test_identity_returns_copy():
    """Identity returns exact copy."""
    Xp = [[1, 2], [3, 4]]
    result = apply_identity(Xp, [(0, 0)])
    assert result == Xp
    assert result is not Xp  # Different object


def test_identity_preserves_all_values():
    """Identity preserves all grid values."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = apply_identity(Xp, [(0, 0), (1, 1), (2, 2)])
    assert result == Xp


# ============================================================================
# Category 3: Edge Cases
# ============================================================================


def test_set_color_empty_coords():
    """Empty coords → identity."""
    Xp = [[1, 2], [3, 4]]
    result = apply_set_color(Xp, [], 5)
    assert result == Xp


def test_set_color_duplicate_coords():
    """Duplicate coords handled deterministically."""
    Xp = [[1, 2], [3, 4]]
    result = apply_set_color(Xp, [(0, 0), (0, 0)], 5)
    assert result == [[5, 2], [3, 4]]


def test_set_color_duplicate_coords_multiple():
    """Multiple duplicate coords."""
    Xp = [[1, 2, 3], [4, 5, 6]]
    result = apply_set_color(Xp, [(0, 0), (1, 1), (0, 0), (1, 1)], 7)
    assert result == [[7, 2, 3], [4, 7, 6]]


def test_set_color_all_coords():
    """Set all coordinates."""
    Xp = [[1, 2], [3, 4]]
    all_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    result = apply_set_color(Xp, all_coords, 7)
    assert result == [[7, 7], [7, 7]]


def test_set_color_border_coords():
    """Set border coordinates only."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    border = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    result = apply_set_color(Xp, border, 0)
    assert result == [[0, 0, 0], [0, 5, 0], [0, 0, 0]]


def test_keep_nonzero_empty_coords():
    """Empty coords → identity."""
    Xp = [[1, 2], [3, 4]]
    result = apply_keep_nonzero(Xp, [])
    assert result == Xp


def test_keep_nonzero_single_zero():
    """Single zero coordinate."""
    Xp = [[1, 0], [3, 4]]
    result = apply_keep_nonzero(Xp, [(0, 1)])
    assert result == [[1, 0], [3, 4]]


def test_keep_nonzero_single_nonzero():
    """Single nonzero coordinate."""
    Xp = [[1, 2], [3, 4]]
    result = apply_keep_nonzero(Xp, [(0, 0)])
    assert result == [[1, 2], [3, 4]]


def test_identity_empty_coords():
    """Identity with empty coords."""
    Xp = [[1, 2], [3, 4]]
    result = apply_identity(Xp, [])
    assert result == Xp


def test_identity_many_coords():
    """Identity ignores coord count."""
    Xp = [[1, 2], [3, 4]]
    many_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    result = apply_identity(Xp, many_coords)
    assert result == Xp


# ============================================================================
# Category 4: Validation Tests
# ============================================================================


def test_set_color_ragged_grid():
    """Ragged grid raises ValueError."""
    Xp = [[1, 2], [3]]  # Ragged
    with pytest.raises(ValueError, match="rectangular"):
        apply_set_color(Xp, [(0, 0)], 5)


def test_set_color_invalid_color_negative():
    """Color < 0 raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of range"):
        apply_set_color(Xp, [(0, 0)], -1)


def test_set_color_invalid_color_too_large():
    """Color > 9 raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of range"):
        apply_set_color(Xp, [(0, 0)], 10)


def test_set_color_invalid_color_eleven():
    """Color = 11 raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of range"):
        apply_set_color(Xp, [(0, 0)], 11)


def test_set_color_coord_out_of_bounds_row():
    """Row out of bounds raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_set_color(Xp, [(2, 0)], 5)  # Row 2 doesn't exist


def test_set_color_coord_out_of_bounds_col():
    """Column out of bounds raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_set_color(Xp, [(0, 2)], 5)  # Col 2 doesn't exist


def test_set_color_coord_negative_row():
    """Negative row raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_set_color(Xp, [(-1, 0)], 5)


def test_set_color_coord_negative_col():
    """Negative column raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_set_color(Xp, [(0, -1)], 5)


def test_keep_nonzero_ragged_grid():
    """Ragged grid raises ValueError."""
    Xp = [[1, 2, 3], [4]]  # Ragged
    with pytest.raises(ValueError, match="rectangular"):
        apply_keep_nonzero(Xp, [(0, 0)])


def test_keep_nonzero_coord_out_of_bounds_row():
    """Row out of bounds raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_keep_nonzero(Xp, [(3, 0)])


def test_keep_nonzero_coord_out_of_bounds_col():
    """Column out of bounds raises ValueError."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_keep_nonzero(Xp, [(0, 5)])


def test_identity_ragged_grid():
    """Ragged grid raises ValueError."""
    Xp = [[1, 2], [3, 4, 5]]  # Ragged
    with pytest.raises(ValueError, match="rectangular"):
        apply_identity(Xp, [(0, 0)])


def test_identity_coord_out_of_bounds():
    """Out of bounds coord raises ValueError (even though ignored)."""
    Xp = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_identity(Xp, [(10, 10)])


# ============================================================================
# Category 5: Determinism Tests
# ============================================================================


def test_set_color_determinism():
    """Repeated calls produce identical results."""
    Xp = [[1, 2], [3, 4]]
    coords = [(0, 0), (1, 1)]
    result1 = apply_set_color(Xp, coords, 5)
    result2 = apply_set_color(Xp, coords, 5)
    assert result1 == result2


def test_set_color_unsorted_coords():
    """Unsorted coords produce same result as sorted."""
    Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    unsorted = [(2, 2), (0, 0), (1, 1)]
    sorted_coords = [(0, 0), (1, 1), (2, 2)]

    result_unsorted = apply_set_color(Xp, unsorted, 5)
    result_sorted = apply_set_color(Xp, sorted_coords, 5)
    assert result_unsorted == result_sorted


def test_set_color_reverse_order_coords():
    """Reverse order coords produce same result."""
    Xp = [[1, 2, 3], [4, 5, 6]]
    forward = [(0, 0), (0, 1), (1, 0), (1, 1)]
    reverse = [(1, 1), (1, 0), (0, 1), (0, 0)]

    result_forward = apply_set_color(Xp, forward, 7)
    result_reverse = apply_set_color(Xp, reverse, 7)
    assert result_forward == result_reverse


def test_keep_nonzero_determinism():
    """Repeated calls produce identical results."""
    Xp = [[1, 0, 3], [0, 5, 0]]
    coords = [(0, 0), (0, 1), (1, 1)]
    result1 = apply_keep_nonzero(Xp, coords)
    result2 = apply_keep_nonzero(Xp, coords)
    assert result1 == result2


def test_keep_nonzero_unsorted_coords():
    """Unsorted coords produce same result as sorted."""
    Xp = [[1, 0, 3], [0, 5, 0]]
    unsorted = [(1, 1), (0, 0), (0, 1)]
    sorted_coords = [(0, 0), (0, 1), (1, 1)]

    result_unsorted = apply_keep_nonzero(Xp, unsorted)
    result_sorted = apply_keep_nonzero(Xp, sorted_coords)
    assert result_unsorted == result_sorted


def test_identity_determinism():
    """Repeated calls produce identical results."""
    Xp = [[1, 2, 3], [4, 5, 6]]
    coords = [(0, 0), (1, 1)]
    result1 = apply_identity(Xp, coords)
    result2 = apply_identity(Xp, coords)
    assert result1 == result2


def test_identity_different_coords_same_result():
    """Different coords produce same result (all ignored)."""
    Xp = [[1, 2], [3, 4]]
    coords1 = [(0, 0)]
    coords2 = [(1, 1)]

    result1 = apply_identity(Xp, coords1)
    result2 = apply_identity(Xp, coords2)
    assert result1 == result2


# ============================================================================
# Category 6: Empty Grid Edge Cases
# ============================================================================


def test_set_color_empty_grid():
    """Empty grid with empty coords."""
    Xp = []
    result = apply_set_color(Xp, [], 5)
    assert result == []


def test_keep_nonzero_empty_grid():
    """Empty grid with empty coords."""
    Xp = []
    result = apply_keep_nonzero(Xp, [])
    assert result == []


def test_identity_empty_grid():
    """Empty grid with empty coords."""
    Xp = []
    result = apply_identity(Xp, [])
    assert result == []


def test_set_color_empty_grid_with_coords_fails():
    """Empty grid with non-empty coords raises ValueError."""
    Xp = []
    with pytest.raises(ValueError, match="out of bounds"):
        apply_set_color(Xp, [(0, 0)], 5)


# ============================================================================
# Category 7: Single Pixel Grid
# ============================================================================


def test_set_color_single_pixel():
    """Single pixel grid."""
    Xp = [[7]]
    result = apply_set_color(Xp, [(0, 0)], 3)
    assert result == [[3]]


def test_keep_nonzero_single_pixel_nonzero():
    """Single nonzero pixel."""
    Xp = [[7]]
    result = apply_keep_nonzero(Xp, [(0, 0)])
    assert result == [[7]]


def test_keep_nonzero_single_pixel_zero():
    """Single zero pixel."""
    Xp = [[0]]
    result = apply_keep_nonzero(Xp, [(0, 0)])
    assert result == [[0]]


def test_identity_single_pixel():
    """Single pixel grid."""
    Xp = [[5]]
    result = apply_identity(Xp, [(0, 0)])
    assert result == [[5]]
