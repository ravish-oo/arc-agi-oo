"""
Tests for Local Content Masks (P4-03).

This module tests is_color_mask and touching_color_mask from signature_builders.py,
which implement local content predicates for Φ features.

Test coverage:
- is_color_mask: correctness, shape, edge cases, validation
- touching_color_mask: 4-neighbor only (not 8), disjointness with color mask, border handling
- Φ.3 stability (input-only features)
- Purity (input unchanged)
- Determinism (repeated calls identical)
- Error handling (ragged grids, invalid colors)
- Adversarial cases
"""

import copy
import pytest
from src.signature_builders import is_color_mask, touching_color_mask


# ============================================================================
# Test Helpers
# ============================================================================


def assert_mask_shape(mask: list[list[int]], expected_shape: tuple[int, int]) -> None:
    """Verify mask has expected shape."""
    h, w = expected_shape
    assert len(mask) == h, f"Mask height {len(mask)} != expected {h}"
    if h > 0:
        for r, row in enumerate(mask):
            assert len(row) == w, f"Row {r} width {len(row)} != expected {w}"


def assert_disjoint(mask1: list[list[int]], mask2: list[list[int]]) -> None:
    """Verify two masks are disjoint (no pixel is 1 in both)."""
    h = len(mask1)
    if h == 0:
        return
    w = len(mask1[0])

    for r in range(h):
        for c in range(w):
            overlap = mask1[r][c] == 1 and mask2[r][c] == 1
            assert not overlap, f"Masks overlap at ({r},{c})"


def count_ones(mask: list[list[int]]) -> int:
    """Count number of 1s in mask."""
    if not mask:
        return 0
    return sum(sum(row) for row in mask)


# ============================================================================
# is_color_mask Tests
# ============================================================================


def test_is_color_mask_empty():
    """Empty grid returns empty mask."""
    result = is_color_mask([], 5)
    assert result == []


def test_is_color_mask_single_pixel_match():
    """1x1 grid with matching color."""
    g = [[3]]
    mask = is_color_mask(g, 3)

    assert mask == [[1]]
    assert_mask_shape(mask, (1, 1))


def test_is_color_mask_single_pixel_no_match():
    """1x1 grid with non-matching color."""
    g = [[3]]
    mask = is_color_mask(g, 5)

    assert mask == [[0]]
    assert_mask_shape(mask, (1, 1))


def test_is_color_mask_simple_2x2():
    """2x2 grid with some matching pixels."""
    g = [[1, 2], [2, 1]]
    mask = is_color_mask(g, 2)

    assert mask == [[0, 1], [1, 0]]
    assert count_ones(mask) == 2


def test_is_color_mask_color_absent():
    """Color not present in grid → all zeros."""
    g = [[1, 2, 3], [4, 5, 6]]
    mask = is_color_mask(g, 9)

    assert mask == [[0, 0, 0], [0, 0, 0]]
    assert count_ones(mask) == 0


def test_is_color_mask_color_everywhere():
    """All pixels match color → all ones."""
    g = [[7, 7], [7, 7], [7, 7]]
    mask = is_color_mask(g, 7)

    assert mask == [[1, 1], [1, 1], [1, 1]]
    assert count_ones(mask) == 6


def test_is_color_mask_shape_matches_input():
    """Mask shape always matches input grid shape."""
    g = [[1, 2, 3, 4], [5, 6, 7, 8]]
    mask = is_color_mask(g, 3)

    assert_mask_shape(mask, (2, 4))
    assert mask == [[0, 0, 1, 0], [0, 0, 0, 0]]


def test_is_color_mask_all_colors():
    """Test all valid ARC colors (0-9)."""
    g = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    for color in range(10):
        mask = is_color_mask(g, color)
        assert_mask_shape(mask, (3, 3))

        # Count should be 1 for colors 0-8, 0 for color 9
        expected_count = 1 if color <= 8 else 0
        assert count_ones(mask) == expected_count


def test_is_color_mask_purity():
    """Input grid unchanged after is_color_mask call."""
    g = [[1, 2], [3, 4]]
    g_copy = copy.deepcopy(g)

    is_color_mask(g, 2)

    assert g == g_copy, "Input grid modified (purity violation)"


def test_is_color_mask_determinism():
    """Repeated calls with same input produce identical output."""
    g = [[1, 2, 3], [4, 5, 6]]

    mask1 = is_color_mask(g, 5)
    mask2 = is_color_mask(g, 5)

    assert mask1 == mask2, "Non-deterministic behavior detected"


def test_is_color_mask_ragged_input():
    """Ragged grid raises ValueError."""
    g_ragged = [[1, 2], [3]]

    with pytest.raises(ValueError, match="Ragged grid"):
        is_color_mask(g_ragged, 1)


def test_is_color_mask_invalid_color_negative():
    """Color < 0 raises ValueError."""
    g = [[1, 2]]

    with pytest.raises(ValueError, match="Color must be in \\[0..9\\]"):
        is_color_mask(g, -1)


def test_is_color_mask_invalid_color_too_large():
    """Color > 9 raises ValueError."""
    g = [[1, 2]]

    with pytest.raises(ValueError, match="Color must be in \\[0..9\\]"):
        is_color_mask(g, 10)


def test_is_color_mask_single_row():
    """Single row grid."""
    g = [[1, 2, 1, 3, 1]]
    mask = is_color_mask(g, 1)

    assert mask == [[1, 0, 1, 0, 1]]
    assert count_ones(mask) == 3


def test_is_color_mask_single_column():
    """Single column grid."""
    g = [[5], [3], [5], [5]]
    mask = is_color_mask(g, 5)

    assert mask == [[1], [0], [1], [1]]
    assert count_ones(mask) == 3


# ============================================================================
# touching_color_mask Tests
# ============================================================================


def test_touching_color_mask_empty():
    """Empty grid returns empty mask."""
    result = touching_color_mask([], 5)
    assert result == []


def test_touching_color_mask_single_pixel():
    """1x1 grid always returns [[0]] (no neighbors)."""
    # Matching color
    g1 = [[5]]
    mask1 = touching_color_mask(g1, 5)
    assert mask1 == [[0]]

    # Non-matching color
    g2 = [[3]]
    mask2 = touching_color_mask(g2, 5)
    assert mask2 == [[0]]


def test_touching_color_mask_center_pixel():
    """3x3 grid with color in center → 4-neighbor cross pattern."""
    g = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]

    mask = touching_color_mask(g, 5)

    # Expected: cross pattern (4-neighbors only, not diagonals)
    expected = [
        [0, 1, 0],  # (0,1) touches (1,1)
        [1, 0, 1],  # (1,0) and (1,2) touch (1,1); (1,1) is color 5 so excluded
        [0, 1, 0],  # (2,1) touches (1,1)
    ]

    assert mask == expected
    assert count_ones(mask) == 4  # 4 touching pixels


def test_touching_color_mask_corner_pixel():
    """Color in corner → only 2 neighbors."""
    g = [
        [3, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    mask = touching_color_mask(g, 3)

    expected = [
        [0, 1, 0],  # (0,1) touches (0,0)
        [1, 0, 0],  # (1,0) touches (0,0)
        [0, 0, 0],
    ]

    assert mask == expected
    assert count_ones(mask) == 2


def test_touching_color_mask_edge_pixel():
    """Color on edge (not corner) → 3 neighbors."""
    g = [
        [0, 7, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    mask = touching_color_mask(g, 7)

    expected = [
        [1, 0, 1],  # (0,0) and (0,2) touch (0,1)
        [0, 1, 0],  # (1,1) touches (0,1)
        [0, 0, 0],
    ]

    assert mask == expected
    assert count_ones(mask) == 3


def test_touching_color_mask_4_neighbor_only():
    """Verify diagonals are NOT neighbors (4-neighbor, not 8)."""
    # Checkerboard pattern: diagonals should NOT touch
    g = [
        [7, 0, 7],
        [0, 7, 0],
        [7, 0, 7],
    ]

    mask = touching_color_mask(g, 7)

    # Expected: only orthogonal neighbors, not diagonals
    expected = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]

    assert mask == expected


def test_touching_color_mask_all_color():
    """All pixels are the color → no touching (touching excludes color cells)."""
    g = [[2, 2], [2, 2]]
    mask = touching_color_mask(g, 2)

    assert mask == [[0, 0], [0, 0]]
    assert count_ones(mask) == 0


def test_touching_color_mask_color_absent():
    """Color not present → all zeros."""
    g = [[1, 2], [3, 4]]
    mask = touching_color_mask(g, 9)

    assert mask == [[0, 0], [0, 0]]
    assert count_ones(mask) == 0


def test_touching_color_mask_disjoint_from_color_mask():
    """T & M are disjoint (no overlap)."""
    g = [
        [0, 5, 0],
        [5, 0, 5],
        [0, 5, 0],
    ]

    color_mask = is_color_mask(g, 5)
    touching_mask = touching_color_mask(g, 5)

    assert_disjoint(color_mask, touching_mask)


def test_touching_color_mask_horizontal_line():
    """Horizontal line of color."""
    g = [[0, 5, 5, 5, 0]]
    mask = touching_color_mask(g, 5)

    # (0,0) and (0,4) touch the ends
    expected = [[1, 0, 0, 0, 1]]
    assert mask == expected


def test_touching_color_mask_vertical_line():
    """Vertical line of color."""
    g = [[0], [5], [5], [5], [0]]
    mask = touching_color_mask(g, 5)

    # (0,0) and (4,0) touch the ends
    expected = [[1], [0], [0], [0], [1]]
    assert mask == expected


def test_touching_color_mask_border_no_wraparound():
    """Color on all borders → verify no wrap-around."""
    g = [
        [3, 3, 3],
        [3, 0, 3],
        [3, 3, 3],
    ]

    mask = touching_color_mask(g, 3)

    # Only center pixel (1,1) touches color 3
    expected = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    assert mask == expected
    assert count_ones(mask) == 1


def test_touching_color_mask_multiple_regions():
    """Multiple separate regions of color."""
    g = [
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
    ]

    mask = touching_color_mask(g, 1)

    # Each 1 should create touching pixels around it
    # (0,1) touches (0,0); (0,2) touches (0,3)
    # (1,0) touches (0,0) and (2,0); (1,3) touches (0,3) and (2,3)
    # (2,1) touches (2,0); (2,2) touches (2,3)
    expected = [
        [0, 1, 1, 0],  # (0,1) touches (0,0); (0,2) touches (0,3)
        [1, 0, 0, 1],  # (1,0) touches (0,0) and (2,0); (1,3) touches (0,3) and (2,3)
        [0, 1, 1, 0],  # (2,1) touches (2,0); (2,2) touches (2,3)
    ]

    assert mask == expected


def test_touching_color_mask_purity():
    """Input grid unchanged after touching_color_mask call."""
    g = [[0, 5, 0], [5, 0, 5]]
    g_copy = copy.deepcopy(g)

    touching_color_mask(g, 5)

    assert g == g_copy, "Input grid modified (purity violation)"


def test_touching_color_mask_determinism():
    """Repeated calls with same input produce identical output."""
    g = [[1, 2, 1], [2, 1, 2]]

    mask1 = touching_color_mask(g, 1)
    mask2 = touching_color_mask(g, 1)

    assert mask1 == mask2, "Non-deterministic behavior detected"


def test_touching_color_mask_ragged_input():
    """Ragged grid raises ValueError."""
    g_ragged = [[1, 2], [3]]

    with pytest.raises(ValueError, match="Ragged grid"):
        touching_color_mask(g_ragged, 1)


def test_touching_color_mask_invalid_color_negative():
    """Color < 0 raises ValueError."""
    g = [[1, 2]]

    with pytest.raises(ValueError, match="Color must be in \\[0..9\\]"):
        touching_color_mask(g, -1)


def test_touching_color_mask_invalid_color_too_large():
    """Color > 9 raises ValueError."""
    g = [[1, 2]]

    with pytest.raises(ValueError, match="Color must be in \\[0..9\\]"):
        touching_color_mask(g, 10)


def test_touching_color_mask_shape_matches_input():
    """Mask shape always matches input grid shape."""
    g = [[1, 2, 3, 4], [5, 6, 7, 8]]
    mask = touching_color_mask(g, 3)

    assert_mask_shape(mask, (2, 4))


# ============================================================================
# Adversarial Cases (from P4-03 context pack)
# ============================================================================


def test_touching_color_mask_single_pixel_center_adversarial():
    """Adversarial: single color pixel in center of 3x3."""
    g = [
        [0, 0, 0],
        [0, 9, 0],
        [0, 0, 0],
    ]

    color_mask = is_color_mask(g, 9)
    touching_mask = touching_color_mask(g, 9)

    # Color mask: only center
    assert color_mask == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    # Touching mask: 4-neighbor cross
    assert touching_mask == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    # Disjoint
    assert_disjoint(color_mask, touching_mask)

    # Cover all 4 neighbors
    assert count_ones(touching_mask) == 4


def test_touching_color_mask_single_pixel_corner_adversarial():
    """Adversarial: single color pixel in corner."""
    g = [
        [8, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    color_mask = is_color_mask(g, 8)
    touching_mask = touching_color_mask(g, 8)

    # Color mask: only top-left
    assert color_mask == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Touching mask: only 2 neighbors (right and down)
    assert touching_mask == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]

    # Disjoint
    assert_disjoint(color_mask, touching_mask)

    # Only 2 neighbors for corner
    assert count_ones(touching_mask) == 2


def test_touching_color_mask_checker_cross_adversarial():
    """Adversarial: verify 4-neighbor only, diagonals not set."""
    g = [
        [6, 0, 6],
        [0, 6, 0],
        [6, 0, 6],
    ]

    touching_mask = touching_color_mask(g, 6)

    # Expected: only orthogonal, NOT diagonals
    expected = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]

    assert touching_mask == expected

    # Diagonals should be 0 (not touching)
    assert touching_mask[0][0] == 0  # Top-left diagonal
    assert touching_mask[0][2] == 0  # Top-right diagonal
    assert touching_mask[2][0] == 0  # Bottom-left diagonal
    assert touching_mask[2][2] == 0  # Bottom-right diagonal


def test_touching_color_mask_color_absent_adversarial():
    """Adversarial: color not in grid → both masks zero."""
    g = [[1, 2, 3], [4, 5, 6]]

    color_mask = is_color_mask(g, 0)
    touching_mask = touching_color_mask(g, 0)

    # Both should be all zeros
    assert color_mask == [[0, 0, 0], [0, 0, 0]]
    assert touching_mask == [[0, 0, 0], [0, 0, 0]]


def test_touching_color_mask_large_grid():
    """Large grid performance check."""
    # 50x50 grid with color 4 in center
    g = [[0] * 50 for _ in range(50)]
    g[25][25] = 4

    mask = touching_color_mask(g, 4)

    # Should only have 4 touching pixels (center has 4 neighbors)
    assert count_ones(mask) == 4
    assert mask[24][25] == 1  # up
    assert mask[26][25] == 1  # down
    assert mask[25][24] == 1  # left
    assert mask[25][26] == 1  # right


def test_touching_color_mask_all_valid_colors():
    """Test touching for all valid ARC colors (0-9)."""
    g = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]

    for color in range(10):
        mask = touching_color_mask(g, color)
        assert_mask_shape(mask, (3, 3))

        # Verify disjoint from color mask
        color_mask = is_color_mask(g, color)
        assert_disjoint(color_mask, mask)


def test_touching_color_mask_complex_pattern():
    """Complex pattern with multiple color regions."""
    g = [
        [3, 3, 0, 0],
        [3, 0, 0, 3],
        [0, 0, 3, 3],
        [0, 3, 3, 0],
    ]

    mask = touching_color_mask(g, 3)

    # Manually verify a few key pixels
    # (0,2) should touch (0,1)
    assert mask[0][2] == 1

    # (1,1) should touch (0,1) and (1,0)
    assert mask[1][1] == 1

    # All color cells should be 0
    assert mask[0][0] == 0
    assert mask[0][1] == 0
    assert mask[1][0] == 0

    # Verify disjoint
    color_mask = is_color_mask(g, 3)
    assert_disjoint(color_mask, mask)
