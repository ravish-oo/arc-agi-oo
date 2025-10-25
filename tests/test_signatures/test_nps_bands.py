"""
Tests for NPS Band Masks (P4-02).

This module tests row_band_masks and col_band_masks from signature_builders.py,
which implement Non-Periodic Segmentation (NPS) based on content-change boundaries.

Test coverage:
- Disjoint+cover property verification
- Empty grids → []
- Single row/col → one band
- All rows/cols equal → one band
- Content changes → multiple bands
- Φ.3 stability (same input structure → identical masks)
- Purity (input unchanged)
- Determinism (repeated calls identical)
- Adversarial cases

All tests verify that masks form disjoint partitions covering the entire grid.
"""

import copy
import pytest
from src.signature_builders import row_band_masks, col_band_masks


# ============================================================================
# Test Helpers
# ============================================================================


def assert_disjoint_and_cover(
    masks: list[list[list[int]]], shape: tuple[int, int]
) -> None:
    """
    Verify that masks form a disjoint partition covering the entire grid.

    For each pixel (r,c), exactly one mask must have value 1.

    Args:
        masks: List of 0/1 masks
        shape: (height, width) of grid

    Raises:
        AssertionError: If disjoint+cover invariant violated
    """
    h, w = shape
    for r in range(h):
        for c in range(w):
            count = sum(masks[i][r][c] for i in range(len(masks)))
            assert count == 1, f"Pixel ({r},{c}) covered {count} times (expected 1)"


def assert_mask_shape(mask: list[list[int]], expected_shape: tuple[int, int]) -> None:
    """Verify mask has expected shape."""
    h, w = expected_shape
    assert len(mask) == h, f"Mask height {len(mask)} != expected {h}"
    if h > 0:
        for r, row in enumerate(mask):
            assert len(row) == w, f"Row {r} width {len(row)} != expected {w}"


# ============================================================================
# row_band_masks Tests
# ============================================================================


def test_row_band_masks_empty():
    """Empty grid returns empty list (no bands)."""
    result = row_band_masks([])
    assert result == []


def test_row_band_masks_single_row():
    """Single row → one band covering entire grid."""
    g = [[1, 2, 3]]
    masks = row_band_masks(g)

    assert len(masks) == 1, "Single row should produce 1 band"
    assert_mask_shape(masks[0], (1, 3))
    assert masks[0] == [[1, 1, 1]]
    assert_disjoint_and_cover(masks, (1, 3))


def test_row_band_masks_single_column():
    """Single column with changing values → multiple bands."""
    g = [[1], [1], [2]]
    masks = row_band_masks(g)

    # Rows 0-1 identical (band 0), row 2 different (band 1)
    assert len(masks) == 2
    assert_mask_shape(masks[0], (3, 1))
    assert_mask_shape(masks[1], (3, 1))

    assert masks[0] == [[1], [1], [0]]  # Band 0: rows 0-1
    assert masks[1] == [[0], [0], [1]]  # Band 1: row 2

    assert_disjoint_and_cover(masks, (3, 1))


def test_row_band_masks_all_rows_equal():
    """All rows identical → one band."""
    g = [[1, 2], [1, 2], [1, 2]]
    masks = row_band_masks(g)

    assert len(masks) == 1, "All equal rows should produce 1 band"
    assert masks[0] == [[1, 1], [1, 1], [1, 1]]
    assert_disjoint_and_cover(masks, (3, 2))


def test_row_band_masks_all_rows_different():
    """Each row different → n bands (one per row)."""
    g = [[1, 1], [2, 2], [3, 3]]
    masks = row_band_masks(g)

    assert len(masks) == 3
    assert masks[0] == [[1, 1], [0, 0], [0, 0]]  # Row 0
    assert masks[1] == [[0, 0], [1, 1], [0, 0]]  # Row 1
    assert masks[2] == [[0, 0], [0, 0], [1, 1]]  # Row 2

    assert_disjoint_and_cover(masks, (3, 2))


def test_row_band_masks_contiguous_bands():
    """Rows with content changes create contiguous bands."""
    g = [
        [1, 2],  # Band 0
        [1, 2],  # Band 0 (same as row 0)
        [3, 4],  # Band 1 (different from row 1)
        [5, 6],  # Band 2 (different from row 2)
        [5, 6],  # Band 2 (same as row 3)
    ]
    masks = row_band_masks(g)

    assert len(masks) == 3
    assert masks[0] == [[1, 1], [1, 1], [0, 0], [0, 0], [0, 0]]  # Rows 0-1
    assert masks[1] == [[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]  # Row 2
    assert masks[2] == [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1]]  # Rows 3-4

    assert_disjoint_and_cover(masks, (5, 2))


def test_row_band_masks_1x1_grid():
    """1x1 grid → single band."""
    g = [[7]]
    masks = row_band_masks(g)

    assert len(masks) == 1
    assert masks[0] == [[1]]
    assert_disjoint_and_cover(masks, (1, 1))


def test_row_band_masks_purity():
    """Input grid unchanged after row_band_masks call."""
    g = [[1, 2], [3, 4], [3, 4]]
    g_copy = copy.deepcopy(g)

    row_band_masks(g)

    assert g == g_copy, "Input grid modified (purity violation)"


def test_row_band_masks_determinism():
    """Repeated calls with same input produce identical output."""
    g = [[1, 1], [2, 2], [1, 1]]

    masks1 = row_band_masks(g)
    masks2 = row_band_masks(g)

    assert masks1 == masks2, "Non-deterministic behavior detected"


def test_row_band_masks_stability():
    """Φ.3 stability: masks depend on structure, not values (input-only)."""
    # Two grids with same structure (row boundaries at same positions)
    g1 = [[1, 2], [1, 2], [3, 4]]
    g2 = [[5, 6], [5, 6], [7, 8]]

    masks1 = row_band_masks(g1)
    masks2 = row_band_masks(g2)

    # Same band structure (band 0: rows 0-1, band 1: row 2)
    assert len(masks1) == len(masks2)
    assert masks1 == masks2, "Masks differ despite same row structure"


def test_row_band_masks_ragged_input():
    """Ragged grid raises ValueError."""
    g_ragged = [[1, 2], [3]]  # Row 1 shorter

    with pytest.raises(ValueError, match="Ragged grid"):
        row_band_masks(g_ragged)


# ============================================================================
# col_band_masks Tests
# ============================================================================


def test_col_band_masks_empty():
    """Empty grid returns empty list (no bands)."""
    result = col_band_masks([])
    assert result == []


def test_col_band_masks_single_column():
    """Single column → one band covering entire grid."""
    g = [[1], [2], [3]]
    masks = col_band_masks(g)

    assert len(masks) == 1, "Single column should produce 1 band"
    assert_mask_shape(masks[0], (3, 1))
    assert masks[0] == [[1], [1], [1]]
    assert_disjoint_and_cover(masks, (3, 1))


def test_col_band_masks_single_row():
    """Single row with changing values → multiple bands."""
    g = [[1, 1, 2]]
    masks = col_band_masks(g)

    # Cols 0-1 identical (band 0), col 2 different (band 1)
    assert len(masks) == 2
    assert_mask_shape(masks[0], (1, 3))
    assert_mask_shape(masks[1], (1, 3))

    assert masks[0] == [[1, 1, 0]]  # Band 0: cols 0-1
    assert masks[1] == [[0, 0, 1]]  # Band 1: col 2

    assert_disjoint_and_cover(masks, (1, 3))


def test_col_band_masks_all_cols_equal():
    """All columns identical → one band."""
    g = [[1, 1, 1], [2, 2, 2]]
    masks = col_band_masks(g)

    assert len(masks) == 1, "All equal columns should produce 1 band"
    assert masks[0] == [[1, 1, 1], [1, 1, 1]]
    assert_disjoint_and_cover(masks, (2, 3))


def test_col_band_masks_all_cols_different():
    """Each column different → n bands (one per column)."""
    g = [[1, 2, 3], [1, 2, 3]]
    masks = col_band_masks(g)

    assert len(masks) == 3
    assert masks[0] == [[1, 0, 0], [1, 0, 0]]  # Col 0
    assert masks[1] == [[0, 1, 0], [0, 1, 0]]  # Col 1
    assert masks[2] == [[0, 0, 1], [0, 0, 1]]  # Col 2

    assert_disjoint_and_cover(masks, (2, 3))


def test_col_band_masks_contiguous_bands():
    """Columns with content changes create contiguous bands."""
    g = [
        [1, 1, 3, 5, 5],  # Col changes: 0-1 same, 2 diff, 3-4 same
        [2, 2, 4, 6, 6],
    ]
    masks = col_band_masks(g)

    assert len(masks) == 3
    assert masks[0] == [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]  # Cols 0-1
    assert masks[1] == [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]  # Col 2
    assert masks[2] == [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]  # Cols 3-4

    assert_disjoint_and_cover(masks, (2, 5))


def test_col_band_masks_1x1_grid():
    """1x1 grid → single band."""
    g = [[7]]
    masks = col_band_masks(g)

    assert len(masks) == 1
    assert masks[0] == [[1]]
    assert_disjoint_and_cover(masks, (1, 1))


def test_col_band_masks_purity():
    """Input grid unchanged after col_band_masks call."""
    g = [[1, 2, 2], [3, 4, 4]]
    g_copy = copy.deepcopy(g)

    col_band_masks(g)

    assert g == g_copy, "Input grid modified (purity violation)"


def test_col_band_masks_determinism():
    """Repeated calls with same input produce identical output."""
    g = [[1, 2, 1], [1, 2, 1]]

    masks1 = col_band_masks(g)
    masks2 = col_band_masks(g)

    assert masks1 == masks2, "Non-deterministic behavior detected"


def test_col_band_masks_stability():
    """Φ.3 stability: masks depend on structure, not values (input-only)."""
    # Two grids with same structure (column boundaries at same positions)
    g1 = [[1, 1, 3], [2, 2, 4]]
    g2 = [[5, 5, 7], [6, 6, 8]]

    masks1 = col_band_masks(g1)
    masks2 = col_band_masks(g2)

    # Same band structure (band 0: cols 0-1, band 1: col 2)
    assert len(masks1) == len(masks2)
    assert masks1 == masks2, "Masks differ despite same column structure"


def test_col_band_masks_ragged_input():
    """Ragged grid raises ValueError."""
    g_ragged = [[1, 2], [3]]  # Row 1 shorter

    with pytest.raises(ValueError, match="Ragged grid"):
        col_band_masks(g_ragged)


# ============================================================================
# Adversarial Cases (from P4-02 context pack)
# ============================================================================


def test_row_band_masks_wide_grid():
    """Wide grid (1 row, many columns) → one band."""
    g = [[i for i in range(100)]]
    masks = row_band_masks(g)

    assert len(masks) == 1
    assert all(masks[0][0][c] == 1 for c in range(100))
    assert_disjoint_and_cover(masks, (1, 100))


def test_col_band_masks_tall_grid():
    """Tall grid (many rows, 1 column) → one band."""
    g = [[i] for i in range(100)]
    masks = col_band_masks(g)

    assert len(masks) == 1
    assert all(masks[0][r][0] == 1 for r in range(100))
    assert_disjoint_and_cover(masks, (100, 1))


def test_row_band_masks_alternating_rows():
    """Alternating rows → many bands (pathological case)."""
    # [A, B, A, B, ...] pattern
    g = [[1, 2] if i % 2 == 0 else [3, 4] for i in range(10)]
    masks = row_band_masks(g)

    # Each row differs from next → 10 bands (one per row)
    assert len(masks) == 10
    assert_disjoint_and_cover(masks, (10, 2))

    # Each band covers exactly one row
    for i, mask in enumerate(masks):
        for r in range(10):
            expected = 1 if r == i else 0
            assert mask[r][0] == expected


def test_col_band_masks_alternating_cols():
    """Alternating columns → many bands (pathological case)."""
    # [[A, B, A, B, ...], [A, B, A, B, ...]]
    g = [[(i % 2) for i in range(10)], [(i % 2) for i in range(10)]]
    masks = col_band_masks(g)

    # Each column differs from next → 10 bands (one per column)
    assert len(masks) == 10
    assert_disjoint_and_cover(masks, (2, 10))

    # Each band covers exactly one column
    for j, mask in enumerate(masks):
        for c in range(10):
            expected = 1 if c == j else 0
            assert mask[0][c] == expected


def test_row_band_masks_complex_pattern():
    """Complex multi-band pattern with varying band sizes."""
    g = [
        [1, 1],  # Band 0 (size 1)
        [2, 2],  # Band 1 (size 3)
        [2, 2],
        [2, 2],
        [3, 3],  # Band 2 (size 2)
        [3, 3],
        [4, 4],  # Band 3 (size 1)
    ]
    masks = row_band_masks(g)

    assert len(masks) == 4

    # Band 0: row 0
    assert sum(masks[0][0]) == 2
    assert sum(sum(row) for row in masks[0]) == 2

    # Band 1: rows 1-3
    assert sum(sum(row) for row in masks[1]) == 6

    # Band 2: rows 4-5
    assert sum(sum(row) for row in masks[2]) == 4

    # Band 3: row 6
    assert sum(sum(row) for row in masks[3]) == 2

    assert_disjoint_and_cover(masks, (7, 2))
