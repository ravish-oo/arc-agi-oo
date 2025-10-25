"""
Tests for Index Predicates (Φ Features) — Phase 4 Work Order P4-01.

Comprehensive test suite covering:
- Disjoint property: ∑Mi[r][c] == 1 for all (r,c)
- Cover property: ∑Mi == ones(shape)
- Correct residues: (r+c)%2, r%k, c%k
- Φ.3 stability: masks independent of grid values
- Purity: inputs unchanged
- Determinism: repeated calls identical
- Edge cases: empty, 1x1, 1xN, Nx1, ragged
"""

import pytest
from src.signature_builders import parity_mask, rowmod_mask, colmod_mask


# =============================
# Helper Functions
# =============================

def assert_disjoint_and_cover(masks: list[list[list[int]]], shape: tuple[int, int]) -> None:
    """
    Verify that masks form a disjoint partition covering all pixels exactly once.

    Args:
        masks: List of 0/1 masks
        shape: Expected (height, width)
    """
    h, w = shape

    for r in range(h):
        for c in range(w):
            # Count how many masks cover this pixel
            count = sum(masks[i][r][c] for i in range(len(masks)))
            assert count == 1, f"Pixel ({r},{c}) covered {count} times (expected 1)"


def assert_parity_tuple_disjoint_and_cover(m0: list[list[int]], m1: list[list[int]], shape: tuple[int, int]) -> None:
    """
    Verify parity masks (tuple format) are disjoint and cover.
    """
    assert_disjoint_and_cover([m0, m1], shape)


def grid_shape(g: list[list[int]]) -> tuple[int, int]:
    """Get (height, width) of grid."""
    if not g:
        return (0, 0)
    return (len(g), len(g[0]))


# =============================
# A. parity_mask Tests
# =============================

def test_parity_mask_empty():
    """Empty grid → empty masks."""
    m0, m1 = parity_mask([])
    assert m0 == []
    assert m1 == []


def test_parity_mask_1x1():
    """1x1 grid: (0,0) has even sum → M0."""
    g = [[5]]
    m0, m1 = parity_mask(g)

    assert m0 == [[1]]
    assert m1 == [[0]]
    assert_parity_tuple_disjoint_and_cover(m0, m1, grid_shape(g))


def test_parity_mask_2x2():
    """2x2 checkerboard pattern."""
    g = [[1, 2], [3, 4]]
    m0, m1 = parity_mask(g)

    # (0,0): 0+0=0 (even) → M0
    # (0,1): 0+1=1 (odd) → M1
    # (1,0): 1+0=1 (odd) → M1
    # (1,1): 1+1=2 (even) → M0
    assert m0 == [[1, 0], [0, 1]]
    assert m1 == [[0, 1], [1, 0]]
    assert_parity_tuple_disjoint_and_cover(m0, m1, grid_shape(g))


def test_parity_mask_3x3():
    """3x3 grid: verify checkerboard."""
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    m0, m1 = parity_mask(g)

    # M0: (0,0), (0,2), (1,1), (2,0), (2,2)
    # M1: (0,1), (1,0), (1,2), (2,1)
    expected_m0 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    expected_m1 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    assert m0 == expected_m0
    assert m1 == expected_m1
    assert_parity_tuple_disjoint_and_cover(m0, m1, grid_shape(g))


def test_parity_mask_wide_1xN():
    """Wide 1xN grid: alternating pattern."""
    g = [[1, 2, 3, 4, 5]]
    m0, m1 = parity_mask(g)

    # (0,0): even, (0,1): odd, (0,2): even, (0,3): odd, (0,4): even
    assert m0 == [[1, 0, 1, 0, 1]]
    assert m1 == [[0, 1, 0, 1, 0]]
    assert_parity_tuple_disjoint_and_cover(m0, m1, grid_shape(g))


def test_parity_mask_tall_Nx1():
    """Tall Nx1 grid: alternating pattern."""
    g = [[1], [2], [3], [4], [5]]
    m0, m1 = parity_mask(g)

    # (0,0): even, (1,0): odd, (2,0): even, (3,0): odd, (4,0): even
    assert m0 == [[1], [0], [1], [0], [1]]
    assert m1 == [[0], [1], [0], [1], [0]]
    assert_parity_tuple_disjoint_and_cover(m0, m1, grid_shape(g))


def test_parity_mask_ragged_raises():
    """Ragged input → ValueError."""
    ragged = [[1, 2, 3], [4, 5]]
    with pytest.raises(ValueError, match="Ragged grid"):
        parity_mask(ragged)


def test_parity_mask_phi3_stability():
    """Φ.3 stability: masks depend only on shape, not grid values."""
    g1 = [[1, 2], [3, 4]]
    g2 = [[9, 0], [7, 5]]

    m0_1, m1_1 = parity_mask(g1)
    m0_2, m1_2 = parity_mask(g2)

    # Masks should be identical (same shape, different values)
    assert m0_1 == m0_2
    assert m1_1 == m1_2


def test_parity_mask_purity():
    """Purity: input grid unchanged."""
    g = [[1, 2], [3, 4]]
    g_copy = [row[:] for row in g]

    parity_mask(g)

    assert g == g_copy


def test_parity_mask_determinism():
    """Determinism: repeated calls identical."""
    g = [[1, 2, 3], [4, 5, 6]]

    m0_1, m1_1 = parity_mask(g)
    m0_2, m1_2 = parity_mask(g)

    assert m0_1 == m0_2
    assert m1_1 == m1_2


# =============================
# B. rowmod_mask Tests
# =============================

def test_rowmod_mask_empty_k2():
    """Empty grid → k empty masks."""
    masks = rowmod_mask([], 2)
    assert masks == [[], []]


def test_rowmod_mask_empty_k3():
    """Empty grid → k empty masks."""
    masks = rowmod_mask([], 3)
    assert masks == [[], [], []]


def test_rowmod_mask_1x1_k2():
    """1x1 grid: row 0 → M0."""
    g = [[5]]
    masks = rowmod_mask(g, 2)

    assert masks == [[[1]], [[0]]]
    assert_disjoint_and_cover(masks, grid_shape(g))


def test_rowmod_mask_3x2_k2():
    """3x2 grid with k=2: alternating rows."""
    g = [[1, 2], [3, 4], [5, 6]]
    masks = rowmod_mask(g, 2)

    # M0: rows 0, 2
    # M1: row 1
    expected_m0 = [[1, 1], [0, 0], [1, 1]]
    expected_m1 = [[0, 0], [1, 1], [0, 0]]

    assert len(masks) == 2
    assert masks[0] == expected_m0
    assert masks[1] == expected_m1
    assert_disjoint_and_cover(masks, grid_shape(g))


def test_rowmod_mask_6x2_k3():
    """6x2 grid with k=3: rows cycle every 3."""
    g = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 1]]
    masks = rowmod_mask(g, 3)

    # M0: rows 0, 3
    # M1: rows 1, 4
    # M2: rows 2, 5
    expected_m0 = [[1, 1], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]
    expected_m1 = [[0, 0], [1, 1], [0, 0], [0, 0], [1, 1], [0, 0]]
    expected_m2 = [[0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [1, 1]]

    assert len(masks) == 3
    assert masks[0] == expected_m0
    assert masks[1] == expected_m1
    assert masks[2] == expected_m2
    assert_disjoint_and_cover(masks, grid_shape(g))


def test_rowmod_mask_invalid_k():
    """Invalid k (not in {2,3}) → ValueError."""
    g = [[1, 2], [3, 4]]

    with pytest.raises(ValueError, match="k must be in"):
        rowmod_mask(g, 1)

    with pytest.raises(ValueError, match="k must be in"):
        rowmod_mask(g, 4)


def test_rowmod_mask_ragged_raises():
    """Ragged input → ValueError."""
    ragged = [[1, 2, 3], [4, 5]]
    with pytest.raises(ValueError, match="Ragged grid"):
        rowmod_mask(ragged, 2)


def test_rowmod_mask_phi3_stability():
    """Φ.3 stability: masks depend only on shape, not grid values."""
    g1 = [[1, 2], [3, 4], [5, 6]]
    g2 = [[9, 0], [7, 8], [6, 5]]

    masks1 = rowmod_mask(g1, 2)
    masks2 = rowmod_mask(g2, 2)

    assert masks1 == masks2


def test_rowmod_mask_purity():
    """Purity: input grid unchanged."""
    g = [[1, 2], [3, 4]]
    g_copy = [row[:] for row in g]

    rowmod_mask(g, 2)

    assert g == g_copy


def test_rowmod_mask_determinism():
    """Determinism: repeated calls identical."""
    g = [[1, 2], [3, 4], [5, 6]]

    masks1 = rowmod_mask(g, 3)
    masks2 = rowmod_mask(g, 3)

    assert masks1 == masks2


# =============================
# C. colmod_mask Tests
# =============================

def test_colmod_mask_empty_k2():
    """Empty grid → k empty masks."""
    masks = colmod_mask([], 2)
    assert masks == [[], []]


def test_colmod_mask_empty_k3():
    """Empty grid → k empty masks."""
    masks = colmod_mask([], 3)
    assert masks == [[], [], []]


def test_colmod_mask_1x1_k2():
    """1x1 grid: col 0 → M0."""
    g = [[5]]
    masks = colmod_mask(g, 2)

    assert masks == [[[1]], [[0]]]
    assert_disjoint_and_cover(masks, grid_shape(g))


def test_colmod_mask_2x4_k2():
    """2x4 grid with k=2: alternating columns."""
    g = [[1, 2, 3, 4], [5, 6, 7, 8]]
    masks = colmod_mask(g, 2)

    # M0: cols 0, 2
    # M1: cols 1, 3
    expected_m0 = [[1, 0, 1, 0], [1, 0, 1, 0]]
    expected_m1 = [[0, 1, 0, 1], [0, 1, 0, 1]]

    assert len(masks) == 2
    assert masks[0] == expected_m0
    assert masks[1] == expected_m1
    assert_disjoint_and_cover(masks, grid_shape(g))


def test_colmod_mask_2x6_k3():
    """2x6 grid with k=3: columns cycle every 3."""
    g = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]]
    masks = colmod_mask(g, 3)

    # M0: cols 0, 3
    # M1: cols 1, 4
    # M2: cols 2, 5
    expected_m0 = [[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]
    expected_m1 = [[0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]]
    expected_m2 = [[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]]

    assert len(masks) == 3
    assert masks[0] == expected_m0
    assert masks[1] == expected_m1
    assert masks[2] == expected_m2
    assert_disjoint_and_cover(masks, grid_shape(g))


def test_colmod_mask_invalid_k():
    """Invalid k (not in {2,3}) → ValueError."""
    g = [[1, 2], [3, 4]]

    with pytest.raises(ValueError, match="k must be in"):
        colmod_mask(g, 1)

    with pytest.raises(ValueError, match="k must be in"):
        colmod_mask(g, 5)


def test_colmod_mask_ragged_raises():
    """Ragged input → ValueError."""
    ragged = [[1, 2, 3], [4, 5]]
    with pytest.raises(ValueError, match="Ragged grid"):
        colmod_mask(ragged, 2)


def test_colmod_mask_phi3_stability():
    """Φ.3 stability: masks depend only on shape, not grid values."""
    g1 = [[1, 2, 3], [4, 5, 6]]
    g2 = [[9, 0, 7], [8, 6, 5]]

    masks1 = colmod_mask(g1, 3)
    masks2 = colmod_mask(g2, 3)

    assert masks1 == masks2


def test_colmod_mask_purity():
    """Purity: input grid unchanged."""
    g = [[1, 2, 3], [4, 5, 6]]
    g_copy = [row[:] for row in g]

    colmod_mask(g, 3)

    assert g == g_copy


def test_colmod_mask_determinism():
    """Determinism: repeated calls identical."""
    g = [[1, 2, 3], [4, 5, 6]]

    masks1 = colmod_mask(g, 3)
    masks2 = colmod_mask(g, 3)

    assert masks1 == masks2


# =============================
# D. Cross-Function Integration Tests
# =============================

def test_cross_parity_vs_rowmod():
    """Verify parity masks are consistent with rowmod k=2."""
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    m0_parity, m1_parity = parity_mask(g)
    masks_row2 = rowmod_mask(g, 2)

    # For parity and rowmod k=2, the patterns should be related
    # (not identical, but internally consistent)
    # Just verify both are disjoint and cover
    assert_parity_tuple_disjoint_and_cover(m0_parity, m1_parity, grid_shape(g))
    assert_disjoint_and_cover(masks_row2, grid_shape(g))


def test_cross_rowmod_vs_colmod():
    """Verify rowmod and colmod are orthogonal partitions."""
    g = [[1, 2, 3], [4, 5, 6]]

    masks_row = rowmod_mask(g, 2)
    masks_col = colmod_mask(g, 3)

    # Both should be disjoint+cover independently
    assert_disjoint_and_cover(masks_row, grid_shape(g))
    assert_disjoint_and_cover(masks_col, grid_shape(g))

    # Each pixel should belong to exactly one row mask and one col mask
    for r in range(2):
        for c in range(3):
            row_count = sum(masks_row[i][r][c] for i in range(2))
            col_count = sum(masks_col[i][r][c] for i in range(3))
            assert row_count == 1
            assert col_count == 1


# =============================
# E. Adversarial Mini Fixtures
# =============================

def test_adversarial_wide_1x7():
    """Wide 1x7: verify residues on columns for k=2,3."""
    g = [[0, 1, 2, 3, 4, 5, 6]]

    # colmod k=2: alternating pattern
    masks_col2 = colmod_mask(g, 2)
    assert masks_col2[0] == [[1, 0, 1, 0, 1, 0, 1]]
    assert masks_col2[1] == [[0, 1, 0, 1, 0, 1, 0]]

    # colmod k=3: cycle every 3
    masks_col3 = colmod_mask(g, 3)
    assert masks_col3[0] == [[1, 0, 0, 1, 0, 0, 1]]
    assert masks_col3[1] == [[0, 1, 0, 0, 1, 0, 0]]
    assert masks_col3[2] == [[0, 0, 1, 0, 0, 1, 0]]


def test_adversarial_tall_7x1():
    """Tall 7x1: verify residues on rows for k=2,3."""
    g = [[0], [1], [2], [3], [4], [5], [6]]

    # rowmod k=2: alternating pattern
    masks_row2 = rowmod_mask(g, 2)
    assert masks_row2[0] == [[1], [0], [1], [0], [1], [0], [1]]
    assert masks_row2[1] == [[0], [1], [0], [1], [0], [1], [0]]

    # rowmod k=3: cycle every 3
    masks_row3 = rowmod_mask(g, 3)
    assert masks_row3[0] == [[1], [0], [0], [1], [0], [0], [1]]
    assert masks_row3[1] == [[0], [1], [0], [0], [1], [0], [0]]
    assert masks_row3[2] == [[0], [0], [1], [0], [0], [1], [0]]


def test_adversarial_rect_3x4():
    """Rectangular 3x4: cross-check parity vs row/col residues."""
    g = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]]

    # All three should independently form disjoint+cover partitions
    m0_parity, m1_parity = parity_mask(g)
    masks_row = rowmod_mask(g, 3)
    masks_col = colmod_mask(g, 2)

    assert_parity_tuple_disjoint_and_cover(m0_parity, m1_parity, grid_shape(g))
    assert_disjoint_and_cover(masks_row, grid_shape(g))
    assert_disjoint_and_cover(masks_col, grid_shape(g))


# =============================
# F. Determinism Re-run Test
# =============================

def test_full_determinism_rerun():
    """Run entire test suite twice; verify byte-for-byte identical."""
    g = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]]

    # First run
    p0_1, p1_1 = parity_mask(g)
    r2_1 = rowmod_mask(g, 2)
    r3_1 = rowmod_mask(g, 3)
    c2_1 = colmod_mask(g, 2)
    c3_1 = colmod_mask(g, 3)

    # Second run
    p0_2, p1_2 = parity_mask(g)
    r2_2 = rowmod_mask(g, 2)
    r3_2 = rowmod_mask(g, 3)
    c2_2 = colmod_mask(g, 2)
    c3_2 = colmod_mask(g, 3)

    # Verify identical
    assert p0_1 == p0_2 and p1_1 == p1_2
    assert r2_1 == r2_2
    assert r3_1 == r3_2
    assert c2_1 == c2_2
    assert c3_1 == c3_2
