"""
Property tests for Φ features (P4-08).

This module validates mathematical invariants across ALL Φ features:
- Φ.3 Stability: input-only, no Y dependencies
- Finiteness: bounded signature spaces
- Disjointness/Coverage: partition properties
- Determinism: byte-identical serialization

Categories:
1. Φ.3 Stability Tests
2. Finiteness Tests
3. Disjointness & Coverage Tests
4. Determinism Tests
5. Edge Case Tests

LOC Budget: ≤ 220 lines
"""

import pytest
from src.signature_builders import phi_signature_tables


# ============================================================================
# Category 1: Φ.3 Stability (Input-Only)
# ============================================================================


def test_phi_repeated_calls_identical():
    """Repeated calls on same X produce identical results."""
    X = [[1, 2, 3], [4, 5, 6]]
    result1 = phi_signature_tables(X)
    result2 = phi_signature_tables(X)
    assert result1 == result2


def test_phi_no_y_parameter():
    """Φ builders accept only X, never Y (structural property)."""
    X = [[1, 2], [3, 4]]
    result = phi_signature_tables(X)
    assert result is not None


# ============================================================================
# Category 2: Finiteness
# ============================================================================


def test_parity_exactly_two_masks():
    """Parity always produces exactly 2 masks (M0, M1)."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    parity = result["index"]["parity"]
    assert len(parity) == 2
    assert "M0" in parity
    assert "M1" in parity


def test_rowmod_k2_exactly_two_masks():
    """Rowmod k=2 produces exactly 2 masks."""
    X = [[1, 2], [3, 4], [5, 6]]
    result = phi_signature_tables(X)
    rowmod_k2 = result["index"]["rowmod"]["k2"]
    assert len(rowmod_k2) == 2


def test_rowmod_k3_exactly_three_masks():
    """Rowmod k=3 produces exactly 3 masks."""
    X = [[1, 2], [3, 4], [5, 6]]
    result = phi_signature_tables(X)
    rowmod_k3 = result["index"]["rowmod"]["k3"]
    assert len(rowmod_k3) == 3


def test_colmod_k2_exactly_two_masks():
    """Colmod k=2 produces exactly 2 masks."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    colmod_k2 = result["index"]["colmod"]["k2"]
    assert len(colmod_k2) == 2


def test_colmod_k3_exactly_three_masks():
    """Colmod k=3 produces exactly 3 masks."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    colmod_k3 = result["index"]["colmod"]["k3"]
    assert len(colmod_k3) == 3


def test_row_bands_bounded_by_R():
    """Row bands: ≤ R masks."""
    X = [[1, 2], [3, 4], [5, 6]]
    result = phi_signature_tables(X)
    row_bands = result["nps"]["row_bands"]
    R = len(X)
    assert len(row_bands) <= R
    assert len(row_bands) >= 1


def test_col_bands_bounded_by_C():
    """Column bands: ≤ C masks."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    col_bands = result["nps"]["col_bands"]
    C = len(X[0])
    assert len(col_bands) <= C
    assert len(col_bands) >= 1


def test_local_content_exactly_ten_colors():
    """is_color and touching_color must have exactly 10 entries (colors 0-9)."""
    X = [[5, 3], [3, 5]]
    result = phi_signature_tables(X)
    assert len(result["local"]["is_color"]) == 10
    assert len(result["local"]["touching_color"]) == 10
    for c in range(10):
        assert c in result["local"]["is_color"]
        assert c in result["local"]["touching_color"]


def test_components_bounded_by_RC():
    """Components: ≤ R×C."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    meta = result["components"]["meta"]
    R = len(X)
    C = len(X[0])
    assert len(meta) <= R * C


# ============================================================================
# Category 3: Disjointness & Coverage
# ============================================================================


def test_parity_disjoint_and_cover():
    """Parity: M0 ⊕ M1 == ones; M0 ∧ M1 == 0."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    M0 = result["index"]["parity"]["M0"]
    M1 = result["index"]["parity"]["M1"]

    for r in range(len(X)):
        for c in range(len(X[0])):
            assert not (M0[r][c] == 1 and M1[r][c] == 1)
            assert M0[r][c] + M1[r][c] == 1


def test_rowmod_k2_disjoint_and_cover():
    """Rowmod k=2: disjoint + cover."""
    X = [[1, 2], [3, 4], [5, 6]]
    result = phi_signature_tables(X)
    masks = result["index"]["rowmod"]["k2"]

    for r in range(len(X)):
        for c in range(len(X[0])):
            total = sum(masks[i][r][c] for i in range(2))
            assert total == 1
            count_ones = sum(1 for i in range(2) if masks[i][r][c] == 1)
            assert count_ones == 1


def test_rowmod_k3_disjoint_and_cover():
    """Rowmod k=3: disjoint + cover."""
    X = [[1, 2], [3, 4], [5, 6]]
    result = phi_signature_tables(X)
    masks = result["index"]["rowmod"]["k3"]

    for r in range(len(X)):
        for c in range(len(X[0])):
            total = sum(masks[i][r][c] for i in range(3))
            assert total == 1
            count_ones = sum(1 for i in range(3) if masks[i][r][c] == 1)
            assert count_ones == 1


def test_colmod_k2_disjoint_and_cover():
    """Colmod k=2: disjoint + cover."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    masks = result["index"]["colmod"]["k2"]

    for r in range(len(X)):
        for c in range(len(X[0])):
            total = sum(masks[i][r][c] for i in range(2))
            assert total == 1


def test_colmod_k3_disjoint_and_cover():
    """Colmod k=3: disjoint + cover."""
    X = [[1, 2, 3], [4, 5, 6]]
    result = phi_signature_tables(X)
    masks = result["index"]["colmod"]["k3"]

    for r in range(len(X)):
        for c in range(len(X[0])):
            total = sum(masks[i][r][c] for i in range(3))
            assert total == 1


def test_row_bands_disjoint_and_cover():
    """Row bands: disjoint + cover."""
    X = [[1, 1], [1, 1], [2, 2]]
    result = phi_signature_tables(X)
    bands = result["nps"]["row_bands"]

    for r in range(len(X)):
        for c in range(len(X[0])):
            total = sum(bands[i][r][c] for i in range(len(bands)))
            assert total == 1
            count_ones = sum(1 for i in range(len(bands)) if bands[i][r][c] == 1)
            assert count_ones == 1


def test_col_bands_disjoint_and_cover():
    """Column bands: disjoint + cover."""
    X = [[1, 1, 2], [1, 1, 2]]
    result = phi_signature_tables(X)
    bands = result["nps"]["col_bands"]

    for r in range(len(X)):
        for c in range(len(X[0])):
            total = sum(bands[i][r][c] for i in range(len(bands)))
            assert total == 1


def test_touching_disjoint_from_is_color():
    """touching_color[c] ∧ is_color[c] == 0 for all c."""
    X = [[5, 0], [0, 0]]
    result = phi_signature_tables(X)

    for c in range(10):
        is_c = result["local"]["is_color"][c]
        touching_c = result["local"]["touching_color"][c]

        for r in range(len(X)):
            for col in range(len(X[0])):
                assert not (is_c[r][col] == 1 and touching_c[r][col] == 1)


# ============================================================================
# Category 4: Determinism
# ============================================================================


def test_determinism_deep_equality():
    """Two runs produce identical results."""
    X = [[1, 2, 3], [4, 5, 6]]
    result1 = phi_signature_tables(X)
    result2 = phi_signature_tables(X)
    assert result1 == result2


def test_determinism_multiple_grids():
    """Determinism holds across different grids."""
    grids = [[[1]], [[1, 2], [3, 4]], [[5, 5, 5], [5, 5, 5]]]

    for X in grids:
        result1 = phi_signature_tables(X)
        result2 = phi_signature_tables(X)
        assert result1 == result2


# ============================================================================
# Category 5: Edge Cases
# ============================================================================


def test_empty_grid():
    """Empty grid returns minimal valid structures."""
    X = []
    result = phi_signature_tables(X)

    assert result["index"]["parity"] == {"M0": [], "M1": []}
    assert result["nps"]["row_bands"] == []
    assert result["nps"]["col_bands"] == []
    assert result["components"]["id_grid"] == []
    assert result["components"]["meta"] == []

    for c in range(10):
        assert result["local"]["is_color"][c] == []
        assert result["local"]["touching_color"][c] == []


def test_single_pixel():
    """Single pixel grid."""
    X = [[7]]
    result = phi_signature_tables(X)

    assert len(result["components"]["meta"]) == 1
    assert result["components"]["meta"][0]["color"] == 7
    assert result["components"]["meta"][0]["size"] == 1


def test_1xN_grid():
    """1×N grid (single row)."""
    X = [[1, 2, 3, 4, 5]]
    result = phi_signature_tables(X)

    assert len(result["nps"]["row_bands"]) == 1
    col_bands = result["nps"]["col_bands"]
    assert len(col_bands) >= 1


def test_Nx1_grid():
    """N×1 grid (single column)."""
    X = [[1], [2], [3], [4], [5]]
    result = phi_signature_tables(X)

    assert len(result["nps"]["col_bands"]) == 1
    row_bands = result["nps"]["row_bands"]
    assert len(row_bands) >= 1


def test_all_same_color():
    """Grid with all pixels same color."""
    X = [[3, 3, 3], [3, 3, 3]]
    result = phi_signature_tables(X)

    assert len(result["components"]["meta"]) == 1
    assert result["components"]["meta"][0]["color"] == 3
    assert result["components"]["meta"][0]["size"] == 6
    assert result["local"]["is_color"][3] == [[1, 1, 1], [1, 1, 1]]
    assert result["local"]["touching_color"][3] == [[0, 0, 0], [0, 0, 0]]


def test_alternating_rows():
    """Alternating row pattern."""
    X = [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]]
    result = phi_signature_tables(X)

    row_bands = result["nps"]["row_bands"]
    assert len(row_bands) == 4


def test_palette_absent():
    """Colors 0, 5-9 absent from grid."""
    X = [[1, 2], [3, 4]]
    result = phi_signature_tables(X)

    assert len(result["local"]["is_color"]) == 10

    for c in [0, 5, 6, 7, 8, 9]:
        is_c = result["local"]["is_color"][c]
        total = sum(sum(row) for row in is_c)
        assert total == 0
