"""
Tests for Patchkey Tables (P4-06).

This module tests patchkey_table from signature_builders.py,
which computes per-pixel canonical patch keys for radii r ∈ {2,3,4}.

Test coverage:
- Radius validation (r ∈ {2,3,4})
- Empty grids
- Too-small grids (all None)
- Exact-fit grids
- Interior key correctness
- Border handling (None at borders)
- OFA locality (palette permutations)
- Different radii
- Purity, determinism
- Error handling
"""

import copy
import pytest
from src.signature_builders import patchkey_table, patch_canonical_key


# ============================================================================
# Test Helpers
# ============================================================================


def count_valid_keys(table: list[list[object]]) -> int:
    """Count number of non-None entries in table."""
    if not table:
        return 0
    return sum(1 for row in table for entry in row if entry is not None)


def count_none(table: list[list[object]]) -> int:
    """Count number of None entries in table."""
    if not table:
        return 0
    return sum(1 for row in table for entry in row if entry is None)


# ============================================================================
# Radius Validation Tests
# ============================================================================


def test_patchkey_table_valid_radii():
    """Valid radii: 2, 3, 4."""
    g = [[i for i in range(10)] for _ in range(10)]  # 10×10 grid

    # All valid radii should work
    for r in [2, 3, 4]:
        table = patchkey_table(g, r)
        assert len(table) == 10
        assert len(table[0]) == 10


def test_patchkey_table_invalid_radius():
    """Invalid radii raise ValueError."""
    g = [[1, 2], [3, 4]]

    # Invalid radii
    for r in [0, 1, 5, 6, -1, 10]:
        with pytest.raises(ValueError, match="Radius must be in"):
            patchkey_table(g, r)


# ============================================================================
# Empty and Small Grid Tests
# ============================================================================


def test_patchkey_table_empty():
    """Empty grid returns empty list."""
    result = patchkey_table([], 2)
    assert result == []


def test_patchkey_table_too_small_2x2_r2():
    """2×2 grid with r=2 (needs 5×5) → all None."""
    g = [[1, 2], [3, 4]]

    table = patchkey_table(g, 2)

    # All entries should be None (grid too small)
    assert table == [[None, None], [None, None]]
    assert count_none(table) == 4
    assert count_valid_keys(table) == 0


def test_patchkey_table_too_small_4x4_r2():
    """4×4 grid with r=2 (needs 5×5) → all None."""
    g = [[i for i in range(4)] for _ in range(4)]

    table = patchkey_table(g, 2)

    # All entries should be None
    assert count_none(table) == 16
    assert count_valid_keys(table) == 0


def test_patchkey_table_too_small_6x6_r3():
    """6×6 grid with r=3 (needs 7×7) → all None."""
    g = [[i for i in range(6)] for _ in range(6)]

    table = patchkey_table(g, 3)

    # All entries should be None
    assert count_none(table) == 36
    assert count_valid_keys(table) == 0


# ============================================================================
# Exact Fit Tests
# ============================================================================


def test_patchkey_table_exact_fit_5x5_r2():
    """5×5 grid with r=2 → only center (2,2) valid."""
    g = [[i + j for j in range(5)] for i in range(5)]

    table = patchkey_table(g, 2)

    # Only center pixel (2,2) should have a valid key
    assert table[2][2] is not None
    assert isinstance(table[2][2], tuple)
    assert len(table[2][2]) == 3  # (R, C, values_tuple)

    # All other pixels should be None
    for i in range(5):
        for j in range(5):
            if (i, j) != (2, 2):
                assert table[i][j] is None

    # Total: 1 valid key, 24 None
    assert count_valid_keys(table) == 1
    assert count_none(table) == 24


def test_patchkey_table_exact_fit_7x7_r3():
    """7×7 grid with r=3 → only center (3,3) valid."""
    g = [[i * j for j in range(7)] for i in range(7)]

    table = patchkey_table(g, 3)

    # Only center pixel (3,3) should have a valid key
    assert table[3][3] is not None

    # All other pixels should be None
    for i in range(7):
        for j in range(7):
            if (i, j) != (3, 3):
                assert table[i][j] is None

    assert count_valid_keys(table) == 1
    assert count_none(table) == 48


def test_patchkey_table_exact_fit_9x9_r4():
    """9×9 grid with r=4 → only center (4,4) valid."""
    g = [[(i + j) % 5 for j in range(9)] for i in range(9)]

    table = patchkey_table(g, 4)

    # Only center pixel (4,4) should have a valid key
    assert table[4][4] is not None

    # All other pixels should be None
    assert count_valid_keys(table) == 1
    assert count_none(table) == 80


# ============================================================================
# Interior Region Tests
# ============================================================================


def test_patchkey_table_7x7_r2():
    """7×7 grid with r=2 → 3×3 interior."""
    g = [[i for i in range(7)] for _ in range(7)]

    table = patchkey_table(g, 2)

    # Valid centers: (2,2) through (4,4) — 3×3 region
    # Interior count: (7-2*2) × (7-2*2) = 3×3 = 9
    assert count_valid_keys(table) == 9

    # Verify specific interior pixels have keys
    for i in range(2, 5):  # rows 2,3,4
        for j in range(2, 5):  # cols 2,3,4
            assert table[i][j] is not None

    # Verify border pixels are None
    assert table[0][0] is None
    assert table[0][6] is None
    assert table[6][0] is None
    assert table[6][6] is None
    assert table[1][3] is None  # row 1 is border
    assert table[3][1] is None  # col 1 is border


def test_patchkey_table_10x10_r2():
    """10×10 grid with r=2 → 6×6 interior."""
    g = [[i * j % 10 for j in range(10)] for i in range(10)]

    table = patchkey_table(g, 2)

    # Interior count: (10-2*2) × (10-2*2) = 6×6 = 36
    assert count_valid_keys(table) == 36

    # Verify interior region
    for i in range(2, 8):  # rows 2-7
        for j in range(2, 8):  # cols 2-7
            assert table[i][j] is not None


def test_patchkey_table_10x10_r3():
    """10×10 grid with r=3 → 4×4 interior."""
    g = [[i + j for j in range(10)] for i in range(10)]

    table = patchkey_table(g, 3)

    # Interior count: (10-2*3) × (10-2*3) = 4×4 = 16
    assert count_valid_keys(table) == 16

    # Verify interior region
    for i in range(3, 7):  # rows 3-6
        for j in range(3, 7):  # cols 3-6
            assert table[i][j] is not None


# ============================================================================
# Border Handling Tests
# ============================================================================


def test_patchkey_table_border_none():
    """Border pixels are always None."""
    g = [[1] * 10 for _ in range(10)]

    for r in [2, 3, 4]:
        table = patchkey_table(g, r)

        # First and last r rows should be all None
        for i in range(r):
            assert all(entry is None for entry in table[i])
            assert all(entry is None for entry in table[-(i + 1)])

        # First and last r cols should be all None
        for i in range(10):
            for j in range(r):
                assert table[i][j] is None
                assert table[i][-(j + 1)] is None


def test_patchkey_table_rectangular_6x8_r2():
    """Rectangular grid (6×8) with r=2."""
    g = [[i * j for j in range(8)] for i in range(6)]

    table = patchkey_table(g, 2)

    # Interior: rows 2-3, cols 2-5 → 2×4 = 8 valid keys
    assert count_valid_keys(table) == 8

    # Verify interior
    for i in range(2, 4):
        for j in range(2, 6):
            assert table[i][j] is not None

    # Verify borders
    for i in [0, 1, 4, 5]:  # border rows
        for j in range(8):
            assert table[i][j] is None

    for j in [0, 1, 6, 7]:  # border cols
        for i in range(6):
            assert table[i][j] is None


# ============================================================================
# Key Correctness Tests
# ============================================================================


def test_patchkey_table_key_matches_patch_canonical_key():
    """Interior keys match patch_canonical_key of extracted windows."""
    g = [[i * j % 7 for j in range(8)] for i in range(8)]
    r = 2

    table = patchkey_table(g, r)

    # For each valid center, verify key matches
    for i in range(r, 8 - r):
        for j in range(r, 8 - r):
            # Extract window manually
            window = [row[j - r : j + r + 1] for row in g[i - r : i + r + 1]]

            # Compute expected key
            expected_key = patch_canonical_key(window)

            # Verify table entry matches
            assert table[i][j] == expected_key


def test_patchkey_table_uniform_grid():
    """Uniform grid (all same color) → all interior keys identical."""
    g = [[5] * 10 for _ in range(10)]
    r = 2

    table = patchkey_table(g, r)

    # All interior keys should be identical (all windows are same)
    interior_keys = [
        table[i][j] for i in range(r, 10 - r) for j in range(r, 10 - r)
    ]

    assert len(interior_keys) == 36  # 6×6 interior
    assert len(set(interior_keys)) == 1  # All identical


# ============================================================================
# OFA Locality Tests
# ============================================================================


def test_patchkey_table_ofa_locality_simple():
    """Palette permutation → same keys at interior."""
    # Two grids with different colors but same pattern
    g1 = [[i % 3 for j in range(7)] for i in range(7)]
    g2 = [[(i + 5) % 3 for j in range(7)] for i in range(7)]

    r = 2

    table1 = patchkey_table(g1, r)
    table2 = patchkey_table(g2, r)

    # Interior keys should be identical (OFA locality)
    # Note: This depends on whether pattern is preserved under palette permutation
    # For this specific example, pattern might not be identical
    # Let me use a clearer example

    # Better example: explicit palette swap
    g1 = [[1, 2, 1, 2, 1, 2, 1], [2, 1, 2, 1, 2, 1, 2]] * 4
    g1 = [row for row in g1[:7]]  # Make it 7 rows

    # Swap 1↔3, 2↔4
    g2 = [[3 if x == 1 else (4 if x == 2 else x) for x in row] for row in g1]

    table1 = patchkey_table(g1, r)
    table2 = patchkey_table(g2, r)

    # Compare interior keys
    for i in range(r, 7 - r):
        for j in range(r, 7 - r):
            # Both should have keys (not None)
            assert table1[i][j] is not None
            assert table2[i][j] is not None

            # Keys should be identical (OFA locality)
            assert table1[i][j] == table2[i][j]


# ============================================================================
# Purity and Determinism Tests
# ============================================================================


def test_patchkey_table_purity():
    """Input grid unchanged after patchkey_table call."""
    g = [[i * j for j in range(8)] for i in range(8)]
    g_copy = copy.deepcopy(g)

    patchkey_table(g, 2)

    assert g == g_copy, "Input grid modified (purity violation)"


def test_patchkey_table_determinism():
    """Repeated calls with same input produce identical output."""
    g = [[i + j for j in range(9)] for i in range(9)]

    table1 = patchkey_table(g, 3)
    table2 = patchkey_table(g, 3)

    assert table1 == table2, "Non-deterministic behavior detected"


def test_patchkey_table_determinism_all_radii():
    """Determinism for all valid radii."""
    g = [[i * j % 5 for j in range(10)] for i in range(10)]

    for r in [2, 3, 4]:
        table1 = patchkey_table(g, r)
        table2 = patchkey_table(g, r)

        assert table1 == table2, f"Non-deterministic for r={r}"


# ============================================================================
# Different Radii Tests
# ============================================================================


def test_patchkey_table_different_radii_same_grid():
    """Different radii produce different interior sizes."""
    g = [[i for i in range(10)] for _ in range(10)]

    table_r2 = patchkey_table(g, 2)
    table_r3 = patchkey_table(g, 3)
    table_r4 = patchkey_table(g, 4)

    # r=2: interior 6×6 = 36
    # r=3: interior 4×4 = 16
    # r=4: interior 2×2 = 4

    assert count_valid_keys(table_r2) == 36
    assert count_valid_keys(table_r3) == 16
    assert count_valid_keys(table_r4) == 4


def test_patchkey_table_r2_vs_r3():
    """r=2 has larger interior than r=3."""
    g = [[1] * 15 for _ in range(15)]

    table_r2 = patchkey_table(g, 2)
    table_r3 = patchkey_table(g, 3)

    # r=2: interior (15-4)×(15-4) = 11×11 = 121
    # r=3: interior (15-6)×(15-6) = 9×9 = 81

    assert count_valid_keys(table_r2) == 121
    assert count_valid_keys(table_r3) == 81


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_patchkey_table_ragged_input():
    """Ragged grid raises ValueError."""
    g_ragged = [[1, 2, 3], [4, 5]]

    with pytest.raises(ValueError, match="Ragged grid"):
        patchkey_table(g_ragged, 2)


def test_patchkey_table_invalid_radius_string():
    """Non-integer radius is invalid."""
    g = [[1, 2], [3, 4]]

    # This should fail even before the r in {2,3,4} check
    # because comparison with integers will work but validation will catch it
    with pytest.raises((ValueError, TypeError)):
        patchkey_table(g, "2")


# ============================================================================
# Shape Preservation Tests
# ============================================================================


def test_patchkey_table_shape_matches_input():
    """Output table has same shape as input grid."""
    for R in [5, 7, 10]:
        for C in [5, 8, 12]:
            g = [[1] * C for _ in range(R)]

            table = patchkey_table(g, 2)

            assert len(table) == R
            assert all(len(row) == C for row in table)


# ============================================================================
# Adversarial Cases
# ============================================================================


def test_patchkey_table_large_grid():
    """Large grid (50×50) with r=2."""
    g = [[i * j % 10 for j in range(50)] for i in range(50)]

    table = patchkey_table(g, 2)

    # Interior: (50-4)×(50-4) = 46×46 = 2116
    assert count_valid_keys(table) == 2116

    # Verify shape
    assert len(table) == 50
    assert all(len(row) == 50 for row in table)


def test_patchkey_table_single_row():
    """Single row grid (1×20)."""
    g = [[i for i in range(20)]]

    # r=2 needs 5 rows, but we only have 1
    table = patchkey_table(g, 2)

    # All entries should be None
    assert count_none(table) == 20
    assert count_valid_keys(table) == 0


def test_patchkey_table_single_column():
    """Single column grid (20×1)."""
    g = [[i] for i in range(20)]

    # r=2 needs 5 cols, but we only have 1
    table = patchkey_table(g, 2)

    # All entries should be None
    assert count_none(table) == 20
    assert count_valid_keys(table) == 0


def test_patchkey_table_minimal_valid_r4():
    """Minimal grid for r=4 (9×9)."""
    g = [[i + j for j in range(9)] for i in range(9)]

    table = patchkey_table(g, 4)

    # Only center (4,4) valid
    assert count_valid_keys(table) == 1
    assert table[4][4] is not None


# ============================================================================
# Integration Tests
# ============================================================================


def test_patchkey_table_integration_all_features():
    """Integration test combining multiple features."""
    g = [[i * j % 7 for j in range(12)] for i in range(12)]

    for r in [2, 3, 4]:
        table = patchkey_table(g, r)

        # Verify shape
        assert len(table) == 12
        assert all(len(row) == 12 for row in table)

        # Verify interior count
        expected_interior = (12 - 2 * r) * (12 - 2 * r)
        assert count_valid_keys(table) == expected_interior

        # Verify border None
        for i in range(r):
            assert all(entry is None for entry in table[i])

        # Verify interior keys match patch_canonical_key
        for i in range(r, 12 - r):
            for j in range(r, 12 - r):
                window = [
                    row[j - r : j + r + 1] for row in g[i - r : i + r + 1]
                ]
                expected_key = patch_canonical_key(window)
                assert table[i][j] == expected_key
