"""
Comprehensive tests for NPS (Non-Periodic Segmentation) boundary detection.

Tests cover:
- boundaries_by_any_change: row/col change boundary detection
- bands_from_boundaries: boundary → band conversion
- Edge cases (empty, single row/col, all equal, alternating)
- Validation (ragged, invalid axis, invalid boundaries)
- Purity (no mutation, deterministic)
- Integration (round-trip, partition correctness)
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.components import boundaries_by_any_change, bands_from_boundaries
from src.utils import copy_grid


# ============================================================================
# Test Fixtures (from context pack P1-06)
# ============================================================================

# Empty grid
g0 = []

# Single pixel
g1 = [[5]]

# Single row, multiple cols
g2 = [[1, 2, 3]]

# Single col, multiple rows
g3 = [[1], [2], [3]]

# All rows identical (no row boundaries)
g4 = [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
]

# Alternating rows
g5 = [
    [1, 1, 1],
    [2, 2, 2],
    [1, 1, 1],
    [2, 2, 2]
]

# Partial row change (boundary if ANY column differs)
g6 = [
    [1, 2, 3],
    [1, 2, 9],  # Only last column differs from row 0
    [1, 2, 9]   # Same as row 1
]

# Ragged (invalid)
g_ragged = [[1, 2], [3]]


# ============================================================================
# boundaries_by_any_change Tests - Validation
# ============================================================================

def test_boundaries_by_any_change_ragged_raises():
    """Ragged grid raises ValueError."""
    with pytest.raises(ValueError, match="rectangular"):
        boundaries_by_any_change(g_ragged, "row")


def test_boundaries_by_any_change_invalid_axis_raises():
    """Invalid axis raises ValueError."""
    with pytest.raises(ValueError, match='axis must be "row" or "col"'):
        boundaries_by_any_change(g2, "diagonal")

    with pytest.raises(ValueError, match='axis must be "row" or "col"'):
        boundaries_by_any_change(g2, "ROW")  # Case-sensitive


# ============================================================================
# boundaries_by_any_change Tests - Edge Cases
# ============================================================================

def test_boundaries_by_any_change_empty_grid():
    """Empty grid returns empty list for both axes."""
    assert boundaries_by_any_change(g0, "row") == []
    assert boundaries_by_any_change(g0, "col") == []


def test_boundaries_by_any_change_single_pixel():
    """Single pixel: no adjacent rows/cols to compare."""
    assert boundaries_by_any_change(g1, "row") == []
    assert boundaries_by_any_change(g1, "col") == []


def test_boundaries_by_any_change_single_row():
    """Single row: no row boundaries, but col boundaries if cols differ."""
    assert boundaries_by_any_change(g2, "row") == []
    # Cols differ: 1→2 at j=0, 2→3 at j=1
    assert boundaries_by_any_change(g2, "col") == [0, 1]


def test_boundaries_by_any_change_single_col():
    """Single col: no col boundaries, but row boundaries if rows differ."""
    # Rows differ: 1→2 at i=0, 2→3 at i=1
    assert boundaries_by_any_change(g3, "row") == [0, 1]
    assert boundaries_by_any_change(g3, "col") == []


def test_boundaries_by_any_change_all_rows_identical():
    """All rows identical: no row boundaries."""
    assert boundaries_by_any_change(g4, "row") == []
    # Cols still differ from each other
    assert boundaries_by_any_change(g4, "col") == [0, 1]


def test_boundaries_by_any_change_all_cols_identical():
    """All cols identical: no col boundaries."""
    g = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ]
    # Rows differ
    assert boundaries_by_any_change(g, "row") == [0, 1]
    # All cols identical
    assert boundaries_by_any_change(g, "col") == []


def test_boundaries_by_any_change_all_same_value():
    """Grid with all same value: no boundaries anywhere."""
    g = [[5, 5], [5, 5], [5, 5]]
    assert boundaries_by_any_change(g, "row") == []
    assert boundaries_by_any_change(g, "col") == []


# ============================================================================
# boundaries_by_any_change Tests - Alternating Patterns
# ============================================================================

def test_boundaries_by_any_change_alternating_rows():
    """Alternating rows: boundary at every index."""
    result = boundaries_by_any_change(g5, "row")
    # 4 rows → 3 possible boundaries [0, 1, 2]
    # All adjacent rows differ → all boundaries present
    assert result == [0, 1, 2]


def test_boundaries_by_any_change_alternating_cols():
    """Alternating cols: boundary at every index."""
    g = [
        [1, 2, 1, 2],
        [1, 2, 1, 2],
        [1, 2, 1, 2]
    ]
    result = boundaries_by_any_change(g, "col")
    # 4 cols → 3 possible boundaries [0, 1, 2]
    # All adjacent cols differ → all boundaries present
    assert result == [0, 1, 2]


# ============================================================================
# boundaries_by_any_change Tests - Partial Changes
# ============================================================================

def test_boundaries_by_any_change_partial_row_change():
    """Boundary if ANY column differs (not all)."""
    result = boundaries_by_any_change(g6, "row")
    # Rows 0→1 differ (last col: 3→9) → boundary at 0
    # Rows 1→2 identical → no boundary at 1
    assert result == [0]


def test_boundaries_by_any_change_partial_col_change():
    """Boundary if ANY row differs (not all)."""
    g = [
        [1, 1, 1],
        [1, 2, 1],  # Only middle col differs from col 0
        [1, 2, 1]
    ]
    result = boundaries_by_any_change(g, "col")
    # Cols 0→1 differ (row 1: 1→2) → boundary at 0
    # Cols 1→2 differ (row 1: 2→1) → boundary at 1
    assert result == [0, 1]


def test_boundaries_by_any_change_block_change():
    """Change localized to sub-rectangle."""
    g = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 9, 1, 1],  # Single cell differs
        [1, 1, 1, 1, 1]
    ]
    result = boundaries_by_any_change(g, "row")
    # Rows 1→2 differ (col 2: 1→9) → boundary at 1
    # Rows 2→3 differ (col 2: 9→1) → boundary at 2
    assert result == [1, 2]


# ============================================================================
# boundaries_by_any_change Tests - Result Properties
# ============================================================================

def test_boundaries_by_any_change_result_sorted():
    """Result is sorted ascending (guaranteed by construction)."""
    g = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ]
    result = boundaries_by_any_change(g, "row")
    # All rows differ → [0, 1, 2]
    assert result == sorted(result)


def test_boundaries_by_any_change_result_unique():
    """Result contains no duplicates."""
    g = [
        [1, 1],
        [2, 2],
        [3, 3]
    ]
    result = boundaries_by_any_change(g, "row")
    assert len(result) == len(set(result))


def test_boundaries_by_any_change_result_in_range():
    """All boundaries in [0..L-2] where L is axis length."""
    g = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ]
    result_row = boundaries_by_any_change(g, "row")
    rows = len(g)
    for b in result_row:
        assert 0 <= b <= rows - 2

    result_col = boundaries_by_any_change(g, "col")
    cols = len(g[0])
    for b in result_col:
        assert 0 <= b <= cols - 2


# ============================================================================
# boundaries_by_any_change Tests - Purity & Determinism
# ============================================================================

def test_boundaries_by_any_change_no_mutation():
    """Input grid is not mutated."""
    g = [[1, 2], [3, 4]]
    g_orig = copy_grid(g)

    _ = boundaries_by_any_change(g, "row")
    _ = boundaries_by_any_change(g, "col")

    assert g == g_orig


def test_boundaries_by_any_change_determinism():
    """Re-run yields identical results."""
    grids = [g0, g1, g2, g3, g4, g5, g6]

    for g in grids:
        result1_row = boundaries_by_any_change(g, "row")
        result2_row = boundaries_by_any_change(g, "row")
        assert result1_row == result2_row

        result1_col = boundaries_by_any_change(g, "col")
        result2_col = boundaries_by_any_change(g, "col")
        assert result1_col == result2_col


# ============================================================================
# bands_from_boundaries Tests - Validation
# ============================================================================

def test_bands_from_boundaries_negative_n_raises():
    """n < 0 raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        bands_from_boundaries(-1, [])


def test_bands_from_boundaries_unsorted_raises():
    """Unsorted boundaries raise ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        bands_from_boundaries(5, [2, 1])


def test_bands_from_boundaries_duplicates_raises():
    """Duplicate boundaries raise ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        bands_from_boundaries(5, [1, 1, 2])


def test_bands_from_boundaries_out_of_range_raises():
    """Boundaries outside [0..n-2] raise ValueError."""
    # Boundary < 0
    with pytest.raises(ValueError, match=r"must be in \[0\.\.3\]"):
        bands_from_boundaries(5, [-1, 1, 2])

    # Boundary > n-2
    with pytest.raises(ValueError, match=r"must be in \[0\.\.3\]"):
        bands_from_boundaries(5, [1, 2, 4])  # 4 > 5-2=3


# ============================================================================
# bands_from_boundaries Tests - Edge Cases
# ============================================================================

def test_bands_from_boundaries_n_zero():
    """n=0 returns empty list."""
    assert bands_from_boundaries(0, []) == []


def test_bands_from_boundaries_n_one():
    """n=1 returns single band [(0,0)]."""
    assert bands_from_boundaries(1, []) == [(0, 0)]


def test_bands_from_boundaries_n_three_no_boundaries():
    """n=3, no boundaries → single band [(0, 2)]."""
    assert bands_from_boundaries(3, []) == [(0, 2)]


def test_bands_from_boundaries_n_five_no_boundaries():
    """n=5, no boundaries → single band [(0, 4)]."""
    assert bands_from_boundaries(5, []) == [(0, 4)]


# ============================================================================
# bands_from_boundaries Tests - Single Boundary
# ============================================================================

def test_bands_from_boundaries_single_boundary():
    """Single boundary splits into two bands."""
    # n=5, boundary at 2 → [(0,2), (3,4)]
    assert bands_from_boundaries(5, [2]) == [(0, 2), (3, 4)]


def test_bands_from_boundaries_boundary_at_start():
    """Boundary at 0 creates single-element first band."""
    # n=5, boundary at 0 → [(0,0), (1,4)]
    assert bands_from_boundaries(5, [0]) == [(0, 0), (1, 4)]


def test_bands_from_boundaries_boundary_at_end():
    """Boundary at n-2 creates single-element last band."""
    # n=5, boundary at 3 → [(0,3), (4,4)]
    assert bands_from_boundaries(5, [3]) == [(0, 3), (4, 4)]


# ============================================================================
# bands_from_boundaries Tests - Multiple Boundaries
# ============================================================================

def test_bands_from_boundaries_multiple_boundaries():
    """Multiple boundaries create multiple bands."""
    # n=7, boundaries=[1, 4, 5]
    # Expected: [(0,1), (2,4), (5,5), (6,6)]
    result = bands_from_boundaries(7, [1, 4, 5])
    expected = [(0, 1), (2, 4), (5, 5), (6, 6)]
    assert result == expected


def test_bands_from_boundaries_consecutive_boundaries():
    """Consecutive boundaries create single-element bands."""
    # n=4, boundaries=[0, 1, 2]
    # Expected: [(0,0), (1,1), (2,2), (3,3)]
    result = bands_from_boundaries(4, [0, 1, 2])
    expected = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert result == expected


def test_bands_from_boundaries_sparse_boundaries():
    """Sparse boundaries create large bands."""
    # n=10, boundaries=[2, 7]
    # Expected: [(0,2), (3,7), (8,9)]
    result = bands_from_boundaries(10, [2, 7])
    expected = [(0, 2), (3, 7), (8, 9)]
    assert result == expected


# ============================================================================
# bands_from_boundaries Tests - Properties
# ============================================================================

def test_bands_from_boundaries_full_coverage():
    """Union of bands equals [0..n-1] (no gaps)."""
    def verify_coverage(n, boundaries):
        bands = bands_from_boundaries(n, boundaries)
        covered = set()
        for start, end in bands:
            for i in range(start, end + 1):
                covered.add(i)
        expected = set(range(n))
        assert covered == expected

    verify_coverage(5, [])
    verify_coverage(5, [2])
    verify_coverage(7, [1, 4, 5])
    verify_coverage(10, [0, 3, 5, 8])  # n=10 → valid boundaries in [0..8]


def test_bands_from_boundaries_disjoint():
    """Bands do not overlap."""
    def verify_disjoint(n, boundaries):
        bands = bands_from_boundaries(n, boundaries)
        for i in range(len(bands) - 1):
            _, end1 = bands[i]
            start2, _ = bands[i + 1]
            # Adjacent bands: end1 + 1 == start2
            assert end1 + 1 == start2

    verify_disjoint(5, [2])
    verify_disjoint(7, [1, 4, 5])
    verify_disjoint(10, [0, 3, 5, 8])  # n=10 → valid boundaries in [0..8]


def test_bands_from_boundaries_start_le_end():
    """All bands have start ≤ end."""
    def verify_start_le_end(n, boundaries):
        bands = bands_from_boundaries(n, boundaries)
        for start, end in bands:
            assert start <= end

    verify_start_le_end(1, [])
    verify_start_le_end(5, [2])
    verify_start_le_end(4, [0, 1, 2])  # Single-element bands


def test_bands_from_boundaries_contiguous():
    """Bands are contiguous and in order."""
    bands = bands_from_boundaries(7, [1, 4, 5])
    # Expected: [(0,1), (2,4), (5,5), (6,6)]

    # First band starts at 0
    assert bands[0][0] == 0

    # Last band ends at n-1
    assert bands[-1][1] == 6

    # Bands are contiguous
    for i in range(len(bands) - 1):
        assert bands[i][1] + 1 == bands[i + 1][0]


# ============================================================================
# bands_from_boundaries Tests - Determinism
# ============================================================================

def test_bands_from_boundaries_determinism():
    """Re-run yields identical results."""
    test_cases = [
        (0, []),
        (1, []),
        (5, []),
        (5, [2]),
        (7, [1, 4, 5]),
        (10, [0, 3, 5, 8])  # n=10 → valid boundaries in [0..8]
    ]

    for n, boundaries in test_cases:
        result1 = bands_from_boundaries(n, boundaries)
        result2 = bands_from_boundaries(n, boundaries)
        assert result1 == result2


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_round_trip_row():
    """boundaries_by_any_change → bands_from_boundaries for rows."""
    g = [
        [1, 1, 1],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [3, 3, 3]
    ]

    boundaries = boundaries_by_any_change(g, "row")
    # Rows 0-1 identical, rows 1→2 differ, rows 2→3 differ, rows 3-4 identical
    # Boundaries at [1, 2]
    assert boundaries == [1, 2]

    bands = bands_from_boundaries(len(g), boundaries)
    # Expected: [(0,1), (2,2), (3,4)]
    assert bands == [(0, 1), (2, 2), (3, 4)]

    # Verify bands correspond to homogeneous row groups
    # Band 0: rows 0-1 (all 1s)
    # Band 1: row 2 (all 2s)
    # Band 2: rows 3-4 (all 3s)


def test_integration_round_trip_col():
    """boundaries_by_any_change → bands_from_boundaries for cols."""
    g = [
        [1, 1, 2, 3, 3],
        [1, 1, 2, 3, 3],
        [1, 1, 2, 3, 3]
    ]

    boundaries = boundaries_by_any_change(g, "col")
    # Cols 0-1 identical, cols 1→2 differ, cols 2→3 differ, cols 3-4 identical
    # Boundaries at [1, 2]
    assert boundaries == [1, 2]

    bands = bands_from_boundaries(len(g[0]), boundaries)
    # Expected: [(0,1), (2,2), (3,4)]
    assert bands == [(0, 1), (2, 2), (3, 4)]


def test_integration_nps_example():
    """NPS example: grid with 3 distinct row-bands produces 2 boundaries."""
    g = [
        [5, 5, 5, 5],
        [5, 5, 5, 5],
        [7, 7, 7, 7],
        [9, 9, 9, 9],
        [9, 9, 9, 9],
        [9, 9, 9, 9]
    ]

    boundaries = boundaries_by_any_change(g, "row")
    # Rows 0-1: all 5s (no boundary between them)
    # Rows 1→2: 5s→7s (boundary at 1)
    # Rows 2→3: 7s→9s (boundary at 2)
    # Rows 3-5: all 9s (no boundaries)
    assert boundaries == [1, 2]

    bands = bands_from_boundaries(len(g), boundaries)
    # Expected: [(0,1), (2,2), (3,5)]
    assert bands == [(0, 1), (2, 2), (3, 5)]

    # 3 bands corresponding to 3 homogeneous regions


def test_integration_no_boundaries_single_band():
    """No boundaries → single band covering entire axis."""
    g = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

    boundaries = boundaries_by_any_change(g, "row")
    assert boundaries == []

    bands = bands_from_boundaries(len(g), boundaries)
    assert bands == [(0, 2)]  # Single band covering all 3 rows
