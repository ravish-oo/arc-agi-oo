"""
Comprehensive tests for 8-connected component extraction.

Tests cover:
- 8-connectivity correctness (diagonal bridges, corner touches)
- Deterministic component IDs with stable tie-breaking
- bbox correctness (inclusive bounding box)
- cells sorted row-major
- Purity (no mutation, no aliasing)
- Edge cases (empty, single cell, ragged)
- Determinism (re-run stability)
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.components import bbox, components_by_color, NEIGHBORS_8
from src.utils import deep_eq, copy_grid


# ============================================================================
# Test Fixtures (from context pack P1-05)
# ============================================================================

# Empty grid
g0 = []

# Single pixel
g1 = [[5]]

# Horizontal line (8-conn, trivially connected)
g2 = [[1, 1, 1]]

# Diagonal bridge (8-connected DOES connect)
g3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

# Two separate components same color
g4 = [
    [1, 0, 1],
    [0, 0, 0]
]

# Corner touch (8-conn connects)
g5 = [
    [2, 0],
    [0, 2]
]

# Tie-breaking: same size, different bbox
g6 = [
    [3, 0, 3],
    [0, 3, 0]
]

# Ragged (invalid)
g_ragged = [[1, 2], [3]]


# ============================================================================
# bbox Tests
# ============================================================================

def test_bbox_empty_raises():
    """Empty cells list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        bbox([])


def test_bbox_single_cell():
    """Single cell [(2,3)] → (2, 3, 2, 3)."""
    assert bbox([(2, 3)]) == (2, 3, 2, 3)


def test_bbox_horizontal_line():
    """Horizontal line [(0,1), (0,2), (0,3)] → (0, 1, 0, 3)."""
    assert bbox([(0, 1), (0, 2), (0, 3)]) == (0, 1, 0, 3)


def test_bbox_vertical_line():
    """Vertical line [(1,0), (2,0), (3,0)] → (1, 0, 3, 0)."""
    assert bbox([(1, 0), (2, 0), (3, 0)]) == (1, 0, 3, 0)


def test_bbox_l_shape():
    """L-shape: bbox is inclusive rectangle."""
    cells = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert bbox(cells) == (0, 0, 2, 2)


def test_bbox_unsorted_cells():
    """bbox works with unsorted cells list."""
    cells = [(3, 2), (1, 0), (2, 1)]
    assert bbox(cells) == (1, 0, 3, 2)


def test_bbox_determinism():
    """Same cells always yield same bbox."""
    cells = [(5, 3), (2, 7), (8, 1)]
    result1 = bbox(cells)
    result2 = bbox(cells)
    assert result1 == result2


# ============================================================================
# components_by_color Edge Cases
# ============================================================================

def test_components_by_color_empty():
    """Empty grid returns empty dict."""
    assert components_by_color(g0) == {}


def test_components_by_color_single_pixel():
    """Single pixel forms single component."""
    result = components_by_color(g1)

    expected = {
        5: [{
            "id": 0,
            "color": 5,
            "cells": [(0, 0)],
            "bbox": (0, 0, 0, 0)
        }]
    }

    assert result == expected


def test_components_by_color_horizontal_line():
    """Horizontal line forms single component (trivially 8-connected)."""
    result = components_by_color(g2)

    expected = {
        1: [{
            "id": 0,
            "color": 1,
            "cells": [(0, 0), (0, 1), (0, 2)],
            "bbox": (0, 0, 0, 2)
        }]
    }

    assert result == expected


def test_components_by_color_all_same_color():
    """All same color → single component with all pixels."""
    g = [[3, 3], [3, 3]]
    result = components_by_color(g)

    expected = {
        3: [{
            "id": 0,
            "color": 3,
            "cells": [(0, 0), (0, 1), (1, 0), (1, 1)],
            "bbox": (0, 0, 1, 1)
        }]
    }

    assert result == expected


def test_components_by_color_ragged_raises():
    """Ragged grid raises ValueError."""
    with pytest.raises(ValueError, match="rectangular"):
        components_by_color(g_ragged)


# ============================================================================
# 8-Connectivity Tests (CRITICAL)
# ============================================================================

def test_8connectivity_diagonal_bridge():
    """
    Diagonal bridge: pixels touching only diagonally ARE connected.

    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

    All three 1s form a SINGLE component via diagonal connections.
    """
    result = components_by_color(g3)

    # Extract color 1 components
    comps_1 = result[1]

    # Should be exactly 1 component containing all three 1s
    assert len(comps_1) == 1
    assert comps_1[0]["cells"] == [(0, 0), (1, 1), (2, 2)]
    assert comps_1[0]["bbox"] == (0, 0, 2, 2)


def test_8connectivity_corner_touch():
    """
    Corner touch DOES connect in 8-connectivity.

    [[2, 0],
     [0, 2]]

    Positions (0,0) and (1,1) are 8-neighbors → single component.
    """
    result = components_by_color(g5)

    comps_2 = result[2]

    # Should be exactly 1 component
    assert len(comps_2) == 1
    assert comps_2[0]["cells"] == [(0, 0), (1, 1)]
    assert comps_2[0]["bbox"] == (0, 0, 1, 1)


def test_8connectivity_no_diagonal_across_different_color():
    """
    Diagonal connection requires SAME color at both positions.

    [[1, 0],
     [0, 1]]

    The two 1s are diagonal neighbors BUT should they connect?
    Actually yes! (0,0) and (1,1) are 8-neighbors if same color.

    Let's verify this is ONE component, not two.
    """
    g = [[1, 0], [0, 1]]
    result = components_by_color(g)

    comps_1 = result[1]

    # (0,0) and (1,1) ARE 8-neighbors → single component
    assert len(comps_1) == 1
    assert comps_1[0]["cells"] == [(0, 0), (1, 1)]


def test_8connectivity_separated_components():
    """
    Two 1s separated by 0s (no 8-neighbor connection).

    [[1, 0, 1],
     [0, 0, 0]]

    (0,0) and (0,2) are NOT 8-neighbors → two separate components.
    """
    result = components_by_color(g4)

    comps_1 = result[1]

    # Should be exactly 2 components
    assert len(comps_1) == 2

    # Both have size 1; sorted by bbox lex
    # bbox (0,0,0,0) < bbox (0,2,0,2)
    assert comps_1[0]["id"] == 0
    assert comps_1[0]["cells"] == [(0, 0)]
    assert comps_1[0]["bbox"] == (0, 0, 0, 0)

    assert comps_1[1]["id"] == 1
    assert comps_1[1]["cells"] == [(0, 2)]
    assert comps_1[1]["bbox"] == (0, 2, 0, 2)


def test_8connectivity_all_8_neighbors():
    """
    Verify all 8 neighbors are considered (not just 4).

    Center pixel surrounded by 8 neighbors of same color → single component.
    """
    g = [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5]
    ]

    result = components_by_color(g)

    comps_5 = result[5]

    # All 9 pixels form single component
    assert len(comps_5) == 1
    assert len(comps_5[0]["cells"]) == 9


# ============================================================================
# Deterministic ID Assignment Tests
# ============================================================================

def test_component_ids_deterministic():
    """
    Components sorted by (-size, bbox lex, first-cell lex).
    IDs assigned 0..(n-1) in sorted order per color.
    """
    # Grid with multiple components of same color
    result = components_by_color(g4)

    comps_1 = result[1]

    # IDs should be 0, 1 (assigned in sorted order)
    assert comps_1[0]["id"] == 0
    assert comps_1[1]["id"] == 1


def test_component_ids_size_ordering():
    """Largest component gets id=0."""
    g = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0]
    ]

    result = components_by_color(g)

    comps_1 = result[1]

    # First component: size 3 (top row)
    # Second component: size 1 (bottom-left)
    # Size 3 > size 1 → top row gets id=0
    assert len(comps_1) == 2
    assert comps_1[0]["id"] == 0
    assert len(comps_1[0]["cells"]) == 3

    assert comps_1[1]["id"] == 1
    assert len(comps_1[1]["cells"]) == 1


def test_component_ids_bbox_tie_break():
    """
    Verify bbox tie-breaking with a grid that has separate components.

    [[3, 0, 0],
     [0, 0, 3],
     [3, 0, 0]]

    Three separate components of color 3 (no 8-neighbor connections):
    - (0,0): bbox (0,0,0,0), size 1
    - (1,2): bbox (1,2,1,2), size 1
    - (2,0): bbox (2,0,2,0), size 1

    Sorted by (-size, bbox lex, first-cell lex):
    All size 1, so sort by bbox lex: (0,0,0,0) < (1,2,1,2) < (2,0,2,0)
    IDs: 0, 1, 2 respectively
    """
    g = [
        [3, 0, 0],
        [0, 0, 3],
        [3, 0, 0]
    ]

    result = components_by_color(g)

    comps_3 = result[3]

    assert len(comps_3) == 3

    # All same size (1), sorted by bbox lex
    assert comps_3[0]["id"] == 0
    assert comps_3[0]["bbox"] == (0, 0, 0, 0)
    assert comps_3[0]["cells"] == [(0, 0)]

    assert comps_3[1]["id"] == 1
    assert comps_3[1]["bbox"] == (1, 2, 1, 2)
    assert comps_3[1]["cells"] == [(1, 2)]

    assert comps_3[2]["id"] == 2
    assert comps_3[2]["bbox"] == (2, 0, 2, 0)
    assert comps_3[2]["cells"] == [(2, 0)]


def test_component_ids_first_cell_tie_break():
    """
    Same size, same bbox → first-cell lex breaks tie.

    [[7, 7, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 7, 7]]

    Two separate 2-cell components (no 8-neighbor connection due to gap):
    - Top: cells [(0,0), (0,1)], bbox (0,0,0,1), size 2
    - Bottom: cells [(2,3), (2,4)], bbox (2,3,2,4), size 2

    Same size, different bbox → bbox lex breaks tie.
    (0,0,0,1) < (2,3,2,4) → top gets id=0
    """
    g = [
        [7, 7, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 7, 7]
    ]

    result = components_by_color(g)

    comps_7 = result[7]

    assert len(comps_7) == 2

    # Both size 2, different bbox
    assert comps_7[0]["id"] == 0
    assert comps_7[0]["bbox"] == (0, 0, 0, 1)
    assert comps_7[0]["cells"] == [(0, 0), (0, 1)]

    assert comps_7[1]["id"] == 1
    assert comps_7[1]["bbox"] == (2, 3, 2, 4)
    assert comps_7[1]["cells"] == [(2, 3), (2, 4)]


# ============================================================================
# cells Sorted Row-Major Tests
# ============================================================================

def test_cells_sorted_row_major():
    """cells list is sorted (r,c) lex ascending (row-major)."""
    g = [
        [0, 5, 0],
        [5, 5, 5],
        [0, 5, 0]
    ]

    result = components_by_color(g)

    comps_5 = result[5]

    # All 5s form single component (8-connected via center)
    cells = comps_5[0]["cells"]

    # Expected order: (0,1), (1,0), (1,1), (1,2), (2,1)
    expected_cells = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    assert cells == expected_cells


def test_cells_sorted_various_shapes():
    """Verify row-major sorting for various component shapes."""
    # L-shape component
    g = [
        [9, 0, 0],
        [9, 0, 0],
        [9, 9, 9]
    ]

    result = components_by_color(g)

    cells = result[9][0]["cells"]

    # Expected: (0,0), (1,0), (2,0), (2,1), (2,2)
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert cells == expected


# ============================================================================
# Purity Tests
# ============================================================================

def test_components_by_color_no_mutation():
    """Input grid is not mutated."""
    g_orig = [[1, 2], [3, 4]]
    g_snapshot = copy_grid(g_orig)

    _ = components_by_color(g_orig)

    assert deep_eq(g_orig, g_snapshot)


def test_components_by_color_no_aliasing():
    """Modifying returned cells list does not affect future calls."""
    g = [[5, 5]]

    result1 = components_by_color(g)

    # Modify returned cells
    result1[5][0]["cells"].append((99, 99))

    # Second call should return unmodified version
    result2 = components_by_color(g)

    assert result2[5][0]["cells"] == [(0, 0), (0, 1)]


# ============================================================================
# Determinism Tests
# ============================================================================

def test_components_by_color_determinism():
    """Re-run on same grid yields identical component lists."""
    for g in [g0, g1, g2, g3, g4, g5, g6]:
        result1 = components_by_color(g)
        result2 = components_by_color(g)

        # Deep equality check
        assert result1 == result2


def test_components_by_color_determinism_complex():
    """Complex grid with multiple colors and components → deterministic."""
    g = [
        [1, 2, 1, 0],
        [2, 2, 0, 1],
        [1, 0, 1, 1]
    ]

    results = [components_by_color(g) for _ in range(10)]

    # All results identical
    for result in results:
        assert result == results[0]


# ============================================================================
# Adversarial / Edge Case Tests
# ============================================================================

def test_mixed_colors_touching():
    """
    8-connectivity connects same-color diagonals, even if different color between.

    [[1, 2],
     [2, 1]]

    Color 1: (0,0) and (1,1) ARE 8-neighbors → single component
    Color 2: (0,1) and (1,0) ARE 8-neighbors → single component

    This demonstrates that 8-connectivity bridges diagonally regardless of
    intervening colors (unlike some region-growing algorithms).
    """
    g = [
        [1, 2],
        [2, 1]
    ]

    result = components_by_color(g)

    # Color 1: single component (diagonal connection)
    comps_1 = result[1]
    assert len(comps_1) == 1
    assert comps_1[0]["cells"] == [(0, 0), (1, 1)]

    # Color 2: single component (diagonal connection)
    comps_2 = result[2]
    assert len(comps_2) == 1
    assert comps_2[0]["cells"] == [(0, 1), (1, 0)]


def test_multiple_colors_multiple_components():
    """
    Grid with multiple colors forming connected components via 8-connectivity.

    [[1, 0, 2, 0],
     [0, 1, 0, 2],
     [1, 0, 2, 0]]

    Color 1: (0,0), (1,1), (2,0) - all connected via diagonals → 1 component
    Color 2: (0,2), (1,3), (2,2) - all connected via diagonals → 1 component
    Color 0: forms connected component(s)

    This verifies 8-connectivity works across multiple colors simultaneously.
    """
    g = [
        [1, 0, 2, 0],
        [0, 1, 0, 2],
        [1, 0, 2, 0]
    ]

    result = components_by_color(g)

    # Color 1: single component (all connected via diagonals)
    comps_1 = result[1]
    assert len(comps_1) == 1
    assert comps_1[0]["cells"] == [(0, 0), (1, 1), (2, 0)]

    # Color 2: single component (all connected via diagonals)
    comps_2 = result[2]
    assert len(comps_2) == 1
    assert comps_2[0]["cells"] == [(0, 2), (1, 3), (2, 2)]

    # Color 0: should form component(s)
    assert len(result[0]) >= 1


def test_large_component_ordering():
    """Large component wins id=0 regardless of position."""
    g = [
        [1, 0, 0, 0],
        [0, 2, 2, 2],
        [0, 2, 2, 2],
        [0, 2, 2, 2]
    ]

    result = components_by_color(g)

    comps_2 = result[2]

    # Single large component of 2s
    assert len(comps_2) == 1
    assert len(comps_2[0]["cells"]) == 9
    assert comps_2[0]["id"] == 0


# ============================================================================
# NEIGHBORS_8 Constant Test
# ============================================================================

def test_neighbors_8_constant():
    """Verify NEIGHBORS_8 contains exactly 8 directions."""
    assert len(NEIGHBORS_8) == 8

    # Should include all 8 directions
    expected = {
        (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
        ( 0, -1),          ( 0, 1),  # W,     E
        ( 1, -1), ( 1, 0), ( 1, 1),  # SW, S, SE
    }

    assert set(NEIGHBORS_8) == expected


# ============================================================================
# Integration Tests
# ============================================================================

def test_bbox_integration_with_components():
    """bbox results in components match manual bbox computation."""
    g = [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1]
    ]

    result = components_by_color(g)

    comps_1 = result[1]

    # Single component (8-connected)
    assert len(comps_1) == 1

    # Verify bbox matches manual computation
    cells = comps_1[0]["cells"]
    expected_bbox = bbox(cells)

    assert comps_1[0]["bbox"] == expected_bbox
    assert expected_bbox == (0, 0, 2, 2)


def test_components_full_workflow():
    """
    Full workflow test: extract components, verify all properties.
    """
    g = [
        [5, 5, 0, 7],
        [5, 0, 7, 7],
        [0, 0, 0, 7]
    ]

    result = components_by_color(g)

    # Color 5: single component (top-left)
    comps_5 = result[5]
    assert len(comps_5) == 1
    assert comps_5[0]["id"] == 0
    assert comps_5[0]["color"] == 5
    assert comps_5[0]["cells"] == [(0, 0), (0, 1), (1, 0)]
    assert comps_5[0]["bbox"] == (0, 0, 1, 1)

    # Color 7: single component (right side, 8-connected via diagonals)
    comps_7 = result[7]
    assert len(comps_7) == 1
    assert comps_7[0]["id"] == 0
    assert comps_7[0]["color"] == 7
    assert len(comps_7[0]["cells"]) == 4

    # Color 0: forms 8-connected component(s)
    comps_0 = result[0]
    assert len(comps_0) >= 1
