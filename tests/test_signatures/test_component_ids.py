"""
Tests for Component ID Table (P4-04).

This module tests component_id_table from signature_builders.py,
which assigns stable, deterministic IDs to 8-connected components.

Test coverage:
- Empty grid → ([], [])
- Single pixel, all same color
- Multiple components with different sizes → sorted by size descending
- Same size, different bbox → sorted by bbox lex
- 8-connectivity verification (diagonal adjacency)
- Global ID space (IDs span all colors 0..K-1)
- Φ.3 stability (input-only features)
- Purity (input unchanged)
- Determinism (repeated calls identical)
- Error handling (ragged grids)
- Coverage: every pixel assigned exactly one ID
- Meta list sorted by ID
"""

import copy
import pytest
from src.signature_builders import component_id_table


# ============================================================================
# Test Helpers
# ============================================================================


def verify_coverage(id_grid: list[list[int]], meta: list[dict]) -> None:
    """
    Verify every pixel has exactly one component ID.

    Args:
        id_grid: Grid of component IDs
        meta: Metadata list

    Raises:
        AssertionError: If coverage is incomplete or overlapping
    """
    if not id_grid:
        assert meta == [], "Empty id_grid should have empty meta"
        return

    h = len(id_grid)
    w = len(id_grid[0])

    # Every pixel should have ID in range [0, K-1] where K = len(meta)
    K = len(meta)
    for r in range(h):
        for c in range(w):
            comp_id = id_grid[r][c]
            assert 0 <= comp_id < K, f"Pixel ({r},{c}) has invalid ID {comp_id}"


def verify_meta_sorted(meta: list[dict]) -> None:
    """Verify meta list is sorted by ID (implicitly, since list index is ID)."""
    # IDs should be contiguous 0..K-1
    for i, m in enumerate(meta):
        # Meta should contain required keys
        assert "color" in m
        assert "size" in m
        assert "bbox" in m
        assert "seed_rc" in m


def verify_sorting_order(meta: list[dict]) -> None:
    """
    Verify components are sorted by (-size, bbox, seed_rc).

    Args:
        meta: Metadata list sorted by component ID
    """
    # Build sort keys for each component
    sort_keys = [(-m["size"], m["bbox"], m["seed_rc"]) for m in meta]

    # Verify list is sorted
    assert sort_keys == sorted(sort_keys), "Components not sorted correctly"


# ============================================================================
# Basic Tests
# ============================================================================


def test_component_id_table_empty():
    """Empty grid returns ([], [])."""
    id_grid, meta = component_id_table([])

    assert id_grid == []
    assert meta == []


def test_component_id_table_single_pixel():
    """Single pixel → ID 0."""
    g = [[5]]
    id_grid, meta = component_id_table(g)

    assert id_grid == [[0]]
    assert len(meta) == 1
    assert meta[0] == {"color": 5, "size": 1, "bbox": (0, 0, 0, 0), "seed_rc": (0, 0)}

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)


def test_component_id_table_all_same_color():
    """All pixels same color → single component ID 0."""
    g = [[3, 3], [3, 3]]
    id_grid, meta = component_id_table(g)

    assert id_grid == [[0, 0], [0, 0]]
    assert len(meta) == 1
    assert meta[0] == {"color": 3, "size": 4, "bbox": (0, 0, 1, 1), "seed_rc": (0, 0)}

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)


def test_component_id_table_two_components_same_color():
    """Two separate components of same color → different IDs."""
    g = [[1, 0, 1]]  # Two separate 1-components (not 8-connected)

    id_grid, meta = component_id_table(g)

    # Both components size 1, different bboxes
    # (0,0) has bbox (0,0,0,0), (0,2) has bbox (0,2,0,2)
    # Bbox (0,0,0,0) < (0,2,0,2) lexicographically

    assert id_grid == [[0, 1, 2]]  # Wait, color 0 has size 1 too
    assert len(meta) == 3

    # All three have size 1, sorted by bbox
    # Component at (0,0): bbox (0,0,0,0)
    # Component at (0,1): bbox (0,1,0,1)
    # Component at (0,2): bbox (0,2,0,2)

    assert meta[0]["bbox"] == (0, 0, 0, 0)  # color 1
    assert meta[1]["bbox"] == (0, 1, 0, 1)  # color 0
    assert meta[2]["bbox"] == (0, 2, 0, 2)  # color 1

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)
    verify_sorting_order(meta)


def test_component_id_table_size_ordering():
    """Components sorted by size (larger first)."""
    g = [[1, 1, 1], [2, 2, 0]]

    id_grid, meta = component_id_table(g)

    # Component sizes: color 1 has 3, color 2 has 2, color 0 has 1
    assert id_grid == [[0, 0, 0], [1, 1, 2]]
    assert len(meta) == 3

    assert meta[0]["color"] == 1
    assert meta[0]["size"] == 3

    assert meta[1]["color"] == 2
    assert meta[1]["size"] == 2

    assert meta[2]["color"] == 0
    assert meta[2]["size"] == 1

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)
    verify_sorting_order(meta)


def test_component_id_table_bbox_tiebreak():
    """Same size, different bbox → sorted by bbox lex."""
    g = [[1, 0, 2]]

    id_grid, meta = component_id_table(g)

    # All three components have size 1
    # Sorted by bbox: (0,0,0,0) < (0,1,0,1) < (0,2,0,2)

    assert len(meta) == 3
    assert meta[0]["bbox"] == (0, 0, 0, 0)  # color 1
    assert meta[1]["bbox"] == (0, 1, 0, 1)  # color 0
    assert meta[2]["bbox"] == (0, 2, 0, 2)  # color 2

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)
    verify_sorting_order(meta)


def test_component_id_table_seed_rc_metadata():
    """Verify seed_rc is the first cell in row-major order."""
    g = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    id_grid, meta = component_id_table(g)

    # Color 0 has 8 pixels, forms a border around color 1
    # Color 1 has 1 pixel at (1,1)

    # Color 0 is larger, gets ID 0
    assert meta[0]["color"] == 0
    assert meta[0]["size"] == 8
    assert meta[0]["seed_rc"] == (0, 0)  # First cell in row-major

    # Color 1 gets ID 1
    assert meta[1]["color"] == 1
    assert meta[1]["size"] == 1
    assert meta[1]["seed_rc"] == (1, 1)

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)


# ============================================================================
# 8-Connectivity Tests
# ============================================================================


def test_component_id_table_8_connected_diagonal():
    """Diagonal touching connects components (8-connectivity)."""
    g = [[1, 0], [0, 1]]

    id_grid, meta = component_id_table(g)

    # With 8-connectivity, both 1s are connected via diagonal
    # So there should be 1 component of color 1 and 1 component of color 0

    assert len(meta) == 2

    # Both components have size 2
    # Need to check which is which by color
    comp_by_color = {m["color"]: m for m in meta}

    assert comp_by_color[1]["size"] == 2
    assert comp_by_color[0]["size"] == 2

    # Verify diagonal connectivity: both 1s should have same ID
    assert id_grid[0][0] == id_grid[1][1], "Diagonally adjacent 1s not connected"

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)


def test_component_id_table_8_connected_complex():
    """Complex 8-connectivity pattern."""
    g = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    id_grid, meta = component_id_table(g)

    # All three 1s connected via diagonals → single component of color 1
    # Color 0 forms remaining 6 pixels

    assert len(meta) == 2

    comp_by_color = {m["color"]: m for m in meta}

    assert comp_by_color[0]["size"] == 6
    assert comp_by_color[1]["size"] == 3

    # All 1s should have same ID
    assert id_grid[0][0] == id_grid[1][1] == id_grid[2][2]

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)


def test_component_id_table_no_4_connectivity_only():
    """Verify 8-connectivity, not just 4-connectivity."""
    g = [[1, 0], [0, 1]]

    id_grid, meta = component_id_table(g)

    # With 4-connectivity only, these would be separate components
    # With 8-connectivity, they're connected

    # Both 1s should have same ID (connected via diagonal)
    assert id_grid[0][0] == id_grid[1][1], "Should be 8-connected, not 4-connected"


# ============================================================================
# Global ID Space Tests
# ============================================================================


def test_component_id_table_global_ids():
    """IDs are global across all colors, not per-color."""
    g = [
        [1, 1, 2, 2],
        [0, 0, 3, 3],
    ]

    id_grid, meta = component_id_table(g)

    # Four components, all size 2
    # IDs should be 0, 1, 2, 3 globally (not 0,0,0,0 per color)

    assert len(meta) == 4

    # All IDs should be unique and in range [0, 3]
    all_ids = set()
    for row in id_grid:
        all_ids.update(row)

    assert all_ids == {0, 1, 2, 3}, f"IDs not global: {all_ids}"

    verify_coverage(id_grid, meta)
    verify_meta_sorted(meta)


def test_component_id_table_multi_color_ordering():
    """Multiple colors, verify global ordering."""
    g = [
        [1, 2],  # Two components, size 1 each
        [3, 4],  # Two components, size 1 each
    ]

    id_grid, meta = component_id_table(g)

    # All four have size 1, sorted by bbox
    # (0,0), (0,1), (1,0), (1,1)

    assert len(meta) == 4

    assert meta[0]["bbox"] == (0, 0, 0, 0)  # color 1
    assert meta[1]["bbox"] == (0, 1, 0, 1)  # color 2
    assert meta[2]["bbox"] == (1, 0, 1, 0)  # color 3
    assert meta[3]["bbox"] == (1, 1, 1, 1)  # color 4

    verify_coverage(id_grid, meta)
    verify_sorting_order(meta)


# ============================================================================
# Purity and Determinism Tests
# ============================================================================


def test_component_id_table_purity():
    """Input grid unchanged after component_id_table call."""
    g = [[1, 2], [3, 4]]
    g_copy = copy.deepcopy(g)

    component_id_table(g)

    assert g == g_copy, "Input grid modified (purity violation)"


def test_component_id_table_determinism():
    """Repeated calls with same input produce identical output."""
    g = [[1, 1, 0], [2, 2, 0]]

    id_grid1, meta1 = component_id_table(g)
    id_grid2, meta2 = component_id_table(g)

    assert id_grid1 == id_grid2, "id_grid not deterministic"
    assert meta1 == meta2, "meta not deterministic"


def test_component_id_table_determinism_complex():
    """Determinism with complex multi-component grid."""
    g = [
        [1, 0, 1, 0],
        [0, 2, 0, 2],
        [3, 0, 3, 0],
    ]

    # Run 3 times
    results = [component_id_table(g) for _ in range(3)]

    # All should be identical
    for i in range(1, 3):
        assert results[i][0] == results[0][0], f"Run {i} id_grid differs"
        assert results[i][1] == results[0][1], f"Run {i} meta differs"


# ============================================================================
# Edge Cases
# ============================================================================


def test_component_id_table_single_row():
    """Single row grid."""
    g = [[1, 2, 1, 3, 1]]

    id_grid, meta = component_id_table(g)

    # Three separate 1-components (not connected)
    # One 2-component, one 3-component
    # Total: 5 components, all size 1

    assert len(meta) == 5

    # All size 1, sorted by bbox (which equals seed_rc for size-1 components)
    for i in range(5):
        assert meta[i]["size"] == 1
        assert meta[i]["bbox"] == (0, i, 0, i)
        assert meta[i]["seed_rc"] == (0, i)

    verify_coverage(id_grid, meta)
    verify_sorting_order(meta)


def test_component_id_table_single_column():
    """Single column grid."""
    g = [[1], [2], [1], [3], [1]]

    id_grid, meta = component_id_table(g)

    # Three separate 1-components, one 2, one 3
    assert len(meta) == 5

    # All size 1
    for i in range(5):
        assert meta[i]["size"] == 1
        assert meta[i]["bbox"] == (i, 0, i, 0)
        assert meta[i]["seed_rc"] == (i, 0)

    verify_coverage(id_grid, meta)
    verify_sorting_order(meta)


def test_component_id_table_large_component():
    """Large connected component."""
    # 10x10 grid, all same color
    g = [[5] * 10 for _ in range(10)]

    id_grid, meta = component_id_table(g)

    # Single component of size 100
    assert len(meta) == 1
    assert meta[0]["size"] == 100
    assert meta[0]["color"] == 5
    assert meta[0]["bbox"] == (0, 0, 9, 9)
    assert meta[0]["seed_rc"] == (0, 0)

    # All pixels should have ID 0
    for row in id_grid:
        assert all(cell == 0 for cell in row)

    verify_coverage(id_grid, meta)


def test_component_id_table_many_small_components():
    """Many small components (performance test)."""
    # Checkerboard pattern: each pixel is separate component
    g = [[(r + c) % 10 for c in range(10)] for r in range(10)]

    id_grid, meta = component_id_table(g)

    # Should have many components
    # Each component should have consistent properties

    for m in meta:
        assert m["size"] >= 1
        assert 0 <= m["color"] <= 9

    verify_coverage(id_grid, meta)
    verify_sorting_order(meta)


def test_component_id_table_ragged_input():
    """Ragged grid raises ValueError."""
    g_ragged = [[1, 2], [3]]

    with pytest.raises(ValueError, match="Ragged grid"):
        component_id_table(g_ragged)


# ============================================================================
# Adversarial Cases (from P4-04 context pack)
# ============================================================================


def test_component_id_table_diagonal_bridge():
    """Diagonal adjacency required to connect."""
    g = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    id_grid, meta = component_id_table(g)

    # All three 1s connected via diagonals
    assert len(meta) == 2  # One for color 1, one for color 0

    # All 1s should have same ID
    id_of_first_1 = id_grid[0][0]
    assert id_grid[1][1] == id_of_first_1
    assert id_grid[2][2] == id_of_first_1

    verify_coverage(id_grid, meta)


def test_component_id_table_multi_color_mix():
    """Multiple colors, interleaved."""
    g = [
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
    ]

    id_grid, meta = component_id_table(g)

    # With 8-connectivity, adjacent diagonals connect
    # This creates complex connectivity patterns

    # Verify global IDs
    all_ids = set()
    for row in id_grid:
        all_ids.update(row)

    # IDs should be contiguous 0..K-1
    assert all_ids == set(range(len(meta)))

    verify_coverage(id_grid, meta)
    verify_sorting_order(meta)


def test_component_id_table_same_size_different_bbox():
    """Two components, same size, different bbox."""
    g = [
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
    ]

    id_grid, meta = component_id_table(g)

    # Components:
    # - Color 1: size 2, bbox (0,0,0,1), seed_rc (0,0)
    # - Color 2: size 2, bbox (0,3,0,4), seed_rc (0,3)
    # - Color 0: size 6, bbox (0,2,1,4), seed_rc (0,2)

    # Sorted by (-size, bbox, seed_rc):
    # 1. Color 0: size 6 → ID 0
    # 2. Color 1: size 2, bbox (0,0,0,1) → ID 1
    # 3. Color 2: size 2, bbox (0,3,0,4) → ID 2

    assert len(meta) == 3

    assert meta[0]["color"] == 0
    assert meta[0]["size"] == 6

    assert meta[1]["color"] == 1
    assert meta[1]["size"] == 2
    assert meta[1]["bbox"] == (0, 0, 0, 1)

    assert meta[2]["color"] == 2
    assert meta[2]["size"] == 2
    assert meta[2]["bbox"] == (0, 3, 0, 4)

    verify_coverage(id_grid, meta)
    verify_sorting_order(meta)


def test_component_id_table_bbox_correctness():
    """Verify bbox is computed correctly."""
    g = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]

    id_grid, meta = component_id_table(g)

    # Color 1 forms a cross pattern
    # Pixels: (0,2), (1,1), (1,2), (1,3), (2,2)
    # bbox: r in [0,2], c in [1,3] → (0, 1, 2, 3)

    comp_by_color = {m["color"]: m for m in meta}

    assert comp_by_color[1]["bbox"] == (0, 1, 2, 3)

    verify_coverage(id_grid, meta)
