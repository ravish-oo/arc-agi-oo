"""
8-connected component extraction for ARC AGI solver.

Provides:
- bbox(cells): Inclusive bounding box for component cells
- components_by_color(g): 8-connected component extraction grouped by color

All functions are pure (no mutation) and deterministic.
"""

from src.utils import dims


# 8-neighbor offsets (all 8 directions including diagonals)
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
    ( 0, -1),          ( 0, 1),  # W,     E
    ( 1, -1), ( 1, 0), ( 1, 1),  # SW, S, SE
]


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """
    Return inclusive bounding box (r0, c0, r1, c1) for a component's cells.

    Args:
        cells: Non-empty list of (row, col) coordinates

    Returns:
        (r0, c0, r1, c1) where:
          r0 = min row, c0 = min col
          r1 = max row, c1 = max col
          All inclusive (component pixels exist at boundaries)

    Raises:
        ValueError: If cells is empty

    Edge cases:
        - Single cell [(2,3)] → (2, 3, 2, 3)
        - Horizontal line [(0,1), (0,2), (0,3)] → (0, 1, 0, 3)
        - Vertical line [(1,0), (2,0), (3,0)] → (1, 0, 3, 0)

    Invariants:
        - Pure function (no mutation)
        - Deterministic (same cells → same bbox)
        - r0 ≤ r1 and c0 ≤ c1 always
    """
    if not cells:
        raise ValueError("Cannot compute bounding box for empty cells list")

    # Extract all rows and columns
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

    # Return (min_row, min_col, max_row, max_col)
    return (min(rows), min(cols), max(rows), max(cols))


def components_by_color(g: list[list[int]]) -> dict[int, list[dict]]:
    """
    Extract 8-connected components grouped by pixel color (0..9).

    8-connectivity: pixels (r1,c1) and (r2,c2) are neighbors if:
        max(abs(r2-r1), abs(c2-c1)) == 1
    This includes all 8 directions: N, S, E, W, NE, NW, SE, SW.
    Diagonal-only touching creates connections (unlike 4-connected).

    Args:
        g: Rectangular grid (list of lists of integers 0..9)

    Returns:
        dict mapping color → list of components, where each component is:
        {
            "id": int,              # 0-based ID within this color (deterministic)
            "color": int,           # pixel value (0..9)
            "cells": list[tuple[int,int]],  # sorted (r,c) row-major ascending
            "bbox": tuple[int,int,int,int]  # (r0, c0, r1, c1) inclusive
        }

        Colors not present in grid are omitted from result.
        Components within each color are sorted by deterministic tie-break:
          1. -size (largest first)
          2. bbox lex: (r0, c0, r1, c1) ascending
          3. first cell (r, c) lex ascending

        IDs are assigned 0..(n-1) following this sorted order per color.

    Raises:
        ValueError: If grid is ragged (non-rectangular)

    Edge cases:
        - [] → {}
        - All same color → {color: [single component with all pixels]}
        - Checkerboard diagonal touch:
            [[1,0],
             [0,1]]
          Color 1 forms TWO separate components (no diagonal bridge across 0s)
        - Corner touch DOES connect:
            [[1,0],
             [0,1]]
          If positions are (0,0) and (1,1), they ARE 8-connected neighbors.
        - Background color 0 typically ignored in ARC (but not special-cased here;
          if 0-pixels exist, they form components like any other color)

    Invariants:
        - Pure: input grid unchanged
        - Deterministic: same grid → identical component lists and IDs
        - 8-connectivity enforced (not 4-connected)
        - Stable tie-breaks ensure reproducible IDs
        - cells lists are sorted row-major: (r,c) lex ascending
        - No aliasing: cells lists are newly allocated
    """
    # Empty grid special case
    if not g:
        return {}

    # Validate rectangularity
    rows, cols = dims(g)  # Raises ValueError if ragged

    # Build position→color mapping for quick lookup
    pos_to_color = {}
    for r in range(rows):
        for c in range(cols):
            pos_to_color[(r, c)] = g[r][c]

    # Track visited positions across all colors
    visited = set()

    # Group components by color
    components_dict = {}  # color → list of raw components

    # BFS flood-fill for each unvisited position
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited:
                continue

            # Start new component
            color = g[r][c]
            component_cells = []
            queue = [(r, c)]
            visited.add((r, c))

            # BFS flood fill with 8-connectivity
            while queue:
                curr_r, curr_c = queue.pop(0)
                component_cells.append((curr_r, curr_c))

                # Check all 8 neighbors
                for dr, dc in NEIGHBORS_8:
                    nr, nc = curr_r + dr, curr_c + dc

                    # Bounds check
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue

                    # Already visited check
                    if (nr, nc) in visited:
                        continue

                    # Same color check
                    if g[nr][nc] != color:
                        continue

                    # Add to component
                    visited.add((nr, nc))
                    queue.append((nr, nc))

            # Sort cells row-major (r,c) lex ascending
            component_cells.sort()

            # Store component by color
            if color not in components_dict:
                components_dict[color] = []
            components_dict[color].append({
                "color": color,
                "cells": component_cells,
            })

    # For each color, compute bbox, sort components, assign IDs
    result = {}
    for color, components in components_dict.items():
        # Compute bbox and size for each component
        for comp in components:
            comp["bbox"] = bbox(comp["cells"])
            comp["size"] = len(comp["cells"])

        # Sort by tie-break key: (-size, bbox, first_cell)
        components.sort(key=lambda c: (
            -c["size"],           # Largest first (negate for ascending sort)
            c["bbox"],            # (r0, c0, r1, c1) lex ascending
            c["cells"][0]         # First cell (r,c) lex ascending
        ))

        # Assign IDs and remove temporary size field
        for i, comp in enumerate(components):
            comp["id"] = i
            del comp["size"]

        result[color] = components

    return result


def boundaries_by_any_change(g: list[list[int]], axis: str) -> list[int]:
    """
    Return sorted boundary indices where adjacent rows or columns differ.

    Non-Periodic Segmentation (NPS): detect content-change boundaries along
    an axis to partition grid into homogeneous bands.

    Args:
        g: Rectangular grid (list of lists of integers 0..9)
        axis: "row" or "col"
          - "row": compare row i and row i+1; boundary at i if any column differs
          - "col": compare col j and col j+1; boundary at j if any row differs

    Returns:
        Strictly increasing list of boundary indices in [0..L-2], where:
          - For axis="row": L = number of rows; boundaries in range [0..rows-2]
          - For axis="col": L = number of cols; boundaries in range [0..cols-2]
        Empty list if no boundaries (all rows/cols identical) or grid too small.

    Raises:
        ValueError: If grid is ragged (non-rectangular)
        ValueError: If axis not in {"row", "col"}

    Edge cases:
        - [] → [] (no boundaries)
        - Single row [[1,2,3]] with axis="row" → [] (no adjacent rows to compare)
        - Single col [[1],[2],[3]] with axis="col" → [] (no adjacent cols)
        - All rows identical → [] (no change boundaries)
        - Alternating rows → [0, 1, 2, ...] (boundary at every index)

    Semantics:
        - axis="row": boundary i means "rows i and i+1 differ in at least one column"
        - axis="col": boundary j means "columns j and j+1 differ in at least one row"
        - Boundaries partition axis into bands; no boundary means single band

    Invariants:
        - Pure function (no mutation)
        - Deterministic (same grid → same boundaries)
        - Input-only (never depends on target Y)
        - Result is sorted ascending, unique, within valid range
    """
    # Empty grid special case
    if not g:
        return []

    # Validate rectangularity and get dimensions
    rows, cols = dims(g)  # Raises ValueError if ragged

    # Validate axis parameter
    if axis not in {"row", "col"}:
        raise ValueError(f'axis must be "row" or "col", got: {axis}')

    boundaries = []

    if axis == "row":
        # Compare each row i with row i+1
        for i in range(rows - 1):
            row_i = g[i]
            row_i1 = g[i + 1]
            # Boundary at i if ANY column differs
            if any(row_i[c] != row_i1[c] for c in range(cols)):
                boundaries.append(i)

    else:  # axis == "col"
        # Compare each column j with column j+1
        for j in range(cols - 1):
            # Extract columns j and j+1
            col_j = [g[r][j] for r in range(rows)]
            col_j1 = [g[r][j + 1] for r in range(rows)]
            # Boundary at j if ANY row differs
            if any(col_j[r] != col_j1[r] for r in range(rows)):
                boundaries.append(j)

    return boundaries  # Already sorted by construction


def bands_from_boundaries(n: int, boundaries: list[int]) -> list[tuple[int, int]]:
    """
    Convert boundary indices to inclusive (start, end) band tuples.

    Given axis length n and sorted boundary indices, partition [0..n-1]
    into contiguous bands split at each boundary.

    Args:
        n: Axis length (rows or cols), must be ≥ 0
        boundaries: Sorted, unique boundary indices within [0..n-2]

    Returns:
        List of (start, end) inclusive band tuples partitioning [0..n-1]:
          - If boundaries = [b0, b1, ..., bk], return:
            [(0, b0), (b0+1, b1), (b1+1, b2), ..., (bk+1, n-1)]
          - Empty boundaries → single band [(0, n-1)] if n > 0, else []
          - n=0 → []
          - n=1 → [(0, 0)]

    Raises:
        ValueError: If n < 0
        ValueError: If boundaries not sorted or contain duplicates
        ValueError: If any boundary not in [0..n-2]

    Edge cases:
        - n=0, boundaries=[] → []
        - n=1, boundaries=[] → [(0, 0)]
        - n=3, boundaries=[] → [(0, 2)]
        - n=3, boundaries=[1] → [(0, 1), (2, 2)]
        - n=5, boundaries=[0, 2, 3] → [(0, 0), (1, 2), (3, 3), (4, 4)]
        - Consecutive boundaries create single-element bands

    Semantics:
        - Bands cover [0..n-1] exactly once (no gaps, no overlaps)
        - start ≤ end always (inclusive on both ends)
        - Bands are contiguous and in order

    Invariants:
        - Pure function (no mutation)
        - Deterministic (same inputs → same bands)
        - Full coverage: union of bands equals [0..n-1] when n > 0
        - Disjoint: bands do not overlap
    """
    # Validate n
    if n < 0:
        raise ValueError(f"n must be non-negative, got: {n}")

    # Empty axis case
    if n == 0:
        return []

    # Validate boundaries
    if boundaries:
        # Check sorted and unique
        for i in range(len(boundaries) - 1):
            if boundaries[i] >= boundaries[i + 1]:
                raise ValueError(f"boundaries must be strictly increasing (sorted, unique), got: {boundaries}")

        # Check range [0..n-2]
        if boundaries[0] < 0:
            raise ValueError(f"boundaries must be in [0..{n-2}], got: {boundaries}")
        if boundaries[-1] > n - 2:
            raise ValueError(f"boundaries must be in [0..{n-2}], got: {boundaries}")

    # No boundaries → single band [0..n-1]
    if not boundaries:
        return [(0, n - 1)]

    # Build bands from boundaries
    bands = []
    prev = 0
    for b in boundaries:
        bands.append((prev, b))
        prev = b + 1

    # Final band from last boundary+1 to n-1
    bands.append((prev, n - 1))

    return bands


def bands_from_boundaries(boundaries: list[int], axis_length: int) -> list[int]:
    """
    Convert boundary list to band index array.

    Args:
        boundaries: sorted list of boundary positions [b0, b1, ..., bn] where b0=0, bn=axis_length
        axis_length: total length of axis (H for rows, W for cols)

    Returns:
        Array of length axis_length where each position i maps to its band index
        Band j spans [boundaries[j], boundaries[j+1])

    Example:
        boundaries = [0, 2, 5, 8], axis_length = 8
        Returns: [0, 0, 1, 1, 1, 2, 2, 2]
                  |--band 0--|--band 1--|--band 2--|

    Algorithm:
        1. Create array of length axis_length initialized to 0
        2. For each band j in range(len(boundaries)-1):
            a. For each position i in range(boundaries[j], boundaries[j+1]):
                - array[i] = j
        3. Return array

    Edge cases:
        - Single band [0, axis_length]: all positions → band 0
        - Empty axis_length=0: return []
        - boundaries = [0, 1, 2, ..., axis_length]: each position is its own band

    Semantics:
        - boundaries[0] must be 0
        - boundaries[-1] must be axis_length
        - boundaries must be strictly increasing
        - No validation performed (caller ensures correctness)

    Purity:
        - Never mutates boundaries
        - Returns new array

    Determinism:
        - Same (boundaries, axis_length) → same output array
        - Iteration order is stable (range())
    """
    if axis_length == 0:
        return []

    # Create result array
    result = [0] * axis_length

    # Assign band indices
    for band_idx in range(len(boundaries) - 1):
        start = boundaries[band_idx]
        end = boundaries[band_idx + 1]
        for pos in range(start, end):
            result[pos] = band_idx

    return result
