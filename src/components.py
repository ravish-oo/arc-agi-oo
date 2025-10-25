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
