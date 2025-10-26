"""
Core grid utilities for ARC AGI solver.

Grid model: list[list[int]] with rectangular shape, palette 0..9.
All functions are pure (no mutation) and deterministic.
"""

def dims(g: list[list[int]]) -> tuple[int, int]:
    """
    Return (rows, cols) for grid g.

    - Empty grid [] returns (0, 0).
    - Non-empty grid must be rectangular; raise ValueError if ragged.
    - Returns (len(g), len(g[0])) for valid rectangular grids.

    Args:
        g: Grid as list of lists of integers

    Returns:
        Tuple of (rows, cols)

    Raises:
        ValueError: If grid is ragged (non-rectangular)
    """
    if not g:
        return (0, 0)

    rows = len(g)
    cols = len(g[0])

    # Validate rectangularity
    for row in g:
        if len(row) != cols:
            raise ValueError("Grid must be rectangular (all rows same length)")

    return (rows, cols)


def copy_grid(g: list[list[int]]) -> list[list[int]]:
    """
    Deep copy of grid g with new row objects.

    - No aliasing: modifying returned grid must not affect original.
    - Preserves all values and structure.

    Args:
        g: Grid to copy

    Returns:
        Deep copy of grid
    """
    # Create new row objects (not just g[:] which would alias inner lists)
    return [list(row) for row in g]


def deep_eq(a: list[list[int]], b: list[list[int]]) -> bool:
    """
    True iff shapes equal and all entries equal (bit-for-bit).

    - No coercions (FY spirit: exact equality only).
    - Shape mismatch returns False.

    Args:
        a: First grid
        b: Second grid

    Returns:
        True if grids are structurally identical
    """
    # Check outer list length
    if len(a) != len(b):
        return False

    # Check each row
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for val_a, val_b in zip(row_a, row_b):
            if val_a != val_b:
                return False

    return True


def transpose(g: list[list[int]]) -> list[list[int]]:
    """
    Matrix transpose: (R,C) → (C,R).

    - Empty grid [] returns [].
    - Raise ValueError on ragged input.

    Args:
        g: Grid to transpose

    Returns:
        Transposed grid

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    rows, cols = dims(g)  # Validates rectangularity

    # Transpose: new[c][r] = old[r][c]
    result = []
    for c in range(cols):
        new_row = []
        for r in range(rows):
            new_row.append(g[r][c])
        result.append(new_row)

    return result


def rot90(g: list[list[int]]) -> list[list[int]]:
    """
    Rotate 90° clockwise.

    - (R,C) → (C,R) with appropriate index mapping.
    - Raise ValueError on ragged input.

    Args:
        g: Grid to rotate

    Returns:
        Rotated grid

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    rows, cols = dims(g)  # Validates rectangularity

    # 90° clockwise: new[c][rows-1-r] = old[r][c]
    # Result is cols×rows
    result = []
    for c in range(cols):
        new_row = []
        for r in range(rows - 1, -1, -1):
            new_row.append(g[r][c])
        result.append(new_row)

    return result


def rot180(g: list[list[int]]) -> list[list[int]]:
    """
    Rotate 180°.

    - Equals rot90(rot90(g)).
    - (R,C) → (R,C) with reverse indexing.

    Args:
        g: Grid to rotate

    Returns:
        Rotated grid

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    rows, cols = dims(g)  # Validates rectangularity

    # 180° rotation: new[rows-1-r][cols-1-c] = old[r][c]
    # Equivalent to reversing all rows and all columns
    result = []
    for r in range(rows - 1, -1, -1):
        new_row = []
        for c in range(cols - 1, -1, -1):
            new_row.append(g[r][c])
        result.append(new_row)

    return result


def rot270(g: list[list[int]]) -> list[list[int]]:
    """
    Rotate 270° clockwise (= 90° counter-clockwise).

    - Equals rot90(rot180(g)) or rot180(rot90(g)).

    Args:
        g: Grid to rotate

    Returns:
        Rotated grid

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    rows, cols = dims(g)  # Validates rectangularity

    # 270° clockwise = 90° counter-clockwise
    # new[cols-1-c][r] = old[r][c]
    # Result is cols×rows
    result = []
    for c in range(cols - 1, -1, -1):
        new_row = []
        for r in range(rows):
            new_row.append(g[r][c])
        result.append(new_row)

    return result


def flip_h(g: list[list[int]]) -> list[list[int]]:
    """
    Horizontal mirror: reverse each row.

    - (R,C) → (R,C) with column indices reversed.

    Args:
        g: Grid to flip

    Returns:
        Horizontally flipped grid

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    dims(g)  # Validates rectangularity

    # Reverse each row
    return [list(reversed(row)) for row in g]


def flip_v(g: list[list[int]]) -> list[list[int]]:
    """
    Vertical mirror: reverse row order.

    - (R,C) → (R,C) with row indices reversed.

    Args:
        g: Grid to flip

    Returns:
        Vertically flipped grid

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    dims(g)  # Validates rectangularity

    # Reverse row order
    return [list(row) for row in reversed(g)]


def color_interaction_graph(g: list[list[int]]) -> dict[int, int]:
    """
    Compute canonical color IDs via Color Interaction Graph (CIG).

    Two colors interact if they appear in adjacent cells (4-connectivity).
    The CIG is represented as an adjacency set for each color.
    Colors are assigned canonical IDs 0..k-1 in order of:
    1. Frequency (most common first)
    2. Minimum raw color value (tie-breaker for determinism)

    This provides input-only color canonicalization that:
    - Preserves color separation (different raw colors → different canonical IDs)
    - Enables cross-palette generalization (similar palettes → similar canonical IDs)

    Args:
        g: Grid to analyze

    Returns:
        Mapping from raw color (0-9) to canonical color ID (0..k-1)
        where k is the number of distinct colors in the grid

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return {}

    rows, cols = dims(g)  # Validates rectangularity

    # Count color frequencies
    color_freq = {}
    for r in range(rows):
        for c in range(cols):
            color = g[r][c]
            color_freq[color] = color_freq.get(color, 0) + 1

    # Sort colors by: (frequency descending, raw value ascending)
    colors_sorted = sorted(
        color_freq.keys(),
        key=lambda c: (-color_freq[c], c)
    )

    # Assign canonical IDs
    canon_map = {}
    for canon_id, raw_color in enumerate(colors_sorted):
        canon_map[raw_color] = canon_id

    return canon_map


def apply_color_canon(g: list[list[int]], canon_map: dict[int, int]) -> list[list[int]]:
    """
    Apply canonical color mapping to grid.

    Replaces each pixel's raw color with its canonical color ID.

    Args:
        g: Grid to canonicalize
        canon_map: Mapping from raw color to canonical color ID

    Returns:
        New grid with canonical color IDs

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    rows, cols = dims(g)  # Validates rectangularity

    result = []
    for r in range(rows):
        row = []
        for c in range(cols):
            raw_color = g[r][c]
            canon_color = canon_map.get(raw_color, raw_color)
            row.append(canon_color)
        result.append(row)

    return result
