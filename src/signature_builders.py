"""
Signature Builders (Φ Features) for ARC AGI Pure Math Solver.

This module implements the canonical signature system used to partition grids
into disjoint equivalence classes. All features are INPUT-ONLY (Φ.3 stability)
and form the finite Boolean basis for Step-2 (P+Φ/GLUE) solver.

Phase 4 Work Orders:
- P4-01: Index Predicates (parity_mask, rowmod_mask, colmod_mask)
- P4-02: NPS Bands (row_band_masks, col_band_masks)

Public API:
- parity_mask(g) → (M0, M1)
- rowmod_mask(g, k) → [M0, ..., M{k-1}]
- colmod_mask(g, k) → [M0, ..., M{k-1}]
- row_band_masks(g) → [B0, ..., Bn]
- col_band_masks(g) → [B0, ..., Bn]

All masks are 0/1 grids with shape(mask) == shape(g), forming disjoint partitions.

Invariants (per primary-anchor.md):
- Φ.1 (Finiteness): Bounded grids → finite signature space
- Φ.2 (Disjointness): Distinct signatures → disjoint pixel sets
- Φ.3 (Stability): Depends only on INPUT (never on target Y)
"""

from src.components import boundaries_by_any_change


def _validate_rectangular(g: list[list[int]]) -> None:
    """
    Validate that grid g is rectangular (all rows same length).

    Raises:
        ValueError: If g is ragged (rows have different lengths).

    Note: Empty grid [] is valid (0 rows).
    """
    if not g:
        return  # Empty grid is valid

    width = len(g[0])
    for i, row in enumerate(g):
        if len(row) != width:
            raise ValueError(
                f"Ragged grid: row {i} has length {len(row)}, expected {width}"
            )


def parity_mask(g: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """
    Compute parity partition: checkerboard pattern based on (r+c) mod 2.

    Returns two disjoint 0/1 masks (M0, M1) where:
    - M0[r][c] == 1 iff (r+c) mod 2 == 0 (even sum)
    - M1[r][c] == 1 iff (r+c) mod 2 == 1 (odd sum)

    Properties:
    - Disjoint: M0[r][c] + M1[r][c] == 1 for all (r,c)
    - Cover: M0 + M1 == ones(shape)
    - Input-only: depends on (r,c) indices, not grid values

    Args:
        g: Input grid (list of lists of ints)

    Returns:
        Tuple (M0, M1) of 0/1 masks with shape == shape(g)

    Raises:
        ValueError: If g is ragged

    Examples:
        >>> parity_mask([])
        ([], [])

        >>> parity_mask([[1]])
        ([[1]], [[0]])

        >>> parity_mask([[1, 2], [3, 4]])
        ([[1, 0], [0, 1]], [[0, 1], [1, 0]])
    """
    _validate_rectangular(g)

    if not g:
        return ([], [])

    h = len(g)
    w = len(g[0])

    # Initialize masks
    m0 = [[0] * w for _ in range(h)]
    m1 = [[0] * w for _ in range(h)]

    # Fill masks based on (r+c) mod 2
    for r in range(h):
        for c in range(w):
            if (r + c) % 2 == 0:
                m0[r][c] = 1
            else:
                m1[r][c] = 1

    return (m0, m1)


def rowmod_mask(g: list[list[int]], k: int) -> list[list[list[int]]]:
    """
    Compute row modulo partition: horizontal bands based on r mod k.

    Returns k disjoint 0/1 masks [M0, M1, ..., M{k-1}] where:
    - Mi[r][c] == 1 iff r mod k == i

    Properties:
    - Disjoint: only one Mi[r][c] == 1 for each (r,c)
    - Cover: ∑Mi == ones(shape)
    - Input-only: depends on row index r, not grid values

    Args:
        g: Input grid (list of lists of ints)
        k: Modulo value, must be in {2, 3}

    Returns:
        List of k masks [M0, M1, ..., M{k-1}], each with shape == shape(g)

    Raises:
        ValueError: If g is ragged or k not in {2, 3}

    Examples:
        >>> rowmod_mask([], 2)
        [[], []]

        >>> rowmod_mask([[1]], 2)
        [[[1]], [[0]]]

        >>> rowmod_mask([[1, 2], [3, 4], [5, 6]], 3)
        [[[1, 1], [0, 0], [0, 0]],
         [[0, 0], [1, 1], [0, 0]],
         [[0, 0], [0, 0], [1, 1]]]
    """
    _validate_rectangular(g)

    if k not in {2, 3}:
        raise ValueError(f"k must be in {{2, 3}}, got {k}")

    if not g:
        # Return k empty masks
        return [[] for _ in range(k)]

    h = len(g)
    w = len(g[0])

    # Initialize k masks
    masks = [[[0] * w for _ in range(h)] for _ in range(k)]

    # Fill masks based on r mod k
    for r in range(h):
        residue = r % k
        for c in range(w):
            masks[residue][r][c] = 1

    return masks


def colmod_mask(g: list[list[int]], k: int) -> list[list[list[int]]]:
    """
    Compute column modulo partition: vertical bands based on c mod k.

    Returns k disjoint 0/1 masks [M0, M1, ..., M{k-1}] where:
    - Mi[r][c] == 1 iff c mod k == i

    Properties:
    - Disjoint: only one Mi[r][c] == 1 for each (r,c)
    - Cover: ∑Mi == ones(shape)
    - Input-only: depends on column index c, not grid values

    Args:
        g: Input grid (list of lists of ints)
        k: Modulo value, must be in {2, 3}

    Returns:
        List of k masks [M0, M1, ..., M{k-1}], each with shape == shape(g)

    Raises:
        ValueError: If g is ragged or k not in {2, 3}

    Examples:
        >>> colmod_mask([], 2)
        [[], []]

        >>> colmod_mask([[1]], 2)
        [[[1]], [[0]]]

        >>> colmod_mask([[1, 2, 3], [4, 5, 6]], 3)
        [[[1, 0, 0], [1, 0, 0]],
         [[0, 1, 0], [0, 1, 0]],
         [[0, 0, 1], [0, 0, 1]]]
    """
    _validate_rectangular(g)

    if k not in {2, 3}:
        raise ValueError(f"k must be in {{2, 3}}, got {k}")

    if not g:
        # Return k empty masks
        return [[] for _ in range(k)]

    h = len(g)
    w = len(g[0])

    # Initialize k masks
    masks = [[[0] * w for _ in range(h)] for _ in range(k)]

    # Fill masks based on c mod k
    for r in range(h):
        for c in range(w):
            residue = c % k
            masks[residue][r][c] = 1

    return masks


def row_band_masks(g: list[list[int]]) -> list[list[list[int]]]:
    """
    Compute row band partition: horizontal bands based on content-change boundaries.

    Uses Non-Periodic Segmentation (NPS) to detect where adjacent rows differ,
    then partitions grid into contiguous horizontal bands split at those boundaries.

    Returns n disjoint 0/1 masks [B0, B1, ..., B{n-1}] where:
    - Bi[r][c] == 1 iff row r belongs to band i
    - Bands are contiguous row ranges

    Properties:
    - Disjoint: only one Bi[r][c] == 1 for each (r,c)
    - Cover: ∑Bi == ones(shape)
    - Input-only: depends on grid structure via boundaries_by_any_change

    Args:
        g: Input grid (list of lists of ints)

    Returns:
        List of n masks [B0, ..., B{n-1}], each with shape == shape(g)
        - n = number of contiguous bands (≥1 if grid non-empty)
        - Empty grid → [] (no bands)

    Raises:
        ValueError: If g is ragged

    Examples:
        >>> row_band_masks([])
        []

        >>> row_band_masks([[1, 2]])
        [[[1, 1]]]

        >>> row_band_masks([[1, 2], [1, 2], [3, 4]])
        [[[1, 1], [1, 1], [0, 0]], [[0, 0], [0, 0], [1, 1]]]
        # Band 0: rows 0-1 (identical), Band 1: row 2 (different)
    """
    _validate_rectangular(g)

    if not g:
        return []

    h = len(g)
    w = len(g[0])

    # Get change boundaries from components.py
    change_boundaries = boundaries_by_any_change(g, "row")

    # Convert to full boundaries list: [0] + [b+1 for b in changes] + [h]
    # This creates inclusive-exclusive ranges
    full_boundaries = [0] + [b + 1 for b in change_boundaries] + [h]

    # Number of bands = len(full_boundaries) - 1
    num_bands = len(full_boundaries) - 1

    # Create masks for each band
    masks = []
    for band_idx in range(num_bands):
        # Band spans rows [start, end)
        start_row = full_boundaries[band_idx]
        end_row = full_boundaries[band_idx + 1]

        # Create mask for this band
        mask = [[0] * w for _ in range(h)]
        for r in range(start_row, end_row):
            for c in range(w):
                mask[r][c] = 1

        masks.append(mask)

    return masks


def col_band_masks(g: list[list[int]]) -> list[list[list[int]]]:
    """
    Compute column band partition: vertical bands based on content-change boundaries.

    Uses Non-Periodic Segmentation (NPS) to detect where adjacent columns differ,
    then partitions grid into contiguous vertical bands split at those boundaries.

    Returns n disjoint 0/1 masks [B0, B1, ..., B{n-1}] where:
    - Bi[r][c] == 1 iff column c belongs to band i
    - Bands are contiguous column ranges

    Properties:
    - Disjoint: only one Bi[r][c] == 1 for each (r,c)
    - Cover: ∑Bi == ones(shape)
    - Input-only: depends on grid structure via boundaries_by_any_change

    Args:
        g: Input grid (list of lists of ints)

    Returns:
        List of n masks [B0, ..., B{n-1}], each with shape == shape(g)
        - n = number of contiguous bands (≥1 if grid non-empty)
        - Empty grid → [] (no bands)

    Raises:
        ValueError: If g is ragged

    Examples:
        >>> col_band_masks([])
        []

        >>> col_band_masks([[1], [2]])
        [[[1], [1]]]

        >>> col_band_masks([[1, 1, 3], [2, 2, 4]])
        [[[1, 1, 0], [1, 1, 0]], [[0, 0, 1], [0, 0, 1]]]
        # Band 0: cols 0-1 (identical), Band 1: col 2 (different)
    """
    _validate_rectangular(g)

    if not g:
        return []

    h = len(g)
    w = len(g[0])

    # Get change boundaries from components.py
    change_boundaries = boundaries_by_any_change(g, "col")

    # Convert to full boundaries list: [0] + [b+1 for b in changes] + [w]
    # This creates inclusive-exclusive ranges
    full_boundaries = [0] + [b + 1 for b in change_boundaries] + [w]

    # Number of bands = len(full_boundaries) - 1
    num_bands = len(full_boundaries) - 1

    # Create masks for each band
    masks = []
    for band_idx in range(num_bands):
        # Band spans columns [start, end)
        start_col = full_boundaries[band_idx]
        end_col = full_boundaries[band_idx + 1]

        # Create mask for this band
        mask = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(start_col, end_col):
                mask[r][c] = 1

        masks.append(mask)

    return masks
