"""
Signature Builders (Φ Features) for ARC AGI Pure Math Solver.

This module implements the canonical signature system used to partition grids
into disjoint equivalence classes. All features are INPUT-ONLY (Φ.3 stability)
and form the finite Boolean basis for Step-2 (P+Φ/GLUE) solver.

Phase 4 Work Orders:
- P4-01: Index Predicates (parity_mask, rowmod_mask, colmod_mask)
- P4-02: NPS Bands (row_band_masks, col_band_masks)
- P4-03: Local Content (is_color_mask, touching_color_mask)
- P4-04: Component IDs (component_id_table)
- P4-05: Patch Canonicalizer Core (patch_canonical_key, patch_canonical_rep, is_valid_patch_size)
- P4-06: Patchkey Tables (patchkey_table)

Public API:
- parity_mask(g) → (M0, M1)
- rowmod_mask(g, k) → [M0, ..., M{k-1}]
- colmod_mask(g, k) → [M0, ..., M{k-1}]
- row_band_masks(g) → [B0, ..., Bn]
- col_band_masks(g) → [B0, ..., Bn]
- is_color_mask(g, color) → M (0/1 mask)
- touching_color_mask(g, color) → T (0/1 mask, 4-neighbor dilation)
- component_id_table(g) → (id_grid, meta) (8-connected, deterministic IDs)
- is_valid_patch_size(n) → bool (True iff n is odd and >= 1)
- patch_canonical_key(p) → (R, C, values_tuple) (OFA + D8 minimal key)
- patch_canonical_rep(p) → canonical patch (OFA + D8 representative)
- patchkey_table(g, r) → table of canonical keys or None (r ∈ {2,3,4})

All masks are 0/1 grids with shape(mask) == shape(g), forming disjoint partitions.

Invariants (per primary-anchor.md):
- Φ.1 (Finiteness): Bounded grids → finite signature space
- Φ.2 (Disjointness): Distinct signatures → disjoint pixel sets
- Φ.3 (Stability): Depends only on INPUT (never on target Y)
"""

from src.components import boundaries_by_any_change, components_by_color
from src.canonicalization import (
    ofa_normalize_patch_colors,
    apply_isometry,
    all_isometries,
)


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


def is_color_mask(g: list[list[int]], color: int) -> list[list[int]]:
    """
    Compute color mask: pixels matching a specific color.

    Returns a 0/1 mask M where:
    - M[r][c] == 1 iff g[r][c] == color
    - M[r][c] == 0 otherwise

    Properties:
    - Input-only: depends on grid values (Φ.3 stability)
    - Shape: shape(M) == shape(g)
    - Color domain: color must be in [0..9] (ARC color range)

    Args:
        g: Input grid (list of lists of ints)
        color: Color value to match (must be in 0..9)

    Returns:
        0/1 mask with shape == shape(g), where 1 indicates color match

    Raises:
        ValueError: If g is ragged or color not in [0..9]

    Examples:
        >>> is_color_mask([], 5)
        []

        >>> is_color_mask([[1]], 1)
        [[1]]

        >>> is_color_mask([[1]], 2)
        [[0]]

        >>> is_color_mask([[1, 2], [2, 1]], 2)
        [[0, 1], [1, 0]]
    """
    _validate_rectangular(g)

    if color < 0 or color > 9:
        raise ValueError(f"Color must be in [0..9], got {color}")

    if not g:
        return []

    h = len(g)
    w = len(g[0])

    # Create mask: 1 where g[r][c] == color, else 0
    mask = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] == color:
                mask[r][c] = 1

    return mask


def touching_color_mask(g: list[list[int]], color: int) -> list[list[int]]:
    """
    Compute touching mask: pixels that are 4-neighbors of a color (but not that color).

    Returns a 0/1 mask T where:
    - T[r][c] == 1 iff:
        - At least one 4-neighbor (r±1,c) or (r,c±1) has g[neighbor] == color
        - AND g[r][c] != color (exclude the color cells themselves)
    - T[r][c] == 0 otherwise

    This is a single-step 4-neighbor dilation of the color set, excluding the color itself.

    Properties:
    - Input-only: depends on grid values and structure (Φ.3 stability)
    - 4-neighbors only: {(r-1,c), (r+1,c), (r,c-1), (r,c+1)} - NO diagonals
    - Single-step: immediate neighbors only, not multi-hop
    - Disjoint from is_color_mask: T & M have no overlap
    - Boundary handling: out-of-bounds neighbors ignored (no wrap-around)

    Args:
        g: Input grid (list of lists of ints)
        color: Color value to check neighbors for (must be in 0..9)

    Returns:
        0/1 mask with shape == shape(g), where 1 indicates touching the color

    Raises:
        ValueError: If g is ragged or color not in [0..9]

    Examples:
        >>> touching_color_mask([], 5)
        []

        >>> touching_color_mask([[1]], 1)
        [[0]]

        >>> touching_color_mask([[5, 0], [0, 0]], 5)
        [[0, 1], [1, 0]]
        # Cell (0,1) and (1,0) are 4-neighbors of (0,0) which has color 5

        >>> touching_color_mask([[0, 5, 0]], 5)
        [[1, 0, 1]]
        # Cells (0,0) and (0,2) touch (0,1) which has color 5

        >>> touching_color_mask([[5, 5], [5, 5]], 5)
        [[0, 0], [0, 0]]
        # All cells ARE color 5, so no touching (touching excludes color cells)
    """
    _validate_rectangular(g)

    if color < 0 or color > 9:
        raise ValueError(f"Color must be in [0..9], got {color}")

    if not g:
        return []

    h = len(g)
    w = len(g[0])

    # Create touching mask
    mask = [[0] * w for _ in range(h)]

    # For each pixel, check if any 4-neighbor has the target color
    for r in range(h):
        for c in range(w):
            # Exclude cells that are already the target color
            if g[r][c] == color:
                mask[r][c] = 0
                continue

            # Check 4-neighbors: (r-1,c), (r+1,c), (r,c-1), (r,c+1)
            neighbors = [
                (r - 1, c),  # up
                (r + 1, c),  # down
                (r, c - 1),  # left
                (r, c + 1),  # right
            ]

            # If any valid neighbor has the target color, mark as touching
            touching = False
            for nr, nc in neighbors:
                # Check bounds
                if 0 <= nr < h and 0 <= nc < w:
                    if g[nr][nc] == color:
                        touching = True
                        break

            mask[r][c] = 1 if touching else 0

    return mask


def component_id_table(
    g: list[list[int]],
) -> tuple[list[list[int]], list[dict]]:
    """
    Assign stable, deterministic IDs to all 8-connected components in grid.

    Returns (id_grid, meta) where:
    - id_grid[r][c] = component ID for pixel (r,c)
    - meta[id] = {"color": int, "size": int, "bbox": tuple, "seed_rc": tuple}

    IDs are assigned globally across all colors (0..K-1 for K components) using
    deterministic 3-level tie-breaking:
    1. Primary: Larger components first (sort by -size)
    2. Secondary: If sizes equal, sort by bbox (r0,c0,r1,c1) lexicographically
    3. Tertiary: If sizes and bboxes equal, sort by seed_rc (first cell row-major)

    Properties:
    - Input-only: depends on grid values and structure (Φ.3 stability)
    - 8-connected: includes diagonal adjacency {(±1,0), (0,±1), (±1,±1)}
    - Global ID space: all components share 0..K-1 namespace (not per-color)
    - Deterministic: stable ordering under identical inputs

    Args:
        g: Input grid (list of lists of ints)

    Returns:
        Tuple (id_grid, meta) where:
        - id_grid: same shape as g, each pixel has its component ID
        - meta: list[dict] where meta[id] contains component metadata:
            - "color": color value of component (0-9)
            - "size": number of pixels in component
            - "bbox": (r0, c0, r1, c1) inclusive bounding box
            - "seed_rc": (r, c) lexicographically first cell in row-major order

    Raises:
        ValueError: If g is ragged

    Examples:
        >>> component_id_table([])
        ([], [])

        >>> component_id_table([[5]])
        ([[0]], [{"color": 5, "size": 1, "bbox": (0,0,0,0), "seed_rc": (0,0)}])

        >>> component_id_table([[3, 3], [3, 3]])
        ([[0, 0], [0, 0]], [{"color": 3, "size": 4, "bbox": (0,0,1,1), "seed_rc": (0,0)}])

        >>> component_id_table([[1, 1, 1], [2, 2, 0]])
        # Returns:
        # id_grid: [[0, 0, 0], [1, 1, 2]]
        # meta: [
        #   {"color": 1, "size": 3, "bbox": (0,0,0,2), "seed_rc": (0,0)},  # ID 0
        #   {"color": 2, "size": 2, "bbox": (1,0,1,1), "seed_rc": (1,0)},  # ID 1
        #   {"color": 0, "size": 1, "bbox": (1,2,1,2), "seed_rc": (1,2)}   # ID 2
        # ]
    """
    _validate_rectangular(g)

    if not g:
        return ([], [])

    h = len(g)
    w = len(g[0])

    # Get all 8-connected components grouped by color
    comp_dict = components_by_color(g)

    # Flatten all components with metadata
    all_components = []
    for color, comp_list in comp_dict.items():
        for comp in comp_list:
            # Each comp is a dict with "cells", "bbox", "color", "id"
            cells = comp["cells"]
            size = len(cells)
            bbox = comp["bbox"]

            # Find seed_rc: lexicographically first cell in row-major order
            # cells is already sorted row-major, so first element is seed
            seed_rc = cells[0]

            all_components.append(
                {
                    "color": color,
                    "size": size,
                    "bbox": bbox,
                    "seed_rc": seed_rc,
                    "pixels": cells,
                }
            )

    # Sort by (-size, bbox, seed_rc) for deterministic ID assignment
    # Larger components first, then bbox lex, then seed_rc lex
    all_components.sort(key=lambda c: (-c["size"], c["bbox"], c["seed_rc"]))

    # Assign IDs and build outputs
    id_grid = [[0] * w for _ in range(h)]
    meta = []

    for comp_id, comp in enumerate(all_components):
        # Fill id_grid with this component's ID
        for r, c in comp["pixels"]:
            id_grid[r][c] = comp_id

        # Build metadata entry (without pixels list)
        meta.append(
            {
                "color": comp["color"],
                "size": comp["size"],
                "bbox": comp["bbox"],
                "seed_rc": comp["seed_rc"],
            }
        )

    return (id_grid, meta)


def is_valid_patch_size(n: int) -> bool:
    """
    Check if n is a valid patch size (odd, positive).

    Patch sizes must be odd positive integers (1, 3, 5, 7, 9, ...).
    This ensures patches have a well-defined center pixel.

    Args:
        n: Patch size to validate

    Returns:
        True if n is odd and >= 1, False otherwise

    Examples:
        >>> is_valid_patch_size(1)
        True

        >>> is_valid_patch_size(5)
        True

        >>> is_valid_patch_size(0)
        False

        >>> is_valid_patch_size(2)
        False

        >>> is_valid_patch_size(-1)
        False
    """
    return isinstance(n, int) and n >= 1 and n % 2 == 1


def _grid_to_key(g: list[list[int]]) -> tuple[int, int, tuple[int, ...]]:
    """
    Helper: Convert grid to canonical key format (R, C, values_tuple).

    Args:
        g: Grid (list of lists of ints)

    Returns:
        Tuple (R, C, values_tuple) where:
        - R = number of rows
        - C = number of columns
        - values_tuple = flattened row-major values as tuple
    """
    if not g:
        return (0, 0, ())

    R = len(g)
    C = len(g[0]) if g else 0

    # Flatten row-major
    values = []
    for row in g:
        values.extend(row)

    return (R, C, tuple(values))


def patch_canonical_key(p: list[list[int]]) -> tuple[int, int, tuple[int, ...]]:
    """
    Compute canonical key for patch under OFA + D8.

    Returns (R, C, values_tuple) of the OFA-normalized, D8-minimal image,
    where the key is chosen by:
    1. Apply OFA (Order of First Appearance) normalization locally to patch
    2. Try all 8 D8 isometries (identity, rotations, flips)
    3. Convert each to (R, C, values_tuple) format
    4. Return lexicographically minimal key (shape-first ordering)
    5. Tie-break by earliest σ from all_isometries() order

    Properties:
    - Input-only: depends on patch structure (Φ.3 stability)
    - OFA locality: palette permutations produce identical keys
    - D8 minimality: key is minimal over all 8 isometries
    - Deterministic: stable ordering, tie-break by σ order
    - Purity: input patch unchanged

    Args:
        p: Input patch (list of lists of ints), must be square

    Returns:
        Tuple (R, C, values_tuple) representing the canonical key

    Raises:
        ValueError: If p is empty, non-square, or ragged

    Examples:
        >>> patch_canonical_key([[5]])
        (1, 1, (0,))

        >>> patch_canonical_key([[3, 3], [3, 3]])
        (2, 2, (0, 0, 0, 0))
    """
    _validate_rectangular(p)

    if not p:
        raise ValueError("Patch cannot be empty")

    h = len(p)
    w = len(p[0])

    if h != w:
        raise ValueError(f"Patch must be square, got {h}x{w}")

    # Apply OFA normalization locally
    ofa_patch = ofa_normalize_patch_colors(p)

    # Try all D8 isometries and find minimal key
    min_key = None

    for sigma in all_isometries():
        # Apply isometry
        transformed = apply_isometry(ofa_patch, sigma)

        # Convert to key format
        key = _grid_to_key(transformed)

        # Track minimum (tuple comparison is lexicographic)
        # Use < (not <=) for tie-breaking by earliest σ
        if min_key is None or key < min_key:
            min_key = key

    return min_key


def patch_canonical_rep(p: list[list[int]]) -> list[list[int]]:
    """
    Return OFA-normalized canonical representative achieving patch_canonical_key(p).

    Returns the actual patch (2D grid) that produces the minimal canonical key.
    This is the OFA-normalized patch transformed by the D8 isometry σ that
    achieves the lexicographically minimal key.

    Properties:
    - Input-only: depends on patch structure (Φ.3 stability)
    - OFA locality: palette permutations produce identical reps
    - D8 minimality: rep achieves minimal key over all 8 isometries
    - Idempotence: patch_canonical_rep(patch_canonical_rep(p)) == patch_canonical_rep(p)
    - Purity: input patch unchanged (newly allocated result)

    Args:
        p: Input patch (list of lists of ints), must be square

    Returns:
        Canonical representative patch (2D grid, newly allocated)

    Raises:
        ValueError: If p is empty, non-square, or ragged

    Examples:
        >>> patch_canonical_rep([[5]])
        [[0]]

        >>> patch_canonical_rep([[3, 3], [3, 3]])
        [[0, 0], [0, 0]]
    """
    _validate_rectangular(p)

    if not p:
        raise ValueError("Patch cannot be empty")

    h = len(p)
    w = len(p[0])

    if h != w:
        raise ValueError(f"Patch must be square, got {h}x{w}")

    # Apply OFA normalization locally
    ofa_patch = ofa_normalize_patch_colors(p)

    # Try all D8 isometries and find the one giving minimal key
    min_key = None
    min_rep = None

    for sigma in all_isometries():
        # Apply isometry
        transformed = apply_isometry(ofa_patch, sigma)

        # Convert to key format
        key = _grid_to_key(transformed)

        # Track minimum
        # Use < (not <=) for tie-breaking by earliest σ
        if min_key is None or key < min_key:
            min_key = key
            min_rep = transformed

    return min_rep


def patchkey_table(g: list[list[int]], r: int) -> list[list[object]]:
    """
    Compute per-pixel canonical patch keys for radius r.

    For each pixel (i,j) in the grid, if a full (2r+1)×(2r+1) window can be
    centered at that pixel, compute the canonical patch key for that window.
    Otherwise, store None (for border pixels where window doesn't fit).

    Window sizes:
    - r=2: 5×5 window
    - r=3: 7×7 window
    - r=4: 9×9 window

    Valid centers for grid of size R×C:
    - Row indices: i ∈ [r, R-1-r] (inclusive)
    - Col indices: j ∈ [r, C-1-r] (inclusive)

    Properties:
    - Input-only: depends on grid structure (Φ.3 stability)
    - OFA locality: palette permutations → identical keys at valid centers
    - Deterministic: stable ordering via patch_canonical_key
    - Purity: input grid unchanged (newly allocated table)
    - No padding/wrapping: border pixels get None

    Args:
        g: Input grid (list of lists of ints), must be rectangular
        r: Radius, must be in {2, 3, 4}

    Returns:
        Table of same shape as g, where each entry is either:
        - (R, C, values_tuple): canonical key if window fits
        - None: if window doesn't fit (border pixels)

    Raises:
        ValueError: If g is ragged or r not in {2, 3, 4}

    Examples:
        >>> patchkey_table([], 2)
        []

        >>> # Too small grid (2×2, needs 5×5)
        >>> patchkey_table([[1, 2], [3, 4]], 2)
        [[None, None], [None, None]]

        >>> # Exact fit (5×5 grid, r=2)
        >>> g = [[i for i in range(5)] for _ in range(5)]
        >>> table = patchkey_table(g, 2)
        >>> # Only center (2,2) has valid window, others are None
        >>> table[2][2] is not None
        True
        >>> table[0][0] is None
        True
    """
    _validate_rectangular(g)

    if r not in {2, 3, 4}:
        raise ValueError(f"Radius must be in {{2, 3, 4}}, got {r}")

    if not g:
        return []

    R = len(g)
    C = len(g[0])

    # Initialize table with None
    table = [[None for _ in range(C)] for _ in range(R)]

    # Compute keys for valid centers only
    # Valid row indices: [r, R-1-r] (inclusive)
    # Valid col indices: [r, C-1-r] (inclusive)
    for i in range(r, R - r):
        for j in range(r, C - r):
            # Extract window centered at (i, j)
            # Window covers [i-r:i+r+1, j-r:j+r+1] (inclusive on both ends)
            window = [row[j - r : j + r + 1] for row in g[i - r : i + r + 1]]

            # Compute canonical key
            key = patch_canonical_key(window)

            # Store in table
            table[i][j] = key

    return table


# ============================================================================
# Pair-Invariant Feature Helpers (Bug B2 Fix)
# ============================================================================


def nps_boundary_masks(
    band_masks: list[list[list[int]]], axis: str
) -> list[list[int]]:
    """
    Compute boundary mask from NPS band masks (pair-invariant).

    A pixel is on a boundary if it's at the edge between two bands.
    For row bands: boundary is where adjacent rows belong to different bands.
    For col bands: boundary is where adjacent cols belong to different bands.

    Args:
        band_masks: List of band masks from row_band_masks or col_band_masks
        axis: "row" or "col" indicating which axis to check

    Returns:
        Binary mask: 1 if pixel is on boundary, 0 otherwise

    Examples:
        >>> # Two row bands: rows 0-1 (band 0), rows 2-3 (band 1)
        >>> bands = [[[1,1],[1,1],[0,0],[0,0]], [[0,0],[0,0],[1,1],[1,1]]]
        >>> nps_boundary_masks(bands, "row")
        [[0, 0], [1, 1], [1, 1], [0, 0]]  # Rows 1 and 2 are on boundary
    """
    if not band_masks:
        return []

    h = len(band_masks[0])
    w = len(band_masks[0][0]) if band_masks[0] else 0

    if h == 0 or w == 0:
        return []

    # Create boundary mask
    boundary = [[0] * w for _ in range(h)]

    if axis == "row":
        # Check row boundaries (adjacent rows in different bands)
        for r in range(h):
            for c in range(w):
                # Check if row r is at boundary with row r-1 or r+1
                is_boundary = False

                if r > 0:
                    # Check if (r, c) and (r-1, c) are in different bands
                    for band_mask in band_masks:
                        if band_mask[r][c] != band_mask[r - 1][c]:
                            is_boundary = True
                            break

                if r < h - 1 and not is_boundary:
                    # Check if (r, c) and (r+1, c) are in different bands
                    for band_mask in band_masks:
                        if band_mask[r][c] != band_mask[r + 1][c]:
                            is_boundary = True
                            break

                boundary[r][c] = 1 if is_boundary else 0

    else:  # axis == "col"
        # Check col boundaries (adjacent cols in different bands)
        for r in range(h):
            for c in range(w):
                # Check if col c is at boundary with col c-1 or c+1
                is_boundary = False

                if c > 0:
                    # Check if (r, c) and (r, c-1) are in different bands
                    for band_mask in band_masks:
                        if band_mask[r][c] != band_mask[r][c - 1]:
                            is_boundary = True
                            break

                if c < w - 1 and not is_boundary:
                    # Check if (r, c) and (r, c+1) are in different bands
                    for band_mask in band_masks:
                        if band_mask[r][c] != band_mask[r][c + 1]:
                            is_boundary = True
                            break

                boundary[r][c] = 1 if is_boundary else 0

    return boundary


def nps_offset_bucket_table(
    band_masks: list[list[list[int]]], axis: str
) -> list[list[int]]:
    """
    Compute normalized offset within band (pair-invariant).

    Returns bucket indicating position within the pixel's band:
    - 0: Start third of band (0-33%)
    - 1: Middle third of band (33-66%)
    - 2: End third of band (66-100%)

    Args:
        band_masks: List of band masks from row_band_masks or col_band_masks
        axis: "row" or "col" indicating which axis

    Returns:
        Grid of bucket values (0, 1, or 2)

    Examples:
        >>> # Row band spanning rows 0-5 (6 rows)
        >>> # Row 0-1: bucket 0, rows 2-3: bucket 1, rows 4-5: bucket 2
    """
    if not band_masks:
        return []

    h = len(band_masks[0])
    w = len(band_masks[0][0]) if band_masks[0] else 0

    if h == 0 or w == 0:
        return []

    # Create offset table
    offset_table = [[0] * w for _ in range(h)]

    # For each band, compute offset for its pixels
    for band_idx, band_mask in enumerate(band_masks):
        if axis == "row":
            # Find row range for this band
            rows_in_band = set()
            for r in range(h):
                if any(band_mask[r][c] == 1 for c in range(w)):
                    rows_in_band.add(r)

            if rows_in_band:
                min_row = min(rows_in_band)
                max_row = max(rows_in_band)
                band_height = max_row - min_row + 1

                # Assign buckets based on position in band
                for r in range(h):
                    for c in range(w):
                        if band_mask[r][c] == 1:
                            # Normalize position within band
                            offset = r - min_row
                            if band_height == 1:
                                bucket = 1  # Single row = middle
                            elif offset < band_height / 3:
                                bucket = 0  # Start third
                            elif offset < 2 * band_height / 3:
                                bucket = 1  # Middle third
                            else:
                                bucket = 2  # End third
                            offset_table[r][c] = bucket

        else:  # axis == "col"
            # Find col range for this band
            cols_in_band = set()
            for c in range(w):
                if any(band_mask[r][c] == 1 for r in range(h)):
                    cols_in_band.add(c)

            if cols_in_band:
                min_col = min(cols_in_band)
                max_col = max(cols_in_band)
                band_width = max_col - min_col + 1

                # Assign buckets based on position in band
                for r in range(h):
                    for c in range(w):
                        if band_mask[r][c] == 1:
                            # Normalize position within band
                            offset = c - min_col
                            if band_width == 1:
                                bucket = 1  # Single col = middle
                            elif offset < band_width / 3:
                                bucket = 0  # Start third
                            elif offset < 2 * band_width / 3:
                                bucket = 1  # Middle third
                            else:
                                bucket = 2  # End third
                            offset_table[r][c] = bucket

    return offset_table


def component_largest_mask(
    id_grid: list[list[int]], meta: list[dict]
) -> list[list[int]]:
    """
    Compute mask of pixels belonging to the largest component (pair-invariant).

    Returns binary mask: 1 if pixel is in the largest component, 0 otherwise.

    Args:
        id_grid: Component ID grid from component_id_table
        meta: Component metadata list

    Returns:
        Binary mask indicating largest component

    Examples:
        >>> # Component 0 has size 100 (largest), component 1 has size 50
        >>> # Pixels with id_grid[r][c] == 0 get mask value 1
    """
    if not id_grid or not meta:
        return []

    h = len(id_grid)
    w = len(id_grid[0]) if id_grid else 0

    if h == 0 or w == 0:
        return [[]]

    # Find largest component(s)
    max_size = max(comp["size"] for comp in meta) if meta else 0

    # Create mask
    mask = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            comp_id = id_grid[r][c]
            if comp_id < len(meta) and meta[comp_id]["size"] == max_size:
                mask[r][c] = 1

    return mask


def component_size_bucket_table(
    id_grid: list[list[int]], meta: list[dict]
) -> list[list[int]]:
    """
    Compute component size buckets (pair-invariant).

    Returns bucket indicating component size:
    - 0: Tiny (≤4 pixels)
    - 1: Small (5-8 pixels)
    - 2: Medium (9-16 pixels)
    - 3: Large (>16 pixels)

    Args:
        id_grid: Component ID grid from component_id_table
        meta: Component metadata list

    Returns:
        Grid of size bucket values (0, 1, 2, or 3)
    """
    if not id_grid or not meta:
        return []

    h = len(id_grid)
    w = len(id_grid[0]) if id_grid else 0

    if h == 0 or w == 0:
        return [[]]

    # Create size bucket table
    size_table = [[0] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            comp_id = id_grid[r][c]
            if comp_id < len(meta):
                size = meta[comp_id]["size"]

                # Assign bucket
                if size <= 4:
                    bucket = 0
                elif size <= 8:
                    bucket = 1
                elif size <= 16:
                    bucket = 2
                else:
                    bucket = 3

                size_table[r][c] = bucket

    return size_table


def component_aspect_bucket_table(
    id_grid: list[list[int]], meta: list[dict]
) -> list[list[int]]:
    """
    Compute component aspect ratio buckets (pair-invariant).

    Returns bucket indicating component shape:
    - 0: Tall (height > width * 1.5)
    - 1: Square (within 1.5 ratio)
    - 2: Wide (width > height * 1.5)

    Args:
        id_grid: Component ID grid from component_id_table
        meta: Component metadata list

    Returns:
        Grid of aspect bucket values (0, 1, or 2)
    """
    if not id_grid or not meta:
        return []

    h = len(id_grid)
    w = len(id_grid[0]) if id_grid else 0

    if h == 0 or w == 0:
        return [[]]

    # Create aspect bucket table
    aspect_table = [[1] * w for _ in range(h)]  # Default to square

    for r in range(h):
        for c in range(w):
            comp_id = id_grid[r][c]
            if comp_id < len(meta):
                bbox = meta[comp_id]["bbox"]
                r0, c0, r1, c1 = bbox

                height = r1 - r0 + 1
                width = c1 - c0 + 1

                # Assign bucket based on aspect ratio
                if height > width * 1.5:
                    bucket = 0  # Tall
                elif width > height * 1.5:
                    bucket = 2  # Wide
                else:
                    bucket = 1  # Square

                aspect_table[r][c] = bucket

    return aspect_table


def phi_signature_tables(X: list[list[int]]) -> dict:
    """
    Aggregate ALL Φ features (P4-01 through P4-06) into single dict.

    Returns a comprehensive signature dict containing:
    - index: Parity and modulo predicates (row/col mod k for k∈{2,3})
    - nps: Non-periodic segmentation bands (row/col)
    - local: Color masks and touching masks for ALL colors 0-9
    - components: Component ID table with metadata
    - patchkeys: Canonical patch keys for radii r∈{2,3,4}

    Properties:
    - Φ.3 (Stability): Input-only, no Y dependencies
    - Fixed key order: deterministic JSON serialization
    - ALL colors 0-9: is_color and touching_color exist even if absent from X
    - Shape consistency: all masks/tables have shape(X)
    - Empty grid handling: returns minimal valid structure

    Args:
        X: Input grid (list of lists of ints)

    Returns:
        Dict with fixed key order ["index", "nps", "local", "components", "patchkeys"]:
        {
            "index": {
                "parity": {"M0": M0, "M1": M1},
                "rowmod": {"k2": [M0, M1], "k3": [M0, M1, M2]},
                "colmod": {"k2": [M0, M1], "k3": [M0, M1, M2]}
            },
            "nps": {
                "row_bands": [B0, ..., Bn],
                "col_bands": [B0, ..., Bn]
            },
            "local": {
                "is_color": {
                    0: M0, 1: M1, ..., 9: M9
                },
                "touching_color": {
                    0: T0, 1: T1, ..., 9: T9
                }
            },
            "components": {
                "id_grid": [[id, ...], ...],
                "meta": [{"color": c, "size": s, "bbox": b, "seed_rc": rc}, ...]
            },
            "patchkeys": {
                "r2": [[key or None, ...], ...],
                "r3": [[key or None, ...], ...],
                "r4": [[key or None, ...], ...]
            }
        }

    Raises:
        ValueError: If X is ragged

    Examples:
        >>> phi_signature_tables([])
        {
            "index": {...},
            "nps": {"row_bands": [], "col_bands": []},
            "local": {"is_color": {...}, "touching_color": {...}},
            "components": {"id_grid": [], "meta": []},
            "patchkeys": {"r2": [], "r3": [], "r4": []}
        }

        >>> result = phi_signature_tables([[5, 5], [5, 5]])
        >>> result["local"]["is_color"][5]
        [[1, 1], [1, 1]]
        >>> result["local"]["is_color"][3]
        [[0, 0], [0, 0]]
    """
    _validate_rectangular(X)

    # P4-01: Index Predicates
    parity_m0, parity_m1 = parity_mask(X)
    rowmod_k2 = rowmod_mask(X, 2)
    rowmod_k3 = rowmod_mask(X, 3)
    colmod_k2 = colmod_mask(X, 2)
    colmod_k3 = colmod_mask(X, 3)

    # P4-02: NPS Bands
    row_bands = row_band_masks(X)
    col_bands = col_band_masks(X)

    # P4-03: Local Content - ALL colors 0-9 (even if absent)
    is_color_dict = {}
    touching_color_dict = {}
    for color in range(10):
        is_color_dict[color] = is_color_mask(X, color)
        touching_color_dict[color] = touching_color_mask(X, color)

    # P4-04: Component IDs
    id_grid, meta = component_id_table(X)

    # P4-06: Patchkey Tables
    patchkey_r2 = patchkey_table(X, 2)
    patchkey_r3 = patchkey_table(X, 3)
    patchkey_r4 = patchkey_table(X, 4)

    # Bug B2 Fix: Compute pair-invariant NPS and component features
    # These replace raw band IDs and component IDs in signatures

    # NPS pair-invariant features
    row_boundary = nps_boundary_masks(row_bands, "row")
    col_boundary = nps_boundary_masks(col_bands, "col")
    row_offset = nps_offset_bucket_table(row_bands, "row")
    col_offset = nps_offset_bucket_table(col_bands, "col")

    # Component pair-invariant features
    largest_comp = component_largest_mask(id_grid, meta)
    comp_size = component_size_bucket_table(id_grid, meta)
    comp_aspect = component_aspect_bucket_table(id_grid, meta)

    # Assemble with FIXED KEY ORDER (for deterministic JSON)
    result = {
        "index": {
            "parity": {"M0": parity_m0, "M1": parity_m1},
            "rowmod": {"k2": rowmod_k2, "k3": rowmod_k3},
            "colmod": {"k2": colmod_k2, "k3": colmod_k3},
        },
        "nps": {
            "row_bands": row_bands,
            "col_bands": col_bands,
            # Pair-invariant features (Bug B2 fix)
            "row_boundary": row_boundary,
            "col_boundary": col_boundary,
            "row_offset": row_offset,
            "col_offset": col_offset,
        },
        "local": {
            "is_color": is_color_dict,
            "touching_color": touching_color_dict,
        },
        "components": {
            "id_grid": id_grid,
            "meta": meta,
            # Pair-invariant features (Bug B2 fix)
            "largest_comp": largest_comp,
            "comp_size": comp_size,
            "comp_aspect": comp_aspect,
        },
        "patchkeys": {
            "r2": patchkey_r2,
            "r3": patchkey_r3,
            "r4": patchkey_r4,
        },
    }

    return result
