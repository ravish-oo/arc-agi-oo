"""
D8 isometry registry and application for ARC AGI solver.

Defines the complete D8 group (dihedral group of the square) with 8 isometries:
- 4 rotations: id, rot90, rot180, rot270
- 4 reflections: flip_h, flip_v, transpose, flip_anti

All functions are pure (no mutation) and deterministic.
"""

from src.utils import transpose, rot90, rot180, rot270, flip_h, flip_v, copy_grid


# Fixed, deterministic order of D8 isometries (NEVER change this order)
ISOMETRIES = [
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "flip_anti"
]


def all_isometries() -> list[str]:
    """
    Return the eight isometry names in a fixed, deterministic order.

    Returns:
        List of 8 isometry names: ["id", "rot90", "rot180", "rot270",
                                    "flip_h", "flip_v", "transpose", "flip_anti"]

    Invariants:
        - Order is FIXED and deterministic
        - len() == 8
        - No duplicates
    """
    return ISOMETRIES


def flip_anti(g: list[list[int]]) -> list[list[int]]:
    """
    Anti-diagonal reflection (swap along bottom-left to top-right diagonal).

    Maps (r, c) → (W-1-c, H-1-r) for H×W grid.
    Result dimensions: W×H (swapped like transpose).

    This is equivalent to: transpose ∘ rot180, or rot180 ∘ transpose.

    Args:
        g: Input grid (H×W)

    Returns:
        Reflected grid (W×H)

    Raises:
        ValueError: If grid is ragged
    """
    if not g:
        return []

    # Validate rectangularity
    H = len(g)
    W = len(g[0])
    for row in g:
        if len(row) != W:
            raise ValueError("Grid must be rectangular (all rows same length)")

    # Create W×H output (dimensions swapped)
    out = [[0] * H for _ in range(W)]

    # Anti-diagonal mapping: (r,c) → (W-1-c, H-1-r)
    for r in range(H):
        for c in range(W):
            out[W - 1 - c][H - 1 - r] = g[r][c]

    return out


def apply_isometry(g: list[list[int]], name: str) -> list[list[int]]:
    """
    Apply the named isometry to grid g and return a newly allocated grid.

    Args:
        g: Input grid (rectangular, non-ragged)
        name: One of all_isometries() values

    Returns:
        New grid with transform applied

    Semantics:
        - "id": identity (copy)
        - "rot90": 90° clockwise rotation (H×W → W×H)
        - "rot180": 180° rotation (H×W → H×W)
        - "rot270": 270° clockwise rotation (H×W → W×H)
        - "flip_h": horizontal flip (left↔right) (H×W → H×W)
        - "flip_v": vertical flip (top↔bottom) (H×W → H×W)
        - "transpose": main diagonal reflection (H×W → W×H)
        - "flip_anti": anti-diagonal reflection (H×W → W×H)

    Edge cases:
        - []: returns []
        - 1×1: all transforms return [[val]]
        - Rectangular: rot90/rot270/transpose/flip_anti swap dimensions

    Invariants:
        - Purity: input grid g is never mutated
        - Purity: output rows are not aliases of input rows
        - Determinism: same (g, name) always yields same output
        - Shape:
            - rot90, rot270, transpose, flip_anti: (H,W) → (W,H)
            - rot180, flip_h, flip_v, id: (H,W) → (H,W)

    Raises:
        KeyError: If name not in all_isometries()
        ValueError: If g is ragged (rows have different lengths)
    """
    # Empty grid special case
    if not g:
        return []

    # Validate rectangularity
    W = len(g[0])
    for row in g:
        if len(row) != W:
            raise ValueError("Grid must be rectangular (all rows same length)")

    # Dispatch table
    transforms = {
        "id": lambda x: copy_grid(x),
        "rot90": rot90,
        "rot180": rot180,
        "rot270": rot270,
        "flip_h": flip_h,
        "flip_v": flip_v,
        "transpose": transpose,
        "flip_anti": flip_anti,
    }

    if name not in transforms:
        raise KeyError(f"Unknown isometry: {name}. Must be one of {all_isometries()}")

    return transforms[name](g)


def canonical_key(g: list[list[int]]) -> tuple[int, int, tuple[int, ...]]:
    """
    Return lexicographic comparison key for grid g.

    The key is (rows, cols, row_major_values) where row_major_values is
    a flat tuple of all grid values in row-major order (left-to-right, top-to-bottom).

    Ordering is shape-first (compare rows, then cols), then values.

    Args:
        g: Grid as list of lists of integers

    Returns:
        Tuple of (rows, cols, values_tuple)

    Examples:
        - [] → (0, 0, ())
        - [[5]] → (1, 1, (5,))
        - [[1,2],[3,4]] → (2, 2, (1,2,3,4))

    Invariants:
        - Pure: input g is never mutated
        - Deterministic: same g → same key
        - Stable ordering: lexicographic comparison on returned tuple

    Raises:
        ValueError: If g is ragged (rows have different lengths)
    """
    if not g:
        return (0, 0, ())

    # Validate rectangularity
    rows = len(g)
    cols = len(g[0])
    for row in g:
        if len(row) != cols:
            raise ValueError("Grid must be rectangular (all rows same length)")

    # Flatten in row-major order
    values = []
    for row in g:
        for val in row:
            values.append(val)

    return (rows, cols, tuple(values))


def canonical_grid(g: list[list[int]]) -> list[list[int]]:
    """
    Return the D8 transform of g with minimal canonical_key.

    This implements the Π (Present) operator: an idempotent canonicalization
    that selects the lexicographically minimal D8 image.

    Algorithm:
        1. Apply all 8 isometries in all_isometries() order
        2. Compute canonical_key for each transform
        3. Select transform with minimal key (lexicographic min)
        4. If multiple transforms share minimal key: choose earliest σ in all_isometries() order

    Args:
        g: Grid to canonicalize

    Returns:
        Grid with minimal canonical_key among all D8 transforms

    Examples:
        - [] → []
        - [[5]] → [[5]] (all D8 transforms equal)
        - [[1,2],[3,4]] → [[1,2],[3,4]] (id wins)

    Invariants:
        - Π² = Π: canonical_grid(canonical_grid(g)) == canonical_grid(g) (idempotence)
        - Pure: input g is never mutated
        - Pure: output is newly allocated (no row aliasing)
        - Deterministic: same g → same output grid
        - Minimality: result has lexicographically minimal key among all D8 transforms

    Raises:
        ValueError: If g is ragged
    """
    if not g:
        return []

    # Collect all D8 transforms with their keys
    candidates = []
    for sigma in all_isometries():
        g_sigma = apply_isometry(g, sigma)
        key = canonical_key(g_sigma)
        candidates.append((key, sigma, g_sigma))

    # Find minimum by (key, sigma_index) for deterministic tie-breaking
    # If keys are equal, earlier sigma in all_isometries() order wins
    best = min(candidates, key=lambda x: (x[0], all_isometries().index(x[1])))

    return best[2]  # Return the grid
