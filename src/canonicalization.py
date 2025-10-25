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
