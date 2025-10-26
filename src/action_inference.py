"""
Action Inference Module (Phase 5).

This module implements action kernels for GLUE stitching in Step 2.

Phases:
- P5-01: Basic action kernels (set_color, keep_nonzero, identity)
- P5-02: Action inference logic (infer_exact_action)
- P5-03: Tuple-based actions (set_all, copy_from)

Mathematical Foundation:
- GLUE-safe: Read from frozen base only, write to new output
- FY exactness: Bit-for-bit deterministic
- Purity: Input never mutated

See docs/context-packs/P5-01.md for complete specification.
"""

from src.utils import copy_grid


# ============================================================================
# Validation Helpers
# ============================================================================


def _validate_rectangular(Xp: list[list[int]]) -> None:
    """Raise ValueError if Xp not rectangular.

    Args:
        Xp: Grid to validate

    Raises:
        ValueError: If any row has different length than first row
    """
    if not Xp:
        return  # Empty grid is valid
    C = len(Xp[0])
    if not all(len(row) == C for row in Xp):
        raise ValueError("Grid must be rectangular")


def _validate_coords(Xp: list[list[int]], coords: list[tuple[int, int]]) -> None:
    """Raise ValueError if any coord out of bounds.

    Args:
        Xp: Grid providing bounds
        coords: List of (row, col) positions to validate

    Raises:
        ValueError: If any coord outside grid bounds
    """
    if not Xp:
        R, C = 0, 0
    else:
        R, C = len(Xp), len(Xp[0])

    for (r, c) in coords:
        if not (0 <= r < R and 0 <= c < C):
            raise ValueError(f"Coordinate ({r}, {c}) out of bounds for {R}×{C} grid")


def _validate_color(color: int) -> None:
    """Raise ValueError if color not in domain {0, 1, ..., 9}.

    Args:
        color: Color value to validate

    Raises:
        ValueError: If color outside [0, 9]
    """
    if not (0 <= color <= 9):
        raise ValueError(f"Color {color} out of range [0, 9]")


# ============================================================================
# Action Kernels I — Basic Write-Only Actions (P5-01)
# ============================================================================


def apply_set_color(
    Xp: list[list[int]],
    coords: list[tuple[int, int]],
    color: int
) -> list[list[int]]:
    """Return new grid with coords set to color, elsewhere equals Xp.

    Args:
        Xp: Rectangular grid (frozen base)
        coords: List of (row, col) positions
        color: Target color ∈ {0, 1, ..., 9}

    Returns:
        New grid Out where:
        - Out[r][c] = color for all (r, c) in coords
        - Out[r][c] = Xp[r][c] otherwise

    Raises:
        ValueError: If Xp not rectangular, color invalid, or any coord out of bounds

    Guarantees:
        - Purity: Xp unchanged
        - GLUE-safe: Reads only from Xp
        - Determinism: Coords processed in sorted row-major order
        - FY exactness: Bit-for-bit reproducible

    Examples:
        >>> Xp = [[1, 2], [3, 4]]
        >>> apply_set_color(Xp, [(0, 0), (1, 1)], 5)
        [[5, 2], [3, 5]]
        >>> Xp  # Unchanged
        [[1, 2], [3, 4]]

    Mathematical Note:
        This implements the SET_COLOR action from spec.md:50.
        For class K with signature σ, apply to all pixels where Φ(x) = σ.
    """
    # Validate inputs
    _validate_rectangular(Xp)
    _validate_color(color)
    _validate_coords(Xp, coords)

    # Create output as deep copy (ensures purity and GLUE safety)
    Out = copy_grid(Xp)

    # Process coords in sorted order for determinism
    sorted_coords = sorted(coords)

    # Apply action: set each coord to target color
    for (r, c) in sorted_coords:
        Out[r][c] = color

    return Out


def apply_keep_nonzero(
    Xp: list[list[int]],
    coords: list[tuple[int, int]]
) -> list[list[int]]:
    """Return new grid: keep nonzero values at coords, zero out zeros.

    Args:
        Xp: Rectangular grid (frozen base)
        coords: List of (row, col) positions

    Returns:
        New grid Out where:
        - Out[r][c] = Xp[r][c] if Xp[r][c] != 0 for (r, c) in coords
        - Out[r][c] = 0 if Xp[r][c] == 0 for (r, c) in coords
        - Out[r][c] = Xp[r][c] otherwise (not in coords)

    Raises:
        ValueError: If Xp not rectangular or any coord out of bounds

    Guarantees:
        - Purity: Xp unchanged
        - GLUE-safe: Reads only from Xp
        - Determinism: Coords processed in sorted row-major order
        - Idempotence: apply_keep_nonzero(apply_keep_nonzero(Xp, C), C)
                       == apply_keep_nonzero(Xp, C)

    Examples:
        >>> Xp = [[1, 0, 3], [0, 5, 0]]
        >>> apply_keep_nonzero(Xp, [(0, 0), (0, 1), (1, 1)])
        [[1, 0, 3], [0, 5, 0]]  # Keep 1, 5; zeros stay zero

    Mathematical Note:
        This implements the KEEP_NONZERO action from spec.md:51.
        Useful for preserving structure while clearing background.

        Since Out starts as deep_copy(Xp), the logic is:
        - For (r,c) in coords: Out[r][c] already equals Xp[r][c]
        - For (r,c) not in coords: Out[r][c] already equals Xp[r][c]

        The semantic contract is "preserve nonzero, ensure zero stays zero"
        which is automatically satisfied by deep_copy. However, we process
        coords for API uniformity and determinism guarantees.
    """
    # Validate inputs
    _validate_rectangular(Xp)
    _validate_coords(Xp, coords)

    # Create output as deep copy (ensures purity and GLUE safety)
    Out = copy_grid(Xp)

    # Process coords in sorted order for determinism
    # Note: This is effectively a no-op since copy_grid already preserves values,
    # but we sort for API uniformity and determinism guarantees
    sorted_coords = sorted(coords)

    # Semantic contract: keep nonzero, zero stays zero
    # This is already satisfied by copy_grid, but we iterate for clarity
    # and future extension points
    for (r, c) in sorted_coords:
        # Out[r][c] already equals Xp[r][c] from copy_grid
        # If Xp[r][c] != 0: keep it (already done)
        # If Xp[r][c] == 0: set to 0 (already done)
        pass  # Explicit no-op for semantic clarity

    return Out


def apply_identity(
    Xp: list[list[int]],
    coords: list[tuple[int, int]]
) -> list[list[int]]:
    """Return deep copy of Xp (no-op action).

    Args:
        Xp: Rectangular grid (frozen base)
        coords: List of (row, col) positions (IGNORED — kept for API uniformity)

    Returns:
        Deep copy of Xp (all pixels unchanged)

    Raises:
        ValueError: If Xp not rectangular or any coord out of bounds

    Guarantees:
        - Purity: Xp unchanged
        - GLUE-safe: Reads only from Xp
        - Determinism: Always returns exact copy
        - Idempotence: apply_identity(apply_identity(Xp, C), C)
                       == apply_identity(Xp, C)

    Examples:
        >>> Xp = [[1, 2], [3, 4]]
        >>> apply_identity(Xp, [(0, 0)])
        [[1, 2], [3, 4]]
        >>> Xp  # Unchanged
        [[1, 2], [3, 4]]

    Mathematical Note:
        This implements the IDENTITY action from spec.md:52.
        Used when a class needs no transformation.
        Coords validated but unused (API uniformity with other actions).
    """
    # Validate inputs (coords validated for API uniformity)
    _validate_rectangular(Xp)
    _validate_coords(Xp, coords)

    # Return deep copy (coords parameter ignored)
    return copy_grid(Xp)
