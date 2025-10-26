"""
GLUE stitching functions for compositional solving.

Phase 6: Residual computation and Phi partition building.
All functions are pure (no mutation) and deterministic.
"""

from src.utils import dims


def compute_residual(Xp: list[list[int]], Y: list[list[int]]) -> list[list[object]]:
    """
    Compute pixelwise residual between transformed input and target.

    Residual R defines which pixels need correction: R[r][c] is None if
    Xp[r][c] already equals Y[r][c], otherwise R[r][c] holds the target
    value Y[r][c] that must be written.

    Mathematical definition:
        R[r][c] = None         if Xp[r][c] = Y[r][c]
        R[r][c] = Y[r][c]      otherwise

    This is the "override table" for Step-2 composition (P + Φ/GLUE).

    Preconditions:
      - Xp is rectangular (all rows have same length)
      - Y is rectangular (all rows have same length)
      - Xp and Y have identical dimensions (R, C)
      - All values in Xp and Y are in range 0..9

    Postconditions:
      - R has shape (R, C)
      - R[r][c] = None        iff Xp[r][c] == Y[r][c]
      - R[r][c] = Y[r][c]     otherwise (value in 0..9)
      - Inputs Xp and Y are never mutated
      - Output rows are newly allocated (no aliasing)

    Args:
        Xp: Transformed input grid (output of some P family transform)
        Y: Target grid (ground truth output)

    Returns:
        New grid R where None indicates "no change needed" and integers
        0..9 indicate "set pixel to this value"

    Raises:
        ValueError: if Xp is ragged (rows have different lengths)
        ValueError: if Y is ragged (rows have different lengths)
        ValueError: if dims(Xp) != dims(Y) (shape mismatch)

    Examples:
        >>> # All equal (no residual)
        >>> compute_residual([[1,2]], [[1,2]])
        [[None, None]]

        >>> # All different (full residual)
        >>> compute_residual([[0,0]], [[1,2]])
        [[1, 2]]

        >>> # Mixed (partial residual)
        >>> compute_residual([[1,0,3]], [[1,2,3]])
        [[None, 2, None]]
    """
    # Validate Xp is rectangular and get shape
    # dims() raises ValueError if ragged
    try:
        h_Xp, w_Xp = dims(Xp)
    except ValueError:
        raise ValueError("Xp is ragged: rows have different lengths")

    # Validate Y is rectangular and get shape
    # dims() raises ValueError if ragged
    try:
        h_Y, w_Y = dims(Y)
    except ValueError:
        raise ValueError("Y is ragged: rows have different lengths")

    # Check shape match
    if (h_Xp, w_Xp) != (h_Y, w_Y):
        raise ValueError(
            f"Shape mismatch: dims(Xp)=({h_Xp},{w_Xp}) != dims(Y)=({h_Y},{w_Y})"
        )

    # Build residual: new grid, no aliasing
    # R[r][c] = None if match, else Y[r][c]
    return [
        [None if Xp[r][c] == Y[r][c] else Y[r][c] for c in range(w_Xp)]
        for r in range(h_Xp)
    ]
