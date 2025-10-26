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


def _validate_rectangular(Xp: list[list[int]], allow_empty: bool = True) -> None:
    """Raise ValueError if Xp not rectangular.

    Args:
        Xp: Grid to validate
        allow_empty: If False, raise error for empty grid

    Raises:
        ValueError: If any row has different length than first row, or if empty when not allowed
    """
    if not Xp:
        if not allow_empty:
            raise ValueError("Cannot mirror empty grid")
        return
    C = len(Xp[0])
    if C == 0 and not allow_empty:
        raise ValueError("Cannot mirror grid with zero columns")
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


# ============================================================================
# Action Kernels II — Mirror Operations (P5-02)
# ============================================================================


def apply_mirror_h(
    Xp: list[list[int]],
    coords: list[tuple[int, int]]
) -> list[list[int]]:
    """Return new grid with coords mirrored horizontally (row r → row R-1-r).

    GLUE-safe: ALL reads from frozen Xp, writes to new Out.
    Grid-global: Mirror axes defined by full grid dimensions R×C.

    Args:
        Xp: Rectangular grid (frozen base, guaranteed non-empty)
        coords: List of (row, col) positions to mirror

    Returns:
        New grid Out where:
        - Out[r][c] = Xp[R-1-r][c] for all (r, c) in coords
        - Out[r][c] = Xp[r][c] otherwise (not in coords)

    Raises:
        ValueError: If Xp not rectangular, empty, or any coord out of bounds

    Guarantees:
        - Purity: Xp unchanged
        - GLUE-safe: Reads ONLY from Xp (never from Out)
        - Determinism: Coords processed in sorted row-major order
        - Grid-global axes: R = len(Xp), mirror row r → row R-1-r
        - FY exactness: Bit-for-bit reproducible

    Examples:
        # 3×3 grid, mirror top-left to bottom-left value
        >>> Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> apply_mirror_h(Xp, [(0, 0)])
        [[7, 2, 3], [4, 5, 6], [7, 8, 9]]  # Xp[0][0] = Xp[2][0] = 7

        # Even grid (4×2)
        >>> Xp = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> apply_mirror_h(Xp, [(0, 1), (3, 0)])
        [[1, 8], [3, 4], [5, 6], [1, 8]]  # (0,1)→(3,1)=8, (3,0)→(0,0)=1

        # 1×1 grid (edge case)
        >>> Xp = [[5]]
        >>> apply_mirror_h(Xp, [(0, 0)])
        [[5]]  # r=0 mirrors to R-1-r = 1-1-0 = 0 (itself)

    Mathematical Note:
        Horizontal mirror flips across horizontal (east-west) axis.
        Row r maps to row R-1-r using FULL GRID height R.
        NOT bbox-relative, NOT patch-local.

        Property: apply_mirror_h(apply_mirror_h(Xp, C), C) = Xp
                  (involution for full-grid coords)

    GLUE Safety:
        Critical test: Two-sided coords including both (r, c) and (R-1-r, c).
        Verifies that processing order does not affect result.
    """
    # Validate inputs
    _validate_rectangular(Xp, allow_empty=False)
    _validate_coords(Xp, coords)

    # Get grid dimensions
    R = len(Xp)
    C = len(Xp[0])

    # Create output as deep copy (ensures purity and GLUE safety)
    Out = copy_grid(Xp)

    # Process coords in sorted order for determinism
    sorted_coords = sorted(coords)

    # Apply horizontal mirror: row r → row R-1-r
    # CRITICAL: Read from Xp only, never from Out (GLUE safety)
    for (r, c) in sorted_coords:
        Out[r][c] = Xp[R - 1 - r][c]  # Read from frozen Xp

    return Out


def apply_mirror_v(
    Xp: list[list[int]],
    coords: list[tuple[int, int]]
) -> list[list[int]]:
    """Return new grid with coords mirrored vertically (col c → col C-1-c).

    GLUE-safe: ALL reads from frozen Xp, writes to new Out.
    Grid-global: Mirror axes defined by full grid dimensions R×C.

    Args:
        Xp: Rectangular grid (frozen base, guaranteed non-empty)
        coords: List of (row, col) positions to mirror

    Returns:
        New grid Out where:
        - Out[r][c] = Xp[r][C-1-c] for all (r, c) in coords
        - Out[r][c] = Xp[r][c] otherwise (not in coords)

    Raises:
        ValueError: If Xp not rectangular, empty, or any coord out of bounds

    Guarantees:
        - Purity: Xp unchanged
        - GLUE-safe: Reads ONLY from Xp (never from Out)
        - Determinism: Coords processed in sorted row-major order
        - Grid-global axes: C = len(Xp[0]), mirror col c → col C-1-c
        - FY exactness: Bit-for-bit reproducible

    Examples:
        # 3×3 grid, mirror top-right to top-left value
        >>> Xp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> apply_mirror_v(Xp, [(0, 2)])
        [[1, 2, 1], [4, 5, 6], [7, 8, 9]]  # Xp[0][2] = Xp[0][0] = 1

        # Even grid (2×4)
        >>> Xp = [[1, 2, 3, 4], [5, 6, 7, 8]]
        >>> apply_mirror_v(Xp, [(0, 0), (1, 3)])
        [[4, 2, 3, 4], [5, 6, 7, 5]]  # (0,0)→(0,3)=4, (1,3)→(1,0)=5

        # 1×1 grid (edge case)
        >>> Xp = [[5]]
        >>> apply_mirror_v(Xp, [(0, 0)])
        [[5]]  # c=0 mirrors to C-1-c = 1-1-0 = 0 (itself)

    Mathematical Note:
        Vertical mirror flips across vertical (north-south) axis.
        Column c maps to column C-1-c using FULL GRID width C.
        NOT bbox-relative, NOT patch-local.

        Property: apply_mirror_v(apply_mirror_v(Xp, C), C) = Xp
                  (involution for full-grid coords)

    GLUE Safety:
        Critical test: Two-sided coords including both (r, c) and (r, C-1-c).
        Verifies that processing order does not affect result.
    """
    # Validate inputs
    _validate_rectangular(Xp, allow_empty=False)
    _validate_coords(Xp, coords)

    # Get grid dimensions
    R = len(Xp)
    C = len(Xp[0])

    # Create output as deep copy (ensures purity and GLUE safety)
    Out = copy_grid(Xp)

    # Process coords in sorted order for determinism
    sorted_coords = sorted(coords)

    # Apply vertical mirror: col c → col C-1-c
    # CRITICAL: Read from Xp only, never from Out (GLUE safety)
    for (r, c) in sorted_coords:
        Out[r][c] = Xp[r][C - 1 - c]  # Read from frozen Xp

    return Out


# ============================================================================
# Action Kernels III — Class Verifier (P5-03)
# ============================================================================


def verify_action_on_class(
    action: tuple[str, object | None],
    items: list[dict],
    class_coords: list[tuple[int, int, int]]
) -> bool:
    """Verify if given action matches ALL pixels of this class across ALL training pairs.

    FY Equality: Returns True iff applying the action to class pixels in Xp yields
    exact match with Y on those pixels for EVERY training pair.

    Args:
        action: Action tuple ("name", param) where:
            - name ∈ {"set_color", "mirror_h", "mirror_v", "keep_nonzero", "identity"}
            - param: color ∈ [0..9] for "set_color", None for others
        items: List of training pair dicts, each with keys:
            - "Xp": Rectangular grid (frozen base)
            - "Y": Rectangular grid (target)
        class_coords: List of (i, r, c) triples indicating class pixels
            - i: train index
            - r, c: pixel coordinates within train i

    Returns:
        True iff action satisfies FY equality on class pixels for all trains
        False otherwise (including validation failures)

    Validation (raises ValueError):
        - Action tuple format: must be (str, param)
        - Action name must be in valid set
        - Color must be in [0..9] for set_color
        - All Xp and Y must be rectangular
        - All coords must be in bounds

    Edge Cases:
        - Empty class_coords → True (vacuous)
        - Some train i has no coords → skip that train (vacuous for i)
        - Duplicates in class_coords → deterministic (sorted coords)

    Guarantees:
        - FY exactness: Bit-for-bit equality (no tolerance)
        - GLUE-safe: Uses existing kernels (read from frozen Xp only)
        - Determinism: Sorted coords per train, stable boolean
        - Purity: No mutation of inputs

    Examples:
        # set_color pass (all class pixels in Y equal the color)
        >>> action = ("set_color", 5)
        >>> items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
        >>> class_coords = [(0, 0, 0)]  # Train 0, pixel (0, 0)
        >>> verify_action_on_class(action, items, class_coords)
        True

        # set_color fail (mismatch in Y)
        >>> items = [{"Xp": [[1, 2]], "Y": [[3, 2]]}]  # Y has 3, not 5
        >>> verify_action_on_class(action, items, class_coords)
        False

        # mirror_h pass
        >>> action = ("mirror_h", None)
        >>> items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [3, 4]]}]
        >>> class_coords = [(0, 0, 0)]  # Top-left mirrors to bottom-left
        >>> verify_action_on_class(action, items, class_coords)
        True  # Y[0][0] = 3 = Xp[1][0]

        # Empty coords (vacuous)
        >>> verify_action_on_class(("identity", None), [], [])
        True

    Mathematical Note:
        This implements the FY verification step from spec.md.
        For a Φ-class with signature σ, verify that action A_σ
        produces exact match with Y on all pixels where Φ(x) = σ.

        This is the foundation for action inference (P5-04), which
        will try each action in order until one satisfies FY.
    """
    # Validate action tuple format
    if not isinstance(action, tuple) or len(action) != 2:
        raise ValueError(f"Action must be a 2-tuple (name, param), got: {action}")

    action_name, param = action

    # Validate action name
    valid_actions = {"set_color", "mirror_h", "mirror_v", "keep_nonzero", "identity"}
    if action_name not in valid_actions:
        raise ValueError(
            f"Invalid action name '{action_name}'. Must be one of: {valid_actions}"
        )

    # Validate parameter for set_color
    if action_name == "set_color":
        if param is None:
            raise ValueError("set_color requires a color parameter (int in [0..9])")
        if not isinstance(param, int) or not (0 <= param <= 9):
            raise ValueError(f"set_color color must be int in [0..9], got: {param}")
    else:
        if param is not None:
            raise ValueError(
                f"Action '{action_name}' requires param=None, got: {param}"
            )

    # Empty class_coords → vacuous truth
    if not class_coords:
        return True

    # Group coords by train index
    coords_by_train: dict[int, list[tuple[int, int]]] = {}
    for (i, r, c) in class_coords:
        if i not in coords_by_train:
            coords_by_train[i] = []
        coords_by_train[i].append((r, c))

    # Verify each train
    for i in sorted(coords_by_train.keys()):
        # Skip if train index out of range
        if i < 0 or i >= len(items):
            raise ValueError(
                f"Train index {i} out of range for {len(items)} training pairs"
            )

        item = items[i]
        Xp = item["Xp"]
        Y = item["Y"]

        # Validate grids
        _validate_rectangular(Xp, allow_empty=False)
        _validate_rectangular(Y, allow_empty=False)

        # Check shape compatibility
        if len(Xp) != len(Y) or (Xp and len(Xp[0]) != len(Y[0])):
            raise ValueError(
                f"Train {i}: Xp and Y must have same shape. "
                f"Xp: {len(Xp)}×{len(Xp[0]) if Xp else 0}, "
                f"Y: {len(Y)}×{len(Y[0]) if Y else 0}"
            )

        # Get coords for this train
        S_i = coords_by_train[i]

        # Validate coords for this train
        _validate_coords(Xp, S_i)

        # Apply action to Xp with coords S_i
        if action_name == "set_color":
            Out = apply_set_color(Xp, S_i, param)
        elif action_name == "mirror_h":
            Out = apply_mirror_h(Xp, S_i)
        elif action_name == "mirror_v":
            Out = apply_mirror_v(Xp, S_i)
        elif action_name == "keep_nonzero":
            Out = apply_keep_nonzero(Xp, S_i)
        elif action_name == "identity":
            Out = apply_identity(Xp, S_i)
        else:
            # Should never reach here (already validated)
            raise ValueError(f"Unknown action: {action_name}")

        # Check FY equality: Out[r][c] == Y[r][c] for all (r, c) in S_i
        for (r, c) in S_i:
            if Out[r][c] != Y[r][c]:
                return False  # Mismatch found

    # All trains satisfy FY equality on class pixels
    return True
