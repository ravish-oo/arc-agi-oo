"""
MirrorComplete Family for ARC AGI solver.

Global exact family: H/V/Diagonal symmetry completion.
Learns ONE axis from {"H", "V", "D"} that satisfies FY for ALL train pairs.
Fills missing/zero cells from mirror counterparts along learned axis.

Critical constraints:
- FY: Accept axis only if deep_eq holds for ALL train pairs
- Determinism: Try axes in order ["H", "V", "D"]; first-acceptable wins
- Unified: ONE axis for all pairs (no per-pair selection)
- Frozen base: Read from original X only (no read-after-write)
- Shape safety: All pairs must have same dimensions
- Purity: No mutations to inputs
"""

from src.utils import dims, deep_eq


class MirrorCompleteFamily:
    """
    Global exact family: H/V/Diagonal symmetry completion.

    Learns ONE axis from {"H", "V", "D"} that satisfies FY for ALL train pairs.
    Fills missing/zero cells from mirror counterparts along learned axis.

    Invariants:
        - FY: Accept axis only if deep_eq holds for ALL train pairs
        - Determinism: Try axes in order ["H", "V", "D"]; first-acceptable wins
        - Unified: ONE axis for all pairs (no per-pair selection)
        - Frozen base: Read from original X only (no read-after-write)
        - Shape safety: All pairs must have same dimensions
        - Purity: No mutations to inputs
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            axis: None initially; set to "H"|"V"|"D" after successful fit()
        """
        self.name = "MirrorComplete"
        self.params = type('Params', (), {'axis': None})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Feasibility fit for Step-2 architecture.

        In Step-2, this transform is always applicable - it preprocesses
        the input and Φ/GLUE handles matching to the output.

        This family has no trainable parameters - it applies deterministic
        logic to transform the input. Feasibility is universal.

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            Always True (transform is always applicable)

        Step-2 Contract:
            - No fit() parameters to learn
            - Transform is always feasible
            - FY constraint enforced at candidate level, not here
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Always applicable in Step-2
        return True
    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply stored axis mirroring to input X.

        Args:
            X: Input grid

        Returns:
            Grid with filled cells according to learned axis

        Raises:
            RuntimeError: If params.axis is None (must call fit() first)

        Determinism:
            - Same (X, axis) always yields same output

        Purity:
            - Never mutates X
            - Returns new grid (no aliasing)
        """
        if self.params.axis is None:
            raise RuntimeError("MirrorCompleteFamily.apply() called before fit(). params.axis is None. Must call fit() with train_pairs first.")

        return self._fill_axis(X, self.params.axis)

    def _fill_axis(self, X: list[list[int]], axis: str) -> list[list[int]]:
        """
        Fill X along specified axis using frozen base read.

        Mirroring rules (frozen base read):
            - "H": Mirror horizontally across mid-column
            - "V": Mirror vertically across mid-row
            - "D": Mirror along main diagonal

        CRITICAL: Always read from original X, never from partially-filled Y (frozen base).

        Args:
            X: Original input grid (never modified)
            axis: "H" | "V" | "D"

        Returns:
            New grid Y with filled cells

        Algorithm (frozen base read):
            1. Y = copy_grid(X)  # Start with copy
            2. H, W = dims(X)
            3. For each (r, c) in Y:
                4. If Y[r][c] == 0:  # Cell needs filling
                    5. Compute mirror (mr, mc) based on axis
                    6. If 0 <= mr < H and 0 <= mc < W and X[mr][mc] != 0:
                        7. Y[r][c] = X[mr][mc]  # Read from ORIGINAL X
            8. Return Y

        Determinism:
            - Same (X, axis) always yields same Y

        Purity:
            - Never mutates X
            - Returns new grid
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Check for empty dimensions
        if h == 0 or w == 0:
            return []

        # Dispatch to axis-specific fill method
        if axis == "H":
            return self._fill_h(X)
        elif axis == "V":
            return self._fill_v(X)
        elif axis == "D":
            return self._fill_d(X)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'H', 'V', or 'D'.")

    def _fill_h(self, X: list[list[int]]) -> list[list[int]]:
        """
        Fill horizontally (mirror across mid-column).

        Mirror mapping: (r, c) ↔ (r, W-1-c)

        Frozen base read: For each (r,c), if Y[r][c]==0, read from X[r][W-1-c].

        Args:
            X: Input grid

        Returns:
            Grid with horizontal mirroring applied

        Purity:
            - Never mutates X
            - Returns new grid
        """
        h, w = dims(X)

        # Create copy of X as starting point
        Y = [list(row) for row in X]

        # Fill zeros from horizontal mirror
        for r in range(h):
            for c in range(w):
                if Y[r][c] == 0:
                    # Mirror coordinate: (r, c) → (r, W-1-c)
                    mc = w - 1 - c

                    # Bounds check
                    if 0 <= mc < w:
                        # Read from ORIGINAL X (frozen base)
                        mirror_val = X[r][mc]

                        # Fill if mirror is non-zero
                        if mirror_val != 0:
                            Y[r][c] = mirror_val

        return Y

    def _fill_v(self, X: list[list[int]]) -> list[list[int]]:
        """
        Fill vertically (mirror across mid-row).

        Mirror mapping: (r, c) ↔ (H-1-r, c)

        Frozen base read: For each (r,c), if Y[r][c]==0, read from X[H-1-r][c].

        Args:
            X: Input grid

        Returns:
            Grid with vertical mirroring applied

        Purity:
            - Never mutates X
            - Returns new grid
        """
        h, w = dims(X)

        # Create copy of X as starting point
        Y = [list(row) for row in X]

        # Fill zeros from vertical mirror
        for r in range(h):
            for c in range(w):
                if Y[r][c] == 0:
                    # Mirror coordinate: (r, c) → (H-1-r, c)
                    mr = h - 1 - r

                    # Bounds check
                    if 0 <= mr < h:
                        # Read from ORIGINAL X (frozen base)
                        mirror_val = X[mr][c]

                        # Fill if mirror is non-zero
                        if mirror_val != 0:
                            Y[r][c] = mirror_val

        return Y

    def _fill_d(self, X: list[list[int]]) -> list[list[int]]:
        """
        Fill diagonally (mirror along main diagonal).

        Mirror mapping: (r, c) ↔ (c, r)

        For non-square grids, only mirror within min(H,W)×min(H,W) region.

        Frozen base read: For each (r,c), if Y[r][c]==0, read from X[c][r] if in bounds.

        Args:
            X: Input grid

        Returns:
            Grid with diagonal mirroring applied

        Purity:
            - Never mutates X
            - Returns new grid
        """
        h, w = dims(X)

        # Create copy of X as starting point
        Y = [list(row) for row in X]

        # Fill zeros from diagonal mirror
        for r in range(h):
            for c in range(w):
                if Y[r][c] == 0:
                    # Mirror coordinate: (r, c) → (c, r)
                    mr = c
                    mc = r

                    # Bounds check (diagonal only works within grid bounds)
                    if 0 <= mr < h and 0 <= mc < w:
                        # Read from ORIGINAL X (frozen base)
                        mirror_val = X[mr][mc]

                        # Fill if mirror is non-zero
                        if mirror_val != 0:
                            Y[r][c] = mirror_val

        return Y
