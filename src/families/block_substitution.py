"""
BlockSubstitution Family for ARC AGI solver.

Global exact family: Per-color k×k glyph expansion (deterministic substitution).
Learns ONE (k, glyphs) from first training pair that works for ALL pairs.
k = expansion factor (Y_rows / X_rows = Y_cols / X_cols)
glyphs = dict mapping color -> k×k grid pattern

Each pixel with color c in input X is replaced by glyphs[c] (k×k block) to produce Y.

Critical constraints:
- k must be same for all colors (uniform square expansion)
- Same color must always map to same glyph (consistency)
- glyphs dict must cover all colors in X (completeness)
"""

from src.utils import dims, deep_eq


class BlockSubstitutionFamily:
    """
    Global exact family: Per-color k×k glyph expansion (deterministic substitution).

    Learns ONE (k, glyphs) from first training pair that works for ALL pairs.
    Each pixel is replaced by its color's k×k glyph pattern.

    Invariants:
        - FY: Accept params only if they reproduce ALL train pairs exactly
        - Square expansion: k = Y_rows/X_rows = Y_cols/X_cols (must be equal)
        - Integer expansion: Y dimensions must be integer multiples of X dimensions
        - Glyph consistency: same color → same glyph everywhere
        - Deterministic color iteration: process colors in sorted ascending order
        - Purity: No mutations; fresh grid allocation
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            k: None initially; int after successful fit() (expansion factor)
            glyphs: None initially; dict[int, list[list[int]]] after successful fit()
        """
        self.name = "BlockSubstitution"
        self.params = type('Params', (), {
            'k': None,
            'glyphs': None,
        })()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Learn ONE (k, glyphs) from first pair that works for ALL pairs.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair and compute expansion factor k
            3. Verify k is integer and square (kH = kW)
            4. Extract glyphs for each color from first pair
            5. Verify glyph consistency within first pair
            6. Verify on all training pairs

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found params that satisfy FY on all pairs; False otherwise

        Determinism:
            - Always learn from first pair
            - Color iteration is sorted ascending
            - Pixel iteration is row-major

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting params
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Extract first pair
        first_pair = train_pairs[0]
        X0 = first_pair["input"]
        Y0 = first_pair["output"]

        # Handle empty grids
        if not X0 or not Y0:
            return False

        # Get dimensions
        hx, wx = dims(X0)
        hy, wy = dims(Y0)

        # Check for empty dimensions
        if hx == 0 or wx == 0 or hy == 0 or wy == 0:
            return False

        # Check integer division for expansion factors
        if hy % hx != 0 or wy % wx != 0:
            return False  # Non-integer expansion

        # Compute expansion factors
        kH = hy // hx
        kW = wy // wx

        # Check square expansion (kH must equal kW)
        if kH != kW:
            return False  # Non-square expansion

        k = kH  # Use single k for square expansion

        # Validate k > 0
        if k <= 0:
            return False

        # Extract all colors from X0 in ascending sorted order
        all_colors = set()
        for r in range(hx):
            for c in range(wx):
                all_colors.add(X0[r][c])

        colors = sorted(all_colors)

        # Build glyphs dict for each color
        glyphs = {}

        for color in colors:
            # Find all positions where this color appears in X0
            for r in range(hx):
                for c in range(wx):
                    if X0[r][c] == color:
                        # Extract corresponding k×k block from Y0
                        glyph_candidate = self._extract_glyph(Y0, r, c, k)

                        if color not in glyphs:
                            # First occurrence defines the glyph for this color
                            glyphs[color] = glyph_candidate
                        else:
                            # Verify consistency: all occurrences must have same glyph
                            if not self._glyphs_equal(glyph_candidate, glyphs[color]):
                                return False  # Inconsistent glyphs for same color

        # Verify on ALL training pairs
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Check dimensions
            if not X or not Y:
                return False

            hx_p, wx_p = dims(X)
            hy_p, wy_p = dims(Y)

            # Check dimension consistency
            if hx_p == 0 or wx_p == 0:
                return False

            # Verify expansion factor matches
            if hy_p != hx_p * k or wy_p != wx_p * k:
                return False

            # Check all colors in X are in glyphs dict
            for r in range(hx_p):
                for c in range(wx_p):
                    if X[r][c] not in glyphs:
                        return False  # Missing color in glyphs

            # Verify FY: apply glyphs and check equality
            Y_predicted = self._apply_with_glyphs(X, k, glyphs)
            if not deep_eq(Y_predicted, Y):
                return False  # FY violation

        # All pairs satisfied - store params and return True
        self.params.k = k
        self.params.glyphs = glyphs
        return True

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply glyph expansion using learned parameters to input X.

        Args:
            X: Input grid

        Returns:
            Grid where each pixel X[r][c] is replaced by glyphs[X[r][c]] (k×k block)
            Output dimensions: (X_rows × k, X_cols × k)

        Raises:
            RuntimeError: If params not set (must call fit() first)
            KeyError: If X contains a color not in glyphs dict

        Determinism:
            - Same (X, params) always yields same output
            - Iteration order is row-major (deterministic)

        Purity:
            - Never mutates X
            - Returns new grid
        """
        if self.params.k is None:
            raise RuntimeError("BlockSubstitutionFamily.apply() called before fit(). params.k is None. Must call fit() with train_pairs first.")

        if self.params.glyphs is None:
            raise RuntimeError("BlockSubstitutionFamily.apply() called before fit(). params.glyphs is None. Must call fit() with train_pairs first.")

        # Use helper method
        return self._apply_with_glyphs(X, self.params.k, self.params.glyphs)

    def _apply_with_glyphs(self, X: list[list[int]], k: int, glyphs: dict) -> list[list[int]]:
        """
        Helper: apply glyph expansion with given params.

        Args:
            X: Input grid
            k: Expansion factor
            glyphs: Glyph dictionary

        Returns:
            Grid with each pixel expanded to its k×k glyph
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Check for empty dimensions
        if h == 0 or w == 0:
            return []

        # Create output grid with dimensions (h*k, w*k)
        output_h = h * k
        output_w = w * k
        result = [[0] * output_w for _ in range(output_h)]

        # For each pixel in input
        for r in range(h):
            for c in range(w):
                color = X[r][c]

                # Get glyph for this color (raises KeyError if missing)
                if color not in glyphs:
                    raise KeyError(f"Color {color} not found in glyphs dictionary. Available colors: {sorted(glyphs.keys())}")

                glyph = glyphs[color]

                # Copy glyph into output at position (r*k, c*k)
                for i in range(k):
                    for j in range(k):
                        result[r * k + i][c * k + j] = glyph[i][j]

        return result

    def _extract_glyph(self, Y: list[list[int]], r: int, c: int, k: int) -> list[list[int]]:
        """
        Helper: extract k×k block from Y at position (r*k, c*k).

        Args:
            Y: Output grid (expanded)
            r: Row index in input grid (pixel position, not block position)
            c: Col index in input grid (pixel position, not block position)
            k: Glyph size

        Returns:
            k×k glyph extracted from Y[r*k:(r+1)*k, c*k:(c+1)*k]

        Determinism:
            - Deterministic indexing (row-major extraction)

        Purity:
            - Never mutates Y
            - Returns new list-of-lists (no aliasing)
        """
        # Compute block start position
        r0 = r * k
        c0 = c * k

        # Extract k×k block
        glyph = []
        for i in range(k):
            row = []
            for j in range(k):
                row.append(Y[r0 + i][c0 + j])
            glyph.append(row)

        return glyph

    def _glyphs_equal(self, glyph1: list[list[int]], glyph2: list[list[int]]) -> bool:
        """
        Helper: check if two glyphs are equal (deep content comparison).

        Args:
            glyph1: First glyph (k×k grid)
            glyph2: Second glyph (k×k grid)

        Returns:
            True if glyphs have same content; False otherwise
        """
        # Check dimensions
        if len(glyph1) != len(glyph2):
            return False

        if not glyph1:
            return True  # Both empty

        if len(glyph1[0]) != len(glyph2[0]):
            return False

        # Compare content
        for i in range(len(glyph1)):
            for j in range(len(glyph1[0])):
                if glyph1[i][j] != glyph2[i][j]:
                    return False

        return True
