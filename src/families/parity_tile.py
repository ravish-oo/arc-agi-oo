"""
ParityTile Family for ARC AGI solver.

Global exact family: Tiling with horizontal/vertical/checkerboard parity flips.
Learns ONE (tiles_v, tiles_h, mode) from first training pair that works for ALL pairs.
tiles_v = number of vertical tile repetitions (Y_rows / X_rows)
tiles_h = number of horizontal tile repetitions (Y_cols / X_cols)
mode ∈ {"none", "h", "v", "hv"}:
  - "none": normal tiling (no flips)
  - "h": flip horizontally on odd column tiles (C&1 == 1)
  - "v": flip vertically on odd row tiles (R&1 == 1)
  - "hv": both flips (checkerboard pattern: (R^C)&1 == 1)
"""

from src.utils import dims, deep_eq, flip_h, flip_v


class ParityTileFamily:
    """
    Global exact family: Tiling with horizontal/vertical/checkerboard parity flips.

    Learns ONE (tiles_v, tiles_h, mode) from first training pair that works for ALL pairs.
    Replicates input X in tiles_v × tiles_h grid with parity-based horizontal/vertical flips.

    Invariants:
        - FY: Accept params only if they reproduce ALL train pairs exactly
        - Unified parameters: ONE (tiles_v, tiles_h, mode) for all pairs
        - Integer factors: tiles_v and tiles_h must be positive integers
        - Deterministic mode search: try ["none", "h", "v", "hv"] in exact order
        - Parity flips: "h" flips on odd C, "v" flips on odd R, "hv" flips on (R^C)&1
        - Purity: No mutations; fresh grid allocation
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            tiles_v: None initially; int after successful fit()
            tiles_h: None initially; int after successful fit()
            mode: None initially; str after successful fit()
        """
        self.name = "ParityTile"
        self.params = type('Params', (), {
            'tiles_v': None,
            'tiles_h': None,
            'mode': None,
        })()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Learn ONE (tiles_v, tiles_h, mode) from first pair that works for ALL pairs.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair and learn tile factors
            3. Try modes in order ["none", "h", "v", "hv"]
            4. First mode that works for ALL pairs wins
            5. Store params and return True if found

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found params that satisfy FY on all pairs; False otherwise

        Determinism:
            - Always learn from first pair (train_pairs[0])
            - Mode search order is fixed ["none", "h", "v", "hv"]
            - Verification order is stable

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

        # Check integer division for tile factors
        if hy % hx != 0 or wy % wx != 0:
            return False  # Non-integer tiling

        # Learn tile factors from first pair
        tiles_v = hy // hx
        tiles_h = wy // wx

        # Validate positive factors
        if tiles_v <= 0 or tiles_h <= 0:
            return False

        # Try each mode in deterministic order
        for mode in ["none", "h", "v", "hv"]:
            # Test this mode on ALL pairs
            if self._try_mode(train_pairs, tiles_v, tiles_h, mode):
                # This mode works for ALL pairs - store and return
                self.params.tiles_v = tiles_v
                self.params.tiles_h = tiles_h
                self.params.mode = mode
                return True

        # No mode satisfied FY on all pairs
        return False

    def _try_mode(self, train_pairs: list[dict], tiles_v: int, tiles_h: int, mode: str) -> bool:
        """
        Test if specific (tiles_v, tiles_h, mode) satisfies FY on ALL pairs.

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts
            tiles_v: vertical tile repetitions
            tiles_h: horizontal tile repetitions
            mode: flip mode to test

        Returns:
            True if this mode produces exact match for all pairs; False otherwise
        """
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Compute predicted output using this mode
            Y_predicted = self._tile(X, tiles_v, tiles_h, mode)

            # Check bit-for-bit equality
            if not deep_eq(Y_predicted, Y):
                return False  # FY violation

        return True  # This mode works for ALL pairs

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply tiling with parity flips using learned parameters to input X.

        Args:
            X: Input grid

        Returns:
            Grid where X is tiled tiles_v × tiles_h times with parity-based flips
            Output dimensions: (X_rows × tiles_v, X_cols × tiles_h)

        Raises:
            RuntimeError: If params not set (must call fit() first)

        Determinism:
            - Same (X, params) always yields same output
            - Iteration order is row-major (deterministic)

        Purity:
            - Never mutates X
            - Returns new grid
        """
        if self.params.tiles_v is None:
            raise RuntimeError("ParityTileFamily.apply() called before fit(). params.tiles_v is None. Must call fit() with train_pairs first.")

        if self.params.tiles_h is None:
            raise RuntimeError("ParityTileFamily.apply() called before fit(). params.tiles_h is None. Must call fit() with train_pairs first.")

        if self.params.mode is None:
            raise RuntimeError("ParityTileFamily.apply() called before fit(). params.mode is None. Must call fit() with train_pairs first.")

        # Use helper method for tiling
        return self._tile(X, self.params.tiles_v, self.params.tiles_h, self.params.mode)

    def _tile(self, X: list[list[int]], tiles_v: int, tiles_h: int, mode: str) -> list[list[int]]:
        """
        Helper: tile X with parity-based flips.

        Args:
            X: Source grid to tile
            tiles_v: Number of vertical repetitions
            tiles_h: Number of horizontal repetitions
            mode: Flip mode ("none", "h", "v", "hv")

        Returns:
            Tiled grid with dimensions (X_rows × tiles_v, X_cols × tiles_h)

        Algorithm:
            For each tile position (R, C):
                - Determine if flip needed based on mode and parity
                - Copy (possibly flipped) tile to output position

        Flip Logic:
            - "none": no flips
            - "h": flip_h if C&1 == 1 (odd column)
            - "v": flip_v if R&1 == 1 (odd row)
            - "hv": checkerboard - when (R^C)&1 == 1:
              flip_h if R even, flip_v if R odd

        Determinism:
            - Row-major tile iteration (R outer, C inner)
            - Flip functions are deterministic

        Purity:
            - Never mutates X
            - Returns new grid (no aliasing)
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Create output grid
        output_h = h * tiles_v
        output_w = w * tiles_h
        result = []

        # Build output row by row
        for out_r in range(output_h):
            row = []
            for out_c in range(output_w):
                # Determine which tile this output position belongs to
                tile_r = out_r // h  # Tile row index (0 to tiles_v-1)
                tile_c = out_c // w  # Tile col index (0 to tiles_h-1)

                # Position within the tile
                within_r = out_r % h
                within_c = out_c % w

                # Determine which version of X to use based on mode and parity
                if mode == "none":
                    # No flip
                    pixel = X[within_r][within_c]
                elif mode == "h":
                    # Flip horizontally on odd column tiles
                    if (tile_c & 1) == 1:
                        # Odd column - flip horizontally
                        pixel = X[within_r][w - 1 - within_c]
                    else:
                        # Even column - no flip
                        pixel = X[within_r][within_c]
                elif mode == "v":
                    # Flip vertically on odd row tiles
                    if (tile_r & 1) == 1:
                        # Odd row - flip vertically
                        pixel = X[h - 1 - within_r][within_c]
                    else:
                        # Even row - no flip
                        pixel = X[within_r][within_c]
                elif mode == "hv":
                    # Checkerboard pattern - flip on (R^C)&1
                    # When (R^C)&1 == 1:
                    #   - If R is even: flip horizontally
                    #   - If R is odd: flip vertically
                    if ((tile_r ^ tile_c) & 1) == 1:
                        if (tile_r & 1) == 0:
                            # Even row, odd column - flip horizontally
                            pixel = X[within_r][w - 1 - within_c]
                        else:
                            # Odd row, even column - flip vertically
                            pixel = X[h - 1 - within_r][within_c]
                    else:
                        # XOR is even - no flip
                        pixel = X[within_r][within_c]
                else:
                    raise ValueError(f"Unknown mode: {mode}. Must be one of ['none', 'h', 'v', 'hv']")

                row.append(pixel)
            result.append(row)

        return result
