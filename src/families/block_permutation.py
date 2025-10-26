"""
BlockPermutation Family for ARC AGI solver.

Global exact family: Deterministic tile reordering (uniform block grid).
Learns ONE (kH, kW, perm) from first training pair that works for ALL pairs.
kH, kW = uniform block tile sizes (height, width)
perm = dict mapping source tile position (r,c) to destination position (R,C)
Tiles enumerated row-major: tile_index = r * ncols_blocks + c

Critical constraint: dims(X) == dims(Y) (no resizing)
"""

from src.utils import dims, deep_eq


class BlockPermutationFamily:
    """
    Global exact family: Deterministic tile reordering (uniform block grid).

    Learns ONE (kH, kW, perm) from first training pair that works for ALL pairs.
    Partitions grid into uniform (kH×kW) tiles and finds permutation that reorders X→Y.

    Invariants:
        - FY: Accept params only if they reproduce ALL train pairs exactly
        - Shape preservation: dims(X) == dims(Y) for all pairs
        - Integer tiling: h%kH==0 and w%kW==0
        - Multiset equality: X and Y have same tiles, just reordered
        - Lexicographic search: try (kH, kW) in lex order, first-match wins
        - Greedy matching: deterministic row-major tile matching
        - Purity: No mutations; fresh grid allocation
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            kH: None initially; int after successful fit() (block height)
            kW: None initially; int after successful fit() (block width)
            perm: None initially; dict after successful fit() {(src_r, src_c): (dst_r, dst_c)}
        """
        self.name = "BlockPermutation"
        self.params = type('Params', (), {
            'kH': None,
            'kW': None,
            'perm': None,
        })()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Learn ONE (kH, kW, perm) from first pair that works for ALL pairs.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair and verify shape equality
            3. Try all (kH, kW) in lexicographic order
            4. For each valid (kH, kW), build greedy permutation
            5. Verify on all training pairs
            6. Store first valid params and return True

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found params that satisfy FY on all pairs; False otherwise

        Determinism:
            - Always learn from first pair
            - Block size search is lexicographic
            - Greedy matching is row-major

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

        # Check shape equality (no resizing)
        if hx != hy or wx != wy:
            return False

        h, w = hx, wx

        # Check for empty dimensions
        if h == 0 or w == 0:
            return False

        # Try all possible block sizes in lexicographic order
        for kH in range(1, h + 1):
            # Skip non-integer vertical tiling
            if h % kH != 0:
                continue

            for kW in range(1, w + 1):
                # Skip non-integer horizontal tiling
                if w % kW != 0:
                    continue

                # Try this block size
                result = self._try_block_size(train_pairs, kH, kW)
                if result is not None:
                    # Found valid params - store and return
                    self.params.kH = kH
                    self.params.kW = kW
                    self.params.perm = result
                    return True

        # No valid (kH, kW, perm) found
        return False

    def _try_block_size(self, train_pairs: list[dict], kH: int, kW: int) -> dict | None:
        """
        Try specific block size and build permutation.

        Args:
            train_pairs: list of training pairs
            kH: block height
            kW: block width

        Returns:
            perm dict if this block size works for ALL pairs; None otherwise
        """
        # Extract tiles from first pair
        X0 = train_pairs[0]["input"]
        Y0 = train_pairs[0]["output"]

        TX = self._extract_tiles(X0, kH, kW)
        TY = self._extract_tiles(Y0, kH, kW)

        # Check multiset equality
        X_tiles_flat = [tile for row in TX for tile in row]
        Y_tiles_flat = [tile for row in TY for tile in row]

        if sorted(X_tiles_flat) != sorted(Y_tiles_flat):
            return None  # Tiles don't match

        # Build permutation using greedy matching
        perm = self._build_permutation(TX, TY)
        if perm is None:
            return None  # Could not build complete permutation

        # Store tile grid dimensions from first pair (Bug B3 fix)
        nrows_first = len(TX)
        ncols_first = len(TX[0]) if TX else 0

        # Verify on ALL training pairs
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Check shape equality
            hx, wx = dims(X)
            hy, wy = dims(Y)
            if hx != hy or wx != wy:
                return None

            # Check dimensions match block size
            if hx % kH != 0 or wx % kW != 0:
                return None

            # Bug B3 fix: Check tile grid dimensions match first pair
            nrows_current = hx // kH
            ncols_current = wx // kW
            if nrows_current != nrows_first or ncols_current != ncols_first:
                return None  # Tile grid size mismatch

            # Apply permutation and check
            Y_pred = self._apply_with_params(X, kH, kW, perm)
            if not deep_eq(Y_pred, Y):
                return None  # FY violation

        # All pairs match
        return perm

    def _build_permutation(self, TX: list[list[tuple]], TY: list[list[tuple]]) -> dict | None:
        """
        Build permutation using greedy row-major matching.

        Args:
            TX: source tiles grid
            TY: destination tiles grid

        Returns:
            perm dict {(src_r, src_c): (dst_r, dst_c)} if complete; None if incomplete
        """
        nrows = len(TX)
        ncols = len(TX[0]) if nrows > 0 else 0

        # Track which source tiles have been used
        used = [[False] * ncols for _ in range(nrows)]

        perm = {}

        # For each destination tile (row-major order)
        for dst_r in range(nrows):
            for dst_c in range(ncols):
                target_tile = TY[dst_r][dst_c]

                # Find first unused matching source tile (row-major order)
                found = False
                for src_r in range(nrows):
                    for src_c in range(ncols):
                        if not used[src_r][src_c] and TX[src_r][src_c] == target_tile:
                            # Match found
                            used[src_r][src_c] = True
                            perm[(src_r, src_c)] = (dst_r, dst_c)
                            found = True
                            break
                    if found:
                        break

                if not found:
                    # Could not find match for this dest tile
                    return None

        return perm

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply tile permutation using learned parameters to input X.

        Args:
            X: Input grid

        Returns:
            Grid where tiles are reordered according to learned permutation
            Output has same dimensions as input

        Raises:
            RuntimeError: If params not set (must call fit() first)

        Determinism:
            - Same (X, params) always yields same output

        Purity:
            - Never mutates X
            - Returns new grid
        """
        if self.params.kH is None:
            raise RuntimeError("BlockPermutationFamily.apply() called before fit(). params.kH is None. Must call fit() with train_pairs first.")

        if self.params.kW is None:
            raise RuntimeError("BlockPermutationFamily.apply() called before fit(). params.kW is None. Must call fit() with train_pairs first.")

        if self.params.perm is None:
            raise RuntimeError("BlockPermutationFamily.apply() called before fit(). params.perm is None. Must call fit() with train_pairs first.")

        # Use helper method
        return self._apply_with_params(X, self.params.kH, self.params.kW, self.params.perm)

    def _apply_with_params(self, X: list[list[int]], kH: int, kW: int, perm: dict) -> list[list[int]]:
        """
        Helper: apply permutation with given params.

        Args:
            X: Input grid
            kH: block height
            kW: block width
            perm: permutation dict

        Returns:
            Grid with tiles reordered according to perm
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Create output grid (start with copy of input)
        result = [row[:] for row in X]

        # Apply each tile mapping
        for (src_r, src_c), (dst_r, dst_c) in perm.items():
            # Extract source tile
            src_tile = []
            for i in range(kH):
                row_idx = src_r * kH + i
                row = []
                for j in range(kW):
                    col_idx = src_c * kW + j
                    row.append(X[row_idx][col_idx])
                src_tile.append(row)

            # Copy source tile to destination position
            for i in range(kH):
                dst_row_idx = dst_r * kH + i
                for j in range(kW):
                    dst_col_idx = dst_c * kW + j
                    result[dst_row_idx][dst_col_idx] = src_tile[i][j]

        return result

    def _extract_tiles(self, X: list[list[int]], kH: int, kW: int) -> list[list[tuple]]:
        """
        Extract tile blocks from grid X using block size (kH, kW).

        Args:
            X: Input grid
            kH: Block height
            kW: Block width

        Returns:
            2D list of tiles: tiles[r][c] = tuple-of-tuples for tile at (r,c)
            Each tile is a kH × kW block from X

        Tile Format:
            - Each tile is tuple of tuples: ((row0), (row1), ..., (rowK-1))
            - This ensures tiles are hashable and comparable

        Determinism:
            - Row-major iteration

        Purity:
            - Never mutates X
            - Returns new data structure
        """
        if not X:
            return []

        h, w = dims(X)

        nrows = h // kH
        ncols = w // kW

        tiles = []

        for R in range(nrows):
            tile_row = []
            for C in range(ncols):
                # Extract block from X[R*kH:(R+1)*kH, C*kW:(C+1)*kW]
                tile_block = []
                for i in range(kH):
                    row_idx = R * kH + i
                    row = []
                    for j in range(kW):
                        col_idx = C * kW + j
                        row.append(X[row_idx][col_idx])
                    tile_block.append(tuple(row))

                # Convert to tuple-of-tuples
                tile = tuple(tile_block)
                tile_row.append(tile)

            tiles.append(tile_row)

        return tiles
