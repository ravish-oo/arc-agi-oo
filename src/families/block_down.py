"""
BlockDown Family for ARC AGI solver.

Global exact family: Uniform block downsampling with deterministic aggregation.
Learns integer block size (kH, kW) and reducer from first training pair.
Input partitioned into kH×kW blocks; each block reduces to single output pixel.
Five reducers tried in order: {center, majority, min, max, first_nonzero}.
"""

from collections import Counter
from src.utils import dims


class BlockDownFamily:
    """
    Global exact family: Uniform block downsampling with deterministic aggregation.

    Learns integer block size (kH, kW) and reducer from first training pair.
    Input is partitioned into kH×kW blocks; each block reduces to single output pixel.
    Five reducers tried in order: {center, majority, min, max, first_nonzero}.
    Accepts ONLY if one reducer reproduces every training pair exactly (FY).

    Invariants:
        - FY: Accept (kH, kW, reducer) only if it reproduces ALL train pairs exactly
        - Unified: ONE (kH, kW, reducer) tuple for all pairs (no per-pair parameters)
        - Integer downsampling: kH and kW must be positive integers (exact division)
        - Deterministic reducer order: try {center, majority, min, max, first_nonzero} in order
        - Deterministic tie-breaking: majority uses smallest color on tie
        - Shape exactness: dims(Y) = (R//kH, C//kW) where (R,C) = dims(X)
        - Purity: No mutations to inputs
    """

    ALLOWED_REDUCERS = ["center", "majority", "min", "max", "first_nonzero"]

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            kH: None initially; set to vertical block height (int > 0) after successful fit()
            kW: None initially; set to horizontal block width (int > 0) after successful fit()
            reducer: None initially; set to one of ALLOWED_REDUCERS after successful fit()
        """
        self.name = "BlockDown"
        self.params = type('Params', (), {'kH': None, 'kW': None, 'reducer': None})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Feasibility fit for Step-2 architecture.

        In Step-2, BlockDown is a preprocessing step combined with Φ/GLUE.
        This method learns consistent integer block sizes (kH, kW) across all
        training pairs based on dimension ratios. It uses a default reducer
        ("center") without verifying FY - that's handled by P + Φ/GLUE composition.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair (X0, Y0) and compute dims
            3. Compute candidate block sizes: kH = R0 // RY0, kW = C0 // CY0
            4. Verify integer division (exact multiple required)
            5. Verify same (kH, kW) dimension ratios hold for all remaining pairs
            6. Store params with default reducer="center" and return True

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found consistent (kH, kW) dimension ratios; False otherwise

        Edge cases:
            - Empty train_pairs: return False
            - Empty grids: return False (cannot infer block size)
            - Identity downsampling (kH=kW=1): accept (valid feasibility)
            - Non-integer ratio: return False
            - Upsampling (Y larger than X): return False
            - Inconsistent block sizes across pairs: return False

        Determinism:
            - Block sizes learned from first pair only
            - Default reducer is always "center"
            - Verification order is train_pairs[1:] (stable)

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting params

        Step-2 Contract:
            - Feasibility check (consistent dimension ratios)
            - Does NOT require apply(X) == Y
            - Uses default reducer (Φ/GLUE will handle pixel-level matching)
            - FY constraint enforced at candidate level
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Extract first pair
        first_pair = train_pairs[0]
        X0 = first_pair["input"]
        Y0 = first_pair["output"]

        # Handle empty grid edge case
        if not X0 or not Y0:
            return False

        # Get dimensions
        R0, C0 = dims(X0)
        RY0, CY0 = dims(Y0)

        # Check for zero dimensions
        if R0 == 0 or C0 == 0 or RY0 == 0 or CY0 == 0:
            return False

        # Check for upsampling (BlockDown is for downsampling only)
        if RY0 > R0 or CY0 > C0:
            return False  # Use PixelReplicate for upsampling

        # Compute candidate block sizes
        # Check if X dimensions are exact multiples of Y dimensions
        if R0 % RY0 != 0 or C0 % CY0 != 0:
            return False  # Non-integer block size

        kH_candidate = R0 // RY0
        kW_candidate = C0 // CY0

        # Verify positive block sizes
        if kH_candidate <= 0 or kW_candidate <= 0:
            return False

        # Verify same (kH, kW) dimension ratios hold for all remaining pairs
        for pair in train_pairs[1:]:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grid in later pair
            if not X or not Y:
                return False

            R, C = dims(X)
            RY, CY = dims(Y)

            # Check for zero dimensions
            if R == 0 or C == 0 or RY == 0 or CY == 0:
                return False

            # Check for upsampling
            if RY > R or CY > C:
                return False

            # Check integer division
            if R % RY != 0 or C % CY != 0:
                return False

            kH_check = R // RY
            kW_check = C // CY

            # Verify same block sizes
            if kH_check != kH_candidate or kW_check != kW_candidate:
                return False  # Inconsistent block sizes

        # Consistent block sizes found - accept as feasible
        # Use default reducer (Φ/GLUE will verify pixel-level correctness)
        self.params.kH = kH_candidate
        self.params.kW = kW_candidate
        self.params.reducer = "center"  # Default reducer for Step-2
        return True

    def _try_reducer(self, train_pairs: list[dict], kH: int, kW: int, reducer: str) -> bool:
        """
        Test if a specific reducer satisfies FY on ALL training pairs.

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts
            kH: vertical block height
            kW: horizontal block width
            reducer: one of ALLOWED_REDUCERS

        Returns:
            True if this reducer produces exact match for all pairs; False otherwise
        """
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Compute predicted output using this reducer
            Y_predicted = self._downsample_with_reducer(X, kH, kW, reducer)

            # Check bit-for-bit equality
            if Y_predicted != Y:
                return False  # FY violation

        return True  # This reducer works for ALL pairs

    def _downsample_with_reducer(self, X: list[list[int]], kH: int, kW: int, reducer: str) -> list[list[int]]:
        """
        Helper: downsample X with given block size and reducer.

        Args:
            X: Input grid
            kH: Vertical block height
            kW: Horizontal block width
            reducer: one of ALLOWED_REDUCERS

        Returns:
            Grid where each kH×kW block is reduced to single pixel
        """
        if not X:
            return []

        R, C = dims(X)

        # Check dimensions are compatible
        if R % kH != 0 or C % kW != 0:
            raise ValueError(f"Input dimensions ({R}, {C}) not divisible by block size ({kH}, {kW})")

        # Compute output dimensions
        R_out = R // kH
        C_out = C // kW

        result = []

        # For each output position
        for r_out in range(R_out):
            output_row = []
            for c_out in range(C_out):
                # Extract block
                r_start = r_out * kH
                r_end = (r_out + 1) * kH
                c_start = c_out * kW
                c_end = (c_out + 1) * kW

                block = [row[c_start:c_end] for row in X[r_start:r_end]]

                # Apply reducer
                pixel_value = self._aggregate_block(block, reducer)
                output_row.append(pixel_value)

            result.append(output_row)

        return result

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply uniform block downsampling with learned reducer to input X.

        Algorithm:
            1. Check params.kH, params.kW, params.reducer are set
            2. Compute input dims (R, C)
            3. Create output grid with dims (R//kH, C//kW)
            4. For each output position (r_out, c_out):
                - Extract block from X
                - Apply reducer to get output pixel
            5. Return output grid

        Args:
            X: Input grid

        Returns:
            Grid where each kH×kW block is reduced to single pixel via params.reducer
            Output shape: (R//kH, C//kW) where (R,C) = dims(X)

        Raises:
            RuntimeError: If params.kH, params.kW, or params.reducer is None (must call fit() first)
            ValueError: If X dimensions not divisible by (kH, kW)

        Edge cases:
            - Empty grid []: returns []
            - Single block: reduces to single pixel
            - Identity downsampling (kH=kW=1): returns deep copy of X

        Determinism:
            - Same (X, kH, kW, reducer) always yields same output
            - Iteration order is row-major (deterministic)

        Purity:
            - Never mutates X
            - Returns new grid (no row aliasing)
        """
        if self.params.kH is None:
            raise RuntimeError("BlockDownFamily.apply() called before fit(). params.kH is None. Must call fit() with train_pairs first.")

        if self.params.kW is None:
            raise RuntimeError("BlockDownFamily.apply() called before fit(). params.kW is None. Must call fit() with train_pairs first.")

        if self.params.reducer is None:
            raise RuntimeError("BlockDownFamily.apply() called before fit(). params.reducer is None. Must call fit() with train_pairs first.")

        # Use helper method for downsampling
        return self._downsample_with_reducer(X, self.params.kH, self.params.kW, self.params.reducer)

    def _aggregate_block(self, block: list[list[int]], reducer: str) -> int:
        """
        Apply reducer to kH×kW block; return single pixel value.

        Args:
            block: kH×kW grid (list of lists)
            reducer: one of {"center", "majority", "min", "max", "first_nonzero"}

        Returns:
            Single integer color value

        Reducers:
            - "center": value at block center position [kH//2][kW//2]
            - "majority": most frequent color; tie → smallest color
            - "min": minimum color value in block
            - "max": maximum color value in block
            - "first_nonzero": first nonzero in row-major scan; all zeros → 0

        Determinism:
            - All reducers are deterministic
            - Majority tie-breaking uses smallest color
            - first_nonzero uses row-major scan

        Purity:
            - Never mutates block
            - Returns single integer
        """
        if reducer == "center":
            # Center position using floor division
            kH = len(block)
            kW = len(block[0]) if block else 0
            return block[kH // 2][kW // 2]

        elif reducer == "majority":
            # Flatten block to 1D list
            flat = [pixel for row in block for pixel in row]

            # Count occurrences
            counts = Counter(flat)

            # Find maximum count
            max_count = max(counts.values())

            # Collect all colors with max count (ties)
            tied_colors = [color for color, count in counts.items() if count == max_count]

            # Return smallest color (deterministic tie-breaking)
            return min(tied_colors)

        elif reducer == "min":
            # Flatten and return minimum
            flat = [pixel for row in block for pixel in row]
            return min(flat)

        elif reducer == "max":
            # Flatten and return maximum
            flat = [pixel for row in block for pixel in row]
            return max(flat)

        elif reducer == "first_nonzero":
            # Flatten to row-major list
            flat = [pixel for row in block for pixel in row]

            # Find first nonzero
            for pixel in flat:
                if pixel != 0:
                    return pixel

            # All zeros
            return 0

        else:
            raise ValueError(f"Unknown reducer: {reducer}. Must be one of {self.ALLOWED_REDUCERS}")
