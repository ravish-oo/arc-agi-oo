"""
PixelReplicate Family for ARC AGI solver.

Global exact family: Uniform pixel upsampling (kH × kW).
Learns integer scaling factors (kH, kW) from first training pair.
Each input pixel X[r][c] is replicated into a kH×kW block in output.
"""

from src.utils import dims


class PixelReplicateFamily:
    """
    Global exact family: Uniform pixel upsampling (kH × kW).

    Learns integer scaling factors (kH, kW) from first training pair.
    Each input pixel X[r][c] is replicated into a kH×kW block in output.
    Accepts ONLY if this scaling reproduces every training pair exactly (FY).

    Invariants:
        - FY: Accept (kH, kW) only if it reproduces ALL train pairs exactly
        - Unified: ONE (kH, kW) pair for all pairs (no per-pair scaling)
        - Integer scaling: kH and kW must be positive integers (exact division)
        - Shape exactness: dims(Y) = (R*kH, C*kW) where (R,C) = dims(X)
        - Purity: No mutations to inputs
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            kH: None initially; set to vertical scaling factor (int > 0) after successful fit()
            kW: None initially; set to horizontal scaling factor (int > 0) after successful fit()
        """
        self.name = "PixelReplicate"
        self.params = type('Params', (), {'kH': None, 'kW': None})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Learn ONE (kH, kW) pair that works for ALL train pairs.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair (X0, Y0) and compute dims
            3. Handle edge cases (empty dimensions)
            4. Compute candidate scaling factors: kH = RY0 / R0, kW = CY0 / C0
            5. Verify integer division (modulo check)
            6. Verify same (kH, kW) works for all remaining pairs
            7. Verify FY on ALL pairs (bit-for-bit equality)
            8. Store params and return True

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found (kH, kW) that satisfies FY on all pairs; False otherwise

        Edge cases:
            - Empty train_pairs: return False
            - Empty grids: handled (check for zero dimensions)
            - Identity scaling (kH=kW=1): pass-through
            - Non-integer ratio: return False
            - Inconsistent scaling across pairs: return False
            - Zero or negative dimensions: return False

        Determinism:
            - Scaling factors learned from first pair only
            - Verification order is train_pairs[1:] (stable)
            - Integer division check is deterministic

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

        # Handle empty grid edge case
        if not X0:
            if not Y0:
                # Both empty - accept with undefined kH/kW (special case)
                # Actually, we can't define kH/kW for empty grids, so return False
                return False
            else:
                # X empty but Y non-empty - impossible
                return False

        # Get dimensions
        R0, C0 = dims(X0)
        RY0, CY0 = dims(Y0)

        # Check for zero dimensions (shouldn't happen after empty check, but be safe)
        if R0 == 0 or C0 == 0:
            return False

        # Compute candidate scaling factors
        # Check if Y dimensions are exact multiples of X dimensions
        if RY0 % R0 != 0 or CY0 % C0 != 0:
            return False  # Non-integer scaling

        kH_candidate = RY0 // R0
        kW_candidate = CY0 // C0

        # Verify positive scaling factors
        if kH_candidate <= 0 or kW_candidate <= 0:
            return False

        # Verify same (kH, kW) works for all remaining pairs
        for pair in train_pairs[1:]:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grid in later pair
            if not X:
                if not Y:
                    continue  # Both empty, skip
                else:
                    return False  # X empty but Y non-empty

            R, C = dims(X)
            RY, CY = dims(Y)

            # Check for zero dimensions
            if R == 0 or C == 0:
                return False

            # Check integer division
            if RY % R != 0 or CY % C != 0:
                return False

            kH_check = RY // R
            kW_check = CY // C

            # Verify same scaling factors
            if kH_check != kH_candidate or kW_check != kW_candidate:
                return False  # Inconsistent scaling

        # Verify FY on ALL pairs (bit-for-bit equality)
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grid edge case
            if not X:
                if not Y:
                    continue  # Both empty, skip
                else:
                    return False  # X empty but Y non-empty

            # Compute predicted output using candidate factors
            Y_predicted = self._replicate_with_factors(X, kH_candidate, kW_candidate)

            # Check bit-for-bit equality
            if Y_predicted != Y:
                return False  # FY violation

        # All pairs have consistent scaling factors AND FY exactness verified
        # Store and return success
        self.params.kH = kH_candidate
        self.params.kW = kW_candidate
        return True

    def _replicate_with_factors(self, X: list[list[int]], kH: int, kW: int) -> list[list[int]]:
        """
        Helper: replicate X with given factors without requiring params to be set.

        Args:
            X: Input grid
            kH: Vertical scaling factor
            kW: Horizontal scaling factor

        Returns:
            Grid where each pixel X[r][c] is replicated into kH×kW block
        """
        if not X:
            return []

        R, C = dims(X)
        result = []

        for r in range(R):
            for dr in range(kH):
                output_row = []
                for c in range(C):
                    pixel = X[r][c]
                    for dc in range(kW):
                        output_row.append(pixel)
                result.append(output_row)

        return result

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply uniform pixel replication to input X.

        Algorithm:
            1. Check params.kH and params.kW are set
            2. Compute input dims (R, C)
            3. Create output grid with dims (R*kH, C*kW)
            4. For each input pixel (r, c):
                - Fill output block [r*kH:(r+1)*kH] × [c*kW:(c+1)*kW] with X[r][c]
            5. Return output grid

        Args:
            X: Input grid

        Returns:
            Grid where each pixel X[r][c] is replicated into kH×kW block
            Output shape: (R*kH, C*kW) where (R,C) = dims(X)

        Raises:
            RuntimeError: If params.kH or params.kW is None (must call fit() first)

        Edge cases:
            - Empty grid []: returns []
            - Single pixel: with kH=kW=2 returns 2×2 block
            - Identity scaling (kH=kW=1): returns deep copy of X
            - Non-uniform scaling (kH≠kW): each pixel becomes kH×kW rectangle

        Determinism:
            - Same (X, kH, kW) always yields same output
            - Iteration order is row-major (deterministic)

        Purity:
            - Never mutates X
            - Returns new grid (no row aliasing)
        """
        if self.params.kH is None:
            raise RuntimeError("PixelReplicateFamily.apply() called before fit(). params.kH is None. Must call fit() with train_pairs first.")

        if self.params.kW is None:
            raise RuntimeError("PixelReplicateFamily.apply() called before fit(). params.kW is None. Must call fit() with train_pairs first.")

        # Use helper method for replication
        return self._replicate_with_factors(X, self.params.kH, self.params.kW)
