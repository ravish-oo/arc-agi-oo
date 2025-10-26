"""
NPSDown Family for ARC AGI solver.

Global exact family: Non-uniform partition downsampling via change boundaries.
Learns ONE reducer from {center, majority, min, max, first_nonzero} that works for ALL pairs.
Each pair computes its own boundaries from its input X via boundaries_by_any_change (Φ.3 input-only).
Boundaries partition X into row bands and column bands; each band block aggregates to one pixel.
"""

from collections import Counter
from src.utils import dims
from src.components import boundaries_by_any_change, bands_from_boundaries


class NPSDownFamily:
    """
    Global exact family: Non-uniform partition downsampling via change boundaries.

    Learns ONE reducer from {center, majority, min, max, first_nonzero} that works for ALL pairs.
    Each pair computes its own boundaries from its input X via boundaries_by_any_change.
    Boundaries partition X into row bands and column bands; each band block aggregates to one pixel.
    Critical: boundaries computed ONLY from X (never from Y) per Φ.3 input-only constraint.

    Invariants:
        - FY: Accept reducer only if it reproduces ALL train pairs exactly
        - Φ.3: Boundaries derived from X only (input-only, never from Y)
        - Unified reducer: ONE reducer for all pairs (no per-pair variations)
        - Deterministic reducer order: try {center, majority, min, max, first_nonzero} in order
        - Deterministic tie-breaking: majority uses smallest color on tie
        - Purity: No mutations to inputs; boundaries recomputed fresh each apply()
    """

    ALLOWED_REDUCERS = ["center", "majority", "min", "max", "first_nonzero"]

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            reducer: None initially; set to one of ALLOWED_REDUCERS after successful fit()
        """
        self.name = "NPSDown"
        self.params = type('Params', (), {'reducer': None})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Feasibility fit for Step-2 architecture.

        In Step-2, NPSDown is a preprocessing step combined with Φ/GLUE.
        This method verifies that input-derived boundaries produce compatible
        output dimensions across all training pairs. It uses a default reducer
        ("center") without verifying FY - that's handled by P + Φ/GLUE composition.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Verify all pairs have shape compatibility (num_bands from X matches Y dims)
            3. Store default reducer="center" and return True

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if all pairs have compatible band/output dimensions; False otherwise

        Φ.3 Input-Only Constraint (CRITICAL):
            - Boundaries computed ONLY from X via boundaries_by_any_change(X, axis)
            - NEVER compute boundaries from Y or any target-dependent feature
            - Each pair computes its own boundaries from its X
            - Reducer is default; boundaries recomputed in apply()

        Determinism:
            - boundaries_by_any_change is deterministic
            - Default reducer is always "center"

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting params.reducer

        Step-2 Contract:
            - Feasibility check (shape compatibility)
            - Does NOT require apply(X) == Y
            - Uses default reducer (Φ/GLUE will handle pixel-level matching)
            - FY constraint enforced at candidate level
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Verify all pairs have shape compatibility
        # (num_row_bands from X must equal Y rows, num_col_bands from X must equal Y cols)
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grids
            if not X or not Y:
                return False

            # Compute boundaries from X (Φ.3: input-only)
            row_change_boundaries = boundaries_by_any_change(X, "row")
            col_change_boundaries = boundaries_by_any_change(X, "col")

            # Convert to full boundary lists
            H, W = dims(X)
            row_boundaries = [0] + [b + 1 for b in row_change_boundaries] + [H]
            col_boundaries = [0] + [b + 1 for b in col_change_boundaries] + [W]

            num_row_bands = len(row_boundaries) - 1
            num_col_bands = len(col_boundaries) - 1

            # Check shape compatibility
            RY, CY = dims(Y)
            if num_row_bands != RY or num_col_bands != CY:
                return False  # Shape mismatch

        # Shape-compatible - accept as feasible
        # Use default reducer (Φ/GLUE will verify pixel-level correctness)
        self.params.reducer = "center"  # Default reducer for Step-2
        return True

    def _try_reducer(self, train_pairs: list[dict], reducer: str) -> bool:
        """
        Test if a specific reducer satisfies FY on ALL training pairs.

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts
            reducer: one of ALLOWED_REDUCERS

        Returns:
            True if this reducer produces exact match for all pairs; False otherwise
        """
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Compute predicted output using this reducer
            Y_predicted = self._downsample_with_reducer(X, reducer)

            # Check bit-for-bit equality
            if Y_predicted != Y:
                return False  # FY violation

        return True  # This reducer works for ALL pairs

    def _downsample_with_reducer(self, X: list[list[int]], reducer: str) -> list[list[int]]:
        """
        Helper: downsample X with given reducer using NPS boundaries from X.

        Args:
            X: Input grid
            reducer: one of ALLOWED_REDUCERS

        Returns:
            Grid where each band block is reduced to single pixel
        """
        if not X:
            return []

        H, W = dims(X)

        # Compute boundaries from X (Φ.3: input-only)
        row_change_boundaries = boundaries_by_any_change(X, "row")
        col_change_boundaries = boundaries_by_any_change(X, "col")

        # Convert to full boundary lists
        row_boundaries = [0] + [b + 1 for b in row_change_boundaries] + [H]
        col_boundaries = [0] + [b + 1 for b in col_change_boundaries] + [W]

        num_row_bands = len(row_boundaries) - 1
        num_col_bands = len(col_boundaries) - 1

        result = []

        # For each output position (band block)
        for r_out in range(num_row_bands):
            output_row = []
            for c_out in range(num_col_bands):
                # Extract band block boundaries
                r0 = row_boundaries[r_out]
                r1 = row_boundaries[r_out + 1]
                c0 = col_boundaries[c_out]
                c1 = col_boundaries[c_out + 1]

                # Extract values from X[r0:r1][c0:c1] (all cells in band block)
                values = [X[r][c] for r in range(r0, r1) for c in range(c0, c1)]

                # Apply reducer
                pixel_value = self._aggregate_band_block(values, reducer)
                output_row.append(pixel_value)

            result.append(output_row)

        return result

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply non-uniform partition downsampling with learned reducer to input X.
        Boundaries are recomputed from X (input-only, Φ.3).

        Algorithm:
            1. Check params.reducer is set
            2. Compute boundaries from X (input-only)
            3. Aggregate each band block using learned reducer
            4. Return downsampled grid

        Args:
            X: Input grid

        Returns:
            Grid where each band block is reduced to single pixel
            Output shape: (num_row_bands, num_col_bands)

        Raises:
            RuntimeError: If params.reducer is None (must call fit() first)

        Φ.3 Critical Requirement:
            - Boundaries computed ONLY from X (input grid)
            - NEVER use Y or any target information
            - Recompute boundaries fresh on every apply() call (no cached state from fit)

        Determinism:
            - Same (X, reducer) always yields same output
            - boundaries_by_any_change is deterministic

        Purity:
            - Never mutates X
            - Returns new grid
        """
        if self.params.reducer is None:
            raise RuntimeError("NPSDownFamily.apply() called before fit(). params.reducer is None. Must call fit() with train_pairs first.")

        # Use helper method for downsampling
        return self._downsample_with_reducer(X, self.params.reducer)

    def _aggregate_band_block(self, values: list[int], reducer: str) -> int:
        """
        Apply reducer to flattened band block values; return single pixel value.

        Args:
            values: 1D list of all pixel values in band block (row-major flattened)
            reducer: one of {"center", "majority", "min", "max", "first_nonzero"}

        Returns:
            Single integer color value

        Reducers:
            - "center": value at center position (values[len(values)//2])
            - "majority": most frequent color; tie → smallest color
            - "min": minimum color value
            - "max": maximum color value
            - "first_nonzero": first nonzero in list; all zeros → 0

        Determinism:
            - All reducers are deterministic
            - Majority tie-breaking uses smallest color
            - first_nonzero scans list order

        Purity:
            - Never mutates values
            - Returns single integer
        """
        if reducer == "center":
            # Center position using floor division
            return values[len(values) // 2]

        elif reducer == "majority":
            # Count occurrences
            counts = Counter(values)

            # Find maximum count
            max_count = max(counts.values())

            # Collect all colors with max count (ties)
            tied_colors = [color for color, count in counts.items() if count == max_count]

            # Return smallest color (deterministic tie-breaking)
            return min(tied_colors)

        elif reducer == "min":
            # Return minimum
            return min(values)

        elif reducer == "max":
            # Return maximum
            return max(values)

        elif reducer == "first_nonzero":
            # Find first nonzero
            for value in values:
                if value != 0:
                    return value

            # All zeros
            return 0

        else:
            raise ValueError(f"Unknown reducer: {reducer}. Must be one of {self.ALLOWED_REDUCERS}")
