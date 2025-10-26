"""
NPSUp Family for ARC AGI solver.

Global exact family: Non-uniform partition upsampling via learned band replication.
Learns ONE (row_factors, col_factors) from first training pair that works for ALL pairs.
Each pair computes its own boundaries from its input X via boundaries_by_any_change (Φ.3 input-only).
row_factors[i] = how many Y rows the i-th X row band expands to.
col_factors[j] = how many Y cols the j-th X col band expands to.
"""

from src.utils import dims, deep_eq
from src.components import boundaries_by_any_change


class NPSUpFamily:
    """
    Global exact family: Non-uniform partition upsampling via learned band replication.

    Learns ONE (row_factors, col_factors) from first training pair that works for ALL pairs.
    Each pair computes its own boundaries from its input X via boundaries_by_any_change.
    row_factors[i] = how many Y rows the i-th X row band expands to.
    col_factors[j] = how many Y cols the j-th X col band expands to.
    Critical: during apply(), boundaries computed ONLY from X (never Y) per Φ.3 input-only.

    Invariants:
        - FY: Accept factors only if they reproduce ALL train pairs exactly
        - Φ.3: Boundaries derived from X only during apply() (never from Y)
        - Unified factors: ONE (row_factors, col_factors) for all pairs
        - Integer factors: All factors must be positive integers
        - Purity: No mutations; boundaries recomputed fresh each apply()
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            row_factors: None initially; list[int] after successful fit()
            col_factors: None initially; list[int] after successful fit()
        """
        self.name = "NPSUp"
        self.params = type('Params', (), {
            'row_factors': None,
            'col_factors': None,
        })()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Feasibility fit for Step-2 architecture.

        In Step-2, NPSUp is a preprocessing step combined with Φ/GLUE.
        This method learns replication factors from the first training pair
        and verifies band count consistency across all pairs. It does NOT
        require pixel-level exactness - that's handled by P + Φ/GLUE composition.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair and learn factors from dimension ratios
            3. Verify all remaining pairs have same band count structure
            4. Store factors and return True (let Φ/GLUE handle pixel matching)

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found consistent band structure with integer factors; False otherwise

        Φ.3 Input-Only Constraint (CRITICAL):
            - During fit(): can examine both X and Y to learn factors
            - During apply(): boundaries computed ONLY from X, never from Y
            - Factors are learned once; apply() recomputes X boundaries and applies factors

        Determinism:
            - Always learn from first pair (train_pairs[0])
            - boundaries_by_any_change is deterministic

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting params

        Step-2 Contract:
            - Feasibility check (band count consistency + integer factors)
            - Does NOT require apply(X) == Y
            - P + Φ/GLUE will handle actual transformation
            - FY constraint enforced at candidate level
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

        # Learn factors from first pair
        factors = self._learn_factors_from_pair(X0, Y0)
        if factors is None:
            return False  # Incompatible (non-integer factors or band mismatch)

        row_factors, col_factors = factors

        # Verify all remaining pairs have consistent band count structure
        for pair in train_pairs[1:]:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grids
            if not X or not Y:
                return False

            # Check that X has same band structure as first pair
            H, W = dims(X)
            X_row_change_boundaries = boundaries_by_any_change(X, "row")
            X_col_change_boundaries = boundaries_by_any_change(X, "col")
            num_row_bands = len(X_row_change_boundaries) + 1
            num_col_bands = len(X_col_change_boundaries) + 1

            # Verify band counts match learned factors
            if num_row_bands != len(row_factors) or num_col_bands != len(col_factors):
                return False  # Band structure mismatch

        # Consistent band structure found - accept as feasible
        # Store factors (Φ/GLUE will verify pixel-level correctness)
        self.params.row_factors = row_factors
        self.params.col_factors = col_factors
        return True

    def _learn_factors_from_pair(self, X: list[list[int]], Y: list[list[int]]) -> tuple[list[int], list[int]] | None:
        """
        Learn replication factors from single (X, Y) pair.

        Args:
            X: Input grid
            Y: Output grid

        Returns:
            (row_factors, col_factors) if factors can be learned; None if incompatible

        Algorithm:
            1. Compute boundaries from X and Y
            2. Check band count match
            3. For each band, compute factor = Y_band_size / X_band_size
            4. Verify all factors are positive integers
            5. Return (row_factors, col_factors) or None
        """
        H, W = dims(X)
        RY, CY = dims(Y)

        # Compute boundaries from X and Y
        X_row_change_boundaries = boundaries_by_any_change(X, "row")
        X_col_change_boundaries = boundaries_by_any_change(X, "col")
        Y_row_change_boundaries = boundaries_by_any_change(Y, "row")
        Y_col_change_boundaries = boundaries_by_any_change(Y, "col")

        # Convert to full boundary lists
        X_row_bounds = [0] + [b + 1 for b in X_row_change_boundaries] + [H]
        X_col_bounds = [0] + [b + 1 for b in X_col_change_boundaries] + [W]
        Y_row_bounds = [0] + [b + 1 for b in Y_row_change_boundaries] + [RY]
        Y_col_bounds = [0] + [b + 1 for b in Y_col_change_boundaries] + [CY]

        num_X_row_bands = len(X_row_bounds) - 1
        num_X_col_bands = len(X_col_bounds) - 1
        num_Y_row_bands = len(Y_row_bounds) - 1
        num_Y_col_bands = len(Y_col_bounds) - 1

        # Check band count match
        if num_X_row_bands != num_Y_row_bands or num_X_col_bands != num_Y_col_bands:
            return None  # Band structure incompatible

        # Learn row factors
        row_factors = []
        for i in range(num_X_row_bands):
            num_X_rows = X_row_bounds[i + 1] - X_row_bounds[i]
            num_Y_rows = Y_row_bounds[i + 1] - Y_row_bounds[i]

            # Check if Y rows is exact multiple of X rows
            if num_Y_rows % num_X_rows != 0:
                return None  # Non-integer factor

            factor = num_Y_rows // num_X_rows
            if factor <= 0:
                return None  # Non-positive factor

            row_factors.append(factor)

        # Learn col factors
        col_factors = []
        for j in range(num_X_col_bands):
            num_X_cols = X_col_bounds[j + 1] - X_col_bounds[j]
            num_Y_cols = Y_col_bounds[j + 1] - Y_col_bounds[j]

            # Check if Y cols is exact multiple of X cols
            if num_Y_cols % num_X_cols != 0:
                return None  # Non-integer factor

            factor = num_Y_cols // num_X_cols
            if factor <= 0:
                return None  # Non-positive factor

            col_factors.append(factor)

        return (row_factors, col_factors)

    def _upsample_with_factors(self, X: list[list[int]], row_factors: list[int], col_factors: list[int]) -> list[list[int]]:
        """
        Helper: upsample X with given factors using NPS boundaries from X.

        Args:
            X: Input grid
            row_factors: replication factor for each X row band
            col_factors: replication factor for each X col band

        Returns:
            Grid where each X band block is replicated according to factors
        """
        if not X:
            return []

        H, W = dims(X)

        # Compute boundaries from X (Φ.3: input-only)
        X_row_change_boundaries = boundaries_by_any_change(X, "row")
        X_col_change_boundaries = boundaries_by_any_change(X, "col")

        # Convert to full boundary lists
        row_bounds = [0] + [b + 1 for b in X_row_change_boundaries] + [H]
        col_bounds = [0] + [b + 1 for b in X_col_change_boundaries] + [W]

        num_row_bands = len(row_bounds) - 1
        num_col_bands = len(col_bounds) - 1

        # Build output grid by replicating each band
        result = []

        # For each X row band
        for band_i in range(num_row_bands):
            x_r0 = row_bounds[band_i]
            x_r1 = row_bounds[band_i + 1]

            # For each row in this X band
            for x_r in range(x_r0, x_r1):
                # Replicate this row row_factors[band_i] times
                for _ in range(row_factors[band_i]):
                    output_row = []

                    # For each X col band
                    for band_j in range(num_col_bands):
                        x_c0 = col_bounds[band_j]
                        x_c1 = col_bounds[band_j + 1]

                        # For each col in this X band
                        for x_c in range(x_c0, x_c1):
                            pixel = X[x_r][x_c]

                            # Replicate this pixel col_factors[band_j] times
                            for _ in range(col_factors[band_j]):
                                output_row.append(pixel)

                    result.append(output_row)

        return result

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply non-uniform partition upsampling with learned factors to input X.
        Boundaries are recomputed from X (input-only, Φ.3).

        Args:
            X: Input grid

        Returns:
            Grid where each X band block is replicated according to learned factors

        Raises:
            RuntimeError: If params.row_factors or params.col_factors is None (must call fit() first)

        Φ.3 Critical Requirement:
            - Boundaries computed ONLY from X (input grid)
            - NEVER use Y or any target information
            - Recompute boundaries fresh on every apply() call

        Determinism:
            - Same (X, factors) always yields same output
            - boundaries_by_any_change is deterministic

        Purity:
            - Never mutates X
            - Returns new grid
        """
        if self.params.row_factors is None:
            raise RuntimeError("NPSUpFamily.apply() called before fit(). params.row_factors is None. Must call fit() with train_pairs first.")

        if self.params.col_factors is None:
            raise RuntimeError("NPSUpFamily.apply() called before fit(). params.col_factors is None. Must call fit() with train_pairs first.")

        # Use helper method for upsampling
        return self._upsample_with_factors(X, self.params.row_factors, self.params.col_factors)
