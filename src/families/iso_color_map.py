"""
IsoColorMap Family for ARC AGI solver.

Global exact family: Isometry + per-color remapping.
Jointly learns one σ ∈ D8 and one color mapping dict such that applying
both sequentially (σ first, then recolor) transforms every train X into Y exactly (FY).
"""

from src.utils import dims
from src.canonicalization import all_isometries, apply_isometry


class IsoColorMapFamily:
    """
    Global exact family: Isometry + per-color remapping.

    Tries all σ ∈ D8 in deterministic all_isometries() order.
    For each σ: learns color mapping from first (apply_isometry(X, σ), Y) pair.
    Accepts FIRST (σ, mapping) where both σ and mapping satisfy FY on ALL train pairs.

    Invariants:
        - FY: Accept (σ, mapping) only if it reproduces ALL train pairs exactly
        - Unified: ONE (σ, mapping) pair for all pairs (no per-pair variations)
        - Determinism: Fixed σ iteration order, first-acceptable wins
        - Composition: σ applied FIRST, then mapping (order matters)
        - No guessing: apply() raises KeyError for unseen colors
        - Purity: No mutations to inputs
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            sigma: None initially; set to σ name (str) after successful fit()
            mapping: None initially; set to dict[int, int] after successful fit()
        """
        self.name = "IsoColorMap"
        self.params = type('Params', (), {'sigma': None, 'mapping': None})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Find ONE (σ, mapping) pair that works for ALL train pairs.

        Algorithm:
            1. If train_pairs is empty: return False
            2. For each σ in all_isometries() (deterministic order):
                3. Apply σ to all training inputs: X_sigma_i = apply_isometry(X_i, σ)
                4. Build candidate mapping from first pair (X_sigma_0, Y_0):
                    - If dims(X_sigma_0) ≠ dims(Y_0): skip this σ (shape mismatch)
                    - For each position: learn mapping[X_sigma_0[r][c]] = Y_0[r][c]
                    - If conflict in first pair: skip this σ
                5. Verify mapping on ALL remaining pairs:
                    - For each pair: apply σ, check mapping works, check for conflicts
                6. If all pairs passed: store (σ, mapping) and return True
            7. Return False (no (σ, mapping) worked for all pairs)

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found (σ, mapping) that satisfies FY on all pairs; False otherwise

        Edge cases:
            - Empty train_pairs: return False
            - Single pair: accept first matching (σ, mapping)
            - Identity case (X==Y): σ="id", mapping={c:c for all c}
            - Isometry only: σ!=id, mapping={c:c}
            - ColorMap only: σ="id", mapping!=identity
            - Mixed transforms: return False if no unified (σ, mapping) works

        Determinism:
            - all_isometries() order is fixed
            - First-acceptable (σ, mapping) wins
            - Mapping construction uses row-major scan

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting params
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Try each σ in deterministic D8 order
        for sigma in all_isometries():
            # Try to learn mapping for this σ
            mapping = self._try_sigma(sigma, train_pairs)

            if mapping is not None:
                # Found a working (σ, mapping) pair!
                self.params.sigma = sigma
                self.params.mapping = mapping
                return True

        # No (σ, mapping) worked for all pairs
        return False

    def _try_sigma(self, sigma: str, train_pairs: list[dict]) -> dict[int, int] | None:
        """
        Try a specific σ: learn mapping from first pair, verify on all pairs.

        Args:
            sigma: Isometry name (from all_isometries())
            train_pairs: Training pairs

        Returns:
            mapping dict if this σ works for all pairs; None if it doesn't
        """
        # Build candidate mapping from first pair
        first_pair = train_pairs[0]
        X0 = first_pair["input"]
        Y0 = first_pair["output"]

        # Handle empty grid edge case
        if not X0:
            if not Y0:
                return {}  # Empty mapping for empty grids
            else:
                return None  # Dimension mismatch

        # Apply σ to first input
        X0_sigma = apply_isometry(X0, sigma)

        # Shape safety: check dimensions match
        if dims(X0_sigma) != dims(Y0):
            return None  # Skip this σ

        rows, cols = dims(X0_sigma)

        # Build mapping from first pair (pixel-by-pixel comparison)
        mapping = {}
        for r in range(rows):
            for c in range(cols):
                old_c = X0_sigma[r][c]
                new_c = Y0[r][c]

                if old_c not in mapping:
                    # New color mapping
                    mapping[old_c] = new_c
                else:
                    # Check for conflict in first pair
                    if mapping[old_c] != new_c:
                        return None  # Conflict: same color maps to different outputs

        # Verify mapping works for all remaining pairs
        for pair in train_pairs[1:]:
            X = pair["input"]
            Y = pair["output"]

            # Apply σ to input
            X_sigma = apply_isometry(X, sigma)

            # Shape safety
            if dims(X_sigma) != dims(Y):
                return None  # Skip this σ

            rows_i, cols_i = dims(X_sigma)

            # Verify every pixel follows the learned mapping
            for r in range(rows_i):
                for c in range(cols_i):
                    old_c = X_sigma[r][c]
                    new_c = Y[r][c]

                    if old_c not in mapping:
                        # New color in later pair not seen in first pair
                        return None

                    if mapping[old_c] != new_c:
                        # Conflict: mapping doesn't work for this pair
                        return None

        # All pairs verified - this (σ, mapping) works!
        return mapping

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply stored σ then mapping to input X.

        Algorithm:
            1. Check params.sigma and params.mapping are set
            2. Apply σ: X_sigma = apply_isometry(X, params.sigma)
            3. Apply mapping: result[r][c] = mapping[X_sigma[r][c]]
            4. Return result

        Args:
            X: Input grid

        Returns:
            Grid with σ applied first, then mapping

        Raises:
            RuntimeError: If params.sigma or params.mapping is None (must call fit() first)
            KeyError: If X_sigma contains color not in mapping (unseen color)

        Edge cases:
            - Empty grid []: returns []
            - Shape change: σ may swap dimensions
            - Unseen color: raises KeyError

        Composition:
            - σ applied FIRST, then mapping (order matters!)

        Determinism:
            - Same (X, σ, mapping) always yields same output

        Purity:
            - Never mutates X
            - Returns new grid (no aliasing)
        """
        if self.params.sigma is None:
            raise RuntimeError("IsoColorMapFamily.apply() called before fit(). params.sigma is None. Must call fit() with train_pairs first.")

        if self.params.mapping is None:
            raise RuntimeError("IsoColorMapFamily.apply() called before fit(). params.mapping is None. Must call fit() with train_pairs first.")

        # Empty grid edge case
        if not X:
            return []

        # Step 1: Apply σ
        X_sigma = apply_isometry(X, self.params.sigma)

        # Step 2: Apply mapping
        rows, cols = dims(X_sigma)
        result = []

        for r in range(rows):
            new_row = []
            for c in range(cols):
                old_c = X_sigma[r][c]
                # KeyError raised automatically if old_c not in mapping
                new_c = self.params.mapping[old_c]
                new_row.append(new_c)
            result.append(new_row)

        return result
