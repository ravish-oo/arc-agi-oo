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
        Feasibility fit for Step-2 architecture.

        In Step-2, IsoColorMap is a preprocessing step combined with Φ/GLUE.
        This method finds a (σ, mapping) pair that is shape-safe and has a
        consistent color mapping across all training pairs. It does NOT require
        that apply(X) == Y exactly - that's handled by P + Φ/GLUE composition.

        Algorithm:
            1. If train_pairs is empty: return False
            2. For each σ in all_isometries() (deterministic order):
                3. Check shape safety: dims(apply_isometry(X, σ)) == dims(Y) for all pairs
                4. Build consistent mapping from all pairs with conflict detection
                5. If shape-safe and conflict-free: store (σ, mapping) and return True
            6. Return False (no feasible (σ, mapping) found)

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found shape-safe, conflict-free (σ, mapping); False otherwise

        Edge cases:
            - Empty train_pairs: return False
            - Identity case: σ="id", mapping learned from training
            - Isometry only: σ!=id, mapping may be identity or custom
            - ColorMap only: σ="id", mapping learned from training

        Determinism:
            - all_isometries() order is fixed
            - First shape-safe, conflict-free (σ, mapping) wins
            - Mapping construction uses row-major scan

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting params

        Step-2 Contract:
            - Feasibility check (shape safety + consistent mapping)
            - Does NOT require apply(X) == Y
            - P + Φ/GLUE will handle actual transformation
            - FY constraint enforced at candidate level
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
        Try a specific σ: check shape safety and learn consistent mapping.

        In Step-2, this method checks feasibility (shape + consistent mapping)
        rather than exact pixel-level matching. The mapping is built from the
        overlapping region across all pairs with conflict detection.

        Args:
            sigma: Isometry name (from all_isometries())
            train_pairs: Training pairs

        Returns:
            mapping dict if this σ is feasible; None if not shape-safe or has conflicts
        """
        # Build candidate mapping from ALL pairs (shape-agnostic like ColorMap)
        mapping = {}

        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grid edge case
            if not X or not Y:
                continue  # Skip empty grids

            # Apply σ to input
            X_sigma = apply_isometry(X, sigma)

            # Shape safety: check dimensions match
            if dims(X_sigma) != dims(Y):
                return None  # Skip this σ (not shape-safe)

            rows, cols = dims(X_sigma)

            # Build/verify mapping from overlapping region
            for r in range(rows):
                for c in range(cols):
                    old_c = X_sigma[r][c]
                    new_c = Y[r][c]

                    if old_c not in mapping:
                        # New color mapping - learn it
                        mapping[old_c] = new_c
                    else:
                        # Check for conflict (consistency requirement)
                        if mapping[old_c] != new_c:
                            return None  # Conflict: same color maps to different outputs

        # Shape-safe and consistent mapping found - accept this σ
        return mapping if mapping else {}

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
