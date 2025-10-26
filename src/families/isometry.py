"""
Isometry Family for ARC AGI solver.

Global exact family: D8 isometry selection.
Tries all σ ∈ D8 in deterministic all_isometries() order.
Accepts FIRST σ where apply_isometry(X, σ) == Y for ALL train pairs.
"""

from src.utils import deep_eq, dims
from src.canonicalization import all_isometries, apply_isometry


class IsometryFamily:
    """
    Global exact family: D8 isometry selection.

    Tries all σ ∈ D8 in deterministic all_isometries() order.
    Accepts FIRST σ where apply_isometry(X, σ) == Y for ALL train pairs.
    Stores params.sigma and uses it for apply(X).

    Invariants:
        - FY: Accept σ only if deep_eq holds for ALL train pairs
        - Determinism: First-acceptable σ in all_isometries() order wins
        - Unified: ONE σ for all pairs (no per-pair selection)
        - Shape safety: Skip σ if dimension mismatch occurs
        - Purity: No mutations to inputs
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            sigma: None initially; set to σ name (str) after successful fit()
        """
        self.name = "Isometry"
        self.params = type('Params', (), {'sigma': None})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Feasibility fit for Step-2 architecture.

        In Step-2, P is a preprocessing step combined with Φ/GLUE, not a complete solver.
        This method checks if there exists at least one σ ∈ D8 that is shape-safe for
        all training pairs. It picks the FIRST shape-safe σ in deterministic order.

        The solver will try P + Φ/GLUE composition - we don't require P(X) == Y here.

        Algorithm:
            1. For each σ in all_isometries() (deterministic order):
                2. Check if σ is shape-safe for ALL train pairs:
                    - For each pair (X, Y): dims(apply_isometry(X, σ)) == dims(Y)
                3. If shape-safe for all pairs: store params.sigma = σ; return True
            4. Return False (no shape-safe σ found)

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found shape-safe σ; False otherwise

        Edge cases:
            - Empty train_pairs: return False
            - All σ cause dimension changes: return False
            - Identity "id" is always shape-safe (dims(X) unchanged)

        Determinism:
            - all_isometries() order is fixed
            - First shape-safe σ wins (stable tie-breaking)

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting self.params.sigma

        Step-2 Contract:
            - Feasibility check (shape safety), not exactness check
            - P + Φ/GLUE will handle the actual transformation
            - FY constraint enforced at candidate level, not here
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Try each σ in deterministic D8 order
        for sigma in all_isometries():
            # Check if this σ is shape-safe for ALL train pairs
            shape_safe_for_all = True

            for pair in train_pairs:
                X = pair["input"]
                Y = pair["output"]

                # Apply σ to input
                X_sigma = apply_isometry(X, sigma)

                # Shape safety: skip σ if dimensions mismatch
                if dims(X_sigma) != dims(Y):
                    shape_safe_for_all = False
                    break

            # If this σ is shape-safe for all pairs, accept it
            if shape_safe_for_all:
                self.params.sigma = sigma
                return True

        # No shape-safe σ found
        return False

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply stored σ to input X.

        Args:
            X: Input grid

        Returns:
            apply_isometry(X, params.sigma)

        Raises:
            RuntimeError: If params.sigma is None (must call fit() first)

        Edge cases:
            - Empty grid []: apply_isometry([], σ) returns []
            - Shape change: rot90/rot270/transpose/flip_anti swap dimensions

        Determinism:
            - Same (X, σ) always yields same output

        Purity:
            - Never mutates X
            - Returns new grid (no aliasing)
        """
        if self.params.sigma is None:
            raise RuntimeError("IsometryFamily.apply() called before fit(). Must call fit() with train_pairs first.")

        return apply_isometry(X, self.params.sigma)
