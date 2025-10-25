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
        Find ONE σ ∈ D8 that works for ALL train pairs.

        Algorithm:
            1. For each σ in all_isometries() (deterministic order):
                2. For each pair (X, Y) in train_pairs:
                    3. Apply σ to X: X_sigma = apply_isometry(X, σ)
                    4. If dims(X_sigma) ≠ dims(Y): skip this σ (shape mismatch)
                    5. If not deep_eq(X_sigma, Y): skip this σ (value mismatch)
                6. If all pairs passed: store params.sigma = σ; return True
            7. Return False (no σ worked for all pairs)

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found σ that satisfies FY on all pairs; False otherwise

        Edge cases:
            - Empty train_pairs: return False
            - Single pair: accept first matching σ
            - All pairs identical (X==Y): "id" should match
            - Mixed transforms (no unified σ): return False

        Determinism:
            - all_isometries() order is fixed
            - First-acceptable wins (stable tie-breaking)

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting self.params.sigma

        FY Principle:
            - Bit-for-bit equality required for ALL pairs
            - Single pixel difference → reject σ
            - All-but-one pairs match → still reject σ
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Try each σ in deterministic D8 order
        for sigma in all_isometries():
            # Check if this σ works for ALL train pairs
            all_pairs_match = True

            for pair in train_pairs:
                X = pair["input"]
                Y = pair["output"]

                # Apply σ to input
                X_sigma = apply_isometry(X, sigma)

                # Shape safety: skip σ if dimensions mismatch
                if dims(X_sigma) != dims(Y):
                    all_pairs_match = False
                    break

                # FY exactness: skip σ if not bit-for-bit equal
                if not deep_eq(X_sigma, Y):
                    all_pairs_match = False
                    break

            # If this σ worked for ALL pairs, accept it
            if all_pairs_match:
                self.params.sigma = sigma
                return True

        # No σ worked for all pairs
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
