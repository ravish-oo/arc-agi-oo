"""
RowPermutation Family for ARC AGI solver.

Global exact family: Deterministic row reordering (learned permutation).
Learns ONE permutation perm from first training pair that works for ALL pairs.
perm = list[int] where perm[i] = j means row i in X moves to position j in Y.

Critical constraints:
- Same permutation must work for all training pairs (unified params)
- All pairs must have equal shapes (same row count, same col count)
- Row multisets must match: sorted(X_rows) == sorted(Y_rows)
- Permutation must be bijection: each row used exactly once
"""

from src.utils import dims, deep_eq


class RowPermutationFamily:
    """
    Global exact family: Deterministic row reordering (learned permutation).

    Learns ONE permutation from first training pair that works for ALL pairs.
    Reorders rows of input X to produce output Y.

    Invariants:
        - FY: Accept params only if they reproduce ALL train pairs exactly
        - Shape preservation: dims(X) == dims(Y) for all pairs
        - Row multiset equality: sorted(rows) must match
        - Permutation bijectivity: each row used exactly once
        - Deterministic tie-break: greedy first-fit for symmetric rows
        - Purity: No mutations; fresh grid allocation
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            perm: None initially; list[int] after successful fit()
                  perm[i] = j means row i in X goes to position j in Y
        """
        self.name = "RowPermutation"
        self.params = type('Params', (), {
            'perm': None,
        })()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Learn ONE permutation from first pair that works for ALL pairs.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair and verify shape
            3. Find permutation using greedy first-fit
            4. Verify on all training pairs

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found permutation that satisfies FY on all pairs; False otherwise

        Determinism:
            - Always learn from first pair
            - Greedy first-fit for tie-breaking

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
        if hx == 0 or wx == 0:
            return False

        # Check shape preservation
        if hx != hy or wx != wy:
            return False  # Shape mismatch

        h, w = hx, wx

        # Find permutation from first pair
        perm = self._find_permutation(X0, Y0)
        if perm is None:
            return False  # No valid permutation

        # Verify on ALL training pairs
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Check dimensions
            if not X or not Y:
                return False

            hx_p, wx_p = dims(X)
            hy_p, wy_p = dims(Y)

            # Check shape consistency
            if hx_p != h or wx_p != w or hy_p != h or wy_p != w:
                return False  # Inconsistent shapes

            # Verify FY: apply permutation and check equality
            Y_predicted = self._apply_with_perm(X, perm)
            if not deep_eq(Y_predicted, Y):
                return False  # FY violation

        # All pairs satisfied - store params and return True
        self.params.perm = perm
        return True

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply row permutation using learned parameters to input X.

        Args:
            X: Input grid

        Returns:
            Grid with rows reordered according to learned permutation
            Output dimensions: same as input (shape preserved)

        Raises:
            RuntimeError: If params.perm is None (must call fit() first)

        Determinism:
            - Same (X, perm) always yields same output

        Purity:
            - Never mutates X
            - Returns new grid
        """
        if self.params.perm is None:
            raise RuntimeError("RowPermutationFamily.apply() called before fit(). params.perm is None. Must call fit() with train_pairs first.")

        # Use helper method
        return self._apply_with_perm(X, self.params.perm)

    def _apply_with_perm(self, X: list[list[int]], perm: list[int]) -> list[list[int]]:
        """
        Helper: apply permutation with given perm.

        Args:
            X: Input grid
            perm: Permutation array

        Returns:
            Grid with rows reordered according to perm
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Check for empty dimensions
        if h == 0 or w == 0:
            return []

        # Compute inverse permutation
        # perm[i] = j means X[i] → Y[j]
        # inv_perm[j] = i means Y[j] ← X[i]
        inv_perm = [None] * h
        for i in range(h):
            j = perm[i]
            inv_perm[j] = i

        # Create output grid using inverse permutation
        result = []
        for j in range(h):
            i = inv_perm[j]
            # Copy row i to position j
            result.append(list(X[i]))

        return result

    def _find_permutation(self, X: list[list[int]], Y: list[list[int]]) -> list[int] | None:
        """
        Helper: find permutation perm such that Y[perm[i]] == X[i] for all i.

        Uses greedy first-fit for deterministic tie-breaking with symmetric rows.

        Args:
            X: Input grid
            Y: Output grid (must have same dims as X)

        Returns:
            List perm where perm[i] = j means X[i] → Y[j]
            None if no valid permutation exists

        Determinism:
            - Greedy first-fit ensures deterministic tie-breaking
            - Iterate Y positions [0..h-1] in order
            - For each Y position, assign first unused X row

        Purity:
            - Read-only on X and Y
        """
        # Get dimensions
        h, w = dims(X)
        hy, wy = dims(Y)

        # Check shape match
        if h != hy or w != wy:
            return None

        # Handle empty case
        if h == 0:
            return []

        # Convert rows to tuples for comparison
        rows_X = [tuple(X[i]) for i in range(h)]
        rows_Y = [tuple(Y[i]) for i in range(h)]

        # Check row multiset equality
        if sorted(rows_X) != sorted(rows_Y):
            return None  # Different row multisets

        # Greedy first-fit to find permutation
        # inv_perm[j] = i means Y[j] comes from X[i]
        used = [False] * h
        inv_perm = [None] * h

        for j in range(h):
            target_row = rows_Y[j]

            # Find first unused row in X that matches target
            found = False
            for i in range(h):
                if not used[i] and rows_X[i] == target_row:
                    used[i] = True
                    inv_perm[j] = i
                    found = True
                    break

            if not found:
                return None  # Should not happen if multiset check passed

        # Convert inv_perm to perm
        # inv_perm[j] = i means X[i] → Y[j]
        # perm[i] = j means X[i] → Y[j]
        perm = [None] * h
        for j in range(h):
            i = inv_perm[j]
            perm[i] = j

        return perm
