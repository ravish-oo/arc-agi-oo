"""
ColPermutation Family for ARC AGI solver.

Global exact family: Deterministic column reordering (learned permutation).
Learns ONE permutation perm from first training pair that works for ALL pairs.
perm = list[int] where perm[j] = k means column j in X moves to position k in Y.

Critical constraints:
- Same permutation must work for all training pairs (unified params)
- All pairs must have equal shapes (same row count, same col count)
- Column multisets must match: sorted(columns) must match
- Permutation must be bijection: each column used exactly once
"""

from src.utils import dims, deep_eq


class ColPermutationFamily:
    """
    Global exact family: Deterministic column reordering (learned permutation).

    Learns ONE permutation from first training pair that works for ALL pairs.
    Reorders columns of input X to produce output Y.

    Invariants:
        - FY: Accept params only if they reproduce ALL train pairs exactly
        - Shape preservation: dims(X) == dims(Y) for all pairs
        - Column multiset equality: sorted(columns) must match
        - Permutation bijectivity: each column used exactly once
        - Deterministic tie-break: greedy first-fit for symmetric columns
        - Purity: No mutations; fresh grid allocation
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            perm: None initially; list[int] after successful fit()
                  perm[j] = k means column j in X goes to position k in Y
        """
        self.name = "ColPermutation"
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
        Apply column permutation using learned parameters to input X.

        Args:
            X: Input grid

        Returns:
            Grid with columns reordered according to learned permutation
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
            raise RuntimeError("ColPermutationFamily.apply() called before fit(). params.perm is None. Must call fit() with train_pairs first.")

        # Use helper method
        return self._apply_with_perm(X, self.params.perm)

    def _apply_with_perm(self, X: list[list[int]], perm: list[int]) -> list[list[int]]:
        """
        Helper: apply permutation with given perm.

        Args:
            X: Input grid
            perm: Permutation array

        Returns:
            Grid with columns reordered according to perm
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Check for empty dimensions
        if h == 0 or w == 0:
            return []

        # Compute inverse permutation
        # perm[j] = k means X[:, j] → Y[:, k]
        # inv_perm[k] = j means Y[:, k] ← X[:, j]
        inv_perm = [None] * w
        for j in range(w):
            k = perm[j]
            inv_perm[k] = j

        # Create output grid using inverse permutation
        result = []
        for r in range(h):
            row = []
            for k in range(w):
                j = inv_perm[k]
                # Copy element from column j to position k
                row.append(X[r][j])
            result.append(row)

        return result

    def _find_permutation(self, X: list[list[int]], Y: list[list[int]]) -> list[int] | None:
        """
        Helper: find permutation perm such that Y[:, perm[j]] == X[:, j] for all j.

        Uses greedy first-fit for deterministic tie-breaking with symmetric columns.

        Args:
            X: Input grid
            Y: Output grid (must have same dims as X)

        Returns:
            List perm where perm[j] = k means X[:, j] → Y[:, k]
            None if no valid permutation exists

        Determinism:
            - Greedy first-fit ensures deterministic tie-breaking
            - Iterate Y column positions [0..w-1] in order
            - For each Y position, assign first unused X column

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
        if w == 0:
            return []

        # Extract columns as tuples for comparison
        cols_X = [tuple(X[r][j] for r in range(h)) for j in range(w)]
        cols_Y = [tuple(Y[r][k] for r in range(h)) for k in range(w)]

        # Check column multiset equality
        if sorted(cols_X) != sorted(cols_Y):
            return None  # Different column multisets

        # Greedy first-fit to find permutation
        # inv_perm[k] = j means Y[:, k] comes from X[:, j]
        used = [False] * w
        inv_perm = [None] * w

        for k in range(w):
            target_col = cols_Y[k]

            # Find first unused column in X that matches target
            found = False
            for j in range(w):
                if not used[j] and cols_X[j] == target_col:
                    used[j] = True
                    inv_perm[k] = j
                    found = True
                    break

            if not found:
                return None  # Should not happen if multiset check passed

        # Convert inv_perm to perm
        # inv_perm[k] = j means X[:, j] → Y[:, k]
        # perm[j] = k means X[:, j] → Y[:, k]
        perm = [None] * w
        for k in range(w):
            j = inv_perm[k]
            perm[j] = k

        return perm
