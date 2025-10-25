"""
SortRowsLex Family for ARC AGI solver.

Global exact family: Lexicographic row sorting (no learned parameters).
Verifies that Y == sorted(X) for all training pairs (rows sorted lexicographically).
Sorting is deterministic, stable, and ascending.

Critical constraints:
- FY: Y must equal sorted(X) bit-for-bit for ALL train pairs
- No parameters to learn (deterministic transformation)
- Lexicographic order: rows compared as tuples
- Stable: equal rows preserve original relative order
- Shape preservation: dims(Y) == dims(X)
"""

from src.utils import dims, deep_eq


class SortRowsLexFamily:
    """
    Global exact family: Lexicographic row sorting (no learned parameters).

    Verifies that Y == sorted(X) for all training pairs.
    Sorting is deterministic, stable, and ascending.

    Invariants:
        - FY: Accept only if Y == sorted(X) for ALL train pairs exactly
        - Shape preservation: dims(Y) == dims(X) for all pairs
        - Lexicographic order: rows compared as sequences
        - Stable sort: equal rows preserve original relative order
        - Determinism: same input always yields same output
        - Purity: No mutations; fresh grid allocation
    """

    def __init__(self):
        """
        Initialize family with name and no parameters.

        Unlike parameterized families, SortRowsLex has NO params to learn.
        It's a fixed transformation (deterministic lexicographic sort).
        """
        self.name = "SortRowsLex"
        # Empty params object for API consistency
        self.params = type('Params', (), {})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Verify that Y == sorted(X) for ALL training pairs.

        No parameters are learned; this is a pure verification step.

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if ALL pairs satisfy Y == sorted(X) exactly; False otherwise

        Algorithm:
            1. If train_pairs is empty: return False
            2. For each pair (X, Y):
                a. Check dims(X) == dims(Y) (sorting preserves shape)
                b. Compute sorted_X = sorted(X rows lexicographically)
                c. If not deep_eq(sorted_X, Y): return False
            3. If all pairs pass: return True

        Determinism:
            - Python's sorted() is stable and deterministic

        Purity:
            - Never mutates train_pairs
            - No side effects
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Verify each pair
        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grids
            if not X and not Y:
                continue  # Both empty is valid

            if not X or not Y:
                return False  # One empty but not the other

            # Get dimensions
            hx, wx = dims(X)
            hy, wy = dims(Y)

            # Check for empty dimensions
            if hx == 0 or wx == 0:
                return False

            # Check shape preservation
            if hx != hy or wx != wy:
                return False  # Sorting preserves dimensions

            # Sort X rows lexicographically
            sorted_X = self._sort_rows(X)

            # Verify FY: sorted(X) == Y
            if not deep_eq(sorted_X, Y):
                return False

        # All pairs satisfied
        return True

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply lexicographic row sorting to input X.

        Args:
            X: Input grid

        Returns:
            Grid with rows sorted lexicographically (ascending)
            Output dimensions: same as input (shape preserved)

        Determinism:
            - Same X always yields same output
            - Stable sort: equal rows preserve original order

        Purity:
            - Never mutates X
            - Returns new grid
        """
        return self._sort_rows(X)

    def _sort_rows(self, X: list[list[int]]) -> list[list[int]]:
        """
        Helper: sort rows lexicographically.

        Args:
            X: Input grid

        Returns:
            Grid with rows sorted in ascending lexicographic order

        Sorting Logic:
            - Rows compared as sequences (lexicographic comparison)
            - Ascending order (smallest row first)
            - Stable: equal rows preserve original relative order
            - Uses Python's sorted() which is deterministic

        Purity:
            - Never mutates X
            - Returns new grid (fresh allocation)
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Check for empty dimensions
        if h == 0 or w == 0:
            return []

        # Sort rows lexicographically
        # Python's sorted() compares lists lexicographically by default
        # and is stable (preserves relative order of equal elements)
        sorted_rows = sorted(X, key=lambda row: tuple(row))

        # Create fresh copies to ensure no aliasing
        result = [list(row) for row in sorted_rows]

        return result
