"""
SortColsLex Family for ARC AGI solver.

Global exact family: Lexicographic column sorting (no learned parameters).
Verifies that Y == sorted_cols(X) for all training pairs (columns sorted lexicographically).
Sorting is deterministic, stable, and ascending.

Critical constraints:
- FY: Y must equal sorted_cols(X) bit-for-bit for ALL train pairs
- No parameters to learn (deterministic transformation)
- Lexicographic order: columns compared as tuples (top-to-bottom)
- Stable: equal columns preserve original relative order
- Shape preservation: dims(Y) == dims(X)
"""

from src.utils import dims, deep_eq


class SortColsLexFamily:
    """
    Global exact family: Lexicographic column sorting (no learned parameters).

    Verifies that Y == sorted_cols(X) for all training pairs.
    Sorting is deterministic, stable, and ascending.

    Invariants:
        - FY: Accept only if Y == sorted_cols(X) for ALL train pairs exactly
        - Shape preservation: dims(Y) == dims(X) for all pairs
        - Lexicographic order: columns compared as tuples (top-to-bottom)
        - Stable sort: equal columns preserve original relative order
        - Determinism: same input always yields same output
        - Purity: No mutations; fresh grid allocation
    """

    def __init__(self):
        """
        Initialize family with name and no parameters.

        Unlike parameterized families, SortColsLex has NO params to learn.
        It's a fixed transformation (deterministic lexicographic column sort).
        """
        self.name = "SortColsLex"
        # Empty params object for API consistency
        self.params = type('Params', (), {})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Verify that Y == sorted_cols(X) for ALL training pairs.

        No parameters are learned; this is a pure verification step.

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if ALL pairs satisfy Y == sorted_cols(X) exactly; False otherwise

        Algorithm:
            1. If train_pairs is empty: return False
            2. For each pair (X, Y):
                a. Check dims(X) == dims(Y) (sorting preserves shape)
                b. Compute sorted_cols_X = sorted_cols(X)
                c. If not deep_eq(sorted_cols_X, Y): return False
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

            # Sort columns of X lexicographically
            sorted_cols_X = self._sort_cols(X)

            # Verify FY: sorted_cols(X) == Y
            if not deep_eq(sorted_cols_X, Y):
                return False

        # All pairs satisfied
        return True

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply lexicographic column sorting to input X.

        Args:
            X: Input grid

        Returns:
            Grid with columns sorted lexicographically (ascending)
            Output dimensions: same as input (shape preserved)

        Determinism:
            - Same X always yields same output
            - Stable sort: equal columns preserve original order

        Purity:
            - Never mutates X
            - Returns new grid
        """
        return self._sort_cols(X)

    def _sort_cols(self, X: list[list[int]]) -> list[list[int]]:
        """
        Helper: sort columns lexicographically.

        Args:
            X: Input grid

        Returns:
            Grid with columns sorted in ascending lexicographic order

        Sorting Logic:
            - Columns extracted as tuples (top-to-bottom)
            - Compared lexicographically as sequences
            - Ascending order (smallest column first)
            - Stable: equal columns preserve original relative order
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

        # Extract columns as tuples (top-to-bottom)
        columns = []
        for j in range(w):
            col_j = tuple(X[r][j] for r in range(h))
            columns.append(col_j)

        # Sort columns lexicographically
        # Python's sorted() compares tuples lexicographically by default
        # and is stable (preserves relative order of equal elements)
        sorted_columns = sorted(columns)

        # Reconstruct grid from sorted columns
        result = []
        for r in range(h):
            row = []
            for c in range(w):
                # sorted_columns[c] is a tuple, so sorted_columns[c][r] is the element at row r
                row.append(sorted_columns[c][r])
            result.append(row)

        return result
