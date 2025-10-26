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
        Feasibility fit for Step-2 architecture.

        In Step-2, this transform is always applicable - it preprocesses
        the input and Φ/GLUE handles matching to the output.

        This family has no trainable parameters - it applies deterministic
        logic to transform the input. Feasibility is universal.

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            Always True (transform is always applicable)

        Step-2 Contract:
            - No fit() parameters to learn
            - Transform is always feasible
            - FY constraint enforced at candidate level, not here
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Always applicable in Step-2
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
