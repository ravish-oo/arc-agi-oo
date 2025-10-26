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
