"""
ColorMap Family for ARC AGI solver.

Global exact family: per-color remapping (LUT).
Learns a single color mapping dict {old_color: new_color} from training pairs.
Accepts ONLY if this mapping reproduces every training pair exactly (FY).
"""

from src.utils import dims


class ColorMapFamily:
    """
    Global exact family: per-color remapping (LUT).

    Learns a single color mapping dict {old_color: new_color} from training pairs.
    Accepts ONLY if this mapping reproduces every training pair exactly (FY).

    Invariants:
        - FY: Accept mapping only if it reproduces ALL train pairs exactly
        - Unified: ONE mapping dict for all pairs (no per-pair variations)
        - Conflict rejection: Same input color mapping to different outputs → reject
        - No guessing: apply() raises KeyError for unseen colors
        - Determinism: Row-major iteration, stable mapping construction
        - Purity: No mutations to inputs
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            mapping: None initially; set to dict[int, int] after successful fit()
        """
        self.name = "ColorMap"
        self.params = type('Params', (), {'mapping': None})()

    def fit(self, train_pairs: list[dict]) -> bool:
        """
        Learn ONE color mapping dict that works for ALL train pairs.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Extract first pair (X0, Y0)
            3. Build candidate mapping by comparing X0 and Y0 pixel-by-pixel:
                - For each position (r, c):
                    - old_c = X0[r][c]
                    - new_c = Y0[r][c]
                    - If old_c not in mapping: mapping[old_c] = new_c
                    - Else if mapping[old_c] != new_c: return False (conflict in first pair)
            4. For each remaining pair (X, Y) in train_pairs[1:]:
                - For each position (r, c):
                    - old_c = X[r][c]
                    - new_c = Y[r][c]
                    - If old_c not in mapping: return False (unseen color in later pair)
                    - If mapping[old_c] != new_c: return False (conflict)
            5. Store params.mapping and return True

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if found mapping that satisfies FY on all pairs; False otherwise

        Edge cases:
            - Empty train_pairs: return False
            - Single pair: learn mapping from that pair
            - Identity mapping (all colors same): {c: c for c in colors}
            - Conflict in first pair (1→2 at pos A, 1→3 at pos B): return False
            - Conflict across pairs (1→2 in pair0, 1→3 in pair1): return False
            - New color in later pair not seen in first pair: return False

        Determinism:
            - Iteration over grid positions is row-major (stable)
            - Mapping dict construction is deterministic (insertion order preserved)

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting self.params.mapping

        FY Principle:
            - Bit-for-bit equality required for ALL pairs
            - Single pixel mismatch → reject mapping
            - All-but-one pixels match → still reject
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Build candidate mapping from first pair
        mapping = {}
        first_pair = train_pairs[0]
        X0 = first_pair["input"]
        Y0 = first_pair["output"]

        # Handle empty grid edge case
        if not X0:
            # Both must be empty for mapping to work
            if not Y0:
                self.params.mapping = {}
                return True
            else:
                return False

        # Validate first pair has same dimensions
        if dims(X0) != dims(Y0):
            return False

        rows, cols = dims(X0)

        # Build mapping from first pair (pixel-by-pixel comparison)
        for r in range(rows):
            for c in range(cols):
                old_c = X0[r][c]
                new_c = Y0[r][c]

                if old_c not in mapping:
                    # New color mapping
                    mapping[old_c] = new_c
                else:
                    # Check for conflict in first pair
                    if mapping[old_c] != new_c:
                        return False  # Same input color maps to different outputs in first pair

        # Verify mapping works for all remaining pairs
        for pair in train_pairs[1:]:
            X = pair["input"]
            Y = pair["output"]

            # Validate dimensions match
            if dims(X) != dims(Y):
                return False

            rows_i, cols_i = dims(X)

            # Verify every pixel follows the learned mapping
            for r in range(rows_i):
                for c in range(cols_i):
                    old_c = X[r][c]
                    new_c = Y[r][c]

                    if old_c not in mapping:
                        # New color in later pair not seen in first pair
                        return False

                    if mapping[old_c] != new_c:
                        # Conflict: mapping doesn't work for this pair
                        return False

        # All pairs verified - store mapping and accept
        self.params.mapping = mapping
        return True

    def apply(self, X: list[list[int]]) -> list[list[int]]:
        """
        Apply stored color mapping to input X.

        Args:
            X: Input grid

        Returns:
            Grid where each pixel X[r][c] is replaced by mapping[X[r][c]]

        Raises:
            RuntimeError: If params.mapping is None (must call fit() first)
            KeyError: If X contains color not in mapping (unseen color)

        Edge cases:
            - Empty grid []: returns []
            - Single pixel [[5]]: returns [[mapping[5]]]
            - All pixels same color: all mapped to same output color

        Determinism:
            - Same (X, mapping) always yields same output

        Purity:
            - Never mutates X
            - Returns new grid (no row aliasing)
        """
        if self.params.mapping is None:
            raise RuntimeError("ColorMapFamily.apply() called before fit(). Must call fit() with train_pairs first.")

        # Empty grid edge case
        if not X:
            return []

        rows, cols = dims(X)
        result = []

        # Map each pixel via learned mapping
        for r in range(rows):
            new_row = []
            for c in range(cols):
                old_c = X[r][c]
                # KeyError raised automatically if old_c not in mapping
                new_c = self.params.mapping[old_c]
                new_row.append(new_c)
            result.append(new_row)

        return result
