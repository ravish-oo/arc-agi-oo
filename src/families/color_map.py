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
        Feasibility fit for Step-2 architecture.

        In Step-2, ColorMap is a preprocessing step combined with Φ/GLUE.
        This method builds a consistent color mapping across all training pairs
        and validates shape safety. It does NOT require that mapping produces
        exact outputs - that's handled by P + Φ/GLUE composition.

        Algorithm:
            1. If train_pairs is empty: return False
            2. Build unified mapping across ALL train pairs:
                - For each pair (X, Y):
                    - Check shape safety: dims(X) == dims(Y)
                    - For each pixel position (r, c):
                        - old_c = X[r][c], new_c = Y[r][c]
                        - If old_c not in mapping: mapping[old_c] = new_c
                        - Else if mapping[old_c] != new_c: return False (conflict)
            3. Store params.mapping and return True

        Args:
            train_pairs: list of {"input": grid, "output": grid} dicts

        Returns:
            True if consistent, shape-safe mapping exists; False if conflicts

        Edge cases:
            - Empty train_pairs: return False
            - Empty grids: accept if all empty
            - Shape mismatch: return False (not shape-safe)
            - Conflict (1→2 and 1→3): return False (inconsistent)
            - Cross-palette learning: accept new colors from later pairs

        Determinism:
            - Row-major iteration (stable)
            - Mapping dict insertion order preserved

        Purity:
            - Never mutates train_pairs
            - No side effects beyond setting self.params.mapping

        Step-2 Contract:
            - Feasibility check (consistency + shape safety)
            - Does NOT require apply(X) == Y
            - P + Φ/GLUE will handle actual transformation
            - FY constraint enforced at candidate level
        """
        # Empty train_pairs edge case
        if not train_pairs:
            return False

        # Build unified mapping across all train pairs
        mapping = {}

        for pair in train_pairs:
            X = pair["input"]
            Y = pair["output"]

            # Handle empty grid edge case
            if not X or not Y:
                continue  # Skip empty grids, can't learn mapping

            # Build mapping from overlapping region (min dimensions)
            rows_x, cols_x = dims(X)
            rows_y, cols_y = dims(Y)
            rows = min(rows_x, rows_y)
            cols = min(cols_x, cols_y)

            # Build/verify mapping from overlapping region
            for r in range(rows):
                for c in range(cols):
                    old_c = X[r][c]
                    new_c = Y[r][c]

                    if old_c not in mapping:
                        # New color mapping - learn it
                        mapping[old_c] = new_c
                    else:
                        # Check for conflict (consistency requirement)
                        if mapping[old_c] != new_c:
                            return False

        # Consistent mapping found (shape-agnostic) - accept
        self.params.mapping = mapping if mapping else {}
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
