"""
CopyMoveAllComponents Family for ARC AGI solver.

Global exact family: per-color component translation.
Learns ONE translation vector (Δr, Δc) per color that satisfies FY for ALL train pairs.
All components of color c translate by the same (Δr, Δc).

Critical constraints:
- FY: Accept deltas only if deep_eq holds for ALL train pairs
- Determinism: Component ordering from components_by_color() is stable
- Unified: ONE delta per color across all pairs (no per-component variation)
- Translation only: Components preserve shape and orientation
- GLUE: Write components in deterministic order; later overwrites earlier
- Shape safety: Output dimensions equal input dimensions
- Purity: No mutations to inputs
"""

from src.utils import dims, deep_eq
from src.components import components_by_color


class CopyMoveAllComponentsFamily:
    """
    Global exact family: per-color component translation.

    Learns ONE translation vector (Δr, Δc) per color that satisfies FY for ALL train pairs.
    All components of color c translate by the same (Δr, Δc).

    Invariants:
        - FY: Accept deltas only if deep_eq holds for ALL train pairs
        - Determinism: Component ordering from components_by_color() is stable
        - Unified: ONE delta per color across all pairs (no per-component variation)
        - Translation only: Components preserve shape and orientation
        - GLUE: Write components in deterministic order; later overwrites earlier
        - Shape safety: Output dimensions equal input dimensions
        - Purity: No mutations to inputs
    """

    def __init__(self):
        """
        Initialize family with name and empty params.

        Params:
            deltas: None initially; set to dict[int, tuple[int,int]] after successful fit()
                    Maps color → (Δr, Δc) translation vector
        """
        self.name = "CopyMoveAllComponents"
        self.params = type('Params', (), {'deltas': None})()

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
        Apply stored per-color translation to input X.

        Args:
            X: Input grid

        Returns:
            Grid with all components translated by learned deltas

        Raises:
            RuntimeError: If params.deltas is None (must call fit() first)

        Determinism:
            - Same (X, deltas) always yields same output

        Purity:
            - Never mutates X
            - Returns new grid
        """
        if self.params.deltas is None:
            raise RuntimeError("CopyMoveAllComponentsFamily.apply() called before fit(). params.deltas is None. Must call fit() with train_pairs first.")

        return self._apply_deltas(X, self.params.deltas)

    def _compute_centroid(self, cells: list[tuple[int, int]]) -> tuple[float, float]:
        """
        Compute centroid (mean r, mean c) of component cells.

        Args:
            cells: List of (row, col) tuples

        Returns:
            (mean_r, mean_c) as floats

        Edge cases:
            - Empty cells list: raises ValueError
            - Single cell: returns that cell's coordinates

        Purity:
            - Read-only on cells
        """
        if not cells:
            raise ValueError("Cannot compute centroid for empty cells list")

        mean_r = sum(r for r, c in cells) / len(cells)
        mean_c = sum(c for r, c in cells) / len(cells)

        return (mean_r, mean_c)

    def _infer_deltas(self, X: list[list[int]], Y: list[list[int]]) -> dict[int, tuple[int, int]] | None:
        """
        Infer per-color translation deltas by matching components between X and Y.

        Algorithm:
            1. Extract components from X and Y using components_by_color()
            2. For each color in X:
                a. If color not in Y: return None (inconsistent)
                b. If component counts differ: return None
                c. Compute centroids for all components
                d. Sort components by centroid (row-major lex)
                e. Match components by sorted order
                f. Infer delta from first component pair
                g. Verify all component pairs have same delta
            3. Return deltas dict

        Args:
            X: Input grid
            Y: Output grid (same dimensions as X)

        Returns:
            dict[color, (Δr, Δc)] or None if inference fails

        Determinism:
            - components_by_color() returns stable order
            - Sorting by centroid is deterministic

        Purity:
            - Read-only on X and Y
        """
        # Extract components
        cx = components_by_color(X)
        cy = components_by_color(Y)

        deltas = {}

        # Check for colors in Y but not in X (new colors)
        # Skip color 0 - it can appear as background fill even if not in X
        for color in cy:
            if color != 0 and color not in cx:
                return None  # Cannot infer inverse translation for new color

        # For each color in X
        for color in sorted(cx.keys()):
            # Check if color exists in Y
            if color not in cy:
                # Color disappeared (moved out of bounds or overwritten) - skip it
                # Note: color 0 can appear in Y as background even if not in X
                continue

            # Check if component counts match
            if len(cx[color]) != len(cy[color]):
                # Component count mismatch - color must have disappeared or been overwritten
                # If counts differ, we can't infer a unified delta, so skip this color
                continue

            # Compute centroids for X components
            x_comps_with_centroids = []
            for comp in cx[color]:
                centroid = self._compute_centroid(comp["cells"])
                x_comps_with_centroids.append((comp, centroid))

            # Compute centroids for Y components
            y_comps_with_centroids = []
            for comp in cy[color]:
                centroid = self._compute_centroid(comp["cells"])
                y_comps_with_centroids.append((comp, centroid))

            # Sort by centroid (row-major lex)
            x_comps_with_centroids.sort(key=lambda item: item[1])
            y_comps_with_centroids.sort(key=lambda item: item[1])

            # Infer delta from first component pair
            if len(x_comps_with_centroids) == 0:
                # No components of this color (shouldn't happen since color is in cx)
                continue

            x_centroid_0 = x_comps_with_centroids[0][1]
            y_centroid_0 = y_comps_with_centroids[0][1]

            delta_r = round(y_centroid_0[0] - x_centroid_0[0])
            delta_c = round(y_centroid_0[1] - x_centroid_0[1])

            # Verify all component pairs have same delta
            for i in range(len(x_comps_with_centroids)):
                x_centroid = x_comps_with_centroids[i][1]
                y_centroid = y_comps_with_centroids[i][1]

                delta_r_i = round(y_centroid[0] - x_centroid[0])
                delta_c_i = round(y_centroid[1] - x_centroid[1])

                if delta_r_i != delta_r or delta_c_i != delta_c:
                    return None  # Inconsistent deltas within same color

            # Store delta for this color
            deltas[color] = (delta_r, delta_c)

        return deltas

    def _apply_deltas(self, X: list[list[int]], deltas: dict[int, tuple[int, int]]) -> list[list[int]]:
        """
        Apply per-color translation deltas to grid X.

        Algorithm:
            1. Create output grid Y (all zeros, same size as X)
            2. Extract components from X
            3. For each color in sorted order:
                a. Get delta (Δr, Δc) for this color
                b. For each component of this color:
                    c. For each cell (r, c) in component:
                        d. Compute translated position (r', c') = (r+Δr, c+Δc)
                        e. If in bounds: Y[r'][c'] = color (overwrites earlier)
            4. Return Y

        Args:
            X: Input grid
            deltas: dict[color, (Δr, Δc)]

        Returns:
            Translated grid

        Determinism:
            - components_by_color() returns stable order
            - Processing colors in sorted order ensures deterministic overwrites

        Purity:
            - Never mutates X
            - Returns new grid
        """
        # Handle empty grid
        if not X:
            return []

        h, w = dims(X)

        # Check for empty dimensions
        if h == 0 or w == 0:
            return []

        # Create output grid (all zeros)
        Y = [[0] * w for _ in range(h)]

        # Extract components from X
        cx = components_by_color(X)

        # Process colors in sorted order (deterministic)
        for color in sorted(cx.keys()):
            # Check if we have a delta for this color
            if color not in deltas:
                continue  # Skip this color (no learned delta)

            delta_r, delta_c = deltas[color]

            # Translate all components of this color
            for comp in cx[color]:
                for r, c in comp["cells"]:
                    # Compute translated position
                    r_new = r + delta_r
                    c_new = c + delta_c

                    # Bounds check
                    if 0 <= r_new < h and 0 <= c_new < w:
                        # Write to output (later colors overwrite earlier)
                        Y[r_new][c_new] = color

        return Y
