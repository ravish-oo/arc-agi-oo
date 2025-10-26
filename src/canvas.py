"""
Canvas transforms: Stage C color alignment choices.

A canvas transform is applied before P (global transform) and determines
which color space the Φ features will operate in.

Interface:
    fit(train_inputs): Learn from training inputs
    apply(grid): Returns (transformed_grid, aux_data)
        - transformed_grid: Grid in the canvas color space
        - aux_data: Dict with metadata including raw_color_id for features
"""

from src.utils import dims


class IdentityColor:
    """
    No-op canvas: colors remain in their original raw palette.

    This is the baseline choice - no color alignment.
    Use this when colorful schemas (use_is_color=True) need raw color values.
    """

    name = "Identity"

    def __init__(self):
        pass

    def fit(self, train_inputs: list[list[list[int]]]):
        """No learning needed for identity transform."""
        pass

    def apply(self, grid: list[list[int]]) -> tuple[list[list[int]], dict]:
        """
        Return grid unchanged with auxiliary data containing raw colors.

        Args:
            grid: Input grid in original color space

        Returns:
            Tuple of (grid, aux_data) where:
            - grid: Same as input (no transformation)
            - aux_data: Dict with 'raw_color_id' grid (same as input)
        """
        if not grid:
            return grid, {"raw_color_id": grid}

        # For identity, transformed grid IS the raw grid
        aux_data = {
            "raw_color_id": grid  # Raw colors for colorful features
        }

        return grid, aux_data
