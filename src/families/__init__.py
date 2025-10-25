"""
ARC AGI Global Families (P) for exact transformations.

Phase 2: 16 global families implementing fit/apply pattern.
Each family tries to find parameters that exactly reproduce ALL train pairs.
"""

from src.families.isometry import IsometryFamily
from src.families.color_map import ColorMapFamily
from src.families.iso_color_map import IsoColorMapFamily
from src.families.pixel_replicate import PixelReplicateFamily
from src.families.block_down import BlockDownFamily
from src.families.nps_down import NPSDownFamily
from src.families.nps_up import NPSUpFamily
from src.families.parity_tile import ParityTileFamily
from src.families.block_permutation import BlockPermutationFamily
from src.families.block_substitution import BlockSubstitutionFamily
from src.families.row_permutation import RowPermutationFamily
from src.families.col_permutation import ColPermutationFamily
from src.families.sort_rows import SortRowsLexFamily
from src.families.sort_cols import SortColsLexFamily

__all__ = ["IsometryFamily", "ColorMapFamily", "IsoColorMapFamily", "PixelReplicateFamily", "BlockDownFamily", "NPSDownFamily", "NPSUpFamily", "ParityTileFamily", "BlockPermutationFamily", "BlockSubstitutionFamily", "RowPermutationFamily", "ColPermutationFamily", "SortRowsLexFamily", "SortColsLexFamily"]
