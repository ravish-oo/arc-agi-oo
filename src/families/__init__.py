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

__all__ = ["IsometryFamily", "ColorMapFamily", "IsoColorMapFamily", "PixelReplicateFamily", "BlockDownFamily"]
