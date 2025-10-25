"""
ARC AGI Global Families (P) for exact transformations.

Phase 2: 16 global families implementing fit/apply pattern.
Each family tries to find parameters that exactly reproduce ALL train pairs.
"""

from src.families.isometry import IsometryFamily

__all__ = ["IsometryFamily"]
