"""
Comprehensive tests for Π (canonical_grid) and canonical_key.

Tests cover:
- Π² = Π (idempotence) - THE critical property
- Minimality (canonical_grid returns argmin over D8)
- Deterministic tie-breaking
- Purity (no mutation, no aliasing)
- Edge cases (empty, 1×1, rectangular, ragged)
- Determinism (re-run stability)
"""

import pytest
import random
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.canonicalization import (
    canonical_key, canonical_grid,
    all_isometries, apply_isometry
)
from src.utils import deep_eq, copy_grid


# ============================================================================
# Test Fixtures
# ============================================================================

# Empty grid
g0 = []

# 1x1 grid (all D8 transforms equal)
g1 = [[5]]

# 2x2 square
g2 = [[1, 2],
      [3, 4]]

# 2x3 rectangular (shape changes under transpose)
g3 = [[5, 6, 7],
      [8, 9, 0]]

# Symmetric 3x3 (tie-break test case)
g4 = [[1, 2, 1],
      [2, 3, 2],
      [1, 2, 1]]

# 1x4 horizontal
g5 = [[1, 2, 3, 4]]

# 4x1 vertical
g6 = [[1], [2], [3], [4]]

# Ragged (invalid)
g_ragged = [[1, 2], [3]]


# ============================================================================
# canonical_key Tests
# ============================================================================

def test_canonical_key_empty():
    """Empty grid returns (0, 0, ())."""
    assert canonical_key(g0) == (0, 0, ())


def test_canonical_key_1x1():
    """Single element grid returns (1, 1, (val,))."""
    assert canonical_key(g1) == (1, 1, (5,))


def test_canonical_key_2x2():
    """2x2 grid returns correct row-major tuple."""
    assert canonical_key(g2) == (2, 2, (1, 2, 3, 4))


def test_canonical_key_2x3():
    """2x3 grid returns correct shape and row-major values."""
    assert canonical_key(g3) == (2, 3, (5, 6, 7, 8, 9, 0))


def test_canonical_key_row_major_order():
    """Values are flattened in row-major order (left-to-right, top-to-bottom)."""
    g = [[1, 2, 3],
         [4, 5, 6]]
    key = canonical_key(g)
    assert key == (2, 3, (1, 2, 3, 4, 5, 6))


def test_canonical_key_ragged_raises():
    """Ragged input raises ValueError."""
    with pytest.raises(ValueError, match="rectangular"):
        canonical_key(g_ragged)


def test_canonical_key_determinism():
    """Same grid always returns same key."""
    for _ in range(10):
        assert canonical_key(g2) == (2, 2, (1, 2, 3, 4))


# ============================================================================
# Idempotence Tests (Π² = Π) - THE CRITICAL PROPERTY
# ============================================================================

def test_canonical_grid_idempotence_fixtures():
    """Π² = Π for all fixtures."""
    for g in [g0, g1, g2, g3, g4, g5, g6]:
        result1 = canonical_grid(g)
        result2 = canonical_grid(result1)
        assert deep_eq(result1, result2), f"Π² != Π for fixture {g}"


def test_canonical_grid_idempotence_random():
    """Π² = Π for 100 random small grids."""
    random.seed(42)  # Deterministic test
    for _ in range(100):
        # Generate random 2x2 to 5x5 grids
        rows = random.randint(2, 5)
        cols = random.randint(2, 5)
        g = [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

        result1 = canonical_grid(g)
        result2 = canonical_grid(result1)
        assert deep_eq(result1, result2), f"Π² != Π for random grid {g}"


# ============================================================================
# Minimality Tests
# ============================================================================

def test_canonical_grid_minimality_fixtures():
    """canonical_grid returns argmin over all D8 transforms."""
    for g in [g0, g1, g2, g3, g4, g5, g6]:
        result = canonical_grid(g)
        result_key = canonical_key(result)

        # Check that result_key is minimal among all D8 transforms
        for sigma in all_isometries():
            g_sigma = apply_isometry(g, sigma)
            sigma_key = canonical_key(g_sigma)
            assert result_key <= sigma_key, \
                f"canonical_grid not minimal for {g}: result_key={result_key}, {sigma}_key={sigma_key}"


def test_canonical_grid_2x2_minimality():
    """For 2x2 grid [[1,2],[3,4]], id should win (minimal key)."""
    g = [[1, 2], [3, 4]]
    result = canonical_grid(g)

    # Expected: id wins because (2,2,(1,2,3,4)) is lexicographically minimal
    assert deep_eq(result, [[1, 2], [3, 4]])


def test_canonical_grid_2x3_shape_comparison():
    """For 2x3 vs 3x2, shape-first ordering applies."""
    g = [[5, 6, 7], [8, 9, 0]]

    # Among all D8 transforms:
    # - (2,3) shapes: id, rot180, flip_h, flip_v
    # - (3,2) shapes: rot90, rot270, transpose, flip_anti
    # Shape-first ordering means (2,3,...) < (3,2,...) regardless of values

    # Among (2,3) shapes, compare values:
    # - id:      (2, 3, (5, 6, 7, 8, 9, 0))
    # - rot180:  (2, 3, (0, 9, 8, 7, 6, 5))  ← starts with 0, wins!
    # - flip_h:  (2, 3, (7, 6, 5, 0, 9, 8))
    # - flip_v:  (2, 3, (8, 9, 0, 5, 6, 7))

    result = canonical_grid(g)
    expected_key = (2, 3, (0, 9, 8, 7, 6, 5))  # rot180 wins
    assert canonical_key(result) == expected_key
    assert deep_eq(result, [[0, 9, 8], [7, 6, 5]])


# ============================================================================
# Deterministic Tie-Breaking Tests
# ============================================================================

def test_canonical_grid_tie_break_symmetric():
    """For symmetric grid, if multiple σ have same key, earliest σ wins."""
    # Create a fully symmetric grid (all same value)
    g = [[5, 5], [5, 5]]

    # All D8 transforms will have the same key: (2, 2, (5,5,5,5))
    result = canonical_grid(g)

    # id is first in all_isometries(), so it should win
    assert deep_eq(result, [[5, 5], [5, 5]])


def test_canonical_grid_tie_break_checkerboard():
    """For checkerboard pattern with symmetry, earliest σ wins."""
    # Checkerboard that looks same after rot180
    g = [[1, 2, 1],
         [2, 3, 2],
         [1, 2, 1]]

    result = canonical_grid(g)

    # Should be deterministic (id wins if keys equal)
    assert deep_eq(result, g)  # id is earliest


# ============================================================================
# Purity Tests
# ============================================================================

def test_canonical_grid_no_mutation():
    """Input grid is not mutated."""
    g_orig = [[1, 2, 3], [4, 5, 6]]
    g_snapshot = copy_grid(g_orig)

    _ = canonical_grid(g_orig)

    assert deep_eq(g_orig, g_snapshot)


def test_canonical_grid_no_aliasing():
    """Output grid has no row aliasing with input."""
    g = [[1, 2], [3, 4]]
    result = canonical_grid(g)

    # Modify result
    result[0][0] = 99

    # Original unchanged
    assert g[0][0] == 1


def test_canonical_key_no_mutation():
    """canonical_key does not mutate input."""
    g_orig = [[1, 2, 3], [4, 5, 6]]
    g_snapshot = copy_grid(g_orig)

    _ = canonical_key(g_orig)

    assert deep_eq(g_orig, g_snapshot)


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_canonical_grid_empty():
    """Empty grid returns empty grid."""
    assert canonical_grid(g0) == []


def test_canonical_grid_1x1():
    """1x1 grid returns itself (all D8 transforms equal)."""
    result = canonical_grid(g1)
    assert deep_eq(result, [[5]])


def test_canonical_grid_1x4():
    """1x4 grid canonicalizes correctly."""
    result = canonical_grid(g5)
    # Should return minimal key among all D8 transforms
    assert canonical_key(result) <= canonical_key(g5)


def test_canonical_grid_4x1():
    """4x1 grid canonicalizes correctly."""
    result = canonical_grid(g6)
    # Should return minimal key among all D8 transforms
    assert canonical_key(result) <= canonical_key(g6)


def test_canonical_grid_ragged_raises():
    """Ragged input raises ValueError."""
    with pytest.raises(ValueError, match="rectangular"):
        canonical_grid(g_ragged)


# ============================================================================
# Determinism Tests
# ============================================================================

def test_canonical_grid_determinism():
    """Running canonical_grid twice yields identical results."""
    for g in [g0, g1, g2, g3, g4, g5, g6]:
        result1 = canonical_grid(g)
        result2 = canonical_grid(g)
        assert deep_eq(result1, result2)


def test_canonical_key_determinism():
    """Running canonical_key twice yields identical results."""
    for g in [g0, g1, g2, g3, g4, g5, g6]:
        key1 = canonical_key(g)
        key2 = canonical_key(g)
        assert key1 == key2


# ============================================================================
# Specific Transformation Tests
# ============================================================================

def test_canonical_grid_selects_lexmin():
    """Verify canonical_grid selects lexicographically minimal transform."""
    # Grid where rot90 has different key than id
    g = [[1, 2], [3, 4]]

    # Collect all keys
    keys = {}
    for sigma in all_isometries():
        g_sigma = apply_isometry(g, sigma)
        keys[sigma] = canonical_key(g_sigma)

    # Find manual minimum
    min_key = min(keys.values())

    # canonical_grid result should have this min key
    result = canonical_grid(g)
    assert canonical_key(result) == min_key


def test_canonical_grid_all_transforms_different():
    """For asymmetric grid, verify each D8 transform has unique key."""
    # Asymmetric grid
    g = [[1, 2, 3],
         [4, 5, 6]]

    # Collect all keys
    keys = []
    for sigma in all_isometries():
        g_sigma = apply_isometry(g, sigma)
        key = canonical_key(g_sigma)
        keys.append(key)

    # Not all keys are the same (asymmetric grid)
    assert len(set(keys)) > 1


# ============================================================================
# Integration Test: Canonical Form Equivalence
# ============================================================================

def test_canonical_grid_d8_equivalents_same_canonical():
    """All D8 transforms of a grid have the same canonical form."""
    g = [[1, 2], [3, 4]]

    # Compute canonical form of original
    canonical_orig = canonical_grid(g)

    # For each D8 transform, canonical form should be the same
    for sigma in all_isometries():
        g_sigma = apply_isometry(g, sigma)
        canonical_sigma = canonical_grid(g_sigma)
        assert deep_eq(canonical_orig, canonical_sigma), \
            f"canonical_grid({sigma}(g)) != canonical_grid(g)"


# ============================================================================
# Property-Based Test: Idempotence Always Holds
# ============================================================================

def test_canonical_grid_idempotence_various_sizes():
    """Π² = Π for various grid sizes."""
    test_grids = [
        [],
        [[1]],
        [[1, 2]],
        [[1], [2]],
        [[1, 2, 3]],
        [[1], [2], [3]],
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2, 3, 4]],
        [[1], [2], [3], [4]],
    ]

    for g in test_grids:
        result1 = canonical_grid(g)
        result2 = canonical_grid(result1)
        assert deep_eq(result1, result2), f"Π² != Π for {g}"
