"""
Comprehensive tests for OFA (Order-of-First-Appearance) patch normalization
and D8 patch canonicalization.

Tests cover:
- OFA locality (palette permutations normalize identically)
- OFA row-major order
- canonical_patch_key correctness
- Π² = Π (idempotence) on patches - THE CRITICAL PROPERTY
- D8 minimality
- Deterministic tie-breaking
- Purity (no mutation, no aliasing)
- Edge cases (empty, 1×1, 1×N, N×1, ragged)
- Determinism (re-run stability)
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.canonicalization import (
    ofa_normalize_patch_colors,
    canonical_patch_key,
    canonical_d8_patch,
    all_isometries,
    apply_isometry
)
from src.utils import deep_eq, copy_grid


# ============================================================================
# Test Fixtures
# ============================================================================

# Empty patch
p0 = []

# Single cell
p1 = [[5]]

# Row with distinct colors (OFA order test)
p2 = [[7, 3, 7, 5]]

# Column
p3 = [[2], [5], [2]]

# 2×2 symmetric (tie-break test)
p4 = [[1, 1], [1, 1]]

# 2×3 asymmetric
p5 = [[9, 1], [3, 1], [3, 9]]

# Palette permutation pairs (OFA locality test)
p6 = [[5, 7], [7, 5]]
p7 = [[3, 2], [2, 3]]

# Ragged (invalid)
p_ragged = [[1, 2], [3]]


# ============================================================================
# OFA Locality Tests (CRITICAL for patch-local property)
# ============================================================================

def test_ofa_locality_palette_permutation():
    """
    Same pattern with different colors normalizes identically.
    [[5,7],[7,5]] and [[3,2],[2,3]] both have same structure.
    """
    result1 = ofa_normalize_patch_colors(p6)
    result2 = ofa_normalize_patch_colors(p7)

    # Both should normalize to [[0,1],[1,0]]
    # (first color in row-major → 0, second → 1)
    expected = [[0, 1], [1, 0]]

    assert deep_eq(result1, expected), f"p6 normalized to {result1}, expected {expected}"
    assert deep_eq(result2, expected), f"p7 normalized to {result2}, expected {expected}"


def test_ofa_locality_various_permutations():
    """Multiple palette permutations of same pattern normalize identically."""
    # Pattern: checkerboard
    patterns = [
        [[1, 2], [2, 1]],
        [[5, 7], [7, 5]],
        [[0, 9], [9, 0]],
        [[3, 2], [2, 3]],
    ]

    expected = [[0, 1], [1, 0]]  # OFA normalizes all to same

    for p in patterns:
        result = ofa_normalize_patch_colors(p)
        assert deep_eq(result, expected), f"Pattern {p} normalized to {result}, expected {expected}"


# ============================================================================
# OFA Row-Major Order Tests
# ============================================================================

def test_ofa_row_major_order_row():
    """OFA scans row-major: [[7,3,7,5]] → 7→0, 3→1, 5→2"""
    result = ofa_normalize_patch_colors(p2)
    expected = [[0, 1, 0, 2]]  # 7 first→0, 3 second→1, 7 again→0, 5 third→2

    assert deep_eq(result, expected)


def test_ofa_row_major_order_column():
    """OFA scans row-major even for column patches: [[2],[5],[2]] → 2→0, 5→1"""
    result = ofa_normalize_patch_colors(p3)
    expected = [[0], [1], [0]]  # 2 first→0, 5 second→1, 2 again→0

    assert deep_eq(result, expected)


def test_ofa_row_major_2x2():
    """OFA scans left→right, top→bottom"""
    p = [[1, 2],
         [3, 1]]
    result = ofa_normalize_patch_colors(p)
    expected = [[0, 1], [2, 0]]  # 1→0, 2→1, 3→2, 1 again→0

    assert deep_eq(result, expected)


# ============================================================================
# canonical_patch_key Tests
# ============================================================================

def test_canonical_patch_key_empty():
    """Empty patch returns (0, 0, ())."""
    assert canonical_patch_key(p0) == (0, 0, ())


def test_canonical_patch_key_single_cell():
    """Single cell [[x]] returns (1, 1, (0,)) — OFA maps any x→0"""
    assert canonical_patch_key(p1) == (1, 1, (0,))
    assert canonical_patch_key([[7]]) == (1, 1, (0,))
    assert canonical_patch_key([[0]]) == (1, 1, (0,))


def test_canonical_patch_key_row():
    """Row [[7,3,7,5]] returns (1, 4, (0,1,0,2))"""
    expected = (1, 4, (0, 1, 0, 2))
    assert canonical_patch_key(p2) == expected


def test_canonical_patch_key_shape_first():
    """Shape-first ordering: lexicographic comparison on (rows, cols, ...)"""
    p_2x1 = [[1], [2]]
    p_1x2 = [[1, 2]]

    key_2x1 = canonical_patch_key(p_2x1)  # (2, 1, (0, 1))
    key_1x2 = canonical_patch_key(p_1x2)  # (1, 2, (0, 1))

    # Lexicographic: compare first element first
    # (2, 1, ...) > (1, 2, ...) because 2 > 1
    assert key_2x1 > key_1x2


def test_canonical_patch_key_ragged_raises():
    """Ragged input raises ValueError."""
    with pytest.raises(ValueError, match="rectangular"):
        canonical_patch_key(p_ragged)


# ============================================================================
# Idempotence Tests (Π² = Π) - THE CRITICAL PROPERTY
# ============================================================================

def test_canonical_d8_patch_idempotence_fixtures():
    """Π² = Π for all fixtures."""
    for p in [p0, p1, p2, p3, p4, p5, p6, p7]:
        result1 = canonical_d8_patch(p)
        result2 = canonical_d8_patch(result1)
        assert deep_eq(result1, result2), f"Π² != Π for patch {p}"


def test_canonical_d8_patch_idempotence_various_sizes():
    """Π² = Π for various patch sizes."""
    test_patches = [
        [],
        [[1]],
        [[1, 2]],
        [[1], [2]],
        [[1, 2, 3]],
        [[1], [2], [3]],
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6]],
        [[5, 7, 2], [1, 5, 1], [2, 7, 5]],
    ]

    for p in test_patches:
        result1 = canonical_d8_patch(p)
        result2 = canonical_d8_patch(result1)
        assert deep_eq(result1, result2), f"Π² != Π for {p}"


# ============================================================================
# D8 Minimality Tests
# ============================================================================

def test_canonical_d8_patch_minimality():
    """canonical_d8_patch returns argmin over all D8 transforms."""
    for p in [p0, p1, p2, p3, p4, p5, p6]:
        result = canonical_d8_patch(p)
        result_key = canonical_patch_key(result)

        # Check that result_key is minimal among all D8 transforms
        for sigma in all_isometries():
            p_sigma = apply_isometry(p, sigma)
            sigma_key = canonical_patch_key(p_sigma)
            assert result_key <= sigma_key, \
                f"canonical_d8_patch not minimal for {p}: result_key={result_key}, {sigma}_key={sigma_key}"


def test_canonical_d8_patch_selects_lexmin():
    """Verify canonical_d8_patch selects lexicographically minimal transform."""
    p = [[1, 2], [3, 4]]

    # Collect all keys
    keys = {}
    for sigma in all_isometries():
        p_sigma = apply_isometry(p, sigma)
        keys[sigma] = canonical_patch_key(p_sigma)

    # Find manual minimum
    min_key = min(keys.values())

    # canonical_d8_patch result should have this min key
    result = canonical_d8_patch(p)
    assert canonical_patch_key(result) == min_key


# ============================================================================
# Deterministic Tie-Breaking Tests
# ============================================================================

def test_canonical_d8_patch_tie_break_all_same():
    """For all-same patch, all D8 transforms equal → id wins (earliest σ)."""
    p = [[5, 5], [5, 5]]

    result = canonical_d8_patch(p)

    # All D8 transforms will have same key: (2, 2, (0,0,0,0))
    # id is first in all_isometries(), so it should win
    # OFA normalizes to [[0,0],[0,0]]
    expected = [[0, 0], [0, 0]]
    assert deep_eq(result, expected)


def test_canonical_d8_patch_tie_break_determinism():
    """Same patch always yields same canonical form (deterministic tie-break)."""
    p = [[1, 1], [1, 1]]

    results = [canonical_d8_patch(p) for _ in range(10)]

    # All results should be identical
    for result in results:
        assert deep_eq(result, results[0])


# ============================================================================
# Purity Tests
# ============================================================================

def test_ofa_normalize_patch_colors_no_mutation():
    """Input patch is not mutated."""
    p_orig = [[7, 3, 7, 5]]
    p_snapshot = copy_grid(p_orig)

    _ = ofa_normalize_patch_colors(p_orig)

    assert deep_eq(p_orig, p_snapshot)


def test_canonical_patch_key_no_mutation():
    """Input patch is not mutated."""
    p_orig = [[7, 3, 7, 5]]
    p_snapshot = copy_grid(p_orig)

    _ = canonical_patch_key(p_orig)

    assert deep_eq(p_orig, p_snapshot)


def test_canonical_d8_patch_no_mutation():
    """Input patch is not mutated."""
    p_orig = [[1, 2], [3, 4]]
    p_snapshot = copy_grid(p_orig)

    _ = canonical_d8_patch(p_orig)

    assert deep_eq(p_orig, p_snapshot)


def test_ofa_normalize_patch_colors_no_aliasing():
    """Output rows are not aliases of input rows."""
    p = [[1, 2], [3, 4]]
    result = ofa_normalize_patch_colors(p)

    # Modify result
    result[0][0] = 99

    # Original unchanged
    assert p[0][0] == 1


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_ofa_normalize_patch_colors_empty():
    """Empty patch returns empty patch."""
    assert ofa_normalize_patch_colors(p0) == []


def test_ofa_normalize_patch_colors_single_cell():
    """Single cell [[x]] maps to [[0]]."""
    assert deep_eq(ofa_normalize_patch_colors(p1), [[0]])
    assert deep_eq(ofa_normalize_patch_colors([[7]]), [[0]])


def test_ofa_normalize_patch_colors_single_color():
    """Patch with single color maps all to 0."""
    p = [[3, 3, 3], [3, 3, 3]]
    result = ofa_normalize_patch_colors(p)
    expected = [[0, 0, 0], [0, 0, 0]]

    assert deep_eq(result, expected)


def test_canonical_d8_patch_empty():
    """Empty patch returns empty patch."""
    assert canonical_d8_patch(p0) == []


def test_canonical_d8_patch_single_cell():
    """Single cell [[x]] returns [[0]] (OFA maps to 0)."""
    result = canonical_d8_patch(p1)
    assert deep_eq(result, [[0]])


def test_ofa_normalize_patch_colors_ragged_raises():
    """Ragged input raises ValueError."""
    with pytest.raises(ValueError, match="rectangular"):
        ofa_normalize_patch_colors(p_ragged)


def test_canonical_d8_patch_ragged_raises():
    """Ragged input raises ValueError."""
    with pytest.raises(ValueError, match="rectangular"):
        canonical_d8_patch(p_ragged)


# ============================================================================
# Determinism Tests
# ============================================================================

def test_ofa_normalize_patch_colors_determinism():
    """Same patch always yields same normalized result."""
    for p in [p1, p2, p3, p4, p5, p6]:
        result1 = ofa_normalize_patch_colors(p)
        result2 = ofa_normalize_patch_colors(p)
        assert deep_eq(result1, result2)


def test_canonical_patch_key_determinism():
    """Same patch always yields same key."""
    for p in [p0, p1, p2, p3, p4, p5]:
        key1 = canonical_patch_key(p)
        key2 = canonical_patch_key(p)
        assert key1 == key2


def test_canonical_d8_patch_determinism():
    """Same patch always yields same canonical form."""
    for p in [p0, p1, p2, p3, p4, p5]:
        result1 = canonical_d8_patch(p)
        result2 = canonical_d8_patch(p)
        assert deep_eq(result1, result2)


# ============================================================================
# Integration Test: D8 Equivalence
# ============================================================================

def test_canonical_d8_patch_d8_equivalents_same_canonical():
    """All D8 transforms of a patch have the same canonical form."""
    p = [[1, 2], [3, 4]]

    # Compute canonical form of original
    canonical_orig = canonical_d8_patch(p)

    # For each D8 transform, canonical form should be the same
    for sigma in all_isometries():
        p_sigma = apply_isometry(p, sigma)
        canonical_sigma = canonical_d8_patch(p_sigma)
        assert deep_eq(canonical_orig, canonical_sigma), \
            f"canonical_d8_patch({sigma}(p)) != canonical_d8_patch(p)"


# ============================================================================
# OFA Compact Palette Test
# ============================================================================

def test_ofa_normalize_patch_colors_compact_palette():
    """OFA uses compact palette {0..k-1} for k distinct colors."""
    p = [[5, 7, 5], [2, 7, 2]]  # 3 distinct colors: 5, 7, 2

    result = ofa_normalize_patch_colors(p)

    # Flatten and check all values are in {0, 1, 2}
    flat = [val for row in result for val in row]
    assert set(flat) == {0, 1, 2}

    # Verify: 5→0, 7→1, 2→2 (row-major order)
    expected = [[0, 1, 0], [2, 1, 2]]
    assert deep_eq(result, expected)


# ============================================================================
# Specific Transformation Tests
# ============================================================================

def test_canonical_d8_patch_returns_ofa_normalized():
    """canonical_d8_patch returns OFA-normalized result (not raw transform)."""
    p = [[5, 7], [7, 5]]

    result = canonical_d8_patch(p)

    # Result should be OFA-normalized (using {0, 1}, not {5, 7})
    flat = [val for row in result for val in row]
    assert max(flat) <= 1  # OFA remaps to {0, 1}
    assert min(flat) >= 0
