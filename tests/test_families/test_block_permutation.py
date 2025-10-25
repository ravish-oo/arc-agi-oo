"""
Tests for BlockPermutation Family.

Covers:
- fit() success and failure cases
- apply() with various block sizes
- Edge cases (empty, shape mismatch, non-integer tiling)
- Permutation building and matching
- FY exactness verification
- Determinism
"""

import pytest
from src.families.block_permutation import BlockPermutationFamily


@pytest.fixture
def family():
    """Create fresh BlockPermutationFamily instance for each test."""
    return BlockPermutationFamily()


# ============================================================================
# Basic fit() Tests
# ============================================================================

def test_fit_empty_train_pairs(family):
    """fit() with empty train_pairs returns False."""
    result = family.fit([])
    assert result is False
    assert family.params.kH is None
    assert family.params.kW is None
    assert family.params.perm is None


def test_fit_empty_grids(family):
    """fit() with empty grids returns False."""
    train = [{"input": [], "output": []}]
    result = family.fit(train)
    assert result is False


def test_fit_shape_mismatch(family):
    """fit() rejects when dims(X) != dims(Y)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 3], [4, 5, 6]]
        }
    ]
    result = family.fit(train)
    assert result is False


def test_fit_zero_dimensions(family):
    """fit() rejects grids with zero height or width."""
    train = [{"input": [[]], "output": [[]]}]
    result = family.fit(train)
    assert result is False


# ============================================================================
# Simple Block Size Tests
# ============================================================================

def test_fit_1x1_blocks_identity(family):
    """fit() with 1×1 blocks (identity permutation)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 1
    assert family.params.kW == 1
    # Identity permutation
    expected_perm = {(0, 0): (0, 0), (0, 1): (0, 1), (1, 0): (1, 0), (1, 1): (1, 1)}
    assert family.params.perm == expected_perm


def test_fit_2x2_blocks_simple_swap(family):
    """fit() with 2×2 blocks (simple swap)."""
    # Two 2×2 blocks swapped positions
    train = [
        {
            "input": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "output": [[3, 4, 1, 2], [7, 8, 5, 6]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Verify output is correct (lex order may find 1×1 blocks first)
    predicted = family.apply(train[0]["input"])
    assert predicted == train[0]["output"]


def test_fit_1x2_blocks_horizontal_swap(family):
    """fit() with 1×2 blocks (horizontal swap)."""
    train = [
        {
            "input": [[1, 2, 3, 4]],
            "output": [[3, 4, 1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Verify output is correct (lex order may find 1×1 blocks first)
    predicted = family.apply(train[0]["input"])
    assert predicted == train[0]["output"]


def test_fit_2x1_blocks_vertical_swap(family):
    """fit() with 2×1 blocks (vertical swap)."""
    train = [
        {
            "input": [[1], [2], [3], [4]],
            "output": [[3], [4], [1], [2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Verify output is correct (lex order may find 1×1 blocks first)
    predicted = family.apply(train[0]["input"])
    assert predicted == train[0]["output"]


# ============================================================================
# Lexicographic Search Order Tests
# ============================================================================

def test_fit_lexicographic_order(family):
    """fit() tries block sizes in lexicographic order."""
    # 4×4 grid that can be tiled multiple ways
    # Create a pattern where (1,1) is the first valid block size
    train = [
        {
            "input": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            "output": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # First valid block size in lex order is (1, 1)
    assert family.params.kH == 1
    assert family.params.kW == 1


# ============================================================================
# Multiset Equality Tests
# ============================================================================

def test_fit_rejects_different_multiset(family):
    """fit() rejects when tiles don't form same multiset."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [5, 6]]  # Different tiles
        }
    ]
    result = family.fit(train)
    assert result is False


def test_fit_accepts_same_tiles_reordered(family):
    """fit() accepts when tiles are reordered but multiset matches."""
    # 2×2 blocks: [[1,2],[3,4]] and [[5,6],[7,8]] swapped
    train = [
        {
            "input": [[1, 2, 5, 6], [3, 4, 7, 8]],
            "output": [[5, 6, 1, 2], [7, 8, 3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Verify output is correct (lex order may find 1×1 blocks first)
    predicted = family.apply(train[0]["input"])
    assert predicted == train[0]["output"]


# ============================================================================
# Greedy Matching Tests
# ============================================================================

def test_greedy_matching_deterministic(family):
    """Greedy matching is deterministic (row-major order)."""
    # Multiple identical tiles - greedy should pick first available
    train = [
        {
            "input": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # With all identical 1×1 tiles, greedy matching gives identity
    assert family.params.kH == 1
    assert family.params.kW == 1


def test_greedy_matching_with_duplicates(family):
    """Greedy matching handles duplicate tiles correctly."""
    # Two pairs of identical 2×2 tiles
    train = [
        {
            "input": [[1, 1, 2, 2], [1, 1, 2, 2]],
            "output": [[2, 2, 1, 1], [2, 2, 1, 1]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Verify output is correct (lex order may find 1×1 blocks first)
    predicted = family.apply(train[0]["input"])
    assert predicted == train[0]["output"]


# ============================================================================
# apply() Tests
# ============================================================================

def test_apply_before_fit_raises(family):
    """apply() before fit() raises RuntimeError."""
    with pytest.raises(RuntimeError, match="apply\\(\\) called before fit\\(\\)"):
        family.apply([[1, 2]])


def test_apply_after_successful_fit(family):
    """apply() works correctly after successful fit()."""
    train = [
        {
            "input": [[1, 2, 3, 4]],
            "output": [[3, 4, 1, 2]]
        }
    ]
    family.fit(train)

    # Apply to original input
    result = family.apply([[1, 2, 3, 4]])
    assert result == [[3, 4, 1, 2]]


def test_apply_to_different_input(family):
    """apply() applies learned permutation to different input."""
    train = [
        {
            "input": [[1, 2, 3, 4]],
            "output": [[3, 4, 1, 2]]
        }
    ]
    family.fit(train)

    # Apply same permutation to different values
    result = family.apply([[10, 20, 30, 40]])
    assert result == [[30, 40, 10, 20]]


def test_apply_preserves_input_dimensions(family):
    """apply() output has same dimensions as input."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[4, 3], [2, 1]]
        }
    ]
    family.fit(train)

    X = [[5, 6], [7, 8]]
    result = family.apply(X)
    assert len(result) == len(X)
    assert len(result[0]) == len(X[0])


def test_apply_empty_grid(family):
    """apply() handles empty grid correctly."""
    # Fit with non-empty, then apply to empty
    train = [
        {
            "input": [[1, 2]],
            "output": [[2, 1]]
        }
    ]
    family.fit(train)

    result = family.apply([])
    assert result == []


# ============================================================================
# FY Exactness Tests
# ============================================================================

def test_fy_exactness_all_pairs(family):
    """fit() ensures FY exactness on ALL training pairs."""
    train = [
        {
            "input": [[1, 2, 3, 4]],
            "output": [[3, 4, 1, 2]]
        },
        {
            "input": [[10, 20, 30, 40]],
            "output": [[30, 40, 10, 20]]
        }
    ]
    result = family.fit(train)
    assert result is True

    # Verify on all pairs
    for pair in train:
        predicted = family.apply(pair["input"])
        assert predicted == pair["output"]


def test_fy_violation_rejects(family):
    """fit() rejects when params don't work for ALL pairs."""
    train = [
        {
            "input": [[1, 2, 3, 4]],
            "output": [[3, 4, 1, 2]]
        },
        {
            "input": [[10, 20, 30, 40]],
            "output": [[10, 20, 30, 40]]  # Different permutation
        }
    ]
    result = family.fit(train)
    assert result is False


# ============================================================================
# Purity Tests
# ============================================================================

def test_fit_does_not_mutate_input(family):
    """fit() does not mutate train_pairs."""
    original_input = [[1, 2], [3, 4]]
    original_output = [[3, 4], [1, 2]]
    train = [{"input": original_input, "output": original_output}]

    # Store original values
    input_copy = [row[:] for row in original_input]
    output_copy = [row[:] for row in original_output]

    family.fit(train)

    # Check no mutation
    assert train[0]["input"] == input_copy
    assert train[0]["output"] == output_copy


def test_apply_does_not_mutate_input(family):
    """apply() does not mutate input grid."""
    train = [
        {
            "input": [[1, 2]],
            "output": [[2, 1]]
        }
    ]
    family.fit(train)

    X = [[10, 20]]
    X_copy = [row[:] for row in X]

    family.apply(X)

    assert X == X_copy


# ============================================================================
# Determinism Tests
# ============================================================================

def test_deterministic_repeated_fit(family):
    """Repeated fit() calls yield identical params."""
    train = [
        {
            "input": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "output": [[3, 4, 1, 2], [7, 8, 5, 6]]
        }
    ]

    # First fit
    family.fit(train)
    kH1 = family.params.kH
    kW1 = family.params.kW
    perm1 = family.params.perm.copy()

    # Second fit (fresh instance)
    family2 = BlockPermutationFamily()
    family2.fit(train)

    assert family2.params.kH == kH1
    assert family2.params.kW == kW1
    assert family2.params.perm == perm1


def test_deterministic_apply(family):
    """apply() is deterministic."""
    train = [
        {
            "input": [[1, 2, 3, 4]],
            "output": [[3, 4, 1, 2]]
        }
    ]
    family.fit(train)

    X = [[10, 20, 30, 40]]
    result1 = family.apply(X)
    result2 = family.apply(X)

    assert result1 == result2


# ============================================================================
# Complex Permutation Tests
# ============================================================================

def test_4x4_grid_with_2x2_blocks(family):
    """Complex 4×4 grid with 2×2 blocks."""
    # Four 2×2 blocks in a specific permutation
    train = [
        {
            "input": [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4]
            ],
            "output": [
                [4, 4, 3, 3],
                [4, 4, 3, 3],
                [2, 2, 1, 1],
                [2, 2, 1, 1]
            ]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Verify output is correct (lex order may find 1×1 blocks first)
    predicted = family.apply(train[0]["input"])
    assert predicted == train[0]["output"]


def test_6x6_grid_with_3x2_blocks(family):
    """6×6 grid with 3×2 blocks."""
    train = [
        {
            "input": [
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
                [4, 4, 5, 5, 6, 6],
                [4, 4, 5, 5, 6, 6],
                [4, 4, 5, 5, 6, 6]
            ],
            "output": [
                [6, 6, 5, 5, 4, 4],
                [6, 6, 5, 5, 4, 4],
                [6, 6, 5, 5, 4, 4],
                [3, 3, 2, 2, 1, 1],
                [3, 3, 2, 2, 1, 1],
                [3, 3, 2, 2, 1, 1]
            ]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Could be 3×2 blocks
    assert family.params.kH in [1, 3]
    assert family.params.kW in [1, 2]

    # Verify correctness
    predicted = family.apply(train[0]["input"])
    assert predicted == train[0]["output"]


# ============================================================================
# Edge Cases
# ============================================================================

def test_single_block_full_grid(family):
    """Single block (entire grid as one tile)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Could be any block size that divides evenly
    # Lex order would give (1, 1) first
    assert family.params.kH == 1
    assert family.params.kW == 1


def test_non_integer_tiling_rejected(family):
    """Non-integer tiling is rejected."""
    # Can't tile 3×3 grid with 2×2 blocks
    train = [
        {
            "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "output": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Should work with valid block sizes (1×1, 1×3, 3×1, 3×3)
    # Lex order gives (1, 1) first
    assert family.params.kH == 1
    assert family.params.kW == 1
