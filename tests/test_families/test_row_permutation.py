"""
Tests for RowPermutation Family.

Covers:
- fit() success and failure cases
- apply() with various permutations
- Edge cases (empty, single row, identity)
- Shape preservation
- Row multiset equality
- Permutation bijectivity
- Deterministic tie-breaking
- FY exactness verification
- Determinism and purity
"""

import pytest
from src.families.row_permutation import RowPermutationFamily


@pytest.fixture
def family():
    """Create fresh RowPermutationFamily instance for each test."""
    return RowPermutationFamily()


# ============================================================================
# Basic fit() Tests
# ============================================================================

def test_fit_empty_train_pairs(family):
    """fit() with empty train_pairs returns False."""
    result = family.fit([])
    assert result is False
    assert family.params.perm is None


def test_fit_empty_grids(family):
    """fit() with empty grids returns False."""
    train = [{"input": [], "output": []}]
    result = family.fit(train)
    assert result is False


def test_fit_zero_dimensions(family):
    """fit() rejects grids with zero dimensions."""
    train = [{"input": [[]], "output": [[]]}]
    result = family.fit(train)
    assert result is False


def test_fit_identity_permutation(family):
    """fit() with identity permutation (Y == X)."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[1, 2], [3, 4], [5, 6]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.perm == [0, 1, 2]


def test_fit_reverse_permutation(family):
    """fit() with reverse permutation."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[5, 6], [3, 4], [1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.perm == [2, 1, 0]


def test_fit_swap_first_last(family):
    """fit() with swap of first and last rows."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[5, 6], [3, 4], [1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # X[0]→Y[2], X[1]→Y[1], X[2]→Y[0]
    assert family.params.perm == [2, 1, 0]


def test_fit_cyclic_rotation(family):
    """fit() with cyclic rotation of rows."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[3, 4], [5, 6], [1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # X[0]→Y[2], X[1]→Y[0], X[2]→Y[1]
    assert family.params.perm == [2, 0, 1]


def test_fit_single_row(family):
    """fit() with single row (identity)."""
    train = [
        {
            "input": [[1, 2, 3]],
            "output": [[1, 2, 3]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.perm == [0]


# ============================================================================
# Shape Preservation Tests
# ============================================================================

def test_reject_different_row_counts(family):
    """fit() rejects when row counts differ."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4], [5, 6]]
        }
    ]
    result = family.fit(train)
    assert result is False


def test_reject_different_column_counts(family):
    """fit() rejects when column counts differ."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 3], [4, 5, 6]]
        }
    ]
    result = family.fit(train)
    assert result is False


def test_accept_same_shape(family):
    """fit() accepts when shapes match."""
    train = [
        {
            "input": [[1, 2, 3], [4, 5, 6]],
            "output": [[4, 5, 6], [1, 2, 3]]
        }
    ]
    result = family.fit(train)
    assert result is True


# ============================================================================
# Row Multiset Equality Tests
# ============================================================================

def test_reject_different_row_multiset(family):
    """fit() rejects when row multisets differ."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [5, 6]]  # Different row
        }
    ]
    result = family.fit(train)
    assert result is False


def test_accept_same_rows_different_order(family):
    """fit() accepts when same rows in different order."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[5, 6], [1, 2], [3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True


# ============================================================================
# Deterministic Tie-Breaking Tests
# ============================================================================

def test_deterministic_tie_break_duplicate_rows(family):
    """fit() uses deterministic tie-breaking for duplicate rows."""
    train = [
        {
            "input": [[1, 2], [1, 2], [3, 4]],
            "output": [[1, 2], [3, 4], [1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Greedy first-fit:
    # Y[0]=[1,2] → first unused X with [1,2] is i=0 → inv_perm[0]=0
    # Y[1]=[3,4] → first unused X with [3,4] is i=2 → inv_perm[1]=2
    # Y[2]=[1,2] → first unused X with [1,2] is i=1 → inv_perm[2]=1
    # inv_perm = [0, 2, 1] → perm = [0, 2, 1]
    assert family.params.perm == [0, 2, 1]


def test_deterministic_repeated_fit(family):
    """Repeated fit() calls yield identical perm."""
    train = [
        {
            "input": [[5, 6], [3, 4], [1, 2]],
            "output": [[1, 2], [5, 6], [3, 4]]
        }
    ]

    # First fit
    family.fit(train)
    perm1 = family.params.perm[:]

    # Second fit (fresh instance)
    family2 = RowPermutationFamily()
    family2.fit(train)

    assert family2.params.perm == perm1


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
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        }
    ]
    family.fit(train)

    # Apply to original input
    result = family.apply([[1, 2], [3, 4]])
    assert result == [[3, 4], [1, 2]]


def test_apply_to_different_input(family):
    """apply() applies learned permutation to different input."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        }
    ]
    family.fit(train)

    # Apply same permutation to different values
    result = family.apply([[10, 20], [30, 40]])
    assert result == [[30, 40], [10, 20]]


def test_apply_preserves_shape(family):
    """apply() output has same shape as input."""
    train = [
        {
            "input": [[1, 2, 3], [4, 5, 6]],
            "output": [[4, 5, 6], [1, 2, 3]]
        }
    ]
    family.fit(train)

    X = [[10, 20, 30], [40, 50, 60]]
    result = family.apply(X)
    assert len(result) == len(X)
    assert len(result[0]) == len(X[0])


def test_apply_empty_grid(family):
    """apply() handles empty grid correctly."""
    train = [
        {
            "input": [[1]],
            "output": [[1]]
        }
    ]
    family.fit(train)

    result = family.apply([])
    assert result == []


# ============================================================================
# Permutation Bijectivity Tests
# ============================================================================

def test_permutation_is_bijection(family):
    """fit() produces a valid bijection."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[5, 6], [1, 2], [3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True

    perm = family.params.perm
    # Check perm is bijection [0..h-1] → [0..h-1]
    assert len(perm) == 3
    assert set(perm) == {0, 1, 2}
    # Each index appears exactly once
    assert len(set(perm)) == len(perm)


def test_inverse_permutation_correctness(family):
    """Verify inverse permutation is computed correctly."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[3, 4], [5, 6], [1, 2]]
        }
    ]
    family.fit(train)
    # perm: X[0]→Y[2], X[1]→Y[0], X[2]→Y[1]
    # perm = [2, 0, 1]

    X = [[1, 2], [3, 4], [5, 6]]
    result = family.apply(X)
    # inv_perm: Y[0]←X[1], Y[1]←X[2], Y[2]←X[0]
    # Y = [[3,4], [5,6], [1,2]]
    assert result == [[3, 4], [5, 6], [1, 2]]


# ============================================================================
# FY Exactness Tests
# ============================================================================

def test_fy_exact_equality_required(family):
    """fit() rejects if output differs by single element."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        },
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 99]]  # Single element different
        }
    ]
    result = family.fit(train)
    assert result is False


def test_fy_all_pairs_must_match(family):
    """fit() requires ALL pairs to match (not just some)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        },
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        },
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4]]  # Different permutation
        }
    ]
    result = family.fit(train)
    assert result is False


# ============================================================================
# Unified Parameters Tests
# ============================================================================

def test_unified_perm_multiple_pairs(family):
    """fit() uses same permutation for all pairs."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        },
        {
            "input": [[10, 20], [30, 40]],
            "output": [[30, 40], [10, 20]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.perm == [1, 0]


def test_reject_inconsistent_permutations(family):
    """fit() rejects when pairs require different permutations."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]  # perm=[1,0]
        },
        {
            "input": [[10, 20], [30, 40]],
            "output": [[10, 20], [30, 40]]  # perm=[0,1]
        }
    ]
    result = family.fit(train)
    assert result is False


# ============================================================================
# Edge Cases
# ============================================================================

def test_two_rows_swap(family):
    """fit() with two rows swapped."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.perm == [1, 0]


def test_large_grid(family):
    """fit() and apply() with larger grid."""
    train = [
        {
            "input": [[1], [2], [3], [4], [5]],
            "output": [[5], [4], [3], [2], [1]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.perm == [4, 3, 2, 1, 0]


def test_all_identical_rows(family):
    """fit() with all identical rows (any perm works)."""
    train = [
        {
            "input": [[1, 2], [1, 2], [1, 2]],
            "output": [[1, 2], [1, 2], [1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # Greedy first-fit produces specific perm
    # Y[0]=[1,2] → X[0], Y[1]=[1,2] → X[1], Y[2]=[1,2] → X[2]
    assert family.params.perm == [0, 1, 2]


# ============================================================================
# Purity Tests
# ============================================================================

def test_fit_does_not_mutate_train_pairs(family):
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
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        }
    ]
    family.fit(train)

    X = [[1, 2], [3, 4]]
    X_copy = [row[:] for row in X]

    family.apply(X)

    assert X == X_copy


def test_apply_no_row_aliasing(family):
    """apply() output has no row aliasing with input."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [1, 2]]
        }
    ]
    family.fit(train)

    X = [[1, 2], [3, 4]]
    result = family.apply(X)

    # Mutating result should not affect X
    result[0][0] = 999
    assert X[0][0] == 1


# ============================================================================
# Determinism Tests
# ============================================================================

def test_deterministic_apply(family):
    """apply() is deterministic."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [[5, 6], [1, 2], [3, 4]]
        }
    ]
    family.fit(train)

    X = [[10, 20], [30, 40], [50, 60]]
    result1 = family.apply(X)
    result2 = family.apply(X)

    assert result1 == result2


# ============================================================================
# Complex Permutation Tests
# ============================================================================

def test_complex_permutation_4_rows(family):
    """fit() with complex 4-row permutation."""
    train = [
        {
            "input": [[1, 1], [2, 2], [3, 3], [4, 4]],
            "output": [[3, 3], [1, 1], [4, 4], [2, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # X[0]→Y[1], X[1]→Y[3], X[2]→Y[0], X[3]→Y[2]
    assert family.params.perm == [1, 3, 0, 2]


def test_wide_grid(family):
    """fit() with wide grid (many columns)."""
    train = [
        {
            "input": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "output": [[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.perm == [1, 0]


def test_rectangular_grid(family):
    """fit() with non-square rectangular grid."""
    train = [
        {
            "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            "output": [[10, 11, 12], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # X[0]→Y[1], X[1]→Y[3], X[2]→Y[2], X[3]→Y[0]
    assert family.params.perm == [1, 3, 2, 0]
