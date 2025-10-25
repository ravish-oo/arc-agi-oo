"""
Tests for SortRowsLex Family — Phase 2 work order P2-13.

Comprehensive test suite covering:
- Basic fit() tests (empty, single row, sorted/unsorted)
- Shape preservation tests
- Lexicographic order tests
- Sorting correctness tests
- FY exactness tests
- Multi-pair tests
- Stable sort tests
- apply() tests
- Determinism tests
- Purity tests
- Edge cases
- No parameters tests
"""

import pytest
from src.families.sort_rows import SortRowsLexFamily
from src.utils import dims, deep_eq


# =============================
# Fixtures
# =============================

@pytest.fixture
def family():
    """Fresh family instance for each test."""
    return SortRowsLexFamily()


# =============================
# Test Grids
# =============================

# Basic grids
g_empty = []
g_single_row = [[5, 6, 7]]
g_already_sorted = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
g_reverse_sorted = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
g_unsorted = [[4, 5, 6], [1, 2, 3], [7, 8, 9]]
g_identical_rows = [[2, 3], [2, 3], [2, 3]]
g_partial_duplicates = [[5, 6], [1, 2], [5, 6], [1, 2]]
g_lexicographic_order = [[2, 1], [1, 3], [1, 2]]

# Sorted equivalents
g_already_sorted_sorted = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
g_reverse_sorted_sorted = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
g_unsorted_sorted = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
g_identical_rows_sorted = [[2, 3], [2, 3], [2, 3]]
g_partial_duplicates_sorted = [[1, 2], [1, 2], [5, 6], [5, 6]]
g_lexicographic_order_sorted = [[1, 2], [1, 3], [2, 1]]


# =============================
# Basic fit() Tests
# =============================

def test_fit_empty_train_pairs(family):
    """fit([]) should return False."""
    result = family.fit([])
    assert result is False


def test_fit_empty_grids(family):
    """Empty grids should be accepted if both are empty."""
    train_pairs = [{"input": [], "output": []}]
    result = family.fit(train_pairs)
    assert result is True


def test_fit_single_row(family):
    """Single row is trivially sorted."""
    train_pairs = [{"input": [[5, 6, 7]], "output": [[5, 6, 7]]}]
    result = family.fit(train_pairs)
    assert result is True


def test_fit_already_sorted(family):
    """Already sorted X should match Y if Y == X."""
    train_pairs = [{"input": g_already_sorted, "output": g_already_sorted_sorted}]
    result = family.fit(train_pairs)
    assert result is True


def test_fit_reverse_order(family):
    """Reverse sorted X should match sorted Y."""
    train_pairs = [{"input": g_reverse_sorted, "output": g_reverse_sorted_sorted}]
    result = family.fit(train_pairs)
    assert result is True


def test_fit_unsorted(family):
    """Unsorted X should match sorted Y."""
    train_pairs = [{"input": g_unsorted, "output": g_unsorted_sorted}]
    result = family.fit(train_pairs)
    assert result is True


def test_fit_identical_rows(family):
    """All identical rows should remain in same order."""
    train_pairs = [{"input": g_identical_rows, "output": g_identical_rows_sorted}]
    result = family.fit(train_pairs)
    assert result is True


# =============================
# Shape Preservation Tests
# =============================

def test_reject_different_row_counts(family):
    """Different row counts should be rejected."""
    train_pairs = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4], [5, 6]]}
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_reject_different_column_counts(family):
    """Different column counts should be rejected."""
    train_pairs = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2, 5], [3, 4, 6]]}
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_accept_same_shape(family):
    """Same shape with correct sorting should be accepted."""
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 4]]}
    ]
    result = family.fit(train_pairs)
    assert result is True


# =============================
# Lexicographic Order Tests
# =============================

def test_lexicographic_comparison_element_0(family):
    """[1,x] < [2,y] for any x, y."""
    train_pairs = [
        {"input": [[2, 9], [1, 0]], "output": [[1, 0], [2, 9]]}
    ]
    result = family.fit(train_pairs)
    assert result is True


def test_lexicographic_comparison_element_1(family):
    """[1,2] < [1,3] (compare second element when first is equal)."""
    train_pairs = [
        {"input": [[1, 3], [1, 2]], "output": [[1, 2], [1, 3]]}
    ]
    result = family.fit(train_pairs)
    assert result is True


def test_lexicographic_comparison_complex(family):
    """Complex lexicographic ordering: [[1,2], [1,3], [2,1]]."""
    train_pairs = [
        {"input": g_lexicographic_order, "output": g_lexicographic_order_sorted}
    ]
    result = family.fit(train_pairs)
    assert result is True


def test_lexicographic_comparison_all_equal(family):
    """Equal rows should preserve original order (stable)."""
    train_pairs = [
        {"input": [[2, 3], [2, 3]], "output": [[2, 3], [2, 3]]}
    ]
    result = family.fit(train_pairs)
    assert result is True


# =============================
# Sorting Correctness Tests
# =============================

def test_sort_ascending(family):
    """Verify rows are in ascending order after sort."""
    X = [[5, 6], [1, 2], [3, 4]]
    Y_expected = [[1, 2], [3, 4], [5, 6]]
    train_pairs = [{"input": X, "output": Y_expected}]
    result = family.fit(train_pairs)
    assert result is True


def test_sort_descending_rejected(family):
    """Descending order should be rejected."""
    train_pairs = [
        {"input": [[1, 2], [3, 4], [5, 6]], "output": [[5, 6], [3, 4], [1, 2]]}
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_sort_partial_rejected(family):
    """Partially sorted output should be rejected."""
    train_pairs = [
        {"input": [[3, 4], [1, 2], [5, 6]], "output": [[1, 2], [5, 6], [3, 4]]}
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_idempotent_sort(family):
    """sorted(sorted(X)) == sorted(X)."""
    # First sort
    X = [[5, 6], [1, 2], [3, 4]]
    sorted_once = family.apply(X)
    # Second sort
    sorted_twice = family.apply(sorted_once)
    assert deep_eq(sorted_once, sorted_twice)


# =============================
# FY Exactness Tests
# =============================

def test_fy_single_element_differs(family):
    """Single element difference should be rejected."""
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 5]]}  # 5 instead of 4
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_fy_all_pairs_must_match(family):
    """All pairs must satisfy Y == sorted(X)."""
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 4]]},  # OK
        {"input": [[7, 8], [5, 6]], "output": [[5, 6], [7, 8]]},  # OK
    ]
    result = family.fit(train_pairs)
    assert result is True


def test_fy_bit_for_bit_equality(family):
    """deep_eq must pass exactly (bit-for-bit equality)."""
    train_pairs = [
        {"input": [[2, 3], [1, 2]], "output": [[1, 2], [2, 3]]}
    ]
    result = family.fit(train_pairs)
    assert result is True


# =============================
# Multi-Pair Tests
# =============================

def test_multi_pair_consistent(family):
    """All pairs with consistent sorting should be accepted."""
    train_pairs = [
        {"input": [[5, 6], [1, 2], [3, 4]], "output": [[1, 2], [3, 4], [5, 6]]},
        {"input": [[9, 0], [2, 3], [7, 8]], "output": [[2, 3], [7, 8], [9, 0]]},
    ]
    result = family.fit(train_pairs)
    assert result is True


def test_multi_pair_one_fails(family):
    """One failing pair should cause rejection."""
    train_pairs = [
        {"input": [[5, 6], [1, 2], [3, 4]], "output": [[1, 2], [3, 4], [5, 6]]},  # OK
        {"input": [[9, 0], [2, 3], [7, 8]], "output": [[9, 0], [2, 3], [7, 8]]},  # NOT sorted
    ]
    result = family.fit(train_pairs)
    assert result is False


def test_multi_pair_different_dimensions(family):
    """Consistent sorting across different grid sizes should work."""
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 4]]},  # 2x2
        {"input": [[7, 8, 9], [4, 5, 6], [1, 2, 3]], "output": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},  # 3x3
    ]
    result = family.fit(train_pairs)
    assert result is True


# =============================
# Stable Sort Tests
# =============================

def test_stable_sort_identical_rows(family):
    """Equal rows should preserve original order."""
    # Can't directly verify order preservation without inspecting indices,
    # but we can verify that sorting identical rows doesn't change result
    train_pairs = [
        {"input": [[2, 3], [2, 3], [2, 3]], "output": [[2, 3], [2, 3], [2, 3]]}
    ]
    result = family.fit(train_pairs)
    assert result is True


def test_stable_sort_partial_duplicates(family):
    """[[5,6],[1,2],[5,6],[1,2]] → [[1,2],[1,2],[5,6],[5,6]]."""
    train_pairs = [
        {"input": g_partial_duplicates, "output": g_partial_duplicates_sorted}
    ]
    result = family.fit(train_pairs)
    assert result is True


def test_stable_sort_determinism(family):
    """Sorting twice should yield identical outputs."""
    X = [[5, 6], [1, 2], [5, 6], [1, 2]]
    result1 = family.apply(X)
    result2 = family.apply(X)
    assert deep_eq(result1, result2)


# =============================
# apply() Tests
# =============================

def test_apply_empty_grid(family):
    """apply([]) should return []."""
    result = family.apply([])
    assert result == []


def test_apply_single_row(family):
    """Single row should remain unchanged."""
    X = [[5, 6, 7]]
    result = family.apply(X)
    assert deep_eq(result, [[5, 6, 7]])


def test_apply_already_sorted(family):
    """Already sorted input should return copy."""
    result = family.apply(g_already_sorted)
    assert deep_eq(result, g_already_sorted_sorted)
    # Verify it's a copy, not the same object
    assert result is not g_already_sorted


def test_apply_reverse_order(family):
    """Reverse sorted input should be reversed."""
    result = family.apply(g_reverse_sorted)
    assert deep_eq(result, g_reverse_sorted_sorted)


def test_apply_unsorted(family):
    """Unsorted input should be sorted."""
    result = family.apply(g_unsorted)
    assert deep_eq(result, g_unsorted_sorted)


def test_apply_preserves_shape(family):
    """dims(apply(X)) should equal dims(X)."""
    X = [[7, 8], [3, 4], [1, 2]]
    result = family.apply(X)
    assert dims(result) == dims(X)


def test_apply_no_fit_required(family):
    """apply() should work without calling fit() first."""
    # Don't call fit()
    X = [[5, 6], [1, 2], [3, 4]]
    result = family.apply(X)
    expected = [[1, 2], [3, 4], [5, 6]]
    assert deep_eq(result, expected)


# =============================
# Determinism Tests
# =============================

def test_deterministic_sort(family):
    """apply(X) twice should yield identical outputs."""
    X = [[7, 8], [3, 4], [1, 2]]
    result1 = family.apply(X)
    result2 = family.apply(X)
    assert deep_eq(result1, result2)


def test_deterministic_fit(family):
    """fit() twice on same data should yield identical result."""
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 4]]}
    ]
    result1 = family.fit(train_pairs)
    result2 = family.fit(train_pairs)
    assert result1 == result2 == True


# =============================
# Purity Tests
# =============================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() should not mutate input train_pairs."""
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 4]]}
    ]
    # Deep copy for comparison
    import copy
    train_pairs_copy = copy.deepcopy(train_pairs)

    family.fit(train_pairs)

    assert deep_eq(train_pairs, train_pairs_copy)


def test_apply_does_not_mutate_input(family):
    """apply() should not mutate input X."""
    X = [[3, 4], [1, 2]]
    import copy
    X_copy = copy.deepcopy(X)

    family.apply(X)

    assert deep_eq(X, X_copy)


def test_apply_no_row_aliasing(family):
    """Output should have fresh allocation, no aliasing with input."""
    X = [[3, 4], [1, 2]]
    result = family.apply(X)

    # Verify no aliasing
    assert result is not X
    for i in range(len(result)):
        # Check that no output row is the same object as any input row
        for j in range(len(X)):
            assert result[i] is not X[j]


# =============================
# Edge Cases
# =============================

def test_two_rows_swap(family):
    """Two rows should be swapped if needed."""
    X = [[5, 6], [1, 2]]
    result = family.apply(X)
    expected = [[1, 2], [5, 6]]
    assert deep_eq(result, expected)


def test_large_grid_many_rows(family):
    """Many rows should be sorted correctly."""
    # Create 100 rows in descending order
    X = [[100 - i, 200 - i] for i in range(100)]
    result = family.apply(X)

    # Verify ascending order
    for i in range(len(result) - 1):
        assert tuple(result[i]) <= tuple(result[i + 1])


def test_wide_grid(family):
    """Few rows, many columns should work."""
    X = [[9, 8, 7, 6, 5], [4, 3, 2, 1, 0], [5, 5, 5, 5, 5]]
    result = family.apply(X)
    expected = [[4, 3, 2, 1, 0], [5, 5, 5, 5, 5], [9, 8, 7, 6, 5]]
    assert deep_eq(result, expected)


def test_identical_values_all_rows(family):
    """All rows with same values should remain unchanged."""
    X = [[7, 7], [7, 7], [7, 7]]
    result = family.apply(X)
    assert deep_eq(result, [[7, 7], [7, 7], [7, 7]])


def test_single_column(family):
    """Single column, many rows should work."""
    X = [[5], [1], [3], [2]]
    result = family.apply(X)
    expected = [[1], [2], [3], [5]]
    assert deep_eq(result, expected)


def test_large_values(family):
    """Rows with large integers should work."""
    X = [[9999, 8888], [1111, 2222], [5555, 6666]]
    result = family.apply(X)
    expected = [[1111, 2222], [5555, 6666], [9999, 8888]]
    assert deep_eq(result, expected)


# =============================
# Comparison with RowPermutation
# =============================

def test_vs_row_permutation(family):
    """SortRowsLex is specific case of RowPermutation (sorted permutation)."""
    # This test verifies that SortRowsLex produces a valid row permutation
    X = [[3, 4], [1, 2], [5, 6]]
    result = family.apply(X)

    # Verify result is a permutation of input rows
    rows_X = [tuple(row) for row in X]
    rows_result = [tuple(row) for row in result]
    assert sorted(rows_X) == sorted(rows_result)


def test_deterministic_vs_learned(family):
    """SortRowsLex is fixed (deterministic); RowPermutation is learned."""
    # SortRowsLex always produces same output for same input
    X = [[3, 4], [1, 2]]
    result1 = family.apply(X)
    result2 = family.apply(X)
    # Always sorted ascending
    assert deep_eq(result1, [[1, 2], [3, 4]])
    assert deep_eq(result2, [[1, 2], [3, 4]])


# =============================
# Integration with utils
# =============================

def test_uses_dims(family):
    """Verify dims() is used correctly."""
    # dims() should work for any rectangular grid
    X = [[1, 2, 3], [4, 5, 6]]
    result = family.apply(X)
    h, w = dims(result)
    assert (h, w) == (2, 3)


def test_uses_deep_eq(family):
    """Verify deep_eq() is used in fit()."""
    # deep_eq() checks bit-for-bit equality
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 4]]}
    ]
    result = family.fit(train_pairs)
    assert result is True


# =============================
# No Parameters Tests
# =============================

def test_no_params_to_set(family):
    """params object exists but has no fields."""
    assert hasattr(family, 'params')
    # Check that params has no meaningful attributes (it's an empty object)
    params_dict = {k: v for k, v in family.params.__dict__.items() if not k.startswith('_')}
    assert params_dict == {}


def test_params_empty_after_fit(family):
    """fit() doesn't populate params."""
    train_pairs = [
        {"input": [[3, 4], [1, 2]], "output": [[1, 2], [3, 4]]}
    ]
    family.fit(train_pairs)

    # Check params still empty
    params_dict = {k: v for k, v in family.params.__dict__.items() if not k.startswith('_')}
    assert params_dict == {}


def test_params_not_used_in_apply(family):
    """apply() doesn't read params (deterministic transformation)."""
    # apply() works without fit() (no params needed)
    X = [[3, 4], [1, 2]]
    result = family.apply(X)
    expected = [[1, 2], [3, 4]]
    assert deep_eq(result, expected)
