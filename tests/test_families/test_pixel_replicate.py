"""
Test suite for PixelReplicateFamily (P2-04).

Tests cover:
- Basic fit/apply workflow
- Integer scaling requirement (critical: no fractional ratios)
- Unified (kH, kW) requirement (critical: ONE pair for ALL pairs)
- FY exactness (bit-for-bit equality)
- Edge cases (empty grids, identity scaling)
- Block replication correctness
- Purity (no mutations)
- Determinism
"""

import pytest
from src.families.pixel_replicate import PixelReplicateFamily
from src.utils import deep_eq, copy_grid


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def family():
    """Fresh PixelReplicateFamily instance for each test."""
    return PixelReplicateFamily()


# ============================================================================
# Basic fit/apply
# ============================================================================

def test_fit_identity_scaling(family):
    """fit() with identity scaling (kH=kW=1) should pass-through."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 1
    assert family.params.kW == 1


def test_fit_uniform_2x_scaling(family):
    """fit() with uniform 2x upsampling should learn kH=kW=2."""
    train = [
        {"input": [[1, 2], [3, 4]],
         "output": [[1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 2
    assert family.params.kW == 2


def test_fit_non_uniform_scaling(family):
    """fit() with non-uniform scaling (kH≠kW) should learn correct factors."""
    # kH=2, kW=3
    train = [
        {"input": [[1, 2]],
         "output": [[1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 2
    assert family.params.kW == 3


def test_apply_after_fit(family):
    """apply() after successful fit() should return correct upsampled grid."""
    train = [
        {"input": [[5]], "output": [[5, 5, 5],
                                     [5, 5, 5],
                                     [5, 5, 5]]}
    ]
    family.fit(train)

    # Apply to new input
    X = [[1, 2]]
    result = family.apply(X)
    expected = [[1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2]]
    assert deep_eq(result, expected)


def test_apply_before_fit_raises(family):
    """apply() before fit() should raise RuntimeError."""
    X = [[1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="params.kH is None"):
        family.apply(X)


# ============================================================================
# Integer scaling requirement (CRITICAL)
# ============================================================================

def test_fit_rejects_non_integer_ratio_rows(family):
    """fit() must reject when Y_rows / X_rows is not an integer."""
    # 2×2 → 3×2: kH would be 1.5 (non-integer)
    train = [
        {"input": [[1, 2], [3, 4]],
         "output": [[1, 2], [3, 4], [5, 6]]}
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.kH is None
    assert family.params.kW is None


def test_fit_rejects_non_integer_ratio_cols(family):
    """fit() must reject when Y_cols / X_cols is not an integer."""
    # 1×2 → 1×3: kW would be 1.5 (non-integer)
    train = [
        {"input": [[1, 2]],
         "output": [[1, 2, 3]]}
    ]
    result = family.fit(train)
    assert result is False


def test_fit_rejects_downsampling(family):
    """fit() must reject downsampling (Y smaller than X)."""
    # 2×2 → 1×1: kH=kW=0.5 (fractional, also downsampling)
    train = [
        {"input": [[1, 2], [3, 4]],
         "output": [[1]]}
    ]
    result = family.fit(train)
    assert result is False


# ============================================================================
# Unified (kH, kW) requirement (CRITICAL)
# ============================================================================

def test_unified_factors_multiple_pairs(family):
    """fit() with multiple pairs using same (kH, kW) should succeed."""
    train = [
        {"input": [[1, 2]], "output": [[1, 1, 2, 2], [1, 1, 2, 2]]},  # kH=2, kW=2
        {"input": [[5, 6]], "output": [[5, 5, 6, 6], [5, 5, 6, 6]]}   # kH=2, kW=2
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 2
    assert family.params.kW == 2


def test_unified_factors_inconsistent_rejects(family):
    """fit() must reject when pairs have inconsistent scaling factors."""
    train = [
        {"input": [[1]], "output": [[1, 1], [1, 1]]},  # kH=2, kW=2
        {"input": [[2]], "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]}  # kH=3, kW=3
    ]
    result = family.fit(train)
    assert result is False


def test_unified_factors_all_pairs_checked():
    """fit() must verify ALL pairs, not just first."""
    train = [
        {"input": [[1]], "output": [[1, 1], [1, 1]]},  # kH=2, kW=2
        {"input": [[2]], "output": [[2, 2], [2, 2]]},  # kH=2, kW=2
        {"input": [[3]], "output": [[3, 3, 3], [3, 3, 3]]}  # kH=2, kW=3 - conflict!
    ]
    family = PixelReplicateFamily()
    result = family.fit(train)
    assert result is False


# ============================================================================
# FY exactness
# ============================================================================

def test_fit_rejects_fy_violation(family):
    """fit() must reject when dimensions match but values don't follow pixel replication."""
    train = [
        {
            "input": [[1, 2]],
            # Dimensions correct: 1*2=2 rows, 2*2=4 cols (kH=2, kW=2)
            # Values wrong: should be [[1,1,2,2], [1,1,2,2]]
            "output": [[1, 1, 2, 2],
                       [1, 1, 3, 3]]  # Last two values incorrect
        }
    ]
    result = family.fit(train)
    assert result is False  # Must reject due to FY violation


def test_fy_exact_replication_required():
    """After successful fit, applying to train inputs must reproduce outputs exactly."""
    train = [
        {"input": [[1, 2]], "output": [[1, 1, 2, 2], [1, 1, 2, 2]]},
        {"input": [[3, 4]], "output": [[3, 3, 4, 4], [3, 3, 4, 4]]}
    ]
    family = PixelReplicateFamily()
    result = family.fit(train)
    assert result is True

    # Verify FY exactness
    for pair in train:
        X = pair["input"]
        Y = pair["output"]
        Y_predicted = family.apply(X)
        assert deep_eq(Y_predicted, Y)


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_train_pairs_returns_false(family):
    """fit([]) with empty train_pairs should return False."""
    result = family.fit([])
    assert result is False
    assert family.params.kH is None
    assert family.params.kW is None


def test_empty_grids_returns_false(family):
    """fit() with empty grids should return False."""
    train = [
        {"input": [], "output": []}
    ]
    result = family.fit(train)
    assert result is False


def test_apply_empty_grid(family):
    """apply([]) with valid (kH, kW) should return []."""
    train = [
        {"input": [[1]], "output": [[1, 1], [1, 1]]}  # kH=2, kW=2
    ]
    family.fit(train)
    result = family.apply([])
    assert result == []


def test_single_pixel_grid(family):
    """fit() with 1x1 grids should work correctly."""
    train = [
        {"input": [[5]], "output": [[5, 5, 5],
                                     [5, 5, 5],
                                     [5, 5, 5]]}  # kH=3, kW=3
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 3
    assert family.params.kW == 3


def test_vertical_scaling_only(family):
    """fit() with vertical-only scaling (kH>1, kW=1) should work."""
    train = [
        {"input": [[1, 2]], "output": [[1, 2], [1, 2], [1, 2]]}  # kH=3, kW=1
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 3
    assert family.params.kW == 1


def test_horizontal_scaling_only(family):
    """fit() with horizontal-only scaling (kH=1, kW>1) should work."""
    train = [
        {"input": [[1], [2]], "output": [[1, 1, 1], [2, 2, 2]]}  # kH=1, kW=3
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 1
    assert family.params.kW == 3


# ============================================================================
# Block replication correctness
# ============================================================================

def test_block_replication_2x2(family):
    """Verify each pixel replicates into correct 2×2 block."""
    train = [
        {"input": [[1, 2], [3, 4]],
         "output": [[1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4]]}
    ]
    family.fit(train)

    X = [[5, 6], [7, 8]]
    result = family.apply(X)
    expected = [[5, 5, 6, 6],
                [5, 5, 6, 6],
                [7, 7, 8, 8],
                [7, 7, 8, 8]]
    assert deep_eq(result, expected)


def test_block_replication_non_uniform(family):
    """Verify non-uniform replication (kH=2, kW=3) fills correct blocks."""
    train = [
        {"input": [[1, 2]],
         "output": [[1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2]]}
    ]
    family.fit(train)

    X = [[9, 0]]
    result = family.apply(X)
    expected = [[9, 9, 9, 0, 0, 0],
                [9, 9, 9, 0, 0, 0]]
    assert deep_eq(result, expected)


def test_block_boundaries_no_overlap(family):
    """Verify block boundaries are correct (no off-by-one errors)."""
    train = [
        {"input": [[1]], "output": [[1, 1], [1, 1]]}  # kH=2, kW=2
    ]
    family.fit(train)

    # 2×2 input with kH=kW=2 → 4×4 output
    X = [[1, 2], [3, 4]]
    result = family.apply(X)
    expected = [[1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4]]
    assert deep_eq(result, expected)

    # Verify dimensions
    assert len(result) == 4
    assert len(result[0]) == 4


# ============================================================================
# Shape safety
# ============================================================================

def test_output_shape_correctness(family):
    """Verify output dims = (R*kH, C*kW) for all apply() calls."""
    train = [
        {"input": [[1]], "output": [[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]]}  # kH=3, kW=3
    ]
    family.fit(train)

    test_cases = [
        ([[1]], (3, 3)),
        ([[1, 2]], (3, 6)),
        ([[1], [2]], (6, 3)),
        ([[1, 2], [3, 4]], (6, 6)),
    ]

    for X, expected_shape in test_cases:
        result = family.apply(X)
        assert len(result) == expected_shape[0]
        assert len(result[0]) == expected_shape[1]


def test_large_scaling_factors(family):
    """Verify large scaling factors work correctly."""
    train = [
        {"input": [[1]], "output": [[1] * 10] * 10}  # kH=10, kW=10
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.kH == 10
    assert family.params.kW == 10

    # Verify apply
    X = [[5]]
    output = family.apply(X)
    assert len(output) == 10
    assert len(output[0]) == 10
    assert all(all(val == 5 for val in row) for row in output)


# ============================================================================
# Purity
# ============================================================================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() must not mutate train_pairs input."""
    train = [
        {"input": [[1, 2]], "output": [[1, 1, 2, 2], [1, 1, 2, 2]]}
    ]
    train_copy = [
        {"input": copy_grid(train[0]["input"]),
         "output": copy_grid(train[0]["output"])}
    ]

    family.fit(train)

    # Verify train_pairs unchanged
    assert deep_eq(train[0]["input"], train_copy[0]["input"])
    assert deep_eq(train[0]["output"], train_copy[0]["output"])


def test_apply_does_not_mutate_input(family):
    """apply() must not mutate input grid X."""
    train = [
        {"input": [[1]], "output": [[1, 1], [1, 1]]}
    ]
    family.fit(train)

    X = [[5, 6]]
    X_copy = copy_grid(X)

    result = family.apply(X)

    # Verify X unchanged
    assert deep_eq(X, X_copy)


def test_apply_no_row_aliasing(family):
    """apply() output must not share row references with input."""
    train = [
        {"input": [[1]], "output": [[1, 1], [1, 1]]}
    ]
    family.fit(train)

    X = [[7]]
    result = family.apply(X)

    # Mutate result
    result[0][0] = 999

    # Verify X unchanged (no aliasing)
    assert X[0][0] == 7


# ============================================================================
# Determinism
# ============================================================================

def test_deterministic_repeated_fit(family):
    """Running fit() twice on same train_pairs yields identical params."""
    train = [
        {"input": [[1, 2]], "output": [[1, 1, 2, 2], [1, 1, 2, 2]]}
    ]

    result1 = family.fit(train)
    kH1 = family.params.kH
    kW1 = family.params.kW

    # Reset and re-fit
    family.params.kH = None
    family.params.kW = None
    result2 = family.fit(train)
    kH2 = family.params.kH
    kW2 = family.params.kW

    assert result1 is True
    assert result2 is True
    assert kH1 == kH2
    assert kW1 == kW2


def test_deterministic_apply(family):
    """apply() with same input yields identical output."""
    train = [
        {"input": [[1]], "output": [[1, 1], [1, 1]]}
    ]
    family.fit(train)

    X = [[5, 6]]
    result1 = family.apply(X)
    result2 = family.apply(X)

    assert deep_eq(result1, result2)


# ============================================================================
# Integration tests
# ============================================================================

def test_complex_workflow_multiple_pairs():
    """Full workflow with multiple pairs and verification."""
    train = [
        {"input": [[1, 2], [3, 4]],
         "output": [[1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4]]},  # kH=3, kW=3
        {"input": [[5, 6], [7, 8]],
         "output": [[5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8]]}
    ]
    family = PixelReplicateFamily()

    result = family.fit(train)
    assert result is True
    assert family.params.kH == 3
    assert family.params.kW == 3

    # Apply on new input
    X = [[9, 0]]
    result = family.apply(X)
    expected = [[9, 9, 9, 0, 0, 0],
                [9, 9, 9, 0, 0, 0],
                [9, 9, 9, 0, 0, 0]]
    assert deep_eq(result, expected)
