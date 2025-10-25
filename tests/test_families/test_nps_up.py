"""
Test suite for NPSUpFamily (P2-07).

Tests cover:
- Basic fit/apply workflow
- Factor learning correctness (integer validation)
- Φ.3 input-only constraint (boundaries from X only)
- Unified factors requirement (ONE set for ALL pairs)
- Band count matching requirement
- FY exactness (bit-for-bit equality)
- Edge cases (uniform grids, various patterns, large factors)
- Purity (no mutations)
- Determinism
"""

import pytest
from src.families.nps_up import NPSUpFamily
from src.utils import deep_eq, copy_grid


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def family():
    """Fresh NPSUpFamily instance for each test."""
    return NPSUpFamily()


# ============================================================================
# Basic fit/apply
# ============================================================================

def test_fit_simple_2x_upsampling(family):
    """fit() with simple 2× upsampling in both dimensions."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [2, 2]
    assert family.params.col_factors == [2, 2]


def test_apply_after_fit(family):
    """apply() after successful fit() should return correct upsampled grid."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True

    # Apply to same input
    X = [[1, 2], [3, 4]]
    Y = family.apply(X)
    assert Y == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]


def test_apply_before_fit_raises(family):
    """apply() before fit() should raise RuntimeError."""
    X = [[1, 2], [3, 4]]

    with pytest.raises(RuntimeError) as exc_info:
        family.apply(X)

    assert "params.row_factors is None" in str(exc_info.value)


# ============================================================================
# Factor learning correctness
# ============================================================================

def test_learn_uniform_factors(family):
    """Learn uniform factors (same factor for all bands)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],  # 2×2, each cell is own band
            "output": [[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2],
                      [3, 3, 3, 4, 4, 4], [3, 3, 3, 4, 4, 4], [3, 3, 3, 4, 4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [3, 3]
    assert family.params.col_factors == [3, 3]


def test_learn_non_uniform_factors(family):
    """Learn non-uniform factors (different factors for different bands)."""
    train = [
        {
            "input": [[1, 1], [2, 2]],  # row change → 2 row bands, 1 col band
            "output": [[1, 1], [1, 1], [2, 2], [2, 2], [2, 2]]  # factors [2, 3], [1]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [2, 3]
    assert family.params.col_factors == [1]


def test_reject_non_integer_factors(family):
    """fit() must reject when Y_band_size is not integer multiple of X_band_size."""
    train = [
        {
            "input": [[1, 1], [2, 2]],  # 2 row bands (rows change), 1 col band (cols uniform)
            "output": [[1, 1], [2, 2], [2, 2]]  # 2 row bands with sizes 1 and 2 - factor 1 and 2 (non-uniform but valid)
        }
    ]
    # Actually this has non-integer col factors: X has 2 cols → 1 band, Y has 2 cols → 1 band, factor=1. Valid!
    # Let me create a truly invalid case: X band has 2 cells, Y band has 3 cells (3/2 not integer)
    train = [
        {
            "input": [[1, 1, 1], [2, 2, 2]],  # 2 row bands, 1 col band (3 uniform cols)
            "output": [[1, 1], [2, 2]]  # This is downsampling, not upsampling - invalid for NPSUp
        }
    ]
    result = family.fit(train)
    assert result is False  # Non-integer or invalid factors


def test_reject_band_count_mismatch(family):
    """fit() must reject when X and Y have different number of bands."""
    train = [
        {
            "input": [[1, 2], [3, 4]],  # 2 row bands, 2 col bands
            "output": [[1, 1, 1], [1, 1, 1]]  # 1 row band, 1 col band (mismatch)
        }
    ]
    result = family.fit(train)
    assert result is False  # Band count mismatch


def test_identity_upsampling(family):
    """Factor of 1 for all bands (identity upsampling)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4]]  # Same size
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [1, 1]
    assert family.params.col_factors == [1, 1]


# ============================================================================
# Φ.3 Input-Only Constraint
# ============================================================================

def test_phi3_boundaries_from_x_only(family):
    """Boundaries computed from X only, never from Y (Φ.3 constraint)."""
    # Same X → same boundaries → same output shape
    train = [
        {
            "input": [[1, 1], [2, 2]],  # row change → 2 row bands, 1 col band
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]  # 2× upsampling
        }
    ]
    result = family.fit(train)
    assert result is True

    # Apply to same X → should get same shape
    X = [[1, 1], [2, 2]]
    Y1 = family.apply(X)
    Y2 = family.apply(X)
    assert Y1 == Y2

    # Apply to different X with SAME boundary structure (2 row bands, 1 col band)
    X2 = [[3, 3], [4, 4]]  # row change → 2 row bands, 1 col band (same structure!)
    Y3 = family.apply(X2)
    # Should have same shape since band structure matches
    assert len(Y3) == 4  # 2 row bands × 2 factor each
    assert len(Y3[0]) == 2  # 1 col band × 2 factor
    # Values should be from X2, not training data (Φ.3: boundaries from X2 only)
    assert Y3[0][0] == 3
    assert Y3[2][0] == 4


def test_phi3_boundaries_recomputed_each_apply(family):
    """apply() recomputes boundaries from X (no cached state from fit)."""
    train = [
        {
            "input": [[1, 1], [2, 2]],  # row change → 2 row bands, 1 col band
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        }
    ]
    family.fit(train)

    # Apply to X with SAME boundary structure (2 row bands, 1 col band)
    # but different boundary positions due to different values
    X_new = [[5, 5], [5, 5], [6, 6]]  # 3 rows, boundaries recomputed from X_new
    # boundaries_by_any_change finds change at row 1 → boundaries = [1]
    # So 2 row bands: [0,2) with 2 rows and [2,3) with 1 row
    Y_new = family.apply(X_new)

    # Band 0 (2 rows) × factor 2 = 4 output rows
    # Band 1 (1 row) × factor 2 = 2 output rows
    # Total: 6 output rows
    assert len(Y_new) == 6
    assert len(Y_new[0]) == 2  # 1 col band × 2 factor
    # Verify values come from X_new (Φ.3: boundaries from input only)
    assert Y_new[0][0] == 5
    assert Y_new[4][0] == 6


# ============================================================================
# Unified factors requirement
# ============================================================================

def test_unified_factors_multiple_pairs(family):
    """fit() must use same factors for all training pairs."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        },
        {
            "input": [[3, 3], [4, 4]],
            "output": [[3, 3], [3, 3], [4, 4], [4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True


def test_unified_factors_different_boundaries_per_pair(family):
    """Pairs with different boundary structures must be rejected."""
    # Pair 1: row change → 2 row bands, 1 col band
    # Pair 2: col change → 1 row band, 2 col bands
    # Different band structures → fit must return False
    train = [
        {
            "input": [[1, 1], [2, 2]],  # row change → 2 row bands, 1 col band
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        },
        {
            "input": [[3, 4], [3, 4]],  # col change → 1 row band, 2 col bands
            "output": [[3, 3, 4, 4], [3, 3, 4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is False  # Different band structures cannot work with unified factors


def test_fit_rejects_incompatible_factors(family):
    """fit() must return False if learned factors don't work for all pairs."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]  # factors [2], [1]
        },
        {
            "input": [[3, 3], [4, 4]],
            "output": [[3, 3], [3, 3], [3, 3], [4, 4], [4, 4], [4, 4]]  # factors [3], [1]
        }
    ]
    result = family.fit(train)
    assert result is False  # Incompatible factors


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_train_pairs_returns_false(family):
    """fit([]) must return False."""
    result = family.fit([])
    assert result is False


def test_empty_grids_returns_false(family):
    """fit() with empty grids must return False."""
    train = [
        {"input": [], "output": []}
    ]
    result = family.fit(train)
    assert result is False


def test_apply_empty_grid(family):
    """apply([]) with valid params should return []."""
    # First fit with valid data
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        }
    ]
    family.fit(train)

    # Then apply to empty grid
    result = family.apply([])
    assert result == []


def test_single_cell_grid(family):
    """Single cell grid with 2× upsampling."""
    train = [
        {
            "input": [[5]],
            "output": [[5, 5], [5, 5]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [2]
    assert family.params.col_factors == [2]

    Y = family.apply([[5]])
    assert Y == [[5, 5], [5, 5]]


def test_uniform_grid_upsampling(family):
    """Uniform grid (no boundaries) forms single band."""
    train = [
        {
            "input": [[7, 7], [7, 7]],  # uniform → 1 row band, 1 col band
            "output": [[7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7]]  # 2× in each
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [2]
    assert family.params.col_factors == [2]


def test_row_bands_only(family):
    """Grid with only row changes → N row bands × 1 col band."""
    train = [
        {
            "input": [[1, 1], [2, 2], [3, 3]],  # 3 row bands, 1 col band
            "output": [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]]  # 2× rows, 1× cols
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [2, 2, 2]
    assert family.params.col_factors == [1]


def test_col_bands_only(family):
    """Grid with only col changes → 1 row band × M col bands."""
    train = [
        {
            "input": [[1, 2, 3], [1, 2, 3]],  # 1 row band, 3 col bands
            "output": [[1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3]]  # 1× rows, 2× cols
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [1]
    assert family.params.col_factors == [2, 2, 2]


def test_large_upsampling_factor(family):
    """Large upsampling factors (10×)."""
    train = [
        {
            "input": [[1]],
            "output": [[1] * 10] * 10  # 10×10 output from 1×1 input
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [10]
    assert family.params.col_factors == [10]


def test_mixed_boundaries(family):
    """Grid with both row and col changes → N×M bands."""
    train = [
        {
            "input": [[1, 2], [3, 4]],  # 2 row bands, 2 col bands
            "output": [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [2, 2]
    assert family.params.col_factors == [2, 2]


# ============================================================================
# FY exactness
# ============================================================================

def test_fy_exact_equality_required(family):
    """After successful fit, applying to train inputs must reproduce outputs exactly."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        },
        {
            "input": [[3, 3], [4, 4]],
            "output": [[3, 3], [3, 3], [4, 4], [4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True

    # Verify FY exactness
    for pair in train:
        X = pair["input"]
        Y = pair["output"]
        Y_predicted = family.apply(X)
        assert deep_eq(Y_predicted, Y)


# ============================================================================
# Determinism
# ============================================================================

def test_deterministic_repeated_fit(family):
    """Running fit() twice on same data yields identical params."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        }
    ]

    result1 = family.fit(train)
    row_factors1 = family.params.row_factors
    col_factors1 = family.params.col_factors

    # Create fresh family and fit again
    family2 = NPSUpFamily()
    result2 = family2.fit(train)
    row_factors2 = family2.params.row_factors
    col_factors2 = family2.params.col_factors

    assert result1 == result2 == True
    assert row_factors1 == row_factors2
    assert col_factors1 == col_factors2


def test_deterministic_apply(family):
    """Applying same input yields identical output."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        }
    ]
    family.fit(train)

    X = [[1, 1], [2, 2]]
    Y1 = family.apply(X)
    Y2 = family.apply(X)

    assert deep_eq(Y1, Y2)


# ============================================================================
# Purity
# ============================================================================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() must not mutate train_pairs."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        }
    ]
    train_copy = copy_grid(train)

    family.fit(train)

    assert deep_eq(train, train_copy)


def test_apply_does_not_mutate_input(family):
    """apply() must not mutate input X."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        }
    ]
    family.fit(train)

    X = [[1, 1], [2, 2]]
    X_copy = copy_grid(X)

    family.apply(X)

    assert deep_eq(X, X_copy)


def test_apply_no_row_aliasing(family):
    """apply() output must have no row aliasing."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        }
    ]
    family.fit(train)

    X = [[1, 1], [2, 2]]
    Y = family.apply(X)

    # Modify output
    if Y and Y[0]:
        original_value = Y[0][0]
        Y[0][0] = 999

        # Apply again
        Y2 = family.apply(X)

        # Verify Y2 not affected by mutation of Y
        assert Y2[0][0] != 999
        assert Y2[0][0] == original_value


# ============================================================================
# Integration
# ============================================================================

def test_complex_workflow_multiple_pairs_different_boundaries(family):
    """Complex workflow with multiple pairs - all must have SAME boundary structure."""
    # All three pairs have same boundary structure: 2 row bands, 1 col band
    train = [
        {
            "input": [[1, 1], [2, 2]],  # row change → 2 row bands, 1 col band
            "output": [[1, 1], [1, 1], [2, 2], [2, 2]]
        },
        {
            "input": [[3, 3], [4, 4]],  # row change → 2 row bands, 1 col band
            "output": [[3, 3], [3, 3], [4, 4], [4, 4]]
        },
        {
            "input": [[5, 5], [6, 6]],  # row change → 2 row bands, 1 col band
            "output": [[5, 5], [5, 5], [6, 6], [6, 6]]
        }
    ]
    result = family.fit(train)
    assert result is True

    # Verify each pair
    for pair in train:
        X = pair["input"]
        Y = pair["output"]
        Y_pred = family.apply(X)
        assert deep_eq(Y_pred, Y)


def test_asymmetric_factors(family):
    """Different upsampling factors for rows vs cols."""
    train = [
        {
            "input": [[1, 2], [3, 4]],  # 2 row bands, 2 col bands
            "output": [[1, 1, 2, 2, 2], [1, 1, 2, 2, 2], [1, 1, 2, 2, 2],
                      [3, 3, 4, 4, 4], [3, 3, 4, 4, 4], [3, 3, 4, 4, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.row_factors == [3, 3]  # rows upsample 3×
    assert family.params.col_factors == [2, 3]  # cols upsample differently: first band 2×, second 3×
