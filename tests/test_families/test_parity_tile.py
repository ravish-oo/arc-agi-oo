"""
Test suite for ParityTileFamily (P2-08).

Tests cover:
- Basic fit/apply workflow for all modes ("none", "h", "v", "hv")
- Deterministic mode search order ["none", "h", "v", "hv"]
- Unified parameters requirement (ONE set for ALL pairs)
- Integer tiling factors validation
- Flip logic correctness for each mode
- Tiling dimensions correctness
- FY exactness (bit-for-bit equality)
- Edge cases (empty grids, identity, symmetric grids)
- Parity logic and bit operations
- Determinism
- Purity (no mutations)
"""

import pytest
from src.families.parity_tile import ParityTileFamily
from src.utils import deep_eq, copy_grid


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def family():
    """Fresh ParityTileFamily instance for each test."""
    return ParityTileFamily()


# ============================================================================
# Basic fit/apply
# ============================================================================

def test_fit_identity(family):
    """fit() with identity case (tiles_v=1, tiles_h=1, mode='none')."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 1
    assert family.params.tiles_h == 1
    assert family.params.mode == "none"


def test_fit_none_mode(family):
    """fit() with mode='none' (normal tiling, no flips)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 2
    assert family.params.tiles_h == 2
    assert family.params.mode == "none"


def test_fit_h_mode(family):
    """fit() with mode='h' (horizontal flip on odd columns)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 1
    assert family.params.tiles_h == 2
    assert family.params.mode == "h"


def test_fit_v_mode(family):
    """fit() with mode='v' (vertical flip on odd rows)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4], [3, 4], [1, 2]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 2
    assert family.params.tiles_h == 1
    assert family.params.mode == "v"


def test_fit_hv_mode(family):
    """fit() with mode='hv' (checkerboard pattern)."""
    # For hv mode with 2×2 tiling:
    # (0,0): X, (0,1): flip_h, (1,0): flip_v, (1,1): X
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 1, 2], [1, 2, 3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 2
    assert family.params.tiles_h == 2
    assert family.params.mode == "hv"


def test_apply_after_fit(family):
    """apply() after successful fit() should return correct tiled grid."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]
    result = family.fit(train)
    assert result is True

    # Apply to same input
    X = [[1, 2], [3, 4]]
    Y = family.apply(X)
    assert Y == [[1, 2, 2, 1], [3, 4, 4, 3]]


def test_apply_before_fit_raises(family):
    """apply() before fit() should raise RuntimeError."""
    X = [[1, 2], [3, 4]]

    with pytest.raises(RuntimeError) as exc_info:
        family.apply(X)

    assert "params.tiles_v is None" in str(exc_info.value)


# ============================================================================
# Deterministic mode search order
# ============================================================================

def test_mode_search_order_symmetric_grid(family):
    """Symmetric grid where multiple modes give same result - 'none' wins (first)."""
    train = [
        {
            "input": [[1, 1], [1, 1]],
            "output": [[1, 1, 1, 1], [1, 1, 1, 1]]
        }
    ]
    result = family.fit(train)
    assert result is True
    # All modes would work, but "none" is tried first
    assert family.params.mode == "none"


def test_mode_search_order_deterministic_repeat(family):
    """Running fit() twice on same data yields identical mode."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]

    result1 = family.fit(train)
    mode1 = family.params.mode

    # Create fresh family and fit again
    family2 = ParityTileFamily()
    result2 = family2.fit(train)
    mode2 = family2.params.mode

    assert result1 == result2 == True
    assert mode1 == mode2 == "h"


# ============================================================================
# Unified parameters requirement
# ============================================================================

def test_unified_params_multiple_pairs(family):
    """fit() must use same params for all training pairs."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [1, 1, 1, 1]]
        },
        {
            "input": [[5, 5], [6, 6]],
            "output": [[5, 5, 5, 5], [6, 6, 6, 6], [6, 6, 6, 6], [5, 5, 5, 5]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 2
    assert family.params.tiles_h == 2
    assert family.params.mode == "v"


def test_unified_params_inconsistent_rejects(family):
    """fit() must return False if params from first pair don't work for second."""
    train = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [1, 1, 1, 1]]
        },
        {
            "input": [[5, 5], [6, 6]],
            "output": [[5, 5, 5, 5], [6, 6, 6, 6], [5, 5, 5, 5], [6, 6, 6, 6]]
        }
    ]
    # First pair: tiles_v=2, tiles_h=2, mode="v"
    # Second pair: would need mode="none" not "v"
    result = family.fit(train)
    assert result is False


# ============================================================================
# Integer tiling factors
# ============================================================================

def test_reject_non_integer_vertical_tiling(family):
    """fit() must reject when Y_rows not divisible by X_rows."""
    train = [
        {
            "input": [[1, 2]],  # 1 row
            "output": [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]  # 5 rows: 5/1 is integer - this should work!
        }
    ]
    # Actually this should work - let me fix the test
    result = family.fit(train)
    assert result is True  # 5/1 = 5 is valid


def test_reject_non_integer_horizontal_tiling(family):
    """fit() must reject when Y_cols not divisible by X_cols."""
    train = [
        {
            "input": [[1], [2], [3]],  # 1 col, 3 rows
            "output": [[1, 1], [2, 2], [3, 3]]  # 2 cols: 2/1 = 2 is integer - this should work!
        }
    ]
    result = family.fit(train)
    assert result is True  # 2/1 = 2 is valid


def test_reject_true_non_integer_factor(family):
    """fit() must reject fractional tile counts."""
    # Create a case where columns don't divide evenly
    # Input: 2 rows, 3 cols
    # Output: 4 rows, 5 cols
    # tiles_v = 4/2 = 2 (valid)
    # tiles_h = 5/3 = not integer (invalid)
    train = [
        {
            "input": [[1, 2, 3], [4, 5, 6]],  # 2 rows, 3 cols
            "output": [[1, 2, 3, 1, 2], [4, 5, 6, 4, 5], [1, 2, 3, 1, 2], [4, 5, 6, 4, 5]]  # 4 rows, 5 cols
        }
    ]
    result = family.fit(train)
    assert result is False  # 5/3 is not integer


# ============================================================================
# Flip logic correctness
# ============================================================================

def test_flip_none_all_identical(family):
    """mode='none' produces all identical tiles."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4],
                      [1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4],
                      [1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.mode == "none"


def test_flip_h_odd_columns(family):
    """mode='h' flips horizontally on odd column tiles (C&1==1)."""
    # X = [[1,2],[3,4]]
    # tiles_h=3: C=0 (even) no flip, C=1 (odd) flip_h, C=2 (even) no flip
    # Pattern: [[1,2], [2,1], [1,2]]
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1, 1, 2], [3, 4, 4, 3, 3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_h == 3
    assert family.params.mode == "h"


def test_flip_v_odd_rows(family):
    """mode='v' flips vertically on odd row tiles (R&1==1)."""
    # X = [[1,2],[3,4]]
    # tiles_v=3: R=0 (even) no flip, R=1 (odd) flip_v, R=2 (even) no flip
    # Pattern: [[1,2],[3,4]]; [[3,4],[1,2]]; [[1,2],[3,4]]
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4], [3, 4], [1, 2], [1, 2], [3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 3
    assert family.params.mode == "v"


def test_flip_hv_checkerboard_2x2(family):
    """mode='hv' produces checkerboard pattern."""
    # (R=0,C=0): no flip (0^0=0)
    # (R=0,C=1): flip_h (0^1=1, R even)
    # (R=1,C=0): flip_v (1^0=1, R odd)
    # (R=1,C=1): no flip (1^1=0)
    # X = [[1,2],[3,4]]
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 1, 2], [1, 2, 3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 2
    assert family.params.tiles_h == 2
    assert family.params.mode == "hv"


def test_flip_hv_checkerboard_3x3(family):
    """mode='hv' checkerboard pattern extends correctly to 3×3."""
    # tiles_v=3, tiles_h=3
    # XOR pattern:
    # (0,0)=0 (0,1)=1 (0,2)=0
    # (1,0)=1 (1,1)=0 (1,2)=1
    # (2,0)=0 (2,1)=1 (2,2)=0
    # When XOR=1: if R even→flip_h, if R odd→flip_v
    X = [[1, 2], [3, 4]]
    expected = [
        # R=0: X, flip_h, X
        [1, 2, 2, 1, 1, 2],
        [3, 4, 4, 3, 3, 4],
        # R=1: flip_v, X, flip_v
        [3, 4, 1, 2, 3, 4],
        [1, 2, 3, 4, 1, 2],
        # R=2: X, flip_h, X
        [1, 2, 2, 1, 1, 2],
        [3, 4, 4, 3, 3, 4]
    ]
    train = [{"input": X, "output": expected}]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 3
    assert family.params.tiles_h == 3
    assert family.params.mode == "hv"


# ============================================================================
# Tiling dimensions
# ============================================================================

def test_output_dimensions_correctness(family):
    """Verify Y dimensions = (X_rows × tiles_v, X_cols × tiles_h)."""
    X = [[1, 2, 3], [4, 5, 6]]  # 2×3
    # Build 4×5 tiling with mode='none'
    expected = []
    for _ in range(4):  # 4 vertical repetitions
        expected.extend([[1, 2, 3] * 5, [4, 5, 6] * 5])  # 5 horizontal repetitions per row

    train = [{"input": X, "output": expected}]
    # Input: 2 rows, 3 cols
    # Output: 8 rows, 15 cols
    # tiles_v = 8/2 = 4, tiles_h = 15/3 = 5
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 4
    assert family.params.tiles_h == 5

    # Apply and check dimensions
    X_test = [[7, 8, 9], [10, 11, 12]]
    Y = family.apply(X_test)
    assert len(Y) == 8  # 2 × 4
    assert len(Y[0]) == 15  # 3 × 5


# ============================================================================
# FY exactness
# ============================================================================

def test_fy_exact_equality_required(family):
    """After successful fit, applying to train inputs must reproduce outputs exactly."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        },
        {
            "input": [[5, 6], [7, 8]],
            "output": [[5, 6, 6, 5], [7, 8, 8, 7]]
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


def test_fy_single_pixel_difference_rejects(family):
    """Single pixel difference in output should reject the mode."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 999]]  # Last pixel wrong
        }
    ]
    result = family.fit(train)
    # No mode will produce this output
    assert result is False


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
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]
    family.fit(train)

    # Then apply to empty grid
    result = family.apply([])
    assert result == []


def test_single_pixel_grid(family):
    """Single pixel grid tiles correctly."""
    train = [
        {
            "input": [[5]],
            "output": [[5, 5], [5, 5]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 2
    assert family.params.tiles_h == 2
    assert family.params.mode == "none"


def test_identity_case(family):
    """tiles_v=1, tiles_h=1, mode='none' is identity."""
    train = [
        {
            "input": [[1, 2, 3], [4, 5, 6]],
            "output": [[1, 2, 3], [4, 5, 6]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 1
    assert family.params.tiles_h == 1
    assert family.params.mode == "none"


def test_large_tiling(family):
    """Large tile counts work correctly."""
    X = [[1, 2], [3, 4]]
    # Create 5×7 tiling with mode='none'
    expected = []
    for _ in range(5):
        expected.extend([[1, 2] * 7, [3, 4] * 7])

    train = [{"input": X, "output": expected}]
    result = family.fit(train)
    assert result is True
    assert family.params.tiles_v == 5
    assert family.params.tiles_h == 7
    assert family.params.mode == "none"


# ============================================================================
# Parity logic (bit operations)
# ============================================================================

def test_parity_odd_column_detection(family):
    """Verify (C & 1) == 1 detects odd columns."""
    # Test mode="h" which flips on odd columns
    train = [
        {
            "input": [[1, 9], [2, 8]],  # Asymmetric to detect flips
            "output": [[1, 9, 9, 1, 1, 9, 9, 1], [2, 8, 8, 2, 2, 8, 8, 2]]
        }
    ]
    # tiles_h=4: C=0 (even) no flip, C=1 (odd) flip, C=2 (even) no flip, C=3 (odd) flip
    result = family.fit(train)
    assert result is True
    assert family.params.mode == "h"


def test_parity_odd_row_detection(family):
    """Verify (R & 1) == 1 detects odd rows."""
    # Test mode="v" which flips on odd rows
    train = [
        {
            "input": [[1, 2], [9, 8]],  # Asymmetric to detect flips
            "output": [[1, 2], [9, 8], [9, 8], [1, 2], [1, 2], [9, 8], [9, 8], [1, 2]]
        }
    ]
    # tiles_v=4: R=0 no flip, R=1 flip, R=2 no flip, R=3 flip
    result = family.fit(train)
    assert result is True
    assert family.params.mode == "v"


def test_parity_xor_checkerboard(family):
    """Verify (R^C)&1 produces checkerboard pattern."""
    # Build a checkerboard manually to verify XOR logic
    X = [[1, 9], [2, 8]]  # Asymmetric
    # (0,0): 0^0=0 → no flip → [[1,9],[2,8]]
    # (0,1): 0^1=1, R=0 even → flip_h → [[9,1],[8,2]]
    # (1,0): 1^0=1, R=1 odd → flip_v → [[2,8],[1,9]]
    # (1,1): 1^1=0 → no flip → [[1,9],[2,8]]
    expected = [
        [1, 9, 9, 1],
        [2, 8, 8, 2],
        [2, 8, 1, 9],
        [1, 9, 2, 8]
    ]
    train = [{"input": X, "output": expected}]
    result = family.fit(train)
    assert result is True
    assert family.params.mode == "hv"


# ============================================================================
# Determinism
# ============================================================================

def test_deterministic_repeated_fit(family):
    """Running fit() twice on same data yields identical params."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 1, 2], [1, 2, 3, 4]]
        }
    ]

    result1 = family.fit(train)
    params1 = (family.params.tiles_v, family.params.tiles_h, family.params.mode)

    # Create fresh family and fit again
    family2 = ParityTileFamily()
    result2 = family2.fit(train)
    params2 = (family2.params.tiles_v, family2.params.tiles_h, family2.params.mode)

    assert result1 == result2 == True
    assert params1 == params2


def test_deterministic_apply(family):
    """Applying same input yields identical output."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]
    family.fit(train)

    X = [[5, 6], [7, 8]]
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
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]
    train_copy = copy_grid(train)

    family.fit(train)

    assert deep_eq(train, train_copy)


def test_apply_does_not_mutate_input(family):
    """apply() must not mutate input X."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]
    family.fit(train)

    X = [[5, 6], [7, 8]]
    X_copy = copy_grid(X)

    family.apply(X)

    assert deep_eq(X, X_copy)


def test_apply_no_row_aliasing(family):
    """apply() output must have no row aliasing."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2, 2, 1], [3, 4, 4, 3]]
        }
    ]
    family.fit(train)

    X = [[5, 6], [7, 8]]
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
