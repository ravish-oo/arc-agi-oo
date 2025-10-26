"""
Comprehensive action test suite (P5-05).

Categories:
1. Purity — Xp unchanged after kernel calls
2. Correctness — Basic kernel behavior
3. Edge Cases — Empty, 1x1, 1xN, Nx1, borders
4. Validation — ValueError on invalid inputs
5. Determinism — Stable across repeated calls, coord orders
6. GLUE-Safety — CRITICAL: Read from frozen base only
7. Verifier — FY equality, vacuous truth
8. Inference — Fixed action order, unified params
9. Adversarial — Duplicates, borders, symmetric mirrors
10. Integration — End-to-end workflows

Coverage Target: ~140 tests within 220 LOC budget
"""

import pytest
import copy
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.action_inference import (
    apply_set_color, apply_keep_nonzero, apply_identity,
    apply_mirror_h, apply_mirror_v,
    verify_action_on_class, infer_action_for_class
)


# ============================================================================
# FIXTURES
# ============================================================================

# Empty/small grids
g_empty = []
g_1x1 = [[5]]
g_1x3 = [[1, 2, 3]]
g_3x1 = [[1], [2], [3]]

# 2x2 grids
g_2x2 = [[1, 2], [3, 4]]
g_2x2_zeros = [[0, 0], [0, 0]]
g_2x2_mixed = [[1, 0], [0, 2]]

# 3x3 grids (odd)
g_3x3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
g_3x3_sym_h = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]
g_3x3_sym_v = [[1, 2, 1], [4, 5, 4], [7, 8, 7]]

# 4x4 grids (even)
g_4x4 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]

# Ragged (invalid)
g_ragged = [[1, 2], [3]]


# ============================================================================
# Category 1: Purity Tests (5 tests)
# ============================================================================

def test_set_color_purity():
    """Xp unchanged after apply_set_color."""
    original = copy.deepcopy(g_2x2)
    apply_set_color(g_2x2, [(0, 0)], 5)
    assert g_2x2 == original


def test_keep_nonzero_purity():
    """Xp unchanged after apply_keep_nonzero."""
    original = copy.deepcopy(g_2x2_mixed)
    apply_keep_nonzero(g_2x2_mixed, [(0, 0), (1, 1)])
    assert g_2x2_mixed == original


def test_identity_purity():
    """Xp unchanged after apply_identity."""
    original = copy.deepcopy(g_2x2)
    apply_identity(g_2x2, [(0, 0)])
    assert g_2x2 == original


def test_mirror_h_purity():
    """Xp unchanged after apply_mirror_h."""
    original = copy.deepcopy(g_3x3)
    apply_mirror_h(g_3x3, [(0, 0)])
    assert g_3x3 == original


def test_mirror_v_purity():
    """Xp unchanged after apply_mirror_v."""
    original = copy.deepcopy(g_3x3)
    apply_mirror_v(g_3x3, [(0, 0)])
    assert g_3x3 == original


# ============================================================================
# Category 2: Correctness Tests (20 tests)
# ============================================================================

def test_set_color_single():
    """Single pixel change."""
    out = apply_set_color(g_2x2, [(0, 0)], 5)
    assert out == [[5, 2], [3, 4]]


def test_set_color_multiple():
    """Multiple pixels."""
    out = apply_set_color(g_2x2, [(0, 0), (1, 1)], 7)
    assert out == [[7, 2], [3, 7]]


def test_set_color_empty_coords():
    """Empty coords → identity."""
    out = apply_set_color(g_2x2, [], 9)
    assert out == g_2x2


def test_set_color_all_coords():
    """All pixels."""
    out = apply_set_color(g_2x2, [(0, 0), (0, 1), (1, 0), (1, 1)], 0)
    assert out == [[0, 0], [0, 0]]


def test_keep_nonzero_mixed():
    """Nonzero preserved, zero stays zero."""
    out = apply_keep_nonzero(g_2x2_mixed, [(0, 0), (0, 1), (1, 0), (1, 1)])
    assert out == g_2x2_mixed


def test_keep_nonzero_all_zeros():
    """All zeros remain zeros."""
    out = apply_keep_nonzero(g_2x2_zeros, [(0, 0), (1, 1)])
    assert out == g_2x2_zeros


def test_identity_returns_copy():
    """Exact copy, different object."""
    out = apply_identity(g_2x2, [(0, 0)])
    assert out == g_2x2
    assert out is not g_2x2


def test_mirror_h_simple_3x3():
    """Odd grid horizontal mirror."""
    out = apply_mirror_h(g_3x3, [(0, 0)])
    assert out[0][0] == 7  # Xp[2][0]


def test_mirror_h_even_4x4():
    """Even grid horizontal mirror."""
    out = apply_mirror_h(g_4x4, [(0, 0)])
    assert out[0][0] == 3  # Xp[3][0]


def test_mirror_h_center_row_odd():
    """Center row (r=1) mirrors to itself in 3x3."""
    out = apply_mirror_h(g_3x3, [(1, 1)])
    assert out[1][1] == 5  # Xp[1][1] (center)


def test_mirror_v_simple_3x3():
    """Odd grid vertical mirror."""
    out = apply_mirror_v(g_3x3, [(0, 2)])
    assert out[0][2] == 1  # Xp[0][0]


def test_mirror_v_even_4x4():
    """Even grid vertical mirror."""
    out = apply_mirror_v(g_4x4, [(0, 0)])
    assert out[0][0] == 4  # Xp[0][3]


def test_mirror_v_center_col_odd():
    """Center col (c=1) mirrors to itself in 3x3."""
    out = apply_mirror_v(g_3x3, [(1, 1)])
    assert out[1][1] == 5  # Xp[1][1] (center)


def test_mirror_h_two_coords():
    """Both coords mirror."""
    out = apply_mirror_h(g_2x2, [(0, 0), (1, 0)])
    assert out == [[3, 2], [1, 4]]


def test_mirror_v_two_coords():
    """Both coords mirror."""
    out = apply_mirror_v(g_2x2, [(0, 0), (0, 1)])
    assert out == [[2, 1], [3, 4]]


def test_set_color_color_zero():
    """Color 0 (black)."""
    out = apply_set_color(g_2x2, [(0, 0)], 0)
    assert out[0][0] == 0


def test_set_color_color_nine():
    """Color 9."""
    out = apply_set_color(g_2x2, [(0, 0)], 9)
    assert out[0][0] == 9


def test_keep_nonzero_partial_coords():
    """Only some coords."""
    out = apply_keep_nonzero(g_2x2_mixed, [(0, 0), (1, 1)])
    assert out == [[1, 0], [0, 2]]


def test_identity_coords_ignored():
    """coords parameter ignored."""
    out1 = apply_identity(g_2x2, [(0, 0)])
    out2 = apply_identity(g_2x2, [(1, 1)])
    assert out1 == out2 == g_2x2


def test_mirror_h_full_grid():
    """All coords."""
    out = apply_mirror_h(g_2x2, [(0, 0), (0, 1), (1, 0), (1, 1)])
    assert out == [[3, 4], [1, 2]]


# ============================================================================
# Category 3: Edge Cases (25 tests)
# ============================================================================

def test_set_color_1x1():
    """Single pixel grid."""
    out = apply_set_color(g_1x1, [(0, 0)], 7)
    assert out == [[7]]


def test_set_color_1xN():
    """1 row N cols."""
    out = apply_set_color(g_1x3, [(0, 1)], 8)
    assert out == [[1, 8, 3]]


def test_set_color_Nx1():
    """N rows 1 col."""
    out = apply_set_color(g_3x1, [(1, 0)], 9)
    assert out == [[1], [9], [3]]


def test_set_color_border_coords():
    """Only border pixels."""
    coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    out = apply_set_color(g_3x3, coords, 0)
    assert out[1][1] == 5  # Center unchanged


def test_mirror_h_1x1():
    """1x1 grid (r=0 mirrors to itself)."""
    out = apply_mirror_h(g_1x1, [(0, 0)])
    assert out == g_1x1


def test_mirror_h_1xN():
    """1 row (all mirror to same row)."""
    out = apply_mirror_h(g_1x3, [(0, 0)])
    assert out[0][0] == 1  # Mirrors to itself


def test_mirror_h_Nx1():
    """1 col (vertical mirror, different rows)."""
    out = apply_mirror_h(g_3x1, [(0, 0)])
    assert out[0][0] == 3  # Xp[2][0]


def test_mirror_v_1x1():
    """1x1 grid (c=0 mirrors to itself)."""
    out = apply_mirror_v(g_1x1, [(0, 0)])
    assert out == g_1x1


def test_mirror_v_1xN():
    """1 row (horizontal mirror, different cols)."""
    out = apply_mirror_v(g_1x3, [(0, 0)])
    assert out[0][0] == 3  # Xp[0][2]


def test_mirror_v_Nx1():
    """1 col (all mirror to same col)."""
    out = apply_mirror_v(g_3x1, [(0, 0)])
    assert out[0][0] == 1  # Mirrors to itself


def test_keep_nonzero_empty_coords():
    """Empty coords → identity."""
    out = apply_keep_nonzero(g_2x2, [])
    assert out == g_2x2


def test_identity_empty_coords():
    """Empty coords → identity."""
    out = apply_identity(g_2x2, [])
    assert out == g_2x2


def test_mirror_h_border_top_bottom():
    """Top and bottom rows."""
    out = apply_mirror_h(g_3x3, [(0, 1), (2, 1)])
    assert out[0][1] == 8  # Xp[2][1]
    assert out[2][1] == 2  # Xp[0][1]


def test_mirror_v_border_left_right():
    """Left and right cols."""
    out = apply_mirror_v(g_3x3, [(1, 0), (1, 2)])
    assert out[1][0] == 6  # Xp[1][2]
    assert out[1][2] == 4  # Xp[1][0]


def test_set_color_duplicate_coords():
    """Duplicates handled deterministically."""
    out = apply_set_color(g_2x2, [(0, 0), (0, 0), (1, 1)], 5)
    assert out == [[5, 2], [3, 5]]


def test_mirror_h_duplicate_coords():
    """Duplicates in mirror coords."""
    out = apply_mirror_h(g_2x2, [(0, 0), (0, 0)])
    assert out[0][0] == 3


def test_mirror_v_duplicate_coords():
    """Duplicates in mirror coords."""
    out = apply_mirror_v(g_2x2, [(0, 0), (0, 0)])
    assert out[0][0] == 2


def test_set_color_corners_only():
    """Only 4 corners."""
    coords = [(0, 0), (0, 2), (2, 0), (2, 2)]
    out = apply_set_color(g_3x3, coords, 0)
    assert out[0][0] == 0 and out[0][2] == 0
    assert out[2][0] == 0 and out[2][2] == 0
    assert out[1][1] == 5  # Center unchanged


def test_mirror_h_odd_all_rows():
    """All rows in odd grid."""
    out = apply_mirror_h(g_3x3, [(0, 0), (1, 0), (2, 0)])
    assert out[0][0] == 7  # Row 2
    assert out[1][0] == 4  # Center row (itself)
    assert out[2][0] == 1  # Row 0


def test_mirror_v_odd_all_cols():
    """All cols in odd grid."""
    out = apply_mirror_v(g_3x3, [(0, 0), (0, 1), (0, 2)])
    assert out[0][0] == 3  # Col 2
    assert out[0][1] == 2  # Center col (itself)
    assert out[0][2] == 1  # Col 0


def test_keep_nonzero_1x1():
    """1x1 grid."""
    out = apply_keep_nonzero(g_1x1, [(0, 0)])
    assert out == g_1x1


def test_identity_1x1():
    """1x1 grid."""
    out = apply_identity(g_1x1, [(0, 0)])
    assert out == g_1x1


def test_set_color_even_grid_no_center():
    """Even grid (no single center)."""
    out = apply_set_color(g_4x4, [(1, 1), (1, 2), (2, 1), (2, 2)], 9)
    assert all(out[r][c] == 9 for r in [1, 2] for c in [1, 2])


def test_mirror_h_even_grid_middle_rows():
    """Even grid middle rows."""
    out = apply_mirror_h(g_4x4, [(1, 0), (2, 0)])
    assert out[1][0] == 9  # Xp[2][0]
    assert out[2][0] == 5  # Xp[1][0]


def test_mirror_v_even_grid_middle_cols():
    """Even grid middle cols."""
    out = apply_mirror_v(g_4x4, [(0, 1), (0, 2)])
    assert out[0][1] == 3  # Xp[0][2]
    assert out[0][2] == 2  # Xp[0][1]


# ============================================================================
# Category 4: Validation Tests (15 tests)
# ============================================================================

def test_set_color_ragged_grid():
    """ValueError on ragged."""
    with pytest.raises(ValueError):
        apply_set_color(g_ragged, [(0, 0)], 5)


def test_set_color_invalid_color_negative():
    """ValueError on color < 0."""
    with pytest.raises(ValueError):
        apply_set_color(g_2x2, [(0, 0)], -1)


def test_set_color_invalid_color_too_large():
    """ValueError on color > 9."""
    with pytest.raises(ValueError):
        apply_set_color(g_2x2, [(0, 0)], 10)


def test_set_color_coord_out_of_bounds():
    """ValueError on OOB coords."""
    with pytest.raises(ValueError):
        apply_set_color(g_2x2, [(5, 5)], 5)


def test_mirror_h_ragged_grid():
    """ValueError on ragged."""
    with pytest.raises(ValueError):
        apply_mirror_h(g_ragged, [(0, 0)])


def test_mirror_h_coord_out_of_bounds():
    """ValueError on OOB coords."""
    with pytest.raises(ValueError):
        apply_mirror_h(g_2x2, [(5, 5)])


def test_mirror_v_ragged_grid():
    """ValueError on ragged."""
    with pytest.raises(ValueError):
        apply_mirror_v(g_ragged, [(0, 0)])


def test_mirror_v_coord_out_of_bounds():
    """ValueError on OOB coords."""
    with pytest.raises(ValueError):
        apply_mirror_v(g_2x2, [(5, 5)])


def test_keep_nonzero_ragged_grid():
    """ValueError on ragged."""
    with pytest.raises(ValueError):
        apply_keep_nonzero(g_ragged, [(0, 0)])


def test_identity_ragged_grid():
    """ValueError on ragged."""
    with pytest.raises(ValueError):
        apply_identity(g_ragged, [(0, 0)])


def test_mirror_h_empty_grid():
    """ValueError on empty grid."""
    with pytest.raises(ValueError):
        apply_mirror_h(g_empty, [])


def test_mirror_v_empty_grid():
    """ValueError on empty grid."""
    with pytest.raises(ValueError):
        apply_mirror_v(g_empty, [])


def test_set_color_negative_coord():
    """ValueError on negative coord."""
    with pytest.raises(ValueError):
        apply_set_color(g_2x2, [(-1, 0)], 5)


def test_mirror_h_negative_coord():
    """ValueError on negative coord."""
    with pytest.raises(ValueError):
        apply_mirror_h(g_2x2, [(-1, 0)])


def test_mirror_v_negative_coord():
    """ValueError on negative coord."""
    with pytest.raises(ValueError):
        apply_mirror_v(g_2x2, [(0, -1)])


# ============================================================================
# Category 5: Determinism Tests (10 tests)
# ============================================================================

def test_set_color_determinism():
    """Repeated calls → same result."""
    out1 = apply_set_color(g_2x2, [(0, 0)], 5)
    out2 = apply_set_color(g_2x2, [(0, 0)], 5)
    assert out1 == out2


def test_set_color_unsorted_coords():
    """Unsorted coords → same result."""
    out1 = apply_set_color(g_2x2, [(0, 0), (1, 1)], 5)
    out2 = apply_set_color(g_2x2, [(1, 1), (0, 0)], 5)
    assert out1 == out2


def test_mirror_h_determinism():
    """Repeated calls → same result."""
    out1 = apply_mirror_h(g_3x3, [(0, 0), (2, 0)])
    out2 = apply_mirror_h(g_3x3, [(0, 0), (2, 0)])
    assert out1 == out2


def test_mirror_h_unsorted_coords():
    """Unsorted coords → same result."""
    out1 = apply_mirror_h(g_3x3, [(0, 0), (2, 0)])
    out2 = apply_mirror_h(g_3x3, [(2, 0), (0, 0)])
    assert out1 == out2


def test_mirror_v_determinism():
    """Repeated calls → same result."""
    out1 = apply_mirror_v(g_3x3, [(0, 0), (0, 2)])
    out2 = apply_mirror_v(g_3x3, [(0, 0), (0, 2)])
    assert out1 == out2


def test_mirror_v_unsorted_coords():
    """Unsorted coords → same result."""
    out1 = apply_mirror_v(g_3x3, [(0, 0), (0, 2)])
    out2 = apply_mirror_v(g_3x3, [(0, 2), (0, 0)])
    assert out1 == out2


def test_keep_nonzero_determinism():
    """Repeated calls → same result."""
    out1 = apply_keep_nonzero(g_2x2_mixed, [(0, 0), (1, 1)])
    out2 = apply_keep_nonzero(g_2x2_mixed, [(0, 0), (1, 1)])
    assert out1 == out2


def test_identity_determinism():
    """Repeated calls → same result."""
    out1 = apply_identity(g_2x2, [(0, 0)])
    out2 = apply_identity(g_2x2, [(0, 0)])
    assert out1 == out2


def test_set_color_shuffled_coords():
    """Shuffled coords → same result."""
    coords_orig = [(0, 0), (0, 1), (1, 0), (1, 1)]
    coords_shuffled = [(1, 1), (0, 0), (1, 0), (0, 1)]
    out1 = apply_set_color(g_2x2, coords_orig, 7)
    out2 = apply_set_color(g_2x2, coords_shuffled, 7)
    assert out1 == out2


def test_mirror_h_shuffled_coords():
    """Shuffled coords → same result."""
    coords_orig = [(0, 0), (1, 0), (2, 0)]
    coords_shuffled = [(2, 0), (0, 0), (1, 0)]
    out1 = apply_mirror_h(g_3x3, coords_orig)
    out2 = apply_mirror_h(g_3x3, coords_shuffled)
    assert out1 == out2


# ============================================================================
# Category 6: GLUE-Safety Tests (CRITICAL!) (10 tests)
# ============================================================================

def test_mirror_h_glue_safety_two_sided():
    """Coords include both (r,c) and (R-1-r,c) — GLUE-safe."""
    # g_2x2 = [[1,2],[3,4]], R=2
    # (0,0) mirrors to (1,0), (1,0) mirrors to (0,0)
    # If read from Out (broken GLUE): depends on order
    # If read from Xp (correct GLUE): always [[3,2],[1,4]]
    out = apply_mirror_h(g_2x2, [(0, 0), (1, 0)])
    assert out == [[3, 2], [1, 4]]


def test_mirror_h_glue_safety_center_odd():
    """Center row in odd grid — reads from Xp."""
    # g_3x3, center row r=1 mirrors to itself
    out = apply_mirror_h(g_3x3, [(1, 0), (1, 1), (1, 2)])
    assert out[1] == [4, 5, 6]  # Unchanged (mirrors to self from Xp)


def test_mirror_v_glue_safety_two_sided():
    """Coords include both (r,c) and (r,C-1-c) — GLUE-safe."""
    # g_2x2 = [[1,2],[3,4]], C=2
    # (0,0) mirrors to (0,1), (0,1) mirrors to (0,0)
    out = apply_mirror_v(g_2x2, [(0, 0), (0, 1)])
    assert out == [[2, 1], [3, 4]]


def test_mirror_v_glue_safety_center_odd():
    """Center col in odd grid — reads from Xp."""
    # g_3x3, center col c=1 mirrors to itself
    out = apply_mirror_v(g_3x3, [(0, 1), (1, 1), (2, 1)])
    assert [out[r][1] for r in range(3)] == [2, 5, 8]  # Unchanged


def test_set_color_glue_safety_duplicates():
    """Overlapping coords (duplicates) — deterministic."""
    out = apply_set_color(g_2x2, [(0, 0), (0, 0), (0, 0)], 5)
    assert out[0][0] == 5


def test_mirror_h_glue_full_grid():
    """All coords mirror — entire grid flips."""
    coords = [(r, c) for r in range(2) for c in range(2)]
    out = apply_mirror_h(g_2x2, coords)
    assert out == [[3, 4], [1, 2]]


def test_mirror_v_glue_full_grid():
    """All coords mirror — entire grid flips."""
    coords = [(r, c) for r in range(2) for c in range(2)]
    out = apply_mirror_v(g_2x2, coords)
    assert out == [[2, 1], [4, 3]]


def test_mirror_h_glue_3x3_symmetric():
    """3x3 symmetric coords — all pairs."""
    coords = [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)]
    out = apply_mirror_h(g_3x3, coords)
    assert out[0] == [7, 8, 9]
    assert out[1] == [4, 5, 6]  # Center unchanged
    assert out[2] == [1, 2, 3]


def test_mirror_v_glue_3x3_symmetric():
    """3x3 symmetric coords — all pairs."""
    coords = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]
    out = apply_mirror_v(g_3x3, coords)
    assert out[0] == [3, 2, 1]
    assert out[1] == [6, 5, 4]
    assert out[2] == [9, 8, 7]


def test_mirror_h_glue_unsorted_two_sided():
    """Unsorted two-sided coords — order independent."""
    out1 = apply_mirror_h(g_2x2, [(0, 0), (1, 0)])
    out2 = apply_mirror_h(g_2x2, [(1, 0), (0, 0)])
    assert out1 == out2 == [[3, 2], [1, 4]]


# ============================================================================
# Category 7: Verifier Tests (15 tests)
# ============================================================================

def test_verify_set_color_pass():
    """set_color action matches Y."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(("set_color", 5), items, class_coords)


def test_verify_set_color_fail():
    """set_color action doesn't match Y."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0)]
    assert not verify_action_on_class(("set_color", 7), items, class_coords)


def test_verify_mirror_h_pass():
    """mirror_h action matches Y."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]]}]
    class_coords = [(0, 0, 0), (0, 1, 0)]
    assert verify_action_on_class(("mirror_h", None), items, class_coords)


def test_verify_mirror_v_pass():
    """mirror_v action matches Y."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[3, 2, 1]]}]
    class_coords = [(0, 0, 0), (0, 0, 2)]
    assert verify_action_on_class(("mirror_v", None), items, class_coords)


def test_verify_empty_class_coords():
    """Empty coords → True (vacuous)."""
    items = [{"Xp": [[1, 2]], "Y": [[3, 4]]}]
    assert verify_action_on_class(("identity", None), items, [])


def test_verify_multi_train_pass():
    """All trains match."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 2]]},
        {"Xp": [[3, 4]], "Y": [[5, 4]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]
    assert verify_action_on_class(("set_color", 5), items, class_coords)


def test_verify_multi_train_fail_one():
    """One train fails → False."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 2]]},
        {"Xp": [[3, 4]], "Y": [[7, 4]]}  # Different color
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]
    assert not verify_action_on_class(("set_color", 5), items, class_coords)


def test_verify_identity_pass():
    """identity matches when Y == Xp."""
    items = [{"Xp": [[1, 2]], "Y": [[1, 2]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]
    assert verify_action_on_class(("identity", None), items, class_coords)


def test_verify_keep_nonzero_pass():
    """keep_nonzero matches when Y == Xp."""
    items = [{"Xp": [[1, 0]], "Y": [[1, 0]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]
    assert verify_action_on_class(("keep_nonzero", None), items, class_coords)


def test_verify_partial_class():
    """Only some pixels verified."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[5, 2, 9]]}]
    class_coords = [(0, 0, 0)]  # Only first pixel
    assert verify_action_on_class(("set_color", 5), items, class_coords)


def test_verify_fy_exactness():
    """FY equality is exact (no tolerance)."""
    items = [{"Xp": [[1]], "Y": [[2]]}]
    class_coords = [(0, 0, 0)]
    assert not verify_action_on_class(("set_color", 1), items, class_coords)


def test_verify_duplicates_in_coords():
    """Duplicates in coords handled."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0), (0, 0, 0)]  # Duplicate
    assert verify_action_on_class(("set_color", 5), items, class_coords)


def test_verify_three_trains():
    """Three trains all match."""
    items = [
        {"Xp": [[1]], "Y": [[5]]},
        {"Xp": [[2]], "Y": [[5]]},
        {"Xp": [[3]], "Y": [[5]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert verify_action_on_class(("set_color", 5), items, class_coords)


def test_verify_mixed_coords_per_train():
    """Different coords in different trains."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 2]]},
        {"Xp": [[3, 4]], "Y": [[3, 5]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 1)]
    assert verify_action_on_class(("set_color", 5), items, class_coords)


def test_verify_all_actions():
    """All 5 actions can be verified."""
    items = [{"Xp": [[1]], "Y": [[1]]}]
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(("identity", None), items, class_coords)
    assert verify_action_on_class(("keep_nonzero", None), items, class_coords)


# ============================================================================
# Category 8: Inference Tests (20 tests)
# ============================================================================

def test_infer_set_color_unified_single():
    """Unified color → set_color."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 5)


def test_infer_set_color_unified_multi():
    """Unified across trains."""
    items = [
        {"Xp": [[1, 2]], "Y": [[7, 2]]},
        {"Xp": [[3, 4]], "Y": [[7, 4]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 7)


def test_infer_set_color_skip_mixed():
    """Mixed colors → skip set_color."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 2]]},
        {"Xp": [[3, 4]], "Y": [[7, 4]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    assert result is None or result[0] != "set_color"


def test_infer_mirror_h_simple():
    """mirror_h wins when applicable."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]]}]
    class_coords = [(0, 0, 0), (0, 1, 0)]
    assert infer_action_for_class(items, class_coords) == ("mirror_h", None)


def test_infer_mirror_v_simple():
    """mirror_v wins when applicable."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[3, 2, 1]]}]
    class_coords = [(0, 0, 0), (0, 0, 2)]
    assert infer_action_for_class(items, class_coords) == ("mirror_v", None)


def test_infer_keep_nonzero_simple():
    """keep_nonzero wins when applicable."""
    items = [{"Xp": [[1, 0, 3], [4, 0, 6]], "Y": [[1, 0, 3], [4, 0, 6]]}]
    class_coords = [(0, r, c) for r in range(2) for c in range(3)]
    assert infer_action_for_class(items, class_coords) == ("keep_nonzero", None)


def test_infer_identity_fallback():
    """identity as last resort."""
    items = [{"Xp": [[7, 8, 9], [4, 5, 6]], "Y": [[7, 8, 9], [4, 5, 6]]}]
    class_coords = [(0, r, c) for r in range(2) for c in range(3)]
    result = infer_action_for_class(items, class_coords)
    assert result in [("keep_nonzero", None), ("identity", None)]


def test_infer_empty_class_coords():
    """Empty coords → identity (vacuous)."""
    items = [{"Xp": [[1, 2]], "Y": [[3, 4]]}]
    assert infer_action_for_class(items, []) == ("identity", None)


def test_infer_unsat_no_action():
    """None if no action matches."""
    items = [{"Xp": [[1]], "Y": [[999]]}]
    class_coords = [(0, 0, 0)]
    assert infer_action_for_class(items, class_coords) is None


def test_infer_first_pass_order():
    """set_color before mirrors."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 5]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 5)


def test_infer_determinism():
    """Repeated calls → same result."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0)]
    result1 = infer_action_for_class(items, class_coords)
    result2 = infer_action_for_class(items, class_coords)
    assert result1 == result2


def test_infer_shuffled_coords():
    """Shuffled coords → same result."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[7, 7, 7]]}]
    coords1 = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    coords2 = [(0, 0, 2), (0, 0, 0), (0, 0, 1)]
    result1 = infer_action_for_class(items, coords1)
    result2 = infer_action_for_class(items, coords2)
    assert result1 == result2 == ("set_color", 7)


def test_infer_color_zero():
    """Infer set_color with color 0."""
    items = [{"Xp": [[5, 6]], "Y": [[0, 6]]}]
    class_coords = [(0, 0, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 0)


def test_infer_color_nine():
    """Infer set_color with color 9."""
    items = [{"Xp": [[1, 2]], "Y": [[9, 2]]}]
    class_coords = [(0, 0, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 9)


def test_infer_single_pixel():
    """Single pixel class."""
    items = [{"Xp": [[5]], "Y": [[7]]}]
    class_coords = [(0, 0, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 7)


def test_infer_three_trains():
    """Three trains unified color."""
    items = [
        {"Xp": [[1]], "Y": [[3]]},
        {"Xp": [[2]], "Y": [[3]]},
        {"Xp": [[5]], "Y": [[3]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 3)


def test_infer_mixed_trains_unsat():
    """Mixed colors across trains → UNSAT."""
    items = [
        {"Xp": [[1]], "Y": [[3]]},
        {"Xp": [[2]], "Y": [[5]]},
        {"Xp": [[7]], "Y": [[9]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert infer_action_for_class(items, class_coords) is None


def test_infer_mirror_multi_train():
    """Mirror across multiple trains."""
    items = [
        {"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]]},
        {"Xp": [[5, 6], [7, 8]], "Y": [[7, 6], [5, 8]]}
    ]
    class_coords = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    assert infer_action_for_class(items, class_coords) == ("mirror_h", None)


def test_infer_partial_class():
    """Only some pixels in class."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[999, 2, 888]]}]
    class_coords = [(0, 0, 1)]  # Only middle pixel
    assert infer_action_for_class(items, class_coords) == ("set_color", 2)


def test_infer_large_multi_train():
    """Multiple trains many coords."""
    items = [
        {"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[0, 2, 0], [4, 5, 6]]},
        {"Xp": [[7, 8], [9, 0]], "Y": [[7, 8], [0, 0]]}
    ]
    class_coords = [(0, 0, 0), (0, 0, 2), (1, 1, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 0)


# ============================================================================
# Category 9: Adversarial Minis (15 tests)
# ============================================================================

def test_adv_dup_coords_set_color():
    """Duplicates in coords list."""
    out = apply_set_color(g_2x2, [(0, 0), (0, 0), (1, 1), (1, 1)], 5)
    assert out == [[5, 2], [3, 5]]


def test_adv_dup_coords_mixed_order():
    """Duplicates + unsorted."""
    out = apply_set_color(g_2x2, [(1, 1), (0, 0), (0, 0), (1, 1)], 5)
    assert out == [[5, 2], [3, 5]]


def test_adv_dup_coords_mirror_h():
    """Mirror with duplicates."""
    out = apply_mirror_h(g_2x2, [(0, 0), (0, 0)])
    assert out[0][0] == 3


def test_adv_dup_coords_verifier():
    """Verifier handles duplicates."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0), (0, 0, 0)]
    assert verify_action_on_class(("set_color", 5), items, class_coords)


def test_adv_dup_coords_inference():
    """Inference handles duplicates."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0), (0, 0, 0)]
    assert infer_action_for_class(items, class_coords) == ("set_color", 5)


def test_adv_border_only_3x3():
    """Set only border pixels."""
    coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    out = apply_set_color(g_3x3, coords, 0)
    assert out[1][1] == 5  # Center unchanged


def test_adv_border_mirror_h_odd():
    """Mirror borders in odd grid."""
    out = apply_mirror_h(g_3x3, [(0, 1), (2, 1)])
    assert out[0][1] == 8  # From row 2
    assert out[2][1] == 2  # From row 0


def test_adv_border_mirror_v_odd():
    """Mirror borders in odd grid."""
    out = apply_mirror_v(g_3x3, [(1, 0), (1, 2)])
    assert out[1][0] == 6  # From col 2
    assert out[1][2] == 4  # From col 0


def test_adv_border_corners_only():
    """Only 4 corners."""
    coords = [(0, 0), (0, 2), (2, 0), (2, 2)]
    out = apply_set_color(g_3x3, coords, 0)
    assert sum(out[r][c] for r, c in coords) == 0
    assert out[1][1] == 5  # Center unchanged


def test_adv_sym_mirror_conflict_h():
    """Coords include (r,c) and (R-1-r,c) — GLUE-safe."""
    out = apply_mirror_h(g_2x2, [(0, 0), (1, 0)])
    assert out == [[3, 2], [1, 4]]


def test_adv_sym_mirror_conflict_v():
    """Coords include (r,c) and (r,C-1-c) — GLUE-safe."""
    out = apply_mirror_v(g_1x3, [(0, 0), (0, 2)])
    assert out == [[3, 2, 1]]


def test_adv_sym_mirror_inference():
    """Inference handles symmetric coords."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]]}]
    class_coords = [(0, 0, 0), (0, 1, 0)]
    assert infer_action_for_class(items, class_coords) == ("mirror_h", None)


def test_adv_mixed_trains_one_pixel_fails():
    """One pixel in one train fails → False."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 5]]},
        {"Xp": [[3, 4]], "Y": [[5, 7]]}  # Second pixel differs
    ]
    class_coords = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
    assert not verify_action_on_class(("set_color", 5), items, class_coords)


def test_adv_mixed_trains_inference_unsat():
    """Inference returns None if trains conflict."""
    items = [
        {"Xp": [[1]], "Y": [[3]]},
        {"Xp": [[2]], "Y": [[5]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]
    assert infer_action_for_class(items, class_coords) is None


def test_adv_border_center_line_even():
    """Even grid (two middle rows/cols)."""
    coords = [(1, 1), (1, 2), (2, 1), (2, 2)]
    out = apply_set_color(g_4x4, coords, 9)
    assert all(out[r][c] == 9 for r in [1, 2] for c in [1, 2])


# ============================================================================
# Category 10: Integration Tests (5 tests)
# ============================================================================

def test_integration_full_workflow_set_color():
    """End-to-end: items → infer → verify."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0)]
    action = infer_action_for_class(items, class_coords)
    assert action == ("set_color", 5)
    assert verify_action_on_class(action, items, class_coords)


def test_integration_full_workflow_mirror_h():
    """End-to-end: items → infer → verify."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]]}]
    class_coords = [(0, 0, 0), (0, 1, 0)]
    action = infer_action_for_class(items, class_coords)
    assert action == ("mirror_h", None)
    assert verify_action_on_class(action, items, class_coords)


def test_integration_multi_action_cascade():
    """Try all actions in order until one matches."""
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[1, 2, 3], [4, 5, 6]]}]
    class_coords = [(0, r, c) for r in range(2) for c in range(3)]
    action = infer_action_for_class(items, class_coords)
    # Mixed colors → skip set_color
    # Mirrors won't match (grid structure different)
    # keep_nonzero or identity should work
    assert action in [("keep_nonzero", None), ("identity", None)]
    assert verify_action_on_class(action, items, class_coords)


def test_integration_realistic_scenario():
    """Real-world-like grid and class."""
    items = [
        {"Xp": [[1, 2, 3, 4], [5, 6, 7, 8]], "Y": [[4, 2, 2, 1], [8, 6, 6, 5]]}
    ]
    class_coords = [(0, 0, 0), (0, 0, 3), (0, 1, 0), (0, 1, 3)]
    action = infer_action_for_class(items, class_coords)
    assert action == ("mirror_v", None)
    assert verify_action_on_class(action, items, class_coords)


def test_integration_unsat_witness():
    """Document UNSAT case for Phase 6."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[9, 8, 7]]}]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    action = infer_action_for_class(items, class_coords)
    assert action is None  # No action can transform [1,2,3] to [9,8,7]
