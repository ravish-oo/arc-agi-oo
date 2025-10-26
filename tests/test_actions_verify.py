"""
Test suite for action verifier (P5-03).

Categories:
1. set_color Tests — Pass/fail scenarios
2. mirror_h Tests — Pass/fail scenarios
3. mirror_v Tests — Pass/fail scenarios
4. keep_nonzero Tests — Pass/fail scenarios
5. identity Tests — Pass/fail scenarios
6. Edge Cases — Empty coords, partial trains, duplicates
7. Validation Tests — Ragged grids, OOB coords, bad action tuples
8. Multi-Train Tests — Mixed satisfaction, multiple trains
9. Determinism Tests — Shuffled coords

Coverage Target: ≥35 tests
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.action_inference import verify_action_on_class


# ============================================================================
# Category 1: set_color Tests (6 tests)
# ============================================================================


def test_set_color_pass_single_train():
    """set_color matches when all class pixels in Y equal the color."""
    action = ("set_color", 5)
    items = [{"Xp": [[1, 2, 3]], "Y": [[5, 2, 3]]}]
    class_coords = [(0, 0, 0)]  # Train 0, pixel (0, 0)
    assert verify_action_on_class(action, items, class_coords) is True


def test_set_color_fail_mismatch():
    """set_color fails when Y pixel doesn't match color."""
    action = ("set_color", 5)
    items = [{"Xp": [[1, 2]], "Y": [[3, 2]]}]  # Y[0][0] = 3, not 5
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is False


def test_set_color_pass_multiple_coords():
    """set_color matches on multiple coords."""
    action = ("set_color", 7)
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[7, 2, 7], [4, 5, 6]]}]
    class_coords = [(0, 0, 0), (0, 0, 2)]  # Two coords
    assert verify_action_on_class(action, items, class_coords) is True


def test_set_color_fail_one_mismatch():
    """set_color fails if even one pixel mismatches."""
    action = ("set_color", 7)
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[7, 2, 5], [4, 5, 6]]}]
    # First coord matches (7), second doesn't (5 != 7)
    class_coords = [(0, 0, 0), (0, 0, 2)]
    assert verify_action_on_class(action, items, class_coords) is False


def test_set_color_pass_multi_train():
    """set_color matches across multiple trains."""
    action = ("set_color", 3)
    items = [
        {"Xp": [[1, 2]], "Y": [[3, 2]]},
        {"Xp": [[4, 5]], "Y": [[4, 3]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 1)]  # Different trains
    assert verify_action_on_class(action, items, class_coords) is True


def test_set_color_fail_one_train_fails():
    """set_color fails if any train fails."""
    action = ("set_color", 3)
    items = [
        {"Xp": [[1, 2]], "Y": [[3, 2]]},  # Train 0 matches
        {"Xp": [[4, 5]], "Y": [[4, 7]]}   # Train 1 doesn't match (7 != 3)
    ]
    class_coords = [(0, 0, 0), (1, 0, 1)]
    assert verify_action_on_class(action, items, class_coords) is False


# ============================================================================
# Category 2: mirror_h Tests (4 tests)
# ============================================================================


def test_mirror_h_pass():
    """mirror_h matches when Y equals mirrored Xp."""
    action = ("mirror_h", None)
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [3, 4]]}]
    # Top-left (0,0) mirrors to bottom-left: Xp[1][0] = 3
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_mirror_h_fail():
    """mirror_h fails when Y doesn't match."""
    action = ("mirror_h", None)
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[5, 2], [3, 4]]}]
    # Top-left should mirror to 3, but Y has 5
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is False


def test_mirror_h_pass_two_sided():
    """mirror_h with both mirror pairs (GLUE safety verification)."""
    action = ("mirror_h", None)
    items = [{"Xp": [[1, 2], [3, 4], [5, 6]], "Y": [[5, 2], [3, 4], [1, 6]]}]
    # Row 0 ↔ Row 2: (0,0)→5, (2,0)→1
    class_coords = [(0, 0, 0), (0, 2, 0)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_mirror_h_fail_two_sided():
    """mirror_h fails if one side incorrect."""
    action = ("mirror_h", None)
    items = [{"Xp": [[1, 2], [3, 4], [5, 6]], "Y": [[5, 2], [3, 4], [999, 6]]}]
    # Row 2 should be 1, but is 999
    class_coords = [(0, 0, 0), (0, 2, 0)]
    assert verify_action_on_class(action, items, class_coords) is False


# ============================================================================
# Category 3: mirror_v Tests (4 tests)
# ============================================================================


def test_mirror_v_pass():
    """mirror_v matches when Y equals mirrored Xp."""
    action = ("mirror_v", None)
    items = [{"Xp": [[1, 2, 3]], "Y": [[3, 2, 3]]}]
    # Left (0,0) mirrors to right: Xp[0][2] = 3
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_mirror_v_fail():
    """mirror_v fails when Y doesn't match."""
    action = ("mirror_v", None)
    items = [{"Xp": [[1, 2, 3]], "Y": [[999, 2, 3]]}]
    # Should be 3, but Y has 999
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is False


def test_mirror_v_pass_two_sided():
    """mirror_v with both mirror pairs."""
    action = ("mirror_v", None)
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[3, 2, 1], [4, 5, 6]]}]
    # Col 0 ↔ Col 2: (0,0)→3, (0,2)→1
    class_coords = [(0, 0, 0), (0, 0, 2)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_mirror_v_fail_two_sided():
    """mirror_v fails if one side incorrect."""
    action = ("mirror_v", None)
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[3, 2, 999], [4, 5, 6]]}]
    # Col 2 should be 1, but is 999
    class_coords = [(0, 0, 0), (0, 0, 2)]
    assert verify_action_on_class(action, items, class_coords) is False


# ============================================================================
# Category 4: keep_nonzero Tests (3 tests)
# ============================================================================


def test_keep_nonzero_pass():
    """keep_nonzero matches when Y preserves nonzero, keeps zero."""
    action = ("keep_nonzero", None)
    items = [{"Xp": [[1, 0, 3]], "Y": [[1, 0, 3]]}]
    # All pixels match (identity for this case)
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_keep_nonzero_fail():
    """keep_nonzero fails when Y changes a value."""
    action = ("keep_nonzero", None)
    items = [{"Xp": [[1, 0, 3]], "Y": [[1, 0, 999]]}]
    # Pixel (0,2) should be 3, but is 999
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    assert verify_action_on_class(action, items, class_coords) is False


def test_keep_nonzero_pass_zeros():
    """keep_nonzero matches when zeros stay zero."""
    action = ("keep_nonzero", None)
    items = [{"Xp": [[0, 0]], "Y": [[0, 0]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]
    assert verify_action_on_class(action, items, class_coords) is True


# ============================================================================
# Category 5: identity Tests (3 tests)
# ============================================================================


def test_identity_pass():
    """identity matches when Y equals Xp on class pixels."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2, 3]], "Y": [[1, 2, 3]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_identity_fail():
    """identity fails when Y differs from Xp."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2, 3]], "Y": [[1, 999, 3]]}]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    assert verify_action_on_class(action, items, class_coords) is False


def test_identity_pass_partial():
    """identity matches on partial pixels."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2, 3]], "Y": [[999, 2, 888]]}]
    # Only check (0, 1), which matches
    class_coords = [(0, 0, 1)]
    assert verify_action_on_class(action, items, class_coords) is True


# ============================================================================
# Category 6: Edge Cases (6 tests)
# ============================================================================


def test_empty_class_coords_vacuous():
    """Empty class_coords returns True (vacuous)."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2]], "Y": [[3, 4]]}]
    class_coords = []
    assert verify_action_on_class(action, items, class_coords) is True


def test_partial_train_coverage():
    """Some trains have no coords (vacuous for those trains)."""
    action = ("set_color", 5)
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 2]]},  # Train 0 has coords
        {"Xp": [[3, 4]], "Y": [[3, 4]]}   # Train 1 has no coords
    ]
    # Only train 0 has coords
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_duplicate_coords():
    """Duplicate coords handled deterministically."""
    action = ("set_color", 7)
    items = [{"Xp": [[1, 2]], "Y": [[7, 2]]}]
    # Duplicate (0,0,0) three times
    class_coords = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_no_items_with_coords():
    """Empty items list with coords raises error."""
    action = ("identity", None)
    items = []
    class_coords = [(0, 0, 0)]  # References train 0
    with pytest.raises(ValueError, match="out of range"):
        verify_action_on_class(action, items, class_coords)


def test_single_pixel_grid():
    """1×1 grid."""
    action = ("set_color", 9)
    items = [{"Xp": [[5]], "Y": [[9]]}]
    class_coords = [(0, 0, 0)]
    assert verify_action_on_class(action, items, class_coords) is True


def test_large_multi_train():
    """Multiple trains with multiple coords each."""
    action = ("set_color", 0)
    items = [
        {"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[0, 2, 0], [4, 5, 6]]},
        {"Xp": [[7, 8], [9, 1]], "Y": [[7, 8], [0, 1]]}
    ]
    class_coords = [(0, 0, 0), (0, 0, 2), (1, 1, 0)]
    assert verify_action_on_class(action, items, class_coords) is True


# ============================================================================
# Category 7: Validation Tests (8 tests)
# ============================================================================


def test_invalid_action_tuple_not_tuple():
    """Action must be a tuple."""
    with pytest.raises(ValueError, match="must be a 2-tuple"):
        verify_action_on_class("not_a_tuple", [], [])


def test_invalid_action_tuple_wrong_length():
    """Action tuple must have exactly 2 elements."""
    with pytest.raises(ValueError, match="must be a 2-tuple"):
        verify_action_on_class(("set_color",), [], [])


def test_invalid_action_name():
    """Action name must be in valid set."""
    with pytest.raises(ValueError, match="Invalid action name"):
        verify_action_on_class(("invalid_action", None), [], [])


def test_set_color_missing_param():
    """set_color requires a color parameter."""
    with pytest.raises(ValueError, match="requires a color parameter"):
        verify_action_on_class(("set_color", None), [], [])


def test_set_color_invalid_color_negative():
    """set_color color must be in [0..9]."""
    with pytest.raises(ValueError, match="must be int in"):
        verify_action_on_class(("set_color", -1), [], [])


def test_set_color_invalid_color_too_large():
    """set_color color must be in [0..9]."""
    with pytest.raises(ValueError, match="must be int in"):
        verify_action_on_class(("set_color", 10), [], [])


def test_mirror_h_param_must_be_none():
    """mirror_h requires param=None."""
    with pytest.raises(ValueError, match="requires param=None"):
        verify_action_on_class(("mirror_h", 5), [], [])


def test_ragged_xp():
    """Ragged Xp raises ValueError."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2], [3]], "Y": [[1, 2], [3, 4]]}]
    class_coords = [(0, 0, 0)]
    with pytest.raises(ValueError, match="rectangular"):
        verify_action_on_class(action, items, class_coords)


def test_ragged_y():
    """Ragged Y raises ValueError."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[1, 2], [3]]}]
    class_coords = [(0, 0, 0)]
    with pytest.raises(ValueError, match="rectangular"):
        verify_action_on_class(action, items, class_coords)


def test_shape_mismatch():
    """Xp and Y must have same shape."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2]], "Y": [[1, 2, 3]]}]
    class_coords = [(0, 0, 0)]
    with pytest.raises(ValueError, match="same shape"):
        verify_action_on_class(action, items, class_coords)


def test_coord_out_of_bounds():
    """Coord out of bounds raises ValueError."""
    action = ("identity", None)
    items = [{"Xp": [[1, 2]], "Y": [[1, 2]]}]
    class_coords = [(0, 0, 5)]  # Col 5 doesn't exist
    with pytest.raises(ValueError, match="out of bounds"):
        verify_action_on_class(action, items, class_coords)


def test_train_index_out_of_range():
    """Train index out of range raises ValueError."""
    action = ("identity", None)
    items = [{"Xp": [[1]], "Y": [[1]]}]
    class_coords = [(5, 0, 0)]  # Train 5 doesn't exist
    with pytest.raises(ValueError, match="out of range"):
        verify_action_on_class(action, items, class_coords)


# ============================================================================
# Category 8: Determinism Tests (2 tests)
# ============================================================================


def test_determinism_repeated_calls():
    """Repeated calls produce same result."""
    action = ("set_color", 3)
    items = [{"Xp": [[1, 2]], "Y": [[3, 2]]}]
    class_coords = [(0, 0, 0)]

    result1 = verify_action_on_class(action, items, class_coords)
    result2 = verify_action_on_class(action, items, class_coords)
    assert result1 == result2


def test_determinism_shuffled_coords():
    """Shuffled coords produce same result (sorted internally)."""
    action = ("set_color", 7)
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[7, 2, 7], [4, 5, 6]]}]

    # Sorted order
    coords_sorted = [(0, 0, 0), (0, 0, 2)]
    # Reverse order
    coords_reversed = [(0, 0, 2), (0, 0, 0)]

    result1 = verify_action_on_class(action, items, coords_sorted)
    result2 = verify_action_on_class(action, items, coords_reversed)
    assert result1 == result2 == True
