"""
Test suite for P6-04: verify_stitched_equality(items, classes, actions_by_cid)

21 tests across 5 categories:
- Basic functionality (7 tests)
- Edge cases (4 tests)
- Error propagation (3 tests)
- Purity and determinism (3 tests)
- Action integration (4 tests)
"""

import pytest
from src.glue import verify_stitched_equality


# ============================================================================
# CATEGORY 1: BASIC FUNCTIONALITY (7 tests)
# ============================================================================

def test_empty_items_returns_true():
    """Empty items list → vacuous truth."""
    result = verify_stitched_equality([], {}, {})
    assert result is True


def test_no_classes_xp_equals_y_returns_true():
    """No classes, Xp == Y → True."""
    items = [
        {"Xp": [[1, 2], [3, 4]], "Y": [[1, 2], [3, 4]], "feats": {}, "residual": [[None, None], [None, None]]},
    ]
    result = verify_stitched_equality(items, {}, {})
    assert result is True


def test_no_classes_xp_not_equal_y_returns_false():
    """No classes, Xp != Y → False."""
    items = [
        {"Xp": [[1, 2]], "Y": [[1, 3]], "feats": {}, "residual": [[None, 3]]},
    ]
    result = verify_stitched_equality(items, {}, {})
    assert result is False


def test_single_class_set_color_exact_match_returns_true():
    """Single class, set_color matches Y → True."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 5]], "feats": {}, "residual": [[5, 5]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions = {0: ("set_color", 5)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True


def test_single_class_set_color_mismatch_returns_false():
    """Single class, stitched != Y → False."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 4]], "feats": {}, "residual": [[5, 4]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions = {0: ("set_color", 5)}  # Sets both to 5, but Y has [5, 4]

    result = verify_stitched_equality(items, classes, actions)
    assert result is False


def test_multiple_classes_all_match_returns_true():
    """Multiple classes, all stitched correctly → True."""
    items = [
        {"Xp": [[1, 2, 3]], "Y": [[5, 2, 7]], "feats": {}, "residual": [[5, None, 7]]},
    ]
    classes = {
        0: [(0, 0, 0)],  # Pixel (0,0)
        1: [(0, 0, 2)],  # Pixel (0,2)
    }
    actions = {
        0: ("set_color", 5),
        1: ("set_color", 7),
    }

    result = verify_stitched_equality(items, classes, actions)
    assert result is True


def test_two_trains_both_match_returns_true():
    """Two trains, both stitched correctly → True."""
    items = [
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
    ]
    classes = {0: [(0, 0, 0), (1, 0, 0)]}  # Both trains
    actions = {0: ("set_color", 1)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True


# ============================================================================
# CATEGORY 2: EDGE CASES (4 tests)
# ============================================================================

def test_two_trains_second_fails_returns_false():
    """Two trains, second doesn't match → False."""
    items = [
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
        {"Xp": [[0]], "Y": [[2]], "feats": {}, "residual": [[2]]},  # Different target
    ]
    classes = {0: [(0, 0, 0), (1, 0, 0)]}  # Both trains get same action
    actions = {0: ("set_color", 1)}  # Sets to 1, but train 1 wants 2

    result = verify_stitched_equality(items, classes, actions)
    assert result is False


def test_some_class_no_coords_for_train_still_returns_true():
    """Class has no coords for train 1, but verification still passes."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 0]], "feats": {}, "residual": [[5, None]]},
        {"Xp": [[0, 0]], "Y": [[0, 0]], "feats": {}, "residual": [[None, None]]},
    ]
    classes = {0: [(0, 0, 0)]}  # Only train 0
    actions = {0: ("set_color", 5)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True


def test_shape_mismatch_after_stitch_returns_false():
    """Shape mismatch between stitched and Y → False."""
    # This is actually impossible with correct implementation (stitcher preserves shape)
    # But we test that deep_eq catches shape mismatches
    items = [
        {"Xp": [[1, 2]], "Y": [[1, 2, 3]], "feats": {}, "residual": [[None, None, None]]},
    ]
    classes = {}
    actions = {}

    # This should raise ValueError from stitcher (shape mismatch in items)
    # But if it doesn't, deep_eq should return False
    result = verify_stitched_equality(items, classes, actions)
    assert result is False


def test_single_pixel_differs_returns_false():
    """Single pixel differs → False."""
    items = [
        {"Xp": [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "Y": [[1, 1, 1], [1, 0, 1], [1, 1, 1]], "feats": {}, "residual": [[None, None, None], [None, 0, None], [None, None, None]]},
    ]
    classes = {0: [(0, 1, 1)]}  # Only pixel (1,1)
    actions = {0: ("set_color", 0)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True

    # Now change Y slightly
    items_modified = [
        {"Xp": [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "Y": [[1, 1, 1], [1, 9, 1], [1, 1, 1]], "feats": {}, "residual": [[None, None, None], [None, 9, None], [None, None, None]]},
    ]

    result_modified = verify_stitched_equality(items_modified, classes, actions)
    assert result_modified is False


# ============================================================================
# CATEGORY 3: ERROR PROPAGATION (3 tests)
# ============================================================================

def test_missing_action_for_class_raises_valueerror():
    """Missing action for class_id → ValueError from stitcher."""
    items = [
        {"Xp": [[0]], "Y": [[5]], "feats": {}, "residual": [[5]]},
    ]
    classes = {0: [(0, 0, 0)]}
    actions = {}  # Missing action for class 0

    with pytest.raises(ValueError, match="Missing action for class_id 0"):
        verify_stitched_equality(items, classes, actions)


def test_ragged_grid_raises_valueerror():
    """Ragged grid → ValueError from stitcher."""
    items = [
        {"Xp": [[1, 2], [3]], "Y": [[1, 2], [3, 4]], "feats": {}, "residual": [[None, None], [None, None]]},
    ]
    classes = {}
    actions = {}

    with pytest.raises(ValueError, match="rectangular"):
        verify_stitched_equality(items, classes, actions)


def test_invalid_action_name_raises_valueerror():
    """Invalid action name → ValueError from stitcher."""
    items = [
        {"Xp": [[0]], "Y": [[5]], "feats": {}, "residual": [[5]]},
    ]
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("invalid_action", None)}

    with pytest.raises(ValueError, match="Unknown action"):
        verify_stitched_equality(items, classes, actions)


# ============================================================================
# CATEGORY 4: PURITY AND DETERMINISM (3 tests)
# ============================================================================

def test_purity_items_unchanged():
    """Items unchanged after verification."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]},
    ]
    items_orig = [
        {"Xp": [[1, 2]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions = {0: ("set_color", 5)}

    verify_stitched_equality(items, classes, actions)

    # Items unchanged
    assert items == items_orig


def test_determinism_same_inputs_same_result():
    """Same inputs → same boolean result."""
    items = [
        {"Xp": [[0, 1]], "Y": [[5, 5]], "feats": {}, "residual": [[5, 5]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions = {0: ("set_color", 5)}

    result1 = verify_stitched_equality(items, classes, actions)
    result2 = verify_stitched_equality(items, classes, actions)

    assert result1 == result2
    assert result1 is True


def test_stable_hash_across_runs():
    """Boolean result stable across multiple runs."""
    items = [
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
    ]
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    results = [verify_stitched_equality(items, classes, actions) for _ in range(10)]

    # All results identical
    assert all(r is True for r in results)


# ============================================================================
# CATEGORY 5: ACTION INTEGRATION (4 tests)
# ============================================================================

def test_mirror_h_action_verification():
    """Mirror_h action matches Y → True."""
    items = [
        {"Xp": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "Y": [[7, 2, 3], [4, 5, 6], [7, 8, 9]], "feats": {}, "residual": [[7, None, None], [None, None, None], [None, None, None]]},
    ]
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("mirror_h", None)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True


def test_mirror_v_action_verification():
    """Mirror_v action matches Y → True."""
    items = [
        {"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[3, 2, 3], [4, 5, 6]], "feats": {}, "residual": [[3, None, None], [None, None, None]]},
    ]
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("mirror_v", None)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True


def test_keep_nonzero_action_verification():
    """Keep_nonzero action matches Y → True."""
    items = [
        {"Xp": [[1, 0, 3]], "Y": [[1, 0, 3]], "feats": {}, "residual": [[None, None, None]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1), (0, 0, 2)]}
    actions = {0: ("keep_nonzero", None)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True


def test_identity_action_verification():
    """Identity action matches Y → True."""
    items = [
        {"Xp": [[1, 2], [3, 4]], "Y": [[1, 2], [3, 4]], "feats": {}, "residual": [[None, None], [None, None]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]}
    actions = {0: ("identity", None)}

    result = verify_stitched_equality(items, classes, actions)
    assert result is True
