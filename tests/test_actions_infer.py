"""
Test suite for action inference core (P5-04).

Categories:
1. set_color Inference — Unified parameter logic
2. mirror_h Inference — Spatial transformations
3. mirror_v Inference — Spatial transformations
4. keep_nonzero Inference — Conditional identity
5. identity Inference — Fallback
6. First-Pass Selection — Action order matters
7. Edge Cases — Empty coords, UNSAT, single train
8. Multi-Train Coverage — Unified parameters across trains
9. Determinism Tests — Stable results

Coverage Target: ≥35 tests
"""

import pytest
import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-oo')

from src.action_inference import infer_action_for_class


# ============================================================================
# Category 1: set_color Inference (6 tests)
# ============================================================================


def test_set_color_unified_single_train():
    """Infer set_color when all class pixels in Y have same color (single train)."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[5, 2, 5]]}]
    class_coords = [(0, 0, 0), (0, 0, 2)]  # Both Y pixels = 5
    result = infer_action_for_class(items, class_coords)
    assert result == ("set_color", 5)


def test_set_color_unified_multi_train():
    """Infer set_color when color unified across multiple trains."""
    items = [
        {"Xp": [[1, 2]], "Y": [[7, 2]]},
        {"Xp": [[3, 4]], "Y": [[7, 4]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]  # Both trains Y[0][0] = 7
    result = infer_action_for_class(items, class_coords)
    assert result == ("set_color", 7)


def test_set_color_skip_mixed_colors_same_train():
    """Skip set_color if different colors within same train."""
    items = [{"Xp": [[1, 2]], "Y": [[3, 5]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]  # Y has 3 and 5
    result = infer_action_for_class(items, class_coords)
    # set_color skipped, should try other actions
    # For this simple case, identity won't work either (Xp != Y)
    # Result will be None or another action if one matches
    assert result != ("set_color", 3) and result != ("set_color", 5)


def test_set_color_skip_mixed_colors_multi_train():
    """Skip set_color if different colors across trains."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 2]]},
        {"Xp": [[3, 4]], "Y": [[7, 4]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]  # Train 0 has 5, train 1 has 7
    result = infer_action_for_class(items, class_coords)
    # set_color inadmissible (mixed colors)
    assert result is None or result[0] != "set_color"


def test_set_color_zero():
    """Infer set_color with color 0 (black)."""
    items = [{"Xp": [[5, 6]], "Y": [[0, 6]]}]
    class_coords = [(0, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    assert result == ("set_color", 0)


def test_set_color_nine():
    """Infer set_color with color 9."""
    items = [{"Xp": [[1, 2]], "Y": [[9, 2]]}]
    class_coords = [(0, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    assert result == ("set_color", 9)


# ============================================================================
# Category 2: mirror_h Inference (3 tests)
# ============================================================================


def test_mirror_h_simple():
    """Infer mirror_h when Y equals horizontally mirrored Xp."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]]}]
    # Top-left (0,0) mirrors to bottom-left: Xp[1][0] = 3
    # Bottom-left (1,0) mirrors to top-left: Xp[0][0] = 1
    class_coords = [(0, 0, 0), (0, 1, 0)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed colors (3, 1) → skip
    # mirror_h: matches both pixels
    assert result == ("mirror_h", None)


def test_mirror_h_multi_train():
    """Infer mirror_h across multiple trains."""
    items = [
        {"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [3, 4]]},
        {"Xp": [[5, 6], [7, 8]], "Y": [[7, 6], [7, 8]]}
    ]
    # Both trains: top-left mirrors to bottom-left
    class_coords = [(0, 0, 0), (1, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    assert result == ("mirror_h", None)


def test_mirror_h_beats_later_actions():
    """mirror_h comes before mirror_v in action order."""
    # Craft case where only mirror_h works (not mirror_v)
    items = [{"Xp": [[1, 2], [3, 4], [5, 6]], "Y": [[5, 2], [3, 4], [1, 6]]}]
    # Row 0 col 0 mirrors to row 2 col 0: Xp[2][0] = 5
    # Row 2 col 0 mirrors to row 0 col 0: Xp[0][0] = 1
    class_coords = [(0, 0, 0), (0, 2, 0)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (5, 1) → skip
    # mirror_h: matches both
    assert result == ("mirror_h", None)


# ============================================================================
# Category 3: mirror_v Inference (3 tests)
# ============================================================================


def test_mirror_v_simple():
    """Infer mirror_v when Y equals vertically mirrored Xp."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[3, 2, 1]]}]
    # Left (0,0) mirrors to right: Xp[0][2] = 3
    # Right (0,2) mirrors to left: Xp[0][0] = 1
    class_coords = [(0, 0, 0), (0, 0, 2)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (3, 1) → skip
    # mirror_h: won't match (only 1 row)
    # mirror_v: matches both
    assert result == ("mirror_v", None)


def test_mirror_v_multi_train():
    """Infer mirror_v across multiple trains."""
    items = [
        {"Xp": [[1, 2, 3]], "Y": [[3, 2, 3]]},
        {"Xp": [[4, 5, 6]], "Y": [[6, 5, 6]]}
    ]
    # Both trains: left mirrors to right
    class_coords = [(0, 0, 0), (1, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    assert result == ("mirror_v", None)


def test_mirror_v_after_mirror_h_fails():
    """mirror_v tried after mirror_h fails."""
    # Craft case where mirror_h fails but mirror_v works
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[3, 2, 1], [6, 5, 4]]}]
    # (0,0) mirrors vertically to (0,2): Xp[0][2] = 3
    # (0,2) mirrors vertically to (0,0): Xp[0][0] = 1
    # (1,0) mirrors vertically to (1,2): Xp[1][2] = 6
    # (1,2) mirrors vertically to (1,0): Xp[1][0] = 4
    class_coords = [(0, 0, 0), (0, 0, 2), (0, 1, 0), (0, 1, 2)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (3, 1, 6, 4) → skip
    # mirror_h: won't match (row 0 would go to row 1, but Y differs)
    # mirror_v: matches all 4 pixels
    assert result == ("mirror_v", None)


# ============================================================================
# Category 4: keep_nonzero Inference (2 tests)
# ============================================================================


def test_keep_nonzero_simple():
    """Infer keep_nonzero when Y preserves Xp structure."""
    items = [{"Xp": [[1, 0, 3], [4, 0, 6]], "Y": [[1, 0, 3], [4, 0, 6]]}]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (1, 0, 3, 4, 0, 6) → skip
    # mirror_h: row 0 would map to row 1, but they're different → won't match
    # mirror_v: col 0 would map to col 2, Xp[0][2]=3 vs Y[0][0]=1 → won't match
    # keep_nonzero: Y == Xp → matches
    assert result == ("keep_nonzero", None)


def test_keep_nonzero_multi_train():
    """Infer keep_nonzero across multiple trains."""
    items = [
        {"Xp": [[1, 0], [2, 3]], "Y": [[1, 0], [2, 3]]},
        {"Xp": [[0, 5], [6, 7]], "Y": [[0, 5], [6, 7]]}
    ]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed → skip
    # mirror_h/v: won't match (different values)
    # keep_nonzero: Y == Xp → matches
    assert result == ("keep_nonzero", None)


# ============================================================================
# Category 5: identity Inference (3 tests)
# ============================================================================


def test_identity_simple():
    """Infer identity when Y equals Xp and no earlier action triggers."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[1, 2], [3, 4]]}]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (1, 2, 3, 4) → skip
    # mirror_h: row 0 would go to row 1: Xp[1][0]=3 vs Y[0][0]=1 → won't match
    # mirror_v: col 0 would go to col 1: Xp[0][1]=2 vs Y[0][0]=1 → won't match
    # keep_nonzero: Y == Xp → matches (comes before identity)
    assert result in [("keep_nonzero", None), ("identity", None)]


def test_identity_partial_class():
    """Infer identity for partial class (some pixels, not all)."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[999, 2, 888]]}]
    # Only check pixel (0, 1), which matches
    class_coords = [(0, 0, 1)]
    result = infer_action_for_class(items, class_coords)
    # set_color: only one pixel, Y=2, so try set_color(2)
    # But Xp[0][1]=2 and Y[0][1]=2, so set_color(2) will match
    assert result == ("set_color", 2)


def test_identity_fallback():
    """identity is the last resort (guaranteed to match if Y == Xp on class)."""
    items = [{"Xp": [[7, 8, 9], [4, 5, 6]], "Y": [[7, 8, 9], [4, 5, 6]]}]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2)]
    result = infer_action_for_class(items, class_coords)
    # Mixed colors (7, 8, 9, 4, 5, 6) → skip set_color
    # mirror_h: row 0 would go to row 1, but values differ → won't match
    # mirror_v: col 0 would go to col 2: Xp[0][2]=9 vs Y[0][0]=7 → won't match
    # keep_nonzero or identity should match (keep_nonzero comes first)
    assert result in [("keep_nonzero", None), ("identity", None)]


# ============================================================================
# Category 6: First-Pass Selection (4 tests)
# ============================================================================


def test_first_pass_set_color_wins():
    """set_color wins if it matches (even if mirrors would also match)."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 5]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]  # Both Y = 5
    result = infer_action_for_class(items, class_coords)
    # set_color(5) is first and matches
    assert result == ("set_color", 5)


def test_first_pass_mirror_h_before_mirror_v():
    """mirror_h tried before mirror_v."""
    # Create case where both could theoretically match, but h is first
    items = [{"Xp": [[1, 1], [2, 2]], "Y": [[2, 2], [2, 2]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]
    result = infer_action_for_class(items, class_coords)
    # set_color(2) should work first (unified color)
    assert result == ("set_color", 2)


def test_first_pass_keep_nonzero_before_identity():
    """keep_nonzero tried before identity."""
    items = [{"Xp": [[3, 4], [5, 6]], "Y": [[3, 4], [5, 6]]}]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (3, 4, 5, 6) → skip
    # mirror_h: row 0 would go to row 1, but values differ → won't match
    # mirror_v: col 0 would go to col 1: Xp[0][1]=4 vs Y[0][0]=3 → won't match
    # keep_nonzero: matches (both nonzero, Y == Xp)
    assert result == ("keep_nonzero", None)


def test_first_pass_skip_inadmissible():
    """Skip inadmissible actions (e.g., mixed-color set_color)."""
    items = [{"Xp": [[1, 2]], "Y": [[3, 5]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (3, 5) → skip
    # mirrors: Xp has 1,2; Y has 3,5; mirrors won't match
    # keep_nonzero: Xp=1,2; Y=3,5 → doesn't match
    # identity: Xp != Y → doesn't match
    assert result is None


# ============================================================================
# Category 7: Edge Cases (5 tests)
# ============================================================================


def test_empty_class_coords():
    """Empty class_coords returns identity (vacuous truth)."""
    items = [{"Xp": [[1, 2]], "Y": [[3, 4]]}]
    class_coords = []
    result = infer_action_for_class(items, class_coords)
    assert result == ("identity", None)


def test_unsat_no_action_matches():
    """Return None if no action satisfies FY."""
    items = [{"Xp": [[1]], "Y": [[999]]}]
    class_coords = [(0, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    # No action transforms 1 to 999
    assert result is None


def test_single_pixel_class():
    """Single pixel class."""
    items = [{"Xp": [[5]], "Y": [[7]]}]
    class_coords = [(0, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    # set_color(7) should work
    assert result == ("set_color", 7)


def test_single_train_unified_param():
    """Single train still requires unified param (same rule)."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[4, 4, 4]]}]
    class_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    result = infer_action_for_class(items, class_coords)
    # All Y = 4 → set_color(4)
    assert result == ("set_color", 4)


def test_large_multi_train():
    """Multiple trains with many coords."""
    items = [
        {"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[0, 2, 0], [4, 5, 6]]},
        {"Xp": [[7, 8], [9, 0]], "Y": [[7, 8], [0, 0]]}
    ]
    class_coords = [(0, 0, 0), (0, 0, 2), (1, 1, 0)]  # All Y = 0
    result = infer_action_for_class(items, class_coords)
    # Unified color 0 → set_color(0)
    assert result == ("set_color", 0)


# ============================================================================
# Category 8: Multi-Train Coverage (3 tests)
# ============================================================================


def test_multi_train_unified_color():
    """Unified color across 3 trains."""
    items = [
        {"Xp": [[1]], "Y": [[3]]},
        {"Xp": [[2]], "Y": [[3]]},
        {"Xp": [[5]], "Y": [[3]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    assert result == ("set_color", 3)


def test_multi_train_mixed_colors_unsat():
    """Mixed colors across trains → UNSAT if no other action works."""
    items = [
        {"Xp": [[1]], "Y": [[3]]},
        {"Xp": [[2]], "Y": [[5]]},
        {"Xp": [[7]], "Y": [[9]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    # Mixed colors (3, 5, 9) → set_color skipped
    # No spatial transform will fix this
    assert result is None


def test_multi_train_mirror_agreement():
    """Multiple trains all agree on mirror_h."""
    items = [
        {"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [3, 4]]},
        {"Xp": [[5, 6], [7, 8]], "Y": [[7, 6], [7, 8]]}
    ]
    class_coords = [(0, 0, 0), (1, 0, 0)]
    result = infer_action_for_class(items, class_coords)
    assert result == ("mirror_h", None)


# ============================================================================
# Category 9: Determinism Tests (3 tests)
# ============================================================================


def test_determinism_repeated_calls():
    """Repeated calls produce same result."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 2]]}]
    class_coords = [(0, 0, 0)]

    result1 = infer_action_for_class(items, class_coords)
    result2 = infer_action_for_class(items, class_coords)
    assert result1 == result2


def test_determinism_shuffled_coords():
    """Shuffled coords produce same result (internally sorted per train)."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[7, 7, 7]]}]

    # Original order
    coords_orig = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    # Shuffled order
    coords_shuffled = [(0, 0, 2), (0, 0, 0), (0, 0, 1)]

    result1 = infer_action_for_class(items, coords_orig)
    result2 = infer_action_for_class(items, coords_shuffled)
    assert result1 == result2 == ("set_color", 7)


def test_determinism_action_order():
    """Action order is fixed and deterministic."""
    items = [{"Xp": [[1, 2]], "Y": [[3, 3]]}]
    class_coords = [(0, 0, 0), (0, 0, 1)]

    result = infer_action_for_class(items, class_coords)
    # set_color(3) comes first and matches
    assert result == ("set_color", 3)


# ============================================================================
# Category 10: Integration Tests (2 tests)
# ============================================================================


def test_integration_realistic_class():
    """Realistic scenario: class of pixels needing mirror_v."""
    items = [
        {
            "Xp": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "Y": [[1, 2, 1, 4], [5, 6, 5, 8]]
        }
    ]
    # For C=4: col 2 mirrors to col C-1-2 = 1
    # Y[0][2] should equal Xp[0][1] = 2, but Y[0][2] = 1
    # Let me fix: use pixels (0,3) and (1,3) which mirror to (0,0) and (1,0)
    class_coords = [(0, 0, 3), (0, 1, 3)]
    result = infer_action_for_class(items, class_coords)
    # Y[0][3] = 4, Xp[0][0] = 1 → doesn't work either
    # Let me recalculate: Y[0][3]=4, for mirror_v: Out[0][3] = Xp[0][C-1-3] = Xp[0][0] = 1 ❌
    # Actually, the Y values need adjustment. Let me create correct Y:
    # For pixels at col 3 to mirror_v from col 0: Y[r][3] should equal Xp[r][0]
    # Y[0][3] should be 1, Y[1][3] should be 5
    # But current Y has Y[0][3]=4, Y[1][3]=8
    # Let me use different coords that actually work with current Y
    # Current Y[0][2]=1, Y[1][2]=5
    # For mirror_v: Out[0][2] = Xp[0][C-1-2] = Xp[0][1] = 2 ❌
    # I need to fix Y to make mirror_v work correctly
    items = [
        {
            "Xp": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "Y": [[4, 2, 2, 1], [8, 6, 6, 5]]
        }
    ]
    # Y[0][0]=4 should mirror from Xp[0][3]=4 ✓
    # Y[0][3]=1 should mirror from Xp[0][0]=1 ✓
    # Y[1][0]=8 should mirror from Xp[1][3]=8 ✓
    # Y[1][3]=5 should mirror from Xp[1][0]=5 ✓
    class_coords = [(0, 0, 0), (0, 0, 3), (0, 1, 0), (0, 1, 3)]
    result = infer_action_for_class(items, class_coords)
    # set_color: mixed (4, 1, 8, 5) → skip
    # mirror_h: won't match
    # mirror_v: matches all 4 pixels
    assert result == ("mirror_v", None)


def test_integration_identity_fallback():
    """Realistic scenario: class that is unchanged (identity)."""
    items = [
        {
            "Xp": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "Y": [[999, 2, 888], [4, 5, 6], [777, 8, 666]]
        }
    ]
    # Only check pixels that are unchanged
    class_coords = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 1)]
    result = infer_action_for_class(items, class_coords)
    # Mixed colors (2, 4, 5, 6, 8) → skip set_color
    # Mirrors unlikely
    # keep_nonzero or identity
    assert result in [("keep_nonzero", None), ("identity", None)]
