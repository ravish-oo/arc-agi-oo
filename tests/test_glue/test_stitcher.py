"""
Test suite for P6-03: stitch_from_classes(items, classes, actions_by_cid)

30+ tests across 5 categories:
- Category A: Basic stitching (10 tests)
- Category B: GLUE frozen-base reads (8 tests)
- Category C: Purity and determinism (5 tests)
- Category D: Optional seam check (3 tests)
- Category E: Edge cases (4 tests)
"""

import pytest
import json
from src.glue import stitch_from_classes


# ============================================================================
# CATEGORY A: BASIC STITCHING (10 tests)
# ============================================================================

def test_empty_classes_returns_deep_copies():
    """Empty classes → stitched outputs == deep copies of Xp."""
    items = [
        {"Xp": [[1, 2]], "Y": [[1, 2]], "feats": {}, "residual": [[None, None]]},
        {"Xp": [[3, 4]], "Y": [[3, 4]], "feats": {}, "residual": [[None, None]]},
    ]
    classes = {}
    actions_by_cid = {}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    assert outs == [[[1, 2]], [[3, 4]]]
    # Verify no aliasing
    assert outs[0] is not items[0]["Xp"]
    assert outs[1] is not items[1]["Xp"]


def test_single_class_set_color():
    """Single class, set_color action."""
    items = [{"Xp": [[0, 0]], "Y": [[5, 5]], "feats": {}, "residual": [[5, 5]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}  # Both pixels
    actions_by_cid = {0: ("set_color", 5)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    assert outs == [[[5, 5]]]


def test_multi_class_disjoint_set_color():
    """Multiple classes, disjoint coords."""
    items = [{"Xp": [[1, 2, 3]], "Y": [[5, 2, 7]], "feats": {}, "residual": [[5, None, 7]]}]
    classes = {
        0: [(0, 0, 0)],  # Pixel (0,0)
        1: [(0, 0, 2)],  # Pixel (0,2)
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 7),
    }

    outs = stitch_from_classes(items, classes, actions_by_cid)

    assert outs == [[[5, 2, 7]]]


def test_identity_action_preserves_xp():
    """Identity action → Out == Xp."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[1, 2], [3, 4]], "feats": {}, "residual": [[None, None], [None, None]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]}
    actions_by_cid = {0: ("identity", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    assert outs == [[[1, 2], [3, 4]]]


def test_keep_nonzero_action():
    """Keep nonzero action."""
    items = [{"Xp": [[1, 0, 3]], "Y": [[1, 0, 3]], "feats": {}, "residual": [[None, None, None]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1), (0, 0, 2)]}
    actions_by_cid = {0: ("keep_nonzero", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # Nonzero preserved, zero stays zero
    assert outs == [[[1, 0, 3]]]


def test_multiple_trains_selective_coords():
    """Multiple trains, classes with varying coords per train."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 0]], "feats": {}, "residual": [[5, None]]},
        {"Xp": [[0, 0]], "Y": [[5, 7]], "feats": {}, "residual": [[5, 7]]},
    ]
    classes = {
        0: [(0, 0, 0), (1, 0, 0)],  # Class 0: pixel (0,0) in both trains
        1: [(1, 0, 1)],  # Class 1: pixel (0,1) in train 1 only
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 7),
    }

    outs = stitch_from_classes(items, classes, actions_by_cid)

    assert outs == [[[5, 0]], [[5, 7]]]


def test_class_order_deterministic():
    """Classes processed in ascending order."""
    items = [{"Xp": [[0, 0, 0]], "Y": [[5, 6, 7]], "feats": {}, "residual": [[5, 6, 7]]}]
    classes = {
        2: [(0, 0, 2)],  # Class 2
        0: [(0, 0, 0)],  # Class 0
        1: [(0, 0, 1)],  # Class 1
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 6),
        2: ("set_color", 7),
    }

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # Should process in order 0, 1, 2
    assert outs == [[[5, 6, 7]]]


def test_no_coords_for_train_skips_gracefully():
    """Class has coords only for train 0, not train 1."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 0]], "feats": {}, "residual": [[5, None]]},
        {"Xp": [[0, 0]], "Y": [[0, 0]], "feats": {}, "residual": [[None, None]]},
    ]
    classes = {0: [(0, 0, 0)]}  # Only train 0
    actions_by_cid = {0: ("set_color", 5)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # Train 0 stitched, train 1 unchanged
    assert outs == [[[5, 0]], [[0, 0]]]


def test_missing_action_raises_valueerror():
    """Missing action for class_id raises ValueError."""
    items = [{"Xp": [[0]], "Y": [[5]], "feats": {}, "residual": [[5]]}]
    classes = {0: [(0, 0, 0)]}
    actions_by_cid = {}  # Missing action for class 0

    with pytest.raises(ValueError, match="Missing action for class_id 0"):
        stitch_from_classes(items, classes, actions_by_cid)


def test_ragged_xp_raises_valueerror():
    """Ragged Xp raises ValueError."""
    items = [{"Xp": [[1, 2], [3]], "Y": [[1, 2], [3, 4]], "feats": {}, "residual": [[None, None], [None, None]]}]
    classes = {}
    actions_by_cid = {}

    with pytest.raises(ValueError, match="rectangular"):
        stitch_from_classes(items, classes, actions_by_cid)


# ============================================================================
# CATEGORY B: GLUE FROZEN-BASE READS (8 tests)
# ============================================================================

def test_mirror_h_reads_from_frozen_xp():
    """Mirror_h reads from frozen Xp, not Out."""
    items = [{"Xp": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "Y": [[7, 2, 3], [4, 5, 6], [7, 8, 9]], "feats": {}, "residual": [[7, None, None], [None, None, None], [None, None, None]]}]
    classes = {0: [(0, 0, 0)]}  # Only pixel (0,0)
    actions_by_cid = {0: ("mirror_h", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # mirror_h: Out[0][0] = Xp[R-1-0][0] = Xp[2][0] = 7
    assert outs == [[[7, 2, 3], [4, 5, 6], [7, 8, 9]]]


def test_mirror_v_reads_from_frozen_xp():
    """Mirror_v reads from frozen Xp, not Out."""
    items = [{"Xp": [[1, 2, 3], [4, 5, 6]], "Y": [[1, 2, 3], [4, 5, 6]], "feats": {}, "residual": [[None, None, 3], [None, None, None]]}]
    classes = {0: [(0, 0, 0)]}  # Only pixel (0,0)
    actions_by_cid = {0: ("mirror_v", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # mirror_v: Out[0][0] = Xp[0][C-1-0] = Xp[0][2] = 3
    assert outs == [[[3, 2, 3], [4, 5, 6]]]


def test_two_sided_mirror_h_glue_correctness():
    """CRITICAL: Both (r,c) and (R-1-r,c) read from frozen Xp, NOT from Out."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]], "feats": {}, "residual": [[3, None], [1, None]]}]
    classes = {0: [(0, 0, 0), (0, 1, 0)]}  # Both (0,0) and (1,0)
    actions_by_cid = {0: ("mirror_h", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # R = 2
    # Out[0][0] = Xp[2-1-0][0] = Xp[1][0] = 3
    # Out[1][0] = Xp[2-1-1][0] = Xp[0][0] = 1  (reads from Xp, NOT from Out[0][0])
    assert outs == [[[3, 2], [1, 4]]]


def test_two_sided_mirror_v_glue_correctness():
    """CRITICAL: Both (r,c) and (r,C-1-c) read from frozen Xp, NOT from Out."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[2, 1], [4, 3]], "feats": {}, "residual": [[2, 1], [4, 3]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}  # Both (0,0) and (0,1)
    actions_by_cid = {0: ("mirror_v", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # C = 2
    # Out[0][0] = Xp[0][2-1-0] = Xp[0][1] = 2
    # Out[0][1] = Xp[0][2-1-1] = Xp[0][0] = 1  (reads from Xp, NOT from Out[0][0])
    assert outs == [[[2, 1], [3, 4]]]


def test_multi_class_mirrors_no_chaining():
    """Multiple classes with mirrors don't chain reads."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]], "feats": {}, "residual": [[3, None], [1, None]]}]
    classes = {
        0: [(0, 0, 0)],  # Class 0: pixel (0,0)
        1: [(0, 1, 0)],  # Class 1: pixel (1,0)
    }
    actions_by_cid = {
        0: ("mirror_h", None),
        1: ("mirror_h", None),
    }

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # Both read from frozen Xp
    # Out[0][0] = Xp[1][0] = 3
    # Out[1][0] = Xp[0][0] = 1  (NOT Out[0][0])
    assert outs == [[[3, 2], [1, 4]]]


def test_1x1_grid_mirror_self():
    """1x1 grid mirrors to itself."""
    items = [{"Xp": [[5]], "Y": [[5]], "feats": {}, "residual": [[None]]}]
    classes = {0: [(0, 0, 0)]}
    actions_by_cid = {0: ("mirror_h", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # R=1, Out[0][0] = Xp[1-1-0][0] = Xp[0][0] = 5
    assert outs == [[[5]]]


def test_even_grid_mirror_h():
    """4x2 grid, mirror_h at multiple rows."""
    items = [{"Xp": [[1, 2], [3, 4], [5, 6], [7, 8]], "Y": [[7, 2], [5, 4], [3, 6], [1, 8]], "feats": {}, "residual": [[7, None], [5, None], [3, None], [1, None]]}]
    classes = {0: [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]}
    actions_by_cid = {0: ("mirror_h", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # R=4
    # Out[0][0] = Xp[3][0] = 7
    # Out[1][0] = Xp[2][0] = 5
    # Out[2][0] = Xp[1][0] = 3
    # Out[3][0] = Xp[0][0] = 1
    assert outs == [[[7, 2], [5, 4], [3, 6], [1, 8]]]


def test_odd_grid_mirror_v():
    """3x3 grid, mirror_v at multiple cols."""
    items = [{"Xp": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "Y": [[3, 2, 1], [6, 5, 4], [9, 8, 7]], "feats": {}, "residual": [[3, None, 1], [6, None, 4], [9, None, 7]]}]
    classes = {0: [(0, 0, 0), (0, 0, 2), (0, 1, 0), (0, 1, 2), (0, 2, 0), (0, 2, 2)]}
    actions_by_cid = {0: ("mirror_v", None)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # C=3
    # Out[r][0] = Xp[r][2] for all r
    # Out[r][2] = Xp[r][0] for all r
    # Middle column unchanged
    assert outs == [[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]


# ============================================================================
# CATEGORY C: PURITY AND DETERMINISM (5 tests)
# ============================================================================

def test_purity_items_unchanged():
    """Items unchanged after stitching."""
    items = [{"Xp": [[1, 2]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]}]
    items_orig = [{"Xp": [[1, 2]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions_by_cid = {0: ("set_color", 5)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # Items unchanged
    assert items == items_orig


def test_no_aliasing_out_rows():
    """Out rows are newly allocated, not aliasing Xp rows."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[1, 2], [3, 4]], "feats": {}, "residual": [[None, None], [None, None]]}]
    classes = {}
    actions_by_cid = {}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # Mutate Out
    outs[0][0][0] = 999

    # Verify Xp unchanged
    assert items[0]["Xp"][0][0] == 1


def test_determinism_same_inputs_same_outputs():
    """Same inputs → identical outputs."""
    items = [{"Xp": [[0, 1], [2, 3]], "Y": [[4, 5], [6, 7]], "feats": {}, "residual": [[4, 5], [6, 7]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]}
    actions_by_cid = {0: ("set_color", 5)}

    outs1 = stitch_from_classes(items, classes, actions_by_cid)
    outs2 = stitch_from_classes(items, classes, actions_by_cid)

    assert outs1 == outs2


def test_coord_ordering_within_class():
    """Results independent of coord order (sorted processing)."""
    items = [{"Xp": [[0, 0, 0]], "Y": [[5, 6, 7]], "feats": {}, "residual": [[5, 6, 7]]}]
    # Coords in random order (should be sorted by implementation)
    classes = {0: [(0, 0, 2), (0, 0, 0), (0, 0, 1)]}
    actions_by_cid = {0: ("set_color", 9)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    assert outs == [[[9, 9, 9]]]


def test_stable_hash_across_runs():
    """Hash of outputs identical across runs."""
    items = [{"Xp": [[0, 1]], "Y": [[2, 3]], "feats": {}, "residual": [[2, 3]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions_by_cid = {0: ("set_color", 5)}

    hashes = []
    for _ in range(10):
        outs = stitch_from_classes(items, classes, actions_by_cid)
        hashes.append(json.dumps(outs, sort_keys=True))

    # All hashes identical
    assert len(set(hashes)) == 1


# ============================================================================
# CATEGORY D: OPTIONAL SEAM CHECK (3 tests)
# ============================================================================

def test_seam_check_passes_disjoint_classes():
    """Seam check enabled with disjoint classes → no error."""
    items = [{"Xp": [[0, 0]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]}]
    classes = {
        0: [(0, 0, 0)],  # Disjoint
        1: [(0, 0, 1)],  # Disjoint
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 6),
    }

    # Should succeed
    outs = stitch_from_classes(items, classes, actions_by_cid, enable_seam_check=True)
    assert outs == [[[5, 6]]]


def test_seam_check_detects_overlap():
    """Seam check detects overlapping classes."""
    items = [{"Xp": [[0]], "Y": [[5]], "feats": {}, "residual": [[5]]}]
    classes = {
        0: [(0, 0, 0)],  # Overlap
        1: [(0, 0, 0)],  # Overlap with class 0
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 7),
    }

    with pytest.raises(ValueError, match="Seam check failed.*class 1.*overlap"):
        stitch_from_classes(items, classes, actions_by_cid, enable_seam_check=True)


def test_seam_check_disabled_allows_overlap():
    """Seam check disabled allows overlap (last write wins)."""
    items = [{"Xp": [[0]], "Y": [[7]], "feats": {}, "residual": [[7]]}]
    classes = {
        0: [(0, 0, 0)],  # Overlap
        1: [(0, 0, 0)],  # Overlap (processed after 0)
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 7),  # This write wins
    }

    # Should succeed (no check)
    outs = stitch_from_classes(items, classes, actions_by_cid, enable_seam_check=False)
    assert outs == [[[7]]]  # Class 1 write wins


# ============================================================================
# CATEGORY E: EDGE CASES (4 tests)
# ============================================================================

def test_empty_items_list():
    """Empty items → returns []."""
    outs = stitch_from_classes([], {}, {})
    assert outs == []


def test_large_number_of_classes():
    """100 classes, each with 1 pixel."""
    Xp = [[i for i in range(10)] for _ in range(10)]
    Y = [[9 - i for i in range(10)] for _ in range(10)]
    items = [{"Xp": Xp, "Y": Y, "feats": {}, "residual": Y}]

    # Create 100 classes (one pixel each)
    classes = {k: [(0, k // 10, k % 10)] for k in range(100)}
    actions_by_cid = {k: ("set_color", 9) for k in range(100)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # All 100 pixels set to 9
    assert all(all(val == 9 for val in row) for row in outs[0])


def test_all_pixels_in_residual():
    """Every pixel in residual → full grid rewritten."""
    items = [{"Xp": [[0, 0], [0, 0]], "Y": [[5, 5], [5, 5]], "feats": {}, "residual": [[5, 5], [5, 5]]}]
    classes = {0: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]}
    actions_by_cid = {0: ("set_color", 5)}

    outs = stitch_from_classes(items, classes, actions_by_cid)

    assert outs == [[[5, 5], [5, 5]]]


def test_partial_residual():
    """50% of pixels in residual → only those stitched."""
    items = [{"Xp": [[1, 2], [3, 4]], "Y": [[5, 2], [3, 6]], "feats": {}, "residual": [[5, None], [None, 6]]}]
    classes = {
        0: [(0, 0, 0)],  # Pixel (0,0)
        1: [(0, 1, 1)],  # Pixel (1,1)
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 6),
    }

    outs = stitch_from_classes(items, classes, actions_by_cid)

    # Only (0,0) and (1,1) changed
    assert outs == [[[5, 2], [3, 6]]]
