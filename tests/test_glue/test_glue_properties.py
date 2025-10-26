"""
Test suite for P6-05: GLUE Property Tests — stitched == one-shot; seam rejection

14 property tests across 4 sections:
- Section 1: GLUE Theorem T3 (stitched == one-shot) - 5 tests
- Section 2: Seam Rejection (overlap detection) - 3 tests
- Section 3: verify_stitched_equality wrapper - 3 tests
- Section 4: Determinism & Purity - 3 tests

These tests prove mathematical properties of the GLUE stitching system,
specifically Theorem T3 from primary-anchor.md: "If class masks are disjoint,
applying class actions and stitching yields the same grid as applying them
'one-shot.'"
"""

import pytest
from src.glue import stitch_from_classes, verify_stitched_equality
from src.utils import copy_grid, deep_eq


# ===========================================================================
# HELPER: Manual one-shot computation for GLUE theorem verification
# ===========================================================================

def compute_one_shot(items, classes, actions_by_cid):
    """
    Manually compute "one-shot" outputs by applying all class actions
    with reads from frozen Xp, for comparison with stitched results.

    This proves GLUE Theorem T3: when classes are disjoint, stitched
    output equals one-shot output (both read from frozen base).
    """
    # Initialize outputs as deep copies of Xp
    outputs = [copy_grid(item["Xp"]) for item in items]

    # Apply all actions in class order (same as stitcher)
    for class_id in sorted(classes.keys()):
        action_name, param = actions_by_cid[class_id]
        coords = classes[class_id]

        # Group by train
        from collections import defaultdict
        coords_by_train = defaultdict(list)
        for train_idx, r, c in coords:
            coords_by_train[train_idx].append((r, c))

        # Apply action to each train
        for train_idx, train_coords in coords_by_train.items():
            Xp = items[train_idx]["Xp"]
            Out = outputs[train_idx]
            from src.utils import dims
            R, C = dims(Xp)

            # Same logic as stitch_from_classes (frozen-base reads)
            if action_name == "set_color":
                for r, c in train_coords:
                    Out[r][c] = param
            elif action_name == "mirror_h":
                for r, c in train_coords:
                    Out[r][c] = Xp[R - 1 - r][c]  # Read from frozen Xp
            elif action_name == "mirror_v":
                for r, c in train_coords:
                    Out[r][c] = Xp[r][C - 1 - c]  # Read from frozen Xp
            elif action_name == "keep_nonzero":
                for r, c in train_coords:
                    Out[r][c] = Xp[r][c] if Xp[r][c] != 0 else 0
            elif action_name == "identity":
                for r, c in train_coords:
                    Out[r][c] = Xp[r][c]

    return outputs


# ===========================================================================
# SECTION 1: GLUE THEOREM T3 (stitched == one-shot) — 5 tests
# ===========================================================================

def test_glue_theorem_empty_classes():
    """Empty classes: stitched == one-shot == deep_copy(Xp)."""
    items = [
        {"Xp": [[1, 2]], "Y": [[1, 2]], "feats": {}, "residual": [[None, None]]},
    ]
    classes = {}
    actions_by_cid = {}

    stitched = stitch_from_classes(items, classes, actions_by_cid)
    one_shot = compute_one_shot(items, classes, actions_by_cid)

    assert stitched == one_shot
    assert stitched == [[[1, 2]]]


def test_glue_theorem_single_class_set_color():
    """Single class, set_color: stitched == one-shot."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 5]], "feats": {}, "residual": [[5, 5]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions_by_cid = {0: ("set_color", 5)}

    stitched = stitch_from_classes(items, classes, actions_by_cid)
    one_shot = compute_one_shot(items, classes, actions_by_cid)

    assert stitched == one_shot
    assert stitched == [[[5, 5]]]


def test_glue_theorem_two_classes_disjoint():
    """Two disjoint classes: stitched == one-shot (GLUE T3)."""
    items = [
        {"Xp": [[1, 2, 3]], "Y": [[7, 2, 9]], "feats": {}, "residual": [[7, None, 9]]},
    ]
    classes = {
        0: [(0, 0, 0)],  # Pixel (0,0)
        1: [(0, 0, 2)],  # Pixel (0,2)
    }
    actions_by_cid = {
        0: ("set_color", 7),
        1: ("set_color", 9),
    }

    stitched = stitch_from_classes(items, classes, actions_by_cid)
    one_shot = compute_one_shot(items, classes, actions_by_cid)

    # GLUE Theorem T3: disjoint classes commute
    assert stitched == one_shot
    assert stitched == [[[7, 2, 9]]]


def test_glue_theorem_mirror_frozen_base():
    """Mirror action with frozen-base reads: stitched == one-shot."""
    items = [
        {"Xp": [[1, 2], [3, 4]], "Y": [[3, 2], [1, 4]], "feats": {}, "residual": [[3, None], [1, None]]},
    ]
    classes = {0: [(0, 0, 0), (0, 1, 0)]}  # Column 0, both rows
    actions_by_cid = {0: ("mirror_h", None)}

    stitched = stitch_from_classes(items, classes, actions_by_cid)
    one_shot = compute_one_shot(items, classes, actions_by_cid)

    # Both read from frozen Xp (not from Out)
    assert stitched == one_shot
    assert stitched == [[[3, 2], [1, 4]]]


def test_glue_theorem_multiple_trains():
    """Multiple trains, class spans both: stitched == one-shot."""
    items = [
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
    ]
    classes = {0: [(0, 0, 0), (1, 0, 0)]}  # Both trains
    actions_by_cid = {0: ("set_color", 1)}

    stitched = stitch_from_classes(items, classes, actions_by_cid)
    one_shot = compute_one_shot(items, classes, actions_by_cid)

    assert stitched == one_shot
    assert stitched == [[[1]], [[1]]]


# ===========================================================================
# SECTION 2: SEAM REJECTION (overlap detection) — 3 tests
# ===========================================================================

def test_seam_rejection_two_classes_overlap():
    """Overlapping classes: seam check raises ValueError."""
    items = [
        {"Xp": [[0]], "Y": [[5]], "feats": {}, "residual": [[5]]},
    ]
    classes = {
        0: [(0, 0, 0)],  # Pixel (0,0)
        1: [(0, 0, 0)],  # Same pixel (OVERLAP)
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 7),
    }

    # Seam check enabled → raises ValueError
    with pytest.raises(ValueError, match="Seam check failed.*class 1.*overlap"):
        stitch_from_classes(items, classes, actions_by_cid, enable_seam_check=True)


def test_seam_rejection_partial_overlap():
    """Partial overlap across classes: seam check raises ValueError."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]},
    ]
    classes = {
        0: [(0, 0, 0), (0, 0, 1)],  # Both pixels
        1: [(0, 0, 1)],              # Pixel (0,1) overlaps
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 7),
    }

    # Seam check enabled → raises ValueError
    with pytest.raises(ValueError, match="Seam check failed"):
        stitch_from_classes(items, classes, actions_by_cid, enable_seam_check=True)


def test_seam_check_disabled_allows_overlap():
    """Seam check disabled: overlap allowed (last write wins)."""
    items = [
        {"Xp": [[0]], "Y": [[7]], "feats": {}, "residual": [[7]]},
    ]
    classes = {
        0: [(0, 0, 0)],  # Pixel (0,0)
        1: [(0, 0, 0)],  # Same pixel (overlap)
    }
    actions_by_cid = {
        0: ("set_color", 5),
        1: ("set_color", 7),  # This write wins (processed after class 0)
    }

    # Seam check disabled (default) → no error, last write wins
    stitched = stitch_from_classes(items, classes, actions_by_cid, enable_seam_check=False)
    assert stitched == [[[7]]]  # Class 1 write wins


# ===========================================================================
# SECTION 3: verify_stitched_equality wrapper — 3 tests
# ===========================================================================

def test_verify_stitched_equality_true_on_disjoint():
    """Disjoint fixture with exact match: verifier returns True."""
    items = [
        {"Xp": [[1, 2, 3]], "Y": [[7, 2, 9]], "feats": {}, "residual": [[7, None, 9]]},
    ]
    classes = {
        0: [(0, 0, 0)],
        1: [(0, 0, 2)],
    }
    actions_by_cid = {
        0: ("set_color", 7),
        1: ("set_color", 9),
    }

    result = verify_stitched_equality(items, classes, actions_by_cid)
    assert result is True


def test_verify_stitched_equality_false_on_altered_y():
    """Single pixel mismatch: verifier returns False (FY strictness)."""
    items = [
        {"Xp": [[0, 0]], "Y": [[5, 4]], "feats": {}, "residual": [[5, 4]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions_by_cid = {0: ("set_color", 5)}  # Sets both to 5, but Y has [5, 4]

    result = verify_stitched_equality(items, classes, actions_by_cid)
    assert result is False


def test_verify_stitched_equality_false_on_partial_train_failure():
    """Partial failure (second train doesn't match): returns False."""
    items = [
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
        {"Xp": [[0]], "Y": [[2]], "feats": {}, "residual": [[2]]},  # Different target
    ]
    classes = {0: [(0, 0, 0), (1, 0, 0)]}
    actions_by_cid = {0: ("set_color", 1)}  # Sets both to 1, but train 1 wants 2

    result = verify_stitched_equality(items, classes, actions_by_cid)
    assert result is False  # ALL trains must match


# ===========================================================================
# SECTION 4: DETERMINISM & PURITY — 3 tests
# ===========================================================================

def test_determinism_identical_runs():
    """Identical runs produce byte-identical results."""
    items = [
        {"Xp": [[0, 1], [2, 3]], "Y": [[5, 5], [5, 5]], "feats": {}, "residual": [[5, 5], [5, 5]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]}
    actions_by_cid = {0: ("set_color", 5)}

    stitched1 = stitch_from_classes(items, classes, actions_by_cid)
    stitched2 = stitch_from_classes(items, classes, actions_by_cid)

    assert stitched1 == stitched2
    assert deep_eq(stitched1[0], stitched2[0])


def test_purity_items_unchanged():
    """Items unchanged after stitching (no mutation)."""
    items = [
        {"Xp": [[1, 2]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]},
    ]
    items_orig = [
        {"Xp": [[1, 2]], "Y": [[5, 6]], "feats": {}, "residual": [[5, 6]]},
    ]
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions_by_cid = {0: ("set_color", 5)}

    stitch_from_classes(items, classes, actions_by_cid)

    # Items unchanged
    assert items == items_orig


def test_determinism_stable_hash():
    """Stable hash across 10 runs (deterministic outputs)."""
    import json

    items = [
        {"Xp": [[0]], "Y": [[1]], "feats": {}, "residual": [[1]]},
    ]
    classes = {0: [(0, 0, 0)]}
    actions_by_cid = {0: ("set_color", 1)}

    hashes = []
    for _ in range(10):
        stitched = stitch_from_classes(items, classes, actions_by_cid)
        hashes.append(json.dumps(stitched, sort_keys=True))

    # All hashes identical
    assert len(set(hashes)) == 1
