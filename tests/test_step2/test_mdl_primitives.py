"""
Test suite for P7-01: MDL Primitives — hash_candidate, compute_mdl

Test categories:
- Basic functionality (3 tests)
- Determinism (2 tests)
- Permutation invariance (2 tests)
- Action type counting (3 tests)
- Edge cases (4 tests)
- Error paths (5 tests)

Total: 19 tests
"""

import pytest
from src.mdl_selection import hash_candidate, compute_mdl


# ============================================================================
# CATEGORY 1: BASIC FUNCTIONALITY (3 tests)
# ============================================================================

def test_hash_candidate_returns_hex_string():
    """hash_candidate returns 64-character hex string."""
    P_desc = {"name": "Isometry", "index": 0}
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions = {0: ("set_color", 5)}

    result = hash_candidate(P_desc, classes, actions)

    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in '0123456789abcdef' for c in result)


def test_compute_mdl_returns_4tuple():
    """compute_mdl returns (num_classes, num_action_types, p_index, hash)."""
    P_desc = {"name": "ColorMap", "index": 2}
    classes = {0: [(0, 1, 2)], 1: [(0, 3, 4)]}
    actions = {0: ("set_color", 7), 1: ("mirror_h", None)}

    result = compute_mdl(P_desc, classes, actions)

    assert isinstance(result, tuple)
    assert len(result) == 4
    num_classes, num_action_types, p_index, stable_hash = result
    assert num_classes == 2
    assert num_action_types == 2  # set_color and mirror_h
    assert p_index == 2
    assert isinstance(stable_hash, str) and len(stable_hash) == 64


def test_compute_mdl_components_correct():
    """compute_mdl components computed correctly."""
    P_desc = {"name": "NPSDown", "index": 5}
    classes = {0: [(0, 0, 0)], 1: [(0, 1, 1)], 2: [(0, 2, 2)]}
    actions = {
        0: ("set_color", 1),
        1: ("set_color", 2),  # Same action name as class 0
        2: ("keep_nonzero", None)
    }

    num_classes, num_action_types, p_index, _ = compute_mdl(P_desc, classes, actions)

    assert num_classes == 3
    assert num_action_types == 2  # set_color and keep_nonzero (not 3)
    assert p_index == 5


# ============================================================================
# CATEGORY 2: DETERMINISM (2 tests)
# ============================================================================

def test_hash_candidate_deterministic():
    """Same inputs → same hash (determinism)."""
    P_desc = {"name": "PixelReplicate", "index": 3}
    classes = {0: [(0, 5, 10), (0, 5, 11)], 1: [(1, 2, 3)]}
    actions = {0: ("mirror_v", None), 1: ("identity", None)}

    hash1 = hash_candidate(P_desc, classes, actions)
    hash2 = hash_candidate(P_desc, classes, actions)

    assert hash1 == hash2


def test_compute_mdl_deterministic():
    """Same inputs → same MDL tuple (determinism)."""
    P_desc = {"name": "BlockDown", "index": 4}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 9)}

    mdl1 = compute_mdl(P_desc, classes, actions)
    mdl2 = compute_mdl(P_desc, classes, actions)

    assert mdl1 == mdl2
    assert mdl1[3] == mdl2[3]  # Hash component stable


# ============================================================================
# CATEGORY 3: PERMUTATION INVARIANCE (2 tests)
# ============================================================================

def test_hash_candidate_dict_order_invariant():
    """Different dict insertion order → same hash."""
    P_desc = {"name": "ParityTile", "index": 7}

    # Version 1: classes in order 0, 1, 2
    classes1 = {0: [(0, 0, 0)], 1: [(0, 1, 1)], 2: [(0, 2, 2)]}
    actions1 = {0: ("set_color", 1), 1: ("set_color", 2), 2: ("set_color", 3)}

    # Version 2: classes in reverse order 2, 1, 0 (different insertion order)
    classes2 = {2: [(0, 2, 2)], 1: [(0, 1, 1)], 0: [(0, 0, 0)]}
    actions2 = {2: ("set_color", 3), 1: ("set_color", 2), 0: ("set_color", 1)}

    hash1 = hash_candidate(P_desc, classes1, actions1)
    hash2 = hash_candidate(P_desc, classes2, actions2)

    assert hash1 == hash2


def test_hash_candidate_coord_order_canonicalized():
    """Coords sorted by (i,r,c) → hash stable regardless of input order."""
    P_desc = {"name": "SortRowsLex", "index": 13}

    # Coords in unsorted order
    classes1 = {0: [(0, 5, 10), (0, 3, 8), (0, 5, 9)]}
    actions1 = {0: ("identity", None)}

    # Same coords in different order
    classes2 = {0: [(0, 3, 8), (0, 5, 9), (0, 5, 10)]}
    actions2 = {0: ("identity", None)}

    hash1 = hash_candidate(P_desc, classes1, actions1)
    hash2 = hash_candidate(P_desc, classes2, actions2)

    assert hash1 == hash2


# ============================================================================
# CATEGORY 4: ACTION TYPE COUNTING (3 tests)
# ============================================================================

def test_action_types_count_unique_names():
    """Action types counted by name only, not params."""
    P_desc = {"name": "Identity", "index": 0}
    classes = {0: [(0, 0, 0)], 1: [(0, 1, 1)], 2: [(0, 2, 2)]}
    actions = {
        0: ("set_color", 1),
        1: ("set_color", 2),  # Same name, different param
        2: ("set_color", 3)   # Same name, different param
    }

    _, num_action_types, _, _ = compute_mdl(P_desc, classes, actions)

    assert num_action_types == 1  # Only "set_color"


def test_action_types_multiple_different_names():
    """Multiple action names → correct count."""
    P_desc = {"name": "MirrorComplete", "index": 11}
    classes = {0: [(0, 0, 0)], 1: [(0, 1, 1)], 2: [(0, 2, 2)]}
    actions = {
        0: ("set_color", 5),
        1: ("mirror_h", None),
        2: ("keep_nonzero", None)
    }

    _, num_action_types, _, _ = compute_mdl(P_desc, classes, actions)

    assert num_action_types == 3


def test_action_types_with_none_and_int_params():
    """Action types counted correctly with mix of None and int params."""
    P_desc = {"name": "RowPermutation", "index": 12}
    classes = {0: [(0, 0, 0)], 1: [(0, 1, 1)]}
    actions = {
        0: ("mirror_v", None),   # None param
        1: ("set_color", 7)      # int param
    }

    _, num_action_types, _, _ = compute_mdl(P_desc, classes, actions)

    assert num_action_types == 2


# ============================================================================
# CATEGORY 5: EDGE CASES (4 tests)
# ============================================================================

def test_empty_classes_and_actions():
    """Empty classes and actions → valid hash and MDL."""
    P_desc = {"name": "Identity", "index": 0}
    classes = {}
    actions = {}

    hash_result = hash_candidate(P_desc, classes, actions)
    mdl_result = compute_mdl(P_desc, classes, actions)

    assert isinstance(hash_result, str) and len(hash_result) == 64
    assert mdl_result == (0, 0, 0, hash_result)


def test_single_class_single_coord():
    """Single class with single coord → correct MDL."""
    P_desc = {"name": "ColorMap", "index": 2}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    num_classes, num_action_types, p_index, hash_val = compute_mdl(P_desc, classes, actions)

    assert num_classes == 1
    assert num_action_types == 1
    assert p_index == 2
    assert len(hash_val) == 64


def test_multiple_coords_same_class():
    """Single class with many coords → counted as 1 class."""
    P_desc = {"name": "NPSUp", "index": 6}
    classes = {0: [(0, i, i) for i in range(10)]}  # 10 coords
    actions = {0: ("mirror_h", None)}

    num_classes, _, _, _ = compute_mdl(P_desc, classes, actions)

    assert num_classes == 1  # Still just 1 class


def test_large_p_index():
    """Large P index → handled correctly."""
    P_desc = {"name": "CopyMoveAllComponents", "index": 15}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("identity", None)}

    _, _, p_index, _ = compute_mdl(P_desc, classes, actions)

    assert p_index == 15


# ============================================================================
# CATEGORY 6: ERROR PATHS (5 tests)
# ============================================================================

def test_missing_p_desc_name():
    """Missing P_desc['name'] → ValueError."""
    P_desc = {"index": 0}  # Missing "name"
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    with pytest.raises(ValueError, match="must contain 'name' and 'index' fields"):
        hash_candidate(P_desc, classes, actions)


def test_missing_p_desc_index():
    """Missing P_desc['index'] → ValueError."""
    P_desc = {"name": "Isometry"}  # Missing "index"
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    with pytest.raises(ValueError, match="must contain 'name' and 'index' fields"):
        hash_candidate(P_desc, classes, actions)


def test_negative_p_index():
    """Negative P index → ValueError."""
    P_desc = {"name": "BadFamily", "index": -1}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    with pytest.raises(ValueError, match="must be non-negative integer"):
        hash_candidate(P_desc, classes, actions)


def test_invalid_coord_format():
    """Malformed coord (not 3-tuple) → ValueError."""
    P_desc = {"name": "Identity", "index": 0}
    classes = {0: [(0, 0)]}  # Only 2 elements
    actions = {0: ("set_color", 1)}

    with pytest.raises(ValueError, match="Invalid coord format"):
        hash_candidate(P_desc, classes, actions)


def test_non_json_serializable_param():
    """Non-JSON-serializable param (float) → TypeError."""
    P_desc = {"name": "Identity", "index": 0}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1.5)}  # Float not allowed

    with pytest.raises(TypeError, match="must be None or int"):
        hash_candidate(P_desc, classes, actions)


# ============================================================================
# COMPREHENSIVE ADVERSARIAL MINI
# ============================================================================

def test_mdl_tie_breaking_order():
    """MDL tuples sort correctly: fewer classes wins, then fewer types, then lower index."""
    # Candidate A: 2 classes, 1 type, index 5
    mdl_A = (2, 1, 5, "aaa")

    # Candidate B: 3 classes, 1 type, index 3 (more classes → worse)
    mdl_B = (3, 1, 3, "bbb")

    # Candidate C: 2 classes, 2 types, index 3 (same classes as A, more types → worse)
    mdl_C = (2, 2, 3, "ccc")

    # Candidate D: 2 classes, 1 type, index 10 (same as A except higher index → worse)
    mdl_D = (2, 1, 10, "ddd")

    candidates = [mdl_B, mdl_D, mdl_A, mdl_C]
    sorted_candidates = sorted(candidates)

    # Best to worst: A < D < C < B
    # (2,1,5) < (2,1,10) < (2,2,3) < (3,1,3)
    assert sorted_candidates[0] == mdl_A
    assert sorted_candidates[1] == mdl_D
    assert sorted_candidates[2] == mdl_C
    assert sorted_candidates[3] == mdl_B
