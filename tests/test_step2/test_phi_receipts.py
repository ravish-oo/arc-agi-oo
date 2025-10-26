"""
Test suite for P7-04: Receipts (Φ mode) — generate_receipt_phi

Test categories:
- Receipt structure validation (3 tests)
- MDL candidates sorting (2 tests)
- Actions histogram counting (3 tests)
- Determinism (2 tests)
- Error paths (4 tests)
- Edge cases (3 tests)

Total: 17 tests
"""

import pytest
import json
from src.receipts import generate_receipt_phi, receipt_stable_hash
from src.mdl_selection import compute_mdl


# ============================================================================
# HELPER: Build candidate dict with proper MDL
# ============================================================================

def build_candidate(P_desc, classes, actions):
    """Helper to build candidate dict with computed MDL."""
    mdl_tuple = compute_mdl(P_desc, classes, actions)
    num_classes, num_action_types, p_index, stable_hash = mdl_tuple

    return {
        "P": P_desc,
        "classes": classes,
        "actions": actions,
        "mdl": {
            "num_classes": num_classes,
            "num_action_types": num_action_types,
            "p_index": p_index,
            "hash": stable_hash
        }
    }


# ============================================================================
# CATEGORY 1: RECEIPT STRUCTURE VALIDATION (3 tests)
# ============================================================================

def test_receipt_has_required_keys():
    """Receipt has exactly required top-level keys."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 5)}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    assert set(receipt.keys()) == {"mode", "chosen_candidate", "mdl_candidates"}
    assert receipt["mode"] == "phi_partition"


def test_chosen_candidate_structure():
    """Chosen candidate has correct structure."""
    P_desc = {"name": "Isometry", "index": 0}
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions = {0: ("set_color", 7)}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    chosen = receipt["chosen_candidate"]
    assert set(chosen.keys()) == {"P", "mdl", "summary", "reason"}
    assert set(chosen["P"].keys()) == {"name", "index"}
    assert set(chosen["mdl"].keys()) == {"num_classes", "num_action_types", "p_index", "hash"}
    assert set(chosen["summary"].keys()) == {"num_classes", "num_action_types", "actions_histogram"}


def test_mdl_candidates_list_structure():
    """MDL candidates list has correct structure."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {}
    actions = {}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    assert isinstance(receipt["mdl_candidates"], list)
    assert len(receipt["mdl_candidates"]) == 1

    cand = receipt["mdl_candidates"][0]
    assert set(cand.keys()) == {"P", "mdl", "summary"}


# ============================================================================
# CATEGORY 2: MDL CANDIDATES SORTING (2 tests)
# ============================================================================

def test_mdl_candidates_sorted_by_tuple():
    """MDL candidates sorted by (num_classes, num_action_types, p_index, hash)."""
    P1 = {"name": "ColorMap", "index": 0}
    P2 = {"name": "Isometry", "index": 1}
    P3 = {"name": "PixelReplicate", "index": 2}

    classes1 = {0: [(0, 0, 0)], 1: [(0, 1, 0)]}
    classes2 = {0: [(0, 0, 0)]}
    classes3 = {0: [(0, 0, 0)], 1: [(0, 1, 0)]}

    actions1 = {0: ("set_color", 1), 1: ("mirror_h", None)}
    actions2 = {0: ("set_color", 3)}
    actions3 = {0: ("set_color", 2), 1: ("set_color", 4)}

    # Build candidates with proper MDL
    mdl_candidates = [
        build_candidate(P1, classes1, actions1),
        build_candidate(P2, classes2, actions2),
        build_candidate(P3, classes3, actions3),
    ]

    receipt = generate_receipt_phi(P2, classes2, actions2, mdl_candidates)

    # Sorted order: P2 (1 class) < P3 (2 classes, 1 type) < P1 (2 classes, 2 types)
    sorted_cands = receipt["mdl_candidates"]
    assert sorted_cands[0]["P"]["name"] == "Isometry"
    assert sorted_cands[1]["P"]["name"] == "PixelReplicate"
    assert sorted_cands[2]["P"]["name"] == "ColorMap"


def test_chosen_candidate_is_first_in_sorted_list():
    """Chosen candidate (MDL best) appears first in sorted mdl_candidates."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    P2 = {"name": "ColorMap", "index": 0}
    classes2 = {0: [(0, 0, 0)], 1: [(0, 1, 0)]}
    actions2 = {0: ("set_color", 2), 1: ("set_color", 3)}

    mdl_candidates = [
        build_candidate(P2, classes2, actions2),
        build_candidate(P_desc, classes, actions),
    ]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    # Identity should be first (1 class < 2 classes)
    assert receipt["mdl_candidates"][0]["P"]["name"] == "Identity"
    assert receipt["chosen_candidate"]["P"]["name"] == "Identity"


# ============================================================================
# CATEGORY 3: ACTIONS HISTOGRAM COUNTING (3 tests)
# ============================================================================

def test_actions_histogram_counts_names_not_params():
    """Histogram counts action names, not params."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {0: [(0, 0, 0)], 1: [(0, 0, 1)], 2: [(0, 0, 2)], 3: [(0, 1, 0)]}
    actions = {
        0: ("set_color", 1),
        1: ("set_color", 2),  # Same name, different param
        2: ("mirror_h", None),
        3: ("set_color", 3),  # Another set_color
    }

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    histogram = receipt["chosen_candidate"]["summary"]["actions_histogram"]
    assert histogram == {"mirror_h": 1, "set_color": 3}  # Sorted by name


def test_actions_histogram_sorted_alphabetically():
    """Histogram sorted alphabetically by action name."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {0: [(0, 0, 0)], 1: [(0, 0, 1)], 2: [(0, 0, 2)]}
    actions = {
        0: ("set_color", 1),
        1: ("keep_nonzero", None),
        2: ("mirror_h", None),
    }

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    histogram = receipt["chosen_candidate"]["summary"]["actions_histogram"]
    # Should be sorted: keep_nonzero < mirror_h < set_color
    assert list(histogram.keys()) == ["keep_nonzero", "mirror_h", "set_color"]


def test_actions_histogram_empty_when_no_actions():
    """Empty histogram when no actions (zero classes)."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {}
    actions = {}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    histogram = receipt["chosen_candidate"]["summary"]["actions_histogram"]
    assert histogram == {}


# ============================================================================
# CATEGORY 4: DETERMINISM (2 tests)
# ============================================================================

def test_determinism_identical_runs():
    """Same inputs → same receipt (dict equality)."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {0: [(0, 0, 0), (0, 0, 1)]}
    actions = {0: ("set_color", 5)}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt1 = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)
    receipt2 = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    assert receipt1 == receipt2


def test_determinism_stable_json_serialization():
    """Same inputs → same JSON serialization (stable hash)."""
    P_desc = {"name": "ColorMap", "index": 0}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 9)}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    hashes = []
    for _ in range(3):
        receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)
        receipt_hash = receipt_stable_hash(receipt)
        hashes.append(receipt_hash)

    # All hashes identical
    assert len(set(hashes)) == 1


# ============================================================================
# CATEGORY 5: ERROR PATHS (4 tests)
# ============================================================================

def test_error_missing_p_desc_name():
    """Missing P_desc['name'] → ValueError."""
    P_desc = {"index": 0}  # Missing "name"
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    mdl_candidates = [{
        "P": {"name": "ColorMap", "index": 0},
        "classes": classes,
        "actions": actions,
        "mdl": {"num_classes": 1, "num_action_types": 1, "p_index": 0, "hash": "err1"}
    }]

    with pytest.raises(ValueError, match="must contain 'name' and 'index' fields"):
        generate_receipt_phi(P_desc, classes, actions, mdl_candidates)


def test_error_negative_p_index_below_minus_one():
    """P_desc['index'] < -1 → ValueError."""
    P_desc = {"name": "BadFamily", "index": -2}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    mdl_candidates = [{
        "P": P_desc,
        "classes": classes,
        "actions": actions,
        "mdl": {"num_classes": 1, "num_action_types": 1, "p_index": -2, "hash": "err2"}
    }]

    with pytest.raises(ValueError, match="must be >= -1"):
        generate_receipt_phi(P_desc, classes, actions, mdl_candidates)


def test_error_empty_mdl_candidates():
    """Empty mdl_candidates list → ValueError."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("set_color", 1)}

    with pytest.raises(ValueError, match="must not be empty"):
        generate_receipt_phi(P_desc, classes, actions, [])


def test_error_chosen_not_in_candidates():
    """Chosen candidate not in mdl_candidates → ValueError."""
    P_chosen = {"name": "Identity", "index": -1}
    classes_chosen = {0: [(0, 0, 0)]}
    actions_chosen = {0: ("set_color", 1)}

    # mdl_candidates has different candidate
    P_other = {"name": "ColorMap", "index": 0}
    classes_other = {0: [(0, 0, 0)], 1: [(0, 1, 0)]}
    actions_other = {0: ("set_color", 2), 1: ("set_color", 3)}

    mdl_candidates = [build_candidate(P_other, classes_other, actions_other)]

    with pytest.raises(ValueError, match="not found in mdl_candidates"):
        generate_receipt_phi(P_chosen, classes_chosen, actions_chosen, mdl_candidates)


# ============================================================================
# CATEGORY 6: EDGE CASES (3 tests)
# ============================================================================

def test_edge_case_zero_classes():
    """Zero classes (Identity with no edits) → valid receipt."""
    P_desc = {"name": "Identity", "index": -1}
    classes = {}
    actions = {}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    assert receipt["chosen_candidate"]["mdl"]["num_classes"] == 0
    assert receipt["chosen_candidate"]["summary"]["actions_histogram"] == {}


def test_edge_case_single_candidate():
    """Single candidate → validates correctly."""
    P_desc = {"name": "Isometry", "index": 0}
    classes = {0: [(0, 0, 0)]}
    actions = {0: ("mirror_v", None)}

    mdl_candidates = [build_candidate(P_desc, classes, actions)]

    receipt = generate_receipt_phi(P_desc, classes, actions, mdl_candidates)

    assert len(receipt["mdl_candidates"]) == 1
    assert receipt["chosen_candidate"]["P"]["name"] == "Isometry"


def test_edge_case_tie_breaking_with_hash():
    """Identical (num_classes, num_action_types, p_index) → hash breaks tie."""
    P1 = {"name": "FamilyA", "index": 5}
    P2 = {"name": "FamilyB", "index": 5}

    classes1 = {0: [(0, 0, 0)]}
    classes2 = {0: [(0, 1, 0)]}

    actions1 = {0: ("set_color", 1)}
    actions2 = {0: ("set_color", 2)}

    # Build candidates with proper MDL (hashes will be different due to different inputs)
    mdl_candidates = [
        build_candidate(P1, classes1, actions1),
        build_candidate(P2, classes2, actions2),
    ]

    # Choose first candidate
    receipt = generate_receipt_phi(P1, classes1, actions1, mdl_candidates)

    # Verify first candidate is chosen
    assert receipt["chosen_candidate"]["P"]["name"] == "FamilyA"
    # Both candidates should be in sorted list
    assert len(receipt["mdl_candidates"]) == 2
