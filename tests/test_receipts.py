"""
Tests for Receipts Module — Phase 3 work order P3-02.

Comprehensive test suite covering:
- generate_receipt_global with/without params
- generate_receipt_unsat with different reasons
- receipt_stable_hash determinism and stability
- task_meta structure validation
- Edge cases: None params, non-serializable params, empty reason
- JSON serialization verification
- Hash collision avoidance
"""

import pytest
import json
from src.receipts import generate_receipt_global, generate_receipt_unsat, receipt_stable_hash


# =============================
# Mock Family Objects
# =============================

class MockFamilyWithParams:
    """Family with JSON-serializable params dict."""
    def __init__(self):
        self.name = "ColorMap"
        self.params = {"color_map": {0: 1, 1: 2, 2: 0}}


class MockFamilyWithObjectParams:
    """Family with params as object with __dict__."""
    def __init__(self):
        self.name = "Isometry"
        # Create params object with instance attributes
        params_obj = type('Params', (), {})()
        params_obj.sigma = 'rot90'
        params_obj.flip = False
        self.params = params_obj


class MockFamilyNoParams:
    """Family with params=None."""
    def __init__(self):
        self.name = "Isometry"
        self.params = None


class MockFamilyBadParams:
    """Family with non-JSON-serializable params."""
    def __init__(self):
        self.name = "ComplexFamily"
        self.params = {"func": lambda x: x}  # Not JSON-serializable


class MockFamilyNonDictParams:
    """Family with params that's not a dict."""
    def __init__(self):
        self.name = "StringFamily"
        self.params = "not_a_dict"


# =============================
# A. generate_receipt_global Tests
# =============================

def test_generate_receipt_global_with_params():
    """PASS receipt with params dict."""
    family = MockFamilyWithParams()
    task_meta = {"task_id": "test1", "train_n": 3, "test_n": 1}

    receipt = generate_receipt_global(family, task_meta)

    assert receipt["mode"] == "global"
    assert receipt["solver"] == "ColorMap"
    assert receipt["task"] == task_meta
    assert "params" in receipt
    assert receipt["params"] == {"color_map": {0: 1, 1: 2, 2: 0}}

    # JSON-serializable check
    json_str = json.dumps(receipt)
    parsed = json.loads(json_str)
    # Note: JSON converts integer dict keys to strings, so we just verify serialization works
    assert parsed["mode"] == receipt["mode"]
    assert parsed["solver"] == receipt["solver"]
    assert parsed["task"] == receipt["task"]


def test_generate_receipt_global_with_object_params():
    """PASS receipt with params as object with __dict__."""
    family = MockFamilyWithObjectParams()
    task_meta = {"task_id": "test_obj", "train_n": 2, "test_n": 1}

    receipt = generate_receipt_global(family, task_meta)

    assert receipt["mode"] == "global"
    assert receipt["solver"] == "Isometry"
    assert receipt["task"] == task_meta
    assert "params" in receipt
    assert receipt["params"] == {"sigma": "rot90", "flip": False}

    # JSON-serializable check
    json_str = json.dumps(receipt)
    assert json.loads(json_str) == receipt


def test_generate_receipt_global_no_params():
    """PASS receipt without params (params=None)."""
    family = MockFamilyNoParams()
    task_meta = {"task_id": "test2", "train_n": 2, "test_n": 1}

    receipt = generate_receipt_global(family, task_meta)

    assert receipt["mode"] == "global"
    assert receipt["solver"] == "Isometry"
    assert receipt["task"] == task_meta
    assert "params" not in receipt  # Omitted when None


def test_generate_receipt_global_bad_params_omitted():
    """PASS receipt with non-serializable params → omit params."""
    family = MockFamilyBadParams()
    task_meta = {"task_id": "test3", "train_n": 1, "test_n": 1}

    receipt = generate_receipt_global(family, task_meta)

    assert receipt["mode"] == "global"
    assert receipt["solver"] == "ComplexFamily"
    assert receipt["task"] == task_meta
    assert "params" not in receipt  # Non-serializable params omitted

    # Receipt should still be JSON-serializable
    json_str = json.dumps(receipt)
    assert json.loads(json_str) == receipt


def test_generate_receipt_global_non_dict_params_omitted():
    """PASS receipt with non-dict params → omit params."""
    family = MockFamilyNonDictParams()
    task_meta = {"task_id": "test_str", "train_n": 1, "test_n": 1}

    receipt = generate_receipt_global(family, task_meta)

    assert receipt["mode"] == "global"
    assert receipt["solver"] == "StringFamily"
    assert receipt["task"] == task_meta
    assert "params" not in receipt  # Non-dict params omitted


# =============================
# B. generate_receipt_unsat Tests
# =============================

def test_generate_receipt_unsat():
    """UNSAT receipt with standard reason."""
    task_meta = {"task_id": "test4", "train_n": 3, "test_n": 2}
    reason = "no_global_P_satisfied_FY"

    receipt = generate_receipt_unsat(reason, task_meta)

    assert receipt["mode"] == "unsat"
    assert receipt["reason"] == reason
    assert receipt["task"] == task_meta

    # JSON-serializable check
    json_str = json.dumps(receipt)
    assert json.loads(json_str) == receipt


def test_unsat_empty_reason():
    """UNSAT receipt with empty reason string (edge case)."""
    task_meta = {"task_id": "empty", "train_n": 0, "test_n": 0}
    receipt = generate_receipt_unsat("", task_meta)

    assert receipt["mode"] == "unsat"
    assert receipt["reason"] == ""
    assert receipt["task"] == task_meta

    # Should still serialize
    json_str = json.dumps(receipt)
    assert json.loads(json_str) == receipt


def test_unsat_different_reasons():
    """UNSAT receipts with different reason strings."""
    task_meta = {"task_id": "multi", "train_n": 2, "test_n": 1}

    reasons = [
        "no_global_P_satisfied_FY",
        "no_train_pairs",
        "malformed_input",
        "shape_mismatch_all_families"
    ]

    for reason in reasons:
        receipt = generate_receipt_unsat(reason, task_meta)
        assert receipt["reason"] == reason
        assert receipt["mode"] == "unsat"
        # All should be JSON-serializable
        json.dumps(receipt)


# =============================
# C. Determinism Tests
# =============================

def test_determinism_pass_receipt():
    """Same input → same PASS receipt."""
    family = MockFamilyWithParams()
    task_meta = {"task_id": "det", "train_n": 2, "test_n": 1}

    receipt1 = generate_receipt_global(family, task_meta)
    receipt2 = generate_receipt_global(family, task_meta)

    assert receipt1 == receipt2
    assert receipt_stable_hash(receipt1) == receipt_stable_hash(receipt2)


def test_determinism_unsat_receipt():
    """Same input → same UNSAT receipt."""
    task_meta = {"task_id": "det_unsat", "train_n": 1, "test_n": 1}
    reason = "no_global_P_satisfied_FY"

    receipt1 = generate_receipt_unsat(reason, task_meta)
    receipt2 = generate_receipt_unsat(reason, task_meta)

    assert receipt1 == receipt2
    assert receipt_stable_hash(receipt1) == receipt_stable_hash(receipt2)


# =============================
# D. receipt_stable_hash Tests
# =============================

def test_stable_hash_deterministic():
    """Same receipt → same hash (multiple calls)."""
    receipt = {
        "mode": "global",
        "solver": "Test",
        "task": {"task_id": "hash", "train_n": 1, "test_n": 1},
        "params": {"a": 1, "b": 2}
    }

    hash1 = receipt_stable_hash(receipt)
    hash2 = receipt_stable_hash(receipt)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest
    assert all(c in '0123456789abcdef' for c in hash1)


def test_different_receipts_different_hashes():
    """Different receipts → different hashes."""
    task_meta1 = {"task_id": "A", "train_n": 1, "test_n": 1}
    task_meta2 = {"task_id": "B", "train_n": 1, "test_n": 1}

    family = MockFamilyNoParams()

    receipt1 = generate_receipt_global(family, task_meta1)
    receipt2 = generate_receipt_global(family, task_meta2)

    hash1 = receipt_stable_hash(receipt1)
    hash2 = receipt_stable_hash(receipt2)

    assert hash1 != hash2  # Different task_id → different hash


def test_stable_hash_nested_structures():
    """Nested dicts/lists → stable hash with canonical ordering."""
    receipt = {
        "mode": "global",
        "solver": "Nested",
        "task": {"task_id": "nest", "train_n": 1, "test_n": 1},
        "params": {
            "nested": {"a": [1, 2, 3], "b": {"c": 4}},
            "list": [5, 6, 7]
        }
    }

    hash_result = receipt_stable_hash(receipt)
    assert len(hash_result) == 64

    # Same receipt built differently (different key order) → same hash
    receipt_reordered = {
        "task": {"test_n": 1, "task_id": "nest", "train_n": 1},  # Different order
        "solver": "Nested",
        "mode": "global",
        "params": {
            "list": [5, 6, 7],
            "nested": {"b": {"c": 4}, "a": [1, 2, 3]}  # Different order
        }
    }

    assert receipt_stable_hash(receipt_reordered) == hash_result


def test_stable_hash_pass_and_unsat():
    """Both PASS and UNSAT receipts can be hashed."""
    family = MockFamilyWithParams()
    task_meta = {"task_id": "both", "train_n": 2, "test_n": 1}

    receipt_pass = generate_receipt_global(family, task_meta)
    receipt_unsat = generate_receipt_unsat("no_match", task_meta)

    hash_pass = receipt_stable_hash(receipt_pass)
    hash_unsat = receipt_stable_hash(receipt_unsat)

    # Both should produce valid hashes
    assert len(hash_pass) == 64
    assert len(hash_unsat) == 64

    # Different modes → different hashes
    assert hash_pass != hash_unsat


# =============================
# E. task_meta Validation
# =============================

def test_task_id_types():
    """task_meta supports both string and int task_id."""
    task_meta_str = {"task_id": "string_id", "train_n": 1, "test_n": 1}
    task_meta_int = {"task_id": 42, "train_n": 1, "test_n": 1}

    family = MockFamilyNoParams()

    receipt_str = generate_receipt_global(family, task_meta_str)
    receipt_int = generate_receipt_global(family, task_meta_int)

    assert receipt_str["task"]["task_id"] == "string_id"
    assert receipt_int["task"]["task_id"] == 42

    # Both should serialize
    json.dumps(receipt_str)
    json.dumps(receipt_int)

    # Different task_id → different hashes
    assert receipt_stable_hash(receipt_str) != receipt_stable_hash(receipt_int)


def test_task_meta_preserved():
    """task_meta is preserved exactly in receipt."""
    task_meta = {
        "task_id": "preserve",
        "train_n": 5,
        "test_n": 3
    }

    family = MockFamilyNoParams()
    receipt = generate_receipt_global(family, task_meta)

    # task_meta should be identical (not a copy issue)
    assert receipt["task"] == task_meta
    assert receipt["task"]["task_id"] == "preserve"
    assert receipt["task"]["train_n"] == 5
    assert receipt["task"]["test_n"] == 3


# =============================
# F. JSON Serializability
# =============================

def test_all_receipts_json_serializable():
    """All receipt types are JSON-serializable."""
    family_with = MockFamilyWithParams()
    family_without = MockFamilyNoParams()
    task_meta = {"task_id": "json_test", "train_n": 2, "test_n": 1}

    # PASS with params - just verify it serializes (JSON converts int keys to strings)
    receipt1 = generate_receipt_global(family_with, task_meta)
    json_str1 = json.dumps(receipt1)
    parsed1 = json.loads(json_str1)
    assert parsed1["mode"] == "global"

    # PASS without params
    receipt2 = generate_receipt_global(family_without, task_meta)
    json_str2 = json.dumps(receipt2)
    assert json.loads(json_str2) == receipt2

    # UNSAT
    receipt3 = generate_receipt_unsat("test_reason", task_meta)
    json_str3 = json.dumps(receipt3)
    assert json.loads(json_str3) == receipt3


# =============================
# G. Hash Collision Avoidance
# =============================

def test_hash_collision_different_solvers():
    """Different solvers → different hashes (even with same task_meta)."""
    task_meta = {"task_id": "collision", "train_n": 1, "test_n": 1}

    family1 = MockFamilyNoParams()
    family1.name = "SolverA"

    family2 = MockFamilyNoParams()
    family2.name = "SolverB"

    receipt1 = generate_receipt_global(family1, task_meta)
    receipt2 = generate_receipt_global(family2, task_meta)

    hash1 = receipt_stable_hash(receipt1)
    hash2 = receipt_stable_hash(receipt2)

    assert hash1 != hash2


def test_hash_collision_different_params():
    """Different params → different hashes."""
    task_meta = {"task_id": "params_diff", "train_n": 1, "test_n": 1}

    family1 = MockFamilyWithParams()
    family1.params = {"color_map": {0: 1}}

    family2 = MockFamilyWithParams()
    family2.params = {"color_map": {0: 2}}

    receipt1 = generate_receipt_global(family1, task_meta)
    receipt2 = generate_receipt_global(family2, task_meta)

    hash1 = receipt_stable_hash(receipt1)
    hash2 = receipt_stable_hash(receipt2)

    assert hash1 != hash2


# =============================
# H. Edge Cases
# =============================

def test_unicode_in_task_id():
    """Unicode strings in task_meta → stable hash."""
    task_meta = {"task_id": "τεστ_üñíçödé", "train_n": 1, "test_n": 1}
    family = MockFamilyNoParams()

    receipt = generate_receipt_global(family, task_meta)

    # Should be JSON-serializable
    json_str = json.dumps(receipt)
    assert json.loads(json_str) == receipt

    # Should produce stable hash
    hash1 = receipt_stable_hash(receipt)
    hash2 = receipt_stable_hash(receipt)
    assert hash1 == hash2


def test_large_params_dict():
    """Large params dict → stable hash."""
    family = MockFamilyWithParams()
    family.params = {str(i): i for i in range(100)}  # 100 key-value pairs

    task_meta = {"task_id": "large", "train_n": 1, "test_n": 1}
    receipt = generate_receipt_global(family, task_meta)

    # Should serialize and hash
    json.dumps(receipt)
    hash_result = receipt_stable_hash(receipt)
    assert len(hash_result) == 64
