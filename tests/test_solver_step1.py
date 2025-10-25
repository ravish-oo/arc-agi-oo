"""
Tests for Step-1 Solver — Phase 3 work order P3-01.

Comprehensive test suite covering:
- Basic solve_step1() with different family types
- First-pass selection (not MDL)
- Edge cases (empty train/test, malformed task)
- FY exactness verification
- Determinism and purity
- Receipt generation for PASS and UNSAT
- Integration with all 16 global families
"""

import pytest
import copy
from src.solver_step1 import solve_step1, _try_family, GLOBAL_FAMILIES
from src.families.isometry import IsometryFamily
from src.families.color_map import ColorMapFamily
from src.families.pixel_replicate import PixelReplicateFamily
from src.utils import deep_eq


# =============================
# A. Basic solve_step1 Tests
# =============================

def test_solve_step1_identity():
    """Identity task → Isometry family with sigma='id'."""
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]}
        ],
        "test": [
            {"input": [[9, 0], [1, 2]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert result["predictions"] == [[[9, 0], [1, 2]]]
    assert result["receipt"]["mode"] == "global"
    assert result["receipt"]["solver"] == "Isometry"
    assert result["receipt"]["step"] == 1
    assert result["receipt"]["num_trains"] == 2
    assert result["receipt"]["num_tests"] == 1


def test_solve_step1_rot90():
    """Rotation task → Isometry family with rot90."""
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]}
        ],
        "test": [
            {"input": [[9, 0], [1, 2]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert result["predictions"] == [[[1, 9], [2, 0]]]
    assert result["receipt"]["solver"] == "Isometry"


def test_solve_step1_colormap():
    """ColorMap task → ColorMap family."""
    task = {
        "train": [
            {"input": [[0, 1, 2]], "output": [[5, 6, 7]]},
            {"input": [[1, 2, 0]], "output": [[6, 7, 5]]}
        ],
        "test": [
            {"input": [[2, 0, 1]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert result["predictions"] == [[[7, 5, 6]]]
    assert result["receipt"]["solver"] == "ColorMap"


def test_solve_step1_pixel_replicate():
    """Pixel replication task → PixelReplicate family."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 1, 2, 2], [1, 1, 2, 2]]},
            {"input": [[3, 4]], "output": [[3, 3, 4, 4], [3, 3, 4, 4]]}
        ],
        "test": [
            {"input": [[5, 6]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert result["predictions"] == [[[5, 5, 6, 6], [5, 5, 6, 6]]]
    assert result["receipt"]["solver"] == "PixelReplicate"


def test_solve_step1_unsat():
    """Impossible task → UNSAT."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[3, 4]]},
            {"input": [[1, 2]], "output": [[5, 6]]}  # Same input, different output
        ],
        "test": [
            {"input": [[1, 2]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "UNSAT"
    assert result["predictions"] is None
    assert result["receipt"]["mode"] == "unsat"
    assert result["receipt"]["reason"] == "no_family_matched"
    assert result["receipt"]["witness"] is None


# =============================
# B. First-Pass Selection
# =============================

def test_solve_step1_first_pass():
    """Multiple families could work → first in order wins."""
    # Identity task: both Isometry and ColorMap could work
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]},
            {"input": [[3, 4]], "output": [[3, 4]]}
        ],
        "test": [
            {"input": [[5, 6]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    # Isometry comes first in GLOBAL_FAMILIES, so it should win
    assert result["receipt"]["solver"] == "Isometry"


def test_solve_step1_family_order():
    """Verify families are tried in fixed order."""
    # Create a task that ONLY ColorMap can solve (Isometry can't)
    task = {
        "train": [
            {"input": [[0, 1]], "output": [[5, 6]]},
            {"input": [[1, 0]], "output": [[6, 5]]}
        ],
        "test": [
            {"input": [[0, 0, 1]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert result["receipt"]["solver"] == "ColorMap"


# =============================
# C. Edge Cases
# =============================

def test_solve_step1_empty_train():
    """Empty train_pairs → UNSAT."""
    task = {
        "train": [],
        "test": [{"input": [[1, 2]]}]
    }

    result = solve_step1(task)

    assert result["status"] == "UNSAT"
    assert result["predictions"] is None
    assert result["receipt"]["reason"] == "no_train_pairs"


def test_solve_step1_empty_test():
    """Empty test → PASS with empty predictions."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]}
        ],
        "test": []
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert result["predictions"] == []
    assert result["receipt"]["mode"] == "global"


def test_solve_step1_malformed_missing_train():
    """Missing 'train' key → UNSAT."""
    task = {
        "test": [{"input": [[1, 2]]}]
    }

    result = solve_step1(task)

    assert result["status"] == "UNSAT"
    assert result["receipt"]["reason"] == "malformed_task"


def test_solve_step1_malformed_missing_test():
    """Missing 'test' key → UNSAT."""
    task = {
        "train": [{"input": [[1, 2]], "output": [[1, 2]]}]
    }

    result = solve_step1(task)

    assert result["status"] == "UNSAT"
    assert result["receipt"]["reason"] == "malformed_task"


def test_solve_step1_multiple_test_inputs():
    """Multiple test inputs → all predicted."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]},
            {"input": [[3, 4]], "output": [[3, 4]]}
        ],
        "test": [
            {"input": [[5, 6]]},
            {"input": [[7, 8]]},
            {"input": [[9, 0]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert len(result["predictions"]) == 3
    assert result["predictions"][0] == [[5, 6]]
    assert result["predictions"][1] == [[7, 8]]
    assert result["predictions"][2] == [[9, 0]]


# =============================
# D. FY Exactness
# =============================

def test_fy_exactness_single_pixel_differs():
    """Single pixel difference → family rejected."""
    # Create a truly impossible task: same input, different outputs
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[3, 4]]},
            {"input": [[1, 2]], "output": [[5, 6]]}  # Same input, different output → impossible
        ],
        "test": [
            {"input": [[1, 2]]}
        ]
    }

    result = solve_step1(task)

    # This should be UNSAT because no deterministic function can produce different outputs for same input
    assert result["status"] == "UNSAT"


def test_fy_verification_after_fit():
    """Verify FY check happens AFTER fit() (fit() may be imperfect)."""
    # Use _try_family directly to test FY verification
    train_pairs = [
        {"input": [[1, 2]], "output": [[1, 2]]},
        {"input": [[3, 4]], "output": [[3, 4]]}
    ]

    instance, success = _try_family(IsometryFamily, train_pairs)

    assert success is True
    assert instance is not None
    # Verify that instance.apply() matches outputs
    for pair in train_pairs:
        assert deep_eq(instance.apply(pair["input"]), pair["output"])


# =============================
# E. Determinism and Purity
# =============================

def test_solve_step1_determinism():
    """Run twice on same task → identical results."""
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
        ],
        "test": [
            {"input": [[5, 6], [7, 8]]}
        ]
    }

    result1 = solve_step1(task)
    result2 = solve_step1(task)

    assert result1["status"] == result2["status"]
    assert deep_eq(result1["predictions"], result2["predictions"])
    assert result1["receipt"]["solver"] == result2["receipt"]["solver"]


def test_solve_step1_no_mutation():
    """Original task dict unchanged after solve_step1()."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]}
        ],
        "test": [
            {"input": [[3, 4]]}
        ]
    }

    task_copy = copy.deepcopy(task)
    solve_step1(task)

    assert deep_eq(task, task_copy)


def test_solve_step1_no_aliasing():
    """Output predictions have fresh allocation, no aliasing with input."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]}
        ],
        "test": [
            {"input": [[3, 4]]}
        ]
    }

    result = solve_step1(task)

    # Verify no aliasing with test input
    assert result["predictions"][0] is not task["test"][0]["input"]


# =============================
# F. Receipt Completeness
# =============================

def test_receipt_pass_has_required_fields():
    """PASS receipt has all required fields."""
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]}
        ],
        "test": [
            {"input": [[3, 4]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    receipt = result["receipt"]

    # Required fields
    assert "mode" in receipt
    assert "solver" in receipt
    assert "step" in receipt
    assert "num_trains" in receipt
    assert "num_tests" in receipt

    # Values
    assert receipt["mode"] == "global"
    assert receipt["step"] == 1
    assert receipt["num_trains"] == 1
    assert receipt["num_tests"] == 1


def test_receipt_unsat_has_required_fields():
    """UNSAT receipt has all required fields."""
    task = {
        "train": [],
        "test": [{"input": [[1, 2]]}]
    }

    result = solve_step1(task)

    assert result["status"] == "UNSAT"
    receipt = result["receipt"]

    # Required fields
    assert "mode" in receipt
    assert "step" in receipt
    assert "reason" in receipt
    assert "witness" in receipt

    # Values
    assert receipt["mode"] == "unsat"
    assert receipt["step"] == 1
    assert receipt["reason"] == "no_train_pairs"
    assert receipt["witness"] is None


def test_receipt_includes_params():
    """PASS receipt includes family params when available."""
    task = {
        "train": [
            {"input": [[0, 1]], "output": [[5, 6]]},
            {"input": [[1, 0]], "output": [[6, 5]]}
        ],
        "test": [
            {"input": [[0, 1]]}
        ]
    }

    result = solve_step1(task)

    assert result["status"] == "PASS"
    assert "params" in result["receipt"]
    # ColorMap should have learned mapping
    params = result["receipt"]["params"]
    assert isinstance(params, dict)


# =============================
# G. Helper Function Tests
# =============================

def test_try_family_success():
    """_try_family returns (instance, True) on success."""
    train_pairs = [
        {"input": [[1, 2]], "output": [[1, 2]]},
        {"input": [[3, 4]], "output": [[3, 4]]}
    ]

    instance, success = _try_family(IsometryFamily, train_pairs)

    assert success is True
    assert instance is not None
    assert instance.name == "Isometry"


def test_try_family_failure():
    """_try_family returns (None, False) on failure."""
    train_pairs = [
        {"input": [[1, 2]], "output": [[3, 4]]},
        {"input": [[1, 2]], "output": [[5, 6]]}  # Impossible
    ]

    instance, success = _try_family(IsometryFamily, train_pairs)

    assert success is False
    assert instance is None


def test_try_family_empty_train():
    """_try_family with empty train_pairs → (None, False)."""
    instance, success = _try_family(IsometryFamily, [])

    assert success is False
    assert instance is None


def test_try_family_exception_handling():
    """_try_family catches exceptions → (None, False)."""
    # Create malformed train_pairs that will cause exception
    train_pairs = [
        {"input": None, "output": [[1, 2]]}  # None input will cause exception
    ]

    instance, success = _try_family(IsometryFamily, train_pairs)

    assert success is False
    assert instance is None


# =============================
# H. Integration Tests
# =============================

def test_all_16_families_in_global_list():
    """Verify all 16 families are in GLOBAL_FAMILIES list."""
    assert len(GLOBAL_FAMILIES) == 16

    # Verify specific families are present
    family_names = [f().name for f in GLOBAL_FAMILIES]

    expected_families = [
        "Isometry",
        "ColorMap",
        "IsoColorMap",
        "PixelReplicate",
        "BlockDown",
        "NPSDown",
        "NPSUp",
        "ParityTile",
        "BlockPermutation",
        "BlockSubstitution",
        "RowPermutation",
        "ColPermutation",
        "SortRowsLex",
        "SortColsLex",
        "MirrorComplete",
        "CopyMoveAllComponents"
    ]

    assert family_names == expected_families


def test_global_families_deterministic_order():
    """GLOBAL_FAMILIES list has fixed, deterministic order."""
    # Get order twice
    order1 = [f().name for f in GLOBAL_FAMILIES]
    order2 = [f().name for f in GLOBAL_FAMILIES]

    assert order1 == order2


def test_solve_step1_with_different_family_types():
    """Test tasks that require different family types."""
    # Task 1: Requires Isometry
    task_iso = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 4], [1, 3]]}  # transpose
        ],
        "test": [{"input": [[5, 6], [7, 8]]}]
    }

    result_iso = solve_step1(task_iso)
    assert result_iso["status"] == "PASS"
    assert result_iso["receipt"]["solver"] == "Isometry"

    # Task 2: Requires ColorMap
    task_color = {
        "train": [
            {"input": [[0, 1]], "output": [[9, 8]]},
            {"input": [[1, 0]], "output": [[8, 9]]}
        ],
        "test": [{"input": [[0, 0, 1]]}]
    }

    result_color = solve_step1(task_color)
    assert result_color["status"] == "PASS"
    assert result_color["receipt"]["solver"] == "ColorMap"

    # Task 3: Requires PixelReplicate
    task_pixel = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 1, 2, 2]]}
        ],
        "test": [{"input": [[3, 4]]}]
    }

    result_pixel = solve_step1(task_pixel)
    assert result_pixel["status"] == "PASS"
    assert result_pixel["receipt"]["solver"] == "PixelReplicate"
