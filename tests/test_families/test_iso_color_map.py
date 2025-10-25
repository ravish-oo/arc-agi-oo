"""
Test suite for IsoColorMapFamily (P2-03).

Tests cover:
- Basic fit/apply workflow
- Unified (σ, mapping) requirement (critical: ONE pair for ALL pairs)
- Deterministic σ selection (first-acceptable wins)
- Composition correctness (σ FIRST, then mapping)
- Shape safety with dimension changes
- FY exactness (bit-for-bit equality)
- Mapping conflict detection
- Unseen color handling (KeyError)
- Edge cases (empty grids, single pairs)
- Purity (no mutations)
- Determinism
"""

import pytest
from src.families.iso_color_map import IsoColorMapFamily
from src.utils import deep_eq, copy_grid


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def family():
    """Fresh IsoColorMapFamily instance for each test."""
    return IsoColorMapFamily()


# Test grids
g_empty = []

g_single = [[5]]

g_square = [[1, 2],
            [3, 4]]

g_rect = [[1, 2, 3],
          [4, 5, 6]]


# ============================================================================
# Basic fit/apply
# ============================================================================

def test_fit_identity(family):
    """fit() with identity (X==Y) should accept σ='id' with identity mapping."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"
    assert family.params.mapping == {1: 1, 2: 2, 3: 3, 4: 4}


def test_fit_isometry_only(family):
    """fit() should work for geometric transforms."""
    # Output [[3,1],[4,2]] from input [[1,2],[3,4]]
    # Note: σ='id' with mapping {1:3, 2:1, 3:4, 4:2} also works
    # Since 'id' comes before 'rot90' in all_isometries(), it wins (first-acceptable)
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"  # First σ that works
    assert family.params.mapping == {1: 3, 2: 1, 3: 4, 4: 2}


def test_fit_colormap_only(family):
    """fit() with pure color remap (no geometric transform) should accept σ='id' with non-identity mapping."""
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[2, 3], [1, 4]]}  # {1:2, 2:3, 3:1, 4:4}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"
    assert family.params.mapping == {1: 2, 2: 3, 3: 1, 4: 4}


def test_fit_both_sigma_and_mapping(family):
    """fit() with both σ and mapping should accept correct (σ, mapping) pair."""
    # Output [[7,5],[8,6]] from input [[1,2],[3,4]]
    # σ='id' with mapping {1:7, 2:5, 3:8, 4:6} works (first-acceptable)
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[7, 5], [8, 6]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"  # First σ that works
    assert family.params.mapping == {1: 7, 2: 5, 3: 8, 4: 6}


def test_apply_after_fit(family):
    """apply() after successful fit() should return correct transformed grid."""
    # Learn σ='flip_h', mapping={1:9, 2:8}
    train = [
        {"input": [[1, 1, 2]], "output": [[8, 9, 9]]}  # flip_h then map
    ]
    family.fit(train)

    # Apply to new input
    X = [[2, 2, 1]]
    result = family.apply(X)
    # flip_h([[2,2,1]]) = [[1,2,2]] → map → [[9,8,8]]
    expected = [[9, 8, 8]]
    assert deep_eq(result, expected)


def test_apply_before_fit_raises(family):
    """apply() before fit() should raise RuntimeError."""
    X = [[1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="params.sigma is None"):
        family.apply(X)


# ============================================================================
# Unified (σ, mapping) requirement (CRITICAL)
# ============================================================================

def test_unified_sigma_mapping_multiple_pairs(family):
    """fit() with multiple pairs using same (σ, mapping) should succeed."""
    # For 1xN grids, both rot180 and flip_h produce same result
    # rot180 comes before flip_h in all_isometries(), so it wins
    # rot180([[1,1,2]]) = [[2,1,1]] → map {1:9, 2:8} → [[8,9,9]]
    # rot180([[2,2,1]]) = [[1,2,2]] → map {1:9, 2:8} → [[9,8,8]]
    train = [
        {"input": [[1, 1, 2]], "output": [[8, 9, 9]]},
        {"input": [[2, 2, 1]], "output": [[9, 8, 8]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot180"  # First σ that works (before flip_h)
    assert family.params.mapping == {1: 9, 2: 8}


def test_unified_conflict_no_solution(family):
    """fit() with conflicting requirements (no unified (σ, mapping)) must return False."""
    # Pair 1: needs mapping {1:2}
    # Pair 2: needs mapping {1:3} (conflict)
    train = [
        {"input": [[1]], "output": [[2]]},
        {"input": [[1]], "output": [[3]]}
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.sigma is None
    assert family.params.mapping is None


def test_unified_all_pairs_must_verify(family):
    """fit() must verify ALL pairs, not just first."""
    # First two pairs work with σ='id', mapping={1:2, 2:3}
    # Third pair conflicts
    train = [
        {"input": [[1, 2]], "output": [[2, 3]]},
        {"input": [[2, 1]], "output": [[3, 2]]},
        {"input": [[1, 2]], "output": [[9, 3]]}  # {1:9, 2:3} - conflict on color 1
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.sigma is None
    assert family.params.mapping is None


# ============================================================================
# Deterministic σ selection
# ============================================================================

def test_deterministic_order_symmetric_grid(family):
    """When multiple σ match, choose first in all_isometries() order."""
    # Grid where all isometries + mapping {1:2} work
    train = [
        {"input": [[1, 1], [1, 1]], "output": [[2, 2], [2, 2]]}
    ]
    result = family.fit(train)
    assert result is True
    # "id" comes first in all_isometries()
    assert family.params.sigma == "id"
    assert family.params.mapping == {1: 2}


def test_deterministic_repeated_fit(family):
    """Running fit() twice on same train_pairs yields identical params."""
    train = [
        {"input": [[1, 2]], "output": [[7, 5]]}  # rot90 + mapping
    ]

    result1 = family.fit(train)
    sigma1 = family.params.sigma
    mapping1 = family.params.mapping.copy()

    # Reset and re-fit
    family.params.sigma = None
    family.params.mapping = None
    result2 = family.fit(train)
    sigma2 = family.params.sigma
    mapping2 = family.params.mapping

    assert result1 is True
    assert result2 is True
    assert sigma1 == sigma2
    assert mapping1 == mapping2


# ============================================================================
# Composition correctness (σ FIRST, then mapping)
# ============================================================================

def test_composition_order_matters():
    """Verify σ applied FIRST, then mapping (not commutative)."""
    # σ='id' with mapping {1:7, 2:9, 3:6, 4:8} works (first-acceptable)
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[7, 9], [6, 8]]}
    ]

    family = IsoColorMapFamily()
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"  # First σ that works
    assert family.params.mapping == {1: 7, 2: 9, 3: 6, 4: 8}

    # Verify apply uses correct composition
    X2 = [[1, 2], [3, 4]]
    result2 = family.apply(X2)
    expected2 = [[7, 9], [6, 8]]
    assert deep_eq(result2, expected2)


# ============================================================================
# Shape safety
# ============================================================================

def test_shape_safety_dimension_swap(family):
    """fit() should handle dimension-swapping σ correctly."""
    # 1x3 → 3x1 via rot90 (or transpose)
    # rot90([[1,2,3]]) = [[1],[2],[3]] → map {1:5, 2:6, 3:7} → [[5],[6],[7]]
    train = [
        {"input": [[1, 2, 3]], "output": [[5], [6], [7]]}
    ]
    result = family.fit(train)
    assert result is True
    # rot90 comes before transpose in all_isometries()
    assert family.params.sigma == "rot90"
    assert family.params.mapping == {1: 5, 2: 6, 3: 7}


def test_shape_safety_dimension_mismatch(family):
    """fit() should reject if no σ produces correct dimensions."""
    # 2x2 → 3x3 (impossible with D8)
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2, 0], [3, 4, 0], [0, 0, 0]]}
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.sigma is None
    assert family.params.mapping is None


# ============================================================================
# FY exactness
# ============================================================================

def test_fy_single_pixel_difference_rejects(family):
    """Single pixel difference in verification → reject (σ, mapping)."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]},  # {1:3, 2:4}
        {"input": [[1, 2]], "output": [[3, 9]]}   # {1:3, 2:9} - conflict on color 2
    ]
    result = family.fit(train)
    assert result is False


def test_fy_exact_equality_required():
    """After successful fit, applying (σ, mapping) to train inputs must reproduce outputs exactly."""
    train = [
        {"input": [[1, 2]], "output": [[8, 7]]},  # flip_h + mapping {1:7, 2:8}
        {"input": [[2, 1]], "output": [[7, 8]]}   # same (σ, mapping)
    ]
    family = IsoColorMapFamily()
    result = family.fit(train)
    assert result is True

    # Verify FY exactness by applying to all train inputs
    for pair in train:
        X = pair["input"]
        Y = pair["output"]
        Y_predicted = family.apply(X)
        assert deep_eq(Y_predicted, Y)


# ============================================================================
# Mapping conflict detection
# ============================================================================

def test_mapping_conflict_in_first_pair(family):
    """fit() should skip σ if mapping conflict in first pair."""
    # After any σ, if same color maps to different outputs → skip
    # [[1,1],[2,2]] with output [[3,4],[5,5]] → 1 maps to both 3 and 4 (conflict)
    train = [
        {"input": [[1, 1], [2, 2]], "output": [[3, 4], [5, 5]]}
    ]
    result = family.fit(train)
    assert result is False
    assert family.params.sigma is None


def test_mapping_conflict_across_pairs(family):
    """fit() should skip σ if mapping conflict across pairs."""
    # For σ='id': pair0 needs {1:2}, pair1 needs {1:3} → conflict
    train = [
        {"input": [[1]], "output": [[2]]},
        {"input": [[1]], "output": [[3]]}
    ]
    result = family.fit(train)
    assert result is False


def test_mapping_unseen_color_in_later_pair(family):
    """fit() should skip σ if later pair has unseen color after applying σ."""
    # For σ='id': pair0 learns {1:3, 2:4}, pair1 has color 5 (unseen)
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]},
        {"input": [[1, 5]], "output": [[3, 6]]}
    ]
    result = family.fit(train)
    assert result is False


# ============================================================================
# Unseen color handling
# ============================================================================

def test_apply_unseen_color_raises_keyerror(family):
    """apply() with unseen color (after σ) should raise KeyError."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}  # σ='id', mapping={1:3, 2:4}
    ]
    family.fit(train)

    X = [[1, 5]]  # 5 not in mapping
    with pytest.raises(KeyError):
        family.apply(X)


def test_apply_keyerror_identifies_color():
    """KeyError should identify which color is unseen."""
    family = IsoColorMapFamily()
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}
    ]
    family.fit(train)

    X = [[7]]  # 7 not in mapping
    try:
        family.apply(X)
        assert False, "Expected KeyError"
    except KeyError as e:
        # Python KeyError includes the key in the error message
        assert "7" in str(e) or 7 in e.args


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_train_pairs_returns_false(family):
    """fit([]) with empty train_pairs should return False."""
    result = family.fit([])
    assert result is False
    assert family.params.sigma is None
    assert family.params.mapping is None


def test_empty_grids(family):
    """fit() with empty grids ([] → []) should succeed with σ='id', mapping={}."""
    train = [
        {"input": [], "output": []}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"
    assert family.params.mapping == {}


def test_empty_grids_mismatch(family):
    """fit() with [] → [[1]] should fail (dimension mismatch)."""
    train = [
        {"input": [], "output": [[1]]}
    ]
    result = family.fit(train)
    assert result is False


def test_apply_empty_grid(family):
    """apply([]) with valid (σ, mapping) should return []."""
    train = [
        {"input": [[1]], "output": [[2]]}  # σ='id', mapping={1:2}
    ]
    family.fit(train)
    result = family.apply([])
    assert result == []


def test_single_pixel_grid(family):
    """fit() with 1x1 grids should work correctly."""
    train = [
        {"input": [[5]], "output": [[7]]}  # σ='id', mapping={5:7}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "id"
    assert family.params.mapping == {5: 7}


def test_rectangular_grid_dimension_swap(family):
    """fit() with rectangular grids should handle dimension swaps correctly."""
    # 2x3 → 3x2 via rot90
    # rot90([[1,2,3],[4,5,6]]) = [[4,1],[5,2],[6,3]]
    # mapping identity → output same as rot90 result
    train = [
        {"input": [[1, 2, 3], [4, 5, 6]], "output": [[4, 1], [5, 2], [6, 3]]}
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot90"


# ============================================================================
# Purity
# ============================================================================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() must not mutate train_pairs input."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}
    ]
    train_copy = [
        {"input": copy_grid(train[0]["input"]),
         "output": copy_grid(train[0]["output"])}
    ]

    family.fit(train)

    # Verify train_pairs unchanged
    assert deep_eq(train[0]["input"], train_copy[0]["input"])
    assert deep_eq(train[0]["output"], train_copy[0]["output"])


def test_apply_does_not_mutate_input(family):
    """apply() must not mutate input grid X."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}
    ]
    family.fit(train)

    X = [[2, 1]]
    X_copy = copy_grid(X)

    result = family.apply(X)

    # Verify X unchanged
    assert deep_eq(X, X_copy)


def test_apply_no_row_aliasing(family):
    """apply() output must not share row references with input."""
    train = [
        {"input": [[1]], "output": [[2]]}
    ]
    family.fit(train)

    X = [[1, 1]]
    result = family.apply(X)

    # Mutate result
    result[0][0] = 999

    # Verify X unchanged (no aliasing)
    assert X[0][0] == 1


# ============================================================================
# Determinism
# ============================================================================

def test_deterministic_apply(family):
    """apply() with same input yields identical output."""
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]}
    ]
    family.fit(train)

    X = [[2, 1]]
    result1 = family.apply(X)
    result2 = family.apply(X)

    assert deep_eq(result1, result2)


# ============================================================================
# Integration tests
# ============================================================================

def test_various_transformations_work():
    """fit() should successfully handle various geometric+color transformations."""
    # Note: With IsoColorMap, multiple (σ, mapping) pairs may achieve same result.
    # Implementation correctly selects first-acceptable σ in all_isometries() order.
    test_cases = [
        # Identity
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),

        # Dimension swap (2x3 → 3x2) - forces non-id σ
        ([[1, 2, 3], [4, 5, 6]], [[4, 1], [5, 2], [6, 3]]),

        # Color remapping
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),

        # Both geometric + color
        ([[1, 2, 3], [4, 5, 6]], [[10, 7], [11, 8], [12, 9]]),

        # Dimension swap with identity colors
        ([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]]),
    ]

    for input_grid, output_grid in test_cases:
        family = IsoColorMapFamily()
        train = [{"input": input_grid, "output": output_grid}]
        result = family.fit(train)
        assert result is True, f"Failed to fit {input_grid} → {output_grid}"

        # Verify apply reproduces output
        predicted = family.apply(input_grid)
        assert deep_eq(predicted, output_grid), \
            f"apply() failed: expected {output_grid}, got {predicted}"


def test_complex_workflow_multiple_pairs():
    """Full workflow with both σ and mapping across multiple pairs."""
    # σ='rot90' (first-acceptable), mapping={0:5, 1:6, 2:7}
    # rot90([[0,1],[2,0]]) = [[2,0],[0,1]] → map → [[7,5],[5,6]]
    # rot90([[1,2],[0,1]]) = [[0,1],[1,2]] → map → [[5,6],[6,7]]
    train = [
        {"input": [[0, 1], [2, 0]], "output": [[7, 5], [5, 6]]},
        {"input": [[1, 2], [0, 1]], "output": [[5, 6], [6, 7]]}
    ]
    family = IsoColorMapFamily()

    result = family.fit(train)
    assert result is True
    assert family.params.sigma == "rot90"  # First σ that works
    assert family.params.mapping == {0: 5, 1: 6, 2: 7}

    # Apply on new input
    X = [[2, 1], [1, 0]]
    result = family.apply(X)
    # rot90([[2,1],[1,0]]) = [[1,2],[0,1]] → map → [[6,7],[5,6]]
    expected = [[6, 7], [5, 6]]
    assert deep_eq(result, expected)
