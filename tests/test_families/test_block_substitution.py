"""
Tests for BlockSubstitution Family.

Covers:
- fit() success and failure cases
- apply() with various glyph sizes
- Edge cases (empty, identity, uniform color)
- Square expansion requirement (kH == kW)
- Integer expansion factors
- Glyph consistency within and across pairs
- Color completeness
- Deterministic color iteration
- FY exactness verification
- Determinism and purity
"""

import pytest
from src.families.block_substitution import BlockSubstitutionFamily


@pytest.fixture
def family():
    """Create fresh BlockSubstitutionFamily instance for each test."""
    return BlockSubstitutionFamily()


# ============================================================================
# Basic fit() Tests
# ============================================================================

def test_fit_empty_train_pairs(family):
    """fit() with empty train_pairs returns False."""
    result = family.fit([])
    assert result is False
    assert family.params.k is None
    assert family.params.glyphs is None


def test_fit_empty_grids(family):
    """fit() with empty grids returns False."""
    train = [{"input": [], "output": []}]
    result = family.fit(train)
    assert result is False


def test_fit_zero_dimensions(family):
    """fit() rejects grids with zero dimensions."""
    train = [{"input": [[]], "output": [[]]}]
    result = family.fit(train)
    assert result is False


def test_fit_identity_case(family):
    """fit() with k=1 (identity case)."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.k == 1
    # Each color maps to 1×1 glyph
    assert family.params.glyphs == {1: [[1]], 2: [[2]], 3: [[3]], 4: [[4]]}


def test_fit_simple_k2_binary(family):
    """fit() with k=2, binary colors."""
    train = [
        {
            "input": [[0, 1], [1, 0]],
            "output": [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0]
            ]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.k == 2
    assert family.params.glyphs == {
        0: [[0, 0], [0, 0]],
        1: [[1, 1], [1, 1]]
    }


def test_fit_k3_cross_pattern(family):
    """fit() with k=3, cross pattern glyphs."""
    train = [
        {
            "input": [[0, 1], [1, 0]],
            "output": [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.k == 3
    assert family.params.glyphs == {
        0: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        1: [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    }


# ============================================================================
# Square Expansion Requirement Tests
# ============================================================================

def test_reject_non_square_expansion_different_factors(family):
    """fit() rejects when kH != kW (non-square expansion)."""
    train = [
        {
            "input": [[1, 2]],
            "output": [
                [11, 12, 21],
                [13, 14, 22]
            ]
        }
    ]
    # hx=1, wx=2, hy=2, wy=3
    # kH = 2/1 = 2, kW = 3/2 = 1.5 (non-integer)
    result = family.fit(train)
    assert result is False


def test_reject_non_square_expansion_integer_but_different(family):
    """fit() rejects when kH and kW are both integers but different."""
    train = [
        {
            "input": [[1, 2], [3, 4], [5, 6]],
            "output": [
                [11, 12, 13, 14, 21, 22],
                [15, 16, 17, 18, 23, 24],
                [31, 32, 33, 34, 41, 42],
                [35, 36, 37, 38, 43, 44],
                [51, 52, 53, 54, 61, 62],
                [55, 56, 57, 58, 63, 64]
            ]
        }
    ]
    # hx=3, wx=2, hy=6, wy=6
    # kH = 6/3 = 2, kW = 6/2 = 3 (both integer but kH != kW)
    result = family.fit(train)
    assert result is False


def test_accept_square_expansion_k2(family):
    """fit() accepts when kH = kW = 2."""
    train = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [
                [11, 12, 21, 22],
                [13, 14, 23, 24],
                [31, 32, 41, 42],
                [33, 34, 43, 44]
            ]
        }
    ]
    # hx=2, wx=2, hy=4, wy=4
    # kH = 4/2 = 2, kW = 4/2 = 2 (square expansion)
    result = family.fit(train)
    assert result is True
    assert family.params.k == 2


def test_accept_square_expansion_k3(family):
    """fit() accepts when kH = kW = 3."""
    train = [
        {
            "input": [[1, 2]],
            "output": [
                [11, 12, 13, 21, 22, 23],
                [14, 15, 16, 24, 25, 26],
                [17, 18, 19, 27, 28, 29]
            ]
        }
    ]
    # hx=1, wx=2, hy=3, wy=6
    # kH = 3/1 = 3, kW = 6/2 = 3 (square expansion)
    result = family.fit(train)
    assert result is True
    assert family.params.k == 3


# ============================================================================
# Integer Expansion Factor Tests
# ============================================================================

def test_reject_non_integer_vertical_expansion(family):
    """fit() rejects when hy % hx != 0."""
    train = [
        {
            "input": [[1], [2], [3]],
            "output": [[11], [12], [13], [14], [21], [22], [23]]
        }
    ]
    # hx=3, hy=7 → 7 % 3 = 1 != 0
    result = family.fit(train)
    assert result is False


def test_reject_non_integer_horizontal_expansion(family):
    """fit() rejects when wy % wx != 0."""
    train = [
        {
            "input": [[1, 2, 3]],
            "output": [[11, 12, 21, 22]]
        }
    ]
    # wx=3, wy=4 → 4 % 3 = 1 != 0
    result = family.fit(train)
    assert result is False


def test_accept_integer_expansion_k4(family):
    """fit() accepts integer expansion k=4."""
    train = [
        {
            "input": [[1]],
            "output": [
                [11, 12, 13, 14],
                [15, 16, 17, 18],
                [19, 20, 21, 22],
                [23, 24, 25, 26]
            ]
        }
    ]
    # hx=1, wx=1, hy=4, wy=4 → k=4
    result = family.fit(train)
    assert result is True
    assert family.params.k == 4


# ============================================================================
# Glyph Consistency Within Pair Tests
# ============================================================================

def test_reject_inconsistent_glyphs_same_color(family):
    """fit() rejects when same color maps to different glyphs within a pair."""
    train = [
        {
            "input": [[1, 1]],
            "output": [
                [11, 12, 21, 22],
                [13, 14, 23, 24]
            ]
        }
    ]
    # First 1 → [[11,12],[13,14]], second 1 → [[21,22],[23,24]] (different)
    result = family.fit(train)
    assert result is False


def test_accept_consistent_glyphs_same_color(family):
    """fit() accepts when same color maps to same glyph everywhere."""
    train = [
        {
            "input": [[1, 1, 1]],
            "output": [
                [11, 12, 11, 12, 11, 12],
                [13, 14, 13, 14, 13, 14]
            ]
        }
    ]
    # All three 1s map to same glyph [[11,12],[13,14]]
    result = family.fit(train)
    assert result is True
    assert family.params.k == 2
    assert family.params.glyphs == {1: [[11, 12], [13, 14]]}


def test_reject_partial_glyph_mismatch(family):
    """fit() rejects when glyphs differ in a single pixel."""
    train = [
        {
            "input": [[2, 2]],
            "output": [
                [5, 5, 5, 5],
                [5, 5, 5, 9]  # Second glyph has different bottom-right
            ]
        }
    ]
    # First 2 → [[5,5],[5,5]], second 2 → [[5,5],[5,9]] (differ in [1][1])
    result = family.fit(train)
    assert result is False


# ============================================================================
# Glyph Consistency Across Pairs Tests
# ============================================================================

def test_accept_multi_pair_consistent_glyphs(family):
    """fit() accepts when glyphs learned from first pair work for all pairs."""
    train = [
        {
            "input": [[0, 1], [1, 0]],
            "output": [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0]
            ]
        },
        {
            "input": [[1, 0], [0, 1]],
            "output": [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1]
            ]
        }
    ]
    # Same glyphs work for both pairs
    result = family.fit(train)
    assert result is True
    assert family.params.k == 2
    assert family.params.glyphs == {0: [[0, 0], [0, 0]], 1: [[1, 1], [1, 1]]}


def test_reject_multi_pair_inconsistent_glyphs(family):
    """fit() rejects when pair2 uses different glyph for same color."""
    train = [
        {
            "input": [[1]],
            "output": [[11, 12], [13, 14]]
        },
        {
            "input": [[1]],
            "output": [[21, 22], [23, 24]]  # Different glyph
        }
    ]
    # First pair: glyphs={1: [[11,12],[13,14]]}
    # Second pair: expects [[21,22],[23,24]] but glyphs has different pattern
    result = family.fit(train)
    assert result is False


# ============================================================================
# Color Completeness Tests
# ============================================================================

def test_reject_new_color_in_later_pair(family):
    """fit() rejects when later pair has color not in first pair."""
    train = [
        {
            "input": [[0, 1]],
            "output": [[0, 0, 1, 1], [0, 0, 1, 1]]
        },
        {
            "input": [[2]],
            "output": [[2, 2], [2, 2]]
        }
    ]
    # Color 2 appears in second pair but not in first pair
    result = family.fit(train)
    assert result is False


def test_accept_all_colors_in_first_pair(family):
    """fit() accepts when all colors appear in first pair."""
    train = [
        {
            "input": [[0, 1, 2]],
            "output": [
                [00, 00, 11, 11, 22, 22],
                [00, 00, 11, 11, 22, 22]
            ]
        },
        {
            "input": [[1, 0]],
            "output": [[11, 11, 00, 00], [11, 11, 00, 00]]
        }
    ]
    # All colors {0, 1, 2} in first pair; second pair only uses {0, 1}
    result = family.fit(train)
    assert result is True


# ============================================================================
# apply() Tests
# ============================================================================

def test_apply_before_fit_raises(family):
    """apply() before fit() raises RuntimeError."""
    with pytest.raises(RuntimeError, match="apply\\(\\) called before fit\\(\\)"):
        family.apply([[1, 2]])


def test_apply_after_successful_fit(family):
    """apply() works correctly after successful fit()."""
    train = [
        {
            "input": [[0, 1]],
            "output": [[0, 0, 1, 1], [0, 0, 1, 1]]
        }
    ]
    family.fit(train)

    # Apply to original input
    result = family.apply([[0, 1]])
    assert result == [[0, 0, 1, 1], [0, 0, 1, 1]]


def test_apply_to_different_input(family):
    """apply() applies learned glyphs to different input."""
    train = [
        {
            "input": [[0, 1]],
            "output": [[0, 0, 1, 1], [0, 0, 1, 1]]
        }
    ]
    family.fit(train)

    # Apply to different arrangement
    result = family.apply([[1, 0, 1]])
    assert result == [[1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1]]


def test_apply_with_unseen_color_raises_keyerror(family):
    """apply() with color not in glyphs raises KeyError."""
    train = [
        {
            "input": [[0, 1]],
            "output": [[0, 0, 1, 1], [0, 0, 1, 1]]
        }
    ]
    family.fit(train)

    # Try to apply with color 2 (not in glyphs)
    with pytest.raises(KeyError, match="Color 2 not found"):
        family.apply([[2]])


def test_apply_empty_grid(family):
    """apply() handles empty grid correctly."""
    train = [
        {
            "input": [[1]],
            "output": [[11, 12], [13, 14]]
        }
    ]
    family.fit(train)

    result = family.apply([])
    assert result == []


def test_apply_output_dimensions(family):
    """apply() output has correct dimensions."""
    train = [
        {
            "input": [[1]],
            "output": [[11, 12, 13], [14, 15, 16], [17, 18, 19]]
        }
    ]
    family.fit(train)

    # Apply to 2×3 input
    result = family.apply([[1, 1, 1], [1, 1, 1]])
    # Expected: (2*3, 3*3) = (6, 9)
    assert len(result) == 6
    assert len(result[0]) == 9


# ============================================================================
# Deterministic Color Iteration Tests
# ============================================================================

def test_deterministic_color_iteration_order(family):
    """fit() processes colors in ascending sorted order."""
    train = [
        {
            "input": [[9, 3, 7, 1]],
            "output": [
                [90, 91, 30, 31, 70, 71, 10, 11],
                [92, 93, 32, 33, 72, 73, 12, 13]
            ]
        }
    ]
    result = family.fit(train)
    assert result is True

    # Colors should be processed in sorted order: [1, 3, 7, 9]
    # Verify glyphs dict has all colors
    assert set(family.params.glyphs.keys()) == {1, 3, 7, 9}


def test_deterministic_repeated_fit(family):
    """Repeated fit() calls yield identical params."""
    train = [
        {
            "input": [[5, 2, 8]],
            "output": [
                [50, 51, 20, 21, 80, 81],
                [52, 53, 22, 23, 82, 83]
            ]
        }
    ]

    # First fit
    family.fit(train)
    k1 = family.params.k
    glyphs1 = {c: [row[:] for row in g] for c, g in family.params.glyphs.items()}

    # Second fit (fresh instance)
    family2 = BlockSubstitutionFamily()
    family2.fit(train)

    assert family2.params.k == k1
    assert set(family2.params.glyphs.keys()) == set(glyphs1.keys())
    for c in glyphs1:
        assert family2.params.glyphs[c] == glyphs1[c]


# ============================================================================
# Glyph Extraction Tests
# ============================================================================

def test_extract_glyph_k1(family):
    """_extract_glyph with k=1 extracts single pixel."""
    Y = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
    glyph = family._extract_glyph(Y, 1, 2, 1)
    assert glyph == [[23]]


def test_extract_glyph_k2(family):
    """_extract_glyph with k=2 extracts 2×2 block."""
    Y = [
        [11, 12, 21, 22],
        [13, 14, 23, 24],
        [31, 32, 41, 42],
        [33, 34, 43, 44]
    ]
    glyph = family._extract_glyph(Y, 0, 1, 2)
    assert glyph == [[21, 22], [23, 24]]


def test_extract_glyph_k3_at_corners(family):
    """_extract_glyph with k=3 at grid corners."""
    Y = [
        [11, 12, 13, 21, 22, 23],
        [14, 15, 16, 24, 25, 26],
        [17, 18, 19, 27, 28, 29],
        [31, 32, 33, 41, 42, 43],
        [34, 35, 36, 44, 45, 46],
        [37, 38, 39, 47, 48, 49]
    ]

    # Top-left
    glyph_tl = family._extract_glyph(Y, 0, 0, 3)
    assert glyph_tl == [[11, 12, 13], [14, 15, 16], [17, 18, 19]]

    # Bottom-right
    glyph_br = family._extract_glyph(Y, 1, 1, 3)
    assert glyph_br == [[41, 42, 43], [44, 45, 46], [47, 48, 49]]


# ============================================================================
# FY Exactness Tests
# ============================================================================

def test_fy_exact_equality_required(family):
    """fit() rejects if output differs by single pixel."""
    train = [
        {
            "input": [[1]],
            "output": [[11, 12], [13, 14]]
        },
        {
            "input": [[1]],
            "output": [[11, 12], [13, 99]]  # Single pixel different
        }
    ]
    result = family.fit(train)
    assert result is False


def test_fy_all_pairs_must_match(family):
    """fit() requires ALL pairs to match (not just some)."""
    train = [
        {
            "input": [[1]],
            "output": [[11, 12], [13, 14]]
        },
        {
            "input": [[1]],
            "output": [[11, 12], [13, 14]]
        },
        {
            "input": [[1]],
            "output": [[21, 22], [23, 24]]  # Third pair different
        }
    ]
    result = family.fit(train)
    assert result is False


# ============================================================================
# Edge Cases
# ============================================================================

def test_single_pixel_input(family):
    """fit() and apply() with 1×1 input."""
    train = [
        {
            "input": [[5]],
            "output": [[51, 52], [53, 54]]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.k == 2

    # Apply returns the single glyph
    result_apply = family.apply([[5]])
    assert result_apply == [[51, 52], [53, 54]]


def test_uniform_color_input(family):
    """fit() and apply() with all pixels same color."""
    train = [
        {
            "input": [[7, 7], [7, 7]],
            "output": [
                [71, 72, 71, 72],
                [73, 74, 73, 74],
                [71, 72, 71, 72],
                [73, 74, 73, 74]
            ]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.k == 2
    assert family.params.glyphs == {7: [[71, 72], [73, 74]]}


def test_many_colors(family):
    """fit() handles many colors correctly."""
    train = [
        {
            "input": [[0, 1, 2, 3, 4]],
            "output": [
                [00, 00, 11, 11, 22, 22, 33, 33, 44, 44],
                [00, 00, 11, 11, 22, 22, 33, 33, 44, 44]
            ]
        }
    ]
    # hx=1, wx=5, hy=2, wy=10 → k=2 (square expansion)
    result = family.fit(train)
    assert result is True
    assert family.params.k == 2
    assert len(family.params.glyphs) == 5


def test_large_k(family):
    """fit() and apply() with large k."""
    # k=5 would create 5×5 glyphs
    train = [
        {
            "input": [[1]],
            "output": [
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
                [26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35]
            ]
        }
    ]
    result = family.fit(train)
    assert result is True
    assert family.params.k == 5


# ============================================================================
# Purity Tests
# ============================================================================

def test_fit_does_not_mutate_train_pairs(family):
    """fit() does not mutate train_pairs."""
    original_input = [[1, 2], [3, 4]]
    original_output = [[11, 12, 21, 22], [13, 14, 23, 24], [31, 32, 41, 42], [33, 34, 43, 44]]
    train = [{"input": original_input, "output": original_output}]

    # Store original values
    input_copy = [row[:] for row in original_input]
    output_copy = [row[:] for row in original_output]

    family.fit(train)

    # Check no mutation
    assert train[0]["input"] == input_copy
    assert train[0]["output"] == output_copy


def test_apply_does_not_mutate_input(family):
    """apply() does not mutate input grid."""
    train = [
        {
            "input": [[1]],
            "output": [[11, 12], [13, 14]]
        }
    ]
    family.fit(train)

    X = [[1, 1]]
    X_copy = [row[:] for row in X]

    family.apply(X)

    assert X == X_copy


def test_apply_no_row_aliasing(family):
    """apply() output has no row aliasing with input or glyphs."""
    train = [
        {
            "input": [[1]],
            "output": [[11, 12], [13, 14]]
        }
    ]
    family.fit(train)

    result = family.apply([[1]])

    # Mutating result should not affect glyphs
    result[0][0] = 999
    assert family.params.glyphs[1][0][0] == 11


# ============================================================================
# Comparison Tests
# ============================================================================

def test_comparison_with_pixel_replicate(family):
    """BlockSubstitution differs from PixelReplicate (different glyphs per color)."""
    # PixelReplicate: all pixels expand uniformly
    # BlockSubstitution: each color has its own glyph
    train = [
        {
            "input": [[0, 1]],
            "output": [[0, 0, 1, 1], [0, 0, 1, 1]]
        }
    ]
    result = family.fit(train)
    assert result is True

    # Different colors have different glyphs
    assert family.params.glyphs[0] != family.params.glyphs[1]
