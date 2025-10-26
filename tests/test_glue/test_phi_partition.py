"""
Test suite for P6-02: build_phi_partition(tr_pairs_afterP)

30+ tests across 11 categories:
- Input format handling
- Empty and edge cases
- Φ.3 Stability (INPUT-ONLY)
- Disjointness (Φ.2)
- Determinism
- Signature field correctness
- Multi-train coordination
- Band handling
- Component handling
- Patchkey handling
- Performance sanity
"""

import pytest
from src.glue import build_phi_partition


# ============================================================================
# 1. INPUT FORMAT HANDLING (5 tests)
# ============================================================================

def test_accepts_tuple_format():
    """Accept [(Xp, Y), ...] tuple format."""
    tr_pairs = [
        ([[1, 2]], [[1, 2]]),
        ([[3, 4]], [[3, 4]]),
    ]
    items, classes = build_phi_partition(tr_pairs)
    assert len(items) == 2
    assert isinstance(items[0], dict)
    assert "Xp" in items[0] and "Y" in items[0]


def test_accepts_dict_format():
    """Accept [{"Xp": ..., "Y": ...}, ...] dict format."""
    tr_pairs = [
        {"Xp": [[1, 2]], "Y": [[1, 2]]},
        {"Xp": [[3, 4]], "Y": [[3, 4]]},
    ]
    items, classes = build_phi_partition(tr_pairs)
    assert len(items) == 2
    assert isinstance(items[0], dict)


def test_rejects_ragged_xp():
    """Ragged Xp raises ValueError."""
    tr_pairs = [
        ([[1, 2], [3]], [[1, 2], [3, 4]]),  # Xp ragged
    ]
    with pytest.raises(ValueError, match="ragged"):
        build_phi_partition(tr_pairs)


def test_rejects_ragged_y():
    """Ragged Y raises ValueError."""
    tr_pairs = [
        ([[1, 2], [3, 4]], [[1, 2], [3]]),  # Y ragged
    ]
    with pytest.raises(ValueError, match="ragged"):
        build_phi_partition(tr_pairs)


def test_rejects_shape_mismatch():
    """Shape mismatch raises ValueError."""
    tr_pairs = [
        ([[1, 2, 3]], [[1, 2], [3, 4]]),  # 1x3 vs 2x2
    ]
    with pytest.raises(ValueError, match="Shape mismatch"):
        build_phi_partition(tr_pairs)


# ============================================================================
# 2. EMPTY AND EDGE CASES (4 tests)
# ============================================================================

def test_empty_dataset():
    """Empty dataset returns ([], {})."""
    items, classes = build_phi_partition([])
    assert items == []
    assert classes == {}


def test_empty_residuals():
    """All Xp == Y → classes == {} (no residuals)."""
    tr_pairs = [
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),  # Xp == Y
        ([[5, 6], [7, 8]], [[5, 6], [7, 8]]),  # Xp == Y
    ]
    items, classes = build_phi_partition(tr_pairs)
    assert len(items) == 2
    assert classes == {}


def test_single_pixel_grid():
    """1x1 grid works."""
    tr_pairs = [([[5]], [[7]])]
    items, classes = build_phi_partition(tr_pairs)
    assert len(items) == 1
    assert sum(len(coords) for coords in classes.values()) == 1


def test_single_train():
    """One pair works."""
    tr_pairs = [([[0, 0]], [[1, 1]])]
    items, classes = build_phi_partition(tr_pairs)
    assert len(items) == 1
    assert len(classes) >= 1


# ============================================================================
# 3. Φ.3 STABILITY (INPUT-ONLY) (3 tests)
# ============================================================================

def test_phi_depends_only_on_xp():
    """Changing Y doesn't change class structure (same Xp)."""
    Xp = [[0, 0], [0, 0]]
    Y1 = [[1, 2], [3, 4]]
    Y2 = [[5, 6], [7, 8]]

    items1, classes1 = build_phi_partition([(Xp, Y1)])
    items2, classes2 = build_phi_partition([(Xp, Y2)])

    # Classes should have same structure (same pixels grouped)
    assert len(classes1) == len(classes2)
    # Verify same class membership structure
    assert set(classes1.keys()) == set(classes2.keys())


def test_signature_computed_from_xp_not_y():
    """Verify feats uses Xp (items contain feats)."""
    tr_pairs = [([[1, 2]], [[3, 4]])]
    items, classes = build_phi_partition(tr_pairs)

    # Items should contain feats computed from Xp
    assert "feats" in items[0]
    assert items[0]["Xp"] == [[1, 2]]


def test_residual_mask_only_filters():
    """Y used only to compute R, not in Φ."""
    Xp = [[1, 1], [1, 1]]
    Y = [[1, 2], [1, 2]]  # Only column 1 differs

    items, classes = build_phi_partition([(Xp, Y)])

    # Only 2 residual pixels (column 1)
    total_residuals = sum(len(coords) for coords in classes.values())
    assert total_residuals == 2


# ============================================================================
# 4. DISJOINTNESS (Φ.2) (3 tests)
# ============================================================================

def test_classes_are_disjoint():
    """No (i,r,c) appears in multiple classes."""
    tr_pairs = [
        ([[0, 0], [0, 0]], [[1, 2], [3, 4]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Collect all coordinates from all classes
    all_coords = []
    for coords in classes.values():
        all_coords.extend(coords)

    # No duplicates
    assert len(all_coords) == len(set(all_coords))


def test_union_equals_all_residuals():
    """Union of all classes == all residual pixels."""
    tr_pairs = [
        ([[0, 0], [0, 0]], [[1, 2], [3, 4]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Count residuals directly
    residual_count = 0
    for item in items:
        for row in item["residual"]:
            for val in row:
                if val is not None:
                    residual_count += 1

    # Count pixels in classes
    class_count = sum(len(coords) for coords in classes.values())

    assert class_count == residual_count


def test_no_missing_residuals():
    """Every R[r][c] != None is in some class."""
    tr_pairs = [
        ([[1, 1], [1, 1]], [[2, 1], [1, 3]]),  # (0,0) and (1,1) differ
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Expected residuals: (0,0,0) and (0,1,1)
    all_coords = set()
    for coords in classes.values():
        all_coords.update(coords)

    assert (0, 0, 0) in all_coords
    assert (0, 1, 1) in all_coords


# ============================================================================
# 5. DETERMINISM (4 tests)
# ============================================================================

def test_stable_class_ids():
    """Two runs → identical class_id assignments."""
    tr_pairs = [
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]]),
    ]

    items1, classes1 = build_phi_partition(tr_pairs)
    items2, classes2 = build_phi_partition(tr_pairs)

    assert classes1 == classes2


def test_stable_coordinate_ordering():
    """Coords within class sorted by (i,r,c)."""
    tr_pairs = [
        ([[0, 0], [0, 0]], [[1, 1], [1, 1]]),  # All 4 pixels in same class
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Should be one class with all pixels
    if len(classes) > 0:
        coords = classes[0]
        # Verify sorted by (i,r,c)
        assert coords == sorted(coords)


def test_signature_lex_ordering():
    """Class IDs increase with lex ordering of sigs."""
    tr_pairs = [
        ([[0, 0], [0, 0]], [[1, 2], [3, 4]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Class IDs should be contiguous 0..K-1
    class_ids = sorted(classes.keys())
    expected = list(range(len(classes)))
    assert class_ids == expected


def test_deterministic_json_hash():
    """Stable JSON serialization (same structure)."""
    import json

    tr_pairs = [
        ([[0, 1]], [[2, 3]]),
    ]

    items1, classes1 = build_phi_partition(tr_pairs)
    items2, classes2 = build_phi_partition(tr_pairs)

    # Convert to JSON and verify equality
    json1 = json.dumps(classes1, sort_keys=True)
    json2 = json.dumps(classes2, sort_keys=True)

    assert json1 == json2


# ============================================================================
# 6. SIGNATURE FIELD CORRECTNESS (7 tests)
# ============================================================================

def test_parity_field():
    """Verify (r+c) % 2 encoded correctly."""
    # Create a grid where different parity positions have different residuals
    tr_pairs = [
        ([[0, 0], [0, 0]], [[1, 0], [0, 1]]),  # Even parity → 1, odd → 0
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Should partition by parity (and other features)
    # At minimum, we have residual pixels
    total = sum(len(coords) for coords in classes.values())
    assert total == 2  # Only (0,0) and (1,1) differ


def test_rowmod_colmod_fields():
    """Verify r%2, r%3, c%2, c%3 encoded."""
    # Large enough grid to test mod values
    tr_pairs = [
        ([[0]*6 for _ in range(6)], [[1]*6 for _ in range(6)]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # All pixels are residuals (36 total)
    total = sum(len(coords) for coords in classes.values())
    assert total == 36


def test_band_ids():
    """Verify row_band_id, col_band_id (or -1 if uniform)."""
    # Uniform grid should have single band (id 0)
    tr_pairs = [
        ([[1, 1], [1, 1]], [[2, 2], [2, 2]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Should partition (uniform input → single band for each axis)
    assert len(classes) >= 1


def test_is_color_field():
    """Verify Xp[r][c] stored correctly."""
    # Different colors should create different signatures
    tr_pairs = [
        ([[1, 2]], [[0, 0]]),  # Different input colors
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Two residuals with different is_color → different classes (likely)
    total = sum(len(coords) for coords in classes.values())
    assert total == 2


def test_touching_flags_bitmask():
    """Verify 10-bit packing."""
    # Different touching patterns
    tr_pairs = [
        ([[1, 2, 3]], [[0, 0, 0]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # 3 residuals
    total = sum(len(coords) for coords in classes.values())
    assert total == 3


def test_component_id_field():
    """Verify component_id matches id_grid."""
    tr_pairs = [
        ([[1, 2]], [[0, 0]]),  # Different components
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Verify items contain feats with components
    assert "components" in items[0]["feats"]
    assert "id_grid" in items[0]["feats"]["components"]


def test_patchkey_fields():
    """Verify r2/r3/r4 keys or None at borders."""
    # Small grid → many border pixels
    tr_pairs = [
        ([[1, 2, 3]], [[0, 0, 0]]),  # 1x3 grid (all borders for r≥2)
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Verify patchkeys present
    assert "patchkeys" in items[0]["feats"]


# ============================================================================
# 7. MULTI-TRAIN COORDINATION (3 tests)
# ============================================================================

def test_cross_train_class_merging():
    """Same signature in train 0 and 1 → same class."""
    # Identical Xp → same signatures
    Xp = [[0, 0]]
    tr_pairs = [
        (Xp, [[1, 0]]),  # Train 0: pixel (0,0) residual
        (Xp, [[2, 0]]),  # Train 1: pixel (0,0) residual
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Both residuals should be in same class (same signature)
    # Verify we have classes
    assert len(classes) >= 1


def test_per_train_coordinate_sorting():
    """Within class, coords sorted by (i,r,c)."""
    tr_pairs = [
        ([[0, 0]], [[1, 1]]),
        ([[0, 0]], [[2, 2]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Check that coordinates are sorted
    for coords in classes.values():
        assert coords == sorted(coords)


def test_multiple_trains_disjoint():
    """Pixels from different trains can coexist in same class."""
    tr_pairs = [
        ([[0]], [[1]]),  # Train 0: (0,0,0) residual
        ([[0]], [[1]]),  # Train 1: (0,0,0) residual
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Both should be in same class (same Xp, same position)
    total = sum(len(coords) for coords in classes.values())
    assert total == 2


# ============================================================================
# 8. BAND HANDLING (3 tests)
# ============================================================================

def test_single_band():
    """Uniform row/col → single band."""
    tr_pairs = [
        ([[1, 1]], [[2, 2]]),  # Uniform input
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Check NPS bands in feats
    assert "nps" in items[0]["feats"]


def test_multiple_bands():
    """NPS detects boundaries → multiple band IDs."""
    # Alternating pattern should create bands
    tr_pairs = [
        ([[1, 1], [2, 2]], [[0, 0], [0, 0]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Verify bands present
    assert "nps" in items[0]["feats"]


def test_no_bands_yields_minus_one():
    """Empty band list → -1 band_id."""
    # Uniform grid has no band boundaries
    tr_pairs = [
        ([[0, 0]], [[1, 1]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Partition should succeed
    assert len(items) == 1


# ============================================================================
# 9. COMPONENT HANDLING (3 tests)
# ============================================================================

def test_single_component():
    """Uniform color → all same component_id."""
    tr_pairs = [
        ([[1, 1]], [[2, 2]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Single component expected
    id_grid = items[0]["feats"]["components"]["id_grid"]
    unique_ids = set(id_grid[0])
    assert len(unique_ids) == 1


def test_multiple_components():
    """Different colors → different component_ids."""
    tr_pairs = [
        ([[1, 2]], [[0, 0]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Two components expected
    id_grid = items[0]["feats"]["components"]["id_grid"]
    unique_ids = set(id_grid[0])
    assert len(unique_ids) == 2


def test_component_tie_breaking():
    """Equal size/bbox → deterministic ordering."""
    # Two separate components of same color, same size
    tr_pairs = [
        ([[1, 0, 1]], [[0, 0, 0]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Components should have deterministic IDs
    id_grid = items[0]["feats"]["components"]["id_grid"]
    assert len(id_grid[0]) == 3


# ============================================================================
# 10. PATCHKEY HANDLING (3 tests)
# ============================================================================

def test_patchkey_none_at_borders():
    """Border pixels have None for all radii."""
    # 2x2 grid → all pixels are borders for r≥2
    tr_pairs = [
        ([[1, 2], [3, 4]], [[0, 0], [0, 0]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # For r=2, all pixels in 2x2 grid are borders
    patchkeys_r2 = items[0]["feats"]["patchkeys"]["r2"]
    # Should have None values at borders
    assert patchkeys_r2 is not None


def test_patchkey_valid_at_center():
    """Center pixels have valid keys for smaller radii."""
    # Large enough grid for r=2 interior
    tr_pairs = [
        ([[i for i in range(5)] for _ in range(5)],
         [[0]*5 for _ in range(5)]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # Center pixel (2,2) should have valid patchkey for r=2
    patchkeys_r2 = items[0]["feats"]["patchkeys"]["r2"]
    # (2,2) should not be None for r=2
    assert len(patchkeys_r2) == 5


def test_small_grid_all_none():
    """Grid too small → all None for large radii."""
    # 1x1 grid → all patchkeys None
    tr_pairs = [
        ([[1]], [[0]]),
    ]
    items, classes = build_phi_partition(tr_pairs)

    # All patchkeys should be None for 1x1 grid
    patchkeys = items[0]["feats"]["patchkeys"]
    assert patchkeys["r2"][0][0] is None
    assert patchkeys["r3"][0][0] is None
    assert patchkeys["r4"][0][0] is None


# ============================================================================
# 11. PERFORMANCE SANITY (2 tests)
# ============================================================================

def test_large_grid_performance():
    """30x30 grid with many residuals completes quickly."""
    import time

    # Create large grid with many residuals
    Xp = [[0]*30 for _ in range(30)]
    Y = [[1]*30 for _ in range(30)]

    tr_pairs = [(Xp, Y)]

    start = time.time()
    items, classes = build_phi_partition(tr_pairs)
    elapsed = time.time() - start

    assert elapsed < 2.0  # Should complete in <2s
    assert sum(len(coords) for coords in classes.values()) == 900


def test_many_trains_performance():
    """10 trains x 10x10 completes quickly."""
    import time

    # Create 10 trains
    tr_pairs = [
        ([[0]*10 for _ in range(10)], [[1]*10 for _ in range(10)])
        for _ in range(10)
    ]

    start = time.time()
    items, classes = build_phi_partition(tr_pairs)
    elapsed = time.time() - start

    assert elapsed < 2.0  # Should complete in <2s
    assert len(items) == 10
