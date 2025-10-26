"""
GLUE stitching functions for compositional solving.

Phase 6: Residual computation and Phi partition building.
All functions are pure (no mutation) and deterministic.
"""

from collections import defaultdict
from src.utils import dims, copy_grid
from src.signature_builders import phi_signature_tables


def compute_residual(Xp: list[list[int]], Y: list[list[int]]) -> list[list[object]]:
    """
    Compute pixelwise residual between transformed input and target.

    Residual R defines which pixels need correction: R[r][c] is None if
    Xp[r][c] already equals Y[r][c], otherwise R[r][c] holds the target
    value Y[r][c] that must be written.

    Mathematical definition:
        R[r][c] = None         if Xp[r][c] = Y[r][c]
        R[r][c] = Y[r][c]      otherwise

    This is the "override table" for Step-2 composition (P + Φ/GLUE).

    Preconditions:
      - Xp is rectangular (all rows have same length)
      - Y is rectangular (all rows have same length)
      - Xp and Y have identical dimensions (R, C)
      - All values in Xp and Y are in range 0..9

    Postconditions:
      - R has shape (R, C)
      - R[r][c] = None        iff Xp[r][c] == Y[r][c]
      - R[r][c] = Y[r][c]     otherwise (value in 0..9)
      - Inputs Xp and Y are never mutated
      - Output rows are newly allocated (no aliasing)

    Args:
        Xp: Transformed input grid (output of some P family transform)
        Y: Target grid (ground truth output)

    Returns:
        New grid R where None indicates "no change needed" and integers
        0..9 indicate "set pixel to this value"

    Raises:
        ValueError: if Xp is ragged (rows have different lengths)
        ValueError: if Y is ragged (rows have different lengths)
        ValueError: if dims(Xp) != dims(Y) (shape mismatch)

    Examples:
        >>> # All equal (no residual)
        >>> compute_residual([[1,2]], [[1,2]])
        [[None, None]]

        >>> # All different (full residual)
        >>> compute_residual([[0,0]], [[1,2]])
        [[1, 2]]

        >>> # Mixed (partial residual)
        >>> compute_residual([[1,0,3]], [[1,2,3]])
        [[None, 2, None]]
    """
    # Validate Xp is rectangular and get shape
    # dims() raises ValueError if ragged
    try:
        h_Xp, w_Xp = dims(Xp)
    except ValueError:
        raise ValueError("Xp is ragged: rows have different lengths")

    # Validate Y is rectangular and get shape
    # dims() raises ValueError if ragged
    try:
        h_Y, w_Y = dims(Y)
    except ValueError:
        raise ValueError("Y is ragged: rows have different lengths")

    # Check shape match
    if (h_Xp, w_Xp) != (h_Y, w_Y):
        raise ValueError(
            f"Shape mismatch: dims(Xp)=({h_Xp},{w_Xp}) != dims(Y)=({h_Y},{w_Y})"
        )

    # Build residual: new grid, no aliasing
    # R[r][c] = None if match, else Y[r][c]
    return [
        [None if Xp[r][c] == Y[r][c] else Y[r][c] for c in range(w_Xp)]
        for r in range(h_Xp)
    ]


def build_phi_partition(
    tr_pairs_afterP: list[tuple[list[list[int]], list[list[int]]]] | list[dict]
) -> tuple[list[dict], dict[int, list[tuple[int, int, int]]]]:
    """
    Build Φ-partition over residual pixels across training pairs.

    Computes residuals R_i = compute_residual(Xp_i, Y_i), builds Φ signatures
    on Xp_i (INPUT-ONLY per Φ.3), and groups residual pixels by canonical
    per-pixel signature. Returns (items, classes) for later action inference
    and stitching.

    Mathematical semantics:
    - For each train i: R_i marks pixels where Xp_i != Y_i (residual set)
    - For each residual pixel (i,r,c): compute signature K[i,r,c] from Φ(Xp_i)
    - Partition: group all residual pixels by signature K (disjoint classes)
    - Class IDs: 0..K-1 assigned via lexicographic ordering of signatures

    This function is PURE partition building only — no action inference,
    no stitching, no MDL selection. It prepares the data structures for
    Step-2 composition (P + Φ/GLUE).

    Args:
        tr_pairs_afterP: Training pairs after applying global P transform.
            - Format A: [(Xp, Y), ...]  (list of tuples)
            - Format B: [{"Xp": ..., "Y": ...}, ...]  (list of dicts)
            - Each Xp, Y is a rectangular grid (list[list[int]])

    Returns:
        Tuple (items, classes) where:

        items: list[dict] with one entry per train:
        [
            {
                "Xp": grid,           # Transformed input
                "Y": grid,            # Target output
                "feats": dict,        # Φ features (from phi_signature_tables)
                "residual": grid      # Residual (None if equal, else Y[r][c])
            },
            ...
        ]

        classes: dict[int, list[tuple[int, int, int]]] mapping class_id → coords:
        {
            0: [(i0, r0, c0), (i0, r1, c1), ...],  # Class 0 pixels
            1: [(i1, r2, c2), ...],                 # Class 1 pixels
            ...
        }
        - class_id: 0..K-1 for K classes (lex-sorted by signature)
        - coords: (train_idx, row, col) sorted by (i, r, c) row-major

    Raises:
        ValueError: If Xp or Y is ragged
        ValueError: If dims(Xp_i) != dims(Y_i) for any train pair

    Postconditions:
        - Φ.3 (Input-only): feats computed on Xp only (never Y)
        - Φ.2 (Disjointness): each (i,r,c) residual belongs to exactly one class
        - Determinism: stable class IDs and coord ordering across runs
        - Empty residual case: classes == {} (no pixels to partition)

    Examples:
        >>> # Empty residuals (all Xp == Y)
        >>> tr_pairs = [([[1,2]], [[1,2]])]
        >>> items, classes = build_phi_partition(tr_pairs)
        >>> len(items)
        1
        >>> classes
        {}

        >>> # Single class (uniform signature)
        >>> tr_pairs = [([[0,0]], [[1,1]])]
        >>> items, classes = build_phi_partition(tr_pairs)
        >>> len(classes)
        1
        >>> len(classes[0])
        2
    """
    # Normalize input: accept both tuple and dict formats
    normalized = []
    for item in tr_pairs_afterP:
        if isinstance(item, dict):
            Xp, Y = item["Xp"], item["Y"]
        else:
            Xp, Y = item
        normalized.append((Xp, Y))

    # Build items: compute residuals and Φ features for each train
    items = []
    for Xp, Y in normalized:
        # Validate and compute residual (raises ValueError if invalid)
        residual = compute_residual(Xp, Y)

        # Compute Φ features on Xp ONLY (Φ.3 input-only constraint)
        feats = phi_signature_tables(Xp)

        items.append({"Xp": Xp, "Y": Y, "feats": feats, "residual": residual})

    # Build signature map: signature → list of (i, r, c) coordinates
    sig_to_pixels = defaultdict(list)

    for i, item in enumerate(items):
        Xp = item["Xp"]
        feats = item["feats"]
        residual = item["residual"]

        h, w = dims(Xp)

        for r in range(h):
            for c in range(w):
                # Only partition residual pixels (where Xp != Y)
                if residual[r][c] is not None:
                    # Build 13-field signature tuple at (r,c)
                    sig = _build_signature(feats, r, c, Xp)
                    sig_to_pixels[sig].append((i, r, c))

    # Assign deterministic class IDs via lexicographic ordering
    sorted_sigs = sorted(sig_to_pixels.keys())  # Lex order of tuples

    classes = {}
    for class_id, sig in enumerate(sorted_sigs):
        # Sort coordinates by (i, r, c) row-major
        coords = sorted(sig_to_pixels[sig])
        classes[class_id] = coords

    return (items, classes)


def _build_signature(
    feats: dict, r: int, c: int, Xp: list[list[int]]
) -> tuple:
    """
    Extract 13-field signature tuple at pixel (r,c).

    Signature fields (in fixed order):
    1. parity: (r+c) % 2
    2. rowmod2: r % 2
    3. rowmod3: r % 3
    4. colmod2: c % 2
    5. colmod3: c % 3
    6. row_band_id: index in row_band_masks, or -1 if no band contains (r,c)
    7. col_band_id: index in col_band_masks, or -1 if no band contains (r,c)
    8. is_color: Xp[r][c] (original color at pixel)
    9. touching_flags: 10-bit bitmask (bit c set if touching_color[c][r][c])
    10. component_id: id_grid[r][c]
    11. patchkey_r2: key from patchkeys["r2"][r][c] (tuple or None)
    12. patchkey_r3: key from patchkeys["r3"][r][c] (tuple or None)
    13. patchkey_r4: key from patchkeys["r4"][r][c] (tuple or None)

    Args:
        feats: Φ features dict from phi_signature_tables(Xp)
        r: Row index
        c: Column index
        Xp: Original transformed grid (for is_color field)

    Returns:
        13-tuple signature in fixed order (deterministic, comparable)
    """
    # Index predicates (fields 1-5)
    parity = 1 if feats["index"]["parity"]["M1"][r][c] else 0
    rowmod2 = 1 if feats["index"]["rowmod"]["k2"][1][r][c] else 0
    rowmod3 = next(i for i in range(3) if feats["index"]["rowmod"]["k3"][i][r][c])
    colmod2 = 1 if feats["index"]["colmod"]["k2"][1][r][c] else 0
    colmod3 = next(i for i in range(3) if feats["index"]["colmod"]["k3"][i][r][c])

    # NPS bands (fields 6-7)
    row_bands = feats["nps"]["row_bands"]
    row_band_id = next((i for i, mask in enumerate(row_bands) if mask[r][c]), -1)

    col_bands = feats["nps"]["col_bands"]
    col_band_id = next((i for i, mask in enumerate(col_bands) if mask[r][c]), -1)

    # Original color (field 8)
    is_color = Xp[r][c]

    # Touching flags: pack 10-bit mask (field 9)
    # Bit position c is set if touching_color[c][r][c] == True
    touching_flags = sum(
        (1 << color)
        for color in range(10)
        if feats["local"]["touching_color"][color][r][c]
    )

    # Component ID (field 10)
    component_id = feats["components"]["id_grid"][r][c]

    # Patchkeys (fields 11-13)
    # Convert None to empty tuple for sortability (None < tuple fails)
    patchkey_r2 = feats["patchkeys"]["r2"][r][c]
    patchkey_r3 = feats["patchkeys"]["r3"][r][c]
    patchkey_r4 = feats["patchkeys"]["r4"][r][c]

    # Replace None with () for lexicographic sorting
    patchkey_r2 = () if patchkey_r2 is None else patchkey_r2
    patchkey_r3 = () if patchkey_r3 is None else patchkey_r3
    patchkey_r4 = () if patchkey_r4 is None else patchkey_r4

    return (
        parity,
        rowmod2,
        rowmod3,
        colmod2,
        colmod3,
        row_band_id,
        col_band_id,
        is_color,
        touching_flags,
        component_id,
        patchkey_r2,
        patchkey_r3,
        patchkey_r4,
    )


def stitch_from_classes(
    items: list[dict],
    classes: dict[int, list[tuple[int, int, int]]],
    actions_by_cid: dict[int, tuple[str, object | None]],
    enable_seam_check: bool = False,
) -> list[list[list[int]]]:
    """
    Apply one action per Φ-class to rebuild outputs from FROZEN base Xp.

    This is the core GLUE stitching function. For each train, it creates a
    fresh output Out_i seeded from Xp_i, then applies class actions in
    deterministic order. All reads come ONLY from Xp_i (frozen base), all
    writes go to Out_i. This ensures GLUE compositionality: stitched output
    equals one-shot application.

    Mathematical semantics:
    - For each train i: Out_i = Xp_i  (initialize)
    - For each class K in ascending order:
      - coords_i = {(r,c) | (i,r,c) ∈ classes[K]}
      - Apply action_K reading from Xp_i, writing to Out_i at coords_i
    - Return [Out_0, Out_1, ..., Out_n]

    GLUE invariant: ALL reads from Xp (frozen), NEVER from Out (being written).

    Args:
        items: List of training pair dicts from build_phi_partition:
            [{"Xp": grid, "Y": grid, "feats": dict, "residual": grid}, ...]
        classes: Dict mapping class_id → list of (train_idx, row, col):
            {0: [(0,0,0), (0,1,1)], 1: [(1,2,3)], ...}
            Coordinates are row-major sorted per train (guaranteed by caller)
        actions_by_cid: Dict mapping class_id → action tuple:
            {0: ("set_color", 5), 1: ("mirror_h", None), ...}
            Action tuples: (name, param) where name ∈ {set_color, mirror_h,
            mirror_v, keep_nonzero, identity}
        enable_seam_check: If True, verify no overlapping writes (Φ.2)

    Returns:
        List of stitched output grids, one per train: [Out_0, Out_1, ...]

    Raises:
        ValueError: If Xp_i is ragged (non-rectangular)
        ValueError: If actions_by_cid missing a class_id from classes
        ValueError: If seam check enabled and overlapping writes detected

    Postconditions:
        - GLUE safety: All reads from Xp, all writes to Out (no read-after-write)
        - Purity: items unchanged, Out_i rows newly allocated
        - Determinism: Fixed class_id order, sorted coords per train
        - Shape: dims(Out_i) == dims(Xp_i) for all i

    Examples:
        >>> # Empty classes → deep copies
        >>> items = [{"Xp": [[1,2]], "Y": [[1,2]], "feats": {}, "residual": [[None,None]]}]
        >>> outs = stitch_from_classes(items, {}, {})
        >>> outs
        [[[1, 2]]]

        >>> # Single class, set_color
        >>> items = [{"Xp": [[0,0]], "Y": [[5,5]], "feats": {}, "residual": [[5,5]]}]
        >>> classes = {0: [(0,0,0), (0,0,1)]}
        >>> actions = {0: ("set_color", 5)}
        >>> outs = stitch_from_classes(items, classes, actions)
        >>> outs
        [[[5, 5]]]
    """
    # Handle empty items edge case
    if not items:
        return []

    # Initialize outputs: deep copy of Xp for each train
    outputs = [copy_grid(item["Xp"]) for item in items]

    # Validate all Xp are rectangular (dims will raise if ragged)
    for i, item in enumerate(items):
        dims(item["Xp"])  # Raises ValueError if ragged

    # Optional seam check: track written coordinates per train
    if enable_seam_check:
        write_masks = [set() for _ in range(len(items))]

    # Process classes in ascending order (deterministic)
    for class_id in sorted(classes.keys()):
        # Check action exists for this class
        if class_id not in actions_by_cid:
            raise ValueError(
                f"Missing action for class_id {class_id}. "
                f"actions_by_cid must contain all class IDs from classes."
            )

        action_name, param = actions_by_cid[class_id]
        coords = classes[class_id]

        # Group coords by train index
        coords_by_train = defaultdict(list)
        for train_idx, r, c in coords:
            coords_by_train[train_idx].append((r, c))

        # Apply action to each train independently
        for train_idx, train_coords in coords_by_train.items():
            Xp = items[train_idx]["Xp"]
            Out = outputs[train_idx]
            R, C = dims(Xp)

            # Optional seam check: verify no overlaps
            if enable_seam_check:
                overlap = set(train_coords) & write_masks[train_idx]
                if overlap:
                    raise ValueError(
                        f"Seam check failed: class {class_id} overlaps with "
                        f"previous writes at train {train_idx}, coords {sorted(overlap)}"
                    )
                write_masks[train_idx].update(train_coords)

            # Apply action (read from frozen Xp, write to Out)
            if action_name == "set_color":
                for r, c in train_coords:
                    Out[r][c] = param

            elif action_name == "mirror_h":
                # Horizontal mirror: Out[r][c] = Xp[R-1-r][c]
                for r, c in train_coords:
                    Out[r][c] = Xp[R - 1 - r][c]

            elif action_name == "mirror_v":
                # Vertical mirror: Out[r][c] = Xp[r][C-1-c]
                for r, c in train_coords:
                    Out[r][c] = Xp[r][C - 1 - c]

            elif action_name == "keep_nonzero":
                # Keep nonzero: Out[r][c] = Xp[r][c] if nonzero else 0
                for r, c in train_coords:
                    Out[r][c] = Xp[r][c] if Xp[r][c] != 0 else 0

            elif action_name == "identity":
                # Identity: Out[r][c] = Xp[r][c]
                for r, c in train_coords:
                    Out[r][c] = Xp[r][c]

            else:
                raise ValueError(f"Unknown action: {action_name}")

    return outputs
