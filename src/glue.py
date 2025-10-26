"""
GLUE stitching functions for compositional solving.

Phase 6: Residual computation and Phi partition building.
All functions are pure (no mutation) and deterministic.
"""

from collections import defaultdict
from src.utils import dims, copy_grid, deep_eq
from src.signature_builders import phi_signature_tables, Schema


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
    tr_pairs_afterP: list[tuple[list[list[int]], list[list[int]]]] | list[dict],
    schema = None
) -> tuple[list[dict], dict[int, list[tuple[int, int, int]]]]:
    """
    Build Φ-partition over residual pixels across training pairs.

    Computes residuals R_i = compute_residual(Xp_i, Y_i), builds Φ signatures
    on Xp_i (INPUT-ONLY per Φ.3), and groups residual pixels by canonical
    per-pixel signature. Returns (items, classes) for later action inference
    and stitching.

    Schema Lattice (MDL Minimality):
    Colorless variants (most coarse, best for palette-permutation tasks):
    - Schema(): 6-tuple (spatial only, no is_color, no patchkeys) - COARSEST
    - Schema(use_patch_r2=True): 7-tuple (+ patchkey_r2)
    - Schema(use_patch_r3=True): 7-tuple (+ patchkey_r3)
    - Schema(use_patch_r4=True): 7-tuple (+ patchkey_r4)

    Colorful variants (legacy, for color-dependent tasks):
    - Schema(use_is_color=True): 7-tuple (+ is_color, no patchkeys)
    - Schema(use_is_color=True, use_patch_r2=True): 8-tuple (+ is_color + patchkey_r2)
    - ... etc (8 variants total in schema lattice)

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
        schema: Schema object or legacy string ('S0'/'S1'/'S2'/'S3') or None
            (default None → Schema() → colorless base)

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
        # Schema determines which patchkeys are included
        feats = phi_signature_tables(Xp, schema)

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
                    # Build schema-dependent signature tuple at (r,c)
                    # S0: 7-tuple, S1/S2/S3: 8-tuple
                    sig = _build_signature(feats, r, c, Xp, schema)
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
    feats: dict, r: int, c: int, Xp: list[list[int]], schema = None
) -> tuple:
    """
    Extract pair-invariant signature tuple at pixel (r,c).

    BUG FIXES (2025-10-26):
    - B1: Removed absolute position features (parity, rowmod, colmod)
    - B2: Replaced pair-specific IDs with pair-invariant features
    - B4: Removed row_offset and col_offset (too position-specific)
    - Schema Lattice: Conditional is_color + patchkey inclusion for MDL minimality

    Schema Lattice (MDL Minimality):
    Colorless variants (most coarse, best for palette-permutation tasks):
    - Schema(): 6-tuple (spatial only, no is_color, no patchkeys) - COARSEST
    - Schema(use_patch_r2=True): 7-tuple (+ patchkey_r2)
    - Schema(use_patch_r3=True): 7-tuple (+ patchkey_r3)
    - Schema(use_patch_r4=True): 7-tuple (+ patchkey_r4)

    Colorful variants (legacy, for color-dependent tasks):
    - Schema(use_is_color=True): 7-tuple (+ is_color, no patchkeys)
    - Schema(use_is_color=True, use_patch_r2=True): 8-tuple (+ is_color + patchkey_r2)
    - ... etc (8 variants total in schema lattice)

    Base signature fields (always present):
    1. row_boundary: 1 if on NPS row boundary, 0 otherwise
    2. col_boundary: 1 if on NPS col boundary, 0 otherwise
    3. touching_flags: 10-bit bitmask (bit c set if touching_color[c][r][c])
    4. largest_comp: 1 if in largest component, 0 otherwise
    5. comp_size: component size bucket (0=tiny, 1=small, 2=medium, 3=large)
    6. comp_aspect: component aspect (0=tall, 1=square, 2=wide)

    Schema-conditional fields:
    +1. is_color: Xp[r][c] (specific color value) - ONLY if schema.use_is_color
    +1. patchkey: canonical patch key or () - ONLY if schema.use_patch_rN

    Args:
        feats: Φ features dict from phi_signature_tables(Xp, schema)
        r: Row index
        c: Column index
        Xp: Original transformed grid (for is_color field, if used)
        schema: Schema object or legacy string ('S0'/'S1'/'S2'/'S3') or None

    Returns:
        Signature tuple (length depends on schema configuration)
    """
    # Handle schema parameter (Schema object or legacy string, or None)
    if schema is None:
        schema = Schema()  # Default: colorless base
    elif isinstance(schema, str):
        # Backward compatibility: legacy string schemas include is_color
        if schema == 'S0':
            schema = Schema(use_is_color=True)
        elif schema == 'S1':
            schema = Schema(use_is_color=True, use_patch_r2=True)
        elif schema == 'S2':
            schema = Schema(use_is_color=True, use_patch_r3=True)
        elif schema == 'S3':
            schema = Schema(use_is_color=True, use_patch_r4=True)
        else:
            raise ValueError(f"Invalid schema: {schema}. Must be 'S0', 'S1', 'S2', or 'S3'.")

    # Base features (always present in all schemas)

    # NPS pair-invariant features
    row_boundary = feats["nps"]["row_boundary"][r][c]
    col_boundary = feats["nps"]["col_boundary"][r][c]

    # Touching flags: pack 10-bit mask
    # Bit position c is set if touching_color[c][r][c] == True
    # NOTE: touching_color is STRUCTURAL (which colors are adjacent), not absolute color ID
    touching_flags = sum(
        (1 << color)
        for color in range(10)
        if feats["local"]["touching_color"][color][r][c]
    )

    # Component pair-invariant features
    largest_comp = feats["components"]["largest_comp"][r][c]
    comp_size = feats["components"]["comp_size"][r][c]
    comp_aspect = feats["components"]["comp_aspect"][r][c]

    # Build signature tuple conditionally based on schema
    sig_parts = [
        row_boundary,
        col_boundary,
        touching_flags,
        largest_comp,
        comp_size,
        comp_aspect,
    ]

    # Schema-conditional: color features (mutually exclusive)
    if schema.use_is_color:
        # Specific color value (palette-specific)
        sig_parts.append(Xp[r][c])
    elif schema.use_canon_color_id:
        # Canonical color ID (enables cross-palette generalization)
        canon_color_map = feats["local"]["canon_color_map"]
        raw_color = Xp[r][c]
        canon_color = canon_color_map.get(raw_color, raw_color)
        sig_parts.append(canon_color)

    # Schema-conditional: patchkey (only ONE of r2/r3/r4 can be True)
    if schema.use_patch_r2:
        patchkey = feats["patchkeys"]["r2"][r][c]
        sig_parts.append(() if patchkey is None else patchkey)
    elif schema.use_patch_r3:
        patchkey = feats["patchkeys"]["r3"][r][c]
        sig_parts.append(() if patchkey is None else patchkey)
    elif schema.use_patch_r4:
        patchkey = feats["patchkeys"]["r4"][r][c]
        sig_parts.append(() if patchkey is None else patchkey)

    return tuple(sig_parts)


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


def verify_stitched_equality(
    items: list[dict],
    classes: dict[int, list[tuple[int, int, int]]],
    actions_by_cid: dict[int, tuple[str, object | None]],
) -> bool:
    """
    Verify GLUE exactness (FY) by stitching per-class actions and comparing to targets.

    Returns True iff stitch_from_classes(...) reproduces every Y_i exactly (bit-for-bit);
    otherwise False. This is a pure verification wrapper—no solver logic, no MDL.

    This function embodies the GLUE theorem (T3): when class supports are disjoint,
    stitching from a frozen base with per-class actions yields exact equality to
    one-shot outputs. We verify this by comparing stitched results to ground truth Y.

    Args:
        items: Training pair dicts from build_phi_partition:
            [{"Xp": grid, "Y": grid, "feats": dict, "residual": grid}, ...]
        classes: Class partitions (from build_phi_partition):
            {class_id: [(train_idx, row, col), ...], ...}
        actions_by_cid: Actions per class (from infer_action_for_class):
            {class_id: (action_name, param), ...}

    Returns:
        True: stitched outputs match all targets Y_i exactly
        False: at least one train fails equality check

    Raises:
        ValueError: Propagated from stitch_from_classes if:
            - Missing action for a class_id
            - Seam check enabled and overlap detected
            - Invalid grid (ragged rows)

    Postconditions:
        - FY equality: returns True ONLY if deep_eq(Out_i, Y_i) for ALL i
        - Purity: items unchanged
        - Determinism: same inputs → same boolean result

    Edge cases:
        - classes == {} → returns True iff every Xp_i == Y_i already
        - Some class has no coords for a train i (skipped) → still can be True
        - Single pixel differs in Y_i → returns False

    Examples:
        >>> # All Xp == Y (no residuals, no classes)
        >>> items = [{"Xp": [[1,2]], "Y": [[1,2]], "feats": {}, "residual": [[None,None]]}]
        >>> verify_stitched_equality(items, {}, {})
        True

        >>> # Single class, set_color action → exact match
        >>> items = [{"Xp": [[0,0]], "Y": [[5,5]], "feats": {}, "residual": [[5,5]]}]
        >>> classes = {0: [(0,0,0), (0,0,1)]}
        >>> actions = {0: ("set_color", 5)}
        >>> verify_stitched_equality(items, classes, actions)
        True

        >>> # Single pixel differs → False
        >>> items = [{"Xp": [[0,0]], "Y": [[5,4]], "feats": {}, "residual": [[5,4]]}]
        >>> classes = {0: [(0,0,0), (0,0,1)]}
        >>> actions = {0: ("set_color", 5)}
        >>> verify_stitched_equality(items, classes, actions)
        False
    """
    # Call stitcher to get outputs (propagates ValueError if invalid inputs)
    stitched = stitch_from_classes(items, classes, actions_by_cid)

    # Verify ALL trains match targets exactly (FY equality)
    return all(deep_eq(stitched[i], items[i]["Y"]) for i in range(len(items)))
