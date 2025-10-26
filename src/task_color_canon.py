"""
Task-Level Color Canonicalization via Weisfeiler-Lehman refinement.

Computes a single, stable canonical color mapping across ALL training inputs
to satisfy FY (exact balance) while enabling cross-palette generalization.

Key insight: Per-input canonicalization causes ID drift across training pairs,
violating FY. Task-level canonicalization creates one reference that applies
consistently to all training and test inputs.
"""

from src.utils import dims
import hashlib


def _compute_components_8conn(g: list[list[int]], target_color: int) -> list[tuple[int, tuple[int, int]]]:
    """
    Find all 8-connected components of target_color.

    Returns list of (area, centroid) tuples for each component.
    Deterministic: processes cells row-major, assigns component IDs in discovery order.
    """
    if not g:
        return []

    rows, cols = dims(g)
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def flood_fill(start_r: int, start_c: int) -> tuple[int, tuple[int, int]]:
        """8-conn flood fill, returns (area, centroid)"""
        stack = [(start_r, start_c)]
        cells = []

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or g[r][c] != target_color:
                continue

            visited[r][c] = True
            cells.append((r, c))

            # 8-connected neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    stack.append((r + dr, c + dc))

        area = len(cells)
        if area == 0:
            return (0, (0, 0))

        centroid_r = sum(r for r, c in cells) / area
        centroid_c = sum(c for r, c in cells) / area
        return (area, (int(centroid_r), int(centroid_c)))

    # Process row-major for determinism
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and g[r][c] == target_color:
                comp = flood_fill(r, c)
                if comp[0] > 0:
                    components.append(comp)

    return components


def _compute_adjacency(g: list[list[int]]) -> dict[int, set[int]]:
    """
    Compute 4-connected color adjacency graph.

    Returns dict mapping each color to the set of colors it touches.
    """
    if not g:
        return {}

    rows, cols = dims(g)
    adjacency = {}

    for r in range(rows):
        for c in range(cols):
            color = g[r][c]
            if color not in adjacency:
                adjacency[color] = set()

            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_color = g[nr][nc]
                    if neighbor_color != color:
                        adjacency[color].add(neighbor_color)

    return adjacency


def _compute_color_fingerprint(g: list[list[int]], color: int) -> tuple:
    """
    Compute input-only structural fingerprint for a color.

    Returns tuple: (frequency, num_components, mean_area, centroid_r, centroid_c, adjacency_degree)
    """
    if not g:
        return (0, 0, 0, 0, 0, 0)

    rows, cols = dims(g)

    # Frequency
    frequency = sum(1 for row in g for c in row if c == color)

    # Components (8-connected)
    components = _compute_components_8conn(g, color)
    num_components = len(components)

    if num_components == 0:
        mean_area = 0
        centroid_r = 0
        centroid_c = 0
    else:
        mean_area = sum(area for area, _ in components) / num_components
        centroid_r = sum(cr for _, (cr, _) in components) / num_components
        centroid_c = sum(cc for _, (_, cc) in components) / num_components

    # Adjacency degree
    adjacency = _compute_adjacency(g)
    adjacency_degree = len(adjacency.get(color, set()))

    return (frequency, num_components, int(mean_area), int(centroid_r), int(centroid_c), adjacency_degree)


class TaskColorCanon:
    """
    Task-level color canonicalization.

    Computes a single canonical color mapping across all training inputs,
    then applies the same mapping to test inputs via WL alignment.
    """

    def __init__(self):
        self.task_reference = None  # Will be dict: wl_label -> canonical_id
        self.wl_rounds = 2  # Number of WL refinement rounds

    def fit(self, train_inputs: list[list[list[int]]]):
        """
        Learn task-level canonical color mapping from training inputs.

        Args:
            train_inputs: List of training input grids
        """
        if not train_inputs:
            self.task_reference = {}
            return

        # Step 1: Compute initial fingerprints for all colors in all training inputs
        all_colors = set()
        for g in train_inputs:
            for row in g:
                all_colors.update(row)

        # Build union graph: collect fingerprints from all training inputs
        color_fingerprints = {}  # color -> list of fingerprints across all train inputs
        for color in all_colors:
            color_fingerprints[color] = []

        for g in train_inputs:
            grid_colors = set()
            for row in g:
                grid_colors.update(row)

            for color in grid_colors:
                fp = _compute_color_fingerprint(g, color)
                color_fingerprints[color].append(fp)

        # Aggregate fingerprints: use mean values across all occurrences
        aggregated_fingerprints = {}
        for color, fps in color_fingerprints.items():
            if not fps:
                continue
            # Average each component
            avg_frequency = sum(fp[0] for fp in fps) / len(fps)
            avg_num_comp = sum(fp[1] for fp in fps) / len(fps)
            avg_mean_area = sum(fp[2] for fp in fps) / len(fps)
            avg_centroid_r = sum(fp[3] for fp in fps) / len(fps)
            avg_centroid_c = sum(fp[4] for fp in fps) / len(fps)
            avg_degree = sum(fp[5] for fp in fps) / len(fps)

            aggregated_fingerprints[color] = (
                int(avg_frequency),
                int(avg_num_comp),
                int(avg_mean_area),
                int(avg_centroid_r),
                int(avg_centroid_c),
                int(avg_degree)
            )

        # Build union adjacency graph
        union_adjacency = {}
        for g in train_inputs:
            grid_adj = _compute_adjacency(g)
            for color, neighbors in grid_adj.items():
                if color not in union_adjacency:
                    union_adjacency[color] = set()
                union_adjacency[color].update(neighbors)

        # Step 2: WL refinement
        # Initial labels: fingerprint tuple
        wl_labels = {color: aggregated_fingerprints[color] for color in all_colors}

        for round_num in range(self.wl_rounds):
            new_labels = {}
            for color in all_colors:
                # Collect neighbor labels
                neighbors = union_adjacency.get(color, set())
                neighbor_labels = sorted([wl_labels[n] for n in neighbors if n in wl_labels])

                # New label: hash of (own label, sorted neighbor labels)
                label_str = str((wl_labels[color], tuple(neighbor_labels)))
                label_hash = hashlib.md5(label_str.encode()).hexdigest()
                new_labels[color] = label_hash

            wl_labels = new_labels

        # Step 3: Create task reference Ξ with lexicographic ordering
        # Sort by (wl_label, fingerprint components)
        color_entries = []
        for color in all_colors:
            fp = aggregated_fingerprints[color]
            wl_label = wl_labels[color]
            # Sort key: (wl_label, frequency desc, degree desc, mean_area, centroid_r, centroid_c)
            sort_key = (wl_label, -fp[0], -fp[5], fp[2], fp[3], fp[4])
            color_entries.append((color, wl_label, sort_key))

        color_entries.sort(key=lambda x: x[2])

        # Assign canonical IDs
        self.task_reference = {}
        self.raw_to_canon = {}  # For training: raw_color -> canonical_id

        for canon_id, (raw_color, wl_label, _) in enumerate(color_entries):
            self.task_reference[wl_label] = canon_id
            self.raw_to_canon[raw_color] = canon_id

    def apply(self, g: list[list[int]]) -> tuple[list[list[int]], dict]:
        """
        Apply task-level canonicalization to a grid.

        For training: uses raw_to_canon mapping.
        For test: computes WL labels and aligns to task_reference.

        Args:
            g: Input grid

        Returns:
            Tuple of (canonicalized grid, aux_data) where aux_data contains:
            - "raw_color_id": Grid with original raw colors (for colorful features)
        """
        if not g or self.task_reference is None:
            return g, {"raw_color_id": g}

        rows, cols = dims(g)

        # Get all colors in this grid
        grid_colors = set()
        for row in g:
            grid_colors.update(row)

        # Compute WL labels for this grid
        color_fingerprints = {}
        for color in grid_colors:
            color_fingerprints[color] = _compute_color_fingerprint(g, color)

        adjacency = _compute_adjacency(g)

        # WL refinement (same number of rounds as training)
        wl_labels = {color: color_fingerprints[color] for color in grid_colors}

        for round_num in range(self.wl_rounds):
            new_labels = {}
            for color in grid_colors:
                neighbors = adjacency.get(color, set())
                neighbor_labels = sorted([wl_labels[n] for n in neighbors if n in wl_labels])

                label_str = str((wl_labels[color], tuple(neighbor_labels)))
                label_hash = hashlib.md5(label_str.encode()).hexdigest()
                new_labels[color] = label_hash

            wl_labels = new_labels

        # Align to task reference
        raw_to_canon = {}
        for raw_color in grid_colors:
            wl_label = wl_labels[raw_color]
            if wl_label in self.task_reference:
                raw_to_canon[raw_color] = self.task_reference[wl_label]
            else:
                # Unseen WL label - fallback to raw color
                # (This should be rare; means test has novel color structure)
                raw_to_canon[raw_color] = raw_color

        # Apply canonicalization
        result = []
        for r in range(rows):
            row = []
            for c in range(cols):
                raw_color = g[r][c]
                canon_color = raw_to_canon.get(raw_color, raw_color)
                row.append(canon_color)
            result.append(row)

        # Return canonicalized grid and aux_data with original raw colors
        aux_data = {"raw_color_id": g}
        return result, aux_data

    def inverse_apply(self, canon_g: list[list[int]], canon_to_raw: dict[int, int]) -> list[list[int]]:
        """
        Inverse canonicalization: convert canonical grid back to original colors.

        Args:
            canon_g: Grid in canonical color space
            canon_to_raw: Mapping from canonical IDs to raw colors (inverse of raw_to_canon)

        Returns:
            Grid in original color space
        """
        if not canon_g:
            return canon_g

        rows, cols = dims(canon_g)
        result = []
        for r in range(rows):
            row = []
            for c in range(cols):
                canon_color = canon_g[r][c]
                raw_color = canon_to_raw.get(canon_color, canon_color)  # Fallback to canon if not in mapping
                row.append(raw_color)
            result.append(row)

        return result
