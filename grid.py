"""
grid.py
=======
Semantic 2.5D grid construction and the semantic-aware variable
resolution grid.

Addresses:
  #7  Dominant semantic class can hide small or important obstacles
      -> create_semantic_grid() no longer takes a pure majority vote.
         If ANY point in a cell belongs to FORCE_NON_TRAVERSABLE_CLASSES
         (building/fence/vegetation/trunk/pole), that cell is forced to
         the most-frequent obstacle label present, even if traversable
         ground points dominate the raw point count - a person standing
         at the edge of a mostly-road cell can no longer be voted away.
  #24 Limited semantic confidence representation
      -> returns a semantic_confidence grid alongside the label grid:
         dominant_label_count / total_points_in_cell. Downstream code
         (cost map, dashboard) can treat a 0.4-confidence cell
         differently from a 0.95-confidence one instead of only ever
         seeing the single winning label.
  #3  Adaptive/variable resolution grid is not used for path planning
      -> rasterize_variable_grid() projects the finer per-point
         semantic_aware_variable_grid cells back onto the base-resolution
         planning grid: for every base cell, if any overlapping
         fine-resolution cell is an obstacle, the base cell is marked
         obstacle too. This is a practical bridge (not a full
         non-uniform-grid A*, which the module docstring below still
         flags as future work) that lets the higher-fidelity detections
         made at 0.25 m near obstacles/dynamic classes actually reach
         the planner, which previously only ever saw the coarse fixed
         grid.
"""

from typing import Dict, Tuple
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from config import (
    Config, CLASS_NAMES, CLASS_COLORS, OBSTACLE_CLASSES,
    FORCE_NON_TRAVERSABLE_CLASSES, HIGH_IMPORTANCE_CLASSES,
    MEDIUM_IMPORTANCE_CLASSES,
)


# ============================================================
# SEMANTIC COLORMAP  (unchanged)
# ============================================================

def get_semantic_colormap(present_labels):
    sorted_labels = sorted(present_labels)
    colors = [CLASS_COLORS.get(label, [0.5, 0.5, 0.5]) for label in sorted_labels]
    cmap = mcolors.ListedColormap(colors)

    boundaries = []
    for i, label in enumerate(sorted_labels):
        if i == 0:
            boundaries.append(label - 0.5)
        boundaries.append(label + 0.5)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLORS.get(l, [0.5, 0.5, 0.5]),
                       edgecolor="black", linewidth=0.5)
        for l in sorted_labels
    ]
    legend_labels = [CLASS_NAMES.get(l, f"class {l}") for l in sorted_labels]
    return cmap, norm, legend_handles, legend_labels


def create_semantic_colors(semantic_labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(semantic_labels), 3), dtype=np.float64)
    for i, label in enumerate(semantic_labels):
        colors[i] = CLASS_COLORS.get(int(label), [0.5, 0.5, 0.5])
    return colors


# ============================================================
# SEMANTIC 2.5D GRID  (#7, #24)
# ============================================================

def create_semantic_grid(
    xyz: np.ndarray,
    semantic_labels: np.ndarray,
    resolution: float,
    x_min: float, x_max: float, y_min: float, y_max: float,
):
    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)

    elevation_grid = np.full((grid_height, grid_width), np.nan)
    semantic_grid = np.full((grid_height, grid_width), -1, dtype=np.int32)
    point_count_grid = np.zeros((grid_height, grid_width), dtype=np.int32)
    confidence_grid = np.zeros((grid_height, grid_width), dtype=np.float64)

    gx = ((xyz[:, 0] - x_min) / resolution).astype(np.int64)
    gy = ((xyz[:, 1] - y_min) / resolution).astype(np.int64)

    valid = (gx >= 0) & (gx < grid_width) & (gy >= 0) & (gy < grid_height)
    gx, gy = gx[valid], gy[valid]
    z_valid = xyz[:, 2][valid]
    labels_valid = semantic_labels[valid]

    if len(labels_valid) == 0:
        return elevation_grid, semantic_grid, point_count_grid, confidence_grid

    flat_cell_idx = gy.astype(np.int64) * grid_width + gx.astype(np.int64)

    # Max-z heightmap per cell (kept here for the point-count/BEV-style
    # panels; terrain slope now comes from ground.build_elevation_grid
    # on ground-only points instead - see #6).
    order_z = np.argsort(flat_cell_idx, kind="stable")
    sorted_cells_z = flat_cell_idx[order_z]
    sorted_z = z_valid[order_z]
    unique_cells_z, start_z, counts_z = np.unique(sorted_cells_z, return_index=True, return_counts=True)
    for cell_flat, start, count in zip(unique_cells_z, start_z, counts_z):
        cy, cx = divmod(int(cell_flat), grid_width)
        elevation_grid[cy, cx] = np.max(sorted_z[start:start + count])
        point_count_grid[cy, cx] = count

    # Vectorized dominant-label vote with an obstacle-priority override.
    order = np.argsort(flat_cell_idx, kind="stable")
    sorted_cells = flat_cell_idx[order]
    sorted_labels = labels_valid[order]
    unique_cells, start_idx, counts = np.unique(sorted_cells, return_index=True, return_counts=True)

    for cell_flat, start, count in zip(unique_cells, start_idx, counts):
        cell_labels = sorted_labels[start:start + count]
        uniq, label_counts = np.unique(cell_labels, return_counts=True)

        # #7: restrict the vote to "always-block" classes first, if any
        # are present at all, so a lone building/vegetation/pole point
        # can't be outvoted by a larger patch of adjacent road/terrain
        # points sharing the same cell.
        force_mask = np.isin(uniq, list(FORCE_NON_TRAVERSABLE_CLASSES))
        if np.any(force_mask):
            uniq_f, counts_f = uniq[force_mask], label_counts[force_mask]
            dominant_label = uniq_f[np.argmax(counts_f)]
            dominant_count = int(counts_f.max())
        else:
            dominant_label = uniq[np.argmax(label_counts)]
            dominant_count = int(label_counts.max())

        cy, cx = divmod(int(cell_flat), grid_width)
        semantic_grid[cy, cx] = dominant_label
        confidence_grid[cy, cx] = dominant_count / int(count)  # #24

    return elevation_grid, semantic_grid, point_count_grid, confidence_grid


# ============================================================
# SEMANTIC-AWARE VARIABLE RESOLUTION  (#4 continuous distance tiers
# kept simple here since per-point resolution must stay one of a small
# discrete set for the hashing scheme to work; the *terrain* adaptive
# resolution in ground.py is the one made fully continuous)
# ============================================================

def semantic_aware_variable_grid(xyz: np.ndarray, semantic_labels: np.ndarray, cfg: Config,
                                  x_min: float, x_max: float, y_min: float, y_max: float):
    adaptive_cells: Dict[Tuple[int, int, float], dict] = {}

    dist = np.hypot(xyz[:, 0], xyz[:, 1])
    in_bounds = (xyz[:, 0] >= x_min) & (xyz[:, 0] < x_max) & (xyz[:, 1] >= y_min) & (xyz[:, 1] < y_max)

    for i in np.nonzero(in_bounds)[0]:
        x, y, z = xyz[i]
        label = int(semantic_labels[i])

        if dist[i] < cfg.adaptive_near_radius_m:
            r = cfg.adaptive_min_resolution
        elif dist[i] < cfg.adaptive_mid_radius_m:
            r = (cfg.adaptive_min_resolution + cfg.adaptive_max_resolution) / 2
        else:
            r = cfg.adaptive_max_resolution

        if label in HIGH_IMPORTANCE_CLASSES:
            r = cfg.adaptive_min_resolution
        elif label in MEDIUM_IMPORTANCE_CLASSES:
            r = min(r, (cfg.adaptive_min_resolution + cfg.adaptive_max_resolution) / 2)

        gx = int(np.floor((x - x_min) / r))
        gy = int(np.floor((y - y_min) / r))
        key = (gx, gy, r)

        if key not in adaptive_cells:
            adaptive_cells[key] = {
                "x": x, "y": y, "resolution": r, "elevation": z,
                "point_count": 1, "semantic_label": label,
            }
        else:
            cell = adaptive_cells[key]
            cell["elevation"] = max(cell["elevation"], z)
            cell["point_count"] += 1

    for cell in adaptive_cells.values():
        cell["semantic_name"] = CLASS_NAMES.get(cell["semantic_label"], "Unknown")

    return adaptive_cells


def variable_resolution_occupancy(adaptive_cells: dict):
    from config import TRAVERSABLE_CLASSES
    out = []
    for cell in adaptive_cells.values():
        label = cell["semantic_label"]
        if label in OBSTACLE_CLASSES:
            occupancy = 1
        elif label in TRAVERSABLE_CLASSES:
            occupancy = 0
        else:
            occupancy = -1
        out.append({**cell, "occupancy": occupancy})
    return out


def rasterize_variable_grid(variable_occupancy, base_shape, resolution, x_min, y_min):
    """
    #3: Project the finer semantic-aware variable grid's obstacle
    detections onto the base planning grid's resolution, so cells the
    base grid's own voting missed (because they were far enough away
    to only get a coarse 1.0 m base cell, while the variable grid still
    resolved a HIGH_IMPORTANCE object there at 0.25 m) still register
    as obstacles for planning.

    This is a bridge, not a true multi-resolution planner: information
    only flows one way (fine -> coarse obstacle override) and the
    planner still searches on the uniform base grid. A proper fix is
    a hierarchical/any-angle planner (e.g. D* Lite over the adaptive
    quad-cells directly) - left as documented future work.
    """
    grid_height, grid_width = base_shape
    fine_obstacle_override = np.zeros(base_shape, dtype=bool)

    for cell in variable_occupancy:
        if cell["occupancy"] != 1:
            continue
        gx = int((cell["x"] - x_min) / resolution)
        gy = int((cell["y"] - y_min) / resolution)
        if 0 <= gx < grid_width and 0 <= gy < grid_height:
            fine_obstacle_override[gy, gx] = True

    return fine_obstacle_override


def variable_grid_statistics(variable_occupancy):
    occupied = sum(c["occupancy"] == 1 for c in variable_occupancy)
    free = sum(c["occupancy"] == 0 for c in variable_occupancy)
    unknown = sum(c["occupancy"] == -1 for c in variable_occupancy)

    resolution_counts: Dict[float, int] = {}
    for cell in variable_occupancy:
        resolution_counts[cell["resolution"]] = resolution_counts.get(cell["resolution"], 0) + 1

    print("\n--- VARIABLE GRID STATISTICS ---")
    print("Total adaptive cells:", len(variable_occupancy))
    for r, count in sorted(resolution_counts.items()):
        print(f"{r:.2f} m cells:", count)
    print("Occupied:", occupied, "| Free:", free, "| Unknown:", unknown)