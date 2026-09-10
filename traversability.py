"""
traversability.py
==================
Semantic + slope -> traversability grid.

Addresses:
  #21 No real-time performance optimization
      -> The original traversability_analysis() ran a Python double
         for-loop over every (gy, gx) cell (H*W Python-level iterations,
         e.g. 200x200=40,000 iterations for a 100m grid at 0.5m
         resolution, each doing dict/set lookups). That is now fully
         vectorized with numpy boolean masks and np.select - one pass
         over the whole grid instead of 40k+ interpreter-level
         iterations. This is the single biggest real-time win in the
         pipeline since traversability_analysis ran on every frame.
  #10 Limited handling of unknown grid cells
      -> traversability values are still {0: blocked, 1: traversable,
         2: unknown}, but planning.py now reads Config.unknown_policy
         to decide how "2" is costed (block / heavy-penalize /
         optimistic) instead of the cost map hardcoding unknown=inf.
"""

import numpy as np

from config import (
    Config, OBSTACLE_CLASSES, TRAVERSABLE_CLASSES,
    FORCE_NON_TRAVERSABLE_CLASSES, FORCE_TRAVERSABLE_CHECK_CLASSES,
    CLASS_MAX_SLOPE, get_class_max_slope,
)


def traversability_analysis(semantic_grid: np.ndarray, slope_degrees: np.ndarray,
                             terrain_unknown: np.ndarray, cfg: Config):
    """
    Vectorized replacement for the original nested-loop implementation.
    Returns an int8 grid: 0 = not traversable, 1 = traversable, 2 = unknown.
    """
    shape = semantic_grid.shape
    traversability = np.full(shape, 2, dtype=np.int8)

    labeled = semantic_grid != -1

    force_block = np.isin(semantic_grid, list(FORCE_NON_TRAVERSABLE_CLASSES))
    obstacle = np.isin(semantic_grid, list(OBSTACLE_CLASSES)) & labeled
    force_check = np.isin(semantic_grid, list(FORCE_TRAVERSABLE_CHECK_CLASSES))
    traversable_class = np.isin(semantic_grid, list(TRAVERSABLE_CLASSES))

    # Per-class slope tolerance, broadcast across the grid.
    max_slope_grid = np.full(shape, cfg.max_slope_deg, dtype=np.float64)
    for label, slope_limit in CLASS_MAX_SLOPE.items():
        max_slope_grid = np.where(semantic_grid == label, slope_limit, max_slope_grid)

    within_slope = slope_degrees <= max_slope_grid
    slope_known = ~terrain_unknown & ~np.isnan(slope_degrees)

    # --- priority order mirrors the original per-cell logic ---
    # 1) unlabeled cell -> unknown
    traversability = np.where(~labeled, 2, traversability)

    # 2) always-blocked classes
    traversability = np.where(labeled & force_block, 0, traversability)

    # 3) generic obstacle classes (not already handled by force_block)
    traversability = np.where(labeled & obstacle & ~force_block, 0, traversability)

    remaining_ground = labeled & ~force_block & ~obstacle

    # 4) engineered ground (road/parking/sidewalk): graded purely on
    #    per-class slope tolerance once terrain is known.
    fc_unknown = remaining_ground & force_check & ~slope_known
    fc_ok = remaining_ground & force_check & slope_known & within_slope
    fc_block = remaining_ground & force_check & slope_known & ~within_slope
    traversability = np.where(fc_unknown, 2, traversability)
    traversability = np.where(fc_ok, 1, traversability)
    traversability = np.where(fc_block, 0, traversability)

    # 5) any other class not in TRAVERSABLE_CLASSES -> blocked
    other_ground = remaining_ground & ~force_check
    not_traversable_class = other_ground & ~traversable_class
    traversability = np.where(not_traversable_class, 0, traversability)

    # 6) remaining traversable-class ground: graded by (per-class) slope
    grade_cells = other_ground & traversable_class
    grade_unknown = grade_cells & ~slope_known
    grade_ok = grade_cells & slope_known & within_slope
    grade_block = grade_cells & slope_known & ~within_slope
    traversability = np.where(grade_unknown, 2, traversability)
    traversability = np.where(grade_ok, 1, traversability)
    traversability = np.where(grade_block, 0, traversability)

    return traversability.astype(np.int8)


def variable_resolution_traversability(variable_occupancy, slope_degrees, cfg: Config,
                                         x_min: float, y_min: float):
    """Per-point-cluster equivalent of traversability_analysis, kept
    as a Python loop since adaptive_cells is a dict of a few thousand
    entries at most (not a full HxW grid) - vectorizing this buys
    little given create_semantic_grid/traversability_analysis above
    already carry the real per-frame cost."""
    grid_height, grid_width = slope_degrees.shape

    for cell in variable_occupancy:
        x, y, label = cell["x"], cell["y"], cell["semantic_label"]
        gx = int((x - x_min) / cfg.resolution)
        gy = int((y - y_min) / cfg.resolution)

        local_slope = np.nan
        if 0 <= gx < grid_width and 0 <= gy < grid_height:
            local_slope = slope_degrees[gy, gx]
        cell["slope"] = local_slope

        if label in FORCE_NON_TRAVERSABLE_CLASSES:
            cell["traversability"] = 0
            continue

        if label in FORCE_TRAVERSABLE_CHECK_CLASSES:
            if np.isnan(local_slope):
                cell["traversability"] = 2
            elif local_slope <= get_class_max_slope(label, cfg.max_slope_deg):
                cell["traversability"] = 1
            else:
                cell["traversability"] = 0
            continue

        if cell["occupancy"] == 1:
            cell["traversability"] = 0
        elif cell["occupancy"] == -1:
            cell["traversability"] = 2
        elif np.isnan(local_slope):
            cell["traversability"] = 2
        elif local_slope > get_class_max_slope(label, cfg.max_slope_deg):
            cell["traversability"] = 0
        else:
            cell["traversability"] = 1

    return variable_occupancy