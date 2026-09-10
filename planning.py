# """
# planning.py
# ===========
# Cost map construction and A* path planning.

# Addresses:
#   #1  A* path planning does not use the probabilistic occupancy map
#       -> build_cost_map() now takes occupancy_probability directly and
#          folds P(occupied) into the per-cell cost (occupancy_prob_cost_weight),
#          and also treats a raycast-occupied cell as blocked even if the
#          semantic vote alone called it traversable (a moving/occluding
#          object the semantic classifier under-weighted still shows up
#          in the raycast).
#   #9  No vehicle footprint or obstacle inflation
#       -> the obstacle mask handed to the planner is the INFLATED mask
#          from occupancy.inflate_obstacles(), not the raw 1-cell mask.
#   #3  Adaptive grid not used for planning
#       -> fine_obstacle_override (from grid.rasterize_variable_grid) is
#          OR'd into the obstacle mask before inflation.
#   #2  A* heuristic is not scaled according to terrain cost
#       -> the heuristic multiplies Euclidean grid distance by the
#          MINIMUM finite cost anywhere in the cost grid, which keeps it
#          admissible (it can never overestimate true remaining cost)
#          while making it far tighter than a plain unweighted distance
#          once real terrain costs are >> 1. Config.heuristic_inflation
#          lets you deliberately trade optimality for speed (>1.0) if
#          a hackathon demo needs faster planning on a big grid.
#   #10 Unknown-cell handling -> Config.unknown_policy (block / penalize
#       / optimistic) instead of a hardcoded inf.
#   #12 A* produces jagged grid-based paths / #13 no path smoothing
#       -> smooth_path() does iterative line-of-sight shortcutting
#          (a lightweight "any-angle" post-process): it repeatedly tries
#          to connect non-adjacent waypoints directly and keeps the
#          shortcut only if every cell the straight segment crosses is
#          passable, which removes the staircase artifacts 8-connected
#          grid search produces on diagonal-ish routes.
#   #14 No collision verification after path generation
#       -> verify_path_collision_free() re-checks the FINAL (possibly
#          smoothed) path against the inflated obstacle grid before it's
#          trusted, independent of the search that produced it.
#   #20 Lack of comprehensive path-planning evaluation metrics
#       -> path_metrics() reports length, turn count, mean/min clearance,
#          total accumulated cost, and planning wall-clock time.
#   #16 Lack of dynamic obstacle handling
#       -> replan_with_new_obstacles() is a minimal, honestly-scoped hook:
#          given a list of newly-observed obstacle cells (e.g. from a
#          later frame or a live perception callback) it patches the
#          cost grid and re-runs A* from the platform's current position.
#          It does NOT do tracking/prediction of moving objects - that
#          needs multi-frame association this single-frame pipeline
#          doesn't have inputs for - so treat this as the integration
#          point a future tracker would call into, not a finished
#          dynamic-obstacle system.
# """

# import math
# import time
# import heapq
# from typing import List, Optional, Tuple

# import numpy as np

# from config import Config, SEMANTIC_RISK_WEIGHTS


# # ============================================================
# # COORDINATE HELPERS
# # ============================================================

# def world_to_grid(x, y, x_min, y_min, resolution) -> Tuple[int, int]:
#     return int((x - x_min) / resolution), int((y - y_min) / resolution)


# def grid_to_world(gx, gy, x_min, y_min, resolution) -> Tuple[float, float]:
#     return x_min + (gx + 0.5) * resolution, y_min + (gy + 0.5) * resolution


# # ============================================================
# # COST MAP  (#1, #9, #3, #10, #15)
# # ============================================================

# def build_cost_map(
#     traversability_grid: np.ndarray,
#     slope_degrees: np.ndarray,
#     semantic_grid: np.ndarray,
#     occupancy_probability: np.ndarray,
#     inflated_obstacle_mask: np.ndarray,
#     cfg: Config,
# ) -> np.ndarray:
#     """
#     Fuse traversability, slope, per-class semantic risk, raycast
#     occupancy probability, and the inflated obstacle footprint into one
#     navigation cost grid (#15: "cost map does not fully integrate all
#     available information" - this is the integration point).
#     """
#     shape = traversability_grid.shape

#     if cfg.unknown_policy == "block":
#         unknown_cost = np.inf
#     elif cfg.unknown_policy == "optimistic":
#         unknown_cost = 1.0
#     else:  # "penalize"
#         unknown_cost = cfg.unknown_penalty_multiplier

#     normalized_slope = np.clip(slope_degrees / max(cfg.max_slope_deg, 1e-6), 0.0, 1.0)
#     normalized_slope = np.where(np.isnan(normalized_slope), 1.0, normalized_slope)

#     risk_weight = np.full(shape, 1.5, dtype=np.float64)  # default for unlisted classes
#     for label, weight in SEMANTIC_RISK_WEIGHTS.items():
#         risk_weight = np.where(semantic_grid == label, weight, risk_weight)

#     occupancy_term = 1.0 + cfg.occupancy_prob_cost_weight * np.nan_to_num(occupancy_probability, nan=0.0)

#     base_cost = (1.0 + normalized_slope) * risk_weight * occupancy_term

#     cost_grid = np.select(
#         [traversability_grid == 0, traversability_grid == 2],
#         [np.inf, unknown_cost],
#         default=base_cost,
#     )

#     # #1/#9/#3: a cell the raycast or the fine variable grid flagged as
#     # (inflated) obstacle is blocked outright, regardless of what the
#     # semantic vote alone said.
#     cost_grid = np.where(inflated_obstacle_mask, np.inf, cost_grid)

#     return cost_grid


# # ============================================================
# # A*  (#2 admissible terrain-scaled heuristic)
# # ============================================================

# _NEIGHBORS = [
#     (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
#     (-1, 0, 1.0),                          (1, 0, 1.0),
#     (-1, 1, math.sqrt(2)),  (0, 1, 1.0),   (1, 1, math.sqrt(2)),
# ]


# def a_star_search(cost_grid: np.ndarray, start_cell, goal_cell, cfg: Config):
#     grid_height, grid_width = cost_grid.shape

#     def in_bounds(gx, gy):
#         return 0 <= gx < grid_width and 0 <= gy < grid_height

#     def passable(gx, gy):
#         return np.isfinite(cost_grid[gy, gx])

#     sx, sy = start_cell
#     tx, ty = goal_cell

#     if not in_bounds(sx, sy) or not in_bounds(tx, ty):
#         print("A*: start or goal is outside the grid bounds.")
#         return None
#     if not passable(sx, sy):
#         print("A*: start cell is not traversable.")
#         return None
#     if not passable(tx, ty):
#         print("A*: goal cell is not traversable.")
#         return None

#     # #2: admissible heuristic = Euclidean grid distance * cheapest
#     # possible per-step cost anywhere on the map. Still never
#     # overestimates true cost-to-go (every real step costs at least
#     # this much), but is far tighter than distance-only once typical
#     # terrain costs are several times 1.0.
#     finite_costs = cost_grid[np.isfinite(cost_grid)]
#     min_cost = float(finite_costs.min()) if finite_costs.size else 1.0
#     min_cost = max(min_cost, 1e-6)

#     def heuristic(gx, gy):
#         return math.hypot(gx - tx, gy - ty) * min_cost * cfg.heuristic_inflation

#     open_heap = [(heuristic(sx, sy), 0.0, (sx, sy))]
#     came_from = {}
#     g_score = {(sx, sy): 0.0}
#     visited = set()

#     while open_heap:
#         _, current_g, current = heapq.heappop(open_heap)
#         if current in visited:
#             continue
#         visited.add(current)

#         if current == (tx, ty):
#             path = [current]
#             while current in came_from:
#                 current = came_from[current]
#                 path.append(current)
#             path.reverse()
#             return path

#         cx, cy = current
#         for dx, dy, step_cost in _NEIGHBORS:
#             nx, ny = cx + dx, cy + dy
#             if not in_bounds(nx, ny) or not passable(nx, ny):
#                 continue

#             move_cost = step_cost * cost_grid[ny, nx]
#             tentative_g = current_g + move_cost
#             neighbor = (nx, ny)

#             if tentative_g < g_score.get(neighbor, np.inf):
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g
#                 f_score = tentative_g + heuristic(nx, ny)
#                 heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

#     print("A*: no path found between start and goal.")
#     return None


# # ============================================================
# # PATH SMOOTHING  (#12, #13) + COLLISION VERIFICATION (#14)
# # ============================================================

# def _line_is_passable(cost_grid: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
#     """Bresenham-walk a straight grid line and confirm every cell it
#     touches is finite-cost (used by both smoothing and verification)."""
#     x0, y0 = a
#     x1, y1 = b
#     n_steps = max(abs(x1 - x0), abs(y1 - y0), 1)

#     for i in range(n_steps + 1):
#         t = i / n_steps
#         gx = int(round(x0 + (x1 - x0) * t))
#         gy = int(round(y0 + (y1 - y0) * t))
#         if not (0 <= gy < cost_grid.shape[0] and 0 <= gx < cost_grid.shape[1]):
#             return False
#         if not np.isfinite(cost_grid[gy, gx]):
#             return False
#     return True


# def smooth_path(cell_path: List[Tuple[int, int]], cost_grid: np.ndarray, cfg: Config):
#     """
#     Greedy line-of-sight shortcutting: from each waypoint, jump as far
#     forward along the path as a straight, fully-passable line allows.
#     Removes the staircase jaggedness inherent to 8-connected grid A*
#     (#12) without needing a different search algorithm, and directly
#     implements the missing smoothing step (#13).
#     """
#     if cell_path is None or len(cell_path) < 3:
#         return cell_path

#     smoothed = [cell_path[0]]
#     i = 0
#     checks = 0

#     while i < len(cell_path) - 1:
#         j = len(cell_path) - 1
#         advanced = False
#         while j > i + 1:
#             checks += 1
#             if checks > cfg.smoothing_max_shortcut_checks:
#                 j = i + 1
#                 break
#             if _line_is_passable(cost_grid, cell_path[i], cell_path[j]):
#                 smoothed.append(cell_path[j])
#                 i = j
#                 advanced = True
#                 break
#             j -= 1
#         if not advanced:
#             i += 1
#             smoothed.append(cell_path[i])

#     return smoothed


# def verify_path_collision_free(cell_path: List[Tuple[int, int]], cost_grid: np.ndarray) -> bool:
#     """#14: independent post-hoc check of the final path (after any
#     smoothing) against the cost grid, segment by segment."""
#     if not cell_path or len(cell_path) < 2:
#         return cell_path is not None
#     return all(
#         _line_is_passable(cost_grid, cell_path[i], cell_path[i + 1])
#         for i in range(len(cell_path) - 1)
#     )


# # ============================================================
# # PLANNING ENTRY POINT
# # ============================================================

# def plan_path(cost_grid: np.ndarray, start_world, goal_world, cfg: Config,
#               x_min: float, y_min: float):
#     t0 = time.perf_counter()

#     start_cell = world_to_grid(*start_world, x_min, y_min, cfg.resolution)
#     goal_cell = world_to_grid(*goal_world, x_min, y_min, cfg.resolution)

#     cell_path = a_star_search(cost_grid, start_cell, goal_cell, cfg)
#     if cell_path is None:
#         return None, None, time.perf_counter() - t0

#     cell_path = smooth_path(cell_path, cost_grid, cfg)

#     if not verify_path_collision_free(cell_path, cost_grid):
#         print("WARNING: smoothed path failed collision verification; "
#               "falling back to the raw (unsmoothed) A* path.")
#         cell_path = a_star_search(cost_grid, start_cell, goal_cell, cfg)
#         if cell_path is None or not verify_path_collision_free(cell_path, cost_grid):
#             return None, None, time.perf_counter() - t0

#     world_path = [grid_to_world(gx, gy, x_min, y_min, cfg.resolution) for gx, gy in cell_path]
#     elapsed = time.perf_counter() - t0
#     return world_path, cell_path, elapsed


# def replan_with_new_obstacles(cost_grid: np.ndarray, new_obstacle_cells: List[Tuple[int, int]],
#                                 current_cell: Tuple[int, int], goal_cell: Tuple[int, int], cfg: Config):
#     """
#     #16 (scoped honestly - see module docstring): patch newly-reported
#     obstacle cells into an existing cost grid and re-run A* from the
#     platform's current cell. This is the hook a live obstacle tracker
#     would call; it does not itself track or predict motion.
#     """
#     patched = cost_grid.copy()
#     for gx, gy in new_obstacle_cells:
#         if 0 <= gy < patched.shape[0] and 0 <= gx < patched.shape[1]:
#             patched[gy, gx] = np.inf

#     cell_path = a_star_search(patched, current_cell, goal_cell, cfg)
#     cell_path = smooth_path(cell_path, patched, cfg) if cell_path else None
#     return patched, cell_path


# # ============================================================
# # PATH METRICS  (#20)
# # ============================================================

# def path_metrics(world_path: Optional[List[Tuple[float, float]]], cost_grid: np.ndarray,
#                   x_min: float, y_min: float, resolution: float, planning_time_s: float):
#     if not world_path or len(world_path) < 2:
#         return {
#             "found": world_path is not None,
#             "length_m": 0.0, "num_waypoints": 0, "num_turns": 0,
#             "total_cost": 0.0, "mean_clearance_cells": 0.0,
#             "min_clearance_cells": 0.0, "planning_time_s": planning_time_s,
#         }

#     pts = np.array(world_path)
#     seg_vectors = np.diff(pts, axis=0)
#     seg_lengths = np.linalg.norm(seg_vectors, axis=1)
#     length_m = float(seg_lengths.sum())

#     headings = np.arctan2(seg_vectors[:, 1], seg_vectors[:, 0])
#     turn_angles = np.abs(np.diff(headings))
#     turn_angles = np.minimum(turn_angles, 2 * np.pi - turn_angles)
#     num_turns = int(np.sum(turn_angles > np.radians(5)))

#     total_cost = 0.0
#     for (x0, y0), (x1, y1) in zip(world_path[:-1], world_path[1:]):
#         gx, gy = world_to_grid(x1, y1, x_min, y_min, resolution)
#         if 0 <= gy < cost_grid.shape[0] and 0 <= gx < cost_grid.shape[1]:
#             c = cost_grid[gy, gx]
#             total_cost += float(c) if np.isfinite(c) else 0.0

#     from scipy import ndimage
#     obstacle_mask = ~np.isfinite(cost_grid)
#     if obstacle_mask.any() and (~obstacle_mask).any():
#         distance_field = ndimage.distance_transform_edt(~obstacle_mask)
#         clearances = [
#             distance_field[gy, gx]
#             for gx, gy in (world_to_grid(x, y, x_min, y_min, resolution) for x, y in world_path)
#             if 0 <= gy < distance_field.shape[0] and 0 <= gx < distance_field.shape[1]
#         ]
#         mean_clearance = float(np.mean(clearances)) if clearances else 0.0
#         min_clearance = float(np.min(clearances)) if clearances else 0.0
#     else:
#         mean_clearance = min_clearance = float("inf")

#     return {
#         "found": True,
#         "length_m": length_m,
#         "num_waypoints": len(world_path),
#         "num_turns": num_turns,
#         "total_cost": total_cost,
#         "mean_clearance_cells": mean_clearance,
#         "min_clearance_cells": min_clearance,
#         "planning_time_s": planning_time_s,
#     }
"""
planning.py
===========
Cost map construction and A* path planning.

Addresses:
  #1  A* path planning does not use the probabilistic occupancy map
      -> build_cost_map() now takes occupancy_probability directly and
         folds P(occupied) into the per-cell cost (occupancy_prob_cost_weight),
         and also treats a raycast-occupied cell as blocked even if the
         semantic vote alone called it traversable (a moving/occluding
         object the semantic classifier under-weighted still shows up
         in the raycast).
  #9  No vehicle footprint or obstacle inflation
      -> the obstacle mask handed to the planner is the INFLATED mask
         from occupancy.inflate_obstacles(), not the raw 1-cell mask.
  #3  Adaptive grid not used for planning
      -> fine_obstacle_override (from grid.rasterize_variable_grid) is
         OR'd into the obstacle mask before inflation.
  #2  A* heuristic is not scaled according to terrain cost
      -> the heuristic multiplies Euclidean grid distance by the
         MINIMUM finite cost anywhere in the cost grid, which keeps it
         admissible (it can never overestimate true remaining cost)
         while making it far tighter than a plain unweighted distance
         once real terrain costs are >> 1. Config.heuristic_inflation
         lets you deliberately trade optimality for speed (>1.0) if
         a hackathon demo needs faster planning on a big grid.
  #10 Unknown-cell handling -> Config.unknown_policy (block / penalize
      / optimistic) instead of a hardcoded inf.
  #12 A* produces jagged grid-based paths / #13 no path smoothing
      -> smooth_path() does iterative line-of-sight shortcutting
         (a lightweight "any-angle" post-process): it repeatedly tries
         to connect non-adjacent waypoints directly and keeps the
         shortcut only if every cell the straight segment crosses is
         passable, which removes the staircase artifacts 8-connected
         grid search produces on diagonal-ish routes.
  #14 No collision verification after path generation
      -> verify_path_collision_free() re-checks the FINAL (possibly
         smoothed) path against the inflated obstacle grid before it's
         trusted, independent of the search that produced it.
  #20 Lack of comprehensive path-planning evaluation metrics
      -> path_metrics() reports length, turn count, mean/min clearance,
         total accumulated cost, and planning wall-clock time.
  #16 Lack of dynamic obstacle handling
      -> replan_with_new_obstacles() is a minimal, honestly-scoped hook:
         given a list of newly-observed obstacle cells (e.g. from a
         later frame or a live perception callback) it patches the
         cost grid and re-runs A* from the platform's current position.
"""

import math
import time
import heapq
from typing import List, Optional, Tuple

import numpy as np

from config import Config, SEMANTIC_RISK_WEIGHTS


# ============================================================
# COORDINATE HELPERS
# ============================================================

def world_to_grid(x, y, x_min, y_min, resolution) -> Tuple[int, int]:
    return int((x - x_min) / resolution), int((y - y_min) / resolution)


def grid_to_world(gx, gy, x_min, y_min, resolution) -> Tuple[float, float]:
    return x_min + (gx + 0.5) * resolution, y_min + (gy + 0.5) * resolution


def find_nearest_passable_cell(cost_grid: np.ndarray, cell: Tuple[int, int], max_radius: int = 5) -> Tuple[int, int]:
    """Snap start or goal cell to the nearest finite-cost cell if trapped in inflation."""
    gx, gy = cell
    height, width = cost_grid.shape

    if 0 <= gy < height and 0 <= gx < width and np.isfinite(cost_grid[gy, gx]):
        return cell

    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= ny < height and 0 <= nx < width:
                    if np.isfinite(cost_grid[ny, nx]):
                        return (nx, ny)
    return cell


# ============================================================
# COST MAP  (#1, #9, #3, #10, #15)
# ============================================================

def build_cost_map(
    traversability_grid: np.ndarray,
    slope_degrees: np.ndarray,
    semantic_grid: np.ndarray,
    occupancy_probability: np.ndarray,
    inflated_obstacle_mask: np.ndarray,
    cfg: Config,
    start_world: Tuple[float, float] = (0.0, 0.0),
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> np.ndarray:
    """
    Fuse traversability, slope, per-class semantic risk, raycast
    occupancy probability, and the inflated obstacle footprint into one
    navigation cost grid.
    """
    shape = traversability_grid.shape

    if cfg.unknown_policy == "block":
        unknown_cost = np.inf
    elif cfg.unknown_policy == "optimistic":
        unknown_cost = 1.0
    else:  # "penalize"
        unknown_cost = cfg.unknown_penalty_multiplier

    normalized_slope = np.clip(slope_degrees / max(cfg.max_slope_deg, 1e-6), 0.0, 1.0)
    normalized_slope = np.where(np.isnan(normalized_slope), 1.0, normalized_slope)

    risk_weight = np.full(shape, 1.5, dtype=np.float64)  # default for unlisted classes
    for label, weight in SEMANTIC_RISK_WEIGHTS.items():
        risk_weight = np.where(semantic_grid == label, weight, risk_weight)

    occupancy_term = 1.0 + cfg.occupancy_prob_cost_weight * np.nan_to_num(occupancy_probability, nan=0.0)

    base_cost = (1.0 + normalized_slope) * risk_weight * occupancy_term

    cost_grid = np.select(
        [traversability_grid == 0, traversability_grid == 2],
        [np.inf, unknown_cost],
        default=base_cost,
    )

    # Make a working copy of the inflated obstacle mask to avoid in-place side effects
    effective_mask = inflated_obstacle_mask.copy()

    # Clear ego vehicle origin footprint so inflation mask doesn't lock out starting point
    sx, sy = world_to_grid(*start_world, x_min, y_min, cfg.resolution)
    if 0 <= sy < shape[0] and 0 <= sx < shape[1]:
        effective_mask[sy, sx] = False

    cost_grid = np.where(effective_mask, np.inf, cost_grid)

    return cost_grid


# ============================================================
# A*  (#2 admissible terrain-scaled heuristic)
# ============================================================

_NEIGHBORS = [
    (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
    (-1, 0, 1.0),                          (1, 0, 1.0),
    (-1, 1, math.sqrt(2)),  (0, 1, 1.0),   (1, 1, math.sqrt(2)),
]


def a_star_search(cost_grid: np.ndarray, start_cell: Tuple[int, int], goal_cell: Tuple[int, int], cfg: Config):
    grid_height, grid_width = cost_grid.shape

    def in_bounds(gx, gy):
        return 0 <= gx < grid_width and 0 <= gy < grid_height

    def passable(gx, gy):
        return np.isfinite(cost_grid[gy, gx])

    sx, sy = start_cell
    tx, ty = goal_cell

    if not in_bounds(sx, sy) or not in_bounds(tx, ty):
        print("A*: start or goal is outside the grid bounds.")
        return None
    if not passable(sx, sy):
        print("A*: start cell is not traversable.")
        return None
    if not passable(tx, ty):
        print("A*: goal cell is not traversable.")
        return None

    finite_costs = cost_grid[np.isfinite(cost_grid)]
    min_cost = float(finite_costs.min()) if finite_costs.size else 1.0
    min_cost = max(min_cost, 1e-6)

    def heuristic(gx, gy):
        return math.hypot(gx - tx, gy - ty) * min_cost * cfg.heuristic_inflation

    open_heap = [(heuristic(sx, sy), 0.0, (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    visited = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == (tx, ty):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, step_cost in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny) or not passable(nx, ny):
                continue

            move_cost = step_cost * cost_grid[ny, nx]
            tentative_g = current_g + move_cost
            neighbor = (nx, ny)

            if tentative_g < g_score.get(neighbor, np.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(nx, ny)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    print("A*: no path found between start and goal.")
    return None


# ============================================================
# PATH SMOOTHING  (#12, #13) + COLLISION VERIFICATION (#14)
# ============================================================

def _line_is_passable(cost_grid: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Bresenham-walk a straight grid line and confirm every cell it touches is finite-cost."""
    x0, y0 = a
    x1, y1 = b
    n_steps = max(abs(x1 - x0), abs(y1 - y0), 1)

    for i in range(n_steps + 1):
        t = i / n_steps
        gx = int(round(x0 + (x1 - x0) * t))
        gy = int(round(y0 + (y1 - y0) * t))
        if not (0 <= gy < cost_grid.shape[0] and 0 <= gx < cost_grid.shape[1]):
            return False
        if not np.isfinite(cost_grid[gy, gx]):
            return False
    return True


def smooth_path(cell_path: List[Tuple[int, int]], cost_grid: np.ndarray, cfg: Config):
    """Greedy line-of-sight shortcutting to eliminate jagged staircase paths."""
    if cell_path is None or len(cell_path) < 3:
        return cell_path

    smoothed = [cell_path[0]]
    i = 0
    checks = 0

    while i < len(cell_path) - 1:
        j = len(cell_path) - 1
        advanced = False
        while j > i + 1:
            checks += 1
            if checks > cfg.smoothing_max_shortcut_checks:
                j = i + 1
                break
            if _line_is_passable(cost_grid, cell_path[i], cell_path[j]):
                smoothed.append(cell_path[j])
                i = j
                advanced = True
                break
            j -= 1
        if not advanced:
            i += 1
            smoothed.append(cell_path[i])

    return smoothed


def verify_path_collision_free(cell_path: List[Tuple[int, int]], cost_grid: np.ndarray) -> bool:
    """Independent post-hoc check of the final path against the cost grid."""
    if not cell_path or len(cell_path) < 2:
        return cell_path is not None
    return all(
        _line_is_passable(cost_grid, cell_path[i], cell_path[i + 1])
        for i in range(len(cell_path) - 1)
    )


# ============================================================
# PLANNING ENTRY POINT
# ============================================================

def plan_path(cost_grid: np.ndarray, start_world: Tuple[float, float], goal_world: Tuple[float, float], cfg: Config,
              x_min: float, y_min: float):
    t0 = time.perf_counter()

    start_cell = world_to_grid(*start_world, x_min, y_min, cfg.resolution)
    goal_cell = world_to_grid(*goal_world, x_min, y_min, cfg.resolution)

    # Snap blocked endpoints to nearest passable cell
    start_cell = find_nearest_passable_cell(cost_grid, start_cell)
    goal_cell = find_nearest_passable_cell(cost_grid, goal_cell)

    cell_path = a_star_search(cost_grid, start_cell, goal_cell, cfg)
    if cell_path is None:
        return None, None, time.perf_counter() - t0

    cell_path = smooth_path(cell_path, cost_grid, cfg)

    if not verify_path_collision_free(cell_path, cost_grid):
        print("WARNING: smoothed path failed collision verification; falling back to raw A* path.")
        cell_path = a_star_search(cost_grid, start_cell, goal_cell, cfg)
        if cell_path is None or not verify_path_collision_free(cell_path, cost_grid):
            return None, None, time.perf_counter() - t0

    world_path = [grid_to_world(gx, gy, x_min, y_min, cfg.resolution) for gx, gy in cell_path]
    elapsed = time.perf_counter() - t0
    return world_path, cell_path, elapsed


def replan_with_new_obstacles(cost_grid: np.ndarray, new_obstacle_cells: List[Tuple[int, int]],
                              current_cell: Tuple[int, int], goal_cell: Tuple[int, int], cfg: Config):
    """Patch dynamic obstacle cells into an existing grid and re-run path planning."""
    patched = cost_grid.copy()
    for gx, gy in new_obstacle_cells:
        if 0 <= gy < patched.shape[0] and 0 <= gx < patched.shape[1]:
            patched[gy, gx] = np.inf

    cell_path = a_star_search(patched, current_cell, goal_cell, cfg)
    cell_path = smooth_path(cell_path, patched, cfg) if cell_path else None
    return patched, cell_path


# ============================================================
# PATH METRICS  (#20)
# ============================================================

def path_metrics(world_path: Optional[List[Tuple[float, float]]], cost_grid: np.ndarray,
                 x_min: float, y_min: float, resolution: float, planning_time_s: float):
    if not world_path or len(world_path) < 2:
        return {
            "found": world_path is not None,
            "length_m": 0.0, "num_waypoints": 0, "num_turns": 0,
            "total_cost": 0.0, "mean_clearance_cells": 0.0,
            "min_clearance_cells": 0.0, "planning_time_s": planning_time_s,
        }

    pts = np.array(world_path)
    seg_vectors = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(seg_vectors, axis=1)
    length_m = float(seg_lengths.sum())

    headings = np.arctan2(seg_vectors[:, 1], seg_vectors[:, 0])
    turn_angles = np.abs(np.diff(headings))
    turn_angles = np.minimum(turn_angles, 2 * np.pi - turn_angles)
    num_turns = int(np.sum(turn_angles > np.radians(5)))

    total_cost = 0.0
    for (x0, y0), (x1, y1) in zip(world_path[:-1], world_path[1:]):
        gx, gy = world_to_grid(x1, y1, x_min, y_min, resolution)
        if 0 <= gy < cost_grid.shape[0] and 0 <= gx < cost_grid.shape[1]:
            c = cost_grid[gy, gx]
            total_cost += float(c) if np.isfinite(c) else 0.0

    from scipy import ndimage
    obstacle_mask = ~np.isfinite(cost_grid)
    if obstacle_mask.any() and (~obstacle_mask).any():
        distance_field = ndimage.distance_transform_edt(~obstacle_mask)
        clearances = [
            distance_field[gy, gx]
            for gx, gy in (world_to_grid(x, y, x_min, y_min, resolution) for x, y in world_path)
            if 0 <= gy < distance_field.shape[0] and 0 <= gx < distance_field.shape[1]
        ]
        mean_clearance = float(np.mean(clearances)) if clearances else 0.0
        min_clearance = float(np.min(clearances)) if clearances else 0.0
    else:
        mean_clearance = min_clearance = float("inf")

    return {
        "found": True,
        "length_m": length_m,
        "num_waypoints": len(world_path),
        "num_turns": num_turns,
        "total_cost": total_cost,
        "mean_clearance_cells": mean_clearance,
        "min_clearance_cells": min_clearance,
        "planning_time_s": planning_time_s,
    }