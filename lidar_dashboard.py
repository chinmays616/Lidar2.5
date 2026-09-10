"""
LiDAR 2.5D Semantic Mapping Dashboard
======================================

A browser-based dashboard (Streamlit + Plotly) for the semantic-KITTI
LiDAR mapping / costmap / path-planning pipeline.

Unlike the original desktop script (matplotlib windows + an Open3D
viewer, one frame at a time, run from the command line), this version:

  * Runs as a website: `streamlit run lidar_dashboard.py` opens a
    browser tab with sliders, tabs, and interactive charts.
  * Works on a whole SEQUENCE of frames, not just one: browse frame by
    frame with Prev/Next, or batch-process a frame range and see
    metrics/timing trends across the whole run.
  * Lets people without a local dataset folder drag-and-drop .bin /
    .label files straight into the browser.
  * Replaces the matplotlib CheckButtons dashboard and the Open3D
    viewer with Plotly figures, which render in-browser and need no
    display server (works on a laptop, a remote server, or a cloud
    deployment).
  * Caches results per frame + per settings, so flipping between tabs
    or re-visiting a frame you've already processed is instant.

Run it with:
    pip install streamlit plotly numpy scipy scikit-learn
    streamlit run lidar_dashboard.py

Expected dataset layout (standard SemanticKITTI):
    <root>/sequences/<seq>/velodyne/000000.bin, 000001.bin, ...
    <root>/sequences/<seq>/labels/000000.label, 000001.label, ...
(Or just drag-and-drop matching .bin/.label pairs in "Upload files" mode.)
"""

from __future__ import annotations

import io
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
)

import plotly.graph_objects as go


# ============================================================
# SEMANTIC CLASS TABLES  (unchanged from the original pipeline)
# ============================================================

TRAVERSABLE_CLASSES = {40, 44, 48, 49, 72}

OBSTACLE_CLASSES = {
    10, 11, 13, 15, 16, 18, 20,
    30, 31, 32,
    50, 51, 52,
    70, 71,
    80, 81,
}

CLASS_MAX_SLOPE = {40: 15.0, 44: 15.0, 48: 10.0, 49: 12.0, 72: 20.0}

FORCE_NON_TRAVERSABLE_CLASSES = {50, 51, 70, 71, 80}
FORCE_TRAVERSABLE_CHECK_CLASSES = {40, 44, 48}

SEMANTIC_RISK_WEIGHTS = {40: 1.0, 44: 1.2, 48: 1.5, 49: 1.8, 72: 2.0}

CLASS_NAMES = {
    0: "unlabeled", 1: "outlier",
    10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 16: "on-rails",
    18: "truck", 20: "other-vehicle",
    30: "person", 31: "bicyclist", 32: "motorcyclist",
    40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
    50: "building", 51: "fence", 52: "other-structure",
    60: "lane-marking",
    70: "vegetation", 71: "trunk", 72: "terrain",
    80: "pole", 81: "traffic-sign",
    99: "other", 254: "moving", 255: "unknown",
}

CLASS_COLORS = {
    0: (140, 140, 140), 1: (140, 140, 140),
    10: (255, 0, 0), 11: (255, 128, 0), 13: (255, 0, 128), 15: (204, 0, 0),
    16: (179, 51, 51), 18: (179, 0, 0), 20: (230, 51, 51),
    30: (255, 255, 0), 31: (255, 204, 0), 32: (204, 204, 0),
    40: (77, 77, 77), 44: (115, 115, 115), 48: (166, 166, 166), 49: (204, 191, 140),
    50: (140, 0, 0), 51: (153, 102, 51), 52: (102, 102, 140),
    60: (230, 230, 51),
    70: (0, 140, 0), 71: (0, 89, 0), 72: (89, 191, 51),
    80: (140, 0, 255), 81: (204, 0, 255),
    99: (204, 204, 204), 254: (255, 255, 255), 255: (51, 51, 51),
    -1: (255, 255, 255),
}


def rgb_str(label: int) -> str:
    r, g, b = CLASS_COLORS.get(int(label), (128, 128, 128))
    return f"rgb({r},{g},{b})"


def get_class_max_slope(label, default):
    return CLASS_MAX_SLOPE.get(int(label), default)


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class PipelineConfig:
    resolution: float = 0.5
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0

    max_slope: float = 15.0
    slope_display_max: float = 45.0

    max_interp_distance: float = 3.0
    interp_k_neighbors: int = 8

    sensor_origin: Tuple[float, float] = (0.0, 0.0)
    p_occ: float = 0.7
    p_free: float = 0.3
    log_odds_clamp: float = 10.0
    occ_prob_threshold: float = 0.65
    free_prob_threshold: float = 0.35
    raycast_stride: int = 1
    raycast_batch_size: int = 20000

    inflation_radius: float = 2.5
    inflation_decay_rate: float = 3.0
    inflation_max_cost: float = 50.0
    enable_inflation: bool = True

    unknown_policy: str = "penalize"          # block | penalize | optimistic
    unknown_penalty_multiplier: float = 6.0

    start_coord: Tuple[float, float] = (0.0, 0.0)
    goal_coord: Tuple[float, float] = (20.0, 15.0)
    max_snap_radius_cells: int = 25

    smooth_path: bool = True
    path_smooth_factor: float = 0.1
    path_smooth_samples: int = 200

    lateral_accel_max: float = 3.0
    default_speed: float = 5.0
    jerk_max: float = 5.0
    obstacle_proximity_threshold: float = 3.0

    max_3d_points: int = 30000


# ============================================================
# CORE PIPELINE (pure numpy/scipy — reusable, no GUI calls)
# ============================================================

def load_pointcloud(path: str):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points, points[:, :3], points[:, 3]


def load_semantic_labels(path: str, num_points: int):
    labels = np.fromfile(path, dtype=np.uint32)
    if num_points != len(labels):
        raise ValueError(
            f"Point/label count mismatch: {num_points} points vs {len(labels)} labels."
        )
    return labels & 0xFFFF


def create_semantic_grid(xyz, semantic_labels, cfg: PipelineConfig):
    r = cfg.resolution
    grid_width = int((cfg.x_max - cfg.x_min) / r)
    grid_height = int((cfg.y_max - cfg.y_min) / r)

    elevation_grid = np.full((grid_height, grid_width), np.nan)
    semantic_grid = np.full((grid_height, grid_width), -1, dtype=np.int32)
    point_count_grid = np.zeros((grid_height, grid_width), dtype=np.int32)

    gx = ((xyz[:, 0] - cfg.x_min) / r).astype(int)
    gy = ((xyz[:, 1] - cfg.y_min) / r).astype(int)
    valid = (gx >= 0) & (gx < grid_width) & (gy >= 0) & (gy < grid_height)

    gx_v, gy_v, z_v, lab_v = gx[valid], gy[valid], xyz[:, 2][valid], semantic_labels[valid]

    for x_, y_, z_ in zip(gx_v, gy_v, z_v):
        if np.isnan(elevation_grid[y_, x_]):
            elevation_grid[y_, x_] = z_
        else:
            elevation_grid[y_, x_] = max(elevation_grid[y_, x_], z_)
        point_count_grid[y_, x_] += 1

    if len(lab_v) > 0:
        flat = gy_v.astype(np.int64) * grid_width + gx_v.astype(np.int64)
        order = np.argsort(flat, kind="stable")
        flat_sorted, lab_sorted = flat[order], lab_v[order]
        uniq_cells, start_idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
        for cell_flat, start, count in zip(uniq_cells, start_idx, counts):
            cell_labels = lab_sorted[start:start + count]
            u, c = np.unique(cell_labels, return_counts=True)
            dominant = u[np.argmax(c)]
            semantic_grid[cell_flat // grid_width, cell_flat % grid_width] = dominant

    return elevation_grid, semantic_grid, point_count_grid


def _interpolate_elevation_idw(elevation_grid, resolution, max_distance, k):
    grid_height, grid_width = elevation_grid.shape
    valid_mask = ~np.isnan(elevation_grid)
    if not np.any(valid_mask):
        return elevation_grid.copy(), np.zeros_like(elevation_grid, dtype=bool)

    yy, xx = np.mgrid[0:grid_height, 0:grid_width]
    valid_coords = np.column_stack([xx[valid_mask], yy[valid_mask]]).astype(float) * resolution
    valid_values = elevation_grid[valid_mask]
    tree = cKDTree(valid_coords)

    all_coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float) * resolution
    k_query = min(k, len(valid_values))
    dist, idx = tree.query(all_coords, k=k_query, distance_upper_bound=max_distance)
    if k_query == 1:
        dist, idx = dist[:, None], idx[:, None]

    n_cells = all_coords.shape[0]
    filled = np.full(n_cells, np.nan)
    observed_mask = np.zeros(n_cells, dtype=bool)
    n_values = len(valid_values)

    for i in range(n_cells):
        d, ix = dist[i], idx[i]
        finite = np.isfinite(d) & (ix < n_values)
        if not np.any(finite):
            continue
        d, ix = d[finite], ix[finite]
        if d[0] < 1e-9:
            filled[i] = valid_values[ix[0]]
        else:
            w = 1.0 / (d ** 2)
            filled[i] = np.sum(w * valid_values[ix]) / np.sum(w)
        observed_mask[i] = True

    filled_grid = filled.reshape(grid_height, grid_width)
    observed_mask = observed_mask.reshape(grid_height, grid_width) | valid_mask
    filled_grid = np.where(observed_mask, filled_grid, np.nan)
    return filled_grid, observed_mask


def _nan_aware_gaussian(grid, sigma):
    nan_mask = np.isnan(grid)
    if not np.any(nan_mask):
        return ndimage.gaussian_filter(grid, sigma=sigma)
    data = np.where(nan_mask, 0.0, grid)
    weight = np.where(nan_mask, 0.0, 1.0)
    data_smooth = ndimage.gaussian_filter(data, sigma=sigma)
    weight_smooth = ndimage.gaussian_filter(weight, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = data_smooth / weight_smooth
    smoothed[weight_smooth < 1e-6] = np.nan
    return smoothed


def terrain_analysis(elevation_grid, cfg: PipelineConfig, smoothing_sigma=1.0):
    valid = ~np.isnan(elevation_grid)
    if not np.any(valid):
        unknown_mask = np.ones_like(elevation_grid, dtype=bool)
        return np.full_like(elevation_grid, np.nan), np.full_like(elevation_grid, np.nan), unknown_mask

    filled_grid, _ = _interpolate_elevation_idw(
        elevation_grid, cfg.resolution, cfg.max_interp_distance, cfg.interp_k_neighbors
    )
    if smoothing_sigma > 0:
        filled_grid = _nan_aware_gaussian(filled_grid, smoothing_sigma)

    grad_y, grad_x = np.gradient(filled_grid, cfg.resolution)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    slope_degrees = np.degrees(np.arctan(slope))

    unknown_mask = np.isnan(filled_grid) | np.isnan(slope_degrees)
    slope = np.where(unknown_mask, np.nan, slope)
    slope_degrees = np.where(unknown_mask, np.nan, slope_degrees)
    return slope, slope_degrees, unknown_mask


def traversability_analysis(elevation_grid, semantic_grid, cfg: PipelineConfig):
    slope, slope_degrees, terrain_unknown = terrain_analysis(elevation_grid, cfg)

    traversability_grid = np.full(semantic_grid.shape, 2, dtype=np.int8)  # 2 = unknown
    known_mask = semantic_grid != -1

    force_non_trav = np.isin(semantic_grid, list(FORCE_NON_TRAVERSABLE_CLASSES))
    traversability_grid[force_non_trav] = 0

    obstacle_mask = np.isin(semantic_grid, list(OBSTACLE_CLASSES)) & ~force_non_trav
    traversability_grid[obstacle_mask] = 0

    max_slope_grid = np.full(semantic_grid.shape, cfg.max_slope, dtype=np.float32)
    for cls_id, cls_max_s in CLASS_MAX_SLOPE.items():
        max_slope_grid[semantic_grid == cls_id] = cls_max_s

    is_trav_class = np.isin(semantic_grid, list(TRAVERSABLE_CLASSES))
    valid_terrain = known_mask & is_trav_class & ~terrain_unknown

    traversability_grid[valid_terrain & (slope_degrees <= max_slope_grid)] = 1
    traversability_grid[valid_terrain & (slope_degrees > max_slope_grid)] = 0

    unclassified = known_mask & ~is_trav_class & ~force_non_trav & ~obstacle_mask
    traversability_grid[unclassified] = 0

    return traversability_grid, slope_degrees


def probabilistic_occupancy_mapping(xyz, cfg: PipelineConfig):
    r = cfg.resolution
    grid_width = int((cfg.x_max - cfg.x_min) / r)
    grid_height = int((cfg.y_max - cfg.y_min) / r)

    l_occ = math.log(cfg.p_occ / (1.0 - cfg.p_occ))
    l_free = math.log(cfg.p_free / (1.0 - cfg.p_free))
    log_odds_flat = np.zeros(grid_height * grid_width, dtype=np.float64)

    ox, oy = cfg.sensor_origin
    origin_gx = min(max(int((ox - cfg.x_min) / r), 0), grid_width - 1)
    origin_gy = min(max(int((oy - cfg.y_min) / r), 0), grid_height - 1)

    stride = max(1, cfg.raycast_stride)
    points = xyz[::stride] if stride > 1 else xyz

    gx_all = ((points[:, 0] - cfg.x_min) / r).astype(np.int64)
    gy_all = ((points[:, 1] - cfg.y_min) / r).astype(np.int64)
    in_bounds = (gx_all >= 0) & (gx_all < grid_width) & (gy_all >= 0) & (gy_all < grid_height)
    gx_all, gy_all = gx_all[in_bounds], gy_all[in_bounds]

    n_points = len(gx_all)
    batch_size = max(1, cfg.raycast_batch_size)

    for start in range(0, n_points, batch_size):
        gxb, gyb = gx_all[start:start + batch_size], gy_all[start:start + batch_size]
        dx, dy = gxb - origin_gx, gyb - origin_gy
        n_steps_actual = np.maximum(np.abs(dx), np.abs(dy))
        n_steps_for_t = np.maximum(n_steps_actual, 1)
        max_steps = int(n_steps_for_t.max())

        k = np.arange(max_steps + 1)[:, None]
        t = k / n_steps_for_t[None, :]
        within_ray = t <= 1.0

        cell_x = np.rint(origin_gx + dx[None, :] * t).astype(np.int64)
        cell_y = np.rint(origin_gy + dy[None, :] * t).astype(np.int64)
        in_grid = (cell_x >= 0) & (cell_x < grid_width) & (cell_y >= 0) & (cell_y < grid_height)

        is_endpoint = within_ray & (k == n_steps_actual[None, :])
        is_free = within_ray & (~is_endpoint)
        free_mask = (is_free & in_grid).ravel()
        occ_mask = (is_endpoint & in_grid).ravel()
        flat_idx = (cell_y * grid_width + cell_x).ravel()

        if np.any(free_mask):
            np.add.at(log_odds_flat, flat_idx[free_mask], l_free)
        if np.any(occ_mask):
            np.add.at(log_odds_flat, flat_idx[occ_mask], l_occ)

    log_odds = log_odds_flat.reshape(grid_height, grid_width)
    np.clip(log_odds, -cfg.log_odds_clamp, cfg.log_odds_clamp, out=log_odds)
    probability_grid = 1.0 - 1.0 / (1.0 + np.exp(log_odds))

    occupancy_grid = np.full((grid_height, grid_width), -1, dtype=np.int8)
    occupancy_grid[probability_grid >= cfg.occ_prob_threshold] = 1
    occupancy_grid[probability_grid <= cfg.free_prob_threshold] = 0
    return occupancy_grid, probability_grid


def inflate_costmap(traversability_grid, cfg: PipelineConfig):
    obstacle_mask = traversability_grid == 0
    if not np.any(obstacle_mask):
        return np.zeros_like(traversability_grid, dtype=np.float64)
    dist_cells = ndimage.distance_transform_edt(~obstacle_mask)
    dist_meters = dist_cells * cfg.resolution
    inflation_cost = cfg.inflation_max_cost * np.exp(
        -cfg.inflation_decay_rate * dist_meters / cfg.inflation_radius
    )
    inflation_cost[dist_meters > cfg.inflation_radius] = 0.0
    inflation_cost[obstacle_mask] = np.inf
    return inflation_cost


def build_cost_map(traversability_grid, slope_degrees, semantic_grid, cfg: PipelineConfig):
    if cfg.unknown_policy == "block":
        unknown_cost = np.inf
    elif cfg.unknown_policy == "optimistic":
        unknown_cost = 1.0
    elif cfg.unknown_policy == "penalize":
        unknown_cost = cfg.unknown_penalty_multiplier
    else:
        raise ValueError(f"Unknown unknown_policy {cfg.unknown_policy!r}")

    normalized_slope = np.clip(slope_degrees / cfg.max_slope, 0.0, 1.0)
    normalized_slope = np.where(np.isnan(normalized_slope), 1.0, normalized_slope)

    risk_weight = np.full(traversability_grid.shape, 1.5, dtype=np.float64)
    for label, weight in SEMANTIC_RISK_WEIGHTS.items():
        risk_weight = np.where(semantic_grid == label, weight, risk_weight)

    traversable_cost = (1.0 + normalized_slope) * risk_weight
    cost_grid = np.select(
        [traversability_grid == 0, traversability_grid == 2],
        [np.inf, unknown_cost],
        default=traversable_cost,
    )

    if cfg.enable_inflation:
        inflation = inflate_costmap(traversability_grid, cfg)
        finite_mask = np.isfinite(cost_grid) & np.isfinite(inflation)
        cost_grid[finite_mask] += inflation[finite_mask]

    return cost_grid.astype(np.float64)


def world_to_grid(x, y, cfg: PipelineConfig):
    return int((x - cfg.x_min) / cfg.resolution), int((y - cfg.y_min) / cfg.resolution)


def grid_to_world(gx, gy, cfg: PipelineConfig):
    return cfg.x_min + (gx + 0.5) * cfg.resolution, cfg.y_min + (gy + 0.5) * cfg.resolution


def find_nearest_passable_cell(cost_grid, cell, max_radius):
    grid_height, grid_width = cost_grid.shape
    cx, cy = cell
    if 0 <= cx < grid_width and 0 <= cy < grid_height and np.isfinite(cost_grid[cy, cx]):
        return cell
    for radius in range(1, max_radius + 1):
        best, best_d2 = None, None
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < grid_width and 0 <= ny < grid_height):
                    continue
                if not np.isfinite(cost_grid[ny, nx]):
                    continue
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best, best_d2 = (nx, ny), d2
        if best is not None:
            return best
    return None


def a_star_search(cost_grid, start_cell, goal_cell):
    import heapq

    grid_height, grid_width = cost_grid.shape

    def in_bounds(gx, gy):
        return 0 <= gx < grid_width and 0 <= gy < grid_height

    def passable(gx, gy):
        return np.isfinite(cost_grid[gy, gx])

    sx, sy = start_cell
    gxg, gyg = goal_cell
    if not (in_bounds(sx, sy) and in_bounds(gxg, gyg)):
        return None
    if not (passable(sx, sy) and passable(gxg, gyg)):
        return None

    finite_costs = cost_grid[np.isfinite(cost_grid)]
    min_cost = max(float(finite_costs.min()) if finite_costs.size else 1.0, 1e-6)

    def heuristic(gx, gy):
        return math.hypot(gx - gxg, gy - gyg) * min_cost

    neighbors = [
        (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
        (-1, 0, 1.0), (1, 0, 1.0),
        (-1, 1, math.sqrt(2)), (0, 1, 1.0), (1, 1, math.sqrt(2)),
    ]

    open_heap = [(heuristic(sx, sy), 0.0, (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    visited = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == (gxg, gyg):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny) or not passable(nx, ny):
                continue
            tentative_g = current_g + step_cost * cost_grid[ny, nx]
            neighbor = (nx, ny)
            if tentative_g < g_score.get(neighbor, np.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_heap, (tentative_g + heuristic(nx, ny), tentative_g, neighbor))

    return None


def smooth_path_bspline(path_world, cfg: PipelineConfig):
    if path_world is None or len(path_world) < 3:
        return path_world
    pts = np.array(path_world, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]

    num_samples = cfg.path_smooth_samples
    if len(path_world) < 5:
        num_samples = len(path_world)
    k = min(3, len(path_world) - 1)

    try:
        tck, _ = splprep([x, y], s=cfg.path_smooth_factor * len(x), k=k)
        u_new = np.linspace(0, 1, num_samples)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new.tolist(), y_new.tolist()))
    except Exception:
        return path_world


def plan_path(cost_grid, cfg: PipelineConfig):
    start_gx, start_gy = world_to_grid(cfg.start_coord[0], cfg.start_coord[1], cfg)
    goal_gx, goal_gy = world_to_grid(cfg.goal_coord[0], cfg.goal_coord[1], cfg)

    start_cell = find_nearest_passable_cell(cost_grid, (start_gx, start_gy), cfg.max_snap_radius_cells)
    goal_cell = find_nearest_passable_cell(cost_grid, (goal_gx, goal_gy), cfg.max_snap_radius_cells)

    notes = []
    if start_cell is None:
        return None, [f"No passable cell within {cfg.max_snap_radius_cells} cells of START {cfg.start_coord}."]
    if goal_cell is None:
        return None, [f"No passable cell within {cfg.max_snap_radius_cells} cells of GOAL {cfg.goal_coord}."]
    if start_cell != (start_gx, start_gy):
        notes.append(f"START snapped to nearest passable cell {start_cell}.")
    if goal_cell != (goal_gx, goal_gy):
        notes.append(f"GOAL snapped to nearest passable cell {goal_cell}.")

    cell_path = a_star_search(cost_grid, start_cell, goal_cell)
    if cell_path is None:
        notes.append("A*: no path found between start and goal.")
        return None, notes

    world_path = [grid_to_world(gx, gy, cfg) for gx, gy in cell_path]
    if cfg.smooth_path:
        raw_count = len(world_path)
        world_path = smooth_path_bspline(world_path, cfg)
        notes.append(f"Path smoothed: {raw_count} raw waypoints -> {len(world_path)} smoothed waypoints.")

    return world_path, notes


def compute_distance_to_nearest_obstacle(path_world, occupancy_grid, cfg: PipelineConfig):
    if path_world is None or len(path_world) == 0:
        return np.array([])
    obstacle_mask = occupancy_grid == 1
    grid_height, grid_width = occupancy_grid.shape
    dist_meters = ndimage.distance_transform_edt(~obstacle_mask) * cfg.resolution

    pts = np.array(path_world, dtype=np.float64)
    distances = np.zeros(len(pts))
    for i in range(len(pts)):
        gx = max(0, min(int((pts[i, 0] - cfg.x_min) / cfg.resolution), grid_width - 1))
        gy = max(0, min(int((pts[i, 1] - cfg.y_min) / cfg.resolution), grid_height - 1))
        distances[i] = dist_meters[gy, gx]
    return distances


def compute_path_curvature_and_speed(path_world, obstacle_distances, cfg: PipelineConfig):
    """Menger-curvature speed limiting with obstacle-proximity de-rating and a
    forward/backward jerk-limited velocity profile. See dashboard docstring."""
    if path_world is None or len(path_world) < 3:
        return {
            "kappa": np.array([]), "v_max": np.array([]), "v_target": np.array([]),
            "v_exceeded": False, "max_kappa": 0.0, "min_v_max": float("inf"),
            "violations": [], "jerk_violations": [],
        }

    pts = np.array(path_world, dtype=np.float64)
    N = len(pts)
    kappa = np.zeros(N)
    v_max_curv = np.full(N, 50.0)

    for i in range(1, N - 1):
        A, B, C = pts[i - 1], pts[i], pts[i + 1]
        AB, BC = B - A, C - B
        len_AB, len_BC, len_CA = np.linalg.norm(AB), np.linalg.norm(BC), np.linalg.norm(A - C)
        cross = AB[0] * BC[1] - AB[1] * BC[0]
        denom = len_AB * len_BC * len_CA
        kappa[i] = (2.0 * abs(cross) / denom) if denom >= 1e-12 else 0.0

        a_lat = cfg.lateral_accel_max
        if obstacle_distances is not None and len(obstacle_distances) > i:
            d_obs = obstacle_distances[i]
            if d_obs < cfg.obstacle_proximity_threshold:
                proximity_factor = max(0.5, d_obs / cfg.obstacle_proximity_threshold)
                a_lat = cfg.lateral_accel_max * proximity_factor

        v_max_curv[i] = math.sqrt(a_lat / kappa[i]) if kappa[i] > 1e-12 else 50.0
        v_max_curv[i] = min(max(v_max_curv[i], 0.5), 50.0)

    if N >= 2:
        kappa[0], v_max_curv[0] = kappa[1], v_max_curv[1]
        kappa[-1], v_max_curv[-1] = kappa[-2], v_max_curv[-2]

    v_target = np.minimum(cfg.default_speed, v_max_curv)

    for i in range(1, N):
        ds = np.linalg.norm(pts[i] - pts[i - 1])
        v_target[i] = min(v_target[i], v_target[i - 1] + cfg.jerk_max * ds)
    for i in range(N - 2, -1, -1):
        ds = np.linalg.norm(pts[i + 1] - pts[i])
        v_target[i] = min(v_target[i], v_target[i + 1] + cfg.jerk_max * ds)

    violations = [
        (i, float(kappa[i]), float(v_target[i]))
        for i in range(N) if v_target[i] < cfg.default_speed - 0.01
    ]
    jerk_violations = []
    for i in range(1, N):
        ds = np.linalg.norm(pts[i] - pts[i - 1])
        if ds < 1e-9:
            continue
        actual_jerk = abs(v_target[i] - v_target[i - 1]) / ds
        if actual_jerk > cfg.jerk_max + 0.01:
            jerk_violations.append((i, float(actual_jerk)))

    return {
        "kappa": kappa, "v_max": v_max_curv, "v_target": v_target,
        "v_exceeded": bool(np.any(v_target < cfg.default_speed - 0.01)),
        "max_kappa": float(np.max(kappa)), "min_v_max": float(np.min(v_target)),
        "violations": violations, "jerk_violations": jerk_violations,
    }


def quantitative_evaluation(xyz, semantic_labels, semantic_grid, cfg: PipelineConfig):
    grid_height, grid_width = semantic_grid.shape
    gx = ((xyz[:, 0] - cfg.x_min) / cfg.resolution).astype(int)
    gy = ((xyz[:, 1] - cfg.y_min) / cfg.resolution).astype(int)
    in_bounds = (gx >= 0) & (gx < grid_width) & (gy >= 0) & (gy < grid_height)

    true_labels = semantic_labels[in_bounds]
    predicted_labels = semantic_grid[gy[in_bounds], gx[in_bounds]]

    def to_occ(labels):
        occ = np.full(len(labels), -1, dtype=np.int8)
        occ[np.isin(labels, list(OBSTACLE_CLASSES))] = 1
        occ[np.isin(labels, list(TRAVERSABLE_CLASSES))] = 0
        return occ

    def to_trav(labels):
        trav = np.full(len(labels), -1, dtype=np.int8)
        trav[np.isin(labels, list(TRAVERSABLE_CLASSES))] = 1
        trav[np.isin(labels, list(OBSTACLE_CLASSES))] = 0
        return trav

    def scores(y_true, y_pred):
        if len(y_true) == 0:
            return dict(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, iou=0.0)
        return dict(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            iou=jaccard_score(y_true, y_pred, zero_division=0),
        )

    gt_occ, pred_occ = to_occ(true_labels), to_occ(predicted_labels)
    m_occ = (gt_occ != -1) & (pred_occ != -1)
    occ_scores = scores(gt_occ[m_occ], pred_occ[m_occ])

    gt_trav, pred_trav = to_trav(true_labels), to_trav(predicted_labels)
    m_trav = (gt_trav != -1) & (pred_trav != -1)
    trav_scores = scores(gt_trav[m_trav], pred_trav[m_trav])

    return {"occupancy": occ_scores, "traversability": trav_scores}


# ============================================================
# FRAME ORCHESTRATION  (cached — the expensive part of the app)
# ============================================================

@st.cache_data(show_spinner=False)
def process_frame(bin_path: str, label_path: str, cfg: PipelineConfig) -> dict:
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    points, xyz, intensity = load_pointcloud(bin_path)
    timings["load_lidar_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    semantic_labels = load_semantic_labels(label_path, len(points))
    timings["load_labels_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    elevation_grid, semantic_grid, point_count_grid = create_semantic_grid(xyz, semantic_labels, cfg)
    timings["semantic_grid_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    occupancy_grid, occupancy_probability = probabilistic_occupancy_mapping(xyz, cfg)
    timings["occupancy_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    traversability_grid, slope_degrees = traversability_analysis(elevation_grid, semantic_grid, cfg)
    timings["traversability_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cost_grid = build_cost_map(traversability_grid, slope_degrees, semantic_grid, cfg)
    timings["costmap_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    path_world, path_notes = plan_path(cost_grid, cfg)
    timings["path_planning_ms"] = (time.perf_counter() - t0) * 1000

    kinodynamic_result = None
    obstacle_dists = None
    t0 = time.perf_counter()
    if path_world is not None and len(path_world) >= 3:
        obstacle_dists = compute_distance_to_nearest_obstacle(path_world, occupancy_grid, cfg)
        kinodynamic_result = compute_path_curvature_and_speed(path_world, obstacle_dists, cfg)
    timings["kinodynamic_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    evaluation = quantitative_evaluation(xyz, semantic_labels, semantic_grid, cfg)
    timings["evaluation_ms"] = (time.perf_counter() - t0) * 1000

    path_length_m = 0.0
    if path_world is not None and len(path_world) >= 2:
        p = np.array(path_world)
        path_length_m = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    passable_fraction = float(np.isfinite(cost_grid).mean())
    unknown_fraction = float(np.mean(occupancy_grid == -1))

    return {
        "n_points": len(points),
        "xyz": xyz,
        "semantic_labels": semantic_labels,
        "elevation_grid": elevation_grid,
        "semantic_grid": semantic_grid,
        "point_count_grid": point_count_grid,
        "occupancy_grid": occupancy_grid,
        "traversability_grid": traversability_grid,
        "slope_degrees": slope_degrees,
        "cost_grid": cost_grid,
        "path_world": path_world,
        "path_notes": path_notes,
        "path_length_m": path_length_m,
        "kinodynamic": kinodynamic_result,
        "obstacle_distances": obstacle_dists,
        "evaluation": evaluation,
        "timings_ms": timings,
        "total_ms": sum(timings.values()),
        "passable_fraction": passable_fraction,
        "unknown_fraction": unknown_fraction,
    }


# ============================================================
# DATASET DISCOVERY
# ============================================================

FRAME_ID_RE = re.compile(r"(\d+)\.bin$")


def discover_sequences(root: str) -> Dict[str, List[str]]:
    """Scan <root>/sequences/<seq>/velodyne/*.bin against .../labels/*.label
    and return {sequence_id: [sorted frame_ids with both files present]}."""
    result: Dict[str, List[str]] = {}
    root_path = Path(root)
    seq_root = root_path / "sequences"
    if not seq_root.is_dir():
        return result

    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        velodyne_dir = seq_dir / "velodyne"
        labels_dir = seq_dir / "labels"
        if not velodyne_dir.is_dir() or not labels_dir.is_dir():
            continue
        bin_stems = {p.stem for p in velodyne_dir.glob("*.bin")}
        label_stems = {p.stem for p in labels_dir.glob("*.label")}
        frame_ids = sorted(bin_stems & label_stems)
        if frame_ids:
            result[seq_dir.name] = frame_ids
    return result


def frame_paths(root: str, seq: str, frame_id: str) -> Tuple[str, str]:
    seq_dir = Path(root) / "sequences" / seq
    return str(seq_dir / "velodyne" / f"{frame_id}.bin"), str(seq_dir / "labels" / f"{frame_id}.label")


def save_uploaded_pairs(bin_files, label_files) -> Dict[str, Tuple[str, str]]:
    """Write uploaded .bin/.label pairs (matched by filename stem) to a temp
    dir and return {frame_id: (bin_path, label_path)}."""
    tmp_dir = st.session_state.setdefault("_upload_tmp_dir", tempfile.mkdtemp(prefix="lidar_upload_"))
    bins = {Path(f.name).stem: f for f in (bin_files or [])}
    labels = {Path(f.name).stem: f for f in (label_files or [])}
    pairs = {}
    for stem in sorted(set(bins) & set(labels)):
        bin_path = os.path.join(tmp_dir, f"{stem}.bin")
        label_path = os.path.join(tmp_dir, f"{stem}.label")
        with open(bin_path, "wb") as f:
            f.write(bins[stem].getbuffer())
        with open(label_path, "wb") as f:
            f.write(labels[stem].getbuffer())
        pairs[stem] = (bin_path, label_path)
    return pairs


# ============================================================
# PLOTLY FIGURE BUILDERS
# ============================================================

def _extent(cfg: PipelineConfig):
    x = np.arange(cfg.x_min, cfg.x_max, cfg.resolution)
    y = np.arange(cfg.y_min, cfg.y_max, cfg.resolution)
    return x, y


def figure_elevation(elevation_grid, cfg: PipelineConfig):
    x, y = _extent(cfg)
    fig = go.Figure(go.Heatmap(z=elevation_grid, x=x, y=y, colorscale="Viridis", colorbar=dict(title="m")))
    fig.update_layout(title="Elevation", xaxis_title="X (m)", yaxis_title="Y (m)",
                       yaxis=dict(scaleanchor="x"), height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def figure_semantic(semantic_grid, cfg: PipelineConfig):
    present = sorted(int(v) for v in np.unique(semantic_grid))
    idx_map = {lab: i for i, lab in enumerate(present)}
    z = np.vectorize(idx_map.get)(semantic_grid)
    n = max(len(present), 1)
    colorscale = []
    for i, lab in enumerate(present):
        r, g, b = CLASS_COLORS.get(lab, (128, 128, 128))
        c = f"rgb({r},{g},{b})"
        colorscale.append([i / n, c])
        colorscale.append([(i + 1) / n, c])

    x, y = _extent(cfg)
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, colorscale=colorscale, zmin=0, zmax=n,
        colorbar=dict(
            tickvals=[i + 0.5 for i in range(n)],
            ticktext=[CLASS_NAMES.get(lab, f"class {lab}") for lab in present],
            len=0.9,
        ),
        hovertemplate="x=%{x:.1f} y=%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(title="Semantic classes", xaxis_title="X (m)", yaxis_title="Y (m)",
                       yaxis=dict(scaleanchor="x"), height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def figure_occupancy(occupancy_grid, cfg: PipelineConfig):
    z = np.where(occupancy_grid == -1, np.nan, occupancy_grid).astype(float)
    x, y = _extent(cfg)
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, zmin=0, zmax=1,
        colorscale=[[0, "#4daf4a"], [1, "#e41a1c"]],
        colorbar=dict(tickvals=[0, 1], ticktext=["Free", "Occupied"]),
    ))
    fig.update_layout(title="Occupancy (probabilistic raycast)", xaxis_title="X (m)", yaxis_title="Y (m)",
                       yaxis=dict(scaleanchor="x"), height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def figure_slope(slope_degrees, cfg: PipelineConfig):
    z = np.clip(slope_degrees, 0, cfg.slope_display_max)
    x, y = _extent(cfg)
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale="Turbo", zmin=0, zmax=cfg.slope_display_max,
                                colorbar=dict(title="deg")))
    fig.update_layout(title=f"Terrain slope (capped at {cfg.slope_display_max:.0f}°)",
                       xaxis_title="X (m)", yaxis_title="Y (m)",
                       yaxis=dict(scaleanchor="x"), height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def figure_traversability(traversability_grid, cfg: PipelineConfig, path_world=None):
    z = np.where(traversability_grid == 2, np.nan, traversability_grid).astype(float)
    x, y = _extent(cfg)
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, zmin=0, zmax=1,
        colorscale=[[0, "#e41a1c"], [1, "#4daf4a"]],
        colorbar=dict(tickvals=[0, 1], ticktext=["Blocked", "Traversable"]),
    ))
    if path_world is not None and len(path_world) > 0:
        px = [p[0] for p in path_world]
        py = [p[1] for p in path_world]
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color="blue", width=3), name="Path"))
        fig.add_trace(go.Scatter(x=[px[0]], y=[py[0]], mode="markers",
                                  marker=dict(color="cyan", size=14, symbol="star", line=dict(color="black", width=1)),
                                  name="Start"))
        fig.add_trace(go.Scatter(x=[px[-1]], y=[py[-1]], mode="markers",
                                  marker=dict(color="magenta", size=14, symbol="star", line=dict(color="black", width=1)),
                                  name="Goal"))
    fig.update_layout(title="Traversability + planned path", xaxis_title="X (m)", yaxis_title="Y (m)",
                       yaxis=dict(scaleanchor="x"), height=460, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def figure_pointcloud_3d(xyz, semantic_labels, cfg: PipelineConfig, path_world=None, v_target=None,
                          elevation_grid=None):
    n = len(xyz)
    if n > cfg.max_3d_points:
        idx = np.random.default_rng(0).choice(n, size=cfg.max_3d_points, replace=False)
    else:
        idx = np.arange(n)

    colors = [rgb_str(l) for l in semantic_labels[idx]]
    fig = go.Figure(go.Scatter3d(
        x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2],
        mode="markers", marker=dict(size=1.4, color=colors, opacity=0.75),
        name="Point cloud", hoverinfo="skip",
    ))

    if path_world is not None and len(path_world) >= 2 and elevation_grid is not None:
        gh, gw = elevation_grid.shape
        px, py, pz = [], [], []
        for x_, y_ in path_world:
            gx = max(0, min(int((x_ - cfg.x_min) / cfg.resolution), gw - 1))
            gy = max(0, min(int((y_ - cfg.y_min) / cfg.resolution), gh - 1))
            z_ = elevation_grid[gy, gx]
            z_ = 0.0 if np.isnan(z_) else z_
            px.append(x_); py.append(y_); pz.append(z_ + 0.2)

        line_kwargs = dict(width=7)
        if v_target is not None and len(v_target) == len(px):
            line_kwargs.update(color=list(v_target), colorscale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                                cmin=0, cmax=cfg.default_speed,
                                colorbar=dict(title="m/s", x=1.08))
        else:
            line_kwargs.update(color="blue")

        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=line_kwargs, name="Path"))

    fig.update_layout(
        title="3D point cloud" + (" + velocity-colored path" if v_target is not None else ""),
        scene=dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
        height=600, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def figure_speed_profile(path_world, kinodynamic, cfg: PipelineConfig):
    pts = np.array(path_world, dtype=np.float64)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=arc, y=kinodynamic["v_target"], mode="lines+markers", name="Target speed",
                              marker=dict(size=4)))
    fig.add_hline(y=cfg.default_speed, line_dash="dash", line_color="gray",
                  annotation_text="cruise speed", annotation_position="top left")
    fig.update_layout(title="Kinodynamic speed profile along path", xaxis_title="Distance along path (m)",
                       yaxis_title="Target speed (m/s)", height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ============================================================
# STREAMLIT APP
# ============================================================

def _sidebar_config(existing: PipelineConfig) -> PipelineConfig:
    st.sidebar.header("⚙️ Settings")

    with st.sidebar.expander("Grid & bounds", expanded=False):
        resolution = st.number_input("Grid resolution (m)", 0.1, 2.0, existing.resolution, 0.1)
        x_min = st.number_input("X min", -200.0, 0.0, existing.x_min, 5.0)
        x_max = st.number_input("X max", 0.0, 200.0, existing.x_max, 5.0)
        y_min = st.number_input("Y min", -200.0, 0.0, existing.y_min, 5.0)
        y_max = st.number_input("Y max", 0.0, 200.0, existing.y_max, 5.0)

    with st.sidebar.expander("Traversability", expanded=False):
        max_slope = st.slider("Max slope (deg)", 5.0, 40.0, existing.max_slope, 1.0)
        slope_display_max = st.slider("Slope display cap (deg)", 10.0, 90.0, existing.slope_display_max, 5.0)

    with st.sidebar.expander("Occupancy (raycast)", expanded=False):
        occ_threshold = st.slider("Occupied probability threshold", 0.5, 0.99, existing.occ_prob_threshold, 0.01)
        free_threshold = st.slider("Free probability threshold", 0.01, 0.5, existing.free_prob_threshold, 0.01)
        raycast_stride = st.number_input("Raycast stride (speed vs. detail)", 1, 10, existing.raycast_stride, 1)

    with st.sidebar.expander("Costmap inflation", expanded=False):
        enable_inflation = st.checkbox("Enable obstacle inflation", existing.enable_inflation)
        inflation_radius = st.slider("Inflation radius (m)", 0.0, 10.0, existing.inflation_radius, 0.5)
        inflation_decay_rate = st.slider("Inflation decay rate", 0.5, 10.0, existing.inflation_decay_rate, 0.5)
        inflation_max_cost = st.slider("Inflation max cost", 0.0, 200.0, existing.inflation_max_cost, 5.0)
        unknown_policy = st.selectbox("Unknown-cell policy", ["penalize", "block", "optimistic"],
                                       index=["penalize", "block", "optimistic"].index(existing.unknown_policy))

    with st.sidebar.expander("Start / goal & path", expanded=True):
        c1, c2 = st.columns(2)
        start_x = c1.number_input("Start X", value=float(existing.start_coord[0]))
        start_y = c2.number_input("Start Y", value=float(existing.start_coord[1]))
        c1, c2 = st.columns(2)
        goal_x = c1.number_input("Goal X", value=float(existing.goal_coord[0]))
        goal_y = c2.number_input("Goal Y", value=float(existing.goal_coord[1]))
        smooth_path = st.checkbox("Smooth path (B-spline)", existing.smooth_path)

    with st.sidebar.expander("Vehicle kinodynamics", expanded=False):
        default_speed = st.slider("Cruise speed (m/s)", 0.5, 20.0, existing.default_speed, 0.5)
        lateral_accel_max = st.slider("Max lateral accel (m/s²)", 0.5, 10.0, existing.lateral_accel_max, 0.5)
        jerk_max = st.slider("Max jerk (m/s² per m)", 0.5, 20.0, existing.jerk_max, 0.5)
        obstacle_proximity_threshold = st.slider("Obstacle proximity de-rate radius (m)", 0.5, 10.0,
                                                   existing.obstacle_proximity_threshold, 0.5)

    with st.sidebar.expander("3D viewer", expanded=False):
        max_3d_points = st.slider("Max points rendered in 3D", 2000, 100000, existing.max_3d_points, 2000)

    return replace(
        existing,
        resolution=resolution, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        max_slope=max_slope, slope_display_max=slope_display_max,
        occ_prob_threshold=occ_threshold, free_prob_threshold=free_threshold, raycast_stride=int(raycast_stride),
        enable_inflation=enable_inflation, inflation_radius=inflation_radius,
        inflation_decay_rate=inflation_decay_rate, inflation_max_cost=inflation_max_cost,
        unknown_policy=unknown_policy,
        start_coord=(start_x, start_y), goal_coord=(goal_x, goal_y), smooth_path=smooth_path,
        default_speed=default_speed, lateral_accel_max=lateral_accel_max, jerk_max=jerk_max,
        obstacle_proximity_threshold=obstacle_proximity_threshold,
        max_3d_points=int(max_3d_points),
    )


def _render_frame_dashboard(result: dict, cfg: PipelineConfig, frame_label: str):
    st.subheader(f"Frame: {frame_label}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LiDAR points", f"{result['n_points']:,}")
    c2.metric("Passable cells", f"{result['passable_fraction'] * 100:.1f}%")
    c3.metric("Occupancy unknown", f"{result['unknown_fraction'] * 100:.1f}%")
    c4.metric("Path length", f"{result['path_length_m']:.1f} m" if result["path_world"] else "no path")
    c5.metric("Processing time", f"{result['total_ms']:.0f} ms")

    if result["path_world"] is None:
        for note in result["path_notes"]:
            st.warning(note)
    else:
        for note in result["path_notes"]:
            st.caption(note)

    kino = result["kinodynamic"]
    if kino is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Min safe speed", f"{kino['min_v_max']:.2f} m/s")
        k2.metric("Max curvature", f"{kino['max_kappa']:.4f} 1/m")
        k3.metric("Curvature-limited pts", f"{len(kino['violations'])}/{len(kino['kappa'])}")
        k4.metric("Jerk violations", f"{len(kino['jerk_violations'])}")

    ev = result["evaluation"]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Occupancy F1", f"{ev['occupancy']['f1']:.3f}")
    e2.metric("Occupancy IoU", f"{ev['occupancy']['iou']:.3f}")
    e3.metric("Traversability F1", f"{ev['traversability']['f1']:.3f}")
    e4.metric("Traversability IoU", f"{ev['traversability']['iou']:.3f}")

    tabs = st.tabs(["🧭 Path & traversability", "🗺️ Layers", "🧊 3D view", "🚗 Speed profile", "⏱️ Timing"])

    with tabs[0]:
        st.plotly_chart(figure_traversability(result["traversability_grid"], cfg, result["path_world"]),
                         use_container_width=True)

    with tabs[1]:
        lc1, lc2 = st.columns(2)
        lc1.plotly_chart(figure_elevation(result["elevation_grid"], cfg), use_container_width=True)
        lc2.plotly_chart(figure_semantic(result["semantic_grid"], cfg), use_container_width=True)
        lc3, lc4 = st.columns(2)
        lc3.plotly_chart(figure_occupancy(result["occupancy_grid"], cfg), use_container_width=True)
        lc4.plotly_chart(figure_slope(result["slope_degrees"], cfg), use_container_width=True)

    with tabs[2]:
        v_target = kino["v_target"] if kino is not None else None
        st.plotly_chart(
            figure_pointcloud_3d(result["xyz"], result["semantic_labels"], cfg,
                                  path_world=result["path_world"], v_target=v_target,
                                  elevation_grid=result["elevation_grid"]),
            use_container_width=True,
        )

    with tabs[3]:
        if kino is not None and result["path_world"] is not None:
            st.plotly_chart(figure_speed_profile(result["path_world"], kino, cfg), use_container_width=True)
        else:
            st.info("No path was found for this frame, so no speed profile is available.")

    with tabs[4]:
        timing_items = list(result["timings_ms"].items())
        fig = go.Figure(go.Bar(x=[k for k, _ in timing_items], y=[v for _, v in timing_items]))
        fig.update_layout(title="Per-stage processing time", yaxis_title="ms", height=350,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


def _batch_process(bin_label_pairs: List[Tuple[str, str, str]], cfg: PipelineConfig):
    """bin_label_pairs: list of (frame_id, bin_path, label_path)."""
    rows = []
    progress = st.progress(0.0, text="Starting batch run...")
    for i, (frame_id, bin_path, label_path) in enumerate(bin_label_pairs):
        progress.progress((i) / max(len(bin_label_pairs), 1), text=f"Processing frame {frame_id}...")
        try:
            r = process_frame(bin_path, label_path, cfg)
            rows.append({
                "frame": frame_id,
                "points": r["n_points"],
                "path_found": r["path_world"] is not None,
                "path_length_m": r["path_length_m"],
                "min_safe_speed_mps": r["kinodynamic"]["min_v_max"] if r["kinodynamic"] else None,
                "occupancy_f1": r["evaluation"]["occupancy"]["f1"],
                "traversability_f1": r["evaluation"]["traversability"]["f1"],
                "passable_fraction": r["passable_fraction"],
                "total_ms": r["total_ms"],
            })
        except Exception as e:
            rows.append({"frame": frame_id, "error": str(e)})
    progress.progress(1.0, text="Batch run complete.")
    return rows


def main():
    st.set_page_config(page_title="LiDAR 2.5D Mapping Dashboard", layout="wide")
    st.title("🛰️ LiDAR 2.5D Semantic Mapping Dashboard")
    st.caption(
        "Browse a whole SemanticKITTI-style sequence frame by frame, or batch-process a "
        "range of frames and see how the map, path, and speed profile evolve."
    )

    if "cfg" not in st.session_state:
        st.session_state.cfg = PipelineConfig()
    cfg = _sidebar_config(st.session_state.cfg)
    st.session_state.cfg = cfg

    st.sidebar.divider()
    source_mode = st.sidebar.radio("Data source", ["Local folder", "Upload files"])

    frame_options: Dict[str, Tuple[str, str]] = {}  # frame_id -> (bin_path, label_path)

    if source_mode == "Local folder":
        root = st.sidebar.text_input(
            "Dataset root folder",
            value=st.session_state.get("root", ""),
            help="Folder containing sequences/<seq>/velodyne/*.bin and .../labels/*.label",
        )
        st.session_state["root"] = root
        if root:
            sequences = discover_sequences(root)
            if not sequences:
                st.sidebar.error("No sequences found. Check the folder structure.")
            else:
                seq = st.sidebar.selectbox("Sequence", list(sequences.keys()))
                frame_ids = sequences[seq]
                for fid in frame_ids:
                    bpath, lpath = frame_paths(root, seq, fid)
                    frame_options[fid] = (bpath, lpath)
        else:
            st.sidebar.info("Enter a dataset folder to begin, or switch to Upload files.")
    else:
        st.sidebar.caption("Upload matching .bin and .label files (matched by filename, e.g. 000009.bin / 000009.label).")
        bin_files = st.sidebar.file_uploader(".bin files", type="bin", accept_multiple_files=True)
        label_files = st.sidebar.file_uploader(".label files", type="label", accept_multiple_files=True)
        if bin_files and label_files:
            pairs = save_uploaded_pairs(bin_files, label_files)
            if not pairs:
                st.sidebar.error("No matching .bin/.label filename pairs found.")
            frame_options = pairs

    if not frame_options:
        st.info("👈 Choose a dataset folder or upload files in the sidebar to get started.")
        return

    frame_ids = list(frame_options.keys())

    mode = st.radio("Mode", ["Single frame", "Batch process sequence"], horizontal=True)

    if mode == "Single frame":
        idx_key = "frame_idx"
        if idx_key not in st.session_state or st.session_state[idx_key] >= len(frame_ids):
            st.session_state[idx_key] = 0

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        if nav1.button("⬅ Prev", use_container_width=True) and st.session_state[idx_key] > 0:
            st.session_state[idx_key] -= 1
        selected = nav2.select_slider("Frame", options=list(range(len(frame_ids))),
                                       value=st.session_state[idx_key],
                                       format_func=lambda i: frame_ids[i])
        st.session_state[idx_key] = selected
        if nav3.button("Next ➡", use_container_width=True) and st.session_state[idx_key] < len(frame_ids) - 1:
            st.session_state[idx_key] += 1

        frame_id = frame_ids[st.session_state[idx_key]]
        bin_path, label_path = frame_options[frame_id]

        with st.spinner(f"Processing frame {frame_id}..."):
            try:
                result = process_frame(bin_path, label_path, cfg)
            except Exception as e:
                st.error(f"Failed to process frame {frame_id}: {e}")
                return

        _render_frame_dashboard(result, cfg, frame_id)

    else:
        st.write(f"**{len(frame_ids)} frames available.** Choose a range to batch-process.")
        c1, c2 = st.columns(2)
        start_i = c1.number_input("From frame index", 0, len(frame_ids) - 1, 0)
        end_i = c2.number_input("To frame index", 0, len(frame_ids) - 1, min(9, len(frame_ids) - 1))
        if end_i < start_i:
            st.error("End index must be >= start index.")
            return

        if st.button("▶ Run batch", type="primary"):
            subset = [(fid, *frame_options[fid]) for fid in frame_ids[start_i:end_i + 1]]
            rows = _batch_process(subset, cfg)
            st.session_state["batch_rows"] = rows

        rows = st.session_state.get("batch_rows")
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download results as CSV", csv_bytes, file_name="batch_results.csv",
                                mime="text/csv")

            if "occupancy_f1" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["frame"], y=df["occupancy_f1"], mode="lines+markers",
                                          name="Occupancy F1"))
                fig.add_trace(go.Scatter(x=df["frame"], y=df["traversability_f1"], mode="lines+markers",
                                          name="Traversability F1"))
                fig.update_layout(title="Quality across frames", yaxis_title="F1 score", height=350,
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if "total_ms" in df.columns:
                fig2 = go.Figure(go.Bar(x=df["frame"], y=df["total_ms"]))
                fig2.update_layout(title="Processing time per frame", yaxis_title="ms", height=300,
                                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()