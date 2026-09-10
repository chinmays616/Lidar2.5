"""
occupancy.py
============
Probabilistic (log-odds raycast) occupancy grid, and obstacle inflation.

Addresses:
  #9  No vehicle footprint or obstacle inflation
      -> inflate_obstacles() grows every occupied cell outward by
         (robot_radius_m + inflation_extra_margin_m) using a disk
         structuring element, so the planner can safely treat "any
         inflated cell" as impassable and use the platform as a point
         mass. Both the raycast occupancy grid AND the semantic
         obstacle mask are inflated before costing (see planning.py).
"""

import math
import numpy as np
from scipy import ndimage

from config import Config


def probabilistic_occupancy_mapping(xyz: np.ndarray, cfg: Config,
                                      x_min: float, x_max: float, y_min: float, y_max: float):
    """
    Vectorized log-odds raycast occupancy grid.
    Returns (occupancy_grid[int8: 1 occ / 0 free / -1 unknown], probability_grid[float]).
    """
    grid_width = int((x_max - x_min) / cfg.resolution)
    grid_height = int((y_max - y_min) / cfg.resolution)

    l_occ = math.log(cfg.p_occ / (1.0 - cfg.p_occ))
    l_free = math.log(cfg.p_free / (1.0 - cfg.p_free))

    log_odds_flat = np.zeros(grid_height * grid_width, dtype=np.float64)

    ox, oy = cfg.sensor_origin
    origin_gx = min(max(int((ox - x_min) / cfg.resolution), 0), grid_width - 1)
    origin_gy = min(max(int((oy - y_min) / cfg.resolution), 0), grid_height - 1)

    points = xyz[::cfg.raycast_stride] if cfg.raycast_stride > 1 else xyz

    gx_all = ((points[:, 0] - x_min) / cfg.resolution).astype(np.int64)
    gy_all = ((points[:, 1] - y_min) / cfg.resolution).astype(np.int64)

    in_bounds = (gx_all >= 0) & (gx_all < grid_width) & (gy_all >= 0) & (gy_all < grid_height)
    gx_all, gy_all = gx_all[in_bounds], gy_all[in_bounds]

    n_points = len(gx_all)
    batch_size = cfg.raycast_batch_size

    for start in range(0, n_points, batch_size):
        gx_batch = gx_all[start:start + batch_size]
        gy_batch = gy_all[start:start + batch_size]

        dx = gx_batch - origin_gx
        dy = gy_batch - origin_gy

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


def inflate_obstacles(obstacle_mask: np.ndarray, resolution: float, cfg: Config) -> np.ndarray:
    """
    #9: Grow an obstacle boolean mask outward by the platform's
    footprint radius so downstream planning can treat the robot as a
    point and any inflated obstacle cell as fully impassable.
    """
    total_radius_m = cfg.robot_radius_m + cfg.inflation_extra_margin_m
    radius_cells = max(1, int(round(total_radius_m / resolution)))

    yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    disk = (xx ** 2 + yy ** 2) <= radius_cells ** 2

    return ndimage.binary_dilation(obstacle_mask, structure=disk)