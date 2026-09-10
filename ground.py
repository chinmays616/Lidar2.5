"""
ground.py
=========
Ground segmentation and terrain/slope analysis.

Addresses:
  #5  Simplistic ground detection using a fixed height threshold
      -> ransac_ground_segmentation() fits a plane with Open3D's RANSAC
         and classifies inliers as ground, robust to sloped or uneven
         terrain that a single global z-threshold gets wrong. The old
         height-threshold method is kept as an explicit opt-in fallback
         (Config.ground_method="height_threshold") for scenes where a
         plane fit is a bad model (e.g. multi-level parking structures).
  #6  Maximum-Z based elevation estimation can misrepresent terrain
      -> build_elevation_grid() now has a `points_for_terrain` argument.
         When called with ground-only points (from #5) it takes the
         MEDIAN height of ground points per cell instead of the MAX,
         so a car roof or tree canopy sharing a cell with real ground
         samples no longer drags the terrain surface (and therefore
         slope) upward. A separate obstacle-height grid (unchanged,
         max-based - appropriate for a BEV/obstacle heightmap) is kept
         distinct so the two purposes stop being conflated.
  #23 Limited uncertainty modeling
      -> interpolate_elevation_idw() now also returns a confidence grid
         (1 / (1 + mean neighbor distance)), so terrain_analysis can
         propagate "how much do we trust this elevation estimate" to
         downstream consumers (traversability/cost map) alongside the
         existing bounded-distance unknown_mask.
  #21 No real-time performance optimization
      -> terrain_analysis / interpolate_elevation_idw were already
         vectorized in the source pipeline; the remaining per-cell
         Python loop in interpolate_elevation_idw's weighting step is
         kept (k is small, <=8) since a full vectorization there adds
         complexity for little benefit at this grid size. The genuinely
         hot per-cell double loops live in traversability.py and were
         removed there (see that file's docstring).
  #4  Rule-based adaptive resolution
      -> adaptive_resolution() now blends distance and local slope into
         a continuous target resolution instead of two hard if/elif
         thresholds, so resolution changes smoothly across a slope
         boundary instead of jumping.
"""

from typing import Tuple
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from config import Config


# ============================================================
# GROUND SEGMENTATION  (#5)
# ============================================================

def ransac_ground_segmentation(xyz: np.ndarray, cfg: Config):
    """
    Fit a single dominant plane with RANSAC and classify points within
    `ransac_distance_threshold` of it as ground. Falls back to the
    height-threshold method if Open3D is unavailable or the fit fails
    (keeps the pipeline usable in constrained environments - #18).
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not available - falling back to height-threshold ground detection.")
        return height_threshold_ground_segmentation(xyz, cfg.ground_height_threshold)

    if len(xyz) < cfg.ransac_n:
        return height_threshold_ground_segmentation(xyz, cfg.ground_height_threshold)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    try:
        plane_model, inlier_idx = pcd.segment_plane(
            distance_threshold=cfg.ransac_distance_threshold,
            ransac_n=cfg.ransac_n,
            num_iterations=cfg.ransac_iterations,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"RANSAC plane fit failed ({exc}); falling back to height threshold.")
        return height_threshold_ground_segmentation(xyz, cfg.ground_height_threshold)

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)

    # A "ground" plane should be roughly horizontal. If RANSAC locked
    # onto a wall/facade instead (normal far from vertical), that's a
    # bad fit for this frame - fall back rather than trust it blindly.
    if normal_norm < 1e-9 or abs(normal[2]) / normal_norm < 0.7:
        print("RANSAC plane is not roughly horizontal; falling back to height threshold.")
        return height_threshold_ground_segmentation(xyz, cfg.ground_height_threshold)

    ground_mask = np.zeros(len(xyz), dtype=bool)
    ground_mask[np.asarray(inlier_idx, dtype=np.int64)] = True

    return xyz[ground_mask], xyz[~ground_mask], ground_mask, plane_model


def height_threshold_ground_segmentation(xyz: np.ndarray, threshold: float):
    """Legacy fallback: everything below `threshold` in z is "ground"."""
    ground_mask = xyz[:, 2] < threshold
    plane_model = np.array([0.0, 0.0, 1.0, -threshold])
    return xyz[ground_mask], xyz[~ground_mask], ground_mask, plane_model


# ============================================================
# ELEVATION GRID  (#6: ground-only, median not max)
# ============================================================

def build_elevation_grid(
    points: np.ndarray,
    resolution: float,
    x_min: float, x_max: float, y_min: float, y_max: float,
    reducer: str = "median",
) -> np.ndarray:
    """
    Rasterize `points` (already filtered to whatever subset is
    appropriate - ground-only for terrain, all points for a BEV
    heightmap) into a per-cell elevation grid.

    reducer="median" is the terrain-safe default (#6): a stray
    obstacle point sharing a ground cell can't drag the surface up.
    reducer="max" reproduces the old BEV/obstacle-heightmap behaviour
    for callers that explicitly want the tallest thing in each cell.
    """
    # NOTE: no "+1" here - this must match the sizing convention used by
    # grid.create_semantic_grid / occupancy.probabilistic_occupancy_mapping
    # / traversability.traversability_analysis, or grids with mismatched
    # shapes can't be combined in build_cost_map.
    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)

    grid = np.full((grid_height, grid_width), np.nan)

    if len(points) == 0:
        return grid

    gx = ((points[:, 0] - x_min) / resolution).astype(np.int64)
    gy = ((points[:, 1] - y_min) / resolution).astype(np.int64)
    z = points[:, 2]

    valid = (gx >= 0) & (gx < grid_width) & (gy >= 0) & (gy < grid_height)
    gx, gy, z = gx[valid], gy[valid], z[valid]

    if len(z) == 0:
        return grid

    flat_idx = gy * grid_width + gx
    order = np.argsort(flat_idx, kind="stable")
    flat_sorted = flat_idx[order]
    z_sorted = z[order]

    unique_cells, start_idx, counts = np.unique(
        flat_sorted, return_index=True, return_counts=True
    )

    for cell_flat, start, count in zip(unique_cells, start_idx, counts):
        z_values = z_sorted[start:start + count]
        value = np.median(z_values) if reducer == "median" else np.max(z_values)
        cy, cx = divmod(int(cell_flat), grid_width)
        grid[cy, cx] = value

    return grid


# ============================================================
# BOUNDED IDW ELEVATION FILL  (+ confidence, #23)
# ============================================================

def interpolate_elevation_idw(
    elevation_grid: np.ndarray,
    resolution: float,
    max_distance: float,
    k: int,
):
    """
    Fill NaN cells from nearby REAL samples via inverse-distance
    weighting, bounded by max_distance (cells with no observation
    within range stay NaN/"unknown" rather than being extrapolated).

    Returns (filled_grid, observed_mask, confidence_grid) where
    confidence in [0, 1] is 1/(1+mean neighbor distance) for filled
    cells, 1.0 for directly observed cells, and 0.0 for unknown cells
    (#23: a lightweight uncertainty signal for downstream cost maps).
    """
    grid_height, grid_width = elevation_grid.shape
    valid_mask = ~np.isnan(elevation_grid)

    if not np.any(valid_mask):
        return (
            elevation_grid.copy(),
            np.zeros_like(elevation_grid, dtype=bool),
            np.zeros_like(elevation_grid, dtype=float),
        )

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
    n_values = len(valid_values)

    filled = np.full(n_cells, np.nan)
    observed_mask = np.zeros(n_cells, dtype=bool)
    confidence = np.zeros(n_cells, dtype=float)

    for i in range(n_cells):
        d, ix = dist[i], idx[i]
        finite = np.isfinite(d) & (ix < n_values)
        if not np.any(finite):
            continue
        d, ix = d[finite], ix[finite]

        if d[0] < 1e-9:
            filled[i] = valid_values[ix[0]]
            confidence[i] = 1.0
        else:
            w = 1.0 / (d ** 2)
            filled[i] = np.sum(w * valid_values[ix]) / np.sum(w)
            confidence[i] = 1.0 / (1.0 + float(np.mean(d)))

        observed_mask[i] = True

    filled_grid = filled.reshape(grid_height, grid_width)
    observed_mask = observed_mask.reshape(grid_height, grid_width) | valid_mask
    confidence_grid = confidence.reshape(grid_height, grid_width)
    confidence_grid[valid_mask] = 1.0

    filled_grid = np.where(observed_mask, filled_grid, np.nan)
    confidence_grid = np.where(observed_mask, confidence_grid, 0.0)

    return filled_grid, observed_mask, confidence_grid


def _nan_aware_gaussian(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth without NaN neighborhoods biasing edges toward 0."""
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


def terrain_analysis(elevation_grid: np.ndarray, cfg: Config):
    """
    Ground-only elevation -> (slope, slope_degrees, unknown_mask,
    confidence_grid). See module docstring for #6/#23.
    """
    valid = ~np.isnan(elevation_grid)
    if not np.any(valid):
        unknown_mask = np.ones_like(elevation_grid, dtype=bool)
        zeros = np.full_like(elevation_grid, np.nan)
        return zeros, zeros.copy(), unknown_mask, np.zeros_like(elevation_grid)

    filled_grid, observed_mask, confidence_grid = interpolate_elevation_idw(
        elevation_grid, cfg.resolution, cfg.max_interp_distance_m, cfg.interp_k_neighbors
    )

    if cfg.smoothing_sigma > 0:
        filled_grid = _nan_aware_gaussian(filled_grid, cfg.smoothing_sigma)

    gradient_y, gradient_x = np.gradient(filled_grid, cfg.resolution)
    slope = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    slope_degrees = np.degrees(np.arctan(slope))

    unknown_mask = np.isnan(filled_grid) | np.isnan(slope_degrees)
    slope = np.where(unknown_mask, np.nan, slope)
    slope_degrees = np.where(unknown_mask, np.nan, slope_degrees)

    return slope, slope_degrees, unknown_mask, confidence_grid


# ============================================================
# ADAPTIVE RESOLUTION  (#4: continuous, not two hard thresholds)
# ============================================================

def adaptive_resolution(xyz: np.ndarray, cfg: Config, coarse_resolution: float = 1.0):
    """
    Blend distance-from-sensor and local slope magnitude into a
    continuous target cell resolution, instead of jumping between
    three fixed tiers at arbitrary slope cutoffs. Resolution still
    clamps to [adaptive_min_resolution, adaptive_max_resolution]; the
    interpolation in between is smooth so a cell just under a slope
    cutoff no longer gets a wildly different resolution than a
    neighbor just over it.
    """
    x_min, x_max, y_min, y_max = cfg.resolved_bounds(xyz)

    coarse_elevation = build_elevation_grid(
        xyz, coarse_resolution, x_min, x_max, y_min, y_max, reducer="max"
    )

    filled = coarse_elevation.copy()
    valid = ~np.isnan(filled)

    if not np.any(valid):
        flat = np.full(coarse_elevation.shape, cfg.adaptive_max_resolution)
        return flat, coarse_elevation, np.zeros_like(coarse_elevation)

    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = filled[tuple(indices)]

    gradient_y, gradient_x = np.gradient(filled, coarse_resolution)
    slope = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # normalized_slope in [0, 1]: 0 -> flat, 1 -> at/above a steep cutoff.
    normalized_slope = np.clip(slope / 0.30, 0.0, 1.0)

    resolution_map = (
        cfg.adaptive_max_resolution
        - normalized_slope * (cfg.adaptive_max_resolution - cfg.adaptive_min_resolution)
    )

    return resolution_map, coarse_elevation, slope