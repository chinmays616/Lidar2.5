
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import CheckButtons
import open3d as o3d

from scipy import ndimage
from scipy.spatial import cKDTree

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BIN_FILE = (
    r"semantic_kitti_sample\dataset\sequences\00"
    r"\velodyne\000009.bin"
)

LABEL_FILE = (
    r"semantic_kitti_sample\dataset\sequences\00"
    r"\labels\000009.label"
)

RESOLUTION = 0.5

X_MIN = -50
X_MAX = 50

Y_MIN = -50
Y_MAX = 50

MAX_SLOPE = 15.0

# Slope display is capped independently of MAX_SLOPE (the traversability
# threshold) so that boundary/extrapolation artifacts at the edge of the
# scanned area don't blow out the color scale.
SLOPE_DISPLAY_MAX = 45.0

# Terrain-fill / slope-interpolation controls (fix 2).
# A cell farther than MAX_INTERP_DISTANCE meters from any real
# elevation sample is left unknown rather than extrapolated.
MAX_INTERP_DISTANCE = 3.0
INTERP_K_NEIGHBORS = 8

# Probabilistic raycasting controls (fix 4, retuned).
#
# P_FREE was originally 0.4, which under the standard log-odds
# update gives l_free = ln(0.4/0.6) = -0.405 -> after a single
# ray crossing, P(occupied) = sigmoid(-0.405) = 0.40, which does
# NOT cross the old FREE_PROB_THRESHOLD of 0.35. That meant a
# free-space cell needed at least two overlapping ray crossings
# before it could ever be resolved as "free" - a bad match for a
# single LiDAR frame over a large (100m x 100m) grid, where most
# cells are only ever crossed once or not at all. P_FREE is
# lowered to 0.3 (l_free = ln(0.3/0.7) = -0.847), so
# P(occupied) after just ONE crossing = sigmoid(-0.847) = 0.30,
# which already clears FREE_PROB_THRESHOLD. P_OCC / OCC_PROB_THRESHOLD
# were already single-pass-resolvable (sigmoid(ln(0.7/0.3)) = 0.70
# clears 0.65) and are left as-is.
SENSOR_ORIGIN = (0.0, 0.0)
P_OCC = 0.7           # inverse sensor model: P(occupied | hit)
P_FREE = 0.3          # inverse sensor model: P(occupied | ray passed through)
P_PRIOR = 0.5
LOG_ODDS_CLAMP = 10.0
OCC_PROB_THRESHOLD = 0.65
FREE_PROB_THRESHOLD = 0.35
# Raycasting is now vectorized (see probabilistic_occupancy_mapping),
# processing RAYCAST_BATCH_SIZE rays at a time with numpy scatter-add
# instead of a per-point Python loop + inner Bresenham while-loop, so
# a full-density pass over every point in a 100m x 100m grid is cheap
# enough to run with no stride at all. RAYCAST_STRIDE is kept as an
# escape hatch for very large multi-scan accumulations, not needed
# for a single frame.
RAYCAST_STRIDE = 1
RAYCAST_BATCH_SIZE = 20000


# ============================================================
# SEMANTIC CLASSES
# ============================================================

TRAVERSABLE_CLASSES = {
    40,   # road
    44,   # parking
    48,   # sidewalk
    49,   # other-ground
    72    # terrain
}


OBSTACLE_CLASSES = {
    10,   # car
    11,   # bicycle
    13,   # bus
    15,   # motorcycle
    16,   # on-rails
    18,   # truck
    20,   # other-vehicle

    30,   # person
    31,   # bicyclist
    32,   # motorcyclist

    50,   # building
    51,   # fence
    52,   # other-structure

    70,   # vegetation
    71,   # trunk

    80,   # pole
    81    # traffic-sign
}


# ============================================================
# SEMANTIC-AWARE SLOPE THRESHOLDS
#
# Obstacle classes (buildings, vegetation, poles, vehicles, ...)
# are already flagged non-traversable purely from their semantic
# label in both traversability_analysis() and
# variable_resolution_traversability() - slope is never even
# consulted for them, so a steep-looking gradient on the side of
# a building can't accidentally get it reclassified as
# traversable.
#
# For the TRAVERSABLE_CLASSES, though, a single global MAX_SLOPE
# doesn't reflect that different ground surface types tolerate
# different grades: a road/parking surface is engineered and can
# be trusted up to the standard MAX_SLOPE, a sidewalk is narrower
# and typically flatter so a stricter cap catches curb-adjacent
# noise, and open terrain is intentionally given more slack since
# off-road capable platforms may still want it flagged traversable
# on moderate grades. Any traversable class not listed here falls
# back to the global MAX_SLOPE.
# ============================================================

CLASS_MAX_SLOPE = {
    40: 15.0,   # road
    44: 15.0,   # parking
    48: 10.0,   # sidewalk - stricter, curbs/steps read as high local slope
    49: 12.0,   # other-ground
    72: 20.0,   # terrain - more permissive, open/off-road ground
}


def get_class_max_slope(label, default=MAX_SLOPE):
    """Per-class slope tolerance for TRAVERSABLE_CLASSES, falling
    back to the global MAX_SLOPE for any class not explicitly
    tuned in CLASS_MAX_SLOPE."""

    return CLASS_MAX_SLOPE.get(int(label), default)


# ============================================================
# SEMANTIC CLASS NAMES
# ============================================================

CLASS_NAMES = {

    0: "unlabeled",
    1: "outlier",

    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",

    30: "person",
    31: "bicyclist",
    32: "motorcyclist",

    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",

    50: "building",
    51: "fence",
    52: "other-structure",

    60: "lane-marking",

    70: "vegetation",
    71: "trunk",
    72: "terrain",

    80: "pole",
    81: "traffic-sign",

    99: "other",
    254: "moving",
    255: "unknown"
}


# ============================================================
# SEMANTIC CLASS COLORS (module-level so every consumer -
# point cloud viewer, 2D semantic map, legends - shares one
# single source of truth and therefore one consistent look)
# ============================================================

CLASS_COLORS = {

    0: [0.55, 0.55, 0.55],
    1: [0.55, 0.55, 0.55],

    10: [1.00, 0.00, 0.00],
    11: [1.00, 0.50, 0.00],
    13: [1.00, 0.00, 0.50],
    15: [0.80, 0.00, 0.00],
    16: [0.70, 0.20, 0.20],
    18: [0.70, 0.00, 0.00],
    20: [0.90, 0.20, 0.20],

    30: [1.00, 1.00, 0.00],
    31: [1.00, 0.80, 0.00],
    32: [0.80, 0.80, 0.00],

    40: [0.30, 0.30, 0.30],   # road - neutral gray (not green!)
    44: [0.45, 0.45, 0.45],   # parking
    48: [0.65, 0.65, 0.65],   # sidewalk
    49: [0.80, 0.75, 0.55],   # other-ground

    50: [0.55, 0.00, 0.00],   # building (distinct from vehicles)
    51: [0.60, 0.40, 0.20],   # fence
    52: [0.40, 0.40, 0.55],   # other-structure

    60: [0.90, 0.90, 0.20],   # lane-marking

    70: [0.00, 0.55, 0.00],   # vegetation - green
    71: [0.00, 0.35, 0.00],   # trunk - dark green
    72: [0.35, 0.75, 0.20],   # terrain - lighter green

    80: [0.55, 0.00, 1.00],   # pole - purple
    81: [0.80, 0.00, 1.00],   # traffic-sign - lighter purple

    99: [0.80, 0.80, 0.80],
    254: [1.00, 1.00, 1.00],
    255: [0.20, 0.20, 0.20],

    -1: [1.00, 1.00, 1.00]    # unlabeled/empty grid cell -> white
}


def get_semantic_colormap(present_labels):
    """
    Build a discrete, per-class categorical colormap for a set of
    semantic label ids that are actually present in the current grid.

    Returns:
        cmap        - matplotlib.colors.ListedColormap
        norm        - matplotlib.colors.BoundaryNorm mapping raw
                      label ids -> the correct color slot
        legend_handles - list of matplotlib Patch objects for a legend
        sorted_labels  - the label ids in the order used by the cmap
    """

    sorted_labels = sorted(present_labels)

    colors = [
        CLASS_COLORS.get(label, [0.5, 0.5, 0.5])
        for label in sorted_labels
    ]

    cmap = mcolors.ListedColormap(colors)

    # Boundaries sit halfway between consecutive label ids so that
    # BoundaryNorm routes each exact label value into its own bin.
    boundaries = []

    for i, label in enumerate(sorted_labels):

        if i == 0:

            boundaries.append(label - 0.5)

        boundaries.append(label + 0.5)

    norm = mcolors.BoundaryNorm(
        boundaries,
        cmap.N
    )

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=CLASS_COLORS.get(label, [0.5, 0.5, 0.5]),
            edgecolor="black",
            linewidth=0.5
        )
        for label in sorted_labels
    ]

    legend_labels = [
        CLASS_NAMES.get(label, f"class {label}")
        for label in sorted_labels
    ]

    return cmap, norm, legend_handles, legend_labels


# ============================================================
# STAGE 1
# LIDAR LOADING
# ============================================================

def load_pointcloud(file):

    points = np.fromfile(
        file,
        dtype=np.float32
    ).reshape(-1, 4)

    xyz = points[:, :3]

    intensity = points[:, 3]

    print("LiDAR points:", len(points))

    return points, xyz, intensity


# ============================================================
# STAGE 2
# BEV PROJECTION
# ============================================================

def create_bev(
    xyz,
    resolution=0.1
):

    x_min = np.min(xyz[:, 0])
    x_max = np.max(xyz[:, 0])

    y_min = np.min(xyz[:, 1])
    y_max = np.max(xyz[:, 1])

    grid_width = int(
        (x_max - x_min) / resolution
    ) + 1

    grid_height = int(
        (y_max - y_min) / resolution
    ) + 1

    bev = np.full(
        (grid_height, grid_width),
        np.nan
    )

    grid_x = (
        (xyz[:, 0] - x_min) /
        resolution
    ).astype(int)

    grid_y = (
        (xyz[:, 1] - y_min) /
        resolution
    ).astype(int)

    valid = (
        (grid_x >= 0) &
        (grid_x < grid_width) &
        (grid_y >= 0) &
        (grid_y < grid_height)
    )

    grid_x = grid_x[valid]
    grid_y = grid_y[valid]

    z = xyz[:, 2][valid]

    for gx, gy, z_value in zip(
        grid_x,
        grid_y,
        z
    ):

        if np.isnan(bev[gy, gx]):

            bev[gy, gx] = z_value

        else:

            bev[gy, gx] = max(
                bev[gy, gx],
                z_value
            )

    return bev


# ============================================================
# STAGE 3
# FIXED RESOLUTION ELEVATION GRID
# ============================================================

def create_elevation_grid(
    xyz,
    resolution=0.5
):

    x_min = np.min(xyz[:, 0])
    x_max = np.max(xyz[:, 0])

    y_min = np.min(xyz[:, 1])
    y_max = np.max(xyz[:, 1])

    grid_width = int(
        (x_max - x_min) / resolution
    ) + 1

    grid_height = int(
        (y_max - y_min) / resolution
    ) + 1

    elevation_grid = np.full(
        (grid_height, grid_width),
        np.nan
    )

    grid_x = (
        (xyz[:, 0] - x_min) /
        resolution
    ).astype(int)

    grid_y = (
        (xyz[:, 1] - y_min) /
        resolution
    ).astype(int)

    valid = (
        (grid_x >= 0) &
        (grid_x < grid_width) &
        (grid_y >= 0) &
        (grid_y < grid_height)
    )

    for gx, gy, z in zip(
        grid_x[valid],
        grid_y[valid],
        xyz[:, 2][valid]
    ):

        if np.isnan(
            elevation_grid[gy, gx]
        ):

            elevation_grid[gy, gx] = z

        else:

            elevation_grid[gy, gx] = max(
                elevation_grid[gy, gx],
                z
            )

    return elevation_grid


# ============================================================
# STAGE 3
# GROUND DETECTION
# ============================================================

def ground_detection(
    xyz,
    ground_threshold=-1.5
):

    ground_mask = (
        xyz[:, 2] < ground_threshold
    )

    ground_points = xyz[
        ground_mask
    ]

    non_ground_points = xyz[
        ~ground_mask
    ]

    return ground_points, non_ground_points


# ============================================================
# STAGE 3 (fix 2)
# BOUNDED IDW ELEVATION FILL
#
# Fills NaN elevation cells from nearby REAL samples only,
# using inverse-distance weighting over the k nearest observed
# cells, and refuses to fill anything farther than
# max_distance meters from an observation. Those far cells stay
# NaN ("unknown") instead of inheriting a flat, arbitrarily
# extrapolated plateau the way nearest-neighbor fill did.
# ============================================================

def interpolate_elevation_idw(
    elevation_grid,
    resolution,
    max_distance=MAX_INTERP_DISTANCE,
    k=INTERP_K_NEIGHBORS
):

    grid_height, grid_width = elevation_grid.shape

    valid_mask = ~np.isnan(elevation_grid)

    if not np.any(valid_mask):

        return (
            elevation_grid.copy(),
            np.zeros_like(elevation_grid, dtype=bool)
        )

    yy, xx = np.mgrid[0:grid_height, 0:grid_width]

    valid_coords = np.column_stack(
        [xx[valid_mask], yy[valid_mask]]
    ).astype(float) * resolution

    valid_values = elevation_grid[valid_mask]

    tree = cKDTree(valid_coords)

    all_coords = np.column_stack(
        [xx.ravel(), yy.ravel()]
    ).astype(float) * resolution

    k_query = min(k, len(valid_values))

    dist, idx = tree.query(
        all_coords,
        k=k_query,
        distance_upper_bound=max_distance
    )

    if k_query == 1:

        dist = dist[:, None]
        idx = idx[:, None]

    n_cells = all_coords.shape[0]

    filled = np.full(n_cells, np.nan)
    observed_mask = np.zeros(n_cells, dtype=bool)

    n_values = len(valid_values)

    for i in range(n_cells):

        d = dist[i]
        ix = idx[i]

        finite = np.isfinite(d) & (ix < n_values)

        if not np.any(finite):
            continue

        d = d[finite]
        ix = ix[finite]

        if d[0] < 1e-9:

            filled[i] = valid_values[ix[0]]

        else:

            w = 1.0 / (d ** 2)

            filled[i] = np.sum(w * valid_values[ix]) / np.sum(w)

        observed_mask[i] = True

    filled_grid = filled.reshape(grid_height, grid_width)

    observed_mask = observed_mask.reshape(
        grid_height, grid_width
    ) | valid_mask

    filled_grid = np.where(
        observed_mask,
        filled_grid,
        np.nan
    )

    return filled_grid, observed_mask


def _nan_aware_gaussian(grid, sigma):
    """
    Gaussian-smooth a grid that may contain NaNs without letting
    NaN neighborhoods bleed a fake near-zero value into valid
    cells (a plain gaussian_filter on NaN-as-zero data biases
    every edge cell toward zero). NaN cells stay NaN.
    """

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


# ============================================================
# STAGE 3
# TERRAIN ANALYSIS  (fix 2: bounded fill, no more slope rays)
#
# Cells with no real elevation sample within max_interp_distance
# are left as NaN in both the smoothed elevation surface and the
# returned slope grids, and are reported via unknown_mask so
# downstream traversability logic can treat them as "unknown"
# instead of silently trusting a fabricated gradient.
# ============================================================

def terrain_analysis(
    elevation_grid,
    resolution,
    smoothing_sigma=1.0,
    max_interp_distance=MAX_INTERP_DISTANCE,
    k_neighbors=INTERP_K_NEIGHBORS
):

    valid = ~np.isnan(elevation_grid)

    if not np.any(valid):

        unknown_mask = np.ones_like(elevation_grid, dtype=bool)

        return (
            np.full_like(elevation_grid, np.nan),
            np.full_like(elevation_grid, np.nan),
            unknown_mask
        )

    filled_grid, observed_mask = interpolate_elevation_idw(
        elevation_grid,
        resolution,
        max_distance=max_interp_distance,
        k=k_neighbors
    )

    if smoothing_sigma > 0:

        filled_grid = _nan_aware_gaussian(
            filled_grid,
            smoothing_sigma
        )

    gradient_y, gradient_x = np.gradient(
        filled_grid,
        resolution
    )

    slope = np.sqrt(
        gradient_x ** 2 +
        gradient_y ** 2
    )

    slope_degrees = np.degrees(
        np.arctan(slope)
    )

    # A cell (or any of its gradient neighbors) that was never
    # within max_interp_distance of a real sample is unknown -
    # NaN propagates through gaussian/gradient math naturally,
    # so this just formalizes it into an explicit mask.
    unknown_mask = np.isnan(filled_grid) | np.isnan(slope_degrees)

    slope = np.where(unknown_mask, np.nan, slope)
    slope_degrees = np.where(unknown_mask, np.nan, slope_degrees)

    return slope, slope_degrees, unknown_mask


# ============================================================
# STAGE 4
# INITIAL ADAPTIVE RESOLUTION
# ============================================================

def adaptive_resolution(
    xyz,
    coarse_resolution=1.0
):

    coarse_elevation = create_elevation_grid(
        xyz,
        coarse_resolution
    )

    filled = coarse_elevation.copy()

    valid = ~np.isnan(filled)

    if np.any(valid):

        indices = ndimage.distance_transform_edt(
            ~valid,
            return_distances=False,
            return_indices=True
        )

        filled = filled[
            tuple(indices)
        ]

    else:

        return (
            np.full(
                coarse_elevation.shape,
                coarse_resolution
            ),
            coarse_elevation,
            np.zeros_like(coarse_elevation)
        )

    gradient_y, gradient_x = np.gradient(
        filled,
        coarse_resolution
    )

    slope = np.sqrt(
        gradient_x ** 2 +
        gradient_y ** 2
    )

    resolution_map = np.full(
        coarse_elevation.shape,
        coarse_resolution
    )

    resolution_map[
        slope > 0.10
    ] = 0.5

    resolution_map[
        slope > 0.30
    ] = 0.25

    return (
        resolution_map,
        coarse_elevation,
        slope
    )


# ============================================================
# STAGE 5
# LOAD SEMANTIC LABELS
# ============================================================

def load_semantic_labels(
    label_file,
    num_points
):

    labels = np.fromfile(
        label_file,
        dtype=np.uint32
    )

    if num_points != len(labels):

        raise ValueError(
            "Point and label counts do not match!"
        )

    semantic_labels = (
        labels & 0xFFFF
    )

    return semantic_labels


def create_semantic_colors(
    semantic_labels
):

    colors = np.zeros(
        (len(semantic_labels), 3),
        dtype=np.float64
    )

    for i, label in enumerate(
        semantic_labels
    ):

        colors[i] = CLASS_COLORS.get(
            int(label),
            [0.5, 0.5, 0.5]
        )

    return colors


# ============================================================
# STAGE 5
# SEMANTIC POINT CLOUD VISUALIZATION
# ============================================================

def visualize_semantic_pointcloud(
    xyz,
    colors
):

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(xyz)
    )

    pcd.colors = (
        o3d.utility.Vector3dVector(colors)
    )

    o3d.visualization.draw_geometries(
        [pcd]
    )


# ============================================================
# STAGE 9
# SEMANTIC 2.5D GRID
# ============================================================

def create_semantic_grid(
    xyz,
    semantic_labels,
    resolution=0.5,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50
):

    grid_width = int(
        (x_max - x_min) /
        resolution
    )

    grid_height = int(
        (y_max - y_min) /
        resolution
    )

    elevation_grid = np.full(
        (grid_height, grid_width),
        np.nan
    )

    semantic_grid = np.full(
        (grid_height, grid_width),
        -1,
        dtype=np.int32
    )

    point_count_grid = np.zeros(
        (grid_height, grid_width),
        dtype=np.int32
    )

    grid_x = (
        (xyz[:, 0] - x_min) /
        resolution
    ).astype(int)

    grid_y = (
        (xyz[:, 1] - y_min) /
        resolution
    ).astype(int)

    valid = (
        (grid_x >= 0) &
        (grid_x < grid_width) &
        (grid_y >= 0) &
        (grid_y < grid_height)
    )

    grid_x_valid = grid_x[valid]
    grid_y_valid = grid_y[valid]

    z_valid = xyz[:, 2][valid]

    labels_valid = semantic_labels[valid]

    for gx, gy, z in zip(
        grid_x_valid,
        grid_y_valid,
        z_valid
    ):

        if np.isnan(
            elevation_grid[gy, gx]
        ):

            elevation_grid[gy, gx] = z

        else:

            elevation_grid[gy, gx] = max(
                elevation_grid[gy, gx],
                z
            )

        point_count_grid[
            gy, gx
        ] += 1

    # Vectorized dominant-label-per-cell vote (replaces an O(H*W*N)
    # nested-loop + boolean-mask rescan that re-tested every point
    # against every cell). Points are bucketed by their flat cell
    # index and the majority label per bucket is taken via
    # np.unique's inverse-index / bincount trick.
    if len(labels_valid) > 0:

        flat_cell_idx = (
            grid_y_valid.astype(np.int64) * grid_width +
            grid_x_valid.astype(np.int64)
        )

        order = np.argsort(flat_cell_idx, kind="stable")

        sorted_cells = flat_cell_idx[order]
        sorted_labels = labels_valid[order]

        unique_cells, start_idx, counts = np.unique(
            sorted_cells,
            return_index=True,
            return_counts=True
        )

        for cell_flat, start, count in zip(
            unique_cells, start_idx, counts
        ):

            cell_labels = sorted_labels[start:start + count]

            uniq, label_counts = np.unique(
                cell_labels,
                return_counts=True
            )

            dominant_label = uniq[np.argmax(label_counts)]

            gy = cell_flat // grid_width
            gx = cell_flat % grid_width

            semantic_grid[gy, gx] = dominant_label

    return (
        elevation_grid,
        semantic_grid,
        point_count_grid
    )


# ============================================================
# STAGE 10
# SEMANTIC-AWARE VARIABLE RESOLUTION
# ============================================================

def semantic_aware_variable_grid(
    xyz,
    semantic_labels,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50
):

    HIGH_IMPORTANCE_CLASSES = {
        10, 11, 13, 15,
        18, 20, 30, 31, 32
    }

    MEDIUM_IMPORTANCE_CLASSES = {
        50, 51, 52,
        70, 71,
        80, 81
    }

    adaptive_cells = {}

    point_resolutions = np.zeros(
        len(xyz)
    )

    for i in range(len(xyz)):

        x = xyz[i, 0]
        y = xyz[i, 1]
        z = xyz[i, 2]

        label = semantic_labels[i]

        if (
            x < x_min or
            x >= x_max or
            y < y_min or
            y >= y_max
        ):
            continue

        distance = np.sqrt(
            x ** 2 + y ** 2
        )

        # Distance-based resolution
        if distance < 20:

            r = 0.25

        elif distance < 40:

            r = 0.50

        else:

            r = 1.00

        # Semantic importance
        if label in HIGH_IMPORTANCE_CLASSES:

            r = 0.25

        elif label in MEDIUM_IMPORTANCE_CLASSES:

            r = min(
                r,
                0.50
            )

        gx = int(
            np.floor(
                (x - x_min) / r
            )
        )

        gy = int(
            np.floor(
                (y - y_min) / r
            )
        )

        key = (
            gx,
            gy,
            r
        )

        if key not in adaptive_cells:

            adaptive_cells[key] = {

                "x": x,

                "y": y,

                "resolution": r,

                "elevation": z,

                "point_count": 1,

                "semantic_label": int(label)
            }

        else:

            cell = adaptive_cells[key]

            cell["elevation"] = max(
                cell["elevation"],
                z
            )

            cell["point_count"] += 1

        point_resolutions[i] = r

    # Add semantic names
    for cell in adaptive_cells.values():

        label = cell[
            "semantic_label"
        ]

        cell[
            "semantic_name"
        ] = CLASS_NAMES.get(
            label,
            "Unknown"
        )

    return (
        adaptive_cells,
        point_resolutions
    )


# ============================================================
# STAGE 10
# VARIABLE-RESOLUTION OCCUPANCY
# ============================================================

def variable_resolution_occupancy(
    adaptive_cells
):

    variable_occupancy = []

    for cell in adaptive_cells.values():

        semantic_label = cell[
            "semantic_label"
        ]

        if semantic_label in OBSTACLE_CLASSES:

            occupancy = 1

        elif semantic_label in TRAVERSABLE_CLASSES:

            occupancy = 0

        else:

            occupancy = -1

        variable_occupancy.append({

            "x": cell["x"],

            "y": cell["y"],

            "resolution": cell["resolution"],

            "semantic_label":
                semantic_label,

            "semantic_name":
                cell["semantic_name"],

            "elevation":
                cell["elevation"],

            "point_count":
                cell["point_count"],

            "occupancy":
                occupancy
        })

    return variable_occupancy


# ============================================================
# STAGE 10
# VARIABLE-RESOLUTION TRAVERSABILITY
# ============================================================

def variable_resolution_traversability(
    variable_occupancy,
    slope_degrees,
    resolution=0.5,
    x_min=-50,
    y_min=-50,
    max_slope=15.0
):

    grid_height, grid_width = (
        slope_degrees.shape
    )

    for cell in variable_occupancy:

        x = cell["x"]
        y = cell["y"]

        gx = int(
            (x - x_min) /
            resolution
        )

        gy = int(
            (y - y_min) /
            resolution
        )

        if (
            gx < 0 or
            gx >= grid_width or
            gy < 0 or
            gy >= grid_height
        ):

            local_slope = np.nan

        else:

            local_slope = slope_degrees[
                gy,
                gx
            ]

        cell["slope"] = local_slope

        # Obstacle
        if cell["occupancy"] == 1:

            cell["traversability"] = 0

        # Unknown occupancy
        elif cell["occupancy"] == -1:

            cell["traversability"] = 2

        # Unknown terrain (outside the bounded slope interpolation)
        elif np.isnan(local_slope):

            cell["traversability"] = 2

        # Excessive slope (semantic-aware: road/parking/sidewalk/
        # terrain each get their own tolerance instead of one
        # global cutoff - see CLASS_MAX_SLOPE)
        elif local_slope > get_class_max_slope(
            cell["semantic_label"], max_slope
        ):

            cell["traversability"] = 0

        # Traversable
        else:

            cell["traversability"] = 1

    return variable_occupancy


# ============================================================
# STAGE 8
# FIXED GRID OCCUPANCY (semantic-rule based)
# ============================================================

def occupancy_mapping(
    xyz,
    semantic_labels,
    resolution=0.5,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50
):

    grid_width = int(
        (x_max - x_min) /
        resolution
    )

    grid_height = int(
        (y_max - y_min) /
        resolution
    )

    occupancy_grid = np.full(
        (grid_height, grid_width),
        -1,
        dtype=np.int8
    )

    grid_x = (
        (xyz[:, 0] - x_min) /
        resolution
    ).astype(int)

    grid_y = (
        (xyz[:, 1] - y_min) /
        resolution
    ).astype(int)

    valid = (
        (grid_x >= 0) &
        (grid_x < grid_width) &
        (grid_y >= 0) &
        (grid_y < grid_height)
    )

    for gx, gy, label in zip(
        grid_x[valid],
        grid_y[valid],
        semantic_labels[valid]
    ):

        if label in OBSTACLE_CLASSES:

            occupancy_grid[
                gy, gx
            ] = 1

        elif (
            label in TRAVERSABLE_CLASSES
            and
            occupancy_grid[gy, gx] != 1
        ):

            occupancy_grid[
                gy, gx
            ] = 0

    obstacle_map = (
        occupancy_grid == 1
    )

    return (
        occupancy_grid,
        obstacle_map
    )

def probabilistic_occupancy_mapping(
    xyz,
    resolution=0.5,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50,
    sensor_origin=SENSOR_ORIGIN,
    p_occ=P_OCC,
    p_free=P_FREE,
    p_prior=P_PRIOR,
    log_odds_clamp=LOG_ODDS_CLAMP,
    occ_threshold=OCC_PROB_THRESHOLD,
    free_threshold=FREE_PROB_THRESHOLD,
    stride=RAYCAST_STRIDE,
    batch_size=RAYCAST_BATCH_SIZE
):
    """
    Returns:
        occupancy_grid - int8 grid: 1 occupied, 0 free, -1 unknown
                         (same encoding as occupancy_mapping(), so
                         it drops into the dashboard / stats code
                         unchanged)
        probability_grid - float grid of P(occupied) in [0, 1]
    """

    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)

    l_occ = math.log(p_occ / (1.0 - p_occ))
    l_free = math.log(p_free / (1.0 - p_free))

    log_odds_flat = np.zeros(grid_height * grid_width, dtype=np.float64)

    ox, oy = sensor_origin

    origin_gx = int((ox - x_min) / resolution)
    origin_gy = int((oy - y_min) / resolution)

    origin_gx = min(max(origin_gx, 0), grid_width - 1)
    origin_gy = min(max(origin_gy, 0), grid_height - 1)

    points = xyz[::stride] if stride > 1 else xyz

    gx_all = ((points[:, 0] - x_min) / resolution).astype(np.int64)
    gy_all = ((points[:, 1] - y_min) / resolution).astype(np.int64)

    in_bounds = (
        (gx_all >= 0) & (gx_all < grid_width) &
        (gy_all >= 0) & (gy_all < grid_height)
    )

    gx_all = gx_all[in_bounds]
    gy_all = gy_all[in_bounds]

    n_points = len(gx_all)

    for start in range(0, n_points, batch_size):

        gx_batch = gx_all[start:start + batch_size]
        gy_batch = gy_all[start:start + batch_size]

        dx = gx_batch - origin_gx
        dy = gy_batch - origin_gy

        # Chebyshev distance = number of discrete steps a Bresenham
        # walk from the origin to this endpoint would actually take.
        n_steps_actual = np.maximum(np.abs(dx), np.abs(dy))

        # Only used as a parametric denominator to avoid a 0/0 for
        # rays that terminate in the origin's own cell; classification
        # below uses n_steps_actual, not this, so it can't miscount
        # a degenerate ray as having free-space cells.
        n_steps_for_t = np.maximum(n_steps_actual, 1)

        max_steps = int(n_steps_for_t.max())

        k = np.arange(max_steps + 1)[:, None]                  # (S+1, 1)
        t = k / n_steps_for_t[None, :]                          # (S+1, B)

        within_ray = t <= 1.0

        cell_x = np.rint(origin_gx + dx[None, :] * t).astype(np.int64)
        cell_y = np.rint(origin_gy + dy[None, :] * t).astype(np.int64)

        in_grid = (
            (cell_x >= 0) & (cell_x < grid_width) &
            (cell_y >= 0) & (cell_y < grid_height)
        )

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

    np.clip(log_odds, -log_odds_clamp, log_odds_clamp, out=log_odds)

    probability_grid = 1.0 - 1.0 / (1.0 + np.exp(log_odds))

    occupancy_grid = np.full((grid_height, grid_width), -1, dtype=np.int8)

    occupancy_grid[probability_grid >= occ_threshold] = 1
    occupancy_grid[probability_grid <= free_threshold] = 0

    return occupancy_grid, probability_grid

def traversability_analysis(
    elevation_grid,
    semantic_grid,
    resolution=0.5,
    max_slope=15.0,
    smoothing_sigma=1.0,
    max_interp_distance=MAX_INTERP_DISTANCE,
    k_neighbors=INTERP_K_NEIGHBORS
):

    slope, slope_degrees, terrain_unknown = terrain_analysis(
        elevation_grid,
        resolution,
        smoothing_sigma=smoothing_sigma,
        max_interp_distance=max_interp_distance,
        k_neighbors=k_neighbors
    )

    traversability_grid = np.full(
        semantic_grid.shape,
        2,
        dtype=np.int8
    )

    for gy in range(
        semantic_grid.shape[0]
    ):

        for gx in range(
            semantic_grid.shape[1]
        ):

            label = semantic_grid[
                gy,
                gx
            ]

            if label == -1:

                traversability_grid[
                    gy, gx
                ] = 2

                continue

            if label in OBSTACLE_CLASSES:

                traversability_grid[
                    gy, gx
                ] = 0

                continue

            if label not in TRAVERSABLE_CLASSES:

                traversability_grid[
                    gy, gx
                ] = 0

                continue

            if terrain_unknown[gy, gx]:

                traversability_grid[
                    gy, gx
                ] = 2

                continue

            # Semantic-aware slope cutoff: road/parking/sidewalk/
            # terrain each get their own tolerance (CLASS_MAX_SLOPE)
            # instead of one blanket MAX_SLOPE for every ground type.
            if slope_degrees[
                gy, gx
            ] > get_class_max_slope(label, max_slope):

                traversability_grid[
                    gy, gx
                ] = 0

            else:

                traversability_grid[
                    gy, gx
                ] = 1

    return (
        traversability_grid,
        slope_degrees
    )

def variable_grid_statistics(
    variable_occupancy
):

    occupied = sum(
        cell["occupancy"] == 1
        for cell in variable_occupancy
    )

    free = sum(
        cell["occupancy"] == 0
        for cell in variable_occupancy
    )

    unknown = sum(
        cell["occupancy"] == -1
        for cell in variable_occupancy
    )

    traversable = sum(
        cell["traversability"] == 1
        for cell in variable_occupancy
    )

    not_traversable = sum(
        cell["traversability"] == 0
        for cell in variable_occupancy
    )

    unknown_traversability = sum(
        cell["traversability"] == 2
        for cell in variable_occupancy
    )

    resolution_counts = {}

    for cell in variable_occupancy:

        r = cell["resolution"]

        resolution_counts[r] = (
            resolution_counts.get(r, 0) + 1
        )

    print("\n")
    print("================================================")
    print("       VARIABLE GRID STATISTICS")
    print("================================================")

    print(
        "Total adaptive cells:",
        len(variable_occupancy)
    )

    print("\n--- RESOLUTION ---")

    for r, count in sorted(
        resolution_counts.items()
    ):

        print(
            f"{r:.2f} m cells:",
            count
        )

    print("\n--- OCCUPANCY ---")

    print(
        "Occupied:",
        occupied
    )

    print(
        "Free:",
        free
    )

    print(
        "Unknown:",
        unknown
    )

    print("\n--- TRAVERSABILITY ---")

    print(
        "Traversable:",
        traversable
    )

    print(
        "Not traversable:",
        not_traversable
    )

    print(
        "Unknown:",
        unknown_traversability
    )

    print(
        "================================================"
    )

def visualization_dashboard(
    xyz,
    elevation_grid,
    semantic_grid,
    point_count_grid,
    occupancy_grid,
    slope_degrees,
    traversability_grid,
    resolution=0.5,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50,
    max_slope=15.0
):

    grid_height, grid_width = (
        elevation_grid.shape
    )

    fig = plt.figure(
        figsize=(17, 13)
    )

    extent = [
        x_min,
        x_max,
        y_min,
        y_max
    ]

    axes = []
    images = []
    titles = []

    # --------------------------------------------------------
    # Elevation
    # --------------------------------------------------------

    ax1 = fig.add_subplot(
        2, 3, 1
    )

    image1 = ax1.imshow(
        elevation_grid,
        origin="lower",
        extent=extent
    )

    ax1.set_title(
        "Elevation Map"
    )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    fig.colorbar(
        image1,
        ax=ax1,
        label="Elevation (m)"
    )

    axes.append(ax1)
    images.append(image1)
    titles.append("Elevation")

    # --------------------------------------------------------
    # Semantic  (discrete categorical colormap; legend moved
    # to a figure-level legend below - see FIX 3 at the bottom
    # of this function)
    # --------------------------------------------------------

    ax2 = fig.add_subplot(
        2, 3, 2
    )

    present_labels = set(
        np.unique(semantic_grid).tolist()
    )

    semantic_cmap, semantic_norm, legend_handles, legend_labels = (
        get_semantic_colormap(present_labels)
    )

    image2 = ax2.imshow(
        semantic_grid,
        origin="lower",
        extent=extent,
        cmap=semantic_cmap,
        norm=semantic_norm,
        interpolation="nearest"
    )

    ax2.set_title(
        "Semantic Map"
    )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    axes.append(ax2)
    images.append(image2)
    titles.append("Semantic")

    # --------------------------------------------------------
    # Point Density
    # --------------------------------------------------------

    ax3 = fig.add_subplot(
        2, 3, 3
    )

    image3 = ax3.imshow(
        point_count_grid,
        origin="lower",
        extent=extent
    )

    ax3.set_title(
        "LiDAR Point Density"
    )

    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")

    fig.colorbar(
        image3,
        ax=ax3,
        label="Points / Cell"
    )

    axes.append(ax3)
    images.append(image3)
    titles.append("Point Density")

    # --------------------------------------------------------
    # Occupancy  (FIX 4: probabilistic raycast grid;
    # FIX 2 (dashboard side): mask Unknown (-1) as transparent)
    # --------------------------------------------------------

    ax4 = fig.add_subplot(
        2, 3, 4
    )

    occupancy_masked = np.ma.masked_where(
        occupancy_grid == -1,
        occupancy_grid
    )

    occupancy_cmap = mcolors.ListedColormap(
        ["#4daf4a", "#e41a1c"]   # 0 free -> green, 1 occupied -> red
    )

    occupancy_cmap.set_bad(color="none")

    image4 = ax4.imshow(
        occupancy_masked,
        origin="lower",
        extent=extent,
        cmap=occupancy_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax4.set_facecolor("white")

    ax4.set_title(
        "Occupancy Map (probabilistic raycast)"
    )

    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")

    occ_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4daf4a", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#e41a1c", edgecolor="black"),
    ]

    ax4.legend(
        occ_handles,
        ["Free", "Occupied"],
        loc="upper right",
        fontsize=7
    )

    axes.append(ax4)
    images.append(image4)
    titles.append("Occupancy")

    # --------------------------------------------------------
    # Slope  (smoothed, bounded-interpolation slope; unknown
    # cells masked transparent instead of extrapolated)
    # --------------------------------------------------------

    ax5 = fig.add_subplot(
        2, 3, 5
    )

    slope_display = np.clip(
        slope_degrees,
        0,
        SLOPE_DISPLAY_MAX
    )

    slope_display = np.ma.masked_invalid(slope_display)

    slope_cmap = plt.get_cmap("turbo").copy()
    slope_cmap.set_bad(color="none")

    image5 = ax5.imshow(
        slope_display,
        origin="lower",
        extent=extent,
        cmap=slope_cmap,
        vmin=0,
        vmax=SLOPE_DISPLAY_MAX
    )

    ax5.set_facecolor("white")

    ax5.set_title(
        f"Terrain Slope (capped at {SLOPE_DISPLAY_MAX:.0f}°, "
        f"unmapped area masked)"
    )

    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")

    fig.colorbar(
        image5,
        ax=ax5,
        label="Slope (degrees)"
    )

    axes.append(ax5)
    images.append(image5)
    titles.append("Slope")

    # --------------------------------------------------------
    # Traversability  (mask Unknown (2) as transparent)
    # --------------------------------------------------------

    ax6 = fig.add_subplot(
        2, 3, 6
    )

    traversability_masked = np.ma.masked_where(
        traversability_grid == 2,
        traversability_grid
    )

    traversability_cmap = mcolors.ListedColormap(
        ["#e41a1c", "#4daf4a"]   # 0 no -> red, 1 yes -> green
    )

    traversability_cmap.set_bad(color="none")

    image6 = ax6.imshow(
        traversability_masked,
        origin="lower",
        extent=extent,
        cmap=traversability_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax6.set_facecolor("white")

    ax6.set_title(
        "Traversability Map"
    )

    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Y (m)")

    trav_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#e41a1c", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#4daf4a", edgecolor="black"),
    ]

    ax6.legend(
        trav_handles,
        ["Not traversable", "Traversable"],
        loc="upper right",
        fontsize=7
    )

    axes.append(ax6)
    images.append(image6)
    titles.append("Traversability")

    fig.suptitle(
        "LiDAR 2.5D Semantic Mapping Dashboard",
        fontsize=18
    )

  
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        fontsize=7,
        frameon=True,
        title="Semantic class"
    )

    plt.tight_layout(
        rect=[0, 0.10, 1, 0.96]
    )

    check_ax = fig.add_axes(
        [0.005, 0.90, 0.09, 0.09]
    )

    check = CheckButtons(
        check_ax,
        titles,
        [True] * len(titles)
    )

    def _toggle(label):

        idx = titles.index(label)

        axes[idx].set_visible(
            not axes[idx].get_visible()
        )

        fig.canvas.draw_idle()

    check.on_clicked(_toggle)

    fig._layer_toggle_widget = check

    plt.show()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_cells = (
        grid_width *
        grid_height
    )

    occupied = np.sum(
        occupancy_grid == 1
    )

    free = np.sum(
        occupancy_grid == 0
    )

    unknown = np.sum(
        occupancy_grid == -1
    )

    traversable = np.sum(
        traversability_grid == 1
    )

    not_traversable = np.sum(
        traversability_grid == 0
    )

    unknown_traversability = np.sum(
        traversability_grid == 2
    )

    print("\n")
    print("================================================")
    print("             LIDAR DASHBOARD")
    print("================================================")

    print(
        "LiDAR points:",
        len(xyz)
    )

    print(
        "Grid resolution:",
        resolution,
        "m"
    )

    print(
        "Grid size:",
        grid_width,
        "x",
        grid_height
    )

    print(
        "Total cells:",
        total_cells
    )

    print("\n--- OCCUPANCY (probabilistic raycast) ---")

    print(
        "Occupied:",
        occupied
    )

    print(
        "Free:",
        free
    )

    print(
        "Unknown:",
        unknown
    )

    print("\n--- TRAVERSABILITY ---")

    print(
        "Traversable:",
        traversable
    )

    print(
        "Not traversable:",
        not_traversable
    )

    print(
        "Unknown:",
        unknown_traversability
    )

    print(
        "Maximum allowed slope:",
        max_slope,
        "degrees"
    )

    print(
        "================================================"
    )

def launch_open3d_viewer(
    xyz,
    semantic_labels
):
    """Interactive 3D point-cloud viewer (rotate/pan/zoom)."""

    colors = create_semantic_colors(
        semantic_labels
    )

    visualize_semantic_pointcloud(
        xyz,
        colors
    )


def build_interactive_composite(
    occupancy_grid,
    traversability_grid,
    semantic_grid,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50,
    resolution=0.5,
    output_path="interactive_dashboard.html"
):
    """
    Build a single top-down 2.5D composite with Occupancy,
    Traversability, and Semantic layers as independently
    toggleable Plotly traces (click a legend entry to
    show/hide that layer). Supports native zoom/pan/hover.
    Writes a standalone HTML file and returns its path.
    """

    import plotly.graph_objects as go

    x_coords = np.arange(
        x_min, x_max, resolution
    )

    y_coords = np.arange(
        y_min, y_max, resolution
    )

    occupancy_masked = np.where(
        occupancy_grid == -1,
        np.nan,
        occupancy_grid
    )

    traversability_masked = np.where(
        traversability_grid == 2,
        np.nan,
        traversability_grid
    )

    semantic_masked = np.where(
        semantic_grid == -1,
        np.nan,
        semantic_grid
    )

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=semantic_masked,
            x=x_coords,
            y=y_coords,
            colorscale="Turbo",
            name="Semantic",
            visible=True,
            showscale=False,
            hovertemplate="x=%{x:.1f} y=%{y:.1f} class=%{z}<extra>Semantic</extra>"
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=occupancy_masked,
            x=x_coords,
            y=y_coords,
            colorscale=[[0, "#4daf4a"], [1, "#e41a1c"]],
            zmin=0,
            zmax=1,
            name="Occupancy",
            visible="legendonly",
            showscale=False,
            hovertemplate="x=%{x:.1f} y=%{y:.1f} occupied=%{z}<extra>Occupancy</extra>"
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=traversability_masked,
            x=x_coords,
            y=y_coords,
            colorscale=[[0, "#e41a1c"], [1, "#4daf4a"]],
            zmin=0,
            zmax=1,
            name="Traversability",
            visible="legendonly",
            showscale=False,
            hovertemplate="x=%{x:.1f} y=%{y:.1f} traversable=%{z}<extra>Traversability</extra>"
        )
    )

    fig.update_layout(
        title="Interactive 2.5D Composite (click legend to toggle layers)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(
            title="Layers (click to toggle)",
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        width=900,
        height=800
    )

    fig.write_html(output_path)

    print(
        "Interactive composite dashboard written to:",
        output_path
    )

    return output_path

def quantitative_evaluation(
    xyz,
    semantic_labels,
    semantic_grid,
    resolution=0.5,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50
):

    grid_height, grid_width = semantic_grid.shape

    grid_x = ((xyz[:, 0] - x_min) / resolution).astype(int)
    grid_y = ((xyz[:, 1] - y_min) / resolution).astype(int)

    in_bounds = (
        (grid_x >= 0) & (grid_x < grid_width) &
        (grid_y >= 0) & (grid_y < grid_height)
    )

    # --------------------------------------------------------
    # Ground truth = each point's own label.
    # Prediction   = the grid cell's dominant label, looked up
    # per point. Out-of-bounds points are dropped (no cell to
    # compare against).
    # --------------------------------------------------------

    true_labels = semantic_labels[in_bounds]

    predicted_labels = semantic_grid[
        grid_y[in_bounds],
        grid_x[in_bounds]
    ]

    def labels_to_occupancy(labels):

        occ = np.full(len(labels), -1, dtype=np.int8)

        obstacle_mask = np.isin(labels, list(OBSTACLE_CLASSES))
        traversable_mask = np.isin(labels, list(TRAVERSABLE_CLASSES))

        occ[obstacle_mask] = 1
        occ[traversable_mask] = 0

        return occ

    ground_truth_occupancy = labels_to_occupancy(true_labels)
    predicted_occupancy = labels_to_occupancy(predicted_labels)

    valid = (
        (ground_truth_occupancy != -1) &
        (predicted_occupancy != -1)
    )

    y_true = ground_truth_occupancy[valid]
    y_pred = predicted_occupancy[valid]

    # --------------------------------------------------------
    # Occupancy Metrics
    # --------------------------------------------------------

    if len(y_true) > 0:

        occupancy_accuracy = accuracy_score(y_true, y_pred)
        occupancy_precision = precision_score(y_true, y_pred, zero_division=0)
        occupancy_recall = recall_score(y_true, y_pred, zero_division=0)
        occupancy_f1 = f1_score(y_true, y_pred, zero_division=0)
        occupancy_iou = jaccard_score(y_true, y_pred, zero_division=0)
        occupancy_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    else:

        occupancy_accuracy = occupancy_precision = occupancy_recall = 0.0
        occupancy_f1 = occupancy_iou = 0.0
        occupancy_cm = np.zeros((2, 2), dtype=int)

    # --------------------------------------------------------
    # Traversability Metrics (same true-point vs. predicted-cell
    # comparison, re-mapped through the traversability rule)
    # --------------------------------------------------------

    def labels_to_traversability(labels):

        trav = np.full(len(labels), -1, dtype=np.int8)

        traversable_mask = np.isin(labels, list(TRAVERSABLE_CLASSES))
        obstacle_mask = np.isin(labels, list(OBSTACLE_CLASSES))

        trav[traversable_mask] = 1
        trav[obstacle_mask] = 0

        return trav

    ground_truth_traversability = labels_to_traversability(true_labels)
    predicted_traversability = labels_to_traversability(predicted_labels)

    valid_traversability = (
        (ground_truth_traversability != -1) &
        (predicted_traversability != -1)
    )

    true_traversability = ground_truth_traversability[valid_traversability]
    pred_traversability = predicted_traversability[valid_traversability]

    if len(true_traversability) > 0:

        traversability_accuracy = accuracy_score(
            true_traversability, pred_traversability
        )
        traversability_precision = precision_score(
            true_traversability, pred_traversability, zero_division=0
        )
        traversability_recall = recall_score(
            true_traversability, pred_traversability, zero_division=0
        )
        traversability_f1 = f1_score(
            true_traversability, pred_traversability, zero_division=0
        )
        traversability_iou = jaccard_score(
            true_traversability, pred_traversability, zero_division=0
        )

    else:

        traversability_accuracy = traversability_precision = 0.0
        traversability_recall = traversability_f1 = traversability_iou = 0.0

    # --------------------------------------------------------
    # Terrain Statistics
    # --------------------------------------------------------

    z = xyz[:, 2]

    ground_mask = np.isin(
        semantic_labels,
        list(TRAVERSABLE_CLASSES)
    )

    ground_z = z[ground_mask]

    if len(ground_z) > 0:

        mean_ground_height = np.mean(ground_z)
        std_ground_height = np.std(ground_z)

    else:

        mean_ground_height = 0
        std_ground_height = 0

    # --------------------------------------------------------
    # Semantic Statistics
    # --------------------------------------------------------

    unique_classes, class_counts = np.unique(
        semantic_labels,
        return_counts=True
    )

    semantic_statistics = dict(
        zip(
            unique_classes.tolist(),
            class_counts.tolist()
        )
    )

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print("\n")
    print("================================================")
    print("     OCCUPANCY EVALUATION (grid discretization")
    print("     fidelity - point label vs. cell majority")
    print("     vote; NOT an independent benchmark)")
    print("================================================")

    print("Accuracy :", round(occupancy_accuracy, 4))
    print("Precision:", round(occupancy_precision, 4))
    print("Recall   :", round(occupancy_recall, 4))
    print("F1-score :", round(occupancy_f1, 4))
    print("IoU      :", round(occupancy_iou, 4))

    print("\nConfusion Matrix")
    print(occupancy_cm)

    print("\n")
    print("================================================")
    print("     TRAVERSABILITY EVALUATION (same caveat)")
    print("================================================")

    print("Accuracy :", round(traversability_accuracy, 4))
    print("Precision:", round(traversability_precision, 4))
    print("Recall   :", round(traversability_recall, 4))
    print("F1-score :", round(traversability_f1, 4))
    print("IoU      :", round(traversability_iou, 4))

    print("\n")
    print("================================================")
    print("             TERRAIN STATISTICS")
    print("================================================")

    print("Ground points:", len(ground_z))
    print("Mean ground elevation:", round(mean_ground_height, 3), "m")
    print(
        "Elevation standard deviation:",
        round(std_ground_height, 3),
        "m"
    )

    print("\n")
    print("================================================")
    print("             SEMANTIC STATISTICS")
    print("================================================")

    for label, count in semantic_statistics.items():

        print(
            CLASS_NAMES.get(label, "Unknown"),
            ":",
            count,
            "points"
        )

    print("\n")
    print("================================================")
    print("           STAGE 14 COMPLETE")
    print("================================================")

    return {

        "occupancy": {
            "accuracy": occupancy_accuracy,
            "precision": occupancy_precision,
            "recall": occupancy_recall,
            "f1": occupancy_f1,
            "iou": occupancy_iou,
            "confusion_matrix": occupancy_cm
        },

        "traversability": {
            "accuracy": traversability_accuracy,
            "precision": traversability_precision,
            "recall": traversability_recall,
            "f1": traversability_f1,
            "iou": traversability_iou
        },

        "terrain": {
            "ground_points": len(ground_z),
            "mean_ground_height": mean_ground_height,
            "std_ground_height": std_ground_height
        },

        "semantic_statistics": semantic_statistics
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("================================================")
    print("       LIDAR 2.5D MAPPING PIPELINE")
    print("================================================")

    # ========================================================
    # STAGE 1
    # ========================================================

    print("\n[STAGE 1] Loading LiDAR...")

    points, xyz, intensity = load_pointcloud(BIN_FILE)

    # ========================================================
    # STAGE 2
    # ========================================================

    print("\n[STAGE 2] Creating BEV...")

    bev = create_bev(xyz, resolution=0.1)

    # ========================================================
    # STAGE 3
    # ========================================================

    print("\n[STAGE 3] Creating elevation grid...")

    elevation_grid = create_elevation_grid(
        xyz,
        resolution=RESOLUTION
    )

    ground_points, non_ground_points = ground_detection(xyz)

    slope, slope_degrees, terrain_unknown_mask = terrain_analysis(
        elevation_grid,
        RESOLUTION
    )

    # ========================================================
    # STAGE 4
    # ========================================================

    print("\n[STAGE 4] Creating initial adaptive resolution...")

    resolution_map, coarse_elevation, coarse_slope = adaptive_resolution(
        xyz,
        coarse_resolution=1.0
    )

    # ========================================================
    # STAGE 5
    # ========================================================

    print("\n[STAGE 5] Loading SemanticKITTI labels...")

    semantic_labels = load_semantic_labels(
        LABEL_FILE,
        len(points)
    )

    semantic_colors = create_semantic_colors(semantic_labels)

    # ========================================================
    # STAGE 9
    # ========================================================

    print("\n[STAGE 9] Creating semantic 2.5D grid...")

    (
        elevation_grid,
        semantic_grid,
        point_count_grid
    ) = create_semantic_grid(

        xyz,
        semantic_labels,
        resolution=RESOLUTION,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX
    )

    # ========================================================
    # STAGE 10
    # ========================================================

    print("\n[STAGE 10] Creating semantic-aware variable grid...")

    adaptive_cells, point_resolutions = semantic_aware_variable_grid(

        xyz,
        semantic_labels,
        X_MIN,
        X_MAX,
        Y_MIN,
        Y_MAX
    )

    print("Adaptive cells:", len(adaptive_cells))

    # ========================================================
    # STAGE 10 - VARIABLE OCCUPANCY
    # ========================================================

    print("\n[STAGE 10] Creating variable-resolution occupancy...")

    variable_occupancy = variable_resolution_occupancy(adaptive_cells)

    # ========================================================
    # STAGE 10 - VARIABLE TRAVERSABILITY
    # ========================================================

    print("\n[STAGE 10] Creating variable-resolution traversability...")

    variable_occupancy = variable_resolution_traversability(

        variable_occupancy,
        slope_degrees,
        resolution=RESOLUTION,
        x_min=X_MIN,
        y_min=Y_MIN,
        max_slope=MAX_SLOPE
    )

    # ========================================================
    # VARIABLE GRID STATISTICS
    # ========================================================

    variable_grid_statistics(variable_occupancy)

    # ========================================================
    # STAGE 8b
    # PROBABILISTIC RAYCAST OCCUPANCY  (fix 4)
    # ========================================================

    print(
        "\n[STAGE 8b] Building probabilistic raycast occupancy "
        "grid..."
    )

    occupancy_grid, occupancy_probability = probabilistic_occupancy_mapping(

        xyz,
        resolution=RESOLUTION,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX
    )

    unknown_fraction = np.mean(occupancy_grid == -1)

    print(
        "Occupancy grid unknown fraction:",
        round(float(unknown_fraction) * 100, 1),
        "%"
    )

    # ========================================================
    # STAGE 12
    # FIXED GRID TRAVERSABILITY
    # ========================================================

    print("\n[STAGE 12] Calculating traversability...")

    traversability_grid, slope_degrees = traversability_analysis(

        elevation_grid,
        semantic_grid,
        resolution=RESOLUTION,
        max_slope=MAX_SLOPE
    )

    # ========================================================
    # STAGE 13
    # ========================================================

    print("\n[STAGE 13] Opening visualization dashboard...")

    visualization_dashboard(

        xyz,
        elevation_grid,
        semantic_grid,
        point_count_grid,
        occupancy_grid,
        slope_degrees,
        traversability_grid,
        resolution=RESOLUTION,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        max_slope=MAX_SLOPE
    )

    # ========================================================
    # STAGE 13b
    # INTERACTIVE VIEWERS
    # ========================================================

    print("\n[STAGE 13b] Building interactive 2.5D composite...")

    build_interactive_composite(

        occupancy_grid,
        traversability_grid,
        semantic_grid,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        resolution=RESOLUTION,
        output_path="interactive_dashboard.html"
    )

    print(
        "\n[STAGE 13b] Launching interactive 3D point-cloud viewer "
        "(close the window to continue)..."
    )

    # Uncomment to pop up the Open3D interactive 3D viewer.
    # It blocks until the window is closed, so it is left
    # opt-in rather than run automatically every pipeline pass.
    # launch_open3d_viewer(xyz, semantic_labels)

    # ========================================================
    # STAGE 14
    # ========================================================

    print("\n[STAGE 14] Running quantitative evaluation...")

    evaluation_results = quantitative_evaluation(

        xyz,
        semantic_labels,
        semantic_grid,
        resolution=RESOLUTION,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("================================================")
    print("          PIPELINE COMPLETE")
    print("================================================")

    print("LiDAR points:", len(points))
    print("Adaptive cells:", len(adaptive_cells))

    print(
        "Variable occupied cells:",
        sum(cell["occupancy"] == 1 for cell in variable_occupancy)
    )

    print(
        "Variable free cells:",
        sum(cell["occupancy"] == 0 for cell in variable_occupancy)
    )

    print(
        "Variable traversable cells:",
        sum(cell["traversability"] == 1 for cell in variable_occupancy)
    )

    print(
        "Grid discretization Occupancy F1:",
        round(evaluation_results["occupancy"]["f1"], 4)
    )

    print(
        "Grid discretization Traversability F1:",
        round(evaluation_results["traversability"]["f1"], 4)
    )

    print("================================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()