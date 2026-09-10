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
    r"\velodyne\000005.bin"
)

LABEL_FILE = (
    r"semantic_kitti_sample\dataset\sequences\00"
    r"\labels\000005.label"
)

RESOLUTION = 0.5

X_MIN = -50
X_MAX = 50

Y_MIN = -50
Y_MAX = 50

MAX_SLOPE = 15.0
SLOPE_DISPLAY_MAX = 45.0

MAX_INTERP_DISTANCE = 3.0
INTERP_K_NEIGHBORS = 8

SENSOR_ORIGIN = (0.0, 0.0)
P_OCC = 0.7           # inverse sensor model: P(occupied | hit)
P_FREE = 0.3          # inverse sensor model: P(occupied | ray passed through)
P_PRIOR = 0.5
LOG_ODDS_CLAMP = 10.0
OCC_PROB_THRESHOLD = 0.65
FREE_PROB_THRESHOLD = 0.35

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


FORCE_NON_TRAVERSABLE_CLASSES = {
    50,   # building
    51,   # fence
    70,   # vegetation
    71,   # trunk
    80    # pole
}

FORCE_TRAVERSABLE_CHECK_CLASSES = {
    40,   # road
    44,   # parking
    48    # sidewalk
}


# ============================================================
# COSTMAP / PATH PLANNING CONFIG (fix 4)
# ============================================================

START_COORD = (0.0, 0.0)     # sensor origin, world (x, y) meters
GOAL_COORD = (20.0, 15.0)    # example goal, world (x, y) meters - edit me

# How the cost map treats traversability_grid == 2 ("unknown" - no
# semantic label, or terrain too far from a real elevation sample).
# The old pipeline always hard-blocked unknown cells with cost=inf.
# For ONE real LiDAR frame that's a trap: large parts of a 100x100 m
# grid are never observed at all, so unknown patches routinely sit
# right across the straight line between START_COORD and GOAL_COORD -
# even though a real robot could just drive across unmapped-but-
# probably-fine ground. "penalize" keeps A* able to route through
# unknown territory when there's no better way, at a real cost, and
# still strictly prefers confirmed-traversable ground when available.
UNKNOWN_TRAVERSAL_POLICY = "penalize"   # "block" | "penalize" | "optimistic"
UNKNOWN_PENALTY_MULTIPLIER = 6.0        # only used when policy == "penalize"

# If the exact START_COORD / GOAL_COORD grid cell isn't passable
# (common: the sensor-origin cell mixes ground + ego-vehicle points,
# or your hand-picked GOAL_COORD happens to land on an unlabeled/
# obstacle cell), plan_path() searches outward up to this many grid
# cells for the nearest passable one instead of just giving up.
MAX_SNAP_RADIUS_CELLS = 25

SEMANTIC_RISK_WEIGHTS = {
    40: 1.0,   # road - lowest cost, preferred route
    44: 1.2,   # parking
    48: 1.5,   # sidewalk
    49: 1.8,   # other-ground
    72: 2.0,   # terrain - passable but costliest ground type
}


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
   

    sorted_labels = sorted(present_labels)

    colors = [
        CLASS_COLORS.get(label, [0.5, 0.5, 0.5])
        for label in sorted_labels
    ]

    cmap = mcolors.ListedColormap(colors)
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
    unknown_mask = np.isnan(filled_grid) | np.isnan(slope_degrees)

    slope = np.where(unknown_mask, np.nan, slope)
    slope_degrees = np.where(unknown_mask, np.nan, slope_degrees)

    return slope, slope_degrees, unknown_mask

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
        label = cell["semantic_label"]

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

        # fix 3: explicit override - always block these classes
        # regardless of local slope.
        if label in FORCE_NON_TRAVERSABLE_CLASSES:

            cell["traversability"] = 0

            continue

        # fix 3: explicit override - road/parking/sidewalk graded
        # purely on their (per-class) slope tolerance.
        if label in FORCE_TRAVERSABLE_CHECK_CLASSES:

            if np.isnan(local_slope):

                cell["traversability"] = 2

            elif local_slope <= get_class_max_slope(label, max_slope):

                cell["traversability"] = 1

            else:

                cell["traversability"] = 0

            continue

        # Obstacle
        if cell["occupancy"] == 1:

            cell["traversability"] = 0

        # Unknown occupancy
        elif cell["occupancy"] == -1:

            cell["traversability"] = 2

        # Unknown terrain (outside the bounded slope interpolation)
        elif np.isnan(local_slope):

            cell["traversability"] = 2

        # Excessive slope (semantic-aware: terrain/other-ground get
        # their own tolerance instead of one global cutoff - see
        # CLASS_MAX_SLOPE)
        elif local_slope > get_class_max_slope(
            label, max_slope
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
    """
    Computes a 2D traversability grid using semantic labels and local terrain slope.
    Returns both the traversability grid and slope_degrees grid for downstream processing/viz.
    """
    slope, slope_degrees, terrain_unknown = terrain_analysis(
        elevation_grid,
        resolution,
        smoothing_sigma=smoothing_sigma,
        max_interp_distance=max_interp_distance,
        k_neighbors=k_neighbors
    )

    traversability_grid = np.full(semantic_grid.shape, 2, dtype=np.int8)
    known_mask = (semantic_grid != -1)

    # 1. Force Non-Traversable Classes
    force_non_trav_mask = np.isin(semantic_grid, list(FORCE_NON_TRAVERSABLE_CLASSES))
    traversability_grid[force_non_trav_mask] = 0

    # 2. General Obstacle Classes
    obstacle_mask = np.isin(semantic_grid, list(OBSTACLE_CLASSES)) & ~force_non_trav_mask
    traversability_grid[obstacle_mask] = 0

    # 3. Process Traversable Ground Classes
    max_slope_grid = np.full(semantic_grid.shape, max_slope, dtype=np.float32)
    for cls_id, cls_max_s in CLASS_MAX_SLOPE.items():
        max_slope_grid[semantic_grid == cls_id] = cls_max_s

    is_traversable_class = np.isin(semantic_grid, list(TRAVERSABLE_CLASSES))
    valid_terrain_mask = known_mask & is_traversable_class & ~terrain_unknown

    traversability_grid[valid_terrain_mask & (slope_degrees <= max_slope_grid)] = 1
    traversability_grid[valid_terrain_mask & (slope_degrees > max_slope_grid)] = 0

    unclassified_mask = known_mask & ~is_traversable_class & ~force_non_trav_mask & ~obstacle_mask
    traversability_grid[unclassified_mask] = 0

    return traversability_grid, slope_degrees

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
    max_slope=15.0,
    path_world=None
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

    # fix 2: lock spatial aspect ratio so X/Y meters render
    # as a true square regardless of window resizing.
    ax1.set_aspect("equal")

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
    # to a figure-level legend outside the right edge - fix 1)
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

    ax2.set_aspect("equal")

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

    ax3.set_aspect("equal")

    fig.colorbar(
        image3,
        ax=ax3,
        label="Points / Cell"
    )

    axes.append(ax3)
    images.append(image3)
    titles.append("Point Density")

    # --------------------------------------------------------
    # Occupancy  (probabilistic raycast grid; Unknown (-1)
    # masked transparent)
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

    ax4.set_aspect("equal")

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
        f"Terrain Slope (capped at {SLOPE_DISPLAY_MAX:.0f}\u00b0, "
        f"unmapped area masked)"
    )

    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")

    ax5.set_aspect("equal")

    fig.colorbar(
        image5,
        ax=ax5,
        label="Slope (degrees)"
    )

    axes.append(ax5)
    images.append(image5)
    titles.append("Slope")

    # --------------------------------------------------------
    # Traversability  (Unknown (2) masked transparent; planned
    # A* path overlaid here when provided - fix 4)
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

    ax6.set_aspect("equal")

    trav_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#e41a1c", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#4daf4a", edgecolor="black"),
    ]

    trav_legend_labels = ["Not traversable", "Traversable"]

    if path_world:

        path_x = [p[0] for p in path_world]
        path_y = [p[1] for p in path_world]

        ax6.plot(
            path_x, path_y,
            color="blue",
            linewidth=2.5,
            zorder=5
        )

        start_point = ax6.plot(
            path_x[0], path_y[0],
            marker="*", color="cyan", markersize=14,
            markeredgecolor="black", linestyle="None",
            zorder=6
        )[0]

        goal_point = ax6.plot(
            path_x[-1], path_y[-1],
            marker="*", color="magenta", markersize=14,
            markeredgecolor="black", linestyle="None",
            zorder=6
        )[0]

        path_line = plt.Line2D([0], [0], color="blue", linewidth=2.5)

        trav_handles = trav_handles + [path_line, start_point, goal_point]
        trav_legend_labels = trav_legend_labels + [
            "Planned path", "Start", "Goal"
        ]

    ax6.legend(
        trav_handles,
        trav_legend_labels,
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

    # fix 1: semantic legend moved outside the figure's right
    # edge (instead of floating over the bottom subplots).
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        bbox_transform=fig.transFigure,
        ncol=1,
        fontsize=7,
        frameon=True,
        title="Semantic class"
    )

    # fix 1: tighter, explicit margins so the legend (right)
    # and layer-toggle checkboxes (left) both have room and
    # don't overlap the plots.
    plt.tight_layout(
        rect=[0.08, 0.05, 0.85, 0.95]
    )

    # fix 1: checkbox overlay moved up/left so it no longer
    # covers the Elevation Map's Y-axis labels.
    check_ax = fig.add_axes(
        [0.01, 0.75, 0.1, 0.2]
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
    output_path="interactive_dashboard.html",
    path_world=None
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

    if path_world:

        path_x = [p[0] for p in path_world]
        path_y = [p[1] for p in path_world]

        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines+markers",
                line=dict(color="blue", width=3),
                marker=dict(size=3),
                name="Planned path",
                visible=True
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
# STAGE 15
# COST MAP GENERATION (fix 4)
# ============================================================

def build_cost_map(
    traversability_grid,
    slope_degrees,
    semantic_grid,
    max_slope=MAX_SLOPE,
    obstacle_cost=np.inf,
    unknown_policy=UNKNOWN_TRAVERSAL_POLICY,
    unknown_penalty_multiplier=UNKNOWN_PENALTY_MULTIPLIER
):
    """
    Combine the slope grid (0-max_slope degrees, normalized) and
    per-class semantic risk weights into a single navigation cost
    matrix for A* search.

    - Non-traversable cells (traversability_grid == 0)
      -> obstacle_cost (blocked)
    - Unknown cells (traversability_grid == 2)
      -> governed by unknown_policy:
           "block"      cost = inf (old behaviour - safest, but on a
                         single real frame this frequently makes the
                         start or goal itself unreachable)
           "penalize"   cost = unknown_penalty_multiplier (default -
                         passable but expensive, so A* still prefers
                         confirmed ground and only crosses unknown
                         territory when there's no real alternative)
           "optimistic" cost = 1.0 (treated like flat known ground)
    - Traversable cells (traversability_grid == 1)
      -> (1 + normalized_slope) * semantic_risk_weight, so
         steeper ground and riskier-but-still-traversable
         surfaces cost more without being blocked outright

    Vectorized (the previous version looped over every grid cell in
    pure Python - correct, but on a 200x200 grid that's 40,000
    interpreter-level iterations every time you re-plan).
    """

    if unknown_policy == "block":
        unknown_cost = np.inf
    elif unknown_policy == "optimistic":
        unknown_cost = 1.0
    elif unknown_policy == "penalize":
        unknown_cost = unknown_penalty_multiplier
    else:
        raise ValueError(
            f"Unknown unknown_policy {unknown_policy!r}; "
            "expected 'block', 'penalize', or 'optimistic'."
        )

    normalized_slope = np.clip(slope_degrees / max_slope, 0.0, 1.0)
    normalized_slope = np.where(np.isnan(normalized_slope), 1.0, normalized_slope)

    risk_weight = np.full(traversability_grid.shape, 1.5, dtype=np.float64)
    for label, weight in SEMANTIC_RISK_WEIGHTS.items():
        risk_weight = np.where(semantic_grid == label, weight, risk_weight)

    traversable_cost = (1.0 + normalized_slope) * risk_weight

    cost_grid = np.select(
        [traversability_grid == 0, traversability_grid == 2],
        [obstacle_cost, unknown_cost],
        default=traversable_cost,
    )

    return cost_grid.astype(np.float64)


def world_to_grid(x, y, x_min, y_min, resolution):

    gx = int((x - x_min) / resolution)
    gy = int((y - y_min) / resolution)

    return gx, gy


def grid_to_world(gx, gy, x_min, y_min, resolution):

    x = x_min + (gx + 0.5) * resolution
    y = y_min + (gy + 0.5) * resolution

    return x, y


def find_nearest_passable_cell(cost_grid, cell, max_radius=MAX_SNAP_RADIUS_CELLS):
    """
    If `cell` is off-grid or has cost=inf, search outward in growing
    square rings (radius 1, 2, 3, ...) and return the nearest cell
    with finite cost. Returns None if nothing passable is found within
    max_radius cells.

    This exists because START_COORD/GOAL_COORD are picked in world
    coordinates by hand, and it's very easy for the exact grid cell
    they discretize to land on a blocked or unknown cell - e.g. the
    sensor-origin cell often mixes ground returns with the ego
    vehicle's own points, and a hand-picked GOAL_COORD has no
    guarantee of landing on mapped ground. Previously this just made
    a_star_search() print "start/goal cell is not traversable" and
    give up immediately.
    """

    grid_height, grid_width = cost_grid.shape
    cx, cy = cell

    if 0 <= cx < grid_width and 0 <= cy < grid_height and np.isfinite(cost_grid[cy, cx]):
        return cell

    for radius in range(1, max_radius + 1):
        best = None
        best_dist2 = None

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue  # only the outer ring of this radius

                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < grid_width and 0 <= ny < grid_height):
                    continue
                if not np.isfinite(cost_grid[ny, nx]):
                    continue

                dist2 = dx * dx + dy * dy
                if best_dist2 is None or dist2 < best_dist2:
                    best, best_dist2 = (nx, ny), dist2

        if best is not None:
            return best

    return None


# ============================================================
# STAGE 15
# A* GRID SEARCH (fix 4)
# ============================================================

def a_star_search(
    cost_grid,
    start_cell,
    goal_cell
):
    """
    8-connected A* over a (height, width) cost grid. Cells whose
    cost is inf are impassable. start_cell / goal_cell are
    (gx, gy) grid indices. Returns a list of (gx, gy) cells from
    start to goal inclusive, or None if no path exists.
    """

    import heapq

    grid_height, grid_width = cost_grid.shape

    def in_bounds(gx, gy):

        return 0 <= gx < grid_width and 0 <= gy < grid_height

    def passable(gx, gy):

        return np.isfinite(cost_grid[gy, gx])

    sx, sy = start_cell
    gx_goal, gy_goal = goal_cell

    if not in_bounds(sx, sy) or not in_bounds(gx_goal, gy_goal):

        print("A*: start or goal is outside the grid bounds.")

        return None

    if not passable(sx, sy):

        print("A*: start cell is not traversable.")

        return None

    if not passable(gx_goal, gy_goal):

        print("A*: goal cell is not traversable.")

        return None

    # Admissible heuristic scaled by the cheapest real step cost on
    # this grid (Euclidean-distance-only heuristics are still
    # admissible but very loose once real per-cell costs are several
    # times 1.0, so A* ends up exploring far more of the grid than it
    # needs to). Multiplying by the minimum finite cost anywhere on
    # the grid can never overestimate the true remaining cost, so the
    # search result is still optimal - just found faster.
    finite_costs = cost_grid[np.isfinite(cost_grid)]
    min_cost = float(finite_costs.min()) if finite_costs.size else 1.0
    min_cost = max(min_cost, 1e-6)

    def heuristic(gx, gy):

        return math.hypot(gx - gx_goal, gy - gy_goal) * min_cost

    neighbors = [
        (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
        (-1, 0, 1.0),                          (1, 0, 1.0),
        (-1, 1, math.sqrt(2)),  (0, 1, 1.0),   (1, 1, math.sqrt(2)),
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

        if current == (gx_goal, gy_goal):

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

            move_cost = step_cost * cost_grid[ny, nx]

            tentative_g = current_g + move_cost

            neighbor = (nx, ny)

            if tentative_g < g_score.get(neighbor, np.inf):

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(nx, ny)

                heapq.heappush(
                    open_heap,
                    (f_score, tentative_g, neighbor)
                )

    total_passable = int(np.isfinite(cost_grid).sum())

    print("A*: no path found between start and goal.")
    print(
        f"A*: explored {len(visited)} reachable cells from start; "
        f"{total_passable}/{cost_grid.size} cells on the whole grid "
        f"are passable ({100 * total_passable / cost_grid.size:.1f}%). "
        "If that reachable count is much smaller than the passable "
        "count, start and goal sit in two disconnected 'islands' of "
        "traversable ground (e.g. separated by a building or an "
        "unmapped gap) - try UNKNOWN_TRAVERSAL_POLICY = 'penalize' or "
        "'optimistic', or pick a GOAL_COORD closer to visibly mapped "
        "ground in the dashboard."
    )

    return None


def plan_path(
    cost_grid,
    start_world,
    goal_world,
    x_min,
    y_min,
    resolution,
    max_snap_radius=MAX_SNAP_RADIUS_CELLS
):
    """
    Convenience wrapper: converts world-frame start/goal coordinates
    to grid cells, snaps either one onto the nearest passable cell if
    it isn't already passable, runs A*, and converts the resulting
    cell path back to world-frame (x, y) coordinates for plotting.
    """

    start_gx, start_gy = world_to_grid(
        start_world[0], start_world[1], x_min, y_min, resolution
    )

    goal_gx, goal_gy = world_to_grid(
        goal_world[0], goal_world[1], x_min, y_min, resolution
    )

    start_cell = find_nearest_passable_cell(cost_grid, (start_gx, start_gy), max_snap_radius)
    goal_cell = find_nearest_passable_cell(cost_grid, (goal_gx, goal_gy), max_snap_radius)

    if start_cell is None:
        print(
            f"plan_path: no passable cell found within {max_snap_radius} "
            f"grid cells of START {start_world}. Try UNKNOWN_TRAVERSAL_POLICY "
            "= 'penalize'/'optimistic' or increase MAX_SNAP_RADIUS_CELLS."
        )
        return None

    if goal_cell is None:
        print(
            f"plan_path: no passable cell found within {max_snap_radius} "
            f"grid cells of GOAL {goal_world}. Try UNKNOWN_TRAVERSAL_POLICY "
            "= 'penalize'/'optimistic', increase MAX_SNAP_RADIUS_CELLS, or "
            "pick a GOAL_COORD closer to visibly mapped ground."
        )
        return None

    if start_cell != (start_gx, start_gy):
        print(f"plan_path: START snapped from grid {(start_gx, start_gy)} to nearest passable cell {start_cell}.")

    if goal_cell != (goal_gx, goal_gy):
        print(f"plan_path: GOAL snapped from grid {(goal_gx, goal_gy)} to nearest passable cell {goal_cell}.")

    cell_path = a_star_search(cost_grid, start_cell, goal_cell)

    if cell_path is None:
        return None

    world_path = [
        grid_to_world(gx, gy, x_min, y_min, resolution)
        for gx, gy in cell_path
    ]

    return world_path
def create_3d_path(path, elevation_grid):
    path_points = []

    for gx, gy in path:

        x = X_MIN + (gx + 0.5) * RESOLUTION
        y = Y_MIN + (gy + 0.5) * RESOLUTION

        z = elevation_grid[gy, gx]

        if np.isnan(z):
            z = 0.0

        # Lift path slightly above surface
        z += 0.15

        path_points.append([x, y, z])

    path_points = np.asarray(path_points)

    lines = [
        [i, i + 1]
        for i in range(len(path_points) - 1)
    ]

    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(path_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.colors = o3d.utility.Vector3dVector(
        np.tile([[1.0, 0.0, 0.0]], (len(lines), 1))
    )

    return line_set
def create_marker(position, radius=0.5):
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius
    )

    mesh.translate(position)

    return mesh
start_marker = create_marker(
    [START_COORD[0], START_COORD[1], 1.0]
)

goal_marker = create_marker(
    [GOAL_COORD[0], GOAL_COORD[1], 1.0]
)


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
    # STAGE 15
    # COST MAP + A* PATH PLANNING (fix 4)
    # ========================================================

    print("\n[STAGE 15] Building cost map and planning path...")

    cost_grid = build_cost_map(
        traversability_grid,
        slope_degrees,
        semantic_grid,
        max_slope=MAX_SLOPE,
        unknown_policy=UNKNOWN_TRAVERSAL_POLICY
    )

    passable_fraction = float(np.isfinite(cost_grid).mean())

    print(
        f"Cost map: {passable_fraction * 100:.1f}% of cells passable "
        f"(unknown_policy='{UNKNOWN_TRAVERSAL_POLICY}')"
    )

    path_world = plan_path(
        cost_grid,
        START_COORD,
        GOAL_COORD,
        x_min=X_MIN,
        y_min=Y_MIN,
        resolution=RESOLUTION
    )

    if path_world is not None:

        print(
            "Path found:",
            len(path_world),
            "waypoints, from",
            START_COORD,
            "to",
            GOAL_COORD
        )

    else:

        print(
            "No path found from",
            START_COORD,
            "to",
            GOAL_COORD,
            "- try a different GOAL_COORD or check for obstacles."
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
        max_slope=MAX_SLOPE,
        path_world=path_world
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
        output_path="interactive_dashboard.html",
        path_world=path_world
    )

    print(
        "\n[STAGE 13b] Launching interactive 3D point-cloud viewer "
        "(close the window to continue)..."
    )

    # Uncomment to pop up the Open3D interactive 3D viewer.
    # It blocks until the window is closed, so it is left
    # opt-in rather than run automatically every pipeline pass.
    launch_open3d_viewer(xyz, semantic_labels)
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