# ============================================================
# STAGE 10
# SEMANTIC-AWARE VARIABLE-RESOLUTION GRID
# ============================================================
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from stage_9 import MAX_SLOPE

# ============================================================
# 1. FILE PATHS
# ============================================================

bin_file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000009.bin"

label_file = r"semantic_kitti_sample\dataset\sequences\00\labels\000009.label"


# ============================================================
# 2. LOAD LiDAR POINT CLOUD
# ============================================================

points = np.fromfile(
    bin_file,
    dtype=np.float32
).reshape(-1, 4)

xyz = points[:, :3]

print("Number of LiDAR points:", len(points))


# ============================================================
# 3. LOAD SEMANTICKITTI LABELS
# ============================================================

labels = np.fromfile(
    label_file,
    dtype=np.uint32
)

if len(points) != len(labels):
    raise ValueError(
        "Number of points and labels do not match!"
    )

semantic_labels = labels & 0xFFFF

print("Points and labels match!")


# ============================================================
# 4. SEMANTICKITTI CLASS NAMES
# ============================================================

class_names = {

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
    81: "traffic-sign"
}


# ============================================================
# 5. DEFINE SEMANTIC IMPORTANCE
# ============================================================

# Important dynamic/obstacle objects
# will receive finer resolution.

HIGH_IMPORTANCE = {

    10,  # car
    11,  # bicycle
    13,  # bus
    15,  # motorcycle
    18,  # truck
    20,  # other vehicle
    30,  # person
    31,  # bicyclist
    32   # motorcyclist
}


# Medium importance objects

MEDIUM_IMPORTANCE = {

    50,  # building
    51,  # fence
    52,  # structure
    70,  # vegetation
    71,  # trunk
    80,  # pole
    81   # traffic sign
}


# ============================================================
# 6. DEFINE DISTANCE-BASED RESOLUTIONS
# ============================================================

# Distance is measured from the LiDAR sensor.

NEAR_DISTANCE = 20
MEDIUM_DISTANCE = 40
FAR_DISTANCE = 80


# Base resolution according to distance

NEAR_RESOLUTION = 0.25
MEDIUM_RESOLUTION = 0.50
FAR_RESOLUTION = 1.00


# ============================================================
# 7. DEFINE MAP RANGE
# ============================================================

x_min = -50
x_max = 50

y_min = -50
y_max = 50


# ============================================================
# 8. CALCULATE DISTANCE OF EACH POINT
# ============================================================

x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]


# Horizontal distance from LiDAR

distance = np.sqrt(
    x**2 + y**2
)


# ============================================================
# 9. ASSIGN BASE RESOLUTION
# ============================================================

resolution = np.zeros(
    len(points),
    dtype=np.float32
)


# Near

near_mask = distance < NEAR_DISTANCE

resolution[near_mask] = NEAR_RESOLUTION


# Medium

medium_mask = (
    (distance >= NEAR_DISTANCE) &
    (distance < MEDIUM_DISTANCE)
)

resolution[medium_mask] = MEDIUM_RESOLUTION


# Far

far_mask = distance >= MEDIUM_DISTANCE

resolution[far_mask] = FAR_RESOLUTION


# ============================================================
# 10. MAKE IMPORTANT OBJECTS HIGH RESOLUTION
# ============================================================

important_mask = np.isin(
    semantic_labels,
    list(HIGH_IMPORTANCE)
)


# Important objects get fine resolution

resolution[important_mask] = NEAR_RESOLUTION


# ============================================================
# 11. MEDIUM-IMPORTANCE OBJECTS
# ============================================================

medium_object_mask = np.isin(
    semantic_labels,
    list(MEDIUM_IMPORTANCE)
)


# Medium importance objects cannot become
# coarser than 0.5 m.

resolution[
    medium_object_mask &
    (resolution > MEDIUM_RESOLUTION)
] = MEDIUM_RESOLUTION


# ============================================================
# 12. DISPLAY RESOLUTION STATISTICS
# ============================================================

print("\nResolution statistics:")

print(
    "0.25 m cells:",
    np.sum(resolution == 0.25)
)

print(
    "0.50 m cells:",
    np.sum(resolution == 0.50)
)

print(
    "1.00 m cells:",
    np.sum(resolution == 1.00)
)


# ============================================================
# 13. CREATE VARIABLE GRID CELLS
# ============================================================

# Dictionary:
#
# key   = (grid_x, grid_y, resolution)
# value = list of point indices

grid = {}


# ============================================================
# 14. ASSIGN EACH POINT TO A VARIABLE-SIZE CELL
# ============================================================

for i in range(len(points)):

    r = resolution[i]

    gx = int(
        np.floor(
            (x[i] - x_min) / r
        )
    )

    gy = int(
        np.floor(
            (y[i] - y_min) / r
        )
    )

    key = (gx, gy, r)

    if key not in grid:

        grid[key] = []

    grid[key].append(i)


# ============================================================
# 15. CREATE INFORMATION FOR EACH CELL
# ============================================================

adaptive_cells = []


for key, indices in grid.items():

    gx, gy, r = key

    indices = np.array(indices)

    cell_x = x[indices]
    cell_y = y[indices]
    cell_z = z[indices]

    cell_labels = semantic_labels[indices]


    # --------------------------------------------------------
    # Elevation
    # --------------------------------------------------------

    elevation = np.max(cell_z)


    # --------------------------------------------------------
    # Point count
    # --------------------------------------------------------

    point_count = len(indices)


    # --------------------------------------------------------
    # Dominant semantic class
    # --------------------------------------------------------

    unique_labels, counts = np.unique(
        cell_labels,
        return_counts=True
    )

    dominant_label = unique_labels[
        np.argmax(counts)
    ]


    # --------------------------------------------------------
    # Cell center
    # --------------------------------------------------------

    center_x = np.mean(cell_x)
    center_y = np.mean(cell_y)


    # --------------------------------------------------------
    # Store cell information
    # --------------------------------------------------------

    adaptive_cells.append({

        "x": center_x,

        "y": center_y,

        "resolution": r,

        "elevation": elevation,

        "point_count": point_count,

        "semantic_label": int(
            dominant_label
        ),

        "semantic_name":
            class_names.get(
                int(dominant_label),
                "unknown"
            )
    })


# ============================================================
# 16. PRINT GRID INFORMATION
# ============================================================

print("\nAdaptive grid created!")

print(
    "Total adaptive cells:",
    len(adaptive_cells)
)


# ============================================================
# 17. PRINT FIRST 20 CELLS
# ============================================================

print("\nFirst 20 adaptive cells:")

for cell in adaptive_cells[:20]:

    print(

        "Position:",
        (
            round(cell["x"], 2),
            round(cell["y"], 2)
        ),

        "| Resolution:",
        cell["resolution"],

        "| Elevation:",
        round(
            cell["elevation"],
            2
        ),

        "| Points:",
        cell["point_count"],

        "| Semantic:",
        cell["semantic_name"]
    )


# ============================================================
# 18. VISUALIZE VARIABLE RESOLUTION
# ============================================================

plt.figure(
    figsize=(10, 10)
)


# Extract cell information

cell_x = np.array([
    cell["x"]
    for cell in adaptive_cells
])

cell_y = np.array([
    cell["y"]
    for cell in adaptive_cells
])

cell_resolution = np.array([
    cell["resolution"]
    for cell in adaptive_cells
])


# Plot resolution

scatter = plt.scatter(

    cell_x,

    cell_y,

    c=cell_resolution,

    s=8
)


plt.colorbar(
    scatter,
    label="Cell Resolution (m)"
)


plt.xlabel(
    "X (meters)"
)

plt.ylabel(
    "Y (meters)"
)

plt.title(
    "Semantic-Aware Variable Resolution Grid"
)

plt.axis("equal")

plt.show()


# ============================================================
# 19. VISUALIZE SEMANTIC INFORMATION
# ============================================================

plt.figure(
    figsize=(10, 10)
)


semantic_values = np.array([

    cell["semantic_label"]

    for cell in adaptive_cells

])


scatter = plt.scatter(

    cell_x,

    cell_y,

    c=semantic_values,

    s=8
)


plt.colorbar(
    scatter,
    label="Semantic Class ID"
)


plt.xlabel(
    "X (meters)"
)

plt.ylabel(
    "Y (meters)"
)

plt.title(
    "Semantic Information on Adaptive Grid"
)

plt.axis("equal")

plt.show()


# ============================================================
# 20. SUMMARY
# ============================================================

print("\n===================================")

print("STAGE 10 COMPLETE")

print("===================================")

print(
    "Original LiDAR points:",
    len(points)
)

print(
    "Adaptive grid cells:",
    len(adaptive_cells)
)

print(
    "Fine resolution points:",
    np.sum(resolution == 0.25)
)

print(
    "Medium resolution points:",
    np.sum(resolution == 0.50)
)

print(
    "Coarse resolution points:",
    np.sum(resolution == 1.00)
)

print("===================================")
print("[STAGE 10] Creating semantic-aware variable grid...")

adaptive_cells, resolution = semantic_aware_variable_grid(
    xyz,
    semantic_labels
)

print("Adaptive cells:", len(adaptive_cells))

print("[STAGE 10] Creating variable-resolution occupancy...")

variable_occupancy = variable_resolution_occupancy(
    adaptive_cells
)

print("[STAGE 10] Creating variable-resolution traversability...")

variable_occupancy = variable_resolution_traversability(
    variable_occupancy,
    MAX_SLOPE
)

# ===================================================================================================================
# Stage 7
# ===================================================================================================================
def semantic_aware_variable_grid(
    xyz,
    semantic_labels,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50
):
    """
    Create a semantic-aware variable-resolution 2.5D grid.

    Resolution depends on:
    1. Distance from LiDAR
    2. Semantic importance of objects

    Each adaptive cell stores:
    - position
    - resolution
    - elevation
    - point count
    - dominant semantic class
    """

    # ========================================================
    # 1. Semantic importance
    # ========================================================

    HIGH_IMPORTANCE = {
        10, 11, 13, 15, 18, 20,
        30, 31, 32
    }

    MEDIUM_IMPORTANCE = {
        50, 51, 52,
        70, 71,
        80, 81
    }

    # ========================================================
    # 2. Distance-based resolution
    # ========================================================

    NEAR_DISTANCE = 20
    MEDIUM_DISTANCE = 40

    NEAR_RESOLUTION = 0.25
    MEDIUM_RESOLUTION = 0.50
    FAR_RESOLUTION = 1.00

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Horizontal distance from LiDAR
    distance = np.sqrt(x**2 + y**2)

    # ========================================================
    # 3. Assign base resolution
    # ========================================================

    resolution = np.zeros(
        len(xyz),
        dtype=np.float32
    )

    # Near
    near_mask = distance < NEAR_DISTANCE
    resolution[near_mask] = NEAR_RESOLUTION

    # Medium
    medium_mask = (
        (distance >= NEAR_DISTANCE) &
        (distance < MEDIUM_DISTANCE)
    )

    resolution[medium_mask] = MEDIUM_RESOLUTION

    # Far
    far_mask = distance >= MEDIUM_DISTANCE
    resolution[far_mask] = FAR_RESOLUTION

    # ========================================================
    # 4. Semantic-aware refinement
    # ========================================================

    important_mask = np.isin(
        semantic_labels,
        list(HIGH_IMPORTANCE)
    )

    # Important objects always receive fine resolution
    resolution[important_mask] = NEAR_RESOLUTION

    medium_object_mask = np.isin(
        semantic_labels,
        list(MEDIUM_IMPORTANCE)
    )

    # Medium objects cannot become 1 m
    resolution[
        medium_object_mask &
        (resolution > MEDIUM_RESOLUTION)
    ] = MEDIUM_RESOLUTION

    # ========================================================
    # 5. Statistics
    # ========================================================

    print("\nResolution statistics:")

    print(
        "0.25 m points:",
        np.sum(resolution == 0.25)
    )

    print(
        "0.50 m points:",
        np.sum(resolution == 0.50)
    )

    print(
        "1.00 m points:",
        np.sum(resolution == 1.00)
    )

    # ========================================================
    # 6. Create adaptive cells
    # ========================================================

    grid = {}

    for i in range(len(xyz)):

        r = resolution[i]

        gx = int(
            np.floor(
                (x[i] - x_min) / r
            )
        )

        gy = int(
            np.floor(
                (y[i] - y_min) / r
            )
        )

        key = (gx, gy, r)

        if key not in grid:
            grid[key] = []

        grid[key].append(i)

    # ========================================================
    # 7. Semantic class names
    # ========================================================

    class_names = {

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
        81: "traffic-sign"
    }

    # ========================================================
    # 8. Calculate information for each cell
    # ========================================================

    adaptive_cells = []

    for key, indices in grid.items():

        gx, gy, r = key

        indices = np.array(indices)

        cell_x = x[indices]
        cell_y = y[indices]
        cell_z = z[indices]

        cell_labels = semantic_labels[indices]

        # Maximum elevation
        elevation = np.max(cell_z)

        # Number of LiDAR points
        point_count = len(indices)

        # Dominant semantic class
        unique_labels, counts = np.unique(
            cell_labels,
            return_counts=True
        )

        dominant_label = unique_labels[
            np.argmax(counts)
        ]

        # Cell center
        center_x = np.mean(cell_x)
        center_y = np.mean(cell_y)

        adaptive_cells.append({

            "x": center_x,
            "y": center_y,
            "resolution": float(r),
            "elevation": float(elevation),
            "point_count": point_count,
            "semantic_label": int(dominant_label),
            "semantic_name": class_names.get(
                int(dominant_label),
                "unknown"
            )
        })

    # ========================================================
    # 9. Statistics
    # ========================================================

    print("\nAdaptive grid created!")

    print(
        "Total adaptive cells:",
        len(adaptive_cells)
    )

    return adaptive_cells, resolution