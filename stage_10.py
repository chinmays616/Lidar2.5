# ============================================================
# STAGE 13
# LIDAR 2.5D VISUALIZATION DASHBOARD
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


# ============================================================
# 1. FILE PATHS
# ============================================================

bin_file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000009.bin"

label_file = r"semantic_kitti_sample\dataset\sequences\00\labels\000009.label"


# ============================================================
# 2. LOAD LIDAR
# ============================================================

points = np.fromfile(
    bin_file,
    dtype=np.float32
).reshape(-1, 4)

xyz = points[:, :3]

intensity = points[:, 3]


print("LiDAR points:", len(points))


# ============================================================
# 3. LOAD SEMANTIC LABELS
# ============================================================

labels = np.fromfile(
    label_file,
    dtype=np.uint32
)

if len(points) != len(labels):

    raise ValueError(
        "Point and label counts do not match!"
    )


semantic_labels = labels & 0xFFFF


# ============================================================
# 4. GRID PARAMETERS
# ============================================================

resolution = 0.5

x_min = -50
x_max = 50

y_min = -50
y_max = 50


grid_width = int(
    (x_max - x_min) / resolution
)

grid_height = int(
    (y_max - y_min) / resolution
)


# ============================================================
# 5. CREATE GRIDS
# ============================================================

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


# -1 = unknown
#  0 = free
#  1 = occupied

occupancy_grid = np.full(
    (grid_height, grid_width),
    -1,
    dtype=np.int8
)


# 0 = not traversable
# 1 = traversable
# 2 = unknown

traversability_grid = np.full(
    (grid_height, grid_width),
    2,
    dtype=np.int8
)


# ============================================================
# 6. SEMANTIC CLASSES
# ============================================================

TRAVERSABLE_CLASSES = {

    40,   # road
    44,   # parking
    48,   # sidewalk
    49,   # other ground
    72    # terrain
}


OBSTACLE_CLASSES = {

    10,   # car
    11,   # bicycle
    13,   # bus
    15,   # motorcycle
    16,   # on rails
    18,   # truck
    20,   # other vehicle

    30,   # person
    31,   # bicyclist
    32,   # motorcyclist

    50,   # building
    51,   # fence
    52,   # structure

    70,   # vegetation
    71,   # trunk

    80,   # pole
    81    # traffic sign
}


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
# 7. CONVERT XYZ → GRID INDICES
# ============================================================

x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]


grid_x = (
    (x - x_min) / resolution
).astype(int)


grid_y = (
    (y - y_min) / resolution
).astype(int)


# ============================================================
# 8. REMOVE POINTS OUTSIDE GRID
# ============================================================

valid = (

    (grid_x >= 0) &
    (grid_x < grid_width) &

    (grid_y >= 0) &
    (grid_y < grid_height)

)


grid_x = grid_x[valid]
grid_y = grid_y[valid]

z_valid = z[valid]

labels_valid = semantic_labels[valid]


# ============================================================
# 9. BUILD ELEVATION + POINT COUNT
# ============================================================

for i in range(len(z_valid)):

    gx = grid_x[i]
    gy = grid_y[i]

    current_z = z_valid[i]


    # Elevation

    if np.isnan(
        elevation_grid[gy, gx]
    ):

        elevation_grid[gy, gx] = current_z

    else:

        elevation_grid[gy, gx] = max(
            elevation_grid[gy, gx],
            current_z
        )


    # Point count

    point_count_grid[
        gy,
        gx
    ] += 1


# ============================================================
# 10. BUILD SEMANTIC GRID
# ============================================================

for gy in range(grid_height):

    for gx in range(grid_width):

        mask = (

            (grid_x == gx) &
            (grid_y == gy)

        )


        if not np.any(mask):

            continue


        cell_labels = labels_valid[mask]


        unique_labels, counts = np.unique(
            cell_labels,
            return_counts=True
        )


        dominant_label = unique_labels[
            np.argmax(counts)
        ]


        semantic_grid[
            gy,
            gx
        ] = dominant_label


# ============================================================
# 11. BUILD OCCUPANCY GRID
# ============================================================

for gy in range(grid_height):

    for gx in range(grid_width):

        label = semantic_grid[
            gy,
            gx
        ]


        if label == -1:

            occupancy_grid[
                gy,
                gx
            ] = -1


        elif label in OBSTACLE_CLASSES:

            occupancy_grid[
                gy,
                gx
            ] = 1


        elif label in TRAVERSABLE_CLASSES:

            occupancy_grid[
                gy,
                gx
            ] = 0


        else:

            occupancy_grid[
                gy,
                gx
            ] = -1


# ============================================================
# 12. CALCULATE SLOPE
# ============================================================

elevation_for_slope = elevation_grid.copy()


mean_elevation = np.nanmean(
    elevation_for_slope
)


elevation_for_slope[
    np.isnan(elevation_for_slope)
] = mean_elevation


gradient_y, gradient_x = np.gradient(
    elevation_for_slope,
    resolution
)


slope_radians = np.arctan(
    np.sqrt(
        gradient_x**2 +
        gradient_y**2
    )
)


slope_degrees = np.degrees(
    slope_radians
)


# ============================================================
# 13. TRAVERSABILITY
# ============================================================

MAX_SLOPE = 15.0


for gy in range(grid_height):

    for gx in range(grid_width):

        label = semantic_grid[
            gy,
            gx
        ]


        # Unknown

        if label == -1:

            traversability_grid[
                gy,
                gx
            ] = 2

            continue


        # Obstacle

        if label in OBSTACLE_CLASSES:

            traversability_grid[
                gy,
                gx
            ] = 0

            continue


        # Non-traversable semantic class

        if label not in TRAVERSABLE_CLASSES:

            traversability_grid[
                gy,
                gx
            ] = 0

            continue


        # Slope

        if slope_degrees[
            gy,
            gx
        ] > MAX_SLOPE:

            traversability_grid[
                gy,
                gx
            ] = 0

        else:

            traversability_grid[
                gy,
                gx
            ] = 1


# ============================================================
# 14. CREATE HEIGHT COLORS FOR 3D CLOUD
# ============================================================

z_min = np.min(xyz[:, 2])
z_max = np.max(xyz[:, 2])


normalized_z = (
    xyz[:, 2] - z_min
) / (
    z_max - z_min + 1e-8
)


height_colors = np.zeros(
    (len(xyz), 3)
)


height_colors[:, 0] = normalized_z

height_colors[:, 1] = (
    1 - normalized_z
)

height_colors[:, 2] = 0.5


# ============================================================
# 15. CREATE OPEN3D POINT CLOUD
# ============================================================

pcd = o3d.geometry.PointCloud()


pcd.points = o3d.utility.Vector3dVector(
    xyz
)


pcd.colors = o3d.utility.Vector3dVector(
    height_colors
)


# ============================================================
# 16. DASHBOARD
# ============================================================

fig = plt.figure(
    figsize=(16, 12)
)


# ============================================================
# 17. ELEVATION
# ============================================================

ax1 = fig.add_subplot(
    2,
    3,
    1
)


image1 = ax1.imshow(

    elevation_grid,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)


ax1.set_title(
    "Elevation Map"
)

ax1.set_xlabel(
    "X (m)"
)

ax1.set_ylabel(
    "Y (m)"
)


fig.colorbar(
    image1,
    ax=ax1,
    label="Elevation (m)"
)


# ============================================================
# 18. SEMANTIC MAP
# ============================================================

ax2 = fig.add_subplot(
    2,
    3,
    2
)


image2 = ax2.imshow(

    semantic_grid,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)


ax2.set_title(
    "Semantic Map"
)

ax2.set_xlabel(
    "X (m)"
)

ax2.set_ylabel(
    "Y (m)"
)


fig.colorbar(
    image2,
    ax=ax2,
    label="Semantic Class ID"
)


# ============================================================
# 19. POINT DENSITY
# ============================================================

ax3 = fig.add_subplot(
    2,
    3,
    3
)


image3 = ax3.imshow(

    point_count_grid,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)


ax3.set_title(
    "LiDAR Point Density"
)

ax3.set_xlabel(
    "X (m)"
)

ax3.set_ylabel(
    "Y (m)"
)


fig.colorbar(
    image3,
    ax=ax3,
    label="Points / Cell"
)


# ============================================================
# 20. OCCUPANCY
# ============================================================

ax4 = fig.add_subplot(
    2,
    3,
    4
)


image4 = ax4.imshow(

    occupancy_grid,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ],

    vmin=-1,
    vmax=1
)


ax4.set_title(
    "Occupancy Map"
)

ax4.set_xlabel(
    "X (m)"
)

ax4.set_ylabel(
    "Y (m)"
)


fig.colorbar(
    image4,
    ax=ax4,
    label="-1 Unknown / 0 Free / 1 Occupied"
)


# ============================================================
# 21. SLOPE
# ============================================================

ax5 = fig.add_subplot(
    2,
    3,
    5
)


image5 = ax5.imshow(

    slope_degrees,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)


ax5.set_title(
    "Terrain Slope"
)

ax5.set_xlabel(
    "X (m)"
)

ax5.set_ylabel(
    "Y (m)"
)


fig.colorbar(
    image5,
    ax=ax5,
    label="Slope (degrees)"
)


# ============================================================
# 22. TRAVERSABILITY
# ============================================================

ax6 = fig.add_subplot(
    2,
    3,
    6
)


image6 = ax6.imshow(

    traversability_grid,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ],

    vmin=0,
    vmax=2
)


ax6.set_title(
    "Traversability Map"
)

ax6.set_xlabel(
    "X (m)"
)

ax6.set_ylabel(
    "Y (m)"
)


fig.colorbar(

    image6,

    ax=ax6,

    label="0 No / 1 Yes / 2 Unknown"
)


# ============================================================
# 23. DASHBOARD TITLE
# ============================================================

fig.suptitle(

    "LiDAR 2.5D Semantic Mapping Dashboard",

    fontsize=18
)


plt.tight_layout()

plt.show()


# ============================================================
# 24. PRINT DASHBOARD STATISTICS
# ============================================================

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
    len(points)
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
    "Maximum allowed slope:",
    MAX_SLOPE,
    "degrees"
)

print("================================================")