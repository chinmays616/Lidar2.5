# ============================================================
# STAGE 12
# TRAVERSABILITY ANALYSIS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


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


# ============================================================
# 4. DEFINE GRID
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

# Maximum elevation in each cell

elevation_grid = np.full(
    (grid_height, grid_width),
    np.nan
)


# Number of points

point_count_grid = np.zeros(
    (grid_height, grid_width),
    dtype=np.int32
)


# Dominant semantic label

semantic_grid = np.full(
    (grid_height, grid_width),
    -1,
    dtype=np.int32
)


# Occupancy
#
# -1 = unknown
#  0 = free
#  1 = occupied

occupancy_grid = np.full(
    (grid_height, grid_width),
    -1,
    dtype=np.int8
)


# ============================================================
# 6. DEFINE SEMANTIC CLASSES
# ============================================================

# Traversable surfaces

TRAVERSABLE_CLASSES = {

    40,  # road
    44,  # parking
    48,  # sidewalk
    49,  # other-ground
    72   # terrain
}


# Obstacles

OBSTACLE_CLASSES = {

    10,  # car
    11,  # bicycle
    13,  # bus
    15,  # motorcycle
    16,  # on-rails
    18,  # truck
    20,  # other vehicle

    30,  # person
    31,  # bicyclist
    32,  # motorcyclist

    50,  # building
    51,  # fence
    52,  # structure

    70,  # vegetation
    71,  # trunk

    80,  # pole
    81   # traffic sign
}


# ============================================================
# 7. EXTRACT XYZ
# ============================================================

x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]


# ============================================================
# 8. CONVERT POINTS TO GRID INDICES
# ============================================================

grid_x = (
    (x - x_min) / resolution
).astype(int)

grid_y = (
    (y - y_min) / resolution
).astype(int)


# ============================================================
# 9. KEEP ONLY POINTS INSIDE GRID
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
# 10. BUILD ELEVATION GRID
# ============================================================

for i in range(len(z_valid)):

    gx = grid_x[i]
    gy = grid_y[i]

    current_z = z_valid[i]

    if np.isnan(
        elevation_grid[gy, gx]
    ):

        elevation_grid[gy, gx] = current_z

    else:

        elevation_grid[gy, gx] = max(
            elevation_grid[gy, gx],
            current_z
        )


# ============================================================
# 11. COUNT POINTS
# ============================================================

for i in range(len(z_valid)):

    gx = grid_x[i]
    gy = grid_y[i]

    point_count_grid[gy, gx] += 1


# ============================================================
# 12. BUILD SEMANTIC GRID
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


        semantic_grid[gy, gx] = (
            dominant_label
        )


# ============================================================
# 13. BUILD OCCUPANCY GRID
# ============================================================

for gy in range(grid_height):

    for gx in range(grid_width):

        label = semantic_grid[gy, gx]


        # No points

        if label == -1:

            occupancy_grid[gy, gx] = -1


        # Obstacle

        elif label in OBSTACLE_CLASSES:

            occupancy_grid[gy, gx] = 1


        # Traversable

        elif label in TRAVERSABLE_CLASSES:

            occupancy_grid[gy, gx] = 0


        # Unknown semantic class

        else:

            occupancy_grid[gy, gx] = -1


# ============================================================
# 14. CALCULATE SLOPE
# ============================================================

# Fill missing elevation temporarily
# using nearest available value through
# simple interpolation-like filling.

elevation_for_slope = elevation_grid.copy()


# Replace NaN with the mean elevation
# for numerical calculation.

mean_elevation = np.nanmean(
    elevation_for_slope
)

elevation_for_slope[
    np.isnan(elevation_for_slope)
] = mean_elevation


# Calculate gradients

gradient_y, gradient_x = np.gradient(
    elevation_for_slope,
    resolution
)


# Calculate slope magnitude

slope_radians = np.arctan(
    np.sqrt(
        gradient_x**2 +
        gradient_y**2
    )
)


# Convert radians → degrees

slope_degrees = np.degrees(
    slope_radians
)


# ============================================================
# 15. DEFINE MAXIMUM ACCEPTABLE SLOPE
# ============================================================

MAX_SLOPE = 15.0


# ============================================================
# 16. CREATE TRAVERSABILITY GRID
# ============================================================

# Values:
#
# 0 = NOT TRAVERSABLE
# 1 = TRAVERSABLE
# 2 = UNKNOWN

traversability_grid = np.full(
    (grid_height, grid_width),
    2,
    dtype=np.int8
)


# ============================================================
# 17. APPLY SEMANTIC CONDITION
# ============================================================

for gy in range(grid_height):

    for gx in range(grid_width):

        label = semantic_grid[gy, gx]


        # Unknown cell

        if label == -1:

            traversability_grid[
                gy, gx
            ] = 2

            continue


        # Obstacle

        if label in OBSTACLE_CLASSES:

            traversability_grid[
                gy, gx
            ] = 0

            continue


        # Non-traversable semantic class

        if label not in TRAVERSABLE_CLASSES:

            traversability_grid[
                gy, gx
            ] = 0

            continue


        # ====================================================
        # SLOPE CONDITION
        # ====================================================

        if slope_degrees[gy, gx] > MAX_SLOPE:

            traversability_grid[
                gy, gx
            ] = 0

        else:

            traversability_grid[
                gy, gx
            ] = 1


# ============================================================
# 18. COUNT TRAVERSABILITY STATES
# ============================================================

traversable = np.sum(
    traversability_grid == 1
)

not_traversable = np.sum(
    traversability_grid == 0
)

unknown = np.sum(
    traversability_grid == 2
)


print("\n===================================")
print("TRAVERSABILITY ANALYSIS")
print("===================================")

print(
    "Traversable cells:",
    traversable
)

print(
    "Non-traversable cells:",
    not_traversable
)

print(
    "Unknown cells:",
    unknown
)

print(
    "Maximum allowed slope:",
    MAX_SLOPE,
    "degrees"
)


# ============================================================
# 19. VISUALIZE SLOPE
# ============================================================

plt.figure(
    figsize=(10, 10)
)


plt.imshow(

    slope_degrees,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)


plt.colorbar(
    label="Slope (degrees)"
)


plt.xlabel(
    "X (meters)"
)

plt.ylabel(
    "Y (meters)"
)

plt.title(
    "Terrain Slope"
)

plt.show()


# ============================================================
# 20. VISUALIZE TRAVERSABILITY
# ============================================================

plt.figure(
    figsize=(10, 10)
)


plt.imshow(

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


plt.colorbar(
    label="Traversability"
)


plt.xlabel(
    "X (meters)"
)

plt.ylabel(
    "Y (meters)"
)

plt.title(
    "Traversability Map"
)

plt.show()


# ============================================================
# 21. VISUALIZE OCCUPANCY
# ============================================================

plt.figure(
    figsize=(10, 10)
)


plt.imshow(

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


plt.colorbar(
    label="Occupancy"
)


plt.xlabel(
    "X (meters)"
)

plt.ylabel(
    "Y (meters)"
)

plt.title(
    "Occupancy Map"
)

plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("STAGE 12 COMPLETE")
print("===================================")

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
    "Maximum slope:",
    MAX_SLOPE,
    "degrees"
)

print("===================================")