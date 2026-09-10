# ============================================================
# STAGE 11
# OCCUPANCY AND OBSTACLE MAPPING
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

print("Number of points:", len(xyz))


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
# 5. CREATE OCCUPANCY GRID
# ============================================================

# -1 = UNKNOWN
#  0 = FREE
#  1 = OCCUPIED

occupancy_grid = np.full(
    (grid_height, grid_width),
    -1,
    dtype=np.int8
)


# ============================================================
# 6. DEFINE SEMANTIC CLASSES
# ============================================================

# Classes considered obstacles

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
    52,  # other structure

    70,  # vegetation
    71,  # trunk

    80,  # pole
    81   # traffic sign
}


# Classes considered traversable/free

FREE_CLASSES = {

    40,  # road
    44,  # parking
    48,  # sidewalk
    49,  # other ground
    72   # terrain
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
# 9. REMOVE POINTS OUTSIDE GRID
# ============================================================

valid = (

    (grid_x >= 0) &
    (grid_x < grid_width) &

    (grid_y >= 0) &
    (grid_y < grid_height)

)


grid_x = grid_x[valid]
grid_y = grid_y[valid]

labels_valid = semantic_labels[valid]


# ============================================================
# 10. CLASSIFY EACH POINT
# ============================================================

for i in range(len(grid_x)):

    gx = grid_x[i]
    gy = grid_y[i]

    label = labels_valid[i]


    # --------------------------------------------------------
    # Obstacle
    # --------------------------------------------------------

    if label in OBSTACLE_CLASSES:

        occupancy_grid[gy, gx] = 1


    # --------------------------------------------------------
    # Free space
    # --------------------------------------------------------

    elif label in FREE_CLASSES:

        # Only mark FREE if an obstacle hasn't
        # already been detected in this cell.

        if occupancy_grid[gy, gx] != 1:

            occupancy_grid[gy, gx] = 0


# ============================================================
# 11. COUNT GRID STATES
# ============================================================

unknown_cells = np.sum(
    occupancy_grid == -1
)

free_cells = np.sum(
    occupancy_grid == 0
)

occupied_cells = np.sum(
    occupancy_grid == 1
)


print("\nOccupancy statistics")
print("--------------------")

print(
    "Unknown cells :",
    unknown_cells
)

print(
    "Free cells    :",
    free_cells
)

print(
    "Occupied cells:",
    occupied_cells
)


# ============================================================
# 12. VISUALIZE OCCUPANCY GRID
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
    label="Occupancy State"
)


plt.xlabel(
    "X (meters)"
)

plt.ylabel(
    "Y (meters)"
)

plt.title(
    "LiDAR Occupancy and Obstacle Map"
)

plt.show()


# ============================================================
# 13. CREATE A SIMPLE OBSTACLE MAP
# ============================================================

obstacle_map = (
    occupancy_grid == 1
)


# ============================================================
# 14. VISUALIZE OBSTACLES ONLY
# ============================================================

plt.figure(
    figsize=(10, 10)
)


plt.imshow(

    obstacle_map,

    origin="lower",

    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)


plt.xlabel(
    "X (meters)"
)

plt.ylabel(
    "Y (meters)"
)

plt.title(
    "Detected Obstacles"
)

plt.show()


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("STAGE 11 COMPLETE")
print("===================================")

print(
    "Grid resolution:",
    resolution,
    "m"
)

print(
    "Total grid cells:",
    grid_width * grid_height
)

print(
    "Unknown:",
    unknown_cells
)

print(
    "Free:",
    free_cells
)

print(
    "Occupied:",
    occupied_cells
)

print("===================================")