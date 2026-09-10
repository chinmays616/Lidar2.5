# =============================================================
# Stage 9 Code
# =============================================================
import numpy as np
import open3d as o3d
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

intensity = points[:, 3]

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

print("Points and labels match!")


# ============================================================
# 4. EXTRACT SEMANTIC LABEL
# ============================================================

semantic_labels = labels & 0xFFFF


# ============================================================
# 5. SEMANTICKITTI CLASS NAMES
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
# 6. DEFINE GRID PARAMETERS
# ============================================================

# Grid resolution
# Each cell represents 0.5m x 0.5m

resolution = 0.5


# X-Y range of the LiDAR map

x_min = -50
x_max = 50

y_min = -50
y_max = 50


# Number of grid cells

grid_width = int(
    (x_max - x_min) / resolution
)

grid_height = int(
    (y_max - y_min) / resolution
)


print("\nGrid size:")
print("Width :", grid_width)
print("Height:", grid_height)


# ============================================================
# 7. CREATE GRID ARRAYS
# ============================================================

# Maximum elevation in each cell

elevation_grid = np.full(
    (grid_height, grid_width),
    np.nan
)


# Semantic label for each cell

semantic_grid = np.full(
    (grid_height, grid_width),
    -1,
    dtype=np.int32
)


# Number of points inside each cell

point_count_grid = np.zeros(
    (grid_height, grid_width),
    dtype=np.int32
)


# ============================================================
# 8. CONVERT X,Y COORDINATES INTO GRID INDICES
# ============================================================

x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]


# Convert real-world coordinates to grid coordinates

grid_x = ((x - x_min) / resolution).astype(int)

grid_y = ((y - y_min) / resolution).astype(int)


# ============================================================
# 9. REMOVE POINTS OUTSIDE OUR GRID
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


print("\nPoints inside grid:", len(z_valid))


# ============================================================
# 10. BUILD ELEVATION GRID
# ============================================================

for i in range(len(z_valid)):

    gx = grid_x[i]
    gy = grid_y[i]

    current_z = z_valid[i]

    # Store the maximum height
    # present inside this cell

    if np.isnan(elevation_grid[gy, gx]):

        elevation_grid[gy, gx] = current_z

    else:

        elevation_grid[gy, gx] = max(
            elevation_grid[gy, gx],
            current_z
        )


# ============================================================
# 11. COUNT POINTS IN EACH CELL
# ============================================================

for i in range(len(z_valid)):

    gx = grid_x[i]
    gy = grid_y[i]

    point_count_grid[gy, gx] += 1


# ============================================================
# 12. DETERMINE SEMANTIC CLASS OF EACH CELL
# ============================================================

# We use the most frequently occurring
# semantic class inside each cell.

for gy in range(grid_height):

    for gx in range(grid_width):

        # Find points belonging to this cell

        mask = (
            (grid_x == gx) &
            (grid_y == gy)
        )

        if not np.any(mask):
            continue

        cell_labels = labels_valid[mask]

        # Find unique labels and their counts

        unique, counts = np.unique(
            cell_labels,
            return_counts=True
        )

        # Select the most common class

        dominant_label = unique[
            np.argmax(counts)
        ]

        semantic_grid[gy, gx] = dominant_label


# ============================================================
# 13. PRINT SOME GRID INFORMATION
# ============================================================

occupied_cells = np.sum(
    point_count_grid > 0
)

print("\n2.5D Semantic Grid")
print("-------------------")

print("Total cells:",
      grid_width * grid_height)

print("Occupied cells:",
      occupied_cells)

print("Empty cells:",
      grid_width * grid_height - occupied_cells)


# ============================================================
# 14. DISPLAY SOME SEMANTIC CELLS
# ============================================================

print("\nSample occupied cells:")

counter = 0

for gy in range(grid_height):

    for gx in range(grid_width):

        label = semantic_grid[gy, gx]

        if label != -1:

            print(
                "Cell:",
                (gx, gy),
                "| Elevation:",
                round(elevation_grid[gy, gx], 2),
                "| Points:",
                point_count_grid[gy, gx],
                "| Semantic:",
                class_names.get(
                    label,
                    "unknown"
                )
            )

            counter += 1

            if counter >= 10:
                break

    if counter >= 10:
        break


# ============================================================
# 15. CREATE SEMANTIC COLOR MAP
# ============================================================

semantic_image = np.zeros(
    (grid_height, grid_width, 3),
    dtype=np.float32
)


# Same colors from Stage 8

class_colors = {

    0:  [0.5, 0.5, 0.5],
    1:  [0.3, 0.3, 0.3],

    10: [1.0, 0.0, 0.0],
    11: [1.0, 0.5, 0.0],
    13: [1.0, 0.0, 1.0],
    15: [0.8, 0.0, 0.0],
    16: [0.9, 0.3, 0.0],
    18: [0.7, 0.0, 0.0],
    20: [1.0, 0.3, 0.2],

    30: [0.0, 0.0, 1.0],
    31: [0.0, 0.5, 1.0],
    32: [0.0, 0.8, 1.0],

    40: [0.0, 1.0, 0.0],
    44: [0.3, 0.8, 0.3],
    48: [0.2, 0.6, 0.2],
    49: [0.4, 0.8, 0.4],

    50: [0.6, 0.4, 0.2],
    51: [0.7, 0.7, 0.2],
    52: [0.5, 0.3, 0.2],

    60: [1.0, 1.0, 1.0],

    70: [0.0, 0.8, 0.0],
    71: [0.2, 0.5, 0.0],
    72: [0.5, 0.8, 0.2],

    80: [0.5, 0.5, 0.5],
    81: [1.0, 1.0, 0.0]
}


for gy in range(grid_height):

    for gx in range(grid_width):

        label = semantic_grid[gy, gx]

        if label in class_colors:

            semantic_image[gy, gx] = (
                class_colors[label]
            )


# ============================================================
# 16. VISUALIZE SEMANTIC 2D GRID
# ============================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    semantic_image,
    origin="lower",
    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.title(
    "SemanticKITTI 2D Semantic Grid"
)

plt.show()


# ============================================================
# 17. VISUALIZE ELEVATION GRID
# ============================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    elevation_grid,
    origin="lower",
    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)

plt.colorbar(
    label="Elevation (m)"
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.title(
    "2.5D Elevation Grid"
)

plt.show()


# ============================================================
# 18. VISUALIZE POINT DENSITY
# ============================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    point_count_grid,
    origin="lower",
    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)

plt.colorbar(
    label="Number of points"
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.title(
    "LiDAR Point Density Grid"
)

plt.show()
def create_semantic_grid(
    xyz,
    semantic_labels,
    resolution=0.5,
    x_min=-50,
    x_max=50,
    y_min=-50,
    y_max=50
):
    """
    Create a 2.5D grid containing:
    - Maximum elevation
    - Dominant semantic class
    - Point density
    """

    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)

    print("\nGrid size:")
    print("Width :", grid_width)
    print("Height:", grid_height)

    # --------------------------------------------------------
    # Create grids
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Convert XYZ → grid indices
    # --------------------------------------------------------

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    grid_x = ((x - x_min) / resolution).astype(int)
    grid_y = ((y - y_min) / resolution).astype(int)

    # --------------------------------------------------------
    # Remove points outside grid
    # --------------------------------------------------------

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

    print("\nPoints inside grid:", len(z_valid))

    # --------------------------------------------------------
    # Build elevation grid
    # --------------------------------------------------------

    for gx, gy, current_z in zip(
        grid_x,
        grid_y,
        z_valid
    ):

        if np.isnan(elevation_grid[gy, gx]):

            elevation_grid[gy, gx] = current_z

        else:

            elevation_grid[gy, gx] = max(
                elevation_grid[gy, gx],
                current_z
            )

    # --------------------------------------------------------
    # Count points
    # --------------------------------------------------------

    for gx, gy in zip(grid_x, grid_y):

        point_count_grid[gy, gx] += 1

    # --------------------------------------------------------
    # Determine dominant semantic class
    # --------------------------------------------------------

    for gy in range(grid_height):

        for gx in range(grid_width):

            mask = (
                (grid_x == gx) &
                (grid_y == gy)
            )

            if not np.any(mask):
                continue

            cell_labels = labels_valid[mask]

            unique, counts = np.unique(
                cell_labels,
                return_counts=True
            )

            dominant_label = unique[
                np.argmax(counts)
            ]

            semantic_grid[gy, gx] = dominant_label

    # --------------------------------------------------------
    # Grid statistics
    # --------------------------------------------------------

    occupied_cells = np.sum(
        point_count_grid > 0
    )

    total_cells = (
        grid_width * grid_height
    )

    print("\n2.5D Semantic Grid")
    print("-------------------")
    print("Total cells:", total_cells)
    print("Occupied cells:", occupied_cells)
    print("Empty cells:", total_cells - occupied_cells)

    return (
        elevation_grid,
        semantic_grid,
        point_count_grid
    )