from sys import implementation
import numpy as np
import matplotlib.pyplot as plt

file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000009.bin"
points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

xyz = points[:, :3]#Take every row, but only columns 0, 1 and 2.

intensity = points[:, 3]#
x_min = xyz[:, 0].min()
x_max = xyz[:, 0].max()

y_min = xyz[:, 1].min()
y_max = xyz[:, 1].max()

resolution = 0.1

grid_width = int((x_max - x_min) / resolution)
grid_height = int((y_max - y_min) / resolution)

elevation_grid = np.full(
    (grid_height, grid_width),
    np.nan
)

grid_x = ((xyz[:, 0] - x_min) / resolution).astype(int)
grid_y = ((xyz[:, 1] - y_min) / resolution).astype(int)

for gx, gy, z in zip(grid_x, grid_y, xyz[:, 2]):

    if 0 <= gx < grid_width and 0 <= gy < grid_height:

        if np.isnan(elevation_grid[gy, gx]):
            elevation_grid[gy, gx] = z

        else:
            elevation_grid[gy, gx] = max(
                elevation_grid[gy, gx],
                z
            )

plt.imshow(
    elevation_grid,
    origin="lower"
)

plt.title("Fixed-Resolution Elevation Grid")
plt.colorbar(label="Height (Z)")

plt.show()


# ==========================================
# GROUND POINT DETECTION
# ==========================================

z = xyz[:, 2]

# Ground threshold
ground_threshold = -1.5

# Ground points
ground_mask = z < ground_threshold

# Non-ground points
non_ground_mask = z >= ground_threshold

ground_points = xyz[ground_mask]
non_ground_points = xyz[non_ground_mask]

print("Total points:", len(xyz))
print("Ground points:", len(ground_points))
print("Non-ground points:", len(non_ground_points))


# ==========================================
# TERRAIN ANALYSIS
# ==========================================

# Replace empty cells with nearest available
# value for basic analysis
valid = ~np.isnan(elevation_grid)

# Calculate gradients
dz_dy, dz_dx = np.gradient(
    np.nan_to_num(elevation_grid, nan=0),
    resolution
)

# Calculate slope magnitude
slope = np.sqrt(
    dz_dx**2 + dz_dy**2
)

# Convert slope to degrees
slope_angle = np.degrees(
    np.arctan(slope)
)

# Visualize slope
plt.figure(figsize=(10, 8))

plt.imshow(
    slope_angle,
    origin="lower",
    cmap="viridis"
)

plt.colorbar(label="Slope (degrees)")

plt.title("Terrain Slope Map")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
# ============================================
# ==========================================
# TERRAIN ANALYSIS
# ==========================================
# ============================================

from scipy import ndimage


# ------------------------------------------------------------
# 1. Find valid and empty cells
# ------------------------------------------------------------

valid = ~np.isnan(elevation_grid)

print("Valid cells:", np.sum(valid))
print("Empty cells:", np.sum(~valid))


# ------------------------------------------------------------
# 2. Replace empty cells with nearest available elevation
# ------------------------------------------------------------

# Find the nearest valid cell for every empty cell
nearest_indices = ndimage.distance_transform_edt(
    ~valid,
    return_distances=False,
    return_indices=True
)

# Create a copy of the elevation grid
filled_elevation = elevation_grid.copy()

# Fill NaN cells using the elevation of their nearest
# valid cell
filled_elevation[~valid] = elevation_grid[
    tuple(nearest_indices[:, ~valid])
]


# ------------------------------------------------------------
# 3. Calculate elevation gradients
# ------------------------------------------------------------

dz_dy, dz_dx = np.gradient(
    filled_elevation,
    resolution
)


# ------------------------------------------------------------
# 4. Calculate slope magnitude
# ------------------------------------------------------------

slope = np.sqrt(
    dz_dx**2 +
    dz_dy**2
)


# ------------------------------------------------------------
# 5. Convert slope to degrees
# ------------------------------------------------------------

slope_angle = np.degrees(
    np.arctan(slope)
)


# ------------------------------------------------------------
# 6. Keep originally empty cells empty
# ------------------------------------------------------------

slope_angle[~valid] = np.nan


# ------------------------------------------------------------
# 7. Display filled elevation grid
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(
    filled_elevation,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap="terrain"
)

plt.colorbar(
    label="Elevation (Z)"
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.title("Filled Elevation Grid")

plt.show()


# ------------------------------------------------------------
# 8. Display terrain slope
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(
    slope_angle,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap="viridis"
)

plt.colorbar(
    label="Slope (degrees)"
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.title("Terrain Slope Map")

plt.show()

def create_elevation_grid(xyz, resolution=0.1):
    """
    Stage 3A:
    Create fixed-resolution elevation grid.
    """

    x_min = xyz[:, 0].min()
    x_max = xyz[:, 0].max()

    y_min = xyz[:, 1].min()
    y_max = xyz[:, 1].max()

    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)

    elevation_grid = np.full(
        (grid_height, grid_width),
        np.nan
    )

    grid_x = (
        (xyz[:, 0] - x_min) / resolution
    ).astype(int)

    grid_y = (
        (xyz[:, 1] - y_min) / resolution
    ).astype(int)

    for gx, gy, z in zip(
        grid_x,
        grid_y,
        xyz[:, 2]
    ):

        if 0 <= gx < grid_width and 0 <= gy < grid_height:

            if np.isnan(elevation_grid[gy, gx]):
                elevation_grid[gy, gx] = z

            else:
                elevation_grid[gy, gx] = max(
                    elevation_grid[gy, gx],
                    z
                )

    return elevation_grid, x_min, x_max, y_min, y_max

def ground_detection(xyz, ground_threshold=-1.5):
    """
    Stage 3B:
    Separate ground and non-ground points.
    """

    z = xyz[:, 2]

    ground_mask = z < ground_threshold
    non_ground_mask = z >= ground_threshold

    ground_points = xyz[ground_mask]
    non_ground_points = xyz[non_ground_mask]

    print("Total points:", len(xyz))
    print("Ground points:", len(ground_points))
    print("Non-ground points:", len(non_ground_points))

    return (
        ground_mask,
        non_ground_mask,
        ground_points,
        non_ground_points
    )
def terrain_analysis(
    elevation_grid,
    resolution
):
    """
    Stage 3C:
    Fill missing elevation cells and calculate terrain slope.
    """

    from scipy import ndimage

    valid = ~np.isnan(elevation_grid)

    print("Valid cells:", np.sum(valid))
    print("Empty cells:", np.sum(~valid))

    # Find nearest valid cell
    nearest_indices = ndimage.distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True
    )

    filled_elevation = elevation_grid.copy()

    # Fill empty cells
    filled_elevation[~valid] = elevation_grid[
        tuple(nearest_indices[:, ~valid])
    ]

    # Calculate gradients
    dz_dy, dz_dx = np.gradient(
        filled_elevation,
        resolution
    )

    # Slope magnitude
    slope = np.sqrt(
        dz_dx**2 +
        dz_dy**2
    )

    # Convert to degrees
    slope_angle = np.degrees(
        np.arctan(slope)
    )

    # Restore original empty cells
    slope_angle[~valid] = np.nan

    return (
        filled_elevation,
        slope,
        slope_angle
    )