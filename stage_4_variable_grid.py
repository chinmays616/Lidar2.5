from sys import implementation

import numpy as np

import numpy as np
import matplotlib.pyplot as plt


file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000009.bin"
points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

xyz = points[:, :3]#Take every row, but only columns 0, 1 and 2.

intensity = points[:, 3]#

# ============================================================
# VARIABLE-RESOLUTION GRID
# ============================================================

# ------------------------------------------------------------
# 1. Define map boundaries
# ------------------------------------------------------------

x_min = xyz[:, 0].min()
x_max = xyz[:, 0].max()

y_min = xyz[:, 1].min()
y_max = xyz[:, 1].max()


# ------------------------------------------------------------
# 2. Create a coarse grid first
# ------------------------------------------------------------

coarse_resolution = 1.0     # 1 meter cells

coarse_width = int(
    np.ceil((x_max - x_min) / coarse_resolution)
)

coarse_height = int(
    np.ceil((y_max - y_min) / coarse_resolution)
)

# Store elevation
coarse_elevation = np.full(
    (coarse_height, coarse_width),
    np.nan
)


# ------------------------------------------------------------
# 3. Put LiDAR points into coarse cells
# ------------------------------------------------------------

grid_x = (
    (xyz[:, 0] - x_min) / coarse_resolution
).astype(int)

grid_y = (
    (xyz[:, 1] - y_min) / coarse_resolution
).astype(int)


for gx, gy, z in zip(
    grid_x,
    grid_y,
    xyz[:, 2]
):

    if (
        0 <= gx < coarse_width and
        0 <= gy < coarse_height
    ):

        if np.isnan(coarse_elevation[gy, gx]):
            coarse_elevation[gy, gx] = z

        else:
            coarse_elevation[gy, gx] = max(
                coarse_elevation[gy, gx],
                z
            )


# ============================================================
# 4. Calculate terrain complexity
# ============================================================

# Replace empty cells temporarily with 0
filled = np.nan_to_num(
    coarse_elevation,
    nan=0
)

dz_dy, dz_dx = np.gradient(
    filled,
    coarse_resolution
)

# Terrain slope
slope = np.sqrt(
    dz_dx**2 + dz_dy**2
)


# ============================================================
# 5. Decide resolution for each cell
# ============================================================

# Start with coarse resolution
resolution_map = np.full(
    coarse_elevation.shape,
    coarse_resolution
)

# Moderate terrain → 0.5 m
resolution_map[slope > 0.10] = 0.5

# Complex terrain → 0.25 m
resolution_map[slope > 0.30] = 0.25


# ============================================================
# 6. Display resolution map
# ============================================================

plt.figure(figsize=(12, 8))

plt.imshow(
    resolution_map,
    origin="lower",
    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ]
)

plt.colorbar(
    label="Cell Resolution (meters)"
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.title(
    "Variable-Resolution Grid"
)

plt.show()
