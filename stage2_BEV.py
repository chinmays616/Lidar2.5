from sys import implementation

import numpy as np

file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000009.bin"
points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)


xyz = points[:, :3]#Take every row, but only columns 0, 1 and 2.
# ============================================================
# BEV (Bird's-Eye View)
# 3D LiDAR Point Cloud → 2D Top-Down Height Map
# ============================================================

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Define the area we want to visualize
# ------------------------------------------------------------

x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()

# Keep only points inside this area
mask = (
    (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
    (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
)

bev_points = xyz[mask]
#filtered_xyz = xyz[mask]

print("Points used for BEV:", len(bev_points))


# ------------------------------------------------------------
# 2. Set BEV resolution
# ------------------------------------------------------------

resolution = 0.1   # 0.1 meter = 10 cm per cell

grid_width = int((x_max - x_min) / resolution)
grid_height = int((y_max - y_min) / resolution)

print("BEV grid size:", grid_width, "x", grid_height)


# ------------------------------------------------------------
# 3. Convert X,Y coordinates to grid coordinates
# ------------------------------------------------------------

grid_x = ((bev_points[:, 0] - x_min) / resolution).astype(int)
grid_y = ((bev_points[:, 1] - y_min) / resolution).astype(int)


# ------------------------------------------------------------
# 4. Create empty BEV grid
# ------------------------------------------------------------

bev = np.full(
    (grid_height, grid_width),
    np.nan
)


# ------------------------------------------------------------
# 5. Put Z(height) into each grid cell
# ------------------------------------------------------------

for gx, gy, z_value in zip(grid_x, grid_y, bev_points[:, 2]):

    if 0 <= gx < grid_width and 0 <= gy < grid_height:

        # Keep the highest point in the cell
        if np.isnan(bev[gy, gx]):
            bev[gy, gx] = z_value
        else:
            bev[gy, gx] = max(bev[gy, gx], z_value)


# ------------------------------------------------------------
# 6. Display BEV
# ------------------------------------------------------------

plt.figure(figsize=(10, 10))

plt.imshow(
    bev,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap="terrain"
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("LiDAR BEV - Height Map")

plt.colorbar(label="Height (Z)")

plt.show()