from sys import implementation

import numpy as np

file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000001.bin"
points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)


#np.fromfile(file, dtype=np.float32)------>Read the binary data and interpret every 4 bytes as a floating-point number.

#reshape(-1, 4)-------->Reshape the array into a 2D array with 4 columns, where -1 means the number of rows is inferred from the length of the array.

xyz = points[:, :3]#Take every row, but only columns 0, 1 and 2.

intensity = points[:, 3]#Take every row but only 3 column that is intensity.

print("Number of points:", len(points))
print("Shape:", points.shape)
print("First 5 points:")
print(points[:5])

z = xyz[:, 2]#Get the height values (Z coordinates) from the point cloud.

# Normalize height to 0-1
z_min = z.min()
z_max = z.max()

z_normalized = (z - z_min) / (z_max - z_min)#Will give a value between 0 and 1 for each point based on its height.

# Create RGB colors
colors = np.zeros((len(xyz), 3))

colors[:, 0] = z_normalized       # Red
colors[:, 1] = 1 - z_normalized   # Green
colors[:, 2] = 0                   # Blue


import open3d as o3d

# # Load SemanticKITTI point cloud
# file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000000.bin"

points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

# Extract XYZ coordinates
xyz = points[:, :3]

print("Number of points:", len(xyz))


pcd = o3d.geometry.PointCloud()# Create Open3D point cloud
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
# o3d.visualization.draw_geometries(#opens the 3D viewer.
#     [pcd],
#     window_name="SemanticKITTI Point Cloud",
#     width=1200,
#     height=800
# )

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Height Colored LiDAR",
    width=1200,
    height=800
)