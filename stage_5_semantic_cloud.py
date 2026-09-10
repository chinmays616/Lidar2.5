from sys import implementation

import numpy as np

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

# Extract XYZ
xyz = points[:, :3]

# Extract intensity
intensity = points[:, 3]

print("Number of LiDAR points:", len(points))
print("Point cloud shape:", points.shape)

print("\nFirst 5 points:")
print(points[:5])


# ============================================================
# 3. LOAD SEMANTICKITTI LABELS
# ============================================================

labels = np.fromfile(
    label_file,
    dtype=np.uint32
)

print("\nNumber of labels:", len(labels))

# Check that every point has one label
if len(points) != len(labels):
    raise ValueError(
        "Number of points and labels do not match!"
    )

print("Points and labels match!")


# ============================================================
# 4. EXTRACT SEMANTIC ID
# ============================================================

# SemanticKITTI label format contains:
#
# lower 16 bits  → semantic class
# upper 16 bits  → instance ID
#
# We only need the semantic class.

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
# 6. PRINT CLASSES PRESENT IN THIS FRAME
# ============================================================

unique_labels = np.unique(semantic_labels)

print("\nSemantic classes present in frame:")

for label in unique_labels:

    if label in class_names:
        print(
            label,
            "->",
            class_names[label]
        )

    else:
        print(
            label,
            "-> unknown"
        )


# ============================================================
# 7. DEFINE COLORS FOR EACH SEMANTIC CLASS
# ============================================================

# RGB values range from 0 to 1

class_colors = {

    # Unlabeled / outlier
    0:  [0.5, 0.5, 0.5],
    1:  [0.3, 0.3, 0.3],

    # Vehicles
    10: [1.0, 0.0, 0.0],       # car
    11: [1.0, 0.5, 0.0],       # bicycle
    13: [1.0, 0.0, 1.0],       # bus
    15: [0.8, 0.0, 0.0],       # motorcycle
    16: [0.9, 0.3, 0.0],       # on-rails
    18: [0.7, 0.0, 0.0],       # truck
    20: [1.0, 0.3, 0.2],       # other vehicle

    # People
    30: [0.0, 0.0, 1.0],       # person
    31: [0.0, 0.5, 1.0],       # bicyclist
    32: [0.0, 0.8, 1.0],       # motorcyclist

    # Ground
    40: [0.0, 1.0, 0.0],       # road
    44: [0.3, 0.8, 0.3],       # parking
    48: [0.2, 0.6, 0.2],       # sidewalk
    49: [0.4, 0.8, 0.4],       # other-ground

    # Structures
    50: [0.6, 0.4, 0.2],       # building
    51: [0.7, 0.7, 0.2],       # fence
    52: [0.5, 0.3, 0.2],       # other structure

    # Lane marking
    60: [1.0, 1.0, 1.0],

    # Nature
    70: [0.0, 0.8, 0.0],       # vegetation
    71: [0.2, 0.5, 0.0],       # trunk
    72: [0.5, 0.8, 0.2],       # terrain

    # Road objects
    80: [0.5, 0.5, 0.5],       # pole
    81: [1.0, 1.0, 0.0]        # traffic sign
}


# ============================================================
# 8. CREATE COLOR FOR EVERY POINT
# ============================================================

colors = np.zeros(
    (len(xyz), 3),
    dtype=np.float64
)

for i, label in enumerate(semantic_labels):

    if label in class_colors:

        colors[i] = class_colors[label]

    else:

        # Unknown class
        colors[i] = [0.5, 0.5, 0.5]


# ============================================================
# 9. CREATE OPEN3D POINT CLOUD
# ============================================================

import open3d as o3d
pcd = o3d.geometry.PointCloud()

# Add XYZ coordinates
pcd.points = o3d.utility.Vector3dVector(xyz)

# Add semantic colors
pcd.colors = o3d.utility.Vector3dVector(colors)


# ============================================================
# 10. VISUALIZE
# ============================================================

o3d.visualization.draw_geometries(

    [pcd],

    window_name="SemanticKITTI Semantic Point Cloud",

    width=1200,

    height=800
)