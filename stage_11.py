# ============================================================
# STAGE 14
# QUANTITATIVE EVALUATION
# ============================================================

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

bin_file = r"semantic_kitti_sample\dataset\sequences\00\velodyne\000009.bin"

label_file = r"semantic_kitti_sample\dataset\sequences\00\labels\000009.label"


# ============================================================
# 2. LOAD LiDAR
# ============================================================

points = np.fromfile(
    bin_file,
    dtype=np.float32
).reshape(-1, 4)

xyz = points[:, :3]


# ============================================================
# 3. LOAD GROUND-TRUTH LABELS
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
# 4. SEMANTIC CLASSES
# ============================================================

TRAVERSABLE_CLASSES = {

    40,   # road
    44,   # parking
    48,   # sidewalk
    49,   # other-ground
    72    # terrain
}


OBSTACLE_CLASSES = {

    10,   # car
    11,   # bicycle
    13,   # bus
    15,   # motorcycle
    16,   # on-rails
    18,   # truck
    20,   # other-vehicle

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


# ============================================================
# 5. CREATE GROUND-TRUTH OCCUPANCY
# ============================================================

# 0 = FREE
# 1 = OCCUPIED
# -1 = UNKNOWN

ground_truth_occupancy = np.full(
    len(semantic_labels),
    -1,
    dtype=np.int8
)


for i, label in enumerate(
    semantic_labels
):

    if label in OBSTACLE_CLASSES:

        ground_truth_occupancy[i] = 1


    elif label in TRAVERSABLE_CLASSES:

        ground_truth_occupancy[i] = 0


# ============================================================
# 6. CREATE PREDICTED OCCUPANCY
# ============================================================

# For this evaluation we classify
# each LiDAR point directly.

predicted_occupancy = np.full(
    len(semantic_labels),
    -1,
    dtype=np.int8
)


for i, label in enumerate(
    semantic_labels
):

    # In your actual system this should come
    # from your occupancy prediction.
    #
    # Here we use the semantic result as
    # the prediction so that the evaluation
    # pipeline can be demonstrated.

    if label in OBSTACLE_CLASSES:

        predicted_occupancy[i] = 1


    elif label in TRAVERSABLE_CLASSES:

        predicted_occupancy[i] = 0


# ============================================================
# 7. REMOVE UNKNOWN POINTS
# ============================================================

valid = (

    (ground_truth_occupancy != -1) &
    (predicted_occupancy != -1)

)


y_true = ground_truth_occupancy[
    valid
]

y_pred = predicted_occupancy[
    valid
]


# ============================================================
# 8. OCCUPANCY ACCURACY
# ============================================================

occupancy_accuracy = accuracy_score(
    y_true,
    y_pred
)


occupancy_precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


occupancy_recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


occupancy_f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


occupancy_iou = jaccard_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# ============================================================
# 10. PRINT OCCUPANCY RESULTS
# ============================================================

print("\n")
print("================================================")
print("             OCCUPANCY EVALUATION")
print("================================================")

print(
    "Accuracy :",
    round(occupancy_accuracy, 4)
)

print(
    "Precision:",
    round(occupancy_precision, 4)
)

print(
    "Recall   :",
    round(occupancy_recall, 4)
)

print(
    "F1-score :",
    round(occupancy_f1, 4)
)

print(
    "IoU      :",
    round(occupancy_iou, 4)
)


print("\nConfusion Matrix")

print(
    cm
)


# ============================================================
# 11. TRAVERSABILITY GROUND TRUTH
# ============================================================

# 1 = TRAVERSABLE
# 0 = NOT TRAVERSABLE
# -1 = UNKNOWN

ground_truth_traversability = np.full(
    len(semantic_labels),
    -1,
    dtype=np.int8
)


for i, label in enumerate(
    semantic_labels
):

    if label in TRAVERSABLE_CLASSES:

        ground_truth_traversability[i] = 1


    elif label in OBSTACLE_CLASSES:

        ground_truth_traversability[i] = 0


# ============================================================
# 12. PREDICTED TRAVERSABILITY
# ============================================================

predicted_traversability = np.full(
    len(semantic_labels),
    -1,
    dtype=np.int8
)


for i, label in enumerate(
    semantic_labels
):

    if label in TRAVERSABLE_CLASSES:

        predicted_traversability[i] = 1


    elif label in OBSTACLE_CLASSES:

        predicted_traversability[i] = 0


# ============================================================
# 13. REMOVE UNKNOWN
# ============================================================

valid_traversability = (

    (ground_truth_traversability != -1) &
    (predicted_traversability != -1)

)


true_traversability = (
    ground_truth_traversability[
        valid_traversability
    ]
)


pred_traversability = (
    predicted_traversability[
        valid_traversability
    ]
)


# ============================================================
# 14. TRAVERSABILITY METRICS
# ============================================================

traversability_accuracy = accuracy_score(
    true_traversability,
    pred_traversability
)


traversability_precision = precision_score(
    true_traversability,
    pred_traversability,
    zero_division=0
)


traversability_recall = recall_score(
    true_traversability,
    pred_traversability,
    zero_division=0
)


traversability_f1 = f1_score(
    true_traversability,
    pred_traversability,
    zero_division=0
)


traversability_iou = jaccard_score(
    true_traversability,
    pred_traversability,
    zero_division=0
)


# ============================================================
# 15. PRINT TRAVERSABILITY RESULTS
# ============================================================

print("\n")
print("================================================")
print("          TRAVERSABILITY EVALUATION")
print("================================================")

print(
    "Accuracy :",
    round(traversability_accuracy, 4)
)

print(
    "Precision:",
    round(traversability_precision, 4)
)

print(
    "Recall   :",
    round(traversability_recall, 4)
)

print(
    "F1-score :",
    round(traversability_f1, 4)
)

print(
    "IoU      :",
    round(traversability_iou, 4)
)


# ============================================================
# 16. ELEVATION STATISTICS
# ============================================================

z = xyz[:, 2]


# Ground points are the best approximation
# for terrain elevation.

ground_mask = np.isin(
    semantic_labels,
    list(TRAVERSABLE_CLASSES)
)


ground_z = z[
    ground_mask
]


if len(ground_z) > 0:

    mean_ground_height = np.mean(
        ground_z
    )

    std_ground_height = np.std(
        ground_z
    )

else:

    mean_ground_height = 0

    std_ground_height = 0


# ============================================================
# 17. PRINT ELEVATION STATISTICS
# ============================================================

print("\n")
print("================================================")
print("             TERRAIN STATISTICS")
print("================================================")

print(
    "Ground points:",
    len(ground_z)
)

print(
    "Mean ground elevation:",
    round(
        mean_ground_height,
        3
    ),
    "m"
)

print(
    "Elevation standard deviation:",
    round(
        std_ground_height,
        3
    ),
    "m"
)


# ============================================================
# 18. POINT CLOUD STATISTICS
# ============================================================

unique_classes, class_counts = np.unique(
    semantic_labels,
    return_counts=True
)


print("\n")
print("================================================")
print("             SEMANTIC STATISTICS")
print("================================================")


for label, count in zip(
    unique_classes,
    class_counts
):

    print(
        "Class",
        label,
        ":",
        count,
        "points"
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n")
print("================================================")
print("           STAGE 14 COMPLETE")
print("================================================")

print("\nOccupancy")
print(
    "Accuracy :",
    round(occupancy_accuracy, 4)
)
print(
    "Precision:",
    round(occupancy_precision, 4)
)
print(
    "Recall   :",
    round(occupancy_recall, 4)
)
print(
    "F1       :",
    round(occupancy_f1, 4)
)
print(
    "IoU      :",
    round(occupancy_iou, 4)
)


print("\nTraversability")
print(
    "Accuracy :",
    round(traversability_accuracy, 4)
)
print(
    "Precision:",
    round(traversability_precision, 4)
)
print(
    "Recall   :",
    round(traversability_recall, 4)
)
print(
    "F1       :",
    round(traversability_f1, 4)
)
print(
    "IoU      :",
    round(traversability_iou, 4)
)

print("================================================")