"""
io_utils.py
===========
Loading raw SemanticKITTI frames, plus a pose-transform utility.

Addresses:
  #22 Lack of sensor pose and coordinate transformations
      -> apply_pose_transform() lets any future multi-frame accumulator
         (ICP/LOAM odometry, KITTI calibration poses, etc.) bring
         several scans into one common world frame before they are
         handed to grid.py / ground.py. Not exercised by app.py yet
         because only a single frame is available in this dataset
         sample, but the pipeline no longer assumes sensor_origin is
         always (0,0) - see Config.sensor_origin.
  #18 Limited robustness to different scenes
      -> load_pointcloud/labels raise clear, actionable errors instead
         of silently producing garbage on malformed input.
"""

from typing import Tuple
import numpy as np


def load_pointcloud(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a SemanticKITTI .bin frame -> (raw_points, xyz, intensity)."""
    raw = np.fromfile(path, dtype=np.float32)

    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(
            f"'{path}' does not look like a valid SemanticKITTI .bin file "
            f"(size {raw.size} floats is not a multiple of 4)."
        )

    points = raw.reshape(-1, 4)
    xyz = points[:, :3]
    intensity = points[:, 3]

    print("LiDAR points:", len(points))
    return points, xyz, intensity


def load_semantic_labels(path: str, num_points: int) -> np.ndarray:
    """Load a SemanticKITTI .label file -> per-point semantic class ids."""
    labels = np.fromfile(path, dtype=np.uint32)

    if num_points != len(labels):
        raise ValueError(
            f"Point count ({num_points}) and label count ({len(labels)}) "
            f"do not match for '{path}'."
        )

    # Lower 16 bits = semantic class, upper 16 bits = instance id.
    return labels & 0xFFFF


def apply_pose_transform(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Transform an (N, 3) point cloud from sensor/local frame into world
    frame using a 4x4 homogeneous pose matrix (e.g. KITTI odometry
    poses, or an ICP-estimated relative transform between consecutive
    scans - see #22/#19 for how this plugs into multi-frame use).
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be a 4x4 matrix, got shape {pose.shape}")

    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([xyz.astype(np.float64), ones])
    world = (pose @ homogeneous.T).T
    return world[:, :3]


def accumulate_frames(frames_xyz, poses) -> np.ndarray:
    """
    Merge several (xyz, pose) scans into one world-frame point cloud.
    A minimal building block for temporal accumulation (#19 multi-frame
    evaluation, #16 groundwork for tracking moving obstacles across
    frames) - not run by the default single-frame pipeline.
    """
    if len(frames_xyz) != len(poses):
        raise ValueError("frames_xyz and poses must be the same length")

    transformed = [
        apply_pose_transform(xyz, pose) for xyz, pose in zip(frames_xyz, poses)
    ]
    return np.vstack(transformed)