"""
Runs Stages 1-4 end to end on a single SemanticKITTI frame:
  load -> ground segmentation -> adaptive ring grid -> query interface -> visualize
"""

from pathlib import Path

from io_utils import load_lidar, load_labels
from ground import segment_ground_plane
from grid import RINGS, build_adaptive_ring_grid
from query import RingGridMap
from visualize import visualize_point_cloud, visualize_ring_grids

SEQ_DIR = Path(r"semantic_kitti_sample/dataset/sequences/00")
FRAME = "000009"
BIN_FILE = SEQ_DIR / "velodyne" / f"{FRAME}.bin"
LABEL_FILE = SEQ_DIR / "labels" / f"{FRAME}.label"


def main():
    xyz, intensity = load_lidar(BIN_FILE)
    n_points = len(xyz)
    print(f"Loaded {n_points} points from {BIN_FILE.name}")

    semantic_ids = load_labels(LABEL_FILE, n_points)
    print("Loaded semantic labels" if semantic_ids is not None else "No labels found")

    ground_mask, height_above_ground, plane_model = segment_ground_plane(xyz)
    print(f"Ground plane: {plane_model}, {ground_mask.sum()}/{n_points} points classified as ground")

    grid_data = build_adaptive_ring_grid(xyz, height_above_ground, ground_mask, semantic_ids, RINGS)

    for ring in RINGS:
        data = grid_data[ring.name]
        if data is None:
            print(f"{ring.name}: no points")
            continue
        occ_pct = 100.0 * data["occupied"].sum() / data["occupied"].size
        print(
            f"{ring.name}: {data['n_points']} pts -> {data['n_cells']}x{data['n_cells']} "
            f"@ {ring.resolution}m ({occ_pct:.1f}% occupied), "
            f"{len(data['refined'])} cells refined to {ring.resolution/2}m"
        )

    grid_map = RingGridMap(grid_data, RINGS)
    for x, y in [(3.0, 2.0), (15.0, -5.0), (60.0, 10.0)]:
        print(f"query({x}, {y}) -> {grid_map.query_point(x, y)}")

    visualize_point_cloud(xyz, ground_mask)
    visualize_ring_grids(grid_data, RINGS)


if __name__ == "__main__":
    main()
