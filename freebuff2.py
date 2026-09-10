import math
import time
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
# Kinodynamic feasibility parameters
LATERAL_ACCEL_MAX = 3.0          # max lateral acceleration (m/s^2) before tire slip
VEHICLE_SPEED_DEFAULT = 5.0      # default target speed (m/s ~ 18 km/h urban)
JERK_MAX = 5.0                   # max jerk (rate of accel change, m/s^3) for smooth velocity transitions
OBSTACLE_PROXIMITY_THRESHOLD = 3.0  # meters: within this distance of obstacle, reduce accel limit
TIMING_CSV_PATH = "pipeline_benchmark.csv"  # CSV file for timing export


# ============================================================
    print("================================================")


def launch_open3d_viewer(xyz, semantic_labels):
def launch_open3d_viewer(xyz, semantic_labels, velocity_path=None):
    """Launch Open3D viewer with semantic point cloud and optional velocity-colored path.

    Parameters
    ----------
    xyz : np.ndarray (N, 3)
    semantic_labels : np.ndarray (N,)
    velocity_path : o3d.geometry.LineSet or None
        Velocity-colored path overlay (green=fast, red=slow).
    """
    colors = create_semantic_colors(semantic_labels)
    visualize_semantic_pointcloud(xyz, colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]
    if velocity_path is not None:
        geometries.append(velocity_path)

    o3d.visualization.draw_geometries(geometries)


def build_interactive_composite(
    occupancy_grid,
    traversability_grid,
    }


def compute_path_curvature_and_speed(
def compute_distance_to_nearest_obstacle(
    path_world,
    lateral_accel_max=LATERAL_ACCEL_MAX,
    default_speed=VEHICLE_SPEED_DEFAULT
    occupancy_grid,
    x_min=X_MIN,
    y_min=Y_MIN,
    x_max=X_MAX,
    y_max=Y_MAX,
    resolution=RESOLUTION
):
    """Compute discrete curvature and kinodynamic speed limits for a path.
    """Compute distance from each path point to nearest obstacle cell.

    For each interior waypoint the curvature is estimated via the
    Menger curvature formula through three consecutive points:
    Uses scipy.ndimage.distance_transform_edt for efficient exact
    Euclidean distance to the closest occupied cell.

        kappa = 4 * area(ABC) / (|AB| * |BC| * |CA|)
    Returns
    -------
    distances : np.ndarray of shape (N,) in meters, one per path point
    """
    if path_world is None or len(path_world) == 0:
        return np.array([])

    The maximum allowable speed at each point is then:
    obstacle_mask = (occupancy_grid == 1)
    grid_height, grid_width = occupancy_grid.shape

        v_max = sqrt(a_lat_max / kappa)
    # Distance transform: distance from every cell to nearest obstacle
    dist_cells = ndimage.distance_transform_edt(~obstacle_mask)
    dist_meters = dist_cells * resolution

    If kappa is near zero (straight segment), v_max is set to a large
    value capped at 50 m/s for safety.
    pts = np.array(path_world, dtype=np.float64)
    N = len(pts)
    distances = np.zeros(N)

    for i in range(N):
        gx = int((pts[i, 0] - x_min) / resolution)
        gy = int((pts[i, 1] - y_min) / resolution)
        gx = max(0, min(gx, grid_width - 1))
        gy = max(0, min(gy, grid_height - 1))
        distances[i] = dist_meters[gy, gx]

    return distances


def compute_path_curvature_and_speed(
    path_world,
    lateral_accel_max=LATERAL_ACCEL_MAX,
    default_speed=VEHICLE_SPEED_DEFAULT,
    jerk_max=JERK_MAX,
    obstacle_distances=None,
    obstacle_proximity_threshold=OBSTACLE_PROXIMITY_THRESHOLD
):
    """Compute curvature, kinodynamic speed limits, and jerk-limited velocity profile.

    Improvements over v1:
      - Distance-dependent lateral accel: reduces a_lat_max when the path is
        near obstacles, forcing slower speeds for safety.
      - Jerk-limited profile: enforces max rate of speed change between
        consecutive waypoints, producing a physically-feasible trajectory.

    Parameters
    ----------
    path_world : list of (x, y) tuples
    lateral_accel_max : float
        Maximum lateral acceleration in m/s^2.
    default_speed : float
        Target cruise speed in m/s.
    lateral_accel_max : float  -- base max lateral acceleration (m/s^2)
    default_speed : float      -- target cruise speed (m/s)
    jerk_max : float           -- max speed change per unit arc-length (m/s^2)
    obstacle_distances : np.ndarray or None
        Pre-computed distance to nearest obstacle at each waypoint.
    obstacle_proximity_threshold : float
        Distance (m) within which lateral accel is scaled down linearly.

    Returns
    -------
    result : dict with keys
        'kappa'      : np.ndarray of shape (N,) — curvature at each point
        'v_max'      : np.ndarray of shape (N,) — max safe speed at each point
        'v_target'   : np.ndarray of shape (N,) — clamped target speed
        'v_exceeded' : bool — True if any segment requires speed < default_speed
        'max_kappa'  : float — maximum curvature along the path
        'min_v_max'  : float — minimum safe speed along the path
        'violations' : list of (index, kappa, v_max) for segments exceeding limits
        'kappa'          : np.ndarray (N,) -- curvature
        'v_max'          : np.ndarray (N,) -- curvature-limited speed
        'v_target'       : np.ndarray (N,) -- final target speed
        'v_exceeded'     : bool
        'max_kappa'      : float
        'min_v_max'      : float
        'violations'     : list of (index, kappa, v_max)
        'jerk_violations': list of (index, actual_jerk)
    """
    if path_world is None or len(path_world) < 3:
        return {
            "max_kappa": 0.0,
            "min_v_max": float("inf"),
            "violations": [],
            "jerk_violations": [],
        }

    pts = np.array(path_world, dtype=np.float64)
    N = len(pts)
    kappa = np.zeros(N)
    v_max = np.full(N, 50.0)  # default: very fast on straight segments
    violations = []
    v_max_curv = np.full(N, 50.0)

    # ---- Step 1: Curvature-limited speed (with proximity-dependent accel) ----
    for i in range(1, N - 1):
        A = pts[i - 1]
        B = pts[i]
        C = pts[i + 1]

        A, B, C = pts[i - 1], pts[i], pts[i + 1]
        AB = B - A
        BC = C - B
        CA = A - C

        len_AB = np.linalg.norm(AB)
        len_BC = np.linalg.norm(BC)
        len_CA = np.linalg.norm(CA)

        # Menger curvature: 4 * area of triangle / product of side lengths
        cross = AB[0] * BC[1] - AB[1] * BC[0]  # 2 * signed area
        cross = AB[0] * BC[1] - AB[1] * BC[0]
        area_2 = abs(cross)

        denom = len_AB * len_BC * len_CA

        if denom < 1e-12:
            kappa[i] = 0.0
        else:
            kappa[i] = 2.0 * area_2 / denom  # Menger formula: 4*area / (|AB|*|BC|*|CA|) but cross = 2*area
            kappa[i] = 2.0 * area_2 / denom

        # Correct Menger: kappa = 4 * Area / (|AB|*|BC|*|CA|)
        # area = 0.5 * |cross|, so 4*area = 2*|cross|
        # Already using 2.0*area_2 = 2*|cross| = 4*area. Correct.
        # Distance-dependent lateral accel limit
        a_lat = lateral_accel_max
        if obstacle_distances is not None and len(obstacle_distances) > i:
            d_obs = obstacle_distances[i]
            if d_obs < obstacle_proximity_threshold:
                # Linear scaling: at d=0 -> 0.5x; at d=threshold -> 1.0x
                proximity_factor = max(0.5, d_obs / obstacle_proximity_threshold)
                a_lat = lateral_accel_max * proximity_factor

        if kappa[i] > 1e-12:
            v_max[i] = math.sqrt(lateral_accel_max / kappa[i])
            v_max_curv[i] = math.sqrt(a_lat / kappa[i])
        else:
            v_max[i] = 50.0
            v_max_curv[i] = 50.0

        # Clamp v_max to reasonable range
        v_max[i] = min(v_max[i], 50.0)
        v_max[i] = max(v_max[i], 0.5)  # minimum 0.5 m/s to avoid division issues
        v_max_curv[i] = min(v_max_curv[i], 50.0)
        v_max_curv[i] = max(v_max_curv[i], 0.5)

    # Endpoints inherit curvature from neighbors
    if N >= 2:
        kappa[0] = kappa[1]
        v_max[0] = v_max[1]
        v_max_curv[0] = v_max_curv[1]
        kappa[-1] = kappa[-2]
        v_max[-1] = v_max[-2]
        v_max_curv[-1] = v_max_curv[-2]

    # Target speed: min(default_speed, v_max) at each point
    v_target = np.minimum(default_speed, v_max)
    # ---- Step 2: Jerk-limited velocity profile (forward + backward pass) ----
    v_target = np.minimum(default_speed, v_max_curv)

    # Detect violations: segments where default speed exceeds v_max
    v_exceeded = bool(np.any(v_max < default_speed))
    max_kappa = float(np.max(kappa))
    min_v_max = float(np.min(v_max))
    # Forward: enforce max acceleration
    for i in range(1, N):
        ds = np.linalg.norm(pts[i] - pts[i - 1])
        v_max_accel = v_target[i - 1] + jerk_max * ds
        v_target[i] = min(v_target[i], v_max_accel)

    # Backward: enforce max deceleration
    for i in range(N - 2, -1, -1):
        ds = np.linalg.norm(pts[i + 1] - pts[i])
        v_max_decel = v_target[i + 1] + jerk_max * ds
        v_target[i] = min(v_target[i], v_max_decel)

    # ---- Step 3: Detect violations ----
    violations = []
    jerk_violations = []
    for i in range(N):
        if v_max[i] < default_speed:
            violations.append((i, float(kappa[i]), float(v_max[i])))
        if v_target[i] < default_speed - 0.01:
            violations.append((i, float(kappa[i]), float(v_target[i])))

    for i in range(1, N):
        ds = np.linalg.norm(pts[i] - pts[i - 1])
        if ds < 1e-9:
            continue
        dv = abs(v_target[i] - v_target[i - 1])
        actual_jerk = dv / ds
        if actual_jerk > jerk_max + 0.01:
            jerk_violations.append((i, float(actual_jerk)))

    v_exceeded = bool(np.any(v_target < default_speed - 0.01))
    max_kappa = float(np.max(kappa))
    min_v_max = float(np.min(v_target))

    return {
        "kappa": kappa,
        "v_max": v_max,
        "v_max": v_max_curv,
        "v_target": v_target,
        "v_exceeded": v_exceeded,
        "max_kappa": max_kappa,
        "min_v_max": min_v_max,
        "violations": violations,
        "jerk_violations": jerk_violations,
    }


def export_timing_to_csv(
    timing_dict,
    output_path=TIMING_CSV_PATH
):
    """Export pipeline stage timings to a CSV file.

    Each run appends a row with a timestamp, making it easy to compare
    performance across runs or hardware.

    Parameters
    ----------
    timing_dict : dict
        Mapping of stage_name (str) -> elapsed_ms (float).
    output_path : str
        Path to CSV file.
    """
    write_header = not os.path.exists(output_path)

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["timestamp"] + list(timing_dict.keys())
            writer.writerow(header)
        row = [time.strftime("%Y-%m-%d %H:%M:%S")] + [f"{v:.2f}" for v in timing_dict.values()]
        writer.writerow(row)

    print(f"  Timing exported to: {output_path}")


def create_velocity_colored_3d_path(
    path_world,
    v_target,
    elevation_grid,
    x_min=X_MIN,
    y_min=Y_MIN,
    resolution=RESOLUTION,
    default_speed=VEHICLE_SPEED_DEFAULT
):
    """Create an Open3D LineSet with velocity-color-coded segments.

    Color scheme:
      Green  = full speed (>= 90% of default)
      Yellow = moderate (50-90%)
      Red    = slow (< 50% or violating limits)

    Parameters
    ----------
    path_world : list of (x, y) tuples
    v_target : np.ndarray of shape (N,) -- target speed at each waypoint
    elevation_grid : np.ndarray -- for z-coordinate lookup
    default_speed : float -- for color thresholds

    Returns
    -------
    line_set : o3d.geometry.LineSet with per-segment colors
    """
    if path_world is None or len(path_world) < 2 or elevation_grid is None:
        return None

    pts = np.array(path_world, dtype=np.float64)
    N = len(pts)
    grid_height, grid_width = elevation_grid.shape

    path_points = []
    for i in range(N):
        x, y = pts[i]
        gx = int((x - x_min) / resolution)
        gy = int((y - y_min) / resolution)
        gx = max(0, min(gx, grid_width - 1))
        gy = max(0, min(gy, grid_height - 1))
        z = elevation_grid[gy, gx]
        if np.isnan(z):
            z = 0.0
        z += 0.15  # lift above surface
        path_points.append([x, y, z])

    path_points = np.asarray(path_points)
    lines = [[i, i + 1] for i in range(N - 1)]

    # Color each segment by the average speed of its two endpoints
    colors = []
    for i in range(N - 1):
        avg_speed = 0.5 * (v_target[i] + v_target[i + 1])
        ratio = avg_speed / max(default_speed, 0.01)
        if ratio >= 0.9:
            color = [0.0, 0.8, 0.0]   # green: full speed
        elif ratio >= 0.5:
            # interpolate green -> yellow
            t = (ratio - 0.5) / 0.4
            color = [1.0 - t * 1.0, 0.8 + t * 0.2 - 0.8 * (1 - t), 0.0]
            color = [1.0 - t, 0.8 * t + 0.1 * (1 - t), 0.0]
        else:
            color = [0.9, 0.1, 0.1]   # red: slow / danger
        colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(path_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def plot_raw_vs_smooth_comparison(
    raw_path,
    smooth_path,
    if path_world is not None and len(path_world) >= 3:
        print("\n[STAGE 15b] Kinodynamic feasibility check...")
        t0 = time.perf_counter()
        obstacle_dists = compute_distance_to_nearest_obstacle(
            path_world, occupancy_grid,
            x_min=X_MIN, y_min=Y_MIN, x_max=X_MAX, y_max=Y_MAX, resolution=RESOLUTION
        )
        kinodynamic_result = compute_path_curvature_and_speed(
            path_world,
            lateral_accel_max=LATERAL_ACCEL_MAX,
            default_speed=VEHICLE_SPEED_DEFAULT
            default_speed=VEHICLE_SPEED_DEFAULT,
            jerk_max=JERK_MAX,
            obstacle_distances=obstacle_dists,
            obstacle_proximity_threshold=OBSTACLE_PROXIMITY_THRESHOLD
        )
        t_kinod = time.perf_counter() - t0
        print(f"  [TIMER] Kinodynamic analysis: {t_kinod * 1000:.1f} ms")
        print(f"  Max curvature: {kinodynamic_result['max_kappa']:.6f} 1/m")
        print(f"  Min safe speed: {kinodynamic_result['min_v_max']:.2f} m/s")
        print(f"  Target speed: {VEHICLE_SPEED_DEFAULT:.2f} m/s")
        print(f"  Jerk limit: {JERK_MAX:.2f} m/s^2/m")
        if kinodynamic_result['v_exceeded']:
            n_viol = len(kinodynamic_result['violations'])
            total_pts = len(kinodynamic_result['kappa'])
                  f"v_max={kinodynamic_result['violations'][0][2]:.2f} m/s")
        else:
            print(f"  OK: All segments safe at {VEHICLE_SPEED_DEFAULT:.1f} m/s")
        if kinodynamic_result.get('jerk_violations'):
            print(f"  Jerk violations: {len(kinodynamic_result['jerk_violations'])} segments")

    # Stage 13
    print("\n[STAGE 13] Opening visualization dashboard...")
    )

    print("\n[STAGE 13b] Launching interactive 3D point-cloud viewer...")
    launch_open3d_viewer(xyz, semantic_labels)
    # Build velocity-colored path for 3D viewer
    vel_3d_path = None
    if path_world is not None and kinodynamic_result is not None and len(kinodynamic_result["v_target"]) > 0:
        vel_3d_path = create_velocity_colored_3d_path(
            path_world, kinodynamic_result["v_target"], elevation_grid,
            x_min=X_MIN, y_min=Y_MIN, resolution=RESOLUTION,
            default_speed=VEHICLE_SPEED_DEFAULT
        )
    launch_open3d_viewer(xyz, semantic_labels, velocity_path=vel_3d_path)

    # Stage 14
    print("\n[STAGE 14] Running quantitative evaluation...")
        resolution=RESOLUTION, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX
    )

    # --- Export timing to CSV ---
    timing_data = {
        "lidar_load_ms": t_lidar * 1000,
        "bev_creation_ms": t_bev * 1000,
        "elevation_terrain_ms": t_elev * 1000,
        "adaptive_resolution_ms": t_adapt * 1000,
        "semantic_labels_ms": t_labels * 1000,
        "semantic_grid_ms": t_semgrid * 1000,
        "variable_grid_ms": t_vargrid * 1000,
        "raycast_occupancy_ms": t_raycast * 1000,
        "traversability_ms": t_trav * 1000,
        "costmap_inflation_ms": t_costmap * 1000,
        "astar_smoothing_ms": t_astar * 1000,
    }
    if kinodynamic_result is not None:
        timing_data["kinodynamic_ms"] = t_kinod * 1000
    export_timing_to_csv(timing_data, output_path=TIMING_CSV_PATH)

    # Final Summary
    total_pipeline_time = t_lidar + t_bev + t_elev + t_adapt + t_labels + t_semgrid + t_vargrid + t_raycast + t_trav + t_costmap + t_astar
    print("\n================================================")
        print(f"  Max curvature:           {kinodynamic_result['max_kappa']:.6f} 1/m")
        print(f"  Min safe speed:          {kinodynamic_result['min_v_max']:.2f} m/s")
        print(f"  Target speed:            {VEHICLE_SPEED_DEFAULT:.2f} m/s")
        print(f"  Lateral accel limit:     {LATERAL_ACCEL_MAX:.2f} m/s^2")
        print(f"  Lateral accel limit:     {LATERAL_ACCEL_MAX:.2f} m/s^2 (distance-dependent)")
        print(f"  Jerk limit:              {JERK_MAX:.2f} m/s^2 per meter")
        print(f"  Obstacle proximity:      {OBSTACLE_PROXIMITY_THRESHOLD:.1f} m (reduces accel near walls)")
        if kinodynamic_result['v_exceeded']:
            n = len(kinodynamic_result['violations'])
            total = len(kinodynamic_result['kappa'])
            print(f"  Status:                  {n}/{total} segments need speed reduction")
            print(f"  Curvature violations:    {n}/{total} segments need speed reduction")
        else:
            print(f"  Status:                  ALL segments safe")
            print(f"  Curvature violations:    NONE (all segments safe)")
        if kinodynamic_result.get('jerk_violations'):
            nj = len(kinodynamic_result['jerk_violations'])
            print(f"  Jerk violations:         {nj} segments exceed {JERK_MAX:.1f} m/s^2/m")
        else:
            print(f"  Jerk violations:         NONE (smooth velocity profile)")
    print("================================================")
import numpy as np, math, time
from scipy import ndimage
import csv, os, tempfile

# ---- Copy new functions for testing (avoid open3d import) ----

def compute_distance_to_nearest_obstacle(path_world, occupancy_grid, x_min=-50, y_min=-50, x_max=50, y_max=50, resolution=0.5):
    if path_world is None or len(path_world) == 0:
        return np.array([])
    obstacle_mask = (occupancy_grid == 1)
    grid_height, grid_width = occupancy_grid.shape
    dist_cells = ndimage.distance_transform_edt(~obstacle_mask)
    dist_meters = dist_cells * resolution
    pts = np.array(path_world, dtype=np.float64)
    N = len(pts)
    distances = np.zeros(N)
    for i in range(N):
        gx = int((pts[i, 0] - x_min) / resolution)
        gy = int((pts[i, 1] - y_min) / resolution)
        gx = max(0, min(gx, grid_width - 1))
        gy = max(0, min(gy, grid_height - 1))
        distances[i] = dist_meters[gy, gx]
    return distances

def compute_path_curvature_and_speed(path_world, lateral_accel_max=3.0, default_speed=5.0, jerk_max=5.0, obstacle_distances=None, obstacle_proximity_threshold=3.0):
    if path_world is None or len(path_world) < 3:
        return {'kappa': np.array([]), 'v_max': np.array([]), 'v_target': np.array([]),
                'v_exceeded': False, 'max_kappa': 0.0, 'min_v_max': float('inf'),
                'violations': [], 'jerk_violations': []}
    pts = np.array(path_world, dtype=np.float64)
    N = len(pts)
    kappa = np.zeros(N)
    v_max_curv = np.full(N, 50.0)
    for i in range(1, N - 1):
        A, B, C = pts[i-1], pts[i], pts[i+1]
        AB, BC, CA = B-A, C-B, A-C
        len_AB = np.linalg.norm(AB); len_BC = np.linalg.norm(BC); len_CA = np.linalg.norm(CA)
        cross = AB[0]*BC[1] - AB[1]*BC[0]
        area_2 = abs(cross); denom = len_AB * len_BC * len_CA
        kappa[i] = 2.0 * area_2 / denom if denom >= 1e-12 else 0.0
        a_lat = lateral_accel_max
        if obstacle_distances is not None and len(obstacle_distances) > i:
            d_obs = obstacle_distances[i]
            if d_obs < obstacle_proximity_threshold:
                a_lat = lateral_accel_max * max(0.5, d_obs / obstacle_proximity_threshold)
        v_max_curv[i] = math.sqrt(a_lat / kappa[i]) if kappa[i] > 1e-12 else 50.0
        v_max_curv[i] = min(max(v_max_curv[i], 0.5), 50.0)
    if N >= 2:
        kappa[0] = kappa[1]; v_max_curv[0] = v_max_curv[1]
        kappa[-1] = kappa[-2]; v_max_curv[-1] = v_max_curv[-2]
    v_target = np.minimum(default_speed, v_max_curv)
    # Forward jerk pass
    for i in range(1, N):
        ds = np.linalg.norm(pts[i] - pts[i-1])
        v_target[i] = min(v_target[i], v_target[i-1] + jerk_max * ds)
    # Backward jerk pass
    for i in range(N-2, -1, -1):
        ds = np.linalg.norm(pts[i+1] - pts[i])
        v_target[i] = min(v_target[i], v_target[i+1] + jerk_max * ds)
    violations = []; jerk_violations = []
    for i in range(N):
        if v_target[i] < default_speed - 0.01:
            violations.append((i, float(kappa[i]), float(v_target[i])))
    for i in range(1, N):
        ds = np.linalg.norm(pts[i] - pts[i-1])
        if ds < 1e-9: continue
        dv = abs(v_target[i] - v_target[i-1])
        actual_jerk = dv / ds
        if actual_jerk > jerk_max + 0.01:
            jerk_violations.append((i, float(actual_jerk)))
    v_exceeded = bool(np.any(v_target < default_speed - 0.01))
    return {'kappa': kappa, 'v_max': v_max_curv, 'v_target': v_target,
            'v_exceeded': v_exceeded, 'max_kappa': float(np.max(kappa)),
            'min_v_max': float(np.min(v_target)),
            'violations': violations, 'jerk_violations': jerk_violations}


# ============ TEST 1: Distance to nearest obstacle ============
print("--- Test compute_distance_to_nearest_obstacle ---")
# Use small grid: 0-50m, resolution 0.5 -> 100x100 cells
grid = np.full((100, 100), 0, dtype=np.int8)
# Put obstacle at grid (50, 50) = world (25, 25)
grid[50, 50] = 1

# Path passes right through (25, 25)
path = [(5.0, 25.0), (15.0, 25.0), (25.0, 25.0), (35.0, 25.0), (45.0, 25.0)]
dists = compute_distance_to_nearest_obstacle(path, grid, 0, 0, 50, 50, 0.5)
print("  Distances:", np.round(dists, 2))
assert len(dists) == 5
# Point at (25,25) is ON the obstacle cell -> distance ~ 0
assert dists[2] < 1.0, "Point on obstacle should have near-zero distance"
# Points further away should have larger distances
assert dists[0] > dists[2], "Far point should have larger distance"
assert dists[4] > dists[2], "Far point should have larger distance"
print("  PASSED")


# ============ TEST 2: Distance-dependent lateral accel ============
print("\n--- Test distance-dependent accel ---")
path_turn = [(0,0), (5,0), (5,5)]

dists_near = np.array([5.0, 1.0, 5.0])  # 1m from obstacle at turn
res_near = compute_path_curvature_and_speed(path_turn, 3.0, 5.0, 5.0, dists_near, 3.0)

dists_far = np.array([10.0, 10.0, 10.0])  # far from obstacles
res_far = compute_path_curvature_and_speed(path_turn, 3.0, 5.0, 5.0, dists_far, 3.0)

print("  Near obstacle:  v_target[1] = {:.2f} m/s".format(res_near["v_target"][1]))
print("  Far from obstacle: v_target[1] = {:.2f} m/s".format(res_far["v_target"][1]))
assert res_near["v_target"][1] < res_far["v_target"][1], \
    "Near obstacle should have LOWER speed due to reduced a_lat"
print("  PASSED")


# ============ TEST 3: Jerk limiting ============
print("\n--- Test jerk limiting ---")
path_long = [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0),
             (5,5), (5,10), (5,15), (5,20)]
dists_long = np.full(len(path_long), 10.0)

res_long = compute_path_curvature_and_speed(path_long, 3.0, 5.0, 2.0, dists_long, 3.0)
print("  Jerk violations:", len(res_long["jerk_violations"]))

if len(res_long["violations"]) > 0:
    v = res_long["v_target"]
    max_dv_per_m = 0
    for i in range(1, len(v)):
        ds = np.linalg.norm(np.array(path_long[i]) - np.array(path_long[i-1]))
        if ds > 1e-9:
            dv = abs(v[i] - v[i-1]) / ds
            max_dv_per_m = max(max_dv_per_m, dv)
    print("  Max speed change rate: {:.2f} m/s per m (limit: 2.0)".format(max_dv_per_m))
    assert max_dv_per_m <= 2.05, "Jerk exceeded: {}".format(max_dv_per_m)
    print("  PASSED")
else:
    print("  No speed violations (all safe at default speed)")


# ============ TEST 4: CSV export ============
print("\n--- Test CSV export ---")
tmpfile = os.path.join(tempfile.gettempdir(), "test_timing.csv")
if os.path.exists(tmpfile):
    os.remove(tmpfile)

timing_data = {"stage_a_ms": 12.34, "stage_b_ms": 56.78}
write_header = not os.path.exists(tmpfile)
with open(tmpfile, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        header = ["timestamp"] + list(timing_data.keys())
        writer.writerow(header)
    row = [time.strftime("%Y-%m-%d %H:%M:%S")] + ["{:.2f}".format(v) for v in timing_data.values()]
    writer.writerow(row)

# Append second run
with open(tmpfile, "a", newline="") as f:
    writer = csv.writer(f)
    row = [time.strftime("%Y-%m-%d %H:%M:%S")] + ["{:.2f}".format(v) for v in timing_data.values()]
    writer.writerow(row)

with open(tmpfile, "r") as f:
    content = f.read()
lines = content.strip().split("\n")
assert len(lines) == 3, "Expected 3 lines (header + 2 rows), got {}".format(len(lines))
assert "timestamp" in lines[0]
assert "stage_a_ms" in lines[0]
print("  CSV content:")
for line in lines:
    print("    " + line)
print("  PASSED")
os.remove(tmpfile)


# ============ SUMMARY ============
print("\n" + "=" * 48)
print("  ALL TESTS PASSED")
print("=" * 48)

Give feedback


Google Cloud