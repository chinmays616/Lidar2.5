"""
visualize.py
============
Matplotlib multi-panel dashboard + interactive Plotly composite.
Functionally the same panels as the original pipeline; adapted to take
a Config object and the new (occupancy_probability, inflated obstacle,
planned path) outputs so the dashboard reflects what the planner
actually saw, not just the raw grids.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import CheckButtons

from config import Config
from grid import get_semantic_colormap


def visualization_dashboard(
    elevation_grid, semantic_grid, point_count_grid,
    occupancy_grid, slope_degrees, traversability_grid,
    cfg: Config, x_min, x_max, y_min, y_max, path_world=None,
):
    extent = [x_min, x_max, y_min, y_max]
    fig = plt.figure(figsize=(17, 13))

    axes, images, titles = [], [], []

    def _panel(pos, data, title, cmap=None, norm=None, vmin=None, vmax=None,
               cbar_label=None, mask_value=None):
        ax = fig.add_subplot(2, 3, pos)
        d = data
        if mask_value is not None:
            d = np.ma.masked_where(data == mask_value, data)
        elif np.issubdtype(np.asarray(data).dtype, np.floating):
            d = np.ma.masked_invalid(data)
        im = ax.imshow(d, origin="lower", extent=extent, cmap=cmap, norm=norm,
                        vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        if cbar_label:
            fig.colorbar(im, ax=ax, label=cbar_label)
        axes.append(ax)
        images.append(im)
        titles.append(title)
        return ax, im

    _panel(1, elevation_grid, "Elevation Map", cbar_label="Elevation (m)")

    ax2 = fig.add_subplot(2, 3, 2)
    present_labels = set(np.unique(semantic_grid).tolist())
    semantic_cmap, semantic_norm, legend_handles, legend_labels = get_semantic_colormap(present_labels)
    im2 = ax2.imshow(semantic_grid, origin="lower", extent=extent, cmap=semantic_cmap,
                      norm=semantic_norm, interpolation="nearest")
    ax2.set_title("Semantic Map")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_aspect("equal")
    axes.append(ax2); images.append(im2); titles.append("Semantic")

    _panel(3, point_count_grid, "LiDAR Point Density", cbar_label="Points / Cell")

    ax4 = fig.add_subplot(2, 3, 4)
    occ_masked = np.ma.masked_where(occupancy_grid == -1, occupancy_grid)
    occ_cmap = mcolors.ListedColormap(["#4daf4a", "#e41a1c"])
    occ_cmap.set_bad(color="none")
    ax4.imshow(occ_masked, origin="lower", extent=extent, cmap=occ_cmap, vmin=0, vmax=1,
               interpolation="nearest")
    ax4.set_facecolor("white")
    ax4.set_title("Occupancy Map (probabilistic raycast)")
    ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)"); ax4.set_aspect("equal")
    occ_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black") for c in ["#4daf4a", "#e41a1c"]]
    ax4.legend(occ_handles, ["Free", "Occupied"], loc="upper right", fontsize=7)
    axes.append(ax4); images.append(None); titles.append("Occupancy")

    ax5 = fig.add_subplot(2, 3, 5)
    slope_display = np.ma.masked_invalid(np.clip(slope_degrees, 0, cfg.slope_display_max_deg))
    slope_cmap = plt.get_cmap("turbo").copy()
    slope_cmap.set_bad(color="none")
    im5 = ax5.imshow(slope_display, origin="lower", extent=extent, cmap=slope_cmap,
                      vmin=0, vmax=cfg.slope_display_max_deg)
    ax5.set_facecolor("white")
    ax5.set_title(f"Terrain Slope (capped at {cfg.slope_display_max_deg:.0f}\u00b0)")
    ax5.set_xlabel("X (m)"); ax5.set_ylabel("Y (m)"); ax5.set_aspect("equal")
    fig.colorbar(im5, ax=ax5, label="Slope (degrees)")
    axes.append(ax5); images.append(im5); titles.append("Slope")

    ax6 = fig.add_subplot(2, 3, 6)
    trav_masked = np.ma.masked_where(traversability_grid == 2, traversability_grid)
    trav_cmap = mcolors.ListedColormap(["#e41a1c", "#4daf4a"])
    trav_cmap.set_bad(color="none")
    ax6.imshow(trav_masked, origin="lower", extent=extent, cmap=trav_cmap, vmin=0, vmax=1,
               interpolation="nearest")
    ax6.set_facecolor("white")
    ax6.set_title("Traversability Map")
    ax6.set_xlabel("X (m)"); ax6.set_ylabel("Y (m)"); ax6.set_aspect("equal")

    trav_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black") for c in ["#e41a1c", "#4daf4a"]]
    trav_labels = ["Not traversable", "Traversable"]

    if path_world:
        path_x = [p[0] for p in path_world]
        path_y = [p[1] for p in path_world]
        ax6.plot(path_x, path_y, color="blue", linewidth=2.5, zorder=5)
        start_pt = ax6.plot(path_x[0], path_y[0], marker="*", color="cyan", markersize=14,
                             markeredgecolor="black", linestyle="None", zorder=6)[0]
        goal_pt = ax6.plot(path_x[-1], path_y[-1], marker="*", color="magenta", markersize=14,
                            markeredgecolor="black", linestyle="None", zorder=6)[0]
        path_line = plt.Line2D([0], [0], color="blue", linewidth=2.5)
        trav_handles += [path_line, start_pt, goal_pt]
        trav_labels += ["Planned path", "Start", "Goal"]

    ax6.legend(trav_handles, trav_labels, loc="upper right", fontsize=7)
    axes.append(ax6); images.append(None); titles.append("Traversability")

    fig.suptitle("LiDAR 2.5D Semantic Mapping Dashboard", fontsize=18)
    fig.legend(legend_handles, legend_labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
               bbox_transform=fig.transFigure, ncol=1, fontsize=7, frameon=True, title="Semantic class")
    plt.tight_layout(rect=[0.08, 0.05, 0.85, 0.95])

    check_ax = fig.add_axes([0.01, 0.75, 0.1, 0.2])
    check = CheckButtons(check_ax, titles, [True] * len(titles))

    def _toggle(label):
        idx = titles.index(label)
        axes[idx].set_visible(not axes[idx].get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(_toggle)
    fig._layer_toggle_widget = check

    plt.show()
    return fig


def build_interactive_composite(occupancy_grid, traversability_grid, semantic_grid,
                                 cfg: Config, x_min, x_max, y_min, y_max,
                                 output_path="interactive_dashboard.html", path_world=None):
    import plotly.graph_objects as go

    x_coords = np.arange(x_min, x_max, cfg.resolution)
    y_coords = np.arange(y_min, y_max, cfg.resolution)

    semantic_masked = np.where(semantic_grid == -1, np.nan, semantic_grid)
    occupancy_masked = np.where(occupancy_grid == -1, np.nan, occupancy_grid)
    traversability_masked = np.where(traversability_grid == 2, np.nan, traversability_grid)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=semantic_masked, x=x_coords, y=y_coords, colorscale="Turbo",
                              name="Semantic", visible=True, showscale=False))
    fig.add_trace(go.Heatmap(z=occupancy_masked, x=x_coords, y=y_coords,
                              colorscale=[[0, "#4daf4a"], [1, "#e41a1c"]], zmin=0, zmax=1,
                              name="Occupancy", visible="legendonly", showscale=False))
    fig.add_trace(go.Heatmap(z=traversability_masked, x=x_coords, y=y_coords,
                              colorscale=[[0, "#e41a1c"], [1, "#4daf4a"]], zmin=0, zmax=1,
                              name="Traversability", visible="legendonly", showscale=False))

    if path_world:
        path_x = [p[0] for p in path_world]
        path_y = [p[1] for p in path_world]
        fig.add_trace(go.Scatter(x=path_x, y=path_y, mode="lines+markers",
                                  line=dict(color="blue", width=3), marker=dict(size=3),
                                  name="Planned path", visible=True))

    fig.update_layout(
        title="Interactive 2.5D Composite (click legend to toggle layers)",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(title="Layers (click to toggle)", itemclick="toggle", itemdoubleclick="toggleothers"),
        width=900, height=800,
    )
    fig.write_html(output_path)
    print("Interactive composite dashboard written to:", output_path)
    return output_path


def launch_open3d_viewer(xyz, semantic_labels):
    """Interactive 3D point-cloud viewer. Import kept lazy/optional (#18)
    so the rest of the pipeline still runs in environments without
    Open3D installed."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed - skipping interactive 3D viewer.")
        return

    from grid import create_semantic_colors
    colors = create_semantic_colors(semantic_labels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])