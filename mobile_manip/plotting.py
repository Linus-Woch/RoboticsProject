# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.plotting
=======================
Top-down (bird's-eye) 2D plotting helpers: the physical scene (tables and
obstacles, drawn at their true footprint size) with the robot's trajectory
overlaid, and a second version with the robot's own footprint stamped along
the path so overlaps with obstacles/tables -- i.e. collisions -- are
visually obvious. Used by the instructor-only solution scripts to sanity
check navigation, not part of the student-facing scaffold.
"""

import os

import numpy as np


def save_scene_and_trajectory_plots(
    name,
    trajectory_xy,
    rectangles,
    footprint_radius,
    scene_dir,
    collision_dir,
    title="Scene",
    n_footprint_stamps=25,
    polylines=None,
    keepouts=None,
    goal_xy=None,
    contact_points=None,
):
    """
    Args:
        name: file name stem (e.g. "mobile_nut").
        trajectory_xy: (T, 2) array of the robot's true [x, y] over time.
        rectangles: list of (center_xy, size_xy, label, color) -- tables and
            obstacles, drawn at true size as axis-aligned rectangles.
        footprint_radius: the robot base's conservative bounding-circle
            radius (see MobilePandaBase.lidar_radius in robots.py -- this is
            the SAME validated "clears the base's own geometry" radius, so
            an overlap here is a genuine, conservative potential-collision
            signal, not a guess).
        scene_dir: folder for the plain scene+trajectory plot.
        collision_dir: SEPARATE folder for the footprint-overlay plot.
        n_footprint_stamps: how many footprint circles to draw along the path.
        polylines: optional list of (points_xy, label, color) drawn as lines
            (e.g. the walking person's patrol path).
        goal_xy: optional (x, y) navigation goal, drawn as a star.
        contact_points: optional list of (x, y) where the robot touched a bin, the
            person or a table (see `ContactMonitor.contact_points()`), drawn as red crosses.
        keepouts: optional list of (center_xy, size_xy, label) drawn as dashed,
            unfilled rectangles (zones that must stay clear, e.g. the
            manipulation zone in front of the task table).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    traj = np.asarray(trajectory_xy)

    def _draw_rectangles(ax):
        for points, label, color in (polylines or []):
            pts = np.asarray(points)
            ax.plot(pts[:, 0], pts[:, 1], linestyle="--", marker="|", color=color,
                    linewidth=2, alpha=0.9, label=label, zorder=3)
        for center_xy, size_xy, label in (keepouts or []):
            cx, cy = center_xy
            w, h = size_xy
            ax.add_patch(patches.Rectangle(
                (cx - w / 2, cy - h / 2), w, h, fill=False, linestyle=":",
                linewidth=2, edgecolor="green", label=label, zorder=2))
        for center_xy, size_xy, label, color in rectangles:
            cx, cy = center_xy
            w, h = size_xy
            rect = patches.Rectangle(
                (cx - w / 2, cy - h / 2), w, h,
                linewidth=1, edgecolor="black", facecolor=color, alpha=0.6, label=label,
            )
            ax.add_patch(rect)

    def _finish(ax, fig, out_dir, filename, plot_title):
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5, label="robot trajectory", zorder=4)
        ax.scatter(*traj[0], c="green", marker="o", s=70, label="start", zorder=5)
        ax.scatter(*traj[-1], c="red", marker="x", s=70, label="end", zorder=5)
        if contact_points is not None and len(contact_points) > 0:
            cp = np.asarray(contact_points)
            ax.scatter(cp[:, 0], cp[:, 1], c="red", marker="P", s=90, edgecolors="black", label="collision", zorder=7)
        if goal_xy is not None:
            ax.scatter(*goal_xy, c="gold", edgecolors="black", marker="*", s=220, label="goal (manipulation pose)", zorder=6)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(plot_title)
        # de-duplicate legend entries (rectangles of the same type repeat the label)
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        # outside the axes so it never covers the trajectory/patrol path
        ax.legend(seen.values(), seen.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        ax.grid(True, alpha=0.25)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {path}")

    # Plot 1: scene (true-size objects) + trajectory
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_rectangles(ax)
    _finish(ax, fig, scene_dir, f"{name}_scene_trajectory.png", title)

    # Plot 2 (separate folder): trajectory with the robot's own footprint
    # stamped along the path, to make overlaps with tables/obstacles -- i.e.
    # collisions -- visually obvious.
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    _draw_rectangles(ax2)
    stamp_idxs = np.linspace(0, len(traj) - 1, min(n_footprint_stamps, len(traj))).astype(int)
    for i in stamp_idxs:
        ax2.add_patch(patches.Circle(traj[i], footprint_radius, color="orange", alpha=0.10, zorder=1))
    # one labeled patch for the legend
    ax2.add_patch(patches.Circle(
        traj[stamp_idxs[-1]], footprint_radius, color="orange", alpha=0.10,
        zorder=1, label=f"base footprint (r={footprint_radius:.2f}m)",
    ))
    _finish(ax2, fig2, collision_dir, f"{name}_trajectory_footprint.png",
            f"{title} -- base footprint along path (overlap with a rectangle = possible collision)")
