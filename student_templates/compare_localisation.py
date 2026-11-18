"""Replay odometry and LiDAR logs and compare both estimates on identical scans."""

import argparse
from pathlib import Path

import numpy as np

from mobile_controller import make_env
from mobile_manip.localisation import LidarMap, PoseEKF, wrap_angle


def compare(trace, lidar_map, output):
    rows = np.atleast_1d(np.genfromtxt(trace, delimiter=",", names=True))
    odometry = np.column_stack([rows[name] for name in ("odom_x_m", "odom_y_m", "odom_yaw_rad")])
    truth = np.column_stack([rows[name] for name in ("true_x_m", "true_y_m", "true_yaw_rad")])
    scan_columns = [name for name in rows.dtype.names if name.startswith("lidar_")]
    scans = np.column_stack([rows[name] for name in scan_columns])
    ekf = PoseEKF(lidar_map)
    estimates = []
    used_counts = []
    for odom, scan in zip(odometry, scans):
        estimates.append(ekf.update(odom, scan))
        used_counts.append(ekf.used_beams.sum())

    position_errors = []
    heading_errors = []
    for label, poses in (("Odometry", odometry), ("EKF", np.array(estimates))):
        xy_error = np.linalg.norm(poses[:, :2] - truth[:, :2], axis=1)
        yaw_error = np.degrees(wrap_angle(poses[:, 2] - truth[:, 2]))
        position_errors.append(xy_error)
        heading_errors.append(np.abs(yaw_error))
        near_goal = rows["goal_distance_m"] < 0.25
        near_text = f", near-goal position RMSE {np.sqrt(np.mean(xy_error[near_goal] ** 2)):.3f} m" if near_goal.any() else ""
        print(f"{label}: position RMSE {np.sqrt(np.mean(xy_error ** 2)):.3f} m, "
              f"heading RMSE {np.sqrt(np.mean(yaw_error ** 2)):.2f} deg, "
              f"final position error {xy_error[-1]:.3f} m{near_text}")
    print(f"Accepted beams per observation: {np.mean(used_counts):.1f} on average")
    plot_comparison(rows["time_s"], position_errors, heading_errors, lidar_map, odometry[0], scans[0], output)


def plot_comparison(times, position_errors, heading_errors, lidar_map, start_pose, first_scan, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (scan_ax, xy_ax, yaw_ax) = plt.subplots(3, 1, figsize=(8, 8), layout="constrained")
    predicted, _, _ = lidar_map.predict(start_pose, len(first_scan))
    beams = np.arange(len(first_scan))
    scan_ax.plot(beams, predicted, "o-", label="Predicted from map")
    scan_ax.plot(beams, np.where(first_scan > 0, first_scan, np.nan), "x", label="Measured")
    scan_ax.set(xlabel="LiDAR beam", ylabel="Range (m)", title="LiDAR scan at the known start pose")
    for label, position, heading in zip(("Odometry", "EKF"), position_errors, heading_errors):
        xy_ax.plot(times, position, label=label)
        yaw_ax.plot(times, heading, label=label)
    xy_ax.axhline(0.05, linestyle=":", color="gray", label="5 cm reference")
    xy_ax.set(xlabel="Time (s)", ylabel="Position error (m)")
    yaw_ax.set(xlabel="Time (s)", ylabel="Heading error (deg)")
    for axis in (scan_ax, xy_ax, yaw_ax):
        axis.legend()
        axis.grid(alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"Saved comparison to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="CSV produced by --log-csv")
    parser.add_argument("--output", type=Path, default=Path("plots/localisation_comparison.png"))
    args = parser.parse_args()

    env = make_env(task="wipe", mobile_phase=True, human_mode="none")
    try:
        env.reset()
        lidar_map = LidarMap(env.wall_segments(), env.table_map())
    finally:
        env.close()
    compare(args.trace, lidar_map, args.output)


if __name__ == "__main__":
    main()
