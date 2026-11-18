# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
student_templates/mobile_controller.py  --  TRACK 1: MOBILE ROBOTICS
=======================================================================
Drive the base from its start pose (about (0.15, 2.0) m) to the manipulation pose in front of
the task table, using only the robot's sensors, without touching the bins, the tables or the
walking person. The arm stays still. The handout explains the task, the sensors and the
success test; this file is where your code goes.

WHAT TO DO
----------
Fill in the two methods of `MobileController` below (search for "TODO"):

  1. `estimate_pose(obs)`: where is the robot? Start by returning the odometry; replace it
     with something that also uses the LIDAR (for example a filter that compares the LIDAR
     ranges with the wall map) so the estimate does not drift.
  2. `compute_action(obs)`: where should it go? Turn your pose estimate and the LIDAR into
     the base command `[vx, vy, wz]` that drives the robot to `self.goal_xy` while avoiding
     obstacles, and stops there.

Run this file to try it:

    python student_templates/mobile_controller.py --human-mode none --steps 400 --log-csv plots/odometry_baseline.csv
    python student_templates/mobile_controller.py --human-mode fast --render    # watch it

At the end it prints the success test of the handout (distance to the goal, yaw, contact steps).
Do your experiments without --render: it is much faster.

WHAT YOU MAY USE INSIDE YOUR CONTROLLER
---------------------------------------
    The start pose of the base: (0.15, 2.0) m, facing +x (yaw 0), the same in every episode. The first
    odometry reading is exactly this pose, so start your estimate there. After that, only the sensors
    below (LIDAR and odometry). You may also hard-code what you know about the layout (for example part
    of the route), but the bins change at every reset and the person moves, so a fixed route can fail.
    obs["robot0_base_lidar"]   (16,) LIDAR ranges in metres. -1 = no hit; 5.0 = range limit
                               (nothing within 5 m, NOT a hit at 5 m). Beam i points at
                               yaw + i*22.5 degrees and starts on a ring of radius 0.25 m
                               around the base origin, at height 0.5 m. The ring is above
                               the floor, so it sees bins, the person, walls and the tables'
                               skirts, but nothing lower than 0.5 m.
    obs["robot0_base_odom"]    (3,) NOISY, DRIFTING pose [x, y, yaw] (m, m, rad).
    obs["mobilebase0_nav_image"], obs["..._depth"]   optional forward camera: make the env with
                               cameras=[NAV_CAMERA]. The depth is NOT in metres: convert it
                               with `mobile_manip.depth_to_meters`.
    env.wall_segments()        (6, 2, 2) the arena walls as line segments: a map to localise with.
                               (The walls are solid: do not drive into them.)
    env.table_map()            the two table footprints (centre, size, skirt inset): a known map.
    env.manipulation_base_xy() the goal (also given to you as `self.goal_xy`).
    env.manipulation_keepout() (centre, size) of the strip in front of the task table that must
                               stay free.

NOT ALLOWED INSIDE YOUR CONTROLLER (it is ground truth, it would defeat the point)
    env.true_base_pose(), obs["robot0_base_pos"], obs["robot0_base_quat"], anything computed
    from them (for example robot0_eef_pos minus robot0_base_to_eef_pos), obs["robot0_proprio-state"]
    (it contains them), and anything read from env.sim (bin or person positions).
    The true pose is for the evaluation at the end of a run and for plots.

DRIVING THE BASE  (this is not what it looks like: read it once)
-----------------------------------------------------------------
  * The command [vx, vy] is NOT in the robot frame and is NOT a velocity: the simulator rotates
    it by -yaw and then ignores the first 0.25 of every world axis (a "dead zone"). A command of
    0.30 gives 0.05 m/s, 1.00 gives 0.74 m/s.
  * So do not write [vx, vy] by hand. Decide the velocity you want in the WORLD frame in m/s
    and convert it:
        vx, vy = world_velocity_mps_to_base_command(v_world, yaw)
    The largest velocity this can give is about 0.45 m/s per world axis.
  * The base is holonomic: it can move in any direction without turning. wz is a yaw-RATE command
    (1.0 is about 76 degrees per second). With wz = 0 the yaw simply stays where it is: on open floor it
    stayed within 0.1 degrees over 24 m. It only changes if the robot is pushed (a contact), so send a
    small feedback term to undo that, for example wz = -1.0 * yaw (it is 0 when the yaw is 0). Turning is harmful: the arm does not follow the base,
    and the manipulation phase expects the arm to face the task table.
  * The chassis is a BOX, not a circle: see CHASSIS_X / CHASSIS_Y below. Measure your clearances
    to the box.
  * The action you return has 3 entries; the code that calls you puts them in entries 7-9 of
    the 10-entry action vector, and leaves the arm and torso entries at 0.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mobile_manip import make_env, ContactMonitor, world_velocity_mps_to_base_command

# ---- Robot geometry, in the ROBOT frame (x forward, y left, origin at the base origin) ----
LIDAR_RING_RADIUS = 0.25        # m: beams start on a ring of this radius
LIDAR_MAX_RANGE = 5.0           # m: a reading of exactly this value means "no hit within range"
CHASSIS_X = (-0.55, 0.15)       # m: the chassis reaches 0.55 m BEHIND the origin, only 0.15 m ahead
CHASSIS_Y = (-0.25, 0.25)       # m: and 0.25 m to each side


def lidar_points(lidar, x, y, yaw):
    """World-frame (x, y) of every LIDAR beam that hit something, shape (n_hits, 2).

    Use your ESTIMATED pose inside the controller. With the true pose this is handy to check
    your localisation (do the points land on the walls and bins?)."""
    theta = yaw + np.arange(len(lidar)) * 2 * np.pi / len(lidar)          # direction of each beam
    hit = (lidar > 0) & (lidar < LIDAR_MAX_RANGE)                         # -1: no hit, 5.0: out of range
    ox = x + LIDAR_RING_RADIUS * np.cos(theta)                            # where each beam starts
    oy = y + LIDAR_RING_RADIUS * np.sin(theta)
    px, py = ox + lidar * np.cos(theta), oy + lidar * np.sin(theta)       # where it ends
    return np.stack([px[hit], py[hit]], axis=1)


def plot_lidar(lidar, x, y, yaw, walls=None, ax=None):
    """Draws one LIDAR scan seen from above, to check what the sensor really gives you.

        plot_lidar(obs["robot0_base_lidar"], *obs["robot0_base_odom"], walls=env.wall_segments())
        import matplotlib.pyplot as plt; plt.show()          # or plt.savefig("scan.png")

    Blue line + red dot = a beam that hit something (the dot is the hit point); dotted grey line = a
    beam with no hit within range (-1 or 5.0). The box is the chassis, the arrow its heading.
    `walls` is optional: pass env.wall_segments() to see the wall map. Returns the matplotlib axes."""
    import matplotlib.pyplot as plt
    from matplotlib import patches
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    n = len(lidar)
    theta = yaw + np.arange(n) * 2 * np.pi / n
    for i in range(n):
        u = np.array([np.cos(theta[i]), np.sin(theta[i])])
        o = np.array([x, y]) + LIDAR_RING_RADIUS * u
        if 0 < lidar[i] < LIDAR_MAX_RANGE:
            end = o + lidar[i] * u
            ax.plot([o[0], end[0]], [o[1], end[1]], color="tab:blue", lw=0.8, alpha=0.7)
            ax.plot(end[0], end[1], "o", color="red", ms=4, zorder=3)
        else:
            end = o + LIDAR_MAX_RANGE * u
            ax.plot([o[0], end[0]], [o[1], end[1]], color="gray", lw=0.6, ls=":")
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[CHASSIS_X[0], CHASSIS_Y[0]], [CHASSIS_X[1], CHASSIS_Y[0]],
                        [CHASSIS_X[1], CHASSIS_Y[1]], [CHASSIS_X[0], CHASSIS_Y[1]]])
    ax.add_patch(patches.Polygon(np.array([x, y]) + corners @ R.T, closed=True, fc="#dddddd", ec="black", zorder=2))
    ax.annotate("", xy=(x + 0.35 * c, y + 0.35 * s), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color="black"), zorder=4)
    if walls is not None:
        for w in np.asarray(walls):
            ax.plot(w[:, 0], w[:, 1], color="black", lw=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)
    return ax


class MobileController:
    def __init__(self, goal_xy):
        self.goal_xy = np.asarray(goal_xy, dtype=float)
        # TODO: put the state of your estimator and planner here (filter covariance, the
        # previous estimate, a map, a path, a counter for "how long have I been stuck", ...).

    def estimate_pose(self, obs):
        """Returns your best estimate of [x, y, yaw] in the world frame.

        Baseline: the raw odometry. It drifts: the longer and farther you drive, the worse it
        gets. Replace it with an estimator that uses obs["robot0_base_lidar"] as a correction,
        for example by predicting each beam's range from the wall map (`env.wall_segments()`, so
        pass the map to __init__) and comparing it with what the LIDAR measured. Beams that hit
        bins, the person or the tables are not in the wall map: they are outliers for such a
        filter, so you have to reject them.

        NOTE: this method is called several times per control step (by compute_action and by
        at_goal). If your estimator has state, update it ONCE per step, not once per call.
        """
        return np.asarray(obs["robot0_base_odom"], dtype=float)

    def compute_action(self, obs):
        """Returns the base command [vx, vy, wz] for this step.

        Baseline: drive straight toward the goal at up to 0.3 m/s using odometry.
        This is the handout's reference controller for comparing later improvements.
        """
        pose = self.estimate_pose(obs)
        to_goal = self.goal_xy - pose[:2]
        distance = np.linalg.norm(to_goal)
        # A gain of 1/s slows the base inside 0.3 m and gives zero translation at the goal.
        v_world = to_goal * (0.3 / max(distance, 0.3))
        vx, vy = world_velocity_mps_to_base_command(v_world, pose[2])
        yaw_error = (pose[2] + np.pi) % (2 * np.pi) - np.pi
        wz = np.clip(-yaw_error, -1.0, 1.0)
        return np.array([vx, vy, wz])

    def at_goal(self, obs, tolerance=0.05, yaw_tolerance=np.radians(10.0)):
        """True when, according to YOUR estimate, the robot is within `tolerance` metres of the goal
        and facing the table (|yaw| < `yaw_tolerance`).

        These defaults are the success test of the handout, but evaluated with your estimate
        instead of the true pose; the true pose is only used for the final evaluation. The yaw
        check matters: the manipulation phase starts with the arm facing the task table."""
        pose = self.estimate_pose(obs)
        yaw_error = (pose[2] + np.pi) % (2 * np.pi) - np.pi
        return bool(np.linalg.norm(pose[:2] - self.goal_xy) < tolerance and abs(yaw_error) < yaw_tolerance)


def evaluate_run(env, goal_xy, monitor):
    """The Track 1 success test of the handout, computed with the TRUE pose (allowed here: this is
    the evaluation, not the controller). Prints the numbers and returns True for a success."""
    x, y, yaw = env.true_base_pose()
    dist = float(np.linalg.norm(np.array([x, y]) - goal_xy))
    yaw_deg = float(np.degrees((yaw + np.pi) % (2 * np.pi) - np.pi))
    contact_steps = monitor.total_steps()
    success = dist < 0.05 and abs(yaw_deg) < 10.0 and contact_steps == 0
    print(f"distance to the goal: {dist:.3f} m (needs < 0.05)   yaw: {yaw_deg:.1f} deg (needs |yaw| < 10)   "
          f"steps in contact: {contact_steps} (needs 0)")
    print("SUCCESS" if success else "not a success (yet)")
    return success


def odometry_log_row(step, env, obs, goal_xy, monitor, arm_drift):
    """Capture one observation and its ground truth for offline evaluation."""
    odom = obs["robot0_base_odom"]
    truth = env.true_base_pose()
    yaw_error = (odom[2] - truth[2] + np.pi) % (2 * np.pi) - np.pi
    return (
        step, 0.05 * step, *odom, *truth,
        np.linalg.norm(odom[:2] - truth[:2]), abs(np.degrees(yaw_error)),
        np.linalg.norm(truth[:2] - goal_xy), monitor.total_steps(), arm_drift,
        *obs["robot0_base_lidar"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run your MobileController in the lab and evaluate it.")
    parser.add_argument("--human-mode", choices=["none", "slow", "fast"], default="none",
                        help="walking person: none, slow (0.1 m/s) or fast (0.3 m/s)")
    parser.add_argument("--steps", type=int, default=1200,
                        help="control steps to run (1 step = 0.05 s). With a person (the long route) 1200 is about right")
    parser.add_argument("--render", action="store_true", help="open a window (slower; for debugging)")
    parser.add_argument("--plot-lidar", action="store_true",
                        help="save a picture of the last LIDAR scan to plots/lidar_scan.png (uses your pose estimate)")
    parser.add_argument("--log-csv", type=Path,
                        help="save per-step odometry, LiDAR, ground truth, and evaluation metrics")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    env = make_env(task="wipe", mobile_phase=True, human_mode=args.human_mode, has_renderer=args.render,
                   horizon=max(1500, args.steps + 10))
    obs = env.reset()
    base_start, base_end = env.robots[0].composite_controller._action_split_indexes["base"]   # entries 7..9
    goal_xy = env.manipulation_base_xy()
    controller = MobileController(goal_xy=goal_xy)
    monitor = ContactMonitor(env)          # counts the steps in contact with bins / person / tables
    arm_start_q = obs["robot0_joint_pos"].copy()      # the arm must stay still: remember where it starts
    arm_drift = 0.0                                   # largest change of any arm joint so far (rad)
    log_rows = []
    if args.log_csv:
        log_rows.append(odometry_log_row(0, env, obs, goal_xy, monitor, arm_drift))

    for step in range(1, args.steps + 1):
        action = np.zeros(env.action_dim)                                  # arm and torso stay at 0
        action[base_start:base_end] = controller.compute_action(obs)       # [vx, vy, wz]
        obs, reward, done, info = env.step(action)
        monitor.update()
        arm_drift = max(arm_drift, float(np.max(np.abs(obs["robot0_joint_pos"] - arm_start_q))))
        if args.log_csv:
            log_rows.append(odometry_log_row(step, env, obs, goal_xy, monitor, arm_drift))
        if args.render:
            env.render()
        if controller.at_goal(obs):
            print(f"arrived (by your estimate) at step {step} = {0.05 * step:.2f} s")
            break
        if done:
            break

    est, true = controller.estimate_pose(obs), env.true_base_pose()
    print(f"run ended after {step} steps ({0.05 * step:.2f} s simulated)")
    print(f"estimated pose {np.round(est, 3)}   true pose {np.round(true, 3)}   "
          f"localisation error {np.linalg.norm(est[:2] - true[:2]):.3f} m")
    print(monitor.report())
    print(f"arm: no joint moved more than {arm_drift:.3f} rad ({np.degrees(arm_drift):.2f} deg) from its starting value")
    evaluate_run(env, goal_xy, monitor)
    if args.log_csv:
        columns = [
            "step", "time_s", "odom_x_m", "odom_y_m", "odom_yaw_rad",
            "true_x_m", "true_y_m", "true_yaw_rad", "position_error_m",
            "heading_error_deg", "goal_distance_m", "contact_steps", "max_arm_drift_rad",
        ]
        columns.extend(f"lidar_{beam}_m" for beam in range(len(obs["robot0_base_lidar"])))
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.log_csv, log_rows, delimiter=",", header=",".join(columns), comments="", fmt="%.9g")
        print(f"odometry trace saved to {args.log_csv}")
    if args.plot_lidar:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(out_dir, exist_ok=True)
        ax = plot_lidar(obs["robot0_base_lidar"], *est, walls=env.wall_segments())
        ax.set_title("LIDAR scan at the end of the run (your pose estimate)")
        plt.savefig(os.path.join(out_dir, "lidar_scan.png"), dpi=150, bbox_inches="tight")
        print("LIDAR scan saved to", os.path.join(out_dir, "lidar_scan.png"))
    env.close()


if __name__ == "__main__":
    main()
