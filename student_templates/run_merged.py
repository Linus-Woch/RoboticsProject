# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
student_templates/run_merged.py  --  RUN BOTH TRACKS ONE AFTER THE OTHER
==========================================================================
The complete application of the handout: drive to the task table (your MobileController), then
wipe the dirt (your ManipulationController). You do NOT need this file to do your own track: the
mobile controller runs on its own with `python student_templates/mobile_controller.py` and the
manipulation controller with `python student_templates/manipulation_controller.py`. This script
shows how the two fit together, in case two groups want to try their work together.

The only thing the two sides share is one signal. The mobile side says "arrived" when, by its own
estimate, the base is within HANDOFF_TOLERANCE of the goal and facing the table
(`MobileController.at_goal`). From then on the manipulation side takes over, with the base
standing still and the arm still in its starting pose, exactly as in Track 2.

    python student_templates/run_merged.py --human-mode slow

With the placeholder controllers nothing happens (the base never arrives): fill them in first.
Results are written to student_templates/trajectory.png (true vs estimated base path) and to
plots/merged_wipe/ (cleaning progress).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mobile_manip import make_env, ContactMonitor, WipeProgressRecorder, WRIST_CAMERA
from mobile_controller import MobileController, evaluate_run
from manipulation_controller import ManipulationController

# How close to the goal the base must be before the arm starts working: the same 5 cm as the
# Track 1 success test. A base that stops 12-15 cm short makes the arm stretch for the table, and
# then the episode often ends early on a joint limit.
HANDOFF_TOLERANCE = 0.05

HORIZON = 3000      # steps: the default 1000 (50 s) is too short for driving AND wiping
MAX_STEPS = 2500


def main():
    parser = argparse.ArgumentParser(description="Navigate to the task table, then wipe the dirt.")
    parser.add_argument("--human-mode", choices=["none", "slow", "fast"], default="none",
                        help="walking person: none, slow (0.1 m/s) or fast (0.3 m/s)")
    parser.add_argument("--camera-size", type=int, default=256, help="wrist camera image size in pixels")
    args = parser.parse_args()

    env = make_env(task="wipe", mobile_phase=True, human_mode=args.human_mode,
                   cameras=[WRIST_CAMERA], camera_size=args.camera_size, horizon=HORIZON)
    obs = env.reset()

    split = env.robots[0].composite_controller._action_split_indexes
    base_start, base_end = split["base"]       # entries 7..9: base command [vx, vy, wz]
    arm_start, arm_end = split["right"]        # entries 0..5: arm delta pose

    goal_xy = env.manipulation_base_xy()
    mobile = MobileController(goal_xy=goal_xy)
    manip = ManipulationController(env)

    monitor = ContactMonitor(env)
    recorder = WipeProgressRecorder(env)
    arm_start_q = obs["robot0_joint_pos"].copy()     # the arm must stay still while the base drives
    arm_drift = 0.0
    phase = "navigate"
    true_traj, est_traj = [], []
    for t in range(MAX_STEPS):
        action = np.zeros(env.action_dim)                      # every entry you do not set stays 0

        if phase == "navigate":
            action[base_start:base_end] = mobile.compute_action(obs)
            if mobile.at_goal(obs, tolerance=HANDOFF_TOLERANCE):
                phase = "manipulate"
                print(f"t={t}: arrived (within {HANDOFF_TOLERANCE} m of the goal and facing the table), handing over to the arm")
                print(f"arm: no joint moved more than {arm_drift:.3f} rad ({np.degrees(arm_drift):.2f} deg) during the drive")
                print("Track 1 result of the drive (true pose, steps in contact so far):")
                evaluate_run(env, goal_xy, monitor)       # the drive is judged here, before the arm starts
        else:
            action[arm_start:arm_end] = np.clip(manip.compute_action(obs), -1, 1)

        obs, reward, done, info = env.step(action)
        monitor.update()
        if phase == "navigate":
            arm_drift = max(arm_drift, float(np.max(np.abs(obs["robot0_joint_pos"] - arm_start_q))))
        true_traj.append(env.true_base_pose()[:2].copy())      # ground truth: for the plot only
        est_traj.append(mobile.estimate_pose(obs)[:2].copy())
        if phase == "manipulate":
            recorder.update(obs, done)
            if manip.is_done(obs):
                print(f"t={t}: the manipulation controller says it is done")
                break
        if done:
            break

    print(f"\nlast phase: {phase}")
    if phase == "navigate":
        pose = mobile.estimate_pose(obs)
        print(f"  never handed over: by your estimate the base ended {np.linalg.norm(pose[:2] - goal_xy):.3f} m from the goal "
              f"at yaw {np.degrees(pose[2]):.1f} deg. The hand-over needs < {HANDOFF_TOLERANCE} m and |yaw| < 10 deg.")
    print(f"localisation error at the end (estimate vs ground truth): "
          f"{np.linalg.norm(mobile.estimate_pose(obs)[:2] - env.true_base_pose()[:2]):.3f} m")
    print("contacts over the whole run (the tool touching the task table while it wipes is normal):")
    print(monitor.report())
    if recorder.step:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "plots", "merged_wipe")
        print(recorder.summary())
        print("cleaning-progress figure saved to", recorder.save(out_dir, title="integrated run (your controllers)"))
    env.close()

    _plot_trajectories(np.array(true_traj), np.array(est_traj))


def _plot_trajectories(true_traj, est_traj):
    """Plots the true and the estimated base path: the kind of figure you need for the
    localisation part of the Track 1 report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib is not installed: skipping the trajectory plot; `pip install matplotlib`)")
        return

    plt.figure(figsize=(6, 6))
    plt.plot(true_traj[:, 0], true_traj[:, 1], label="true path (ground truth)")
    plt.plot(est_traj[:, 0], est_traj[:, 1], label="estimated path (your controller)", linestyle="--")
    plt.scatter(*true_traj[0], c="green", marker="o", label="start", zorder=5)
    plt.scatter(*true_traj[-1], c="red", marker="x", label="end", zorder=5)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Base path: true vs estimated")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), "trajectory.png")
    plt.savefig(out_path, dpi=150)
    print(f"trajectory plot saved to {out_path}")


if __name__ == "__main__":
    main()
