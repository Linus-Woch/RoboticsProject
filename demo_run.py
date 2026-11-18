# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
demo_run.py  --  the smallest possible example of using the lab
================================================================
Creates the environment, prints what it gives you, and steps it with a do-nothing action. Read
this file to see how an environment is created, how an action is built, and where the
observations are; then copy what you need into your own controller.

Usage:
    python demo_run.py --task wipe --steps 20                       # Track 1 set-up (drive to the table)
    python demo_run.py --task wipe --phase manipulation_only --cameras --steps 20   # Track 2 set-up
    python demo_run.py --task wipe --human-mode fast --render       # watch it (window; slower)

`--task nut` is a second task that is NOT used in this project.
"""

import argparse

import numpy as np

from mobile_manip import make_env, split_observations, NAV_CAMERA, WRIST_CAMERA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["wipe", "nut"], default="wipe", help="'wipe' is the task of this project")
    parser.add_argument("--controller", default="BASIC",
                        help="leave as BASIC (operational-space arm control + base velocity command)")
    parser.add_argument(
        "--phase", choices=["full", "manipulation_only"], default="full",
        help="'full' = the whole lab, the robot starts far from the task table (Track 1). "
             "'manipulation_only' = the robot is already parked at the task table (Track 2).",
    )
    parser.add_argument("--human-mode", choices=["none", "slow", "fast"], default="none",
                        help="walking person: none, slow (0.1 m/s) or fast (0.3 m/s)")
    parser.add_argument("--person", action="store_true", help="same as --human-mode fast")
    parser.add_argument("--cameras", action="store_true",
                        help="switch on the navigation camera and the wrist camera (RGB-D). Off by default: "
                             "rendering images is the slowest part of the simulation.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="open a window (slower)")
    args = parser.parse_args()

    # 1. Create the environment. `make_env` is the only function you need to build it.
    env = make_env(
        task=args.task,
        controller=args.controller,
        mobile_phase=(args.phase == "full"),
        human_mode="fast" if (args.person and args.human_mode == "none") else args.human_mode,
        cameras=[NAV_CAMERA, WRIST_CAMERA] if args.cameras else None,
        has_renderer=args.render,
    )

    # 2. reset() starts an episode and returns the first observation: a dictionary of NumPy arrays.
    obs = env.reset()
    mobile_obs, manip_obs, other_obs = split_observations(obs)

    print("\nObservations for the MOBILE side (Track 1):")
    for k, v in mobile_obs.items():
        print(f"    {k:30s} shape={np.shape(v)}")
    print("Observations for the MANIPULATION side (Track 2):")
    for k, v in manip_obs.items():
        print(f"    {k:30s} shape={np.shape(v)}")
    print("(print(obs.keys()) lists all of them.)")

    # 3. The action is ONE vector. Which entries drive what? Do not hard-code the numbers:
    split = env.robots[0].composite_controller._action_split_indexes
    arm, torso, base = split["right"], split["torso"], split["base"]
    print(f"\nThe action has {env.action_dim} entries:")
    print(f"    arm   entries {arm[0]}..{arm[1] - 1}: [dx, dy, dz, d_rx, d_ry, d_rz], a delta pose of the tool")
    print(f"    torso entry   {torso[0]}: unused, keep it 0")
    print(f"    base  entries {base[0]}..{base[1] - 1}: [vx, vy, wz], see world_velocity_mps_to_base_command")
    print("    (no gripper entry: the wiping tool has no fingers)")

    # 4. The control loop. One env.step() = one control step = 0.05 s (20 Hz).
    print(f"\nRunning {args.steps} steps with a zero action (replace it with your controller)...")
    for t in range(args.steps):
        action = np.zeros(env.action_dim)       # all zeros = do nothing; fill in the entries you control
        obs, reward, done, info = env.step(action)
        if args.render:
            env.render()
        if done:                                # the horizon was reached (or the wipe episode ended early)
            obs = env.reset()

    # env.true_base_pose() is the GROUND TRUTH pose: use it to evaluate, never inside a controller.
    print(f"\nFinal localization error (odometry vs ground truth): "
          f"{np.linalg.norm(obs['robot0_base_odom'][:2] - env.true_base_pose()[:2]):.3f} m")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
