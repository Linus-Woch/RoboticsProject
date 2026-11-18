# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
student_templates/manipulation_controller.py  --  TRACK 2: MANIPULATION
=========================================================================
The robot is parked in front of the task table. A winding trail of dirt (100 spots, 4 cm each)
lies on the tabletop, in a different place and shape at every reset. Find it with the wrist
camera and wipe it off, as completely and as quickly as possible. The handout explains the
task, the controller and the reward; this file is where your code goes.

WHAT TO DO
----------
Fill in `ManipulationController.compute_action` (search for "TODO"). It is called once per
control step (0.05 s) and returns the arm action: six numbers in [-1, 1]. A typical structure:
  1. LOOK: move to a pose from where the camera sees the whole trail, and find the dirt in the
     image (`detect_dirt` below does that for you; read what it does).
  2. PLAN: decide the ORDER and the PATH in which to visit the dirt (this is the interesting part:
     a circle around the middle of the dirt misses most of a winding trail).
  3. WIPE: follow the path with the tool pressed on the table, without pressing too hard and
     without hitting a joint limit (the episode ends early if you do).

Run this file to try it (a picture of what you cleaned is saved at the end):

    python student_templates/manipulation_controller.py --steps 700
    python student_templates/manipulation_controller.py --steps 700 --render       # watch it

WHAT YOU MAY USE
----------------
    obs["robot0_eef_pos"], obs["robot0_eef_quat"]   tool pose in the world frame (quaternion x, y, z, w)
    obs["robot0_base_to_eef_pos"]                   tool position in the robot base frame
    obs["robot0_joint_pos"], obs["robot0_joint_vel"]   the 7 arm joints (rad, rad/s)
    obs["robot0_contact"]                           True while the tool touches something
    obs["robot0_eye_in_hand_image"], obs["..._depth"]  wrist camera (needs cameras=[WRIST_CAMERA])
    obs["proportion_wiped"]                         fraction of the dirt cleaned, 0..1: for
                                                    logging, plots and deciding when to stop

NOT ALLOWED to locate the dirt: obs["wipe_centroid"], obs["wipe_radius"],
obs["gripper_to_wipe_centroid"], obs["object-state"] (they tell you where the dirt is without
looking), or anything read from env.sim / env.model. The dirt positions your planner uses must
come from the camera image.

THE ACTION
----------
`env.action_dim` is 10. Entries 0-5 move the arm, entry 6 (torso) and entries 7-9 (base) must
stay 0, and there is no gripper entry (the wiping tool has no fingers). You only return the six
arm numbers; the code that calls you puts them in entries 0-5.

    [dx, dy, dz, d_rx, d_ry, d_rz]    each in [-1, 1]

It is a DELTA command in the robot base frame: the controller aims at a pose a small step away
from where the tool is NOW (up to 5 cm and 0.5 rad), and pulls the tool there like a spring.
So 0 means "hold here", and a small command repeated every step moves the tool slowly but
steadily. With the base at yaw 0 the base axes are parallel to the world axes, so an error
measured in world coordinates can be used directly:

    action = np.clip(10.0 * (target_xyz - obs["robot0_eef_pos"]), -1, 1)    # [dx, dy, dz]

(gain 10: an error of 0.1 m gives the full command 1). Keep the three rotation entries at 0 to
keep the tool flat on the table.

THE CAMERA
----------
  * robosuite returns images upside down: flip the image AND the depth with [::-1].
  * The depth is NOT in metres: convert it with `mobile_manip.depth_to_meters(env, depth)`.
  * `mobile_manip.pixel_to_world(env, WRIST_CAMERA, (row, col), depth_m, height, width)` gives the
    world position of one pixel (row, col of the FLIPPED image).
  * The wrist image also shows the robot's own base, the floor and the LIDAR rays that the
    renderer draws. Their red end points look like dirt to a colour test; they lie on the floor, so
    `detect_dirt` removes them with a check on the height of the points.
  * The camera loses the dirt from view when the tool is within about 6 cm of it: look from a
    standoff pose first, then plan.
  * Rendering images is the slowest part of a step: ask for a small `camera_size` if you are short
    of time (the default here is 256 pixels).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mobile_manip import (make_env, WRIST_CAMERA, WipeProgressRecorder,
                          depth_to_meters, pixel_to_world)

TABLE_Z = 0.9          # m: height of the tabletop in the world frame (the dirt lies on it)


def detect_dirt(env, obs, stride=10):
    """Finds the dirt in the wrist camera image. Returns an (N, 3) array of world (x, y, z) points
    that lie on the dirt (every `stride`-th dirt pixel, to keep it fast); (0, 3) if none is seen.

    How it works: the dirt is dark brown on a white table, so a colour test finds the dirt pixels;
    each one is turned into a world point with its depth (`pixel_to_world`); points that are not
    at table height are dropped (they are not dirt, see "THE CAMERA" above). Change the colour
    thresholds or the stride if you need to."""
    image = obs["robot0_eye_in_hand_image"][::-1]                              # (H, W, 3) upright
    depth = depth_to_meters(env, obs["robot0_eye_in_hand_depth"])[::-1]        # (H, W, 1) in metres, upright
    height, width = image.shape[:2]
    r, g, b = (image[..., k].astype(int) for k in range(3))
    dirt_mask = (r - b > 15) & (r + g + b < 450)                               # brownish and dark
    rows, cols = np.nonzero(dirt_mask)
    if len(rows) == 0:
        return np.zeros((0, 3))
    points = np.array([pixel_to_world(env, WRIST_CAMERA, (row, col), depth, height, width)
                       for row, col in zip(rows[::stride], cols[::stride])])
    return points[np.abs(points[:, 2] - TABLE_Z) < 0.03]


class ManipulationController:
    def __init__(self, env):
        self.env = env      # needed by the camera helpers (detect_dirt, depth_to_meters, pixel_to_world)
        # TODO: put your state here: the phase you are in (look / plan / wipe), the detected dirt
        # points, the planned path and which waypoint you are heading to, ...

    def compute_action(self, obs):
        """Returns the arm action [dx, dy, dz, d_rx, d_ry, d_rz], shape (6,), each in [-1, 1].

        Placeholder: holds still. Replace it with your look -> plan -> wipe logic. A first step, to
        see the pieces working (move above the middle of the dirt and stay there):

            if self.dirt_points is None:                 # (set self.dirt_points = None in __init__)
                self.dirt_points = detect_dirt(self.env, obs)
            target = np.array([*self.dirt_points[:, :2].mean(axis=0), 0.97])   # 7 cm above the table
            action = np.zeros(6)
            action[:3] = np.clip(10.0 * (target - obs["robot0_eef_pos"]), -1, 1)
            return action

        The tool pad wipes a spot when it is pressed onto the table (z of the tool close to
        TABLE_Z); 0.97 is a hovering height. Think about how to go down gently and stay down.
        """
        # TODO: compute the six numbers
        return np.zeros(6)

    def is_done(self, obs):
        """TODO: return True when you want to stop (for example when the wiping is finished).
        `obs["proportion_wiped"]` (0..1) is allowed for this. Placeholder: never."""
        return False


def main():
    parser = argparse.ArgumentParser(description="Run your ManipulationController in the lab and draw what it cleaned.")
    parser.add_argument("--steps", type=int, default=200, help="control steps to run (1 step = 0.05 s); e.g. 700 for a real attempt")
    parser.add_argument("--camera-size", type=int, default=256, help="wrist camera image size in pixels")
    parser.add_argument("--render", action="store_true", help="open a window (slower; for debugging)")
    args = parser.parse_args()

    env = make_env(task="wipe", mobile_phase=False, cameras=[WRIST_CAMERA], camera_size=args.camera_size,
                   has_renderer=args.render, horizon=max(1000, args.steps + 10))
    obs = env.reset()
    recorder = WipeProgressRecorder(env)       # records the cleaned fraction and the tool path
    arm_start, arm_end = env.robots[0].composite_controller._action_split_indexes["right"]   # entries 0..5
    controller = ManipulationController(env)

    for t in range(args.steps):
        action = np.zeros(env.action_dim)                        # base and torso stay at 0
        arm_action = np.asarray(controller.compute_action(obs), dtype=float)
        assert arm_action.shape == (6,), "compute_action must return 6 numbers: [dx, dy, dz, d_rx, d_ry, d_rz]"
        action[arm_start:arm_end] = np.clip(arm_action, -1, 1)
        obs, reward, done, info = env.step(action)
        recorder.update(obs, done)
        if args.render:
            env.render()
        if done or controller.is_done(obs):
            break

    print(f"after {t + 1} steps: reward of the last step = {reward:.3f}")
    print(recorder.summary())
    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots", "manipulation_wipe_progress")
    print("cleaning-progress figure saved to", recorder.save(out_dir, title="your ManipulationController"))
    env.close()


if __name__ == "__main__":
    main()
