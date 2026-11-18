# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip
============
The simulated laboratory for the mobile-manipulation project: a wheeled Franka Panda
("MobilePanda") in robosuite, with a task table, a pickup table, three bins on the route
and, optionally, a walking person. See the course handout for the two tracks.

Quick start
-----------
    from mobile_manip import make_env, WRIST_CAMERA

    # Track 1 (mobile robotics): the robot starts far from the task table and must drive to it
    env = make_env(task="wipe", mobile_phase=True, human_mode="slow")   # human_mode: "none" | "slow" | "fast"
    obs = env.reset()
    obs["robot0_base_lidar"]   # (16,) metres, -1 = no hit, 5.0 = range limit. Beam i points at i*22.5 deg
                               # in the robot frame and starts on a ring 0.25 m from the base origin, so
                               # the range is measured from the ring, not from the origin.
    obs["robot0_base_odom"]    # (3,) NOISY, drifting [x, y, yaw]: the only pose you may use while driving
    env.true_base_pose()       # (3,) ground truth [x, y, yaw]: for evaluation and plots ONLY

    # Track 2 (manipulation): the robot is already parked at the table, with a wrist camera
    env = make_env(task="wipe", mobile_phase=False, cameras=[WRIST_CAMERA], camera_size=256)

Both tracks: `action = np.zeros(env.action_dim)`, fill in your entries, `env.step(action)`.
Entries 0-5 move the arm, 6 is unused, 7-9 drive the base (the handout has the table).

Software versions: robosuite==1.5.2, mujoco==3.3.0
"""

from .robots import (MobilePanda, MobilePandaBase, NoisyOdometry, get_controller_config,
                     world_velocity_to_base_command, world_velocity_mps_to_base_command,
                     BASE_COMMAND_DEADZONE, BASE_MAX_AXIS_SPEED, NAV_CAMERA, WRIST_CAMERA)
from .tasks import TwoTableTaskMixin, LidarObservableMixin, split_observations
from .moving_obstacles import PersonMixin
from .environments import MobileNutAssembly, MobileWipe, make_env
from .plotting import save_scene_and_trajectory_plots
from .contacts import ContactMonitor
from .wipe_progress import WipeProgressRecorder
from .camera_utils import depth_to_meters, pixel_to_world, world_to_pixel, point_in_camera_view

__all__ = [
    "MobilePanda", "MobilePandaBase", "NoisyOdometry", "get_controller_config",
    "world_velocity_to_base_command", "world_velocity_mps_to_base_command",
    "BASE_COMMAND_DEADZONE", "BASE_MAX_AXIS_SPEED", "NAV_CAMERA", "WRIST_CAMERA",
    "TwoTableTaskMixin", "LidarObservableMixin", "split_observations", "PersonMixin",
    "MobileNutAssembly", "MobileWipe", "make_env",
    "save_scene_and_trajectory_plots",
    "ContactMonitor", "WipeProgressRecorder", "depth_to_meters", "pixel_to_world", "world_to_pixel", "point_in_camera_view",
]
