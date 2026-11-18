# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.robots
====================
Everything about the *robot itself*: a Panda arm mounted on a wheeled base,
plus the sensors that give the mobile subgroup something to localize and
navigate with -- a 2D LIDAR ring, a forward-facing RGB-D camera, and a
noisy-odometry model. This is the one file you'd touch to change what
hardware the robot has.

Design notes on the LIDAR (read this before changing lidar_radius/lidar_height)
--------------------------------------------------------------------------
We use MuJoCo's native `<rangefinder/>` sensor (the same mechanism
hello-robot/stretch_mujoco uses for Stretch's simulated 2D spinning lidar),
NOT a hand-rolled Python ray-cast loop. A `<rangefinder/>` is just a site
that raycasts along its local **+Z** axis and reports the distance to whatever
it hits. (+Z, NOT -Z: cameras and lights look down -Z, so it is easy to assume
the same here. This ring was first built with -Z pointing outward, which made
every beam point INWARD -- the "LIDAR" spent its whole life reading the robot's
own chassis and the far side of the scene, never what was in front of it. The
sensor and a direct `mujoco.mj_ray` along the site's +Z axis agree exactly,
which is how it was caught; if you touch the ring, check that equality.)

The catch: a rangefinder does NOT automatically ignore the rest of its own
robot -- it only ignores geoms on the exact body its site sits on. The
chassis (a low box extending ~0.55m behind and ~0.25m to each side of the
base origin, top at z=0.38m) and the arm's mount column belong to OTHER
bodies, so a ring placed inside or below them would range to the robot itself.

Mount used here (lidar_height=0.50, lidar_radius=0.25): above the chassis top
and below the arm, at the chassis' lateral edge like a real LIDAR puck.
Validated against the real sensor -- 50 random arm poses x random base
positions/yaws, 800 finite-range beams: 0 self-hits. (z=0.60 starts hitting
the arm: 3 self-hits in the same sweep; z=0.30 needs radius ~0.8 to clear the
chassis.)

Consequences of the mount worth knowing:
  * Anything within `lidar_radius` of the base centre is inside the ring and
    invisible (beams start at the ring and point away). Keep the radius small:
    an earlier ring at radius 0.8 had a 0.8m blind zone, so the robot lost
    every obstacle just before it mattered.
  * The ring plane (z=0.5m) is above small things. Obstacles meant to be seen
    must be taller than that (mobile_manip's floor obstacles are 0.6m); tabletops
    (0.82-0.9m) are above it too, so a stock table is seen only through its thin
    legs (one central post, for a pickup table small enough that all four legs
    coincide -- easy for 16 beams to miss). TwoTableTaskMixin fills the space
    under both tables' slabs with a solid, LIDAR-visible "skirt" for exactly
    this reason -- see `arenas.py`.
  * `footprint_radius` (0.60m) is a separate number: the smallest circle about
    the base origin containing the chassis. Use it for clearance/plots, not the
    ring radius.
  * The earlier design was validated with ray-casts along the site's -Z axis,
    but the sensor fires along +Z -- so it never tested what the sensor
    actually did. If you touch the ring, compare the sensor to `mujoco.mj_ray`
    along the site's +Z axis and re-run the sweep against the SENSOR.

Odometry
--------
We do NOT expose ground-truth pose as an observation -- that would make an
EKF/localization exercise pointless. Instead:
  * `robot0_base_odom`: a NOISY, DRIFTING pose estimate obtained by
    integrating the base's commanded wheel velocities with configurable
    Gaussian noise -- this is what a real wheel encoder gives you, and is
    the intended INPUT to a student-implemented EKF.
  * The true pose is still computed (for grading / sanity-checking), but
    lives under `env._true_base_pose()`, not in the observation dict.
"""

import numpy as np
import xml.etree.ElementTree as ET

from robosuite.models.robots import Panda
from robosuite.models.bases import OmronMobileBase, register_base
from robosuite.robots import register_robot_class
from robosuite.controllers import load_composite_controller_config


# ---------------------------------------------------------------------------
# Quaternion helper: build a horizontal ring of sensing directions.
# ---------------------------------------------------------------------------
def _quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# Fixed quaternion that maps local +Z -> world +X: a 90 degree rotation about Y.
# MuJoCo rangefinders fire along the site's +Z axis, so this makes the beam
# point straight out horizontally. Rotating it about world Z by theta gives a
# beam pointing at angle theta with no vertical component. (Do NOT reuse the
# camera convention here -- cameras look along -Z.)
_Q_LOOK_OUT = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])


def _ring_quat(theta):
    half = theta / 2.0
    q_z = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    return _quat_mult(q_z, _Q_LOOK_OUT)


# ---------------------------------------------------------------------------
# Mobile base: OmronMobileBase + a ring of rangefinders + a nav camera.
# ---------------------------------------------------------------------------
@register_base
class MobilePandaBase(OmronMobileBase):
    """
    OmronMobileBase with:
      * a ring of `n_rays` native <rangefinder/> sensors (2D LIDAR), verified
        self-collision-free (see module docstring),
      * a forward-facing RGB-D navigation camera (`<prefix>nav`), the
        base-mounted equivalent of stretch_mujoco's `cam_nav`.
    """

    n_rays = 16
    lidar_height = 0.50      # above the chassis top (0.38m), below the arm -- see module docstring
    lidar_radius = 0.25      # small on purpose: the ring's blind zone is everything inside it
    footprint_radius = 0.60  # bounding circle of the chassis about the base origin (rear 0.55m, half-width 0.25m)
    lidar_max_range = 5.0

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        prefix = self.naming_prefix  # e.g. "mobilebase0_"
        base_body = self.worldbody.find(f".//body[@name='{prefix}base']")

        angles = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        for i, theta in enumerate(angles):
            site_name = f"{prefix}lidar_site{i}"
            x = self.lidar_radius * np.cos(theta)
            y = self.lidar_radius * np.sin(theta)
            q = _ring_quat(theta)

            site = ET.SubElement(base_body, "site")
            site.set("name", site_name)
            site.set("pos", f"{x:.4f} {y:.4f} {self.lidar_height:.4f}")
            site.set("quat", " ".join(f"{v:.6f}" for v in q))
            site.set("size", "0.006")
            site.set("rgba", "1 0 0 0.3")  # visible in the interactive viewer

            rf = ET.SubElement(self.sensor, "rangefinder")
            rf.set("name", f"{prefix}lidar{i}")
            rf.set("site", site_name)
            rf.set("cutoff", str(self.lidar_max_range))

        # Forward-facing RGB-D nav camera, mounted above the lidar ring.
        nav_cam = ET.SubElement(base_body, "camera")
        nav_cam.set("name", f"{prefix}nav")
        nav_cam.set("pos", "0.28 0 0.45")
        nav_cam.set("euler", "0 -90 -90")  # local -Z -> base's +X, upright
        nav_cam.set("fovy", "60")


# ---------------------------------------------------------------------------
# Robot: Panda arm + MobilePandaBase.
# ---------------------------------------------------------------------------
@register_robot_class("WheeledRobot")
class MobilePanda(Panda):
    @property
    def default_base(self):
        return "MobilePandaBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def init_qpos(self):
        # Stock Panda's default resting pose puts the gripper at roughly
        # (table_x, table_y, table_z - 0.02) -- fine for the original
        # single-table task, where the robot is parked well clear of the
        # table until it deliberately reaches in. Here the robot is parked
        # RIGHT NEXT TO the pickup table (see TwoTableTaskMixin), so that
        # same resting pose puts the gripper embedded in the pickup table's
        # surface at reset (found by testing: eef_pos landed within 1cm of
        # the table's center, in x/y AND z) -- the arm spends the entire
        # navigation phase in continuous contact with the table, which can
        # resist the base's own motion too, since MuJoCo solves robot+base
        # as one connected system. A more elbow-up resting pose keeps the
        # gripper around 1.2-1.3m up regardless of table position below it.
        return np.array([0, -0.5, 0, -2.0, 0, 1.5, 0.785])


def get_controller_config(controller="BASIC"):
    """
    Composite controller config for MobilePanda. `controller` can be any
    composite controller robosuite ships:
        "BASIC"               -> arm: Operational Space Control, base: joint velocity
        "WHOLE_BODY_MINK_IK"  -> whole-body IK (requires the `mink` package)
    """
    return load_composite_controller_config(controller=controller)


def world_velocity_to_base_command(v_world_xy, yaw, init_yaw=0.0):
    """
    Converts a desired base velocity expressed in WORLD coordinates into the
    `[vx, vy]` part of the base action.

    Why this exists: the base's [vx, vy] action is NOT in the robot's own
    frame, whatever the docs of the underlying controller suggest.
    robosuite's `MobileBaseJointVelocityController.set_goal` rotates the
    command by -theta (theta = yaw - initial yaw; the source comment says "do
    the reverse of theta rotation") before it reaches slide joints whose axes
    are fixed in the initial-base frame. Net effect: to move along world
    direction v you must send R(+theta) v, not R(-theta) v. Sending R(-theta)
    (the natural "world -> robot frame" conversion) makes the executed motion
    R(-2*theta) v -- correct at yaw 0 and yaw 180 (which is why a single check
    near 180 degrees looked fine), and increasingly wrong in between; a
    heading-following controller then spirals, because turning changes yaw,
    which changes how wrong its own commands are.

    Args:
        v_world_xy: desired (vx, vy) in world coordinates.
        yaw: the base's current yaw (rad), e.g. from your pose estimate.
        init_yaw: the base's yaw at the start of the episode (0 in this project).

    Returns:
        np.ndarray (2,): the [vx, vy] to put in the base action slice.
    """
    theta = yaw - init_yaw
    c, s = np.cos(theta), np.sin(theta)
    vx, vy = v_world_xy
    return np.array([c * vx - s * vy, s * vx + c * vy])


# --- base velocity calibration (measured on the real sim; see STATUS.md) ---
# The base's [vx, vy] action is NOT a velocity: each WORLD axis has a dead zone,
#     speed_axis = max(0, |command_axis| - BASE_COMMAND_DEADZONE)  [m/s]
# (command 0.30 -> 0.05 m/s, 0.50 -> 0.25, 1.00 -> 0.74; per axis, so a command
# toward 30 degrees executes as ~10 degrees -- the small component is swallowed).
# Verified independent of the base's yaw: the dead zone sits on world axes.
BASE_COMMAND_DEADZONE = 0.25
# The rotated command's components must stay within +-1 (the env clips them), and
# a world-axis command c has norm up to sqrt(2)*c, so cap each world axis at
# (1/sqrt(2) - deadzone): ~0.45 m/s.
BASE_MAX_AXIS_SPEED = 1.0 / np.sqrt(2.0) - BASE_COMMAND_DEADZONE


def world_velocity_mps_to_base_command(v_world_mps, yaw, init_yaw=0.0):
    """
    Converts a desired base velocity in WORLD coordinates, in real m/s, into the
    `[vx, vy]` part of the base action -- undoing both the -yaw rotation (see
    `world_velocity_to_base_command`) and the per-axis dead zone, so the base
    actually moves at the velocity you ask for (up to BASE_MAX_AXIS_SPEED per
    axis; larger requests are clipped per axis).

    Use this instead of `world_velocity_to_base_command` whenever you think in
    m/s. The raw version remains for code that wants the unmodified command.
    """
    v = np.clip(np.asarray(v_world_mps, dtype=float), -BASE_MAX_AXIS_SPEED, BASE_MAX_AXIS_SPEED)
    c_world = np.sign(v) * (np.abs(v) + BASE_COMMAND_DEADZONE) * (np.abs(v) > 1e-6)
    return world_velocity_to_base_command(c_world, yaw, init_yaw)


NAV_CAMERA = "mobilebase0_nav"        # base-mounted, forward-facing -- mobile subgroup
WRIST_CAMERA = "robot0_eye_in_hand"   # built into the Panda gripper -- manipulation subgroup


class NoisyOdometry:
    """
    Synthesizes a realistic, DRIFTING dead-reckoning pose estimate from a
    sequence of true relative motions, using the standard "odometry motion
    model" from probabilistic robotics (Thrun et al.): the error added at
    each physics sub-step is proportional to the MOTION in that sub-step
    (translation/rotation), plus a small constant floor (5e-4). The floor
    means a parked robot drifts too, slowly, like a random walk: typically
    about 5 cm after 12 s and 9 cm after 50 s. Driving adds error in
    proportion to the distance driven.

    `estimate[t] = estimate[t-1] + noisy(true_pose[t] - true_pose[t-1])`,
    where the noise stddev on each relative motion scales with the size of
    that motion. This sidesteps needing to reproduce robosuite's specific
    base joint/kinematic conventions (we tried that too; the base's
    kinematic chain couples rotation and translation in a way that made
    reading raw joint velocities fragile) -- whatever the base actually
    does, differencing the true trajectory senses it correctly.

    Feed this the true pose each step via `step(true_pose, dt)`; read the
    result off `.x`, `.y`, `.yaw`. Do NOT feed it a policy's own state.
    """

    def __init__(self, trans_noise_frac=0.03, rot_noise_frac=0.03,
                 min_trans_std=0.0005, min_rot_std=0.0005, seed=None):
        self.trans_noise_frac = trans_noise_frac  # stddev as a fraction of distance moved
        self.rot_noise_frac = rot_noise_frac      # stddev as a fraction of (distance + rotation)
        self.min_trans_std = min_trans_std        # floor, so exactly-zero motion isn't exactly noise-free
        self.min_rot_std = min_rot_std
        self._rng = np.random.default_rng(seed)
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw
        self._prev_true = np.array([x, y, yaw])

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(self, true_pose, dt):
        true_pose = np.asarray(true_pose, dtype=float)
        dx, dy = true_pose[:2] - self._prev_true[:2]
        dyaw = self._wrap_to_pi(true_pose[2] - self._prev_true[2])
        self._prev_true = true_pose

        trans = np.hypot(dx, dy)
        motion = trans + abs(dyaw)
        trans_std = self.min_trans_std + self.trans_noise_frac * trans
        rot_std = self.min_rot_std + self.rot_noise_frac * motion

        dx += self._rng.normal(0, trans_std)
        dy += self._rng.normal(0, trans_std)
        dyaw += self._rng.normal(0, rot_std)

        self.x += dx
        self.y += dy
        self.yaw = self._wrap_to_pi(self.yaw + dyaw)
        return self.x, self.y, self.yaw
