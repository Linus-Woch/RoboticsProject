<!--
Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
Barcelona East School of Engineering (EEBE).
Robotics Systems MSc course 295II235, 2026-2027.
All rights reserved.
-->

# mobile_manip

The simulated laboratory for the mobile-manipulation project: a wheeled Franka Panda
(`MobilePanda`) in robosuite/MuJoCo, with a task table, a pickup table, three bins on the route
between them and, optionally, a walking person. **The handout is the reference for the project**
(the two tracks, what you may use, how it is graded); this file tells you where things are.

- Installation: see **INSTALL.md** (Ubuntu; on Windows use WSL2, the steps are there).
- Software: `robosuite==1.5.2`, `mujoco==3.3.0`, Python 3.12.

## What is in the package

| Path | What it is |
|---|---|
| `student_templates/mobile_controller.py` | **Track 1 starting point.** A `MobileController` class with two methods to fill in, and a command-line test that ends with the success test of the handout. |
| `student_templates/manipulation_controller.py` | **Track 2 starting point.** A `ManipulationController` class with one method to fill in, a `detect_dirt` helper, and a command-line test that draws what you cleaned. |
| `student_templates/run_merged.py` | Runs both controllers one after the other (drive, then wipe). Optional: you do not need it for your own track. |
| `demo_run.py` | The smallest example of creating and stepping the environment; prints the observations and the layout of the action vector. |
| `mobile_manip/` | The environment itself (robot, sensors, tables, bins, person, helper functions). You do not need to edit it. `make_env(...)` in `environments.py` creates the environments. |

The templates are optional; the environment is what counts. Every file starts with a comment
that explains what it does and how to run it.

## Quick start

```python
import numpy as np
from mobile_manip import make_env, WRIST_CAMERA

# Track 1: the robot starts far from the task table and must drive to it
env = make_env(task="wipe", mobile_phase=True, human_mode="none")    # human_mode: "none" | "slow" | "fast"
# Track 2: the robot is already parked at the task table, with a wrist camera
env = make_env(task="wipe", mobile_phase=False, cameras=[WRIST_CAMERA], camera_size=256)

obs = env.reset()                                  # dictionary of NumPy arrays; print(obs.keys())
action = np.zeros(env.action_dim)                  # 10 entries; fill in only the ones you control
obs, reward, done, info = env.step(action)         # one step = 0.05 s
```

Where the entries of the action vector are (do not hard-code them, read them):

```python
split = env.robots[0].composite_controller._action_split_indexes
arm_start, arm_end = split["right"]    # 0..5   arm: delta pose of the tool [dx, dy, dz, d_rx, d_ry, d_rz]
                                       # 6      torso: unused, keep it 0
base_start, base_end = split["base"]   # 7..9   base: [vx, vy, wz] (see below)
```

There is no gripper entry: the wiping tool has no fingers. `done` becomes True after `horizon`
steps (1000 = 50 s by default; pass `horizon=...` to `make_env`) and, in the wipe task, when the
episode ends early (see "Reward").

## Track 1: driving the base

- **What you know at the start:** the base starts at (0.15, 2.0) facing +x in every episode, and the first
  odometry reading is exactly that pose. After that you only have the sensors (LIDAR and odometry), plus the
  goal and the maps below. You may hard-code what you know about the layout, but the bins change at every
  reset and the person moves, so a fixed route can still fail.
- **Sensors you may use:** `obs["robot0_base_lidar"]`, `obs["robot0_base_odom"]`, the optional
  navigation camera, `env.wall_segments()`, `env.table_map()`, `env.manipulation_base_xy()`,
  `env.manipulation_keepout()`.
- **Never inside your controller:** `env.true_base_pose()`, `obs["robot0_base_pos"]`,
  `obs["robot0_base_quat"]`, `obs["robot0_proprio-state"]` and anything computed from them, and
  anything read from `env.sim`. The true pose is for evaluating a run and for plots.
- **The base command is not what it looks like.** `[vx, vy]` is not in the robot frame and is not
  a velocity: the simulator rotates it by `-yaw` and then ignores the first 0.25 of each world
  axis (a command of 0.30 moves the base at 0.05 m/s, 1.00 at 0.74 m/s). Use
  `world_velocity_mps_to_base_command(v_world, yaw)`: give it the velocity you want in the world
  frame, in m/s, and it returns the `[vx, vy]` to send (up to about 0.45 m/s per axis).
- **Keep the yaw near 0** (`wz = -1.0 * yaw`): the base is holonomic, and the arm does not follow
  the base when it turns, so the arm must keep facing the table.
- **The chassis is a box**: 0.7 x 0.5 m, reaching 0.55 m behind the base origin and 0.15 m ahead
  of it (0.25 m to each side). Measure clearances to the box.
- **Odometry** (`robot0_base_odom`) is the true motion plus noise that grows with the motion
  (the standard odometry motion model, Thrun et al.). It starts exact, drifts as you drive, and
  also drifts slowly when parked (typically about 5 cm after 12 s and 9 cm after 50 s). The true
  pose is never in `obs`.
- **LIDAR** (`robot0_base_lidar`): 16 beams, beam `i` points at `yaw + i*22.5` degrees and starts
  on a ring of radius 0.25 m around the base origin, 0.5 m above the floor, so ranges are measured
  from the ring and nothing lower than 0.5 m is seen. `-1` = no hit; `5.0` = range limit (nothing
  within 5 m). The tables show up through their solid skirts. The walls are solid: the base
  cannot drive through them, touching one counts as a contact, and they are a good map to localise against.
  Stay inside the arena.
- **See what the LIDAR sees.** `student_templates/mobile_controller.py` has `lidar_points(lidar, x, y, yaw)` (world-frame
  hit points) and `plot_lidar(lidar, x, y, yaw, walls)` (draws the beams, the hit points and the chassis).
  `python student_templates/mobile_controller.py --plot-lidar` saves one scan to `plots/lidar_scan.png`.
- **The arm must stay still.** Remember `obs["robot0_joint_pos"]` at the start and track the largest change of
  any joint; the Track 1 template prints it at the end of a run.
- **The bins.** Without a person (stage 1) the three bins are scattered on the straight route, so a
  short detour is enough. With a person (stages 2 and 3) two bins form a barrier that closes the short
  way into the manipulation zone, and the third is somewhere on the long way: you have to go east
  around the pickup table, south of the task table, and enter the zone from the south (about 6 m
  instead of 3 m). Pass `course="simple"` or `course="long"` to `make_env` to choose it yourself.
  The layout changes at every reset.
- **The person** walks back and forth on a line 0.5 m beyond the west edge of the manipulation
  zone (behind the parked robot), at 0.1 m/s (`human_mode="slow"`) or 0.3 m/s (`"fast"`).
  `person_mode="static"` makes it stand there instead.
- **Success:** within 5 cm of `env.manipulation_base_xy()`, |yaw| < 10 degrees, and no contact,
  all evaluated with the true pose. `ContactMonitor` counts the contacts:

```python
from mobile_manip import ContactMonitor
monitor = ContactMonitor(env)        # right after env.reset()
...                                  # monitor.update() after every env.step(...)
print(monitor.report())              # e.g. "chassis vs bin: 12 steps (first at step 140)"
```

## Track 2: wiping the dirt

- **Allowed:** `robot0_eef_pos/quat`, `robot0_base_to_eef_pos`, `robot0_joint_pos/vel`,
  `robot0_contact`, the wrist camera, and `proportion_wiped` (for logging, plots and deciding when
  to stop).
- **Not allowed to locate the dirt:** `wipe_centroid`, `wipe_radius`, `gripper_to_wipe_centroid`,
  `object-state`, or anything read from `env.sim`. The dirt positions your planner uses must come
  from the camera image.
- **The arm action is a delta pose**: the controller aims at a pose up to 5 cm / 0.5 rad away from
  where the tool is now, so 0 means "hold here". With the base at yaw 0 you can use an error in
  world coordinates directly: `np.clip(10.0 * (target - obs["robot0_eef_pos"]), -1, 1)`.
- **Cameras** give RGB and depth, `obs["<camera>_image"]` (uint8) and `obs["<camera>_depth"]`.
  robosuite returns images **upside down** (flip image and depth with `[::-1]`), and the depth is
  **not in metres** (convert it with `depth_to_meters(env, depth)`). `pixel_to_world(env,
  WRIST_CAMERA, (row, col), depth_m, height, width)` gives the world position of a pixel of the
  flipped image. The wrist image also shows the robot's base, the floor and the LIDAR rays that
  the renderer draws; their red end points look like dirt to a colour test, so keep only points
  at table height (z about 0.9 m). `detect_dirt` in the template does all of this. The camera loses
  the dirt from view when the tool is within about 6 cm of it.
- **Reward.** After every step the environment returns a *reward*: a number that scores the step
  (higher is better): a large positive term for each dirt spot wiped, a term that grows as the
  tool gets closer to the dirt that is left, a small positive term while the tool presses on the
  table (and a penalty if it presses too hard), and a penalty of -10 if the arm collides with
  something or reaches a joint limit. In that last case the episode **ends early**. All rewards
  are scaled by a constant that depends on `horizon`, so compare rewards only between runs with
  the same horizon.
- **See what you cleaned:** `WipeProgressRecorder` turns `proportion_wiped` into a figure (percent
  cleaned over time, and a top-down view of the dirt with the path of the tool):

```python
from mobile_manip import WipeProgressRecorder
rec = WipeProgressRecorder(env)          # right after env.reset()
...                                      # rec.update(obs, done) after every env.step(...)
print(rec.summary())                     # "Cleaned 23% (23 of 100 dirt spots) in 700 steps"
rec.save("plots/my_wipe")                # -> plots/my_wipe/wipe_progress.png
```

## Tips

- **Speed.** Rendering camera images is by far the slowest part of the simulation. Do not ask for
  cameras in Track 1 unless you use them, keep `camera_size` as small as works, and run without
  `has_renderer=True` when you collect results. How fast camera images are depends on your
  graphics set-up; if a run with a camera is very slow, tell us.
- **Results.** Layout, dirt trail and odometry noise change at every `env.reset()`: run at least
  10 episodes before you draw conclusions (the handout says what the report needs).
- **Viewer.** `make_env(..., has_renderer=True)` and `env.render()` after each step open a window.
  It starts on a free camera that should move with the mouse (left-drag, right-drag, scroll); if
  it does not, tell us. See INSTALL.md if the window does not open at all.
