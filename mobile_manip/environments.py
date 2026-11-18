# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.environments
===========================
The two concrete, ready-to-use environments for the project's task options.

Registration: robosuite auto-registers *any* subclass of its MujocoEnv base
(via the EnvMeta metaclass) the moment the class is defined -- just importing
this module is enough to make these names usable directly with robosuite's
own factory:

    import mobile_manip  # registers everything as a side effect
    import robosuite as suite
    env = suite.make("MobileNutAssembly", robots="MobilePanda", ...)

`make_env(...)` below is a documented, opinionated convenience wrapper
around that. Both work -- use whichever fits your code.
"""

from robosuite.environments.manipulation.nut_assembly import NutAssemblySquare
from robosuite.environments.manipulation.wipe import Wipe

from .tasks import TwoTableTaskMixin, LidarObservableMixin
from .moving_obstacles import PersonMixin
from .robots import get_controller_config, NAV_CAMERA, WRIST_CAMERA


class MobileNutAssembly(PersonMixin, TwoTableTaskMixin, LidarObservableMixin, NutAssemblySquare):
    """Fetch the square nut from the pickup table, fit it onto the peg on
    the task table. Registered with robosuite as "MobileNutAssembly"."""
    pass


class MobileWipe(PersonMixin, TwoTableTaskMixin, LidarObservableMixin, Wipe):
    """Wipe the dirt path off the task table. Registered with robosuite as
    "MobileWipe"."""
    pass


_TASK_ENVS = {"nut": MobileNutAssembly, "wipe": MobileWipe}


def make_env(
    task="wipe",                # "wipe" (the task used in this project) or "nut" (not used)
    controller="BASIC",         # leave as is: BASIC = operational-space (OSC) arm + base velocity command
    mobile_phase=True,          # True = the full two-table lab with bins (Track 1).
                                # False = robot already parked at the task table (Track 2)
    add_person=False,           # older switch for the person; use human_mode instead
    person_mode="lateral",      # where the person walks: "lateral" (back and forth behind the parked robot,
                                # the default), or "static" (stands still there)
    human_mode=None,            # walking person: "none" (nobody), "slow" (0.1 m/s) or "fast" (0.3 m/s)
    course=None,                # where the bins go: "simple" (scattered on the route: a short detour) or
                                # "long" (a barrier that forces a long detour). Default: "long" when a
                                # person is present (the harder set-up), "simple" otherwise
    cameras=None,                # e.g. [WRIST_CAMERA] or [NAV_CAMERA, WRIST_CAMERA]; None = no camera images
                                # (rendering images is the slowest part of the simulation: only ask for what you use)
    camera_size=84,             # camera images are camera_size x camera_size pixels
    has_renderer=False,         # True = open a window to watch the simulation (slower; use it to debug, not to collect results)
    render_camera=None,         # None = FREE camera you can drag/pan/zoom with the mouse (default);
                                # a camera name (e.g. "frontview", NAV_CAMERA) = FIXED view that
                                # cannot be moved. See the note in the docstring.
    **kwargs,
):
    """Creates the environment for this project in one call.

    Typical use:
        env = make_env(task="wipe", mobile_phase=True,  human_mode="none")        # Track 1: drive to the table
        env = make_env(task="wipe", mobile_phase=False, cameras=[WRIST_CAMERA])   # Track 2: wipe the dirt
        obs = env.reset()
        obs, reward, done, info = env.step(action)     # action = np.zeros(env.action_dim), then fill in your entries

    Any other keyword argument is passed on to the environment. The most useful is `horizon`, the
    number of control steps after which `done` becomes True. If you do not give it, it is 1000 steps
    (50 s), or 1500 steps (75 s) when the bins form the long barrier (see `course`: the route is about
    twice as long).

    On-screen viewer (`has_renderer=True`): robosuite's own default is the FIXED
    camera "frontview", and MuJoCo's viewer cannot be mouse-controlled while a
    fixed camera is selected -- which looked like "I can't move the view". With
    `render_camera=None` (the default here) the viewer starts on a free camera
    framed for this layout (overview for the navigation phase, a side view of the
    table for manipulation-only), and left-drag / right-drag / scroll move it.
    Pass `render_camera="frontview"` (or any camera name) to get a fixed view back.
    """
    env_cls = _TASK_ENVS[task]
    controller_config = get_controller_config(controller)

    use_camera_obs = cameras is not None and len(cameras) > 0
    if "horizon" not in kwargs:
        person_present = (human_mode in ("slow", "fast")) or (human_mode is None and add_person)
        long_course = (course == "long") or (course is None and person_present)
        kwargs["horizon"] = 1500 if (mobile_phase and long_course) else 1000
    if has_renderer and render_camera is None and "renderer_config" not in kwargs:
        # MuJoCo azimuth = direction the camera LOOKS toward (90 deg = looking +y), so
        # azimuth 90 gives a map-like view: x to the right, y up the screen.
        kwargs["renderer_config"] = {"cam_config": (
            {"lookat": [0.0, 0.6, 0.4], "distance": 6.5, "azimuth": 90, "elevation": -60} if mobile_phase
            else {"lookat": [-0.1, 0.0, 1.0], "distance": 2.2, "azimuth": 90, "elevation": -20})}
    return env_cls(
        robots="MobilePanda",
        controller_configs=controller_config,
        mobile_phase=mobile_phase,
        course=course,
        add_person=add_person,
        person_mode=person_mode,
        human_mode=human_mode,
        has_renderer=has_renderer,
        render_camera=render_camera,
        has_offscreen_renderer=use_camera_obs,
        use_camera_obs=use_camera_obs,
        camera_names=cameras if use_camera_obs else None,
        camera_heights=camera_size,
        camera_widths=camera_size,
        camera_depths=use_camera_obs,
        control_freq=20,
        **kwargs,
    )
