# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.contacts
=====================
A small tool for answering "did the robot hit anything?" with a number instead
of by eye. Call `ContactMonitor.update()` once after every `env.step(...)`:

    monitor = ContactMonitor(env)
    for t in range(steps):
        obs, *_ = env.step(action)
        monitor.update()
    print(monitor.report())

It counts, per (robot part, thing hit), how many steps were in contact, and
records WHERE each contact episode began (world x, y) so it can be drawn on the
trajectory plot (`save_scene_and_trajectory_plots(..., contact_points=...)`).

What counts as "the robot": chassis (`mobilebase0*`), arm (`robot0*`) and
gripper (`gripper0*`). What counts as "a thing": the floor bins, the person,
the walls, and both tables (slab, skirt, legs). The floor, the nuts and pegs are ignored.
"""

import mujoco

_ROBOT_PREFIXES = {"mobilebase0": "chassis", "robot0": "arm", "gripper0": "gripper"}


def _robot_part(name):
    for prefix, part in _ROBOT_PREFIXES.items():
        if name.startswith(prefix):
            return part
    return None


def _other_kind(name):
    if "obstacle" in name:
        return "bin"
    if "person" in name:
        return "person"
    if name.startswith("wall_") and name.endswith("_solid"):
        return "wall"
    if "pickup" in name:
        return "pickup table"
    if name.startswith("table"):
        return "task table"
    return None


class ContactMonitor:
    """Counts the control steps in which the robot touches a bin, the person or a table.

    Create it right after `env.reset()`, call `update()` after every `env.step(...)`, and read
    `report()` (text), `total_steps()` (one number) or `steps` (a dict) at the end. Use it for the
    "touched nothing" part of the Track 1 success condition. Evaluation tool: do not use it inside
    your controller.
    """

    def __init__(self, env, episode_gap=10):
        self.env = env
        self._model = env.sim.model._model
        self._data = env.sim.data._data
        self._episode_gap = episode_gap     # steps without contact before a new episode starts
        self._t = 0
        self._last_seen = {}
        self._geom_cache = {}
        self.steps = {}                     # (part, other) -> number of steps in contact
        self.episodes = []                  # dicts: t, part, other, xy

    def _classify(self, geom_id):
        if geom_id not in self._geom_cache:
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            self._geom_cache[geom_id] = (_robot_part(name), _other_kind(name))
        return self._geom_cache[geom_id]

    def update(self):
        """Call once after every `env.step(...)`."""
        seen = {}
        d = self._data
        for i in range(d.ncon):
            c = d.contact[i]
            for robot_geom, other_geom in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                part = self._classify(robot_geom)[0]
                other = self._classify(other_geom)[1]
                if part and other:
                    seen.setdefault((part, other), c.pos[:2].copy())
        for key, xy in seen.items():
            self.steps[key] = self.steps.get(key, 0) + 1
            if self._t - self._last_seen.get(key, -10**9) > self._episode_gap:
                self.episodes.append({"t": self._t, "part": key[0], "other": key[1], "xy": xy})
            self._last_seen[key] = self._t
        self._t += 1

    def total_steps(self):
        """Total contact-steps over all (part, thing) pairs."""
        return int(sum(self.steps.values()))

    def contact_points(self):
        """World (x, y) of the start of every contact episode, for plotting."""
        return [e["xy"] for e in self.episodes]

    def report(self):
        """Text summary: for each (robot part, thing touched) the number of steps in contact."""
        if not self.steps:
            return "Collisions: none (no contact with bins, person, walls or tables)."
        lines = ["Collisions (steps in contact, first at step):"]
        first = {}
        for e in self.episodes:
            first.setdefault((e["part"], e["other"]), e["t"])
        for key, n in sorted(self.steps.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {key[0]:7s} vs {key[1]:12s}: {n:4d} steps (first at step {first.get(key, '?')})")
        return "\n".join(lines)
