# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.moving_obstacles
================================
Scripted (mocap-driven) dynamic obstacles, for testing whether an avoidance
policy handles a MOVING obstacle, not just static ones.

We use a MuJoCo **mocap body** rather than a normal free-jointed object.
Mocap bodies are fully collidable (visible to the LIDAR raycasts and
cameras) but are NOT simulated by physics -- gravity and contact forces
don't move them. We reposition a mocap body directly via `sim.data.mocap_pos`
every step, which is the standard MuJoCo pattern for a scripted-motion
obstacle. Using a normal free-joint object instead would work at first, but
contacts / being bumped by the robot would knock it off its intended path --
a real pedestrian doesn't stop walking because a robot grazed them.
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models.tasks import ManipulationTask


def add_moving_person(worldbody, name="person", height=1.7, radius=0.18, start_pos=(0, 0, 0)):
    """
    Adds a simple two-geom "person" (torso cylinder + head sphere) as a
    mocap body directly under `worldbody`.
    """
    body = ET.SubElement(worldbody, "body")
    body.set("name", name)
    body.set("mocap", "true")
    body.set("pos", f"{start_pos[0]:.4f} {start_pos[1]:.4f} {start_pos[2]:.4f}")

    torso_half_h = max(height / 2 - radius, 0.05)
    torso = ET.SubElement(body, "geom")
    torso.set("name", f"{name}_torso")
    torso.set("type", "cylinder")
    torso.set("size", f"{radius:.3f} {torso_half_h:.3f}")
    torso.set("pos", f"0 0 {height / 2:.3f}")
    torso.set("rgba", "0.85 0.65 0.45 1")
    # group="1": robosuite's own render context hides group 0 (its
    # convention for COLLISION meshes -- e.g. every "*_collision"/"*_col"
    # geom in this scaffold) from camera images, showing only group 1
    # ("_visual" meshes elsewhere in this codebase). Without this, the
    # person compiled and simulated correctly (confirmed via direct
    # sim.data queries -- position, size, rgba all correct) but was
    # silently absent from every rendered image. Found by diffing every
    # geom in the compiled model against the set actually reaching the
    # render scene graph: the missing set was exactly "every group-0 geom"
    # (obstacles, table collision meshes, robot link collision meshes, and
    # this), not a positioning, FOV, or mocap-specific issue as it first
    # appeared.
    torso.set("group", "1")

    head = ET.SubElement(body, "geom")
    head.set("name", f"{name}_head")
    head.set("type", "sphere")
    head.set("size", f"{radius * 0.65:.3f}")
    head.set("pos", f"0 0 {height - radius * 0.65:.3f}")
    head.set("rgba", "0.85 0.65 0.45 1")
    head.set("group", "1")

    return name


class WalkingPath:
    """
    A simple back-and-forth ("ping-pong") walk along a sequence of 2D
    waypoints, sampled by simulation time.
    """

    def __init__(self, waypoints, speed=0.4):
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.speed = speed
        n = len(self.waypoints)
        seg_lengths = [
            np.linalg.norm(self.waypoints[(i + 1) % n] - self.waypoints[i]) for i in range(n)
        ]
        self._seg_lengths = np.array(seg_lengths[:-1]) if n > 1 else np.array([0.0])
        self._cum = np.cumsum(self._seg_lengths)
        self._total = self._cum[-1] if len(self._cum) else 0.0

    def position_at(self, t):
        if self._total < 1e-6:
            return self.waypoints[0]
        dist = (t * self.speed) % (2 * self._total)
        if dist > self._total:
            dist = 2 * self._total - dist
        seg_idx = int(np.searchsorted(self._cum, dist))
        seg_idx = min(seg_idx, len(self.waypoints) - 2)
        prev_cum = self._cum[seg_idx - 1] if seg_idx > 0 else 0.0
        local = dist - prev_cum
        a = self.waypoints[seg_idx]
        b = self.waypoints[seg_idx + 1]
        seg_len = self._seg_lengths[seg_idx]
        frac = 0.0 if seg_len < 1e-9 else local / seg_len
        return a + frac * (b - a)


class PersonMixin:
    """
    Mixin: adds a single person (mocap body) to a `TwoTableTaskMixin` env.
    Toggle with `add_person=True/False` (default False); pick the behavior
    with `person_mode`:

      "lateral" (default): walks back and forth along the WEST flank of the
          manipulation zone, behind the parked robot and never in front of the
          table, but moving in the LIDAR's view.
      "static": stands still at the middle of that same flank.

    Speed is set separately, either directly (`person_speed`, m/s) or via the
    three-way `human_mode`: "none" (no person), "slow" (0.1 m/s), "fast"
    (0.3 m/s -- the original and, before this, only speed). "slow" and "fast"
    use whatever `person_mode` is otherwise set to (default "lateral").

    Combine after `TwoTableTaskMixin` in the MRO.
    """

    person_height = 1.7
    person_radius = 0.18
    person_speed = 0.3
    # Placement: along the WEST flank of the manipulation
    # zone (the side the base's chassis reaches back toward, not the side the
    # arm works on), `person_west_clearance` beyond the zone's west edge, spanning
    # the zone's own y-extent plus `person_patrol_margin`. An earlier placement
    # (a line 0.5m off the task table's south side) sat 0.22m from the zone's south
    # edge -- a gap the chassis cannot pass. The whole swath is also registered with
    # the course-solvability check (`_extra_blockers`), so a route that avoids it
    # always exists.
    person_west_clearance = 0.5
    person_patrol_margin = 0.1

    _PERSON_MODES = ("static", "lateral")
    _HUMAN_SPEEDS = {"slow": 0.1, "fast": 0.3}   # "fast" matches the original, only-ever speed

    def __init__(self, *args, add_person=False, person_mode="lateral",
                 human_mode=None, person_speed=None, **kwargs):
        """
        `human_mode` is the requested top-level control -- "none" (no
        person), "slow", or "fast" (0.3 m/s, the original and only speed
        this had before) -- and simply sets `add_person`/`person_speed` for
        you. It overrides them if given; leave it as None (the default) and
        set `add_person`/`person_speed` yourself for finer control (e.g. a
        custom speed, or "static" via `person_mode`, which `human_mode`
        does not touch).
        """
        if human_mode is not None:
            if human_mode not in ("none",) + tuple(self._HUMAN_SPEEDS):
                raise ValueError(f"human_mode must be 'none', {list(self._HUMAN_SPEEDS)}, got {human_mode!r}")
            add_person = human_mode != "none"
            if human_mode in self._HUMAN_SPEEDS:
                person_speed = self._HUMAN_SPEEDS[human_mode]
        if person_mode not in self._PERSON_MODES:
            raise ValueError(f"person_mode must be one of {self._PERSON_MODES}, got {person_mode!r}")
        self.add_person = add_person
        self.person_mode = person_mode
        if person_speed is not None:
            self.person_speed = person_speed
        super().__init__(*args, **kwargs)

    def _person_waypoints_for_mode(self):
        keep_center, keep_size = self.manipulation_keepout()
        west_x = keep_center[0] - keep_size[0] / 2 - self.person_west_clearance
        y_lo = keep_center[1] - keep_size[1] / 2 - self.person_patrol_margin
        y_hi = keep_center[1] + keep_size[1] / 2 + self.person_patrol_margin
        if self.person_mode == "static":
            return [(west_x, keep_center[1])]
        return [(west_x, y_lo), (west_x, y_hi)]       # "lateral"

    def _extra_blockers(self):
        """The person's whole patrol swath (a rectangle around the waypoints, grown by
        the person's radius), so the solvability check requires a route that avoids
        it entirely -- a course is never solvable only when the person happens to be
        somewhere else."""
        if not (self.add_person and hasattr(self, "person_waypoints")):
            return []
        pts = np.array(self.person_waypoints, dtype=float)
        lo, hi = pts.min(axis=0) - self.person_radius, pts.max(axis=0) + self.person_radius
        return [((lo + hi) / 2, (hi - lo) / 2)]

    def _load_model(self):
        super()._load_model()
        if not (self.add_person and getattr(self, "mobile_phase", False)):
            return

        waypoints = self._person_waypoints_for_mode()
        self.person_waypoints = [np.array(w, dtype=float) for w in waypoints]
        self._person_path = WalkingPath(waypoints=waypoints, speed=self.person_speed)
        start = waypoints[0]
        self._person_name = add_moving_person(
            self.model.mujoco_arena.worldbody,
            name="person",
            height=self.person_height,
            radius=self.person_radius,
            start_pos=(start[0], start[1], 0.0),
        )

        existing_objects = list(getattr(self.model, "mujoco_objects", []))
        self.model = ManipulationTask(
            mujoco_arena=self.model.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=existing_objects,
        )

    def _reset_internal(self):
        super()._reset_internal()
        if self.add_person and getattr(self, "mobile_phase", False):
            body_id = self.sim.model.body_name2id(self._person_name)
            self._person_mocap_id = self.sim.model.body_mocapid[body_id]
            self._update_person(t=0.0)

    def _update_person(self, t):
        x, y = self._person_path.position_at(t)
        self.sim.data.mocap_pos[self._person_mocap_id] = [x, y, 0.0]

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        if self.add_person and getattr(self, "mobile_phase", False):
            self._update_person(self.sim.data.time)
