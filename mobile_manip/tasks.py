# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.tasks
====================
Task-level logic shared by both project variants (nut assembly / wipe):
turning a stock single-table robosuite manipulation env into the two-table,
obstacle-avoiding layout, and exposing the sensors (LIDAR + noisy odometry)
students actually implement localization/avoidance against. This is mixed
into a concrete environment in `environments.py`; it never runs on its own.
"""

import mujoco
import numpy as np

from robosuite.utils.observables import Observable, sensor
from robosuite.models.tasks import ManipulationTask

from ._feasibility import route_exists, route_length
from .arenas import add_pickup_table, harden_table_legs, add_task_table_skirt, add_solid_walls
from .obstacles import make_obstacles
from .robots import NoisyOdometry


# Keys (or key prefixes) that belong to each subgroup's observation set.
# Used by `split_observations()` below.
MOBILE_OBS_KEYS = ("base_odom", "base_lidar", "nav_image", "nav_depth")
MANIP_OBS_KEYS = ("eef_pos", "eef_quat", "gripper_qpos", "gripper_qvel",
                   "eye_in_hand_image", "eye_in_hand_depth")


def split_observations(obs):
    """
    Splits a robosuite observation dict into (mobile_obs, manip_obs,
    other_obs) by key-name pattern matching.
    """
    mobile_obs, manip_obs, other_obs = {}, {}, {}
    for k, v in obs.items():
        if any(tag in k for tag in MOBILE_OBS_KEYS):
            mobile_obs[k] = v
        elif any(tag in k for tag in MANIP_OBS_KEYS):
            manip_obs[k] = v
        else:
            other_obs[k] = v
    return mobile_obs, manip_obs, other_obs


class LidarObservableMixin:
    """
    Exposes two things the mobile subgroup needs:
      * `<prefix>base_lidar`: the native rangefinder ring on MobilePandaBase
        (see robots.py). Shape (n_rays,), meters, -1 = nothing in range.
      * `<prefix>base_odom`: a NOISY, DRIFTING [x, y, yaw] pose estimate,
        built by adding per-step Gaussian noise to the robot's own true
        relative motion and integrating (see `NoisyOdometry` in robots.py)
        -- this is the intended input to a student-written EKF, not a free
        localization solution.

    Ground truth is deliberately NOT exposed as an observation (that would
    make a localization exercise pointless). It's still available, for
    grading/sanity-checking only, via `env.true_base_pose()`.
    """

    odom_trans_noise_frac = 0.03  # stddev as a fraction of distance moved per step
    odom_rot_noise_frac = 0.03    # stddev as a fraction of motion magnitude per step

    def __init__(self, *args, odom_seed=None, **kwargs):
        self._odom = NoisyOdometry(
            trans_noise_frac=self.odom_trans_noise_frac,
            rot_noise_frac=self.odom_rot_noise_frac,
            seed=odom_seed,
        )
        super().__init__(*args, **kwargs)

    def wall_segments(self):
        """
        The arena walls as a 2D line map: an (N, 2, 2) array of segment
        endpoints [[x0, y0], [x1, y1]] (four walls plus the two 45-degree
        corner pieces that chamfer the rear corners). This is the map to
        localize against.

        Why walls and not the tables: the LIDAR ring sits at z=0.5m, BELOW the
        tabletops (0.82-0.9m), so it never sees a table's slab -- only its thin
        legs, which a 16-beam ring almost always slips between. The walls are
        the large, always-visible structure the ring can actually range to.
        (The walls are solid: the base cannot drive through them, and the
        LIDAR sees them.)
        """
        m = self.sim.model._model
        d = self.sim.data._data
        segments = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not (name.startswith("wall_") and name.endswith("_visual")):
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            normal = R[:2, 2]                      # walls are boxes, thin along local z
            if np.linalg.norm(normal) < 0.5:
                continue                           # not a vertical wall
            tangent = np.array([-normal[1], normal[0]]) / np.linalg.norm(normal)
            half = sum(m.geom_size[g][k] * abs(R[:2, k] @ tangent) for k in (0, 1))
            centre = d.geom_xpos[g][:2]
            segments.append([centre - half * tangent, centre + half * tangent])
        return np.array(segments)

    def true_base_pose(self):
        """Ground truth [x, y, yaw] of the mobile base, in world coordinates.
        For grading/plotting ONLY -- do not feed this to your policy."""
        prefix = self.robots[0].robot_model.base.naming_prefix
        body_id = self.sim.model.body_name2id(f"{prefix}base")
        pos = self.sim.data.body_xpos[body_id]
        quat_wxyz = self.sim.data.body_xquat[body_id]
        w, x, y, z = quat_wxyz
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([pos[0], pos[1], yaw])

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        # Called once per PHYSICS substep (25x per env.step() at the default
        # 20Hz control / 500Hz sim rate) -- fine, since we're differencing
        # the true trajectory rather than integrating a held action, so
        # calling this more often just means finer-grained (and smaller)
        # noise draws, which is the physically correct thing to do.
        self._odom.step(self.true_base_pose(), dt=self.sim.model.opt.timestep)

    def _reset_internal(self):
        super()._reset_internal()
        # Calibrate the odometry to the true starting pose -- a robot always
        # knows where it started, it's the DRIFT after that which an EKF has
        # to correct for using the LIDAR.
        x, y, yaw = self.true_base_pose()
        self._odom.reset(x, y, yaw)

    def _setup_observables(self):
        observables = super()._setup_observables()

        robot = self.robots[0]
        arm_prefix = robot.robot_model.naming_prefix     # e.g. "robot0_"
        base_model = robot.robot_model.base
        base_prefix = base_model.naming_prefix           # e.g. "mobilebase0_"
        n_rays = getattr(base_model, "n_rays", 16)
        sensor_names = [f"{base_prefix}lidar{i}" for i in range(n_rays)]

        @sensor(modality=f"{arm_prefix}proprio")
        def base_lidar(obs_cache):
            return np.concatenate([robot.get_sensor_measurement(name) for name in sensor_names])

        @sensor(modality=f"{arm_prefix}proprio")
        def base_odom(obs_cache):
            return np.array([self._odom.x, self._odom.y, self._odom.yaw])

        observables[f"{arm_prefix}base_lidar"] = Observable(
            name=f"{arm_prefix}base_lidar", sensor=base_lidar, sampling_rate=self.control_freq,
        )
        observables[f"{arm_prefix}base_odom"] = Observable(
            name=f"{arm_prefix}base_odom", sensor=base_odom, sampling_rate=self.control_freq,
        )
        return observables


class TwoTableTaskMixin:
    """
    Turns a single-table robosuite manipulation env into the two-table
    layout described in arenas.py, plus floor obstacles between the tables.

    `mobile_phase=True` (default): full pipeline -- robot starts parked at
        the PICKUP table and has to navigate to the TASK table.
    `mobile_phase=False`: manipulation-only -- robot stays parked at the
        TASK table exactly as the stock robosuite env would place it, object
        already there. Lets the manipulation subgroup iterate without ever
        touching navigation.
    """

    # Smaller pickup table, repositioned, with explicit extra parking
    # clearance on top of what `base_xpos_offset` gives -- that formula
    # assumes a compact, pedestal-mounted base and left only ~0.16m between
    # the parked robot and the pickup table's edge, while this base's own
    # wheeled chassis extends up to ~0.6-0.8m in some directions (see
    # MobilePandaBase's LIDAR-radius validation in robots.py for where that
    # number comes from). At 0.16m the base's own body was found to be
    # essentially touching/overlapping the table -- confirmed by testing,
    # not assumed -- leaving no room to turn or back away cleanly.
    #
    # The numbers below were computed (not guessed) to balance clearance to
    # the pickup table against clearance to the arena's rear wall (fixed at
    # x=-2.0) within the limited space available: the wall-to-task-table
    # budget is only 1.6m total, and the original layout's 0.7m corridor
    # between the wall and a 0.6m table centered 1.0m out was already less
    # than the base's own footprint radius, so no parking spot in that gap
    # could have been fully clear. This still gives ~0.45m clearance on
    # both sides, not the full 0.8m worst case -- verified empirically (see
    # robots.py / test suite) to be enough in practice for this base, since
    # the 0.8m figure is a conservative bound covering every direction at
    # once, not the specific direction that matters when parked facing the
    # task table.
    pickup_table_full_size = (0.4, 0.4, 0.05)
    # The robot's start pose and the pickup table's position are INDEPENDENT --
    # earlier rounds parked the robot right beside the pickup table, which
    # repeatedly forced tradeoffs between clearance to the table, clearance to
    # the arena's walls, and (once obstacles were actually visible to the
    # fixed LIDAR) crowding the route with all three. Decoupling them and
    # moving the start into the open middle of the arena removes that
    # tension entirely: neither point is now within 2m of any wall.
    #
    # start=(0, 2) [+ task table's own x], pickup=(0.6, 0.75): both offsets
    # from the task table's own (x, y). Checked, not just placed by eye:
    #   - route length (start to goal) ~2.0-2.1m (nut/wipe) -- was ~1.5-1.8m,
    #     so there's more room for 3 obstacles without crowding the goal.
    #   - lateral clearance from the straight route to the pickup table's
    #     centre, at the table's own y: ~0.95m (nut and wipe) -- well past the
    #     ~0.7-0.8m a 0.5m-radius chassis passing a 0.4m table needs.
    #   - clearance from the start pose to the nearest wall: >=1.0m (nut and
    #     wipe); from the pickup table to the nearest wall: >=2.3m. (The
    #     previous layout's worst clearance was 0.24-0.44m, because it sat in
    #     a corner to also get distance from the pickup table.)
    #   - zero contacts at reset, both tasks (`sim.data.ncon`).
    robot_start_offset_delta = np.array([0.0, 2.0, 0.0])
    pickup_table_offset_delta = np.array([0.6, 0.75, 0.0])
    n_obstacles = 3
    # Bin-sized, and deliberately TALLER than the LIDAR plane (z=0.5m). The
    # obstacles were originally 0.2m tall: they spawn slightly above the floor,
    # are seen by the ring for a fraction of a second, then fall to z in
    # [0, 0.2] -- entirely below the ring -- so the robot was blind to them for
    # the whole run and shoved them aside (~100-200 steps of contact per run).
    # They are also wider than before: at ~1.4m from the base centre the 16
    # beams are ~0.55m apart, so an 8cm object is usually slipped between.
    obstacle_half_size = (0.08, 0.08, 0.30)
    # Obstacles are scattered ALONG THE TRANSIT ROUTE (the straight line from the
    # robot's parking spot to the manipulation pose), with a lateral jitter --
    # not in a rectangle derived from table edges. That earlier scheme left a
    # strip only 0.15m tall once the pickup table moved, which both crashed the
    # placement sampler (RandomizationError) and squeezed the bins into the gap
    # between the route and the table. They are kept OUT of the manipulation
    # zone (that is where the base must end up; a bin on the parking spot would
    # make the goal unreachable) and away from the pickup table.
    # obstacle_route_frac's upper bound matters a lot: the manipulation keep-out
    # zone is 0.8m deep, which on a ~2m route already covers roughly the last
    # 40% (fraction 0.6-1.0) -- an earlier version scattered obstacles up to
    # fraction 0.60, i.e. right up against the zone's own boundary, so on
    # average 1.5-2 of the 3 bins landed within 0.3m of the entrance every
    # single reset (measured over 30 resets per task) -- exactly a wall of
    # obstacles blocking the doorway. Stopping at 0.45 leaves a substantial,
    # genuinely obstacle-free final approach.
    # (Values re-measured for the new, longer route: 100/100 placements
    # succeed with the numbers below.)
    obstacle_route_frac = (0.05, 0.45)   # portion of the route to scatter over -- start exclusion is a radius (below), not this
    obstacle_lateral_range = 0.50        # +- jitter across the route (m)
    obstacle_min_spacing = 0.35          # min centre-to-centre distance (bins are 0.16m wide)
    obstacle_keepout_gap = 0.20          # stay this far outside the manipulation zone
    obstacle_pickup_gap = 0.10           # stay this far from the pickup table's footprint
    # Radius around the START pose to keep clear, separate from the route-fraction
    # bounds above: those measure distance ALONG the route, but the chassis
    # extends ~0.55m behind its own origin and ~0.25m to each side, so a bin at
    # even 10% of a 2m route (0.2m along-route) can already be well inside the
    # chassis footprint depending on which side of the route it lands on -- found
    # by testing: the pedestal was in contact with an obstacle at ~35% of resets.
    obstacle_start_gap = 0.75
    # The base is holonomic with yaw held at 0 (see GridPlannerPolicy), so its
    # footprint about the base origin is a FIXED rectangle: the chassis, 0.55m
    # behind the origin, 0.15m ahead, 0.25m to each side.
    chassis_footprint = ((-0.55, -0.25), (0.15, 0.25))
    # Every generated course must be SOLVABLE: a BFS on the true geometry has to
    # find a route from start to goal with this much extra clearance (larger than
    # the planner's own 0.12m, so a solvable course is solvable with room to
    # spare). Unsolvable draws are rejected and re-sampled.
    course_feasibility_margin = 0.15
    course_max_resamples = 200
    # The "simple" course must really be the easy one: its shortest route is at most this long. (With
    # the solid corner walls, many random scatterings leave only a ~5.5 m way round, almost as long as
    # the "long" course; those are rejected.)
    simple_course_max_route = 3.6
    # Where the base ORIGIN can be now that the walls are solid: the arena is x in [-2, 3],
    # y in [-3, 3]. Behind the origin the chassis reaches 0.55 m, to each side 0.25 m, and AHEAD the
    # folded arm reaches 0.43 m (measured; the chassis alone only 0.15 m, but the walls are 3 m
    # tall, so it is the arm that touches the east wall), plus a 5 cm gap.
    # (The chamfered rear corners are ignored: nothing is placed there.)
    course_bounds = ((-1.40, 2.50), (-2.70, 2.70))
    # The two rear corners of the arena are cut by diagonal walls: y = x + 3.5 (north-west) and
    # y = -x - 3.5 (south-west). The chassis' worst corner (0.55 m behind the origin, 0.25 m to the
    # side) is 0.8 m "further out" along the diagonal's normal axis, plus the 5 cm gap (x 1.41):
    # the origin must keep  -x + y <= 2.62  and  -x - y <= 2.62.  Each tuple is (a, b, c): a*x + b*y > c is blocked.
    course_half_planes = ((-1.0, 1.0, 2.62), (-1.0, -1.0, 2.62))

    # ---- the two kinds of course (`course` argument of the env; see `_course_kind`) ----
    # "simple": the three bins are scattered along the straight route (above): a short detour.
    # "long":   two bins form a BARRIER that closes the north and west approaches to the
    #           manipulation zone (a bin blocks far more than its 16 cm: the chassis is 0.7 m
    #           long, so each bin shuts a ~1.2 x 1.0 m region to the base origin), so the short
    #           way is gone and the only route goes east around the pickup table, south of the
    #           task table, and up into the zone from the south. The third bin is scattered
    #           along that long route. Every layout is checked: a route must exist, and the
    #           shortest one must be at least `long_course_min_route` metres (the empty
    #           course's shortest route is ~2.3 m). That length is measured twice: with the
    #           usual 15 cm clearance margin, and with a NEGATIVE margin: the model of bins and
    #           chassis is a little conservative (bins are modelled as their worst-case rotated
    #           square, the grid is 5 cm), and a barrier that only seals at margin 0 can be
    #           driven through in practice (measured). Requiring it to hold at -8 cm makes the
    #           short way really closed, not just narrow.
    long_course_min_route = 5.0
    long_course_seal_margin = -0.08
    # The long route must also be comfortably passable: the shortest route has to exist with every bin
    # grown by this much (a third bin 3 cm from the pickup table left a gap only one grid cell wide,
    # which the nominal check accepted but a planner that keeps 10 cm from tables cannot use).
    long_course_extra_clearance = 0.10
    long_course_keepout_gap = 0.05     # barrier bins may sit this close to the zone (the zone itself stays free)
    # Barrier bins, relative to the manipulation zone (x_west = its west edge, y_top = its north
    # edge): (x offset from x_west, y offset from y_top), as (low, high) ranges.
    long_course_bin_a = ((0.15, 0.35), (0.18, 0.38))
    long_course_bin_b = ((-0.55, -0.30), (0.50, 0.80))
    # Region (world x, y) for the third bin: east of the barrier, around the pickup table.
    long_course_bin_c = ((0.2, 2.4), (-1.6, 1.6))
    long_course_person_gap = 0.10      # bins stay this far (beyond their own size) from the person's swath

    # Strip in front of the task table's near edge where the base has to
    # park to manipulate (stock robosuite parks it 0.16m from that edge; its
    # chassis extends ~0.5-0.8m behind that point, and ~0.5m to each side).
    # Nothing scripted -- the walking person in particular -- should stand
    # or walk in here, or the integrated (navigate-then-manipulate) run can
    # be blocked from ever starting its manipulation phase.
    manipulation_keepout_depth = 0.8
    manipulation_keepout_side_margin = 0.1

    def manipulation_base_xy(self):
        """World (x, y) where the stock env parks the base for the task table --
        i.e. the pose the manipulation phase is developed at (mobile_phase=False).
        This, not the table's centre, is the navigation goal: the arm's shoulder
        stops the base ~0.05m short of the tabletop, so the centre (0.56m from
        this pose in both tasks) can never be reached."""
        fn = self.robots[0].robot_model.base_xpos_offset["table"]
        return np.array(fn(self.table_full_size[0])[:2], dtype=float)

    def table_map(self):
        """The two tables as a static map, for planners: a list of
        (center_xy, full_size_xy, skirt_inset) tuples. The slab (full size) is
        what the arm's shoulder link can hit; the solid skirt, recessed by
        `skirt_inset` on each side, is what the chassis and the LIDAR see.
        Fixed over an episode, so it is legitimate prior knowledge."""
        return [
            (np.asarray(self.pickup_table_offset[:2], dtype=float),
             np.asarray(self.pickup_table_full_size[:2], dtype=float), 0.02),
            (np.asarray(self.table_offset[:2], dtype=float),
             np.asarray(self.table_full_size[:2], dtype=float), 0.08),
        ]

    def manipulation_keepout(self):
        """(center_xy, size_xy) of the axis-aligned zone in front of the task
        table that must stay clear for the manipulation phase."""
        half = np.array(self.table_full_size[:2], dtype=float) / 2
        front_x = self.table_offset[0] - half[0]
        depth = self.manipulation_keepout_depth
        center = np.array([front_x - depth / 2, self.table_offset[1]])
        size = np.array([depth, 2 * (half[1] + self.manipulation_keepout_side_margin)])
        return center, size

    def __init__(self, *args, mobile_phase=True, course=None, **kwargs):
        # Must be set before super().__init__(), which triggers _load_model().
        if course not in (None, "simple", "long"):
            raise ValueError(f"course must be None, 'simple' or 'long', got {course!r}")
        self.mobile_phase = mobile_phase
        self.course = course
        super().__init__(*args, **kwargs)

    def _course_kind(self):
        """"simple" or "long". If `course` was not given: "long" whenever a person is present (the
        harder set-up), "simple" otherwise."""
        if self.course is not None:
            return self.course
        return "long" if self._extra_blockers() else "simple"

    def _load_model(self):
        super()._load_model()  # builds the task table + task objects as normal

        if not self.mobile_phase:
            return  # manipulation-only: leave everything exactly as stock robosuite built it

        self.pickup_table_offset = add_pickup_table(
            self.model.mujoco_arena,
            task_table_offset=self.table_offset,
            pickup_full_size=self.pickup_table_full_size,
            pickup_offset_delta=self.pickup_table_offset_delta,
        )

        # Both tables' legs are visual-only in robosuite's TableArena family
        # (reasonable for the original fixed-arm, single-table tasks; not
        # for a mobile base driving past them at floor level -- confirmed by
        # a user visually observing the base drive straight through one).
        harden_table_legs(self.model.mujoco_arena.worldbody)
        add_task_table_skirt(self.model.mujoco_arena.worldbody)
        add_solid_walls(self.model.mujoco_arena.worldbody)     # the walls stop the base (see arenas.py)

        # Start the robot away from the pickup table (see class docstring) -- it
        # has to drive itself to the task table regardless of where it starts,
        # so unlike the task table's own manipulation pose there's no
        # "approach formula" to match; place it directly.
        self.robot_start_xy = (np.asarray(self.table_offset[:2], dtype=float)
                                + self.robot_start_offset_delta[:2])
        self.robots[0].robot_model.set_base_xpos(
            np.array([self.robot_start_xy[0], self.robot_start_xy[1], 0.0]))

        # Floor obstacles between the two tables.
        self.obstacles = make_obstacles(self.n_obstacles, half_size=self.obstacle_half_size)

        existing_objects = list(getattr(self.model, "mujoco_objects", []))
        self.model = ManipulationTask(
            mujoco_arena=self.model.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=existing_objects + self.obstacles,
        )

    def _sample_obstacle_xy_once(self):
        """Random obstacle centres along the transit route (see class attributes)."""
        start = np.asarray(self.robot_start_xy, dtype=float)
        route = self.manipulation_base_xy() - start
        u = route / np.linalg.norm(route)
        normal = np.array([-u[1], u[0]])
        kc, ks = self.manipulation_keepout()
        h = self.obstacle_half_size[0]
        k_lo = kc - ks / 2 - (self.obstacle_keepout_gap + h)
        k_hi = kc + ks / 2 + (self.obstacle_keepout_gap + h)
        pc = np.asarray(self.pickup_table_offset[:2], dtype=float)
        ph = np.array(self.pickup_table_full_size[:2], dtype=float) / 2
        p_lo = pc - ph - (self.obstacle_pickup_gap + h)
        p_hi = pc + ph + (self.obstacle_pickup_gap + h)
        start_gap = self.obstacle_start_gap + h
        for spacing_scale in (1.0, 0.75, 0.5):     # relax spacing rather than crash reset() on a tight layout
            spacing = self.obstacle_min_spacing * spacing_scale
            chosen = []
            for _ in range(3000):
                f = np.random.uniform(*self.obstacle_route_frac)
                lateral = np.random.uniform(-self.obstacle_lateral_range, self.obstacle_lateral_range)
                p = start + f * route + lateral * normal
                if np.linalg.norm(p - start) < start_gap:
                    continue
                if np.all(p > k_lo) and np.all(p < k_hi):
                    continue
                if np.all(p > p_lo) and np.all(p < p_hi):
                    continue
                if any(np.linalg.norm(p - q) < spacing for q in chosen):
                    continue
                chosen.append(p)
                if len(chosen) == self.n_obstacles:
                    return chosen
        raise RuntimeError(
            f"could not place {self.n_obstacles} obstacles along the route with spacing "
            f"{self.obstacle_min_spacing}m -- relax obstacle_min_spacing / the exclusion gaps")

    def _sample_long_course_xy_once(self):
        """Bin centres for the "long" course (see the class attributes): two barrier bins north-west
        of the manipulation zone and one bin somewhere on the long route. None if a draw is unusable."""
        kc, ks = self.manipulation_keepout()
        x_west, y_top = kc[0] - ks[0] / 2, kc[1] + ks[1] / 2
        h = self.obstacle_half_size[0]
        u = np.random.uniform
        a = np.array([x_west + u(*self.long_course_bin_a[0]), y_top + u(*self.long_course_bin_a[1])])
        b = np.array([x_west + u(*self.long_course_bin_b[0]), y_top + u(*self.long_course_bin_b[1])])
        chosen = [a, b]

        start = np.asarray(self.robot_start_xy, dtype=float)
        k_lo = kc - ks / 2 - (self.long_course_keepout_gap + h)
        k_hi = kc + ks / 2 + (self.long_course_keepout_gap + h)
        pc = np.asarray(self.pickup_table_offset[:2], dtype=float)
        ph = np.array(self.pickup_table_full_size[:2], dtype=float) / 2
        p_lo = pc - ph - (self.obstacle_pickup_gap + h)
        p_hi = pc + ph + (self.obstacle_pickup_gap + h)
        person = [(np.asarray(c, dtype=float), np.asarray(half, dtype=float) + h + self.long_course_person_gap)
                  for c, half in self._extra_blockers()]

        def acceptable(p, others):
            if np.linalg.norm(p - start) < self.obstacle_start_gap + h:
                return False
            if np.all(p > k_lo) and np.all(p < k_hi):
                return False
            if np.all(p > p_lo) and np.all(p < p_hi):
                return False
            if any(np.all(np.abs(p - c) < half) for c, half in person):
                return False
            return all(np.linalg.norm(p - q) >= self.obstacle_min_spacing for q in others)

        if not all(acceptable(p, [q for q in chosen if q is not p]) for p in chosen):
            return None                                   # barrier bins collide with something: draw again
        for _ in range(500):
            c = np.array([u(*self.long_course_bin_c[0]), u(*self.long_course_bin_c[1])])
            if acceptable(c, chosen):
                chosen.append(c)
                break
        else:
            return None
        order = np.random.permutation(len(chosen))       # the bins are identical: do not always put the same one first
        return [chosen[i] for i in order]

    def _extra_blockers(self):
        """Extra (center_xy, half_xy) rectangles that must stay avoidable, beyond the
        bins and tables -- e.g. the person's whole patrol swath. Overridden by
        PersonMixin; empty here."""
        return []

    def course_is_solvable(self, bin_xy):
        """True if a collision-free route start -> goal exists for these bin
        positions (and the tables and any `_extra_blockers`), with
        `course_feasibility_margin` of clearance."""
        h = self.obstacle_half_size[0] * np.sqrt(2.0)      # bins are randomly rotated: circumscribed half-extent
        rects = [(np.asarray(p, dtype=float), (h, h)) for p in bin_xy] + list(self._extra_blockers())
        return route_exists(self.robot_start_xy, self.manipulation_base_xy(), rects, self.table_map(),
                            self.chassis_footprint, self.course_feasibility_margin, bounds=self.course_bounds,
                            half_planes=self.course_half_planes)

    def course_route_length(self, bin_xy, margin=None, bin_grow=0.0):
        """Length (m) of the shortest collision-free route start -> goal for these bin positions
        (same free space as `course_is_solvable`, with clearance `margin`, default
        `course_feasibility_margin`, and every bin grown by `bin_grow`), or None if there is none."""
        h = self.obstacle_half_size[0] * np.sqrt(2.0) + bin_grow
        rects = [(np.asarray(p, dtype=float), (h, h)) for p in bin_xy] + list(self._extra_blockers())
        return route_length(self.robot_start_xy, self.manipulation_base_xy(), rects, self.table_map(),
                            self.chassis_footprint, self.course_feasibility_margin if margin is None else margin,
                            bounds=self.course_bounds, half_planes=self.course_half_planes)

    def _sample_obstacle_xy(self):
        """Bin positions for this reset: random, but only accepted if the course is solvable (and,
        for the "long" course, if the shortest route is at least `long_course_min_route` metres, also
        with the negative seal margin). Records how many draws that took (`last_course_draws`) and
        the shortest route length (`last_course_route_length`, metres)."""
        kind = self._course_kind()
        for attempt in range(1, self.course_max_resamples + 1):
            xy = self._sample_long_course_xy_once() if kind == "long" else self._sample_obstacle_xy_once()
            if xy is None or not self.course_is_solvable(xy):
                continue
            length = self.course_route_length(xy)
            if kind == "simple" and (length is None or length > self.simple_course_max_route):
                continue
            if kind == "long":
                sealed = self.course_route_length(xy, margin=self.long_course_seal_margin)
                roomy = self.course_route_length(xy, bin_grow=self.long_course_extra_clearance)
                if any(v is None or v < self.long_course_min_route for v in (length, sealed, roomy)):
                    continue
            self.last_course_draws = attempt
            self.last_course_route_length = length
            return xy
        raise RuntimeError(
            f"no solvable {kind} bin layout found in {self.course_max_resamples} draws -- the course "
            f"geometry (start, goal, tables, person) may itself be unsolvable")

    def _reset_internal(self):
        super()._reset_internal()
        if self.mobile_phase and getattr(self, "obstacles", None):
            for obstacle, (x, y) in zip(self.obstacles, self._sample_obstacle_xy()):
                yaw = np.random.uniform(-np.pi, np.pi)
                quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                # dropped from a hair above the floor so they settle without bouncing
                self.sim.data.set_joint_qpos(
                    obstacle.joints[0], np.array([x, y, self.obstacle_half_size[2] + 0.01, *quat]))
