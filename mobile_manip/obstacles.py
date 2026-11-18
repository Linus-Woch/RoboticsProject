# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.obstacles
=======================
Static floor obstacles the MOBILE subgroup needs to avoid while driving
from the pickup table to the task table.
"""

from robosuite.models.objects import BoxObject
from robosuite.utils.placement_samplers import UniformRandomSampler


def make_obstacles(n, half_size=(0.05, 0.05, 0.1), rgba=(0.8, 0.2, 0.2, 1)):
    """Free-floating box obstacles with a real joint, so physics (falling,
    getting bumped, resting on the floor) behaves normally -- no manual
    contact hacks required."""
    return [
        BoxObject(name=f"obstacle{i}", size=half_size, joints="default", rgba=list(rgba))
        for i in range(n)
    ]


def make_obstacle_sampler(obstacles, x_range, y_range, z_offset):
    """
    Samples obstacle placements uniformly in a strip between the two tables.
    Uses robosuite's own placement-validity checking, so obstacles never
    spawn overlapping each other or the tables.
    """
    return UniformRandomSampler(
        name="ObstacleSampler",
        mujoco_objects=obstacles,
        x_range=x_range,
        y_range=y_range,
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=z_offset,
    )
