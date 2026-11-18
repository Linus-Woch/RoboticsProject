# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.arenas
====================
Builds the physical two-table layout:

    [ Table A: pickup ] ... open floor with obstacles ... [ Table B: task ]

Table B is whatever table the wrapped robosuite task (NutAssembly / Wipe)
already builds -- we don't touch it, so all of its built-in peg/dirt logic
keeps working untouched. Table A is a second, plain table we graft into the
same arena purely as a pickup surface.
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models.arenas import TableArena


def add_pickup_table(mujoco_arena, task_table_offset, pickup_full_size, pickup_offset_delta):
    """
    Grafts a second, plain table (the "pickup" table) into an existing arena
    that already contains the task table, without duplicating the arena's
    floor/walls/cameras.

    Returns:
        np.ndarray: absolute (x, y, z) world offset of the pickup table.
    """
    pickup_table_offset = np.array(task_table_offset) + np.array(pickup_offset_delta)

    pickup_table_source = TableArena(
        table_full_size=pickup_full_size,
        table_offset=tuple(pickup_table_offset),
    )

    # Every named element under the table body ("table", "table_collision",
    # leg names, ...) needs a unique name or MuJoCo will reject the merged
    # model as having duplicate names -- the task table already used them.
    pickup_body = pickup_table_source.table_body
    for el in pickup_body.iter():
        name = el.get("name")
        if name is not None:
            el.set("name", f"pickup_{name}")

    _add_solid_skirt(pickup_body, prefix="pickup_")
    mujoco_arena.worldbody.append(pickup_body)
    return pickup_table_offset


def _add_solid_skirt(table_body, prefix="", inset=0.02):
    """
    Fills the space under a table's slab with a solid box (a counter instead of
    a stand). Two reasons, both found by testing:
      * The 2D LIDAR ring is at z=0.5m, below the slab (~0.88m), so it cannot
        see a slab at all -- and a table this small has all four legs coincide
        as ONE 5cm post at its centre, which a 16-beam ring almost never hits.
        The robot was blind to the table it starts beside, while its shoulder
        link (which IS at slab height) kept meeting the slab: 65-319 steps of
        contact per run, and most of the failures.
      * It stops the low chassis from ever driving under the table.
    Adds a collision box (group 0) and a matching visual box (group 1).
    `prefix` selects which table ("pickup_" for ours, "" for the stock task table).
    """
    slab = table_body.find(f"./geom[@name='{prefix}table_collision']")
    body_z = float(table_body.get("pos").split()[2])
    slab_z = float(slab.get("pos").split()[2]) if slab.get("pos") else 0.0
    hx, hy, hz = (float(v) for v in slab.get("size").split())
    slab_bottom = body_z + slab_z - hz                 # world height of the slab's underside
    half_h = slab_bottom / 2.0
    for suffix, group, contype, rgba in (("collision", "0", "1", "0.75 0.75 0.78 1"),
                                          ("visual", "1", "0", "0.75 0.75 0.78 1")):
        geom = ET.SubElement(table_body, "geom")
        geom.set("name", f"{prefix}table_skirt_{suffix}")
        geom.set("type", "box")
        geom.set("size", f"{hx - inset:.4f} {hy - inset:.4f} {half_h:.4f}")
        geom.set("pos", f"0 0 {half_h - body_z:.4f}")   # body frame: floor at -body_z
        geom.set("contype", contype)
        geom.set("conaffinity", contype)
        geom.set("group", group)
        geom.set("rgba", rgba)


def harden_table_legs(worldbody):
    """
    Gives real collision to any purely-visual table legs found under
    `worldbody`. robosuite's TableArena (and its subclasses: PegsArena,
    WipeArena, and the plain TableArena we use for the pickup table) model
    each table leg as a VISUAL-ONLY cylinder (contype=0, conaffinity=0) --
    only the flat tabletop slab has real collision. That's a reasonable
    simplification for the original single-table, fixed-arm tasks (a
    manipulator reaching down from above never interacts with the legs at
    all) but means a mobile base driving past at floor level can drive
    straight through a table's legs with zero resistance -- confirmed by a
    user visually observing exactly that happening, not a hypothetical.

    Adds a same-size, same-position collision-only cylinder alongside each
    visual leg found (matched by "leg" in the geom name, contype="0") --
    real geometry for physics/LIDAR, but kept on group 0 (this codebase's
    convention for collision meshes, hidden from clean camera observations)
    and not otherwise touching the existing visual geom, so nothing about
    rendering changes.

    Returns the number of collision geoms added (0 is not necessarily wrong
    -- it just means no visual-only legs were found under `worldbody`).
    """
    added = 0
    for body in worldbody.iter("body"):
        for geom in list(body.findall("geom")):
            name = geom.get("name") or ""
            if "leg" not in name.lower() or geom.get("contype") != "0":
                continue
            collision = ET.SubElement(body, "geom")
            collision.set("name", f"{name}_added_collision")
            collision.set("type", geom.get("type", "cylinder"))
            collision.set("size", geom.get("size", "0.025 0.4375"))
            for attr in ("pos", "quat", "euler"):
                if geom.get(attr) is not None:
                    collision.set(attr, geom.get(attr))
            collision.set("contype", "1")
            collision.set("conaffinity", "1")
            collision.set("group", "0")
            # NOT alpha=0: mj_ray/rangefinder treat fully-transparent geoms
            # as invisible to raycasts too, which would silently exclude
            # this leg from the LIDAR right after fixing physical
            # collision with it -- group=0 alone is enough to hide it from
            # camera observations per this codebase's convention.
            collision.set("rgba", "0.5 0.5 0.5 1")
            added += 1
    return added


def add_task_table_skirt(worldbody):
    """Same solid skirt for the stock task table, for the navigation phase (see
    _add_solid_skirt): the robot arrives at the manipulation pose diagonally and
    its shoulder link -- at slab height, above the LIDAR plane -- otherwise meets
    a slab the LIDAR cannot see (240+ steps of contact in some runs)."""
    body = worldbody.find(".//body[@name='table']")
    if body is None:
        return False
    # Recessed 8cm like a counter's toe-kick: the stock parking pose puts the
    # chassis front only ~0.01m from the slab edge, so a flush skirt would leave
    # a 3cm margin at the goal. The LIDAR still sees the (slightly recessed) face.
    _add_solid_skirt(body, prefix="", inset=0.08)
    return True


def add_solid_walls(worldbody, half_thickness=0.05):
    """
    Makes the arena walls solid. robosuite's walls are thin visual-only boxes, so a base driving
    at one simply went through it (and a "long way round" could be avoided by leaving the arena).
    For every `wall_*_visual` geom this adds an invisible collision box (`wall_*_solid`, group 0)
    of the same size and orientation but `half_thickness` thick, shifted OUTWARD so that its inner
    face coincides with the inner face of the visual wall: the LIDAR still measures the same range
    to the same wall, and `wall_segments()` (which only reads the `_visual` geoms) is unchanged.
    """
    visuals = [g for g in worldbody.iter("geom") if (g.get("name") or "").startswith("wall_")
               and (g.get("name") or "").endswith("_visual")]
    centers = np.array([[float(v) for v in g.get("pos").split()][:2] for g in visuals])
    room_center = centers.mean(axis=0)
    for g in visuals:
        pos = np.array([float(v) for v in g.get("pos").split()])
        w, x, y, z = (float(v) for v in g.get("quat").split())
        normal = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])   # local z axis in the world frame
        outward = 1.0 if np.dot(normal[:2], pos[:2] - room_center) > 0 else -1.0
        size = [float(v) for v in g.get("size").split()]
        solid_pos = pos + outward * normal * (half_thickness - size[2])
        ET.SubElement(worldbody, "geom", {
            "name": g.get("name").replace("_visual", "_solid"),
            "type": "box",
            "pos": " ".join(f"{v:.6f}" for v in solid_pos),
            "quat": g.get("quat"),
            "size": f"{size[0]} {size[1]} {half_thickness}",
            "contype": "1", "conaffinity": "1", "group": "0",
            "rgba": "0.5 0.5 0.5 0",
        })
