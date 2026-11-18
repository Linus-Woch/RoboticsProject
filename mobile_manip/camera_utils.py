# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
mobile_manip.camera_utils
============================
Thin, VERIFIED wrappers around robosuite's own camera math
(`robosuite.utils.camera_utils`) for the two things a camera-based
manipulation policy needs: turning the raw depth image into real distances,
and turning a (pixel, depth) pair into a 3D world point.

Why this file exists instead of just pointing at robosuite's functions
directly: `transform_from_pixels_to_world` needs the INVERSE of the FULL
(intrinsic + extrinsic) camera matrix as its `camera_to_world_transform`
argument. `get_camera_extrinsic_matrix` alone -- which is the more obvious
choice, since it's the one literally described as "camera pose in the world
frame" -- gives an answer that LOOKS plausible (right order of magnitude
pixel, no crash) but is wrong by tens of metres. This was caught by a
round-trip test (project a known 3D point to a pixel, unproject it back,
compare) -- catching it by inspection alone would have been easy to miss,
since nothing about calling it "wrong" throws an error.
"""

import numpy as np

from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    get_real_depth_map,
    project_points_from_world_to_camera,
    transform_from_pixels_to_world as _transform_from_pixels_to_world,
)


def depth_to_meters(env, depth_image):
    """Converts MuJoCo's raw depth image (normalized to [0, 1]) to real
    distances in meters. `depth_image` is `obs["<camera>_depth"]`."""
    return get_real_depth_map(env.sim, depth_image)


def pixel_to_world(env, camera_name, pixel_rc, depth_image_meters, camera_height, camera_width):
    """
    The 3D world position of one pixel, given its depth.

    Args:
        env: the robosuite/mobile_manip environment (needs `env.sim`).
        camera_name: e.g. `mobile_manip.WRIST_CAMERA`.
        pixel_rc: (row, col) pixel coordinates -- e.g. from
            `world_to_pixel`, or from your own detector (mind the
            row/col vs x/y order: row is the vertical/height index).
        depth_image_meters: the REAL depth image (from `depth_to_meters`,
            not the raw `obs["<camera>_depth"]`), shape (H, W, 1) or (H, W).
        camera_height, camera_width: pixel dimensions of that image.

    Returns:
        np.ndarray (3,): world-frame (x, y, z) in meters.
    """
    world_to_pixel_matrix = get_camera_transform_matrix(env.sim, camera_name, camera_height, camera_width)
    # NOT get_camera_extrinsic_matrix() -- see module docstring.
    pixel_to_world_matrix = np.linalg.inv(world_to_pixel_matrix)
    depth_image_meters = np.asarray(depth_image_meters).reshape(camera_height, camera_width, 1)
    pixel_rc = np.asarray(pixel_rc, dtype=float).reshape(1, 2)
    point = _transform_from_pixels_to_world(
        pixel_rc, depth_image_meters.reshape(1, camera_height, camera_width, 1), pixel_to_world_matrix
    )
    return point.reshape(3)


def world_to_pixel(env, camera_name, point_xyz, camera_height, camera_width):
    """The (row, col) pixel a world point projects to (for testing your own
    `pixel_to_world` calls, or checking whether a known point is in frame --
    see `point_in_camera_view`)."""
    world_to_pixel_matrix = get_camera_transform_matrix(env.sim, camera_name, camera_height, camera_width)
    return project_points_from_world_to_camera(np.asarray(point_xyz, dtype=float),
                                                world_to_pixel_matrix, camera_height, camera_width)


def point_in_camera_view(env, camera_name, point_xyz, camera_height, camera_width, margin=0):
    """True if `point_xyz` projects inside the camera's frame (and is in
    front of it, not behind). `margin` (pixels) shrinks the frame inward,
    for "is this comfortably in view" rather than "is it right at the edge"."""
    row, col = world_to_pixel(env, camera_name, point_xyz, camera_height, camera_width)
    if not (margin <= row < camera_height - margin and margin <= col < camera_width - margin):
        return False
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = env.sim.data.cam_xpos[cam_id]
    cam_forward = -env.sim.data.cam_xmat[cam_id].reshape(3, 3)[:, 2]  # camera looks down its own -Z
    return bool(np.dot(np.asarray(point_xyz) - cam_pos, cam_forward) > 0)
