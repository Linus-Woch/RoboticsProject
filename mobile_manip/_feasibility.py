# Copyright (c) 2026 Universitat Politècnica de Catalunya (UPC),
# Barcelona East School of Engineering (EEBE).
# Robotics Systems MSc course 295II235, 2026-2027.
# All rights reserved.

"""
Private helper: is there ANY collision-free route for the base from its start to
its goal, given the true bin/table/person geometry, and how long is the shortest
one? Used by the environment to reject unsolvable random courses at reset (a
course the planner cannot solve says nothing about the planner) and to make sure
a "long" course really is long.

Deliberately a coarse grid search -- it is a validity check, not a planner, and it
uses a larger safety margin than any reasonable planner needs. The base is
holonomic and its yaw is held at 0, so the footprint orientation is fixed and the
free space for the base ORIGIN is exactly: each obstacle dilated by the reflected
chassis rectangle plus a margin.
"""

import heapq
from collections import deque

import numpy as np


def _blocked_grid(rects, tables, footprint, margin, bounds, res, shoulder_radius, table_margin, half_planes=()):
    """The occupancy grid of the free space for the base ORIGIN: returns (blocked, to_cell) where
    `blocked[i, j]` is True for a cell the origin cannot be in and `to_cell(xy)` gives the (i, j) of
    a world point."""
    (x0, x1), (y0, y1) = bounds
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    cx = (x0 + (np.arange(nx) + 0.5) * res)[:, None]
    cy = (y0 + (np.arange(ny) + 0.5) * res)[None, :]

    def block(center, half, offs):
        """Origin cells that make the chassis overlap the rectangle (center, half). Computed directly,
        NOT by shifting a grid mask, so an obstacle outside the grid (a bin near the wall) still blocks
        the cells of the grid that its no-go region reaches."""
        return ((cx >= center[0] - half[0] + offs[0]) & (cx <= center[0] + half[0] + offs[1]) &
                (cy >= center[1] - half[1] + offs[2]) & (cy <= center[1] + half[1] + offs[3]))

    lo, hi = np.array(footprint[0], dtype=float), np.array(footprint[1], dtype=float)
    offsets = (-hi[0] - margin, -lo[0] + margin, -hi[1] - margin, -lo[1] + margin)
    table_offsets = (-hi[0] - table_margin, -lo[0] + table_margin, -hi[1] - table_margin, -lo[1] + table_margin)
    shoulder = (-shoulder_radius, shoulder_radius, -shoulder_radius, shoulder_radius)
    blocked = np.zeros((nx, ny), dtype=bool)
    for a, b, c in half_planes:                       # e.g. the arena's diagonal corner walls: a*x + b*y > c is blocked
        blocked |= (a * cx + b * cy) > c
    for center, half in rects:
        blocked |= block(center, half, offsets)
    for center, full_size, inset in tables:
        half = np.asarray(full_size, dtype=float) / 2
        blocked |= block(center, half - inset, table_offsets)
        blocked |= block(center, half, shoulder)

    def to_cell(xy):
        return int(np.floor((xy[0] - x0) / res)), int(np.floor((xy[1] - y0) / res))

    return blocked, to_cell


_MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def route_exists(start_xy, goal_xy, rects, tables, footprint, margin,
                 bounds=((-1.9, 2.9), (-2.9, 2.9)), res=0.05, shoulder_radius=0.10, table_margin=0.03,
                 half_planes=()):
    """
    rects:  list of (center_xy, half_xy) obstacles that block the CHASSIS (bins,
            the person's patrol swath).
    tables: list of (center_xy, full_size_xy, skirt_inset): the skirt blocks the
            chassis, the full slab blocks the arm's shoulder (a symmetric disc).
    footprint: ((x_lo, y_lo), (x_hi, y_hi)) of the chassis about the base origin.
    margin: clearance around bins/the person (may be negative to ask "would it still be closed if
            my model were this much too conservative?"). `table_margin` (small) applies to the
            tables instead: the goal is, by design, only ~0.09m from the task table's
            skirt, so the tables' margin must be small or the goal itself is blocked.
    """
    blocked, to_cell = _blocked_grid(rects, tables, footprint, margin, bounds, res, shoulder_radius, table_margin,
                                     half_planes)
    nx, ny = blocked.shape
    start, goal = to_cell(start_xy), to_cell(goal_xy)
    for c in (start, goal):
        if not (0 <= c[0] < nx and 0 <= c[1] < ny) or blocked[c]:
            return False
    seen = np.zeros((nx, ny), dtype=bool)
    seen[start] = True
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in _MOVES:
            X, Y = x + dx, y + dy
            if not (0 <= X < nx and 0 <= Y < ny) or seen[X, Y] or blocked[X, Y]:
                continue
            if dx and dy and (blocked[x + dx, y] or blocked[x, y + dy]):
                continue
            seen[X, Y] = True
            queue.append((X, Y))
    return False


def route_length(start_xy, goal_xy, rects, tables, footprint, margin,
                 bounds=((-1.9, 2.9), (-2.9, 2.9)), res=0.05, shoulder_radius=0.10, table_margin=0.03,
                 half_planes=()):
    """Length in metres of the SHORTEST collision-free route (same free space as `route_exists`), or
    None if there is none. Dijkstra on the same grid with 8-neighbour moves, so it slightly
    overestimates a smooth path (by a few percent)."""
    blocked, to_cell = _blocked_grid(rects, tables, footprint, margin, bounds, res, shoulder_radius, table_margin,
                                     half_planes)
    nx, ny = blocked.shape
    start, goal = to_cell(start_xy), to_cell(goal_xy)
    for c in (start, goal):
        if not (0 <= c[0] < nx and 0 <= c[1] < ny) or blocked[c]:
            return None
    dist = np.full((nx, ny), np.inf)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, (x, y) = heapq.heappop(heap)
        if d > dist[x, y]:
            continue
        if (x, y) == goal:
            return float(d) * res
        for dx, dy in _MOVES:
            X, Y = x + dx, y + dy
            if not (0 <= X < nx and 0 <= Y < ny) or blocked[X, Y]:
                continue
            if dx and dy and (blocked[x + dx, y] or blocked[x, y + dy]):
                continue
            nd = d + (1.4142135623730951 if dx and dy else 1.0)
            if nd < dist[X, Y]:
                dist[X, Y] = nd
                heapq.heappush(heap, (nd, (X, Y)))
    return None
