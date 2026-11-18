"""Grid planning around static geometry and recent LiDAR detections."""

from collections import deque
from heapq import heappop, heappush
from math import hypot, sqrt

import numpy as np

from ._feasibility import _blocked_grid, _MOVES
from .localisation import LidarMap, MAX_RANGE_M, RING_RADIUS_M, wrap_angle
from .robots import world_velocity_mps_to_base_command


CELL_M = 0.05
CHASSIS = ((-0.55, -0.25), (0.15, 0.25))
BIN_HALF_M = 0.08
BIN_MARGIN_M = 0.09
DETECTION_MEMORY_STEPS = 500
PARK_TOLERANCE_M = 0.025
PARK_YAW_RAD = np.deg2rad(4)
SWEEP_EXTRA_RAD = np.deg2rad(4)  # Room for residual heading and sampling error.
SWEEP_YAW_TOLERANCE_RAD = np.deg2rad(1)
SWEEP_MAX_COMMAND = 0.35


def astar(blocked, start, goal):
    """Return an eight-connected route in grid cells, with no corner cutting."""
    nx, ny = blocked.shape
    for x, y in (start, goal):
        if not (0 <= x < nx and 0 <= y < ny) or blocked[x, y]:
            return []
    frontier = [(hypot(goal[0] - start[0], goal[1] - start[1]), 0.0, start)]
    distance = {start: 0.0}
    parent = {}
    while frontier:
        _, cost, cell = heappop(frontier)
        if cost > distance[cell]:
            continue
        if cell == goal:
            route = [goal]
            while route[-1] != start:
                route.append(parent[route[-1]])
            return route[::-1]

        x, y = cell
        for dx, dy in _MOVES:
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x < nx and 0 <= new_y < ny) or blocked[new_x, new_y]:
                continue
            if dx and dy and (blocked[new_x, y] or blocked[x, new_y]):
                continue
            neighbour = (new_x, new_y)
            new_cost = cost + (sqrt(2) if dx and dy else 1.0)
            if new_cost < distance.get(neighbour, np.inf):
                distance[neighbour] = new_cost
                parent[neighbour] = cell
                priority = new_cost + hypot(goal[0] - new_x, goal[1] - new_y)
                heappush(frontier, (priority, new_cost, neighbour))
    return []


class OccupancyGrid:
    """Track forbidden positions of the base origin for a near-zero heading."""

    def __init__(self, walls, tables, lidar_map):
        self.walls = walls
        self.tables = tables
        self.lidar_map = lidar_map
        self.origin = np.floor(walls.min(axis=(0, 1)) / CELL_M) * CELL_M
        upper = np.ceil(walls.max(axis=(0, 1)) / CELL_M) * CELL_M
        self.bounds = tuple(zip(self.origin, upper))

        # Move each wall inward by the footprint's reach in that direction.
        # The folded arm reaches farther forward than the chassis.
        inside = walls.mean(axis=(0, 1))
        corners = np.array([[-0.55, -0.25], [-0.55, 0.25], [0.43, -0.25], [0.43, 0.25]])
        planes = []
        for start, end in walls:
            edge = end - start
            normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            if normal @ (inside - start) > 0:
                normal = -normal
            limit = normal @ start - np.max(corners @ normal) - 0.07
            planes.append((*normal, limit))
        self.static, self.cell = _blocked_grid(
            [], tables, CHASSIS, 0, self.bounds, CELL_M,
            shoulder_radius=0.16, table_margin=0.07, half_planes=planes,
        )
        self.blocked = self.static.copy()
        self.retreat_blocked = self.static.copy()
        self.detections = {}

    def centre(self, cell):
        return self.origin + (np.asarray(cell) + 0.5) * CELL_M

    def update(self, pose, ranges, step):
        """Remember short returns that cannot be explained by the static map."""
        predicted, _, _ = self.lidar_map.predict(pose, len(ranges))
        angles = pose[2] + np.arange(len(ranges)) * (2 * np.pi / len(ranges))
        rays = np.column_stack((np.cos(angles), np.sin(angles)))
        hits = (ranges > 0) & (ranges < MAX_RANGE_M) & (predicted - ranges > 0.07)
        previous = self.detections.keys()
        detections = {
            cell: seen for cell, seen in self.detections.items()
            if step - seen < DETECTION_MEMORY_STEPS
        }
        points = pose[:2] + (RING_RADIUS_M + ranges[hits, None]) * rays[hits]
        # A ray can just miss a mapped corner while the measured endpoint is
        # still on that wall or skirt. Check point-to-segment distance as well.
        offsets = points[:, None, :] - self.lidar_map.starts
        edges = self.lidar_map.edges
        along = np.sum(offsets * edges, axis=2) / self.lidar_map.lengths ** 2
        separation = offsets - np.clip(along, 0, 1)[..., None] * edges
        unmapped = np.all(np.linalg.norm(separation, axis=2) > 0.055, axis=1)
        for point, ray in zip(points[unmapped], rays[hits][unmapped]):
            # Shift the visible face slightly along the beam, toward the bin's interior.
            cell = self.cell(point + 0.055 * ray)
            if self.inside(cell):
                detections[cell] = step

        if detections.keys() != previous:
            self.blocked = self.inflate_bins(detections, BIN_MARGIN_M)
            self.retreat_blocked = self.inflate_bins(detections, 0.015)
        self.detections = detections

    def inflate_bins(self, detections, margin):
        blocked = self.static.copy()
        half = BIN_HALF_M + margin + CELL_M / 2
        low, high = np.array(CHASSIS)
        for cell in detections:
            centre = self.centre(cell)
            lo = np.maximum(self.cell(centre - half - high), 0)
            hi = np.minimum(np.asarray(self.cell(centre + half - low)) + 1, blocked.shape)
            blocked[lo[0]:hi[0], lo[1]:hi[1]] = True
        return blocked

    def inside(self, cell):
        x, y = cell
        return 0 <= x < self.blocked.shape[0] and 0 <= y < self.blocked.shape[1]

    def segment_clear(self, start, end, *, retreat=False):
        """Check a shortcut, including corners between diagonal grid cells."""
        blocked = self.retreat_blocked if retreat else self.blocked
        delta = end - start
        count = max(1, int(np.ceil(np.linalg.norm(delta) / (CELL_M / 4))))
        indices = np.floor((np.linspace(start, end, count + 1) - self.origin) / CELL_M).astype(int)
        if np.any(indices < 0) or np.any(indices >= blocked.shape):
            return False
        x, y = indices.T
        if np.any(blocked[x, y]):
            return False
        diagonal = (x[1:] != x[:-1]) & (y[1:] != y[:-1])
        return not np.any(blocked[x[1:][diagonal], y[:-1][diagonal]]
                          | blocked[x[:-1][diagonal], y[1:][diagonal]])


class GridNavigator:
    def __init__(self, goal, walls, tables, *, startup_sweep=True):
        self.goal = np.asarray(goal)
        self.grid = OccupancyGrid(walls, tables, LidarMap(walls, tables))
        self.step = 0
        self.route = []
        self.plans = 0
        self.first_route = None
        self.recent_positions = deque(maxlen=64)
        self.startup_sweep = startup_sweep
        self.sweep_start_yaw = None
        self.sweep_phase = 0
        self.sweep_min_yaw = 0.0
        self.sweep_max_yaw = 0.0
        self.sweep_steps = None
        self.sweep_detections = 0

    def command(self, pose, scan):
        self.step += 1
        start = pose[:2]
        self.grid.update(pose, np.asarray(scan), self.step)

        # The grid assumes a near-zero heading for translation. Collect scans
        # while turning in place, then settle back at the driving heading.
        if self.startup_sweep and self.sweep_steps is None:
            yaw_command = self.sweep_command(pose[2], len(scan))
            if yaw_command is not None:
                return np.array([0.0, 0.0, yaw_command])

        cell = self.grid.cell(start)
        yaw_command = np.clip(-2.0 * wrap_angle(pose[2]), -1, 1)
        if not self.grid.inside(cell):
            return np.array([0.0, 0.0, yaw_command])
        if self.grid.blocked[cell]:
            self.route = []
            velocity = self.retreat_velocity(start)
            vx, vy = world_velocity_mps_to_base_command(velocity, pose[2])
            return np.array([vx, vy, yaw_command])
        if not self.recent_positions or np.linalg.norm(start - self.recent_positions[-1]) > CELL_M / 2:
            self.recent_positions.append(start.copy())

        # Keep the current route until its remaining segments become blocked.
        if not self.route or not self.route_clear(start):
            cells = astar(self.grid.blocked, self.grid.cell(start), self.grid.cell(self.goal))
            self.route = [self.grid.centre(cell) for cell in cells[1:]]
            if cells:
                self.route.append(self.goal)
                self.plans += 1
                if self.first_route is None:
                    self.first_route = (start.copy(), np.array(self.route), self.grid.blocked.copy())

        velocity = np.zeros(2)
        if self.route:
            # Follow the furthest reachable waypoint within a short lookahead.
            # Only discard waypoints when their replacement is collision-free.
            last_visible = 0
            for index in range(1, len(self.route)):
                candidate = self.route[index]
                if np.linalg.norm(candidate - start) > 0.45 or not self.grid.segment_clear(start, candidate):
                    break
                last_visible = index
            if last_visible:
                self.route = self.route[last_visible:]
            target = self.route[0]
            if self.grid.segment_clear(start, target):
                delta = target - start
                distance = np.linalg.norm(delta)
                speed = min(0.26, 2.0 * distance) if len(self.route) > 1 else min(0.20, 1.5 * distance)
                if distance > 1e-6:
                    velocity = speed * delta / distance

        vx, vy = world_velocity_mps_to_base_command(velocity, pose[2])
        return np.array([vx, vy, yaw_command])

    def sweep_command(self, yaw, beam_count):
        """Return a turn command until the outward sweep and return finish."""
        if self.sweep_start_yaw is None:
            self.sweep_start_yaw = yaw
        swept = wrap_angle(yaw - self.sweep_start_yaw)
        self.sweep_min_yaw = min(self.sweep_min_yaw, swept)
        self.sweep_max_yaw = max(self.sweep_max_yaw, swept)
        half_span = (2 * np.pi / beam_count + SWEEP_EXTRA_RAD) / 2
        # Sweep symmetrically to limit the arm excursion at either extreme.
        targets = (-half_span, half_span, -self.sweep_start_yaw)
        error = wrap_angle(targets[self.sweep_phase] - swept)
        if abs(error) <= SWEEP_YAW_TOLERANCE_RAD and self.sweep_phase < 2:
            self.sweep_phase += 1
            error = wrap_angle(targets[self.sweep_phase] - swept)
        if abs(error) > SWEEP_YAW_TOLERANCE_RAD:
            return np.sign(error) * np.clip(3.0 * abs(error), 0.10, SWEEP_MAX_COMMAND)
        self.sweep_steps = self.step - 1
        self.sweep_detections = len(self.grid.detections)
        return None

    def retreat_velocity(self, start):
        """Back out slowly if a new detection puts the base in safety padding."""
        for target in reversed(self.recent_positions):
            distance = np.linalg.norm(target - start)
            if distance > 0.35:
                break
            if self.grid.blocked[self.grid.cell(target)]:
                continue
            if self.grid.segment_clear(start, target, retreat=True):
                return (target - start) * min(2.0, 0.12 / max(distance, 1e-6))
        return np.zeros(2)

    def route_clear(self, start):
        return all(self.grid.segment_clear(a, b) for a, b in zip([start, *self.route], self.route))


def plot_navigation(navigator, trajectory, output):
    """Plot the first route and occupancy grid, plus the actual path for evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if navigator.first_route is None:
        print("No valid A* route to plot.")
        return
    start, route, blocked = navigator.first_route
    first_path = np.vstack((start, route))
    driven = np.asarray(trajectory)
    (x_min, x_max), (y_min, y_max) = navigator.grid.bounds
    figure, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    axis.imshow(blocked.T, extent=(x_min, x_max, y_min, y_max), origin="lower",
                cmap="Greys", vmin=0, vmax=1, alpha=0.25, interpolation="nearest")
    for wall in navigator.grid.walls:
        axis.plot(*wall.T, color="black", linewidth=1)
    for centre, size, _ in navigator.grid.tables:
        axis.add_patch(Rectangle(centre - size / 2, *size, color="0.35"))
    chassis_low, chassis_high = np.array(CHASSIS)
    axis.add_patch(Rectangle(start + chassis_low, *(chassis_high - chassis_low),
                             fill=False, edgecolor="tab:blue", label="Starting chassis"))
    axis.plot(*first_path.T, "--", color="tab:blue", label="First A* route")
    axis.plot(*driven.T, color="tab:green", label="Driven path (evaluation)")
    axis.scatter(*navigator.goal, marker="*", color="tab:red", s=110, label="Goal")
    axis.set(xlabel="x (m)", ylabel="y (m)", title="A* route and cells blocked for the base origin")
    axis.legend(loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"Plan plot saved to {output}")
