"""Pose tracking with world-frame odometry and a static 2D LiDAR map."""

import numpy as np


RING_RADIUS_M = 0.25
MAX_RANGE_M = 5.0
RANGE_STD_M = 0.035  # Includes map approximation and scan/odometry timing differences.
GATE_SIGMA = 3.0
MAX_RESIDUAL_M = 0.15


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def cross_xy(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class LidarMap:
    def __init__(self, walls, tables):
        segments = list(walls)
        for centre, size, inset in tables:
            corners = centre + (size / 2 - inset) * np.array(
                [[-1, -1], [1, -1], [1, 1], [-1, 1]]
            )
            segments.extend(zip(corners, np.roll(corners, -1, axis=0)))
        self.starts, ends = np.asarray(segments, dtype=float).transpose(1, 0, 2)
        self.edges = ends - self.starts
        self.lengths = np.linalg.norm(self.edges, axis=1)

    def predict(self, pose, beam_count):
        """Predict nearest ranges and their pose derivatives for stable hits."""
        angles = pose[2] + np.arange(beam_count) * (2 * np.pi / beam_count)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        direction_derivatives = np.column_stack((-directions[:, 1], directions[:, 0]))

        # Solve pose_xy + distance * direction = start + fraction * edge.
        offsets = self.starts - pose[:2]
        denominators = cross_xy(directions[:, None, :], self.edges)
        nonparallel = np.abs(denominators) > 1e-10
        safe_denominators = np.where(nonparallel, denominators, 1.0)
        distances = cross_xy(offsets, self.edges) / safe_denominators
        fractions = cross_xy(offsets, directions[:, None, :]) / safe_denominators
        intersects = nonparallel & (distances >= RING_RADIUS_M) & (fractions >= 0) & (fractions <= 1)
        candidates = np.where(intersects, distances - RING_RADIUS_M, np.inf)
        nearest = np.argmin(candidates, axis=1)
        beams = np.arange(beam_count)
        ranges = np.minimum(candidates[beams, nearest], MAX_RANGE_M)

        # Near a corner or at a shallow angle, the nearest edge can change
        # abruptly with pose. Skip those hits when linearising the range.
        edge = self.edges[nearest]
        length = self.lengths[nearest]
        fraction = fractions[beams, nearest]
        denominator = safe_denominators[beams, nearest]
        stable = (
            (ranges < MAX_RANGE_M)
            & (np.minimum(fraction, 1 - fraction) * length > 0.05)
            & (np.abs(denominator) > 0.25 * length)
        )
        jacobian = np.column_stack((
            -edge[:, 1],
            edge[:, 0],
            -(ranges + RING_RADIUS_M) * cross_xy(direction_derivatives, edge),
        )) / denominator[:, None]
        return ranges, jacobian, stable


class PoseEKF:
    def __init__(self, lidar_map):
        self.lidar_map = lidar_map
        self.previous_odom = None
        self.pose = None
        # The first odometry reading gives the calibrated starting pose.
        self.covariance = np.zeros((3, 3))
        self.used_beams = np.array([], dtype=bool)

    def update(self, odometry, scan):
        odometry = np.asarray(odometry, dtype=float)
        scan = np.asarray(scan, dtype=float)
        if self.previous_odom is None:
            self.pose = odometry.copy()
            self.previous_odom = odometry.copy()
            self.used_beams = np.zeros(scan.size, dtype=bool)
            return self.pose.copy()

        # Odometry increments are already world-aligned. The motion Jacobian
        # is the identity, so prediction adds motion noise to the covariance.
        delta = odometry - self.previous_odom
        delta[2] = wrap_angle(delta[2])
        self.previous_odom = odometry.copy()
        self.pose += delta
        self.pose[2] = wrap_angle(self.pose[2])
        distance = np.linalg.norm(delta[:2])
        motion_std = np.array([
            0.0025 + 0.01 * distance,
            0.0025 + 0.01 * distance,
            0.0025 + 0.01 * (distance + abs(delta[2])),
        ])
        self.covariance += np.diag(motion_std ** 2)

        predicted, jacobian, stable = self.lidar_map.predict(self.pose, scan.size)
        residual = scan - predicted
        variance = RANGE_STD_M ** 2
        innovation_std = np.sqrt(np.sum((jacobian @ self.covariance) * jacobian, axis=1) + variance)
        gate = np.minimum(GATE_SIGMA * innovation_std, MAX_RESIDUAL_M)
        self.used_beams = stable & (scan > 0) & (scan < MAX_RANGE_M) & (np.abs(residual) <= gate)
        if np.count_nonzero(self.used_beams) < 2:
            self.used_beams[:] = False
            return self.pose.copy()

        # Correct one beam at a time, keeping the same linearisation for
        # this scan. Subtract the effect of corrections from earlier beams.
        correction = np.zeros(3)
        for row, error in zip(jacobian[self.used_beams], residual[self.used_beams]):
            projected = self.covariance @ row
            gain = projected / (row @ projected + variance)
            correction += gain * (error - row @ correction)
            self.covariance -= np.outer(projected, gain)
        self.pose += correction
        self.pose[2] = wrap_angle(self.pose[2])
        return self.pose.copy()
