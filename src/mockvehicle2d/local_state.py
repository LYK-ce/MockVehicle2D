"""Vehicle-owned anchor, odometry estimate, and locally observed occupancy."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from mockvehicle2d.scan import LaserPoint, ScanConfig


UNKNOWN = -1
FREE = 0
OCCUPIED = 1
LOCALIZATION_QUALITIES = frozenset(("nominal", "degraded", "lost"))


def _wrapped(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite(*values: float) -> bool:
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values)


@dataclass(frozen=True)
class AnchorSpec:
    """Known birth pose of ``anchor_map`` in ``global_map``."""

    anchor_id: str
    global_x_m: float
    global_y_m: float
    global_yaw_rad: float
    position_stddev_m: float = 0.0
    yaw_stddev_rad: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_id, str) or not self.anchor_id:
            raise ValueError("anchor_id cannot be empty")
        values = (
            self.global_x_m,
            self.global_y_m,
            self.global_yaw_rad,
            self.position_stddev_m,
            self.yaw_stddev_rad,
        )
        if not _finite(*values) or min(self.position_stddev_m, self.yaw_stddev_rad) < 0:
            raise ValueError("anchor values must be finite and uncertainties cannot be negative")

    def anchor_to_global(self, x_m: float, y_m: float, yaw_rad: float) -> tuple[float, float, float]:
        if not _finite(x_m, y_m, yaw_rad):
            raise ValueError("pose values must be finite")
        cosine, sine = math.cos(self.global_yaw_rad), math.sin(self.global_yaw_rad)
        return (
            self.global_x_m + cosine * x_m - sine * y_m,
            self.global_y_m + sine * x_m + cosine * y_m,
            _wrapped(self.global_yaw_rad + yaw_rad),
        )

    def global_to_anchor(self, x_m: float, y_m: float, yaw_rad: float = 0.0) -> tuple[float, float, float]:
        if not _finite(x_m, y_m, yaw_rad):
            raise ValueError("pose values must be finite")
        dx, dy = x_m - self.global_x_m, y_m - self.global_y_m
        cosine, sine = math.cos(self.global_yaw_rad), math.sin(self.global_yaw_rad)
        return (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            _wrapped(yaw_rad - self.global_yaw_rad),
        )


@dataclass(frozen=True)
class PoseEstimate:
    """Estimated ``base_link`` pose in stable vehicle ``anchor_map`` coordinates."""

    anchor_id: str
    x_m: float
    y_m: float
    yaw_rad: float
    covariance: tuple[float, float, float]
    quality: str
    timestamp: float
    revision: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.anchor_id, str)
            or not self.anchor_id
            or self.quality not in LOCALIZATION_QUALITIES
            or len(self.covariance) != 3
            or not _finite(self.x_m, self.y_m, self.yaw_rad, *self.covariance, self.timestamp)
            or min(self.covariance) < 0
            or type(self.revision) is not int
            or self.revision < 0
        ):
            raise ValueError("invalid pose estimate")

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_id": "anchor_map",
            "anchor_id": self.anchor_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "yaw_rad": self.yaw_rad,
            "covariance_diagonal": list(self.covariance),
            "quality": self.quality,
            "timestamp": self.timestamp,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class OdometryConfig:
    translation_noise_stddev_m: float = 0.0
    yaw_noise_stddev_rad: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not _finite(self.translation_noise_stddev_m, self.yaw_noise_stddev_rad)
            or min(self.translation_noise_stddev_m, self.yaw_noise_stddev_rad) < 0
            or type(self.seed) is not int
        ):
            raise ValueError("odometry noise must be finite and non-negative; seed must be an integer")


class AnchoredOdometry:
    """Integrate simulator motion increments without exposing absolute truth pose."""

    def __init__(
        self,
        anchor: AnchorSpec,
        truth_x_m: float,
        truth_y_m: float,
        truth_yaw_rad: float,
        *,
        config: OdometryConfig = OdometryConfig(),
        timestamp: float,
    ) -> None:
        if not _finite(truth_x_m, truth_y_m, truth_yaw_rad, timestamp):
            raise ValueError("initial odometry values must be finite")
        self.anchor = anchor
        self.config = config
        self._rng = random.Random(config.seed)
        self._truth_x_m = truth_x_m
        self._truth_y_m = truth_y_m
        self._truth_yaw_rad = truth_yaw_rad
        self._birth_truth_yaw_rad = truth_yaw_rad
        self._pose = PoseEstimate(
            anchor.anchor_id,
            0.0,
            0.0,
            0.0,
            (anchor.position_stddev_m**2, anchor.position_stddev_m**2, anchor.yaw_stddev_rad**2),
            "nominal",
            timestamp,
            0,
        )

    @property
    def pose(self) -> PoseEstimate:
        return self._pose

    def update(
        self, truth_x_m: float, truth_y_m: float, truth_yaw_rad: float, *, timestamp: float
    ) -> PoseEstimate:
        if not _finite(truth_x_m, truth_y_m, truth_yaw_rad, timestamp):
            raise ValueError("odometry update values must be finite")
        dx, dy = truth_x_m - self._truth_x_m, truth_y_m - self._truth_y_m
        yaw_delta = _wrapped(truth_yaw_rad - self._truth_yaw_rad)
        cosine, sine = math.cos(self._birth_truth_yaw_rad), math.sin(self._birth_truth_yaw_rad)
        local_dx = cosine * dx + sine * dy
        local_dy = -sine * dx + cosine * dy
        moved = local_dx != 0.0 or local_dy != 0.0
        turned = yaw_delta != 0.0
        if moved and self.config.translation_noise_stddev_m:
            local_dx += self._rng.gauss(0.0, self.config.translation_noise_stddev_m)
            local_dy += self._rng.gauss(0.0, self.config.translation_noise_stddev_m)
        if turned and self.config.yaw_noise_stddev_rad:
            yaw_delta += self._rng.gauss(0.0, self.config.yaw_noise_stddev_rad)

        covariance = list(self._pose.covariance)
        if moved:
            covariance[0] += self.config.translation_noise_stddev_m**2
            covariance[1] += self.config.translation_noise_stddev_m**2
        if turned:
            covariance[2] += self.config.yaw_noise_stddev_rad**2
        self._truth_x_m, self._truth_y_m, self._truth_yaw_rad = truth_x_m, truth_y_m, truth_yaw_rad
        self._pose = PoseEstimate(
            self.anchor.anchor_id,
            self._pose.x_m + local_dx,
            self._pose.y_m + local_dy,
            _wrapped(self._pose.yaw_rad + yaw_delta),
            tuple(covariance),
            self._pose.quality,
            timestamp,
            self._pose.revision + 1,
        )
        return self._pose

    def set_quality(self, quality: str, *, timestamp: float) -> PoseEstimate:
        if quality not in LOCALIZATION_QUALITIES or not _finite(timestamp):
            raise ValueError("invalid localization quality or timestamp")
        self._pose = PoseEstimate(
            self._pose.anchor_id,
            self._pose.x_m,
            self._pose.y_m,
            self._pose.yaw_rad,
            self._pose.covariance,
            quality,
            timestamp,
            self._pose.revision + 1,
        )
        return self._pose


@dataclass(frozen=True)
class MapCellUpdate:
    gx: int
    gy: int
    state: int

    def as_dict(self) -> dict[str, int]:
        return {"gx": self.gx, "gy": self.gy, "state": self.state}


@dataclass(frozen=True)
class LocalMapDelta:
    anchor_id: str
    revision: int
    pose_revision: int
    observed_at: float
    changed_cells: tuple[MapCellUpdate, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "revision": self.revision,
            "pose_revision": self.pose_revision,
            "observed_at": self.observed_at,
            "changed_cells": [cell.as_dict() for cell in self.changed_cells],
        }


class ObservedGrid:
    """Sparse, stable ``anchor_map`` occupancy built only from local scans."""

    def __init__(self, anchor: AnchorSpec, *, resolution_m: float = 1.0) -> None:
        if not _finite(resolution_m) or resolution_m <= 0:
            raise ValueError("resolution_m must be finite and positive")
        self.anchor = anchor
        self.resolution_m = resolution_m
        self.revision = 0
        self._cells: dict[tuple[int, int], int] = {}

    def get_cell(self, gx: int, gy: int) -> int:
        if type(gx) is not int or type(gy) is not int:
            raise ValueError("grid coordinates must be integers")
        return self._cells.get((gx, gy), UNKNOWN)

    def integrate_scan(
        self,
        points: Iterable[LaserPoint],
        pose: PoseEstimate,
        observed_at: float,
        config: ScanConfig,
    ) -> LocalMapDelta:
        if pose.anchor_id != self.anchor.anchor_id:
            raise ValueError("scan pose belongs to a different anchor")
        if not _finite(observed_at):
            raise ValueError("observed_at must be finite")

        updates: dict[tuple[int, int], int] = {}
        for point in points:
            if not _finite(point.angle, point.range, point.intensity) or point.range < 0:
                raise ValueError("scan points must be finite and ranges cannot be negative")
            if point.range > config.max_range:
                raise ValueError("scan range exceeds configured maximum")
            hit = point.range > 0
            distance = point.range if hit else config.max_range
            world_angle = pose.yaw_rad + point.angle
            direction_x, direction_y = math.cos(world_angle), math.sin(world_angle)
            if math.isclose(direction_x, 0.0, abs_tol=1e-12):
                direction_x = 0.0
            if math.isclose(direction_y, 0.0, abs_tol=1e-12):
                direction_y = 0.0
            start = self._cell(pose.x_m, pose.y_m)
            end_x = pose.x_m + distance * direction_x
            end_y = pose.y_m + distance * direction_y
            if hit:
                if direction_x:
                    end_x = math.nextafter(end_x, math.copysign(math.inf, direction_x))
                if direction_y:
                    end_y = math.nextafter(end_y, math.copysign(math.inf, direction_y))
            end = self._cell(end_x, end_y)
            ray = tuple(_bresenham(*start, *end))
            for cell in ray[:-1] if hit else ray:
                updates.setdefault(cell, FREE)
            if hit:
                updates[ray[-1]] = OCCUPIED

        original = {cell: self._cells.get(cell, UNKNOWN) for cell in updates}
        self._cells.update(updates)
        changed = tuple(
            MapCellUpdate(gx, gy, self._cells[(gx, gy)])
            for gx, gy in sorted(
                (cell for cell, state in original.items() if self._cells[cell] != state),
                key=lambda cell: (cell[1], cell[0]),
            )
        )
        if changed:
            self.revision += 1
        return LocalMapDelta(
            self.anchor.anchor_id,
            self.revision,
            pose.revision,
            observed_at,
            changed,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor.anchor_id,
            "frame_id": "anchor_map",
            "resolution_m": self.resolution_m,
            "revision": self.revision,
            "cells": [
                MapCellUpdate(gx, gy, state).as_dict()
                for (gx, gy), state in sorted(self._cells.items(), key=lambda item: (item[0][1], item[0][0]))
            ],
        }

    def _cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        return math.floor(x_m / self.resolution_m), math.floor(y_m / self.resolution_m)


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


class AnchoredLocalState:
    """One vehicle's persistent local estimate and observed map."""

    def __init__(
        self,
        anchor: AnchorSpec,
        *,
        truth_x_m: float,
        truth_y_m: float,
        truth_yaw_rad: float,
        odometry_config: OdometryConfig = OdometryConfig(),
        timestamp: float,
        map_resolution_m: float = 1.0,
    ) -> None:
        self.anchor = anchor
        self.odometry = AnchoredOdometry(
            anchor,
            truth_x_m,
            truth_y_m,
            truth_yaw_rad,
            config=odometry_config,
            timestamp=timestamp,
        )
        self.local_map = ObservedGrid(anchor, resolution_m=map_resolution_m)
        self.last_map_delta: LocalMapDelta | None = None

    @property
    def pose(self) -> PoseEstimate:
        return self.odometry.pose

    def update_from_truth(
        self, truth_x_m: float, truth_y_m: float, truth_yaw_rad: float, *, timestamp: float
    ) -> PoseEstimate:
        return self.odometry.update(truth_x_m, truth_y_m, truth_yaw_rad, timestamp=timestamp)

    def set_localization_quality(self, quality: str, *, timestamp: float) -> PoseEstimate:
        return self.odometry.set_quality(quality, timestamp=timestamp)

    def integrate_scan(
        self, points: Iterable[LaserPoint], observed_at: float, config: ScanConfig
    ) -> LocalMapDelta | None:
        if self.pose.quality == "lost":
            return None
        self.last_map_delta = self.local_map.integrate_scan(points, self.pose, observed_at, config)
        return self.last_map_delta
