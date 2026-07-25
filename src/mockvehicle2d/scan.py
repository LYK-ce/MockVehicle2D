"""Deterministic 2D grid scans using the YDLidar Tmini measurement profile."""

from dataclasses import dataclass
import math
from typing import Iterable

from mockvehicle2d.map_grid import MapGrid


@dataclass(frozen=True)
class LaserPoint:
    """One YDLidar-style return: radians, metres, and unitless intensity."""

    angle: float
    range: float
    intensity: float

    def as_dict(self) -> dict[str, float]:
        return {"angle": self.angle, "range": self.range, "intensity": self.intensity}


@dataclass(frozen=True)
class ScanConfig:
    """LaserScan metadata for one deterministic planar sweep."""

    min_angle: float = 0.0
    max_angle: float = 2 * math.pi * 666 / 667
    angle_increment: float = 2 * math.pi / 667
    scan_time: float = 1 / 6
    min_range: float = 0.02
    max_range: float = 12.0
    model: str = "ydlidar_tmini"
    range_sample_rate_hz: int = 4000
    scan_rate_hz: int = 6

    def __post_init__(self) -> None:
        if self.max_angle < self.min_angle:
            raise ValueError("max_angle must be greater than or equal to min_angle")
        if self.angle_increment <= 0 or self.scan_time <= 0:
            raise ValueError("angle_increment and scan_time must be positive")
        if self.range_sample_rate_hz <= 0 or self.scan_rate_hz <= 0:
            raise ValueError("scan rates must be positive")
        if self.min_range < 0 or self.max_range <= self.min_range:
            raise ValueError("range limits must satisfy 0 <= min_range < max_range")

    def sample_count(self) -> int:
        return int(math.floor((self.max_angle - self.min_angle) / self.angle_increment + 1e-9)) + 1

    def as_dict(self) -> dict[str, float | int | str | dict[str, float]]:
        count = self.sample_count()
        return {
            "min_angle": self.min_angle,
            "max_angle": self.min_angle + (count - 1) * self.angle_increment,
            "angle_increment": self.angle_increment,
            "time_increment": self.scan_time / count,
            "scan_time": self.scan_time,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "point_count": count,
            "model": self.model,
            "range_sample_rate_hz": self.range_sample_rate_hz,
            "scan_rate_hz": self.scan_rate_hz,
            "angle_unit": "rad",
            "range_unit": "m",
            "angle_direction": "clockwise_from_forward",
            "no_return": {"range": 0.0, "intensity": 0.0},
        }


TMINI_SCAN_CONFIG = ScanConfig()
DEFAULT_SCAN_CONFIG = TMINI_SCAN_CONFIG


def _first_wall_range(
    grid: MapGrid, x: float, y: float, world_angle: float, config: ScanConfig
) -> float | None:
    """Return the first wall-boundary distance, or None for no return."""

    cell_x, cell_y = math.floor(x), math.floor(y)
    if not grid.in_bounds(cell_x, cell_y):
        return None
    if grid.is_wall(cell_x, cell_y):
        return config.min_range

    direction_x, direction_y = math.cos(world_angle), math.sin(world_angle)
    if math.isclose(direction_x, 0.0, abs_tol=1e-12):
        direction_x = 0.0
    if math.isclose(direction_y, 0.0, abs_tol=1e-12):
        direction_y = 0.0
    step_x = 1 if direction_x > 0 else -1
    step_y = 1 if direction_y > 0 else -1
    delta_x = abs(1 / direction_x) if direction_x else math.inf
    delta_y = abs(1 / direction_y) if direction_y else math.inf
    next_x = ((cell_x + 1 - x) if direction_x > 0 else (x - cell_x)) * delta_x if direction_x else math.inf
    next_y = ((cell_y + 1 - y) if direction_y > 0 else (y - cell_y)) * delta_y if direction_y else math.inf

    while True:
        if math.isclose(next_x, next_y, abs_tol=1e-12):
            distance = next_x
            candidates = ((cell_x + step_x, cell_y), (cell_x, cell_y + step_y), (cell_x + step_x, cell_y + step_y))
            cell_x += step_x
            cell_y += step_y
            next_x += delta_x
            next_y += delta_y
        elif next_x < next_y:
            distance = next_x
            cell_x += step_x
            next_x += delta_x
            candidates = ((cell_x, cell_y),)
        else:
            distance = next_y
            cell_y += step_y
            next_y += delta_y
            candidates = ((cell_x, cell_y),)

        if distance > config.max_range:
            return None
        if any(not grid.in_bounds(cx, cy) for cx, cy in candidates):
            return None
        if any(grid.is_wall(cx, cy) for cx, cy in candidates):
            return max(config.min_range, distance)


def scan_grid(
    grid: MapGrid, x: float, y: float, yaw: float, config: ScanConfig = DEFAULT_SCAN_CONFIG
) -> list[LaserPoint]:
    """Cast a full local scan from pose ``(x, y, yaw)`` through ``grid``.

    The simulator uses screen-style grid coordinates: yaw=0 and angle=0 face +x;
    positive angles turn toward +y, i.e. clockwise when viewed from above.
    """

    points = []
    for index in range(config.sample_count()):
        angle = config.min_angle + index * config.angle_increment
        distance = _first_wall_range(grid, x, y, yaw + angle, config)
        points.append(
            LaserPoint(
                angle,
                round(distance, 2) if distance is not None else 0.0,
                1.0 if distance is not None else 0.0,
            )
        )
    return points


def scan_message(
    grid: MapGrid,
    x: float,
    y: float,
    yaw: float,
    timestamp: float,
    config: ScanConfig = DEFAULT_SCAN_CONFIG,
    points: Iterable[LaserPoint] | None = None,
) -> dict[str, object]:
    """Build the JSON-compatible local scan frame used by the WebSocket server."""

    scan_points = scan_grid(grid, x, y, yaw, config) if points is None else tuple(points)
    return {
        "type": "scan",
        "timestamp_s": timestamp,
        "ts": timestamp,
        "frame_id": "laser",
        "config": config.as_dict(),
        "points": [point.as_dict() for point in scan_points],
    }
