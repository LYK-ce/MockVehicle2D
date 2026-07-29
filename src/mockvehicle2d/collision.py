"""Continuous circular-vehicle collision geometry."""

import math

from mockvehicle2d.map_grid import MapGrid


def is_strict_overlap(distance_squared: float, radius_squared: float) -> bool:
    """Treat floating-point representations of exact tangency as non-overlap."""
    return distance_squared < radius_squared and not math.isclose(
        distance_squared,
        radius_squared,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def cell_overlaps_circle(gx: int, gy: int, cx: float, cy: float, r2: float) -> bool:
    """圆是否与 cell [gx, gx+1] × [gy, gy+1] 重叠。

    取 cell AABB 上离圆心最近的点，判断距离是否 < r。
    """
    closest_x = max(gx, min(cx, gx + 1))
    closest_y = max(gy, min(cy, gy + 1))
    dx = closest_x - cx
    dy = closest_y - cy
    return is_strict_overlap(dx * dx + dy * dy, r2)


def _point_segment_distance_squared(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return (px - x1) ** 2 + (py - y1) ** 2
    ratio = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_squared))
    nearest_x, nearest_y = x1 + ratio * dx, y1 + ratio * dy
    return (px - nearest_x) ** 2 + (py - nearest_y) ** 2


def _point_aabb_distance_squared(
    px: float, py: float, min_x: float, min_y: float, max_x: float, max_y: float
) -> float:
    dx = max(min_x - px, 0.0, px - max_x)
    dy = max(min_y - py, 0.0, py - max_y)
    return dx * dx + dy * dy


def _segment_intersects_aabb(
    x1: float, y1: float, x2: float, y2: float, min_x: float, min_y: float, max_x: float, max_y: float
) -> bool:
    low, high = 0.0, 1.0
    for start, delta, lower, upper in (
        (x1, x2 - x1, min_x, max_x),
        (y1, y2 - y1, min_y, max_y),
    ):
        if delta == 0:
            if start < lower or start > upper:
                return False
            continue
        enter, leave = (lower - start) / delta, (upper - start) / delta
        if enter > leave:
            enter, leave = leave, enter
        low, high = max(low, enter), min(high, leave)
        if low > high:
            return False
    return True


def segment_aabb_distance_squared(
    x1: float, y1: float, x2: float, y2: float, min_x: float, min_y: float, max_x: float, max_y: float
) -> float:
    if _segment_intersects_aabb(x1, y1, x2, y2, min_x, min_y, max_x, max_y):
        return 0.0
    return min(
        _point_aabb_distance_squared(x1, y1, min_x, min_y, max_x, max_y),
        _point_aabb_distance_squared(x2, y2, min_x, min_y, max_x, max_y),
        *(
            _point_segment_distance_squared(px, py, x1, y1, x2, y2)
            for px, py in ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))
        ),
    )


def is_swept_circle_passable(
    grid: MapGrid, x1: float, y1: float, x2: float, y2: float, radius: float
) -> bool:
    """Check the whole circular-vehicle translation, treating only strict overlap as collision."""
    radius_squared = radius * radius
    for gy in range(math.floor(min(y1, y2) - radius), math.floor(max(y1, y2) + radius) + 1):
        for gx in range(math.floor(min(x1, x2) - radius), math.floor(max(x1, x2) + radius) + 1):
            if grid.is_passable(gx, gy):
                continue
            distance_squared = segment_aabb_distance_squared(
                x1, y1, x2, y2, gx, gy, gx + 1, gy + 1
            )
            if is_strict_overlap(distance_squared, radius_squared):
                return False
    return True
