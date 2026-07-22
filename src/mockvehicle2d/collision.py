"""
mock_collision.py — 碰撞检测函数

提供:
  - raycast():    Bresenham 线段碰撞检测
  - is_circle_passable(): 圆形车辆碰撞检测 (AABB vs Circle)
  - is_swept_circle_passable(): 连续平移检测 (AABB vs swept circle)
  - get_blocking_cells(): 调试用，返回圆形区域内所有阻挡格子
  - CollisionResult: 碰撞结果数据结构

车辆参数: 圆形, radius=0.5, 有航向角 yaw
碰撞检测: AABB vs 圆心最近点距离
"""

import math
from dataclasses import dataclass
from typing import Optional

from mockvehicle2d.map_grid import MapGrid


@dataclass
class CollisionResult:
    """碰撞检测结果"""
    hit: bool
    x: Optional[int] = None
    y: Optional[int] = None


# ── Bresenham 线段碰撞 ──────────────────────────────────

def raycast(grid: MapGrid, x1: int, y1: int, x2: int, y2: int) -> CollisionResult:
    """Bresenham 直线算法 + 逐格碰撞检测。"""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    err = dx - dy

    x, y = x1, y1
    while True:
        if not grid.is_passable(x, y):
            return CollisionResult(hit=True, x=x, y=y)
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return CollisionResult(hit=False)


# ── 圆形碰撞检测 (AABB vs Circle) ──────────────────────

def _cell_overlaps_circle(gx: int, gy: int, cx: float, cy: float, r2: float) -> bool:
    """圆是否与 cell [gx, gx+1] × [gy, gy+1] 重叠。

    取 cell AABB 上离圆心最近的点，判断距离是否 < r。
    """
    closest_x = max(gx, min(cx, gx + 1))
    closest_y = max(gy, min(cy, gy + 1))
    dx = closest_x - cx
    dy = closest_y - cy
    return dx * dx + dy * dy < r2


def is_circle_passable(grid: MapGrid, cx: float, cy: float, radius: float) -> bool:
    """检查圆形车辆区域是否可通行。

    Args:
        grid: 栅格地图
        cx, cy: 圆心坐标（世界坐标，浮点）
        radius: 车辆半径（世界坐标单位）

    Returns:
        True = 全通行, False = 碰撞或越界
    """
    r_int = math.ceil(radius)
    r2 = radius * radius

    x_min = int(cx) - r_int
    x_max = int(cx) + r_int
    y_min = int(cy) - r_int
    y_max = int(cy) + r_int

    for gy in range(y_min, y_max + 1):
        for gx in range(x_min, x_max + 1):
            if not _cell_overlaps_circle(gx, gy, cx, cy, r2):
                continue
            if not grid.is_passable(gx, gy):
                return False

    return True


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


def _segment_aabb_distance_squared(
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
            if _segment_aabb_distance_squared(x1, y1, x2, y2, gx, gy, gx + 1, gy + 1) < radius_squared:
                return False
    return True


def get_blocking_cells(grid: MapGrid, cx: float, cy: float, radius: float) -> list[tuple[int, int]]:
    """返回圆形区域内所有阻挡 cell（调试用）。"""
    r_int = math.ceil(radius)
    r2 = radius * radius

    x_min = int(cx) - r_int
    x_max = int(cx) + r_int
    y_min = int(cy) - r_int
    y_max = int(cy) + r_int

    walls = []
    for gy in range(y_min, y_max + 1):
        for gx in range(x_min, x_max + 1):
            if not _cell_overlaps_circle(gx, gy, cx, cy, r2):
                continue
            if not grid.is_passable(gx, gy):
                walls.append((gx, gy))

    return walls
