"""
mock_collision.py — 碰撞检测函数

提供:
  - raycast():    Bresenham 线段碰撞检测
  - is_circle_passable(): 圆形车辆碰撞检测 (AABB vs Circle)
  - get_blocking_cells(): 调试用，返回圆形区域内所有阻挡格子
  - CollisionResult: 碰撞结果数据结构

车辆参数: 圆形, radius=0.5, 有航向角 yaw
碰撞检测: AABB vs 圆心最近点距离
"""

import math
from dataclasses import dataclass
from typing import Optional

from mock_map_grid import MapGrid


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
        if grid.is_wall(x, y):
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
            if not grid.in_bounds(gx, gy):
                return False
            if grid.is_wall(gx, gy):
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
            if not grid.in_bounds(gx, gy) or grid.is_wall(gx, gy):
                walls.append((gx, gy))

    return walls
