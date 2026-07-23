"""
a_star.py — A* pathfinding on a MapGrid.

Eight-connected grid search with Euclidean heuristic, corner-cutting prevention,
and 1-cell wall inflation for a circular vehicle (r ≈ 0.5).
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

from mockvehicle2d.map_grid import MapGrid

# ── move directions (dx, dy, cost) ──────────────────────────

_CARDINALS: list[tuple[int, int, float]] = [
    (1, 0, 1.0),
    (0, 1, 1.0),
    (-1, 0, 1.0),
    (0, -1, 1.0),
]

_DIAGONALS: list[tuple[int, int, float]] = [
    (1, 1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
]

_MOVES: list[tuple[int, int, float]] = _CARDINALS + _DIAGONALS

# For each diagonal (dx, dy), the two cardinal neighbours that must be free.
_DIAGONAL_GATE: dict[tuple[int, int], list[tuple[int, int]]] = {
    (1, 1):   [(1, 0), (0, 1)],
    (-1, 1):  [(-1, 0), (0, 1)],
    (-1, -1): [(-1, 0), (0, -1)],
    (1, -1):  [(1, 0), (0, -1)],
}


def _inflate_blocked(grid: MapGrid) -> set[tuple[int, int]]:
    """Return the set of cells that are walls *or* adjacent to a wall.

    With vehicle radius 0.5, a vehicle whose centre is in a cell adjacent to a
    wall will overlap the wall cell, so those neighbours are also impassable.
    """
    blocked: set[tuple[int, int]] = set()
    for x in range(grid.width):
        for y in range(grid.height):
            if grid.is_wall(x, y):
                blocked.add((x, y))
                for dx, dy, _ in _MOVES:
                    nx, ny = x + dx, y + dy
                    if grid.in_bounds(nx, ny):
                        blocked.add((nx, ny))
    return blocked


def a_star_search(
    grid: MapGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    vehicle_radius: float = 0.5,
) -> Optional[list[tuple[int, int]]]:
    """Find an eight-connected shortest path from *start* to *goal*.

    Returns a list of grid coordinates (including start and goal) or ``None``
    when no path exists.  Walls are inflated by one cell when
    *vehicle_radius* > 0 so that the returned path keeps the vehicle centre at
    least 0.5 away from any wall.
    """
    if not grid.in_bounds(*start):
        raise ValueError(f"start {start} out of bounds")
    if not grid.in_bounds(*goal):
        raise ValueError(f"goal {goal} out of bounds")

    inflate = vehicle_radius > 0
    blocked = _inflate_blocked(grid) if inflate else None

    def _passable(cell: tuple[int, int]) -> bool:
        if inflate:
            return cell not in blocked
        return grid.is_passable(*cell)

    if not _passable(start):
        return None
    if not _passable(goal):
        return None

    # A* state
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_set: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_set, (_heuristic(start, goal), start))

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return _reconstruct_path(came_from, current)

        cx, cy = current
        for dx, dy, cost in _MOVES:
            neighbour = (cx + dx, cy + dy)
            if not grid.in_bounds(*neighbour):
                continue
            if not _passable(neighbour):
                continue

            # Corner-cutting prevention: diagonal move requires both cardinal
            # intermediate cells to be passable as well.
            if (dx, dy) in _DIAGONAL_GATE:
                if not all(_passable((cx + gx, cy + gy)) for gx, gy in _DIAGONAL_GATE[(dx, dy)]):
                    continue

            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                heapq.heappush(open_set, (tentative + _heuristic(neighbour, goal), neighbour))

    return None


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance (admissible for eight-connected grid)."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
