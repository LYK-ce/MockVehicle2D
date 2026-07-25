"""Incremental D* Lite checks on a sparse, finite local occupancy map."""

import heapq
import math
from pathlib import Path
import random
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.local_state import AnchorSpec, FREE, MapCellUpdate, OCCUPIED, ObservedGrid
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner


ANCHOR = AnchorSpec("planner-test", 0.0, 0.0, 0.0)


def planner(**kwargs) -> DStarLitePlanner:
    return DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.0,
        bounds_margin_m=kwargs.pop("bounds_margin_m", 3.0),
        **kwargs,
    )


def path_cost(path: list[tuple[int, int]] | None, unknown_cost: float = 3.0) -> float:
    if path is None:
        return math.inf
    return sum(
        (math.sqrt(2) if a[0] != b[0] and a[1] != b[1] else 1.0) * unknown_cost
        for a, b in zip(path, path[1:])
    )


def geometric_cost(path: list[tuple[int, int]] | None) -> float:
    return path_cost(path, unknown_cost=1.0)


def test_unknown_space_is_explorable_and_deterministic() -> None:
    first = planner()
    path = first.plan((0, 0), (5, 3))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (5, 3)
    assert path == planner().plan((0, 0), (5, 3))
    assert first.stats["resets"] == 1


def test_obstacle_insert_and_remove_reuses_search_state() -> None:
    search = planner(bounds_margin_m=2.0)
    original = search.plan((0, 0), (6, 0))
    resets = search.stats["resets"]

    detour = search.plan(
        (0, 0),
        (6, 0),
        changed_cells=(MapCellUpdate(3, 0, OCCUPIED),),
    )
    after_insert = dict(search.stats)
    restored = search.plan(
        (0, 0),
        (6, 0),
        changed_cells=(MapCellUpdate(3, 0, FREE),),
    )
    search.plan((1, 0), (6, 0))

    assert original is not None and detour is not None and restored is not None
    assert (3, 0) not in detour
    assert geometric_cost(detour) > geometric_cost(original)
    assert geometric_cost(restored) < geometric_cost(detour)
    assert after_insert["incremental_updates"] > 0
    assert search.stats["resets"] == resets
    assert search.stats["key_modifier_cost"] > 0


def test_diagonal_corner_cutting_is_forbidden() -> None:
    search = planner(bounds_margin_m=0.0)
    assert (
        search.plan(
            (0, 0),
            (1, 1),
            changed_cells=(
                MapCellUpdate(1, 0, OCCUPIED),
                MapCellUpdate(0, 1, OCCUPIED),
            ),
        )
        is None
    )


def test_vehicle_radius_inflates_obstacles() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=1.0,
        bounds_margin_m=2.0,
    )
    path = search.plan(
        (0, 0),
        (6, 0),
        changed_cells=(MapCellUpdate(3, 1, OCCUPIED),),
    )
    assert path is not None
    assert all(max(abs(x - 3), abs(y - 1)) > 1 for x, y in path)


def test_no_route_returns_none() -> None:
    search = planner(bounds_margin_m=0.0)
    wall = tuple(MapCellUpdate(2, y, OCCUPIED) for y in range(0, 4))
    assert search.plan((0, 0), (4, 3), changed_cells=wall) is None


@pytest.mark.parametrize("blocked", [(0, 0), (4, 0)])
def test_occupied_start_or_goal_returns_none(blocked: tuple[int, int]) -> None:
    search = planner()
    assert search.plan(
        (0, 0),
        (4, 0),
        changed_cells=(MapCellUpdate(*blocked, OCCUPIED),),
    ) is None


def test_blocked_start_equals_goal_returns_none_but_free_returns_singleton() -> None:
    search = planner()
    assert search.plan((2, 2), (2, 2)) == [(2, 2)]
    assert search.plan(
        (2, 2),
        (2, 2),
        changed_cells=(MapCellUpdate(2, 2, OCCUPIED),),
    ) is None


def test_inflated_start_or_goal_and_overlapping_removal_stay_blocked() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=1.0,
        bounds_margin_m=3.0,
    )
    assert search.plan(
        (0, 0),
        (4, 0),
        changed_cells=(
            MapCellUpdate(4, -1, OCCUPIED),
            MapCellUpdate(4, 1, OCCUPIED),
        ),
    ) is None
    assert search.plan(
        (0, 0),
        (4, 0),
        changed_cells=(MapCellUpdate(4, -1, FREE),),
    ) is None
    assert search.plan(
        (0, 0),
        (4, 0),
        changed_cells=(MapCellUpdate(4, 1, FREE),),
    ) is not None

    assert search.plan(
        (0, 0),
        (4, 0),
        changed_cells=(MapCellUpdate(0, 1, OCCUPIED),),
    ) is None


def test_goal_and_cell_budget_are_validated() -> None:
    search = planner(max_goal_distance_m=10.0, max_cells=100)
    with pytest.raises(ValueError, match="goal"):
        search.plan((0, 0), (11, 0))
    search = planner(max_goal_distance_m=20.0, max_cells=100)
    with pytest.raises(ValueError, match="cell"):
        search.plan((0, 0), (9, 9))


def _reference_cost(
    start: tuple[int, int],
    goal: tuple[int, int],
    occupied: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> float:
    min_x, min_y, max_x, max_y = bounds
    distances = {start: 0.0}
    queue = [(0.0, start)]
    moves = tuple(
        (dx, dy, math.sqrt(2) if dx and dy else 1.0)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    )
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distances[current]:
            continue
        if current == goal:
            return cost
        for dx, dy, step in moves:
            nxt = current[0] + dx, current[1] + dy
            if not (min_x <= nxt[0] <= max_x and min_y <= nxt[1] <= max_y):
                continue
            if nxt in occupied:
                continue
            if dx and dy and (
                (current[0] + dx, current[1]) in occupied
                or (current[0], current[1] + dy) in occupied
            ):
                continue
            candidate = cost + step * 3.0
            if candidate < distances.get(nxt, math.inf):
                distances[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    return math.inf


def test_random_finite_graph_cost_matches_reference() -> None:
    rng = random.Random(123)
    for _ in range(30):
        occupied = {
            (x, y)
            for x in range(6)
            for y in range(6)
            if (x, y) not in {(0, 0), (5, 5)} and rng.random() < 0.22
        }
        search = planner(bounds_margin_m=0.0)
        path = search.plan(
            (0, 0),
            (5, 5),
            changed_cells=tuple(MapCellUpdate(x, y, OCCUPIED) for x, y in sorted(occupied)),
        )
        assert path_cost(path) == pytest.approx(
            _reference_cost((0, 0), (5, 5), occupied, (0, 0, 5, 5))
        )


def test_local_change_expands_less_state_than_a_fresh_long_route() -> None:
    incremental = planner(bounds_margin_m=6.0)
    assert incremental.plan((0, 0), (80, 0)) is not None
    before = incremental.stats["expansions"]
    assert incremental.plan(
        (1, 0),
        (80, 0),
        changed_cells=(MapCellUpdate(40, 0, OCCUPIED),),
    ) is not None
    incremental_expansions = incremental.stats["expansions"] - before

    fresh = planner(bounds_margin_m=6.0)
    assert fresh.plan(
        (1, 0),
        (80, 0),
        changed_cells=(MapCellUpdate(40, 0, OCCUPIED),),
    ) is not None

    assert incremental.stats["resets"] == 1
    assert incremental_expansions < fresh.stats["expansions"]
