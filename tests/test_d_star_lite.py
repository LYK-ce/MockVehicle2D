"""Incremental D* Lite checks on a sparse, finite local occupancy map."""

import heapq
import math
from pathlib import Path
import random
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.collision import is_swept_circle_passable
from mockvehicle2d.local_state import (
    AnchorSpec,
    FORBIDDEN,
    FREE,
    MapCellUpdate,
    OCCUPIED,
    ObservedGrid,
    UNKNOWN,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner, _key_less
from mockvehicle2d.safety import AUTOMATIC_MINIMUM_CLEARANCE_M


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


def test_bounded_planning_preserves_state_until_the_path_is_ready() -> None:
    search = planner(bounds_margin_m=6.0)
    statuses = []

    while True:
        before = search.stats["expansions"]
        progress = search.advance_plan(
            (0, 0),
            (80, 0),
            expansion_budget=7,
        )
        statuses.append(progress.status)
        assert search.stats["expansions"] - before <= 7
        if progress.status != "pending":
            break

    assert "pending" in statuses
    assert progress.status == "ready"
    assert progress.path == planner(bounds_margin_m=6.0).plan((0, 0), (80, 0))
    assert search.stats["resets"] == 1


def test_bounded_planning_absorbs_pose_and_map_changes_while_pending() -> None:
    search = planner(bounds_margin_m=6.0)
    assert search.advance_plan(
        (0, 0),
        (80, 0),
        expansion_budget=1,
    ).status == "pending"
    obstacle = MapCellUpdate(40, 0, OCCUPIED)

    progress = search.advance_plan(
        (1, 0),
        (80, 0),
        changed_cells=(obstacle,),
        expansion_budget=7,
    )
    while progress.status == "pending":
        progress = search.advance_plan(
            (1, 0),
            (80, 0),
            expansion_budget=7,
        )

    expected = planner(bounds_margin_m=6.0).plan(
        (1, 0),
        (80, 0),
        changed_cells=(obstacle,),
    )
    assert progress.status == "ready"
    assert progress.path == expected
    assert search.stats["resets"] == 1
    assert search.stats["key_modifier_cost"] == pytest.approx(1.0)


def test_bounded_planning_reports_unreachable_after_pending() -> None:
    search = planner(bounds_margin_m=0.0)
    wall = tuple(MapCellUpdate(2, y, OCCUPIED) for y in range(4))
    progress = search.advance_plan(
        (0, 0),
        (4, 3),
        changed_cells=wall,
        expansion_budget=1,
    )
    assert progress.status == "pending"

    while progress.status == "pending":
        progress = search.advance_plan(
            (0, 0),
            (4, 3),
            expansion_budget=1,
        )

    assert progress.status == "unreachable"
    assert progress.path is None
    assert search.last_failure == "search_exhausted"


def test_bounded_planning_stops_at_the_cross_frame_expansion_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = planner(bounds_margin_m=0.0, max_cells=3)

    def never_finishes(expansion_budget: int) -> bool:
        search._expansions += expansion_budget
        return False

    monkeypatch.setattr(search, "_advance_shortest_path", never_finishes)
    for _ in range(20):
        before = search.stats["expansions"]
        progress = search.advance_plan(
            (0, 0),
            (2, 0),
            expansion_budget=4,
        )
        assert search.stats["expansions"] - before <= 4
        if progress.status != "pending":
            break

    assert progress.status == "unreachable"
    assert progress.path is None
    assert search.last_failure == "expansion_limit"
    assert search.stats["expansions"] == 3 * 20


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


def test_observed_change_is_absorbed_without_changing_route_endpoints() -> None:
    search = planner(bounds_margin_m=2.0)
    original = search.plan((0, 0), (6, 0))
    resets = search.stats["resets"]

    changes = search.observe_changes((MapCellUpdate(3, 0, OCCUPIED),))
    detour = search.plan((0, 0), (6, 0))

    assert changes == (MapCellUpdate(3, 0, OCCUPIED),)
    assert original is not None and detour is not None
    assert (3, 0) not in detour
    assert geometric_cost(detour) > geometric_cost(original)
    assert search.stats["resets"] == resets


def test_peer_forbidden_cells_are_not_inflated_twice_and_move_cleanly() -> None:
    grid = ObservedGrid(ANCHOR, resolution_m=0.5)
    peer = DStarLitePlanner(
        grid,
        vehicle_radius_m=0.5,
        hard_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        bounds_margin_m=2.0,
    )
    peer.set_peer_forbidden_cells(((2, 0),))
    peer.plan((-2, 0), (6, 0))

    assert peer._blocked((2, 0))
    assert not peer._blocked((0, 0))

    peer.set_peer_forbidden_cells(((4, 0),))

    assert not peer._blocked((2, 0))
    assert peer._blocked((4, 0))

    static = DStarLitePlanner(
        grid,
        vehicle_radius_m=0.5,
        hard_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        bounds_margin_m=2.0,
    )
    static.plan(
        (-2, 0),
        (6, 0),
        changed_cells=(MapCellUpdate(2, 0, OCCUPIED),),
    )
    assert static._blocked((0, 0))


def test_peer_forbidden_changes_repair_an_existing_search() -> None:
    search = planner(bounds_margin_m=2.0)
    original = search.plan((0, 0), (6, 0))
    resets = search.stats["resets"]

    search.set_peer_forbidden_cells(((1, 0),))
    detour = search.plan((0, 0), (6, 0))
    search.set_peer_forbidden_cells(())
    restored = search.plan((0, 0), (6, 0))

    assert original is not None and detour is not None
    assert (1, 0) not in detour
    assert geometric_cost(detour) > geometric_cost(original)
    assert restored == original
    assert search.stats["resets"] == resets


def test_peer_overlay_never_hides_base_forbidden_inflation() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR, resolution_m=0.5),
        vehicle_radius_m=0.5,
        hard_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        bounds_margin_m=2.0,
    )
    search.plan(
        (-2, 0),
        (6, 0),
        changed_cells=(MapCellUpdate(0, 0, FORBIDDEN),),
    )

    search.set_peer_forbidden_cells(((0, 0),))
    assert search._blocked((2, 0))

    search.set_peer_forbidden_cells(((4, 0),))
    assert search._blocked((2, 0))


def test_peer_overlay_does_not_hide_static_obstacle_inflation() -> None:
    class PeerOverStaticGrid(ObservedGrid):
        def snapshot(self):
            return {
                "cells": [{"gx": 0, "gy": 0, "state": FORBIDDEN}],
                "peer_forbidden_cells": [{"gx": 0, "gy": 0}],
            }

        def cell_without_peers(self, gx, gy):
            return OCCUPIED if (gx, gy) == (0, 0) else UNKNOWN

    search = DStarLitePlanner(
        PeerOverStaticGrid(ANCHOR, resolution_m=0.5),
        vehicle_radius_m=0.5,
        hard_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        bounds_margin_m=2.0,
    )
    search.plan((-2, 0), (6, 0))

    assert search._blocked((2, 0))
    search.set_peer_forbidden_cells(())
    search.observe_changes((MapCellUpdate(0, 0, OCCUPIED),))
    assert search._blocked((2, 0))


def test_peer_circle_rechecks_continuous_access_segments() -> None:
    class PeerGrid(ObservedGrid):
        def peer_exclusion_circles(self):
            return ((1.0, 0.0, 0.5),)

    search = DStarLitePlanner(
        PeerGrid(ANCHOR, resolution_m=0.5),
        vehicle_radius_m=0.0,
    )

    assert not search.is_segment_passable((0.0, 0.0), (2.0, 0.0))
    assert search.is_segment_passable((0.0, 0.6), (2.0, 0.6))


@pytest.mark.parametrize(
    ("base_state", "expected"),
    ((FREE, True), (UNKNOWN, False)),
)
def test_peer_overlay_preserves_base_observation_semantics(
    base_state: int,
    expected: bool,
) -> None:
    class PeerGrid(ObservedGrid):
        def snapshot(self):
            return {
                "cells": [{"gx": 0, "gy": 0, "state": FORBIDDEN}],
                "peer_forbidden_cells": [{"gx": 0, "gy": 0}],
            }

        def cell_without_peers(self, gx, gy):
            return base_state if (gx, gy) == (0, 0) else UNKNOWN

        def peer_exclusion_circles(self):
            return ()

    search = DStarLitePlanner(
        PeerGrid(ANCHOR, resolution_m=0.5),
        vehicle_radius_m=0.1,
    )

    assert (
        search.is_segment_passable(
            (0.25, 0.25),
            (0.25, 0.25),
            require_observed=True,
        )
        is expected
    )


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


def test_vehicle_radius_uses_circle_geometry_at_touching_cell_boundary() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.5,
        bounds_margin_m=2.0,
    )

    path = search.plan(
        (-2, 0),
        (0, 0),
        changed_cells=(MapCellUpdate(1, 0, OCCUPIED),),
    )

    assert path is not None and path[-1] == (0, 0)


def test_exact_off_centre_segment_can_recentre_before_safe_grid_edge() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.5,
        bounds_margin_m=2.0,
    )
    search.plan(
        (0, 0),
        (2, 2),
        changed_cells=(MapCellUpdate(1, -1, OCCUPIED),),
    )

    assert search.is_segment_passable((0.5, 0.5), (1.5, 1.5))
    assert not search.is_segment_passable(
        (0.483010918, 0.054379872),
        (1.5, 1.5),
    )
    assert search.is_segment_passable(
        (0.483010918, 0.054379872),
        (0.5, 0.5),
    )


def test_floating_tangency_matches_runtime_but_real_penetration_blocks() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.5,
        bounds_margin_m=2.0,
    )
    search.plan(
        (0, 0),
        (0, 2),
        changed_cells=(MapCellUpdate(1, 0, OCCUPIED),),
    )
    truth = MapGrid.from_wall_set(4, 4, {(1, 0)})
    tangent = (0.5 + 7e-15, 0.5, 0.5, 1.5)
    safety_tangent = (0.25 + 7e-15, 0.5, 0.25 + 7e-15, 1.5)
    penetration = (0.51, 0.5, 0.51, 1.5)

    assert search.is_segment_passable(tangent[:2], tangent[2:])
    assert is_swept_circle_passable(truth, *tangent, 0.5)
    assert search.is_segment_passable(
        safety_tangent[:2], safety_tangent[2:]
    )
    assert not search.is_segment_passable(
        safety_tangent[:2],
        safety_tangent[2:],
        extra_clearance_m=0.25,
    )
    assert not search.is_segment_passable(penetration[:2], penetration[2:])
    assert not is_swept_circle_passable(truth, *penetration, 0.5)


def test_automatic_planning_clearance_matches_safety_stop_boundary() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.5,
        hard_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        bounds_margin_m=2.0,
    )
    search.plan(
        (0, 0),
        (0, 2),
        changed_cells=(MapCellUpdate(1, 0, OCCUPIED),),
    )
    clearance_025 = (0.25, 0.5, 0.25, 1.5)
    clearance_030 = (0.20, 0.5, 0.20, 1.5)
    clearance_031 = (0.19, 0.5, 0.19, 1.5)

    assert not search.is_segment_passable(
        clearance_025[:2],
        clearance_025[2:],
        extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
    )
    assert not search.is_segment_passable(
        clearance_030[:2],
        clearance_030[2:],
        extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
    )
    assert search.is_segment_passable(
        clearance_031[:2],
        clearance_031[2:],
        extra_clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
    )


def test_hard_clearance_keeps_two_metre_corridor_traversable() -> None:
    observed = ObservedGrid(ANCHOR, resolution_m=0.5)
    search = DStarLitePlanner(
        observed,
        vehicle_radius_m=0.5,
        hard_clearance_m=0.25,
        bounds_margin_m=2.0,
    )
    corridor_walls = tuple(
        MapCellUpdate(x, y, OCCUPIED)
        for x in range(-2, 15)
        for y in (7, 12)
    )

    path = search.plan(
        (0, 9),
        (12, 9),
        changed_cells=corridor_walls,
    )

    assert path is not None
    assert path[0] == (0, 9)
    assert path[-1] == (12, 9)


def test_actual_pose_can_egress_from_planning_clearance_envelope() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR, resolution_m=0.5),
        vehicle_radius_m=0.5,
        hard_clearance_m=0.25,
        bounds_margin_m=2.0,
    )

    path = search.plan(
        (0, 0),
        (-3, 0),
        changed_cells=(MapCellUpdate(1, 0, OCCUPIED),),
        start_position_m=(0.0, 0.25),
    )

    assert path is not None
    assert path[:2] == [(0, 0), (-1, 0)]
    assert search.best_start_connection((0.0, 0.25), (0, 0)) == (-1, 0)
    assert not search.is_segment_passable(
        (0.0, 0.25),
        (0.0, 0.75),
        extra_clearance_m=0.25,
    )
    assert not search._segment_blocked(
        (0.2, 0.5),
        (-0.5, 0.5),
        allow_clearance_egress=True,
    )
    assert search._segment_blocked(
        (0.2, 0.5),
        (1.5, 0.5),
        allow_clearance_egress=True,
    )


def test_blocked_start_without_confirmed_actual_pose_cannot_egress() -> None:
    search = DStarLitePlanner(
        ObservedGrid(ANCHOR, resolution_m=0.5),
        vehicle_radius_m=0.5,
        hard_clearance_m=0.25,
        bounds_margin_m=2.0,
    )

    assert search.plan(
        (0, 0),
        (-3, 0),
        changed_cells=(MapCellUpdate(1, 0, OCCUPIED),),
    ) is None
    assert search.last_failure == "start_blocked"


def test_confirmed_free_clearance_rejects_unknown_and_occupied_envelopes() -> None:
    observed = ObservedGrid(AnchorSpec("dstar-anchor", 0.0, 0.0, 0.0))
    search = DStarLitePlanner(observed, vehicle_radius_m=0.5)
    point = (0.5, 0.5)
    search.plan(
        (0, 0),
        (0, 0),
        changed_cells=(MapCellUpdate(0, 0, FREE),),
    )

    assert search.is_segment_passable(
        point,
        point,
        extra_clearance_m=0.25,
    )
    assert not search.is_segment_passable(
        point,
        point,
        extra_clearance_m=0.25,
        require_observed=True,
    )

    search.plan(
        (0, 0),
        (0, 0),
        changed_cells=tuple(
            MapCellUpdate(gx, gy, FREE)
            for gx in range(-1, 2)
            for gy in range(-1, 2)
            if (gx, gy) != (0, 0)
        ),
    )
    assert search.is_segment_passable(
        point,
        point,
        extra_clearance_m=0.25,
        require_observed=True,
    )

    search.plan(
        (0, 0),
        (0, 0),
        changed_cells=(MapCellUpdate(1, 0, OCCUPIED),),
    )
    assert not search.is_segment_passable(
        point,
        point,
        extra_clearance_m=0.25,
        require_observed=True,
    )


@pytest.mark.parametrize("offset", ((-1, 0), (1, 0), (0, -1), (0, 1)))
@pytest.mark.parametrize("state", (UNKNOWN, OCCUPIED, FORBIDDEN))
def test_confirmed_clearance_tangent_is_symmetric(
    offset: tuple[int, int],
    state: int,
) -> None:
    point = (0.5, 0.5)
    updates = [
        MapCellUpdate(gx, gy, FREE)
        for gx in range(-1, 2)
        for gy in range(-1, 2)
        if (gx, gy) != offset
    ]
    if state != UNKNOWN:
        updates.append(MapCellUpdate(*offset, state))
    search = DStarLitePlanner(ObservedGrid(ANCHOR), vehicle_radius_m=0.25)
    search.plan((0, 0), (0, 0), changed_cells=updates)

    assert not search.is_segment_passable(
        point,
        point,
        extra_clearance_m=0.25,
        require_observed=True,
    )

    strict_tangent = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.5,
    )
    strict_tangent.plan((0, 0), (0, 0), changed_cells=updates)
    assert strict_tangent.is_segment_passable(point, point)


def test_planning_budget_query_covers_distance_and_cell_limits() -> None:
    distance_limited = DStarLitePlanner(ObservedGrid(ANCHOR))
    cells_limited = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        bounds_margin_m=3.0,
        max_cells=100,
    )

    assert distance_limited.planning_budget_allows((0, 0), (256, 0))
    assert not distance_limited.planning_budget_allows((0, 0), (257, 0))
    assert cells_limited.planning_budget_allows((0, 0), (2, 2))
    assert not cells_limited.planning_budget_allows((0, 0), (4, 4))


def test_forbidden_egress_only_allows_immediate_motion_away() -> None:
    edge = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.5,
        bounds_margin_m=2.0,
    )
    edge.plan(
        (0, 0),
        (-2, 0),
        changed_cells=(MapCellUpdate(1, 0, FORBIDDEN),),
    )
    source = (0.75, 0.5)

    assert not edge._segment_blocked(
        source,
        (0.5, 0.5),
        allow_forbidden_egress=True,
    )
    assert edge._segment_blocked(
        source,
        (0.9, 0.5),
        allow_forbidden_egress=True,
    )
    assert edge._segment_blocked(
        source,
        (0.75, 1.5),
        allow_forbidden_egress=True,
    )

    wall = DStarLitePlanner(
        ObservedGrid(ANCHOR),
        vehicle_radius_m=0.5,
        bounds_margin_m=2.0,
    )
    wall.plan(
        (0, 0),
        (-2, 0),
        changed_cells=(MapCellUpdate(1, 0, OCCUPIED),),
    )
    assert wall._segment_blocked(
        source,
        (0.5, 0.5),
        allow_forbidden_egress=True,
    )


def test_no_route_returns_none() -> None:
    search = planner(bounds_margin_m=0.0)
    wall = tuple(MapCellUpdate(2, y, OCCUPIED) for y in range(0, 4))
    assert search.plan((0, 0), (4, 3), changed_cells=wall) is None
    assert search.last_failure == "search_exhausted"


@pytest.mark.parametrize("blocked", [(0, 0), (4, 0)])
def test_occupied_start_or_goal_returns_none(blocked: tuple[int, int]) -> None:
    search = planner()
    assert search.plan(
        (0, 0),
        (4, 0),
        changed_cells=(MapCellUpdate(*blocked, OCCUPIED),),
    ) is None
    assert search.last_failure == (
        "start_blocked" if blocked == (0, 0) else "goal_blocked"
    )


def test_finite_search_without_extractable_path_is_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = planner()
    monkeypatch.setattr(search, "_extract_path", lambda: None)

    assert search.plan((0, 0), (4, 0)) is None
    assert search.last_failure == "path_extraction"


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


def test_near_equal_key_keeps_lexicographic_second_component_order() -> None:
    lower_second = (57.154328932550705, 46.42640687119285)
    higher_second = (57.154328932550684, 55.154328932550705)

    assert lower_second > higher_second
    assert _key_less(lower_second, higher_second)


def test_long_start_movement_reuses_one_search_and_accumulates_key_modifier() -> None:
    search = planner(bounds_margin_m=6.0)

    for start_x in range(41):
        path = search.plan((start_x, 0), (80, 0))
        assert path is not None
        assert path[0] == (start_x, 0)
        assert path[-1] == (80, 0)

    assert search.stats["resets"] == 1
    assert search.stats["key_modifier_cost"] == pytest.approx(40.0)
