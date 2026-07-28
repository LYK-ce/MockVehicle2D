"""Finite-view goto closes scan, local SLAM, D* Lite, and safety in one loop."""

from collections import deque
import math
from pathlib import Path
import sys
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.local_state import (
    AnchorSpec,
    AnchoredLocalState,
    FREE,
    FORBIDDEN,
    LocalMapDelta,
    MapCellUpdate,
    OCCUPIED,
    ObservedGrid,
    OdometryConfig,
    PoseEstimate,
)
from mockvehicle2d.map_grid import MapGrid, VOID
import mockvehicle2d.navigation as navigation_module
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner, PlanProgress
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.server import VehicleRuntime, handle_command_message
from mockvehicle2d.vehicle import Vehicle


def pose(x: float = 0.0, y: float = 0.0, quality: str = "nominal") -> PoseEstimate:
    return PoseEstimate("nav-anchor", x, y, 0.0, (0.0, 0.0, 0.0), quality, 1.0, 1)


def delta(*updates: MapCellUpdate, revision: int = 1) -> LocalMapDelta:
    return LocalMapDelta("nav-anchor", revision, 1, 1.0, updates)


def reference_reachable(
    planner: DStarLitePlanner,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    if planner._blocked(start) or planner._blocked(goal):
        return False
    frontier = deque((start,))
    visited = {start}
    while frontier:
        current = frontier.popleft()
        if current == goal:
            return True
        for neighbour in planner._neighbours(current):
            if (
                neighbour not in visited
                and math.isfinite(planner._cost(current, neighbour))
            ):
                visited.add(neighbour)
                frontier.append(neighbour)
    return False


def test_goto_plans_through_unknown_then_incrementally_detours() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, now=0.0)
    world = MapGrid.from_wall_set(30, 30, set())
    safety = LocalSafetyRuntime()

    navigation.start(
        6.0,
        0.0,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=0.0,
    )
    original = navigation.snapshot()["path"]
    vehicle.advance(world, 0.1)
    navigation.update(
        vehicle,
        world,
        0.1,
        safety,
        pose=pose(),
        advance_result=SafetyAdvanceResult(),
        local_map=observed,
        map_delta=delta(MapCellUpdate(3, 0, OCCUPIED)),
    )
    snapshot = navigation.snapshot()

    assert navigation.status == "active"
    assert snapshot["algorithm"] == "d_star_lite"
    assert snapshot["replan_count"] == 1
    assert snapshot["path_revision"] >= 2
    assert snapshot["path"] != original
    assert {"x_m": 3.5, "y_m": 0.5} not in snapshot["path"]
    assert 0 < vehicle.body_velocities()[0] <= vehicle.linear_speed * 0.5


def test_dynamic_obstacle_clear_restores_shorter_path_without_reset() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    navigation.start(
        6.0,
        0.0,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=0.0,
    )
    navigation.replan(pose(), delta(MapCellUpdate(3, 0, OCCUPIED)), observed)
    detour = navigation.snapshot()["path"]
    resets = navigation.snapshot()["planner_stats"]["resets"]
    navigation.replan(
        pose(1.0),
        delta(MapCellUpdate(3, 0, FREE), revision=2),
        observed,
    )
    restored = navigation.snapshot()["path"]

    assert len(restored) < len(detour)
    assert navigation.snapshot()["planner_stats"]["resets"] == resets


def test_nearby_safe_goal_is_within_one_metre_of_the_vehicle_body() -> None:
    def resolve(requested_x: float) -> GotoController:
        observed = ObservedGrid(
            AnchorSpec("nav-anchor", 0.0, 0.0, 0.0),
            resolution_m=0.5,
        )
        navigation = GotoController()
        vehicle = Vehicle(10.0, 10.0, radius=0.25, now=0.0)
        navigation.start(
            requested_x,
            0.25,
            local_map=observed,
            pose=pose(),
            vehicle_radius_m=vehicle.radius,
        )
        updates = tuple(
            MapCellUpdate(gx, gy, FREE)
            for gx in range(1, 4)
            for gy in range(-1, 2)
        ) + tuple(
            MapCellUpdate(4, gy, OCCUPIED)
            for gy in range(-32, 33)
        )
        navigation.update(
            vehicle,
            MapGrid.from_wall_set(30, 30, set()),
            0.0,
            pose=pose(),
            advance_result=SafetyAdvanceResult(),
            local_map=observed,
            map_delta=delta(*updates),
        )
        for _ in range(10):
            if navigation.snapshot()["planning"] is False:
                break
            navigation.update(
                vehicle,
                MapGrid.from_wall_set(30, 30, set()),
                0.0,
                pose=pose(),
                advance_result=SafetyAdvanceResult(),
                local_map=observed,
            )
        return navigation

    at_limit = resolve(2.25)
    assert at_limit.status == "active"
    assert at_limit.goal_mode == "nearby_safe"
    assert at_limit.goal == (1.25, 0.25)
    assert at_limit.snapshot()["approach_distance_m"] == 0.75

    body_edge_limit = resolve(2.75)
    assert body_edge_limit.status == "active"
    assert body_edge_limit.goal_mode == "approaching_safe_stop", (
        body_edge_limit.snapshot()["planning"],
        body_edge_limit.snapshot()["planner_stats"],
        body_edge_limit.snapshot()["detail"],
        body_edge_limit.snapshot()["path"],
    )
    assert body_edge_limit.snapshot()["approach_distance_m"] <= 1.0


def test_nearby_safe_goal_is_reselected_when_new_evidence_blocks_it() -> None:
    observed = ObservedGrid(
        AnchorSpec("nav-anchor", 0.0, 0.0, 0.0),
        resolution_m=0.5,
    )
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, radius=0.25, now=0.0)
    navigation.start(
        2.25,
        0.25,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=vehicle.radius,
    )
    initial_evidence = tuple(
        MapCellUpdate(gx, gy, FREE)
        for gx in range(-2, 9)
        for gy in range(-5, 6)
        if (gx, gy) != (4, 0)
    ) + (MapCellUpdate(4, 0, OCCUPIED),)
    navigation.update(
        vehicle,
        MapGrid.from_wall_set(30, 30, set()),
        0.0,
        pose=pose(),
        advance_result=SafetyAdvanceResult(),
        local_map=observed,
        map_delta=delta(*initial_evidence),
    )
    first_goal = navigation.goal
    assert navigation.goal_mode == "nearby_safe"
    assert first_goal is not None

    navigation.update(
        vehicle,
        MapGrid.from_wall_set(30, 30, set()),
        0.0,
        pose=pose(),
        advance_result=SafetyAdvanceResult(),
        local_map=observed,
        map_delta=delta(
            MapCellUpdate(
                math.floor(first_goal[0] / observed.resolution_m),
                math.floor(first_goal[1] / observed.resolution_m),
                OCCUPIED,
            ),
            revision=2,
        ),
    )

    assert navigation.status == "active"
    assert navigation.goal_mode == "nearby_safe"
    assert navigation.goal != first_goal


def test_fully_blocked_nearby_body_edge_region_stops_as_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        navigation_module,
        "CANDIDATE_INSPECTIONS_PER_UPDATE",
        4,
        raising=False,
    )
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, radius=0.5, now=0.0)
    navigation.start(
        4.5,
        0.5,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=vehicle.radius,
    )

    world = MapGrid.from_wall_set(30, 30, set())
    inspections = navigation.snapshot()["planner_stats"]["candidate_inspections"]
    navigation.update(
        vehicle,
        world,
        0.0,
        pose=pose(),
        advance_result=SafetyAdvanceResult(),
        local_map=observed,
        map_delta=delta(
            *(
                MapCellUpdate(gx, gy, OCCUPIED)
                for gx in range(3, 7)
                for gy in range(-2, 3)
            ),
        ),
    )

    current_inspections = navigation.snapshot()["planner_stats"][
        "candidate_inspections"
    ]
    assert current_inspections - inspections <= 4
    assert navigation.status == "active"
    assert navigation.snapshot()["planning"] is True
    for tick in range(1, 100):
        inspections = current_inspections
        navigation.update(
            vehicle,
            world,
            tick / 6,
            pose=pose(),
            advance_result=SafetyAdvanceResult(),
            local_map=observed,
        )
        current_inspections = navigation.snapshot()["planner_stats"][
            "candidate_inspections"
        ]
        assert current_inspections - inspections <= 4
        if navigation.status == "blocked":
            break

    assert (
        navigation.status,
        navigation.reason,
        navigation.detail,
    ) == ("blocked", "no_path", "nearby_safe_goal_unavailable")
    assert vehicle.command == "stop"
    assert vehicle.body_velocities() == (0.0, 0.0)


def test_nearby_candidate_at_planning_range_limit_stays_within_budget(
    monkeypatch,
) -> None:
    budget = 256
    monkeypatch.setattr(
        navigation_module,
        "PLANNING_EXPANSIONS_PER_UPDATE",
        budget,
        raising=False,
    )
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, radius=0.01, now=0.0)
    started = perf_counter()
    navigation.start(
        256.0,
        0.5,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=vehicle.radius,
    )
    durations = [perf_counter() - started]
    assert navigation.snapshot()["planner_stats"]["expansions"] <= budget
    assert navigation.snapshot()["planning"] is True
    assert vehicle.command == "stop"

    world = MapGrid.from_wall_set(30, 30, set())
    for tick in range(1, 100):
        before = navigation.snapshot()["planner_stats"]["expansions"]
        vehicle.advance(world, tick / 6)
        started = perf_counter()
        navigation.update(
            vehicle,
            world,
            tick / 6,
            pose=pose(),
            advance_result=SafetyAdvanceResult(),
            local_map=observed,
            map_delta=(
                delta(MapCellUpdate(256, 0, OCCUPIED))
                if tick == 1
                else None
            ),
        )
        durations.append(perf_counter() - started)
        after = navigation.snapshot()["planner_stats"]["expansions"]
        assert after - before <= budget
        if navigation.snapshot()["planning"] is False:
            break

    assert navigation.status == "active"
    assert navigation.goal_mode == "approaching_safe_stop"
    assert navigation.snapshot()["approach_distance_m"] <= 1.0
    assert navigation.snapshot()["planning"] is False
    assert max(durations) < 0.5


def test_manual_command_cancels_pending_planning() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, radius=0.01, now=0.0)
    world = MapGrid.from_wall_set(30, 30, set())
    navigation.start(
        256.0,
        0.5,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=vehicle.radius,
    )
    assert navigation.snapshot()["planning"] is True

    reply = handle_command_message(
        '{"type":"drive","seq":2,"linear_mps":0.1,"angular_rps":0}',
        vehicle,
        world,
        0.0,
        1.0,
        navigation,
    )

    assert reply["accepted"] is True
    assert (navigation.status, navigation.reason) == (
        "cancelled",
        "manual_override",
    )
    assert navigation.snapshot()["planning"] is False
    assert vehicle.command == "drive"


def test_expansion_limit_blocks_instead_of_searching_fallbacks(monkeypatch) -> None:
    def exhausted(planner, *args, **kwargs) -> PlanProgress:
        planner.last_failure = "expansion_limit"
        return PlanProgress("unreachable")

    monkeypatch.setattr(DStarLitePlanner, "advance_plan", exhausted)
    navigation = GotoController()
    navigation.start(
        4.0,
        0.0,
        local_map=ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0)),
        pose=pose(),
    )

    assert (
        navigation.status,
        navigation.reason,
        navigation.detail,
    ) == ("blocked", "no_path", "expansion_limit")
    assert navigation.snapshot()["planning"] is False


def test_replan_does_not_restart_cancelled_goal() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    navigation.start(
        256.0,
        0.5,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=0.01,
    )
    navigation.cancel("manual_override")

    navigation.replan(pose(), None, observed)

    assert (navigation.status, navigation.reason) == (
        "cancelled",
        "manual_override",
    )
    assert navigation.snapshot()["planning"] is False


def test_replan_does_not_restart_collision_or_localization_block() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    world = MapGrid.from_wall_set(30, 30, set())

    for reason in ("collision", "localization_lost"):
        navigation = GotoController()
        vehicle = Vehicle(10.0, 10.0, now=0.0)
        navigation.start(4.0, 0.0, local_map=observed, pose=pose())
        if reason == "collision":
            navigation.update(
                vehicle,
                world,
                0.0,
                pose=pose(),
                advance_result=SafetyAdvanceResult(collided=True),
                local_map=observed,
            )
        else:
            assert navigation.block_for_localization_loss(
                vehicle,
                pose(quality="lost"),
                0.0,
            )

        navigation.replan(pose(), None, observed)

        assert (navigation.status, navigation.reason) == ("blocked", reason)
        assert navigation.snapshot()["planning"] is False


def test_unobstructed_off_centre_pose_keeps_the_next_planned_waypoint() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, now=0.0)
    off_centre = pose(0.1, 0.1)
    navigation.start(
        3.0,
        2.0,
        local_map=observed,
        pose=off_centre,
        vehicle_radius_m=0.5,
    )
    next_cell = navigation._path[1]
    world = MapGrid.from_wall_set(30, 30, set())
    vehicle.advance(world, 0.1)

    navigation.update(
        vehicle,
        world,
        0.1,
        pose=off_centre,
        advance_result=SafetyAdvanceResult(),
        local_map=observed,
    )

    assert navigation._current_waypoint == (
        next_cell[0] + 0.5,
        next_cell[1] + 0.5,
    )


def test_unsafe_exact_pose_connection_blocks_instead_of_moving_blindly() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, now=0.0)
    navigation.start(
        3.0,
        2.0,
        local_map=observed,
        pose=pose(),
        vehicle_radius_m=0.5,
    )

    navigation.update(
        vehicle,
        MapGrid.from_wall_set(30, 30, set()),
        0.1,
        pose=pose(0.9, 0.1),
        advance_result=SafetyAdvanceResult(),
        local_map=observed,
        map_delta=delta(MapCellUpdate(1, -1, OCCUPIED)),
    )

    assert (
        navigation.status,
        navigation.reason,
        navigation.detail,
    ) == ("blocked", "no_path", "start_connection_unsafe")
    assert vehicle.body_velocities() == (0.0, 0.0)


def test_lost_pose_blocks_and_does_not_reuse_old_velocity() -> None:
    observed = ObservedGrid(AnchorSpec("nav-anchor", 0.0, 0.0, 0.0))
    navigation = GotoController()
    vehicle = Vehicle(10.0, 10.0, now=0.0)
    navigation.start(4.0, 0.0, local_map=observed, pose=pose())
    vehicle.install_drive(0.5, 0.0, 0.0)

    navigation.update(
        vehicle,
        MapGrid.from_wall_set(30, 30, set()),
        0.1,
        pose=pose(quality="lost"),
        advance_result=SafetyAdvanceResult(),
        local_map=observed,
    )

    assert navigation.snapshot()["reason"] == "localization_lost"
    assert vehicle.body_velocities() == (0.0, 0.0)


def test_runtime_discovers_hidden_route_obstacle_before_collision_and_reaches_goal() -> None:
    world = MapGrid.from_wall_set(32, 12, {(16, 5)})
    vehicle = Vehicle(2.5, 5.5, radius=0.5, command_timeout=1.0, now=0.0)
    anchor = AnchorSpec("nav-anchor", vehicle.x, vehicle.y, 0.0)
    runtime = VehicleRuntime(
        [],
        world,
        vehicle,
        GotoController(),
        LocalSafetyRuntime(),
        AnchoredLocalState(
            anchor,
            truth_x_m=vehicle.x,
            truth_y_m=vehicle.y,
            truth_yaw_rad=vehicle.yaw,
            odometry_config=OdometryConfig(),
            timestamp=0.0,
        ),
    )
    runtime.navigation.start(
        26.5 - anchor.global_x_m,
        5.5 - anchor.global_y_m,
        reported_goal=(26.5, 5.5),
        local_map=runtime.local_state.local_map,
        pose=runtime.local_state.pose,
        vehicle_radius_m=vehicle.radius,
    )
    initial_path = runtime.navigation.snapshot()["path"]

    first_replan_x = None
    for tick in range(1, 900):
        now = tick / 6
        frame = runtime.update(now, now)
        assert frame.pose_timestamp == frame.scan_timestamp == now
        assert not runtime.vehicle.collision
        if runtime.navigation.snapshot()["replan_count"] and first_replan_x is None:
            first_replan_x = runtime.vehicle.x
        if runtime.navigation.status == "reached":
            break

    assert first_replan_x is not None
    assert first_replan_x < 16 - vehicle.radius
    assert runtime.navigation.snapshot()["path"] != initial_path
    assert runtime.navigation.status == "reached"
    assert math.hypot(runtime.vehicle.x - 26.5, runtime.vehicle.y - 5.5) <= 0.2


def test_runtime_maps_a_hidden_drop_and_replans_around_it() -> None:
    world = MapGrid.from_wall_set(20, 12, set())
    vehicle = Vehicle(2.5, 5.5, radius=0.5, command_timeout=1.0, now=0.0)
    anchor = AnchorSpec("nav-anchor", vehicle.x, vehicle.y, 0.0)
    runtime = VehicleRuntime(
        [],
        world,
        vehicle,
        GotoController(),
        LocalSafetyRuntime(),
        AnchoredLocalState(
            anchor,
            truth_x_m=vehicle.x,
            truth_y_m=vehicle.y,
            truth_yaw_rad=vehicle.yaw,
            timestamp=0.0,
        ),
    )
    runtime.navigation.start(
        10.0,
        0.0,
        reported_goal=(12.5, 5.5),
        local_map=runtime.local_state.local_map,
        pose=runtime.local_state.pose,
        vehicle_radius_m=vehicle.radius,
    )

    saw_edge_stop = False
    inserted_drop = False
    for tick in range(1, 900):
        now = tick / 6
        if not inserted_drop and runtime.vehicle.x >= 6.35:
            for y in range(3, 9):
                world.set_cell(7, y, VOID)
            inserted_drop = True
        frame = runtime.update(now, now)
        saw_edge_stop |= frame.safety_stop == "safety_edge"
        assert not runtime.vehicle.collision
        if runtime.navigation.status == "reached":
            break

    assert inserted_drop and saw_edge_stop, (
        runtime.navigation.snapshot(),
        runtime.safety.snapshot(),
        runtime.vehicle.x,
        runtime.vehicle.y,
    )
    assert any(
        cell["state"] == FORBIDDEN
        for cell in runtime.local_state.local_map.snapshot()["cells"]
    )
    assert runtime.navigation.snapshot()["replan_count"] >= 1
    assert runtime.navigation.status == "reached"
    assert math.hypot(runtime.vehicle.x - 12.5, runtime.vehicle.y - 5.5) <= 0.2


def test_runtime_detours_to_safe_stop_when_exact_goal_becomes_unsafe() -> None:
    world = MapGrid.from_wall_set(16, 12, {(8, 5)})
    vehicle = Vehicle(2.5, 5.5, radius=0.5, command_timeout=1.0, now=0.0)
    anchor = AnchorSpec("nav-anchor", vehicle.x, vehicle.y, 0.0)
    runtime = VehicleRuntime(
        [],
        world,
        vehicle,
        GotoController(),
        LocalSafetyRuntime(),
        AnchoredLocalState(
            anchor,
            truth_x_m=vehicle.x,
            truth_y_m=vehicle.y,
            truth_yaw_rad=vehicle.yaw,
            timestamp=0.0,
            map_resolution_m=0.5,
        ),
    )
    goal = (6.0, 0.0)
    assert runtime.local_state.local_map.is_unknown(12, 0)
    runtime.navigation.start(
        *goal,
        reported_goal=(8.5, 5.5),
        local_map=runtime.local_state.local_map,
        pose=runtime.local_state.pose,
        vehicle_radius_m=vehicle.radius,
    )
    assert runtime.navigation.status == "active"

    fallback_goals = []
    for tick in range(1, 900):
        runtime.update(tick / 6, tick / 6)
        if runtime.navigation.snapshot()["goal_mode"] == "nearby_safe":
            fallback_goals.append(runtime.navigation.goal)
        if runtime.navigation.status != "active":
            break
    assert runtime.local_state.local_map.occupied_cells()
    assert (
        runtime.navigation.status,
        runtime.navigation.reason,
        runtime.navigation.detail,
    ) == ("reached", "nearby_safe_stop", "goal_blocked"), (
        runtime.navigation.snapshot(),
        runtime.local_state.pose,
        fallback_goals[-3:],
        runtime.local_state.local_map.occupied_cells(),
    )
    assert fallback_goals
    assert runtime.navigation.requested_goal == goal
    assert runtime.navigation.reported_goal == (8.5, 5.5)
    assert runtime.navigation.goal != goal
    assert math.dist(runtime.navigation.goal, goal) - vehicle.radius <= 1.0 + 1e-9
    assert math.hypot(
        runtime.local_state.pose.x_m - runtime.navigation.goal[0],
        runtime.local_state.pose.y_m - runtime.navigation.goal[1],
    ) <= runtime.navigation.goal_tolerance_m
    assert runtime.vehicle.command == "stop"
    assert runtime.vehicle.body_velocities() == (0.0, 0.0)


def test_default_resolution_wall_goal_keeps_approaching_a_body_edge_safe_stop() -> None:
    world = MapGrid.from_wall_set(24, 20, {(16, 10)})
    vehicle = Vehicle(10.0, 10.0, radius=0.5, command_timeout=1.0, now=0.0)
    anchor = AnchorSpec("nav-anchor", vehicle.x, vehicle.y, 0.0)
    runtime = VehicleRuntime(
        [],
        world,
        vehicle,
        GotoController(),
        LocalSafetyRuntime(),
        AnchoredLocalState(
            anchor,
            truth_x_m=vehicle.x,
            truth_y_m=vehicle.y,
            truth_yaw_rad=vehicle.yaw,
            timestamp=0.0,
        ),
    )
    requested_goal = (6.5, 0.5)
    runtime.navigation.start(
        *requested_goal,
        reported_goal=(16.5, 10.5),
        local_map=runtime.local_state.local_map,
        pose=runtime.local_state.pose,
        vehicle_radius_m=vehicle.radius,
    )

    runtime.update(1 / 6, 1 / 6)

    snapshot = runtime.navigation.snapshot()
    assert snapshot["status"] == "active"
    assert snapshot["goal_mode"] in {"approaching_safe_stop", "nearby_safe"}
    assert snapshot["goal"] == {"x_m": 16.5, "y_m": 10.5}
    assert snapshot["requested_goal"] == {
        "frame_id": "anchor_map",
        "x_m": 6.5,
        "y_m": 0.5,
    }
    assert snapshot["approach_distance_m"] <= 1.0
    assert runtime.vehicle.command == "drive"

    for tick in range(2, 900):
        runtime.update(tick / 6, tick / 6)
        assert not runtime.vehicle.collision
        if runtime.navigation.status != "active":
            break

    snapshot = runtime.navigation.snapshot()
    center_distance_m = math.dist(
        (runtime.local_state.pose.x_m, runtime.local_state.pose.y_m),
        requested_goal,
    )
    assert (
        snapshot["status"],
        snapshot["reason"],
        snapshot["detail"],
    ) == ("reached", "nearby_safe_stop", "goal_blocked"), (
        snapshot,
        runtime.local_state.pose,
        runtime.local_state.local_map.occupied_cells(),
    )
    assert max(0.0, center_distance_m - vehicle.radius) <= 1.0 + 1e-9
    assert center_distance_m > 1.0
    assert runtime.vehicle.command == "stop"


def test_default_runtime_long_goto_does_not_end_in_no_path() -> None:
    runtime = VehicleRuntime.create(
        started_at=0.0,
        timestamp=0.0,
        anchor=AnchorSpec("mock_vehicle_01_anchor", 10.0, 10.0, 0.0),
        odometry_config=OdometryConfig(),
    )
    for tick in range(1, 7):
        runtime.update(tick / 6, tick / 6)

    def drive_to_global(x_m: float, y_m: float) -> None:
        nonlocal tick
        local_x_m, local_y_m, _ = runtime.local_state.anchor.global_to_anchor(
            x_m, y_m
        )
        runtime.navigation.start(
            local_x_m,
            local_y_m,
            reported_goal=(x_m, y_m),
            local_map=runtime.local_state.local_map,
            pose=runtime.local_state.pose,
            vehicle_radius_m=runtime.vehicle.radius,
        )
        deadline = tick + 1200
        while runtime.navigation.status == "active" and tick < deadline:
            tick += 1
            runtime.update(tick / 6, tick / 6)

        planner = runtime.navigation._planner
        assert planner is not None
        start = runtime.navigation._pose_cell(
            runtime.local_state.pose, runtime.local_state.local_map
        )
        goal = runtime.navigation._goal_cell(runtime.local_state.local_map)
        fresh = DStarLitePlanner(
            runtime.local_state.local_map,
            vehicle_radius_m=runtime.vehicle.radius,
        )
        differential = (
            reference_reachable(planner, start, goal),
            fresh.plan(start, goal) is not None,
            planner._extract_path() is not None,
        )
        assert runtime.navigation.status == "reached", (
            (x_m, y_m),
            runtime.navigation.snapshot(),
            differential,
        )
        assert differential == (True, True, True)
        assert planner.stats["resets"] == 1

    for goal in ((30.0, 30.0), (20.0, 20.0), (22.0, 28.0)):
        drive_to_global(*goal)
