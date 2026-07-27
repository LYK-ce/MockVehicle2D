"""Finite-view goto closes scan, local SLAM, D* Lite, and safety in one loop."""

from collections import deque
import math
from pathlib import Path
import sys


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
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.server import VehicleRuntime
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


def test_runtime_blocks_goal_when_observed_obstacle_makes_arrival_unsafe() -> None:
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
        ),
    )
    goal = (7.0, 0.5)
    assert runtime.local_state.local_map.is_unknown(7, 0)
    runtime.navigation.start(
        *goal,
        reported_goal=(9.5, 6.0),
        local_map=runtime.local_state.local_map,
        pose=runtime.local_state.pose,
        vehicle_radius_m=vehicle.radius,
    )
    assert runtime.navigation.status == "active"

    for tick in range(1, 121):
        runtime.update(tick / 6, tick / 6)
        if runtime.navigation.status != "active":
            break
    assert runtime.local_state.local_map.occupied_cells()
    assert (
        runtime.navigation.status,
        runtime.navigation.reason,
        runtime.navigation.detail,
    ) == ("blocked", "no_path", "goal_blocked")
    assert runtime.vehicle.command == "stop"
    assert runtime.vehicle.body_velocities() == (0.0, 0.0)


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
