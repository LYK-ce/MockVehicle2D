"""Finite-view goto closes scan, local SLAM, D* Lite, and safety in one loop."""

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
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.server import VehicleRuntime
from mockvehicle2d.vehicle import Vehicle


def pose(x: float = 0.0, y: float = 0.0, quality: str = "nominal") -> PoseEstimate:
    return PoseEstimate("nav-anchor", x, y, 0.0, (0.0, 0.0, 0.0), quality, 1.0, 1)


def delta(*updates: MapCellUpdate, revision: int = 1) -> LocalMapDelta:
    return LocalMapDelta("nav-anchor", revision, 1, 1.0, updates)


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
