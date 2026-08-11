"""PIBT-inspired leased-cell coordination contracts."""

import unittest
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mockvehicle2d.coordination import TimedCell
from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    AutoState,
    CoverageMission,
    RobotController,
    GotoMission,
    ModeAction,
    ModeCommand,
    OpMode,
    PatrolMission,
    corridor_descriptors_conflict,
    _front_corridor_waiter,
    _global_coordination_cells,
    _motion_intents_conflict,
    inherit_motion_priority,
    motion_intent_precedes,
)
from mockvehicle2d.episode import run_episode
from mockvehicle2d.fleet import (
    AnchorPose,
    FleetRuntime,
    FleetScenario,
    FleetVehicleSpec,
    _TransientPlanningGrid,
)
from mockvehicle2d.local_state import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    AnchorSpec,
    LocalMapDelta,
    MapCellUpdate,
    ObservedGrid,
    PoseEstimate,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.map_sync import (
    CorridorDescriptor,
    MAX_GRID_COORDINATE,
    MOTION_COMMIT_HORIZON_S,
    MOTION_INTENT_PROTOCOL,
    MOTION_INTENT_TTL_S,
    MapSyncState,
    PeerMotionIntent,
    PeerVehicleState,
    VacateRequest,
)
from mockvehicle2d.navigation import (
    GotoController,
    _point_route_distance,
    _point_segment_distance,
)
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.safety import (
    AUTOMATIC_MINIMUM_CLEARANCE_M,
    LocalSafetyRuntime,
    SafetyAdvanceResult,
    SafetyDecision,
)
from mockvehicle2d.scan import LaserPoint
from mockvehicle2d.vehicle import Vehicle


def inject_active_goto(controller, mission_id, x_m, y_m):
    mission = GotoMission(mission_id, "global_map", x_m, y_m, 1)
    controller.active_mission = mission
    controller._active_subgoals = mission.subgoals


def intent(
    vehicle_id: str,
    *,
    current: tuple[int, int],
    target: tuple[int, int] | None,
    wait_ticks: int,
    owner: str | None = None,
    reserved: bool = False,
    corridor: CorridorDescriptor | None = None,
    task_age_ticks: int = 0,
) -> PeerMotionIntent:
    return PeerMotionIntent(
        vehicle_id,
        1,
        1,
        1.0,
        0.35,
        current,
        target,
        wait_ticks,
        owner or vehicle_id,
        reserved,
        corridor,
        task_age_ticks=task_age_ticks,
    )


def vacate_intent(
    source_vehicle_id: str = "vehicle_a",
    *,
    request_vehicle_id: str = "vehicle_d",
    request_cell: tuple[int, int] = (2, 0),
    route: tuple[tuple[int, int], ...] = ((0, 0), (2, 0), (4, 0)),
    current: tuple[int, int] = (0, 0),
    wait_ticks: int = 4,
    **changes: object,
) -> PeerMotionIntent:
    fields = {
        "task_sequence": 1,
        "vacate_request": VacateRequest(request_vehicle_id, request_cell, route),
        "received_at_s": 1.0,
    }
    fields.update(changes)
    return replace(
        intent(
            source_vehicle_id,
            current=current,
            target=None,
            wait_ticks=wait_ticks,
        ),
        **fields,
    )


def peer_state(
    vehicle_id: str,
    x_m: float,
    y_m: float,
    vx_mps: float,
    vy_mps: float = 0.0,
) -> PeerVehicleState:
    return PeerVehicleState(
        vehicle_id,
        1,
        1,
        1,
        1.0,
        x_m,
        y_m,
        0.0,
        (0.0, 0.0, 0.0),
        "nominal",
        vx_mps,
        vy_mps,
        0.0,
        0.5,
    )


def coordinate_once(
    controller: RobotController,
    vehicle: Vehicle,
    anchor: AnchorSpec,
    local_map: ObservedGrid,
    *,
    vehicle_id: str,
    now: float,
    position_m: tuple[float, float],
    peer_states: tuple[PeerVehicleState, ...],
    peer_intents: tuple[PeerMotionIntent, ...],
    desired: tuple[float, float] = (0.5, 0.0),
) -> tuple[float, float]:
    return controller._coordinate_desired(
        desired,
        vehicle=vehicle,
        vehicle_id=vehicle_id,
        anchor=anchor,
        pose=PoseEstimate(
            anchor.anchor_id,
            *position_m,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            now,
            round(now * 10),
        ),
        local_map=local_map,
        now=now,
        peer_states=peer_states,
        peer_motion_intents=peer_intents,
        coordination_ready=True,
        expected_peer_vehicle_ids=tuple(
            intent.source_vehicle_id for intent in peer_intents
        ),
    )


def parked_idle_request_once(
    *,
    route: tuple[tuple[int, int], ...] | None,
    own_vehicle_id: str = "vehicle_a",
    peer_vehicle_id: str = "vehicle_b",
    peer_current: tuple[int, int] = (2, 0),
    peer_position_m: tuple[float, float] = (2.5, 0.5),
    peer_state_generation: int = 1,
    peer_intent_generation: int = 1,
) -> tuple[RobotController, tuple[float, float]]:
    navigation = Mock(
        transient_peer_blocked=True,
        motion_target=None,
        coordination_corridor=Mock(return_value=None),
        coordination_path_cells=Mock(return_value=route),
    )
    controller = RobotController(navigation)
    inject_active_goto(controller, f"goto-{own_vehicle_id}", 4.5, 0.5)
    controller.mode = OpMode.AUTO
    controller.auto_state = AutoState.ACTIVE
    desired = coordinate_parked_idle_blocker(
        controller,
        route=route,
        own_vehicle_id=own_vehicle_id,
        peer_vehicle_id=peer_vehicle_id,
        peer_current=peer_current,
        peer_position_m=peer_position_m,
        peer_state_generation=peer_state_generation,
        peer_intent_generation=peer_intent_generation,
    )
    return controller, desired


def coordinate_parked_idle_blocker(
    controller: RobotController,
    *,
    route: tuple[tuple[int, int], ...] | None,
    own_vehicle_id: str = "vehicle_a",
    peer_vehicle_id: str = "vehicle_b",
    peer_current: tuple[int, int] = (2, 0),
    peer_position_m: tuple[float, float] = (2.5, 0.5),
    peer_state_generation: int = 1,
    peer_intent_generation: int = 1,
) -> tuple[float, float]:
    anchor = AnchorSpec(f"spawn_{own_vehicle_id}", 0.0, 0.0, 0.0)
    local_map = ObservedGrid(anchor)
    controller.navigation.coordination_path_cells.return_value = route
    state = peer_state(peer_vehicle_id, *peer_position_m, 0.0)
    if peer_state_generation != state.state_generation:
        state = replace(state, state_generation=peer_state_generation)
    blocker = PeerMotionIntent(
        peer_vehicle_id,
        peer_intent_generation,
        1,
        1.0,
        MOTION_INTENT_TTL_S,
        peer_current,
        None,
        0,
        peer_vehicle_id,
        False,
        task_sequence=(1 << 64) - 1,
    )
    return coordinate_once(
        controller,
        Vehicle(0.5, 0.5, radius=0.5, now=0.0),
        anchor,
        local_map,
        vehicle_id=own_vehicle_id,
        now=1.0,
        position_m=(0.5, 0.5),
        peer_states=(state,),
        peer_intents=(blocker,),
        desired=(0.0, 0.0),
    )


def motion_sync_states() -> tuple[MapSyncState, MapSyncState, PoseEstimate]:
    source_anchor = AnchorSpec("spawn_1", 0.0, 0.0, 0.0)
    receiver_anchor = AnchorSpec("spawn_2", 0.0, 0.0, 0.0)
    source = MapSyncState(
        "session_1",
        "vehicle_1",
        source_anchor,
        1.0,
        state_generation=1,
    )
    receiver = MapSyncState(
        "session_1",
        "vehicle_2",
        receiver_anchor,
        1.0,
        clock=lambda: 1.0,
        state_generation=2,
    )
    source.configure_network(
        "peer_1", {"vehicle_2": ("peer_2", receiver_anchor)}
    )
    receiver.configure_network(
        "peer_2", {"vehicle_1": ("peer_1", source_anchor)}
    )
    pose = PoseEstimate(
        source_anchor.anchor_id,
        0.5,
        0.5,
        0.0,
        (0.0, 0.0, 0.0),
        "nominal",
        1.0,
        1,
    )
    return source, receiver, pose


def vacate_request_payload() -> tuple[MapSyncState, dict[str, object]]:
    source, receiver, pose = motion_sync_states()
    source.record_motion_intent(
        pose,
        target_m=None,
        wait_ticks=0,
        priority_owner_id="vehicle_1",
        reserved=False,
        timestamp_s=1.0,
        vacate_request=VacateRequest(
            "vehicle_2",
            (1, 0),
            ((0, 0), (1, 0), (3, 0)),
        ),
    )
    payload = source.prepare_motion_intent()
    assert payload is not None
    return receiver, payload


def idle_vacate_tick(
    requester: PeerMotionIntent,
    requester_state: PeerVehicleState,
    *,
    mode: OpMode = OpMode.AUTO,
    auto_state: AutoState = AutoState.IDLE,
    active_mission: GotoMission | None = None,
    pending_mission: GotoMission | None = None,
    detours: tuple[tuple[float, float], ...] = ((2.5, 1.5),),
    vacate_path: tuple[tuple[float, float], ...] = (),
    corridor: tuple[tuple[int, int], tuple[int, int]] | None = None,
    walls: frozenset[tuple[int, int]] = frozenset(),
    now: float = 1.0,
    pose_quality: str = "nominal",
    pose_yaw_rad: float = 0.0,
) -> tuple[RobotController, Vehicle, LocalSafetyRuntime, Mock]:
    anchor = AnchorSpec("idle-responder", 0.0, 0.0, 0.0)
    pose = PoseEstimate(
        anchor.anchor_id,
        2.5,
        0.5,
        pose_yaw_rad,
        (0.0, 0.0, 0.0),
        pose_quality,
        now,
        round(now * 10),
    )
    navigation = Mock(
        status="reached",
        motion_target=None,
        coordination_corridor=Mock(return_value=corridor),
        coordination_detours=Mock(return_value=detours),
        coordination_vacate_path=Mock(return_value=vacate_path),
        coordination_path_cells=Mock(return_value=None),
    )
    controller = RobotController(navigation)
    controller.mode = mode
    controller.auto_state = auto_state
    if active_mission is not None:
        controller.active_mission = active_mission
        controller._active_subgoals = active_mission.subgoals
    if pending_mission is not None:
        controller._pending.append(pending_mission)
    vehicle = Vehicle(2.5, 0.5, yaw=pose_yaw_rad, now=now)
    safety = LocalSafetyRuntime()
    controller.tick(
        vehicle=vehicle,
        grid=MapGrid.from_wall_set(8, 8, set(walls)),
        safety=safety,
        anchor=anchor,
        pose=pose,
        local_map=ObservedGrid(anchor),
        map_delta=None,
        advance_result=SafetyAdvanceResult(),
        now=now,
        vehicle_id="vehicle_d",
        peer_states=(requester_state,),
        peer_motion_intents=(requester,),
        coordination_ready=True,
        expected_peer_vehicle_ids=(requester.source_vehicle_id,),
    )
    return controller, vehicle, safety, navigation


def tick_real_idle_vacate(
    controller: RobotController,
    vehicle: Vehicle,
    safety: LocalSafetyRuntime,
    anchor: AnchorSpec,
    local_map: ObservedGrid,
    *,
    position_m: tuple[float, float],
    now: float,
    requester: PeerMotionIntent | None,
    requester_state: PeerVehicleState | None,
    safety_scan_healthy: bool = True,
    yaw_rad: float | None = None,
    walls: frozenset[tuple[int, int]] = frozenset(),
    map_delta: LocalMapDelta | None = None,
    advance_result: SafetyAdvanceResult = SafetyAdvanceResult(),
    pose_quality: str = "nominal",
    coordination_map: ObservedGrid | None = None,
) -> None:
    grid = MapGrid.from_wall_set(8, 8, set(walls))
    vehicle.advance(grid, now)
    vehicle.x, vehicle.y = position_m
    if yaw_rad is not None:
        vehicle.yaw = yaw_rad
    controller.tick(
        vehicle=vehicle,
        grid=grid,
        safety=safety,
        anchor=anchor,
        pose=PoseEstimate(
            anchor.anchor_id,
            *position_m,
            vehicle.yaw,
            (0.0, 0.0, 0.0),
            pose_quality,
            now,
            round(now * 10),
        ),
        local_map=local_map,
        map_delta=map_delta,
        advance_result=advance_result,
        now=now,
        safety_scan_healthy=safety_scan_healthy,
        vehicle_id="vehicle_d",
        peer_states=(() if requester_state is None else (requester_state,)),
        peer_motion_intents=(() if requester is None else (requester,)),
        coordination_ready=True,
        expected_peer_vehicle_ids=("vehicle_a",),
        coordination_map=coordination_map,
    )


def real_idle_vacate_session() -> tuple[
    RobotController,
    Vehicle,
    LocalSafetyRuntime,
    AnchorSpec,
    ObservedGrid,
    PeerMotionIntent,
]:
    anchor = AnchorSpec("idle-responder", 0.0, 0.0, 0.0)
    local_map = ObservedGrid(anchor)
    origin = PoseEstimate(
        anchor.anchor_id,
        2.5,
        0.5,
        0.0,
        (0.0, 0.0, 0.0),
        "nominal",
        1.0,
        10,
    )
    navigation = GotoController()
    navigation.start(2.5, 0.5, local_map=local_map, pose=origin)
    for _ in range(4):
        navigation.update(
            pose=origin,
            local_map=local_map,
            max_linear_mps=0.5,
            max_angular_rps=math.pi / 2,
        )
        if navigation.status == "reached":
            break
    assert navigation.status == "reached"
    controller = RobotController(navigation)
    controller.mode = OpMode.AUTO
    vehicle = Vehicle(2.5, 0.5, now=1.0)
    safety = LocalSafetyRuntime()
    requester = vacate_intent()
    tick_real_idle_vacate(
        controller,
        vehicle,
        safety,
        anchor,
        local_map,
        position_m=(2.5, 0.5),
        now=1.0,
        requester=requester,
        requester_state=peer_state("vehicle_a", 0.5, 0.5, 0.0),
    )
    assert controller.is_automatic_motion_active
    return controller, vehicle, safety, anchor, local_map, requester


def enter_real_idle_vacate_session(
    controller: RobotController,
    vehicle: Vehicle,
    safety: LocalSafetyRuntime,
    anchor: AnchorSpec,
    local_map: ObservedGrid,
    requester: PeerMotionIntent,
) -> PeerMotionIntent:
    entered = replace(
        requester,
        sequence=2,
        timestamp_s=1.1,
        received_at_s=1.1,
        current_cell=(2, 0),
    )
    entered_state = replace(
        peer_state("vehicle_a", 2.5, 0.5, 0.0),
        sequence=2,
        timestamp_s=1.1,
    )
    tick_real_idle_vacate(
        controller,
        vehicle,
        safety,
        anchor,
        local_map,
        position_m=(2.5, 2.5),
        now=1.1,
        requester=entered,
        requester_state=entered_state,
    )
    return entered


_CLEAR_VACATE_TRAJECTORY = (
    TimedCell((5, 0), 0.0, 0.0),
    TimedCell((5, 1), 1.0, 4.0),
)


def clear_vacate_evidence(
    requester: PeerMotionIntent,
    *,
    now: float,
    sequence: int,
    trajectory: tuple[TimedCell, ...] = _CLEAR_VACATE_TRAJECTORY,
    generation: int = 1,
    **changes: object,
) -> tuple[PeerMotionIntent, PeerVehicleState]:
    evidence = replace(
        requester,
        intent_generation=generation,
        sequence=sequence,
        timestamp_s=now,
        received_at_s=now,
        current_cell=(5, 0),
        target_cell=trajectory[-1].cell,
        trajectory=trajectory,
        vacate_request=None,
        **changes,
    )
    state = replace(
        peer_state(requester.source_vehicle_id, 5.5, 0.5, 0.0),
        state_generation=generation,
        sequence=sequence,
        timestamp_s=now,
    )
    return evidence, state


def release_real_idle_vacate_session() -> tuple[
    RobotController,
    Vehicle,
    LocalSafetyRuntime,
    AnchorSpec,
    ObservedGrid,
    PeerMotionIntent,
    PeerVehicleState,
]:
    (
        controller,
        vehicle,
        safety,
        anchor,
        local_map,
        requester,
    ) = real_idle_vacate_session()
    entered = enter_real_idle_vacate_session(
        controller,
        vehicle,
        safety,
        anchor,
        local_map,
        requester,
    )
    cleared = entered
    for clear_tick in range(1, 4):
        now = 1.1 + clear_tick / 10
        cleared, state = clear_vacate_evidence(
            entered,
            now=now,
            sequence=2 + clear_tick,
        )
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 2.5),
            now=now,
            requester=cleared,
            requester_state=state,
        )
    assert controller.navigation.status == "active"
    return controller, vehicle, safety, anchor, local_map, cleared, state


class TestMotionCoordination(unittest.TestCase):
    def test_off_axis_route_detects_corridor_from_static_own_map(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        walls = {
            (gx, gy)
            for gx in range(5, 10)
            for gy in (*range(0, 4), *range(7, 11))
        } | {
            (gx, gy)
            for gx in range(15)
            for gy in (0, 10)
        } | {
            (gx, gy)
            for gx in (0, 14)
            for gy in range(11)
        }
        cells = {
            (gx, gy): OCCUPIED if (gx, gy) in walls else FREE
            for gx in range(15)
            for gy in range(11)
        }
        cells.pop((10, 3))
        cells.pop((10, 7))
        local_map = Mock(resolution_m=1.0, revision=1)
        local_map.cell_without_peers = None
        local_map.get_cell.side_effect = lambda gx, gy: cells.get(
            (gx, gy), UNKNOWN
        )
        local_map.snapshot.return_value = {
            "anchor_id": anchor.anchor_id,
            "frame_id": "anchor_map",
            "resolution_m": 1.0,
            "revision": 1,
            "cells": [
                {"gx": gx, "gy": gy, "state": state}
                for (gx, gy), state in sorted(cells.items())
            ],
        }
        pose = PoseEstimate(
            anchor.anchor_id,
            2.5,
            2.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = GotoController()
        navigation.status = "active"
        navigation.requested_goal = (12.5, 7.5)
        navigation.goal = navigation.requested_goal
        navigation._vehicle_radius_m = 0.5
        dynamic_planner = Mock()
        navigation._planner = dynamic_planner

        corridor = None
        for _ in range(16):
            corridor = navigation.coordination_corridor(pose, local_map)
            if corridor is not None:
                break

        self.assertEqual(corridor, ((5, 5), (9, 5)))
        self.assertIs(navigation._planner, dynamic_planner)
        self.assertEqual(navigation._path, [])
        self.assertEqual(dynamic_planner.method_calls, [])

        local_map.snapshot.reset_mock()
        self.assertIsNotNone(navigation.coordination_path_cells(pose, local_map))
        self.assertEqual(local_map.snapshot.call_count, 0)

        cells.pop((4, 3))
        local_map.revision = 2
        local_map.snapshot.return_value = {
            **local_map.snapshot.return_value,
            "revision": 2,
            "cells": [
                {"gx": gx, "gy": gy, "state": state}
                for (gx, gy), state in sorted(cells.items())
            ],
        }
        self.assertTrue(
            all(
                navigation.coordination_corridor(pose, local_map) is None
                for _ in range(16)
            )
        )

    def test_vacate_path_can_look_past_a_wall_corner(self) -> None:
        anchor = AnchorSpec("vacate-corner", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        local_map._cells.update(
            {
                (0, 0): FREE,
                (1, 0): FREE,
                (1, 1): FREE,
                (1, 2): FREE,
            }
        )
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = GotoController()
        navigation.status = "active"
        planner = Mock()
        planner.is_segment_passable.side_effect = (
            lambda source, target, **_: not (
                source == (0.5, 0.5) and target == (1.5, 1.5)
            )
        )
        navigation._planner = planner

        with patch.object(
            navigation,
            "_advance_coordination_path",
            return_value=[(0, 0), (1, 0), (2, 0)],
        ):
            path = navigation.coordination_vacate_path(
                pose,
                local_map,
                required_clearance_m=2.0,
            )

        self.assertEqual(
            path,
            ((1.5, 0.5), (1.5, 1.5), (1.5, 2.5)),
        )

    def test_implicit_vacate_path_requires_clearance_from_route_axis(
        self,
    ) -> None:
        anchor = AnchorSpec("vacate-route-axis", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        local_map._cells.update({(gx, 0): FREE for gx in range(5)})
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = GotoController()
        navigation.status = "active"
        navigation._planner = Mock(is_segment_passable=Mock(return_value=True))

        with patch.object(
            navigation,
            "_advance_coordination_path",
            return_value=[(0, 0), (1, 0), (2, 0)],
        ):
            path = navigation.coordination_vacate_path(
                pose,
                local_map,
                required_clearance_m=1.0,
            )

        self.assertEqual(path, ())

    def test_curved_route_window_requires_clearance_from_every_cell(self) -> None:
        route = tuple((gx, 13) for gx in range(37, 22, -1)) + tuple(
            (23, gy) for gy in range(14, 18)
        )
        self.assertEqual(len(route), 19)
        resolution_m = 0.5

        def center(cell: tuple[int, int]) -> tuple[float, float]:
            return tuple((coordinate + 0.5) * resolution_m for coordinate in cell)

        blocked = center((21, 16))
        old_points = tuple(center(cell) for cell in (route[0], (23, 33), route[-1]))
        self.assertGreater(
            min(
                _point_segment_distance(blocked, start, end)
                for start, end in zip(old_points, old_points[1:])
            ),
            0.9,
        )
        self.assertLessEqual(
            _point_route_distance(blocked, route, resolution_m),
            0.9,
        )
        self.assertGreater(
            _point_route_distance(center((20, 19)), route, resolution_m),
            0.9,
        )

        anchor = AnchorSpec("curved-route-window", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor, resolution_m=resolution_m)
        local_map._cells.update(
            {(21, 16): FREE, (20, 17): FREE, (20, 18): FREE, (20, 19): FREE}
        )
        pose = PoseEstimate(
            anchor.anchor_id,
            *center((21, 16)),
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = GotoController()
        navigation.status = "reached"
        navigation._planner = Mock(is_segment_passable=Mock(return_value=True))

        path = navigation.coordination_vacate_path(
            pose,
            local_map,
            0.9,
            clearance_at_m=lambda point: _point_route_distance(
                point,
                route,
                resolution_m,
            ),
            allow_reached=True,
        )

        self.assertTrue(path)
        self.assertLess(
            _point_route_distance(center((21, 16)), route, resolution_m),
            0.9,
        )
        self.assertGreaterEqual(
            _point_route_distance(path[-1], route, resolution_m),
            0.9,
        )

    def test_vacate_path_does_not_cross_unknown_or_static_cells(self) -> None:
        anchor = AnchorSpec("vacate-blocked", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        for state in (UNKNOWN, OCCUPIED):
            with self.subTest(state=state):
                local_map = ObservedGrid(anchor)
                local_map._cells[(0, 0)] = FREE
                if state != UNKNOWN:
                    local_map._cells.update(
                        {
                            (gx, gy): state
                            for gx in range(-1, 2)
                            for gy in range(-1, 2)
                            if (gx, gy) != (0, 0)
                        }
                    )
                navigation = GotoController()
                navigation.status = "active"
                navigation._planner = Mock(
                    is_segment_passable=Mock(return_value=True)
                )
                with patch.object(
                    navigation,
                    "_advance_coordination_path",
                    return_value=[(0, 0), (1, 0), (2, 0)],
                ):
                    self.assertEqual(
                        navigation.coordination_vacate_path(
                            pose,
                            local_map,
                            required_clearance_m=1.0,
                        ),
                        (),
                    )

    def test_vacate_path_search_is_bounded_to_four_metres(self) -> None:
        anchor = AnchorSpec("vacate-bounded", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        local_map._cells.update(
            {
                (gx, gy): FREE
                for gx in range(-6, 7)
                for gy in range(-6, 7)
            }
        )
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = GotoController()
        navigation.status = "active"
        planner = Mock(is_segment_passable=Mock(return_value=True))
        navigation._planner = planner
        with patch.object(
            navigation,
            "_advance_coordination_path",
            return_value=[(0, 0), (1, 0), (2, 0)],
        ):
            path = navigation.coordination_vacate_path(
                pose,
                local_map,
                required_clearance_m=4.5,
            )

        self.assertEqual(path, ())
        self.assertTrue(
            all(
                math.dist((0.5, 0.5), call.args[1]) <= 4.0 + 1e-12
                for call in planner.is_segment_passable.call_args_list
            )
        )

    def test_explicit_vacate_path_reaches_a_safe_bay_beyond_local_detours(
        self,
    ) -> None:
        anchor = AnchorSpec("explicit-safe-bay", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        local_map._cells.update(
            {
                (gx, gy): FREE
                for gx in range(9)
                for gy in range(2, 5)
            }
            | {
                (gx, gy): FREE
                for gx in range(2, 5)
                for gy in range(2, 8)
            }
        )
        pose = PoseEstimate(
            anchor.anchor_id,
            3.5,
            6.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = GotoController()
        navigation.start(
            pose.x_m,
            pose.y_m,
            local_map=local_map,
            pose=pose,
            vehicle_radius_m=0.5,
        )
        navigation.status = "reached"
        required_clearance_m = 2.0
        approach_line_m = ((3.5, 0.5), (3.5, 6.5))

        detours = navigation.coordination_detours(
            pose,
            local_map,
            allow_reached=True,
        )
        self.assertTrue(detours)
        self.assertTrue(
            all(
                abs(detour[0] - approach_line_m[0][0])
                < required_clearance_m
                for detour in detours
            )
        )

        path = navigation.coordination_vacate_path(
            pose,
            local_map,
            required_clearance_m,
            clearance_at_m=lambda point: _point_segment_distance(
                point,
                *approach_line_m,
            ),
            allow_reached=True,
        )

        self.assertGreater(len(path), 1)
        self.assertGreaterEqual(
            abs(path[-1][0] - approach_line_m[0][0]),
            required_clearance_m,
        )
        self.assertEqual(
            navigation.coordination_vacate_path(
                pose,
                local_map,
                required_clearance_m,
                clearance_at_m=lambda point: _point_segment_distance(
                    point,
                    *approach_line_m,
                ),
                allow_reached=True,
            ),
            path,
        )

    def test_corridor_matching_is_symmetric_and_local_to_overlapping_axis(self) -> None:
        eastbound = CorridorDescriptor((5, 5), (15, 5))
        westbound_partial = CorridorDescriptor((14, 5), (6, 5))
        separate_same_axis = CorridorDescriptor((20, 5), (28, 5))
        separate_parallel = CorridorDescriptor((5, 15), (15, 15))

        self.assertTrue(
            corridor_descriptors_conflict(eastbound, westbound_partial)
        )
        self.assertTrue(
            corridor_descriptors_conflict(westbound_partial, eastbound)
        )
        self.assertFalse(
            corridor_descriptors_conflict(eastbound, separate_same_axis)
        )
        self.assertFalse(
            corridor_descriptors_conflict(eastbound, separate_parallel)
        )

    def test_front_corridor_waiter_is_nearest_then_vehicle_id(self) -> None:
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                CorridorDescriptor((10, 0), (5, 0)),
                (12, 0),
                (14, 0),
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                CorridorDescriptor((5, 0), (10, 0)),
                (3, 0),
                (1, 0),
            ),
        )
        for owner_corridor, waiter_corridor, front_cell, rear_cell in cases:
            with self.subTest(owner_corridor=owner_corridor):
                owner = intent(
                    "vehicle_1",
                    current=owner_corridor.entry_cell,
                    target=owner_corridor.exit_cell,
                    wait_ticks=0,
                    reserved=True,
                    corridor=owner_corridor,
                )
                front = intent(
                    "vehicle_3",
                    current=front_cell,
                    target=waiter_corridor.entry_cell,
                    wait_ticks=0,
                    owner="vehicle_1",
                    corridor=waiter_corridor,
                )
                rear = intent(
                    "vehicle_2",
                    current=rear_cell,
                    target=front_cell,
                    wait_ticks=0,
                    owner="vehicle_1",
                    corridor=waiter_corridor,
                )
                self.assertIs(
                    _front_corridor_waiter(owner, (rear, front)),
                    front,
                )
                tied = intent(
                    "vehicle_2",
                    current=front_cell,
                    target=waiter_corridor.entry_cell,
                    wait_ticks=0,
                    owner="vehicle_1",
                    corridor=waiter_corridor,
                )
                self.assertIs(
                    _front_corridor_waiter(owner, (front, tied)),
                    tied,
                )

    def test_only_front_waiter_stages_in_each_direction(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                CorridorDescriptor((10, 0), (5, 0)),
                11.5,
                13.5,
                (5, 0),
                (6, 0),
                5.5,
                (9.5, 0.5),
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                CorridorDescriptor((5, 0), (10, 0)),
                4.5,
                2.5,
                (10, 0),
                (9, 0),
                10.5,
                (5.5, 0.5),
            ),
        )
        for (
            owner_corridor,
            waiter_corridor,
            front_x_m,
            rear_x_m,
            owner_current,
            owner_target,
            owner_x_m,
            motion_target,
        ) in cases:
            with self.subTest(owner_corridor=owner_corridor):
                owner = intent(
                    "vehicle_1",
                    current=owner_current,
                    target=owner_target,
                    wait_ticks=0,
                    reserved=True,
                    corridor=owner_corridor,
                )
                front = intent(
                    "vehicle_2",
                    current=(math.floor(front_x_m), 0),
                    target=waiter_corridor.entry_cell,
                    wait_ticks=1,
                    owner="vehicle_1",
                    corridor=waiter_corridor,
                )
                rear = intent(
                    "vehicle_3",
                    current=(math.floor(rear_x_m), 0),
                    target=(math.floor(front_x_m), 0),
                    wait_ticks=1,
                    owner="vehicle_1",
                    corridor=waiter_corridor,
                )

                def coordinate(
                    vehicle_id: str,
                    x_m: float,
                    other: PeerMotionIntent,
                ) -> tuple[tuple[float, float], RobotController]:
                    navigation = Mock(
                        status="active",
                        motion_target=motion_target,
                        coordination_corridor=Mock(return_value=None),
                    )
                    navigation.coordination_detours.return_value = (
                        (x_m, 2.5),
                    )
                    controller = RobotController(navigation)
                    controller._corridor = waiter_corridor
                    pose = PoseEstimate(
                        anchor.anchor_id,
                        x_m,
                        0.5,
                        0.0,
                        (0.0, 0.0, 0.0),
                        "nominal",
                        1.0,
                        1,
                    )
                    result = controller._coordinate_desired(
                        (0.5, 0.0),
                        vehicle=Vehicle(x_m, 0.5, now=1.0),
                        vehicle_id=vehicle_id,
                        anchor=anchor,
                        pose=pose,
                        local_map=local_map,
                        now=1.0,
                        peer_states=(
                            peer_state("vehicle_1", owner_x_m, 0.5, 0.0),
                        ),
                        peer_motion_intents=(owner, other),
                    )
                    return result, controller

                front_result, front_controller = coordinate(
                    "vehicle_2",
                    front_x_m,
                    rear,
                )
                rear_result, rear_controller = coordinate(
                    "vehicle_3",
                    rear_x_m,
                    front,
                )
                self.assertNotEqual(front_result, (0.0, 0.0))
                self.assertEqual(
                    front_controller._corridor_rejoin_target_m,
                    (front_x_m, 0.5),
                )
                self.assertEqual(rear_result, (0.0, 0.0))
                self.assertIsNone(rear_controller._corridor_rejoin_target_m)

    def test_owner_entry_gate_waits_for_front_stage_or_peer_expiry(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                CorridorDescriptor((10, 0), (5, 0)),
                4.5,
                3.5,
                (5.5, 0.5),
                (11, 0),
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                CorridorDescriptor((5, 0), (10, 0)),
                11.5,
                12.5,
                (10.5, 0.5),
                (4, 0),
            ),
        )
        for (
            owner_corridor,
            waiter_corridor,
            gate_x_m,
            approach_x_m,
            motion_target,
            waiter_current,
        ) in cases:
            with self.subTest(owner_corridor=owner_corridor):
                acknowledgement = intent(
                    "vehicle_2",
                    current=waiter_current,
                    target=(waiter_current[0], 2),
                    wait_ticks=1,
                    owner="vehicle_1",
                    corridor=waiter_corridor,
                )

                def controller_at(x_m: float) -> tuple[
                    RobotController,
                    Vehicle,
                    PoseEstimate,
                ]:
                    navigation = Mock(
                        status="active",
                        motion_target=motion_target,
                        coordination_corridor=Mock(return_value=None),
                    )
                    controller = RobotController(navigation)
                    controller._corridor = owner_corridor
                    controller._corridor_reserved = True
                    vehicle = Vehicle(x_m, 0.5, now=1.0)
                    pose = PoseEstimate(
                        anchor.anchor_id,
                        x_m,
                        0.5,
                        0.0,
                        (0.0, 0.0, 0.0),
                        "nominal",
                        1.0,
                        1,
                    )
                    return controller, vehicle, pose

                approach, approach_vehicle, approach_pose = controller_at(
                    approach_x_m
                )
                self.assertEqual(
                    approach._coordinate_desired(
                        (0.5, 0.0),
                        vehicle=approach_vehicle,
                        vehicle_id="vehicle_1",
                        anchor=anchor,
                        pose=approach_pose,
                        local_map=local_map,
                        now=1.0,
                        peer_states=(
                            peer_state("vehicle_2", waiter_current[0] + 0.5, 0.5, 0.0),
                        ),
                        peer_motion_intents=(acknowledgement,),
                    ),
                    (0.5, 0.0),
                )

                controller, vehicle, pose = controller_at(gate_x_m)
                blocked = controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=vehicle,
                    vehicle_id="vehicle_1",
                    anchor=anchor,
                    pose=pose,
                    local_map=local_map,
                    now=1.0,
                    peer_states=(
                        peer_state("vehicle_2", waiter_current[0] + 0.5, 0.5, 0.0),
                    ),
                    peer_motion_intents=(acknowledgement,),
                )
                self.assertEqual(blocked, (0.0, 0.0))
                self.assertTrue(controller._corridor_admission_confirmed)

                staged, _, staged_pose = controller_at(gate_x_m)
                staged_result = staged._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=Vehicle(gate_x_m, 0.5, now=1.0),
                    vehicle_id="vehicle_1",
                    anchor=anchor,
                    pose=staged_pose,
                    local_map=local_map,
                    now=1.0,
                    peer_states=(
                        peer_state("vehicle_2", waiter_current[0] + 0.5, 1.8, 0.0),
                    ),
                    peer_motion_intents=(acknowledgement,),
                )
                self.assertEqual(staged_result, (0.5, 0.0))

                expired = controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=vehicle,
                    vehicle_id="vehicle_1",
                    anchor=anchor,
                    pose=pose,
                    local_map=local_map,
                    now=1.4,
                    peer_states=(),
                    peer_motion_intents=(),
                )
                self.assertEqual(expired, (0.5, 0.0))

    def test_partial_corridor_release_extends_monotonically_past_peer_entry(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            motion_target=(6.5, 0.5),
            coordination_corridor=Mock(return_value=None),
        )
        controller = RobotController(navigation)
        controller._corridor = CorridorDescriptor((5, 0), (10, 0))
        peer = intent(
            "vehicle_2",
            current=(16, 0),
            target=(15, 0),
            wait_ticks=0,
            corridor=CorridorDescriptor((15, 0), (8, 0)),
        )

        controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=Vehicle(4.5, 0.5, now=0.0),
            vehicle_id="vehicle_1",
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                4.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                1,
            ),
            local_map=local_map,
            now=1.0,
            peer_states=(),
            peer_motion_intents=(peer,),
        )
        merged = controller.motion_intent[4]
        self.assertEqual(merged, CorridorDescriptor((5, 0), (15, 0)))
        self.assertTrue(controller.motion_intent[3])

        controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=Vehicle(11.5, 0.5, now=1.0),
            vehicle_id="vehicle_1",
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                11.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                2.0,
                2,
            ),
            local_map=local_map,
            now=2.0,
            peer_states=(),
            peer_motion_intents=(),
        )
        self.assertEqual(
            controller.motion_intent[4],
            CorridorDescriptor((5, 0), (15, 0)),
        )

    def test_corridor_release_waits_for_body_and_clearance_past_far_face(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        radius_m = 0.5
        margin_m = radius_m + AUTOMATIC_MINIMUM_CLEARANCE_M
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                11.0 + margin_m,
                11.0 + margin_m + 1e-6,
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                5.0 - margin_m,
                5.0 - margin_m - 1e-6,
            ),
        )

        for corridor, held_x_m, released_x_m in cases:
            with self.subTest(corridor=corridor):
                navigation = Mock(
                    motion_target=None,
                    coordination_corridor=Mock(return_value=None),
                )
                controller = RobotController(navigation)
                controller._corridor = corridor
                vehicle = Vehicle(held_x_m, 0.5, now=0.0)

                controller._coordinate_desired(
                    (0.0, 0.0),
                    vehicle=vehicle,
                    vehicle_id="vehicle_1",
                    anchor=anchor,
                    pose=PoseEstimate(
                        anchor.anchor_id,
                        held_x_m,
                        0.5,
                        0.0,
                        (0.0, 0.0, 0.0),
                        "nominal",
                        1.0,
                        1,
                    ),
                    local_map=local_map,
                    now=1.0,
                    peer_states=(),
                    peer_motion_intents=(),
                )
                self.assertEqual(controller.motion_intent[4], corridor)

                controller._coordinate_desired(
                    (0.0, 0.0),
                    vehicle=Vehicle(released_x_m, 0.5, now=1.0),
                    vehicle_id="vehicle_1",
                    anchor=anchor,
                    pose=PoseEstimate(
                        anchor.anchor_id,
                        released_x_m,
                        0.5,
                        0.0,
                        (0.0, 0.0, 0.0),
                        "nominal",
                        2.0,
                        2,
                    ),
                    local_map=local_map,
                    now=2.0,
                    peer_states=(),
                    peer_motion_intents=(),
                )
                self.assertIsNone(controller.motion_intent[4])

    def test_only_confirmed_owner_ignores_matching_yielded_peer_for_planning(
        self,
    ) -> None:
        controller = RobotController()
        controller._corridor = CorridorDescriptor((5, 0), (15, 0))
        controller._corridor_reserved = True
        controller._corridor_admission_confirmed = True
        controller._intent_priority_owner_id = "vehicle_1"
        matching = intent(
            "vehicle_2",
            current=(16, 0),
            target=(15, 0),
            wait_ticks=1,
            owner="vehicle_1",
            corridor=CorridorDescriptor((14, 0), (6, 0)),
        )
        same_tick_unconfirmed = intent(
            "vehicle_3",
            current=(16, 0),
            target=(15, 0),
            wait_ticks=0,
            owner="vehicle_3",
            reserved=True,
            corridor=CorridorDescriptor((14, 0), (6, 0)),
        )
        nonmatching = intent(
            "vehicle_4",
            current=(16, 4),
            target=(15, 4),
            wait_ticks=1,
            owner="vehicle_1",
            corridor=CorridorDescriptor((14, 4), (6, 4)),
        )

        self.assertEqual(
            controller.planning_ignored_peer_ids(
                "vehicle_1",
                (matching, same_tick_unconfirmed, nonmatching),
            ),
            frozenset(("vehicle_2",)),
        )
        controller._corridor_reserved = False
        self.assertEqual(
            controller.planning_ignored_peer_ids("vehicle_1", (matching,)),
            frozenset(),
        )

    def test_corridor_entry_requires_fresh_intents_from_expected_peers(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)

        def controller_at_entry() -> tuple[
            RobotController,
            Vehicle,
            PoseEstimate,
        ]:
            navigation = Mock(
                status="active",
                motion_target=(5.5, 0.5),
                coordination_corridor=Mock(return_value=None),
            )
            controller = RobotController(navigation)
            controller._corridor = CorridorDescriptor((5, 0), (10, 0))
            return (
                controller,
                Vehicle(4.5, 0.5, now=1.0),
                PoseEstimate(
                    anchor.anchor_id,
                    4.5,
                    0.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.0,
                    1,
                ),
            )

        def coordinate(
            controller: RobotController,
            vehicle: Vehicle,
            pose: PoseEstimate,
            *,
            ready: bool | None,
            expected: tuple[str, ...],
            intents: tuple[PeerMotionIntent, ...] = (),
        ) -> tuple[float, float]:
            return controller._coordinate_desired(
                (0.5, 0.0),
                vehicle=vehicle,
                vehicle_id="vehicle_1",
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                now=1.0,
                peer_states=(),
                peer_motion_intents=intents,
                coordination_ready=ready,
                expected_peer_vehicle_ids=expected,
            )

        standalone, standalone_vehicle, standalone_pose = controller_at_entry()
        standalone_results = [
            coordinate(
                standalone,
                standalone_vehicle,
                standalone_pose,
                ready=None,
                expected=(),
            )
            for _ in range(3)
        ]
        self.assertEqual(standalone_results[-1], (0.5, 0.0))
        self.assertTrue(standalone._corridor_admission_confirmed)

        disconnected, disconnected_vehicle, disconnected_pose = (
            controller_at_entry()
        )
        for _ in range(5):
            self.assertEqual(
                coordinate(
                    disconnected,
                    disconnected_vehicle,
                    disconnected_pose,
                    ready=False,
                    expected=(),
                ),
                (0.0, 0.0),
            )
        self.assertFalse(disconnected._corridor_admission_confirmed)

        partitioned, partitioned_vehicle, partitioned_pose = controller_at_entry()
        for _ in range(5):
            self.assertEqual(
                coordinate(
                    partitioned,
                    partitioned_vehicle,
                    partitioned_pose,
                    ready=True,
                    expected=("vehicle_2",),
                ),
                (0.0, 0.0),
            )
        self.assertFalse(partitioned._corridor_admission_confirmed)

        unrelated = intent(
            "vehicle_2",
            current=(20, 5),
            target=(21, 5),
            wait_ticks=0,
        )
        self.assertEqual(
            coordinate(
                partitioned,
                partitioned_vehicle,
                partitioned_pose,
                ready=True,
                expected=("vehicle_2",),
                intents=(unrelated,),
            ),
            (0.5, 0.0),
        )
        self.assertTrue(partitioned._corridor_admission_confirmed)

    def test_expired_corridor_winner_cannot_be_replaced_during_partition(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            status="active",
            motion_target=(5.5, 0.5),
            coordination_corridor=Mock(return_value=None),
        )
        navigation.coordination_detours.return_value = ()
        controller = RobotController(navigation)
        controller._corridor = CorridorDescriptor((5, 0), (10, 0))
        vehicle = Vehicle(4.5, 0.5, now=1.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            4.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        winner = intent(
            "vehicle_1",
            current=(10, 0),
            target=(9, 0),
            wait_ticks=0,
            reserved=True,
            corridor=CorridorDescriptor((10, 0), (5, 0)),
        )

        controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=vehicle,
            vehicle_id="vehicle_2",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.0,
            peer_states=(),
            peer_motion_intents=(winner,),
            coordination_ready=True,
            expected_peer_vehicle_ids=("vehicle_1",),
        )
        self.assertEqual(controller._coordination_wait_owner_id, "vehicle_1")

        for tick in range(1, 6):
            result = controller._coordinate_desired(
                (0.5, 0.0),
                vehicle=vehicle,
                vehicle_id="vehicle_2",
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                now=1.0 + tick * 0.1,
                peer_states=(),
                peer_motion_intents=(),
                coordination_ready=True,
                expected_peer_vehicle_ids=("vehicle_1",),
            )
            self.assertEqual(result, (0.0, 0.0))
        self.assertFalse(controller._corridor_admission_confirmed)

    def test_partitioned_corridor_entrants_reconnect_to_one_owner(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        setup = {
            "vehicle_1": (
                4.5,
                (5.5, 0.5),
                CorridorDescriptor((5, 0), (10, 0)),
            ),
            "vehicle_2": (
                11.5,
                (10.5, 0.5),
                CorridorDescriptor((10, 0), (5, 0)),
            ),
        }
        controllers = {}
        for vehicle_id, (_, target_m, corridor) in setup.items():
            navigation = Mock(
                status="active",
                motion_target=target_m,
                coordination_corridor=Mock(return_value=None),
            )
            navigation.coordination_detours.return_value = ()
            controller = RobotController(navigation)
            controller._corridor = corridor
            controllers[vehicle_id] = controller

        def coordinate(
            vehicle_id: str,
            peer_intents: tuple[PeerMotionIntent, ...],
        ) -> tuple[float, float]:
            x_m, _, _ = setup[vehicle_id]
            return controllers[vehicle_id]._coordinate_desired(
                (0.5, 0.0),
                vehicle=Vehicle(x_m, 0.5, now=1.0),
                vehicle_id=vehicle_id,
                anchor=anchor,
                pose=PoseEstimate(
                    anchor.anchor_id,
                    x_m,
                    0.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.0,
                    1,
                ),
                local_map=local_map,
                now=1.0,
                peer_states=(),
                peer_motion_intents=peer_intents,
                coordination_ready=True,
                expected_peer_vehicle_ids=(
                    "vehicle_2" if vehicle_id == "vehicle_1" else "vehicle_1",
                ),
            )

        def published(vehicle_id: str) -> PeerMotionIntent:
            x_m, target_m, _ = setup[vehicle_id]
            _, wait_ticks, owner_id, reserved, corridor = (
                controllers[vehicle_id].motion_intent
            )
            return intent(
                vehicle_id,
                current=(math.floor(x_m), 0),
                target=(math.floor(target_m[0]), 0),
                wait_ticks=wait_ticks,
                owner=owner_id,
                reserved=reserved,
                corridor=corridor,
            )

        for _ in range(5):
            self.assertEqual(coordinate("vehicle_1", ()), (0.0, 0.0))
            self.assertEqual(coordinate("vehicle_2", ()), (0.0, 0.0))
        self.assertTrue(
            all(
                controller.snapshot()["coordination"]["state"] == "tentative"
                for controller in controllers.values()
            )
        )

        announced = {
            vehicle_id: published(vehicle_id)
            for vehicle_id in sorted(controllers)
        }
        for _ in range(2):
            for vehicle_id in sorted(controllers):
                peer_id = (
                    "vehicle_2" if vehicle_id == "vehicle_1" else "vehicle_1"
                )
                coordinate(vehicle_id, (announced[peer_id],))
            announced = {
                vehicle_id: published(vehicle_id)
                for vehicle_id in sorted(controllers)
            }

        self.assertEqual(
            [
                vehicle_id
                for vehicle_id, controller in sorted(controllers.items())
                if controller.snapshot()["coordination"]["state"] == "reserved"
            ],
            ["vehicle_1"],
        )
        self.assertEqual(
            controllers["vehicle_2"].snapshot()["coordination"],
            {
                "state": "waiting",
                "reason": "corridor_lease",
                "priority_owner_vehicle_id": "vehicle_1",
            },
        )

    def test_confirmed_corridor_owner_clears_partition_without_revocation(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            status="active",
            motion_target=(7.5, 0.5),
            coordination_corridor=Mock(return_value=None),
        )
        controller = RobotController(navigation)
        controller._corridor = CorridorDescriptor((5, 0), (10, 0))
        controller._corridor_reserved = True
        controller._corridor_admission_confirmed = True

        result = controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=Vehicle(6.5, 0.5, now=1.0),
            vehicle_id="vehicle_1",
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                6.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                1,
            ),
            local_map=local_map,
            now=1.0,
            peer_states=(),
            peer_motion_intents=(),
            coordination_ready=False,
            expected_peer_vehicle_ids=("vehicle_2",),
        )

        self.assertEqual(result, (0.5, 0.0))
        self.assertTrue(controller._corridor_admission_confirmed)

    def test_live_corridor_lease_defers_no_path_until_the_lease_expires(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            4.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            status="active",
            reason=None,
            detail=None,
            motion_target=(5.5, 0.5),
        )
        navigation.coordination_detours.return_value = ()
        navigation.snapshot.return_value = {"status": "active"}

        def block_no_path(**_: object) -> tuple[float, float]:
            navigation.status = "blocked"
            navigation.reason = "no_path"
            navigation.detail = "unreachable"
            return 0.0, 0.0

        navigation.update.side_effect = block_no_path
        controller = RobotController(navigation)
        controller.mode = OpMode.AUTO
        controller.auto_state = AutoState.ACTIVE
        inject_active_goto(controller, "corridor", 16.0, 0.5)
        controller._corridor = CorridorDescriptor((10, 1), (30, 1))
        controller._intent_priority_owner_id = "vehicle_1"
        vehicle = Vehicle(4.5, 0.5, now=1.0)
        grid = MapGrid.from_wall_set(40, 20, set())
        safety = LocalSafetyRuntime()
        owner = intent(
            "vehicle_1",
            current=(20, 1),
            target=(21, 1),
            wait_ticks=0,
            reserved=True,
            corridor=CorridorDescriptor((10, 1), (30, 1)),
        )

        controller.tick(
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=1.0,
            vehicle_id="vehicle_2",
            peer_motion_intents=(owner,),
        )

        navigation.update.assert_not_called()
        self.assertEqual(controller.auto_state, AutoState.ACTIVE)
        self.assertEqual(
            controller.snapshot()["coordination"],
            {
                "state": "waiting",
                "reason": "corridor_lease",
                "priority_owner_vehicle_id": "vehicle_1",
            },
        )

        # MapSync omits the intent after its bounded TTL.  Navigation resumes
        # on that tick and a real no_path result is terminal as usual.
        expired_at = 1.0 + MOTION_INTENT_TTL_S + 0.01
        vehicle.advance(grid, expired_at)
        controller.tick(
            vehicle=vehicle,
            grid=grid,
            safety=safety,
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=expired_at,
            vehicle_id="vehicle_2",
            peer_motion_intents=(),
        )
        navigation.update.assert_called_once()
        self.assertEqual(controller.auto_state, AutoState.BLOCKED)
        self.assertEqual(controller.navigation.reason, "no_path")
        self.assertEqual(controller.snapshot()["coordination"]["state"], "idle")

    def test_staged_waiter_rejoins_saved_pose_before_navigation_resumes(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            status="active",
            reason=None,
            detail=None,
            motion_target=(5.5, 0.5),
        )
        navigation.coordination_corridor.return_value = None
        navigation.coordination_detours.return_value = ((4.5, 2.5),)
        navigation.update.return_value = (0.5, 0.0)
        navigation.snapshot.return_value = {"status": "active"}
        controller = RobotController(navigation)
        controller._corridor = CorridorDescriptor((5, 0), (10, 0))
        waiting_pose = PoseEstimate(
            anchor.anchor_id,
            4.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        owner = intent(
            "vehicle_1",
            current=(10, 0),
            target=(9, 0),
            wait_ticks=0,
            reserved=True,
            corridor=CorridorDescriptor((10, 0), (5, 0)),
        )

        staged_desired = controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=Vehicle(4.5, 0.5, now=1.0),
            vehicle_id="vehicle_2",
            anchor=anchor,
            pose=waiting_pose,
            local_map=local_map,
            now=1.0,
            peer_states=(),
            peer_motion_intents=(owner,),
        )

        self.assertNotEqual(staged_desired, (0.0, 0.0))
        self.assertEqual(controller._corridor_rejoin_target_m, (4.5, 0.5))

        controller.mode = OpMode.AUTO
        controller.auto_state = AutoState.ACTIVE
        inject_active_goto(controller, "corridor", 16.0, 0.5)
        staged_pose = PoseEstimate(
            anchor.anchor_id,
            4.5,
            2.5,
            -math.pi / 2,
            (0.0, 0.0, 0.0),
            "nominal",
            2.0,
            2,
        )
        acknowledgement = intent(
            "vehicle_1",
            current=(10, 0),
            target=(9, 0),
            wait_ticks=0,
            owner="vehicle_2",
            corridor=CorridorDescriptor((10, 0), (5, 0)),
        )
        staged_vehicle = Vehicle(4.5, 2.5, -math.pi / 2, now=2.0)
        grid = MapGrid.from_wall_set(20, 20, set())
        safety = LocalSafetyRuntime()

        controller.tick(
            vehicle=staged_vehicle,
            grid=grid,
            safety=safety,
            anchor=anchor,
            pose=staged_pose,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=2.0,
            vehicle_id="vehicle_2",
            peer_motion_intents=(acknowledgement,),
        )

        navigation.update.assert_not_called()
        self.assertEqual(staged_vehicle.target_velocities(), (0.5, 0.0))
        self.assertEqual(controller.auto_state, AutoState.ACTIVE)

        controller._coordinate_desired(
            (0.0, 0.0),
            vehicle=Vehicle(4.5, 0.5, now=3.0),
            vehicle_id="vehicle_2",
            anchor=anchor,
            pose=waiting_pose,
            local_map=local_map,
            now=3.0,
            peer_states=(),
            peer_motion_intents=(acknowledgement,),
        )
        self.assertIsNone(controller._corridor_rejoin_target_m)

        controller.tick(
            vehicle=Vehicle(4.5, 0.5, now=3.1),
            grid=grid,
            safety=safety,
            anchor=anchor,
            pose=waiting_pose,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=3.1,
            vehicle_id="vehicle_2",
            peer_motion_intents=(acknowledgement,),
        )
        navigation.update.assert_called_once()

    def test_front_waiter_stages_before_owner_reaches_apron(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        margin_m = 0.5 + AUTOMATIC_MINIMUM_CLEARANCE_M
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                CorridorDescriptor((10, 0), (5, 0)),
                5.0 - margin_m,
                (5.5, 0.5),
                (10, 0),
                (9, 0),
                10.5,
                (7, 0),
                (6, 0),
                7.5,
                -0.5,
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                CorridorDescriptor((5, 0), (10, 0)),
                11.0 + margin_m,
                (10.5, 0.5),
                (5, 0),
                (6, 0),
                5.5,
                (8, 0),
                (9, 0),
                8.5,
                0.5,
            ),
        )

        for (
            own_corridor,
            owner_corridor,
            wait_x_m,
            motion_target,
            far_current,
            far_target,
            far_x_m,
            near_current,
            near_target,
            near_x_m,
            near_vx_mps,
        ) in cases:
            with self.subTest(owner_corridor=owner_corridor):
                navigation = Mock(
                    status="active",
                    motion_target=motion_target,
                    coordination_corridor=Mock(return_value=None),
                )
                navigation.coordination_detours.return_value = (
                    (wait_x_m, 2.5),
                )
                controller = RobotController(navigation)
                controller._corridor = own_corridor
                vehicle = Vehicle(wait_x_m, 0.5, now=1.0)
                waiting_pose = PoseEstimate(
                    anchor.anchor_id,
                    wait_x_m,
                    0.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.0,
                    1,
                )
                far_owner = intent(
                    "vehicle_1",
                    current=far_current,
                    target=far_target,
                    wait_ticks=0,
                    reserved=True,
                    corridor=owner_corridor,
                )

                far_result = controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=vehicle,
                    vehicle_id="vehicle_2",
                    anchor=anchor,
                    pose=waiting_pose,
                    local_map=local_map,
                    now=1.0,
                    peer_states=(
                        peer_state("vehicle_1", far_x_m, 0.5, 0.0),
                    ),
                    peer_motion_intents=(far_owner,),
                )

                self.assertNotEqual(far_result, (0.0, 0.0))
                self.assertEqual(
                    controller._corridor_rejoin_target_m,
                    (wait_x_m, 0.5),
                )

                approaching_owner = intent(
                    "vehicle_1",
                    current=near_current,
                    target=near_target,
                    wait_ticks=0,
                    reserved=True,
                    corridor=owner_corridor,
                )
                stage_result = controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=vehicle,
                    vehicle_id="vehicle_2",
                    anchor=anchor,
                    pose=waiting_pose,
                    local_map=local_map,
                    now=1.1,
                    peer_states=(
                        peer_state(
                            "vehicle_1",
                            near_x_m,
                            0.5,
                            near_vx_mps,
                        ),
                    ),
                    peer_motion_intents=(approaching_owner,),
                )

                self.assertNotEqual(stage_result, (0.0, 0.0))
                self.assertEqual(
                    controller._corridor_rejoin_target_m,
                    (wait_x_m, 0.5),
                )

                staged_pose = PoseEstimate(
                    anchor.anchor_id,
                    wait_x_m,
                    2.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.2,
                    2,
                )
                staged_result = controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=Vehicle(wait_x_m, 2.5, now=1.2),
                    vehicle_id="vehicle_2",
                    anchor=anchor,
                    pose=staged_pose,
                    local_map=local_map,
                    now=1.2,
                    peer_states=(
                        peer_state(
                            "vehicle_1",
                            near_x_m,
                            0.5,
                            near_vx_mps,
                        ),
                    ),
                    peer_motion_intents=(approaching_owner,),
                )

                self.assertEqual(staged_result, (0.0, 0.0))
                self.assertEqual(
                    controller._corridor_rejoin_target_m,
                    (wait_x_m, 0.5),
                )

    def test_staging_clearance_is_continuous_in_both_directions(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        required_m = 1.0 + AUTOMATIC_MINIMUM_CLEARANCE_M
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                CorridorDescriptor((10, 0), (5, 0)),
                4.2,
                (5.5, 0.5),
                (10, 0),
                (9, 0),
                10.5,
                -0.1,
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                CorridorDescriptor((5, 0), (10, 0)),
                11.8,
                (10.5, 0.5),
                (5, 0),
                (6, 0),
                5.5,
                0.1,
            ),
        )

        for (
            own_corridor,
            owner_corridor,
            wait_x_m,
            motion_target,
            owner_current,
            owner_target,
            owner_x_m,
            owner_vx_mps,
        ) in cases:
            owner = intent(
                "vehicle_1",
                current=owner_current,
                target=owner_target,
                wait_ticks=0,
                reserved=True,
                corridor=owner_corridor,
            )

            def coordinate(y_m: float) -> tuple[
                tuple[float, float], RobotController
            ]:
                navigation = Mock(
                    status="active",
                    motion_target=motion_target,
                    coordination_corridor=Mock(return_value=None),
                )
                navigation.coordination_detours.return_value = (
                    (wait_x_m, 3.5),
                )
                controller = RobotController(navigation)
                controller._corridor = own_corridor
                controller._corridor_rejoin_target_m = (wait_x_m, 0.5)
                result = controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=Vehicle(wait_x_m, y_m, now=1.0),
                    vehicle_id="vehicle_2",
                    anchor=anchor,
                    pose=PoseEstimate(
                        anchor.anchor_id,
                        wait_x_m,
                        y_m,
                        0.0,
                        (0.0, 0.0, 0.0),
                        "nominal",
                        1.0,
                        1,
                    ),
                    local_map=local_map,
                    now=1.0,
                    peer_states=(
                        peer_state(
                            "vehicle_1",
                            owner_x_m,
                            0.5,
                            owner_vx_mps,
                        ),
                    ),
                    peer_motion_intents=(owner,),
                )
                return result, controller

            with self.subTest(owner_corridor=owner_corridor):
                below, below_controller = coordinate(
                    0.5 + required_m - 1e-6
                )
                self.assertNotEqual(below, (0.0, 0.0))
                self.assertEqual(
                    below_controller._corridor_rejoin_target_m,
                    (wait_x_m, 0.5),
                )
                at_boundary, boundary_controller = coordinate(
                    0.5 + required_m
                )
                self.assertEqual(at_boundary, (0.0, 0.0))
                self.assertEqual(
                    boundary_controller._corridor_rejoin_target_m,
                    (wait_x_m, 0.5),
                )

    def test_curved_owner_sweep_stages_apron_outside_waiter(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        margin_m = 0.5 + AUTOMATIC_MINIMUM_CLEARANCE_M
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                CorridorDescriptor((10, 0), (5, 0)),
                5.0 - margin_m,
                (5.5, 0.5),
                (7, 0),
                (6, 0),
                7.5,
                -0.75,
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                CorridorDescriptor((5, 0), (10, 0)),
                11.0 + margin_m,
                (10.5, 0.5),
                (8, 0),
                (9, 0),
                8.5,
                0.75,
            ),
        )

        for (
            own_corridor,
            owner_corridor,
            wait_x_m,
            motion_target,
            owner_current,
            owner_target,
            owner_x_m,
            owner_vx_mps,
        ) in cases:
            with self.subTest(owner_corridor=owner_corridor):
                navigation = Mock(
                    status="active",
                    motion_target=motion_target,
                    coordination_corridor=Mock(return_value=None),
                )
                navigation.coordination_detours.return_value = (
                    (wait_x_m, 3.5),
                )
                controller = RobotController(navigation)
                controller._corridor = own_corridor
                waiting_pose = PoseEstimate(
                    anchor.anchor_id,
                    wait_x_m,
                    1.9,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.0,
                    1,
                )
                owner = intent(
                    "vehicle_1",
                    current=owner_current,
                    target=owner_target,
                    wait_ticks=0,
                    reserved=True,
                    corridor=owner_corridor,
                )

                result = controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=Vehicle(wait_x_m, 1.9, now=1.0),
                    vehicle_id="vehicle_2",
                    anchor=anchor,
                    pose=waiting_pose,
                    local_map=local_map,
                    now=1.0,
                    peer_states=(
                        peer_state(
                            "vehicle_1",
                            owner_x_m,
                            0.5,
                            owner_vx_mps,
                            0.25,
                        ),
                    ),
                    peer_motion_intents=(owner,),
                )

                self.assertNotEqual(result, (0.0, 0.0))
                self.assertEqual(
                    controller._corridor_rejoin_target_m,
                    (wait_x_m, 1.9),
                )

    def test_dynamic_peer_can_temporarily_block_corridor_rejoin(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            4.5,
            2.5,
            -math.pi / 2,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = Mock(
            status="active",
            reason=None,
            detail=None,
            motion_target=(5.5, 0.5),
        )
        navigation.snapshot.return_value = {"status": "active"}
        controller = RobotController(navigation)
        controller.mode = OpMode.AUTO
        controller.auto_state = AutoState.ACTIVE
        inject_active_goto(controller, "corridor", 16.0, 0.5)
        controller._corridor = CorridorDescriptor((5, 0), (10, 0))
        controller._corridor_reserved = True
        controller._corridor_admission_confirmed = True
        controller._intent_priority_owner_id = "vehicle_2"
        controller._corridor_rejoin_target_m = (4.5, 0.5)
        safety = Mock()
        safety.evaluate.return_value = SafetyDecision(
            0.0,
            0.0,
            "stopped",
            "safety_obstacle",
        )
        acknowledgement = intent(
            "vehicle_1",
            current=(10, 0),
            target=(9, 0),
            wait_ticks=0,
            owner="vehicle_2",
            corridor=CorridorDescriptor((10, 0), (5, 0)),
        )

        controller.tick(
            vehicle=Vehicle(4.5, 2.5, -math.pi / 2, now=1.0),
            grid=MapGrid.from_wall_set(20, 20, set()),
            safety=safety,
            anchor=anchor,
            pose=pose,
            local_map=ObservedGrid(anchor),
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=1.0,
            safety_scan_points=(LaserPoint(0.0, 0.5, 1.0, dynamic=True),),
            vehicle_id="vehicle_2",
            peer_motion_intents=(acknowledgement,),
        )

        navigation.update.assert_not_called()
        navigation.block.assert_not_called()
        self.assertEqual(controller.auto_state, AutoState.ACTIVE)
        self.assertEqual(controller._corridor_rejoin_target_m, (4.5, 0.5))

    def test_rejoin_bypasses_gate_only_for_segment_outside_entry(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        cases = (
            (
                CorridorDescriptor((5, 0), (10, 0)),
                CorridorDescriptor((10, 0), (5, 0)),
                4.5,
                5.5,
            ),
            (
                CorridorDescriptor((10, 0), (5, 0)),
                CorridorDescriptor((5, 0), (10, 0)),
                11.5,
                10.5,
            ),
        )
        for own_corridor, peer_corridor, outside_x_m, inside_x_m in cases:
            with self.subTest(own_corridor=own_corridor):
                acknowledgement = intent(
                    "vehicle_1",
                    current=peer_corridor.entry_cell,
                    target=peer_corridor.exit_cell,
                    wait_ticks=0,
                    owner="vehicle_2",
                    corridor=peer_corridor,
                )

                def coordinate(rejoin_x_m: float) -> tuple[
                    tuple[float, float],
                    RobotController,
                ]:
                    navigation = Mock(
                        status="active",
                        motion_target=(inside_x_m, 0.5),
                        coordination_corridor=Mock(return_value=None),
                    )
                    controller = RobotController(navigation)
                    controller._corridor = own_corridor
                    controller._corridor_reserved = True
                    controller._corridor_admission_confirmed = True
                    controller._corridor_rejoin_target_m = (
                        rejoin_x_m,
                        0.5,
                    )
                    pose = PoseEstimate(
                        anchor.anchor_id,
                        outside_x_m,
                        2.5,
                        -math.pi / 2,
                        (0.0, 0.0, 0.0),
                        "nominal",
                        1.0,
                        1,
                    )
                    result = controller._coordinate_desired(
                        (0.0, 0.0),
                        vehicle=Vehicle(
                            outside_x_m,
                            2.5,
                            -math.pi / 2,
                            now=1.0,
                        ),
                        vehicle_id="vehicle_2",
                        anchor=anchor,
                        pose=pose,
                        local_map=local_map,
                        now=1.0,
                        peer_states=(),
                        peer_motion_intents=(acknowledgement,),
                    )
                    return result, controller

                outside, outside_controller = coordinate(outside_x_m)
                crossing, crossing_controller = coordinate(inside_x_m)
                self.assertEqual(outside, (0.5, 0.0))
                self.assertEqual(
                    outside_controller._corridor_rejoin_target_m,
                    (outside_x_m, 0.5),
                )
                self.assertEqual(crossing, (0.0, 0.0))
                self.assertEqual(
                    crossing_controller._corridor_rejoin_target_m,
                    (inside_x_m, 0.5),
                )

    def test_occupied_rejoin_segment_falls_back_to_navigation(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            status="active",
            motion_target=(5.5, 0.5),
            coordination_corridor=Mock(return_value=None),
        )
        controller = RobotController(navigation)
        controller._corridor = CorridorDescriptor((5, 0), (10, 0))
        controller._corridor_reserved = True
        controller._corridor_admission_confirmed = True
        controller._corridor_rejoin_target_m = (4.5, 0.5)
        pose = PoseEstimate(
            anchor.anchor_id,
            4.5,
            2.5,
            -math.pi / 2,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )

        result = controller._coordinate_desired(
            (0.0, 0.0),
            vehicle=Vehicle(4.5, 2.5, -math.pi / 2, now=1.0),
            vehicle_id="vehicle_2",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.0,
            peer_states=(peer_state("vehicle_1", 4.5, 1.0, 0.0),),
            peer_motion_intents=(),
        )

        self.assertEqual(result, (0.0, 0.0))
        self.assertIsNone(controller._corridor_rejoin_target_m)
        self.assertIsNone(controller._intent_target_m)

    def test_clearing_coordination_discards_saved_rejoin_pose(self) -> None:
        controller = RobotController()
        controller._corridor_rejoin_target_m = (4.5, 0.5)
        controller._vacate_request = VacateRequest(
            "vehicle_b", (2, 0), ((0, 0), (2, 0), (4, 0))
        )

        controller._clear_yield()

        self.assertIsNone(controller._corridor_rejoin_target_m)
        self.assertIsNone(controller.vacate_request)

    def test_mode_takeover_and_cancel_do_not_publish_an_old_vacate_request(
        self,
    ) -> None:
        for action in ("manual", "cancel"):
            with self.subTest(action=action):
                controller, _ = parked_idle_request_once(
                    route=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
                )
                self.assertIsNotNone(controller.vacate_request)
                vehicle = Vehicle(0.5, 0.5, radius=0.5, now=0.0)

                if action == "manual":
                    controller._handle_mode(
                        ModeCommand(1, ModeAction.SWITCH_TO_MANUAL),
                        vehicle,
                        1.1,
                    )
                else:
                    controller._handle_auto(
                        AutoCommand(1, AutoAction.CANCEL_ALL),
                        vehicle,
                        1.1,
                    )

                self.assertIsNone(controller.vacate_request)

    def test_transient_no_path_waits_when_own_map_still_has_a_route(self) -> None:
        persistent = ObservedGrid(AnchorSpec("spawn", 0.0, 0.0, 0.0))
        planning_map = _TransientPlanningGrid(persistent)
        planning_map.update({(2, 0)}, None)
        planner = DStarLitePlanner(
            planning_map,
            vehicle_radius_m=0.0,
            bounds_margin_m=0.0,
        )
        self.assertIsNone(planner.plan((0, 0), (4, 0)))
        self.assertEqual(planner.last_failure, "search_exhausted")
        navigation = GotoController()
        navigation.status = "active"
        navigation.requested_goal = (4.5, 0.5)
        navigation.goal = navigation.requested_goal
        navigation.goal_mode = "exact"
        navigation._vehicle_radius_m = 0.0
        navigation._planner = planner
        pose = PoseEstimate(
            persistent.anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )

        navigation._block_no_path("dynamic_obstacle")
        outcome = navigation.classify_no_path_against_persistent(
            pose,
            persistent,
            planning_map,
            transient_active=True,
            attributed_peer_active=True,
        )

        self.assertEqual(outcome, "transient")
        self.assertEqual(navigation.status, "active")
        self.assertTrue(navigation._waiting_for_peer_replan)

        cleared = planning_map.update(set(), None)
        navigation.update(
            pose=pose,
            local_map=planning_map,
            max_linear_mps=0.5,
            max_angular_rps=math.pi / 2,
            advance_result=SafetyAdvanceResult(),
            map_delta=cleared,
        )

        self.assertEqual(navigation.status, "active")
        self.assertFalse(navigation._waiting_for_peer_replan)
        self.assertNotEqual(navigation.motion_target, None)

    def test_anonymous_dynamic_no_path_gets_one_restart_grace(self) -> None:
        def setup() -> tuple[
            ObservedGrid,
            _TransientPlanningGrid,
            GotoController,
            PoseEstimate,
        ]:
            persistent = ObservedGrid(
                AnchorSpec("spawn", 0.0, 0.0, 0.0)
            )
            planning_map = _TransientPlanningGrid(persistent)
            planning_map.update({(2, 0)}, None)
            planner = DStarLitePlanner(
                planning_map,
                vehicle_radius_m=0.0,
                bounds_margin_m=0.0,
            )
            self.assertIsNone(planner.plan((0, 0), (4, 0)))
            navigation = GotoController()
            navigation.status = "active"
            navigation.requested_goal = (4.5, 0.5)
            navigation.goal = navigation.requested_goal
            navigation.goal_mode = "exact"
            navigation._vehicle_radius_m = 0.0
            navigation._planner = planner
            pose = PoseEstimate(
                persistent.anchor.anchor_id,
                0.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                1,
            )
            navigation._block_no_path("anonymous_dynamic")
            return persistent, planning_map, navigation, pose

        persistent, planning_map, navigation, pose = setup()
        outcome = navigation.classify_no_path_against_persistent(
            pose,
            persistent,
            planning_map,
            transient_active=True,
            attributed_peer_active=False,
        )

        self.assertEqual(outcome, "transient")
        self.assertTrue(navigation._anonymous_replan_grace_used)
        self.assertFalse(navigation._waiting_for_peer_replan)
        cleared = planning_map.update(set(), None)
        navigation.update(
            pose=pose,
            local_map=planning_map,
            max_linear_mps=0.5,
            max_angular_rps=math.pi / 2,
            advance_result=SafetyAdvanceResult(),
            map_delta=cleared,
        )
        self.assertEqual(navigation.status, "active")
        self.assertIsNotNone(navigation.motion_target)

        persistent, planning_map, navigation, pose = setup()
        navigation.classify_no_path_against_persistent(
            pose,
            persistent,
            planning_map,
            transient_active=True,
            attributed_peer_active=False,
        )
        for _ in range(20):
            navigation.update(
                pose=pose,
                local_map=planning_map,
                max_linear_mps=0.5,
                max_angular_rps=math.pi / 2,
                advance_result=SafetyAdvanceResult(),
                map_delta=None,
            )
            if navigation.status == "blocked":
                break
        self.assertEqual(navigation.status, "blocked")
        outcome = navigation.classify_no_path_against_persistent(
            pose,
            persistent,
            planning_map,
            transient_active=True,
            attributed_peer_active=False,
        )
        self.assertEqual(outcome, "static")
        self.assertEqual(navigation.status, "blocked")

    def test_static_no_path_remains_terminal_with_a_transient_grid(self) -> None:
        persistent = ObservedGrid(AnchorSpec("spawn", 0.0, 0.0, 0.0))
        wall = tuple((2, gy) for gy in range(-10, 11))
        persistent._cells.update((cell, OCCUPIED) for cell in wall)
        persistent.revision += 1
        planning_map = _TransientPlanningGrid(persistent)
        planning_map.update(
            set(),
            LocalMapDelta(
                tuple(MapCellUpdate(gx, gy, OCCUPIED) for gx, gy in wall)
            ),
        )
        planner = DStarLitePlanner(
            planning_map,
            vehicle_radius_m=0.0,
            bounds_margin_m=0.0,
        )
        self.assertIsNone(planner.plan((0, 0), (4, 0)))
        navigation = GotoController()
        navigation.status = "active"
        navigation.requested_goal = (4.5, 0.5)
        navigation.goal = navigation.requested_goal
        navigation.goal_mode = "exact"
        navigation._vehicle_radius_m = 0.0
        navigation._planner = planner
        pose = PoseEstimate(
            persistent.anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )

        navigation._block_no_path("static_obstacle")
        outcome = navigation.classify_no_path_against_persistent(
            pose,
            persistent,
            planning_map,
            transient_active=False,
            attributed_peer_active=False,
        )

        self.assertEqual(outcome, "static")
        self.assertEqual(navigation.status, "blocked")
        self.assertFalse(navigation._waiting_for_peer_replan)

    def test_static_no_path_is_not_hidden_without_a_live_corridor_lease(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = Mock(
            status="active",
            reason=None,
            detail=None,
            motion_target=None,
        )
        navigation.snapshot.return_value = {"status": "active"}

        def block_no_path(**_: object) -> tuple[float, float]:
            navigation.status = "blocked"
            navigation.reason = "no_path"
            navigation.detail = "static_unreachable"
            return 0.0, 0.0

        navigation.update.side_effect = block_no_path
        controller = RobotController(navigation)
        controller.mode = OpMode.AUTO
        controller.auto_state = AutoState.ACTIVE
        inject_active_goto(controller, "blocked", 10.0, 0.5)
        vehicle = Vehicle(0.5, 0.5, now=0.0)

        controller.tick(
            vehicle=vehicle,
            grid=MapGrid.from_wall_set(20, 20, set()),
            safety=LocalSafetyRuntime(),
            anchor=anchor,
            pose=pose,
            local_map=ObservedGrid(anchor),
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=1.0,
            vehicle_id="vehicle_2",
            peer_motion_intents=(),
        )

        navigation.update.assert_called_once()
        self.assertEqual(controller.auto_state, AutoState.BLOCKED)
        self.assertEqual(controller.navigation.reason, "no_path")

    def test_older_waiter_precedes_lexically_smaller_newcomer(self) -> None:
        older = intent(
            "vehicle_4",
            current=(2, 1),
            target=(1, 1),
            wait_ticks=8,
        )
        newcomer = intent(
            "vehicle_1",
            current=(0, 1),
            target=(1, 1),
            wait_ticks=0,
        )

        self.assertTrue(motion_intent_precedes(older, newcomer))
        self.assertFalse(motion_intent_precedes(newcomer, older))

        older_task = intent(
            "vehicle_4",
            current=(2, 1),
            target=(1, 1),
            wait_ticks=0,
            task_age_ticks=12,
        )
        self.assertFalse(motion_intent_precedes(older_task, newcomer))
        self.assertTrue(motion_intent_precedes(newcomer, older_task))

    def test_vertex_edge_swap_and_one_hop_priority_inheritance(self) -> None:
        requester = intent(
            "vehicle_1",
            current=(0, 0),
            target=(1, 0),
            wait_ticks=6,
            reserved=True,
        )
        blocker = intent(
            "vehicle_2",
            current=(1, 0),
            target=(0, 0),
            wait_ticks=1,
        )
        same_vertex = intent(
            "vehicle_3",
            current=(1, 1),
            target=(1, 0),
            wait_ticks=0,
        )

        self.assertTrue(_motion_intents_conflict(requester, blocker))
        self.assertTrue(_motion_intents_conflict(requester, same_vertex))
        inherited = inherit_motion_priority(blocker, (requester,))
        self.assertEqual(inherited.priority_owner_id, "vehicle_1")
        self.assertEqual(inherited.wait_ticks, 6)
        self.assertFalse(inherited.reserved)

    def test_priority_inheritance_follows_a_bounded_request_chain(self) -> None:
        upstream = intent(
            "vehicle_4",
            current=(0, 0),
            target=(1, 0),
            wait_ticks=9,
            reserved=True,
        )
        middle = intent(
            "vehicle_2",
            current=(1, 0),
            target=(2, 0),
            wait_ticks=2,
        )
        blocker = intent(
            "vehicle_1",
            current=(2, 0),
            target=(3, 0),
            wait_ticks=0,
        )

        inherited = inherit_motion_priority(
            blocker,
            (middle,),
            (upstream, middle),
        )

        self.assertEqual(inherited.priority_owner_id, "vehicle_4")
        self.assertEqual(inherited.wait_ticks, 9)

    def test_delayed_task_age_still_elects_one_corridor_owner(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        first_corridor = CorridorDescriptor((9, 5), (13, 5))
        second_corridor = CorridorDescriptor((13, 5), (9, 5))

        def coordinate(
            vehicle_id: str,
            x_m: float,
            target_x_m: float,
            own_corridor: CorridorDescriptor,
            peer_id: str,
            peer_x_m: float,
            peer_target: tuple[int, int],
            peer_corridor: CorridorDescriptor,
            own_task_age_ticks: int,
            peer_task_age_ticks: int,
            own_wait_ticks: int,
            peer_wait_ticks: int,
        ) -> RobotController:
            navigation = Mock(
                motion_target=(target_x_m, 5.5),
                coordination_detours=Mock(return_value=()),
            )
            controller = RobotController(navigation)
            controller._corridor = own_corridor
            controller._corridor_reserved = True
            controller._intent_reserved = True
            controller._active_task_age_ticks = own_task_age_ticks
            controller._reservation_wait_ticks = own_wait_ticks
            pose = PoseEstimate(
                anchor.anchor_id,
                x_m,
                5.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                0.3,
                1,
            )
            controller._coordinate_desired(
                (0.5, 0.0),
                vehicle=Vehicle(x_m, 5.5, now=0.0),
                vehicle_id=vehicle_id,
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                now=0.3,
                peer_states=(peer_state(peer_id, peer_x_m, 5.5, 0.0),),
                peer_motion_intents=(
                    intent(
                        peer_id,
                        current=(math.floor(peer_x_m), 5),
                        target=peer_target,
                        wait_ticks=peer_wait_ticks,
                        reserved=True,
                        corridor=peer_corridor,
                        task_age_ticks=peer_task_age_ticks,
                    ),
                ),
                coordination_ready=True,
                expected_peer_vehicle_ids=(peer_id,),
            )
            return controller

        for own_age, peer_age, own_wait, peer_wait in (
            (3, 1, 0, 0),
            (5, 3, 0, 0),
            (40, 1, 7, 2),
        ):
            with self.subTest(
                own_age=own_age,
                peer_age=peer_age,
                own_wait=own_wait,
                peer_wait=peer_wait,
            ):
                first = coordinate(
                    "vehicle_1",
                    7.5,
                    8.5,
                    first_corridor,
                    "vehicle_2",
                    15.5,
                    (14, 5),
                    second_corridor,
                    own_age,
                    peer_age,
                    own_wait,
                    peer_wait,
                )
                second = coordinate(
                    "vehicle_2",
                    15.5,
                    14.5,
                    second_corridor,
                    "vehicle_1",
                    7.5,
                    (8, 5),
                    first_corridor,
                    own_age,
                    peer_age,
                    own_wait,
                    peer_wait,
                )

                self.assertEqual(
                    (first._corridor_reserved, second._corridor_reserved),
                    (True, False),
                )
                self.assertEqual(
                    second.snapshot()["coordination"][
                        "priority_owner_vehicle_id"
                    ],
                    "vehicle_1",
                )

    def test_temporal_quorum_fails_closed_when_peer_evidence_is_incomplete(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_1", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        local_map = ObservedGrid(anchor)
        remote_state = peer_state("vehicle_2", 10.5, 0.5, 0.0)
        remote_intent = intent(
            "vehicle_2",
            current=(10, 0),
            target=(11, 0),
            wait_ticks=0,
        )

        def coordinate(
            states: tuple[PeerVehicleState, ...],
            intents: tuple[PeerMotionIntent, ...],
            ready: bool,
        ) -> tuple[tuple[float, float], RobotController]:
            navigation = Mock(
                motion_target=(1.5, 0.5),
                coordination_path_cells=Mock(
                    return_value=((0, 0), (1, 0))
                ),
                coordination_detours=Mock(return_value=()),
            )
            controller = RobotController(navigation)
            result = controller._coordinate_desired(
                (0.5, 0.0),
                vehicle=Vehicle(0.5, 0.5, now=0.0),
                vehicle_id="vehicle_1",
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                now=1.0,
                peer_states=states,
                peer_motion_intents=intents,
                coordination_ready=ready,
                expected_peer_vehicle_ids=("vehicle_2",),
            )
            return result, controller

        for states, intents, ready in (
            ((remote_state,), (), True),
            ((), (remote_intent,), True),
            ((remote_state,), (remote_intent,), False),
        ):
            with self.subTest(states=bool(states), intents=bool(intents), ready=ready):
                result, controller = coordinate(states, intents, ready)
                self.assertEqual(result, (0.0, 0.0))
                self.assertEqual(
                    controller.snapshot()["coordination"]["reason"],
                    "reservation_sync",
                )

        result, controller = coordinate(
            (remote_state,),
            (remote_intent,),
            True,
        )
        self.assertEqual(result, (0.5, 0.0))
        self.assertIsNone(controller.snapshot()["coordination"]["reason"])

    def test_temporal_quorum_fails_closed_during_peer_restart_topic_skew(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_1", 0.0, 0.0, 0.0)
        peer_anchor = AnchorSpec("spawn_2", 0.0, 0.0, 0.0)
        receiver = MapSyncState(
            "session_1",
            "vehicle_1",
            anchor,
            1.0,
            clock=lambda: 1.0,
            state_generation=1,
        )
        old_peer = MapSyncState(
            "session_1",
            "vehicle_2",
            peer_anchor,
            1.0,
            state_generation=1,
        )
        restarted_peer = MapSyncState(
            "session_1",
            "vehicle_2",
            peer_anchor,
            1.0,
            state_generation=2,
        )
        receiver.configure_network(
            "peer_1",
            {"vehicle_2": ("peer_2", peer_anchor)},
        )
        for source in (old_peer, restarted_peer):
            source.configure_network(
                "peer_2",
                {"vehicle_1": ("peer_1", anchor)},
            )
        peer_pose = PoseEstimate(
            peer_anchor.anchor_id,
            10.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        old_peer.record_motion_intent(
            peer_pose,
            target_m=(11.5, 0.5),
            wait_ticks=0,
            priority_owner_id="vehicle_2",
            reserved=True,
            timestamp_s=1.0,
        )
        restarted_peer.record_vehicle_state(
            peer_pose,
            radius_m=0.5,
            linear_mps=0.0,
            omega_rps=0.0,
        )
        self.assertTrue(
            receiver.receive_transport(
                "peer_2",
                "vehicle_2",
                old_peer.prepare_motion_intent(),
            )
        )
        self.assertTrue(
            receiver.receive_transport(
                "peer_2",
                "vehicle_2",
                restarted_peer.prepare_peer_state(),
            )
        )
        reverse_receiver = MapSyncState(
            "session_1",
            "vehicle_1",
            anchor,
            1.0,
            clock=lambda: 1.0,
            state_generation=1,
        )
        reverse_receiver.configure_network(
            "peer_1",
            {"vehicle_2": ("peer_2", peer_anchor)},
        )
        old_peer.record_vehicle_state(
            peer_pose,
            radius_m=0.5,
            linear_mps=0.0,
            omega_rps=0.0,
        )
        restarted_peer.record_motion_intent(
            peer_pose,
            target_m=(11.5, 0.5),
            wait_ticks=0,
            priority_owner_id="vehicle_2",
            reserved=True,
            timestamp_s=1.0,
        )
        self.assertTrue(
            reverse_receiver.receive_transport(
                "peer_2",
                "vehicle_2",
                old_peer.prepare_peer_state(),
            )
        )
        self.assertTrue(
            reverse_receiver.receive_transport(
                "peer_2",
                "vehicle_2",
                restarted_peer.prepare_motion_intent(),
            )
        )

        local_pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        for skewed_receiver in (receiver, reverse_receiver):
            controller = RobotController(
                Mock(
                    motion_target=(1.5, 0.5),
                    coordination_path_cells=Mock(return_value=((0, 0), (1, 0))),
                    coordination_detours=Mock(return_value=()),
                )
            )
            desired = controller._coordinate_desired(
                (0.5, 0.0),
                vehicle=Vehicle(0.5, 0.5, now=0.0),
                vehicle_id="vehicle_1",
                anchor=anchor,
                pose=local_pose,
                local_map=ObservedGrid(anchor),
                now=1.0,
                peer_states=skewed_receiver.peer_vehicle_states(),
                peer_motion_intents=skewed_receiver.peer_motion_intents(),
                coordination_ready=True,
                expected_peer_vehicle_ids=("vehicle_2",),
            )

            self.assertEqual(desired, (0.0, 0.0))
            self.assertEqual(
                controller.snapshot()["coordination"]["reason"],
                "reservation_sync",
            )

    def test_temporal_commit_survives_cell_progress_then_expires(self) -> None:
        anchor = AnchorSpec("spawn_1", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            motion_target=(1.5, 0.5),
            coordination_path_cells=Mock(
                return_value=((0, 0), (1, 0), (2, 0))
            ),
            coordination_detours=Mock(return_value=()),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(0.5, 0.5, radius=0.1, now=0.0)
        remote_state = PeerVehicleState(
            "vehicle_2",
            1,
            1,
            1,
            1.0,
            10.5,
            10.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            0.0,
            0.0,
            0.0,
            0.1,
        )

        def coordinate(
            *,
            now: float,
            x_m: float,
            peer_wait_ticks: int,
        ) -> tuple[float, float]:
            current_gx = math.floor(x_m)
            navigation.motion_target = (current_gx + 1.5, 0.5)
            navigation.coordination_path_cells.return_value = tuple(
                (gx, 0) for gx in range(current_gx, current_gx + 3)
            )
            pose = PoseEstimate(
                anchor.anchor_id,
                x_m,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                now,
                1,
            )
            remote_intent = intent(
                "vehicle_2",
                current=(current_gx + 1, 1),
                target=(current_gx + 1, 0),
                wait_ticks=peer_wait_ticks,
            )
            return controller._coordinate_desired(
                (0.5, 0.0),
                vehicle=vehicle,
                vehicle_id="vehicle_1",
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                now=now,
                peer_states=(remote_state,),
                peer_motion_intents=(remote_intent,),
                coordination_ready=True,
                expected_peer_vehicle_ids=("vehicle_2",),
            )

        self.assertEqual(
            coordinate(now=1.0, x_m=0.5, peer_wait_ticks=0),
            (0.5, 0.0),
        )
        self.assertTrue(controller.motion_intent[3])

        self.assertEqual(
            coordinate(now=1.4, x_m=1.5, peer_wait_ticks=99),
            (0.5, 0.0),
        )
        self.assertTrue(controller.motion_intent[3])

        self.assertEqual(
            coordinate(now=1.9, x_m=1.5, peer_wait_ticks=99),
            (0.0, 0.0),
        )
        self.assertFalse(controller.motion_intent[3])
        self.assertEqual(
            controller.snapshot()["coordination"]["reason"],
            "space_time_reservation",
        )

    def test_inherited_detour_is_rescheduled_before_intent_publish(self) -> None:
        anchor = AnchorSpec("spawn_1", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            motion_target=None,
            coordination_corridor=Mock(return_value=None),
            coordination_path_cells=Mock(return_value=((0, 0),)),
            coordination_detours=Mock(return_value=((1.5, 0.5),)),
        )
        controller = RobotController(navigation)
        requester = intent(
            "vehicle_2",
            current=(2, 0),
            target=(0, 0),
            wait_ticks=5,
            reserved=True,
        )
        controller._coordinate_desired(
            (0.0, 0.0),
            vehicle=Vehicle(0.5, 0.5, radius=0.1, now=0.0),
            vehicle_id="vehicle_1",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.0,
            peer_states=(peer_state("vehicle_2", 2.5, 0.5, 0.0),),
            peer_motion_intents=(requester,),
            coordination_ready=True,
            expected_peer_vehicle_ids=("vehicle_2",),
        )

        target_m, wait_ticks, owner_id, reserved, corridor = (
            controller.motion_intent
        )
        (
            _,
            task_sequence,
            task_age_ticks,
            trajectory,
            committed_until_offset_s,
            goal_hold,
            safety_time_margin_s,
        ) = controller.temporal_motion_intent
        state = MapSyncState("session", "vehicle_1", anchor, 1.0)
        state.configure_network(
            "peer_1",
            {"vehicle_2": ("peer_2", anchor)},
        )
        state.record_motion_intent(
            pose,
            target_m=target_m,
            wait_ticks=wait_ticks,
            priority_owner_id=owner_id or "vehicle_1",
            reserved=reserved,
            corridor=corridor,
            timestamp_s=1.0,
            task_sequence=task_sequence,
            task_age_ticks=task_age_ticks,
            trajectory=trajectory,
            committed_until_offset_s=committed_until_offset_s,
            goal_hold=goal_hold,
            safety_time_margin_s=safety_time_margin_s,
        )

        published = state.prepare_motion_intent()
        self.assertIsNotNone(published)
        assert published is not None
        self.assertEqual(published["target_cell"], {"gx": 1, "gy": 0})
        self.assertEqual(
            published["trajectory"][1]["cell"],
            published["target_cell"],
        )

    def test_peer_vacate_request_survives_a_target_change_before_entry(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_2", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            motion_target=None,
            coordination_path_cells=Mock(return_value=((1, 0),)),
            coordination_detours=Mock(return_value=((1.5, 1.5),)),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(1.5, 0.5, radius=0.2, now=0.0)

        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_2",
            now=1.0,
            position_m=(1.5, 0.5),
            peer_states=(peer_state("vehicle_1", 0.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_1",
                    current=(0, 0),
                    target=(2, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
            ),
            desired=(0.0, 0.0),
        )
        self.assertEqual(controller._peer_vacate_request_cell, (1, 0))

        navigation.motion_target = (2.5, 1.5)
        held = coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_2",
            now=1.1,
            position_m=(1.5, 1.5),
            peer_states=(peer_state("vehicle_1", 0.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_1",
                    current=(0, 0),
                    target=(2, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
            ),
        )

        self.assertEqual(held, (0.0, 0.0))

        entered = coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_2",
            now=1.2,
            position_m=(1.5, 1.5),
            peer_states=(peer_state("vehicle_1", 1.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_1",
                    current=(1, 0),
                    target=(2, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
            ),
        )
        self.assertEqual(entered, (0.0, 0.0))

        navigation.coordination_path_cells.return_value = ((1, 1), (2, 1))
        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_2",
            now=1.3,
            position_m=(1.5, 1.5),
            peer_states=(peer_state("vehicle_1", 3.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_1",
                    current=(3, 0),
                    target=(4, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
            ),
        )

        self.assertIsNone(controller._peer_vacate_request_cell)

    def test_peer_vacate_request_can_be_explicitly_retracted_before_entry(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_2", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            motion_target=None,
            coordination_path_cells=Mock(return_value=((1, 0),)),
            coordination_detours=Mock(return_value=((1.5, 1.5),)),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(1.5, 0.5, radius=0.1, now=0.0)

        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_2",
            now=1.0,
            position_m=(1.5, 0.5),
            peer_states=(peer_state("vehicle_1", 0.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_1",
                    current=(0, 0),
                    target=(1, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
            ),
            desired=(0.0, 0.0),
        )

        navigation.motion_target = (2.5, 1.5)
        navigation.coordination_path_cells.return_value = ((1, 1), (2, 1))
        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_2",
            now=1.1,
            position_m=(1.5, 1.5),
            peer_states=(peer_state("vehicle_1", 0.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_1",
                    current=(0, 0),
                    target=None,
                    wait_ticks=5,
                    reserved=False,
                ),
            ),
        )

        self.assertIsNone(controller._peer_vacate_request_cell)

    def test_implicit_vacate_rebinds_to_a_same_owner_explicit_request(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_c", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            transient_peer_blocked=True,
            motion_target=None,
            coordination_path_cells=Mock(
                return_value=((1, 0), (2, 0), (3, 0))
            ),
            coordination_vacate_path=Mock(return_value=((1.5, 1.5),)),
            coordination_detours=Mock(return_value=((1.5, 2.5),)),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(1.5, 0.5, radius=0.1, now=0.0)

        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_c",
            now=1.0,
            position_m=(1.5, 0.5),
            peer_states=(peer_state("vehicle_a", 0.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_a",
                    current=(0, 0),
                    target=(3, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
            ),
            desired=(0.0, 0.0),
        )
        self.assertEqual(
            controller._yielding_for,
            "vehicle_a",
        )
        self.assertIsNone(controller._peer_vacate_request_cell)

        navigation.transient_peer_blocked = False
        navigation.motion_target = (2.5, 1.5)
        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_c",
            now=1.1,
            position_m=(1.5, 1.5),
            peer_states=(
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                peer_state("vehicle_b", 2.5, 1.5, 0.0),
            ),
            peer_intents=(
                intent(
                    "vehicle_a",
                    current=(0, 0),
                    target=(3, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
                intent(
                    "vehicle_b",
                    current=(2, 1),
                    target=(1, 1),
                    wait_ticks=7,
                    owner="vehicle_a",
                ),
            ),
        )

        self.assertEqual(
            controller._yielding_for,
            "vehicle_b",
        )
        self.assertEqual(controller.motion_intent[2], "vehicle_a")
        self.assertEqual(controller._peer_vacate_request_cell, (1, 1))
        self.assertFalse(controller._peer_vacate_request_entered)

    def test_inactive_peer_vacate_preserves_generic_yield_clear_debounce(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_b", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            transient_peer_blocked=False,
            motion_target=(1.5, 0.5),
            coordination_corridor=Mock(return_value=None),
        )
        controller = RobotController(navigation)
        controller._yielding_for = "vehicle_a"
        controller._yield_requires_intent = True
        controller._yield_clear_ticks = 1
        controller._schedule_temporal_motion = Mock(
            side_effect=lambda desired, **kwargs: (
                desired,
                kwargs["own"].target_cell,
                kwargs["own"],
                False,
            )
        )
        vehicle = Vehicle(0.5, 0.5, radius=0.5, now=0.0)
        remote_state = peer_state("vehicle_a", 8.5, 8.5, 0.0)
        remote_intent = intent(
            "vehicle_a",
            current=(8, 8),
            target=(9, 8),
            wait_ticks=0,
        )

        held = coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_b",
            now=1.0,
            position_m=(0.5, 0.5),
            peer_states=(remote_state,),
            peer_intents=(remote_intent,),
        )

        self.assertEqual(held, (0.0, 0.0))
        self.assertEqual(controller._yield_clear_ticks, 2)
        released = coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_b",
            now=1.1,
            position_m=(0.5, 0.5),
            peer_states=(remote_state,),
            peer_intents=(remote_intent,),
        )

        self.assertEqual(released, (0.5, 0.0))
        self.assertIsNone(controller._yielding_for)

    def test_implicit_vacate_does_not_yield_forever_to_a_parked_idle_peer(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_c", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)

        def blocker(
            *,
            target: tuple[int, int] | None = None,
            reserved: bool = False,
            goal_hold: bool = False,
            task_sequence: int = (1 << 64) - 1,
        ) -> PeerMotionIntent:
            return PeerMotionIntent(
                "vehicle_b",
                1,
                1,
                1.0,
                0.35,
                (4, 0),
                target,
                5,
                "vehicle_b",
                reserved,
                task_sequence=task_sequence,
                trajectory=(TimedCell((4, 0), 0.0, 4.0),),
                goal_hold=goal_hold,
            )

        def seeded_controller() -> tuple[RobotController, Mock, Vehicle]:
            navigation = Mock(
                transient_peer_blocked=True,
                motion_target=None,
                coordination_path_cells=Mock(
                    return_value=((2, 0), (3, 0), (4, 0))
                ),
                coordination_vacate_path=Mock(return_value=((0.5, 3.5),)),
                coordination_detours=Mock(return_value=()),
            )
            controller = RobotController(navigation)
            controller._schedule_temporal_motion = Mock(
                side_effect=lambda desired, **kwargs: (
                    desired,
                    kwargs["own"].target_cell,
                    kwargs["own"],
                    False,
                )
            )
            vehicle = Vehicle(0.5, 0.5, radius=0.5, now=0.0)
            controller._transient_peer_vacate(
                own=intent(
                    "vehicle_c",
                    current=(0, 0),
                    target=(1, 0),
                    wait_ticks=0,
                ),
                vehicle=vehicle,
                anchor=anchor,
                pose=PoseEstimate(
                    anchor.anchor_id,
                    0.5,
                    0.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.0,
                    10,
                ),
                local_map=local_map,
                peers={"vehicle_b": peer_state("vehicle_b", 4.5, 0.5, 0.0)},
                peer_motion_intents=(
                    blocker(target=(3, 0), task_sequence=1),
                ),
                coordination_map=None,
                now=1.0,
            )
            navigation.transient_peer_blocked = False
            return controller, navigation, vehicle

        cases = (
            ("parked idle", blocker(), True),
            ("moving target", blocker(target=(3, 0)), False),
            ("reserved", blocker(reserved=True), False),
            ("goal hold", blocker(goal_hold=True), False),
        )
        for label, peer_intent, should_release in cases:
            with self.subTest(label=label):
                controller, _, vehicle = seeded_controller()
                for tick in range(3):
                    held = controller._transient_peer_vacate(
                        own=intent(
                            "vehicle_c",
                            current=(0, 3),
                            target=(1, 3),
                            wait_ticks=0,
                        ),
                        vehicle=vehicle,
                        anchor=anchor,
                        pose=PoseEstimate(
                            anchor.anchor_id,
                            0.5,
                            3.5,
                            0.0,
                            (0.0, 0.0, 0.0),
                            "nominal",
                            1.1 + tick / 10,
                            11 + tick,
                        ),
                        local_map=local_map,
                        peers={
                            "vehicle_b": peer_state(
                                "vehicle_b", 4.5, 0.5, 0.0
                            )
                        },
                        peer_motion_intents=(peer_intent,),
                        coordination_map=None,
                        now=1.1 + tick / 10,
                    )
                    if tick < 2 or not should_release:
                        self.assertEqual(held, (0.0, 0.0))

                if should_release:
                    self.assertIsNone(held)
                    self.assertIsNone(controller._peer_vacate_origin_cell)
                else:
                    self.assertEqual(controller._peer_vacate_origin_cell, (0, 0))

    def test_higher_priority_transient_requester_asks_parked_blocker_to_vacate(
        self,
    ) -> None:
        controller, desired = parked_idle_request_once(
            route=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        )

        self.assertEqual(desired, (0.0, 0.0))
        self.assertEqual(
            controller.vacate_request,
            VacateRequest(
                "vehicle_b",
                (2, 0),
                ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
            ),
        )
        self.assertIsNone(controller.motion_intent[0])
        trajectory = controller.temporal_motion_intent[3]
        self.assertEqual(tuple(item.cell for item in trajectory), ((0, 0),))

    def test_corridor_owner_with_transient_no_path_requests_parked_blocker(
        self,
    ) -> None:
        navigation = Mock(
            transient_peer_blocked=True,
            motion_target=None,
            coordination_path_cells=Mock(
                return_value=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
            ),
        )
        controller = RobotController(navigation)
        inject_active_goto(controller, "goto-vehicle_a", 4.5, 0.5)
        controller.mode = OpMode.AUTO
        controller.auto_state = AutoState.ACTIVE
        controller._corridor = CorridorDescriptor((0, 0), (4, 0))
        controller._corridor_reserved = True
        controller._corridor_admission_confirmed = True

        desired = coordinate_parked_idle_blocker(
            controller,
            route=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        )

        self.assertEqual(desired, (0.0, 0.0))
        self.assertEqual(
            controller.vacate_request,
            VacateRequest(
                "vehicle_b",
                (2, 0),
                ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
            ),
        )

    def test_request_keeps_route_end_when_blocker_footprint_is_beside_route(
        self,
    ) -> None:
        controller, desired = parked_idle_request_once(
            route=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
            peer_current=(2, 1),
            peer_position_m=(2.5, 1.5),
        )

        self.assertEqual(desired, (0.0, 0.0))
        self.assertEqual(
            controller.vacate_request,
            VacateRequest(
                "vehicle_b",
                (2, 1),
                ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
            ),
        )

    def test_rotated_anchor_vacate_request_publishes_from_runtime_current_cell(
        self,
    ) -> None:
        anchor = AnchorSpec("rotated-requester", 0.0, 0.0, math.pi / 4)
        local_map = ObservedGrid(anchor)
        local_route = tuple((index, index) for index in range(64))
        global_route = _global_coordination_cells(anchor, local_route, 1.0)
        self.assertEqual(len(global_route), 64)
        blocker_cell = global_route[16]
        navigation = Mock(
            transient_peer_blocked=True,
            motion_target=None,
            coordination_corridor=Mock(return_value=None),
            coordination_path_cells=Mock(return_value=local_route),
        )
        controller = RobotController(navigation)
        inject_active_goto(controller, "rotated-goto", 4.0, 0.0)
        controller.mode = OpMode.AUTO
        controller.auto_state = AutoState.ACTIVE
        pose = PoseEstimate(
            anchor.anchor_id,
            0.1,
            0.9,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )

        controller._coordinate_desired(
            (0.0, 0.0),
            vehicle=Vehicle(0.1, 0.9, radius=0.5, now=0.0),
            vehicle_id="vehicle_a",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.0,
            peer_states=(
                peer_state(
                    "vehicle_b",
                    blocker_cell[0] + 0.5,
                    blocker_cell[1] + 0.5,
                    0.0,
                ),
            ),
            peer_motion_intents=(
                PeerMotionIntent(
                    "vehicle_b",
                    1,
                    1,
                    1.0,
                    MOTION_INTENT_TTL_S,
                    blocker_cell,
                    None,
                    0,
                    "vehicle_b",
                    task_sequence=(1 << 64) - 1,
                ),
            ),
            coordination_ready=True,
            expected_peer_vehicle_ids=("vehicle_b",),
        )
        request = controller.vacate_request
        assert request is not None
        self.assertEqual(len(request.route_cells), 64)
        self.assertIn(blocker_cell, request.route_cells)
        sync = MapSyncState(
            "session", "vehicle_a", anchor, 1.0, state_generation=1
        )
        sync.configure_network(
            "peer_a", {"vehicle_b": ("peer_b", anchor)}
        )

        sync.record_motion_intent(
            pose,
            target_m=None,
            wait_ticks=0,
            priority_owner_id="vehicle_a",
            reserved=False,
            timestamp_s=1.0,
            vacate_request=request,
        )
        payload = sync.prepare_motion_intent()
        assert payload is not None

        self.assertEqual(payload["current_cell"], {"gx": -1, "gy": 0})
        self.assertEqual(
            payload["vacate_request"]["route_cells"][0],
            payload["current_cell"],
        )

    def test_transient_vacate_request_requires_a_complete_route_and_priority(
        self,
    ) -> None:
        route = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
        cases = (
            ("route ends at blocker", {"route": route[:3]}),
            ("own map has no route", {"route": None}),
            (
                "requester has lower priority",
                {
                    "route": route,
                    "own_vehicle_id": "vehicle_b",
                    "peer_vehicle_id": "vehicle_a",
                },
            ),
        )
        for label, kwargs in cases:
            with self.subTest(label=label):
                controller, desired = parked_idle_request_once(**kwargs)
                self.assertEqual(desired, (0.0, 0.0))
                self.assertIsNone(controller.vacate_request)

    def test_active_vacate_request_retracts_when_peer_evidence_no_longer_matches(
        self,
    ) -> None:
        route = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
        cases = (
            (
                "peer moved off route",
                {"peer_current": (2, 3), "peer_position_m": (2.5, 3.5)},
            ),
            (
                "peer generation changed",
                {"peer_state_generation": 2, "peer_intent_generation": 1},
            ),
            (
                "peer pose and intent cell disagree",
                {"peer_current": (9, 9), "peer_position_m": (2.5, 0.5)},
            ),
        )
        for label, kwargs in cases:
            with self.subTest(label=label):
                controller, _ = parked_idle_request_once(route=route)
                self.assertIsNotNone(controller.vacate_request)
                desired = coordinate_parked_idle_blocker(
                    controller,
                    route=route,
                    **kwargs,
                )
                self.assertEqual(desired, (0.0, 0.0))
                self.assertIsNone(controller.vacate_request)

    def test_addressed_higher_priority_request_moves_auto_idle_blocker_through_safety(
        self,
    ) -> None:
        requester = vacate_intent()
        controller, vehicle, safety, _ = idle_vacate_tick(
            requester,
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
        )

        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))
        self.assertEqual(controller.motion_intent[0], (2.5, 1.5))
        self.assertEqual(
            tuple(item.cell for item in controller.temporal_motion_intent[3]),
            ((2, 0), (2, 1)),
        )
        self.assertTrue(controller.is_automatic_motion_active)
        self.assertEqual(safety.decision.state, "clear")

    def test_active_wall_clock_request_moves_simulation_clock_idle_blocker(
        self,
    ) -> None:
        requester = vacate_intent()
        active_state = replace(
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
            timestamp_s=1_800_000_000.0,
        )

        controller, vehicle, _, _ = idle_vacate_tick(
            requester,
            active_state,
            now=1.0,
        )

        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))
        self.assertTrue(controller.is_automatic_motion_active)

    def test_target_cell_responder_keeps_requester_vacate_request(
        self,
    ) -> None:
        controller, _ = parked_idle_request_once(
            route=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        )
        request = controller.vacate_request
        assert request is not None
        controller.navigation.transient_peer_blocked = False
        responder = replace(
            intent(
                "vehicle_b",
                current=(2, 0),
                target=(2, 1),
                wait_ticks=0,
                owner="vehicle_a",
            ),
            task_sequence=(1 << 64) - 1,
            received_at_s=1.1,
        )
        active_state = replace(
            peer_state("vehicle_b", 2.5, 1.5, 0.0),
            timestamp_s=1_800_000_000.0,
        )

        coordinate_once(
            controller,
            Vehicle(0.5, 0.5, radius=0.5, now=0.0),
            AnchorSpec("spawn_vehicle_a", 0.0, 0.0, 0.0),
            ObservedGrid(AnchorSpec("spawn_vehicle_a", 0.0, 0.0, 0.0)),
            vehicle_id="vehicle_a",
            now=1.1,
            position_m=(0.5, 0.5),
            peer_states=(active_state,),
            peer_intents=(responder,),
            desired=(0.0, 0.0),
        )

        self.assertEqual(controller.vacate_request, request)

    def test_curved_route_keeps_request_until_requester_passes_blocker(
        self,
    ) -> None:
        route = (
            (0, 0),
            (0, 2),
            (2, 2),
            (2, 0),
        )
        controller, _ = parked_idle_request_once(
            route=route,
            peer_current=(2, 2),
            peer_position_m=(2.5, 2.5),
        )
        request = controller.vacate_request
        assert request is not None
        controller.navigation.transient_peer_blocked = False
        responder = replace(
            intent(
                "vehicle_b",
                current=(3, 2),
                target=None,
                wait_ticks=0,
                owner="vehicle_a",
            ),
            task_sequence=(1 << 64) - 1,
        )
        responder_state = peer_state("vehicle_b", 3.5, 2.5, 0.0)
        anchor = AnchorSpec("spawn_vehicle_a", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)

        coordinate_once(
            controller,
            Vehicle(0.5, 0.5, radius=0.5, now=0.0),
            anchor,
            local_map,
            vehicle_id="vehicle_a",
            now=1.1,
            position_m=(0.5, 0.5),
            peer_states=(responder_state,),
            peer_intents=(responder,),
            desired=(0.0, 0.0),
        )
        self.assertEqual(controller.vacate_request, request)

        coordinate_once(
            controller,
            Vehicle(0.5, 1.5, radius=0.5, now=0.0),
            anchor,
            local_map,
            vehicle_id="vehicle_a",
            now=1.2,
            position_m=(0.5, 1.5),
            peer_states=(responder_state,),
            peer_intents=(responder,),
            desired=(0.0, 0.0),
        )
        self.assertEqual(
            controller.vacate_request,
            replace(
                request,
                route_cells=((0, 1), (0, 2), (2, 2), (2, 0)),
            ),
        )

        coordinate_once(
            controller,
            Vehicle(2.5, 0.5, radius=0.5, now=0.0),
            anchor,
            local_map,
            vehicle_id="vehicle_a",
            now=1.3,
            position_m=(2.5, 0.5),
            peer_states=(responder_state,),
            peer_intents=(responder,),
            desired=(0.0, 0.0),
        )
        self.assertIsNone(controller.vacate_request)

    def test_vacate_request_progress_covers_legal_two_cell_route_steps(
        self,
    ) -> None:
        cases = (
            (((0, 0), (2, 0), (4, 0)), (1, 0)),
            (((0, 0), (2, 1), (4, 1)), (1, 0)),
            (((0, 0), (2, 1), (4, 1)), (1, 1)),
            (((0, 0), (1, 2), (1, 4)), (0, 1)),
            (((0, 0), (1, 2), (1, 4)), (1, 1)),
        )
        anchor = AnchorSpec("spawn_vehicle_a", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        for route, current_cell in cases:
            with self.subTest(route=route, current_cell=current_cell):
                blocker_cell = route[1]
                controller, _ = parked_idle_request_once(
                    route=route,
                    peer_current=blocker_cell,
                    peer_position_m=(
                        blocker_cell[0] + 0.5,
                        blocker_cell[1] + 0.5,
                    ),
                )
                request = controller.vacate_request
                assert request is not None
                controller.navigation.transient_peer_blocked = False
                responder_cell = (blocker_cell[0], blocker_cell[1] + 1)
                responder = replace(
                    intent(
                        "vehicle_b",
                        current=responder_cell,
                        target=None,
                        wait_ticks=0,
                        owner="vehicle_a",
                    ),
                    task_sequence=(1 << 64) - 1,
                )

                coordinate_once(
                    controller,
                    Vehicle(
                        current_cell[0] + 0.5,
                        current_cell[1] + 0.5,
                        radius=0.5,
                        now=0.0,
                    ),
                    anchor,
                    local_map,
                    vehicle_id="vehicle_a",
                    now=1.1,
                    position_m=(
                        current_cell[0] + 0.5,
                        current_cell[1] + 0.5,
                    ),
                    peer_states=(
                        peer_state(
                            "vehicle_b",
                            responder_cell[0] + 0.5,
                            responder_cell[1] + 0.5,
                            0.0,
                        ),
                    ),
                    peer_intents=(responder,),
                    desired=(0.0, 0.0),
                )

                self.assertEqual(
                    controller.vacate_request,
                    replace(
                        request,
                        route_cells=(current_cell, *route[1:]),
                    ),
                )

    def test_explicit_idle_vacate_defers_new_corridor_detection(self) -> None:
        requester = vacate_intent()

        controller, vehicle, _, navigation = idle_vacate_tick(
            requester,
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
            corridor=((0, 0), (6, 0)),
        )

        self.assertIsNone(controller._corridor)
        self.assertEqual(controller.motion_intent[0], (2.5, 1.5))
        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))
        navigation.coordination_corridor.assert_not_called()

    def test_idle_responder_requires_addressed_fresh_higher_priority_request(
        self,
    ) -> None:
        requester = vacate_intent()
        mission = GotoMission("existing", "global_map", 6.5, 0.5, 1)
        cases = (
            (
                "request addresses another vehicle",
                vacate_intent(request_vehicle_id="vehicle_c"),
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {},
            ),
            (
                "self request",
                vacate_intent("vehicle_d", request_vehicle_id="vehicle_d"),
                peer_state("vehicle_d", 0.5, 0.5, 0.0),
                {},
            ),
            (
                "requested cell is outside footprint",
                vacate_intent(
                    request_cell=(3, 0),
                    route=((0, 0), (2, 0), (4, 0), (5, 0)),
                ),
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {},
            ),
            (
                "requester has lower priority",
                vacate_intent("vehicle_z", wait_ticks=0),
                peer_state("vehicle_z", 0.5, 0.5, 0.0),
                {},
            ),
            (
                "requester generation mismatches pose",
                vacate_intent(intent_generation=2),
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {},
            ),
            (
                "manual mode",
                requester,
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {"mode": OpMode.MANUAL},
            ),
            (
                "paused auto",
                requester,
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {"auto_state": AutoState.PAUSED},
            ),
            (
                "blocked auto",
                requester,
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {"auto_state": AutoState.BLOCKED},
            ),
            (
                "active mission",
                requester,
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {"active_mission": mission},
            ),
            (
                "pending mission",
                requester,
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {"pending_mission": mission},
            ),
        )
        for label, candidate, state, kwargs in cases:
            with self.subTest(label=label):
                controller, vehicle, _, _ = idle_vacate_tick(
                    candidate,
                    state,
                    **kwargs,
                )
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
                self.assertIsNone(controller.motion_intent[0])
                self.assertFalse(controller.is_automatic_motion_active)

    def test_idle_vacate_does_not_start_without_a_safe_executable_detour(
        self,
    ) -> None:
        cases = (
            (
                "no safe detour",
                vacate_intent(),
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {"detours": ()},
                "clear",
            ),
            (
                "local safety rejects detour",
                vacate_intent(
                    current=(2, 2),
                    route=((2, 2), (2, 0), (2, -2)),
                ),
                peer_state("vehicle_a", 2.5, 2.5, 0.0),
                {
                    "detours": ((3.5, 0.5),),
                    "walls": frozenset(((3, 0),)),
                },
                "stopped",
            ),
            (
                "localization lost",
                vacate_intent(),
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                {"pose_quality": "lost"},
                "clear",
            ),
        )
        for label, requester, state, kwargs, safety_state in cases:
            with self.subTest(label=label):
                controller, vehicle, safety, _ = idle_vacate_tick(
                    requester,
                    state,
                    **kwargs,
                )
                self.assertEqual(safety.decision.state, safety_state)
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
                self.assertIsNone(controller.motion_intent[0])
                self.assertEqual(controller.temporal_motion_intent[3], ())
                self.assertEqual(
                    controller.snapshot()["coordination"]["state"], "idle"
                )
                self.assertFalse(controller.is_automatic_motion_active)

    def test_degraded_localization_caps_idle_vacate_linear_speed(self) -> None:
        requester = vacate_intent()

        controller, vehicle, _, _ = idle_vacate_tick(
            requester,
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
            pose_quality="degraded",
            pose_yaw_rad=math.pi / 2,
        )

        self.assertEqual(vehicle.target_velocities(), (0.25, 0.0))
        self.assertTrue(controller.is_automatic_motion_active)

    def test_active_idle_vacate_session_stops_when_request_expires(self) -> None:
        requester = vacate_intent()
        controller, vehicle, safety, _ = idle_vacate_tick(
            requester,
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
        )
        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))

        anchor = AnchorSpec("idle-responder", 0.0, 0.0, 0.0)
        controller.tick(
            vehicle=vehicle,
            grid=MapGrid.from_wall_set(8, 8, set()),
            safety=safety,
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                2.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.4,
                14,
            ),
            local_map=ObservedGrid(anchor),
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=1.4,
            vehicle_id="vehicle_d",
            peer_states=(),
            peer_motion_intents=(),
            coordination_ready=True,
            expected_peer_vehicle_ids=("vehicle_a",),
        )

        self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
        self.assertIsNone(controller.motion_intent[0])
        self.assertTrue(controller.is_automatic_motion_active)

    def test_active_idle_vacate_freezes_when_motion_is_unsafe(self) -> None:
        for advance_result, pose_quality in (
            (SafetyAdvanceResult(collided=True), "nominal"),
            (
                SafetyAdvanceResult(stopped=True, reason="safety_obstacle"),
                "nominal",
            ),
            (SafetyAdvanceResult(), "lost"),
        ):
            with self.subTest(
                advance_result=advance_result,
                pose_quality=pose_quality,
            ):
                (
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    requester,
                ) = real_idle_vacate_session()
                fresh = replace(
                    requester,
                    sequence=2,
                    timestamp_s=1.1,
                    received_at_s=1.1,
                )
                fresh_state = replace(
                    peer_state("vehicle_a", 0.5, 0.5, 0.0),
                    sequence=2,
                    timestamp_s=1.1,
                )
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 0.5),
                    now=1.1,
                    requester=fresh,
                    requester_state=fresh_state,
                    advance_result=advance_result,
                    pose_quality=pose_quality,
                )

                self.assertEqual(controller.auto_state, AutoState.PAUSED)
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
                self.assertTrue(controller.is_automatic_motion_active)
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 0.5),
                    now=1.2,
                    requester=replace(
                        fresh,
                        sequence=3,
                        timestamp_s=1.2,
                        received_at_s=1.2,
                    ),
                    requester_state=replace(
                        fresh_state,
                        sequence=3,
                        timestamp_s=1.2,
                    ),
                )

                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))

    def test_active_idle_vacate_replans_when_route_window_changes(self) -> None:
        old_route = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))
        new_route = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
        requester = vacate_intent(route=old_route)
        controller, vehicle, safety, navigation = idle_vacate_tick(
            requester,
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
            vacate_path=((2.5, 1.5),),
        )
        self.assertEqual(controller._peer_vacate_request_route_cells, old_route)
        navigation.coordination_vacate_path.reset_mock()

        updated = replace(
            requester,
            sequence=2,
            timestamp_s=1.1,
            received_at_s=1.1,
            plan_generation=2,
            vacate_request=VacateRequest("vehicle_d", (2, 0), new_route),
        )
        anchor = AnchorSpec("idle-responder", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 0.5),
            now=1.1,
            requester=updated,
            requester_state=replace(
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                sequence=2,
                timestamp_s=1.1,
            ),
        )

        self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
        self.assertEqual(controller._peer_vacate_request_route_cells, new_route)
        self.assertEqual(controller._peer_vacate_path_m, ())
        self.assertIsNone(controller.motion_intent[0])
        self.assertEqual(controller.temporal_motion_intent[3], ())
        navigation.coordination_vacate_path.assert_not_called()

        navigation.coordination_vacate_path.return_value = ((2.5, 1.5),)
        continued = replace(
            updated,
            sequence=3,
            timestamp_s=1.2,
            received_at_s=1.2,
        )
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 0.5),
            now=1.2,
            requester=continued,
            requester_state=replace(
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                sequence=3,
                timestamp_s=1.2,
            ),
        )

        navigation.coordination_vacate_path.assert_called_once()
        self.assertEqual(controller.motion_intent[0], (2.5, 1.5))

    def test_fresh_retraction_before_entry_returns_after_three_clear_ticks(
        self,
    ) -> None:
        controller, vehicle, safety, anchor, local_map, _ = (
            real_idle_vacate_session()
        )
        clock = [1.0]
        requester_anchor = AnchorSpec("requester", 0.0, 0.0, 0.0)
        source = MapSyncState(
            "session",
            "vehicle_a",
            requester_anchor,
            1.0,
            state_generation=1,
        )
        receiver = MapSyncState(
            "session",
            "vehicle_d",
            anchor,
            1.0,
            clock=lambda: clock[0],
            state_generation=1,
        )
        source.configure_network(
            "peer_a",
            {"vehicle_d": ("peer_d", anchor)},
        )
        receiver.configure_network(
            "peer_d",
            {"vehicle_a": ("peer_a", requester_anchor)},
        )

        for clear_tick in range(1, 4):
            clock[0] = 1.0 + clear_tick / 10
            pose = PoseEstimate(
                requester_anchor.anchor_id,
                5.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                clock[0],
                clear_tick,
            )
            source.record_vehicle_state(
                pose,
                radius_m=0.5,
                linear_mps=0.0,
                omega_rps=0.0,
            )
            source.record_motion_intent(
                pose,
                target_m=(5.5, 1.5),
                wait_ticks=0,
                priority_owner_id="vehicle_a",
                reserved=False,
                timestamp_s=clock[0],
                task_sequence=1,
                vacate_request=None,
            )
            state_payload = source.prepare_peer_state()
            intent_payload = source.prepare_motion_intent()
            assert state_payload is not None and intent_payload is not None
            self.assertTrue(
                receiver.receive_transport("peer_a", "vehicle_a", state_payload)
            )
            self.assertTrue(
                receiver.receive_transport("peer_a", "vehicle_a", intent_payload)
            )
            source.publish_peer_state_result(state_payload["sequence"], True)
            source.publish_motion_intent_result(intent_payload["sequence"], True)
            peer_intents = receiver.peer_motion_intents()
            self.assertIsNone(peer_intents[0].vacate_request)

            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                local_map,
                position_m=(2.5, 2.5),
                now=clock[0],
                requester=peer_intents[0],
                requester_state=receiver.peer_vehicle_states()[0],
            )
            if clear_tick < 3:
                self.assertEqual(controller.navigation.status, "reached")

        self.assertEqual(controller.navigation.status, "active")
        self.assertEqual(controller.navigation.requested_goal, (2.5, 0.5))
        self.assertEqual(controller.events_after(0), ())

    def test_idle_vacate_returns_only_after_three_clear_ticks(self) -> None:
        (
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            requester,
        ) = real_idle_vacate_session()
        entered = enter_real_idle_vacate_session(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            requester,
        )
        for clear_tick in range(1, 4):
            now = 1.1 + clear_tick / 10
            cleared, clear_state = clear_vacate_evidence(
                entered,
                now=now,
                sequence=2 + clear_tick,
                plan_generation=1 + clear_tick,
            )
            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                local_map,
                position_m=(2.5, 2.5),
                now=now,
                requester=cleared,
                requester_state=clear_state,
            )
            self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
            if clear_tick < 3:
                self.assertEqual(controller.navigation.status, "reached")

        self.assertEqual(controller.navigation.status, "active")
        self.assertEqual(controller.navigation.requested_goal, (2.5, 0.5))
        self.assertTrue(controller.is_automatic_motion_active)
        self.assertEqual(controller.events_after(0), ())

    def test_idle_vacate_trajectory_overlap_resets_clear_debounce(self) -> None:
        (
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            requester,
        ) = real_idle_vacate_session()
        entered = enter_real_idle_vacate_session(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            requester,
        )
        overlap_trajectory = (
            TimedCell((5, 0), 0.0, 0.0),
            TimedCell((2, 0), 1.0, 4.0),
        )
        trajectories = (
            _CLEAR_VACATE_TRAJECTORY,
            overlap_trajectory,
            _CLEAR_VACATE_TRAJECTORY,
            _CLEAR_VACATE_TRAJECTORY,
        )
        for index, trajectory in enumerate(trajectories, start=1):
            now = 1.1 + index / 10
            evidence, state = clear_vacate_evidence(
                entered,
                now=now,
                sequence=2 + index,
                trajectory=trajectory,
            )
            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                local_map,
                position_m=(2.5, 2.5),
                now=now,
                requester=evidence,
                requester_state=state,
            )

        self.assertEqual(controller.navigation.status, "reached")
        now = 1.6
        final_clear, final_state = clear_vacate_evidence(
            entered,
            now=now,
            sequence=7,
        )
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 2.5),
            now=now,
            requester=final_clear,
            requester_state=final_state,
        )

        self.assertEqual(controller.navigation.status, "active")

    def test_idle_vacate_generation_change_resets_clear_debounce(self) -> None:
        (
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            requester,
        ) = real_idle_vacate_session()
        entered = enter_real_idle_vacate_session(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            requester,
        )
        def send_clear(
            *,
            now: float,
            sequence: int,
            generation: int = 1,
        ) -> None:
            evidence, state = clear_vacate_evidence(
                entered,
                now=now,
                sequence=sequence,
                generation=generation,
            )
            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                local_map,
                position_m=(2.5, 2.5),
                now=now,
                requester=evidence,
                requester_state=state,
            )

        send_clear(now=1.2, sequence=3)
        send_clear(now=1.3, sequence=4, generation=2)
        send_clear(now=1.4, sequence=5)
        send_clear(now=1.5, sequence=6)
        self.assertEqual(controller.navigation.status, "reached")

        send_clear(now=1.6, sequence=7)
        self.assertEqual(controller.navigation.status, "active")

    def test_invalid_idle_vacate_evidence_resets_clear_debounce(self) -> None:
        invalid_cases = (
            "missing",
            "lost",
            "generation_mismatch",
            "physical_cell_mismatch",
            "other_requester",
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                (
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    requester,
                ) = real_idle_vacate_session()
                entered = enter_real_idle_vacate_session(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    requester,
                )

                first, first_state = clear_vacate_evidence(
                    entered, now=1.2, sequence=3
                )
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 2.5),
                    now=1.2,
                    requester=first,
                    requester_state=first_state,
                )
                invalid, invalid_state = clear_vacate_evidence(
                    entered, now=1.3, sequence=4
                )
                if case == "missing":
                    invalid = invalid_state = None
                elif case == "lost":
                    invalid_state = replace(invalid_state, quality="lost")
                elif case == "generation_mismatch":
                    invalid_state = replace(invalid_state, state_generation=2)
                elif case == "physical_cell_mismatch":
                    invalid_state = replace(invalid_state, global_x_m=4.5)
                else:
                    invalid = replace(invalid, source_vehicle_id="vehicle_b")
                    invalid_state = replace(
                        invalid_state,
                        source_vehicle_id="vehicle_b",
                    )
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 2.5),
                    now=1.3,
                    requester=invalid,
                    requester_state=invalid_state,
                )
                for now, sequence in ((1.4, 5), (1.5, 6)):
                    evidence, state = clear_vacate_evidence(
                        entered, now=now, sequence=sequence
                    )
                    tick_real_idle_vacate(
                        controller,
                        vehicle,
                        safety,
                        anchor,
                        local_map,
                        position_m=(2.5, 2.5),
                        now=now,
                        requester=evidence,
                        requester_state=state,
                    )

                self.assertEqual(controller.navigation.status, "reached")
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
                self.assertTrue(controller.is_automatic_motion_active)

    def test_idle_vacate_returns_through_real_goto_without_mission_events(
        self,
    ) -> None:
        (
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            cleared,
            state,
        ) = release_real_idle_vacate_session()

        for tick in range(1, 9):
            now = 1.4 + tick / 10
            fresh = replace(
                cleared,
                sequence=5 + tick,
                timestamp_s=now,
                received_at_s=now,
            )
            fresh_state = replace(
                state,
                sequence=5 + tick,
                timestamp_s=now,
            )
            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                local_map,
                position_m=(2.5, 2.5),
                now=now,
                requester=fresh,
                requester_state=fresh_state,
            )
            if vehicle.target_velocities() != (0.0, 0.0):
                break

        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))
        now += 0.1
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 0.5),
            now=now,
            requester=replace(
                cleared,
                sequence=20,
                timestamp_s=now,
                received_at_s=now,
            ),
            requester_state=replace(
                state,
                sequence=20,
                timestamp_s=now,
            ),
        )

        self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
        self.assertFalse(controller.is_automatic_motion_active)
        self.assertEqual(controller.events_after(0), ())

    def test_idle_vacate_return_resumes_after_peer_map_clears(self) -> None:
        (
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            cleared,
            state,
        ) = release_real_idle_vacate_session()
        planning_map = _TransientPlanningGrid(local_map)
        forbidden = {
            (gx, gy)
            for gx in range(1, 4)
            for gy in range(1, 4)
        }
        planning_map.update(
            set(),
            None,
            peer_forbidden_cells=forbidden,
        )
        controller.navigation.block(
            "no_path",
            "dynamic overlay seeded when return started",
        )
        now = 1.5
        fresh = replace(
            cleared,
            sequence=6,
            timestamp_s=now,
            received_at_s=now,
        )
        fresh_state = replace(state, sequence=6, timestamp_s=now)
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            planning_map,
            position_m=(2.5, 2.5),
            now=now,
            requester=fresh,
            requester_state=fresh_state,
            map_delta=None,
            coordination_map=local_map,
        )

        self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
        self.assertTrue(controller.navigation.transient_peer_blocked)

        cleared_delta = planning_map.update(
            set(),
            None,
            peer_forbidden_cells=set(),
        )
        for tick in range(1, 9):
            now = 1.5 + tick / 10
            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                planning_map,
                position_m=(2.5, 2.5),
                now=now,
                requester=replace(
                    fresh,
                    sequence=6 + tick,
                    timestamp_s=now,
                    received_at_s=now,
                ),
                requester_state=replace(
                    fresh_state,
                    sequence=6 + tick,
                    timestamp_s=now,
                ),
                map_delta=(cleared_delta if tick == 1 else LocalMapDelta((), ())),
                coordination_map=local_map,
            )
            if vehicle.target_velocities() != (0.0, 0.0):
                break

        self.assertFalse(controller.navigation.transient_peer_blocked)
        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))

    def test_idle_vacate_return_safety_stop_keeps_the_session(self) -> None:
        for expected_state in ("fault", "stopped"):
            with self.subTest(expected_state=expected_state):
                (
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    cleared,
                    state,
                ) = release_real_idle_vacate_session()
                now = 1.5
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 2.5),
                    now=now,
                    requester=replace(
                        cleared,
                        sequence=6,
                        timestamp_s=now,
                        received_at_s=now,
                    ),
                    requester_state=replace(
                        state,
                        sequence=6,
                        timestamp_s=now,
                    ),
                    safety_scan_healthy=expected_state != "fault",
                    yaw_rad=-math.pi / 2,
                    walls=(
                        frozenset(((2, 1),))
                        if expected_state == "stopped"
                        else frozenset()
                    ),
                )

                self.assertEqual(safety.decision.state, expected_state)
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
                self.assertEqual(controller.navigation.status, "active")
                self.assertTrue(controller.is_automatic_motion_active)
                self.assertEqual(controller.events_after(0), ())

    def test_manual_takeover_and_cancel_clear_idle_vacate_session(self) -> None:
        for action in ("manual", "cancel"):
            with self.subTest(action=action):
                controller, vehicle, _, _, _, _ = real_idle_vacate_session()
                if action == "manual":
                    controller.handle(
                        ModeCommand(2, ModeAction.SWITCH_TO_MANUAL),
                        vehicle=vehicle,
                        grid=MapGrid.from_wall_set(8, 8, set()),
                        safety=LocalSafetyRuntime(),
                        now=1.1,
                    )
                else:
                    controller.handle(
                        AutoCommand(2, AutoAction.CANCEL_ALL),
                        vehicle=vehicle,
                        grid=MapGrid.from_wall_set(8, 8, set()),
                        safety=LocalSafetyRuntime(),
                        now=1.1,
                    )

                self.assertFalse(controller.is_automatic_motion_active)
                self.assertIsNone(controller.motion_intent[0])

    def test_pause_and_stop_freeze_then_resume_idle_vacate_session(self) -> None:
        for action in ("pause", "stop"):
            with self.subTest(action=action):
                (
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    requester,
                ) = real_idle_vacate_session()
                command: AutoCommand | ModeCommand = (
                    AutoCommand(2, AutoAction.PAUSE)
                    if action == "pause"
                    else ModeCommand(2, ModeAction.STOP_MOTION)
                )
                controller.handle(
                    command,
                    vehicle=vehicle,
                    grid=MapGrid.from_wall_set(8, 8, set()),
                    safety=safety,
                    now=1.1,
                )

                self.assertEqual(controller.auto_state, AutoState.PAUSED)
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
                self.assertIsNone(controller.motion_intent[0])
                self.assertTrue(controller.is_automatic_motion_active)
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 0.5),
                    now=1.2,
                    requester=None,
                    requester_state=None,
                )
                self.assertEqual(controller.auto_state, AutoState.PAUSED)
                self.assertTrue(controller.is_automatic_motion_active)

                controller.handle(
                    AutoCommand(3, AutoAction.RESUME),
                    vehicle=vehicle,
                    grid=MapGrid.from_wall_set(8, 8, set()),
                    safety=safety,
                    now=1.3,
                )
                fresh = replace(
                    requester,
                    sequence=3,
                    timestamp_s=1.3,
                    received_at_s=1.3,
                )
                fresh_state = replace(
                    peer_state("vehicle_a", 0.5, 0.5, 0.0),
                    sequence=3,
                    timestamp_s=1.3,
                )
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 0.5),
                    now=1.3,
                    requester=fresh,
                    requester_state=fresh_state,
                )

                self.assertEqual(controller.auto_state, AutoState.IDLE)
                self.assertIsNotNone(controller.motion_intent[0])
                self.assertTrue(controller.is_automatic_motion_active)

    def test_pause_resets_idle_vacate_clear_debounce(self) -> None:
        controller, vehicle, safety, anchor, local_map, requester = (
            real_idle_vacate_session()
        )
        entered = enter_real_idle_vacate_session(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            requester,
        )
        def send_clear(now: float, sequence: int) -> None:
            evidence, state = clear_vacate_evidence(
                entered, now=now, sequence=sequence
            )
            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                local_map,
                position_m=(2.5, 2.5),
                now=now,
                requester=evidence,
                requester_state=state,
            )

        send_clear(1.2, 3)
        send_clear(1.3, 4)
        controller.handle(
            AutoCommand(2, AutoAction.PAUSE),
            vehicle=vehicle,
            grid=MapGrid.from_wall_set(8, 8, set()),
            safety=safety,
            now=1.35,
        )
        controller.handle(
            AutoCommand(3, AutoAction.RESUME),
            vehicle=vehicle,
            grid=MapGrid.from_wall_set(8, 8, set()),
            safety=safety,
            now=1.4,
        )

        for clear_tick in range(1, 4):
            send_clear(1.4 + clear_tick / 10, 4 + clear_tick)
            if clear_tick < 3:
                self.assertEqual(controller.navigation.status, "reached")

        self.assertEqual(controller.navigation.status, "active")

    def test_route_update_resets_idle_vacate_clear_debounce(self) -> None:
        controller, vehicle, safety, anchor, local_map, requester = (
            real_idle_vacate_session()
        )
        def send_clear(now: float, sequence: int) -> None:
            evidence, state = clear_vacate_evidence(
                requester, now=now, sequence=sequence
            )
            tick_real_idle_vacate(
                controller,
                vehicle,
                safety,
                anchor,
                local_map,
                position_m=(2.5, 2.5),
                now=now,
                requester=evidence,
                requester_state=state,
            )

        send_clear(1.1, 2)
        send_clear(1.2, 3)
        updated_route = (
            (5, 0),
            (5, 1),
            (4, 1),
            (3, 1),
            (2, 1),
            (2, 0),
            (1, 0),
        )
        updated = replace(
            requester,
            sequence=4,
            timestamp_s=1.3,
            received_at_s=1.3,
            plan_generation=2,
            current_cell=(5, 0),
            vacate_request=VacateRequest("vehicle_d", (2, 0), updated_route),
        )
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 2.5),
            now=1.3,
            requester=updated,
            requester_state=replace(
                peer_state("vehicle_a", 5.5, 0.5, 0.0),
                sequence=4,
                timestamp_s=1.3,
            ),
        )

        for clear_tick in range(1, 4):
            send_clear(1.3 + clear_tick / 10, 4 + clear_tick)
            if clear_tick < 3:
                self.assertEqual(controller.navigation.status, "reached")

        self.assertEqual(controller.navigation.status, "active")

    def test_disconnect_and_fail_safe_stop_freeze_idle_vacate_session(
        self,
    ) -> None:
        for action in ("disconnect", "fail_safe_stop"):
            with self.subTest(action=action):
                (
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    requester,
                ) = real_idle_vacate_session()
                if action == "disconnect":
                    controller.disconnect(vehicle)
                else:
                    controller.fail_safe_stop(vehicle, "transport_lost")

                fresh = replace(
                    requester,
                    sequence=2,
                    timestamp_s=1.2,
                    received_at_s=1.2,
                )
                tick_real_idle_vacate(
                    controller,
                    vehicle,
                    safety,
                    anchor,
                    local_map,
                    position_m=(2.5, 0.5),
                    now=1.2,
                    requester=fresh,
                    requester_state=replace(
                        peer_state("vehicle_a", 0.5, 0.5, 0.0),
                        sequence=2,
                        timestamp_s=1.2,
                    ),
                )

                self.assertEqual(controller.auto_state, AutoState.PAUSED)
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
                self.assertTrue(controller.is_automatic_motion_active)

    def test_reached_mission_releases_corridor_before_idle_vacate(self) -> None:
        anchor = AnchorSpec("idle-responder", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            status="active",
            motion_target=None,
            coordination_corridor=Mock(return_value=None),
            coordination_detours=Mock(return_value=((2.5, 1.5),)),
            coordination_vacate_path=Mock(return_value=()),
            coordination_path_cells=Mock(return_value=None),
        )

        def reach(**_: object) -> tuple[float, float]:
            navigation.status = "reached"
            return 0.0, 0.0

        navigation.update.side_effect = reach
        controller = RobotController(navigation)
        controller.mode = OpMode.AUTO
        controller.auto_state = AutoState.ACTIVE
        inject_active_goto(controller, "completed", 2.5, 0.5)
        controller._corridor = CorridorDescriptor((2, 0), (6, 0))
        vehicle = Vehicle(2.5, 0.5, now=1.0)
        safety = LocalSafetyRuntime()

        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 0.5),
            now=1.0,
            requester=None,
            requester_state=None,
        )
        requester = vacate_intent(
            received_at_s=1.1,
            timestamp_s=1.1,
            sequence=2,
        )
        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            local_map,
            position_m=(2.5, 0.5),
            now=1.1,
            requester=requester,
            requester_state=replace(
                peer_state("vehicle_a", 0.5, 0.5, 0.0),
                sequence=2,
                timestamp_s=1.1,
            ),
        )

        self.assertIsNone(controller._corridor)
        self.assertTrue(controller.is_automatic_motion_active)
        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))

    def test_reached_goto_planner_supplies_real_idle_responder_detour(self) -> None:
        anchor = AnchorSpec("idle-responder", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        pose = PoseEstimate(
            anchor.anchor_id,
            2.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = GotoController()
        navigation.start(2.5, 0.5, local_map=local_map, pose=pose)
        for _ in range(4):
            navigation.update(
                pose=pose,
                local_map=local_map,
                max_linear_mps=1.0,
                max_angular_rps=1.0,
            )
            if navigation.status == "reached":
                break
        self.assertEqual(navigation.status, "reached")
        self.assertEqual(navigation.coordination_detours(pose, local_map), ())
        self.assertTrue(
            navigation.coordination_detours(
                pose,
                local_map,
                allow_reached=True,
            )
        )

        controller = RobotController(navigation)
        controller.mode = OpMode.AUTO
        requester = vacate_intent()
        vehicle = Vehicle(2.5, 0.5, now=1.0)

        controller.tick(
            vehicle=vehicle,
            grid=MapGrid.from_wall_set(8, 8, set()),
            safety=LocalSafetyRuntime(),
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            map_delta=None,
            advance_result=SafetyAdvanceResult(),
            now=1.0,
            vehicle_id="vehicle_d",
            peer_states=(peer_state("vehicle_a", 0.5, 0.5, 0.0),),
            peer_motion_intents=(requester,),
            coordination_ready=True,
            expected_peer_vehicle_ids=("vehicle_a",),
        )

        self.assertIsNotNone(
            controller.motion_intent[0],
            controller.snapshot(),
        )
        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))
        self.assertTrue(controller.is_automatic_motion_active)

    def test_active_idle_responder_continues_to_same_cell_metric_detour(
        self,
    ) -> None:
        requester = vacate_intent()
        controller, vehicle, safety, _ = idle_vacate_tick(
            requester,
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
        )
        self.assertEqual(controller.motion_intent[0], (2.5, 1.5))
        fresh = replace(
            requester,
            sequence=2,
            timestamp_s=1.1,
            received_at_s=1.1,
        )
        state = replace(
            peer_state("vehicle_a", 0.5, 0.5, 0.0),
            sequence=2,
            timestamp_s=1.1,
        )
        anchor = AnchorSpec("idle-responder", 0.0, 0.0, 0.0)

        tick_real_idle_vacate(
            controller,
            vehicle,
            safety,
            anchor,
            ObservedGrid(anchor),
            position_m=(2.5, 1.1),
            now=1.1,
            requester=fresh,
            requester_state=state,
        )

        self.assertEqual(safety.decision.state, "clear")
        self.assertEqual(controller.motion_intent[0], (2.5, 1.5))
        self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))

    def test_real_t_junction_fallback_prioritizes_approach_and_small_dip(
        self,
    ) -> None:
        resolution_m = 0.5
        anchor = AnchorSpec("vehicle_d_spawn", 11.5, 16.5, -math.pi / 2)
        route = ((25, 15), (24, 16)) + tuple(
            (23, gy) for gy in range(17, 34)
        )
        self.assertEqual(len(route), 19)
        world_free = {
            (x, y)
            for x in range(1, 22)
            for y in range(4, 9)
        } | {
            (x, y)
            for x in range(10, 13)
            for y in range(4, 22)
        }
        local_map = ObservedGrid(anchor, resolution_m=resolution_m)
        for gx in range(-30, 31):
            for gy in range(-30, 31):
                global_x_m, global_y_m, _ = anchor.anchor_to_global(
                    (gx + 0.5) * resolution_m,
                    (gy + 0.5) * resolution_m,
                    0.0,
                )
                local_map._cells[gx, gy] = (
                    FREE
                    if (math.floor(global_x_m), math.floor(global_y_m))
                    in world_free
                    else OCCUPIED
                )

        def route_clearance(point_m: tuple[float, float]) -> float:
            return _point_route_distance(
                anchor.anchor_to_global(*point_m, 0.0)[:2],
                route,
                resolution_m,
            )

        grid = MapGrid.from_wall_set(
            23,
            23,
            {
                (x, y)
                for x in range(23)
                for y in range(23)
                if (x, y) not in world_free
            },
        )

        def run_trace_point(
            pose_x_m: float,
            pose_y_m: float,
            goal_y_m: float,
            request_cell: tuple[int, int],
            *,
            block_approach: bool = False,
        ) -> tuple[
            tuple[float, float] | None,
            tuple[tuple[float, float], ...],
            float,
        ]:
            pose = PoseEstimate(
                anchor.anchor_id,
                pose_x_m,
                pose_y_m,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                1,
            )
            navigation = GotoController()
            navigation.start(
                pose.x_m,
                goal_y_m,
                local_map=local_map,
                pose=pose,
                vehicle_radius_m=0.5,
            )
            navigation.update(
                pose=pose,
                local_map=local_map,
                max_linear_mps=1.0,
                max_angular_rps=math.pi / 2,
            )
            self.assertEqual(navigation.status, "reached")
            self.assertEqual(
                navigation.coordination_vacate_path(
                    pose,
                    local_map,
                    1.25,
                    clearance_at_m=route_clearance,
                    allow_reached=True,
                ),
                (),
            )
            detours = navigation.coordination_detours(
                pose,
                local_map,
                allow_reached=True,
            )
            requester = vacate_intent(
                current=route[0],
                request_cell=request_cell,
                route=route,
                trajectory=(TimedCell(route[0], 0.0, 4.0),),
            )
            peer_states = [peer_state("vehicle_a", 12.676, 7.75, 0.0)]
            peer_intents = [requester]
            if block_approach:
                peer_states.append(
                    replace(
                        peer_state("vehicle_c", 11.25, 12.25, 0.0),
                        radius_m=0.01,
                    )
                )
                peer_intents.append(
                    replace(
                        intent(
                            "vehicle_c",
                            current=(22, 24),
                            target=None,
                            wait_ticks=0,
                            reserved=True,
                        ),
                        task_sequence=2,
                        trajectory=(TimedCell((22, 24), 0.0, 4.0),),
                        received_at_s=1.0,
                    )
                )
            controller = RobotController(navigation)
            controller.mode = OpMode.AUTO
            global_pose = anchor.anchor_to_global(
                pose.x_m,
                pose.y_m,
                pose.yaw_rad,
            )
            vehicle = Vehicle(*global_pose, radius=0.5, now=1.0)
            safety = LocalSafetyRuntime()
            controller.tick(
                vehicle=vehicle,
                grid=grid,
                safety=safety,
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                map_delta=None,
                advance_result=SafetyAdvanceResult(),
                now=1.0,
                vehicle_id="vehicle_d",
                peer_states=tuple(peer_states),
                peer_motion_intents=tuple(peer_intents),
                coordination_ready=True,
                expected_peer_vehicle_ids=tuple(
                    peer.source_vehicle_id for peer in peer_intents
                ),
            )
            self.assertEqual(safety.decision.state, "clear")
            if block_approach:
                self.assertEqual(vehicle.target_velocities(), (0.0, 0.0))
            else:
                self.assertNotEqual(vehicle.target_velocities(), (0.0, 0.0))
            return (
                controller.motion_intent[0],
                detours,
                route_clearance((pose.x_m, pose.y_m)),
            )

        target_m, detours, clearance_m = run_trace_point(
            3.5651637580680973,
            -0.08294834497886505,
            0.0,
            (22, 25),
        )
        self.assertEqual(target_m, (4.25, -0.25))
        self.assertLess(detours.index((3.25, -0.25)), detours.index(target_m))
        self.assertGreater(route_clearance(target_m), clearance_m)

        target_m, _, _ = run_trace_point(
            3.5651637580680973,
            -0.08294834497886505,
            0.0,
            (22, 25),
            block_approach=True,
        )
        self.assertIsNone(target_m)

        target_m, _, clearance_m = run_trace_point(
            4.663763237887391,
            -0.26254399030422704,
            -0.26254399030422704,
            (22, 23),
        )
        self.assertEqual(target_m, (5.25, -0.25))
        self.assertLess(route_clearance(target_m), clearance_m)

    def test_explicit_idle_detour_uses_route_clearance_with_rotated_anchor(
        self,
    ) -> None:
        resolution_m = 0.5
        route = tuple((gx, 13) for gx in range(37, 22, -1)) + tuple(
            (23, gy) for gy in range(14, 18)
        )

        def center(cell: tuple[int, int]) -> tuple[float, float]:
            return tuple((coordinate + 0.5) * resolution_m for coordinate in cell)

        anchor = AnchorSpec("rotated-route-window", 20.0, 10.0, math.pi / 2)
        pose_m = anchor.global_to_anchor(*center((21, 15)), 0.0)[:2]
        detour_m = anchor.global_to_anchor(*center((20, 18)), 0.0)[:2]
        local_map = ObservedGrid(anchor, resolution_m=resolution_m)
        navigation = Mock(
            transient_peer_blocked=False,
            coordination_vacate_path=Mock(return_value=()),
            coordination_detours=Mock(return_value=(detour_m,)),
        )
        controller = RobotController(navigation)
        controller._peer_vacate_request_cell = (23, 33)
        controller._peer_vacate_request_route_cells = route
        controller._yielding_for = "vehicle_a"
        controller._coordination_wait_reason = "peer_vacate"
        controller._coordination_wait_owner_id = "vehicle_a"
        controller._schedule_temporal_motion = Mock(
            side_effect=lambda desired, **kwargs: (
                desired,
                kwargs["own"].target_cell,
                kwargs["own"],
                False,
            )
        )
        requester = vacate_intent(
            current=route[0],
            request_cell=(23, 33),
            route=route,
        )
        pose = PoseEstimate(
            anchor.anchor_id,
            *pose_m,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )

        desired = controller._transient_peer_vacate(
            own=replace(
                intent(
                    "vehicle_d",
                    current=(21, 15),
                    target=None,
                    wait_ticks=0,
                ),
                task_sequence=(1 << 64) - 1,
            ),
            vehicle=Vehicle(*pose_m, radius=0.5, now=1.0),
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            peers={
                "vehicle_a": replace(
                    peer_state("vehicle_a", *center(route[0]), 0.0),
                    radius_m=0.1,
                )
            },
            peer_motion_intents=(requester,),
            coordination_map=None,
            now=1.0,
            idle_vacate_requester_id="vehicle_a",
        )

        self.assertNotEqual(desired, (0.0, 0.0))
        self.assertEqual(controller.motion_intent[0], detour_m)

    def test_implicit_vacate_uses_navigation_detour_order(self) -> None:
        anchor = AnchorSpec("implicit-vacate", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            transient_peer_blocked=False,
            coordination_detours=Mock(
                return_value=((3.5, 1.5), (1.5, 1.5))
            ),
        )
        controller = RobotController(navigation)
        controller._schedule_temporal_motion = Mock(
            side_effect=lambda desired, **kwargs: (
                desired,
                kwargs["own"].target_cell,
                kwargs["own"],
                False,
            )
        )
        owner = replace(
            intent(
                "vehicle_a",
                current=(0, 0),
                target=(2, 0),
                wait_ticks=4,
            ),
            task_sequence=1,
        )

        controller._transient_peer_vacate(
            own=replace(
                intent(
                    "vehicle_d",
                    current=(2, 0),
                    target=None,
                    wait_ticks=0,
                ),
                task_sequence=2,
            ),
            vehicle=Vehicle(2.5, 0.5, now=1.0),
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                2.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                1,
            ),
            local_map=local_map,
            peers={"vehicle_a": peer_state("vehicle_a", 0.5, 0.5, 0.0)},
            peer_motion_intents=(owner,),
            coordination_map=None,
            now=1.0,
        )

        self.assertEqual(controller.motion_intent[0], (3.5, 1.5))

    def test_transient_no_path_does_not_vacate_for_a_self_clearing_blocker(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_b", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor, resolution_m=0.5)
        temporal_prefix = (TimedCell((0, 0), 0.0, 4.0),)

        def peer_intent(
            *,
            target: tuple[int, int] | None,
            task_sequence: int,
        ) -> PeerMotionIntent:
            trajectory = (
                (TimedCell((6, 0), 0.0, 4.0),)
                if target is None
                else (
                    TimedCell((6, 2), 0.0, 0.5),
                    TimedCell((5, 2), 0.5, 1.0),
                    TimedCell((4, 2), 1.0, 4.0),
                )
            )
            return PeerMotionIntent(
                "vehicle_a",
                1,
                1,
                1.0,
                0.35,
                (6, 0) if target is None else (6, 2),
                target,
                5,
                "vehicle_a",
                False,
                task_sequence=task_sequence,
                trajectory=trajectory,
            )

        def own(current: tuple[int, int]) -> PeerMotionIntent:
            return PeerMotionIntent(
                "vehicle_b",
                1,
                1,
                1.0,
                0.35,
                current,
                (1, current[1]),
                0,
                "vehicle_b",
                task_sequence=2,
            )

        def controller_fixture() -> tuple[RobotController, Mock, Vehicle]:
            navigation = Mock(
                transient_peer_blocked=True,
                motion_target=None,
                coordination_path_cells=Mock(
                    return_value=tuple((gx, 0) for gx in range(7))
                ),
                coordination_vacate_path=Mock(return_value=((0.25, 1.25),)),
                coordination_detours=Mock(return_value=()),
            )
            controller = RobotController(navigation)
            controller._temporal_trajectory = temporal_prefix
            controller._temporal_commit_deadline_s = 5.0
            controller._intent_reserved = True
            controller._schedule_temporal_motion = Mock(
                side_effect=lambda desired, **kwargs: (
                    desired,
                    kwargs["own"].target_cell,
                    kwargs["own"],
                    False,
                )
            )
            return controller, navigation, Vehicle(
                0.25, 0.25, radius=0.5, now=0.0
            )

        def coordinate(
            controller: RobotController,
            vehicle: Vehicle,
            peer: PeerMotionIntent,
            *,
            now: float,
            own_cell: tuple[int, int] = (0, 0),
        ) -> tuple[float, float] | None:
            return controller._transient_peer_vacate(
                own=own(own_cell),
                vehicle=vehicle,
                anchor=anchor,
                pose=PoseEstimate(
                    anchor.anchor_id,
                    (own_cell[0] + 0.5) * local_map.resolution_m,
                    (own_cell[1] + 0.5) * local_map.resolution_m,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    now,
                    round(now * 10),
                ),
                local_map=local_map,
                peers={
                    "vehicle_a": peer_state(
                        "vehicle_a",
                        peer.current_cell[0] + 0.5,
                        peer.current_cell[1] + 0.5,
                        0.0,
                    )
                },
                peer_motion_intents=(peer,),
                coordination_map=None,
                now=now,
            )

        moving = peer_intent(target=(5, 2), task_sequence=1)
        controller, _, vehicle = controller_fixture()
        self.assertIsNone(coordinate(controller, vehicle, moving, now=1.0))
        self.assertIsNone(controller._peer_vacate_origin_cell)
        self.assertEqual(controller._temporal_trajectory, temporal_prefix)
        self.assertTrue(controller._intent_reserved)

        stationary = peer_intent(target=None, task_sequence=1)
        controller, navigation, vehicle = controller_fixture()
        self.assertIsNotNone(coordinate(controller, vehicle, stationary, now=1.0))
        self.assertEqual(controller._peer_vacate_origin_cell, (0, 0))
        navigation.transient_peer_blocked = False
        self.assertIsNotNone(
            coordinate(
                controller,
                vehicle,
                moving,
                now=1.1,
                own_cell=(0, 2),
            )
        )
        self.assertEqual(controller._peer_vacate_origin_cell, (0, 0))

        direct = PeerMotionIntent(
            "vehicle_a",
            1,
            1,
            1.0,
            0.35,
            (6, 2),
            (6, 1),
            5,
            "vehicle_a",
            task_sequence=1,
            trajectory=(
                TimedCell((6, 2), 0.0, 0.5),
                TimedCell((6, 1), 0.5, 1.0),
                TimedCell((6, 0), 1.0, 4.0),
            ),
        )
        controller, _, vehicle = controller_fixture()
        self.assertIsNotNone(coordinate(controller, vehicle, direct, now=1.0))
        self.assertEqual(controller._peer_vacate_origin_cell, (0, 0))

        idle = peer_intent(target=None, task_sequence=(1 << 64) - 1)
        controller, _, vehicle = controller_fixture()
        self.assertIsNone(coordinate(controller, vehicle, idle, now=1.0))
        self.assertIsNone(controller._peer_vacate_origin_cell)

    def test_priority_root_does_not_vacate_for_its_descendant(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_a", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor, resolution_m=0.5)

        def controller_fixture() -> tuple[RobotController, Vehicle]:
            navigation = Mock(
                transient_peer_blocked=True,
                motion_target=None,
                coordination_path_cells=Mock(
                    return_value=tuple((gx, 0) for gx in range(7))
                ),
                coordination_corridor=Mock(return_value=None),
                coordination_vacate_path=Mock(return_value=((0.25, 1.75),)),
                coordination_detours=Mock(return_value=()),
            )
            controller = RobotController(navigation)
            controller._schedule_temporal_motion = Mock(
                side_effect=lambda desired, **kwargs: (
                    desired,
                    kwargs["own"].target_cell,
                    kwargs["own"],
                    False,
                )
            )
            return controller, Vehicle(0.25, 0.25, radius=0.5, now=0.0)

        own = PeerMotionIntent(
            "vehicle_a",
            1,
            1,
            1.0,
            0.35,
            (0, 0),
            (1, 0),
            0,
            "vehicle_a",
            task_sequence=1,
        )

        def blocker(*, root: str, reserved: bool) -> PeerMotionIntent:
            return PeerMotionIntent(
                "vehicle_b",
                1,
                1,
                1.0,
                0.35,
                (6, 0),
                (5, 0),
                10,
                root,
                reserved,
                task_sequence=2,
                trajectory=(
                    TimedCell((6, 0), 0.0, 0.5),
                    TimedCell((5, 0), 0.5, 4.0),
                ),
            )

        def coordinate(
            controller: RobotController,
            vehicle: Vehicle,
            peer: PeerMotionIntent,
        ) -> tuple[float, float] | None:
            return controller._transient_peer_vacate(
                own=own,
                vehicle=vehicle,
                anchor=anchor,
                pose=PoseEstimate(
                    anchor.anchor_id,
                    0.25,
                    0.25,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.0,
                    1,
                ),
                local_map=local_map,
                peers={
                    "vehicle_b": peer_state(
                        "vehicle_b", 3.25, 0.25, 0.0
                    )
                },
                peer_motion_intents=(peer,),
                coordination_map=None,
                now=1.0,
            )

        descendant_controller, descendant_vehicle = controller_fixture()
        descendant_result = coordinate(
            descendant_controller,
            descendant_vehicle,
            blocker(root="vehicle_a", reserved=False),
        )
        independent_controller, independent_vehicle = controller_fixture()
        independent_result = coordinate(
            independent_controller,
            independent_vehicle,
            blocker(root="vehicle_b", reserved=True),
        )

        self.assertIsNone(descendant_result)
        self.assertIsNone(descendant_controller._peer_vacate_origin_cell)
        self.assertIsNotNone(independent_result)
        self.assertEqual(independent_controller._peer_vacate_origin_cell, (0, 0))

        independent_controller.navigation.transient_peer_blocked = False
        independent_controller.navigation.motion_target = (0.75, 0.25)
        independent_controller._schedule_temporal_motion = Mock(
            side_effect=lambda desired, **kwargs: (
                desired,
                kwargs["own"].target_cell,
                kwargs["own"],
                True,
            )
        )
        inherited_result = independent_controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=independent_vehicle,
            vehicle_id="vehicle_a",
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                0.25,
                0.25,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.1,
                2,
            ),
            local_map=local_map,
            now=1.1,
            peer_states=(peer_state("vehicle_b", 3.25, 0.25, 0.0),),
            peer_motion_intents=(
                blocker(root="vehicle_a", reserved=False),
            ),
        )

        retained_controller, retained_vehicle = controller_fixture()
        self.assertIsNotNone(
            coordinate(
                retained_controller,
                retained_vehicle,
                blocker(root="vehicle_b", reserved=True),
            )
        )
        retained_controller.navigation.transient_peer_blocked = False
        retained_controller.navigation.motion_target = (0.75, 0.25)
        retained_controller._schedule_temporal_motion = Mock(
            side_effect=lambda desired, **kwargs: (
                desired,
                kwargs["own"].target_cell,
                kwargs["own"],
                True,
            )
        )
        retained_result = retained_controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=retained_vehicle,
            vehicle_id="vehicle_a",
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                0.25,
                0.25,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.1,
                2,
            ),
            local_map=local_map,
            now=1.1,
            peer_states=(peer_state("vehicle_b", 3.25, 0.25, 0.0),),
            peer_motion_intents=(
                blocker(root="vehicle_b", reserved=True),
            ),
        )

        self.assertEqual(inherited_result, (0.0, 0.0))
        self.assertEqual(
            independent_controller._coordination_wait_reason,
            "space_time_reservation",
        )
        self.assertIsNone(independent_controller._peer_vacate_origin_cell)
        self.assertEqual(retained_result, (0.0, 0.0))
        self.assertEqual(
            retained_controller._coordination_wait_reason,
            "peer_vacate",
        )
        self.assertEqual(retained_controller._peer_vacate_origin_cell, (0, 0))

    def test_implicit_vacate_debounces_clear_before_route_blocker_rebind(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_c", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            transient_peer_blocked=True,
            motion_target=None,
            coordination_path_cells=Mock(
                return_value=((2, 0), (3, 0), (4, 0))
            ),
            coordination_vacate_path=Mock(return_value=((0.5, 3.5),)),
            coordination_detours=Mock(return_value=()),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(0.5, 0.5, radius=0.5, now=0.0)
        controller._schedule_temporal_motion = Mock(
            side_effect=lambda desired, **kwargs: (
                desired,
                kwargs["own"].target_cell,
                kwargs["own"],
                False,
            )
        )

        def linked_intent(
            vehicle_id: str,
            *,
            current: tuple[int, int],
            target: tuple[int, int] | None,
            trajectory: tuple[TimedCell, ...],
        ) -> PeerMotionIntent:
            return PeerMotionIntent(
                vehicle_id,
                1,
                1,
                1.0,
                0.35,
                current,
                target,
                5,
                "vehicle_a",
                True,
                trajectory=trajectory,
            )

        root = linked_intent(
            "vehicle_a",
            current=(8, 0),
            target=(9, 0),
            trajectory=(TimedCell((8, 0), 0.0, 4.0),),
        )
        far_same_owner = linked_intent(
            "vehicle_d",
            current=(8, 8),
            target=(9, 8),
            trajectory=(TimedCell((8, 8), 0.0, 4.0),),
        )
        controller._transient_peer_vacate(
            own=intent(
                "vehicle_c",
                current=(0, 0),
                target=(1, 0),
                wait_ticks=0,
            ),
            vehicle=vehicle,
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                0.5,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                10,
            ),
            local_map=local_map,
            peers={"vehicle_a": peer_state("vehicle_a", 8.5, 0.5, 0.0)},
            peer_motion_intents=(root,),
            coordination_map=None,
            now=1.0,
        )
        navigation.transient_peer_blocked = False

        def coordinate(
            now: float,
            peer_intents: tuple[PeerMotionIntent, ...],
        ) -> tuple[float, float] | None:
            states = tuple(
                peer_state(
                    peer.source_vehicle_id,
                    peer.current_cell[0] + 0.5,
                    peer.current_cell[1] + 0.5,
                    0.0,
                )
                for peer in peer_intents
            )
            return controller._transient_peer_vacate(
                own=intent(
                    "vehicle_c",
                    current=(0, 3),
                    target=(1, 3),
                    wait_ticks=0,
                ),
                vehicle=vehicle,
                anchor=anchor,
                pose=PoseEstimate(
                    anchor.anchor_id,
                    0.5,
                    3.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    now,
                    round(now * 10),
                ),
                local_map=local_map,
                peers={state.source_vehicle_id: state for state in states},
                peer_motion_intents=peer_intents,
                coordination_map=None,
                now=now,
            )

        self.assertEqual(
            coordinate(1.1, (root, far_same_owner)),
            (0.0, 0.0),
        )
        self.assertEqual(controller._yield_clear_ticks, 1)
        self.assertEqual(controller._peer_vacate_origin_cell, (0, 0))
        self.assertEqual(
            controller._peer_vacate_route_cells,
            ((2, 0), (3, 0), (4, 0)),
        )

        route_blocker = linked_intent(
            "vehicle_b",
            current=(5, 1),
            target=(3, 0),
            trajectory=(
                TimedCell((5, 1), 0.0, 0.5),
                TimedCell((4, 0), 0.5, 1.0),
                TimedCell((3, 0), 1.0, 4.0),
            ),
        )
        self.assertEqual(
            coordinate(1.2, (root, route_blocker, far_same_owner)),
            (0.0, 0.0),
        )
        self.assertEqual(controller._yielding_for, "vehicle_b")
        self.assertEqual(controller.motion_intent[2], "vehicle_a")
        self.assertEqual(controller._yield_clear_ticks, 0)

        blocker_clear = linked_intent(
            "vehicle_b",
            current=(8, 7),
            target=(9, 7),
            trajectory=(TimedCell((8, 7), 0.0, 4.0),),
        )
        clear_peers = root, blocker_clear, far_same_owner
        for now in (1.3, 1.4):
            self.assertEqual(coordinate(now, clear_peers), (0.0, 0.0))
            self.assertIsNotNone(controller._peer_vacate_origin_cell)

        self.assertIsNone(coordinate(1.5, clear_peers))
        self.assertIsNone(controller._peer_vacate_origin_cell)
        self.assertIsNone(controller._yielding_for)

    def test_implicit_vacate_releases_only_after_the_source_is_clear(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_b", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            transient_peer_blocked=True,
            motion_target=None,
            coordination_path_cells=Mock(
                return_value=((1, 0), (2, 0), (3, 0))
            ),
            coordination_vacate_path=Mock(return_value=((1.5, 1.5),)),
            coordination_detours=Mock(return_value=()),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(1.5, 0.5, radius=0.1, now=0.0)

        def coordinate(
            *,
            now: float,
            pose_y_m: float,
            source_current: tuple[int, int],
            source_target: tuple[int, int],
        ) -> tuple[float, float]:
            return coordinate_once(
                controller,
                vehicle,
                anchor,
                local_map,
                vehicle_id="vehicle_b",
                now=now,
                position_m=(1.5, pose_y_m),
                peer_states=(
                    peer_state(
                        "vehicle_a",
                        source_current[0] + 0.5,
                        source_current[1] + 0.5,
                        0.0,
                    ),
                ),
                peer_intents=(
                    intent(
                        "vehicle_a",
                        current=source_current,
                        target=source_target,
                        wait_ticks=5,
                        reserved=True,
                    ),
                ),
            )

        coordinate(
            now=1.0,
            pose_y_m=0.5,
            source_current=(1, 0),
            source_target=(3, 0),
        )
        navigation.transient_peer_blocked = False
        navigation.motion_target = (2.5, 1.5)
        held = coordinate(
            now=1.1,
            pose_y_m=1.5,
            source_current=(1, 0),
            source_target=(3, 0),
        )

        self.assertEqual(held, (0.0, 0.0))
        self.assertEqual(
            controller._yielding_for,
            "vehicle_a",
        )

        navigation.coordination_path_cells.return_value = ((1, 1), (2, 1))
        coordinate(
            now=1.2,
            pose_y_m=1.5,
            source_current=(4, 0),
            source_target=(5, 0),
        )
        coordinate(
            now=1.3,
            pose_y_m=1.5,
            source_current=(4, 0),
            source_target=(5, 0),
        )
        coordinate(
            now=1.4,
            pose_y_m=1.5,
            source_current=(4, 0),
            source_target=(5, 0),
        )

        self.assertIsNone(controller._yielding_for)

    def test_implicit_vacate_releases_after_source_clear_without_an_escape_path(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_c", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        vehicle = Vehicle(0.5, 0.5, radius=0.1, now=0.0)
        blocking = intent(
            "vehicle_a",
            current=(1, 0),
            target=(2, 0),
            wait_ticks=5,
            reserved=True,
        )
        clear = intent(
            "vehicle_a",
            current=(8, 0),
            target=(9, 0),
            wait_ticks=5,
            reserved=True,
        )

        def active_controller() -> tuple[RobotController, Mock]:
            navigation = Mock(
                transient_peer_blocked=True,
                motion_target=None,
                coordination_path_cells=Mock(
                    return_value=((0, 0), (1, 0), (2, 0))
                ),
                coordination_vacate_path=Mock(return_value=()),
                coordination_detours=Mock(return_value=()),
            )
            controller = RobotController(navigation)
            self.assertEqual(
                coordinate(controller, navigation, 1.0, blocking),
                (0.0, 0.0),
            )
            self.assertEqual(controller._peer_vacate_origin_cell, (0, 0))
            return controller, navigation

        def coordinate(
            controller: RobotController,
            navigation: Mock,
            now: float,
            source: PeerMotionIntent | None,
        ) -> tuple[float, float] | None:
            return controller._transient_peer_vacate(
                own=intent(
                    "vehicle_c",
                    current=(0, 0),
                    target=(1, 0),
                    wait_ticks=0,
                ),
                vehicle=vehicle,
                anchor=anchor,
                pose=PoseEstimate(
                    anchor.anchor_id,
                    0.5,
                    0.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    now,
                    round(now * 10),
                ),
                local_map=local_map,
                peers=(
                    {}
                    if source is None
                    else {
                        source.source_vehicle_id: peer_state(
                            source.source_vehicle_id,
                            source.current_cell[0] + 0.5,
                            source.current_cell[1] + 0.5,
                            0.0,
                        )
                    }
                ),
                peer_motion_intents=(() if source is None else (source,)),
                coordination_map=None,
                now=now,
            )

        controller, navigation = active_controller()
        navigation.transient_peer_blocked = False
        for now in (1.1, 1.2):
            self.assertEqual(
                coordinate(controller, navigation, now, clear),
                (0.0, 0.0),
            )
            self.assertIsNotNone(controller._peer_vacate_origin_cell)
        self.assertIsNone(coordinate(controller, navigation, 1.3, clear))
        self.assertIsNone(controller._peer_vacate_origin_cell)

        for case_id, source, transient in (
            ("missing-source", None, False),
            ("route-overlap", blocking, False),
            ("still-transient", clear, True),
        ):
            with self.subTest(case_id=case_id):
                controller, navigation = active_controller()
                navigation.transient_peer_blocked = transient
                for tick in range(1, 5):
                    self.assertEqual(
                        coordinate(
                            controller,
                            navigation,
                            2.0 + tick / 10,
                            source,
                        ),
                        (0.0, 0.0),
                    )
                self.assertIsNotNone(controller._peer_vacate_origin_cell)

    def test_implicit_vacate_keeps_the_planner_route_axis(self) -> None:
        anchor = AnchorSpec("spawn_b", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            transient_peer_blocked=True,
            motion_target=None,
            coordination_path_cells=Mock(
                return_value=((30, 10), (29, 10), (28, 10))
            ),
            coordination_vacate_path=Mock(
                side_effect=(((27.5, 7.5),), ((31.5, 11.5),))
            ),
            coordination_detours=Mock(return_value=()),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(31.5, 11.5, radius=0.8, now=0.0)
        controller._schedule_temporal_motion = Mock(
            side_effect=lambda desired, **kwargs: (
                desired,
                kwargs["own"].target_cell,
                kwargs["own"],
                False,
            )
        )
        owner = intent(
            "vehicle_a",
            current=(32, 11),
            target=(33, 11),
            wait_ticks=5,
            reserved=True,
        )

        first = controller._transient_peer_vacate(
            own=intent(
                "vehicle_b",
                current=(31, 11),
                target=(30, 10),
                wait_ticks=0,
            ),
            vehicle=vehicle,
            anchor=anchor,
            pose=PoseEstimate(
                anchor.anchor_id,
                31.5,
                11.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                10,
            ),
            local_map=local_map,
            peers={"vehicle_a": peer_state("vehicle_a", 32.5, 11.5, 0.0)},
            peer_motion_intents=(owner,),
            coordination_map=None,
            now=1.0,
        )

        self.assertIsNotNone(first)
        navigation.transient_peer_blocked = False

        def clear(now: float) -> tuple[float, float] | None:
            return controller._transient_peer_vacate(
                own=intent(
                    "vehicle_b",
                    current=(27, 7),
                    target=(28, 8),
                    wait_ticks=0,
                ),
                vehicle=vehicle,
                anchor=anchor,
                pose=PoseEstimate(
                    anchor.anchor_id,
                    27.5,
                    7.5,
                    0.0,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    now,
                    round(now * 10),
                ),
                local_map=local_map,
                peers={
                    "vehicle_a": peer_state("vehicle_a", 40.5, 11.5, 0.0)
                },
                peer_motion_intents=(
                    intent(
                        "vehicle_a",
                        current=(40, 11),
                        target=(41, 11),
                        wait_ticks=5,
                        reserved=True,
                    ),
                ),
                coordination_map=None,
                now=now,
            )

        self.assertEqual(clear(1.1), (0.0, 0.0))
        self.assertEqual(clear(1.2), (0.0, 0.0))
        self.assertEqual(navigation.coordination_vacate_path.call_count, 1)
        self.assertIsNone(clear(1.3))
        self.assertIsNone(controller._yielding_for)

    def test_implicit_vacate_attributes_the_blocking_goal_reservation(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn_b", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            transient_peer_blocked=True,
            motion_target=None,
            coordination_path_cells=Mock(
                return_value=((1, 0), (2, 0), (3, 0))
            ),
            coordination_vacate_path=Mock(return_value=((1.5, 1.5),)),
            coordination_detours=Mock(return_value=()),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(1.5, 0.5, radius=0.1, now=0.0)

        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_b",
            now=1.0,
            position_m=(1.5, 0.5),
            peer_states=(peer_state("vehicle_a", 0.5, 0.5, 0.0),),
            peer_intents=(
                intent(
                    "vehicle_a",
                    current=(0, 0),
                    target=(3, 0),
                    wait_ticks=5,
                    reserved=True,
                ),
            ),
            desired=(0.0, 0.0),
        )
        self.assertEqual(controller.motion_intent[2], "vehicle_a")

        far_owner = PeerMotionIntent(
            "vehicle_a",
            1,
            2,
            1.1,
            0.35,
            (8, 8),
            (9, 8),
            5,
            "vehicle_a",
            True,
            trajectory=(
                TimedCell((8, 8), 0.0, 0.5),
                TimedCell((9, 8), 0.5, 4.0),
            ),
        )
        blocking_goal = PeerMotionIntent(
            "vehicle_c",
            1,
            2,
            1.1,
            0.35,
            (2, 0),
            None,
            0,
            "vehicle_c",
            False,
            trajectory=(TimedCell((2, 0), 0.0, 4.0),),
            goal_hold=True,
        )
        coordinate_once(
            controller,
            vehicle,
            anchor,
            local_map,
            vehicle_id="vehicle_b",
            now=1.1,
            position_m=(1.5, 1.5),
            peer_states=(
                peer_state("vehicle_a", 8.5, 8.5, 0.0),
                peer_state("vehicle_c", 2.5, 0.5, 0.0),
            ),
            peer_intents=(far_owner, blocking_goal),
            desired=(0.0, 0.0),
        )

        self.assertEqual(
            controller._yielding_for,
            "vehicle_c",
        )
        self.assertEqual(controller.motion_intent[2], "vehicle_a")
        self.assertIsNone(controller._peer_vacate_request_cell)

    def test_entering_corridor_discards_stale_sipp_trajectory(self) -> None:
        anchor = AnchorSpec("spawn_3", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            7.5,
            5.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        local_map = ObservedGrid(anchor)
        navigation = Mock(
            motion_target=None,
            coordination_corridor=Mock(
                side_effect=(None, ((9, 5), (13, 5)))
            ),
            coordination_path_cells=Mock(return_value=((7, 5),)),
            coordination_detours=Mock(return_value=((7.5, 7.5),)),
        )
        controller = RobotController(navigation)
        vehicle = Vehicle(7.5, 5.5, now=0.0)
        remote_state = peer_state("vehicle_1", 15.5, 5.5, 0.0)

        controller._coordinate_desired(
            (0.0, 0.0),
            vehicle=vehicle,
            vehicle_id="vehicle_3",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.0,
            peer_states=(remote_state,),
            peer_motion_intents=(
                intent(
                    "vehicle_1",
                    current=(15, 5),
                    target=(15, 5),
                    wait_ticks=0,
                ),
            ),
            coordination_ready=True,
            expected_peer_vehicle_ids=("vehicle_1",),
        )
        self.assertEqual(len(controller.temporal_motion_intent[3]), 1)

        controller._coordinate_desired(
            (0.0, 0.0),
            vehicle=vehicle,
            vehicle_id="vehicle_3",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.1,
            peer_states=(remote_state,),
            peer_motion_intents=(
                intent(
                    "vehicle_1",
                    current=(15, 5),
                    target=(14, 5),
                    wait_ticks=0,
                    reserved=True,
                    corridor=CorridorDescriptor((13, 5), (9, 5)),
                ),
            ),
            coordination_ready=True,
            expected_peer_vehicle_ids=("vehicle_1",),
        )

        self.assertEqual(controller.motion_intent[0], (7.5, 7.5))
        self.assertEqual(controller.temporal_motion_intent[3], ())

    def test_intent_transport_is_strict_ordered_isolated_and_leased(self) -> None:
        first_anchor = AnchorSpec("spawn_1", 10.0, 5.0, 0.0)
        second_anchor = AnchorSpec("spawn_2", 20.0, 5.0, 0.0)
        now = [4.0]
        source = MapSyncState(
            "session_1",
            "vehicle_1",
            first_anchor,
            1.0,
            state_generation=1,
        )
        receiver = MapSyncState(
            "session_1",
            "vehicle_2",
            second_anchor,
            1.0,
            clock=lambda: now[0],
            state_generation=2,
        )
        source.configure_network(
            "peer_1", {"vehicle_2": ("peer_2", second_anchor)}
        )
        receiver.configure_network(
            "peer_2", {"vehicle_1": ("peer_1", first_anchor)}
        )
        source_pose = PoseEstimate(
            first_anchor.anchor_id,
            0.25,
            0.25,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            4.0,
            1,
        )
        source.record_motion_intent(
            source_pose,
            target_m=(1.5, 0.5),
            wait_ticks=3,
            priority_owner_id="vehicle_2",
            reserved=True,
            corridor=CorridorDescriptor((10, 5), (15, 5)),
            timestamp_s=4.0,
        )
        payload = source.prepare_motion_intent()
        with self.assertRaises(ValueError):
            source.record_motion_intent(
                source_pose,
                target_m=(1.5, 0.5),
                wait_ticks=0,
                priority_owner_id="vehicle_1",
                reserved=False,
                timestamp_s=4.0,
                trajectory=(
                    TimedCell((10, 5), 0.0, 0.0),
                    TimedCell((11, 5), 0.0, 4.0),
                ),
            )
        with self.assertRaises(ValueError):
            source.record_motion_intent(
                source_pose,
                target_m=(1.5, 0.5),
                wait_ticks=0,
                priority_owner_id="vehicle_1",
                reserved=False,
                timestamp_s=4.0,
                committed_until_offset_s=math.nextafter(
                    MOTION_COMMIT_HORIZON_S,
                    math.inf,
                ),
            )

        self.assertEqual(payload["protocol"], MOTION_INTENT_PROTOCOL)
        self.assertIsNone(payload["vacate_request"])
        self.assertEqual(
            payload["committed_until_offset_s"],
            MOTION_COMMIT_HORIZON_S,
        )
        self.assertEqual(payload["plan_generation"], 1)
        self.assertEqual(payload["priority"]["task_age_ticks"], 0)
        self.assertEqual(
            payload["trajectory"],
            [
                {
                    "cell": {"gx": 10, "gy": 5},
                    "enter_offset_s": 0.0,
                    "leave_offset_s": 0.0,
                },
                {
                    "cell": {"gx": 11, "gy": 5},
                    "enter_offset_s": 1.0,
                    "leave_offset_s": 4.0,
                },
            ],
        )
        self.assertEqual(
            payload["corridor"],
            {
                "entry_cell": {"gx": 10, "gy": 5},
                "exit_cell": {"gx": 15, "gy": 5},
            },
        )
        unknown_owner = deepcopy(payload)
        unknown_owner["priority"]["owner_vehicle_id"] = "unknown_vehicle"
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", unknown_owner)
        )
        malformed_corridors = []
        missing_exit = deepcopy(payload)
        missing_exit["corridor"].pop("exit_cell")
        malformed_corridors.append(missing_exit)
        unexpected_field = deepcopy(payload)
        unexpected_field["corridor"]["extra"] = True
        malformed_corridors.append(unexpected_field)
        diagonal = deepcopy(payload)
        diagonal["corridor"]["exit_cell"] = {"gx": 15, "gy": 6}
        malformed_corridors.append(diagonal)
        equal_endpoints = deepcopy(payload)
        equal_endpoints["corridor"]["exit_cell"] = {"gx": 10, "gy": 5}
        malformed_corridors.append(equal_endpoints)
        boolean_coordinate = deepcopy(payload)
        boolean_coordinate["corridor"]["entry_cell"]["gx"] = True
        malformed_corridors.append(boolean_coordinate)
        missing_corridor = deepcopy(payload)
        missing_corridor.pop("corridor")
        malformed_corridors.append(missing_corridor)
        missing_plan_generation = deepcopy(payload)
        missing_plan_generation.pop("plan_generation")
        malformed_corridors.append(missing_plan_generation)
        reversed_times = deepcopy(payload)
        reversed_times["trajectory"][1]["enter_offset_s"] = -0.1
        malformed_corridors.append(reversed_times)
        zero_travel_time = deepcopy(payload)
        zero_travel_time["trajectory"][1]["enter_offset_s"] = 0.0
        malformed_corridors.append(zero_travel_time)
        commit_past_horizon = deepcopy(payload)
        commit_past_horizon["committed_until_offset_s"] = math.nextafter(
            MOTION_COMMIT_HORIZON_S,
            math.inf,
        )
        malformed_corridors.append(commit_past_horizon)
        excessive_margin = deepcopy(payload)
        excessive_margin["safety_time_margin_s"] = 10.1
        malformed_corridors.append(excessive_margin)
        invalid_task_age = deepcopy(payload)
        invalid_task_age["priority"]["task_age_ticks"] = -1
        malformed_corridors.append(invalid_task_age)
        for malformed in malformed_corridors:
            self.assertFalse(
                receiver.receive_transport("peer_1", "vehicle_1", malformed)
            )
        self.assertTrue(
            receiver.receive_transport("peer_1", "vehicle_1", payload)
        )
        received = receiver.peer_motion_intents()[0]
        self.assertEqual(received.current_cell, (10, 5))
        self.assertEqual(received.target_cell, (11, 5))
        self.assertEqual(received.priority_owner_id, "vehicle_2")
        self.assertEqual(received.received_at_s, 4.0)
        self.assertTrue(received.reserved)
        self.assertEqual(
            received.corridor,
            CorridorDescriptor((10, 5), (15, 5)),
        )
        single_cell_hold = deepcopy(payload)
        single_cell_hold["sequence"] = 2
        single_cell_hold["timestamp_s"] = 4.1
        single_cell_hold["plan_generation"] = 2
        single_cell_hold["target_cell"] = None
        single_cell_hold["trajectory"] = [
            {
                "cell": {"gx": 10, "gy": 5},
                "enter_offset_s": 0.0,
                "leave_offset_s": 4.0,
            }
        ]
        self.assertTrue(
            receiver.receive_transport(
                "peer_1",
                "vehicle_1",
                single_cell_hold,
            )
        )
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", payload)
        )
        changed_same_plan = deepcopy(payload)
        changed_same_plan["sequence"] += 1
        changed_same_plan["timestamp_s"] += 0.1
        changed_same_plan["target_cell"] = {"gx": 12, "gy": 5}
        changed_same_plan["trajectory"][1]["cell"] = {"gx": 12, "gy": 5}
        self.assertFalse(
            receiver.receive_transport(
                "peer_1",
                "vehicle_1",
                changed_same_plan,
            )
        )
        unexpected = deepcopy(payload)
        unexpected["sequence"] += 1
        unexpected["extra"] = True
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", unexpected)
        )
        self.assertEqual(receiver.peer_evidence("vehicle_1"), {})
        self.assertIsNone(receiver.prepare_delta())
        now[0] += MOTION_INTENT_TTL_S + 0.01
        self.assertEqual(receiver.peer_motion_intents(), ())

    def test_vacate_request_round_trips_with_motion_intent(self) -> None:
        self.assertEqual(MOTION_INTENT_PROTOCOL, "mockvehicle2d-motion-intent/4")
        request = VacateRequest(
            "vehicle_2",
            (1, 0),
            ((0, 0), (1, 0), (3, 0)),
        )
        receiver, payload = vacate_request_payload()
        self.assertEqual(
            payload["vacate_request"],
            {
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1, "gy": 0},
                "route_cells": [
                    {"gx": 0, "gy": 0},
                    {"gx": 1, "gy": 0},
                    {"gx": 3, "gy": 0},
                ],
            },
        )
        self.assertTrue(receiver.receive_transport("peer_1", "vehicle_1", payload))
        self.assertEqual(receiver.peer_motion_intents()[0].vacate_request, request)

    def test_vacate_request_is_required_and_has_exact_fields(self) -> None:
        receiver, payload = vacate_request_payload()
        malformed_payloads = []
        missing_request = deepcopy(payload)
        missing_request.pop("vacate_request")
        malformed_payloads.append(missing_request)
        missing_cell = deepcopy(payload)
        missing_cell["vacate_request"].pop("cell")
        malformed_payloads.append(missing_cell)
        missing_route = deepcopy(payload)
        missing_route["vacate_request"].pop("route_cells")
        malformed_payloads.append(missing_route)
        extra_field = deepcopy(payload)
        extra_field["vacate_request"]["extra"] = True
        malformed_payloads.append(extra_field)
        invalid_type = deepcopy(payload)
        invalid_type["vacate_request"] = True
        malformed_payloads.append(invalid_type)

        for malformed in malformed_payloads:
            self.assertFalse(
                receiver.receive_transport("peer_1", "vehicle_1", malformed)
            )
        self.assertTrue(receiver.receive_transport("peer_1", "vehicle_1", payload))

    def test_vacate_request_route_is_an_array_of_two_to_sixty_four_cells(
        self,
    ) -> None:
        cell = (0, 0)
        for route in ((cell,), (cell,) * 65):
            with self.subTest(local_length=len(route)), self.assertRaises(
                ValueError
            ):
                VacateRequest("vehicle_2", (1, 0), route)

        receiver, payload = vacate_request_payload()
        for route in (
            None,
            {},
            [],
            [{"gx": 0, "gy": 0}],
            [{"gx": 0, "gy": 0}] * 65,
        ):
            malformed = deepcopy(payload)
            malformed["vacate_request"]["route_cells"] = route
            with self.subTest(remote_route=route):
                self.assertFalse(
                    receiver.receive_transport("peer_1", "vehicle_1", malformed)
                )

    def test_vacate_request_cell_rejects_boolean_and_out_of_range_coordinates(
        self,
    ) -> None:
        receiver, payload = vacate_request_payload()
        for field in ("cell", "route_cells"):
            for coordinate in (True, MAX_GRID_COORDINATE + 1):
                route = ((0, 0), (coordinate, 0))
                with self.subTest(
                    endpoint="local",
                    field=field,
                    coordinate=coordinate,
                ), self.assertRaises(ValueError):
                    VacateRequest(
                        "vehicle_2",
                        (coordinate, 0) if field == "cell" else (1, 0),
                        route if field == "route_cells" else ((0, 0), (1, 0)),
                    )
                malformed = deepcopy(payload)
                cell = (
                    malformed["vacate_request"]["cell"]
                    if field == "cell"
                    else malformed["vacate_request"]["route_cells"][1]
                )
                cell["gx"] = coordinate
                with self.subTest(
                    endpoint="remote",
                    field=field,
                    coordinate=coordinate,
                ):
                    self.assertFalse(
                        receiver.receive_transport(
                            "peer_1", "vehicle_1", malformed
                        )
                    )
        self.assertTrue(receiver.receive_transport("peer_1", "vehicle_1", payload))

    def test_vacate_request_route_is_contiguous_after_anchor_quantization(
        self,
    ) -> None:
        quantized = _global_coordination_cells(
            AnchorSpec("rotated", 0.0, 0.0, math.pi / 4),
            ((0, 0), (1, 1)),
            0.5,
        )
        self.assertEqual(quantized, ((0, 0), (0, 2)))
        self.assertIsInstance(
            VacateRequest("vehicle_2", (1, 0), quantized),
            VacateRequest,
        )

        for route in (((0, 0), (0, 0)), ((0, 0), (3, 0))):
            with self.subTest(local_route=route), self.assertRaises(ValueError):
                VacateRequest("vehicle_2", (1, 0), route)

        receiver, payload = vacate_request_payload()
        for route in (
            [{"gx": 0, "gy": 0}, {"gx": 0, "gy": 0}],
            [{"gx": 0, "gy": 0}, {"gx": 3, "gy": 0}],
        ):
            malformed = deepcopy(payload)
            malformed["vacate_request"]["route_cells"] = route
            with self.subTest(remote_route=route):
                self.assertFalse(
                    receiver.receive_transport("peer_1", "vehicle_1", malformed)
                )

    def test_vacate_request_route_starts_at_outer_current_cell(self) -> None:
        source, _, pose = motion_sync_states()
        with self.assertRaises(ValueError):
            source.record_motion_intent(
                pose,
                target_m=None,
                wait_ticks=0,
                priority_owner_id="vehicle_1",
                reserved=False,
                timestamp_s=1.0,
                vacate_request=VacateRequest(
                    "vehicle_2",
                    (1, 0),
                    ((1, 0), (2, 0)),
                ),
            )

        receiver, payload = vacate_request_payload()
        payload["vacate_request"]["route_cells"][0] = {"gx": 1, "gy": 0}
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", payload)
        )

    def test_motion_intent_v4_rejects_v3(self) -> None:
        receiver, payload = vacate_request_payload()
        malformed = deepcopy(payload)
        malformed["protocol"] = "mockvehicle2d-motion-intent/3"
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", malformed)
        )
        self.assertTrue(receiver.receive_transport("peer_1", "vehicle_1", payload))

    def test_vacate_request_rejects_invalid_vehicle_identifier(self) -> None:
        with self.assertRaises(ValueError):
            VacateRequest(
                "invalid vehicle",
                (1, 0),
                ((0, 0), (1, 0)),
            )

    def test_vacate_request_targets_a_known_remote_vehicle_on_both_endpoints(
        self,
    ) -> None:
        source, _, pose = motion_sync_states()
        for vehicle_id in ("vehicle_1", "unknown_vehicle"):
            with self.subTest(
                endpoint="local", vehicle_id=vehicle_id
            ), self.assertRaises(ValueError):
                source.record_motion_intent(
                    pose,
                    target_m=None,
                    wait_ticks=0,
                    priority_owner_id="vehicle_1",
                    reserved=False,
                    timestamp_s=1.0,
                    vacate_request=VacateRequest(
                        vehicle_id,
                        (1, 0),
                        ((0, 0), (1, 0), (3, 0)),
                    ),
                )

        receiver, payload = vacate_request_payload()
        for vehicle_id in ("vehicle_1", "unknown_vehicle"):
            malformed = deepcopy(payload)
            malformed["vacate_request"]["vehicle_id"] = vehicle_id
            with self.subTest(endpoint="remote", vehicle_id=vehicle_id):
                self.assertFalse(
                    receiver.receive_transport("peer_1", "vehicle_1", malformed)
                )
        self.assertTrue(receiver.receive_transport("peer_1", "vehicle_1", payload))

    def test_explicit_plan_generation_overflow_does_not_poison_sender(self) -> None:
        source, _, pose = motion_sync_states()
        common = {
            "wait_ticks": 0,
            "priority_owner_id": "vehicle_1",
            "reserved": True,
            "timestamp_s": 1.0,
        }
        source.record_motion_intent(
            pose,
            target_m=(1.5, 0.5),
            plan_generation=7,
            **common,
        )
        first = source.prepare_motion_intent()
        assert first is not None
        source.publish_motion_intent_result(first["sequence"], True)

        for invalid in (True, 0, -1, 1 << 64, "8", 8.0):
            with self.subTest(plan_generation=invalid), self.assertRaises(
                ValueError
            ):
                source.record_motion_intent(
                    pose,
                    target_m=(2.5, 0.5),
                    plan_generation=invalid,
                    **common,
                )
        with self.assertRaises(ValueError):
            source.record_motion_intent(
                pose,
                target_m=(2.5, 0.5),
                plan_generation=8,
                task_sequence=-1,
                **common,
            )
        with self.assertRaises(ValueError):
            source.record_motion_intent(
                pose,
                target_m=(2.5, 0.5),
                plan_generation=8,
                trajectory=(
                    TimedCell((0, 0), 0.0, 0.0),
                    TimedCell((2, 0), 0.0, 4.0),
                ),
                **common,
            )
        with self.assertRaises(ValueError):
            source.record_motion_intent(
                pose,
                target_m=None,
                plan_generation=8,
                trajectory=(TimedCell((0, 0), 0.0, 0.1),),
                committed_until_offset_s=0.8,
                goal_hold=True,
                **common,
            )

        source.record_motion_intent(
            pose,
            target_m=(3.5, 0.5),
            plan_generation=8,
            **common,
        )
        recovered = source.prepare_motion_intent()
        assert recovered is not None
        self.assertEqual(recovered["plan_generation"], 8)
        self.assertEqual(recovered["target_cell"], {"gx": 3, "gy": 0})

    def test_same_plan_generation_has_one_signature_on_both_endpoints(self) -> None:
        source, receiver, pose = motion_sync_states()
        common = {
            "wait_ticks": 0,
            "priority_owner_id": "vehicle_1",
            "reserved": True,
            "timestamp_s": 1.0,
            "plan_generation": 7,
        }
        source.record_motion_intent(
            pose,
            target_m=(1.5, 0.5),
            **common,
        )
        first = source.prepare_motion_intent()
        assert first is not None
        self.assertTrue(
            receiver.receive_transport("peer_1", "vehicle_1", first)
        )
        source.publish_motion_intent_result(first["sequence"], True)

        changed_records = (
            {"target_m": (2.5, 0.5)},
            {"target_m": (1.5, 0.5), "task_sequence": 1},
            {"target_m": (1.5, 0.5), "goal_hold": True},
        )
        for changed in changed_records:
            with self.subTest(sender=changed), self.assertRaises(ValueError):
                source.record_motion_intent(pose, **common, **changed)

        changed_payloads = []
        changed_path = deepcopy(first)
        changed_path["target_cell"] = {"gx": 2, "gy": 0}
        changed_path["trajectory"][1]["cell"] = {"gx": 2, "gy": 0}
        changed_payloads.append(changed_path)
        changed_task = deepcopy(first)
        changed_task["priority"]["task_sequence"] = 1
        changed_payloads.append(changed_task)
        changed_goal = deepcopy(first)
        changed_goal["goal_hold"] = True
        changed_payloads.append(changed_goal)
        for changed in changed_payloads:
            changed["sequence"] = 2
            changed["timestamp_s"] = 1.1
            with self.subTest(receiver=changed):
                self.assertFalse(
                    receiver.receive_transport("peer_1", "vehicle_1", changed)
                )

        source.record_motion_intent(
            pose,
            target_m=(2.5, 0.5),
            **{**common, "plan_generation": 8},
        )
        next_plan = source.prepare_motion_intent()
        assert next_plan is not None
        self.assertTrue(
            receiver.receive_transport("peer_1", "vehicle_1", next_plan)
        )

    def test_vacate_request_changes_require_a_new_plan_generation(self) -> None:
        source, receiver, pose = motion_sync_states()
        common = {
            "target_m": None,
            "wait_ticks": 0,
            "priority_owner_id": "vehicle_1",
            "reserved": False,
            "timestamp_s": 1.0,
            "plan_generation": 7,
        }
        source.record_motion_intent(
            pose,
            vacate_request=VacateRequest(
                "vehicle_2",
                (1, 0),
                ((0, 0), (1, 0), (3, 0)),
            ),
            **common,
        )
        first = source.prepare_motion_intent()
        assert first is not None
        self.assertTrue(receiver.receive_transport("peer_1", "vehicle_1", first))
        source.publish_motion_intent_result(first["sequence"], True)

        for request in (
            VacateRequest(
                "vehicle_2",
                (1, 0),
                ((0, 0), (2, 0), (3, 0)),
            ),
            VacateRequest(
                "vehicle_2",
                (2, 0),
                ((0, 0), (1, 0), (3, 0)),
            ),
            None,
        ):
            with self.subTest(sender=request), self.assertRaises(ValueError):
                source.record_motion_intent(
                    pose,
                    vacate_request=request,
                    **common,
                )

        for request in (
            {
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1, "gy": 0},
                "route_cells": [
                    {"gx": 0, "gy": 0},
                    {"gx": 2, "gy": 0},
                    {"gx": 3, "gy": 0},
                ],
            },
            {
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 2, "gy": 0},
                "route_cells": [
                    {"gx": 0, "gy": 0},
                    {"gx": 1, "gy": 0},
                    {"gx": 3, "gy": 0},
                ],
            },
            None,
        ):
            changed = deepcopy(first)
            changed["sequence"] = 2
            changed["timestamp_s"] = 1.1
            changed["vacate_request"] = request
            with self.subTest(receiver=request):
                self.assertFalse(
                    receiver.receive_transport("peer_1", "vehicle_1", changed)
                )

        source.record_motion_intent(
            pose,
            vacate_request=None,
            **{**common, "plan_generation": 8},
        )
        next_plan = source.prepare_motion_intent()
        assert next_plan is not None
        self.assertTrue(receiver.receive_transport("peer_1", "vehicle_1", next_plan))

    def test_automatic_plan_generation_overflow_is_atomic(self) -> None:
        source, _, pose = motion_sync_states()
        common = {
            "wait_ticks": 0,
            "priority_owner_id": "vehicle_1",
            "reserved": True,
            "timestamp_s": 1.0,
        }
        source.record_motion_intent(
            pose,
            target_m=(1.5, 0.5),
            plan_generation=(1 << 64) - 1,
            **common,
        )
        first = source.prepare_motion_intent()
        assert first is not None
        source.publish_motion_intent_result(first["sequence"], True)

        with self.assertRaises(ValueError):
            source.record_motion_intent(
                pose,
                target_m=(2.5, 0.5),
                **common,
            )

        source.record_motion_intent(
            pose,
            target_m=(1.5, 0.5),
            plan_generation=(1 << 64) - 1,
            **common,
        )
        recovered = source.prepare_motion_intent()
        assert recovered is not None
        self.assertEqual(recovered["plan_generation"], (1 << 64) - 1)
        self.assertEqual(recovered["target_cell"], {"gx": 1, "gy": 0})

    def test_short_hold_commit_must_not_outlive_trajectory(self) -> None:
        source, receiver, pose = motion_sync_states()
        trajectory = (TimedCell((0, 0), 0.0, 0.1),)
        common = {
            "target_m": None,
            "wait_ticks": 0,
            "priority_owner_id": "vehicle_1",
            "reserved": True,
            "timestamp_s": 1.0,
            "plan_generation": 1,
            "trajectory": trajectory,
            "goal_hold": True,
        }

        with self.assertRaises(ValueError):
            source.record_motion_intent(
                pose,
                committed_until_offset_s=0.8,
                **common,
            )

        source.record_motion_intent(
            pose,
            committed_until_offset_s=0.1,
            **common,
        )
        payload = source.prepare_motion_intent()
        assert payload is not None
        self.assertEqual(payload["committed_until_offset_s"], 0.1)
        self.assertTrue(
            receiver.receive_transport("peer_1", "vehicle_1", payload)
        )

        excessive = deepcopy(payload)
        excessive["sequence"] = 2
        excessive["timestamp_s"] = 1.1
        excessive["committed_until_offset_s"] = 0.8
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", excessive)
        )

    def test_runtime_arbitration_promotes_an_older_high_id_waiter(self) -> None:
        anchor = AnchorSpec("spawn_4", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            0.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        local_map = ObservedGrid(anchor)
        peer_intent = intent(
            "vehicle_1",
            current=(2, 0),
            target=(1, 0),
            wait_ticks=0,
        )
        peer = peer_state("vehicle_1", 2.5, 0.5, -0.5)

        older = RobotController(
            Mock(
                motion_target=(1.5, 0.5),
                coordination_detours=Mock(return_value=()),
            )
        )
        older._reservation_wait_ticks = 8
        older._last_coordination_cell = (0, 0)
        desired = older._coordinate_desired(
            (0.5, 0.0),
            vehicle=Vehicle(0.5, 0.5, now=0.0),
            vehicle_id="vehicle_4",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.0,
            peer_states=(peer,),
            peer_motion_intents=(peer_intent,),
        )
        newcomer = RobotController(
            Mock(
                motion_target=(1.5, 0.5),
                coordination_detours=Mock(return_value=()),
            )
        )
        lexical_result = newcomer._coordinate_desired(
            (0.5, 0.0),
            vehicle=Vehicle(0.5, 0.5, now=0.0),
            vehicle_id="vehicle_4",
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=1.0,
            peer_states=(peer,),
            peer_motion_intents=(peer_intent,),
        )

        self.assertEqual(desired, (0.5, 0.0))
        self.assertTrue(older.motion_intent[3])
        self.assertEqual(lexical_result, (0.0, 0.0))
        self.assertTrue(newcomer.is_yielding)

    def test_edge_swap_inherits_priority_and_uses_one_bounded_detour(self) -> None:
        anchor = AnchorSpec("spawn_2", 0.0, 0.0, 0.0)
        pose = PoseEstimate(
            anchor.anchor_id,
            1.5,
            0.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        navigation = Mock(
            motion_target=(0.5, 0.5),
            coordination_detours=Mock(return_value=((1.5, 1.5),)),
        )
        controller = RobotController(navigation)
        requester = intent(
            "vehicle_1",
            current=(0, 0),
            target=(1, 0),
            wait_ticks=5,
            reserved=True,
        )

        controller._coordinate_desired(
            (0.5, 0.0),
            vehicle=Vehicle(1.5, 0.5, now=0.0),
            vehicle_id="vehicle_2",
            anchor=anchor,
            pose=pose,
            local_map=ObservedGrid(anchor),
            now=1.0,
            peer_states=(peer_state("vehicle_1", 0.5, 0.5, 0.5),),
            peer_motion_intents=(requester,),
        )

        target_m, _, owner_id, reserved, _ = controller.motion_intent
        self.assertEqual(target_m, (1.5, 1.5))
        self.assertEqual(owner_id, "vehicle_1")
        self.assertTrue(reserved)
        navigation.coordination_detours.assert_called_once()

    def test_two_independent_conflict_clusters_are_not_globally_serialized(self) -> None:
        local_map = ObservedGrid(AnchorSpec("spawn", 0.0, 0.0, 0.0))
        intents = (
            intent("vehicle_1", current=(0, 0), target=(1, 0), wait_ticks=0),
            intent("vehicle_2", current=(2, 0), target=(1, 0), wait_ticks=0),
            intent("vehicle_3", current=(10, 0), target=(11, 0), wait_ticks=0),
            intent("vehicle_4", current=(12, 0), target=(11, 0), wait_ticks=0),
        )

        for vehicle_id, x_m, peer_id, peer_x_m in (
            ("vehicle_1", 0.5, "vehicle_2", 2.5),
            ("vehicle_3", 10.5, "vehicle_4", 12.5),
        ):
            anchor = AnchorSpec(f"spawn_{vehicle_id}", 0.0, 0.0, 0.0)
            pose = PoseEstimate(
                anchor.anchor_id,
                x_m,
                0.5,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                1,
            )
            controller = RobotController(Mock(motion_target=(x_m + 1.0, 0.5)))
            desired = controller._coordinate_desired(
                (0.5, 0.0),
                vehicle=Vehicle(x_m, 0.5, now=0.0),
                vehicle_id=vehicle_id,
                anchor=anchor,
                pose=pose,
                local_map=local_map,
                now=1.0,
                peer_states=(peer_state(peer_id, peer_x_m, 0.5, -0.5),),
                peer_motion_intents=intents,
            )
            self.assertEqual(desired, (0.5, 0.0))
            self.assertTrue(controller.motion_intent[3])

    def test_four_corridor_waiters_take_over_fairly_after_each_owner_leaves(
        self,
    ) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        eastbound = CorridorDescriptor((10, 1), (30, 1))
        westbound = CorridorDescriptor((30, 1), (10, 1))
        positions = {
            "vehicle_1": ((8, 1), (9, 1), eastbound),
            "vehicle_2": ((7, 1), (8, 1), eastbound),
            "vehicle_3": ((32, 1), (31, 1), westbound),
            "vehicle_4": ((33, 1), (32, 1), westbound),
        }
        controllers: dict[str, RobotController] = {}
        for vehicle_id, (_, target_cell, corridor) in positions.items():
            target_m = tuple(
                (coordinate + 0.5) * local_map.resolution_m
                for coordinate in target_cell
            )
            navigation = Mock(motion_target=target_m)
            navigation.coordination_detours.return_value = ()
            controller = RobotController(navigation)
            controller._corridor = corridor
            controllers[vehicle_id] = controller

        active = list(sorted(controllers))
        published = tuple(
            intent(
                vehicle_id,
                current=current,
                target=target,
                wait_ticks=0,
                corridor=corridor,
            )
            for vehicle_id, (current, target, corridor) in positions.items()
        )
        owner_order = []
        last_wait = {vehicle_id: 0 for vehicle_id in active}
        retired: set[str] = set()

        def retired_intent(vehicle_id: str) -> PeerMotionIntent:
            offset = 100 + int(vehicle_id[-1])
            return intent(
                vehicle_id,
                current=(offset, 100),
                target=(offset + 1, 100),
                wait_ticks=0,
            )

        while active:
            confirmed = []
            for _ in range(4):
                next_published = [
                    retired_intent(vehicle_id)
                    for vehicle_id in sorted(retired)
                ]
                for vehicle_id in active:
                    current, target, _ = positions[vehicle_id]
                    pose_m = tuple(
                        (coordinate + 0.5) * local_map.resolution_m
                        for coordinate in current
                    )
                    controller = controllers[vehicle_id]
                    controller._coordinate_desired(
                        (0.5, 0.0),
                        vehicle=Vehicle(*pose_m, now=0.0),
                        vehicle_id=vehicle_id,
                        anchor=anchor,
                        pose=PoseEstimate(
                            anchor.anchor_id,
                            *pose_m,
                            0.0,
                            (0.0, 0.0, 0.0),
                            "nominal",
                            1.0,
                            1,
                        ),
                        local_map=local_map,
                        now=1.0,
                        peer_states=(),
                        peer_motion_intents=tuple(
                            item
                            for item in published
                            if item.source_vehicle_id != vehicle_id
                        ),
                        coordination_ready=True,
                        expected_peer_vehicle_ids=tuple(
                            peer_id
                            for peer_id in sorted(controllers)
                            if peer_id != vehicle_id
                        ),
                    )
                    _, wait_ticks, owner_id, reserved, corridor = (
                        controller.motion_intent
                    )
                    self.assertGreaterEqual(wait_ticks, last_wait[vehicle_id])
                    last_wait[vehicle_id] = wait_ticks
                    next_published.append(
                        intent(
                            vehicle_id,
                            current=current,
                            target=target,
                            wait_ticks=wait_ticks,
                            owner=owner_id,
                            reserved=reserved,
                            corridor=corridor,
                        )
                    )
                published = tuple(next_published)
                confirmed = [
                    vehicle_id
                    for vehicle_id in active
                    if controllers[vehicle_id].snapshot()["coordination"]["state"]
                    == "reserved"
                ]
                self.assertLessEqual(len(confirmed), 1)
                if confirmed:
                    break
            self.assertEqual(len(confirmed), 1)
            owner = confirmed[0]
            owner_order.append(owner)
            active.remove(owner)
            retired.add(owner)
            # A vehicle that leaves this corridor remains a fleet member and
            # publishes a fresh null descriptor.  Disappearance would be a
            # partition and must not authorize a replacement owner.
            published = tuple(
                item
                for item in published
                if item.source_vehicle_id in active
            ) + tuple(
                retired_intent(vehicle_id)
                for vehicle_id in sorted(retired)
            )

        self.assertEqual(
            owner_order,
            ["vehicle_1", "vehicle_2", "vehicle_3", "vehicle_4"],
        )

    def test_two_independent_corridors_can_hold_one_lease_each(self) -> None:
        anchor = AnchorSpec("spawn", 0.0, 0.0, 0.0)
        local_map = ObservedGrid(anchor)
        positions = {
            "vehicle_1": ((8, 1), CorridorDescriptor((10, 1), (30, 1))),
            "vehicle_2": ((32, 1), CorridorDescriptor((30, 1), (10, 1))),
            "vehicle_3": ((8, 5), CorridorDescriptor((10, 5), (30, 5))),
            "vehicle_4": ((32, 5), CorridorDescriptor((30, 5), (10, 5))),
        }
        initial = tuple(
            intent(
                vehicle_id,
                current=current,
                target=(current[0] + (1 if current[0] < 10 else -1), current[1]),
                wait_ticks=0,
                corridor=corridor,
            )
            for vehicle_id, (current, corridor) in positions.items()
        )
        controllers = {}
        published = initial
        for vehicle_id, (current, corridor) in positions.items():
            target = current[0] + (1 if current[0] < 10 else -1), current[1]
            target_m = tuple(
                (coordinate + 0.5) * local_map.resolution_m
                for coordinate in target
            )
            navigation = Mock(motion_target=target_m)
            navigation.coordination_detours.return_value = ()
            controller = RobotController(navigation)
            controller._corridor = corridor
            controllers[vehicle_id] = controller

        for _ in range(2):
            next_published = []
            for vehicle_id, (current, _) in positions.items():
                target = (
                    current[0] + (1 if current[0] < 10 else -1),
                    current[1],
                )
                pose_m = tuple(
                    (coordinate + 0.5) * local_map.resolution_m
                    for coordinate in current
                )
                controller = controllers[vehicle_id]
                controller._coordinate_desired(
                    (0.5, 0.0),
                    vehicle=Vehicle(*pose_m, now=0.0),
                    vehicle_id=vehicle_id,
                    anchor=anchor,
                    pose=PoseEstimate(
                        anchor.anchor_id,
                        *pose_m,
                        0.0,
                        (0.0, 0.0, 0.0),
                        "nominal",
                        1.0,
                        1,
                    ),
                    local_map=local_map,
                    now=1.0,
                    peer_states=(),
                    peer_motion_intents=tuple(
                        item
                        for item in published
                        if item.source_vehicle_id != vehicle_id
                    ),
                    coordination_ready=True,
                    expected_peer_vehicle_ids=tuple(
                        peer_id
                        for peer_id in sorted(controllers)
                        if peer_id != vehicle_id
                    ),
                )
                _, wait_ticks, owner_id, reserved, corridor = (
                    controller.motion_intent
                )
                next_published.append(
                    intent(
                        vehicle_id,
                        current=current,
                        target=target,
                        wait_ticks=wait_ticks,
                        owner=owner_id,
                        reserved=reserved,
                        corridor=corridor,
                    )
                )
            published = tuple(next_published)

        self.assertEqual(
            [
                vehicle_id
                for vehicle_id, controller in sorted(controllers.items())
                if controller.snapshot()["coordination"]["state"] == "reserved"
            ],
            ["vehicle_1", "vehicle_3"],
        )

    def test_two_vehicle_narrow_corridor_swap_completes(self) -> None:
        corridor = FleetScenario(
            "leased_corridor",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(5.0, 6.0, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    19091,
                    "spawn_2",
                    AnchorPose(14.0, 6.0, 3.141592653589793),
                ),
            ),
            100,
        )
        walls = {
            (x, y)
            for x in range(2, 18)
            for y in (3, 8)
        } | {(x, y) for x in (2, 17) for y in range(3, 9)}
        result = run_episode(
            corridor,
            {
                "vehicle_1": (
                    GotoMission("corridor-1", "global_map", 12.0, 4.5, 2),
                ),
                "vehicle_2": (
                    GotoMission("corridor-2", "global_map", 7.0, 7.0, 2),
                ),
            },
            max_simulation_s=70.0,
            grid=MapGrid.from_wall_set(20, 12, walls),
        )

        self.assertTrue(result.success, result.as_dict())
        self.assertGreaterEqual(result.minimum_inter_vehicle_clearance_m, 0.3)
        self.assertTrue(
            all(
                not vehicle["collision_occurred"]
                and not vehicle["blocked"]
                and vehicle["missions"][0]["status"] == "reached"
                for vehicle in result.vehicles
            )
        )

    def test_three_metre_corridor_is_statically_feasible_for_one_vehicle(
        self,
    ) -> None:
        scenario = FleetScenario(
            "three_metre_single_vehicle",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(5.0, 5.5, 0.0),
                ),
            ),
            100,
        )
        walls = {
            (x, y)
            for x in range(7, 14)
            for y in (*range(0, 4), *range(7, 11))
        } | {
            (x, y)
            for x in range(21)
            for y in (0, 10)
        } | {
            (x, y)
            for x in (0, 20)
            for y in range(11)
        }

        result = run_episode(
            scenario,
            {
                "vehicle_1": (
                    GotoMission("corridor", "global_map", 16.0, 5.5, 2),
                ),
            },
            max_simulation_s=40.0,
            linear_speed=1.0,
            grid=MapGrid.from_wall_set(21, 11, walls),
        )

        self.assertTrue(result.success, result.as_dict())
        self.assertEqual(result.termination_reason, "completed")
        self.assertFalse(result.vehicles[0]["collision_occurred"])

    def test_three_metre_single_lane_swap_uses_one_owner_at_a_time(self) -> None:
        corridor = FleetScenario(
            "three_metre_single_lane",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(5.0, 5.5, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    19091,
                    "spawn_2",
                    AnchorPose(16.0, 5.5, 3.141592653589793),
                ),
            ),
            100,
        )
        walls = {
            (x, y)
            for x in range(7, 14)
            for y in (*range(0, 4), *range(7, 11))
        } | {
            (x, y)
            for x in range(21)
            for y in (0, 10)
        } | {
            (x, y)
            for x in (0, 20)
            for y in range(11)
        }
        traces: list[
            tuple[
                dict[str, tuple[float, float, float]],
                dict[
                    str,
                    tuple[
                        tuple[float, float] | None,
                        int,
                        str | None,
                        bool,
                        CorridorDescriptor | None,
                    ],
                ],
                dict[str, dict[str, object]],
            ]
        ] = []
        original_tick = FleetRuntime.tick

        def recording_tick(runtime: FleetRuntime, timestamp: float) -> None:
            original_tick(runtime, timestamp)
            traces.append(
                (
                    runtime.world.truth_snapshot(),
                    {
                        vehicle_id: node.controller.motion_intent
                        for vehicle_id, node in sorted(runtime.nodes.items())
                    },
                    {
                        vehicle_id: node.controller.snapshot()["coordination"]
                        for vehicle_id, node in sorted(runtime.nodes.items())
                    },
                )
            )

        with patch.object(FleetRuntime, "tick", recording_tick):
            result = run_episode(
                corridor,
                {
                    "vehicle_1": (
                        GotoMission("corridor-1", "global_map", 16.0, 5.5, 2),
                    ),
                    "vehicle_2": (
                        GotoMission("corridor-2", "global_map", 5.0, 5.5, 2),
                    ),
                },
                max_simulation_s=80.0,
                linear_speed=1.0,
                grid=MapGrid.from_wall_set(21, 11, walls),
            )

        self.assertTrue(result.success, result.as_dict())
        occupants = [
            tuple(
                vehicle_id
                for vehicle_id, pose in sorted(poses.items())
                if 7.0 < pose[0] < 14.0
            )
            for poses, _, _ in traces
        ]
        self.assertTrue(any(occupants))
        self.assertTrue(all(len(active) <= 1 for active in occupants))
        descriptor_pair = next(
            (
                intents["vehicle_1"][4],
                intents["vehicle_2"][4],
            )
            for _, intents, _ in traces
            if intents["vehicle_1"][4] is not None
            and intents["vehicle_2"][4] is not None
        )
        first_descriptor, second_descriptor = descriptor_pair
        assert first_descriptor is not None and second_descriptor is not None
        self.assertEqual(first_descriptor.entry_cell, second_descriptor.exit_cell)
        self.assertEqual(first_descriptor.exit_cell, second_descriptor.entry_cell)

        claims_by_tick = [
            tuple(
                vehicle_id
                for vehicle_id, values in sorted(intents.items())
                if values[3] and values[4] is not None
            )
            for _, intents, _ in traces
        ]
        owners_by_tick = [
            tuple(
                vehicle_id
                for vehicle_id, state in sorted(coordination.items())
                if state["state"] == "reserved"
            )
            for _, _, coordination in traces
        ]
        converged_index = next(
            index
            for index, (_, _, coordination) in enumerate(traces)
            if any(
                state["reason"] == "corridor_lease"
                for state in coordination.values()
            )
        )
        self.assertTrue(
            all(
                not active
                for active, owners in zip(
                    occupants[:converged_index],
                    claims_by_tick[:converged_index],
                    strict=True,
                )
                if len(owners) > 1
            )
        )
        self.assertTrue(
            all(len(owners) <= 1 for owners in owners_by_tick)
        )
        owner_order = []
        for owners in owners_by_tick:
            if owners and owners[0] not in owner_order:
                owner_order.append(owners[0])
        self.assertEqual(owner_order, ["vehicle_1", "vehicle_2"])
        self.assertTrue(
            any(
                coordination["vehicle_2"]
                == {
                    "state": "waiting",
                    "reason": "corridor_lease",
                    "priority_owner_vehicle_id": "vehicle_1",
                }
                for _, _, coordination in traces
            )
        )
        for active, owners in zip(occupants, owners_by_tick, strict=True):
            if active and owners:
                self.assertEqual(active, owners)

        # The owner keeps its merged descriptor for every tick inside the
        # physical bottleneck, then clears it only after crossing the release.
        for vehicle_id in ("vehicle_1", "vehicle_2"):
            inside_intents = [
                intents[vehicle_id]
                for poses, intents, _ in traces
                if 7.0 < poses[vehicle_id][0] < 14.0
            ]
            self.assertTrue(inside_intents)
            self.assertTrue(
                all(values[3] and values[4] is not None for values in inside_intents)
            )
        self.assertTrue(
            any(
                poses["vehicle_1"][0] > 14.0
                and intents["vehicle_1"][4] is None
                for poses, intents, _ in traces
            )
        )
        self.assertTrue(
            any(
                poses["vehicle_2"][0] < 7.0
                and intents["vehicle_2"][4] is None
                for poses, intents, _ in traces
            )
        )
        self.assertGreaterEqual(result.minimum_inter_vehicle_clearance_m, 0.3)

    def test_near_entry_claims_converge_before_either_vehicle_enters(self) -> None:
        scenario = FleetScenario(
            "near_entry_corridor_admission",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(6.4, 5.5, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    19091,
                    "spawn_2",
                    AnchorPose(14.6, 5.5, 3.141592653589793),
                ),
            ),
            100,
        )
        walls = {
            (x, y)
            for x in range(7, 14)
            for y in (*range(0, 4), *range(7, 11))
        } | {
            (x, y)
            for x in range(21)
            for y in (0, 10)
        } | {
            (x, y)
            for x in (0, 20)
            for y in range(11)
        }
        trace = []
        original_tick = FleetRuntime.tick

        def recording_tick(runtime: FleetRuntime, timestamp: float) -> None:
            original_tick(runtime, timestamp)
            trace.append(
                (
                    runtime.world.truth_snapshot(),
                    {
                        vehicle_id: node.controller.snapshot()["coordination"]
                        for vehicle_id, node in sorted(runtime.nodes.items())
                    },
                )
            )

        with patch.object(FleetRuntime, "tick", recording_tick):
            result = run_episode(
                scenario,
                {
                    "vehicle_1": (
                        GotoMission("near-1", "global_map", 16.0, 5.5, 2),
                    ),
                    "vehicle_2": (
                        GotoMission("near-2", "global_map", 5.0, 5.5, 2),
                    ),
                },
                max_simulation_s=2.0,
                linear_speed=1.0,
                grid=MapGrid.from_wall_set(21, 11, walls),
            )

        self.assertEqual(result.termination_reason, "timeout")
        double_tentative = [
            index
            for index, (_, coordination) in enumerate(trace)
            if all(
                state["state"] == "tentative"
                for state in coordination.values()
            )
        ]
        self.assertTrue(double_tentative)
        for index in double_tentative:
            self.assertTrue(
                all(
                    not 6.5 < pose[0] < 14.5
                    for pose in trace[index][0].values()
                )
            )
        first_occupied = next(
            index
            for index, (poses, _) in enumerate(trace)
            if any(6.5 < pose[0] < 14.5 for pose in poses.values())
        )
        confirmed = [
            vehicle_id
            for vehicle_id, state in trace[first_occupied][1].items()
            if state["state"] == "reserved"
        ]
        self.assertEqual(confirmed, ["vehicle_1"])
        self.assertTrue(
            all(not vehicle["collision_occurred"] for vehicle in result.vehicles)
        )

    def test_two_vehicles_already_inside_single_lane_fail_safely(self) -> None:
        scenario = FleetScenario(
            "infeasible_already_inside_single_lane",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(9.0, 5.5, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    19091,
                    "spawn_2",
                    AnchorPose(12.0, 5.5, 3.141592653589793),
                ),
            ),
            100,
        )
        walls = {
            (x, y)
            for x in range(7, 14)
            for y in (*range(0, 4), *range(7, 11))
        } | {
            (x, y)
            for x in range(21)
            for y in (0, 10)
        } | {
            (x, y)
            for x in (0, 20)
            for y in range(11)
        }

        result = run_episode(
            scenario,
            {
                "vehicle_1": (
                    GotoMission("inside-1", "global_map", 16.0, 5.5, 2),
                ),
                "vehicle_2": (
                    GotoMission("inside-2", "global_map", 5.0, 5.5, 2),
                ),
            },
            max_simulation_s=10.0,
            linear_speed=1.0,
            grid=MapGrid.from_wall_set(21, 11, walls),
        )

        self.assertFalse(result.success, result.as_dict())
        self.assertEqual(result.termination_reason, "timeout")
        self.assertTrue(
            all(not vehicle["collision_occurred"] for vehicle in result.vehicles)
        )
        self.assertGreaterEqual(result.minimum_inter_vehicle_clearance_m, 0.3)
        self.assertFalse(
            all(
                vehicle["missions"][0]["status"] == "reached"
                for vehicle in result.vehicles
            )
        )

    def test_goto_patrol_and_coverage_share_the_corridor_lease(self) -> None:
        scenario = FleetScenario(
            "mixed_missions_single_lane",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(5.0, 5.5, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    19091,
                    "spawn_2",
                    AnchorPose(16.0, 5.5, 3.141592653589793),
                ),
            ),
            100,
        )
        walls = {
            (x, y)
            for x in range(7, 14)
            for y in (*range(0, 4), *range(7, 11))
        } | {
            (x, y)
            for x in range(21)
            for y in (0, 10)
        } | {
            (x, y)
            for x in (0, 20)
            for y in range(11)
        }
        owners = []
        original_tick = FleetRuntime.tick

        def recording_tick(runtime: FleetRuntime, timestamp: float) -> None:
            original_tick(runtime, timestamp)
            reserved = tuple(
                vehicle_id
                for vehicle_id, node in sorted(runtime.nodes.items())
                if node.controller.snapshot()["coordination"]["state"]
                == "reserved"
            )
            self.assertLessEqual(len(reserved), 1)
            if reserved and reserved[0] not in owners:
                owners.append(reserved[0])

        with patch.object(FleetRuntime, "tick", recording_tick):
            result = run_episode(
                scenario,
                {
                    "vehicle_1": (
                        GotoMission(
                            "mixed-goto",
                            "global_map",
                            16.0,
                            8.0,
                            2,
                        ),
                        PatrolMission(
                            "mixed-patrol",
                            "global_map",
                            ((18.0, 8.0), (18.0, 6.5)),
                            1,
                            3,
                        ),
                    ),
                    "vehicle_2": (
                        CoverageMission(
                            "mixed-coverage",
                            "global_map",
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            2.0,
                            2,
                        ),
                    ),
                },
                max_simulation_s=100.0,
                linear_speed=1.0,
                grid=MapGrid.from_wall_set(21, 11, walls),
            )

        self.assertTrue(result.success, result.as_dict())
        self.assertEqual(owners, ["vehicle_1", "vehicle_2"])
        self.assertGreaterEqual(result.minimum_inter_vehicle_clearance_m, 0.3)
        self.assertEqual(
            [
                mission["type"]
                for vehicle in result.vehicles
                for mission in vehicle["missions"]
            ],
            ["goto", "patrol", "coverage"],
        )
        self.assertTrue(
            all(
                mission["status"] == "reached"
                for vehicle in result.vehicles
                for mission in vehicle["missions"]
            )
        )

    def test_four_vehicle_corridor_routes_are_individually_feasible(self) -> None:
        walls = {
            (x, y)
            for x in range(9, 14)
            for y in (*range(0, 4), *range(7, 11))
        } | {
            (x, y)
            for x in range(21)
            for y in (0, 10)
        } | {
            (x, y)
            for x in (0, 20)
            for y in range(11)
        }
        routes = (
            ("vehicle_1", AnchorPose(7.0, 5.5, 0.0), (16.0, 8.5)),
            ("vehicle_2", AnchorPose(16.0, 5.5, math.pi), (7.0, 8.5)),
            ("vehicle_3", AnchorPose(5.5, 5.5, 0.0), (17.0, 4.0)),
            ("vehicle_4", AnchorPose(17.5, 5.5, math.pi), (6.0, 4.0)),
        )

        for index, (vehicle_id, start, goal) in enumerate(routes):
            with self.subTest(vehicle_id=vehicle_id):
                scenario = FleetScenario(
                    f"single_route_{vehicle_id}",
                    (
                        FleetVehicleSpec(
                            vehicle_id,
                            19090 + index,
                            f"spawn_{index + 1}",
                            start,
                        ),
                    ),
                    100,
                )
                result = run_episode(
                    scenario,
                    {
                        vehicle_id: (
                            GotoMission(
                                f"route-{index + 1}",
                                "global_map",
                                *goal,
                                2,
                            ),
                        ),
                    },
                    max_simulation_s=35.0,
                    linear_speed=1.2,
                    grid=MapGrid.from_wall_set(21, 11, walls),
                )
                self.assertTrue(result.success, result.as_dict())
                self.assertLess(result.simulation_duration_s, 35.0)
                self.assertFalse(result.vehicles[0]["collision_occurred"])

    @pytest.mark.extended
    def test_four_vehicles_share_one_corridor_with_bounded_fair_waiting(
        self,
    ) -> None:
        scenario = FleetScenario(
            "four_vehicle_single_lane_contention",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    19090,
                    "spawn_1",
                    AnchorPose(7.0, 5.5, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    19091,
                    "spawn_2",
                    AnchorPose(16.0, 5.5, math.pi),
                ),
                FleetVehicleSpec(
                    "vehicle_3",
                    19092,
                    "spawn_3",
                    AnchorPose(5.5, 5.5, 0.0),
                ),
                FleetVehicleSpec(
                    "vehicle_4",
                    19093,
                    "spawn_4",
                    AnchorPose(17.5, 5.5, math.pi),
                ),
            ),
            100,
        )
        goals = {
            "vehicle_1": (16.0, 8.5),
            "vehicle_2": (7.0, 8.5),
            "vehicle_3": (17.0, 4.0),
            "vehicle_4": (6.0, 4.0),
        }
        walls = {
            (x, y)
            for x in range(9, 14)
            for y in (*range(0, 4), *range(7, 11))
        } | {
            (x, y)
            for x in range(21)
            for y in (0, 10)
        } | {
            (x, y)
            for x in (0, 20)
            for y in range(11)
        }
        owner_order = []
        active_stall_ticks = {vehicle_id: 0 for vehicle_id in goals}
        longest_active_stall_ticks = {vehicle_id: 0 for vehicle_id in goals}
        previous_poses = {
            spec.vehicle_id: (
                spec.anchor_pose.x_m,
                spec.anchor_pose.y_m,
                spec.anchor_pose.yaw_rad,
            )
            for spec in scenario.vehicles
        }
        original_tick = FleetRuntime.tick

        def recording_tick(runtime: FleetRuntime, timestamp: float) -> None:
            original_tick(runtime, timestamp)
            poses = runtime.world.truth_snapshot()
            confirmed = tuple(
                vehicle_id
                for vehicle_id, node in sorted(runtime.nodes.items())
                if node.controller.snapshot()["coordination"]["state"]
                == "reserved"
            )
            self.assertLessEqual(len(confirmed), 1)
            if confirmed and (not owner_order or owner_order[-1] != confirmed[0]):
                owner_order.append(confirmed[0])
            occupants = tuple(
                vehicle_id
                for vehicle_id, pose in sorted(poses.items())
                if 9.0 < pose[0] < 14.0
            )
            self.assertLessEqual(len(occupants), 1)
            if occupants:
                self.assertEqual(occupants, confirmed)
            for vehicle_id, pose in poses.items():
                moved = math.dist(pose[:2], previous_poses[vehicle_id][:2])
                active = (
                    runtime.nodes[vehicle_id].controller.auto_state
                    is AutoState.ACTIVE
                )
                if active and moved < 0.001:
                    active_stall_ticks[vehicle_id] += 1
                    longest_active_stall_ticks[vehicle_id] = max(
                        longest_active_stall_ticks[vehicle_id],
                        active_stall_ticks[vehicle_id],
                    )
                else:
                    active_stall_ticks[vehicle_id] = 0
            previous_poses.update(poses)

        with patch.object(FleetRuntime, "tick", recording_tick):
            result = run_episode(
                scenario,
                {
                    vehicle_id: (
                        GotoMission(
                            f"contention-{vehicle_id[-1]}",
                            "global_map",
                            *goal,
                            2,
                        ),
                    )
                    for vehicle_id, goal in goals.items()
                },
                max_simulation_s=130.0,
                linear_speed=1.2,
                grid=MapGrid.from_wall_set(21, 11, walls),
            )

        diagnostic = {
            **result.as_dict(),
            "owner_order": owner_order,
            "longest_active_stall_s": {
                vehicle_id: ticks * scenario.tick_ms / 1000.0
                for vehicle_id, ticks in longest_active_stall_ticks.items()
            },
        }
        self.assertTrue(result.success, diagnostic)
        first_acquisitions = []
        for owner in owner_order:
            if owner not in first_acquisitions:
                first_acquisitions.append(owner)
        self.assertEqual(set(first_acquisitions), set(goals))
        self.assertEqual(len(first_acquisitions), len(goals))
        self.assertGreaterEqual(len(owner_order) - 1, 3)
        self.assertTrue(
            all(
                not vehicle["collision_occurred"]
                and not vehicle["blocked"]
                and vehicle["missions"][0]["status"] == "reached"
                for vehicle in result.vehicles
            )
        )
        self.assertGreaterEqual(result.minimum_inter_vehicle_clearance_m, 0.3)
        # This is deliberately strict one-at-a-time admission, not a convoy.
        # The 130 s episode budget includes approach, staging/rejoin,
        # exit-conflict gaps, and the final route tail.  Ninety seconds bounds
        # the fourth vehicle's three-predecessor queue wait without disguising
        # unbounded starvation.
        self.assertLessEqual(
            max(longest_active_stall_ticks.values()) * scenario.tick_ms / 1000.0,
            90.0,
            diagnostic,
        )

    def test_four_vehicle_shared_crossing_completes_with_bounded_wait(self) -> None:
        scenario = FleetScenario.load(
            Path(__file__).parent / "fixtures" / "four_vehicle_crossing_episode.json"
        )
        goals = {
            "mock_vehicle_01": (15.0, 10.0),
            "mock_vehicle_02": (5.0, 10.0),
            "mock_vehicle_03": (10.0, 15.0),
            "mock_vehicle_04": (10.0, 5.0),
        }
        result = run_episode(
            scenario,
            {
                vehicle_id: (
                    GotoMission(
                        f"leased-crossing-{vehicle_id[-2:]}",
                        "global_map",
                        *goal,
                        2,
                    ),
                )
                for vehicle_id, goal in goals.items()
            },
            max_simulation_s=90.0,
            grid=MapGrid.from_wall_set(20, 20, set()),
        )

        self.assertTrue(result.success, result.as_dict())
        self.assertGreaterEqual(result.minimum_inter_vehicle_clearance_m, 0.3)
        longest_waits = [
            vehicle["longest_no_progress_duration_s"]
            for vehicle in result.vehicles
        ]
        self.assertLessEqual(max(longest_waits), 30.0, result.as_dict())
        self.assertTrue(
            all(
                not vehicle["collision_occurred"]
                and not vehicle["blocked"]
                and vehicle["missions"][0]["status"] == "reached"
                for vehicle in result.vehicles
            )
        )


if __name__ == "__main__":
    unittest.main()
