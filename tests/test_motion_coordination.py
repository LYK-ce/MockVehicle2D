"""PIBT-inspired leased-cell coordination contracts."""

import unittest
from copy import deepcopy
import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mockvehicle2d.controller import (
    AutoState,
    CoverageMission,
    RobotController,
    GotoMission,
    OpMode,
    PatrolMission,
    corridor_descriptors_conflict,
    _front_corridor_waiter,
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
    MOTION_INTENT_PROTOCOL,
    MOTION_INTENT_TTL_S,
    MapSyncState,
    PeerMotionIntent,
    PeerVehicleState,
)
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.safety import (
    AUTOMATIC_MINIMUM_CLEARANCE_M,
    LocalSafetyRuntime,
    SafetyAdvanceResult,
    SafetyDecision,
)
from mockvehicle2d.scan import LaserPoint
from mockvehicle2d.vehicle import Vehicle


def intent(
    vehicle_id: str,
    *,
    current: tuple[int, int],
    target: tuple[int, int],
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
        controller.active_mission = GotoMission(
            "corridor",
            "global_map",
            16.0,
            0.5,
            1,
        )
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
        controller.active_mission = GotoMission(
            "corridor",
            "global_map",
            16.0,
            0.5,
            1,
        )
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
        controller.active_mission = GotoMission(
            "corridor",
            "global_map",
            16.0,
            0.5,
            1,
        )
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

        controller._clear_yield()

        self.assertIsNone(controller._corridor_rejoin_target_m)

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
        controller.active_mission = GotoMission(
            "blocked",
            "global_map",
            10.0,
            0.5,
            1,
        )
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
        source.record_motion_intent(
            PoseEstimate(
                first_anchor.anchor_id,
                0.25,
                0.25,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                4.0,
                1,
            ),
            target_m=(1.5, 0.5),
            wait_ticks=3,
            priority_owner_id="vehicle_2",
            reserved=True,
            corridor=CorridorDescriptor((10, 5), (15, 5)),
            timestamp_s=4.0,
        )
        payload = source.prepare_motion_intent()

        self.assertEqual(payload["protocol"], MOTION_INTENT_PROTOCOL)
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
        commit_past_horizon = deepcopy(payload)
        commit_past_horizon["committed_until_offset_s"] = 4.1
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

        older = RobotController(Mock(motion_target=(1.5, 0.5)))
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
        newcomer = RobotController(Mock(motion_target=(1.5, 0.5)))
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
