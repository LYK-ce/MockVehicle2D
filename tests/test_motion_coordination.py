"""PIBT-inspired leased-cell coordination contracts."""

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

from mockvehicle2d.controller import (
    RobotController,
    GotoMission,
    _motion_intents_conflict,
    inherit_motion_priority,
    motion_intent_precedes,
)
from mockvehicle2d.episode import run_episode
from mockvehicle2d.fleet import AnchorPose, FleetScenario, FleetVehicleSpec
from mockvehicle2d.local_state import AnchorSpec, ObservedGrid, PoseEstimate
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.map_sync import (
    MOTION_INTENT_PROTOCOL,
    MOTION_INTENT_TTL_S,
    MapSyncState,
    PeerMotionIntent,
    PeerVehicleState,
)
from mockvehicle2d.vehicle import Vehicle


def intent(
    vehicle_id: str,
    *,
    current: tuple[int, int],
    target: tuple[int, int],
    wait_ticks: int,
    owner: str | None = None,
    reserved: bool = False,
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
    )


def peer_state(
    vehicle_id: str,
    x_m: float,
    y_m: float,
    vx_mps: float,
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
        0.0,
        0.0,
        0.5,
    )


class TestMotionCoordination(unittest.TestCase):
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
            timestamp_s=4.0,
        )
        payload = source.prepare_motion_intent()

        self.assertEqual(payload["protocol"], MOTION_INTENT_PROTOCOL)
        unknown_owner = deepcopy(payload)
        unknown_owner["priority"]["owner_vehicle_id"] = "unknown_vehicle"
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", unknown_owner)
        )
        self.assertTrue(
            receiver.receive_transport("peer_1", "vehicle_1", payload)
        )
        received = receiver.peer_motion_intents()[0]
        self.assertEqual(received.current_cell, (10, 5))
        self.assertEqual(received.target_cell, (11, 5))
        self.assertEqual(received.priority_owner_id, "vehicle_2")
        self.assertTrue(received.reserved)
        self.assertFalse(
            receiver.receive_transport("peer_1", "vehicle_1", payload)
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

        target_m, _, owner_id, reserved = controller.motion_intent
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
