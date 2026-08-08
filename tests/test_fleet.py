"""Shared-world multi-vehicle simulation invariants."""

import asyncio
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mockvehicle2d.collision import is_strict_overlap
from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    GotoMission,
    ManualAction,
    ManualCommand,
    ModeAction,
    ModeCommand,
)
from mockvehicle2d.fleet import (
    AnchorPose,
    FleetRuntime,
    FleetScenario,
    FleetVehicleSpec,
    fleet_handler,
)
from mockvehicle2d.local_state import (
    OCCUPIED,
    AnchorSpec,
    LocalMapDelta,
    OdometryConfig,
    PoseEstimate,
)
from mockvehicle2d.map_grid import WALL, MapGrid
from mockvehicle2d.map_sync import PEER_STATE_TTL_S, P2PSettings
from mockvehicle2d.scan import TMINI_SCAN_CONFIG
from mockvehicle2d.safety import (
    AUTOMATIC_MINIMUM_CLEARANCE_M,
    HARD_STOP_CLEARANCE_M,
)
from mockvehicle2d.server import VehicleRuntime, generate_map
from mockvehicle2d.vehicle import Vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]


def spec(
    number: int,
    x_m: float,
    y_m: float,
    yaw_rad: float = 0.0,
) -> FleetVehicleSpec:
    return FleetVehicleSpec(
        f"vehicle_{number}",
        19089 + number,
        f"spawn_{number}",
        AnchorPose(x_m, y_m, yaw_rad),
    )


def scenario(*vehicles: FleetVehicleSpec, tick_ms: int = 100) -> FleetScenario:
    return FleetScenario("test_scenario", tuple(vehicles), tick_ms)


def free_grid(size: int = 40) -> MapGrid:
    return MapGrid.from_wall_set(size, size, set())


def peer_fleet(
    first: AnchorPose,
    second: AnchorPose,
    *,
    grid: MapGrid | None = None,
    voxels: list[dict[str, object]] | None = None,
    realtime_factor: float = 1.0,
) -> FleetRuntime:
    specs = tuple(
        FleetVehicleSpec(
            f"vehicle_{number}",
            19089 + number,
            f"spawn_{number}",
            pose,
            20089 + number,
        )
        for number, pose in enumerate((first, second), 1)
    )
    fleet = FleetRuntime.create(
        FleetScenario(
            "peer_test",
            specs,
            100,
            P2PSettings(Path("map-sync-node"), Path("runtime")),
        ),
        grid=free_grid() if grid is None else grid,
        voxels=voxels,
        realtime_factor=realtime_factor,
    )
    for vehicle_id, node in fleet.nodes.items():
        node.map_sync.configure_network(
            f"peer_{vehicle_id[-1]}",
            {
                other_id: (f"peer_{other_id[-1]}", other.local_state.anchor)
                for other_id, other in fleet.nodes.items()
                if other_id != vehicle_id
            },
        )
    return fleet


def relay_peer_states(fleet: FleetRuntime) -> None:
    outbound = {
        vehicle_id: node.map_sync.prepare_peer_state()
        for vehicle_id, node in fleet.nodes.items()
    }
    for source_id, payload in outbound.items():
        assert payload is not None
        for receiver_id, receiver in fleet.nodes.items():
            if receiver_id != source_id:
                receiver.map_sync.receive_peer_state(
                    f"peer_{source_id[-1]}",
                    source_id,
                    payload,
                )
        fleet.nodes[source_id].map_sync.publish_peer_state_result(
            payload["sequence"],
            True,
        )


class TestFleetScenario(unittest.TestCase):
    def test_robot_node_passes_network_membership_to_coordination(self) -> None:
        fleet = peer_fleet(AnchorPose(5.0, 5.0, 0.0), AnchorPose(15.0, 5.0, 0.0))
        node = fleet.nodes["vehicle_1"]
        node.map_sync.set_health(
            ready=True,
            connected_vehicle_ids=("vehicle_2",),
        )
        controller = Mock()
        controller.planning_ignored_peer_ids.return_value = frozenset()
        controller.motion_intent = (None, 0, "vehicle_1", False, None)
        node.controller = controller

        node.control(
            fleet.world.vehicle("vehicle_1"),
            fleet.world.sensor_grid("vehicle_1"),
            fleet.world.now,
        )

        self.assertIs(controller.tick.call_args.kwargs["coordination_ready"], True)
        self.assertEqual(
            controller.tick.call_args.kwargs["expected_peer_vehicle_ids"],
            ("vehicle_2",),
        )

    def test_example_declares_four_unique_endpoints_and_spawns(self) -> None:
        loaded = FleetScenario.load(
            REPO_ROOT / "examples" / "four_vehicle_scenario.json"
        )

        self.assertEqual(len(loaded.vehicles), 4)
        self.assertEqual(
            [vehicle.operator_port for vehicle in loaded.vehicles],
            [19090, 19091, 19092, 19093],
        )
        self.assertEqual(len({vehicle.spawn_id for vehicle in loaded.vehicles}), 4)

    def test_strict_json_and_cardinality_validation(self) -> None:
        valid_vehicle = {
            "vehicle_id": "vehicle_1",
            "operator_port": 19090,
            "spawn_id": "spawn_1",
            "anchor_pose": {"x_m": 5.0, "y_m": 5.0, "yaw_rad": 0.0},
        }
        invalid_cases = (
            {"scenario_id": "empty", "vehicles": []},
            {
                "scenario_id": "too_many",
                "vehicles": [
                    {
                        **valid_vehicle,
                        "vehicle_id": f"v_{index}",
                        "operator_port": 19090 + index,
                        "spawn_id": f"s_{index}",
                    }
                    for index in range(5)
                ],
            },
            {
                "scenario_id": "duplicate",
                "vehicles": [valid_vehicle, valid_vehicle],
            },
            {
                "scenario_id": "extra",
                "vehicles": [valid_vehicle],
                "unexpected": True,
            },
        )
        for value in invalid_cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                FleetScenario.from_json(value)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"vehicles": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                FleetScenario.load(path)

    def test_world_atomically_rejects_unsafe_spawns(self) -> None:
        cases = (
            (scenario(spec(1, 0.2, 5.0)), free_grid(), "outside"),
            (
                scenario(spec(1, 5.5, 5.5)),
                MapGrid.from_wall_set(20, 20, {(5, 5)}),
                "static",
            ),
            (
                scenario(spec(1, 5.0, 5.0), spec(2, 6.0, 5.0)),
                free_grid(),
                "overlap",
            ),
        )
        for fleet_scenario, grid, reason in cases:
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                FleetRuntime.create(fleet_scenario, grid=grid)


class TestFleetRuntime(unittest.TestCase):
    def test_peer_state_ttl_uses_fleet_simulation_time(self) -> None:
        fleet = peer_fleet(
            AnchorPose(5.0, 5.0, 0.0),
            AnchorPose(10.0, 10.0, 0.0),
            realtime_factor=3.0,
        )
        relay_peer_states(fleet)
        receiver = fleet.nodes["vehicle_2"].map_sync

        for _ in range(3):
            fleet.tick(fleet.timestamp_at(fleet.world.now + fleet.tick_s))
        self.assertEqual(len(receiver.peer_vehicle_states()), 1)

        fleet.tick(fleet.timestamp_at(fleet.world.now + fleet.tick_s))
        self.assertEqual(receiver.peer_vehicle_states(), ())

    def test_realtime_factor_changes_wall_pacing_not_simulation_ticks(self) -> None:
        fleet_scenario = scenario(spec(1, 5.0, 5.0))
        normal = FleetRuntime.create(
            fleet_scenario,
            grid=free_grid(),
            timestamp=1_000.0,
        )
        fast = FleetRuntime.create(
            fleet_scenario,
            grid=free_grid(),
            timestamp=1_000.0,
            realtime_factor=3.0,
        )
        for fleet in (normal, fast):
            fleet.handle_command(
                "vehicle_1",
                ManualCommand(1, ManualAction.DRIVE, 0.5, 0.0),
            )
            fleet.tick(fleet.timestamp_at(fleet.world.now + fleet.tick_s))
            fleet.tick(fleet.timestamp_at(fleet.world.now + fleet.tick_s))

        normal_vehicle = normal.world.vehicle("vehicle_1")
        fast_vehicle = fast.world.vehicle("vehicle_1")
        self.assertEqual(normal.world.now, fast.world.now)
        self.assertEqual(normal_vehicle.linear_speed, fast_vehicle.linear_speed)
        self.assertAlmostEqual(normal_vehicle.x, fast_vehicle.x)
        self.assertAlmostEqual(normal_vehicle.y, fast_vehicle.y)
        self.assertAlmostEqual(normal_vehicle.yaw, fast_vehicle.yaw)
        self.assertEqual(
            [frame.frame.timestamp for frame in normal.nodes["vehicle_1"]._frames],
            [frame.frame.timestamp for frame in fast.nodes["vehicle_1"]._frames],
        )

        stop = asyncio.Event()
        timeouts: list[float] = []

        async def time_out(awaitable, *, timeout: float):
            awaitable.close()
            timeouts.append(timeout)
            raise asyncio.TimeoutError

        fast.tick = Mock(side_effect=lambda _: stop.set())
        with patch("mockvehicle2d.fleet.asyncio.wait_for", side_effect=time_out):
            asyncio.run(fast.run(stop))

        self.assertEqual(fast.tick.call_count, 1)
        self.assertAlmostEqual(timeouts[0], fleet_scenario.tick_s / 3.0)

        for factor in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                FleetRuntime.create(
                    fleet_scenario,
                    grid=free_grid(),
                    realtime_factor=factor,
                )

    def test_missing_or_stale_tmini_scan_blocks_automatic_motion(self) -> None:
        for latest_scan_time in (
            None,
            -TMINI_SCAN_CONFIG.scan_time - 0.01,
        ):
            with self.subTest(latest_scan_time=latest_scan_time):
                fleet = FleetRuntime.create(
                    scenario(spec(1, 5.0, 5.0)),
                    grid=free_grid(),
                )
                node = fleet.nodes["vehicle_1"]
                fleet.handle_command(
                    "vehicle_1",
                    ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
                )
                fleet.handle_command(
                    "vehicle_1",
                    AutoCommand(
                        2,
                        AutoAction.PUSH,
                        (GotoMission("scan-health", "global_map", 8.0, 5.0, 2),),
                    ),
                )
                node._latest_scan_monotonic_s = latest_scan_time

                fleet.tick(fleet.tick_s)

                self.assertEqual(node.controller.auto_state.value, "blocked")
                self.assertEqual(
                    node.controller.navigation.reason,
                    "safety_sensor_fault",
                )
                self.assertEqual(
                    fleet.world.vehicle("vehicle_1").body_velocities(),
                    (0.0, 0.0),
                )

    def test_peer_forbidden_delta_distinguishes_unchanged_from_cleared(self) -> None:
        self.assertIsNone(LocalMapDelta(()).peer_forbidden_cells)
        fleet = peer_fleet(
            AnchorPose(5.0, 5.0, 0.0),
            AnchorPose(10.0, 10.0, 0.0),
        )
        source = fleet.nodes["vehicle_1"].map_sync
        receiver = fleet.nodes["vehicle_2"]
        payload = source.prepare_peer_state()
        self.assertTrue(
            receiver.map_sync.receive_peer_state(
                "peer_1",
                "vehicle_1",
                payload,
                received_at_s=1.0,
            )
        )

        peer_hit = (-10, -10)
        outside_peer_footprint = (-8, -10)
        receiver._lidar_dynamic_cells = {
            peer_hit,
            outside_peer_footprint,
        }
        receiver._update_planning_map(
            peer_states=receiver.map_sync.peer_vehicle_states(now_s=1.0)
        )
        self.assertTrue(receiver._pending_map_delta.peer_forbidden_cells)
        self.assertEqual(
            receiver._planning_map.cell_without_peers(*peer_hit),
            receiver.local_state.local_map.get_cell(*peer_hit),
        )
        self.assertEqual(
            receiver._planning_map.cell_without_peers(*outside_peer_footprint),
            OCCUPIED,
        )
        receiver._pending_map_delta = None
        receiver._update_planning_map(peer_states=())

        self.assertEqual(receiver._pending_map_delta.peer_forbidden_cells, ())
        self.assertEqual(
            receiver._planning_map.cell_without_peers(*peer_hit),
            OCCUPIED,
        )
        self.assertEqual(
            receiver._planning_map.peer_exclusion_circles(),
            (),
        )

    def test_ignored_corridor_peer_keeps_static_and_unattributed_obstacles(self) -> None:
        fleet = peer_fleet(
            AnchorPose(5.0, 5.0, 0.0),
            AnchorPose(10.0, 10.0, 0.0),
        )
        source = fleet.nodes["vehicle_1"].map_sync
        receiver = fleet.nodes["vehicle_2"]
        payload = source.prepare_peer_state()
        self.assertTrue(
            receiver.map_sync.receive_peer_state(
                "peer_1",
                "vehicle_1",
                payload,
                received_at_s=1.0,
            )
        )
        active = receiver.map_sync.peer_vehicle_states(now_s=1.0)
        peer_hit = (-10, -10)
        unattributed_hit = (-8, -10)
        static_obstacle = (-6, -10)
        receiver.local_state.local_map._cells[static_obstacle] = OCCUPIED
        receiver.local_state.local_map.revision += 1
        receiver._lidar_dynamic_cells = {peer_hit, unattributed_hit}

        receiver._update_planning_map(
            peer_states=active,
            ignored_peer_vehicle_ids=frozenset(("vehicle_1",)),
        )

        self.assertEqual(receiver._pending_map_delta.peer_forbidden_cells, ())
        self.assertNotEqual(
            receiver._planning_map.cell_without_peers(*peer_hit),
            OCCUPIED,
        )
        self.assertEqual(
            receiver._planning_map.cell_without_peers(*unattributed_hit),
            OCCUPIED,
        )
        self.assertEqual(
            receiver._planning_map.get_cell(*static_obstacle),
            OCCUPIED,
        )

    def test_lidar_peer_dedup_uses_exact_cell_circle_intersection(self) -> None:
        fleet = peer_fleet(
            AnchorPose(5.0, 5.0, 0.0),
            AnchorPose(10.0, 10.0, 0.0),
        )
        source_node = fleet.nodes["vehicle_1"]
        source = source_node.map_sync
        node = fleet.nodes["vehicle_2"]
        source.record_vehicle_state(
            PoseEstimate(
                source_node.local_state.anchor.anchor_id,
                4.95,
                5.25,
                0.0,
                (0.0, 0.0, 0.0),
                "nominal",
                1.0,
                1,
            ),
            radius_m=0.5,
            linear_mps=0.0,
            omega_rps=0.0,
        )
        payload = source.prepare_peer_state()
        self.assertIsNotNone(payload)
        self.assertTrue(
            node.map_sync.receive_peer_state(
                "peer_1",
                "vehicle_1",
                payload,
                received_at_s=1.0,
            )
        )
        active = node.map_sync.peer_vehicle_states(now_s=1.0)
        self.assertEqual(len(active), 1)
        self.assertAlmostEqual(active[0].global_x_m, 9.95)
        self.assertAlmostEqual(active[0].global_y_m, 10.25)
        intersects = (0, 0)
        outside = (1, 0)
        node._lidar_dynamic_cells = {intersects, outside}

        node._update_planning_map(peer_states=active)

        self.assertNotEqual(
            node._planning_map.cell_without_peers(*intersects),
            OCCUPIED,
        )
        self.assertEqual(
            node._planning_map.cell_without_peers(*outside),
            OCCUPIED,
        )

        expired = node.map_sync.peer_vehicle_states(
            now_s=1.0 + PEER_STATE_TTL_S + 0.01
        )
        self.assertEqual(expired, ())
        node._update_planning_map(peer_states=expired)
        self.assertEqual(
            node._planning_map.cell_without_peers(*intersects),
            OCCUPIED,
        )
        self.assertEqual(
            node._planning_map.cell_without_peers(*outside),
            OCCUPIED,
        )

    def test_each_vehicle_starts_at_truth_anchor_with_zero_local_odometry(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 6.0, math.pi / 2)),
            grid=free_grid(),
        )

        self.assertEqual(
            fleet.world.truth_snapshot()["vehicle_1"],
            (5.0, 6.0, math.pi / 2),
        )
        pose = fleet.nodes["vehicle_1"].local_state.pose
        self.assertEqual((pose.x_m, pose.y_m, pose.yaw_rad), (0.0, 0.0, 0.0))
        self.assertEqual(pose.anchor_id, "spawn_1")

    def test_tmini_sees_other_vehicle_without_persisting_it_in_own_map(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), spec(2, 7.0, 5.0, math.pi)),
            grid=free_grid(),
        )

        forward = fleet.world.scan("vehicle_1")[0]
        self.assertTrue(forward.dynamic)
        self.assertAlmostEqual(forward.range, 1.5)
        self.assertEqual(
            fleet.nodes["vehicle_1"].local_state.local_map.occupied_cells(),
            (),
        )

    def test_dynamic_vehicle_cells_are_removed_and_restore_the_persistent_map(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.5, 5.5), spec(2, 7.5, 5.5)),
            grid=free_grid(),
        )
        node = fleet.nodes["vehicle_1"]
        persistent = node.local_state.local_map
        transient = {
            (cell["gx"], cell["gy"])
            for cell in node._planning_map.snapshot()["cells"]
            if cell["state"] == WALL
            and persistent.get_cell(cell["gx"], cell["gy"]) != WALL
        }
        self.assertTrue(transient)
        self.assertEqual(persistent.occupied_cells(), ())

        fleet.world.vehicle("vehicle_2").x = 30.5
        fleet._sample_all(1.0)

        self.assertTrue(
            all(
                node._planning_map.get_cell(*cell) == persistent.get_cell(*cell)
                for cell in transient
            )
        )
        pending = {
            (update.gx, update.gy): update.state
            for update in node._pending_map_delta.changed_cells
        }
        self.assertTrue(
            all(pending.get(cell) == persistent.get_cell(*cell) for cell in transient)
        )
        self.assertEqual(persistent.occupied_cells(), ())

    def test_same_goal_without_peer_identity_blocks_in_bounded_time(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 10.5, 10.5), spec(2, 7.5, 10.5)),
            grid=free_grid(),
        )
        goal = 15.5, 10.5
        for number, vehicle_id in enumerate(sorted(fleet.nodes), 1):
            self.assertTrue(
                fleet.handle_command(
                    vehicle_id,
                    ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
                ).accepted
            )
            self.assertTrue(
                fleet.handle_command(
                    vehicle_id,
                    AutoCommand(
                        2,
                        AutoAction.PUSH,
                        (
                            GotoMission(
                                f"same-goal-{number}",
                                "global_map",
                                *goal,
                                2,
                            ),
                        ),
                    ),
                ).accepted
            )

        minimum_separation = math.inf
        for tick in range(600):
            fleet.tick((tick + 1) * fleet.tick_s)
            first = fleet.world.vehicle("vehicle_1")
            second = fleet.world.vehicle("vehicle_2")
            minimum_separation = min(
                minimum_separation,
                math.hypot(first.x - second.x, first.y - second.y),
            )
            self.assertFalse(first.collision)
            self.assertFalse(second.collision)
            if all(
                node.controller.auto_state.value != "active"
                for node in fleet.nodes.values()
            ):
                break
        else:
            self.fail(
                {
                    vehicle_id: node.controller.snapshot()
                    for vehicle_id, node in fleet.nodes.items()
                }
            )

        self.assertTrue(
            all(
                fleet.world.vehicle(vehicle_id).target_velocities() == (0.0, 0.0)
                for vehicle_id in fleet.nodes
            )
        )
        for drain_tick in range(10):
            if all(
                fleet.world.vehicle(vehicle_id).body_velocities() == (0.0, 0.0)
                for vehicle_id in fleet.nodes
            ):
                break
            fleet.tick((tick + drain_tick + 2) * fleet.tick_s)
            first = fleet.world.vehicle("vehicle_1")
            second = fleet.world.vehicle("vehicle_2")
            minimum_separation = min(
                minimum_separation,
                math.hypot(first.x - second.x, first.y - second.y),
            )
            self.assertFalse(first.collision)
            self.assertFalse(second.collision)
        else:
            self.fail("terminal fleet did not finish braking")

        front = fleet.world.vehicle("vehicle_1")
        trailing = fleet.world.vehicle("vehicle_2")
        self.assertLessEqual(math.dist((front.x, front.y), goal), 0.11)
        # Without P2P identity the rear robot only has anonymous, flickering
        # LiDAR cells, so it must stop conservatively instead of looping forever.
        trailing_controller = fleet.nodes["vehicle_2"].controller
        self.assertEqual(trailing_controller.auto_state.value, "blocked")
        self.assertEqual(trailing_controller.navigation.status, "blocked")
        self.assertEqual(
            trailing_controller.navigation.reason,
            "no_path",
        )
        self.assertEqual(
            trailing_controller.navigation.detail,
            "nearby_safe_goal_unavailable",
        )
        self.assertEqual(front.body_velocities(), (0.0, 0.0))
        self.assertEqual(trailing.body_velocities(), (0.0, 0.0))
        self.assertGreaterEqual(
            minimum_separation,
            front.radius
            + trailing.radius
            + AUTOMATIC_MINIMUM_CLEARANCE_M
            - 1e-9,
        )
        self.assertTrue(
            all(
                not node.local_state.local_map.occupied_cells()
                for node in fleet.nodes.values()
            )
        )

    def test_occupied_goal_finishes_at_a_nearby_safe_pose(self) -> None:
        voxels, grid = generate_map(size=40)
        goal = 15.5, 10.5
        follower_start = 14.299, 12.124
        fleet = peer_fleet(
            AnchorPose(*follower_start, math.atan2(-1.624, 1.201)),
            AnchorPose(*goal, 0.0),
            grid=grid,
            voxels=voxels,
        )
        follower_id = "vehicle_1"
        leader_id = "vehicle_2"
        fleet.handle_command(
            follower_id,
            ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
        )
        fleet.handle_command(
            follower_id,
            AutoCommand(
                2,
                AutoAction.PUSH,
                (GotoMission("occupied-goal", "global_map", *goal, 2),),
            ),
        )

        minimum_separation = math.inf
        final_approach_distances = []
        final_approach_started = False
        for tick in range(180):
            relay_peer_states(fleet)
            fleet.tick((tick + 1) * fleet.tick_s)
            follower = fleet.world.vehicle(follower_id)
            leader = fleet.world.vehicle(leader_id)
            minimum_separation = min(
                minimum_separation,
                math.dist((follower.x, follower.y), (leader.x, leader.y)),
            )
            self.assertFalse(follower.collision)
            self.assertFalse(leader.collision)
            controller = fleet.nodes[follower_id].controller
            snapshot = controller.navigation.snapshot()
            final_approach = snapshot["final_approach"]
            if final_approach:
                final_approach_started = True
                final_approach_distances.append(
                    math.dist((follower.x, follower.y), goal)
                )
            elif final_approach_started and controller.auto_state.value != "idle":
                self.fail("safe final approach returned to its access cell")
            if controller.auto_state.value == "idle":
                break
        else:
            self.fail(fleet.nodes[follower_id].controller.snapshot())

        self.assertEqual(
            fleet.world.vehicle(follower_id).target_velocities(),
            (0.0, 0.0),
        )
        for drain_tick in range(10):
            if fleet.world.vehicle(follower_id).body_velocities() == (0.0, 0.0):
                break
            relay_peer_states(fleet)
            fleet.tick((tick + drain_tick + 2) * fleet.tick_s)
            follower = fleet.world.vehicle(follower_id)
            leader = fleet.world.vehicle(leader_id)
            minimum_separation = min(
                minimum_separation,
                math.dist((follower.x, follower.y), (leader.x, leader.y)),
            )
        else:
            self.fail("nearby safe stop did not finish braking")

        follower = fleet.world.vehicle(follower_id)
        leader = fleet.world.vehicle(leader_id)
        controller = fleet.nodes[follower_id].controller
        self.assertEqual(controller.auto_state.value, "idle")
        self.assertIsNone(controller.active_mission)
        self.assertEqual(controller.snapshot()["mission_queue"]["size"], 0)
        self.assertEqual(controller.navigation.status, "reached")
        self.assertEqual(controller.navigation.reason, "nearby_safe_stop")
        self.assertTrue(final_approach_distances)
        self.assertTrue(
            all(
                after <= before + 1e-9
                for before, after in zip(
                    final_approach_distances,
                    final_approach_distances[1:],
                )
            )
        )
        self.assertLessEqual(
            math.dist((follower.x, follower.y), goal) - follower.radius,
            1.0 + 1e-9,
        )
        self.assertGreaterEqual(
            minimum_separation,
            follower.radius
            + leader.radius
            + AUTOMATIC_MINIMUM_CLEARANCE_M
            - 1e-9,
        )
        self.assertEqual(follower.body_velocities(), (0.0, 0.0))
        self.assertEqual(leader.body_velocities(), (0.0, 0.0))
        self.assertEqual(
            controller.events_after(0)[-1].status,
            "reached",
        )

    def test_two_vehicles_from_example_spawns_settle_at_one_goal(self) -> None:
        fleet = peer_fleet(
            AnchorPose(9.0, 9.0, 0.0),
            AnchorPose(11.0, 11.0, math.pi),
        )
        goal = 13.5, 8.5
        for number, vehicle_id in enumerate(sorted(fleet.nodes), 1):
            fleet.handle_command(
                vehicle_id,
                ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
            )
            fleet.handle_command(
                vehicle_id,
                AutoCommand(
                    2,
                    AutoAction.PUSH,
                    (GotoMission(f"angled-same-goal-{number}", "global_map", *goal, 2),),
                ),
            )

        settled_tick = None
        minimum_separation = math.inf
        tail_positions = {vehicle_id: [] for vehicle_id in fleet.nodes}
        revision_at_tail_start = None
        for tick in range(600):
            relay_peer_states(fleet)
            fleet.tick((tick + 1) * fleet.tick_s)
            first = fleet.world.vehicle("vehicle_1")
            second = fleet.world.vehicle("vehicle_2")
            minimum_separation = min(
                minimum_separation,
                math.dist((first.x, first.y), (second.x, second.y)),
            )
            self.assertFalse(first.collision)
            self.assertFalse(second.collision)
            if tick == 499:
                revision_at_tail_start = {
                    vehicle_id: node.controller.navigation.snapshot()["path_revision"]
                    for vehicle_id, node in fleet.nodes.items()
                }
            if tick >= 500:
                for vehicle_id in fleet.nodes:
                    vehicle = fleet.world.vehicle(vehicle_id)
                    tail_positions[vehicle_id].append((vehicle.x, vehicle.y))
            if settled_tick is None and all(
                node.controller.snapshot()["auto_state"] == "idle"
                for node in fleet.nodes.values()
            ):
                settled_tick = tick + 1

        snapshots = {
            vehicle_id: node.controller.snapshot()
            for vehicle_id, node in fleet.nodes.items()
        }
        self.assertIsNotNone(
            settled_tick,
            {
                "controllers": snapshots,
                "tail_motion_m": {
                    vehicle_id: max(
                        math.dist(points[0], point) for point in points
                    )
                    for vehicle_id, points in tail_positions.items()
                },
                "tail_path_revisions": {
                    vehicle_id: node.controller.snapshot()
                    ["navigation"]["path_revision"]
                    - revision_at_tail_start[vehicle_id]
                    for vehicle_id, node in fleet.nodes.items()
                },
            },
        )
        self.assertTrue(
            all(
                math.dist(points[0], point) <= 0.01
                for points in tail_positions.values()
                for point in points
            )
        )
        self.assertTrue(
            all(
                node.controller.navigation.snapshot()["path_revision"]
                == revision_at_tail_start[vehicle_id]
                for vehicle_id, node in fleet.nodes.items()
            )
        )

        self.assertTrue(
            all(
                node.controller.navigation.status == "reached"
                and node.controller.navigation.goal_mode in {"exact", "nearby_safe"}
                for node in fleet.nodes.values()
            )
        )
        nearby_updates = []
        for vehicle_id, node in fleet.nodes.items():
            if node.controller.navigation.goal_mode == "nearby_safe":
                vehicle = fleet.world.vehicle(vehicle_id)
                self.assertLessEqual(
                    math.dist((vehicle.x, vehicle.y), goal) - vehicle.radius,
                    1.0,
                )
                terminal = node.controller.events_after(0)[-1].as_dict(
                    fleet.timestamp_at()
                )
                nearby_updates.append(terminal)
                self.assertEqual(terminal["status"], "reached")
                self.assertEqual(terminal["reason"], "nearby_safe_stop")
                self.assertEqual(
                    terminal["goal"],
                    {
                        "frame_id": "global_map",
                        "x_m": goal[0],
                        "y_m": goal[1],
                    },
                )
                navigation = terminal["navigation"]
                self.assertEqual(navigation["goal_mode"], "nearby_safe")
                self.assertEqual(navigation["reason"], "nearby_safe_stop")
                self.assertNotEqual(
                    navigation["effective_goal"],
                    navigation["requested_goal"],
                )
                self.assertLessEqual(navigation["approach_distance_m"], 1.0)
        self.assertTrue(nearby_updates)
        self.assertGreaterEqual(
            minimum_separation,
            first.radius + second.radius + HARD_STOP_CLEARANCE_M - 1e-9,
        )
        self.assertTrue(
            all(
                not node.local_state.local_map.occupied_cells()
                and not node.map_sync.peer_evidence("vehicle_1")
                and not node.map_sync.peer_evidence("vehicle_2")
                for node in fleet.nodes.values()
            ),
        )

    def test_unbounded_peer_state_is_rejected_before_planning_projection(self) -> None:
        fleet = peer_fleet(
            AnchorPose(5.0, 5.0, 0.0),
            AnchorPose(10.0, 10.0, 0.0),
        )
        fleet.tick(0.1)
        source = fleet.nodes["vehicle_1"].map_sync
        receiver = fleet.nodes["vehicle_2"]
        payload = source.prepare_peer_state()
        before = receiver._planning_map.snapshot()

        huge_position = json.loads(json.dumps(payload))
        huge_position["pose"]["x_m"] = 1e308
        huge_covariance = json.loads(json.dumps(payload))
        huge_covariance["pose"]["covariance_diagonal"][0] = 1e308
        huge_radius = json.loads(json.dumps(payload))
        huge_radius["footprint_radius_m"] = 1e308
        huge_velocity = json.loads(json.dumps(payload))
        huge_velocity["velocity"]["vx_mps"] = 1e308

        for invalid in (
            huge_position,
            huge_covariance,
            huge_radius,
            huge_velocity,
        ):
            self.assertFalse(
                receiver.map_sync.receive_peer_state(
                    "peer_1",
                    "vehicle_1",
                    invalid,
                    received_at_s=1.0,
                )
            )
        receiver._update_planning_map()

        self.assertEqual(receiver.map_sync.peer_vehicle_states(now_s=1.0), ())
        self.assertEqual(receiver._planning_map.snapshot(), before)

    def test_stale_peer_observation_does_not_refresh_ttl_or_planning(self) -> None:
        fleet = peer_fleet(
            AnchorPose(5.0, 5.0, 0.0),
            AnchorPose(10.0, 10.0, 0.0),
        )
        fleet.tick(0.1)
        source_node = fleet.nodes["vehicle_1"]
        receiver = fleet.nodes["vehicle_2"]
        source = source_node.map_sync
        baseline = receiver._planning_map.snapshot()

        first = source.prepare_peer_state()
        self.assertTrue(
            receiver.map_sync.receive_peer_state(
                "peer_1",
                "vehicle_1",
                first,
                received_at_s=1.0,
            )
        )
        active = receiver.map_sync.peer_vehicle_states(now_s=1.0)
        receiver._update_planning_map(peer_states=active)
        self.assertNotEqual(receiver._planning_map.snapshot(), baseline)
        source.publish_peer_state_result(first["sequence"], True)

        for received_at_s in (1.2, 1.0 + PEER_STATE_TTL_S + 0.1):
            stale = source.prepare_peer_state()
            self.assertFalse(
                receiver.map_sync.receive_peer_state(
                    "peer_1",
                    "vehicle_1",
                    stale,
                    received_at_s=received_at_s,
                )
            )
            source.publish_peer_state_result(stale["sequence"], True)

        expired = receiver.map_sync.peer_vehicle_states(
            now_s=1.0 + PEER_STATE_TTL_S + 0.1
        )
        receiver._update_planning_map(peer_states=expired)
        self.assertEqual(expired, ())
        self.assertEqual(receiver._planning_map.snapshot(), baseline)

        pose = source_node.local_state.pose
        source.record_vehicle_state(
            PoseEstimate(
                pose.anchor_id,
                pose.x_m,
                pose.y_m,
                pose.yaw_rad,
                pose.covariance,
                pose.quality,
                pose.timestamp + 1.0,
                pose.revision + 1,
            ),
            radius_m=fleet.world.vehicle("vehicle_1").radius,
            linear_mps=0.0,
            omega_rps=0.0,
        )
        resumed = source.prepare_peer_state()
        self.assertTrue(
            receiver.map_sync.receive_peer_state(
                "peer_1",
                "vehicle_1",
                resumed,
                received_at_s=1.0 + PEER_STATE_TTL_S + 0.2,
            )
        )
        receiver._update_planning_map(
            peer_states=receiver.map_sync.peer_vehicle_states(
                now_s=1.0 + PEER_STATE_TTL_S + 0.2
            )
        )
        self.assertNotEqual(receiver._planning_map.snapshot(), baseline)

    def test_four_vehicle_control_and_local_state_are_isolated(self) -> None:
        fleet = FleetRuntime.create(
            scenario(
                spec(1, 5.0, 5.0),
                spec(2, 20.0, 5.0),
                spec(3, 5.0, 20.0),
                spec(4, 20.0, 20.0),
            ),
            grid=free_grid(),
            command_timeout=10.0,
        )
        before = fleet.world.truth_snapshot()

        accepted = fleet.handle_command(
            "vehicle_1",
            ManualCommand(1, ManualAction.DRIVE, 0.5, 0.0),
        )
        fleet.tick(0.1)
        after = fleet.world.truth_snapshot()

        self.assertTrue(accepted.accepted)
        self.assertGreater(after["vehicle_1"][0], before["vehicle_1"][0])
        self.assertEqual(after["vehicle_2"], before["vehicle_2"])
        self.assertEqual(after["vehicle_3"], before["vehicle_3"])
        self.assertEqual(after["vehicle_4"], before["vehicle_4"])
        fleet.tick(0.2)
        self.assertGreater(fleet.nodes["vehicle_1"].local_state.pose.x_m, 0.0)
        self.assertEqual(fleet.nodes["vehicle_2"].local_state.pose.x_m, 0.0)
        self.assertEqual(len({id(node.controller) for node in fleet.nodes.values()}), 4)
        self.assertEqual(len({id(node.local_state.local_map) for node in fleet.nodes.values()}), 4)

    def test_fixed_tick_is_repeatable(self) -> None:
        fleet_scenario = scenario(
            spec(1, 5.0, 5.0),
            spec(2, 20.0, 5.0),
        )
        first = FleetRuntime.create(
            fleet_scenario,
            grid=free_grid(),
            command_timeout=10.0,
        )
        second = FleetRuntime.create(
            fleet_scenario,
            grid=free_grid(),
            command_timeout=10.0,
        )
        for fleet in (first, second):
            fleet.handle_command(
                "vehicle_1",
                ManualCommand(1, ManualAction.DRIVE, 0.4, 0.2),
            )
            for tick in range(10):
                fleet.tick(tick / 10)

        self.assertEqual(first.world.truth_snapshot(), second.world.truth_snapshot())
        self.assertEqual(first.world.now, second.world.now)

    def test_odometry_noise_is_vehicle_specific_and_order_independent(self) -> None:
        ordered = scenario(spec(1, 5.0, 5.0), spec(2, 20.0, 5.0))
        reversed_order = scenario(spec(2, 20.0, 5.0), spec(1, 5.0, 5.0))
        config = OdometryConfig(0.1, 0.05, 42)

        poses_by_run = []
        seeds_by_run = []
        for fleet_scenario in (ordered, reversed_order):
            fleet = FleetRuntime.create(
                fleet_scenario,
                grid=free_grid(),
                odometry_config=config,
            )
            poses = {}
            seeds = {}
            for vehicle_id in sorted(fleet.nodes):
                vehicle = fleet.world.vehicle(vehicle_id)
                local_state = fleet.nodes[vehicle_id].local_state
                poses[vehicle_id] = local_state.update_from_truth(
                    vehicle.x + 1.0,
                    vehicle.y,
                    vehicle.yaw + 0.1,
                    timestamp=1.0,
                )
                seeds[vehicle_id] = local_state.odometry.config.seed
            poses_by_run.append(poses)
            seeds_by_run.append(seeds)

        self.assertNotEqual(seeds_by_run[0]["vehicle_1"], seeds_by_run[0]["vehicle_2"])
        self.assertNotEqual(
            (
                poses_by_run[0]["vehicle_1"].x_m,
                poses_by_run[0]["vehicle_1"].y_m,
                poses_by_run[0]["vehicle_1"].yaw_rad,
            ),
            (
                poses_by_run[0]["vehicle_2"].x_m,
                poses_by_run[0]["vehicle_2"].y_m,
                poses_by_run[0]["vehicle_2"].yaw_rad,
            ),
        )
        self.assertEqual(seeds_by_run[0], seeds_by_run[1])
        self.assertEqual(poses_by_run[0], poses_by_run[1])

    def test_simultaneous_arbitration_prevents_order_dependent_overlap(self) -> None:
        fleet = FleetRuntime.create(
            scenario(
                spec(1, 10.0, 10.0),
                spec(2, 13.0, 10.0, math.pi),
                tick_ms=1000,
            ),
            grid=free_grid(),
            linear_speed=5.0,
            command_timeout=10.0,
        )
        starts = fleet.world.truth_snapshot()
        for vehicle_id in fleet.nodes:
            fleet.handle_command(
                vehicle_id,
                ManualCommand(1, ManualAction.DRIVE, 5.0, 0.0),
            )

        fleet.tick(1.0)
        fleet.tick(2.0)
        poses = fleet.world.truth_snapshot()
        distance_squared = (
            (poses["vehicle_1"][0] - poses["vehicle_2"][0]) ** 2
            + (poses["vehicle_1"][1] - poses["vehicle_2"][1]) ** 2
        )

        self.assertGreater(poses["vehicle_1"][0], starts["vehicle_1"][0])
        self.assertLess(poses["vehicle_2"][0], starts["vehicle_2"][0])
        self.assertAlmostEqual(
            poses["vehicle_1"][0] - starts["vehicle_1"][0],
            starts["vehicle_2"][0] - poses["vehicle_2"][0],
        )
        self.assertFalse(is_strict_overlap(distance_squared, 1.0))
        self.assertEqual(
            fleet.world.vehicle("vehicle_1").body_velocities(),
            (0.0, 0.0),
        )
        self.assertEqual(
            fleet.world.vehicle("vehicle_2").body_velocities(),
            (0.0, 0.0),
        )

        reversed_fleet = FleetRuntime.create(
            scenario(
                spec(2, 13.0, 10.0, math.pi),
                spec(1, 10.0, 10.0),
                tick_ms=1000,
            ),
            grid=free_grid(),
            linear_speed=5.0,
            command_timeout=10.0,
        )
        for vehicle_id in reversed_fleet.nodes:
            reversed_fleet.handle_command(
                vehicle_id,
                ManualCommand(1, ManualAction.DRIVE, 5.0, 0.0),
            )
        reversed_fleet.tick(1.0)
        reversed_fleet.tick(2.0)

        self.assertEqual(reversed_fleet.world.truth_snapshot(), poses)
        self.assertTrue(
            all(
                reversed_fleet.world.vehicle(vehicle_id).body_velocities()
                == (0.0, 0.0)
                for vehicle_id in reversed_fleet.nodes
            )
        )

    def test_accelerating_trajectories_use_physical_time_for_arbitration(self) -> None:
        fleet = FleetRuntime.create(
            scenario(
                spec(1, 4.7, 5.0),
                spec(2, 5.0, 4.775, math.pi / 2),
                tick_ms=1000,
            ),
            grid=free_grid(),
            radius=0.01,
            linear_speed=1.0,
            linear_acceleration_mps2=1.0,
            command_timeout=2.0,
            spawn_safety_margin_m=0.0,
        )
        first = fleet.world.vehicle("vehicle_1")
        second = fleet.world.vehicle("vehicle_2")
        first.install_drive(1.0, 0.0, 0.0)
        second.install_drive(0.5, 0.0, 0.0)

        results = fleet.world.advance_to(1.0)

        self.assertTrue(all(not result.stopped for result in results.values()))
        poses = fleet.world.truth_snapshot()
        self.assertEqual(poses["vehicle_1"], (5.2, 5.0, 0.0))
        self.assertAlmostEqual(poses["vehicle_2"][0], 5.0)
        self.assertAlmostEqual(poses["vehicle_2"][1], 5.15)
        self.assertAlmostEqual(poses["vehicle_2"][2], math.pi / 2)

    def test_coarse_accelerating_collision_matches_fine_time_steps(self) -> None:
        def collided(tick_s: float) -> dict[str, bool]:
            fleet = FleetRuntime.create(
                scenario(
                    spec(1, 4.71875, 5.0),
                    spec(2, 5.0, 4.75, math.pi / 2),
                    tick_ms=round(tick_s * 1000),
                ),
                grid=free_grid(),
                radius=0.01,
                linear_speed=1.0,
                linear_acceleration_mps2=1.0,
                command_timeout=2.0,
                spawn_safety_margin_m=0.0,
            )
            fleet.world.vehicle("vehicle_1").install_drive(1.0, 0.0, 0.0)
            fleet.world.vehicle("vehicle_2").install_drive(0.5, 0.0, 0.0)
            stopped = {vehicle_id: False for vehicle_id in fleet.nodes}
            now = 0.0
            while now < 1.0:
                now = min(1.0, now + tick_s)
                for vehicle_id, result in fleet.world.advance_to(now).items():
                    stopped[vehicle_id] |= result.stopped
            return stopped

        self.assertEqual(
            collided(1.0),
            collided(0.01),
        )
        self.assertEqual(
            collided(1.0),
            {"vehicle_1": True, "vehicle_2": True},
        )

    def test_low_speed_reversal_collision_matches_fine_time_steps(self) -> None:
        def stopped(tick_s: float) -> dict[str, bool]:
            fleet = FleetRuntime.create(
                scenario(
                    spec(1, 2.492, 5.0),
                    spec(2, 2.511, 5.0),
                    tick_ms=round(tick_s * 1000),
                ),
                grid=free_grid(),
                radius=0.005,
                linear_speed=0.1,
                linear_acceleration_mps2=1.0,
                linear_deceleration_mps2=1.0,
                command_timeout=5.0,
                spawn_safety_margin_m=0.0,
            )
            fleet.world.vehicle("vehicle_1").install_drive(0.1, 0.0, 0.0)
            fleet.world.advance_to(0.1)
            fleet.world.vehicle("vehicle_1").install_drive(-0.1, 0.0, 0.1)
            results = {vehicle_id: False for vehicle_id in fleet.nodes}
            now = 0.1
            while now < 0.3:
                now = min(0.3, now + tick_s)
                for vehicle_id, result in fleet.world.advance_to(now).items():
                    results[vehicle_id] |= result.stopped
            return results

        self.assertEqual(stopped(0.2), stopped(0.01))
        self.assertEqual(
            stopped(0.2),
            {"vehicle_1": True, "vehicle_2": True},
        )

    def test_reversal_safety_observes_the_executed_direction(self) -> None:
        grid = free_grid()
        fleet = FleetRuntime.create(
            scenario(
                spec(1, 2.4, 5.0),
                spec(2, 20.0, 5.0),
            ),
            grid=grid,
            linear_speed=1.0,
            linear_deceleration_mps2=1.0,
            command_timeout=10.0,
        )
        fleet.handle_command(
            "vehicle_1",
            ManualCommand(1, ManualAction.DRIVE, 1.0, 0.0),
        )
        for tick in range(10):
            fleet.tick((tick + 1) * fleet.tick_s)
        grid.set_cell(4, 5, WALL)

        fleet.handle_command(
            "vehicle_1",
            ManualCommand(2, ManualAction.DRIVE, -1.0, 0.0),
        )
        fleet.tick(1.1)

        observation = fleet.nodes["vehicle_1"].safety.observation
        self.assertIsNotNone(observation.obstacle_clearance_m)
        self.assertGreater(
            fleet.world.vehicle("vehicle_1").body_velocities()[0],
            0.0,
        )

    def test_serve_and_fleet_use_the_same_safety_preparation(self) -> None:
        grid = free_grid()
        served = VehicleRuntime.create(
            started_at=0.0,
            anchor=AnchorSpec("serve", 2.4, 5.0, 0.0),
            odometry_config=OdometryConfig(),
            linear_speed=1.0,
            linear_deceleration_mps2=1.0,
            command_timeout=10.0,
        )
        served.grid = grid
        served.vehicle = Vehicle(
            2.4,
            5.0,
            linear_speed=1.0,
            linear_deceleration_mps2=1.0,
            command_timeout=10.0,
            now=0.0,
        )
        fleet = FleetRuntime.create(
            scenario(spec(1, 2.4, 5.0), spec(2, 20.0, 5.0)),
            grid=grid,
            linear_speed=1.0,
            linear_deceleration_mps2=1.0,
            command_timeout=10.0,
        )
        fleet_vehicle = fleet.world.vehicle("vehicle_1")
        for vehicle in (served.vehicle, fleet_vehicle):
            vehicle.install_drive(1.0, 0.0, 0.0)
        served.vehicle.advance(grid, 1.0)
        fleet.world.advance_to(1.0)
        fleet_vehicle = fleet.world.vehicle("vehicle_1")
        grid.set_cell(4, 5, WALL)

        served.advance_to(1.1)
        fleet._advance_world(1.1)
        fleet_vehicle = fleet.world.vehicle("vehicle_1")

        self.assertEqual(
            (
                served.vehicle.x,
                served.vehicle.body_velocities(),
                served.vehicle.target_velocities(),
                served.safety.decision,
            ),
            (
                fleet_vehicle.x,
                fleet_vehicle.body_velocities(),
                fleet_vehicle.target_velocities(),
                fleet.nodes["vehicle_1"].safety.decision,
            ),
        )

    def test_curved_motion_cannot_pass_through_another_vehicle(self) -> None:
        for tick_ms in (100, 1000):
            with self.subTest(tick_ms=tick_ms):
                fleet = FleetRuntime.create(
                    scenario(
                        spec(1, 5.0, 5.0),
                        spec(2, 5.897, 4.421),
                        tick_ms=tick_ms,
                    ),
                    grid=free_grid(),
                    command_timeout=10.0,
                    spawn_safety_margin_m=0.0,
                )
                moving = fleet.world.vehicle("vehicle_1")
                moving.install_drive(0.5, math.pi / 2, fleet.world.now)
                stopped = False

                for _ in range(1000 // tick_ms):
                    results = fleet.world.advance_to(
                        fleet.world.now + tick_ms / 1000
                    )
                    stopped = stopped or results["vehicle_1"].stopped
                    first = fleet.world.vehicle("vehicle_1")
                    second = fleet.world.vehicle("vehicle_2")
                    distance_squared = (
                        (first.x - second.x) ** 2 + (first.y - second.y) ** 2
                    )
                    self.assertFalse(is_strict_overlap(distance_squared, 1.0))

                self.assertTrue(stopped)
                self.assertEqual(
                    fleet.world.vehicle("vehicle_1").body_velocities(),
                    (0.0, 0.0),
                )

    def test_disconnecting_one_endpoint_does_not_stop_another_vehicle(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), spec(2, 20.0, 5.0)),
            grid=free_grid(),
            command_timeout=10.0,
        )
        for vehicle_id in fleet.nodes:
            fleet.handle_command(
                vehicle_id,
                ManualCommand(1, ManualAction.DRIVE, 0.5, 0.0),
            )
        fleet.disconnect("vehicle_1")

        fleet.tick(0.1)

        self.assertEqual(fleet.world.truth_snapshot()["vehicle_1"][:2], (5.0, 5.0))
        self.assertGreater(fleet.world.truth_snapshot()["vehicle_2"][0], 20.0)

    def test_tmini_keeps_six_hz_schedule_independent_of_control_tick(self) -> None:
        duration_s = 2.0
        expected_offsets = [index / 6 for index in range(1, 13)]
        observed_by_tick = {}

        for tick_ms in (50, 100, 250, 1000):
            fleet = FleetRuntime.create(
                scenario(spec(1, 5.0, 5.0), tick_ms=tick_ms),
                grid=free_grid(),
                started_at=10.0,
                timestamp=1_000.0,
                command_timeout=10.0,
            )
            node = fleet.nodes["vehicle_1"]
            _, initial_scan = fleet.telemetry_messages("vehicle_1")
            self.assertEqual(initial_scan["seq"], 0)
            self.assertEqual(initial_scan["timestamp_s"], 1_000.0)
            self.assertEqual(initial_scan["config"]["scan_rate_hz"], 6)
            self.assertAlmostEqual(initial_scan["config"]["scan_time"], 1 / 6)
            original_sample = node.sample
            timestamps = []

            def record_sample(*args, **kwargs):
                timestamps.append(args[2])
                return original_sample(*args, **kwargs)

            node.sample = record_sample
            for _ in range(round(duration_s / fleet.tick_s)):
                fleet.tick(-123.0)

            observed_by_tick[tick_ms] = timestamps
            self.assertEqual(len(timestamps), 12)
            for actual, offset in zip(timestamps, expected_offsets):
                self.assertAlmostEqual(actual, 1_000.0 + offset, places=9)
            self.assertEqual(node.frame_sequence, 12)
            self.assertAlmostEqual(node.latest_frame.timestamp, 1_002.0, places=9)

        baseline = observed_by_tick[50]
        for timestamps in observed_by_tick.values():
            self.assertEqual(len(timestamps), len(baseline))
            for actual, expected in zip(timestamps, baseline):
                self.assertAlmostEqual(actual, expected, places=9)

    def test_large_tick_samples_intermediate_curved_poses(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), tick_ms=1000),
            grid=free_grid(),
            linear_speed=1.0,
            angular_speed=math.pi / 2,
            command_timeout=10.0,
        )
        fleet.handle_command(
            "vehicle_1",
            ManualCommand(1, ManualAction.DRIVE, 1.0, math.pi / 2),
        )
        sampled_poses = []

        from mockvehicle2d import fleet as fleet_module

        real_scan_grid = fleet_module.scan_grid

        def record_scan(grid, x, y, yaw, config, *, circles=()):
            sampled_poses.append((x, y, yaw))
            return real_scan_grid(grid, x, y, yaw, config, circles=circles)

        with patch("mockvehicle2d.fleet.scan_grid", side_effect=record_scan):
            fleet.tick(999.0)

        self.assertEqual(len(sampled_poses), 6)
        self.assertNotEqual(sampled_poses[0], sampled_poses[-1])
        self.assertLess(sampled_poses[0][0], sampled_poses[-1][0])
        self.assertLess(sampled_poses[0][1], sampled_poses[-1][1])
        self.assertAlmostEqual(sampled_poses[0][2], math.pi / 72, places=9)
        self.assertAlmostEqual(sampled_poses[-1][2], 3 * math.pi / 8, places=9)

    def test_dynamic_vehicle_is_raycast_at_the_same_sensor_time(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), spec(2, 7.0, 5.0), tick_ms=1000),
            grid=free_grid(),
            linear_speed=0.5,
            command_timeout=10.0,
            spawn_safety_margin_m=0.0,
        )
        for vehicle_id in fleet.nodes:
            fleet.handle_command(
                vehicle_id,
                ManualCommand(1, ManualAction.DRIVE, 0.5, 0.0),
            )
        ranges = []
        node = fleet.nodes["vehicle_1"]
        original_sample = node.sample

        def record_sample(*args, **kwargs):
            ranges.append(args[1][0].range)
            return original_sample(*args, **kwargs)

        node.sample = record_sample
        fleet.tick(1.0)

        self.assertEqual(len(ranges), 6)
        for measured in ranges:
            self.assertAlmostEqual(measured, 1.5, places=9)

    def test_coarse_tick_frames_use_the_command_state_at_scan_time(self) -> None:
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), tick_ms=1000),
            grid=free_grid(),
            linear_speed=1.0,
            command_timeout=0.5,
        )
        fleet.handle_command(
            "vehicle_1",
            ManualCommand(1, ManualAction.DRIVE, 1.0, 0.0),
        )

        fleet.tick(1.0)

        frames = fleet.nodes["vehicle_1"].frames_after(0)
        self.assertEqual(len(frames), 6)
        self.assertEqual(
            [frame.runtime_state["actuator_command"] for frame in frames],
            ["drive", "drive", "stop", "stop", "stop", "stop"],
        )
        for actual, expected in zip(
            [frame.runtime_state["linear_mps"] for frame in frames],
            (1 / 6, 1 / 3, 1 / 2, 1 / 3, 1 / 6, 0.0),
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [
                frame.runtime_state["controller"]["manual_setpoint_active"]
                for frame in frames
            ],
            [True, True, False, False, False, False],
        )
        self.assertAlmostEqual(frames[1].truth_pose[0], 5.0 + 1 / 18)
        self.assertAlmostEqual(frames[-1].truth_pose[0], 5.25)

    def test_coarse_tick_frames_show_bounded_stop_before_a_wall(self) -> None:
        grid = free_grid()
        grid.set_cell(6, 5, WALL)
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.2, 5.0), tick_ms=1000),
            grid=grid,
            linear_speed=1.0,
            command_timeout=10.0,
            spawn_safety_margin_m=0.0,
        )
        fleet.handle_command(
            "vehicle_1",
            ManualCommand(1, ManualAction.DRIVE, 1.0, 0.0),
        )

        fleet.tick(1.0)

        frames = fleet.nodes["vehicle_1"].frames_after(0)
        stopped_at = next(
            index
            for index, frame in enumerate(frames)
            if frame.runtime_state["actuator_command"] == "stop"
        )
        self.assertGreater(stopped_at, 0)
        self.assertTrue(
            all(not frame.runtime_state["collision"] for frame in frames[:stopped_at])
        )
        self.assertTrue(
            all(not frame.runtime_state["collision"] for frame in frames)
        )
        self.assertEqual(frames[-1].runtime_state["linear_mps"], 0.0)
        self.assertLessEqual(
            frames[-1].truth_pose[0] + 0.5,
            6.0 - HARD_STOP_CLEARANCE_M,
        )


class TestFleetTelemetryWebSocket(unittest.IsolatedAsyncioTestCase):
    async def test_reconnected_clients_continue_the_vehicle_frame_sequence(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), tick_ms=1000),
            grid=free_grid(),
            command_timeout=20.0,
            realtime_factor=3.0,
        )

        async def handler(websocket) -> None:
            await fleet_handler(websocket, fleet=fleet, vehicle_id="vehicle_1")

        async def receive_scan(websocket, sequence: int) -> dict[str, object]:
            while True:
                raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if message.get("type") == "scan" and message.get("seq") == sequence:
                    return message

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}") as first:
                hello = json.loads(await first.recv())
                self.assertEqual(
                    hello["mission_types"], ["goto", "patrol", "coverage"]
                )
                self.assertEqual(hello["realtime_factor"], 3.0)
                await receive_scan(first, 0)
                fleet.tick(1.0)
                first_sequences = [
                    (await receive_scan(first, sequence))["seq"]
                    for sequence in range(1, 7)
                ]
            for _ in range(100):
                if not fleet.nodes["vehicle_1"].controller_lease.locked():
                    break
                await asyncio.sleep(0.001)

            async with connect(f"ws://127.0.0.1:{port}") as second:
                await receive_scan(second, 6)
                fleet.tick(2.0)
                second_sequences = [
                    (await receive_scan(second, sequence))["seq"]
                    for sequence in range(7, 13)
                ]

            self.assertEqual(first_sequences + second_sequences, list(range(1, 13)))
        finally:
            server.close()
            await server.wait_closed()

    async def test_every_scheduled_tmini_frame_reaches_the_client(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        for tick_ms in (50, 100, 250, 1000):
            with self.subTest(tick_ms=tick_ms):
                fleet = FleetRuntime.create(
                    scenario(spec(1, 5.0, 5.0), tick_ms=tick_ms),
                    grid=free_grid(),
                    started_at=10.0,
                    timestamp=1_000.0,
                    command_timeout=10.0,
                )

                async def handler(websocket) -> None:
                    await fleet_handler(
                        websocket,
                        fleet=fleet,
                        vehicle_id="vehicle_1",
                    )

                server = await serve(handler, "127.0.0.1", 0)
                port = server.sockets[0].getsockname()[1]
                try:
                    async with connect(f"ws://127.0.0.1:{port}") as websocket:
                        while True:
                            raw = await websocket.recv()
                            if isinstance(raw, bytes):
                                continue
                            initial = json.loads(raw)
                            if initial.get("type") == "scan":
                                self.assertEqual(initial["seq"], 0)
                                break

                        for _ in range(round(2.0 / fleet.tick_s)):
                            fleet.tick(-123.0)

                        scans = []
                        while len(scans) < 12:
                            message = json.loads(
                                await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            )
                            if message.get("type") == "scan":
                                scans.append(message)

                        self.assertEqual([scan["seq"] for scan in scans], list(range(1, 13)))
                        for sequence, scan in enumerate(scans, 1):
                            self.assertAlmostEqual(
                                scan["timestamp_s"],
                                1_000.0 + sequence / 6,
                                places=9,
                            )
                finally:
                    server.close()
                    await server.wait_closed()

    async def test_backlogged_frames_keep_the_state_from_their_scan_time(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        with tempfile.TemporaryDirectory() as temporary:
            vehicle_spec = FleetVehicleSpec(
                "vehicle_1",
                19090,
                "spawn_1",
                AnchorPose(5.0, 5.0, 0.0),
                20090,
            )
            fleet = FleetRuntime.create(
                FleetScenario(
                    "snapshot_scenario",
                    (vehicle_spec,),
                    1000,
                    P2PSettings(Path("/bin/true"), Path(temporary)),
                ),
                grid=free_grid(),
                started_at=10.0,
                timestamp=1_000.0,
                command_timeout=20.0,
            )
            node = fleet.nodes["vehicle_1"]

            async def handler(websocket) -> None:
                await fleet_handler(websocket, fleet=fleet, vehicle_id="vehicle_1")

            server = await serve(handler, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    while True:
                        raw = await websocket.recv()
                        if isinstance(raw, str) and json.loads(raw).get("type") == "scan":
                            break

                    node.map_sync.published_deltas = 1
                    fleet.handle_command(
                        "vehicle_1",
                        ManualCommand(1, ManualAction.DRIVE, 0.5, 0.0),
                    )
                    fleet.tick(-123.0)
                    node.map_sync.published_deltas = 2
                    fleet.handle_command(
                        "vehicle_1",
                        ManualCommand(2, ManualAction.STOP),
                    )
                    node.safety.healthy = False
                    fleet.tick(-123.0)
                    node.map_sync.published_deltas = 99

                    poses = []
                    while len(poses) < 12:
                        message = json.loads(
                            await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        )
                        if message.get("type") == "pose" and message["seq"]:
                            poses.append(message)

                    self.assertEqual([pose["seq"] for pose in poses], list(range(1, 13)))
                    self.assertEqual(
                        [pose["actuator_command"] for pose in poses],
                        ["drive"] * 6 + ["stop"] * 6,
                    )
                    self.assertEqual(
                        [pose["controller"]["manual_setpoint_active"] for pose in poses],
                        [True] * 6 + [False] * 6,
                    )
                    self.assertEqual(
                        [pose["safety"]["reason"] for pose in poses],
                        [None] * 6 + ["safety_sensor_fault"] * 6,
                    )
                    self.assertEqual(
                        [pose["p2p_map_sync"]["published_deltas"] for pose in poses],
                        [1] * 6 + [2] * 6,
                    )
                    self.assertEqual(
                        [pose["localization"]["scan_match"]["revision"] for pose in poses],
                        list(range(2, 14)),
                    )
                    for pose, expected_vx in zip(
                        poses,
                        (
                            1 / 6,
                            1 / 3,
                            1 / 2,
                            1 / 2,
                            1 / 2,
                            1 / 2,
                            1 / 3,
                            1 / 6,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ),
                    ):
                        self.assertAlmostEqual(pose["vx_mps"], expected_vx)
                        self.assertEqual(pose["vy_mps"], 0.0)
            finally:
                server.close()
                await server.wait_closed()

    async def test_slow_client_gets_explicit_overflow_without_blocking_ticks(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), tick_ms=1000),
            grid=free_grid(),
            started_at=10.0,
            timestamp=1_000.0,
            command_timeout=20.0,
        )

        async def handler(websocket) -> None:
            await fleet_handler(websocket, fleet=fleet, vehicle_id="vehicle_1")

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                while True:
                    raw = await websocket.recv()
                    if isinstance(raw, str):
                        initial = json.loads(raw)
                        if initial.get("type") == "scan":
                            break

                for _ in range(11):
                    fleet.tick(-123.0)

                self.assertAlmostEqual(fleet.world.now, 21.0)
                while True:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    if isinstance(raw, str):
                        message = json.loads(raw)
                        if message.get("type") == "error":
                            break
                self.assertEqual(message["code"], "telemetry_overflow")
                self.assertEqual(message["oldest_available_seq"], 3)
                self.assertEqual(message["latest_available_seq"], 66)
                self.assertEqual(
                    message["latest_available_seq"]
                    - message["oldest_available_seq"]
                    + 1,
                    64,
                )
        finally:
            server.close()
            await server.wait_closed()


class TestFleetMainCleanup(unittest.IsolatedAsyncioTestCase):
    def configured_runtime(self, temporary: str):
        ports = (19101, 19102, 20101, 20102)
        settings = P2PSettings(Path("/bin/true"), Path(temporary))
        fleet_scenario = FleetScenario(
            "cleanup_scenario",
            (
                FleetVehicleSpec(
                    "vehicle_1",
                    ports[0],
                    "spawn_1",
                    AnchorPose(5.0, 5.0, 0.0),
                    ports[2],
                ),
                FleetVehicleSpec(
                    "vehicle_2",
                    ports[1],
                    "spawn_2",
                    AnchorPose(20.0, 5.0, 0.0),
                    ports[3],
                ),
            ),
            100,
            settings,
        )
        fleet = FleetRuntime.create(fleet_scenario, grid=free_grid())
        fleet.run = AsyncMock(side_effect=ValueError("injected tick failure"))
        fleet.disconnect = Mock(side_effect=[RuntimeError("disconnect one"), None])
        return fleet_scenario, fleet

    async def test_main_preserves_original_and_collects_every_cleanup_failure(self) -> None:
        class Server:
            def __init__(self, close_error=None, wait_error=None):
                self.close_error = close_error
                self.wait_error = wait_error
                self.close_called = False
                self.wait_called = False

            def close(self):
                self.close_called = True
                if self.close_error is not None:
                    raise self.close_error

            async def wait_closed(self):
                self.wait_called = True
                if self.wait_error is not None:
                    raise self.wait_error

        with tempfile.TemporaryDirectory() as temporary:
            fleet_scenario, fleet = self.configured_runtime(temporary)
            first = Server(close_error=RuntimeError("server close one"))
            second = Server(wait_error=RuntimeError("server wait two"))
            sync = Mock()
            sync.close = AsyncMock(side_effect=RuntimeError("p2p close"))

            from mockvehicle2d import fleet as fleet_module

            with (
                patch.object(fleet_module.FleetScenario, "load", return_value=fleet_scenario),
                patch.object(fleet_module.FleetRuntime, "create", return_value=fleet),
                patch.object(
                    fleet_module.P2PFleetSync,
                    "start",
                    new=AsyncMock(return_value=sync),
                ),
                patch(
                    "websockets.asyncio.server.serve",
                    new=AsyncMock(side_effect=[first, second]),
                ),
                self.assertRaisesRegex(ValueError, "injected tick failure") as raised,
            ):
                await fleet_module.main("unused.json")

            cause = raised.exception.__cause__
            self.assertIsNotNone(cause)
            details = str(cause)
            for message in (
                "server close one",
                "server wait two",
                "p2p close",
                "disconnect one",
            ):
                self.assertIn(message, details)
            self.assertTrue(first.close_called and second.close_called)
            self.assertTrue(first.wait_called and second.wait_called)
            sync.close.assert_awaited_once()
            self.assertEqual(fleet.disconnect.call_count, 2)

    async def test_main_bounds_a_hanging_server_wait_and_continues_cleanup(self) -> None:
        class HangingServer:
            def __init__(self):
                self.close_called = False
                self.wait_started = asyncio.Event()

            def close(self):
                self.close_called = True

            async def wait_closed(self):
                self.wait_started.set()
                await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as temporary:
            fleet_scenario, fleet = self.configured_runtime(temporary)
            fleet.disconnect = Mock()
            servers = (HangingServer(), HangingServer())
            sync = Mock()
            sync.close = AsyncMock()

            from mockvehicle2d import fleet as fleet_module

            with (
                patch.object(fleet_module.FleetScenario, "load", return_value=fleet_scenario),
                patch.object(fleet_module.FleetRuntime, "create", return_value=fleet),
                patch.object(
                    fleet_module.P2PFleetSync,
                    "start",
                    new=AsyncMock(return_value=sync),
                ),
                patch(
                    "websockets.asyncio.server.serve",
                    new=AsyncMock(side_effect=servers),
                ),
                patch.object(fleet_module, "FLEET_CLEANUP_TIMEOUT_S", 0.05),
                self.assertRaisesRegex(ValueError, "injected tick failure") as raised,
            ):
                await asyncio.wait_for(fleet_module.main("unused.json"), timeout=0.5)

            self.assertIn("WebSocket server wait did not stop", str(raised.exception.__cause__))
            sync.close.assert_awaited_once()
            self.assertEqual(fleet.disconnect.call_count, 2)

    async def test_main_waits_for_a_valid_slow_p2p_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fleet_scenario, fleet = self.configured_runtime(temporary)
            fleet.disconnect = Mock()
            closed = asyncio.Event()

            async def slow_close() -> None:
                await asyncio.sleep(2.05)
                closed.set()

            sync = Mock()
            sync.close = AsyncMock(side_effect=slow_close)
            servers = []
            for _ in fleet_scenario.vehicles:
                server = Mock()
                server.wait_closed = AsyncMock()
                servers.append(server)

            from mockvehicle2d import fleet as fleet_module

            started = asyncio.get_running_loop().time()
            try:
                with (
                    patch.object(fleet_module.FleetScenario, "load", return_value=fleet_scenario),
                    patch.object(fleet_module.FleetRuntime, "create", return_value=fleet),
                    patch.object(
                        fleet_module.P2PFleetSync,
                        "start",
                        new=AsyncMock(return_value=sync),
                    ),
                    patch(
                        "websockets.asyncio.server.serve",
                        new=AsyncMock(side_effect=servers),
                    ),
                    self.assertRaisesRegex(ValueError, "injected tick failure"),
                ):
                    await fleet_module.main("unused.json")
            finally:
                if not closed.is_set():
                    try:
                        await asyncio.wait_for(closed.wait(), timeout=0.2)
                    except asyncio.TimeoutError:
                        pass

            self.assertTrue(closed.is_set())
            self.assertGreaterEqual(asyncio.get_running_loop().time() - started, 2.05)
            sync.close.assert_awaited_once()

    async def test_main_waits_for_cancel_resistant_p2p_cleanup_to_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fleet_scenario, fleet = self.configured_runtime(temporary)
            fleet.disconnect = Mock()
            cancelled = asyncio.Event()
            closed = asyncio.Event()

            async def resistant_close() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await asyncio.sleep(0.03)
                    closed.set()

            sync = Mock()
            sync.close = AsyncMock(side_effect=resistant_close)
            servers = []
            for _ in fleet_scenario.vehicles:
                server = Mock()
                server.wait_closed = AsyncMock()
                servers.append(server)

            from mockvehicle2d import fleet as fleet_module

            with (
                patch.object(fleet_module.FleetScenario, "load", return_value=fleet_scenario),
                patch.object(fleet_module.FleetRuntime, "create", return_value=fleet),
                patch.object(
                    fleet_module.P2PFleetSync,
                    "start",
                    new=AsyncMock(return_value=sync),
                ),
                patch(
                    "websockets.asyncio.server.serve",
                    new=AsyncMock(side_effect=servers),
                ),
                patch("mockvehicle2d.map_sync.PROCESS_STOP_TIMEOUT_S", 0.01),
                patch.object(fleet_module, "FLEET_CLEANUP_TIMEOUT_S", 0.05),
                self.assertRaisesRegex(ValueError, "injected tick failure") as raised,
            ):
                await asyncio.wait_for(fleet_module.main("unused.json"), timeout=0.6)

            self.assertTrue(cancelled.is_set())
            self.assertTrue(closed.is_set())
            self.assertIn("libp2p fleet sync close did not stop", str(raised.exception.__cause__))
            sync.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
