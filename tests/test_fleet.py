"""Shared-world multi-vehicle simulation invariants."""

import asyncio
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mockvehicle2d.collision import is_strict_overlap
from mockvehicle2d.controller import ManualAction, ManualCommand
from mockvehicle2d.fleet import (
    AnchorPose,
    FleetRuntime,
    FleetScenario,
    FleetVehicleSpec,
    fleet_handler,
)
from mockvehicle2d.local_state import OdometryConfig
from mockvehicle2d.map_grid import WALL, MapGrid
from mockvehicle2d.map_sync import P2PSettings


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


class TestFleetScenario(unittest.TestCase):
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
        self.assertAlmostEqual(sampled_poses[0][2], math.pi / 12, places=2)
        self.assertAlmostEqual(sampled_poses[-1][2], math.pi / 2, places=9)

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
        self.assertEqual(
            [frame.runtime_state["linear_mps"] for frame in frames],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            [
                frame.runtime_state["controller"]["manual_setpoint_active"]
                for frame in frames
            ],
            [True, True, False, False, False, False],
        )
        self.assertAlmostEqual(frames[1].truth_pose[0], 5.0 + 1 / 3)
        self.assertAlmostEqual(frames[-1].truth_pose[0], 5.5)

    def test_coarse_tick_frames_change_state_only_after_a_mid_tick_collision(self) -> None:
        grid = free_grid()
        grid.set_cell(6, 5, WALL)
        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), tick_ms=1000),
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
        self.assertTrue(frames[stopped_at].runtime_state["collision"])
        self.assertEqual(
            len({frame.truth_pose for frame in frames[stopped_at:]}),
            1,
        )


class TestFleetTelemetryWebSocket(unittest.IsolatedAsyncioTestCase):
    async def test_reconnected_clients_continue_the_vehicle_frame_sequence(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        fleet = FleetRuntime.create(
            scenario(spec(1, 5.0, 5.0), tick_ms=1000),
            grid=free_grid(),
            command_timeout=20.0,
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
                    node.safety.decision = type(node.safety.decision)(
                        0.0,
                        0.0,
                        "fault",
                        "injected_after_first_tick",
                    )
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
                        [None] * 6 + ["injected_after_first_tick"] * 6,
                    )
                    self.assertEqual(
                        [pose["p2p_map_sync"]["published_deltas"] for pose in poses],
                        [1] * 6 + [2] * 6,
                    )
                    self.assertEqual(
                        [pose["localization"]["scan_match"]["revision"] for pose in poses],
                        list(range(2, 14)),
                    )
                    for pose in poses[:6]:
                        self.assertAlmostEqual(pose["vx_mps"], 0.5)
                    for pose in poses[6:]:
                        self.assertEqual((pose["vx_mps"], pose["vy_mps"]), (0.0, 0.0))
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
