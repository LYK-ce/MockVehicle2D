"""Anchored odometry and vehicle-owned observed-map checks."""

import asyncio
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.local_state import (
    FREE,
    FORBIDDEN,
    OCCUPIED,
    UNKNOWN,
    AnchorSpec,
    AnchoredLocalState,
    AnchoredOdometry,
    OdometryConfig,
    ObservedGrid,
    PoseEstimate,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.scan import LaserPoint, ScanConfig, scan_grid
from mockvehicle2d.server import (
    VehicleRuntime,
    handle_command_message,
    handler,
    telemetry_messages,
)
from mockvehicle2d.vehicle import Vehicle


class AnchorTransformTest(unittest.TestCase):
    def test_global_anchor_round_trip_includes_heading(self) -> None:
        anchor = AnchorSpec("vehicle-1", 100.0, 50.0, math.pi / 2, 0.2, 0.05)

        global_pose = anchor.anchor_to_global(2.0, -1.0, 0.25)
        self.assertAlmostEqual(global_pose[0], 101.0)
        self.assertAlmostEqual(global_pose[1], 52.0)
        local_pose = anchor.global_to_anchor(*global_pose)
        for actual, expected in zip(local_pose, (2.0, -1.0, 0.25), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_odometry_is_relative_to_birth_not_world_origin(self) -> None:
        anchor = AnchorSpec("vehicle-1", 100.0, 50.0, 0.0)
        near = AnchoredOdometry(anchor, 10.0, 20.0, 0.3, timestamp=0.0)
        far = AnchoredOdometry(anchor, 1010.0, -480.0, 0.3, timestamp=0.0)

        near_pose = near.update(11.0, 22.0, 0.5, timestamp=1.0)
        far_pose = far.update(1011.0, -478.0, 0.5, timestamp=1.0)
        self.assertEqual(
            (near_pose.x_m, near_pose.y_m, near_pose.yaw_rad),
            (far_pose.x_m, far_pose.y_m, far_pose.yaw_rad),
        )

    def test_zero_noise_and_fixed_seed_noise(self) -> None:
        anchor = AnchorSpec("vehicle-1", 10.0, 10.0, 0.0)
        exact = AnchoredOdometry(anchor, 10.0, 10.0, 0.0, timestamp=0.0)
        pose = exact.update(11.5, 9.5, 0.2, timestamp=1.0)
        self.assertAlmostEqual(pose.x_m, 1.5)
        self.assertAlmostEqual(pose.y_m, -0.5)
        self.assertAlmostEqual(pose.yaw_rad, 0.2)

        config = OdometryConfig(translation_noise_stddev_m=0.1, yaw_noise_stddev_rad=0.02, seed=7)
        first = AnchoredOdometry(anchor, 10.0, 10.0, 0.0, config=config, timestamp=0.0)
        second = AnchoredOdometry(anchor, 10.0, 10.0, 0.0, config=config, timestamp=0.0)
        first_pose = first.update(11.5, 9.5, 0.2, timestamp=1.0)
        second_pose = second.update(11.5, 9.5, 0.2, timestamp=1.0)
        self.assertEqual(first_pose, second_pose)
        self.assertNotEqual(
            (first_pose.x_m, first_pose.y_m, first_pose.yaw_rad),
            (pose.x_m, pose.y_m, pose.yaw_rad),
        )


class ObservedGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = AnchorSpec("vehicle-1", 10.0, 10.0, 0.0)
        self.pose = PoseEstimate("vehicle-1", 0.5, 0.5, 0.0, (0.0, 0.0, 0.0), "nominal", 1.0, 3)
        self.config = ScanConfig(
            min_angle=0.0,
            max_angle=0.0,
            angle_increment=1.0,
            scan_time=1.0,
            min_range=0.02,
            max_range=4.0,
            range_sample_rate_hz=1,
            scan_rate_hz=1,
        )

    def test_hit_marks_ray_free_endpoint_occupied_and_keeps_occlusion_unknown(self) -> None:
        grid = ObservedGrid(self.anchor, resolution_m=1.0)

        delta = grid.integrate_scan([LaserPoint(0.0, 2.5, 1.0)], self.pose, 2.0, self.config)

        self.assertEqual(grid.get_cell(0, 0), FREE)
        self.assertEqual(grid.get_cell(1, 0), FREE)
        self.assertEqual(grid.get_cell(2, 0), FREE)
        self.assertEqual(grid.get_cell(3, 0), OCCUPIED)
        self.assertEqual(grid.get_cell(4, 0), UNKNOWN)
        self.assertEqual(delta.revision, grid.revision)
        self.assertEqual(delta.pose_revision, self.pose.revision)
        self.assertEqual(delta.anchor_id, self.anchor.anchor_id)
        self.assertEqual({cell.state for cell in delta.changed_cells}, {FREE, OCCUPIED})

    def test_quantized_hit_is_assigned_to_the_wall_cell_not_the_nearer_cell(self) -> None:
        truth = MapGrid.from_wall_set(8, 4, {(4, 1)})
        pose = PoseEstimate(
            "vehicle-1",
            1.496,
            1.5,
            0.0,
            (0.0, 0.0, 0.0),
            "nominal",
            1.0,
            1,
        )
        point = scan_grid(truth, pose.x_m, pose.y_m, pose.yaw_rad, self.config)[0]
        observed = ObservedGrid(self.anchor)

        observed.integrate_scan((point,), pose, 2.0, self.config)

        self.assertAlmostEqual(point.range, 2.504)
        self.assertEqual(observed.get_cell(3, 1), FREE)
        self.assertEqual(observed.get_cell(4, 1), OCCUPIED)

    def test_hit_on_grid_boundary_uses_cell_entered_by_ray(self) -> None:
        cases = (
            ("negative x", (1.5, 0.5, 0.0), math.pi, 0.5, (0, 0), (1, 0)),
            ("positive x", (0.5, 0.5, 0.0), 0.0, 0.5, (1, 0), (0, 0)),
            ("negative y", (0.5, 1.5, 0.0), -math.pi / 2, 0.5, (0, 0), (0, 1)),
            ("positive y", (0.5, 0.5, 0.0), math.pi / 2, 0.5, (0, 1), (0, 0)),
            ("rotated negative x", (1.5, 0.5, math.pi / 4), 3 * math.pi / 4, 0.5, (0, 0), (1, 0)),
            ("inside negative x", (1.5001, 0.5, 0.0), math.pi, 0.5002, (0, 0), (1, 0)),
            ("inside positive x", (0.4999, 0.5, 0.0), 0.0, 0.5002, (1, 0), (0, 0)),
        )
        for name, (x_m, y_m, yaw_rad), angle, distance, hit_cell, start_cell in cases:
            with self.subTest(name=name):
                grid = ObservedGrid(self.anchor, resolution_m=1.0)
                pose = PoseEstimate(
                    "vehicle-1",
                    x_m,
                    y_m,
                    yaw_rad,
                    (0.0, 0.0, 0.0),
                    "nominal",
                    1.0,
                    3,
                )

                grid.integrate_scan(
                    [LaserPoint(angle, distance, 1.0)],
                    pose,
                    2.0,
                    self.config,
                )

                self.assertEqual(grid.get_cell(*hit_cell), OCCUPIED)
                self.assertEqual(grid.get_cell(*start_cell), FREE)

    def test_no_return_marks_to_max_range_free(self) -> None:
        grid = ObservedGrid(self.anchor, resolution_m=1.0)

        grid.integrate_scan([LaserPoint(0.0, 0.0, 0.0)], self.pose, 2.0, self.config)

        for x in range(5):
            self.assertEqual(grid.get_cell(x, 0), FREE)
        self.assertEqual(grid.get_cell(5, 0), UNKNOWN)

    def test_hit_wins_over_free_ray_within_one_scan(self) -> None:
        grid = ObservedGrid(self.anchor, resolution_m=1.0)

        grid.integrate_scan(
            [LaserPoint(0.0, 2.5, 0.0), LaserPoint(0.0, 0.0, 0.0)],
            self.pose,
            2.0,
            self.config,
        )

        self.assertEqual(grid.get_cell(3, 0), OCCUPIED)

    def test_downward_edge_evidence_wins_over_horizontal_free_space(self) -> None:
        grid = ObservedGrid(self.anchor, resolution_m=1.0)

        grid.integrate_scan(
            [LaserPoint(0.0, 0.0, 0.0)],
            self.pose,
            2.0,
            self.config,
            forbidden_points_vehicle_m=((2.5, 0.0),),
        )
        grid.integrate_scan(
            [LaserPoint(0.0, 0.0, 0.0)],
            self.pose,
            3.0,
            self.config,
        )

        self.assertEqual(grid.get_cell(2, 0), FORBIDDEN)
        self.assertEqual(grid.get_cell(3, 0), FORBIDDEN)

    def test_non_boundary_edge_evidence_marks_one_cell_and_repeats_stably(self) -> None:
        grid = ObservedGrid(self.anchor, resolution_m=1.0)
        first = grid.integrate_scan(
            [LaserPoint(0.0, 0.0, 0.0)],
            self.pose,
            2.0,
            self.config,
            forbidden_points_vehicle_m=((2.25, 0.25),),
        )
        second = grid.integrate_scan(
            [LaserPoint(0.0, 0.0, 0.0)],
            self.pose,
            3.0,
            self.config,
            forbidden_points_vehicle_m=((2.25, 0.25),),
        )

        forbidden = {
            (cell["gx"], cell["gy"])
            for cell in grid.snapshot()["cells"]
            if cell["state"] == FORBIDDEN
        }
        self.assertEqual(forbidden, {(2, 0)})
        self.assertTrue(first.changed_cells)
        self.assertEqual(second.changed_cells, ())

    def test_revision_changes_only_when_cells_change_and_snapshot_is_stable(self) -> None:
        grid = ObservedGrid(self.anchor, resolution_m=1.0)
        first = grid.integrate_scan([LaserPoint(0.0, 2.5, 1.0)], self.pose, 2.0, self.config)
        second = grid.integrate_scan([LaserPoint(0.0, 2.5, 1.0)], self.pose, 3.0, self.config)

        self.assertEqual(first.revision, 1)
        self.assertEqual(second.revision, 1)
        self.assertEqual(second.changed_cells, ())
        snapshot = grid.snapshot()
        self.assertEqual(snapshot["anchor_id"], "vehicle-1")
        self.assertEqual(snapshot["revision"], 1)
        self.assertEqual(snapshot["cells"][0], {"gx": 0, "gy": 0, "state": FREE})

    def test_lost_localization_does_not_write_map(self) -> None:
        state = AnchoredLocalState(
            self.anchor,
            truth_x_m=10.0,
            truth_y_m=10.0,
            truth_yaw_rad=0.0,
            timestamp=0.0,
        )
        state.set_localization_quality("lost", timestamp=1.0)

        delta = state.integrate_scan([LaserPoint(0.0, 2.5, 1.0)], 2.0, self.config)

        self.assertIsNone(delta)
        self.assertEqual(state.local_map.revision, 0)
        self.assertEqual(state.local_map.snapshot()["cells"], [])


class _StopAfterScanSocket:
    remote_address = ("test", 0)

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            return
        self.messages.append(json.loads(payload))
        if len(self.messages) == 3:
            raise RuntimeError("stop after scan")


class _HeldControllerSocket:
    remote_address = ("owner", 0)

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.ready = asyncio.Event()
        self.disconnect = asyncio.Event()

    async def send(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            return
        self.messages.append(json.loads(payload))
        if len(self.messages) == 3:
            self.ready.set()

    async def recv(self) -> str:
        await self.disconnect.wait()
        raise RuntimeError("owner disconnected")


class _BusyControllerSocket:
    remote_address = ("busy", 0)

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            return
        self.messages.append(json.loads(payload))
        if len(self.messages) == 3:
            raise RuntimeError("stop unguarded second handler")


class RuntimeIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = MapGrid.from_wall_set(30, 30, set())
        self.anchor = AnchorSpec("vehicle-1", 10.0, 10.0, 0.0)
        self.vehicle = Vehicle(10.0, 10.0, now=0.0)
        self.state = AnchoredLocalState(
            self.anchor,
            truth_x_m=10.0,
            truth_y_m=10.0,
            truth_yaw_rad=0.0,
            timestamp=0.0,
        )

    def test_degraded_limits_automatic_speed_and_lost_blocks_it(self) -> None:
        runtime = VehicleRuntime.create(
            started_at=0.0,
            anchor=self.anchor,
            odometry_config=OdometryConfig(),
        )
        runtime.navigation.start(
            1.0,
            0.0,
            local_map=runtime.local_state.local_map,
            pose=runtime.local_state.pose,
            vehicle_radius_m=runtime.vehicle.radius,
        )
        runtime.local_state.set_localization_quality("degraded", timestamp=0.0)

        runtime.update(0.0, 1.0)

        self.assertEqual(
            (runtime.navigation.status, runtime.navigation.reason),
            ("active", None),
        )
        self.assertAlmostEqual(
            runtime.vehicle.body_velocities()[0], runtime.vehicle.linear_speed / 2
        )

        x_before_loss = runtime.vehicle.x
        runtime.local_state.set_localization_quality("lost", timestamp=1.1)
        runtime.update(0.1, 1.1)
        self.assertEqual(runtime.vehicle.x, x_before_loss)
        self.assertEqual(
            (runtime.navigation.status, runtime.navigation.reason),
            ("blocked", "localization_lost"),
        )
        self.assertEqual(runtime.vehicle.body_velocities(), (0.0, 0.0))

    def test_default_noiseless_scan_never_marks_a_truth_free_hit_occupied(self) -> None:
        runtime = VehicleRuntime.create(
            started_at=0.0,
            timestamp=0.0,
            anchor=self.anchor,
            odometry_config=OdometryConfig(),
        )

        runtime.update(1 / 6, 1 / 6)

        occupied = {
            (cell["gx"], cell["gy"])
            for cell in runtime.local_state.local_map.snapshot()["cells"]
            if cell["state"] == OCCUPIED
        }
        self.assertTrue(occupied)
        false_hits = {
            (gx, gy)
            for gx, gy in occupied
            if not runtime.grid.is_wall(
                gx + int(self.anchor.global_x_m),
                gy + int(self.anchor.global_y_m),
            )
        }
        self.assertFalse(false_hits, false_hits)

    def test_pose_uses_anchor_estimate_and_never_labels_truth(self) -> None:
        self.state.update_from_truth(11.0, 10.0, 0.0, timestamp=1.0)

        pose, _scan = telemetry_messages(
            self.vehicle,
            self.grid,
            1,
            123.0,
            local_state=self.state,
        )

        self.assertEqual((pose["x"], pose["y"], pose["yaw"]), (11.0, 10.0, 0.0))
        self.assertEqual(pose["source"], "anchored_odometry")
        self.assertNotIn("truth", json.dumps(pose).lower())
        self.assertEqual(pose["localization"]["anchor_id"], "vehicle-1")
        self.assertGreater(self.state.local_map.revision, 0)

    def test_goto_converts_global_goal_and_rejects_lost_localization(self) -> None:
        anchor = AnchorSpec("vehicle-1", 100.0, 50.0, math.pi / 2)
        state = AnchoredLocalState(
            anchor,
            truth_x_m=10.0,
            truth_y_m=10.0,
            truth_yaw_rad=0.0,
            timestamp=0.0,
        )
        navigation = GotoController()
        accepted = handle_command_message(
            '{"type":"goto","seq":1,"x_m":100,"y_m":52}',
            self.vehicle,
            self.grid,
            0.0,
            10.0,
            navigation,
            local_state=state,
        )
        self.assertEqual(accepted["accepted"], True)
        self.assertAlmostEqual(navigation.goal[0], 2.0)
        self.assertAlmostEqual(navigation.goal[1], 0.0)
        self.assertEqual(
            navigation.snapshot()["goal"],
            {"x_m": 100.0, "y_m": 52.0},
        )

        state.set_localization_quality("lost", timestamp=11.0)
        rejected = handle_command_message(
            '{"type":"goto","seq":2,"x_m":101,"y_m":52}',
            self.vehicle,
            self.grid,
            0.0,
            11.0,
            navigation,
            local_state=state,
        )
        self.assertEqual(
            rejected,
            {
                "type": "goto_ack",
                "ts": 11.0,
                "seq": 2,
                "goal": {"x_m": 101.0, "y_m": 52.0},
                "accepted": False,
                "reason": "localization_lost",
            },
        )

    def test_lost_command_handoff_does_not_advance_old_automatic_motion(self) -> None:
        cases = (
            ('{"type":"cmd","seq":1,"cmd":"stop"}', "cmd_ack"),
            ('{"type":"cmd","seq":2,"cmd":"forward"}', "cmd_ack"),
            ('{"type":"goto","seq":3,"x_m":15,"y_m":10}', "goto_ack"),
            ('{"type":"cmd","seq":4,"cmd":"invalid"}', "error"),
        )
        for raw, reply_type in cases:
            with self.subTest(reply_type=reply_type, raw=raw):
                vehicle = Vehicle(10.0, 10.0, now=0.0)
                navigation = GotoController()
                state = AnchoredLocalState(
                    self.anchor,
                    truth_x_m=10.0,
                    truth_y_m=10.0,
                    truth_yaw_rad=0.0,
                    timestamp=0.0,
                )
                navigation.start(
                    5.0,
                    0.0,
                    local_map=state.local_map,
                    pose=state.pose,
                    vehicle_radius_m=vehicle.radius,
                )
                vehicle.install_drive(0.5, 0.0, 0.0)
                state.set_localization_quality("lost", timestamp=0.1)

                reply = handle_command_message(
                    raw,
                    vehicle,
                    self.grid,
                    0.5,
                    0.5,
                    navigation,
                    local_state=state,
                )

                self.assertEqual(vehicle.x, 10.0)
                self.assertEqual(reply["type"], reply_type)
                if reply_type == "goto_ack":
                    self.assertEqual(
                        (reply["accepted"], reply["reason"]),
                        (False, "localization_lost"),
                    )
                elif '"forward"' in raw:
                    self.assertEqual(
                        (reply["accepted"], vehicle.command),
                        (True, "forward"),
                    )

    def test_only_one_controller_can_mutate_shared_runtime(self) -> None:
        async def scenario() -> None:
            runtime = VehicleRuntime.create(
                started_at=0.0,
                anchor=self.anchor,
                odometry_config=OdometryConfig(),
            )
            owner = _HeldControllerSocket()
            owner_task = asyncio.create_task(
                handler(
                    owner,
                    _runtime=runtime,
                    _monotonic=lambda: 0.0,
                    _wall_time=lambda: 10.0,
                )
            )
            await asyncio.wait_for(owner.ready.wait(), timeout=1.0)
            runtime.navigation.start(
                5.0,
                0.0,
                local_map=runtime.local_state.local_map,
                pose=runtime.local_state.pose,
                vehicle_radius_m=runtime.vehicle.radius,
            )
            runtime.vehicle.install_drive(0.5, 0.0, 0.0)
            before = (
                runtime.vehicle.x,
                runtime.vehicle.command,
                runtime.navigation.status,
                runtime.local_state.pose.revision,
                runtime.local_state.local_map.revision,
                runtime.frame_sequence,
            )

            busy = _BusyControllerSocket()
            await handler(
                busy,
                _runtime=runtime,
                _monotonic=lambda: 0.5,
                _wall_time=lambda: 20.0,
            )

            self.assertEqual(
                busy.messages,
                [
                    {
                        "type": "error",
                        "ts": 20.0,
                        "seq": None,
                        "code": "vehicle_busy",
                        "message": "another controller is active",
                    }
                ],
            )
            self.assertEqual(
                (
                    runtime.vehicle.x,
                    runtime.vehicle.command,
                    runtime.navigation.status,
                    runtime.local_state.pose.revision,
                    runtime.local_state.local_map.revision,
                    runtime.frame_sequence,
                ),
                before,
            )

            owner.disconnect.set()
            await asyncio.wait_for(owner_task, timeout=1.0)
            self.assertEqual(runtime.vehicle.command, "stop")
            self.assertEqual(
                (runtime.navigation.status, runtime.navigation.reason),
                ("cancelled", "disconnected"),
            )

            pose_revision = runtime.local_state.pose.revision
            map_revision = runtime.local_state.local_map.revision
            replacement = _StopAfterScanSocket()
            await handler(
                replacement,
                _runtime=runtime,
                _monotonic=lambda: 1.0,
                _wall_time=lambda: 30.0,
            )
            self.assertGreater(runtime.local_state.pose.revision, pose_revision)
            self.assertGreaterEqual(runtime.local_state.local_map.revision, map_revision)
            self.assertEqual(
                replacement.messages[1]["localization"]["revision"],
                runtime.local_state.pose.revision,
            )

        asyncio.run(scenario())

    def test_runtime_keeps_pose_and_local_map_across_reconnect(self) -> None:
        runtime = VehicleRuntime.create(
            started_at=0.0,
            anchor=self.anchor,
            odometry_config=OdometryConfig(),
        )
        first = _StopAfterScanSocket()
        asyncio.run(handler(first, _runtime=runtime, _monotonic=lambda: 0.0, _wall_time=lambda: 10.0))
        first_pose_revision = runtime.local_state.pose.revision
        first_map_revision = runtime.local_state.local_map.revision

        second = _StopAfterScanSocket()
        asyncio.run(handler(second, _runtime=runtime, _monotonic=lambda: 1.0, _wall_time=lambda: 11.0))

        self.assertGreater(runtime.local_state.pose.revision, first_pose_revision)
        self.assertGreater(first_map_revision, 0)
        self.assertGreaterEqual(runtime.local_state.local_map.revision, first_map_revision)
        self.assertEqual(second.messages[1]["localization"]["revision"], runtime.local_state.pose.revision)

    def test_controller_lease_is_released_when_pipeline_initialization_fails(self) -> None:
        runtime = VehicleRuntime.create(
            started_at=0.0,
            anchor=self.anchor,
            odometry_config=OdometryConfig(),
        )

        with patch(
            "mockvehicle2d.server.SchemaValidator",
            side_effect=RuntimeError("schema unavailable"),
        ):
            asyncio.run(
                handler(
                    _StopAfterScanSocket(),
                    _runtime=runtime,
                    _monotonic=lambda: 0.0,
                    _wall_time=lambda: 10.0,
                )
            )
        self.assertFalse(runtime.controller_lease.locked())

        replacement = _StopAfterScanSocket()
        asyncio.run(
            handler(
                replacement,
                _runtime=runtime,
                _monotonic=lambda: 1.0,
                _wall_time=lambda: 11.0,
            )
        )
        self.assertFalse(runtime.controller_lease.locked())
        self.assertTrue(replacement.messages)

    def test_navigation_does_not_query_world_grid_cells(self) -> None:
        source = (REPO_ROOT / "src/mockvehicle2d/navigation.py").read_text(encoding="utf-8")
        for forbidden in (".get_cell(", ".is_wall(", ".is_passable(", ".is_void("):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("update_from_truth", source)


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
