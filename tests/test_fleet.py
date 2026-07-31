"""Shared-world multi-vehicle simulation invariants."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from mockvehicle2d.collision import is_strict_overlap
from mockvehicle2d.controller import ManualAction, ManualCommand
from mockvehicle2d.fleet import (
    AnchorPose,
    FleetRuntime,
    FleetScenario,
    FleetVehicleSpec,
)
from mockvehicle2d.local_state import OdometryConfig
from mockvehicle2d.map_grid import MapGrid


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

        self.assertEqual(poses, starts)
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


if __name__ == "__main__":
    unittest.main()
