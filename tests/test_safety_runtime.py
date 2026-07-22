"""Safety runtime integration checks."""

import json
import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.map_grid import FREE, MapGrid, VOID
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.safety import LocalSafetyRuntime
from mockvehicle2d.server import generate_map, handle_command_message, telemetry_messages
from mockvehicle2d.vehicle import Vehicle


def wall_grid(wall_x: int) -> MapGrid:
    return MapGrid.from_wall_set(20, 20, {(wall_x, y) for y in range(20)})


class SafetyRuntimeTest(unittest.TestCase):
    def vehicle(self, x: float = 2.0) -> Vehicle:
        return Vehicle(x, 5.5, radius=0.5, command_timeout=10.0, now=0.0)

    def test_automatic_slow_zone_limits_drive_but_stays_active(self) -> None:
        vehicle = self.vehicle(1.8)
        navigation = GotoController()
        safety = LocalSafetyRuntime()
        navigation.start(10.0, 5.5)

        navigation.update(vehicle, wall_grid(3), 0.0, safety)

        self.assertEqual(navigation.status, "active")
        self.assertEqual(safety.snapshot()["state"], "limited")
        self.assertGreater(vehicle.body_velocities()[0], 0.0)
        self.assertLess(vehicle.body_velocities()[0], vehicle.linear_speed)

    def test_automatic_obstacle_edge_and_fault_block_without_restart(self) -> None:
        cases: list[tuple[MapGrid, Vehicle, LocalSafetyRuntime, str]] = []
        cases.append((wall_grid(3), self.vehicle(2.3), LocalSafetyRuntime(), "safety_obstacle"))

        edge_grid = MapGrid(20, 20)
        edge_grid.set_cell(3, 5, VOID)
        cases.append((edge_grid, self.vehicle(2.3), LocalSafetyRuntime(), "safety_edge"))
        cases.append((MapGrid(20, 20), self.vehicle(), LocalSafetyRuntime(healthy=False), "safety_sensor_fault"))

        for grid, vehicle, safety, reason in cases:
            with self.subTest(reason=reason):
                navigation = GotoController()
                navigation.start(10.0, 5.5)
                navigation.update(vehicle, grid, 0.0, safety)
                self.assertEqual((navigation.status, navigation.reason), ("blocked", reason))
                self.assertEqual(vehicle.body_velocities(), (0.0, 0.0))
                stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)

                if grid.in_bounds(3, 5):
                    grid.set_cell(3, 5, FREE)
                safety.healthy = True
                navigation.update(vehicle, grid, 1.0, safety)
                self.assertEqual((vehicle.x, vehicle.y, vehicle.yaw), stopped_pose)
                self.assertEqual(navigation.status, "blocked")

    def test_manual_hard_stop_is_immediate_and_reverse_clears_latch(self) -> None:
        grid = wall_grid(3)
        vehicle = self.vehicle(2.3)
        safety = LocalSafetyRuntime()

        ack = handle_command_message(
            '{"type":"drive","seq":1,"linear_mps":0.5,"angular_rps":0.2}',
            vehicle,
            grid,
            0.0,
            10.0,
            safety=safety,
        )
        self.assertEqual(
            ack,
            {"type": "cmd_ack", "ts": 10.0, "seq": 1, "cmd": "drive", "accepted": True},
        )
        self.assertEqual(vehicle.body_velocities(), (0.0, 0.0))
        self.assertEqual((safety.snapshot()["state"], safety.snapshot()["reason"]), ("stopped", "safety_obstacle"))

        handle_command_message(
            '{"type":"drive","seq":2,"linear_mps":-0.5,"angular_rps":0}',
            vehicle,
            grid,
            0.1,
            10.1,
            safety=safety,
        )
        self.assertEqual(vehicle.body_velocities(), (-0.5, 0.0))
        self.assertEqual((safety.snapshot()["state"], safety.snapshot()["reason"]), ("clear", None))

        faulted_vehicle = self.vehicle()
        faulted_safety = LocalSafetyRuntime(healthy=False)
        handle_command_message(
            '{"type":"drive","seq":6,"linear_mps":0.5,"angular_rps":0.2}',
            faulted_vehicle,
            MapGrid(20, 20),
            0.0,
            10.2,
            safety=faulted_safety,
        )
        self.assertEqual(faulted_vehicle.body_velocities(), (0.0, 0.0))
        self.assertEqual(
            (faulted_safety.snapshot()["state"], faulted_safety.snapshot()["reason"]),
            ("fault", "safety_sensor_fault"),
        )

    def test_manual_slow_zone_is_not_throttled_and_pure_rotation_is_allowed(self) -> None:
        grid = wall_grid(3)
        vehicle = self.vehicle(1.8)
        safety = LocalSafetyRuntime()

        handle_command_message(
            '{"type":"drive","seq":3,"linear_mps":0.5,"angular_rps":0}',
            vehicle,
            grid,
            0.0,
            11.0,
            safety=safety,
        )
        self.assertEqual(vehicle.body_velocities(), (0.5, 0.0))
        self.assertEqual(safety.snapshot()["state"], "clear")

        close_edge = MapGrid(20, 20)
        close_edge.set_cell(3, 5, VOID)
        vehicle = self.vehicle(2.3)
        safety = LocalSafetyRuntime()
        handle_command_message(
            json.dumps({"type": "cmd", "seq": 4, "cmd": "spin_left"}),
            vehicle,
            close_edge,
            0.0,
            11.1,
            safety=safety,
        )
        self.assertEqual(vehicle.body_velocities(), (0.0, -vehicle.angular_speed))
        self.assertEqual(safety.snapshot()["state"], "clear")

    def test_manual_periodic_recheck_stops_and_does_not_resume(self) -> None:
        grid = wall_grid(4)
        vehicle = self.vehicle()
        safety = LocalSafetyRuntime()
        handle_command_message(
            '{"type":"drive","seq":5,"linear_mps":0.5,"angular_rps":0}',
            vehicle,
            grid,
            0.0,
            12.0,
            safety=safety,
        )

        vehicle.advance(grid, 2.5)
        safety.enforce_manual(vehicle, grid)
        self.assertEqual(vehicle.body_velocities(), (0.0, 0.0))
        self.assertEqual(safety.snapshot()["state"], "stopped")
        stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)

        vehicle.advance(grid, 3.0)
        safety.enforce_manual(vehicle, grid)
        self.assertEqual((vehicle.x, vehicle.y, vehicle.yaw), stopped_pose)
        self.assertEqual(safety.snapshot()["state"], "stopped")

    def test_pose_snapshot_and_generated_void_patch_are_observable(self) -> None:
        safety = LocalSafetyRuntime()
        pose, _scan = telemetry_messages(
            self.vehicle(), MapGrid(20, 20), 1, 13.0, safety=safety
        )
        self.assertEqual(
            pose["safety"],
            {
                "state": "clear",
                "reason": None,
                "obstacle_clearance_m": None,
                "edge_clearance_m": None,
            },
        )

        voxels, grid = generate_map(size=32, seed=42)
        voids = {(voxel["gx"], voxel["gy"]) for voxel in voxels if voxel["state"] == VOID}
        self.assertTrue(voids)
        self.assertTrue(all(grid.is_void(x, y) for x, y in voids))
        self.assertFalse(any(8 <= x <= 12 and 8 <= y <= 12 for x, y in voids))


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SafetyRuntimeTest)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
