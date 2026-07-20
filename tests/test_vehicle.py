"""Deterministic command, motion, watchdog, and collision checks."""

import json
import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.server import CommandMessageError, handle_command_message, parse_command_message
from mockvehicle2d.vehicle import Vehicle


class VehicleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = MapGrid.from_wall_set(20, 20, set())

    def vehicle(self, **kwargs) -> Vehicle:
        return Vehicle(5.0, 5.0, now=0.0, **kwargs)

    def test_forward_backward_and_actual_elapsed_time(self) -> None:
        forward = self.vehicle()
        forward.apply_command(self.grid, "forward", 0.0)
        forward.advance(self.grid, 0.4)
        self.assertAlmostEqual(forward.x, 5.2)
        self.assertAlmostEqual(forward.y, 5.0)

        backward = self.vehicle()
        backward.apply_command(self.grid, "backward", 0.0)
        backward.advance(self.grid, 0.4)
        self.assertAlmostEqual(backward.x, 4.8)

        forward.apply_command(self.grid, "backward", 0.6)
        self.assertAlmostEqual(forward.x, 5.3)
        forward.advance(self.grid, 0.8)
        self.assertAlmostEqual(forward.x, 5.2)

    def test_left_and_right_follow_screen_coordinate_signs(self) -> None:
        left = self.vehicle()
        left.apply_command(self.grid, "spin_left", 0.0)
        left.advance(self.grid, 0.5)
        self.assertAlmostEqual(left.yaw, -math.pi / 4)
        self.assertAlmostEqual(left.velocities()[2], -math.pi / 2)

        right = self.vehicle()
        right.apply_command(self.grid, "spin_right", 0.0)
        right.advance(self.grid, 0.5)
        self.assertAlmostEqual(right.yaw, math.pi / 4)
        self.assertAlmostEqual(right.velocities()[2], math.pi / 2)

    def test_watchdog_integrates_only_until_timeout(self) -> None:
        vehicle = self.vehicle(command_timeout=1.0)
        vehicle.apply_command(self.grid, "forward", 0.0)
        vehicle.advance(self.grid, 2.0)
        self.assertAlmostEqual(vehicle.x, 5.5)
        self.assertEqual(vehicle.command, "stop")
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

    def test_substeps_stop_at_last_safe_position(self) -> None:
        grid = MapGrid.from_wall_set(20, 20, {(4, y) for y in range(20)})
        vehicle = Vehicle(2.5, 5.5, linear_speed=10.0, command_timeout=2.0, now=0.0)
        vehicle.apply_command(grid, "forward", 0.0)
        vehicle.advance(grid, 1.0)
        self.assertLessEqual(vehicle.x, 3.5)
        self.assertGreater(vehicle.x, 2.5)
        self.assertTrue(vehicle.collision)
        self.assertEqual(vehicle.command, "stop")
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

    def test_canonical_and_legacy_commands_are_acknowledged(self) -> None:
        vehicle = self.vehicle()
        ack = handle_command_message(
            json.dumps({"type": "cmd", "seq": 7, "cmd": "forward"}), vehicle, self.grid, 0.0, 12.5
        )
        self.assertEqual(
            ack,
            {"type": "cmd_ack", "ts": 12.5, "seq": 7, "cmd": "forward", "accepted": True},
        )
        legacy = handle_command_message('{"cmd":"stop"}', vehicle, self.grid, 0.1, 12.6)
        self.assertEqual(legacy["type"], "cmd_ack")
        self.assertIsNone(legacy["seq"])

    def test_invalid_command_stops_and_returns_safe_sequence(self) -> None:
        vehicle = self.vehicle()
        vehicle.apply_command(self.grid, "forward", 0.0)
        error = handle_command_message(
            '{"type":"cmd","seq":9,"cmd":"fly"}', vehicle, self.grid, 0.25, 13.0
        )
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["seq"], 9)
        self.assertEqual(error["code"], "invalid_cmd")
        self.assertAlmostEqual(vehicle.x, 5.125)
        self.assertEqual(vehicle.command, "stop")

    def test_parser_strictly_rejects_invalid_shapes_and_fields(self) -> None:
        invalid = [
            b'{"cmd":"forward"}',
            "not-json",
            "[]",
            '{"type":"pose","seq":1,"cmd":"forward"}',
            '{"type":"cmd","seq":-1,"cmd":"forward"}',
            '{"type":"cmd","seq":true,"cmd":"forward"}',
            '{"type":"cmd","cmd":"forward"}',
            '{"cmd":"forward","extra":1}',
            '{"type":"cmd","seq":1,"cmd":"forward","extra":1}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(CommandMessageError):
                parse_command_message(raw)


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VehicleTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
