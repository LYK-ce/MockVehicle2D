"""Deterministic command, motion, watchdog, and collision checks."""

import json
import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.collision import is_circle_passable, is_swept_circle_passable
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.server import CommandMessageError, handle_command_message, parse_command_message, parse_drive_message
from mockvehicle2d.vehicle import Vehicle, command_from_axes


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

    def test_combined_commands_drive_arcs_and_can_switch_to_straight(self) -> None:
        vehicle = self.vehicle()
        vehicle.apply_command(self.grid, "forward_right", 0.0)
        vehicle.advance(self.grid, 0.5)

        self.assertGreater(vehicle.x, 5.0)
        self.assertGreater(vehicle.y, 5.0)
        self.assertAlmostEqual(vehicle.yaw, math.pi / 4)
        vx, vy, omega = vehicle.velocities()
        self.assertGreater(vx, 0.0)
        self.assertGreater(vy, 0.0)
        self.assertGreater(omega, 0.0)

        vehicle.apply_command(self.grid, "forward", 0.5)
        x, y = vehicle.x, vehicle.y
        vehicle.advance(self.grid, 0.7)
        self.assertGreater(vehicle.x, x)
        self.assertGreater(vehicle.y, y)
        self.assertEqual(vehicle.command, "forward")
        self.assertEqual(vehicle.velocities()[2], 0.0)

    def test_backward_combined_commands_follow_turn_signs(self) -> None:
        left = self.vehicle()
        left.apply_command(self.grid, "backward_left", 0.0)
        left.advance(self.grid, 0.5)
        self.assertLess(left.x, 5.0)
        self.assertGreater(left.y, 5.0)
        self.assertLess(left.yaw, 0.0)

        right = self.vehicle()
        right.apply_command(self.grid, "backward_right", 0.0)
        right.advance(self.grid, 0.5)
        self.assertLess(right.x, 5.0)
        self.assertLess(right.y, 5.0)
        self.assertGreater(right.yaw, 0.0)

    def test_opposite_input_axes_cancel(self) -> None:
        self.assertEqual(command_from_axes(True, True, False, False), "stop")
        self.assertEqual(command_from_axes(True, True, True, False), "spin_left")
        self.assertEqual(command_from_axes(True, False, True, True), "forward")

    def test_large_pure_rotation_is_normalized_without_translation(self) -> None:
        vehicle = self.vehicle(angular_speed=1_000_000.0, command_timeout=2.0)
        vehicle.apply_command(self.grid, "spin_right", 0.0)

        vehicle.advance(self.grid, 1.0)

        self.assertEqual((vehicle.x, vehicle.y), (5.0, 5.0))
        self.assertAlmostEqual(vehicle.yaw, math.atan2(math.sin(1_000_000.0), math.cos(1_000_000.0)))
        self.assertEqual(vehicle.command, "spin_right")

    def test_watchdog_integrates_only_until_timeout(self) -> None:
        vehicle = self.vehicle(command_timeout=1.0)
        vehicle.apply_command(self.grid, "forward", 0.0)
        vehicle.advance(self.grid, 2.0)
        self.assertAlmostEqual(vehicle.x, 5.5)
        self.assertEqual(vehicle.command, "stop")
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

    def test_continuous_drive_handles_straight_arcs_and_watchdog(self) -> None:
        straight = self.vehicle(command_timeout=1.0)
        straight.apply_drive(self.grid, 0.25, 0.0, 0.0)
        straight.advance(self.grid, 0.4)
        self.assertAlmostEqual(straight.x, 5.1)
        self.assertAlmostEqual(straight.y, 5.0)
        self.assertEqual(straight.command, "drive")
        straight.apply_drive(self.grid, 0.0, 0.0, 0.4)
        self.assertEqual((straight.command, straight.velocities()), ("stop", (0.0, 0.0, 0.0)))
        straight.apply_drive(self.grid, 0.25, 0.1, 0.4)
        straight.reset(4.0, 4.0, 0.0, 0.5)
        self.assertEqual((straight.command, straight.velocities()), ("stop", (0.0, 0.0, 0.0)))

        arc = self.vehicle(command_timeout=1.0)
        arc.apply_drive(self.grid, 0.4, 0.5, 0.0)
        arc.advance(self.grid, 0.5)
        self.assertGreater(arc.x, 5.0)
        self.assertGreater(arc.y, 5.0)
        self.assertAlmostEqual(arc.yaw, 0.25)
        arc.advance(self.grid, 2.0)
        self.assertEqual(arc.command, "stop")
        self.assertEqual(arc.velocities(), (0.0, 0.0, 0.0))

    def test_continuous_drive_collision_clears_velocity(self) -> None:
        grid = MapGrid.from_wall_set(20, 20, {(4, y) for y in range(20)})
        vehicle = Vehicle(2.5, 5.5, command_timeout=5.0, now=0.0)
        vehicle.apply_drive(grid, 0.5, 0.2, 0.0)
        vehicle.advance(grid, 4.0)
        self.assertTrue(vehicle.collision)
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

    def test_arc_collision_stops_at_last_safe_pose(self) -> None:
        grid = MapGrid.from_wall_set(20, 20, {(4, y) for y in range(20)})
        vehicle = Vehicle(2.5, 5.5, linear_speed=2.0, command_timeout=2.0, now=0.0)
        vehicle.apply_command(grid, "forward_right", 0.0)
        vehicle.advance(grid, 1.0)

        self.assertGreater(vehicle.x, 2.5)
        self.assertLessEqual(vehicle.x, 3.5)
        self.assertGreater(vehicle.y, 5.5)
        self.assertGreater(vehicle.yaw, 0.0)
        self.assertTrue(vehicle.collision)
        self.assertEqual(vehicle.command, "stop")

    def test_swept_circle_catches_wall_corner_between_safe_endpoints(self) -> None:
        grid = MapGrid.from_wall_set(10, 10, {(5, 5)})
        start = (4.561612, 4.738388)
        end = (start[0] + 0.25 / math.sqrt(2), start[1] - 0.25 / math.sqrt(2))
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        self.assertTrue(is_circle_passable(grid, *start, 0.5))
        self.assertTrue(is_circle_passable(grid, *end, 0.5))
        self.assertFalse(is_circle_passable(grid, *midpoint, 0.5))
        self.assertTrue(is_swept_circle_passable(grid, 4.5, 4.0, 4.5, 7.0, 0.5))
        vehicle = Vehicle(*start, yaw=-math.pi / 4, linear_speed=0.25, command_timeout=2.0, now=0.0)

        vehicle.apply_command(grid, "forward", 0.0)
        vehicle.advance(grid, 1.0)

        self.assertEqual((vehicle.x, vehicle.y), start)
        self.assertTrue(vehicle.collision)
        clear = Vehicle(2.0, 2.0, linear_speed=0.25, command_timeout=2.0, now=0.0)
        clear.apply_command(grid, "forward", 0.0)
        clear.advance(grid, 1.0)
        self.assertAlmostEqual(clear.x, 2.25)
        self.assertAlmostEqual(clear.y, 2.0)

    def test_repeated_command_does_not_clear_collision_without_motion(self) -> None:
        grid = MapGrid.from_wall_set(10, 10, {(4, y) for y in range(10)})
        vehicle = Vehicle(3.5, 5.5, now=0.0)

        vehicle.apply_command(grid, "forward", 0.0)
        vehicle.apply_command(grid, "forward", 0.5)

        self.assertTrue(vehicle.collision)
        vehicle.apply_command(grid, "backward", 0.5)
        vehicle.advance(grid, 0.75)
        self.assertFalse(vehicle.collision)

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

        self.assertEqual(
            parse_command_message('{"type":"cmd","seq":8,"cmd":"forward_right"}'),
            ("forward_right", 8),
        )
        self.assertEqual(parse_command_message('{"cmd":"backward_left"}'), ("backward_left", None))
        with self.assertRaises(CommandMessageError):
            parse_command_message('{"type":"cmd","seq":9,"cmd":"forward_up"}')

    def test_drive_message_is_bounded_and_acknowledged(self) -> None:
        self.assertEqual(
            parse_drive_message(
                '{"type":"drive","seq":10,"linear_mps":-0.5,"angular_rps":1.5}', 0.5, 1.5
            ),
            (-0.5, 1.5, 10),
        )
        vehicle = self.vehicle()
        ack = handle_command_message(
            '{"type":"drive","seq":11,"linear_mps":0.25,"angular_rps":-0.4}',
            vehicle,
            self.grid,
            0.0,
            13.0,
        )
        self.assertEqual(
            ack,
            {"type": "cmd_ack", "ts": 13.0, "seq": 11, "cmd": "drive", "accepted": True},
        )
        self.assertEqual(vehicle.command, "drive")
        self.assertEqual(vehicle.velocities(), (0.25, 0.0, -0.4))

    def test_drive_parser_rejects_bool_nonfinite_bounds_and_bad_fields(self) -> None:
        invalid = [
            '{"type":"drive","seq":1,"linear_mps":true,"angular_rps":0}',
            '{"type":"drive","seq":1,"linear_mps":NaN,"angular_rps":0}',
            '{"type":"drive","seq":1,"linear_mps":0,"angular_rps":Infinity}',
            '{"type":"drive","seq":1,"linear_mps":0.51,"angular_rps":0}',
            '{"type":"drive","seq":1,"linear_mps":0,"angular_rps":-1.51}',
            '{"type":"drive","seq":1,"linear_mps":' + "9" * 4000 + ',"angular_rps":0}',
            '{"type":"drive","seq":true,"linear_mps":0,"angular_rps":0}',
            '{"type":"drive","seq":1,"linear_mps":0}',
            '{"type":"drive","seq":1,"linear_mps":0,"angular_rps":0,"extra":1}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(CommandMessageError):
                parse_drive_message(raw, 0.5, 1.5)

        vehicle = self.vehicle()
        vehicle.apply_drive(self.grid, 0.5, 0.0, 0.0)
        error = handle_command_message(invalid[3], vehicle, self.grid, 0.2, 13.0)
        self.assertEqual((error["type"], error["seq"], error["code"]), ("error", 1, "drive_out_of_range"))
        self.assertAlmostEqual(vehicle.x, 5.1)
        self.assertEqual(vehicle.command, "stop")
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

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

    def test_oversized_json_integer_fails_safe(self) -> None:
        vehicle = self.vehicle()
        vehicle.apply_command(self.grid, "forward", 0.0)
        raw = '{"type":"cmd","seq":' + "1" * 5000 + ',"cmd":"forward"}'

        error = handle_command_message(raw, vehicle, self.grid, 0.25, 13.0)

        self.assertEqual(error["type"], "error")
        self.assertEqual(error["code"], "invalid_json")
        self.assertEqual(vehicle.command, "stop")


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VehicleTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
