"""Go-to-goal protocol, controller, and mode-arbitration checks."""

import asyncio
import json
import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.server import (
    CommandMessageError,
    handler,
    handle_command_message,
    parse_goto_message,
    telemetry_messages,
)
from mockvehicle2d.vehicle import Vehicle


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class _GotoSocket:
    remote_address = ("test", 0)

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.messages: list[dict[str, object]] = []
        self.receive_count = 0

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))
        if len(self.messages) == 7:
            raise RuntimeError("stop after autonomous telemetry")

    async def recv(self) -> str:
        self.receive_count += 1
        if self.receive_count == 1:
            return '{"type":"goto","seq":21,"x_m":12,"y_m":10}'
        self.clock.now = 1 / 6
        raise asyncio.TimeoutError


class GotoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = MapGrid.from_wall_set(30, 30, set())

    def vehicle(self, **kwargs) -> Vehicle:
        return Vehicle(5.0, 5.0, command_timeout=0.25, now=0.0, **kwargs)

    def test_goto_parser_accepts_only_exact_finite_coordinates(self) -> None:
        self.assertEqual(
            parse_goto_message('{"type":"goto","seq":12,"x_m":8,"y_m":3.5}'),
            (8.0, 3.5, 12),
        )
        invalid = [
            '{"type":"goto","seq":true,"x_m":8,"y_m":3}',
            '{"type":"goto","seq":1,"x_m":true,"y_m":3}',
            '{"type":"goto","seq":1,"x_m":NaN,"y_m":3}',
            '{"type":"goto","seq":1,"x_m":8,"y_m":Infinity}',
            '{"type":"goto","seq":1,"x_m":8}',
            '{"type":"goto","seq":1,"x_m":8,"y_m":3,"extra":0}',
            '{"type":"goto","x_m":8,"y_m":3}',
            '{"type":"drive","seq":1,"x_m":8,"y_m":3}',
            '{"type":"goto","seq":1,"x_m":' + "9" * 4000 + ',"y_m":3}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(CommandMessageError):
                parse_goto_message(raw)

    def test_controller_turns_then_reaches_and_stays_stopped(self) -> None:
        vehicle = self.vehicle(yaw=math.pi)
        navigation = GotoController()
        navigation.start(7.0, 5.0)

        navigation.update(vehicle, self.grid, 0.0)
        self.assertEqual(navigation.status, "active")
        self.assertEqual(vehicle.velocities()[:2], (0.0, 0.0))
        self.assertLess(vehicle.velocities()[2], 0.0)

        for step in range(1, 301):
            navigation.update(vehicle, self.grid, step * 0.05)
            if navigation.status == "reached":
                break

        self.assertEqual(navigation.status, "reached")
        self.assertEqual(navigation.goal, (7.0, 5.0))
        self.assertEqual(navigation.reason, "goal_tolerance")
        self.assertLessEqual(math.hypot(vehicle.x - 7.0, vehicle.y - 5.0), navigation.goal_tolerance_m)
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))
        stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)
        navigation.update(vehicle, self.grid, 20.0)
        self.assertEqual((vehicle.x, vehicle.y, vehicle.yaw), stopped_pose)

    def test_controller_slows_near_goal_and_refreshes_watchdog(self) -> None:
        vehicle = self.vehicle()
        navigation = GotoController()
        navigation.start(5.2, 5.0)
        navigation.update(vehicle, self.grid, 0.0)
        self.assertGreater(vehicle.velocities()[0], 0.0)
        self.assertLess(vehicle.velocities()[0], vehicle.linear_speed)

        navigation.start(20.0, 5.0)
        for step in range(1, 11):
            navigation.update(vehicle, self.grid, step * 0.1)
        self.assertEqual(navigation.status, "active")
        self.assertEqual(vehicle.command, "drive")
        self.assertGreater(vehicle.x, 5.4)

    def test_collision_blocks_goal_without_restart(self) -> None:
        grid = MapGrid.from_wall_set(30, 30, {(7, y) for y in range(30)})
        vehicle = self.vehicle()
        navigation = GotoController()
        navigation.start(10.0, 5.0)

        for step in range(40):
            navigation.update(vehicle, grid, step * 0.1)
            if navigation.status == "blocked":
                break

        self.assertEqual((navigation.status, navigation.reason), ("blocked", "collision"))
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))
        stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)
        navigation.update(vehicle, grid, 5.0)
        self.assertEqual((vehicle.x, vehicle.y, vehicle.yaw), stopped_pose)

    def test_manual_or_invalid_input_cancels_active_goal_without_resume(self) -> None:
        vehicle = self.vehicle()
        navigation = GotoController()
        navigation.start(10.0, 5.0)
        navigation.update(vehicle, self.grid, 0.0)

        ack = handle_command_message(
            '{"type":"drive","seq":2,"linear_mps":0,"angular_rps":0}',
            vehicle,
            self.grid,
            0.1,
            10.0,
            navigation,
        )
        self.assertEqual(ack["type"], "cmd_ack")
        self.assertEqual((navigation.status, navigation.reason), ("cancelled", "manual_override"))
        navigation.update(vehicle, self.grid, 0.2)
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

        navigation.start(10.0, 5.0)
        navigation.update(vehicle, self.grid, 0.3)
        error = handle_command_message("not-json", vehicle, self.grid, 0.4, 10.1, navigation)
        self.assertEqual(error["type"], "error")
        self.assertEqual((navigation.status, navigation.reason), ("cancelled", "invalid_command"))
        navigation.update(vehicle, self.grid, 0.5)
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

    def test_goto_ack_replacement_and_pose_status_are_observable(self) -> None:
        vehicle = self.vehicle()
        navigation = GotoController()
        first = handle_command_message(
            json.dumps({"type": "goto", "seq": 3, "x_m": 8, "y_m": 6}),
            vehicle,
            self.grid,
            0.0,
            11.0,
            navigation,
        )
        self.assertEqual(
            first,
            {
                "type": "goto_ack",
                "ts": 11.0,
                "seq": 3,
                "goal": {"x_m": 8.0, "y_m": 6.0},
                "accepted": True,
            },
        )
        handle_command_message(
            '{"type":"goto","seq":4,"x_m":9,"y_m":5}',
            vehicle,
            self.grid,
            0.1,
            11.1,
            navigation,
        )
        self.assertEqual(navigation.goal, (9.0, 5.0))

        navigation.update(vehicle, self.grid, 0.1)
        pose, _scan = telemetry_messages(vehicle, self.grid, 1, 11.2, navigation)
        self.assertEqual(pose["control_mode"], "autonomous")
        self.assertEqual(
            pose["navigation"],
            {"status": "active", "goal": {"x_m": 9.0, "y_m": 5.0}, "reason": None},
        )

        legacy = handle_command_message('{"cmd":"stop"}', vehicle, self.grid, 0.2, 11.3, navigation)
        self.assertEqual(
            legacy,
            {"type": "cmd_ack", "ts": 11.3, "seq": None, "cmd": "stop", "accepted": True},
        )
        pose, _scan = telemetry_messages(vehicle, self.grid, 2, 11.4, navigation)
        self.assertEqual(pose["control_mode"], "manual")
        self.assertEqual(pose["navigation"]["status"], "cancelled")

    def test_goto_handoff_collision_blocks_replacement_goal(self) -> None:
        grid = MapGrid.from_wall_set(30, 30, {(7, y) for y in range(30)})
        vehicle = Vehicle(6.4, 5.0, command_timeout=0.25, now=0.0)
        navigation = GotoController()
        handle_command_message(
            '{"type":"goto","seq":5,"x_m":10,"y_m":5}',
            vehicle,
            grid,
            0.0,
            12.0,
            navigation,
        )
        navigation.update(vehicle, grid, 0.0)

        ack = handle_command_message(
            '{"type":"goto","seq":6,"x_m":4,"y_m":5}',
            vehicle,
            grid,
            0.25,
            12.25,
            navigation,
        )

        self.assertEqual(ack["type"], "goto_ack")
        self.assertEqual(navigation.goal, (4.0, 5.0))
        self.assertEqual((navigation.status, navigation.reason), ("blocked", "collision"))
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))
        stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)
        navigation.update(vehicle, grid, 0.3)
        self.assertEqual((vehicle.x, vehicle.y, vehicle.yaw), stopped_pose)
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

    def test_websocket_goto_ack_precedes_autonomous_pose(self) -> None:
        clock = _Clock()
        websocket = _GotoSocket(clock)
        asyncio.run(
            handler(
                websocket,
                _monotonic=clock.monotonic,
                _wall_time=lambda: 12.0,
            )
        )

        self.assertEqual(
            [message["type"] for message in websocket.messages],
            ["hello", "map_full", "pose", "scan", "goto_ack", "pose", "scan"],
        )
        self.assertEqual(websocket.messages[4]["seq"], 21)
        self.assertEqual(websocket.messages[5]["control_mode"], "autonomous")
        self.assertEqual(websocket.messages[5]["navigation"]["status"], "active")


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GotoTest)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
