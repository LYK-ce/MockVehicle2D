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
from mockvehicle2d.local_state import AnchorSpec, AnchoredLocalState, ObservedGrid, PoseEstimate
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

    async def send(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            return  # skip binary frames (map_full chunks)
        self.messages.append(json.loads(payload))
        if len(self.messages) == 6:
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

    @staticmethod
    def observed() -> ObservedGrid:
        return ObservedGrid(AnchorSpec("goto-test", 0.0, 0.0, 0.0))

    @staticmethod
    def pose(vehicle: Vehicle, timestamp: float = 0.0) -> PoseEstimate:
        return PoseEstimate(
            "goto-test",
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            (0.0, 0.0, 0.0),
            "nominal",
            timestamp,
            0,
        )

    def start(
        self,
        navigation: GotoController,
        vehicle: Vehicle,
        local_map: ObservedGrid,
        x_m: float,
        y_m: float,
    ) -> None:
        navigation.start(
            x_m,
            y_m,
            local_map=local_map,
            pose=self.pose(vehicle),
            vehicle_radius_m=vehicle.radius,
        )

    def update(
        self,
        navigation: GotoController,
        vehicle: Vehicle,
        grid: MapGrid,
        local_map: ObservedGrid,
        now: float,
    ) -> None:
        navigation.update(
            vehicle,
            grid,
            now,
            pose=self.pose(vehicle, now),
            local_map=local_map,
        )

    @staticmethod
    def local_state(vehicle: Vehicle) -> AnchoredLocalState:
        return AnchoredLocalState(
            AnchorSpec("goto-runtime", vehicle.x, vehicle.y, vehicle.yaw),
            truth_x_m=vehicle.x,
            truth_y_m=vehicle.y,
            truth_yaw_rad=vehicle.yaw,
            timestamp=0.0,
        )

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
        local_map = self.observed()
        self.start(navigation, vehicle, local_map, 7.0, 5.0)

        self.update(navigation, vehicle, self.grid, local_map, 0.0)
        self.assertEqual(navigation.status, "active")
        self.assertEqual(vehicle.velocities()[:2], (0.0, 0.0))
        self.assertLess(vehicle.velocities()[2], 0.0)

        for step in range(1, 301):
            self.update(navigation, vehicle, self.grid, local_map, step * 0.05)
            if navigation.status == "reached":
                break

        self.assertEqual(navigation.status, "reached")
        self.assertEqual(navigation.goal, (7.0, 5.0))
        self.assertEqual(navigation.reason, "goal_tolerance")
        self.assertLessEqual(math.hypot(vehicle.x - 7.0, vehicle.y - 5.0), navigation.goal_tolerance_m)
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))
        stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)
        self.update(navigation, vehicle, self.grid, local_map, 20.0)
        self.assertEqual((vehicle.x, vehicle.y, vehicle.yaw), stopped_pose)

    def test_controller_slows_near_goal_and_refreshes_watchdog(self) -> None:
        vehicle = self.vehicle()
        navigation = GotoController()
        local_map = self.observed()
        self.start(navigation, vehicle, local_map, 5.2, 5.0)
        self.update(navigation, vehicle, self.grid, local_map, 0.0)
        self.assertGreater(vehicle.velocities()[0], 0.0)
        self.assertLess(vehicle.velocities()[0], vehicle.linear_speed)

        self.start(navigation, vehicle, local_map, 20.0, 5.0)
        for step in range(1, 31):
            self.update(navigation, vehicle, self.grid, local_map, step * 0.1)
        self.assertEqual(navigation.status, "active")
        self.assertEqual(vehicle.command, "drive")
        self.assertGreater(vehicle.x, 5.4)

    def test_collision_blocks_goal_without_restart(self) -> None:
        grid = MapGrid.from_wall_set(30, 30, {(7, y) for y in range(30)})
        vehicle = self.vehicle()
        navigation = GotoController()
        local_map = self.observed()
        self.start(navigation, vehicle, local_map, 10.0, 5.0)

        for step in range(200):
            self.update(navigation, vehicle, grid, local_map, step * 0.1)
            if navigation.status == "blocked":
                break

        self.assertEqual((navigation.status, navigation.reason), ("blocked", "collision"))
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))
        stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)
        self.update(navigation, vehicle, grid, local_map, (step + 1) * 0.1)
        self.assertEqual((vehicle.x, vehicle.y, vehicle.yaw), stopped_pose)

    def test_manual_or_invalid_input_cancels_active_goal_without_resume(self) -> None:
        vehicle = self.vehicle()
        navigation = GotoController()
        local_map = self.observed()
        self.start(navigation, vehicle, local_map, 10.0, 5.0)
        self.update(navigation, vehicle, self.grid, local_map, 0.0)

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
        self.update(navigation, vehicle, self.grid, local_map, 0.2)
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

        self.start(navigation, vehicle, local_map, 10.0, 5.0)
        self.update(navigation, vehicle, self.grid, local_map, 0.3)
        error = handle_command_message("not-json", vehicle, self.grid, 0.4, 10.1, navigation)
        self.assertEqual(error["type"], "error")
        self.assertEqual((navigation.status, navigation.reason), ("cancelled", "invalid_command"))
        self.update(navigation, vehicle, self.grid, local_map, 0.5)
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))

    def test_goto_ack_replacement_and_pose_status_are_observable(self) -> None:
        vehicle = self.vehicle()
        navigation = GotoController()
        local_state = self.local_state(vehicle)
        first = handle_command_message(
            json.dumps({"type": "goto", "seq": 3, "x_m": 8, "y_m": 6}),
            vehicle,
            self.grid,
            0.0,
            11.0,
            navigation,
            local_state=local_state,
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
            local_state=local_state,
        )
        self.assertEqual(navigation.reported_goal, (9.0, 5.0))

        navigation.update(
            vehicle,
            self.grid,
            0.1,
            pose=local_state.pose,
            local_map=local_state.local_map,
        )
        local_state.update_from_truth(
            vehicle.x, vehicle.y, vehicle.yaw, timestamp=11.2
        )
        pose, _scan = telemetry_messages(
            vehicle,
            self.grid,
            1,
            11.2,
            navigation,
            local_state=local_state,
        )
        self.assertEqual(pose["control_mode"], "autonomous")
        self.assertEqual(pose["navigation"]["status"], "active")
        self.assertEqual(pose["navigation"]["goal"], {"x_m": 9.0, "y_m": 5.0})
        self.assertEqual(pose["navigation"]["algorithm"], "d_star_lite")

        legacy = handle_command_message(
            '{"cmd":"stop"}',
            vehicle,
            self.grid,
            0.2,
            11.3,
            navigation,
            local_state=local_state,
        )
        self.assertEqual(
            legacy,
            {"type": "cmd_ack", "ts": 11.3, "seq": None, "cmd": "stop", "accepted": True},
        )
        pose, _scan = telemetry_messages(
            vehicle,
            self.grid,
            2,
            11.4,
            navigation,
            local_state=local_state,
        )
        self.assertEqual(pose["control_mode"], "manual")
        self.assertEqual(pose["navigation"]["status"], "cancelled")

    def test_goto_handoff_collision_blocks_replacement_goal(self) -> None:
        grid = MapGrid.from_wall_set(30, 30, {(7, y) for y in range(30)})
        vehicle = Vehicle(6.49, 5.0, command_timeout=0.25, now=0.0)
        navigation = GotoController()
        local_state = self.local_state(vehicle)
        handle_command_message(
            '{"type":"goto","seq":5,"x_m":10,"y_m":5}',
            vehicle,
            grid,
            0.0,
            12.0,
            navigation,
            local_state=local_state,
        )
        navigation.update(
            vehicle,
            grid,
            0.0,
            pose=local_state.pose,
            local_map=local_state.local_map,
        )

        ack = handle_command_message(
            '{"type":"goto","seq":6,"x_m":4,"y_m":5}',
            vehicle,
            grid,
            0.25,
            12.25,
            navigation,
            local_state=local_state,
        )

        self.assertEqual(
            ack,
            {
                "type": "goto_ack",
                "ts": 12.25,
                "seq": 6,
                "goal": {"x_m": 4.0, "y_m": 5.0},
                "accepted": False,
                "reason": "collision",
            },
        )
        self.assertEqual(navigation.reported_goal, (4.0, 5.0))
        self.assertEqual((navigation.status, navigation.reason), ("blocked", "collision"))
        self.assertEqual(vehicle.velocities(), (0.0, 0.0, 0.0))
        stopped_pose = (vehicle.x, vehicle.y, vehicle.yaw)
        navigation.update(
            vehicle,
            grid,
            0.3,
            pose=local_state.pose,
            local_map=local_state.local_map,
        )
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
            ["hello", "pose", "scan", "goto_ack", "pose", "scan"],
        )
        self.assertEqual(websocket.messages[3]["seq"], 21)
        self.assertEqual(websocket.messages[4]["control_mode"], "autonomous")
        self.assertEqual(websocket.messages[4]["navigation"]["status"], "active")


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GotoTest)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
