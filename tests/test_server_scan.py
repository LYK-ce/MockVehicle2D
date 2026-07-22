"""Wire-format and send-order checks for the WebSocket scan frame."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.test_collision import main as collision_main
from mockvehicle2d.cli.main import _port
from mockvehicle2d.collision import is_circle_passable
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.scan import scan_message
from mockvehicle2d.server import _next_deadline, generate_map, handler, main as server_main, telemetry_messages, validate_vehicle_id
from mockvehicle2d.vehicle import Vehicle


class _StopAfterScanSocket:
    remote_address = ("test", 0)

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))
        if len(self.messages) == 4:
            raise RuntimeError("stop after first scan")


class _CommandSocket:
    remote_address = ("test", 0)

    def __init__(self, command: str) -> None:
        self.command = command
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))
        if len(self.messages) == 5:
            raise RuntimeError("stop after command reply")

    async def recv(self) -> str:
        return self.command


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class _IdleTimeoutSocket:
    remote_address = ("test", 0)

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))
        if len(self.messages) == 6:
            raise RuntimeError("stop after telemetry following idle timeout")

    async def recv(self) -> str:
        self.clock.now += 1 / 6
        raise asyncio.TimeoutError


class _ImmediateEvent:
    def set(self) -> None:
        pass

    async def wait(self) -> None:
        pass


class _ServerContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        pass


class ScanMessageTest(unittest.TestCase):
    def test_cli_port_bounds_and_custom_server_port(self) -> None:
        self.assertEqual((_port("1"), _port("65535")), (1, 65535))
        for invalid in ("", "0", "65536", "1.5", "not-a-port"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                _port(invalid)

        async def run_server() -> None:
            with (
                patch("mockvehicle2d.server.asyncio.Event", return_value=_ImmediateEvent()),
                patch("mockvehicle2d.server.signal.signal"),
                patch("websockets.asyncio.server.serve", return_value=_ServerContext()) as serve,
            ):
                await server_main(port=19090)
                self.assertEqual(serve.call_args.args[1:], ("0.0.0.0", 19090))

        asyncio.run(run_server())

    def test_existing_collision_suite_still_passes(self) -> None:
        self.assertEqual(collision_main(), 0)

    def test_scan_message_contains_laserscan_metadata_and_points(self) -> None:
        grid = MapGrid.from_wall_set(8, 4, {(4, 1)})
        message = scan_message(grid, 1.5, 1.5, 0.0, 1717800000.124)
        self.assertEqual(message["type"], "scan")
        self.assertEqual(message["frame_id"], "laser")
        self.assertEqual(message["config"]["model"], "ydlidar_tmini")
        self.assertEqual(message["config"]["no_return"], {"range": 0.0, "intensity": 0.0})
        self.assertEqual(len(message["points"]), message["config"]["point_count"])
        forward = next(point for point in message["points"] if abs(point["angle"]) < 1e-9)
        self.assertAlmostEqual(forward["range"], 2.5)
        self.assertEqual(forward["intensity"], 1.0)

    def test_server_sends_tmini_scan_immediately_after_pose(self) -> None:
        websocket = _StopAfterScanSocket()
        asyncio.run(handler(websocket, vehicle_id="pictor_test-1"))
        self.assertEqual(
            [message["type"] for message in websocket.messages], ["hello", "map_full", "pose", "scan"]
        )
        self.assertEqual(websocket.messages[0], {"type": "hello", "vehicle_id": "pictor_test-1"})
        self.assertEqual(websocket.messages[2]["x"], 10.0)
        self.assertEqual(websocket.messages[1]["source"], "simulator_ground_truth")
        self.assertEqual(websocket.messages[2]["source"], "simulator_ground_truth")
        self.assertEqual(websocket.messages[2]["command"], "stop")
        self.assertEqual(websocket.messages[2]["seq"], websocket.messages[3]["seq"])
        self.assertEqual(websocket.messages[2]["ts"], websocket.messages[3]["ts"])
        self.assertEqual(websocket.messages[-1]["config"]["model"], "ydlidar_tmini")

    def test_vehicle_id_is_safe_for_pictor_names_and_logs(self) -> None:
        self.assertEqual(validate_vehicle_id("mock.Vehicle_01-test"), "mock.Vehicle_01-test")
        for invalid in ("", "a" * 65, "vehicle/01", "vehicle 01", "小车"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_vehicle_id(invalid)

    def test_timing_integrates_elapsed_time_and_skips_stale_deadlines(self) -> None:
        deadline = _next_deadline(100.0, 100.5, 1 / 6)
        self.assertGreater(deadline, 100.5)
        self.assertAlmostEqual(deadline, 100.0 + 4 / 6)

    def test_pose_and_scan_are_one_snapshot(self) -> None:
        grid = MapGrid.from_wall_set(20, 20, set())
        vehicle = Vehicle(10.0, 10.0, now=4.0)
        pose, scan = telemetry_messages(vehicle, grid, 12, 1717800000.5)
        self.assertEqual((pose["seq"], scan["seq"]), (12, 12))
        self.assertEqual((pose["ts"], scan["ts"]), (1717800000.5, 1717800000.5))
        self.assertEqual((pose["x"], pose["y"]), (10.0, 10.0))

    def test_spawn_is_clear_and_seed_is_deterministic(self) -> None:
        voxels, grid = generate_map(size=20, seed=42)
        again, _ = generate_map(size=20, seed=42)
        self.assertEqual(voxels, again)
        self.assertTrue(is_circle_passable(grid, 10.0, 10.0, 0.5))

    def test_handler_sends_immediate_ack_or_error_without_parallel_sender(self) -> None:
        accepted = _CommandSocket(
            '{"type":"drive","seq":3,"linear_mps":0.25,"angular_rps":-0.4}'
        )
        asyncio.run(handler(accepted))
        self.assertEqual(
            [message["type"] for message in accepted.messages], ["hello", "map_full", "pose", "scan", "cmd_ack"]
        )
        self.assertEqual(accepted.messages[-1]["seq"], 3)
        self.assertEqual(accepted.messages[-1]["cmd"], "drive")

        rejected = _CommandSocket(
            '{"type":"drive","seq":4,"linear_mps":0.51,"angular_rps":0}'
        )
        asyncio.run(handler(rejected))
        self.assertEqual(
            [message["type"] for message in rejected.messages], ["hello", "map_full", "pose", "scan", "error"]
        )
        self.assertEqual(rejected.messages[-1]["seq"], 4)
        self.assertEqual(rejected.messages[-1]["code"], "drive_out_of_range")

    def test_idle_receive_timeout_continues_telemetry_without_sleeping(self) -> None:
        class LegacyAsyncioTimeoutError(Exception):
            pass

        clock = _Clock()
        websocket = _IdleTimeoutSocket(clock)
        with patch.object(asyncio, "TimeoutError", LegacyAsyncioTimeoutError):
            asyncio.run(handler(websocket, _monotonic=clock.monotonic, _wall_time=lambda: 123.0))

        self.assertEqual(
            [message["type"] for message in websocket.messages],
            ["hello", "map_full", "pose", "scan", "pose", "scan"],
        )


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScanMessageTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
