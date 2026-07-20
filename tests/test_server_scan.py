"""Wire-format and send-order checks for the WebSocket scan frame."""

import asyncio
import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.test_collision import main as collision_main
from mockvehicle2d.collision import is_circle_passable
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.scan import scan_message
from mockvehicle2d.server import _next_deadline, generate_map, handler, telemetry_messages
from mockvehicle2d.vehicle import Vehicle


class _StopAfterScanSocket:
    remote_address = ("test", 0)

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))
        if len(self.messages) == 3:
            raise RuntimeError("stop after first scan")


class _CommandSocket:
    remote_address = ("test", 0)

    def __init__(self, command: str) -> None:
        self.command = command
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))
        if len(self.messages) == 4:
            raise RuntimeError("stop after command reply")

    async def recv(self) -> str:
        return self.command


class ScanMessageTest(unittest.TestCase):
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
        asyncio.run(handler(websocket))
        self.assertEqual([message["type"] for message in websocket.messages], ["map_full", "pose", "scan"])
        self.assertEqual(websocket.messages[1]["x"], 10.0)
        self.assertEqual(websocket.messages[0]["source"], "simulator_ground_truth")
        self.assertEqual(websocket.messages[1]["source"], "simulator_ground_truth")
        self.assertEqual(websocket.messages[1]["command"], "stop")
        self.assertEqual(websocket.messages[1]["seq"], websocket.messages[2]["seq"])
        self.assertEqual(websocket.messages[1]["ts"], websocket.messages[2]["ts"])
        self.assertEqual(websocket.messages[-1]["config"]["model"], "ydlidar_tmini")

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
        accepted = _CommandSocket('{"type":"cmd","seq":3,"cmd":"forward"}')
        asyncio.run(handler(accepted))
        self.assertEqual([message["type"] for message in accepted.messages], ["map_full", "pose", "scan", "cmd_ack"])
        self.assertEqual(accepted.messages[-1]["seq"], 3)

        rejected = _CommandSocket('{"type":"pose","seq":4,"cmd":"forward"}')
        asyncio.run(handler(rejected))
        self.assertEqual([message["type"] for message in rejected.messages], ["map_full", "pose", "scan", "error"])
        self.assertEqual(rejected.messages[-1]["seq"], 4)


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScanMessageTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
