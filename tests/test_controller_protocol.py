"""Canonical v4 WebSocket command boundary."""

import asyncio
import json
import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.controller import (
    AutoAction,
    AutoCommand,
    ManualAction,
    ManualCommand,
    ModeAction,
    ModeCommand,
)
from mockvehicle2d.local_state import AnchorSpec, OdometryConfig
from mockvehicle2d.protocol import ProtocolError, parse_command
from mockvehicle2d.server import VehicleRuntime, handler


def parse(raw: object):
    return parse_command(
        raw,
        linear_limit_mps=0.5,
        angular_limit_rps=math.pi / 2,
        mission_batch_limit=3,
    )


class Socket:
    remote_address = ("test", 0)

    def __init__(self, commands: list[str], stop_type: str) -> None:
        self.commands = iter(commands)
        self.stop_type = stop_type
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            return
        message = json.loads(payload)
        self.messages.append(message)
        if message["type"] == self.stop_type:
            raise RuntimeError("test complete")

    async def recv(self) -> str:
        return next(self.commands)


class AckCountSocket:
    remote_address = ("test", 0)

    def __init__(self, commands: list[str], stop_after: int) -> None:
        self.commands = iter(commands)
        self.stop_after = stop_after
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            return
        message = json.loads(payload)
        self.messages.append(message)
        if (
            message["type"] == "command_ack"
            and sum(item["type"] == "command_ack" for item in self.messages)
            == self.stop_after
        ):
            raise RuntimeError("test complete")

    async def recv(self) -> str:
        return next(self.commands)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class TestControllerProtocol(unittest.TestCase):
    def test_parses_only_the_three_command_families(self) -> None:
        self.assertEqual(
            parse('{"type":"mode","seq":1,"action":"switch_to_auto"}'),
            ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
        )
        self.assertEqual(
            parse('{"type":"mode","seq":2,"action":"stop_motion"}'),
            ModeCommand(2, ModeAction.STOP_MOTION),
        )
        self.assertEqual(
            parse(
                '{"type":"manual","seq":3,"action":"drive",'
                '"linear_mps":0.2,"angular_rps":-0.3}'
            ),
            ManualCommand(3, ManualAction.DRIVE, 0.2, -0.3),
        )
        self.assertEqual(
            parse('{"type":"manual","seq":4,"action":"stop"}'),
            ManualCommand(4, ManualAction.STOP),
        )
        command = parse(
            '{"type":"auto","seq":5,"action":"push","missions":['
            '{"mission_id":"goto-1","type":"goto","frame_id":"global_map",'
            '"x_m":12.5,"y_m":8.25}]}'
        )
        self.assertIsInstance(command, AutoCommand)
        self.assertIs(command.action, AutoAction.PUSH)
        self.assertEqual(command.missions[0].mission_id, "goto-1")
        self.assertEqual(command.missions[0].submitted_seq, 5)

    def test_rejects_legacy_ambiguous_or_unsafe_messages(self) -> None:
        cases = [
            ('{"type":"goto","seq":1,"x_m":1,"y_m":2}', "invalid_type"),
            ('{"type":"cmd","seq":1,"cmd":"forward"}', "invalid_type"),
            ('{"type":"nl_command","seq":1,"text":"go"}', "invalid_type"),
            ('{"type":"mode","action":"switch_to_auto"}', "missing_seq"),
            ('{"type":"mode","seq":true,"action":"switch_to_auto"}', "invalid_seq"),
            ('{"type":"mode","seq":-1,"action":"switch_to_auto"}', "invalid_seq"),
            (
                '{"type":"mode","seq":1,"action":"switch_to_auto","extra":0}',
                "invalid_fields",
            ),
            (
                '{"type":"manual","seq":1,"action":"drive",'
                '"linear_mps":0.6,"angular_rps":0}',
                "drive_out_of_range",
            ),
            (
                '{"type":"manual","seq":1,"action":"drive",'
                '"linear_mps":NaN,"angular_rps":0}',
                "invalid_json",
            ),
            ('{"type":"manual","seq":1,"seq":2,"action":"stop"}', "invalid_json"),
            (
                '{"type":"auto","seq":1,"action":"push","missions":[]}',
                "invalid_missions",
            ),
            (
                '{"type":"auto","seq":1,"action":"push","missions":['
                '{"mission_id":"m","type":"goto","frame_id":"anchor_map",'
                '"x_m":1,"y_m":2}]}',
                "invalid_mission",
            ),
            (
                '{"type":"auto","seq":1,"action":"push","missions":['
                '{"mission_id":"m","type":"patrol","frame_id":"global_map",'
                '"x_m":1,"y_m":2}]}',
                "invalid_mission_type",
            ),
        ]
        for raw, code in cases:
            with self.subTest(raw=raw), self.assertRaises(ProtocolError) as caught:
                parse(raw)
            self.assertEqual(caught.exception.code, code)

    def test_push_batch_is_strict_duplicate_free_and_bounded(self) -> None:
        duplicate = (
            '{"type":"auto","seq":1,"action":"push","missions":['
            '{"mission_id":"same","type":"goto","frame_id":"global_map",'
            '"x_m":1,"y_m":2},'
            '{"mission_id":"same","type":"goto","frame_id":"global_map",'
            '"x_m":3,"y_m":4}]}'
        )
        with self.assertRaises(ProtocolError) as caught:
            parse(duplicate)
        self.assertEqual(caught.exception.code, "duplicate_mission_id")

        too_many = {
            "type": "auto",
            "seq": 2,
            "action": "push",
            "missions": [
                {
                    "mission_id": f"m-{index}",
                    "type": "goto",
                    "frame_id": "global_map",
                    "x_m": index,
                    "y_m": index,
                }
                for index in range(4)
            ],
        }
        with self.assertRaises(ProtocolError) as caught:
            parse(json.dumps(too_many))
        self.assertEqual(caught.exception.code, "mission_batch_too_large")

    def test_handler_snapshot_and_mode_ack(self) -> None:
        socket = Socket(
            ['{"type":"mode","seq":1,"action":"switch_to_auto"}'],
            "command_ack",
        )
        clock = Clock()
        asyncio.run(
            handler(
                socket,
                _monotonic=clock.monotonic,
                _wall_time=clock.monotonic,
            )
        )
        hello = next(
            message for message in socket.messages if message["type"] == "hello"
        )
        ack = next(
            message
            for message in socket.messages
            if message["type"] == "command_ack"
        )
        pose = next(
            message for message in socket.messages if message["type"] == "pose"
        )
        scan = next(
            message for message in socket.messages if message["type"] == "scan"
        )
        self.assertEqual(
            [message["type"] for message in socket.messages[:4]],
            ["hello", "pose", "scan", "command_ack"],
        )
        self.assertEqual(hello["protocol_version"], 4)
        self.assertEqual(hello["controller"]["mode"], "manual")
        self.assertEqual(
            (pose["seq"], pose["timestamp_s"]),
            (scan["seq"], scan["timestamp_s"]),
        )
        self.assertTrue(ack["accepted"])
        self.assertEqual(ack["controller"]["mode"], "auto")

    def test_handler_rejects_old_goto_and_fails_safe(self) -> None:
        socket = Socket(
            ['{"type":"goto","seq":1,"x_m":12,"y_m":10}'],
            "error",
        )
        clock = Clock()
        asyncio.run(
            handler(
                socket,
                _monotonic=clock.monotonic,
                _wall_time=clock.monotonic,
            )
        )
        error = next(
            message for message in socket.messages if message["type"] == "error"
        )
        self.assertEqual(error["code"], "invalid_type")

    def test_handler_rejects_replayed_sequence_numbers(self) -> None:
        socket = Socket(
            [
                '{"type":"mode","seq":7,"action":"switch_to_auto"}',
                '{"type":"auto","seq":7,"action":"pause"}',
            ],
            "error",
        )
        clock = Clock()
        asyncio.run(
            handler(
                socket,
                _monotonic=clock.monotonic,
                _wall_time=clock.monotonic,
            )
        )
        errors = [
            message for message in socket.messages if message["type"] == "error"
        ]
        self.assertEqual(errors[-1]["code"], "stale_seq")

    def test_busy_connection_uses_one_timestamp_and_does_not_mutate_runtime(
        self,
    ) -> None:
        async def scenario() -> None:
            runtime = VehicleRuntime.create(
                started_at=0.0,
                timestamp=10.0,
                anchor=AnchorSpec("busy-test", 10.0, 10.0, 0.0),
                odometry_config=OdometryConfig(),
            )
            await runtime.controller_lease.acquire()
            before = runtime.controller.snapshot()
            values = iter((20.0, 21.0))
            socket = Socket([], "error")
            await handler(
                socket,
                _runtime=runtime,
                _monotonic=lambda: 0.0,
                _wall_time=lambda: next(values),
            )
            error = socket.messages[-1]
            self.assertEqual(error["code"], "vehicle_busy")
            self.assertEqual(error["timestamp_s"], error["ts"])
            self.assertEqual(runtime.controller.snapshot(), before)
            self.assertTrue(runtime.controller_lease.locked())
            runtime.controller_lease.release()

        asyncio.run(scenario())

    def test_handler_stop_motion_pauses_auto_without_mode_assumptions(self) -> None:
        socket = AckCountSocket(
            [
                '{"type":"mode","seq":1,"action":"switch_to_auto"}',
                '{"type":"auto","seq":2,"action":"push","missions":['
                '{"mission_id":"hold","type":"goto","frame_id":"global_map",'
                '"x_m":12,"y_m":10}]}',
                '{"type":"mode","seq":3,"action":"stop_motion"}',
            ],
            3,
        )
        clock = Clock()
        asyncio.run(
            handler(
                socket,
                _monotonic=clock.monotonic,
                _wall_time=clock.monotonic,
            )
        )
        acknowledgements = [
            message
            for message in socket.messages
            if message["type"] == "command_ack"
        ]
        stopped = acknowledgements[-1]
        self.assertTrue(stopped["accepted"])
        self.assertEqual(
            stopped["command"],
            {"type": "mode", "action": "stop_motion"},
        )
        self.assertEqual(stopped["controller"]["mode"], "auto")
        self.assertEqual(stopped["controller"]["auto_state"], "paused")
        self.assertEqual(
            stopped["controller"]["mission_queue"]["mission_ids"],
            ["hold"],
        )


if __name__ == "__main__":
    unittest.main()
