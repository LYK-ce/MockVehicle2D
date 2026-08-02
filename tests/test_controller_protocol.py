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
    CoverageMission,
    GotoMission,
    ManualAction,
    ManualCommand,
    ModeAction,
    ModeCommand,
    PatrolMission,
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


class EventSocket:
    remote_address = ("test", 0)

    def __init__(
        self,
        commands: list[str] | None = None,
        *,
        fail_type: str | None = None,
        fail_status: str | None = None,
        record_failure: bool = False,
        clock: Clock | None = None,
        advance_after_commands: float | None = None,
    ) -> None:
        self.commands = iter(commands or ())
        self.fail_type = fail_type
        self.fail_status = fail_status
        self.record_failure = record_failure
        self.clock = clock
        self.advance_after_commands = advance_after_commands
        self.messages: list[dict[str, object]] = []

    async def send(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            return
        message = json.loads(payload)
        failed = message["type"] == self.fail_type or (
            message["type"] == "mission_update"
            and message.get("status") == self.fail_status
        )
        if not failed or self.record_failure:
            self.messages.append(message)
        if failed:
            raise ConnectionError("injected send failure")

    async def recv(self) -> str:
        try:
            return next(self.commands)
        except StopIteration:
            if self.clock is not None and self.advance_after_commands is not None:
                self.clock.now = self.advance_after_commands
                raise asyncio.TimeoutError
            raise


def active_runtime(anchor_id: str) -> VehicleRuntime:
    runtime = VehicleRuntime.create(
        started_at=0.0,
        timestamp=0.0,
        anchor=AnchorSpec(anchor_id, 10.0, 10.0, 0.0),
        odometry_config=OdometryConfig(),
    )
    mode = runtime.handle_command(
        ModeCommand(1, ModeAction.SWITCH_TO_AUTO),
        monotonic_now=0.0,
    )
    pushed = runtime.handle_command(
        AutoCommand(
            2,
            AutoAction.PUSH,
            (GotoMission("active", "global_map", 12.0, 10.0, 2),),
        ),
        monotonic_now=0.0,
    )
    runtime.update(0.0, 0.0)
    assert mode.accepted and pushed.accepted
    return runtime


def serve_socket(
    socket: EventSocket,
    runtime: VehicleRuntime,
    clock: Clock,
) -> None:
    asyncio.run(
        handler(
            socket,
            _runtime=runtime,
            _monotonic=clock.monotonic,
            _wall_time=clock.monotonic,
        )
    )


def mission_updates(socket: EventSocket) -> list[dict[str, object]]:
    return [
        message
        for message in socket.messages
        if message["type"] == "mission_update"
    ]


class TestControllerProtocol(unittest.TestCase):
    def test_runtime_uses_half_metre_local_planning_grid(self) -> None:
        runtime = VehicleRuntime.create(
            started_at=0.0,
            timestamp=0.0,
            anchor=AnchorSpec("resolution-test", 10.0, 10.0, 0.0),
            odometry_config=OdometryConfig(),
        )

        self.assertEqual(runtime.local_state.local_map.resolution_m, 0.5)

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

    def test_parses_patrol_and_deterministic_coverage_routes(self) -> None:
        command = parse(
            json.dumps(
                {
                    "type": "auto",
                    "seq": 6,
                    "action": "push",
                    "missions": [
                        {
                            "mission_id": "patrol-1",
                            "type": "patrol",
                            "frame_id": "global_map",
                            "waypoints": [
                                {"x_m": 1, "y_m": 2},
                                {"x_m": 3, "y_m": 4},
                            ],
                            "cycles": 2,
                        },
                        {
                            "mission_id": "coverage-1",
                            "type": "coverage",
                            "frame_id": "global_map",
                            "area": {
                                "min_x_m": 0,
                                "min_y_m": 0,
                                "max_x_m": 4,
                                "max_y_m": 2,
                            },
                            "lane_spacing_m": 1.5,
                        },
                    ],
                }
            )
        )

        patrol, coverage = command.missions
        self.assertIsInstance(patrol, PatrolMission)
        self.assertEqual(
            patrol.subgoals,
            ((1.0, 2.0), (3.0, 4.0), (1.0, 2.0), (3.0, 4.0)),
        )
        self.assertIsInstance(coverage, CoverageMission)
        self.assertEqual(
            coverage.subgoals,
            (
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.5),
                (0.0, 1.5),
                (0.0, 2.0),
                (4.0, 2.0),
            ),
        )

        vertical = parse(
            '{"type":"auto","seq":7,"action":"push","missions":['
            '{"mission_id":"coverage-2","type":"coverage",'
            '"frame_id":"global_map","area":{"min_x_m":0,"min_y_m":0,'
            '"max_x_m":2,"max_y_m":4},"lane_spacing_m":3}]}'
        ).missions[0]
        self.assertEqual(
            vertical.subgoals,
            ((0.0, 0.0), (0.0, 4.0), (2.0, 4.0), (2.0, 0.0)),
        )

    def test_rejects_invalid_or_oversized_high_level_missions_atomically(self) -> None:
        invalid_missions = [
            {
                "mission_id": "patrol-empty",
                "type": "patrol",
                "frame_id": "global_map",
                "waypoints": [],
                "cycles": 1,
            },
            {
                "mission_id": "patrol-cycles",
                "type": "patrol",
                "frame_id": "global_map",
                "waypoints": [{"x_m": 0, "y_m": 0}],
                "cycles": True,
            },
            {
                "mission_id": "patrol-large",
                "type": "patrol",
                "frame_id": "global_map",
                "waypoints": [{"x_m": 0, "y_m": 0}, {"x_m": 1, "y_m": 1}],
                "cycles": 513,
            },
            {
                "mission_id": "coverage-area",
                "type": "coverage",
                "frame_id": "global_map",
                "area": {
                    "min_x_m": 1,
                    "min_y_m": 0,
                    "max_x_m": 1,
                    "max_y_m": 2,
                },
                "lane_spacing_m": 1,
            },
            {
                "mission_id": "coverage-spacing",
                "type": "coverage",
                "frame_id": "global_map",
                "area": {
                    "min_x_m": 0,
                    "min_y_m": 0,
                    "max_x_m": 1,
                    "max_y_m": 1,
                },
                "lane_spacing_m": 0,
            },
            {
                "mission_id": "coverage-large",
                "type": "coverage",
                "frame_id": "global_map",
                "area": {
                    "min_x_m": 0,
                    "min_y_m": 0,
                    "max_x_m": 1,
                    "max_y_m": 1,
                },
                "lane_spacing_m": 1e-300,
            },
        ]
        valid = {
            "mission_id": "still-not-queued",
            "type": "goto",
            "frame_id": "global_map",
            "x_m": 1,
            "y_m": 2,
        }
        for invalid in invalid_missions:
            raw = json.dumps(
                {
                    "type": "auto",
                    "seq": 8,
                    "action": "push",
                    "missions": [valid, invalid],
                }
            )
            with self.subTest(mission_id=invalid["mission_id"]), self.assertRaises(
                ProtocolError
            ) as caught:
                parse(raw)
            self.assertEqual(caught.exception.code, "invalid_mission")

        strict_cases = [
            (
                {
                    "mission_id": "waypoint-extra",
                    "type": "patrol",
                    "frame_id": "global_map",
                    "waypoints": [{"x_m": 0, "y_m": 0, "z_m": 0}],
                    "cycles": 1,
                },
                "invalid_fields",
            ),
            (
                {
                    "mission_id": "area-extra",
                    "type": "coverage",
                    "frame_id": "global_map",
                    "area": {
                        "min_x_m": 0,
                        "min_y_m": 0,
                        "max_x_m": 1,
                        "max_y_m": 1,
                        "yaw_rad": 0,
                    },
                    "lane_spacing_m": 1,
                },
                "invalid_fields",
            ),
            (
                {
                    "mission_id": "far-waypoint",
                    "type": "patrol",
                    "frame_id": "global_map",
                    "waypoints": [{"x_m": 1_000_001, "y_m": 0}],
                    "cycles": 1,
                },
                "goal_out_of_range",
            ),
        ]
        for invalid, code in strict_cases:
            raw = json.dumps(
                {
                    "type": "auto",
                    "seq": 9,
                    "action": "push",
                    "missions": [invalid],
                }
            )
            with self.subTest(mission_id=invalid["mission_id"]), self.assertRaises(
                ProtocolError
            ) as caught:
                parse(raw)
            self.assertEqual(caught.exception.code, code)

    def test_rejects_legacy_ambiguous_or_unsafe_messages(self) -> None:
        cases = [
            ('{"type":"goto","seq":1,"x_m":1,"y_m":2}', "invalid_type"),
            ('{"type":"cmd","seq":1,"cmd":"forward"}', "invalid_type"),
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
                '{"mission_id":"m","type":"orbit","frame_id":"global_map",'
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
        self.assertEqual(hello["mission_types"], ["goto", "patrol", "coverage"])
        self.assertEqual(hello["controller"]["mode"], "manual")
        event_info = hello["controller"]["mission_events"]
        self.assertEqual(event_info["latest_event_seq"], 0)
        self.assertEqual(event_info["retention"], "process_lifetime")
        self.assertRegex(event_info["event_epoch"], r"^[0-9a-f]{32}$")
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
            socket = Socket([], "error")
            await handler(
                socket,
                _runtime=runtime,
                _monotonic=lambda: 0.0,
                _wall_time=lambda: 20.0,
            )
            error = socket.messages[-1]
            self.assertEqual(error["code"], "vehicle_busy")
            self.assertEqual(error["timestamp_s"], 20.0)
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

    def test_reconnect_replays_command_path_result_after_send_failure(self) -> None:
        runtime = VehicleRuntime.create(
            started_at=0.0,
            timestamp=0.0,
            anchor=AnchorSpec("event-command", 10.0, 10.0, 0.0),
            odometry_config=OdometryConfig(),
        )
        clock = Clock()
        first = EventSocket(
            [
                '{"type":"mode","seq":1,"action":"switch_to_auto"}',
                '{"type":"auto","seq":2,"action":"push","missions":['
                '{"mission_id":"cancel-me","type":"goto",'
                '"frame_id":"global_map","x_m":12,"y_m":10}]}',
                '{"type":"auto","seq":3,"action":"cancel_all"}',
            ],
            fail_status="cancelled",
        )
        serve_socket(first, runtime, clock)
        self.assertFalse(
            any(
                message.get("status") == "cancelled"
                for message in first.messages
            )
        )

        reconnect = EventSocket(fail_status="cancelled", record_failure=True)
        serve_socket(reconnect, runtime, clock)
        updates = mission_updates(reconnect)
        self.assertEqual(
            [(message["mission_id"], message["status"]) for message in updates],
            [("cancel-me", "queued"), ("cancel-me", "cancelled")],
        )
        self.assertEqual(
            [message["event_seq"] for message in updates],
            [1, 2],
        )
        self.assertEqual(
            {message["event_epoch"] for message in updates},
            {runtime.controller.event_epoch},
        )

    def test_reconnect_replays_frame_path_result_when_pose_send_precedes_it(
        self,
    ) -> None:
        runtime = VehicleRuntime.create(
            started_at=0.0,
            timestamp=0.0,
            anchor=AnchorSpec("event-frame", 10.0, 10.0, 0.0),
            odometry_config=OdometryConfig(),
        )
        clock = Clock()
        first = EventSocket(
            [
                '{"type":"mode","seq":1,"action":"switch_to_auto"}',
                '{"type":"auto","seq":2,"action":"push","missions":['
                '{"mission_id":"already-there","type":"goto",'
                '"frame_id":"global_map","x_m":10,"y_m":10}]}',
            ],
            fail_status="reached",
            clock=clock,
            advance_after_commands=1.0,
        )
        serve_socket(first, runtime, clock)
        self.assertFalse(
            any(message.get("status") == "reached" for message in first.messages)
        )

        reconnect = EventSocket(fail_status="reached", record_failure=True)
        serve_socket(reconnect, runtime, clock)
        updates = mission_updates(reconnect)
        self.assertEqual(
            [(message["mission_id"], message["status"]) for message in updates],
            [
                ("already-there", "queued"),
                ("already-there", "active"),
                ("already-there", "reached"),
            ],
        )
        self.assertEqual(
            [message["event_seq"] for message in updates],
            [1, 2, 3],
        )
        self.assertEqual(
            {message["event_epoch"] for message in updates},
            {runtime.controller.event_epoch},
        )

    def test_disconnect_pause_survives_telemetry_send_failure(self) -> None:
        runtime = active_runtime("event-disconnect")
        clock = Clock()
        serve_socket(EventSocket(fail_type="pose"), runtime, clock)

        reconnect = EventSocket(fail_status="paused", record_failure=True)
        serve_socket(reconnect, runtime, clock)
        updates = mission_updates(reconnect)
        self.assertEqual(
            [(message["event_seq"], message["status"]) for message in updates],
            [(1, "queued"), (2, "active"), (3, "paused")],
        )
        self.assertEqual(updates[-1]["reason"], "controller_disconnected")

    def test_invalid_command_pause_is_sent_once_in_order(self) -> None:
        runtime = active_runtime("event-invalid")
        clock = Clock()
        socket = EventSocket(
            ['{"type":"goto","seq":3,"x_m":12,"y_m":10}'],
            fail_status="paused",
            record_failure=True,
        )
        serve_socket(socket, runtime, clock)
        updates = mission_updates(socket)
        self.assertEqual(
            [(message["event_seq"], message["status"]) for message in updates],
            [(1, "queued"), (2, "active"), (3, "paused")],
        )
        self.assertEqual(updates[-1]["reason"], "invalid_command")


if __name__ == "__main__":
    unittest.main()
