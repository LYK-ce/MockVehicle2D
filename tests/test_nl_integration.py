"""Integration tests for NL command pipeline in server.py.

Tests the NL → parse → validate → execute pipeline without WebSocket.
Uses a deterministic test parser for reproducible results.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import mockvehicle2d.server as server_module
from mockvehicle2d.instruction.state_machine import InstructionState, InstructionStateMachine
from mockvehicle2d.instruction.validator import SchemaValidator, SemanticValidator
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.local_state import AnchorSpec, AnchoredLocalState, OdometryConfig
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.safety import LocalSafetyRuntime
from mockvehicle2d.server import (
    _handle_nl_command,
    _nl_completion_reason,
    _process_next_in_queue,
    handle_command_message,
    handler,
    VehicleRuntime,
)
from mockvehicle2d.vehicle import Vehicle


# ═══════════════════════════════════════════════════════════════
# Test helper — minimal deterministic parser for pipeline tests
# ═══════════════════════════════════════════════════════════════

import re as _re


class _TestParser:
    """Minimal deterministic NL parser for integration testing.

    Replaces LLMClient with a simple regex-based parser
    that covers the inputs used in these tests.
    """

    _STOP_WORDS = {"停", "停下", "停止", "紧急停止", "别动了"}
    _GOTO_PAT = _re.compile(r"去.*?(\d+(?:\.\d+)?).*?(\d+(?:\.\d+)?)")
    _PATROL_WORDS = {"开始巡逻", "巡逻", "启动巡逻"}
    _CLARIFY_WORDS = {"开到那边去"}

    def parse(self, text: str) -> list[dict]:
        text = text.strip()
        if not text:
            return [{"intent": "clarify", "parameters": {"question": "请输入指令"}}]

        if text in self._STOP_WORDS:
            return [{"intent": "stop", "parameters": {}}]

        if text in self._PATROL_WORDS:
            return [{"intent": "patrol", "parameters": {}}]

        if text in self._CLARIFY_WORDS:
            return [{"intent": "clarify", "parameters": {"question": "请指定坐标"}}]

        # goto patterns
        m = self._GOTO_PAT.search(text)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            return [{"intent": "goto", "parameters": {"x_m": x, "y_m": y}}]

        return [{"intent": "clarify", "parameters": {"question": "请指定坐标"}}]


class _AsyncTestParser(_TestParser):
    async def parse(self, text: str) -> list[dict]:
        return super().parse(text)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def empty_grid():
    return MapGrid(256, 256)


@pytest.fixture
def vehicle():
    return Vehicle(10.0, 10.0, now=time.monotonic())


@pytest.fixture(autouse=True)
def estimated_nl_runtime(monkeypatch, vehicle):
    """Route every NL test through the same estimated state as the real handler."""
    state = AnchoredLocalState(
        AnchorSpec("nl-test", vehicle.x, vehicle.y, vehicle.yaw),
        truth_x_m=vehicle.x,
        truth_y_m=vehicle.y,
        truth_yaw_rad=vehicle.yaw,
        timestamp=0.0,
    )
    original = _handle_nl_command

    def with_estimate(*args, **kwargs):
        state.update_from_truth(
            vehicle.x, vehicle.y, vehicle.yaw, timestamp=time.time()
        )
        kwargs.setdefault("local_state", state)
        return original(*args, **kwargs)

    monkeypatch.setitem(globals(), "_handle_nl_command", with_estimate)
    return state


@pytest.fixture
def navigation():
    return GotoController()


@pytest.fixture
def safety():
    return LocalSafetyRuntime(healthy=True)


@pytest.fixture
def nl_client():
    return _TestParser()


@pytest.fixture
def schema_v():
    return SchemaValidator()


@pytest.fixture
def semantic_v(empty_grid):
    return SemanticValidator(empty_grid)


@pytest.fixture
def state_machine():
    return InstructionStateMachine()


class _Clock:
    now = 0.0

    def monotonic(self):
        return self.now


class _InvalidSeqSocket:
    remote_address = ("nl-seq", 0)

    def __init__(self, clock, seq):
        self.clock = clock
        self.seq = seq
        self.messages = []
        self.received = False

    async def send(self, payload):
        if isinstance(payload, bytes):
            return
        message = json.loads(payload)
        self.messages.append(message)
        if message.get("type") == "nl_task_update" and message.get("status") == "completed":
            raise RuntimeError("test complete")

    async def recv(self):
        if not self.received:
            self.received = True
            return json.dumps(
                {"type": "nl_command", "seq": self.seq, "text": "去坐标 (10.1, 10)"}
            )
        self.clock.now = 0.2
        raise asyncio.TimeoutError


class _ActiveOwnerSocket:
    remote_address = ("nl-owner", 0)

    def __init__(self, clock, second_type, outcome):
        self.clock = clock
        self.second_type = second_type
        self.outcome = outcome
        self.messages = []
        self.receive_count = 0
        self.runtime = None

    async def send(self, payload):
        if isinstance(payload, bytes):
            return
        message = json.loads(payload)
        self.messages.append(message)
        if (
            message.get("type") == "nl_task_update"
            and message.get("status") == self.outcome
        ):
            raise RuntimeError("test complete")

    async def recv(self):
        self.receive_count += 1
        if self.receive_count == 1:
            return json.dumps(
                {"type": "nl_command", "seq": 1, "text": "去坐标 (10.1, 10)"}
            )
        if self.receive_count == 2:
            return json.dumps(
                {"type": self.second_type, "seq": 2, "text": "停"}
            )
        if self.outcome == "blocked":
            self.runtime.navigation.status = "blocked"
            self.runtime.navigation.reason = "test_block"
        self.clock.now = 0.2
        raise asyncio.TimeoutError


# ═══════════════════════════════════════════════════════════════
# NL command execution tests
# ═══════════════════════════════════════════════════════════════

class TestNlCommandStop:
    """NL '停' → vehicle stops."""

    def test_stop_stops_vehicle(self, vehicle, empty_grid, navigation, nl_client,
                                 schema_v, semantic_v, state_machine):
        # Give vehicle some velocity by advancing then installing
        now = time.monotonic()
        vehicle.advance(empty_grid, now)
        vehicle.install_command("forward", now)
        assert vehicle.command == "forward"

        msg = {"type": "nl_command", "seq": 1, "text": "停"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert vehicle.command == "stop"
        assert len(replies) >= 2  # nl_parse_result + nl_task_update
        # First reply should be accepted parse result
        assert replies[0]["type"] == "nl_parse_result"
        assert replies[0]["accepted"] is True
        # Last reply should be task update with completed
        task_update = [r for r in replies if r["type"] == "nl_task_update"]
        assert len(task_update) == 1
        assert task_update[0]["status"] == "completed"
        assert "stopped" in str(task_update[0].get("reason", ""))

    def test_stop_handoff_integrates_the_same_mid_tick_motion_as_direct_command(
        self, empty_grid, nl_client, schema_v, semantic_v
    ):
        direct = Vehicle(10.0, 10.0, now=0.0)
        natural = Vehicle(10.0, 10.0, now=0.0)
        direct.install_command("forward", 0.0)
        natural.install_command("forward", 0.0)
        direct_state = AnchoredLocalState(
            AnchorSpec("direct", 10.0, 10.0, 0.0),
            truth_x_m=10.0,
            truth_y_m=10.0,
            truth_yaw_rad=0.0,
            timestamp=0.0,
        )
        natural_state = AnchoredLocalState(
            AnchorSpec("natural", 10.0, 10.0, 0.0),
            truth_x_m=10.0,
            truth_y_m=10.0,
            truth_yaw_rad=0.0,
            timestamp=0.0,
        )

        handle_command_message(
            '{"type":"cmd","seq":1,"cmd":"stop"}',
            direct,
            empty_grid,
            0.5,
            10.5,
            GotoController(),
            local_state=direct_state,
        )
        _handle_nl_command(
            {"type": "nl_command", "seq": 2, "text": "停"},
            natural,
            empty_grid,
            GotoController(),
            10.5,
            0.5,
            nl_client,
            schema_v,
            semantic_v,
            InstructionStateMachine(),
            local_state=natural_state,
        )

        assert natural.x == pytest.approx(direct.x)
        assert natural_state.pose.x_m == pytest.approx(direct_state.pose.x_m)
        assert natural.command == direct.command == "stop"


class TestNlCommandGoto:
    """NL '去坐标 (50, 50)' → GotoController.start(50, 50)."""

    def test_goto_starts_navigation(self, vehicle, empty_grid, navigation, nl_client,
                                     schema_v, semantic_v, state_machine):
        msg = {"type": "nl_command", "seq": 2, "text": "去坐标 (50, 50)"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert navigation.status == "active"
        assert navigation.reported_goal == (50.0, 50.0)
        assert state_machine.current_state == InstructionState.ACTIVE

        # Check replies
        assert len(replies) == 2  # nl_parse_result(accepted) + nl_task_update(active)
        parse_results = [r for r in replies if r["type"] == "nl_parse_result"]
        assert len(parse_results) == 1
        assert parse_results[0]["accepted"] is True

        task_updates = [r for r in replies if r["type"] == "nl_task_update"]
        assert len(task_updates) == 1
        assert task_updates[0]["status"] == "active"

    def test_goto_respects_goal(self, vehicle, empty_grid, navigation, nl_client,
                                 schema_v, semantic_v, state_machine):
        msg = {"type": "nl_command", "seq": 3, "text": "去 (100, 200)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert navigation.reported_goal == (100.0, 200.0)
        assert navigation.status == "active"

    def test_goto_without_local_state_fails_closed(
        self,
        vehicle,
        empty_grid,
        navigation,
        nl_client,
        schema_v,
        semantic_v,
    ):
        state_machine = InstructionStateMachine()

        replies = server_module._handle_nl_command(
            {"type": "nl_command", "seq": 4, "text": "去坐标 (20, 20)"},
            vehicle,
            empty_grid,
            navigation,
            time.time(),
            time.monotonic(),
            nl_client,
            schema_v,
            semantic_v,
            state_machine,
        )

        assert replies[-1]["status"] == "blocked"
        assert replies[-1]["reason"] == "local_state_unavailable"
        assert navigation.status != "active"
        assert navigation.snapshot()["planning"] is False
        assert not state_machine.has_more()

    def test_websocket_stays_open_after_missing_local_state_goto(self):
        clock = _Clock()
        runtime = VehicleRuntime.create(
            started_at=0.0,
            anchor=AnchorSpec("nl-missing-state", 10.0, 10.0, 0.0),
            odometry_config=OdometryConfig(),
        )

        class Socket:
            remote_address = ("nl-missing-state", 0)

            def __init__(self):
                self.messages = []
                self.receive_count = 0

            async def send(self, payload):
                if isinstance(payload, bytes):
                    return
                message = json.loads(payload)
                self.messages.append(message)
                if message.get("type") == "cmd_ack":
                    raise RuntimeError("test complete")

            async def recv(self):
                self.receive_count += 1
                if self.receive_count == 1:
                    runtime.local_state = None
                    return json.dumps(
                        {"type": "nl_command", "seq": 5, "text": "去坐标 (20, 20)"}
                    )
                return '{"type":"cmd","seq":6,"cmd":"stop"}'

        socket = Socket()
        asyncio.run(
            handler(
                socket,
                _runtime=runtime,
                _monotonic=clock.monotonic,
                _wall_time=lambda: 10.0,
                _nl_client=_TestParser(),
            )
        )

        assert any(
            message.get("status") == "blocked"
            and message.get("reason") == "local_state_unavailable"
            for message in socket.messages
        )
        assert any(
            message.get("type") == "cmd_ack" and message.get("seq") == 6
            for message in socket.messages
        )
        assert runtime.navigation.status != "active"
        assert runtime.navigation.snapshot()["planning"] is False


class TestNlCommandClarify:
    """NL '开到那边去' → clarify response."""

    def test_clarify_sends_confirm_request(self, vehicle, empty_grid, navigation, nl_client,
                                            schema_v, semantic_v, state_machine):
        msg = {"type": "nl_command", "seq": 7, "text": "开到那边去"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert state_machine.current_state == InstructionState.CONFIRMING
        assert len(replies) == 1
        assert replies[0]["type"] == "nl_confirm_request"
        assert "question" in replies[0]
        assert isinstance(replies[0]["missing"], list)


@pytest.mark.parametrize(
    ("text", "use_safety", "reason"),
    (
        ("去坐标 (10, 5.5)", False, "collision"),
        ("去坐标 (10, 5.5)", True, "safety_obstacle"),
    ),
)
def test_nl_automatic_command_rejects_terminal_handoff(
    text, use_safety, reason, nl_client, schema_v, semantic_v
):
    grid = MapGrid.from_wall_set(30, 30, {(4, y) for y in range(30)})
    vehicle = Vehicle(
        2.0,
        5.5,
        radius=0.5,
        linear_speed=5.0,
        command_timeout=10.0,
        now=0.0,
    )
    navigation = GotoController()
    safety = LocalSafetyRuntime() if use_safety else None
    local_state = AnchoredLocalState(
        AnchorSpec("nl-handoff", vehicle.x, vehicle.y, vehicle.yaw),
        truth_x_m=vehicle.x,
        truth_y_m=vehicle.y,
        truth_yaw_rad=vehicle.yaw,
        timestamp=0.0,
    )
    drive_ack = handle_command_message(
        '{"type":"drive","seq":90,"linear_mps":5,"angular_rps":0}',
        vehicle,
        grid,
        0.0,
        12.0,
        navigation,
        safety,
        local_state=local_state,
    )
    assert drive_ack["accepted"]

    replies = _handle_nl_command(
        {"type": "nl_command", "seq": 91, "text": text},
        vehicle,
        grid,
        navigation,
        13.0,
        1.0,
        nl_client,
        schema_v,
        semantic_v,
        InstructionStateMachine(),
        local_state=local_state,
        safety=safety,
    )

    assert replies[0]["accepted"] is False
    assert replies[0]["reason"] == reason
    assert replies[-1]["status"] == "blocked"
    assert replies[-1]["reason"] == reason
    assert (navigation.status, navigation.reason) == (
        "blocked",
        reason,
    )
    assert navigation.snapshot()["planning"] is False
    assert navigation.reported_goal is None
    assert navigation.reported_yaw_goal_rad is None
    assert vehicle.command == "stop"


@pytest.mark.parametrize(
    (
        "text",
        "use_safety",
        "response_type",
        "navigation_status",
        "navigation_reason",
    ),
    (
        ("开到那边去", False, "nl_confirm_request", "blocked", "collision"),
        ("停", False, "nl_task_update", "cancelled", "nl_stop"),
    ),
)
def test_nl_early_response_settles_terminal_goto_handoff(
    text,
    use_safety,
    response_type,
    navigation_status,
    navigation_reason,
    nl_client,
    schema_v,
):
    grid = MapGrid.from_wall_set(256, 256, {(4, y) for y in range(256)})
    vehicle = Vehicle(
        2.0,
        5.5,
        radius=0.5,
        linear_speed=5.0,
        command_timeout=10.0,
        now=0.0,
    )
    navigation = GotoController()
    local_state = AnchoredLocalState(
        AnchorSpec("nl-early-handoff", vehicle.x, vehicle.y, vehicle.yaw),
        truth_x_m=vehicle.x,
        truth_y_m=vehicle.y,
        truth_yaw_rad=vehicle.yaw,
        timestamp=0.0,
    )
    ack = handle_command_message(
        '{"type":"goto","seq":92,"x_m":250,"y_m":5.5}',
        vehicle,
        grid,
        0.0,
        12.0,
        navigation,
        local_state=local_state,
    )
    assert ack["accepted"]
    assert navigation.snapshot()["planning"] is True
    vehicle.install_drive(5.0, 0.0, 0.0)

    replies = _handle_nl_command(
        {"type": "nl_command", "seq": 93, "text": text},
        vehicle,
        grid,
        navigation,
        13.0,
        1.0,
        nl_client,
        schema_v,
        SemanticValidator(grid),
        InstructionStateMachine(),
        local_state=local_state,
        safety=LocalSafetyRuntime() if use_safety else None,
    )

    assert any(reply["type"] == response_type for reply in replies)
    assert not any(reply.get("status") == "blocked" for reply in replies)
    assert (navigation.status, navigation.reason) == (
        navigation_status,
        navigation_reason,
    )
    assert navigation.snapshot()["planning"] is False
    assert vehicle.command == "stop"

    navigation.replan(local_state.pose, None, local_state.local_map)
    navigation.update(
        vehicle,
        grid,
        1.1,
        pose=local_state.pose,
        local_map=local_state.local_map,
    )
    assert (navigation.status, navigation.reason) == (
        navigation_status,
        navigation_reason,
    )
    assert navigation.snapshot()["planning"] is False


class TestNlCommandPatrol:
    """NL '开始巡逻' → patrol started."""

    def test_patrol_starts(self, vehicle, empty_grid, navigation, nl_client,
                            schema_v, semantic_v, state_machine):
        msg = {"type": "nl_command", "seq": 20, "text": "开始巡逻"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert state_machine.current_state == InstructionState.ACTIVE
        task_updates = [r for r in replies if r["type"] == "nl_task_update"]
        assert len(task_updates) == 1
        assert task_updates[0]["status"] == "active"
        assert "patrol" in str(task_updates[0].get("reason", ""))

    def test_sequence_goto_then_stop_clears_remaining_patrol(
        self,
        vehicle,
        empty_grid,
        navigation,
        schema_v,
        semantic_v,
        state_machine,
        estimated_nl_runtime,
    ):
        class SequenceClient:
            def parse(self, _text):
                return [
                    {"intent": "goto", "parameters": {"x_m": 20.0, "y_m": 20.0}},
                    {"intent": "stop", "parameters": {}},
                    {"intent": "patrol", "parameters": {}},
                ]

        class CaptureSocket:
            def __init__(self):
                self.messages = []

            async def send(self, payload):
                self.messages.append(json.loads(payload))

        replies = _handle_nl_command(
            {"type": "nl_command", "seq": 21, "text": "执行序列"},
            vehicle,
            empty_grid,
            navigation,
            time.time(),
            time.monotonic(),
            SequenceClient(),
            schema_v,
            semantic_v,
            state_machine,
            local_state=estimated_nl_runtime,
        )
        assert any(reply.get("status") == "active" for reply in replies)
        assert state_machine.has_more()

        state_machine.transition(InstructionState.COMPLETED)
        state_machine.transition(InstructionState.IDLE)
        socket = CaptureSocket()
        asyncio.run(
            _process_next_in_queue(
                socket,
                vehicle,
                empty_grid,
                navigation,
                time.time(),
                time.monotonic(),
                schema_v,
                semantic_v,
                state_machine,
                local_state=estimated_nl_runtime,
            )
        )

        assert any(
            message.get("status") == "completed"
            and message.get("reason") == "vehicle stopped"
            for message in socket.messages
        )
        assert state_machine.current_state == InstructionState.IDLE
        assert not state_machine.has_more()

    def test_sequence_validation_failure_clears_remaining_instructions(
        self,
        vehicle,
        empty_grid,
        navigation,
        schema_v,
        semantic_v,
        state_machine,
    ):
        class InvalidSequenceClient:
            def parse(self, _text):
                return [
                    {"intent": "goto_point", "parameters": {"x_m": 20.0, "y_m": 20.0}},
                    {"intent": "patrol", "parameters": {}},
                ]

        replies = _handle_nl_command(
            {"type": "nl_command", "seq": 22, "text": "无效序列"},
            vehicle,
            empty_grid,
            navigation,
            time.time(),
            time.monotonic(),
            InvalidSequenceClient(),
            schema_v,
            semantic_v,
            state_machine,
        )

        assert replies[0]["accepted"] is False
        assert "schema validation failed" in replies[0]["reason"]
        assert state_machine.current_state == InstructionState.IDLE
        assert not state_machine.has_more()


class TestSafetyBlock:
    """Safety block during agent task → agent blocked."""

    def test_safety_block_transitions_to_blocked(self, vehicle, empty_grid, navigation,
                                                  nl_client, schema_v, semantic_v,
                                                  state_machine):
        # Start an NL goto task
        msg = {"type": "nl_command", "seq": 12, "text": "去坐标 (50, 50)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert state_machine.current_state == InstructionState.ACTIVE

        # Simulate safety block by setting navigation status
        navigation.status = "blocked"
        navigation.reason = "collision"

        # The telemetry loop would detect this and transition
        # We simulate that here
        assert navigation.status == "blocked"
        state_machine.transition(InstructionState.BLOCKED)
        assert state_machine.current_state == InstructionState.BLOCKED

        # Should be able to go back to IDLE
        state_machine.transition(InstructionState.IDLE)
        assert state_machine.current_state == InstructionState.IDLE


class TestTaskCompletion:
    """Agent task completion → COMPLETED state."""

    def test_task_reached_completes(self, vehicle, empty_grid, navigation,
                                     nl_client, schema_v, semantic_v,
                                     state_machine):
        # Start goto
        msg = {"type": "nl_command", "seq": 13, "text": "去坐标 (50, 50)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert state_machine.current_state == InstructionState.ACTIVE

        # Simulate navigation reaching goal
        navigation.status = "reached"
        navigation.reason = "goal_tolerance"

        # Telemetry loop would detect and transition
        state_machine.transition(InstructionState.COMPLETED)
        assert state_machine.current_state == InstructionState.COMPLETED
        state_machine.transition(InstructionState.IDLE)
        assert state_machine.current_state == InstructionState.IDLE

    @pytest.mark.parametrize(
        ("mode", "goal_mode", "reason", "expected"),
        (
            ("position", "exact", "goal_tolerance", "goal_reached"),
            ("rotation", None, "yaw_tolerance", "goal_reached"),
            (
                "position",
                "nearby_safe",
                "nearby_safe_stop",
                "nearby_safe_stop",
            ),
        ),
    )
    def test_nl_completion_reason_preserves_exact_and_marks_nearby(
        self,
        mode,
        goal_mode,
        reason,
        expected,
    ):
        navigation = GotoController()
        navigation.mode = mode
        navigation.goal_mode = goal_mode
        navigation.reason = reason

        assert _nl_completion_reason(navigation) == expected


# ═══════════════════════════════════════════════════════════════
# Validation failure tests
# ═══════════════════════════════════════════════════════════════

class TestValidationFailures:
    """Tests for validation rejection paths."""

    def test_goto_wall_rejected(self, vehicle, navigation, nl_client,
                                 schema_v, state_machine):
        """Going to a wall cell should be rejected by semantic validation."""
        grid = MapGrid(256, 256)
        grid.set_cell(50, 50, 1)  # WALL = 1
        semantic_v = SemanticValidator(grid)

        msg = {"type": "nl_command", "seq": 14, "text": "去坐标 (50, 50)"}
        replies = _handle_nl_command(
            msg, vehicle, grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        # Should be rejected
        parse_results = [r for r in replies if r["type"] == "nl_parse_result"]
        assert len(parse_results) == 1
        assert parse_results[0]["accepted"] is False
        assert "wall" in str(parse_results[0].get("reason", "")).lower()
        assert state_machine.current_state == InstructionState.IDLE

    def test_empty_text_rejected(self, vehicle, empty_grid, navigation, nl_client,
                                  schema_v, semantic_v, state_machine):
        msg = {"type": "nl_command", "seq": 15, "text": ""}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert len(replies) == 1
        assert replies[0]["type"] == "nl_parse_result"
        assert replies[0]["accepted"] is False

    @pytest.mark.parametrize("seq", ("bad", True, -1, 2**64, 10**100))
    def test_invalid_sequence_is_normalized_in_every_reply(
        self,
        seq,
        vehicle,
        empty_grid,
        navigation,
        nl_client,
        schema_v,
        semantic_v,
    ):
        replies = _handle_nl_command(
            {"type": "nl_command", "seq": seq, "text": "前进 1 米"},
            vehicle,
            empty_grid,
            navigation,
            time.time(),
            time.monotonic(),
            nl_client,
            schema_v,
            semantic_v,
            InstructionStateMachine(),
        )
        assert replies
        assert {reply["seq"] for reply in replies} == {0}

    @pytest.mark.parametrize("seq", ("bad", True, -1, 2**64, 10**100))
    def test_invalid_sequence_stays_normalized_on_async_completion(self, seq):
        clock = _Clock()
        socket = _InvalidSeqSocket(clock, seq)
        runtime = VehicleRuntime.create(
            started_at=0.0,
            anchor=AnchorSpec("nl-seq", 10.0, 10.0, 0.0),
            odometry_config=OdometryConfig(),
        )

        asyncio.run(
            handler(
                socket,
                _runtime=runtime,
                _monotonic=clock.monotonic,
                _wall_time=lambda: 10.0 + clock.now,
                _nl_client=_AsyncTestParser(),
            )
        )

        task_updates = [
            message
            for message in socket.messages
            if message.get("type") == "nl_task_update"
        ]
        assert task_updates
        assert {message["seq"] for message in task_updates} == {0}

    @pytest.mark.parametrize(
        "second_type", ("nl_command", "nl_clarify_response")
    )
    @pytest.mark.parametrize("outcome", ("completed", "blocked"))
    def test_busy_request_does_not_take_active_task_sequence(
        self, second_type, outcome
    ):
        clock = _Clock()
        socket = _ActiveOwnerSocket(clock, second_type, outcome)
        runtime = VehicleRuntime.create(
            started_at=0.0,
            anchor=AnchorSpec("nl-owner", 10.0, 10.0, 0.0),
            odometry_config=OdometryConfig(),
        )
        socket.runtime = runtime

        asyncio.run(
            handler(
                socket,
                _runtime=runtime,
                _monotonic=clock.monotonic,
                _wall_time=lambda: 10.0 + clock.now,
                _nl_client=_TestParser(),
            )
        )

        busy = [
            message
            for message in socket.messages
            if message.get("type") == "nl_parse_result"
            and not message.get("accepted")
        ]
        terminal = [
            message
            for message in socket.messages
            if message.get("type") == "nl_task_update"
            and message.get("status") == outcome
        ]
        assert busy and busy[-1]["seq"] == 2
        assert "busy" in busy[-1]["reason"]
        assert terminal and terminal[-1]["seq"] == 1
        if outcome == "completed":
            assert terminal[-1]["reason"] == "goal_reached"

    def test_busy_state_rejected(self, vehicle, empty_grid, navigation, nl_client,
                                  schema_v, semantic_v, state_machine):
        """Cannot send nl_command while PARSING/VALIDATING."""
        # Force state to PARSING
        state_machine.transition(InstructionState.PARSING)

        msg = {"type": "nl_command", "seq": 16, "text": "停"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert replies[0]["accepted"] is False
        assert "busy" in str(replies[0].get("reason", "")).lower()


# ═══════════════════════════════════════════════════════════════
# Full pipeline integration tests (parse→validate→task)
# ═══════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end: nl text → parsed → validated → task executed."""

    def test_pipeline_stop(self, vehicle, empty_grid, navigation,
                           nl_client, schema_v, semantic_v, state_machine):
        now = time.monotonic()
        vehicle.advance(empty_grid, now)
        vehicle.install_command("forward", now)
        msg = {"type": "nl_command", "seq": 20, "text": "停"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert vehicle.command == "stop"
        assert state_machine.current_state == InstructionState.IDLE
        assert any(r["type"] == "nl_task_update" and r["status"] == "completed" for r in replies)

    def test_pipeline_clarify_then_goto(self, vehicle, empty_grid, navigation,
                                         nl_client, schema_v, semantic_v, state_machine):
        # First: clarify
        msg1 = {"type": "nl_command", "seq": 22, "text": "开到那边去"}
        _handle_nl_command(
            msg1, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert state_machine.current_state == InstructionState.CONFIRMING

        # Second: provide real command (from CONFIRMING, a new nl_command is allowed)
        msg2 = {"type": "nl_command", "seq": 23, "text": "去坐标 (30, 40)"}
        _handle_nl_command(
            msg2, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert navigation.reported_goal == (30.0, 40.0)
        assert navigation.status == "active"
        assert state_machine.current_state == InstructionState.ACTIVE

    def test_pipeline_patrol(self, vehicle, empty_grid, navigation,
                              nl_client, schema_v, semantic_v, state_machine):
        msg = {"type": "nl_command", "seq": 24, "text": "开始巡逻"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert state_machine.current_state == InstructionState.ACTIVE
        assert any(r["type"] == "nl_task_update" and r["status"] == "active" for r in replies)
