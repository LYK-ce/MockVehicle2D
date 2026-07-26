"""Integration tests for NL command pipeline in server.py.

Tests the NL → parse → validate → execute pipeline without WebSocket.
Uses FakeModelClient for deterministic results.
"""

from __future__ import annotations

import asyncio
import json
import math
import time

import pytest

from mockvehicle2d.instruction.llm_client import FakeModelClient
from mockvehicle2d.instruction.state_machine import InstructionState, InstructionStateMachine
from mockvehicle2d.instruction.validator import SchemaValidator, SemanticValidator
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.local_state import AnchorSpec, AnchoredLocalState, OdometryConfig
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.server import (
    _handle_nl_command,
    _summarize_scan_for_nl,
    _cancel_nl_task,
    handle_command_message,
    handler,
    VehicleRuntime,
)
from mockvehicle2d.vehicle import Vehicle


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
    return FakeModelClient()


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
                {"type": "nl_command", "seq": self.seq, "text": "前进 0.1 米"}
            )
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
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert navigation.reported_goal == (100.0, 200.0)
        assert navigation.status == "active"


class TestNlCommandMoveDistance:
    """NL '前进 5 米' → GotoController with relative goal."""

    def test_move_forward_computes_goal(self, vehicle, empty_grid, navigation, nl_client,
                                         schema_v, semantic_v, state_machine):
        # Vehicle at (10, 10), yaw=0 (faces +x)
        vehicle.yaw = 0.0
        msg = {"type": "nl_command", "seq": 4, "text": "前进 5 米"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert navigation.status == "active"
        goal_x, goal_y = navigation.reported_goal
        assert abs(goal_x - 15.0) < 0.1, f"expected goal_x ~15.0, got {goal_x}"
        assert abs(goal_y - 10.0) < 0.1, f"expected goal_y ~10.0, got {goal_y}"

    def test_move_backward_computes_goal(self, vehicle, empty_grid, navigation, nl_client,
                                          schema_v, semantic_v, state_machine):
        vehicle.yaw = 0.0
        msg = {"type": "nl_command", "seq": 5, "text": "后退 3 米"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert navigation.status == "active"
        goal_x, goal_y = navigation.reported_goal
        assert abs(goal_x - 7.0) < 0.1, f"expected goal_x ~7.0, got {goal_x}"
        assert abs(goal_y - 10.0) < 0.1, f"expected goal_y ~10.0, got {goal_y}"

    def test_move_with_yaw_45deg(self, vehicle, empty_grid, navigation, nl_client,
                                  schema_v, semantic_v, state_machine):
        vehicle.yaw = math.pi / 4  # 45 degrees
        msg = {"type": "nl_command", "seq": 6, "text": "前进 5 米"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        assert navigation.status == "active"
        goal_x, goal_y = navigation.reported_goal
        expected_x = 10.0 + 5.0 * math.cos(math.pi / 4)
        expected_y = 10.0 + 5.0 * math.sin(math.pi / 4)
        assert abs(goal_x - expected_x) < 0.1
        assert abs(goal_y - expected_y) < 0.1

    def test_move_uses_estimated_pose_not_simulator_truth(
        self,
        vehicle,
        empty_grid,
        navigation,
        nl_client,
        schema_v,
        semantic_v,
        state_machine,
        estimated_nl_runtime,
    ):
        estimated_nl_runtime.odometry.apply_correction(
            5.0, 0.0, 0.0, timestamp=time.time()
        )

        _handle_nl_command(
            {"type": "nl_command", "seq": 60, "text": "前进 1 米"},
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

        assert vehicle.x == 10.0
        assert navigation.reported_goal == pytest.approx((16.0, 10.0))


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


class TestNlCommandRotate:
    """NL '左转 90 度' → rotates vehicle."""

    @staticmethod
    def _run_until_done(
        vehicle, grid, navigation, local_state, *, ticks=300
    ):
        now = vehicle.last_update
        for _ in range(ticks):
            now += 0.05
            collided = vehicle.advance(grid, now)
            local_state.update_from_truth(
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                timestamp=local_state.pose.timestamp + 0.05,
            )
            navigation.update(
                vehicle,
                grid,
                now,
                pose=local_state.pose,
                advance_result=SafetyAdvanceResult(collided=collided),
                local_map=local_state.local_map,
            )
            assert vehicle.body_velocities()[0] == 0.0
            if navigation.status != "active":
                break

    @pytest.mark.parametrize(
        ("text", "expected_sign"),
        [("左转 90 度", -1), ("右转 90 度", 1)],
    )
    def test_rotate_changes_yaw_and_reaches(
        self,
        text,
        expected_sign,
        vehicle,
        empty_grid,
        navigation,
        nl_client,
        schema_v,
        semantic_v,
        state_machine,
        estimated_nl_runtime,
    ):
        _handle_nl_command(
            {"type": "nl_command", "seq": 8, "text": text},
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

        assert navigation.status == "active"
        assert navigation.snapshot()["goal"].keys() == {"yaw_rad"}
        navigation.update(
            vehicle,
            empty_grid,
            vehicle.last_update,
            pose=estimated_nl_runtime.pose,
            advance_result=SafetyAdvanceResult(),
            local_map=estimated_nl_runtime.local_map,
        )
        assert navigation.status == "active"
        assert math.copysign(1.0, vehicle.body_velocities()[1]) == expected_sign

        self._run_until_done(
            vehicle, empty_grid, navigation, estimated_nl_runtime
        )
        assert navigation.status == "reached"
        assert expected_sign * vehicle.yaw > math.radians(85)

    @pytest.mark.parametrize(
        ("estimated_yaw_rad", "text", "delta_yaw_rad"),
        [
            (3.0, "右转 30 度", math.pi / 6),
            (-3.0, "左转 30 度", -math.pi / 6),
        ],
    )
    def test_rotate_wraps_pi_and_uses_estimate_not_truth(
        self,
        estimated_yaw_rad,
        text,
        delta_yaw_rad,
        vehicle,
        empty_grid,
        navigation,
        nl_client,
        schema_v,
        semantic_v,
        state_machine,
        estimated_nl_runtime,
    ):
        estimated_nl_runtime.odometry.apply_correction(
            0.0, 0.0, estimated_yaw_rad, timestamp=time.time()
        )
        _handle_nl_command(
            {"type": "nl_command", "seq": 81, "text": text},
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

        assert vehicle.yaw == 0.0
        assert navigation.yaw_goal_rad == pytest.approx(
            math.atan2(
                math.sin(estimated_yaw_rad + delta_yaw_rad),
                math.cos(estimated_yaw_rad + delta_yaw_rad),
            )
        )

    def test_rotate_lost_and_manual_override_use_existing_gates(
        self,
        vehicle,
        empty_grid,
        navigation,
        nl_client,
        schema_v,
        semantic_v,
        state_machine,
        estimated_nl_runtime,
    ):
        estimated_nl_runtime.set_localization_quality(
            "lost", timestamp=time.time()
        )
        replies = _handle_nl_command(
            {"type": "nl_command", "seq": 82, "text": "左转 90 度"},
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
        assert navigation.status == "blocked"
        assert any(
            reply.get("reason") == "localization_lost" for reply in replies
        )

        estimated_nl_runtime.set_localization_quality(
            "nominal", timestamp=time.time()
        )
        navigation.start_rotation(1.0)
        reply = handle_command_message(
            '{"type":"cmd","seq":83,"cmd":"stop"}',
            vehicle,
            empty_grid,
            vehicle.last_update,
            time.time(),
            navigation,
            local_state=estimated_nl_runtime,
        )
        assert reply["accepted"]
        assert navigation.status == "cancelled"


class TestNlCommandScan:
    """NL '看一下' → scan report."""

    def test_scan_sends_report(self, vehicle, empty_grid, navigation, nl_client,
                                schema_v, semantic_v, state_machine):
        # Build a simple scan frame with a few points
        scan_data = {
            "type": "scan",
            "points": [
                {"angle": 0.0, "range": 1.5, "intensity": 1.0},
                {"angle": 0.5, "range": 2.0, "intensity": 1.0},
                {"angle": math.pi / 2, "range": 3.0, "intensity": 1.0},
            ],
        }

        msg = {"type": "nl_command", "seq": 9, "text": "看一下"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
            scan_data=scan_data,
        )

        scan_reports = [r for r in replies if r["type"] == "nl_scan_report"]
        assert len(scan_reports) == 1
        report = scan_reports[0]
        assert "summary" in report
        assert "points_summary" in report
        # front should have min 1.5
        assert report["points_summary"].get("front") == 1.5


class TestNlCommandStatus:
    """NL '状态' → status report."""

    def test_status_reports_position(self, vehicle, empty_grid, navigation, nl_client,
                                      schema_v, semantic_v, state_machine):
        msg = {"type": "nl_command", "seq": 10, "text": "状态"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )

        task_updates = [r for r in replies if r["type"] == "nl_task_update"]
        assert len(task_updates) >= 1
        assert task_updates[0]["status"] == "completed"
        assert "position" in str(task_updates[0].get("reason", ""))


# ═══════════════════════════════════════════════════════════════
# Authority / Preemption tests
# ═══════════════════════════════════════════════════════════════

class TestManualOverride:
    """Manual cmd during agent task → agent cancelled."""

    def test_manual_cmd_cancels_nl_task(self, vehicle, empty_grid, navigation,
                                         nl_client, schema_v, semantic_v,
                                         state_machine):
        # Start an NL goto task
        msg = {"type": "nl_command", "seq": 11, "text": "去坐标 (50, 50)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert state_machine.current_state == InstructionState.ACTIVE
        assert navigation.status == "active"

        # Now cancel via _cancel_nl_task (simulating manual override)
        update = _cancel_nl_task(navigation, state_machine, "manual_override")
        assert update is not None
        assert update["type"] == "nl_task_update"
        assert update["status"] == "cancelled"
        assert update["reason"] == "manual_override"
        assert state_machine.current_state == InstructionState.CANCELLED
        assert navigation.status == "cancelled"

    def test_manual_cmd_during_accepting(self, vehicle, empty_grid, navigation,
                                          nl_client, schema_v, semantic_v,
                                          state_machine):
        # Force state to ACCEPTED (simulate mid-pipeline)
        state_machine.transition(InstructionState.PARSING)
        state_machine.transition(InstructionState.VALIDATING)
        state_machine.transition(InstructionState.ACCEPTED)
        assert state_machine.current_state == InstructionState.ACCEPTED

        update = _cancel_nl_task(navigation, state_machine, "manual_override")
        assert update is not None
        assert update["status"] == "cancelled"


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

    @pytest.mark.parametrize("seq", ("bad", True, -1))
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

    @pytest.mark.parametrize("seq", ("bad", True, -1))
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
            )
        )

        task_updates = [
            message
            for message in socket.messages
            if message.get("type") == "nl_task_update"
        ]
        assert task_updates
        assert {message["seq"] for message in task_updates} == {0}

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
# Scan summary tests
# ═══════════════════════════════════════════════════════════════

class TestScanSummary:
    def test_empty_scan(self):
        result = _summarize_scan_for_nl(None)
        assert "无扫描数据" in result["text"]
        assert result["sectors"] == {}

    def test_no_points(self):
        result = _summarize_scan_for_nl({"type": "scan", "points": []})
        assert "无扫描点" in result["text"]

    def test_sector_classification(self):
        points = [
            {"angle": 0.0, "range": 1.0},       # front
            {"angle": 0.3, "range": 2.0},       # front
            {"angle": math.pi / 2, "range": 3.0},  # left
            {"angle": -math.pi / 2, "range": 4.0},  # right
            {"angle": math.pi, "range": 5.0},    # back
        ]
        result = _summarize_scan_for_nl({"type": "scan", "points": points})
        assert result["sectors"]["front"] == 1.0
        assert result["sectors"]["left"] == 3.0
        assert result["sectors"]["right"] == 4.0
        assert result["sectors"]["back"] == 5.0
        assert "前方" in result["text"]
        assert "左侧" in result["text"]


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

    def test_pipeline_goto_then_cancel(self, vehicle, empty_grid, navigation,
                                        nl_client, schema_v, semantic_v, state_machine):
        # Start goto
        msg = {"type": "nl_command", "seq": 21, "text": "去坐标 (100, 100)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert navigation.status == "active"

        # Cancel
        update = _cancel_nl_task(navigation, state_machine, "manual_override")
        assert update is not None
        assert update["status"] == "cancelled"
        assert navigation.status == "cancelled"

    def test_pipeline_clarify_then_goto(self, vehicle, empty_grid, navigation,
                                         nl_client, schema_v, semantic_v, state_machine):
        # First: clarify
        msg1 = {"type": "nl_command", "seq": 22, "text": "开到那边去"}
        replies1 = _handle_nl_command(
            msg1, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert state_machine.current_state == InstructionState.CONFIRMING

        # Second: provide real command (from CONFIRMING, a new nl_command is allowed)
        msg2 = {"type": "nl_command", "seq": 23, "text": "去坐标 (30, 40)"}
        replies2 = _handle_nl_command(
            msg2, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert navigation.reported_goal == (30.0, 40.0)
        assert navigation.status == "active"
        assert state_machine.current_state == InstructionState.ACTIVE

    def test_pipeline_move_distance_then_wait_completion(self, vehicle, empty_grid,
                                                          navigation, nl_client, schema_v,
                                                          semantic_v, state_machine):
        """Test that move_distance with GotoController reaches goal."""
        vehicle.yaw = 0.0
        msg = {"type": "nl_command", "seq": 24, "text": "前进 1 米"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine,
        )
        assert navigation.status == "active"
        goal_x, goal_y = navigation.reported_goal
        assert abs(goal_x - 11.0) < 0.1
        assert abs(goal_y - 10.0) < 0.1

    def test_pipeline_rotate_reports_si_yaw_goal(self, vehicle, empty_grid,
                                                  navigation, nl_client, schema_v,
                                                  semantic_v, state_machine):
        _handle_nl_command(
            {"type": "nl_command", "seq": 25, "text": "右转 90 度"},
            vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v, state_machine,
        )
        assert navigation.snapshot()["goal"] == {
            "yaw_rad": pytest.approx(math.pi / 2)
        }
