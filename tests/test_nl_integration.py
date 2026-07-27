"""Integration tests for NL command pipeline in server.py.

Tests the NL → parse → validate → execute pipeline without WebSocket.
Uses a deterministic test parser for reproducible results.
"""

from __future__ import annotations

import math
import time

import pytest

from mockvehicle2d.instruction.compiler import TaskCompiler
from mockvehicle2d.instruction.state_machine import InstructionState, InstructionStateMachine
from mockvehicle2d.instruction.validator import SchemaValidator, SemanticValidator
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.safety import LocalSafetyRuntime
from mockvehicle2d.server import (
    _handle_nl_command,
    _summarize_scan_for_nl,
    _cancel_nl_task,
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

    def parse(self, text: str) -> dict | None:
        text = text.strip()
        if not text:
            return {"intent": "clarify", "parameters": {"question": "请输入指令"}}

        if text in self._STOP_WORDS:
            return {"intent": "stop", "parameters": {}}

        if text in self._PATROL_WORDS:
            return {"intent": "patrol", "parameters": {}}

        if text in self._CLARIFY_WORDS:
            return {"intent": "clarify", "parameters": {"question": "请指定坐标"}}

        # goto patterns
        m = self._GOTO_PAT.search(text)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            return {"intent": "goto", "parameters": {"x_m": x, "y_m": y}}

        return {"intent": "clarify", "parameters": {"question": "请指定坐标"}}


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def empty_grid():
    return MapGrid(256, 256)


@pytest.fixture
def vehicle():
    return Vehicle(10.0, 10.0, now=time.monotonic())


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


@pytest.fixture
def task_compiler():
    return TaskCompiler()


# ═══════════════════════════════════════════════════════════════
# NL command execution tests
# ═══════════════════════════════════════════════════════════════

class TestNlCommandStop:
    """NL '停' → vehicle stops."""

    def test_stop_stops_vehicle(self, vehicle, empty_grid, navigation, nl_client,
                                 schema_v, semantic_v, state_machine, task_compiler):
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
            state_machine, task_compiler,
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


class TestNlCommandGoto:
    """NL '去坐标 (50, 50)' → GotoController.start(50, 50)."""

    def test_goto_starts_navigation(self, vehicle, empty_grid, navigation, nl_client,
                                     schema_v, semantic_v, state_machine, task_compiler):
        msg = {"type": "nl_command", "seq": 2, "text": "去坐标 (50, 50)"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )

        assert navigation.status == "active"
        assert navigation.goal == (50.0, 50.0)
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
                                 schema_v, semantic_v, state_machine, task_compiler):
        msg = {"type": "nl_command", "seq": 3, "text": "去 (100, 200)"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )

        assert navigation.goal == (100.0, 200.0)
        assert navigation.status == "active"


class TestNlCommandClarify:
    """NL '开到那边去' → clarify response."""

    def test_clarify_sends_confirm_request(self, vehicle, empty_grid, navigation, nl_client,
                                            schema_v, semantic_v, state_machine, task_compiler):
        msg = {"type": "nl_command", "seq": 7, "text": "开到那边去"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )

        assert state_machine.current_state == InstructionState.CONFIRMING
        assert len(replies) == 1
        assert replies[0]["type"] == "nl_confirm_request"
        assert "question" in replies[0]
        assert isinstance(replies[0]["missing"], list)


class TestNlCommandPatrol:
    """NL '开始巡逻' → patrol started."""

    def test_patrol_starts(self, vehicle, empty_grid, navigation, nl_client,
                            schema_v, semantic_v, state_machine, task_compiler):
        msg = {"type": "nl_command", "seq": 20, "text": "开始巡逻"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )

        assert state_machine.current_state == InstructionState.ACTIVE
        task_updates = [r for r in replies if r["type"] == "nl_task_update"]
        assert len(task_updates) == 1
        assert task_updates[0]["status"] == "active"
        assert "patrol" in str(task_updates[0].get("reason", ""))


# ═══════════════════════════════════════════════════════════════
# Authority / Preemption tests
# ═══════════════════════════════════════════════════════════════

class TestManualOverride:
    """Manual cmd during agent task → agent cancelled."""

    def test_manual_cmd_cancels_nl_task(self, vehicle, empty_grid, navigation,
                                         nl_client, schema_v, semantic_v,
                                         state_machine, task_compiler):
        # Start an NL goto task
        msg = {"type": "nl_command", "seq": 11, "text": "去坐标 (50, 50)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
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
                                          state_machine, task_compiler):
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
                                                  state_machine, task_compiler):
        # Start an NL goto task
        msg = {"type": "nl_command", "seq": 12, "text": "去坐标 (50, 50)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
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
                                     state_machine, task_compiler):
        # Start goto
        msg = {"type": "nl_command", "seq": 13, "text": "去坐标 (50, 50)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
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
                                 schema_v, state_machine, task_compiler):
        """Going to a wall cell should be rejected by semantic validation."""
        grid = MapGrid(256, 256)
        grid.set_cell(50, 50, 1)  # WALL = 1
        semantic_v = SemanticValidator(grid)

        msg = {"type": "nl_command", "seq": 14, "text": "去坐标 (50, 50)"}
        replies = _handle_nl_command(
            msg, vehicle, grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )

        # Should be rejected
        parse_results = [r for r in replies if r["type"] == "nl_parse_result"]
        assert len(parse_results) == 1
        assert parse_results[0]["accepted"] is False
        assert "wall" in str(parse_results[0].get("reason", "")).lower()
        assert state_machine.current_state == InstructionState.IDLE

    def test_empty_text_rejected(self, vehicle, empty_grid, navigation, nl_client,
                                  schema_v, semantic_v, state_machine, task_compiler):
        msg = {"type": "nl_command", "seq": 15, "text": ""}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )

        assert len(replies) == 1
        assert replies[0]["type"] == "nl_parse_result"
        assert replies[0]["accepted"] is False

    def test_busy_state_rejected(self, vehicle, empty_grid, navigation, nl_client,
                                  schema_v, semantic_v, state_machine, task_compiler):
        """Cannot send nl_command while PARSING/VALIDATING."""
        # Force state to PARSING
        state_machine.transition(InstructionState.PARSING)

        msg = {"type": "nl_command", "seq": 16, "text": "停"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
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
                           nl_client, schema_v, semantic_v, state_machine, task_compiler):
        now = time.monotonic()
        vehicle.advance(empty_grid, now)
        vehicle.install_command("forward", now)
        msg = {"type": "nl_command", "seq": 20, "text": "停"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )
        assert vehicle.command == "stop"
        assert state_machine.current_state == InstructionState.IDLE
        assert any(r["type"] == "nl_task_update" and r["status"] == "completed" for r in replies)

    def test_pipeline_goto_then_cancel(self, vehicle, empty_grid, navigation,
                                        nl_client, schema_v, semantic_v, state_machine, task_compiler):
        # Start goto
        msg = {"type": "nl_command", "seq": 21, "text": "去坐标 (100, 100)"}
        _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )
        assert navigation.status == "active"

        # Cancel
        update = _cancel_nl_task(navigation, state_machine, "manual_override")
        assert update is not None
        assert update["status"] == "cancelled"
        assert navigation.status == "cancelled"

    def test_pipeline_clarify_then_goto(self, vehicle, empty_grid, navigation,
                                         nl_client, schema_v, semantic_v, state_machine, task_compiler):
        # First: clarify
        msg1 = {"type": "nl_command", "seq": 22, "text": "开到那边去"}
        replies1 = _handle_nl_command(
            msg1, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )
        assert state_machine.current_state == InstructionState.CONFIRMING

        # Second: provide real command (from CONFIRMING, a new nl_command is allowed)
        msg2 = {"type": "nl_command", "seq": 23, "text": "去坐标 (30, 40)"}
        replies2 = _handle_nl_command(
            msg2, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )
        assert navigation.goal == (30.0, 40.0)
        assert navigation.status == "active"
        assert state_machine.current_state == InstructionState.ACTIVE

    def test_pipeline_patrol(self, vehicle, empty_grid, navigation,
                              nl_client, schema_v, semantic_v, state_machine, task_compiler):
        msg = {"type": "nl_command", "seq": 24, "text": "开始巡逻"}
        replies = _handle_nl_command(
            msg, vehicle, empty_grid, navigation,
            time.time(), time.monotonic(),
            nl_client, schema_v, semantic_v,
            state_machine, task_compiler,
        )
        assert state_machine.current_state == InstructionState.ACTIVE
        assert any(r["type"] == "nl_task_update" and r["status"] == "active" for r in replies)
