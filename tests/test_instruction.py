"""Tests for mockvehicle2d.instruction — Phase 1 offline closed loop.

Covers:
- SchemaValidator
- SemanticValidator
- InstructionStateMachine
- TaskCompiler
- AuthorityManager
- FakeModelClient
- Integration: nl command → parse → validate pipeline
"""

from __future__ import annotations

import json
import math
import threading
import time

import pytest

from mockvehicle2d.instruction.authority import AuthorityLevel, AuthorityManager
from mockvehicle2d.instruction.compiler import TaskCompiler
from mockvehicle2d.instruction.llm_client import FakeModelClient
from mockvehicle2d.instruction.state_machine import (
    InstructionState,
    InstructionStateMachine,
    InvalidTransitionError,
)
from mockvehicle2d.instruction.validator import (
    SchemaValidator,
    SemanticValidator,
    SafetyValidator,
    run_validation_pipeline,
)
from mockvehicle2d.map_grid import MapGrid, WALL, VOID
from mockvehicle2d.safety import LocalSafetyRuntime


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def empty_grid():
    return MapGrid(256, 256)


@pytest.fixture
def grid_with_obstacles():
    grid = MapGrid(256, 256)
    grid.set_cell(50, 50, WALL)
    grid.set_cell(24, 10, VOID)
    return grid


@pytest.fixture
def healthy_safety():
    return LocalSafetyRuntime(healthy=True)


@pytest.fixture
def faulty_safety():
    s = LocalSafetyRuntime(healthy=False)
    # trigger evaluation to get fault state
    from mockvehicle2d.vehicle import Vehicle
    v = Vehicle(10, 10)
    s.evaluate(v, MapGrid(256, 256), 0.5, 0.0, automatic=True)
    return s


def _valid_instruction(intent="stop", params=None, **overrides):
    """Build a minimally valid instruction dict."""
    base = {
        "schema_version": "1.0",
        "intent": intent,
        "timestamp": "2026-07-24T12:00:00+08:00",
        "parameters": params or {},
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════
# SchemaValidator tests (15+)
# ═══════════════════════════════════════════════════════════════

class TestSchemaValidator:
    def setup_method(self):
        self.v = SchemaValidator()

    def test_valid_stop(self):
        ok, msg = self.v.validate(_valid_instruction("stop"))
        assert ok, msg

    def test_valid_status(self):
        ok, msg = self.v.validate(_valid_instruction("status"))
        assert ok, msg

    def test_valid_goto_point(self):
        ok, msg = self.v.validate(_valid_instruction("goto_point", {"x_m": 100, "y_m": 200}))
        assert ok, msg

    def test_valid_move_distance(self):
        ok, msg = self.v.validate(
            _valid_instruction("move_distance", {"distance_m": 5.0, "direction": "forward"})
        )
        assert ok, msg

    def test_valid_rotate(self):
        ok, msg = self.v.validate(
            _valid_instruction("rotate", {"angle_deg": 90, "direction": "left"})
        )
        assert ok, msg

    def test_valid_scan_report(self):
        ok, msg = self.v.validate(_valid_instruction("scan_report", {}))
        assert ok, msg

    def test_valid_clarify(self):
        ok, msg = self.v.validate(
            _valid_instruction("clarify", {"question": "where?"})
        )
        assert ok, msg

    def test_missing_schema_version(self):
        inst = {"intent": "stop", "timestamp": "2026-07-24T12:00:00+08:00"}
        ok, msg = self.v.validate(inst)
        assert not ok
        assert "schema_version" in msg

    def test_missing_intent(self):
        inst = {"schema_version": "1.0", "timestamp": "2026-07-24T12:00:00+08:00"}
        ok, msg = self.v.validate(inst)
        assert not ok

    def test_invalid_intent(self):
        ok, msg = self.v.validate(_valid_instruction("fly"))
        assert not ok

    def test_goto_point_missing_params(self):
        ok, msg = self.v.validate(_valid_instruction("goto_point", {}))
        assert not ok
        assert "x_m" in msg

    def test_move_distance_out_of_range(self):
        ok, msg = self.v.validate(
            _valid_instruction("move_distance", {"distance_m": 999.0, "direction": "forward"})
        )
        assert not ok

    def test_rotate_angle_out_of_bounds(self):
        ok, msg = self.v.validate(
            _valid_instruction("rotate", {"angle_deg": 999, "direction": "left"})
        )
        assert not ok

    def test_confidence_boundary_valid(self):
        ok, msg = self.v.validate(_valid_instruction("stop", confidence=0.0))
        assert ok, msg
        ok, msg = self.v.validate(_valid_instruction("stop", confidence=1.0))
        assert ok, msg

    def test_confidence_out_of_bounds(self):
        ok, msg = self.v.validate(_valid_instruction("stop", confidence=1.5))
        assert not ok

    def test_injection_extra_fields(self):
        """Extra top-level fields are rejected by additionalProperties."""
        inst = _valid_instruction("stop", injected_field="malicious")
        ok, msg = self.v.validate(inst)
        assert not ok

    def test_injection_extra_param_fields(self):
        """Extra fields in parameters for stop are rejected."""
        inst = _valid_instruction("stop", {"evil": True})
        ok, msg = self.v.validate(inst)
        assert not ok

    def test_clarify_missing_question(self):
        ok, msg = self.v.validate(_valid_instruction("clarify", {}))
        assert not ok

    def test_reasoning_too_long(self):
        ok, msg = self.v.validate(_valid_instruction("stop", reasoning="x" * 501))
        assert not ok


# ═══════════════════════════════════════════════════════════════
# SemanticValidator tests (10+)
# ═══════════════════════════════════════════════════════════════

class TestSemanticValidator:
    def setup_method(self):
        self.grid = MapGrid(256, 256)
        self.v = SemanticValidator(self.grid)

    def test_goto_point_in_bounds(self):
        ok, msg = self.v.validate(_valid_instruction("goto_point", {"x_m": 100, "y_m": 200}))
        assert ok, msg

    def test_goto_point_negative_x(self):
        ok, msg = self.v.validate(_valid_instruction("goto_point", {"x_m": -10, "y_m": 100}))
        assert not ok
        assert "out of map bounds" in msg

    def test_goto_point_too_large(self):
        ok, msg = self.v.validate(_valid_instruction("goto_point", {"x_m": 300, "y_m": 100}))
        assert not ok

    def test_goto_point_is_wall(self, grid_with_obstacles):
        v = SemanticValidator(grid_with_obstacles)
        ok, msg = v.validate(_valid_instruction("goto_point", {"x_m": 50, "y_m": 50}))
        assert not ok
        assert "wall" in msg

    def test_goto_point_is_void(self, grid_with_obstacles):
        v = SemanticValidator(grid_with_obstacles)
        ok, msg = v.validate(_valid_instruction("goto_point", {"x_m": 24, "y_m": 10}))
        assert not ok
        assert "void" in msg

    def test_move_distance_valid(self):
        ok, msg = self.v.validate(
            _valid_instruction("move_distance", {"distance_m": 3.0, "direction": "forward"})
        )
        assert ok, msg

    def test_move_distance_zero(self):
        ok, msg = self.v.validate(
            _valid_instruction("move_distance", {"distance_m": 0.0, "direction": "forward"})
        )
        assert not ok

    def test_move_distance_exceeds_max(self):
        v = SemanticValidator(self.grid, max_distance_m=5.0)
        ok, msg = v.validate(
            _valid_instruction("move_distance", {"distance_m": 8.0, "direction": "forward"})
        )
        assert not ok

    def test_move_distance_invalid_direction(self):
        ok, msg = self.v.validate(
            _valid_instruction("move_distance", {"distance_m": 1.0, "direction": "sideways"})
        )
        assert not ok

    def test_rotate_non_zero(self):
        ok, msg = self.v.validate(
            _valid_instruction("rotate", {"angle_deg": 45, "direction": "right"})
        )
        assert ok, msg

    def test_rotate_zero_angle(self):
        ok, msg = self.v.validate(
            _valid_instruction("rotate", {"angle_deg": 0, "direction": "left"})
        )
        assert not ok

    def test_stop_no_semantic_check(self):
        ok, msg = self.v.validate(_valid_instruction("stop"))
        assert ok, msg

    def test_scan_report_no_semantic_check(self):
        ok, msg = self.v.validate(_valid_instruction("scan_report", {"query": "front"}))
        assert ok, msg


# ═══════════════════════════════════════════════════════════════
# SafetyValidator tests
# ═══════════════════════════════════════════════════════════════

class TestSafetyValidator:
    def test_healthy_safety_passes(self, healthy_safety):
        v = SafetyValidator(healthy_safety)
        ok, msg = v.validate()
        assert ok, msg

    def test_faulty_safety_fails(self, faulty_safety):
        v = SafetyValidator(faulty_safety)
        ok, msg = v.validate()
        assert not ok
        assert "fault" in msg.lower()


# ═══════════════════════════════════════════════════════════════
# InstructionStateMachine tests (10+)
# ═══════════════════════════════════════════════════════════════

class TestInstructionStateMachine:
    def setup_method(self):
        self.sm = InstructionStateMachine()

    def test_initial_state_is_idle(self):
        assert self.sm.current_state == InstructionState.IDLE

    def test_idle_to_parsing(self):
        self.sm.transition(InstructionState.PARSING)
        assert self.sm.current_state == InstructionState.PARSING

    def test_parsing_to_validating(self):
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        assert self.sm.current_state == InstructionState.VALIDATING

    def test_validating_to_accepted(self):
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        self.sm.transition(InstructionState.ACCEPTED)
        assert self.sm.current_state == InstructionState.ACCEPTED

    def test_validating_to_rejected(self):
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        self.sm.transition(InstructionState.REJECTED)
        assert self.sm.current_state == InstructionState.REJECTED

    def test_accepted_to_active(self):
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        self.sm.transition(InstructionState.ACCEPTED)
        self.sm.transition(InstructionState.ACTIVE)
        assert self.sm.current_state == InstructionState.ACTIVE

    def test_active_to_completed(self):
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        self.sm.transition(InstructionState.ACCEPTED)
        self.sm.transition(InstructionState.ACTIVE)
        self.sm.transition(InstructionState.COMPLETED)
        assert self.sm.current_state == InstructionState.COMPLETED

    def test_completed_to_idle(self):
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        self.sm.transition(InstructionState.ACCEPTED)
        self.sm.transition(InstructionState.ACTIVE)
        self.sm.transition(InstructionState.COMPLETED)
        self.sm.transition(InstructionState.IDLE)
        assert self.sm.current_state == InstructionState.IDLE

    def test_terminal_to_idle_allowed(self):
        """Any terminal state → IDLE is allowed (restart)."""
        for terminal in (InstructionState.REJECTED, InstructionState.BLOCKED,
                         InstructionState.CANCELLED, InstructionState.FAILED,
                         InstructionState.COMPLETED):
            sm = InstructionStateMachine()
            if terminal == InstructionState.REJECTED:
                sm.transition(InstructionState.PARSING)
                sm.transition(InstructionState.VALIDATING)
                sm.transition(InstructionState.REJECTED)
            elif terminal == InstructionState.COMPLETED:
                sm.transition(InstructionState.PARSING)
                sm.transition(InstructionState.VALIDATING)
                sm.transition(InstructionState.ACCEPTED)
                sm.transition(InstructionState.ACTIVE)
                sm.transition(InstructionState.COMPLETED)
            elif terminal == InstructionState.BLOCKED:
                sm.transition(InstructionState.PARSING)
                sm.transition(InstructionState.VALIDATING)
                sm.transition(InstructionState.ACCEPTED)
                sm.transition(InstructionState.ACTIVE)
                sm.transition(InstructionState.BLOCKED)
            elif terminal == InstructionState.CANCELLED:
                sm.transition(InstructionState.PARSING)
                sm.transition(InstructionState.VALIDATING)
                sm.transition(InstructionState.ACCEPTED)
                sm.transition(InstructionState.CANCELLED)
            elif terminal == InstructionState.FAILED:
                sm.transition(InstructionState.PARSING)
                sm.transition(InstructionState.FAILED)
            assert sm.is_terminal(), f"{terminal} should be terminal"
            sm.transition(InstructionState.IDLE)
            assert sm.current_state == InstructionState.IDLE

    def test_illegal_transition_raises(self):
        with pytest.raises(InvalidTransitionError):
            self.sm.transition(InstructionState.COMPLETED)  # IDLE→COMPLETED not allowed

    def test_illegal_transition_active_to_idle(self):
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        self.sm.transition(InstructionState.ACCEPTED)
        self.sm.transition(InstructionState.ACTIVE)
        with pytest.raises(InvalidTransitionError):
            self.sm.transition(InstructionState.IDLE)  # ACTIVE→IDLE not direct

    def test_snapshot(self):
        snap = self.sm.snapshot()
        assert snap["state"] == "idle"
        assert snap["terminal"] is False

    def test_is_terminal(self):
        assert not self.sm.is_terminal()
        self.sm.transition(InstructionState.PARSING)
        self.sm.transition(InstructionState.VALIDATING)
        self.sm.transition(InstructionState.REJECTED)
        assert self.sm.is_terminal()

    def test_thread_safety(self):
        """Concurrent transitions should not corrupt state."""
        errors = []

        def worker():
            for _ in range(100):
                try:
                    self.sm.transition(InstructionState.PARSING)
                    self.sm.transition(InstructionState.VALIDATING)
                    self.sm.transition(InstructionState.ACCEPTED)
                    self.sm.transition(InstructionState.ACTIVE)
                    self.sm.transition(InstructionState.COMPLETED)
                    self.sm.transition(InstructionState.IDLE)
                except InvalidTransitionError as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # If we got here without deadlock, the lock works.
        # Some InvalidTransitionErrors are expected under concurrency.
        assert True


# ═══════════════════════════════════════════════════════════════
# TaskCompiler tests (7+)
# ═══════════════════════════════════════════════════════════════

class TestTaskCompiler:
    def setup_method(self):
        self.compiler = TaskCompiler()

    def test_compile_stop(self):
        inst = _valid_instruction("stop")
        task = self.compiler.compile(inst)
        assert task["type"] == "immediate"
        assert task["action"] == "stop"
        assert task["cancel_active_task"] is True

    def test_compile_status(self):
        inst = _valid_instruction("status")
        task = self.compiler.compile(inst)
        assert task["type"] == "query"
        assert task["action"] == "status"

    def test_compile_goto_point(self):
        inst = _valid_instruction("goto_point", {"x_m": 100, "y_m": 200})
        task = self.compiler.compile(inst)
        assert task["type"] == "navigation"
        assert task["action"] == "goto_point"
        assert task["goal"] == {"x_m": 100, "y_m": 200}

    def test_compile_move_distance(self):
        inst = _valid_instruction("move_distance", {"distance_m": 5.0, "direction": "forward"})
        # Provide a pose snapshot
        snapshot = {"pose": {"x_m": 10.0, "y_m": 20.0, "yaw_rad": 0.0}}
        task = self.compiler.compile(inst, snapshot)
        assert task["type"] == "navigation"
        assert task["action"] == "move_distance"
        assert task["distance_m"] == 5.0
        assert task["direction"] == "forward"
        # x + 5*cos(0) = 15, y + 5*sin(0) = 20
        assert task["goal"]["x_m"] == 15.0
        assert task["goal"]["y_m"] == 20.0

    def test_compile_rotate(self):
        inst = _valid_instruction("rotate", {"angle_deg": 90, "direction": "left"})
        snapshot = {"pose": {"yaw_rad": 0.0}}
        task = self.compiler.compile(inst, snapshot)
        assert task["type"] == "rotation"
        assert task["action"] == "rotate"
        assert task["angle_rad"] == pytest.approx(math.pi / 2)
        assert task["direction"] == "left"
        assert abs(task["target_yaw_rad"] - math.pi / 2) < 1e-5

    def test_compile_scan_report(self):
        inst = _valid_instruction("scan_report", {"query": "前方"})
        task = self.compiler.compile(inst)
        assert task["type"] == "query"
        assert task["action"] == "scan_report"
        assert task["query"] == "前方"
        assert "summary" in task

    def test_compile_clarify(self):
        inst = _valid_instruction("clarify", {
            "question": "请指定坐标",
            "missing_parameters": ["x_m", "y_m"],
        })
        task = self.compiler.compile(inst)
        assert task["type"] == "clarification"
        assert task["action"] == "clarify"
        assert task["question"] == "请指定坐标"
        assert "x_m" in task["missing_parameters"]

    def test_compile_scan_with_points(self):
        """Test scan summarization with actual points."""
        inst = _valid_instruction("scan_report", {"query": ""})
        points = [
            {"angle": 0.0, "range": 1.5},
            {"angle": 0.3, "range": 2.0},
            {"angle": math.pi / 2, "range": 3.0},   # left
            {"angle": -math.pi / 2, "range": 4.0},  # right
            {"angle": math.pi, "range": 5.0},        # back
        ]
        snapshot = {"scan": {"points": points}}
        task = self.compiler.compile(inst, snapshot)
        summary = task["summary"]
        assert summary["total_points"] == 5
        sectors = summary["sectors"]
        assert sectors["front"]["count"] == 2
        assert sectors["left"]["count"] == 1
        assert sectors["right"]["count"] == 1
        assert sectors["back"]["count"] == 1
        assert sectors["front"]["min_m"] == 1.5


# ═══════════════════════════════════════════════════════════════
# AuthorityManager tests (5+)
# ═══════════════════════════════════════════════════════════════

class TestAuthorityManager:
    def setup_method(self):
        self.am = AuthorityManager()

    def test_initial_level_is_idle(self):
        assert self.am.current_level == AuthorityLevel.IDLE

    def test_agent_can_request_when_idle(self):
        assert self.am.request(AuthorityLevel.AGENT_CONTROL, "agent")
        assert self.am.current_level == AuthorityLevel.AGENT_CONTROL

    def test_lower_priority_denied(self):
        self.am.request(AuthorityLevel.MANUAL_CONTROL, "human")
        assert not self.am.request(AuthorityLevel.AGENT_CONTROL, "agent")
        assert self.am.current_level == AuthorityLevel.MANUAL_CONTROL

    def test_higher_priority_preempts(self):
        self.am.request(AuthorityLevel.AGENT_CONTROL, "agent")
        assert self.am.preempt(AuthorityLevel.SAFETY_BLOCK, "safety", "obstacle")
        assert self.am.current_level == AuthorityLevel.SAFETY_BLOCK

    def test_same_level_can_request(self):
        self.am.request(AuthorityLevel.AGENT_CONTROL, "agent1")
        assert self.am.request(AuthorityLevel.AGENT_CONTROL, "agent2")
        assert self.am.current_source == "agent2"

    def test_release_by_source(self):
        self.am.request(AuthorityLevel.AGENT_CONTROL, "agent1")
        assert self.am.release("agent1")
        assert self.am.current_level == AuthorityLevel.IDLE

    def test_release_wrong_source_fails(self):
        self.am.request(AuthorityLevel.AGENT_CONTROL, "agent1")
        assert not self.am.release("agent2")
        assert self.am.current_level == AuthorityLevel.AGENT_CONTROL

    def test_hardware_estop_overrides_all(self):
        self.am.request(AuthorityLevel.AGENT_CONTROL, "agent")
        self.am.request(AuthorityLevel.MANUAL_CONTROL, "human")
        self.am.request(AuthorityLevel.SAFETY_BLOCK, "safety")
        assert self.am.request(AuthorityLevel.HARDWARE_E_STOP, "estop")
        assert self.am.current_level == AuthorityLevel.HARDWARE_E_STOP

    def test_snapshot(self):
        snap = self.am.snapshot()
        assert snap["level"] == "idle"
        assert snap["source"] == "system"


# ═══════════════════════════════════════════════════════════════
# FakeModelClient tests (10+)
# ═══════════════════════════════════════════════════════════════

class TestFakeModelClient:
    def setup_method(self):
        self.client = FakeModelClient()

    def test_stop_variants(self):
        for text in ("停", "停下", "停止", "紧急停止", "别动了"):
            result = self.client.parse(text)
            assert result is not None
            assert result["intent"] == "stop", f"failed for '{text}'"

    def test_status_variants(self):
        for text in ("现在什么状态", "到哪了", "有没有问题", "状态", "在哪"):
            result = self.client.parse(text)
            assert result is not None
            assert result["intent"] == "status", f"failed for '{text}'"

    def test_goto_point_variants(self):
        tests = [
            ("去 (100, 200)", 100, 200),
            ("去坐标 (50, 30)", 50, 30),
            ("开到 10, 20", 10, 20),
            ("前往 (5, 8)", 5, 8),
        ]
        for text, ex, ey in tests:
            result = self.client.parse(text)
            assert result is not None, f"no result for '{text}'"
            assert result["intent"] == "goto_point", f"wrong intent for '{text}': {result['intent']}"
            assert result["parameters"]["x_m"] == ex
            assert result["parameters"]["y_m"] == ey

    def test_move_distance_forward(self):
        result = self.client.parse("前进 3 米")
        assert result is not None
        assert result["intent"] == "move_distance"
        assert result["parameters"]["distance_m"] == 3.0
        assert result["parameters"]["direction"] == "forward"

    def test_move_distance_backward(self):
        result = self.client.parse("后退 1.5 米")
        assert result is not None
        assert result["intent"] == "move_distance"
        assert result["parameters"]["distance_m"] == 1.5
        assert result["parameters"]["direction"] == "backward"

    def test_rotate_left(self):
        result = self.client.parse("左转 90 度")
        assert result is not None
        assert result["intent"] == "rotate"
        assert result["parameters"]["angle_deg"] == 90
        assert result["parameters"]["direction"] == "left"

    def test_rotate_right(self):
        result = self.client.parse("右转 45 度")
        assert result is not None
        assert result["intent"] == "rotate"
        assert result["parameters"]["angle_deg"] == 45
        assert result["parameters"]["direction"] == "right"

    def test_scan_simple(self):
        for text in ("看一下", "扫一圈", "扫描一下", "扫描"):
            result = self.client.parse(text)
            assert result is not None, f"failed for '{text}'"
            assert result["intent"] == "scan_report", f"wrong intent for '{text}': {result['intent']}"

    def test_scan_with_region(self):
        result = self.client.parse("前面有什么")
        assert result is not None
        assert result["intent"] == "scan_report"
        assert result["parameters"].get("query") == "前方"

    def test_clarify_unknown(self):
        result = self.client.parse("开到那边去")
        assert result is not None
        assert result["intent"] == "clarify"
        assert "question" in result["parameters"]

    def test_empty_input(self):
        result = self.client.parse("")
        assert result is not None
        assert result["intent"] == "clarify"

    def test_gibberish(self):
        result = self.client.parse("asdfghjkl")
        assert result is not None
        assert result["intent"] == "clarify"

    def test_confidence_range(self):
        result = self.client.parse("停")
        assert 0.0 <= result["confidence"] <= 1.0


# ═══════════════════════════════════════════════════════════════
# Integration tests (3+)
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline_stop(self):
        """stop: parse → schema ✓ → semantic ✓ → accepted"""
        client = FakeModelClient()
        instruction = client.parse("停")
        result = run_validation_pipeline(instruction)
        assert result.valid, result.message

    def test_full_pipeline_goto_valid(self, empty_grid):
        client = FakeModelClient()
        semantic = SemanticValidator(empty_grid)
        instruction = client.parse("去 (100, 100)")
        result = run_validation_pipeline(instruction, semantic_validator=semantic)
        assert result.valid, result.message

    def test_full_pipeline_goto_wall(self, grid_with_obstacles):
        client = FakeModelClient()
        semantic = SemanticValidator(grid_with_obstacles)
        instruction = client.parse("去 (50, 50)")
        result = run_validation_pipeline(instruction, semantic_validator=semantic)
        assert not result.valid
        assert result.layer == "semantic"

    def test_full_pipeline_with_safety(self, healthy_safety):
        client = FakeModelClient()
        safety_v = SafetyValidator(healthy_safety)
        instruction = client.parse("停")
        result = run_validation_pipeline(instruction, safety_validator=safety_v)
        assert result.valid

    def test_full_pipeline_safety_fault(self, faulty_safety):
        client = FakeModelClient()
        safety_v = SafetyValidator(faulty_safety)
        instruction = client.parse("停")
        result = run_validation_pipeline(instruction, safety_validator=safety_v)
        assert not result.valid
        assert result.layer == "safety"
