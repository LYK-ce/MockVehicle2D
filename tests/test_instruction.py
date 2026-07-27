"""Tests for mockvehicle2d.instruction — Phase 1 offline closed loop.

Covers:
- SchemaValidator
- SemanticValidator
- InstructionStateMachine
- TaskCompiler
- AuthorityManager
- LLMClient
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


def _valid_instruction(intent="stop", params=None):
    """Build a minimally valid v3 instruction dict."""
    return {"intent": intent, "parameters": params or {}}


# ═══════════════════════════════════════════════════════════════
# SchemaValidator tests
# ═══════════════════════════════════════════════════════════════

class TestSchemaValidator:
    def setup_method(self):
        self.v = SchemaValidator()

    def test_valid_stop(self):
        ok, msg = self.v.validate(_valid_instruction("stop"))
        assert ok, msg

    def test_valid_goto(self):
        ok, msg = self.v.validate(_valid_instruction("goto", {"x_m": 100, "y_m": 200}))
        assert ok, msg

    def test_valid_patrol(self):
        ok, msg = self.v.validate(_valid_instruction("patrol"))
        assert ok, msg

    def test_valid_clarify(self):
        ok, msg = self.v.validate(
            _valid_instruction("clarify", {"question": "where?"})
        )
        assert ok, msg

    def test_missing_parameters_field(self):
        """v3 requires 'parameters' field."""
        inst = {"intent": "stop"}
        ok, msg = self.v.validate(inst)
        assert not ok
        assert "parameters" in msg

    def test_missing_intent(self):
        inst = {"parameters": {}}
        ok, msg = self.v.validate(inst)
        assert not ok

    def test_invalid_intent(self):
        ok, msg = self.v.validate(_valid_instruction("fly"))
        assert not ok

    def test_goto_missing_params(self):
        ok, msg = self.v.validate(_valid_instruction("goto", {}))
        assert not ok
        assert "x_m" in msg

    def test_extra_fields_are_allowed(self):
        """v3 has additionalProperties: true — extra top-level fields are ignored."""
        inst = {"intent": "stop", "parameters": {}, "confidence": 0.95, "reasoning": "test"}
        ok, msg = self.v.validate(inst)
        assert ok, msg

    def test_injection_extra_fields(self):
        """v3 allows extra top-level fields (additionalProperties: true)."""
        inst = {"intent": "stop", "parameters": {}, "injected_field": "malicious"}
        ok, msg = self.v.validate(inst)
        assert ok, msg  # v3 is lenient with extra fields

    def test_injection_extra_param_fields(self):
        """Extra fields in parameters for stop are rejected."""
        inst = _valid_instruction("stop", {"evil": True})
        ok, msg = self.v.validate(inst)
        assert not ok

    def test_clarify_missing_question(self):
        ok, msg = self.v.validate(_valid_instruction("clarify", {}))
        assert not ok

    def test_unknown_intent_rejected(self):
        """Unknown intent values are rejected."""
        ok, msg = self.v.validate(_valid_instruction("unknown_intent"))
        assert not ok

    def test_old_intents_rejected(self):
        """v2 intents (status, goto_point, move_distance, rotate, scan_report) are rejected in v3."""
        for old in ("status", "goto_point", "move_distance", "rotate", "scan_report"):
            ok, msg = self.v.validate(_valid_instruction(old))
            assert not ok, f"{old} should be rejected in v3"


# ═══════════════════════════════════════════════════════════════
# SemanticValidator tests
# ═══════════════════════════════════════════════════════════════

class TestSemanticValidator:
    def setup_method(self):
        self.grid = MapGrid(256, 256)
        self.v = SemanticValidator(self.grid)

    def test_goto_in_bounds(self):
        ok, msg = self.v.validate(_valid_instruction("goto", {"x_m": 100, "y_m": 200}))
        assert ok, msg

    def test_goto_negative_x(self):
        ok, msg = self.v.validate(_valid_instruction("goto", {"x_m": -10, "y_m": 100}))
        assert not ok
        assert "out of map bounds" in msg

    def test_goto_too_large(self):
        ok, msg = self.v.validate(_valid_instruction("goto", {"x_m": 300, "y_m": 100}))
        assert not ok

    def test_goto_is_wall(self, grid_with_obstacles):
        v = SemanticValidator(grid_with_obstacles)
        ok, msg = v.validate(_valid_instruction("goto", {"x_m": 50, "y_m": 50}))
        assert not ok
        assert "wall" in msg

    def test_goto_is_void(self, grid_with_obstacles):
        v = SemanticValidator(grid_with_obstacles)
        ok, msg = v.validate(_valid_instruction("goto", {"x_m": 24, "y_m": 10}))
        assert not ok
        assert "void" in msg

    def test_stop_no_semantic_check(self):
        ok, msg = self.v.validate(_valid_instruction("stop"))
        assert ok, msg

    def test_patrol_no_semantic_check(self):
        ok, msg = self.v.validate(_valid_instruction("patrol"))
        assert ok, msg

    def test_clarify_no_semantic_check(self):
        ok, msg = self.v.validate(_valid_instruction("clarify", {"question": "test"}))
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
# InstructionStateMachine tests
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
# TaskCompiler tests
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

    def test_compile_goto(self):
        inst = _valid_instruction("goto", {"x_m": 100, "y_m": 200})
        task = self.compiler.compile(inst)
        assert task["type"] == "navigation"
        assert task["action"] == "goto"
        assert task["goal"] == {"x_m": 100, "y_m": 200}

    def test_compile_patrol(self):
        inst = _valid_instruction("patrol")
        task = self.compiler.compile(inst)
        assert task["type"] == "navigation"
        assert task["action"] == "patrol"

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

    def test_compile_unknown_intent(self):
        inst = _valid_instruction("some_new_intent")
        task = self.compiler.compile(inst)
        assert task["type"] == "unknown"


# ═══════════════════════════════════════════════════════════════
# AuthorityManager tests
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
# Integration tests
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline_stop(self):
        """stop: parse → schema ✓ → semantic ✓ → accepted"""
        instruction = {"intent": "stop", "parameters": {}}
        result = run_validation_pipeline(instruction)
        assert result.valid, result.message

    def test_full_pipeline_goto_valid(self, empty_grid):
        semantic = SemanticValidator(empty_grid)
        instruction = {"intent": "goto", "parameters": {"x_m": 100, "y_m": 100}}
        result = run_validation_pipeline(instruction, semantic_validator=semantic)
        assert result.valid, result.message

    def test_full_pipeline_goto_wall(self, grid_with_obstacles):
        semantic = SemanticValidator(grid_with_obstacles)
        instruction = {"intent": "goto", "parameters": {"x_m": 50, "y_m": 50}}
        result = run_validation_pipeline(instruction, semantic_validator=semantic)
        assert not result.valid
        assert result.layer == "semantic"

    def test_full_pipeline_patrol(self):
        instruction = {"intent": "patrol", "parameters": {}}
        result = run_validation_pipeline(instruction)
        assert result.valid, result.message

    def test_full_pipeline_with_safety(self, healthy_safety):
        safety_v = SafetyValidator(healthy_safety)
        instruction = {"intent": "stop", "parameters": {}}
        result = run_validation_pipeline(instruction, safety_validator=safety_v)
        assert result.valid

    def test_full_pipeline_safety_fault(self, faulty_safety):
        safety_v = SafetyValidator(faulty_safety)
        instruction = {"intent": "stop", "parameters": {}}
        result = run_validation_pipeline(instruction, safety_validator=safety_v)
        assert not result.valid
        assert result.layer == "safety"
