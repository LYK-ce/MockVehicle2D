"""Instruction state machine — 11 states for instruction lifecycle tracking.

Thread-safe state transitions with snapshot support.
"""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import ClassVar


class InstructionState(Enum):
    """Eleven states of the instruction lifecycle."""

    IDLE = auto()
    PARSING = auto()
    VALIDATING = auto()
    CONFIRMING = auto()
    REJECTED = auto()
    ACCEPTED = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    BLOCKED = auto()
    CANCELLED = auto()
    FAILED = auto()


# Allowed transitions: from → set of valid next states
_VALID_TRANSITIONS: dict[InstructionState, frozenset[InstructionState]] = {
    InstructionState.IDLE: frozenset({InstructionState.PARSING, InstructionState.IDLE}),
    InstructionState.PARSING: frozenset(
        {InstructionState.VALIDATING, InstructionState.FAILED, InstructionState.IDLE}
    ),
    InstructionState.VALIDATING: frozenset(
        {InstructionState.CONFIRMING, InstructionState.REJECTED, InstructionState.ACCEPTED}
    ),
    InstructionState.CONFIRMING: frozenset(
        {InstructionState.ACCEPTED, InstructionState.REJECTED, InstructionState.CANCELLED}
    ),
    InstructionState.REJECTED: frozenset({InstructionState.IDLE}),
    InstructionState.ACCEPTED: frozenset({InstructionState.ACTIVE, InstructionState.CANCELLED}),
    InstructionState.ACTIVE: frozenset(
        {InstructionState.COMPLETED, InstructionState.BLOCKED, InstructionState.CANCELLED, InstructionState.FAILED}
    ),
    InstructionState.COMPLETED: frozenset({InstructionState.IDLE}),
    InstructionState.BLOCKED: frozenset({InstructionState.IDLE, InstructionState.CANCELLED}),
    InstructionState.CANCELLED: frozenset({InstructionState.IDLE}),
    InstructionState.FAILED: frozenset({InstructionState.IDLE}),
}

_TERMINAL_STATES: frozenset[InstructionState] = frozenset(
    {
        InstructionState.REJECTED,
        InstructionState.COMPLETED,
        InstructionState.BLOCKED,
        InstructionState.CANCELLED,
        InstructionState.FAILED,
    }
)


class InvalidTransitionError(ValueError):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, current: InstructionState, target: InstructionState) -> None:
        super().__init__(f"invalid transition: {current.name} → {target.name}")
        self.current = current
        self.target = target


class InstructionStateMachine:
    """Thread-safe instruction state machine with 11 states.

    IDLE ← any terminal state (new nl_command can restart).
    """

    def __init__(self) -> None:
        self._state = InstructionState.IDLE
        self._lock = threading.Lock()

    @property
    def current_state(self) -> InstructionState:
        with self._lock:
            return self._state

    def transition(self, new_state: InstructionState) -> None:
        """Transition to *new_state*, raising InvalidTransitionError if illegal."""
        with self._lock:
            current = self._state
            valid_next = _VALID_TRANSITIONS.get(current, frozenset())
            # Allow IDLE ← any terminal state (new nl_command can restart)
            if current in _TERMINAL_STATES and new_state == InstructionState.IDLE:
                self._state = new_state
                return
            if new_state not in valid_next:
                raise InvalidTransitionError(current, new_state)
            self._state = new_state

    def is_terminal(self) -> bool:
        """True if the current state is a terminal state."""
        return self.current_state in _TERMINAL_STATES

    def snapshot(self) -> dict[str, object]:
        """Return a telemetry-friendly snapshot."""
        with self._lock:
            return {
                "state": self._state.name.lower(),
                "terminal": self._state in _TERMINAL_STATES,
            }

    def __repr__(self) -> str:
        return f"InstructionStateMachine(state={self.current_state.name})"
