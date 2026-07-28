"""Authority manager — priority-based control arbitration.

Five priority levels ensure safety > manual > agent > idle.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import IntEnum


class AuthorityLevel(IntEnum):
    """Priority levels for control authority (higher = more authority)."""

    IDLE = 0
    AGENT_CONTROL = 1
    MANUAL_CONTROL = 2
    SAFETY_BLOCK = 3
    HARDWARE_E_STOP = 4

    @classmethod
    def label(cls, level: int) -> str:
        try:
            return cls(level).name.lower()
        except ValueError:
            return "unknown"


@dataclass
class AuthorityRecord:
    level: AuthorityLevel
    source: str
    timestamp: float = 0.0


class AuthorityManager:
    """Central authority arbitration with five priority levels.

    Priority order (highest to lowest):
    1. HARDWARE_E_STOP
    2. SAFETY_BLOCK
    3. MANUAL_CONTROL
    4. AGENT_CONTROL
    5. IDLE

    Only a higher-or-equal priority level can preempt the current holder.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = AuthorityRecord(AuthorityLevel.IDLE, "system")
        self._history: list[AuthorityRecord] = []

    @property
    def current_level(self) -> AuthorityLevel:
        with self._lock:
            return self._current.level

    @property
    def current_source(self) -> str:
        with self._lock:
            return self._current.source

    def request(self, level: AuthorityLevel, source: str) -> bool:
        """Request authority at *level* for *source*.

        Returns True if granted, False if denied (lower priority).
        """
        with self._lock:
            if level >= self._current.level:
                old = self._current
                self._current = AuthorityRecord(level, source)
                self._history.append(old)
                return True
            return False

    def preempt(self, level: AuthorityLevel, source: str, reason: str) -> bool:
        """Preempt the current authority holder.

        Cancels the current holder if it has lower priority.
        Returns True if preemption succeeded.
        """
        with self._lock:
            if level >= self._current.level:
                old = self._current
                self._current = AuthorityRecord(level, source)
                self._history.append(old)
                return True
            return False

    def release(self, source: str) -> bool:
        """Release authority if *source* is the current holder.

        Falls back to IDLE. Returns True if released.
        """
        with self._lock:
            if self._current.source == source:
                old = self._current
                self._current = AuthorityRecord(AuthorityLevel.IDLE, "system")
                self._history.append(old)
                return True
            return False

    def snapshot(self) -> dict[str, object]:
        """Return a telemetry-friendly snapshot."""
        with self._lock:
            return {
                "level": self._current.level.name.lower(),
                "level_value": self._current.level.value,
                "source": self._current.source,
                "history_count": len(self._history),
            }

    def __repr__(self) -> str:
        return (
            f"AuthorityManager(level={self.current_level.name}, "
            f"source={self.current_source!r})"
        )
