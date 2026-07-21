"""Shared deterministic motion state for the server and Pygame viewer."""

from __future__ import annotations

import math

from mockvehicle2d.collision import is_swept_circle_passable
from mockvehicle2d.map_grid import MapGrid


COMMANDS = frozenset(
    {
        "forward",
        "forward_left",
        "forward_right",
        "backward",
        "backward_left",
        "backward_right",
        "spin_left",
        "spin_right",
        "stop",
    }
)


def command_from_axes(forward: bool, backward: bool, left: bool, right: bool) -> str:
    """Convert held directional inputs into one canonical command."""
    linear = int(bool(forward)) - int(bool(backward))
    turn = int(bool(right)) - int(bool(left))
    if linear:
        command = "forward" if linear > 0 else "backward"
        return command + ("_right" if turn > 0 else "_left" if turn < 0 else "")
    return "spin_right" if turn > 0 else "spin_left" if turn < 0 else "stop"


class Vehicle:
    """A circular differential-drive vehicle in the simulator's screen coordinates."""

    def __init__(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        *,
        linear_speed: float = 0.5,
        angular_speed: float = math.pi / 2,
        radius: float = 0.5,
        command_timeout: float = 1.0,
        now: float = 0.0,
    ) -> None:
        parameters = (linear_speed, angular_speed, radius, command_timeout, now)
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("vehicle parameters must be finite")
        if min(linear_speed, angular_speed, radius, command_timeout) <= 0:
            raise ValueError("vehicle speeds, radius, and command timeout must be positive")
        self.x = x
        self.y = y
        self.yaw = yaw
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.radius = radius
        self.command_timeout = command_timeout
        self.command = "stop"
        self.collision = False
        self._last_update = now
        self._command_deadline: float | None = None

    def reset(self, x: float, y: float, yaw: float, now: float) -> None:
        self.x, self.y, self.yaw = x, y, yaw
        self.command = "stop"
        self.collision = False
        self._last_update = now
        self._command_deadline = None

    def apply_command(self, grid: MapGrid, command: str, now: float) -> None:
        """Advance the old command to ``now``, then install the new command."""
        if command not in COMMANDS:
            raise ValueError(f"unsupported command: {command}")
        self.advance(grid, now)
        self.command = command
        self._command_deadline = now + self.command_timeout if command != "stop" else None

    def stop(self) -> None:
        self.command = "stop"
        self._command_deadline = None

    def advance(self, grid: MapGrid, now: float) -> None:
        """Integrate commanded motion through ``now`` using actual monotonic time."""
        if now < self._last_update:
            raise ValueError("monotonic time moved backwards")

        motion_until = min(now, self._command_deadline) if self._command_deadline is not None else now
        elapsed = motion_until - self._last_update
        if elapsed > 0:
            linear, angular = self._command_velocities()
            if (linear or angular) and not self._move(grid, linear * elapsed, angular * elapsed):
                self.collision = True
                self.stop()

        self._last_update = now
        if self._command_deadline is not None and now >= self._command_deadline:
            self.stop()

    def velocities(self) -> tuple[float, float, float]:
        linear, angular = self._command_velocities()
        return linear * math.cos(self.yaw), linear * math.sin(self.yaw), angular

    def _command_velocities(self) -> tuple[float, float]:
        if self.command == "forward":
            return self.linear_speed, 0.0
        if self.command == "forward_left":
            return self.linear_speed, -self.angular_speed
        if self.command == "forward_right":
            return self.linear_speed, self.angular_speed
        if self.command == "backward":
            return -self.linear_speed, 0.0
        if self.command == "backward_left":
            return -self.linear_speed, -self.angular_speed
        if self.command == "backward_right":
            return -self.linear_speed, self.angular_speed
        if self.command == "spin_left":
            return 0.0, -self.angular_speed
        if self.command == "spin_right":
            return 0.0, self.angular_speed
        return 0.0, 0.0

    def _move(self, grid: MapGrid, distance: float, rotation: float) -> bool:
        if distance == 0:
            self.yaw = math.atan2(math.sin(self.yaw + rotation), math.cos(self.yaw + rotation))
            self.collision = False
            return True

        # Short chords retain a nearby last-safe pose while approximating an arc.
        max_step = max(0.01, min(0.25, self.radius / 2))
        steps = max(
            1,
            math.ceil(abs(distance) / max_step),
            math.ceil(abs(rotation) / (math.pi / 18)),
        )
        step_distance = distance / steps
        step_rotation = rotation / steps
        for _ in range(steps):
            mid_yaw = self.yaw + step_rotation / 2
            x = self.x + step_distance * math.cos(mid_yaw)
            y = self.y + step_distance * math.sin(mid_yaw)
            if step_distance and not is_swept_circle_passable(grid, self.x, self.y, x, y, self.radius):
                return False
            self.x, self.y = x, y
            self.yaw = math.atan2(math.sin(self.yaw + step_rotation), math.cos(self.yaw + step_rotation))
        self.collision = False
        return True
