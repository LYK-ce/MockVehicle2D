"""Shared deterministic motion state for the server and Pygame viewer."""

from __future__ import annotations

import math

from mockvehicle2d.collision import is_circle_passable
from mockvehicle2d.map_grid import MapGrid


COMMANDS = frozenset({"forward", "backward", "spin_left", "spin_right", "stop"})


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
        if command != "stop":
            self.collision = False

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
            if linear and not self._translate(grid, linear * elapsed):
                self.collision = True
                self.stop()
            elif angular:
                self.yaw = math.atan2(math.sin(self.yaw + angular * elapsed), math.cos(self.yaw + angular * elapsed))
                self.collision = False

        self._last_update = now
        if self._command_deadline is not None and now >= self._command_deadline:
            self.stop()

    def velocities(self) -> tuple[float, float, float]:
        linear, angular = self._command_velocities()
        return linear * math.cos(self.yaw), linear * math.sin(self.yaw), angular

    def _command_velocities(self) -> tuple[float, float]:
        if self.command == "forward":
            return self.linear_speed, 0.0
        if self.command == "backward":
            return -self.linear_speed, 0.0
        if self.command == "spin_left":
            return 0.0, -self.angular_speed
        if self.command == "spin_right":
            return 0.0, self.angular_speed
        return 0.0, 0.0

    def _translate(self, grid: MapGrid, distance: float) -> bool:
        # ponytail: fixed substeps suit the 1 m grid; use swept collision if maps become continuous.
        max_step = max(0.01, min(0.25, self.radius / 2))
        steps = max(1, math.ceil(abs(distance) / max_step))
        step = distance / steps
        for _ in range(steps):
            x = self.x + step * math.cos(self.yaw)
            y = self.y + step * math.sin(self.yaw)
            if not is_circle_passable(grid, x, y, self.radius):
                return False
            self.x, self.y = x, y
        self.collision = False
        return True
