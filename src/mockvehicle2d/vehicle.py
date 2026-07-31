"""Deterministic motion and actuator state for the robot controller."""

from __future__ import annotations

import math

from mockvehicle2d.collision import is_swept_circle_passable
from mockvehicle2d.map_grid import MapGrid


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
        self._linear_mps = 0.0
        self._angular_rps = 0.0
        self._last_update = now
        self._command_deadline: float | None = None

    def install_drive(self, linear_mps: float, angular_rps: float, now: float) -> None:
        """Install bounded velocities after the vehicle has already advanced to ``now``."""
        linear, angular = self._validated_drive_velocities(linear_mps, angular_rps)
        self._install_velocities(linear, angular, now)

    def _validated_drive_velocities(
        self, linear_mps: float, angular_rps: float
    ) -> tuple[float, float]:
        values = (linear_mps, angular_rps)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("drive velocities must be numbers")
        if any(isinstance(value, float) and not math.isfinite(value) for value in values):
            raise ValueError("drive velocities must be finite")
        if abs(linear_mps) > self.linear_speed or abs(angular_rps) > self.angular_speed:
            raise ValueError("drive velocities exceed configured limits")
        return float(linear_mps), float(angular_rps)

    def stop(self, now: float | None = None) -> None:
        """Stop immediately; ``now`` discards any unintegrated prior motion."""
        if now is not None:
            if now < self._last_update:
                raise ValueError("monotonic time moved backwards")
            self._last_update = now
        self.command = "stop"
        self._linear_mps = 0.0
        self._angular_rps = 0.0
        self._command_deadline = None

    def advance(
        self,
        grid: MapGrid,
        now: float,
        *,
        limited_velocities: tuple[float, float] | None = None,
        trajectory: list[tuple[float, float, float]] | None = None,
    ) -> bool:
        """Integrate through ``now`` and optionally record timed positions."""
        if now < self._last_update:
            raise ValueError("monotonic time moved backwards")
        if trajectory is not None:
            if trajectory:
                raise ValueError("trajectory output must be empty")
            trajectory.append((self._last_update, self.x, self.y))

        collided = False
        motion_until = min(now, self._command_deadline) if self._command_deadline is not None else now
        elapsed = motion_until - self._last_update
        if elapsed > 0:
            linear, angular = self.body_velocities()
            if limited_velocities is not None:
                limited_linear, limited_angular = limited_velocities
                if (
                    not all(math.isfinite(value) for value in limited_velocities)
                    or abs(limited_linear) > abs(linear)
                    or abs(limited_angular) > abs(angular)
                    or limited_linear * linear < 0
                    or limited_angular * angular < 0
                ):
                    raise ValueError("limited velocities must reduce the active command")
                linear, angular = limited_linear, limited_angular
            if (linear or angular) and not self._move(
                grid,
                linear * elapsed,
                angular * elapsed,
                started_at=self._last_update,
                ended_at=motion_until,
                trajectory=trajectory,
            ):
                self.collision = True
                self.stop()
                collided = True

        if trajectory is not None and trajectory[-1][0] < now:
            trajectory.append((now, self.x, self.y))

        self._last_update = now
        if self._command_deadline is not None and now >= self._command_deadline:
            self.stop()
        return collided

    def body_velocities(self) -> tuple[float, float]:
        """Return the currently commanded linear and angular velocities."""
        return self._linear_mps, self._angular_rps

    @property
    def last_update(self) -> float:
        return self._last_update

    @property
    def command_deadline(self) -> float | None:
        return self._command_deadline

    def _install_velocities(self, linear_mps: float, angular_rps: float, now: float) -> None:
        if now != self._last_update:
            raise ValueError("vehicle must be advanced to now before installing velocities")
        if linear_mps == 0 and angular_rps == 0:
            self.stop()
            return
        self.command = "drive"
        self._linear_mps = linear_mps
        self._angular_rps = angular_rps
        self._command_deadline = now + self.command_timeout

    def _move(
        self,
        grid: MapGrid,
        distance: float,
        rotation: float,
        *,
        started_at: float,
        ended_at: float,
        trajectory: list[tuple[float, float, float]] | None,
    ) -> bool:
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
        for step in range(steps):
            mid_yaw = self.yaw + step_rotation / 2
            x = self.x + step_distance * math.cos(mid_yaw)
            y = self.y + step_distance * math.sin(mid_yaw)
            if step_distance and not is_swept_circle_passable(grid, self.x, self.y, x, y, self.radius):
                return False
            self.x, self.y = x, y
            self.yaw = math.atan2(math.sin(self.yaw + step_rotation), math.cos(self.yaw + step_rotation))
            if trajectory is not None:
                timestamp = (
                    ended_at
                    if step + 1 == steps
                    else started_at + (ended_at - started_at) * (step + 1) / steps
                )
                trajectory.append(
                    (timestamp, self.x, self.y)
                )
        self.collision = False
        return True
