"""Deterministic motion and actuator state for the robot controller."""

from __future__ import annotations

import math

from mockvehicle2d.collision import is_swept_circle_passable
from mockvehicle2d.map_grid import MapGrid


TimedPose = tuple[float, float, float, float]

DEFAULT_LINEAR_ACCELERATION_MPS2 = 1.0
DEFAULT_LINEAR_DECELERATION_MPS2 = 1.0
DEFAULT_ANGULAR_ACCELERATION_RPS2 = math.pi


def _ramp_velocity(
    current: float,
    target: float,
    rate: float,
    elapsed: float,
) -> tuple[float, float]:
    """Return the ending velocity and exact trapezoidal integral."""
    if current == target or elapsed == 0:
        return current, current * elapsed
    ramp_time = abs(target - current) / rate
    if elapsed >= ramp_time or math.isclose(
        elapsed,
        ramp_time,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return target, (current + target) * ramp_time / 2 + target * (
            elapsed - ramp_time
        )
    ending = current + math.copysign(rate * elapsed, target - current)
    return ending, (current + ending) * elapsed / 2


def _integrate_velocity(
    current: float,
    target: float,
    acceleration: float,
    deceleration: float,
    elapsed: float,
) -> tuple[float, float]:
    """Approach a target without jumping through zero on reversals."""
    if current * target < 0:
        stopping_time = abs(current) / deceleration
        if elapsed <= stopping_time:
            return _ramp_velocity(current, 0.0, deceleration, elapsed)
        _, stopping_integral = _ramp_velocity(
            current,
            0.0,
            deceleration,
            stopping_time,
        )
        ending, starting_integral = _ramp_velocity(
            0.0,
            target,
            acceleration,
            elapsed - stopping_time,
        )
        return ending, stopping_integral + starting_integral
    rate = acceleration if abs(target) > abs(current) else deceleration
    return _ramp_velocity(current, target, rate, elapsed)


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
        linear_acceleration_mps2: float = DEFAULT_LINEAR_ACCELERATION_MPS2,
        linear_deceleration_mps2: float = DEFAULT_LINEAR_DECELERATION_MPS2,
        angular_acceleration_rps2: float = DEFAULT_ANGULAR_ACCELERATION_RPS2,
        radius: float = 0.5,
        command_timeout: float = 1.0,
        now: float = 0.0,
    ) -> None:
        parameters = (
            linear_speed,
            angular_speed,
            linear_acceleration_mps2,
            linear_deceleration_mps2,
            angular_acceleration_rps2,
            radius,
            command_timeout,
            now,
        )
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("vehicle parameters must be finite")
        if min(parameters[:-1]) <= 0:
            raise ValueError("vehicle motion limits, radius, and timeout must be positive")
        self.x = x
        self.y = y
        self.yaw = yaw
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.linear_acceleration_mps2 = linear_acceleration_mps2
        self.linear_deceleration_mps2 = linear_deceleration_mps2
        self.angular_acceleration_rps2 = angular_acceleration_rps2
        self.radius = radius
        self.command_timeout = command_timeout
        self.command = "stop"
        self.collision = False
        self._linear_mps = 0.0
        self._angular_rps = 0.0
        self._target_linear_mps = 0.0
        self._target_angular_rps = 0.0
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
        """Request bounded braking; ``now`` discards unintegrated prior motion."""
        if now is not None:
            if not math.isfinite(now) or now < self._last_update:
                raise ValueError("monotonic time moved backwards")
            self._last_update = now
        self.command = "stop"
        self._target_linear_mps = 0.0
        self._target_angular_rps = 0.0
        self._command_deadline = None

    def force_stop(self, now: float | None = None) -> None:
        """Immediately clamp motion after physics rejects a candidate trajectory."""
        self.stop(now)
        self._linear_mps = 0.0
        self._angular_rps = 0.0

    def advance(
        self,
        grid: MapGrid,
        now: float,
        *,
        limited_velocities: tuple[float, float] | None = None,
        trajectory: list[TimedPose] | None = None,
    ) -> bool:
        """Integrate through ``now`` and optionally record timed positions."""
        if now < self._last_update:
            raise ValueError("monotonic time moved backwards")
        if trajectory is not None:
            if trajectory:
                raise ValueError("trajectory output must be empty")
            trajectory.append((self._last_update, self.x, self.y, self.yaw))

        limited = self._validated_limited_velocities(limited_velocities)
        collided = False
        deadline = self._command_deadline
        if deadline is not None and self._last_update < deadline < now:
            collided = not self._advance_interval(grid, deadline, limited, trajectory)
            if not collided:
                self.stop()
        if not collided and self._last_update < now:
            if deadline is not None and deadline <= self._last_update:
                self.stop()
            collided = not self._advance_interval(grid, now, limited, trajectory)

        if trajectory is not None and trajectory[-1][0] < now:
            trajectory.append((now, self.x, self.y, self.yaw))

        self._last_update = now
        if self._command_deadline is not None and now >= self._command_deadline:
            self.stop()
        return collided

    def body_velocities(self) -> tuple[float, float]:
        """Return the currently executed linear and angular velocities."""
        return self._linear_mps, self._angular_rps

    def target_velocities(self) -> tuple[float, float]:
        """Return the controller-requested linear and angular velocities."""
        return self._target_linear_mps, self._target_angular_rps

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
        self._target_linear_mps = linear_mps
        self._target_angular_rps = angular_rps
        self._command_deadline = now + self.command_timeout

    def _validated_limited_velocities(
        self,
        limited_velocities: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if limited_velocities is None:
            return None
        limited_linear, limited_angular = limited_velocities
        if not all(math.isfinite(value) for value in limited_velocities):
            raise ValueError("limited velocities must be finite")
        for limited, current, target in (
            (limited_linear, self._linear_mps, self._target_linear_mps),
            (limited_angular, self._angular_rps, self._target_angular_rps),
        ):
            reference = current if current else target
            if limited * reference < 0 or abs(limited) > max(abs(current), abs(target)):
                raise ValueError("limited velocities must reduce the active motion")
        return float(limited_linear), float(limited_angular)

    def _advance_interval(
        self,
        grid: MapGrid,
        ended_at: float,
        limited_velocities: tuple[float, float] | None,
        trajectory: list[TimedPose] | None,
    ) -> bool:
        started_at = self._last_update
        elapsed = ended_at - started_at
        target_linear, target_angular = (
            self.target_velocities()
            if limited_velocities is None or self.command == "stop"
            else limited_velocities
        )
        ending_linear, distance = _integrate_velocity(
            self._linear_mps,
            target_linear,
            self.linear_acceleration_mps2,
            self.linear_deceleration_mps2,
            elapsed,
        )
        ending_angular, rotation = _integrate_velocity(
            self._angular_rps,
            target_angular,
            self.angular_acceleration_rps2,
            self.angular_acceleration_rps2,
            elapsed,
        )
        if (distance or rotation) and not self._move(
            grid,
            distance,
            rotation,
            started_at=started_at,
            ended_at=ended_at,
            trajectory=trajectory,
        ):
            self.collision = True
            self.force_stop()
            self._last_update = ended_at
            return False
        self._linear_mps = ending_linear
        self._angular_rps = ending_angular
        self._last_update = ended_at
        return True

    def _move(
        self,
        grid: MapGrid,
        distance: float,
        rotation: float,
        *,
        started_at: float,
        ended_at: float,
        trajectory: list[TimedPose] | None,
    ) -> bool:
        if distance == 0:
            self.yaw = math.atan2(math.sin(self.yaw + rotation), math.cos(self.yaw + rotation))
            self.collision = False
            if trajectory is not None and ended_at > trajectory[-1][0]:
                trajectory.append((ended_at, self.x, self.y, self.yaw))
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
                if trajectory is not None:
                    timestamp = started_at + (ended_at - started_at) * (step + 1) / steps
                    if timestamp > trajectory[-1][0]:
                        trajectory.append((timestamp, self.x, self.y, self.yaw))
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
                    (timestamp, self.x, self.y, self.yaw)
                )
        self.collision = False
        return True
