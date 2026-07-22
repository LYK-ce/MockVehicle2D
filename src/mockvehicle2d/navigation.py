"""Minimal local-odometry go-to-goal controller."""

from __future__ import annotations

import math

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import LocalSafetyRuntime
from mockvehicle2d.vehicle import Vehicle


class GotoController:
    """Drive directly toward one goal; stop on arrival, collision, or override."""

    goal_tolerance_m = 0.1
    turn_in_place_threshold_rad = math.radians(20)

    def __init__(self) -> None:
        self.status = "idle"
        self.goal: tuple[float, float] | None = None
        self.reason: str | None = None

    @property
    def control_mode(self) -> str:
        return "autonomous" if self.status == "active" else "manual"

    def start(self, x_m: float, y_m: float) -> None:
        self.goal = (x_m, y_m)
        self.status = "active"
        self.reason = None

    def cancel(self, reason: str) -> None:
        if self.status == "active":
            self.status = "cancelled"
            self.reason = reason

    def snapshot(self) -> dict[str, object]:
        goal = None if self.goal is None else {"x_m": self.goal[0], "y_m": self.goal[1]}
        return {"status": self.status, "goal": goal, "reason": self.reason}

    def update(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        now: float,
        safety: LocalSafetyRuntime | None = None,
    ) -> None:
        collided = vehicle.advance(grid, now)
        if self.status != "active":
            return
        if collided:
            self.status = "blocked"
            self.reason = "collision"
            vehicle.stop()
            return

        assert self.goal is not None
        dx, dy = self.goal[0] - vehicle.x, self.goal[1] - vehicle.y
        distance = math.hypot(dx, dy)
        if distance <= self.goal_tolerance_m:
            self.status = "reached"
            self.reason = "goal_tolerance"
            vehicle.stop()
            return

        desired_yaw = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(desired_yaw - vehicle.yaw), math.cos(desired_yaw - vehicle.yaw)
        )
        angular_rps = max(-vehicle.angular_speed, min(vehicle.angular_speed, 2 * heading_error))
        linear_mps = (
            0.0
            if abs(heading_error) > self.turn_in_place_threshold_rad
            else min(vehicle.linear_speed, distance)
        )
        if safety is not None:
            decision = safety.evaluate(
                vehicle, grid, linear_mps, angular_rps, automatic=True
            )
            if decision.state in {"stopped", "fault"}:
                self.status = "blocked"
                self.reason = decision.reason
                vehicle.stop()
                return
            linear_mps, angular_rps = decision.linear_mps, decision.angular_rps
        vehicle.apply_drive(grid, linear_mps, angular_rps, now)
