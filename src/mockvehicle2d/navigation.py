"""Minimal local-odometry go-to-goal controller."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.vehicle import Vehicle

if TYPE_CHECKING:
    from mockvehicle2d.local_state import PoseEstimate


DEGRADED_LINEAR_SCALE = 0.5


class GotoController:
    """Drive directly toward one goal; stop on arrival, collision, or override."""

    goal_tolerance_m = 0.1
    turn_in_place_threshold_rad = math.radians(20)

    def __init__(self) -> None:
        self.status = "idle"
        self.goal: tuple[float, float] | None = None
        self.reported_goal: tuple[float, float] | None = None
        self.reason: str | None = None

    @property
    def control_mode(self) -> str:
        return "autonomous" if self.status == "active" else "manual"

    def start(
        self, x_m: float, y_m: float, *, reported_goal: tuple[float, float] | None = None
    ) -> None:
        self.goal = (x_m, y_m)
        self.reported_goal = self.goal if reported_goal is None else reported_goal
        self.status = "active"
        self.reason = None

    def cancel(self, reason: str) -> None:
        if self.status == "active":
            self.status = "cancelled"
            self.reason = reason

    def snapshot(self) -> dict[str, object]:
        goal = (
            None
            if self.reported_goal is None
            else {"x_m": self.reported_goal[0], "y_m": self.reported_goal[1]}
        )
        return {"status": self.status, "goal": goal, "reason": self.reason}

    def update(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        now: float,
        safety: LocalSafetyRuntime | None = None,
        *,
        pose: PoseEstimate | None = None,
        advance_result: SafetyAdvanceResult | None = None,
    ) -> None:
        was_active = self.status == "active"
        if advance_result is not None:
            collided = advance_result.collided
            safety_stop = advance_result.reason if advance_result.stopped else None
        elif safety is None:
            collided = vehicle.advance(grid, now)
            safety_stop = None
        else:
            result = safety.advance(vehicle, grid, now, automatic=was_active)
            collided = result.collided
            safety_stop = result.reason if result.stopped else None
        if not was_active:
            return
        if pose is not None and pose.quality == "lost":
            vehicle.stop()
            self.status = "blocked"
            self.reason = "localization_lost"
            return
        if collided:
            self.status = "blocked"
            self.reason = "collision"
            vehicle.stop()
            return
        if safety_stop is not None:
            self.status = "blocked"
            self.reason = safety_stop
            vehicle.stop()
            return

        assert self.goal is not None
        x_m, y_m, yaw_rad = (
            (pose.x_m, pose.y_m, pose.yaw_rad)
            if pose is not None
            else (vehicle.x, vehicle.y, vehicle.yaw)
        )
        dx, dy = self.goal[0] - x_m, self.goal[1] - y_m
        distance = math.hypot(dx, dy)
        if distance <= self.goal_tolerance_m:
            self.status = "reached"
            self.reason = "goal_tolerance"
            vehicle.stop()
            return

        desired_yaw = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(desired_yaw - yaw_rad), math.cos(desired_yaw - yaw_rad)
        )
        angular_rps = max(-vehicle.angular_speed, min(vehicle.angular_speed, 2 * heading_error))
        linear_mps = (
            0.0
            if abs(heading_error) > self.turn_in_place_threshold_rad
            else min(vehicle.linear_speed, distance)
        )
        if pose is not None and pose.quality == "degraded":
            linear_mps *= DEGRADED_LINEAR_SCALE
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
        vehicle.install_drive(linear_mps, angular_rps, now)
