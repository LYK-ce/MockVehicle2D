"""PathFollowingController — Follow an A* planned path with safety integration.

Pattern-matched from ``GotoController``: same ``start / update / cancel / snapshot``
interface, but iterates over ``WaypointFollower`` waypoints instead of driving a
straight line toward a single goal.
"""

from __future__ import annotations

import math

from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.pathfinding.waypoint_follower import WaypointFollower
from mockvehicle2d.safety import LocalSafetyRuntime
from mockvehicle2d.vehicle import Vehicle


class PathFollowingController:
    """Follow a pre-planned A* path with safety integration.

    Each ``update()`` tick:
      1. Advance the vehicle via safety or raw collision.
      2. If not active, return.
      3. If collided or safety-stopped → ``blocked``.
      4. Poll ``WaypointFollower`` for next command.
      5. Convert *forward / spin_left / spin_right* into velocity.
      6. Evaluate safety on the candidate velocities.
      7. Install the (possibly limited) velocities.
    """

    goal_tolerance_m = 0.1          # same as GotoController
    turn_in_place_threshold_rad = math.radians(20)
    waypoint_distance = 0.5         # advance to next waypoint
    angle_tolerance = math.radians(10)

    def __init__(self) -> None:
        self.status = "idle"
        self._follower: WaypointFollower | None = None
        self._path: list[tuple[int, int]] = []
        self.reason: str | None = None

    @property
    def control_mode(self) -> str:
        return "autonomous" if self.status == "active" else "manual"

    @property
    def path(self) -> list[tuple[int, int]]:
        return list(self._path)

    @property
    def goal(self) -> tuple[int, int] | None:
        if self._follower is not None:
            return self._follower.goal
        return None

    @property
    def current_target(self) -> tuple[int, int] | None:
        if self._follower is not None:
            return self._follower.current_target
        return None

    def start(self, path: list[tuple[int, int]]) -> None:
        """Begin following *path* (grid coordinates)."""
        if len(path) < 2:
            raise ValueError("path must contain at least two waypoints")
        self._path = path
        self._follower = WaypointFollower(
            path,
            waypoint_distance=self.waypoint_distance,
            angle_tolerance=self.angle_tolerance,
        )
        self.status = "active"
        self.reason = None

    def cancel(self, reason: str) -> None:
        if self.status == "active":
            self.status = "cancelled"
            self.reason = reason

    def snapshot(self) -> dict[str, object]:
        return {
            "status": self.status,
            "goal": (
                {"x_m": self._follower.goal[0], "y_m": self._follower.goal[1]}
                if self._follower is not None
                else None
            ),
            "current_target": (
                {"x": self.current_target[0], "y": self.current_target[1]}
                if self.current_target is not None
                else None
            ),
            "path_length": len(self._path),
            "reason": self.reason,
        }

    def update(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        now: float,
        safety: LocalSafetyRuntime | None = None,
    ) -> None:
        """Called each tick — advance vehicle, then steer along the path."""
        was_active = self.status == "active"

        # Step 1 — advance physics
        if safety is None:
            collided = vehicle.advance(grid, now)
            safety_stop = None
        else:
            result = safety.advance(vehicle, grid, now, automatic=was_active)
            collided = result.collided
            safety_stop = result.reason if result.stopped else None

        if not was_active:
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

        assert self._follower is not None

        # Step 2 — poll waypoint follower
        cmd, reached = self._follower.next_cmd(vehicle.x, vehicle.y, vehicle.yaw)

        if reached:
            self.status = "reached"
            self.reason = "goal_tolerance"
            vehicle.stop()
            return

        # Step 3 — convert WaypointFollower command to velocities
        linear_mps, angular_rps = 0.0, 0.0

        if cmd == "forward":
            # Drive forward toward current target
            tx, ty = self._follower.current_target
            desired_yaw = math.atan2(ty - vehicle.y, tx - vehicle.x)
            heading_error = math.atan2(
                math.sin(desired_yaw - vehicle.yaw),
                math.cos(desired_yaw - vehicle.yaw),
            )
            angular_rps = max(
                -vehicle.angular_speed,
                min(vehicle.angular_speed, 2 * heading_error),
            )
            if abs(heading_error) > self.turn_in_place_threshold_rad:
                linear_mps = 0.0
            else:
                distance = math.hypot(tx - vehicle.x, ty - vehicle.y)
                linear_mps = min(vehicle.linear_speed, distance)
        elif cmd == "spin_left":
            linear_mps = 0.0
            angular_rps = -vehicle.angular_speed
        elif cmd == "spin_right":
            linear_mps = 0.0
            angular_rps = vehicle.angular_speed

        # Step 4 — safety evaluation
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

        # Step 5 — apply
        vehicle.install_drive(linear_mps, angular_rps, now)
