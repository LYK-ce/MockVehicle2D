"""
waypoint_follower.py — Convert a grid path into Vehicle commands.

Tracks progress along a list of waypoints and returns the appropriate cmd
(forward / spin_left / spin_right / stop) to guide a Vehicle along the path.
"""

from __future__ import annotations

import math

# Default thresholds
ARRIVAL_DISTANCE = 0.5      # metres — considered "arrived" at goal
WAYPOINT_DISTANCE = 0.5     # metres — advance to next waypoint
ANGLE_TOLERANCE = math.radians(10)  # radians — "facing the right way"


class WaypointFollower:
    """Stateful follower that converts a path into per-frame commands.

    Call ``next_cmd(x, y, yaw)`` each physics tick; it returns a command name
    and a boolean ``reached_goal``.
    """

    def __init__(
        self,
        path: list[tuple[float, float]],
        *,
        arrival_distance: float = ARRIVAL_DISTANCE,
        waypoint_distance: float = WAYPOINT_DISTANCE,
        angle_tolerance: float = ANGLE_TOLERANCE,
    ) -> None:
        if len(path) < 2:
            raise ValueError("path must contain at least two waypoints")
        self._path = path
        self._idx = 1          # next waypoint index (0 = start)
        self._goal = path[-1]
        self._arrival_distance = arrival_distance
        self._waypoint_distance = waypoint_distance
        self._angle_tolerance = angle_tolerance

    @property
    def goal(self) -> tuple[float, float]:
        return self._goal

    @property
    def current_target(self) -> tuple[float, float]:
        return self._path[self._idx]

    @property
    def path(self) -> list[tuple[float, float]]:
        return list(self._path)

    def next_cmd(self, x: float, y: float, yaw: float) -> tuple[str, bool]:
        """Return ``(cmd, reached_goal)`` for the current vehicle pose.

        *cmd* is one of ``forward``, ``spin_left``, ``spin_right``, ``stop``.
        *reached_goal* is ``True`` when the vehicle is within *arrival_distance*
        of the final waypoint.
        """
        # Check if we've reached the final goal.
        dist_to_goal = math.hypot(x - self._goal[0], y - self._goal[1])
        if dist_to_goal < self._arrival_distance:
            return "stop", True

        # Advance waypoint index when within range of current target.
        tx, ty = self._path[self._idx]
        dist_to_target = math.hypot(x - tx, y - ty)
        while dist_to_target < self._waypoint_distance and self._idx < len(self._path) - 1:
            self._idx += 1
            tx, ty = self._path[self._idx]
            dist_to_target = math.hypot(x - tx, y - ty)

        # Compute angle to target and decide turn / forward.
        desired_yaw = math.atan2(ty - y, tx - x)
        delta = _normalize_angle(desired_yaw - yaw)

        if abs(delta) < self._angle_tolerance:
            return "forward", False
        return "spin_left" if delta > 0 else "spin_right", False

    def reset(self, path: list[tuple[float, float]]) -> None:
        """Replace the current path and restart from its first segment."""
        if len(path) < 2:
            raise ValueError("path must contain at least two waypoints")
        self._path = path
        self._idx = 1
        self._goal = path[-1]


def _normalize_angle(a: float) -> float:
    """Wrap *a* to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))
