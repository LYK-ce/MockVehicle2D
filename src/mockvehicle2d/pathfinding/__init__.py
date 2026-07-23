"""MockVehicle2D pathfinding algorithms.

Provides A* search on MapGrid and a WaypointFollower for autonomous navigation.

Usage::

    from mockvehicle2d.pathfinding import a_star_search, WaypointFollower

    path = a_star_search(grid, (10, 10), (200, 200))
    if path:
        follower = WaypointFollower(path)
        cmd, done = follower.next_cmd(vehicle.x, vehicle.y, vehicle.yaw)
"""

from mockvehicle2d.pathfinding.a_star import a_star_search
from mockvehicle2d.pathfinding.waypoint_follower import ARRIVAL_DISTANCE, ANGLE_TOLERANCE, WAYPOINT_DISTANCE, WaypointFollower

__all__ = [
    "a_star_search",
    "WaypointFollower",
    "ARRIVAL_DISTANCE",
    "WAYPOINT_DISTANCE",
    "ANGLE_TOLERANCE",
]
