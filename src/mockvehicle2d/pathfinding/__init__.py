"""MockVehicle2D pathfinding algorithms.

Provides A* search on MapGrid, WaypointFollower for autonomous navigation,
and PathFollowingController for safety-integrated path execution.

Usage::

    from mockvehicle2d.pathfinding import a_star_search, WaypointFollower, PathFollowingController

    path = a_star_search(grid, (10, 10), (200, 200))
    if path:
        controller = PathFollowingController()
        controller.start(path)
        controller.update(vehicle, grid, now, safety)
"""

from mockvehicle2d.pathfinding.a_star import a_star_search
from mockvehicle2d.pathfinding.path_following_controller import PathFollowingController
from mockvehicle2d.pathfinding.waypoint_follower import ARRIVAL_DISTANCE, ANGLE_TOLERANCE, WAYPOINT_DISTANCE, WaypointFollower

__all__ = [
    "a_star_search",
    "PathFollowingController",
    "WaypointFollower",
    "ARRIVAL_DISTANCE",
    "WAYPOINT_DISTANCE",
    "ANGLE_TOLERANCE",
]
