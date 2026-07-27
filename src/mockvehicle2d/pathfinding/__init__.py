"""MockVehicle2D pathfinding algorithms.

Provides A* search on MapGrid, WaypointFollower for autonomous navigation,
and PathFollowingController for safety-integrated path execution.

Usage::

    from mockvehicle2d.pathfinding import a_star_search, WaypointFollower, PathFollowingController

    path_cells = a_star_search(grid, (10, 10), (200, 200))
    if path_cells:
        path_m = [
            ((gx + 0.5) * grid.resolution_m, (gy + 0.5) * grid.resolution_m)
            for gx, gy in path_cells
        ]
        controller = PathFollowingController()
        controller.start(path_m)
        controller.update(vehicle, grid, now, safety)
"""

from mockvehicle2d.pathfinding.a_star import a_star_search
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.pathfinding.path_following_controller import PathFollowingController
from mockvehicle2d.pathfinding.waypoint_follower import ARRIVAL_DISTANCE, ANGLE_TOLERANCE, WAYPOINT_DISTANCE, WaypointFollower

__all__ = [
    "a_star_search",
    "DStarLitePlanner",
    "PathFollowingController",
    "WaypointFollower",
    "ARRIVAL_DISTANCE",
    "WAYPOINT_DISTANCE",
    "ANGLE_TOLERANCE",
]
