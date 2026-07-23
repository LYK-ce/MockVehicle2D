"""
cli.py — Unified CLI entry point for MockVehicle2D.

Usage:
    mockvehicle2d serve       Start WebSocket mock server
    mockvehicle2d visual      Launch Pygame visualization
    mockvehicle2d test        Run motion, collision, and Tmini scan tests
"""

import argparse
import asyncio
import math
import sys


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535")
    return port


def _vehicle_id(value: str) -> str:
    from mockvehicle2d.server import validate_vehicle_id

    try:
        return validate_vehicle_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _cmd_serve(args):
    """Start the WebSocket mock server."""
    from mockvehicle2d.server import main as server_main

    asyncio.run(
        server_main(
            port=args.port,
            vehicle_id=args.vehicle_id,
            linear_speed=args.linear_speed,
            angular_speed=math.radians(args.angular_speed),
            radius=args.vehicle_radius,
            command_timeout=args.command_timeout,
        )
    )


def _cmd_visual(_args):
    """Launch the Pygame visualization."""
    from mockvehicle2d.visual import main as visual_main

    visual_main()


def _coords(value: str) -> tuple[int, int]:
    """Parse 'x,y' coordinate pair."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("coordinates must be in the form x,y")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("coordinates must be integers") from error


def _cmd_pathfind(args):
    """Run A* pathfinding on a generated map and print the path."""
    import math
    import random
    from mockvehicle2d.map_grid import MapGrid
    from mockvehicle2d.pathfinding import a_star_search

    # Generate a deterministic map (same as server default), but clear
    # walls around the start position so the vehicle can spawn safely
    # (matching server.generate_map's spawn-area clearing).
    random.seed(42)
    sx, sy = args.start
    margin = math.ceil(args.vehicle_radius) + 1  # 2-cell radius for r=0.5
    clear_x0, clear_x1 = sx - margin, sx + margin
    clear_y0, clear_y1 = sy - margin, sy + margin

    voxels = []
    for gx in range(256):
        for gy in range(256):
            in_clear = clear_x0 <= gx <= clear_x1 and clear_y0 <= gy <= clear_y1
            is_wall = random.random() < 0.05
            voxels.append({"gx": gx, "gy": gy, "gz": 0,
                           "state": 1 if is_wall and not in_clear else 0, "conf": 1.0})
    grid = MapGrid.from_voxels(voxels)

    path = a_star_search(grid, args.start, args.goal, vehicle_radius=args.vehicle_radius)
    if path is None:
        print(f"No path found from {args.start} to {args.goal}")
        sys.exit(1)
    print(f"Path found: {len(path)} waypoints")
    if args.verbose:
        for i, wp in enumerate(path):
            print(f"  [{i}] {wp}")
    sys.exit(0)


def _cmd_test(_args):
    """Run deterministic motion, collision, and local scan tests."""
    import os

    # Add repo root to path so tests/ is importable (src/ layout)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from tests.test_collision import main as collision_main
    from tests.test_goto import main as goto_main
    from tests.test_safety import main as safety_main
    from tests.test_safety_runtime import main as safety_runtime_main
    from tests.test_scan import main as scan_main
    from tests.test_server_scan import main as server_scan_main
    from tests.test_vehicle import main as vehicle_main

    sys.exit(
        collision_main()
        or scan_main()
        or vehicle_main()
        or goto_main()
        or safety_main()
        or safety_runtime_main()
        or server_scan_main()
    )


def main():
    parser = argparse.ArgumentParser(
        prog="mockvehicle2d",
        description="2D vehicle simulator for Pictor WebSocket protocol testing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start controllable WebSocket server with YDLidar Tmini scans")
    serve.add_argument("--port", type=_port, default=19090, metavar="PORT")
    serve.add_argument("--vehicle-id", type=_vehicle_id, default="mock_vehicle_01", metavar="ID")
    serve.add_argument("--linear-speed", type=_positive_float, default=0.5, metavar="MPS")
    serve.add_argument("--angular-speed", type=_positive_float, default=90.0, metavar="DEG_PER_SECOND")
    serve.add_argument("--vehicle-radius", type=_positive_float, default=0.5, metavar="METRES")
    serve.add_argument("--command-timeout", type=_positive_float, default=1.0, metavar="SECONDS")
    sub.add_parser("visual", help="Launch Pygame visualization (W/S/A/D driving)")
    sub.add_parser("test", help="Run motion, collision, and Tmini scan tests")
    pathfind = sub.add_parser("pathfind", help="Run A* pathfinding on generated map")
    pathfind.add_argument("--start", type=_coords, default=(10, 10), metavar="X,Y",
                          help="Start grid coordinate (default: 10,10)")
    pathfind.add_argument("--goal", type=_coords, default=(200, 200), metavar="X,Y",
                          help="Goal grid coordinate (default: 200,200)")
    pathfind.add_argument("--vehicle-radius", type=_positive_float, default=0.5, metavar="METRES")
    pathfind.add_argument("--verbose", "-v", action="store_true", help="Print each waypoint")

    args = parser.parse_args()

    commands = {
        "serve": _cmd_serve,
        "visual": _cmd_visual,
        "test": _cmd_test,
        "pathfind": _cmd_pathfind,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
