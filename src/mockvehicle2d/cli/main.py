"""MockVehicle2D command-line interface."""

import argparse
import asyncio
import math
from pathlib import Path
import random


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def _nonnegative_float(value: str) -> float:
    number = _finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return number


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error


def _port(value: str) -> int:
    port = _integer(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be from 1 to 65535")
    return port


def _vehicle_id(value: str) -> str:
    from mockvehicle2d.server import validate_vehicle_id

    try:
        return validate_vehicle_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _coords_m(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("coordinates must be X_M,Y_M")
    try:
        coordinates = float(parts[0]), float(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("coordinates must be metre numbers") from error
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise argparse.ArgumentTypeError("coordinates must be finite")
    return coordinates


def _cmd_serve(args) -> None:
    from mockvehicle2d.server import main as server_main

    asyncio.run(
        server_main(
            port=args.port,
            vehicle_id=args.vehicle_id,
            linear_speed=args.linear_speed_mps,
            angular_speed=args.angular_speed_rps,
            radius=args.vehicle_radius_m,
            command_timeout=args.command_timeout_s,
            mission_capacity=args.mission_capacity,
            anchor_id=args.anchor_id,
            anchor_x_m=args.anchor_x_m,
            anchor_y_m=args.anchor_y_m,
            anchor_yaw_rad=args.anchor_yaw_rad,
            odometry_translation_noise_stddev_m=args.odom_translation_noise_m,
            odometry_yaw_noise_stddev_rad=args.odom_yaw_noise_rad,
            odometry_seed=args.odom_seed,
        )
    )


def _cmd_pathfind(args) -> None:
    from mockvehicle2d.map_grid import MapGrid
    from mockvehicle2d.pathfinding import a_star_search

    rng = random.Random(42)
    resolution_m = 1.0
    start = tuple(math.floor(value / resolution_m) for value in args.start_m)
    goal = tuple(math.floor(value / resolution_m) for value in args.goal_m)
    margin = math.ceil(args.vehicle_radius_m / resolution_m) + 1
    voxels = []
    for gx in range(256):
        for gy in range(256):
            near_start = (
                start[0] - margin <= gx <= start[0] + margin
                and start[1] - margin <= gy <= start[1] + margin
            )
            voxels.append(
                {
                    "gx": gx,
                    "gy": gy,
                    "gz": 0,
                    "state": int(rng.random() < 0.05 and not near_start),
                    "conf": 1.0,
                }
            )
    path = a_star_search(
        MapGrid.from_voxels(voxels),
        start,
        goal,
        vehicle_radius=args.vehicle_radius_m,
        resolution_m=resolution_m,
    )
    if path is None:
        print(f"No path found from {args.start_m} m to {args.goal_m} m")
        raise SystemExit(1)
    print(f"Path found: {len(path)} waypoints")
    if args.verbose:
        for index, waypoint in enumerate(path):
            print(
                f"  [{index}] x_m={waypoint[0] * resolution_m:.3f} "
                f"y_m={waypoint[1] * resolution_m:.3f}"
            )


def _cmd_fleet(args) -> None:
    from mockvehicle2d.fleet import main as fleet_main

    asyncio.run(
        fleet_main(
            args.scenario,
            linear_speed=args.linear_speed_mps,
            angular_speed=args.angular_speed_rps,
            radius=args.vehicle_radius_m,
            command_timeout=args.command_timeout_s,
            mission_capacity=args.mission_capacity,
            odometry_translation_noise_stddev_m=args.odom_translation_noise_m,
            odometry_yaw_noise_stddev_rad=args.odom_yaw_noise_rad,
            odometry_seed=args.odom_seed,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mockvehicle2d",
        description="2D autonomous robot controller simulator",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="Start the WebSocket controller")
    serve.add_argument("--port", type=_port, default=19090, metavar="PORT")
    serve.add_argument(
        "--vehicle-id",
        type=_vehicle_id,
        default="mock_vehicle_01",
        metavar="ID",
    )
    serve.add_argument(
        "--linear-speed-mps",
        type=_positive_float,
        default=0.5,
        metavar="MPS",
    )
    serve.add_argument(
        "--angular-speed-rps",
        type=_positive_float,
        default=math.pi / 2,
        metavar="RPS",
    )
    serve.add_argument(
        "--vehicle-radius-m",
        type=_positive_float,
        default=0.5,
        metavar="M",
    )
    serve.add_argument(
        "--command-timeout-s",
        type=_positive_float,
        default=1.0,
        metavar="S",
    )
    serve.add_argument(
        "--mission-capacity",
        type=_positive_integer,
        default=16,
        metavar="N",
    )
    serve.add_argument("--anchor-id", type=_vehicle_id, default=None, metavar="ID")
    serve.add_argument("--anchor-x-m", type=_finite_float, default=10.0, metavar="M")
    serve.add_argument("--anchor-y-m", type=_finite_float, default=10.0, metavar="M")
    serve.add_argument(
        "--anchor-yaw-rad",
        type=_finite_float,
        default=0.0,
        metavar="RAD",
    )
    serve.add_argument(
        "--odom-translation-noise-m",
        type=_nonnegative_float,
        default=0.0,
        metavar="M",
    )
    serve.add_argument(
        "--odom-yaw-noise-rad",
        type=_nonnegative_float,
        default=0.0,
        metavar="RAD",
    )
    serve.add_argument("--odom-seed", type=_integer, default=0, metavar="INTEGER")

    fleet = subcommands.add_parser(
        "fleet",
        help="Start 1-4 isolated vehicles in one deterministic shared world",
    )
    fleet.add_argument("--scenario", type=Path, required=True, metavar="JSON")
    fleet.add_argument(
        "--linear-speed-mps",
        type=_positive_float,
        default=0.5,
        metavar="MPS",
    )
    fleet.add_argument(
        "--angular-speed-rps",
        type=_positive_float,
        default=math.pi / 2,
        metavar="RPS",
    )
    fleet.add_argument(
        "--vehicle-radius-m",
        type=_positive_float,
        default=0.5,
        metavar="M",
    )
    fleet.add_argument(
        "--command-timeout-s",
        type=_positive_float,
        default=1.0,
        metavar="S",
    )
    fleet.add_argument(
        "--mission-capacity",
        type=_positive_integer,
        default=16,
        metavar="N",
    )
    fleet.add_argument(
        "--odom-translation-noise-m",
        type=_nonnegative_float,
        default=0.0,
        metavar="M",
    )
    fleet.add_argument(
        "--odom-yaw-noise-rad",
        type=_nonnegative_float,
        default=0.0,
        metavar="RAD",
    )
    fleet.add_argument("--odom-seed", type=_integer, default=0, metavar="INTEGER")

    pathfind = subcommands.add_parser(
        "pathfind",
        help="Run the full-truth A* debug tool",
    )
    pathfind.add_argument(
        "--start-m",
        type=_coords_m,
        default=(10.0, 10.0),
        metavar="X_M,Y_M",
    )
    pathfind.add_argument(
        "--goal-m",
        type=_coords_m,
        default=(200.0, 200.0),
        metavar="X_M,Y_M",
    )
    pathfind.add_argument(
        "--vehicle-radius-m",
        type=_positive_float,
        default=0.5,
        metavar="M",
    )
    pathfind.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    commands = {
        "serve": _cmd_serve,
        "fleet": _cmd_fleet,
        "pathfind": _cmd_pathfind,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
