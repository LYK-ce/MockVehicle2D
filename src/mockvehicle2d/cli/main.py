"""
cli.py — Unified CLI entry point for MockVehicle2D.

Usage:
    mockvehicle2d serve       Start WebSocket mock server
    mockvehicle2d visual      Launch Pygame visualization
    mockvehicle2d nl          Parse natural language vehicle commands
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


def _integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error


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
            linear_speed=args.linear_speed_mps,
            angular_speed=args.angular_speed_rps,
            radius=args.vehicle_radius_m,
            command_timeout=args.command_timeout_s,
            anchor_id=args.anchor_id,
            anchor_x_m=args.anchor_x_m,
            anchor_y_m=args.anchor_y_m,
            anchor_yaw_rad=args.anchor_yaw_rad,
            odometry_translation_noise_stddev_m=args.odom_translation_noise_m,
            odometry_yaw_noise_stddev_rad=args.odom_yaw_noise_rad,
            odometry_seed=args.odom_seed,
        )
    )


def _cmd_visual(_args):
    """Launch the Pygame visualization."""
    from mockvehicle2d.visual import main as visual_main

    visual_main()


def _coords_m(value: str) -> tuple[float, float]:
    """Parse one finite ``x_m,y_m`` coordinate pair."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("coordinates must be in the form x,y")
    try:
        coordinates = float(parts[0]), float(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("coordinates must be metre numbers") from error
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise argparse.ArgumentTypeError("coordinates must be finite")
    return coordinates


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
    resolution_m = 1.0
    start = tuple(math.floor(value / resolution_m) for value in args.start_m)
    goal = tuple(math.floor(value / resolution_m) for value in args.goal_m)
    sx, sy = start
    margin = math.ceil(args.vehicle_radius_m / resolution_m) + 1
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

    path = a_star_search(
        grid,
        start,
        goal,
        vehicle_radius=args.vehicle_radius_m,
        resolution_m=resolution_m,
    )
    if path is None:
        print(f"No path found from {args.start_m} m to {args.goal_m} m")
        sys.exit(1)
    print(f"Path found: {len(path)} waypoints")
    if args.verbose:
        for i, wp in enumerate(path):
            print(
                f"  [{i}] x_m={wp[0] * resolution_m:.3f} "
                f"y_m={wp[1] * resolution_m:.3f}"
            )
    sys.exit(0)


def _cmd_nl(args):
    """Parse a natural language vehicle command through the offline pipeline."""
    import json
    import os
    import random

    from mockvehicle2d.instruction.llm_client import FakeModelClient
    from mockvehicle2d.instruction.validator import (
        SchemaValidator,
        SemanticValidator,
        run_validation_pipeline,
    )
    from mockvehicle2d.map_grid import MapGrid

    # Build a deterministic test map
    random.seed(42)
    voxels = []
    for gx in range(256):
        for gy in range(256):
            is_wall = random.random() < 0.05
            voxels.append({
                "gx": gx, "gy": gy, "gz": 0,
                "state": 1 if is_wall else 0, "conf": 1.0,
            })
    grid = MapGrid.from_voxels(voxels)

    schema_v = SchemaValidator()
    semantic_v = SemanticValidator(grid)

    # --- interactive mode ---
    if args.interactive:
        client = FakeModelClient()
        print("NL Instruction REPL — type 'quit' to exit")
        while True:
            try:
                text = input("nl> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.strip().lower() in ("quit", "exit", "q"):
                break
            if not text.strip():
                continue
            instruction = client.parse(text)
            if instruction is None:
                print("  parse failed: no result")
                continue
            print(f"  intent: {instruction.get('intent')}")
            print(f"  params: {json.dumps(instruction.get('parameters', {}), ensure_ascii=False)}")
            result = run_validation_pipeline(
                instruction, schema_validator=schema_v, semantic_validator=semantic_v
            )
            if result.valid:
                print(f"  ✓ valid")
            else:
                print(f"  ✗ {result.layer}: {result.message}")
        return

    # --- eval mode ---
    if args.eval:
        dataset_path = args.dataset
        if not os.path.exists(dataset_path):
            print(f"error: dataset not found: {dataset_path}")
            sys.exit(1)
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
        return _run_eval(dataset, schema_v, semantic_v)

    # --- single command mode ---
    text = args.text
    if text is None:
        print("error: missing NL text (use --interactive or provide text argument)")
        sys.exit(1)

    if args.vllm:
        import asyncio
        from mockvehicle2d.instruction.llm_client import VLLMClient
        client = VLLMClient()
        instruction = asyncio.run(client.parse(text))
    else:
        client = FakeModelClient()
        instruction = client.parse(text)

    if instruction is None:
        print("parse failed: no result")
        sys.exit(1)

    print(json.dumps(instruction, ensure_ascii=False, indent=2))

    result = run_validation_pipeline(
        instruction, schema_validator=schema_v, semantic_validator=semantic_v
    )
    if result.valid:
        print("validation: ✓ passed")
        sys.exit(0)
    else:
        print(f"validation: ✗ {result.layer} — {result.message}")
        sys.exit(1)


def _run_eval(dataset, schema_v, semantic_v):
    """Run offline evaluation: compare FakeModelClient parse vs expected."""
    import json

    from mockvehicle2d.instruction.llm_client import FakeModelClient
    from mockvehicle2d.instruction.validator import run_validation_pipeline

    client = FakeModelClient()
    total = len(dataset)
    intent_correct = 0
    schema_pass = 0
    semantic_pass = 0
    details: list[dict] = []

    for entry in dataset:
        text = entry["input"]
        expected = entry["expected"]
        instruction = client.parse(text)
        parsed_intent = instruction.get("intent") if instruction else None
        expected_intent = expected.get("intent")

        intent_ok = parsed_intent == expected_intent
        if intent_ok:
            intent_correct += 1

        schema_ok = False
        semantic_ok = False
        if instruction is not None:
            result = run_validation_pipeline(
                instruction, schema_validator=schema_v, semantic_validator=semantic_v
            )
            schema_ok = result.layer != "schema"
            semantic_ok = result.valid

        if schema_ok:
            schema_pass += 1
        if semantic_ok:
            semantic_pass += 1

        details.append({
            "input": text,
            "expected_intent": expected_intent,
            "parsed_intent": parsed_intent,
            "intent_ok": intent_ok,
            "schema_ok": schema_ok,
            "semantic_ok": semantic_ok,
        })

    # Print metrics
    intent_acc = intent_correct / total * 100 if total else 0
    schema_acc = schema_pass / total * 100 if total else 0
    semantic_acc = semantic_pass / total * 100 if total else 0

    print(f"Evaluation on {total} entries:")
    print(f"  Intent accuracy:  {intent_correct}/{total} ({intent_acc:.1f}%)")
    print(f"  Schema pass:      {schema_pass}/{total} ({schema_acc:.1f}%)")
    print(f"  Semantic pass:    {semantic_pass}/{total} ({semantic_acc:.1f}%)")

    # Print failures
    failures = [d for d in details if not d["intent_ok"]]
    if failures:
        print(f"\nIntent mismatches ({len(failures)}):")
        for d in failures[:10]:
            print(f"  '{d['input']}' → expected {d['expected_intent']}, got {d['parsed_intent']}")

    sys.exit(0 if intent_acc >= 90 else 1)


def main():
    parser = argparse.ArgumentParser(
        prog="mockvehicle2d",
        description="2D vehicle simulator for Pictor WebSocket protocol testing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start controllable WebSocket server with YDLidar Tmini scans")
    serve.add_argument("--port", type=_port, default=19090, metavar="PORT")
    serve.add_argument("--vehicle-id", type=_vehicle_id, default="mock_vehicle_01", metavar="ID")
    serve.add_argument("--linear-speed-mps", type=_positive_float, default=0.5, metavar="MPS")
    serve.add_argument("--angular-speed-rps", type=_positive_float, default=math.pi / 2, metavar="RPS")
    serve.add_argument("--vehicle-radius-m", type=_positive_float, default=0.5, metavar="M")
    serve.add_argument("--command-timeout-s", type=_positive_float, default=1.0, metavar="S")
    serve.add_argument(
        "--anchor-id",
        type=_vehicle_id,
        default=None,
        metavar="ID",
        help="Anchor id (default: <vehicle-id>_anchor)",
    )
    serve.add_argument("--anchor-x-m", type=_finite_float, default=10.0, metavar="M")
    serve.add_argument("--anchor-y-m", type=_finite_float, default=10.0, metavar="M")
    serve.add_argument("--anchor-yaw-rad", type=_finite_float, default=0.0, metavar="RAD")
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
    sub.add_parser("visual", help="Launch Pygame visualization (W/S/A/D driving)")
    pathfind = sub.add_parser("pathfind", help="Run A* pathfinding on generated map")
    pathfind.add_argument("--start-m", type=_coords_m, default=(10.0, 10.0), metavar="X_M,Y_M",
                          help="Start position in metres (default: 10,10)")
    pathfind.add_argument("--goal-m", type=_coords_m, default=(200.0, 200.0), metavar="X_M,Y_M",
                          help="Goal position in metres (default: 200,200)")
    pathfind.add_argument("--vehicle-radius-m", type=_positive_float, default=0.5, metavar="M")
    pathfind.add_argument("--verbose", "-v", action="store_true", help="Print each waypoint")

    # ── nl subcommand ───────────────────────────────────────
    nl_cmd = sub.add_parser("nl", help="Parse natural language vehicle commands")
    nl_cmd.add_argument("text", nargs="?", default=None, metavar="TEXT",
                        help="Natural language command text")
    nl_cmd.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive REPL mode")
    nl_cmd.add_argument("--eval", action="store_true",
                        help="Run offline evaluation against a dataset")
    nl_cmd.add_argument("--dataset", type=str, default="tests/nl_eval.json", metavar="PATH",
                        help="Path to evaluation dataset JSON")
    nl_cmd.add_argument("--vllm", action="store_true",
                        help="Use VLLMClient (requires local vLLM)")

    args = parser.parse_args()

    commands = {
        "serve": _cmd_serve,
        "visual": _cmd_visual,
        "pathfind": _cmd_pathfind,
        "nl": _cmd_nl,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
