"""
cli.py — Unified CLI entry point for MockVehicle2D.

Usage:
    mockvehicle2d serve       Start WebSocket mock server
    mockvehicle2d visual      Launch Pygame visualization
    mockvehicle2d test        Run motion, collision, and Tmini scan tests
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

    if args.llm:
        import asyncio
        from mockvehicle2d.instruction.llm_client import LLMClient
        client = LLMClient(model=args.model, schema_validator=schema_v)
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
    serve.add_argument("--linear-speed", type=_positive_float, default=0.5, metavar="MPS")
    serve.add_argument("--angular-speed", type=_positive_float, default=90.0, metavar="DEG_PER_SECOND")
    serve.add_argument("--vehicle-radius", type=_positive_float, default=0.5, metavar="METRES")
    serve.add_argument("--command-timeout", type=_positive_float, default=1.0, metavar="SECONDS")
    serve.add_argument("--nl", action="store_true", default=True,
                       help="Enable natural language command processing (default: on)")
    sub.add_parser("visual", help="Launch Pygame visualization (W/S/A/D driving)")
    sub.add_parser("test", help="Run motion, collision, and Tmini scan tests")
    pathfind = sub.add_parser("pathfind", help="Run A* pathfinding on generated map")
    pathfind.add_argument("--start", type=_coords, default=(10, 10), metavar="X,Y",
                          help="Start grid coordinate (default: 10,10)")
    pathfind.add_argument("--goal", type=_coords, default=(200, 200), metavar="X,Y",
                          help="Goal grid coordinate (default: 200,200)")
    pathfind.add_argument("--vehicle-radius", type=_positive_float, default=0.5, metavar="METRES")
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
    nl_cmd.add_argument("--fake", action="store_true", default=True,
                        help="Use FakeModelClient (default)")
    nl_cmd.add_argument("--llm", action="store_true",
                        help="Use LLMClient (requires local llama.cpp server)")
    nl_cmd.add_argument("--model", type=str, default="Qwen3-8B-Q4_K_M", metavar="MODEL",
                        help="Model name for LLMClient (default: Qwen3-8B-Q4_K_M)")

    args = parser.parse_args()

    commands = {
        "serve": _cmd_serve,
        "visual": _cmd_visual,
        "test": _cmd_test,
        "pathfind": _cmd_pathfind,
        "nl": _cmd_nl,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
