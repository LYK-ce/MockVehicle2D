"""
cli.py — Unified CLI entry point for MockVehicle2D.

Usage:
    mockvehicle2d serve       Start WebSocket mock server
    mockvehicle2d visual      Launch Pygame visualization
    mockvehicle2d test        Run collision and Tmini scan tests
"""

import argparse
import asyncio
import sys


def _cmd_serve(_args):
    """Start the WebSocket mock server."""
    from mockvehicle2d.server import main as server_main

    asyncio.run(server_main())


def _cmd_visual(_args):
    """Launch the Pygame visualization."""
    from mockvehicle2d.visual import main as visual_main

    visual_main()


def _cmd_test(_args):
    """Run collision detection and local scan tests."""
    import os

    # Add repo root to path so tests/ is importable (src/ layout)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from tests.test_collision import main as collision_main
    from tests.test_scan import main as scan_main
    from tests.test_server_scan import main as server_scan_main

    sys.exit(collision_main() or scan_main() or server_scan_main())


def main():
    parser = argparse.ArgumentParser(
        prog="mockvehicle2d",
        description="2D vehicle simulator for Pictor WebSocket protocol testing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Start WebSocket mock server with YDLidar Tmini scans on ws://0.0.0.0:9090")
    sub.add_parser("visual", help="Launch Pygame visualization (W/S/A/D driving)")
    sub.add_parser("test", help="Run collision and Tmini scan tests")

    args = parser.parse_args()

    commands = {
        "serve": _cmd_serve,
        "visual": _cmd_visual,
        "test": _cmd_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
