"""Canonical public configuration and telemetry use explicit SI units."""

import json
import inspect
import math
from pathlib import Path
import re
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mockvehicle2d.cli.main import _cmd_test, main
from mockvehicle2d.instruction.compiler import TaskCompiler
from mockvehicle2d.local_state import (
    AnchorSpec,
    AnchoredLocalState,
    ObservedGrid,
    PoseEstimate,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.pathfinding.d_star_lite import DStarLitePlanner
from mockvehicle2d.server import (
    _handle_nl_command,
    handle_command_message,
    handler,
    telemetry_messages,
)
from mockvehicle2d.vehicle import Vehicle


def _help(monkeypatch, capsys, command: str) -> set[str]:
    monkeypatch.setattr(sys, "argv", ["mockvehicle2d", command, "--help"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 0
    return set(re.findall(r"--[a-z][a-z0-9-]*", capsys.readouterr().out))


def test_serve_and_pathfind_flags_are_explicit_si(monkeypatch, capsys) -> None:
    serve = _help(monkeypatch, capsys, "serve")
    assert {
        "--linear-speed-mps",
        "--angular-speed-rps",
        "--vehicle-radius-m",
        "--command-timeout-s",
        "--anchor-x-m",
        "--anchor-y-m",
        "--anchor-yaw-rad",
        "--odom-translation-noise-m",
        "--odom-yaw-noise-rad",
    } <= serve
    assert not {
        "--linear-speed",
        "--angular-speed",
        "--vehicle-radius",
        "--command-timeout",
        "--anchor-x",
        "--anchor-y",
        "--anchor-yaw",
        "--odom-translation-noise",
        "--odom-yaw-noise",
    } & serve

    pathfind = _help(monkeypatch, capsys, "pathfind")
    assert {"--start-m", "--goal-m", "--vehicle-radius-m"} <= pathfind
    assert not {"--start", "--goal", "--vehicle-radius"} & pathfind


def test_pose_and_scan_have_canonical_si_fields_with_equal_legacy_aliases() -> None:
    vehicle = Vehicle(2.0, 3.0, yaw=0.4, now=0.0)
    state = AnchoredLocalState(
        AnchorSpec("telemetry-units", vehicle.x, vehicle.y, vehicle.yaw),
        truth_x_m=vehicle.x,
        truth_y_m=vehicle.y,
        truth_yaw_rad=vehicle.yaw,
        timestamp=0.0,
    )
    pose, scan = telemetry_messages(
        vehicle,
        MapGrid.from_wall_set(8, 8, set()),
        7,
        12.5,
        local_state=state,
    )
    assert {
        "timestamp_s",
        "x_m",
        "y_m",
        "z_m",
        "yaw_rad",
        "vx_mps",
        "vy_mps",
        "omega_rps",
    } <= pose.keys()
    for canonical, legacy in (
        ("timestamp_s", "ts"),
        ("x_m", "x"),
        ("y_m", "y"),
        ("z_m", "z"),
        ("yaw_rad", "yaw"),
        ("vx_mps", "vx"),
        ("vy_mps", "vy"),
        ("omega_rps", "omega"),
    ):
        assert pose[canonical] == pose[legacy]
    assert scan["timestamp_s"] == scan["ts"] == 12.5
    assert pose["localization"]["timestamp_s"] == pose["localization"]["timestamp"]


def test_navigation_and_compiler_paths_are_labelled_metric_waypoints() -> None:
    observed = ObservedGrid(AnchorSpec("units", 0.0, 0.0, 0.0))
    pose = PoseEstimate("units", 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), "nominal", 0.0, 0)
    navigation = GotoController()
    navigation.start(3.0, 0.0, local_map=observed, pose=pose, vehicle_radius_m=0.0)
    assert all(set(waypoint) == {"x_m", "y_m"} for waypoint in navigation.snapshot()["path"])

    grid = MapGrid.from_wall_set(8, 8, {(2, 0)})
    task = TaskCompiler(grid).compile(
        {
            "schema_version": "1.0",
            "intent": "goto_point",
            "timestamp": "2026-07-26T00:00:00+08:00",
            "parameters": {"x_m": 4.0, "y_m": 0.0},
        },
        {"pose": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}},
    )
    assert all(set(waypoint) == {"x_m", "y_m"} for waypoint in task["path"])

    rotation = TaskCompiler().compile(
        {
            "schema_version": "1.0",
            "intent": "rotate",
            "timestamp": "2026-07-26T00:00:00+08:00",
            "parameters": {"angle_rad": math.pi / 2, "direction": "left"},
        }
    )
    assert rotation["angle_rad"] == pytest.approx(math.pi / 2)
    assert "angle_deg" not in rotation
    json.dumps(rotation)


def test_navigation_fails_closed_without_estimated_pose_or_observed_map() -> None:
    observed = ObservedGrid(AnchorSpec("required-local", 0.0, 0.0, 0.0))
    pose = PoseEstimate(
        "required-local",
        0.0,
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        "nominal",
        0.0,
        0,
    )
    navigation = GotoController()
    navigation.start(3.0, 0.0, local_map=observed, pose=pose)
    vehicle = Vehicle(0.0, 0.0, now=0.0)
    vehicle.install_drive(0.5, 0.0, 0.0)

    navigation.update(vehicle, MapGrid(8, 8), 0.1)

    assert (navigation.status, navigation.reason) == (
        "blocked",
        "local_state_unavailable",
    )
    assert vehicle.body_velocities() == (0.0, 0.0)
    assert (vehicle.x, vehicle.y) == (0.0, 0.0)


def test_goto_rejects_missing_state_and_resource_exhausting_goal() -> None:
    vehicle = Vehicle(2.0, 3.0, now=0.0)
    grid = MapGrid(8, 8)

    missing = handle_command_message(
        '{"type":"goto","seq":1,"x_m":4,"y_m":3}',
        vehicle,
        grid,
        0.0,
        1.0,
        GotoController(),
    )
    assert missing["code"] == "goto_unavailable"

    state = AnchoredLocalState(
        AnchorSpec("bounded-goal", vehicle.x, vehicle.y, vehicle.yaw),
        truth_x_m=vehicle.x,
        truth_y_m=vehicle.y,
        truth_yaw_rad=vehicle.yaw,
        timestamp=0.0,
    )
    too_far = handle_command_message(
        '{"type":"goto","seq":2,"x_m":400,"y_m":3}',
        vehicle,
        grid,
        0.0,
        1.0,
        GotoController(),
        local_state=state,
    )
    assert too_far["type"] == "error"
    assert too_far["code"] == "invalid_goto"
    assert "maximum distance" in too_far["message"]


def test_shifted_anchor_direct_and_nl_goto_share_relative_bounds() -> None:
    vehicle = Vehicle(10.0, 10.0, now=0.0)
    grid = MapGrid(32, 32)
    state = AnchoredLocalState(
        AnchorSpec("shifted-anchor", 1000.0, 1000.0, 0.0),
        truth_x_m=vehicle.x,
        truth_y_m=vehicle.y,
        truth_yaw_rad=vehicle.yaw,
        timestamp=0.0,
    )
    direct = handle_command_message(
        '{"type":"goto","seq":3,"x_m":1001,"y_m":1000}',
        vehicle,
        grid,
        0.0,
        1.0,
        GotoController(),
        local_state=state,
    )
    assert direct["accepted"]

    from mockvehicle2d.instruction.llm_client import FakeModelClient
    from mockvehicle2d.instruction.state_machine import InstructionStateMachine
    from mockvehicle2d.instruction.validator import SchemaValidator, SemanticValidator

    near_navigation = GotoController()
    near = _handle_nl_command(
        {"type": "nl_command", "seq": 4, "text": "去坐标 (1001, 1000)"},
        vehicle,
        grid,
        near_navigation,
        1.0,
        0.0,
        FakeModelClient(),
        SchemaValidator(),
        SemanticValidator(None),
        InstructionStateMachine(),
        local_state=state,
    )
    assert near_navigation.status == "active"
    assert near[0]["accepted"]

    far_direct = handle_command_message(
        '{"type":"goto","seq":5,"x_m":1400,"y_m":1000}',
        vehicle,
        grid,
        0.0,
        1.0,
        GotoController(),
        local_state=state,
    )
    assert far_direct["type"] == "error"
    far_navigation = GotoController()
    far = _handle_nl_command(
        {"type": "nl_command", "seq": 6, "text": "去坐标 (1400, 1000)"},
        vehicle,
        grid,
        far_navigation,
        1.0,
        0.0,
        FakeModelClient(),
        SchemaValidator(),
        SemanticValidator(None),
        InstructionStateMachine(),
        local_state=state,
    )
    assert far_navigation.status == "blocked"
    assert any(reply.get("status") == "blocked" for reply in far)


def test_test_command_runs_full_pytest_and_propagates_failure(monkeypatch) -> None:
    captured = {}

    def fail(command, *, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        return type("Result", (), {"returncode": 7})()

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(SystemExit) as stopped:
        _cmd_test(None)
    assert stopped.value.code == 7
    assert captured["command"][:3] == [sys.executable, "-m", "pytest"]
    assert "-p" in captured["command"]
    assert "no:cacheprovider" in captured["command"]
    assert Path(captured["command"][-1]).name == "tests"
    assert Path(captured["cwd"]) == REPO_ROOT


def test_production_navigation_source_has_no_truth_or_legacy_astar_route() -> None:
    navigation_source = inspect.getsource(GotoController)
    planner_source = inspect.getsource(DStarLitePlanner)
    nl_source = inspect.getsource(_handle_nl_command)
    handler_source = inspect.getsource(handler)

    assert all(
        token not in navigation_source
        for token in ("vehicle.x", "vehicle.y", "vehicle.yaw", "simulator_ground_truth")
    )
    assert all(
        token not in planner_source
        for token in ("MapGrid", "vehicle.", "simulator_ground_truth")
    )
    assert "task_compiler.compile" not in nl_source
    assert "path_following.start" not in nl_source
    assert "PathFollowingController()" not in handler_source
