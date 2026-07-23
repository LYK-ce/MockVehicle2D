#!/usr/bin/env python3
"""Controllable 2D vehicle and Tmini-style WebSocket simulator."""

import asyncio
import json
import math
import random
import re
import signal
import time

from mockvehicle2d.map_grid import MapGrid, VOID
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.safety import LocalSafetyRuntime
from mockvehicle2d.scan import TMINI_SCAN_CONFIG, scan_message
from mockvehicle2d.vehicle import COMMANDS, Vehicle


HOST = "0.0.0.0"
PORT = 19090
DEFAULT_VEHICLE_ID = "mock_vehicle_01"
SPAWN_X = 10.0
SPAWN_Y = 10.0
MAX_JSON_INTEGER_DIGITS = 4300
VEHICLE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


class CommandMessageError(ValueError):
    def __init__(self, code: str, message: str, seq: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.seq = seq


def validate_vehicle_id(value: str) -> str:
    if not VEHICLE_ID_PATTERN.fullmatch(value):
        raise ValueError("vehicle id must be 1-64 ASCII letters, digits, dots, underscores, or hyphens")
    return value


def _next_deadline(deadline: float, now: float, period: float) -> float:
    if now < deadline:
        return deadline
    return deadline + (math.floor((now - deadline) / period) + 1) * period


def _safe_seq(message: object) -> int | None:
    if not isinstance(message, dict):
        return None
    seq = message.get("seq")
    return seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 0 else None


def _bounded_json_int(value: str) -> int:
    if len(value.lstrip("-")) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError("JSON integer is too long")
    return int(value)


def _decode_message(raw: object) -> dict[str, object]:
    if not isinstance(raw, str):
        raise CommandMessageError("invalid_json_text", "command must be a JSON text message")
    try:
        message = json.loads(raw, parse_int=_bounded_json_int)
    except (ValueError, RecursionError) as error:
        raise CommandMessageError("invalid_json", "command is not valid JSON text") from error
    if not isinstance(message, dict):
        raise CommandMessageError("invalid_message", "command JSON must be an object")
    return message


def _parse_command_object(message: dict[str, object]) -> tuple[str, int | None]:
    seq = _safe_seq(message)
    if set(message) == {"cmd"}:
        command = message["cmd"]
        if not isinstance(command, str) or command not in COMMANDS:
            raise CommandMessageError("invalid_cmd", "unsupported cmd", None)
        return command, None

    if "type" not in message:
        raise CommandMessageError("missing_type", "canonical command requires type", seq)
    if message["type"] != "cmd":
        raise CommandMessageError("invalid_type", "type must be cmd", seq)
    if "seq" not in message:
        raise CommandMessageError("missing_seq", "canonical command requires seq", None)
    if seq is None:
        raise CommandMessageError("invalid_seq", "seq must be a non-negative integer", None)
    if "cmd" not in message or not isinstance(message["cmd"], str) or message["cmd"] not in COMMANDS:
        raise CommandMessageError("invalid_cmd", "unsupported cmd", seq)
    if set(message) != {"type", "seq", "cmd"}:
        raise CommandMessageError("invalid_fields", "canonical command has unexpected fields", seq)
    return message["cmd"], seq


def parse_command_message(raw: object) -> tuple[str, int | None]:
    """Validate canonical commands and the exact legacy ``{"cmd": ...}`` form."""
    return _parse_command_object(_decode_message(raw))


def _parse_drive_object(
    message: dict[str, object], linear_limit: float, angular_limit: float
) -> tuple[float, float, int]:
    seq = _safe_seq(message)
    if message.get("type") != "drive":
        raise CommandMessageError("invalid_type", "type must be drive", seq)
    if "seq" not in message:
        raise CommandMessageError("missing_seq", "drive command requires seq", None)
    if seq is None:
        raise CommandMessageError("invalid_seq", "seq must be a non-negative integer", None)
    if set(message) != {"type", "seq", "linear_mps", "angular_rps"}:
        raise CommandMessageError("invalid_fields", "drive command has missing or unexpected fields", seq)

    linear_mps = message["linear_mps"]
    angular_rps = message["angular_rps"]
    values = (linear_mps, angular_rps)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise CommandMessageError("invalid_drive", "drive velocities must be JSON numbers", seq)
    if any(isinstance(value, float) and not math.isfinite(value) for value in values):
        raise CommandMessageError("invalid_drive", "drive velocities must be finite", seq)
    if abs(linear_mps) > linear_limit or abs(angular_rps) > angular_limit:
        raise CommandMessageError("drive_out_of_range", "drive velocities exceed configured limits", seq)
    return float(linear_mps), float(angular_rps), seq


def parse_drive_message(
    raw: object, linear_limit: float, angular_limit: float
) -> tuple[float, float, int]:
    """Validate one bounded continuous-velocity command."""
    return _parse_drive_object(_decode_message(raw), linear_limit, angular_limit)


def _parse_goto_object(message: dict[str, object]) -> tuple[float, float, int]:
    seq = _safe_seq(message)
    if message.get("type") != "goto":
        raise CommandMessageError("invalid_type", "type must be goto", seq)
    if "seq" not in message:
        raise CommandMessageError("missing_seq", "goto command requires seq", None)
    if seq is None:
        raise CommandMessageError("invalid_seq", "seq must be a non-negative integer", None)
    if set(message) != {"type", "seq", "x_m", "y_m"}:
        raise CommandMessageError("invalid_fields", "goto command has missing or unexpected fields", seq)

    values = (message["x_m"], message["y_m"])
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise CommandMessageError("invalid_goto", "goto coordinates must be JSON numbers", seq)
    try:
        x_m, y_m = (float(value) for value in values)
    except OverflowError as error:
        raise CommandMessageError("invalid_goto", "goto coordinates must be finite", seq) from error
    if not math.isfinite(x_m) or not math.isfinite(y_m):
        raise CommandMessageError("invalid_goto", "goto coordinates must be finite", seq)
    return x_m, y_m, seq


def parse_goto_message(raw: object) -> tuple[float, float, int]:
    """Validate one local-odometry go-to-goal command."""
    return _parse_goto_object(_decode_message(raw))


def handle_command_message(
    raw: object,
    vehicle: Vehicle,
    grid: MapGrid,
    monotonic_now: float,
    wall_timestamp: float,
    navigation: GotoController | None = None,
    safety: LocalSafetyRuntime | None = None,
) -> dict[str, object]:
    """Advance the prior command, then acknowledge or fail-safe stop."""
    if safety is None:
        handoff_collided = vehicle.advance(grid, monotonic_now)
        handoff_safety_stop = None
    else:
        handoff = safety.advance(
            vehicle,
            grid,
            monotonic_now,
            automatic=navigation is not None and navigation.status == "active",
        )
        handoff_collided = handoff.collided
        handoff_safety_stop = handoff.reason if handoff.stopped else None
    rejection_reason: str | None = None
    try:
        message = _decode_message(raw)
        if message.get("type") == "goto":
            x_m, y_m, seq = _parse_goto_object(message)
            if navigation is None:
                raise CommandMessageError("goto_unavailable", "goto controller is unavailable", seq)
            vehicle.stop()
            navigation.start(x_m, y_m)
            if handoff_collided:
                navigation.status = "blocked"
                navigation.reason = "collision"
            elif handoff_safety_stop is not None:
                navigation.status = "blocked"
                navigation.reason = handoff_safety_stop
            accepted = navigation.status == "active"
            reply = {
                "type": "goto_ack",
                "ts": wall_timestamp,
                "seq": seq,
                "goal": {"x_m": x_m, "y_m": y_m},
                "accepted": accepted,
            }
            if not accepted:
                reply["reason"] = navigation.reason
            return reply
        if message.get("type") == "drive":
            linear_mps, angular_rps, seq = _parse_drive_object(
                message, vehicle.linear_speed, vehicle.angular_speed
            )
            command = "drive"
        else:
            command, seq = _parse_command_object(message)
            linear_mps, angular_rps = vehicle.velocities_for_command(command)
        if navigation is not None:
            navigation.cancel("manual_override")
        decision = (
            safety.enforce_manual(vehicle, grid, (linear_mps, angular_rps))
            if safety is not None
            else None
        )
        if handoff_collided:
            rejection_reason = "collision"
        elif decision is not None and decision.state in {"stopped", "fault"}:
            rejection_reason = decision.reason or "safety_rejected"
        elif command == "drive":
            vehicle.install_drive(linear_mps, angular_rps, monotonic_now)
        else:
            vehicle.install_command(command, monotonic_now)
    except CommandMessageError as error:
        vehicle.stop()
        if navigation is not None:
            navigation.cancel("invalid_command")
        return {
            "type": "error",
            "ts": wall_timestamp,
            "seq": error.seq,
            "code": error.code,
            "message": str(error),
        }

    reply = {
        "type": "cmd_ack",
        "ts": wall_timestamp,
        "seq": seq,
        "cmd": command,
        "accepted": rejection_reason is None,
    }
    if rejection_reason is not None:
        reply["reason"] = rejection_reason
    return reply


def generate_map(size: int = 256, seed: int = 42, radius: float = 0.5) -> tuple[list[dict[str, object]], MapGrid]:
    """Create the deterministic ground-truth grid and clear the spawn area."""
    rng = random.Random(seed)
    clear_min_x = math.floor(SPAWN_X - radius) - 1
    clear_max_x = math.ceil(SPAWN_X + radius) + 1
    clear_min_y = math.floor(SPAWN_Y - radius) - 1
    clear_max_y = math.ceil(SPAWN_Y + radius) + 1
    voxels = []
    for gx in range(size):
        for gy in range(size):
            in_spawn = clear_min_x <= gx <= clear_max_x and clear_min_y <= gy <= clear_max_y
            is_wall = rng.random() < 0.05
            is_void = size >= 32 and 24 <= gx <= 26 and 9 <= gy <= 12
            state = VOID if is_void and not in_spawn else int(is_wall and not in_spawn)
            voxels.append(
                {
                    "gx": gx,
                    "gy": gy,
                    "gz": 0,
                    "state": state,
                    "conf": 1.0,
                }
            )
    return voxels, MapGrid.from_voxels(voxels)


def telemetry_messages(
    vehicle: Vehicle,
    grid: MapGrid,
    sequence: int,
    timestamp: float,
    navigation: GotoController | None = None,
    safety: LocalSafetyRuntime | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build a pose/scan pair from one state snapshot and wall-clock timestamp."""
    vx, vy, omega = vehicle.velocities()
    pose = {
        "type": "pose",
        "ts": timestamp,
        "seq": sequence,
        "source": "simulator_ground_truth",
        "x": vehicle.x,
        "y": vehicle.y,
        "z": 0.0,
        "yaw": vehicle.yaw,
        "vx": vx,
        "vy": vy,
        "omega": omega,
        "collision": vehicle.collision,
        "command": vehicle.command,
        "control_mode": navigation.control_mode if navigation is not None else "manual",
        "navigation": (
            navigation.snapshot()
            if navigation is not None
            else {"status": "idle", "goal": None, "reason": None}
        ),
        "safety": (
            safety.snapshot()
            if safety is not None
            else {
                "state": "clear",
                "reason": None,
                "obstacle_clearance_m": None,
                "edge_clearance_m": None,
            }
        ),
    }
    scan = scan_message(grid, vehicle.x, vehicle.y, vehicle.yaw, timestamp, TMINI_SCAN_CONFIG)
    scan["seq"] = sequence
    return pose, scan


async def handler(
    websocket,
    *,
    vehicle_id: str = DEFAULT_VEHICLE_ID,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
    _monotonic=time.monotonic,
    _wall_time=time.time,
    _safety_healthy: bool = True,
) -> None:
    """Serve one client; all receives and sends stay serialized in this coroutine."""
    vehicle_id = validate_vehicle_id(vehicle_id)
    addr = websocket.remote_address
    print(f"[+] client connected: {addr}")
    started_at = _monotonic()
    voxels, grid = generate_map(radius=radius)
    vehicle = Vehicle(
        SPAWN_X,
        SPAWN_Y,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        radius=radius,
        command_timeout=command_timeout,
        now=started_at,
    )
    navigation = GotoController()
    safety = LocalSafetyRuntime(healthy=_safety_healthy)
    frame_sequence = 0
    next_deadline = started_at

    try:
        await websocket.send(json.dumps({"type": "hello", "vehicle_id": vehicle_id}))
        map_message = {
            "type": "map_full",
            "ts": _wall_time(),
            "source": "simulator_ground_truth",
            "voxels": voxels,
        }
        payload = json.dumps(map_message)
        print(f"[→] sending map_full ({len(payload)} bytes, {len(voxels)} cells)")
        await websocket.send(payload)

        while True:
            now = _monotonic()
            if now >= next_deadline:
                navigation.update(vehicle, grid, now, safety)
                timestamp = _wall_time()
                pose, scan = telemetry_messages(
                    vehicle, grid, frame_sequence, timestamp, navigation, safety
                )
                await websocket.send(json.dumps(pose))
                await websocket.send(json.dumps(scan))
                print(f"[→] pose #{frame_sequence}: x={vehicle.x:.2f} y={vehicle.y:.2f} cmd={vehicle.command}")
                frame_sequence += 1
                next_deadline = _next_deadline(next_deadline, _monotonic(), TMINI_SCAN_CONFIG.scan_time)
                continue

            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=next_deadline - now)
            except asyncio.TimeoutError:
                continue
            reply = handle_command_message(
                raw, vehicle, grid, _monotonic(), _wall_time(), navigation, safety
            )
            await websocket.send(json.dumps(reply))
    except Exception as error:
        print(f"[!] connection ended: {error}")
    finally:
        navigation.cancel("disconnected")
        vehicle.stop()
        print(f"[-] client disconnected: {addr}")


async def main(
    *,
    port: int = PORT,
    vehicle_id: str = DEFAULT_VEHICLE_ID,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
) -> None:
    from websockets.asyncio.server import serve

    vehicle_id = validate_vehicle_id(vehicle_id)
    stop = asyncio.Event()
    _shutting_down = False

    def _sig_handler():
        nonlocal _shutting_down
        if not _shutting_down:
            _shutting_down = True
            print("\n[!] shutting down...")
            stop.set()
        else:
            # Second Ctrl+C while already shutting down — force exit.
            print("\n[!] forcing exit...")
            import os as _os

            _os._exit(1)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sig_handler)

    async def configured_handler(websocket):
        await handler(
            websocket,
            vehicle_id=vehicle_id,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            radius=radius,
            command_timeout=command_timeout,
        )

    try:
        async with serve(configured_handler, HOST, port):
            print(f"Mock Vehicle Server listening on ws://{HOST}:{port}")
            print("Waiting for a controller connection...\n")
            await stop.wait()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    asyncio.run(main())
