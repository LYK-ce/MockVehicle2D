#!/usr/bin/env python3
"""Controllable 2D vehicle and Tmini-style WebSocket simulator."""

import asyncio
import json
import math
import random
import re
import signal
import time

from mockvehicle2d.map_grid import MapGrid
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


def parse_command_message(raw: object) -> tuple[str, int | None]:
    """Validate canonical commands and the exact legacy ``{"cmd": ...}`` form."""
    if not isinstance(raw, str):
        raise CommandMessageError("invalid_json_text", "command must be a JSON text message")
    try:
        message = json.loads(raw, parse_int=_bounded_json_int)
    except (ValueError, RecursionError) as error:
        raise CommandMessageError("invalid_json", "command is not valid JSON text") from error
    if not isinstance(message, dict):
        raise CommandMessageError("invalid_message", "command JSON must be an object")

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


def handle_command_message(
    raw: object, vehicle: Vehicle, grid: MapGrid, monotonic_now: float, wall_timestamp: float
) -> dict[str, object]:
    """Advance the prior command, then acknowledge or fail-safe stop."""
    try:
        command, seq = parse_command_message(raw)
    except CommandMessageError as error:
        vehicle.advance(grid, monotonic_now)
        vehicle.stop()
        return {
            "type": "error",
            "ts": wall_timestamp,
            "seq": error.seq,
            "code": error.code,
            "message": str(error),
        }

    vehicle.apply_command(grid, command, monotonic_now)
    return {
        "type": "cmd_ack",
        "ts": wall_timestamp,
        "seq": seq,
        "cmd": command,
        "accepted": True,
    }


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
            voxels.append(
                {"gx": gx, "gy": gy, "gz": 0, "state": 1 if is_wall and not in_spawn else 0, "conf": 1.0}
            )
    return voxels, MapGrid.from_voxels(voxels)


def telemetry_messages(
    vehicle: Vehicle, grid: MapGrid, sequence: int, timestamp: float
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
                vehicle.advance(grid, now)
                timestamp = _wall_time()
                pose, scan = telemetry_messages(vehicle, grid, frame_sequence, timestamp)
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
            reply = handle_command_message(raw, vehicle, grid, _monotonic(), _wall_time())
            await websocket.send(json.dumps(reply))
    except Exception as error:
        print(f"[!] connection ended: {error}")
    finally:
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
