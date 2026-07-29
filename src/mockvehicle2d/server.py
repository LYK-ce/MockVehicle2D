#!/usr/bin/env python3
"""Robot Controller WebSocket server backed by the 2D simulator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import math
import random
import re
import signal
import struct
import time

from mockvehicle2d.controller import Command, CommandResult, RobotController
from mockvehicle2d.local_state import (
    AnchorSpec,
    AnchoredLocalState,
    OdometryConfig,
)
from mockvehicle2d.map_grid import MapGrid, VOID
from mockvehicle2d.protocol import (
    ProtocolError,
    command_ack,
    error_message,
    parse_command,
)
from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
from mockvehicle2d.scan import (
    LaserPoint,
    TMINI_SCAN_CONFIG,
    scan_grid,
    scan_message,
)
from mockvehicle2d.vehicle import Vehicle


HOST = "0.0.0.0"
PORT = 19090
DEFAULT_VEHICLE_ID = "mock_vehicle_01"
SPAWN_X = 10.0
SPAWN_Y = 10.0
MAP_RESOLUTION_M = 1.0
SEND_TIMEOUT_S = 1.0
VEHICLE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


@dataclass(frozen=True)
class RuntimeFrame:
    scan_points: tuple[LaserPoint, ...]
    timestamp: float


@dataclass
class VehicleRuntime:
    """Persistent state of one simulated robot."""

    voxels: list[dict[str, object]]
    grid: MapGrid
    vehicle: Vehicle
    controller: RobotController
    safety: LocalSafetyRuntime
    local_state: AnchoredLocalState
    frame_sequence: int = 0
    controller_lease: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
        compare=False,
    )
    _pending_advance: SafetyAdvanceResult = field(
        default_factory=SafetyAdvanceResult,
        repr=False,
        compare=False,
    )

    def advance_to(self, monotonic_now: float) -> None:
        result = self.safety.advance(
            self.vehicle,
            self.grid,
            monotonic_now,
            automatic=self.controller.is_automatic_motion_active,
        )
        self._pending_advance = _merge_advance(self._pending_advance, result)

    def handle_command(
        self,
        command: Command,
        *,
        monotonic_now: float,
    ) -> CommandResult:
        self.advance_to(monotonic_now)
        return self.controller.handle(
            command,
            vehicle=self.vehicle,
            grid=self.grid,
            safety=self.safety,
            now=monotonic_now,
        )

    def fail_safe_stop(self, monotonic_now: float) -> None:
        self.advance_to(monotonic_now)
        self.controller.fail_safe_stop(self.vehicle, "invalid_command")

    def update(self, monotonic_now: float, wall_timestamp: float) -> RuntimeFrame:
        self.advance_to(monotonic_now)
        advance_result = self._pending_advance
        self._pending_advance = SafetyAdvanceResult()
        scan_points = tuple(
            scan_grid(
                self.grid,
                self.vehicle.x,
                self.vehicle.y,
                self.vehicle.yaw,
                TMINI_SCAN_CONFIG,
            )
        )
        self.local_state.update_from_truth(
            self.vehicle.x,
            self.vehicle.y,
            self.vehicle.yaw,
            timestamp=wall_timestamp,
        )
        map_delta = self.local_state.match_and_integrate_scan(
            scan_points,
            wall_timestamp,
            TMINI_SCAN_CONFIG,
            forbidden_points_vehicle_m=(
                ()
                if self.safety.observation.edge_point_vehicle_m is None
                else (self.safety.observation.edge_point_vehicle_m,)
            ),
        )
        self.controller.tick(
            vehicle=self.vehicle,
            grid=self.grid,
            safety=self.safety,
            anchor=self.local_state.anchor,
            pose=self.local_state.pose,
            local_map=self.local_state.local_map,
            map_delta=map_delta,
            advance_result=advance_result,
            now=monotonic_now,
        )
        return RuntimeFrame(scan_points, wall_timestamp)

    @classmethod
    def create(
        cls,
        *,
        started_at: float,
        timestamp: float | None = None,
        anchor: AnchorSpec,
        odometry_config: OdometryConfig,
        linear_speed: float = 0.5,
        angular_speed: float = math.pi / 2,
        radius: float = 0.5,
        command_timeout: float = 1.0,
        mission_capacity: int = 16,
        safety_healthy: bool = True,
    ) -> "VehicleRuntime":
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
        return cls(
            voxels,
            grid,
            vehicle,
            RobotController(mission_capacity=mission_capacity),
            LocalSafetyRuntime(healthy=safety_healthy),
            AnchoredLocalState(
                anchor,
                truth_x_m=vehicle.x,
                truth_y_m=vehicle.y,
                truth_yaw_rad=vehicle.yaw,
                odometry_config=odometry_config,
                timestamp=started_at if timestamp is None else timestamp,
            ),
        )


def validate_vehicle_id(value: str) -> str:
    if not VEHICLE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "vehicle id must be 1-64 ASCII letters, digits, dots, "
            "underscores, or hyphens"
        )
    return value


def _next_deadline(deadline: float, now: float, period: float) -> float:
    if now < deadline:
        return deadline
    return deadline + (math.floor((now - deadline) / period) + 1) * period


def _merge_advance(
    previous: SafetyAdvanceResult,
    current: SafetyAdvanceResult,
) -> SafetyAdvanceResult:
    if previous.collided or current.collided:
        return SafetyAdvanceResult(collided=True)
    if previous.stopped:
        return previous
    return current


def _encode_map_chunks(
    voxels: list[dict[str, object]],
    map_size: int,
) -> list[bytes]:
    chunk_size = 256
    state = {(voxel["gx"], voxel["gy"]): voxel.get("state", 0) for voxel in voxels}
    chunks = []
    for chunk_y in range(0, map_size, chunk_size):
        for chunk_x in range(0, map_size, chunk_size):
            cells = bytearray(chunk_size * chunk_size)
            for gy in range(chunk_size):
                absolute_y = chunk_y + gy
                row_offset = gy * chunk_size
                for gx in range(chunk_size):
                    cells[row_offset + gx] = state.get(
                        (chunk_x + gx, absolute_y),
                        0,
                    )
            chunks.append(struct.pack(">Bii", 0, chunk_x, chunk_y) + bytes(cells))
    return chunks


def _map_metadata(grid: MapGrid, anchor: AnchorSpec) -> dict[str, object]:
    origin_x_m, origin_y_m, origin_yaw_rad = anchor.anchor_to_global(
        -SPAWN_X,
        -SPAWN_Y,
        0.0,
    )
    return {
        "source": "simulator_ground_truth",
        "frame_id": "simulator_map",
        "resolution_m": MAP_RESOLUTION_M,
        "width_cells": grid.width,
        "height_cells": grid.height,
        "transform_to_global_map": {
            "x_m": origin_x_m,
            "y_m": origin_y_m,
            "yaw_rad": origin_yaw_rad,
        },
        "binary_chunks": {
            "type": 0,
            "chunk_size_cells": 256,
            "header": ">Bii",
            "byte_order": "big",
            "payload_order": "row_major_y_x",
        },
    }


def generate_map(
    size: int = 256,
    seed: int = 42,
    radius: float = 0.5,
) -> tuple[list[dict[str, object]], MapGrid]:
    """Create the deterministic truth environment and clear the spawn area."""
    rng = random.Random(seed)
    clear_min_x = math.floor(SPAWN_X - radius) - 1
    clear_max_x = math.ceil(SPAWN_X + radius) + 1
    clear_min_y = math.floor(SPAWN_Y - radius) - 1
    clear_max_y = math.ceil(SPAWN_Y + radius) + 1
    voxels = []
    for gx in range(size):
        for gy in range(size):
            in_spawn = (
                clear_min_x <= gx <= clear_max_x
                and clear_min_y <= gy <= clear_max_y
            )
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
    runtime: VehicleRuntime,
    frame: RuntimeFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one pose/scan pair from the same sampled frame."""
    estimate = runtime.local_state.pose
    x_m, y_m, yaw_rad = runtime.local_state.anchor.anchor_to_global(
        estimate.x_m,
        estimate.y_m,
        estimate.yaw_rad,
    )
    linear_mps, omega_rps = runtime.vehicle.body_velocities()
    pose = {
        "type": "pose",
        "timestamp_s": frame.timestamp,
        "seq": runtime.frame_sequence,
        "source": "anchored_odometry",
        "frame_id": "global_map",
        "x_m": x_m,
        "y_m": y_m,
        "z_m": 0.0,
        "yaw_rad": yaw_rad,
        "vx_mps": linear_mps * math.cos(yaw_rad),
        "vy_mps": linear_mps * math.sin(yaw_rad),
        "omega_rps": omega_rps,
        "collision": runtime.vehicle.collision,
        "actuator_command": runtime.vehicle.command,
        "controller": runtime.controller.snapshot(),
        "safety": runtime.safety.snapshot(),
        "localization": {
            **estimate.as_dict(),
            "local_map_revision": runtime.local_state.local_map.revision,
            **(
                {}
                if runtime.local_state.last_scan_match is None
                else {
                    "scan_match": runtime.local_state.last_scan_match.as_dict(),
                }
            ),
        },
    }
    scan = scan_message(
        runtime.grid,
        runtime.vehicle.x,
        runtime.vehicle.y,
        runtime.vehicle.yaw,
        frame.timestamp,
        TMINI_SCAN_CONFIG,
        frame.scan_points,
    )
    scan["seq"] = runtime.frame_sequence
    return pose, scan


async def _send_json(websocket, message: dict[str, object]) -> None:
    await asyncio.wait_for(
        websocket.send(json.dumps(message, separators=(",", ":"))),
        timeout=SEND_TIMEOUT_S,
    )


async def _send_map_chunks(websocket, chunks: list[bytes]) -> None:
    for chunk in chunks:
        await asyncio.wait_for(websocket.send(chunk), timeout=SEND_TIMEOUT_S)


async def _send_pending_events(
    websocket,
    controller: RobotController,
    after_event_seq: int,
    timestamp: float,
) -> int:
    for event in controller.events_after(after_event_seq):
        await _send_json(websocket, event.as_dict(timestamp))
        after_event_seq = event.event_seq
    return after_event_seq


async def handler(
    websocket,
    *,
    vehicle_id: str = DEFAULT_VEHICLE_ID,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
    mission_capacity: int = 16,
    _monotonic=time.monotonic,
    _wall_time=time.time,
    _safety_healthy: bool = True,
    _localization_quality: str | None = None,
    _runtime: VehicleRuntime | None = None,
) -> None:
    """Serve one exclusive controller connection."""
    vehicle_id = validate_vehicle_id(vehicle_id)
    address = websocket.remote_address
    started_at = _monotonic()
    runtime = _runtime or VehicleRuntime.create(
        started_at=started_at,
        timestamp=_wall_time(),
        anchor=AnchorSpec(f"{vehicle_id}_anchor", SPAWN_X, SPAWN_Y, 0.0),
        odometry_config=OdometryConfig(),
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        radius=radius,
        command_timeout=command_timeout,
        mission_capacity=mission_capacity,
        safety_healthy=_safety_healthy,
    )
    if runtime.controller_lease.locked():
        timestamp = _wall_time()
        try:
            await _send_json(
                websocket,
                {
                    "type": "error",
                    "timestamp_s": timestamp,
                    "seq": None,
                    "code": "vehicle_busy",
                    "message": "another controller owns the vehicle lease",
                },
            )
        except Exception:
            pass
        return

    await runtime.controller_lease.acquire()
    print(f"[+] controller connected: {address}")
    try:
        if (
            _localization_quality is not None
            and _localization_quality != runtime.local_state.pose.quality
        ):
            runtime.local_state.set_localization_quality(
                _localization_quality,
                timestamp=_wall_time(),
            )
        timestamp = _wall_time()
        await _send_json(
            websocket,
            {
                "type": "hello",
                "protocol_version": 4,
                "vehicle_id": vehicle_id,
                "control_lease": "exclusive",
                "mission_frame_id": "global_map",
                "map": _map_metadata(runtime.grid, runtime.local_state.anchor),
                "controller": runtime.controller.snapshot(),
            },
        )
        event_cursor = await _send_pending_events(
            websocket,
            runtime.controller,
            0,
            timestamp,
        )
        await _send_map_chunks(websocket, _encode_map_chunks(runtime.voxels, 256))

        next_deadline = started_at
        last_seq: int | None = None
        while True:
            monotonic_now = _monotonic()
            if monotonic_now >= next_deadline:
                timestamp = _wall_time()
                frame = runtime.update(monotonic_now, timestamp)
                pose, scan = telemetry_messages(runtime, frame)
                await _send_json(websocket, pose)
                await _send_json(websocket, scan)
                event_cursor = await _send_pending_events(
                    websocket,
                    runtime.controller,
                    event_cursor,
                    timestamp,
                )
                runtime.frame_sequence += 1
                next_deadline = _next_deadline(
                    next_deadline,
                    _monotonic(),
                    TMINI_SCAN_CONFIG.scan_time,
                )
                continue

            try:
                raw = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=next_deadline - monotonic_now,
                )
            except asyncio.TimeoutError:
                continue

            timestamp = _wall_time()
            try:
                command = parse_command(
                    raw,
                    linear_limit_mps=runtime.vehicle.linear_speed,
                    angular_limit_rps=runtime.vehicle.angular_speed,
                    mission_batch_limit=runtime.controller.mission_capacity,
                )
                if last_seq is not None and command.seq <= last_seq:
                    raise ProtocolError(
                        "stale_seq",
                        "seq must increase within one controller session",
                        command.seq,
                    )
                last_seq = command.seq
                result = runtime.handle_command(
                    command,
                    monotonic_now=_monotonic(),
                )
                await _send_json(
                    websocket,
                    command_ack(
                        command,
                        result,
                        timestamp=timestamp,
                        controller=runtime.controller.snapshot(),
                    ),
                )
                event_cursor = await _send_pending_events(
                    websocket,
                    runtime.controller,
                    event_cursor,
                    timestamp,
                )
            except ProtocolError as error:
                runtime.fail_safe_stop(_monotonic())
                await _send_json(websocket, error_message(error, timestamp=timestamp))
                event_cursor = await _send_pending_events(
                    websocket,
                    runtime.controller,
                    event_cursor,
                    timestamp,
                )
    except Exception as error:
        print(f"[!] controller connection ended: {error}")
    finally:
        runtime.controller.disconnect(runtime.vehicle)
        runtime.controller_lease.release()
        print(f"[-] controller disconnected: {address}")


async def main(
    *,
    port: int = PORT,
    vehicle_id: str = DEFAULT_VEHICLE_ID,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
    mission_capacity: int = 16,
    anchor_id: str | None = None,
    anchor_x_m: float = SPAWN_X,
    anchor_y_m: float = SPAWN_Y,
    anchor_yaw_rad: float = 0.0,
    odometry_translation_noise_stddev_m: float = 0.0,
    odometry_yaw_noise_stddev_rad: float = 0.0,
    odometry_seed: int = 0,
) -> None:
    from websockets.asyncio.server import serve

    vehicle_id = validate_vehicle_id(vehicle_id)
    runtime = VehicleRuntime.create(
        started_at=time.monotonic(),
        timestamp=time.time(),
        anchor=AnchorSpec(
            anchor_id or f"{vehicle_id}_anchor",
            anchor_x_m,
            anchor_y_m,
            anchor_yaw_rad,
        ),
        odometry_config=OdometryConfig(
            odometry_translation_noise_stddev_m,
            odometry_yaw_noise_stddev_rad,
            odometry_seed,
        ),
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        radius=radius,
        command_timeout=command_timeout,
        mission_capacity=mission_capacity,
    )
    stop = asyncio.Event()
    shutting_down = False

    def signal_handler() -> None:
        nonlocal shutting_down
        if not shutting_down:
            shutting_down = True
            stop.set()
            return
        raise KeyboardInterrupt

    loop = asyncio.get_running_loop()
    for caught_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(caught_signal, signal_handler)

    async def configured_handler(websocket) -> None:
        await handler(
            websocket,
            vehicle_id=vehicle_id,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            radius=radius,
            command_timeout=command_timeout,
            mission_capacity=mission_capacity,
            _runtime=runtime,
        )

    try:
        async with serve(configured_handler, HOST, port):
            print(f"Mock Vehicle Server listening on ws://{HOST}:{port}")
            await stop.wait()
    finally:
        runtime.controller.disconnect(runtime.vehicle)
        for caught_signal in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(caught_signal)


if __name__ == "__main__":
    asyncio.run(main())
