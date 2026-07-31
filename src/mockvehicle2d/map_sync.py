"""Vehicle-owned map deltas and the optional rust-libp2p sidecar lifecycle."""

from __future__ import annotations

import asyncio
import errno
import fcntl
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import stat
import tempfile
from typing import Iterable

from mockvehicle2d.local_state import (
    FREE,
    FORBIDDEN,
    OCCUPIED,
    AnchorSpec,
    LocalMapDelta,
    MapCellUpdate,
)


SIDECAR_PROTOCOL = "mockvehicle2d-map-sync-sidecar/1"
DELTA_PROTOCOL = "mockvehicle2d-map-delta/1"
MAX_DELTA_CELLS = 512
MAX_MESSAGE_BYTES = 256 * 1024
MAX_GRID_COORDINATE = 1_000_000
MAP_EPOCH = 1
TRANSFORM_EPOCH = 1
ALLOWED_CELL_STATES = frozenset((FREE, OCCUPIED, FORBIDDEN))
PROCESS_STOP_TIMEOUT_S = 2.0


class _CleanupError(RuntimeError):
    def __init__(self, context: str, errors: Iterable[BaseException]) -> None:
        self.errors = tuple(errors)
        details = "; ".join(
            f"{type(error).__name__}: {error}" for error in self.errors
        )
        super().__init__(f"{context}: {details}")


class _CleanupPending(_CleanupError):
    """Cleanup stopped at its deadline while owned work was still alive."""


async def _finish_cleanup(
    task: asyncio.Task[None],
) -> tuple[asyncio.CancelledError | None, BaseException | None]:
    cancellation: asyncio.CancelledError | None = None
    waiter = asyncio.create_task(
        asyncio.wait((task,), timeout=PROCESS_STOP_TIMEOUT_S * 8)
    )
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
        except BaseException:
            break
    if waiter.cancelled():
        return cancellation, RuntimeError("cleanup waiter was cancelled")
    done, _ = waiter.result()
    if task not in done:
        task.cancel()
        return cancellation, TimeoutError("cleanup exceeded its bounded deadline")
    try:
        task.result()
    except BaseException as error:
        return cancellation, error
    return cancellation, None


async def _terminate_and_reap(
    process: asyncio.subprocess.Process,
    *,
    terminate_first: bool,
) -> None:
    errors: list[BaseException] = []
    pending_wait = False

    async def wait_once() -> bool:
        nonlocal pending_wait
        wait_task = asyncio.ensure_future(process.wait())
        done, pending = await asyncio.wait(
            (wait_task,),
            timeout=PROCESS_STOP_TIMEOUT_S,
        )
        if pending:
            wait_task.cancel()
            errors.append(TimeoutError("map-sync child wait did not stop"))
            cancelled, still_pending = await asyncio.wait(
                (wait_task,),
                timeout=PROCESS_STOP_TIMEOUT_S,
            )
            if still_pending:
                wait_task.add_done_callback(_consume_task_result)
                pending_wait = True
            elif wait_task in cancelled and not wait_task.cancelled():
                try:
                    wait_task.result()
                except BaseException:
                    pass
            return False
        try:
            wait_task.result()
        except BaseException as error:
            errors.append(error)
            return False
        return True

    if terminate_first and process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append(error)
    if process.returncode is None:
        if not await wait_once() and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except BaseException as error:
                errors.append(error)
            if process.returncode is None:
                await wait_once()
    if errors:
        error_type = _CleanupPending if pending_wait else _CleanupError
        raise error_type("cannot reap map-sync child", errors)


def _consume_task_result(task: asyncio.Future[object]) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except BaseException:
        pass


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _strict_object(
    value: object,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object")
    fields = frozenset(value)
    if not required <= fields or not fields <= allowed:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class P2PSettings:
    sidecar_path: Path
    runtime_dir: Path
    startup_timeout_s: float = 10.0
    sync_interval_ms: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.sidecar_path, Path) or not isinstance(self.runtime_dir, Path):
            raise ValueError("sidecar_path and runtime_dir must be paths")
        if not self.sidecar_path.name or not self.runtime_dir.name:
            raise ValueError("sidecar_path and runtime_dir cannot be empty")
        if not math.isfinite(self.startup_timeout_s) or self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be finite and positive")
        if (
            isinstance(self.sync_interval_ms, bool)
            or not isinstance(self.sync_interval_ms, int)
            or not 20 <= self.sync_interval_ms <= 1000
        ):
            raise ValueError("sync_interval_ms must be from 20 to 1000")

    @classmethod
    def from_json(cls, value: object) -> "P2PSettings":
        body = _strict_object(
            value,
            required=frozenset(("sidecar_path", "runtime_dir")),
            allowed=frozenset(
                (
                    "sidecar_path",
                    "runtime_dir",
                    "startup_timeout_s",
                    "sync_interval_ms",
                )
            ),
            name="p2p",
        )
        sidecar_path = body["sidecar_path"]
        runtime_dir = body["runtime_dir"]
        if not isinstance(sidecar_path, str) or not isinstance(runtime_dir, str):
            raise ValueError("p2p paths must be strings")
        sync_interval_ms = body.get("sync_interval_ms", 100)
        if isinstance(sync_interval_ms, bool) or not isinstance(sync_interval_ms, int):
            raise ValueError("sync_interval_ms must be an integer")
        return cls(
            Path(sidecar_path),
            Path(runtime_dir),
            _finite(body.get("startup_timeout_s", 10.0), "startup_timeout_s"),
            sync_interval_ms,
        )


@dataclass(frozen=True)
class P2PVehicleConfig:
    vehicle_id: str
    listen_port: int
    anchor: AnchorSpec

    def __post_init__(self) -> None:
        if (
            not self.vehicle_id
            or isinstance(self.listen_port, bool)
            or not isinstance(self.listen_port, int)
            or not 1 <= self.listen_port <= 65535
            or not isinstance(self.anchor, AnchorSpec)
        ):
            raise ValueError("invalid P2P vehicle configuration")


class MapSyncState:
    """Keeps local evidence, remote evidence and a derived read-only view separate."""

    def __init__(self, session_id: str, vehicle_id: str, anchor: AnchorSpec, resolution_m: float) -> None:
        if not session_id or not vehicle_id:
            raise ValueError("session_id and vehicle_id cannot be empty")
        if not math.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("resolution_m must be finite and positive")
        self.session_id = session_id
        self.vehicle_id = vehicle_id
        self.anchor = anchor
        self.resolution_m = resolution_m
        self.local_peer_id: str | None = None
        self._expected_peers: dict[str, tuple[str, AnchorSpec]] = {}
        self._own_cells: dict[tuple[int, int], int] = {}
        self._dirty: dict[tuple[int, int], int] = {}
        self._inflight: dict[str, object] | None = None
        self._sequence = 0
        self._peer_evidence: dict[str, dict[tuple[int, int], int]] = {}
        self._peer_epoch: dict[str, int] = {}
        self._peer_sequence: dict[str, int] = {}
        self._connected_vehicle_ids: tuple[str, ...] = ()
        self.ready = False
        self.published_deltas = 0
        self.received_deltas = 0
        self.rejected_deltas = 0
        self.publish_failures = 0
        self.sequence_gaps = 0
        self._collaborative_cache: dict[tuple[int, int], int] | None = None

    def configure_network(
        self,
        local_peer_id: str,
        expected_peers: dict[str, tuple[str, AnchorSpec]],
    ) -> None:
        if self.local_peer_id is not None:
            raise RuntimeError("map sync network is already configured")
        if self.vehicle_id in expected_peers:
            raise ValueError("expected peers cannot include the local vehicle")
        if len({peer_id for peer_id, _ in expected_peers.values()}) != len(expected_peers):
            raise ValueError("peer ids must be unique")
        self.local_peer_id = local_peer_id
        self._expected_peers = dict(expected_peers)

    def record_local(self, delta: LocalMapDelta | None) -> None:
        if delta is None:
            return
        if not isinstance(delta, LocalMapDelta):
            raise ValueError("local delta must be LocalMapDelta")
        for cell in delta.changed_cells:
            if (
                not isinstance(cell, MapCellUpdate)
                or cell.state not in ALLOWED_CELL_STATES
                or abs(cell.gx) > MAX_GRID_COORDINATE
                or abs(cell.gy) > MAX_GRID_COORDINATE
            ):
                raise ValueError("invalid local map cell")
            coordinate = (cell.gx, cell.gy)
            self._own_cells[coordinate] = cell.state
            self._dirty[coordinate] = cell.state
        if delta.changed_cells:
            self._collaborative_cache = None

    def prepare_delta(self) -> dict[str, object] | None:
        if self._inflight is not None or not self._dirty:
            return None
        cells = tuple(sorted(self._dirty.items(), key=lambda item: (item[0][1], item[0][0])))[:MAX_DELTA_CELLS]
        payload: dict[str, object] = {
            "protocol": DELTA_PROTOCOL,
            "session_id": self.session_id,
            "source_vehicle_id": self.vehicle_id,
            "map_epoch": MAP_EPOCH,
            "sequence": self._sequence + 1,
            "source_frame": "anchor_map",
            "anchor_id": self.anchor.anchor_id,
            "transform_epoch": TRANSFORM_EPOCH,
            "transform_to_global_map": {
                "x_m": self.anchor.global_x_m,
                "y_m": self.anchor.global_y_m,
                "yaw_rad": self.anchor.global_yaw_rad,
            },
            "resolution_m": self.resolution_m,
            "cells": [
                {"gx": gx, "gy": gy, "state": state}
                for (gx, gy), state in cells
            ],
        }
        if len(json.dumps(payload, separators=(",", ":")).encode()) > MAX_MESSAGE_BYTES:
            raise RuntimeError("bounded map delta unexpectedly exceeds message limit")
        self._inflight = payload
        return payload

    def publish_result(self, sequence: int, accepted: bool) -> None:
        if self._inflight is None or self._inflight["sequence"] != sequence:
            self.rejected_deltas += 1
            return
        payload = self._inflight
        self._inflight = None
        if not accepted:
            self.publish_failures += 1
            return
        for cell in payload["cells"]:
            coordinate = (cell["gx"], cell["gy"])
            if self._dirty.get(coordinate) == cell["state"]:
                self._dirty.pop(coordinate)
        self._sequence = sequence
        self.published_deltas += 1

    def network_disconnected(self) -> None:
        self.ready = False
        self._connected_vehicle_ids = ()
        if self._inflight is not None:
            self.publish_result(int(self._inflight["sequence"]), False)

    def set_health(self, *, ready: bool, connected_vehicle_ids: Iterable[str] = ()) -> None:
        connected = tuple(sorted(set(connected_vehicle_ids)))
        if any(vehicle_id not in self._expected_peers for vehicle_id in connected):
            raise ValueError("peer health contains an unknown vehicle")
        self.ready = ready
        self._connected_vehicle_ids = connected

    def receive(
        self,
        source_peer_id: str,
        reported_source_vehicle_id: str,
        payload: object,
    ) -> bool:
        try:
            parsed = self._validate_remote(
                source_peer_id,
                reported_source_vehicle_id,
                payload,
            )
        except (TypeError, ValueError):
            self.rejected_deltas += 1
            return False

        source, map_epoch, sequence, cells = parsed
        previous_epoch = self._peer_epoch.get(source, 0)
        if map_epoch < previous_epoch:
            self.rejected_deltas += 1
            return False
        if map_epoch > previous_epoch:
            self._peer_evidence[source] = {}
            self._peer_epoch[source] = map_epoch
            self._peer_sequence[source] = 0
        previous_sequence = self._peer_sequence[source]
        if sequence <= previous_sequence:
            self.rejected_deltas += 1
            return False
        self.sequence_gaps += max(0, sequence - previous_sequence - 1)
        evidence = self._peer_evidence.setdefault(source, {})
        evidence.update(cells)
        self._peer_sequence[source] = sequence
        self.received_deltas += 1
        self._collaborative_cache = None
        return True

    def _validate_remote(
        self,
        source_peer_id: str,
        reported_source_vehicle_id: str,
        payload: object,
    ) -> tuple[str, int, int, dict[tuple[int, int], int]]:
        if len(json.dumps(payload, separators=(",", ":")).encode()) > MAX_MESSAGE_BYTES:
            raise ValueError("remote delta exceeds size limit")
        body = _strict_object(
            payload,
            required=frozenset(
                (
                    "protocol",
                    "session_id",
                    "source_vehicle_id",
                    "map_epoch",
                    "sequence",
                    "source_frame",
                    "anchor_id",
                    "transform_epoch",
                    "transform_to_global_map",
                    "resolution_m",
                    "cells",
                )
            ),
            allowed=frozenset(
                (
                    "protocol",
                    "session_id",
                    "source_vehicle_id",
                    "map_epoch",
                    "sequence",
                    "source_frame",
                    "anchor_id",
                    "transform_epoch",
                    "transform_to_global_map",
                    "resolution_m",
                    "cells",
                )
            ),
            name="map_delta",
        )
        source = body["source_vehicle_id"]
        if not isinstance(source, str) or source != reported_source_vehicle_id:
            raise ValueError("source vehicle mismatch")
        expected = self._expected_peers.get(source)
        if expected is None or expected[0] != source_peer_id:
            raise ValueError("source peer is not allowed")
        expected_anchor = expected[1]
        if (
            body["protocol"] != DELTA_PROTOCOL
            or body["session_id"] != self.session_id
            or body["source_frame"] != "anchor_map"
            or body["anchor_id"] != expected_anchor.anchor_id
            or _positive_integer(body["transform_epoch"], "transform_epoch") != TRANSFORM_EPOCH
        ):
            raise ValueError("remote protocol, session, frame or transform is incompatible")
        resolution = _finite(body["resolution_m"], "resolution_m")
        if not math.isclose(resolution, self.resolution_m, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("remote map resolution is incompatible")
        transform = _strict_object(
            body["transform_to_global_map"],
            required=frozenset(("x_m", "y_m", "yaw_rad")),
            allowed=frozenset(("x_m", "y_m", "yaw_rad")),
            name="transform_to_global_map",
        )
        received_transform = tuple(
            _finite(transform[name], f"transform_to_global_map.{name}")
            for name in ("x_m", "y_m", "yaw_rad")
        )
        expected_transform = (
            expected_anchor.global_x_m,
            expected_anchor.global_y_m,
            expected_anchor.global_yaw_rad,
        )
        if any(
            not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected_value in zip(received_transform, expected_transform)
        ):
            raise ValueError("remote anchor transform does not match the allowlist")
        map_epoch = _positive_integer(body["map_epoch"], "map_epoch")
        sequence = _positive_integer(body["sequence"], "sequence")
        raw_cells = body["cells"]
        if not isinstance(raw_cells, list) or not 1 <= len(raw_cells) <= MAX_DELTA_CELLS:
            raise ValueError("cells must be a bounded non-empty array")
        cells: dict[tuple[int, int], int] = {}
        for raw_cell in raw_cells:
            cell = _strict_object(
                raw_cell,
                required=frozenset(("gx", "gy", "state")),
                allowed=frozenset(("gx", "gy", "state")),
                name="map_cell",
            )
            gx, gy, state = cell["gx"], cell["gy"], cell["state"]
            if (
                type(gx) is not int
                or type(gy) is not int
                or type(state) is not int
                or abs(gx) > MAX_GRID_COORDINATE
                or abs(gy) > MAX_GRID_COORDINATE
                or state not in ALLOWED_CELL_STATES
                or (gx, gy) in cells
            ):
                raise ValueError("invalid or duplicate map cell")
            cells[gx, gy] = state
        return source, map_epoch, sequence, cells

    @property
    def dirty_count(self) -> int:
        return len(self._dirty)

    @property
    def connected_count(self) -> int:
        return len(self._connected_vehicle_ids)

    def peer_evidence(self, source_vehicle_id: str) -> dict[tuple[int, int], int]:
        return dict(self._peer_evidence.get(source_vehicle_id, {}))

    def collaborative_cells(self) -> dict[tuple[int, int], int]:
        if self._collaborative_cache is None:
            combined: dict[tuple[int, int], int] = {}
            self._project(combined, self._own_cells, self.anchor)
            for source, cells in self._peer_evidence.items():
                self._project(combined, cells, self._expected_peers[source][1])
            self._collaborative_cache = combined
        return dict(self._collaborative_cache)

    def _project(
        self,
        destination: dict[tuple[int, int], int],
        cells: dict[tuple[int, int], int],
        anchor: AnchorSpec,
    ) -> None:
        for (gx, gy), state in cells.items():
            local_x = (gx + 0.5) * self.resolution_m
            local_y = (gy + 0.5) * self.resolution_m
            global_x, global_y, _ = anchor.anchor_to_global(local_x, local_y, 0.0)
            coordinate = (
                math.floor(global_x / self.resolution_m),
                math.floor(global_y / self.resolution_m),
            )
            destination[coordinate] = max(destination.get(coordinate, FREE), state)

    def snapshot(self) -> dict[str, object]:
        collaborative_current = self._collaborative_cache is not None
        return {
            "enabled": True,
            "ready": self.ready,
            "peer_id": self.local_peer_id,
            "connected_vehicle_ids": list(self._connected_vehicle_ids),
            "own_known_cells": len(self._own_cells),
            "own_dirty_cells": len(self._dirty),
            "published_deltas": self.published_deltas,
            "received_deltas": self.received_deltas,
            "rejected_deltas": self.rejected_deltas,
            "publish_failures": self.publish_failures,
            "sequence_gaps": self.sequence_gaps,
            "peer_sources": {
                source: {
                    "map_epoch": self._peer_epoch[source],
                    "last_sequence": self._peer_sequence[source],
                    "known_cells": len(cells),
                }
                for source, cells in sorted(self._peer_evidence.items())
            },
            "collaborative_evidence_cells": len(self._own_cells)
            + sum(len(cells) for cells in self._peer_evidence.values()),
            "collaborative_view_current": collaborative_current,
            "collaborative_known_cells": (
                len(self._collaborative_cache) if collaborative_current else None
            ),
        }


class _NodeBridge:
    def __init__(self, vehicle_id: str, socket_path: Path, state: MapSyncState) -> None:
        self.vehicle_id = vehicle_id
        self.socket_path = socket_path
        self.state = state
        self.server: asyncio.AbstractServer | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.ready_event = asyncio.Event()
        self._writer: asyncio.StreamWriter | None = None
        self._outbound: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=1)
        self._tasks: set[asyncio.Task[object]] = set()
        self._owned_socket: tuple[int, int] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start_server(self) -> None:
        self._remove_stale_socket()

        def accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self._track(asyncio.create_task(self._read_connection(reader, writer)))

        self.server = await asyncio.start_unix_server(accepted, path=self.socket_path)
        created = self.socket_path.lstat()
        if not stat.S_ISSOCK(created.st_mode):
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            raise RuntimeError(f"map-sync path is not a Unix socket: {self.socket_path}")
        self._owned_socket = (created.st_dev, created.st_ino)
        os.chmod(self.socket_path, 0o600)
        self._track(asyncio.create_task(self._send_loop()))

    def _remove_stale_socket(self) -> None:
        try:
            found = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(found.st_mode):
            raise RuntimeError(f"map-sync path is not a Unix socket: {self.socket_path}")

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        try:
            result = probe.connect_ex(str(self.socket_path))
        except (OSError, TimeoutError) as error:
            raise RuntimeError(
                f"cannot confirm stale map-sync socket: {self.socket_path}"
            ) from error
        finally:
            probe.close()
        if result == 0:
            raise RuntimeError(f"map-sync socket is already active: {self.socket_path}")
        if result == errno.ENOENT:
            return
        if result != errno.ECONNREFUSED:
            raise RuntimeError(
                f"cannot confirm stale map-sync socket {self.socket_path}: {os.strerror(result)}"
            )

        try:
            current = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (
            not stat.S_ISSOCK(current.st_mode)
            or (current.st_dev, current.st_ino) != (found.st_dev, found.st_ino)
        ):
            raise RuntimeError(f"map-sync socket changed during cleanup: {self.socket_path}")
        self.socket_path.unlink()

    def _remove_owned_socket(self) -> None:
        owned = self._owned_socket
        self._owned_socket = None
        if owned is None:
            return
        try:
            current = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(current.st_mode) and (current.st_dev, current.st_ino) == owned:
            self.socket_path.unlink()

    def _track(self, task: asyncio.Task[object]) -> None:
        self._tasks.add(task)

        def finished(completed: asyncio.Task[object]) -> None:
            self._tasks.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                print(f"[!] {self.vehicle_id} map-sync bridge ended: {error}")

        task.add_done_callback(finished)

    def enqueue(self, payload: dict[str, object]) -> bool:
        try:
            self._outbound.put_nowait({"type": "publish", "payload": payload})
            return True
        except asyncio.QueueFull:
            return False

    async def _read_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            return
        self._writer = writer
        try:
            while raw := await reader.readline():
                if len(raw) > MAX_MESSAGE_BYTES:
                    raise ValueError("sidecar event exceeds size limit")
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise ValueError("sidecar event must be an object")
                event_type = event.get("type")
                if event_type == "ready":
                    if (
                        event.get("protocol") != SIDECAR_PROTOCOL
                        or event.get("vehicle_id") != self.vehicle_id
                        or event.get("peer_id") != self.state.local_peer_id
                    ):
                        raise ValueError("sidecar identity does not match its configuration")
                    self.state.set_health(ready=True)
                    self.ready_event.set()
                elif event_type == "peer_health":
                    connected = event.get("connected_vehicle_ids")
                    if event.get("vehicle_id") != self.vehicle_id or not isinstance(connected, list):
                        raise ValueError("invalid peer health event")
                    self.state.set_health(ready=True, connected_vehicle_ids=connected)
                elif event_type == "publish_result":
                    sequence = event.get("sequence")
                    accepted = event.get("accepted")
                    if type(sequence) is not int or type(accepted) is not bool:
                        raise ValueError("invalid publish result")
                    self.state.publish_result(sequence, accepted)
                elif event_type == "received":
                    source_peer_id = event.get("source_peer_id")
                    source_vehicle_id = event.get("source_vehicle_id")
                    if not isinstance(source_peer_id, str) or not isinstance(source_vehicle_id, str):
                        raise ValueError("invalid received event source")
                    self.state.receive(source_peer_id, source_vehicle_id, event.get("payload"))
                else:
                    raise ValueError("unknown sidecar event")
        except (ConnectionError, json.JSONDecodeError, ValueError):
            pass
        finally:
            if self._writer is writer:
                self._writer = None
                self.state.network_disconnected()
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def _send_loop(self) -> None:
        while True:
            command = await self._outbound.get()
            while self._writer is None:
                await asyncio.sleep(0.01)
            try:
                encoded = json.dumps(command, separators=(",", ":")).encode() + b"\n"
                if len(encoded) > MAX_MESSAGE_BYTES:
                    raise ValueError("local publish command exceeds size limit")
                self._writer.write(encoded)
                await self._writer.drain()
            except (ConnectionError, ValueError):
                payload = command.get("payload")
                if isinstance(payload, dict) and type(payload.get("sequence")) is int:
                    self.state.publish_result(payload["sequence"], False)

    async def _close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        pending_cleanup = False
        writer = self._writer
        shutdown_sent = False
        if writer is not None:
            try:
                writer.write(b'{"type":"shutdown"}\n')
            except ConnectionError:
                pass
            except BaseException as error:
                errors.append(error)
            else:
                drain_task = asyncio.create_task(writer.drain())
                done, pending = await asyncio.wait(
                    (drain_task,),
                    timeout=PROCESS_STOP_TIMEOUT_S,
                )
                if pending:
                    drain_task.cancel()
                    self._track(drain_task)
                    pending_cleanup = True
                    errors.append(TimeoutError("map-sync writer drain did not stop"))
                else:
                    try:
                        drain_task.result()
                        shutdown_sent = True
                    except ConnectionError:
                        pass
                    except BaseException as error:
                        errors.append(error)
        if self.process is not None:
            try:
                await _terminate_and_reap(
                    self.process,
                    terminate_first=not shutdown_sent,
                )
            except _CleanupPending as error:
                pending_cleanup = True
                errors.append(error)
            except BaseException as error:
                errors.append(error)
        server = self.server
        self.server = None
        if server is not None:
            try:
                server.close()
            except BaseException as error:
                errors.append(error)
        if writer is not None:
            try:
                writer.close()
            except BaseException as error:
                errors.append(error)
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=PROCESS_STOP_TIMEOUT_S)
            if pending:
                pending_cleanup = True
                errors.append(TimeoutError("map-sync bridge tasks did not stop"))
            for task in done:
                if task.cancelled():
                    continue
                try:
                    task.result()
                except BaseException as error:
                    errors.append(error)
        if server is not None:
            wait_task = asyncio.create_task(server.wait_closed())
            done, pending = await asyncio.wait(
                (wait_task,),
                timeout=PROCESS_STOP_TIMEOUT_S,
            )
            if pending:
                wait_task.cancel()
                self._track(wait_task)
                pending_cleanup = True
                errors.append(TimeoutError("map-sync server wait did not stop"))
            else:
                try:
                    wait_task.result()
                except BaseException as error:
                    errors.append(error)
        try:
            self._remove_owned_socket()
        except BaseException as error:
            errors.append(error)
        try:
            self.state.network_disconnected()
        except BaseException as error:
            errors.append(error)
        if not pending_cleanup:
            self._writer = None
            self._closed = True
        if errors:
            error_type = _CleanupPending if pending_cleanup else _CleanupError
            raise error_type(
                f"cannot close {self.vehicle_id} map-sync bridge",
                errors,
            )

    async def close(self) -> None:
        if self._closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        close_task = self._close_task
        cancellation, error = await _finish_cleanup(close_task)
        if close_task.done() and self._close_task is close_task:
            self._close_task = None
        if cancellation is not None:
            if error is not None:
                raise cancellation from error
            raise cancellation
        if error is not None:
            raise error


class P2PFleetSync:
    """Starts one independent libp2p sidecar per logical robot node."""

    def __init__(
        self,
        session_id: str,
        settings: P2PSettings,
        vehicles: tuple[P2PVehicleConfig, ...],
        states: dict[str, MapSyncState],
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.vehicles = vehicles
        self.states = states
        self.runtime_dir = settings.runtime_dir.expanduser().resolve()
        self.sidecar_path = settings.sidecar_path.expanduser().resolve()
        self._bridges: dict[str, _NodeBridge] = {}
        self._config_paths: list[Path] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._runtime_lock_fd: int | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def start(
        cls,
        session_id: str,
        settings: P2PSettings,
        vehicles: tuple[P2PVehicleConfig, ...],
        states: dict[str, MapSyncState],
    ) -> "P2PFleetSync":
        runtime = cls(session_id, settings, vehicles, states)
        try:
            await runtime._start()
            return runtime
        except BaseException as original:
            try:
                await runtime.close()
            except asyncio.CancelledError as cancellation:
                if isinstance(original, asyncio.CancelledError):
                    raise original from cancellation.__cause__
                raise cancellation from original
            except BaseException as cleanup_error:
                raise original from cleanup_error
            raise

    async def _start(self) -> None:
        if not self.sidecar_path.is_file() or not os.access(self.sidecar_path, os.X_OK):
            raise ValueError(f"map-sync sidecar is not executable: {self.sidecar_path}")
        self._acquire_runtime_lease()
        peer_ids = [
            await self._ensure_identity(vehicle.vehicle_id)
            for vehicle in self.vehicles
        ]
        if len(set(peer_ids)) != len(peer_ids):
            raise RuntimeError("map-sync sidecars must have unique peer identities")
        peer_id_by_vehicle = dict(zip((vehicle.vehicle_id for vehicle in self.vehicles), peer_ids))

        for index, vehicle in enumerate(self.vehicles, 1):
            expected_peers = {
                remote.vehicle_id: (peer_id_by_vehicle[remote.vehicle_id], remote.anchor)
                for remote in self.vehicles
                if remote.vehicle_id != vehicle.vehicle_id
            }
            self.states[vehicle.vehicle_id].configure_network(
                peer_id_by_vehicle[vehicle.vehicle_id],
                expected_peers,
            )
            bridge = _NodeBridge(
                vehicle.vehicle_id,
                self.runtime_dir / f"p{index}.sock",
                self.states[vehicle.vehicle_id],
            )
            self._bridges[vehicle.vehicle_id] = bridge
            await bridge.start_server()

        for vehicle in self.vehicles:
            bridge = self._bridges[vehicle.vehicle_id]
            config_path = self.runtime_dir / f"{vehicle.vehicle_id}.json"
            config = {
                "protocol": SIDECAR_PROTOCOL,
                "vehicle_id": vehicle.vehicle_id,
                "session_id": self.session_id,
                "listen_port": vehicle.listen_port,
                "uds_path": str(bridge.socket_path),
                "identity_path": str(self.runtime_dir / f"{vehicle.vehicle_id}.key"),
                "peers": [
                    {
                        "vehicle_id": remote.vehicle_id,
                        "peer_id": peer_id_by_vehicle[remote.vehicle_id],
                        "address": (
                            f"/ip4/127.0.0.1/tcp/{remote.listen_port}"
                            f"/p2p/{peer_id_by_vehicle[remote.vehicle_id]}"
                        ),
                    }
                    for remote in self.vehicles
                    if remote.vehicle_id != vehicle.vehicle_id
                ],
            }
            self._config_paths.append(config_path)
            self._write_config(config_path, config)
            bridge.process = await asyncio.create_subprocess_exec(
                str(self.sidecar_path),
                "run",
                str(config_path),
            )

        timeout = self.settings.startup_timeout_s
        await asyncio.wait_for(
            asyncio.gather(*(bridge.ready_event.wait() for bridge in self._bridges.values())),
            timeout=timeout,
        )
        expected_connections = len(self.vehicles) - 1
        if expected_connections:
            await asyncio.wait_for(self._wait_for_full_mesh(expected_connections), timeout=timeout)
        self._flush_task = asyncio.create_task(self._flush_loop())

    def _acquire_runtime_lease(self) -> None:
        if self._runtime_lock_fd is not None:
            raise RuntimeError("map-sync runtime lease is already held")
        try:
            self.runtime_dir.mkdir(parents=True, mode=0o700)
        except FileExistsError:
            pass
        else:
            os.chmod(self.runtime_dir, 0o700)
        directory = self.runtime_dir.stat()
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != os.geteuid()
            or directory.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise RuntimeError(
                "map-sync runtime directory must be owned by this user and not group/world writable"
            )

        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        lock_path = self.runtime_dir / ".fleet.lock"
        fd = os.open(lock_path, flags, 0o600)
        try:
            lock = os.fstat(fd)
            if (
                not stat.S_ISREG(lock.st_mode)
                or lock.st_uid != os.geteuid()
                or lock.st_nlink != 1
            ):
                raise RuntimeError("map-sync runtime lock must be an owned regular file")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"map-sync runtime directory is already in use: {self.runtime_dir}"
                ) from error
        except BaseException:
            os.close(fd)
            raise
        self._runtime_lock_fd = fd

    def _write_config(self, path: Path, value: dict[str, object]) -> None:
        fd = -1
        temporary_path: Path | None = None
        try:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=self.runtime_dir,
            )
            temporary_path = Path(temporary_name)
            self._config_paths.append(temporary_path)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                fd = -1
                json.dump(value, output, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
            self._config_paths.remove(temporary_path)
            temporary_path = None
            _sync_directory(self.runtime_dir)
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                finally:
                    if temporary_path in self._config_paths:
                        self._config_paths.remove(temporary_path)

    def _release_runtime_lease(self) -> None:
        fd = self._runtime_lock_fd
        self._runtime_lock_fd = None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    async def _ensure_identity(self, vehicle_id: str) -> str:
        process = await asyncio.create_subprocess_exec(
            str(self.sidecar_path),
            "identity",
            str(self.runtime_dir / f"{vehicle_id}.key"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.settings.startup_timeout_s,
            )
        except asyncio.TimeoutError:
            cleanup_task = asyncio.create_task(
                _terminate_and_reap(process, terminate_first=True)
            )
            cancellation, cleanup_error = await _finish_cleanup(cleanup_task)
            error = RuntimeError(f"timed out initializing {vehicle_id} libp2p identity")
            if cancellation is not None:
                raise cancellation from cleanup_error or error
            if cleanup_error is not None:
                raise error from cleanup_error
            raise error from None
        except BaseException as original:
            cleanup_task = asyncio.create_task(
                _terminate_and_reap(process, terminate_first=True)
            )
            cancellation, cleanup_error = await _finish_cleanup(cleanup_task)
            if isinstance(original, asyncio.CancelledError):
                raise original from cleanup_error
            if cancellation is not None:
                raise cancellation from cleanup_error or original
            if cleanup_error is not None:
                raise original from cleanup_error
            raise
        if process.returncode != 0:
            raise RuntimeError(
                f"cannot initialize {vehicle_id} libp2p identity: {stderr.decode(errors='replace').strip()}"
            )
        peer_id = stdout.decode().strip()
        if not peer_id or any(character.isspace() for character in peer_id):
            raise RuntimeError(f"invalid PeerId returned for {vehicle_id}")
        return peer_id

    async def _wait_for_full_mesh(self, expected_connections: int) -> None:
        while True:
            if all(
                state.connected_count >= expected_connections
                for state in self.states.values()
            ):
                return
            failed = [
                vehicle_id
                for vehicle_id, bridge in self._bridges.items()
                if bridge.process is not None and bridge.process.returncode is not None
            ]
            if failed:
                raise RuntimeError(f"map-sync sidecars exited during startup: {failed}")
            await asyncio.sleep(0.05)

    def flush_once(self) -> None:
        if len(self.vehicles) <= 1:
            return
        for vehicle_id, state in self.states.items():
            payload = state.prepare_delta()
            if payload is None:
                continue
            if not self._bridges[vehicle_id].enqueue(payload):
                state.publish_result(int(payload["sequence"]), False)

    async def _flush_loop(self) -> None:
        interval_s = self.settings.sync_interval_ms / 1000
        while True:
            started = asyncio.get_running_loop().time()
            self.flush_once()
            await asyncio.sleep(max(0.0, interval_s - (asyncio.get_running_loop().time() - started)))

    async def _close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        pending_cleanup = False
        if self._flush_task is not None:
            flush_task = self._flush_task
            flush_task.cancel()
            done, pending = await asyncio.wait(
                (flush_task,),
                timeout=PROCESS_STOP_TIMEOUT_S,
            )
            if pending:
                pending_cleanup = True
                errors.append(TimeoutError("map-sync flush task did not stop"))
            else:
                self._flush_task = None
                if not flush_task.cancelled():
                    try:
                        flush_task.result()
                    except BaseException as error:
                        errors.append(error)
        if self._bridges:
            close_tasks = {
                vehicle_id: asyncio.create_task(bridge.close())
                for vehicle_id, bridge in self._bridges.items()
            }
            done, pending = await asyncio.wait(
                tuple(close_tasks.values()),
                timeout=PROCESS_STOP_TIMEOUT_S * 3,
            )
            for task in pending:
                task.cancel()
                pending_cleanup = True
                errors.append(TimeoutError("map-sync bridge close did not stop"))
            for vehicle_id, task in close_tasks.items():
                if task not in done:
                    continue
                if task.cancelled():
                    pending_cleanup = True
                    errors.append(asyncio.CancelledError())
                    continue
                try:
                    task.result()
                except _CleanupPending as error:
                    pending_cleanup = True
                    errors.append(error)
                    continue
                except BaseException as error:
                    errors.append(error)
                self._bridges.pop(vehicle_id, None)
        for path in tuple(dict.fromkeys(self._config_paths)):
            try:
                path.unlink(missing_ok=True)
            except BaseException as error:
                errors.append(error)
        self._config_paths.clear()
        if not pending_cleanup and self._flush_task is None and not self._bridges:
            try:
                self._release_runtime_lease()
            except BaseException as error:
                errors.append(error)
            self._closed = True
        if errors:
            error_type = _CleanupPending if pending_cleanup else _CleanupError
            raise error_type("cannot close libp2p fleet sync", errors)

    async def close(self) -> None:
        if self._closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        close_task = self._close_task
        cancellation, error = await _finish_cleanup(close_task)
        if close_task.done() and self._close_task is close_task:
            self._close_task = None
        if cancellation is not None:
            if error is not None:
                raise cancellation from error
            raise cancellation
        if error is not None:
            raise error
