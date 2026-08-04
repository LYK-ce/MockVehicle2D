"""Deterministic shared-world simulation for one to four isolated robot nodes."""

from __future__ import annotations

import asyncio
from collections import deque
import copy
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import signal
import time

from mockvehicle2d.collision import (
    is_strict_overlap,
    is_swept_circle_passable,
    swept_trajectories_overlap,
)
from mockvehicle2d.controller import (
    SUPPORTED_MISSION_TYPES,
    Command,
    CommandResult,
    RobotController,
)
from mockvehicle2d.local_state import (
    FORBIDDEN,
    OCCUPIED,
    AnchorSpec,
    AnchoredLocalState,
    LocalMapDelta,
    MapCellUpdate,
    OdometryConfig,
    ObservedGrid,
    PoseEstimate,
    _hit_axis_cell,
)
from mockvehicle2d.map_grid import MapGrid
from mockvehicle2d.map_sync import (
    MapSyncState,
    P2PFleetSync,
    P2PSettings,
    P2PVehicleConfig,
    PeerVehicleState,
    _CleanupError,
    _CleanupPending,
    _finish_cleanup,
    fleet_sync_cleanup_timeout_s,
)
from mockvehicle2d.protocol import ProtocolError, command_ack, error_message, parse_command
from mockvehicle2d.safety import (
    AUTOMATIC_MINIMUM_CLEARANCE_M,
    LocalSafetyRuntime,
    SafetyAdvanceResult,
)
from mockvehicle2d.scan import LaserPoint, TMINI_SCAN_CONFIG, scan_grid, scan_message
from mockvehicle2d.server import (
    DEFAULT_REALTIME_FACTOR,
    HOST,
    LOCAL_MAP_RESOLUTION_M,
    MAP_RESOLUTION_M,
    RuntimeFrame,
    _encode_map_chunks,
    _send_json,
    _send_map_chunks,
    _send_pending_events,
    generate_map,
    validate_vehicle_id,
)
from mockvehicle2d.vehicle import (
    DEFAULT_ANGULAR_ACCELERATION_RPS2,
    DEFAULT_LINEAR_ACCELERATION_MPS2,
    DEFAULT_LINEAR_DECELERATION_MPS2,
    TimedPose,
    Vehicle,
)


MAX_VEHICLES = 4
DEFAULT_TICK_MS = 100
DEFAULT_SPAWN_SAFETY_MARGIN_M = 0.25
FLEET_CLEANUP_TIMEOUT_S = 2.0
TELEMETRY_BUFFER_FRAMES = 64
PeerExclusion = tuple[float, float, float]


class TelemetryOverflowError(RuntimeError):
    def __init__(self, requested_after: int, oldest: int, latest: int) -> None:
        super().__init__(
            f"telemetry cursor {requested_after} is older than buffered range "
            f"{oldest}..{latest}"
        )
        self.oldest = oldest
        self.latest = latest


def _vehicle_odometry_config(config: OdometryConfig, vehicle_id: str) -> OdometryConfig:
    material = f"{config.seed}\0{vehicle_id}".encode()
    seed = int.from_bytes(sha256(material).digest()[:8], "big")
    return replace(config, seed=seed)


def _strict_object(
    value: object,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    fields = frozenset(value)
    if not required <= fields or not fields <= allowed:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class AnchorPose:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.x_m, "anchor_pose.x_m"),
            (self.y_m, "anchor_pose.y_m"),
            (self.yaw_rad, "anchor_pose.yaw_rad"),
        ):
            _finite_number(value, name)

    @classmethod
    def from_json(cls, value: object) -> "AnchorPose":
        body = _strict_object(
            value,
            required=frozenset(("x_m", "y_m", "yaw_rad")),
            allowed=frozenset(("x_m", "y_m", "yaw_rad")),
            name="anchor_pose",
        )
        return cls(
            _finite_number(body["x_m"], "anchor_pose.x_m"),
            _finite_number(body["y_m"], "anchor_pose.y_m"),
            _finite_number(body["yaw_rad"], "anchor_pose.yaw_rad"),
        )


@dataclass(frozen=True)
class FleetVehicleSpec:
    vehicle_id: str
    operator_port: int
    spawn_id: str
    anchor_pose: AnchorPose
    p2p_port: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or not isinstance(self.spawn_id, str):
            raise ValueError("vehicle_id and spawn_id must be strings")
        validate_vehicle_id(self.vehicle_id)
        validate_vehicle_id(self.spawn_id)
        if (
            isinstance(self.operator_port, bool)
            or not isinstance(self.operator_port, int)
            or not 1 <= self.operator_port <= 65535
        ):
            raise ValueError("operator_port must be an integer from 1 to 65535")
        if not isinstance(self.anchor_pose, AnchorPose):
            raise ValueError("anchor_pose must be an AnchorPose")
        if self.p2p_port is not None and (
            isinstance(self.p2p_port, bool)
            or not isinstance(self.p2p_port, int)
            or not 1 <= self.p2p_port <= 65535
        ):
            raise ValueError("p2p_port must be an integer from 1 to 65535")

    @classmethod
    def from_json(cls, value: object) -> "FleetVehicleSpec":
        body = _strict_object(
            value,
            required=frozenset(("vehicle_id", "operator_port", "spawn_id", "anchor_pose")),
            allowed=frozenset(
                ("vehicle_id", "operator_port", "spawn_id", "anchor_pose", "p2p_port")
            ),
            name="vehicle",
        )
        vehicle_id = body["vehicle_id"]
        spawn_id = body["spawn_id"]
        port = body["operator_port"]
        if not isinstance(vehicle_id, str):
            raise ValueError("vehicle_id must be a string")
        if not isinstance(spawn_id, str):
            raise ValueError("spawn_id must be a string")
        validate_vehicle_id(vehicle_id)
        validate_vehicle_id(spawn_id)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("operator_port must be an integer from 1 to 65535")
        p2p_port = body.get("p2p_port")
        if p2p_port is not None and (
            isinstance(p2p_port, bool)
            or not isinstance(p2p_port, int)
            or not 1 <= p2p_port <= 65535
        ):
            raise ValueError("p2p_port must be an integer from 1 to 65535")
        return cls(
            vehicle_id,
            port,
            spawn_id,
            AnchorPose.from_json(body["anchor_pose"]),
            p2p_port,
        )


@dataclass(frozen=True)
class FleetScenario:
    scenario_id: str
    vehicles: tuple[FleetVehicleSpec, ...]
    tick_ms: int = DEFAULT_TICK_MS
    p2p: P2PSettings | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str):
            raise ValueError("scenario_id must be a string")
        validate_vehicle_id(self.scenario_id)
        if not isinstance(self.vehicles, tuple) or any(
            not isinstance(spec, FleetVehicleSpec) for spec in self.vehicles
        ):
            raise ValueError("vehicles must be a tuple of FleetVehicleSpec values")
        if not 1 <= len(self.vehicles) <= MAX_VEHICLES:
            raise ValueError("scenario must contain from 1 to 4 vehicles")
        if isinstance(self.tick_ms, bool) or not isinstance(self.tick_ms, int):
            raise ValueError("tick_ms must be an integer")
        if not 10 <= self.tick_ms <= 1000:
            raise ValueError("tick_ms must be from 10 to 1000")
        if self.p2p is not None and not isinstance(self.p2p, P2PSettings):
            raise ValueError("p2p must be P2PSettings")
        for field_name, values in (
            ("vehicle_id", [spec.vehicle_id for spec in self.vehicles]),
            ("operator_port", [spec.operator_port for spec in self.vehicles]),
            ("spawn_id", [spec.spawn_id for spec in self.vehicles]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"scenario {field_name} values must be unique")
        p2p_ports = [spec.p2p_port for spec in self.vehicles]
        if self.p2p is None and any(port is not None for port in p2p_ports):
            raise ValueError("p2p_port requires top-level p2p configuration")
        if self.p2p is not None:
            if any(port is None for port in p2p_ports):
                raise ValueError("every vehicle requires p2p_port when p2p is enabled")
            if len(p2p_ports) != len(set(p2p_ports)):
                raise ValueError("scenario p2p_port values must be unique")
            if set(p2p_ports) & {spec.operator_port for spec in self.vehicles}:
                raise ValueError("operator and p2p ports must not overlap")

    @property
    def tick_s(self) -> float:
        return self.tick_ms / 1000

    @classmethod
    def from_json(cls, value: object) -> "FleetScenario":
        body = _strict_object(
            value,
            required=frozenset(("scenario_id", "vehicles")),
            allowed=frozenset(("scenario_id", "vehicles", "tick_ms", "p2p")),
            name="scenario",
        )
        scenario_id = body["scenario_id"]
        vehicles = body["vehicles"]
        tick_ms = body.get("tick_ms", DEFAULT_TICK_MS)
        if not isinstance(scenario_id, str):
            raise ValueError("scenario_id must be a string")
        if not isinstance(vehicles, list):
            raise ValueError("vehicles must be an array")
        if isinstance(tick_ms, bool) or not isinstance(tick_ms, int):
            raise ValueError("tick_ms must be an integer")
        return cls(
            scenario_id,
            tuple(FleetVehicleSpec.from_json(item) for item in vehicles),
            tick_ms,
            None if "p2p" not in body else P2PSettings.from_json(body["p2p"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FleetScenario":
        try:
            raw = Path(path).read_text(encoding="utf-8")
            return cls.from_json(json.loads(raw))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load fleet scenario: {error}") from error


class SharedWorld:
    """Simulator-only truth, physics, sensing and simultaneous motion arbitration."""

    def __init__(
        self,
        grid: MapGrid,
        voxels: list[dict[str, object]],
        specs: tuple[FleetVehicleSpec, ...],
        *,
        radius: float,
        linear_speed: float,
        angular_speed: float,
        linear_acceleration_mps2: float,
        linear_deceleration_mps2: float,
        angular_acceleration_rps2: float,
        command_timeout: float,
        started_at: float,
        spawn_safety_margin_m: float = DEFAULT_SPAWN_SAFETY_MARGIN_M,
    ) -> None:
        parameters = (
            radius,
            linear_speed,
            angular_speed,
            linear_acceleration_mps2,
            linear_deceleration_mps2,
            angular_acceleration_rps2,
            command_timeout,
            started_at,
            spawn_safety_margin_m,
        )
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("world parameters must be finite")
        if min(
            radius,
            linear_speed,
            angular_speed,
            linear_acceleration_mps2,
            linear_deceleration_mps2,
            angular_acceleration_rps2,
            command_timeout,
        ) <= 0:
            raise ValueError("vehicle motion limits, radius, and timeout must be positive")
        if spawn_safety_margin_m < 0:
            raise ValueError("spawn safety margin cannot be negative")
        if not 1 <= len(specs) <= MAX_VEHICLES:
            raise ValueError("world must contain from 1 to 4 vehicles")

        for field_name, values in (
            ("vehicle_id", [spec.vehicle_id for spec in specs]),
            ("operator_port", [spec.operator_port for spec in specs]),
            ("spawn_id", [spec.spawn_id for spec in specs]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"world {field_name} values must be unique")

        expanded_radius = radius + spawn_safety_margin_m
        for spec in specs:
            pose = spec.anchor_pose
            if not is_swept_circle_passable(
                grid,
                pose.x_m,
                pose.y_m,
                pose.x_m,
                pose.y_m,
                expanded_radius,
            ):
                raise ValueError(
                    f"spawn {spec.spawn_id} is outside the world or intersects static terrain"
                )
        for index, first in enumerate(specs):
            for second in specs[index + 1 :]:
                distance_squared = (
                    (first.anchor_pose.x_m - second.anchor_pose.x_m) ** 2
                    + (first.anchor_pose.y_m - second.anchor_pose.y_m) ** 2
                )
                if is_strict_overlap(distance_squared, (2 * expanded_radius) ** 2):
                    raise ValueError(
                        f"spawns {first.spawn_id} and {second.spawn_id} overlap"
                    )

        self._grid = grid
        self._voxels = voxels
        self._vehicles = {
            spec.vehicle_id: Vehicle(
                spec.anchor_pose.x_m,
                spec.anchor_pose.y_m,
                spec.anchor_pose.yaw_rad,
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                linear_acceleration_mps2=linear_acceleration_mps2,
                linear_deceleration_mps2=linear_deceleration_mps2,
                angular_acceleration_rps2=angular_acceleration_rps2,
                radius=radius,
                command_timeout=command_timeout,
                now=started_at,
            )
            for spec in specs
        }
        self.now = started_at

    @property
    def debug_grid(self) -> MapGrid:
        """Ground truth for physics and operator-only map display."""
        return self._grid

    @property
    def debug_voxels(self) -> list[dict[str, object]]:
        return self._voxels

    def vehicle(self, vehicle_id: str) -> Vehicle:
        return self._vehicles[vehicle_id]

    def truth_snapshot(self) -> dict[str, tuple[float, float, float]]:
        return {
            vehicle_id: (vehicle.x, vehicle.y, vehicle.yaw)
            for vehicle_id, vehicle in sorted(self._vehicles.items())
        }

    def scan(self, vehicle_id: str) -> tuple[LaserPoint, ...]:
        vehicle = self.vehicle(vehicle_id)
        circles = tuple(
            (other.x, other.y, other.radius)
            for other_id, other in sorted(self._vehicles.items())
            if other_id != vehicle_id
        )
        return tuple(
            scan_grid(
                self._grid,
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                TMINI_SCAN_CONFIG,
                circles=circles,
            )
        )

    def sensor_grid(self, vehicle_id: str) -> MapGrid:
        """Build the bounded truth projection consumed only by local safety sensors."""
        vehicle = self.vehicle(vehicle_id)
        sensed = MapGrid(self._grid.width, self._grid.height)
        reach = math.ceil(TMINI_SCAN_CONFIG.max_range + vehicle.radius) + 1
        center_x, center_y = math.floor(vehicle.x), math.floor(vehicle.y)
        for gy in range(max(0, center_y - reach), min(self._grid.height, center_y + reach + 1)):
            for gx in range(max(0, center_x - reach), min(self._grid.width, center_x + reach + 1)):
                sensed.set_cell(gx, gy, self._grid.get_cell(gx, gy))

        return sensed

    def advance_to(self, target_time: float) -> dict[str, SafetyAdvanceResult]:
        if not math.isfinite(target_time) or target_time < self.now:
            raise ValueError("world time must be finite and monotonic")
        starts = {
            vehicle_id: (vehicle.x, vehicle.y)
            for vehicle_id, vehicle in self._vehicles.items()
        }
        candidates: dict[str, Vehicle] = {}
        trajectories: dict[str, tuple[TimedPose, ...]] = {}
        results: dict[str, SafetyAdvanceResult] = {}
        for vehicle_id, vehicle in sorted(self._vehicles.items()):
            candidate = copy.copy(vehicle)
            trajectory: list[TimedPose] = []
            collided = candidate.advance(
                self._grid,
                target_time,
                trajectory=trajectory,
            )
            candidates[vehicle_id] = candidate
            trajectories[vehicle_id] = tuple(trajectory)
            results[vehicle_id] = SafetyAdvanceResult(collided=collided)

        blocked: set[str] = set()
        vehicle_ids = sorted(candidates)
        stationary = {
            vehicle_id: (
                (
                    (self.now, *start, self._vehicles[vehicle_id].yaw),
                    (target_time, *start, self._vehicles[vehicle_id].yaw),
                )
                if target_time > self.now
                else ((self.now, *start, self._vehicles[vehicle_id].yaw),)
            )
            for vehicle_id, start in starts.items()
        }
        while True:
            newly_blocked: set[str] = set()
            for index, first_id in enumerate(vehicle_ids):
                first = candidates[first_id]
                first_trajectory = (
                    stationary[first_id]
                    if first_id in blocked
                    else trajectories[first_id]
                )
                for second_id in vehicle_ids[index + 1 :]:
                    second = candidates[second_id]
                    second_trajectory = (
                        stationary[second_id]
                        if second_id in blocked
                        else trajectories[second_id]
                    )
                    if swept_trajectories_overlap(
                        first_trajectory,
                        first.radius,
                        second_trajectory,
                        second.radius,
                    ):
                        newly_blocked.update((first_id, second_id))
            newly_blocked -= blocked
            if not newly_blocked:
                break
            blocked.update(newly_blocked)

        for vehicle_id in blocked:
            stopped = copy.copy(self._vehicles[vehicle_id])
            stopped.force_stop(target_time)
            stopped.collision = False
            candidates[vehicle_id] = stopped
            trajectories[vehicle_id] = stationary[vehicle_id]
            results[vehicle_id] = SafetyAdvanceResult(
                stopped=True,
                reason="safety_obstacle",
            )
        self._vehicles = candidates
        self.now = target_time
        return results


@dataclass
class FleetSensorFrame:
    sequence: int
    frame: RuntimeFrame
    truth_pose: tuple[float, float, float]
    estimate: PoseEstimate
    runtime_state: dict[str, object]


class _TransientPlanningGrid:
    """Overlay current dynamic LiDAR hits without changing the persistent map."""

    def __init__(self, persistent: ObservedGrid) -> None:
        self._persistent = persistent
        self._cells: set[tuple[int, int]] = set()
        self._occupied_cells: set[tuple[int, int]] = set()
        self._peer_forbidden_cells: set[tuple[int, int]] = set()
        self._peer_exclusion_circles: tuple[PeerExclusion, ...] = ()
        self.resolution_m = persistent.resolution_m

    def update(
        self,
        cells: set[tuple[int, int]],
        persistent_delta: LocalMapDelta | None,
        *,
        peer_forbidden_cells: set[tuple[int, int]] | None = None,
        peer_exclusion_circles: tuple[PeerExclusion, ...] = (),
    ) -> LocalMapDelta:
        peers = set() if peer_forbidden_cells is None else peer_forbidden_cells
        combined = cells | peers
        changed = self._cells ^ combined
        changed.update(self._peer_forbidden_cells ^ peers)
        if persistent_delta is not None:
            changed.update(
                (update.gx, update.gy)
                for update in persistent_delta.changed_cells
            )
        self._cells = combined
        self._occupied_cells = set(cells)
        self._peer_forbidden_cells = set(peers)
        self._peer_exclusion_circles = peer_exclusion_circles
        return LocalMapDelta(
            tuple(
                MapCellUpdate(gx, gy, self.get_cell(gx, gy))
                for gx, gy in sorted(changed, key=lambda cell: (cell[1], cell[0]))
            ),
            tuple(sorted(peers, key=lambda cell: (cell[1], cell[0]))),
        )

    def get_cell(self, gx: int, gy: int) -> int:
        if (gx, gy) in self._peer_forbidden_cells:
            return FORBIDDEN
        return (
            OCCUPIED
            if (gx, gy) in self._cells
            else self._persistent.get_cell(gx, gy)
        )

    def is_unknown(self, gx: int, gy: int) -> bool:
        return (gx, gy) not in self._cells and self._persistent.is_unknown(gx, gy)

    def is_forbidden(self, gx: int, gy: int) -> bool:
        return (gx, gy) in self._peer_forbidden_cells or (
            (gx, gy) not in self._cells and self._persistent.is_forbidden(gx, gy)
        )

    def cell_without_peers(self, gx: int, gy: int) -> int:
        return (
            OCCUPIED
            if (gx, gy) in self._occupied_cells
            else self._persistent.get_cell(gx, gy)
        )

    def peer_exclusion_circles(self) -> tuple[PeerExclusion, ...]:
        return self._peer_exclusion_circles

    def snapshot(self) -> dict[str, object]:
        snapshot = self._persistent.snapshot()
        states = {
            (cell["gx"], cell["gy"]): cell["state"]
            for cell in snapshot["cells"]
        }
        states.update((cell, OCCUPIED) for cell in self._cells)
        states.update((cell, FORBIDDEN) for cell in self._peer_forbidden_cells)
        snapshot["cells"] = [
            MapCellUpdate(gx, gy, state).as_dict()
            for (gx, gy), state in sorted(
                states.items(), key=lambda item: (item[0][1], item[0][0])
            )
        ]
        snapshot["peer_forbidden_cells"] = [
            {"gx": gx, "gy": gy}
            for gx, gy in sorted(
                self._peer_forbidden_cells,
                key=lambda cell: (cell[1], cell[0]),
            )
        ]
        return snapshot


def _dynamic_hit_cells(
    points: tuple[LaserPoint, ...],
    pose: PoseEstimate,
    resolution_m: float,
) -> set[tuple[int, int]]:
    cells = set()
    for point in points:
        if not point.dynamic or point.range <= 0:
            continue
        angle = pose.yaw_rad + point.angle
        direction_x, direction_y = math.cos(angle), math.sin(angle)
        cells.add(
            (
                _hit_axis_cell(
                    pose.x_m + point.range * direction_x,
                    direction_x,
                    resolution_m,
                ),
                _hit_axis_cell(
                    pose.y_m + point.range * direction_y,
                    direction_y,
                    resolution_m,
                ),
            )
        )
    return cells


def _peer_vehicle_exclusions(
    states: tuple[PeerVehicleState, ...],
    anchor: AnchorSpec,
    resolution_m: float,
    own_radius_m: float,
) -> tuple[
    set[tuple[int, int]],
    tuple[PeerExclusion, ...],
    tuple[PeerExclusion, ...],
]:
    cells = set()
    circles = []
    hit_envelopes = []
    for state in states:
        x_m, y_m, _ = anchor.global_to_anchor(
            state.global_x_m,
            state.global_y_m,
            state.global_yaw_rad,
        )
        uncertainty_m = math.sqrt(max(state.covariance[:2]))
        exclusion_radius_m = (
            own_radius_m
            + state.radius_m
            + uncertainty_m
            + AUTOMATIC_MINIMUM_CLEARANCE_M
        )
        circles.append((x_m, y_m, exclusion_radius_m))
        hit_envelopes.append(
            (
                x_m,
                y_m,
                state.radius_m
                + uncertainty_m,
            )
        )
        min_gx = math.ceil(
            (x_m - exclusion_radius_m) / resolution_m - 0.5 - 1e-12
        )
        max_gx = math.floor(
            (x_m + exclusion_radius_m) / resolution_m - 0.5 + 1e-12
        )
        min_gy = math.ceil(
            (y_m - exclusion_radius_m) / resolution_m - 0.5 - 1e-12
        )
        max_gy = math.floor(
            (y_m + exclusion_radius_m) / resolution_m - 0.5 + 1e-12
        )
        cells.update(
            (gx, gy)
            for gy in range(min_gy, max_gy + 1)
            for gx in range(min_gx, max_gx + 1)
            if math.hypot(
                (gx + 0.5) * resolution_m - x_m,
                (gy + 0.5) * resolution_m - y_m,
            )
            <= exclusion_radius_m + 1e-12
        )
    return cells, tuple(circles), tuple(hit_envelopes)


def _cell_intersects_circle(
    cell: tuple[int, int],
    resolution_m: float,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
) -> bool:
    min_x, min_y = cell[0] * resolution_m, cell[1] * resolution_m
    max_x, max_y = min_x + resolution_m, min_y + resolution_m
    closest_x = max(min_x, min(center_x_m, max_x))
    closest_y = max(min_y, min(center_y_m, max_y))
    return math.hypot(closest_x - center_x_m, closest_y - center_y_m) <= (
        radius_m + 1e-12
    )


@dataclass
class RobotNode:
    """Vehicle-owned controller, odometry and local map; no shared-world truth."""

    spec: FleetVehicleSpec
    controller: RobotController
    safety: LocalSafetyRuntime
    local_state: AnchoredLocalState
    map_sync: MapSyncState | None = None
    frame_sequence: int = 0
    latest_frame: RuntimeFrame | None = None
    _frames: deque[FleetSensorFrame] = field(
        default_factory=lambda: deque(maxlen=TELEMETRY_BUFFER_FRAMES),
        repr=False,
    )
    controller_lease: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _pending_map_delta: LocalMapDelta | None = field(default=None, repr=False)
    _pending_advance: SafetyAdvanceResult = field(
        default_factory=SafetyAdvanceResult,
        repr=False,
    )
    _planning_map: _TransientPlanningGrid = field(init=False, repr=False)
    _lidar_dynamic_cells: set[tuple[int, int]] = field(
        default_factory=set,
        repr=False,
    )
    _own_radius_m: float = field(default=0.5, init=False, repr=False)
    _latest_scan_monotonic_s: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._planning_map = _TransientPlanningGrid(self.local_state.local_map)

    def control(self, vehicle: Vehicle, sensor_grid: MapGrid, now: float) -> None:
        self._own_radius_m = vehicle.radius
        peer_states = (
            () if self.map_sync is None else self.map_sync.peer_vehicle_states()
        )
        self._update_planning_map(peer_states=peer_states)
        scan_age_s = (
            None
            if self._latest_scan_monotonic_s is None
            else now - self._latest_scan_monotonic_s
        )
        scan_fresh = (
            self.latest_frame is not None
            and scan_age_s is not None
            and -1e-12 <= scan_age_s <= TMINI_SCAN_CONFIG.scan_time + 1e-12
        )
        self.controller.tick(
            vehicle=vehicle,
            grid=sensor_grid,
            safety=self.safety,
            anchor=self.local_state.anchor,
            pose=self.local_state.pose,
            local_map=self._planning_map,
            map_delta=self._pending_map_delta,
            advance_result=self._pending_advance,
            now=now,
            safety_scan_points=(
                self.latest_frame.scan_points if scan_fresh else ()
            ),
            safety_scan_healthy=scan_fresh,
        )
        self._pending_map_delta = None
        self._pending_advance = SafetyAdvanceResult()

    def _update_planning_map(
        self,
        *,
        peer_states: tuple[PeerVehicleState, ...] | None = None,
        persistent_delta: LocalMapDelta | None = None,
    ) -> None:
        if peer_states is None:
            peer_states = (
                () if self.map_sync is None else self.map_sync.peer_vehicle_states()
            )
        peer_cells, peer_circles, peer_hit_envelopes = _peer_vehicle_exclusions(
            peer_states,
            self.local_state.anchor,
            self.local_state.local_map.resolution_m,
            self._own_radius_m,
        )
        dynamic_cells = {
            cell
            for cell in self._lidar_dynamic_cells
            if all(
                not _cell_intersects_circle(
                    cell,
                    self.local_state.local_map.resolution_m,
                    peer_x_m,
                    peer_y_m,
                    radius_m,
                )
                for peer_x_m, peer_y_m, radius_m in peer_hit_envelopes
            )
        }
        planning_delta = self._planning_map.update(
            dynamic_cells,
            persistent_delta,
            peer_forbidden_cells=peer_cells,
            peer_exclusion_circles=peer_circles,
        )
        if not planning_delta.changed_cells:
            return
        pending = {
            (update.gx, update.gy): update
            for update in (
                ()
                if self._pending_map_delta is None
                else self._pending_map_delta.changed_cells
            )
        }
        pending.update(
            ((update.gx, update.gy), update)
            for update in planning_delta.changed_cells
        )
        self._pending_map_delta = LocalMapDelta(
            tuple(
                pending[cell]
                for cell in sorted(pending, key=lambda cell: (cell[1], cell[0]))
            ),
            planning_delta.peer_forbidden_cells,
        )

    def sample(
        self,
        truth_pose: tuple[float, float, float],
        scan_points: tuple[LaserPoint, ...],
        wall_timestamp: float,
        monotonic_timestamp: float,
        vehicle: Vehicle,
    ) -> None:
        truth_x_m, truth_y_m, truth_yaw_rad = truth_pose
        self._own_radius_m = vehicle.radius
        self._latest_scan_monotonic_s = monotonic_timestamp
        self.local_state.update_from_truth(
            truth_x_m,
            truth_y_m,
            truth_yaw_rad,
            timestamp=wall_timestamp,
        )
        persistent_delta = self.local_state.match_and_integrate_scan(
            scan_points,
            wall_timestamp,
            TMINI_SCAN_CONFIG,
            forbidden_points_vehicle_m=(
                ()
                if self.safety.observation.edge_point_vehicle_m is None
                else (self.safety.observation.edge_point_vehicle_m,)
            ),
        )
        self._lidar_dynamic_cells = _dynamic_hit_cells(
            scan_points,
            self.local_state.pose,
            self.local_state.local_map.resolution_m,
        )
        linear_mps, omega_rps = vehicle.body_velocities()
        if self.map_sync is not None:
            self.map_sync.record_local(persistent_delta)
            self.map_sync.record_vehicle_state(
                self.local_state.pose,
                radius_m=vehicle.radius,
                linear_mps=linear_mps,
                omega_rps=omega_rps,
            )
        self._update_planning_map(persistent_delta=persistent_delta)
        self.latest_frame = RuntimeFrame(scan_points, wall_timestamp)
        estimate = self.local_state.pose
        runtime_state = {
            "linear_mps": linear_mps,
            "omega_rps": omega_rps,
            "collision": vehicle.collision,
            "actuator_command": vehicle.command,
            "controller": self.controller.snapshot(now=vehicle.last_update),
            "safety": self.safety.snapshot(),
            "localization": {
                **estimate.as_dict(),
                "local_map_revision": self.local_state.local_map.revision,
                **(
                    {}
                    if self.local_state.last_scan_match is None
                    else {"scan_match": self.local_state.last_scan_match.as_dict()}
                ),
            },
            "p2p_map_sync": (
                {"enabled": False}
                if self.map_sync is None
                else self.map_sync.snapshot()
            ),
        }
        self._frames.append(
            FleetSensorFrame(
                self.frame_sequence,
                self.latest_frame,
                truth_pose,
                estimate,
                runtime_state,
            )
        )

    def frames_after(self, sequence: int) -> tuple[FleetSensorFrame, ...]:
        oldest = self._frames[0].sequence
        latest = self._frames[-1].sequence
        if sequence < oldest - 1:
            raise TelemetryOverflowError(sequence, oldest, latest)
        return tuple(frame for frame in self._frames if frame.sequence > sequence)

    def record_advance(self, result: SafetyAdvanceResult) -> None:
        previous = self._pending_advance
        self._pending_advance = SafetyAdvanceResult(
            collided=previous.collided or result.collided,
            stopped=previous.stopped or result.stopped,
            reason=previous.reason or result.reason,
        )


class FleetRuntime:
    """One deterministic clock coordinating isolated robot nodes in a shared world."""

    def __init__(
        self,
        scenario: FleetScenario,
        world: SharedWorld,
        nodes: dict[str, RobotNode],
        wall_time_offset: float,
        realtime_factor: float,
    ) -> None:
        self.scenario = scenario
        self.world = world
        self.nodes = nodes
        self._wall_time_offset = wall_time_offset
        self.realtime_factor = realtime_factor
        self._sensor_epoch = world.now
        self._next_scan_index = {vehicle_id: 1 for vehicle_id in nodes}
        self.map_chunks = _encode_map_chunks(world.debug_voxels, world.debug_grid.width)

    @classmethod
    def create(
        cls,
        scenario: FleetScenario,
        *,
        started_at: float = 0.0,
        timestamp: float = 0.0,
        grid: MapGrid | None = None,
        voxels: list[dict[str, object]] | None = None,
        linear_speed: float = 0.5,
        angular_speed: float = math.pi / 2,
        linear_acceleration_mps2: float = DEFAULT_LINEAR_ACCELERATION_MPS2,
        linear_deceleration_mps2: float = DEFAULT_LINEAR_DECELERATION_MPS2,
        angular_acceleration_rps2: float = DEFAULT_ANGULAR_ACCELERATION_RPS2,
        radius: float = 0.5,
        command_timeout: float = 1.0,
        mission_capacity: int = 16,
        odometry_config: OdometryConfig = OdometryConfig(),
        safety_healthy: bool = True,
        spawn_safety_margin_m: float = DEFAULT_SPAWN_SAFETY_MARGIN_M,
        realtime_factor: float = 1.0,
    ) -> "FleetRuntime":
        if not math.isfinite(realtime_factor) or realtime_factor <= 0:
            raise ValueError("realtime factor must be finite and positive")
        if grid is None:
            generated_voxels, grid = generate_map(radius=radius)
            voxels = generated_voxels
        elif voxels is None:
            voxels = [
                {
                    "gx": gx,
                    "gy": gy,
                    "gz": 0,
                    "state": grid.get_cell(gx, gy),
                    "conf": 1.0,
                }
                for gy in range(grid.height)
                for gx in range(grid.width)
            ]
        assert voxels is not None
        world = SharedWorld(
            grid,
            voxels,
            scenario.vehicles,
            radius=radius,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            linear_acceleration_mps2=linear_acceleration_mps2,
            linear_deceleration_mps2=linear_deceleration_mps2,
            angular_acceleration_rps2=angular_acceleration_rps2,
            command_timeout=command_timeout,
            started_at=started_at,
            spawn_safety_margin_m=spawn_safety_margin_m,
        )
        nodes = {}
        for spec in scenario.vehicles:
            pose = spec.anchor_pose
            anchor = AnchorSpec(spec.spawn_id, pose.x_m, pose.y_m, pose.yaw_rad)
            local_state = AnchoredLocalState(
                anchor,
                truth_x_m=pose.x_m,
                truth_y_m=pose.y_m,
                truth_yaw_rad=pose.yaw_rad,
                odometry_config=_vehicle_odometry_config(
                    odometry_config,
                    spec.vehicle_id,
                ),
                timestamp=timestamp,
                map_resolution_m=LOCAL_MAP_RESOLUTION_M,
            )
            nodes[spec.vehicle_id] = RobotNode(
                spec,
                RobotController(mission_capacity=mission_capacity),
                LocalSafetyRuntime(healthy=safety_healthy),
                local_state,
                (
                    None
                    if scenario.p2p is None
                    else MapSyncState(
                        scenario.scenario_id,
                        spec.vehicle_id,
                        anchor,
                        local_state.local_map.resolution_m,
                        clock=lambda world=world: world.now,
                    )
                ),
            )
        runtime = cls(
            scenario,
            world,
            nodes,
            timestamp - started_at,
            realtime_factor,
        )
        runtime._sample_all(timestamp)
        return runtime

    @property
    def tick_s(self) -> float:
        return self.scenario.tick_s

    def timestamp_at(self, simulation_now: float | None = None) -> float:
        return self._wall_time_offset + (
            self.world.now if simulation_now is None else simulation_now
        )

    def handle_command(self, vehicle_id: str, command: Command) -> CommandResult:
        node = self.nodes[vehicle_id]
        vehicle = self.world.vehicle(vehicle_id)
        return node.controller.handle(
            command,
            vehicle=vehicle,
            grid=self.world.sensor_grid(vehicle_id),
            safety=node.safety,
            now=self.world.now,
        )

    def fail_safe_stop(self, vehicle_id: str) -> None:
        node = self.nodes[vehicle_id]
        node.controller.fail_safe_stop(
            self.world.vehicle(vehicle_id),
            "invalid_command",
        )

    def disconnect(self, vehicle_id: str) -> None:
        node = self.nodes[vehicle_id]
        node.controller.disconnect(self.world.vehicle(vehicle_id))

    def tick(self, wall_timestamp: float) -> None:
        if not math.isfinite(wall_timestamp):
            raise ValueError("wall timestamp must be finite")
        sensor_grids = {
            vehicle_id: self.world.sensor_grid(vehicle_id)
            for vehicle_id in sorted(self.nodes)
        }
        for vehicle_id in sorted(self.nodes):
            self.nodes[vehicle_id].control(
                self.world.vehicle(vehicle_id),
                sensor_grids[vehicle_id],
                self.world.now,
            )
        target_time = self.world.now + self.tick_s
        self._sample_due_scans(target_time)
        if self.world.now < target_time:
            self._advance_world(target_time)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self.tick_s / self.realtime_factor,
                )
            except asyncio.TimeoutError:
                self.tick(time.time())

    def telemetry_messages(
        self,
        vehicle_id: str,
        sampled: FleetSensorFrame | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        node = self.nodes[vehicle_id]
        if sampled is None:
            sampled = node._frames[-1]
        frame = sampled.frame
        estimate = sampled.estimate
        state = sampled.runtime_state
        x_m, y_m, yaw_rad = node.local_state.anchor.anchor_to_global(
            estimate.x_m,
            estimate.y_m,
            estimate.yaw_rad,
        )
        linear_mps = state["linear_mps"]
        pose = {
            "type": "pose",
            "timestamp_s": frame.timestamp,
            "seq": sampled.sequence,
            "source": "anchored_odometry",
            "frame_id": "global_map",
            "x_m": x_m,
            "y_m": y_m,
            "z_m": 0.0,
            "yaw_rad": yaw_rad,
            "vx_mps": linear_mps * math.cos(yaw_rad),
            "vy_mps": linear_mps * math.sin(yaw_rad),
            "omega_rps": state["omega_rps"],
            "collision": state["collision"],
            "actuator_command": state["actuator_command"],
            "controller": state["controller"],
            "safety": state["safety"],
            "localization": state["localization"],
            "p2p_map_sync": state["p2p_map_sync"],
        }
        truth_x_m, truth_y_m, truth_yaw_rad = sampled.truth_pose
        scan = scan_message(
            self.world.debug_grid,
            truth_x_m,
            truth_y_m,
            truth_yaw_rad,
            frame.timestamp,
            TMINI_SCAN_CONFIG,
            frame.scan_points,
        )
        scan["seq"] = sampled.sequence
        return pose, scan

    def _sample_all(self, wall_timestamp: float) -> None:
        scans = {
            vehicle_id: self.world.scan(vehicle_id)
            for vehicle_id in sorted(self.nodes)
        }
        for vehicle_id in sorted(self.nodes):
            self.nodes[vehicle_id].sample(
                (
                    self.world.vehicle(vehicle_id).x,
                    self.world.vehicle(vehicle_id).y,
                    self.world.vehicle(vehicle_id).yaw,
                ),
                scans[vehicle_id],
                wall_timestamp,
                self.world.now,
                self.world.vehicle(vehicle_id),
            )

    def _advance_world(self, target_time: float) -> None:
        results = self.world.advance_to(target_time)
        for vehicle_id, result in results.items():
            self.nodes[vehicle_id].record_advance(result)

    def _sample_due_scans(self, target_time: float) -> None:
        period_s = TMINI_SCAN_CONFIG.scan_time
        epsilon = 1e-12
        while True:
            scan_time = min(
                self._sensor_epoch + index * period_s
                for index in self._next_scan_index.values()
            )
            if scan_time > target_time + epsilon:
                return
            if scan_time < self.world.now - epsilon:
                raise RuntimeError("sensor schedule moved behind simulation time")
            self._advance_world(max(self.world.now, min(scan_time, target_time)))
            due = (
                vehicle_id
                for vehicle_id in sorted(self.nodes)
                if math.isclose(
                    self._sensor_epoch
                    + self._next_scan_index[vehicle_id] * period_s,
                    scan_time,
                    rel_tol=0.0,
                    abs_tol=epsilon,
                )
            )
            for vehicle_id in due:
                vehicle = self.world.vehicle(vehicle_id)
                node = self.nodes[vehicle_id]
                node.frame_sequence += 1
                node.sample(
                    (vehicle.x, vehicle.y, vehicle.yaw),
                    self.world.scan(vehicle_id),
                    self._wall_time_offset + scan_time,
                    scan_time,
                    vehicle,
                )
                self._next_scan_index[vehicle_id] += 1


def _map_metadata(grid: MapGrid) -> dict[str, object]:
    return {
        "source": "simulator_ground_truth",
        "frame_id": "simulator_map",
        "resolution_m": MAP_RESOLUTION_M,
        "width_cells": grid.width,
        "height_cells": grid.height,
        "transform_to_global_map": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        "binary_chunks": {
            "type": 0,
            "chunk_size_cells": 256,
            "header": ">Bii",
            "byte_order": "big",
            "payload_order": "row_major_y_x",
        },
    }


async def fleet_handler(websocket, *, fleet: FleetRuntime, vehicle_id: str) -> None:
    """Serve one vehicle-specific WebSocket endpoint without advancing world time."""
    node = fleet.nodes[vehicle_id]
    address = websocket.remote_address
    if node.controller_lease.locked():
        try:
            await _send_json(
                websocket,
                {
                    "type": "error",
                    "timestamp_s": fleet.timestamp_at(),
                    "seq": None,
                    "code": "vehicle_busy",
                    "message": "another controller owns the vehicle lease",
                },
            )
        except Exception:
            pass
        return

    await node.controller_lease.acquire()
    print(f"[+] {vehicle_id} controller connected: {address}")
    try:
        await _send_json(
            websocket,
            {
                "type": "hello",
                "protocol_version": 4,
                "vehicle_id": vehicle_id,
                "control_lease": "exclusive",
                "mission_frame_id": "global_map",
                "mission_types": list(SUPPORTED_MISSION_TYPES),
                "realtime_factor": fleet.realtime_factor,
                "birth_anchor": {
                    "anchor_id": node.spec.spawn_id,
                    "x_m": node.spec.anchor_pose.x_m,
                    "y_m": node.spec.anchor_pose.y_m,
                    "yaw_rad": node.spec.anchor_pose.yaw_rad,
                },
                "map": _map_metadata(fleet.world.debug_grid),
                "controller": node.controller.snapshot(),
            },
        )
        event_cursor = await _send_pending_events(
            websocket,
            node.controller,
            0,
            fleet.timestamp_at(),
        )
        await _send_map_chunks(websocket, fleet.map_chunks)
        pose, scan = fleet.telemetry_messages(vehicle_id)
        await _send_json(websocket, pose)
        await _send_json(websocket, scan)
        sent_frame_sequence = int(scan["seq"])
        last_command_sequence: int | None = None

        while True:
            try:
                pending_frames = node.frames_after(sent_frame_sequence)
            except TelemetryOverflowError as error:
                await _send_json(
                    websocket,
                    {
                        "type": "error",
                        "timestamp_s": fleet.timestamp_at(),
                        "seq": None,
                        "code": "telemetry_overflow",
                        "message": str(error),
                        "oldest_available_seq": error.oldest,
                        "latest_available_seq": error.latest,
                    },
                )
                return
            for sampled in pending_frames:
                pose, scan = fleet.telemetry_messages(vehicle_id, sampled)
                await _send_json(websocket, pose)
                await _send_json(websocket, scan)
                event_cursor = await _send_pending_events(
                    websocket,
                    node.controller,
                    event_cursor,
                    pose["timestamp_s"],
                )
                sent_frame_sequence = sampled.sequence

            try:
                raw = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=min(0.05, fleet.tick_s) / fleet.realtime_factor,
                )
            except asyncio.TimeoutError:
                continue

            timestamp = fleet.timestamp_at()
            try:
                vehicle = fleet.world.vehicle(vehicle_id)
                command = parse_command(
                    raw,
                    linear_limit_mps=vehicle.linear_speed,
                    angular_limit_rps=vehicle.angular_speed,
                    mission_batch_limit=node.controller.mission_capacity,
                )
                if (
                    last_command_sequence is not None
                    and command.seq <= last_command_sequence
                ):
                    raise ProtocolError(
                        "stale_seq",
                        "seq must increase within one controller session",
                        command.seq,
                    )
                last_command_sequence = command.seq
                result = fleet.handle_command(vehicle_id, command)
                await _send_json(
                    websocket,
                    command_ack(
                        command,
                        result,
                        timestamp=timestamp,
                        controller=node.controller.snapshot(),
                    ),
                )
                event_cursor = await _send_pending_events(
                    websocket,
                    node.controller,
                    event_cursor,
                    timestamp,
                )
            except ProtocolError as error:
                fleet.fail_safe_stop(vehicle_id)
                await _send_json(websocket, error_message(error, timestamp=timestamp))
                event_cursor = await _send_pending_events(
                    websocket,
                    node.controller,
                    event_cursor,
                    timestamp,
                )
    except Exception as error:
        print(f"[!] {vehicle_id} controller connection ended: {error}")
    finally:
        fleet.disconnect(vehicle_id)
        node.controller_lease.release()
        print(f"[-] {vehicle_id} controller disconnected: {address}")


def _consume_background_result(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except BaseException:
        pass


async def _wait_cleanup_tasks(
    tasks: tuple[asyncio.Task[object], ...],
    *,
    context: str,
    errors: list[BaseException],
    timeout_s: float | None = None,
) -> None:
    if not tasks:
        return
    timeout_s = FLEET_CLEANUP_TIMEOUT_S if timeout_s is None else timeout_s
    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    timed_out = bool(pending)
    for task in pending:
        task.cancel()
    if pending:
        cancelled, still_pending = await asyncio.wait(
            pending,
            timeout=FLEET_CLEANUP_TIMEOUT_S,
        )
        done.update(cancelled)
        pending = still_pending
    if timed_out:
        errors.append(TimeoutError(f"{context} did not stop"))
    for task in pending:
        task.add_done_callback(_consume_background_result)
    if pending:
        errors.append(
            _CleanupPending(
                f"{context} remains alive",
                (TimeoutError("task ignored cancellation"),),
            )
        )
    for task in done:
        if task.cancelled():
            continue
        try:
            task.result()
        except BaseException as error:
            errors.append(error)


async def _cleanup_fleet_main(
    *,
    fleet: FleetRuntime,
    stop: asyncio.Event,
    tick_task: asyncio.Task[None] | None,
    stop_task: asyncio.Task[bool] | None,
    servers: tuple[object, ...],
    p2p_sync: P2PFleetSync | None,
    loop: asyncio.AbstractEventLoop,
    registered_signals: tuple[signal.Signals, ...],
) -> None:
    errors: list[BaseException] = []
    stop.set()

    background = []
    if tick_task is not None:
        background.append(tick_task)
    if stop_task is not None:
        stop_task.cancel()
        background.append(stop_task)
    await _wait_cleanup_tasks(
        tuple(background),
        context="fleet background task",
        errors=errors,
    )

    for server in servers:
        try:
            server.close()
        except BaseException as error:
            errors.append(error)
    server_waiters = []
    for server in servers:
        try:
            server_waiters.append(asyncio.create_task(server.wait_closed()))
        except BaseException as error:
            errors.append(error)
    await _wait_cleanup_tasks(
        tuple(server_waiters),
        context="WebSocket server wait",
        errors=errors,
    )

    if p2p_sync is not None:
        try:
            p2p_task = asyncio.create_task(p2p_sync.close())
        except BaseException as error:
            errors.append(error)
        else:
            await _wait_cleanup_tasks(
                (p2p_task,),
                context="libp2p fleet sync close",
                errors=errors,
                timeout_s=fleet_sync_cleanup_timeout_s()
                + FLEET_CLEANUP_TIMEOUT_S,
            )

    for vehicle_id in fleet.nodes:
        try:
            fleet.disconnect(vehicle_id)
        except BaseException as error:
            errors.append(error)
    for caught_signal in registered_signals:
        try:
            loop.remove_signal_handler(caught_signal)
        except BaseException as error:
            errors.append(error)
    if errors:
        error_type = (
            _CleanupPending
            if any(isinstance(error, _CleanupPending) for error in errors)
            else _CleanupError
        )
        raise error_type("cannot close fleet runtime", errors)


async def main(
    scenario_path: str | Path,
    *,
    linear_speed: float = 0.5,
    angular_speed: float = math.pi / 2,
    linear_acceleration_mps2: float = DEFAULT_LINEAR_ACCELERATION_MPS2,
    linear_deceleration_mps2: float = DEFAULT_LINEAR_DECELERATION_MPS2,
    angular_acceleration_rps2: float = DEFAULT_ANGULAR_ACCELERATION_RPS2,
    radius: float = 0.5,
    command_timeout: float = 1.0,
    mission_capacity: int = 16,
    odometry_translation_noise_stddev_m: float = 0.0,
    odometry_yaw_noise_stddev_rad: float = 0.0,
    odometry_seed: int = 0,
    realtime_factor: float = DEFAULT_REALTIME_FACTOR,
) -> None:
    from websockets.asyncio.server import serve

    scenario = FleetScenario.load(scenario_path)
    fleet = FleetRuntime.create(
        scenario,
        timestamp=time.time(),
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        linear_acceleration_mps2=linear_acceleration_mps2,
        linear_deceleration_mps2=linear_deceleration_mps2,
        angular_acceleration_rps2=angular_acceleration_rps2,
        radius=radius,
        command_timeout=command_timeout,
        mission_capacity=mission_capacity,
        realtime_factor=realtime_factor,
        odometry_config=OdometryConfig(
            odometry_translation_noise_stddev_m,
            odometry_yaw_noise_stddev_rad,
            odometry_seed,
        ),
    )
    p2p_sync: P2PFleetSync | None = None
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
    registered_signals = []
    servers = []
    tick_task: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[bool] | None = None
    original_error: BaseException | None = None
    original_traceback = None
    try:
        for caught_signal in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(caught_signal, signal_handler)
            registered_signals.append(caught_signal)
        if scenario.p2p is not None:
            p2p_sync = await P2PFleetSync.start(
                scenario.scenario_id,
                scenario.p2p,
                tuple(
                    P2PVehicleConfig(
                        spec.vehicle_id,
                        spec.p2p_port,
                        fleet.nodes[spec.vehicle_id].local_state.anchor,
                    )
                    for spec in scenario.vehicles
                ),
                {
                    vehicle_id: node.map_sync
                    for vehicle_id, node in fleet.nodes.items()
                    if node.map_sync is not None
                },
                realtime_factor=realtime_factor,
            )
            print(
                f"libp2p map sync ready for {len(scenario.vehicles)} vehicle(s) "
                f"in session {scenario.scenario_id}"
            )
        for spec in scenario.vehicles:
            async def configured_handler(
                websocket,
                vehicle_id: str = spec.vehicle_id,
            ) -> None:
                await fleet_handler(websocket, fleet=fleet, vehicle_id=vehicle_id)

            servers.append(await serve(configured_handler, HOST, spec.operator_port))
            print(
                f"{spec.vehicle_id} listening on ws://{HOST}:{spec.operator_port} "
                f"at {spec.spawn_id}"
            )
        tick_task = asyncio.create_task(fleet.run(stop))
        stop_task = asyncio.create_task(stop.wait())
        completed, _ = await asyncio.wait(
            (tick_task, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if tick_task in completed:
            await tick_task
    except BaseException as error:
        original_error = error
        original_traceback = error.__traceback__

    cleanup_task = asyncio.create_task(
        _cleanup_fleet_main(
            fleet=fleet,
            stop=stop,
            tick_task=tick_task,
            stop_task=stop_task,
            servers=tuple(servers),
            p2p_sync=p2p_sync,
            loop=loop,
            registered_signals=tuple(registered_signals),
        )
    )
    cancellation, cleanup_error = await _finish_cleanup(
        cleanup_task,
        timeout_s=(
            fleet_sync_cleanup_timeout_s()
            + FLEET_CLEANUP_TIMEOUT_S * 6
        ),
    )
    if original_error is not None:
        if cleanup_error is not None:
            raise original_error.with_traceback(original_traceback) from cleanup_error
        raise original_error.with_traceback(original_traceback)
    if cancellation is not None:
        if cleanup_error is not None:
            raise cancellation from cleanup_error
        raise cancellation
    if cleanup_error is not None:
        raise cleanup_error
