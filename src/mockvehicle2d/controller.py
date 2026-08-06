"""Single authority for manual motion and autonomous missions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import TYPE_CHECKING, ClassVar
import uuid

from mockvehicle2d.navigation import GotoController
from mockvehicle2d.map_sync import (
    MAX_INTENT_WAIT_TICKS,
    MOTION_INTENT_TTL_S,
    PeerMotionIntent,
)
from mockvehicle2d.safety import AUTOMATIC_MINIMUM_CLEARANCE_M

if TYPE_CHECKING:
    from mockvehicle2d.local_state import (
        AnchorSpec,
        LocalMapDelta,
        ObservedGrid,
        PoseEstimate,
    )
    from mockvehicle2d.map_grid import MapGrid
    from mockvehicle2d.map_sync import PeerVehicleState
    from mockvehicle2d.scan import LaserPoint
    from mockvehicle2d.safety import LocalSafetyRuntime, SafetyAdvanceResult
    from mockvehicle2d.vehicle import Vehicle


MISSION_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}")
MISSION_FRAME = "global_map"
MAX_MISSION_SUBGOALS = 1024
SUPPORTED_MISSION_TYPES = ("goto", "patrol", "coverage")
PEER_CONFLICT_HORIZON_S = 4.0
PEER_YIELD_CLEAR_TICKS = 3
PEER_ESCAPE_LINEAR_MPS = 0.2
PEER_RESERVATION_MAX_HOLD_TICKS = 40
Goal = tuple[float, float]


class OpMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class AutoState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"


class ModeAction(str, Enum):
    SWITCH_TO_MANUAL = "switch_to_manual"
    SWITCH_TO_AUTO = "switch_to_auto"
    STOP_MOTION = "stop_motion"


class ManualAction(str, Enum):
    DRIVE = "drive"
    STOP = "stop"


class AutoAction(str, Enum):
    PUSH = "push"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL_ALL = "cancel_all"


@dataclass(frozen=True)
class GotoMission:
    mission_type: ClassVar[str] = "goto"
    mission_id: str
    frame_id: str
    x_m: float
    y_m: float
    submitted_seq: int

    def __post_init__(self) -> None:
        _validate_mission_header(self.mission_id, self.frame_id, self.submitted_seq)
        _validate_goal((self.x_m, self.y_m))

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return self.mission_type, self.frame_id, self.x_m, self.y_m

    @property
    def subgoals(self) -> tuple[Goal, ...]:
        return ((self.x_m, self.y_m),)

    def as_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "type": self.mission_type,
            "frame_id": self.frame_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "submitted_seq": self.submitted_seq,
        }


@dataclass(frozen=True)
class PatrolMission:
    mission_type: ClassVar[str] = "patrol"
    mission_id: str
    frame_id: str
    waypoints: tuple[Goal, ...]
    cycles: int
    submitted_seq: int
    _subgoals: tuple[Goal, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_mission_header(self.mission_id, self.frame_id, self.submitted_seq)
        if not self.waypoints:
            raise ValueError("waypoints must not be empty")
        for waypoint in self.waypoints:
            _validate_goal(waypoint)
        if (
            isinstance(self.cycles, bool)
            or not isinstance(self.cycles, int)
            or self.cycles <= 0
        ):
            raise ValueError("cycles must be a positive integer")
        if self.cycles > MAX_MISSION_SUBGOALS // len(self.waypoints):
            raise ValueError(
                f"mission must generate at most {MAX_MISSION_SUBGOALS} subgoals"
            )
        object.__setattr__(self, "_subgoals", self.waypoints * self.cycles)

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return self.mission_type, self.frame_id, self.waypoints, self.cycles

    @property
    def subgoals(self) -> tuple[Goal, ...]:
        return self._subgoals

    def as_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "type": self.mission_type,
            "frame_id": self.frame_id,
            "waypoints": [
                {"x_m": x_m, "y_m": y_m} for x_m, y_m in self.waypoints
            ],
            "cycles": self.cycles,
            "submitted_seq": self.submitted_seq,
        }


@dataclass(frozen=True)
class CoverageMission:
    mission_type: ClassVar[str] = "coverage"
    mission_id: str
    frame_id: str
    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float
    lane_spacing_m: float
    submitted_seq: int
    _subgoals: tuple[Goal, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_mission_header(self.mission_id, self.frame_id, self.submitted_seq)
        for goal in (
            (self.min_x_m, self.min_y_m),
            (self.max_x_m, self.max_y_m),
        ):
            _validate_goal(goal)
        if self.min_x_m >= self.max_x_m or self.min_y_m >= self.max_y_m:
            raise ValueError("coverage area minimums must be below maximums")
        if (
            isinstance(self.lane_spacing_m, bool)
            or not isinstance(self.lane_spacing_m, (int, float))
            or not math.isfinite(self.lane_spacing_m)
            or self.lane_spacing_m <= 0
        ):
            raise ValueError("lane_spacing_m must be finite and positive")
        object.__setattr__(self, "_subgoals", self._coverage_subgoals())

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.mission_type,
            self.frame_id,
            self.min_x_m,
            self.min_y_m,
            self.max_x_m,
            self.max_y_m,
            self.lane_spacing_m,
        )

    @property
    def subgoals(self) -> tuple[Goal, ...]:
        return self._subgoals

    def as_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "type": self.mission_type,
            "frame_id": self.frame_id,
            "area": {
                "min_x_m": self.min_x_m,
                "min_y_m": self.min_y_m,
                "max_x_m": self.max_x_m,
                "max_y_m": self.max_y_m,
            },
            "lane_spacing_m": self.lane_spacing_m,
            "submitted_seq": self.submitted_seq,
        }

    def _coverage_subgoals(self) -> tuple[Goal, ...]:
        width = self.max_x_m - self.min_x_m
        height = self.max_y_m - self.min_y_m
        along_x = width >= height
        short_span = height if along_x else width
        ratio = short_span / self.lane_spacing_m
        max_segments = MAX_MISSION_SUBGOALS // 2 - 1
        if not math.isfinite(ratio) or ratio > max_segments:
            raise ValueError(
                f"mission must generate at most {MAX_MISSION_SUBGOALS} subgoals"
            )
        segments = max(1, math.ceil(ratio))
        goals: list[Goal] = []
        for index in range(segments + 1):
            lane = (
                (self.max_y_m if along_x else self.max_x_m)
                if index == segments
                else (self.min_y_m if along_x else self.min_x_m)
                + index * self.lane_spacing_m
            )
            if along_x:
                endpoints = ((self.min_x_m, lane), (self.max_x_m, lane))
            else:
                endpoints = ((lane, self.min_y_m), (lane, self.max_y_m))
            goals.extend(endpoints if index % 2 == 0 else reversed(endpoints))
        return tuple(goals)


Mission = GotoMission | PatrolMission | CoverageMission


def _validate_mission_header(
    mission_id: object,
    frame_id: object,
    submitted_seq: object,
) -> None:
    if not isinstance(mission_id, str) or not MISSION_ID_PATTERN.fullmatch(mission_id):
        raise ValueError("invalid mission_id")
    if frame_id != MISSION_FRAME:
        raise ValueError(f"frame_id must be {MISSION_FRAME}")
    if (
        isinstance(submitted_seq, bool)
        or not isinstance(submitted_seq, int)
        or not 0 <= submitted_seq <= 2**64 - 1
    ):
        raise ValueError("submitted_seq must be an unsigned 64-bit integer")


def _validate_goal(goal: object) -> None:
    if not isinstance(goal, tuple) or len(goal) != 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in goal
    ):
        raise ValueError("mission coordinates must be finite")


def _peer_motion_conflicts(
    desired: tuple[float, float],
    vehicle: Vehicle,
    anchor: AnchorSpec,
    pose: PoseEstimate,
    peer: PeerVehicleState,
    travel_limit_m: float | None,
) -> bool:
    own_x_m, own_y_m, own_yaw_rad = anchor.anchor_to_global(
        pose.x_m,
        pose.y_m,
        pose.yaw_rad,
    )
    executed_linear_mps, _ = vehicle.body_velocities()
    projected_linear_mps = (
        desired[0]
        if abs(desired[0]) >= abs(executed_linear_mps)
        else executed_linear_mps
    )
    own_vx = projected_linear_mps * math.cos(own_yaw_rad)
    own_vy = projected_linear_mps * math.sin(own_yaw_rad)
    relative_x = peer.global_x_m - own_x_m
    relative_y = peer.global_y_m - own_y_m
    relative_vx = peer.vx_mps - own_vx
    relative_vy = peer.vy_mps - own_vy
    conflict_distance_m = _peer_conflict_distance(
        vehicle,
        peer,
        executed_linear_mps,
    )
    if (
        math.hypot(relative_x, relative_y) <= conflict_distance_m + 1e-12
        and relative_x * relative_vx + relative_y * relative_vy > 1e-12
    ):
        return False
    own_motion_s = PEER_CONFLICT_HORIZON_S
    if travel_limit_m is not None and abs(projected_linear_mps) > 1e-12:
        own_motion_s = min(
            own_motion_s,
            travel_limit_m / abs(projected_linear_mps),
        )
    closest_distance_m = _closest_approach_distance(
        relative_x,
        relative_y,
        relative_vx,
        relative_vy,
        own_motion_s,
    )
    if own_motion_s < PEER_CONFLICT_HORIZON_S:
        relative_x += relative_vx * own_motion_s
        relative_y += relative_vy * own_motion_s
        closest_distance_m = min(
            closest_distance_m,
            _closest_approach_distance(
                relative_x,
                relative_y,
                peer.vx_mps,
                peer.vy_mps,
                PEER_CONFLICT_HORIZON_S - own_motion_s,
            ),
        )
    return closest_distance_m <= conflict_distance_m + 1e-12


def motion_intent_precedes(
    first: PeerMotionIntent,
    second: PeerMotionIntent,
) -> bool:
    """Return the deterministic winner; a live reservation is a short lease."""
    if first.reserved != second.reserved:
        return first.reserved
    if first.wait_ticks != second.wait_ticks:
        return first.wait_ticks > second.wait_ticks
    if first.priority_owner_id != second.priority_owner_id:
        return first.priority_owner_id < second.priority_owner_id
    return first.source_vehicle_id < second.source_vehicle_id


def inherit_motion_priority(
    own: PeerMotionIntent,
    requesters: tuple[PeerMotionIntent, ...],
) -> PeerMotionIntent:
    inherited = own
    for requester in sorted(requesters, key=lambda intent: intent.source_vehicle_id):
        if motion_intent_precedes(requester, inherited):
            inherited = requester
    if inherited is own:
        return own
    return PeerMotionIntent(
        own.source_vehicle_id,
        own.intent_generation,
        own.sequence,
        own.timestamp_s,
        own.lease_duration_s,
        own.current_cell,
        own.target_cell,
        max(own.wait_ticks, inherited.wait_ticks),
        inherited.priority_owner_id,
        own.reserved,
    )


def _motion_intents_conflict(
    first: PeerMotionIntent,
    second: PeerMotionIntent,
) -> bool:
    if first.target_cell is None or second.target_cell is None:
        return False
    return first.target_cell == second.target_cell or (
        first.current_cell == second.target_cell
        and first.target_cell == second.current_cell
    )


def _coordination_cell(
    anchor: AnchorSpec,
    point_m: tuple[float, float],
    resolution_m: float,
) -> tuple[int, int]:
    global_x_m, global_y_m, _ = anchor.anchor_to_global(*point_m, 0.0)
    return (
        math.floor(global_x_m / resolution_m),
        math.floor(global_y_m / resolution_m),
    )


def _intent_setpoint(
    target_m: tuple[float, float],
    pose: PoseEstimate,
    vehicle: Vehicle,
) -> tuple[float, float]:
    dx, dy = target_m[0] - pose.x_m, target_m[1] - pose.y_m
    distance_m = math.hypot(dx, dy)
    heading_error = math.atan2(
        math.sin(math.atan2(dy, dx) - pose.yaw_rad),
        math.cos(math.atan2(dy, dx) - pose.yaw_rad),
    )
    return (
        0.0
        if abs(heading_error) > GotoController.turn_in_place_threshold_rad
        else min(vehicle.linear_speed, distance_m),
        max(-vehicle.angular_speed, min(vehicle.angular_speed, 2 * heading_error)),
    )


def _peer_conflict_distance(
    vehicle: Vehicle,
    peer: PeerVehicleState,
    executed_linear_mps: float,
) -> float:
    return (
        vehicle.radius
        + peer.radius_m
        + math.sqrt(max(peer.covariance[:2]))
        + AUTOMATIC_MINIMUM_CLEARANCE_M
        + executed_linear_mps**2
        / (2 * vehicle.linear_deceleration_mps2)
    )


def _peer_escape_setpoint(
    vehicle: Vehicle,
    anchor: AnchorSpec,
    pose: PoseEstimate,
    peer: PeerVehicleState,
) -> tuple[float, float] | None:
    own_x_m, own_y_m, own_yaw_rad = anchor.anchor_to_global(
        pose.x_m,
        pose.y_m,
        pose.yaw_rad,
    )
    current_distance_m = math.hypot(
        peer.global_x_m - own_x_m,
        peer.global_y_m - own_y_m,
    )
    conflict_distance_m = _peer_conflict_distance(
        vehicle,
        peer,
        vehicle.body_velocities()[0],
    )
    if current_distance_m > conflict_distance_m + 1e-12:
        return None
    away_yaw_rad = math.atan2(
        own_y_m - peer.global_y_m,
        own_x_m - peer.global_x_m,
    )
    heading_error = math.atan2(
        math.sin(away_yaw_rad - own_yaw_rad),
        math.cos(away_yaw_rad - own_yaw_rad),
    )
    angular_rps = max(
        -vehicle.angular_speed,
        min(vehicle.angular_speed, 2 * heading_error),
    )
    return (
        0.0
        if abs(heading_error) > GotoController.turn_in_place_threshold_rad
        else min(vehicle.linear_speed, PEER_ESCAPE_LINEAR_MPS),
        angular_rps,
    )


def _closest_approach_distance(
    relative_x: float,
    relative_y: float,
    relative_vx: float,
    relative_vy: float,
    duration_s: float,
) -> float:
    relative_speed_squared = relative_vx**2 + relative_vy**2
    closest_time_s = (
        0.0
        if relative_speed_squared <= 1e-12
        else max(
            0.0,
            min(
                duration_s,
                -(relative_x * relative_vx + relative_y * relative_vy)
                / relative_speed_squared,
            ),
        )
    )
    return math.hypot(
        relative_x + relative_vx * closest_time_s,
        relative_y + relative_vy * closest_time_s,
    )


@dataclass(frozen=True)
class ModeCommand:
    seq: int
    action: ModeAction


@dataclass(frozen=True)
class ManualCommand:
    seq: int
    action: ManualAction
    linear_mps: float = 0.0
    angular_rps: float = 0.0


@dataclass(frozen=True)
class AutoCommand:
    seq: int
    action: AutoAction
    missions: tuple[Mission, ...] = ()


Command = ModeCommand | ManualCommand | AutoCommand


@dataclass(frozen=True)
class ControllerEvent:
    event_seq: int
    event_epoch: str
    mission: Mission
    subgoal_index: int
    status: str
    reason: str | None = None
    detail: str | None = None
    navigation: dict[str, object] | None = None

    def as_dict(self, timestamp: float) -> dict[str, object]:
        goal_x_m, goal_y_m = self.mission.subgoals[self.subgoal_index]
        message: dict[str, object] = {
            "type": "mission_update",
            "event_seq": self.event_seq,
            "event_epoch": self.event_epoch,
            "timestamp_s": timestamp,
            "mission_id": self.mission.mission_id,
            "mission_type": self.mission.mission_type,
            "submitted_seq": self.mission.submitted_seq,
            "status": self.status,
            "subgoal_index": self.subgoal_index,
            "subgoal_count": len(self.mission.subgoals),
            "goal": {
                "frame_id": self.mission.frame_id,
                "x_m": goal_x_m,
                "y_m": goal_y_m,
            },
        }
        if self.reason is not None:
            message["reason"] = self.reason
        if self.detail is not None:
            message["detail"] = self.detail
        if self.navigation is not None:
            message["navigation"] = self.navigation
        return message


@dataclass(frozen=True)
class CommandResult:
    accepted: bool
    reason: str | None = None


class RobotController:
    """Own mode, mission lifecycle, navigation and the only actuator output."""

    def __init__(
        self,
        navigation: GotoController | None = None,
        *,
        mission_capacity: int = 16,
    ) -> None:
        if isinstance(mission_capacity, bool) or not isinstance(mission_capacity, int):
            raise ValueError("mission_capacity must be an integer")
        if mission_capacity <= 0:
            raise ValueError("mission_capacity must be positive")
        self.mode = OpMode.MANUAL
        self.auto_state = AutoState.IDLE
        self.navigation = navigation or GotoController()
        self.mission_capacity = mission_capacity
        self.active_mission: Mission | None = None
        self._pending: deque[Mission] = deque()
        # ponytail: process-lifetime ledgers fit the simulator; add persistence and
        # explicit retention only when long-running deployment volume requires it.
        self._mission_history: dict[str, tuple[object, ...]] = {}
        self._events: list[ControllerEvent] = []
        self.event_epoch = uuid.uuid4().hex
        self._manual_setpoint: tuple[float, float] | None = None
        self._manual_deadline: float | None = None
        self._needs_start = False
        self._subgoal_index = 0
        self._deferred_edge_cell: tuple[int, int] | None = None
        self._yielding_for: str | None = None
        self._yield_requires_intent = False
        self._yield_clear_ticks = 0
        self._reservation_wait_ticks = 0
        self._intent_priority_owner_id: str | None = None
        self._intent_target_m: tuple[float, float] | None = None
        self._intent_reserved = False
        self._reservation_cells: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._reservation_hold_ticks = 0
        self._last_coordination_cell: tuple[int, int] | None = None

    @property
    def is_automatic_motion_active(self) -> bool:
        return self.mode is OpMode.AUTO and self.auto_state is AutoState.ACTIVE

    @property
    def is_yielding(self) -> bool:
        return self._yielding_for is not None

    @property
    def motion_intent(self) -> tuple[
        tuple[float, float] | None,
        int,
        str | None,
        bool,
    ]:
        return (
            self._intent_target_m,
            self._reservation_wait_ticks,
            self._intent_priority_owner_id,
            self._intent_reserved,
        )

    def handle(
        self,
        command: Command,
        *,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        now: float,
    ) -> CommandResult:
        if isinstance(command, ModeCommand):
            return self._handle_mode(command, vehicle, now)
        if isinstance(command, ManualCommand):
            return self._handle_manual(command, vehicle, grid, safety, now)
        if isinstance(command, AutoCommand):
            return self._handle_auto(command, vehicle, now)
        raise TypeError("unsupported controller command")

    def tick(
        self,
        *,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        anchor: AnchorSpec,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        map_delta: LocalMapDelta | None,
        advance_result: SafetyAdvanceResult,
        now: float,
        safety_scan_points: tuple[LaserPoint, ...] | None = None,
        safety_scan_healthy: bool = True,
        vehicle_id: str | None = None,
        peer_states: tuple[PeerVehicleState, ...] = (),
        peer_motion_intents: tuple[PeerMotionIntent, ...] = (),
    ) -> None:
        if self.mode is OpMode.MANUAL:
            self._clear_yield()
            self._tick_manual(
                vehicle,
                grid,
                safety,
                now,
                safety_scan_points,
                safety_scan_healthy,
            )
            return
        self._manual_setpoint = None
        self._manual_deadline = None
        if self.auto_state is not AutoState.ACTIVE:
            self._clear_yield()
            vehicle.stop()
            return

        if self._needs_start:
            self._start_or_resume(anchor, pose, local_map, vehicle.radius)
        if self.auto_state is not AutoState.ACTIVE or self.active_mission is None:
            vehicle.stop()
            return

        desired = self.navigation.update(
            pose=pose,
            local_map=local_map,
            max_linear_mps=vehicle.linear_speed,
            max_angular_rps=vehicle.angular_speed,
            advance_result=advance_result,
            map_delta=map_delta,
            safety=safety,
        )
        if self.navigation.status == "reached":
            self._complete_subgoal(vehicle, anchor, pose, local_map)
            return
        if self.navigation.status == "blocked":
            self._finish_blocked(vehicle)
            return

        desired = self._coordinate_desired(
            desired,
            vehicle=vehicle,
            vehicle_id=vehicle_id,
            anchor=anchor,
            pose=pose,
            local_map=local_map,
            now=now,
            peer_states=peer_states,
            peer_motion_intents=peer_motion_intents,
        )

        decision = safety.evaluate(
            vehicle,
            grid,
            desired[0],
            desired[1],
            automatic=True,
            scan_points=safety_scan_points,
            scan_healthy=safety_scan_healthy,
        )
        if decision.state == "fault":
            vehicle.stop()
            self.navigation.block(decision.reason or "safety_sensor_fault")
            self._finish_blocked(vehicle)
            return
        if decision.state == "stopped":
            vehicle.stop(now)
            if self.navigation.finish_nearby_safe_stop(pose, decision.reason):
                self._complete_subgoal(vehicle, anchor, pose, local_map)
                return
            edge_cell = (
                self.navigation.unmapped_edge_evidence_cell(safety, pose, local_map)
                if decision.reason == "safety_edge"
                else None
            )
            if edge_cell is not None and edge_cell != self._deferred_edge_cell:
                self._deferred_edge_cell = edge_cell
                return
            self.navigation.block(decision.reason or "safety_obstacle")
            self._finish_blocked(vehicle)
            return
        vehicle.install_drive(decision.linear_mps, decision.angular_rps, now)

    def _coordinate_desired(
        self,
        desired: tuple[float, float],
        *,
        vehicle: Vehicle,
        vehicle_id: str | None,
        anchor: AnchorSpec,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        now: float,
        peer_states: tuple[PeerVehicleState, ...],
        peer_motion_intents: tuple[PeerMotionIntent, ...],
    ) -> tuple[float, float]:
        if vehicle_id is None:
            self._clear_yield()
            return desired

        peers = {state.source_vehicle_id: state for state in peer_states}
        intents = {
            intent.source_vehicle_id: intent for intent in peer_motion_intents
        }
        motion_target = self.navigation.motion_target
        self._intent_target_m = motion_target
        self._intent_priority_owner_id = vehicle_id
        current_cell = _coordination_cell(
            anchor,
            (pose.x_m, pose.y_m),
            local_map.resolution_m,
        )
        if self._last_coordination_cell != current_cell:
            self._last_coordination_cell = current_cell
            self._reservation_wait_ticks = 0
            self._reservation_hold_ticks = 0
            self._reservation_cells = None
        target_cell = (
            None
            if motion_target is None
            else _coordination_cell(anchor, motion_target, local_map.resolution_m)
        )
        own = PeerMotionIntent(
            vehicle_id,
            1,
            1,
            now,
            MOTION_INTENT_TTL_S,
            current_cell,
            target_cell,
            self._reservation_wait_ticks,
            vehicle_id,
            self._intent_reserved
            and self._reservation_cells == (current_cell, target_cell)
            and self._reservation_hold_ticks < PEER_RESERVATION_MAX_HOLD_TICKS,
        )

        requesters = [
            intent
            for intent in peer_motion_intents
            if intent.target_cell == current_cell
            and intent.current_cell != current_cell
        ]
        inherited_from = None
        inherited_own = inherit_motion_priority(own, tuple(requesters))
        if inherited_own.priority_owner_id != own.priority_owner_id:
            inherited_from = next(
                requester
                for requester in requesters
                if requester.priority_owner_id == inherited_own.priority_owner_id
            )
            own = inherited_own
            self._intent_priority_owner_id = own.priority_owner_id

        swap_request = next(
            (
                requester
                for requester in requesters
                if target_cell is None or requester.current_cell == target_cell
            ),
            None,
        )
        if swap_request is not None:
            unavailable = {
                cell
                for intent in peer_motion_intents
                for cell in (intent.current_cell, intent.target_cell)
                if cell is not None
            }
            for detour_m in self.navigation.coordination_detours(pose, local_map):
                detour_cell = _coordination_cell(
                    anchor,
                    detour_m,
                    local_map.resolution_m,
                )
                if detour_cell in unavailable:
                    continue
                motion_target = detour_m
                target_cell = detour_cell
                desired = _intent_setpoint(detour_m, pose, vehicle)
                self._intent_target_m = detour_m
                own = PeerMotionIntent(
                    vehicle_id,
                    1,
                    1,
                    now,
                    MOTION_INTENT_TTL_S,
                    current_cell,
                    target_cell,
                    own.wait_ticks,
                    own.priority_owner_id,
                    own.reserved,
                )
                break

        vacating_for = (
            None
            if inherited_from is None
            or target_cell in {None, current_cell, inherited_from.current_cell}
            else inherited_from.source_vehicle_id
        )

        travel_limit_m = (
            None
            if motion_target is None
            else math.dist((pose.x_m, pose.y_m), motion_target)
        )
        if self._yielding_for is not None:
            peer = peers.get(self._yielding_for)
            peer_intent = intents.get(self._yielding_for)
            if peer is None or (
                self._yield_requires_intent and peer_intent is None
            ):
                self._reservation_wait_ticks = min(
                    MAX_INTENT_WAIT_TICKS,
                    self._reservation_wait_ticks + 1,
                )
                self._intent_reserved = False
                return 0.0, 0.0
            still_conflicts = _peer_motion_conflicts(
                desired, vehicle, anchor, pose, peer, travel_limit_m
            ) or (
                peer_intent is not None
                and (
                    own.target_cell == peer_intent.current_cell
                    or _motion_intents_conflict(own, peer_intent)
                )
            )
            if still_conflicts and (
                peer_intent is None
                or own.target_cell == peer_intent.current_cell
                or motion_intent_precedes(peer_intent, own)
            ):
                self._yield_clear_ticks = 0
                self._reservation_wait_ticks = min(
                    MAX_INTENT_WAIT_TICKS,
                    self._reservation_wait_ticks + 1,
                )
                self._intent_reserved = False
                if desired[0] == 0.0 and desired[1] != 0.0:
                    return desired
                if desired == (0.0, 0.0):
                    escape = _peer_escape_setpoint(vehicle, anchor, pose, peer)
                    if escape is not None:
                        return escape
                return 0.0, 0.0
            self._yield_clear_ticks += 1
            if self._yield_clear_ticks < PEER_YIELD_CLEAR_TICKS:
                return 0.0, 0.0
            self._yielding_for = None
            self._yield_requires_intent = False
            self._yield_clear_ticks = 0

        intent_conflicts = []
        for peer_intent in peer_motion_intents:
            peer = peers.get(peer_intent.source_vehicle_id)
            if peer is None:
                continue
            if peer_intent.source_vehicle_id == vacating_for:
                continue
            target_occupied = own.target_cell == peer_intent.current_cell
            cell_conflict = _motion_intents_conflict(own, peer_intent)
            trajectory_conflict = _peer_motion_conflicts(
                desired,
                vehicle,
                anchor,
                pose,
                peer,
                travel_limit_m,
            )
            if target_occupied or (
                (cell_conflict or trajectory_conflict)
                and motion_intent_precedes(peer_intent, own)
            ):
                intent_conflicts.append(peer_intent.source_vehicle_id)
        fallback_conflicts = [
            state.source_vehicle_id
            for state in peer_states
            if state.source_vehicle_id not in intents
            and vehicle_id > state.source_vehicle_id
            and _peer_motion_conflicts(
                desired,
                vehicle,
                anchor,
                pose,
                state,
                travel_limit_m,
            )
        ]
        conflicts = sorted({*intent_conflicts, *fallback_conflicts})
        if conflicts:
            self._yielding_for = conflicts[0]
            self._yield_requires_intent = conflicts[0] in intents
            self._yield_clear_ticks = 0
            self._reservation_wait_ticks = min(
                MAX_INTENT_WAIT_TICKS,
                self._reservation_wait_ticks + 1,
            )
            self._intent_reserved = False
            return 0.0, 0.0

        cells = current_cell, target_cell
        if target_cell is not None and target_cell != current_cell:
            if self._reservation_cells == cells:
                self._reservation_hold_ticks += 1
            else:
                self._reservation_cells = cells
                self._reservation_hold_ticks = 0
            self._intent_reserved = (
                self._reservation_hold_ticks < PEER_RESERVATION_MAX_HOLD_TICKS
            )
        else:
            self._intent_reserved = False
            self._reservation_cells = None
            self._reservation_hold_ticks = 0
        return desired

    def _clear_yield(self) -> None:
        self._yielding_for = None
        self._yield_requires_intent = False
        self._yield_clear_ticks = 0
        self._reservation_wait_ticks = 0
        self._intent_priority_owner_id = None
        self._intent_target_m = None
        self._intent_reserved = False
        self._reservation_cells = None
        self._reservation_hold_ticks = 0
        self._last_coordination_cell = None

    def disconnect(self, vehicle: Vehicle) -> None:
        vehicle.stop()
        self._manual_setpoint = None
        self._manual_deadline = None
        if self.mode is OpMode.AUTO and (
            self.active_mission is not None or self._pending
        ):
            self._pause_active("controller_disconnected")

    def fail_safe_stop(self, vehicle: Vehicle, reason: str) -> None:
        vehicle.stop()
        self._manual_setpoint = None
        self._manual_deadline = None
        if self.mode is OpMode.AUTO and (
            self.active_mission is not None or self._pending
        ):
            self._pause_active(reason)

    @property
    def latest_event_seq(self) -> int:
        return len(self._events)

    def events_after(self, event_seq: int) -> tuple[ControllerEvent, ...]:
        if (
            isinstance(event_seq, bool)
            or not isinstance(event_seq, int)
            or event_seq < 0
        ):
            raise ValueError("event_seq must be a non-negative integer")
        return tuple(self._events[event_seq:])

    def snapshot(self, *, now: float | None = None) -> dict[str, object]:
        manual_setpoint_active = self._manual_setpoint is not None and (
            now is None
            or self._manual_deadline is None
            or now < self._manual_deadline
        )
        return {
            "mode": self.mode.value,
            "auto_state": self.auto_state.value,
            "active_mission": (
                None
                if self.active_mission is None
                else self._active_mission_snapshot()
            ),
            "mission_queue": {
                "size": len(self._pending),
                "capacity": self.mission_capacity,
                "mission_ids": [mission.mission_id for mission in self._pending],
            },
            "manual_setpoint_active": manual_setpoint_active,
            "navigation": self.navigation.snapshot(),
            "mission_events": {
                "event_epoch": self.event_epoch,
                "latest_event_seq": self.latest_event_seq,
                "retention": "process_lifetime",
            },
        }

    def _handle_mode(
        self,
        command: ModeCommand,
        vehicle: Vehicle,
        now: float,
    ) -> CommandResult:
        if command.action is ModeAction.STOP_MOTION:
            vehicle.stop(now)
            self._manual_setpoint = None
            self._manual_deadline = None
            if self.mode is OpMode.AUTO:
                if self.active_mission is not None or self._pending:
                    self._pause_active("stop_motion")
                else:
                    self.auto_state = AutoState.IDLE
            return CommandResult(True)

        if command.action is ModeAction.SWITCH_TO_MANUAL:
            if self.mode is OpMode.MANUAL:
                return CommandResult(True)
            vehicle.stop(now)
            self._manual_setpoint = None
            self._manual_deadline = None
            if self.active_mission is not None or self._pending:
                self._pause_active("manual_takeover")
            else:
                self.auto_state = AutoState.IDLE
            self.mode = OpMode.MANUAL
            return CommandResult(True)

        if self.mode is OpMode.AUTO:
            return CommandResult(True)
        vehicle.stop(now)
        self._manual_setpoint = None
        self._manual_deadline = None
        self.mode = OpMode.AUTO
        if self.active_mission is not None or self._pending:
            self.auto_state = AutoState.PAUSED
            self._needs_start = False
        else:
            self.auto_state = AutoState.IDLE
        return CommandResult(True)

    def _handle_manual(
        self,
        command: ManualCommand,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        now: float,
    ) -> CommandResult:
        if self.mode is not OpMode.MANUAL:
            return CommandResult(False, "wrong_mode")
        if command.action is ManualAction.STOP:
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop(now)
            return CommandResult(True)

        desired = command.linear_mps, command.angular_rps
        decision = safety.evaluate(vehicle, grid, *desired, automatic=False)
        if decision.state == "fault" or (
            decision.state == "stopped" and decision.angular_rps == 0.0
        ):
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop(now)
            return CommandResult(False, decision.reason or "safety_rejected")
        self._manual_setpoint = desired
        self._manual_deadline = now + vehicle.command_timeout
        vehicle.install_drive(decision.linear_mps, decision.angular_rps, now)
        return CommandResult(True)

    def _handle_auto(
        self,
        command: AutoCommand,
        vehicle: Vehicle,
        now: float,
    ) -> CommandResult:
        if self.mode is not OpMode.AUTO:
            return CommandResult(False, "wrong_mode")
        if command.action is AutoAction.PUSH:
            return self._push(command.missions)
        if command.action is AutoAction.PAUSE:
            vehicle.stop(now)
            if self.active_mission is not None or self._pending:
                self._pause_active("paused")
            else:
                self.auto_state = AutoState.IDLE
                self._needs_start = False
            return CommandResult(True)
        if command.action is AutoAction.RESUME:
            if self.auto_state is AutoState.ACTIVE:
                return CommandResult(True)
            vehicle.stop(now)
            if self.active_mission is None and not self._pending:
                self.auto_state = AutoState.IDLE
                self._needs_start = False
                return CommandResult(True)
            self.auto_state = AutoState.ACTIVE
            self._needs_start = True
            return CommandResult(True)

        vehicle.stop(now)
        self._cancel_all("cancelled")
        return CommandResult(True)

    def _push(self, missions: tuple[Mission, ...]) -> CommandResult:
        if not missions:
            return CommandResult(False, "empty_mission_batch")
        ids = [mission.mission_id for mission in missions]
        if len(ids) != len(set(ids)):
            return CommandResult(False, "duplicate_mission_id")
        new_missions = []
        for mission in missions:
            known = self._mission_history.get(mission.mission_id)
            if known is None:
                new_missions.append(mission)
                continue
            if known != mission.fingerprint:
                return CommandResult(False, "mission_id_conflict")
        if len(self._pending) + len(new_missions) > self.mission_capacity:
            return CommandResult(False, "mission_queue_full")

        for mission in new_missions:
            self._pending.append(mission)
            self._mission_history[mission.mission_id] = mission.fingerprint
            self._emit(mission, "queued")
        if new_missions and self.auto_state is AutoState.IDLE:
            self.auto_state = AutoState.ACTIVE
            self._needs_start = True
        return CommandResult(True)

    def _tick_manual(
        self,
        vehicle: Vehicle,
        grid: MapGrid,
        safety: LocalSafetyRuntime,
        now: float,
        scan_points: tuple[LaserPoint, ...] | None,
        scan_healthy: bool,
    ) -> None:
        if (
            self._manual_setpoint is None
            or self._manual_deadline is None
            or now >= self._manual_deadline
        ):
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop()
            return
        decision = safety.evaluate(
            vehicle,
            grid,
            *self._manual_setpoint,
            automatic=False,
            scan_points=scan_points,
            scan_healthy=scan_healthy,
        )
        if decision.state == "fault" or (
            decision.state == "stopped" and decision.angular_rps == 0.0
        ):
            self._manual_setpoint = None
            self._manual_deadline = None
            vehicle.stop()
            return
        vehicle.install_drive(decision.linear_mps, decision.angular_rps, now)

    def _start_or_resume(
        self,
        anchor: AnchorSpec,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        vehicle_radius_m: float,
        *,
        emit_event: bool = True,
    ) -> None:
        self._needs_start = False
        self._deferred_edge_cell = None
        self._clear_yield()
        if self.active_mission is None:
            if not self._pending:
                self.auto_state = AutoState.IDLE
                return
            self.active_mission = self._pending.popleft()
            self._subgoal_index = 0
        mission = self.active_mission
        goal_x_m, goal_y_m = mission.subgoals[self._subgoal_index]
        local_x_m, local_y_m, _ = anchor.global_to_anchor(
            goal_x_m, goal_y_m
        )
        try:
            self.navigation.start(
                local_x_m,
                local_y_m,
                reported_goal=(goal_x_m, goal_y_m),
                local_map=local_map,
                pose=pose,
                vehicle_radius_m=vehicle_radius_m,
            )
        except ValueError as error:
            self.navigation.block("invalid_goal", str(error))
            self._finish_blocked_without_vehicle()
            return
        if pose.quality == "lost":
            self.navigation.block("localization_lost")
            self._finish_blocked_without_vehicle()
            return
        if emit_event:
            self._emit(
                mission,
                "active",
                navigation=self.navigation.snapshot(),
            )

    def _advance_subgoal(self, vehicle: Vehicle) -> bool:
        assert self.active_mission is not None
        if self._subgoal_index + 1 >= len(self.active_mission.subgoals):
            return False
        vehicle.stop()
        self._subgoal_index += 1
        self._needs_start = True
        return True

    def _complete_subgoal(
        self,
        vehicle: Vehicle,
        anchor: AnchorSpec,
        pose: PoseEstimate,
        local_map: ObservedGrid,
    ) -> None:
        if self._advance_subgoal(vehicle):
            self._start_or_resume(
                anchor,
                pose,
                local_map,
                vehicle.radius,
                emit_event=False,
            )
        else:
            self._finish_reached(vehicle)

    def _finish_reached(self, vehicle: Vehicle) -> None:
        assert self.active_mission is not None
        mission = self.active_mission
        vehicle.stop()
        self._emit(
            mission,
            "reached",
            self.navigation.reason,
            self.navigation.detail,
            self.navigation.snapshot(),
        )
        self.active_mission = None
        self._subgoal_index = 0
        self._needs_start = bool(self._pending)
        self.auto_state = (
            AutoState.ACTIVE if self._needs_start else AutoState.IDLE
        )

    def _finish_blocked(self, vehicle: Vehicle) -> None:
        vehicle.stop()
        self._finish_blocked_without_vehicle()

    def _finish_blocked_without_vehicle(self) -> None:
        assert self.active_mission is not None
        self.auto_state = AutoState.BLOCKED
        self._needs_start = False
        self._emit(
            self.active_mission,
            "blocked",
            self.navigation.reason or "blocked",
            self.navigation.detail,
            self.navigation.snapshot(),
        )

    def _pause_active(self, reason: str) -> None:
        self._clear_yield()
        if self.auto_state is AutoState.PAUSED:
            self._needs_start = False
            return
        if self.active_mission is not None:
            if self.navigation.status == "active":
                self.navigation.cancel(reason)
            self._emit(
                self.active_mission,
                "paused",
                reason,
                navigation=self.navigation.snapshot(),
            )
        self.auto_state = AutoState.PAUSED
        self._needs_start = False

    def _cancel_all(self, reason: str) -> None:
        self._clear_yield()
        missions = (
            (() if self.active_mission is None else (self.active_mission,))
            + tuple(self._pending)
        )
        if self.navigation.status in {"active", "blocked"}:
            self.navigation.cancel(reason)
        for mission in missions:
            self._emit(mission, "cancelled", reason)
        self.active_mission = None
        self._subgoal_index = 0
        self._pending.clear()
        self.auto_state = AutoState.IDLE
        self._needs_start = False

    def _emit(
        self,
        mission: Mission,
        status: str,
        reason: str | None = None,
        detail: str | None = None,
        navigation: dict[str, object] | None = None,
    ) -> None:
        self._events.append(
            ControllerEvent(
                self.latest_event_seq + 1,
                self.event_epoch,
                mission,
                self._subgoal_index if mission is self.active_mission else 0,
                status,
                reason,
                detail,
                navigation,
            )
        )

    def _active_mission_snapshot(self) -> dict[str, object]:
        assert self.active_mission is not None
        mission = self.active_mission
        goal_x_m, goal_y_m = mission.subgoals[self._subgoal_index]
        return {
            **mission.as_dict(),
            "subgoal_index": self._subgoal_index,
            "subgoal_count": len(mission.subgoals),
            "current_goal": {
                "frame_id": mission.frame_id,
                "x_m": goal_x_m,
                "y_m": goal_y_m,
            },
        }
