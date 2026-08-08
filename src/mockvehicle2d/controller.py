"""Single authority for manual motion and autonomous missions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
import math
import re
from typing import TYPE_CHECKING, ClassVar
import uuid

from mockvehicle2d.coordination import ReservationTable, TimedCell, prioritized_sipp
from mockvehicle2d.navigation import GotoController
from mockvehicle2d.map_sync import (
    CorridorDescriptor,
    MAX_INTENT_WAIT_TICKS,
    MOTION_COMMIT_HORIZON_S,
    MOTION_INTENT_TTL_S,
    MOTION_PLAN_HORIZON_S,
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
CORRIDOR_REJOIN_TOLERANCE_M = 0.1
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
    return _motion_intent_priority_key(first) < _motion_intent_priority_key(second)


def _motion_intent_priority_key(
    intent: PeerMotionIntent,
) -> tuple[int, int, str, str]:
    return (
        0 if intent.reserved else 1,
        0 if intent.reserved else -intent.wait_ticks,
        intent.priority_owner_id,
        intent.source_vehicle_id,
    )


def inherit_motion_priority(
    own: PeerMotionIntent,
    requesters: tuple[PeerMotionIntent, ...],
    intents: tuple[PeerMotionIntent, ...] = (),
) -> PeerMotionIntent:
    def chain_winner(
        candidate: PeerMotionIntent,
        visited: frozenset[str],
    ) -> PeerMotionIntent:
        winner = candidate
        next_visited = visited | {candidate.source_vehicle_id}
        upstream = (
            intent
            for intent in intents
            if intent.source_vehicle_id not in next_visited
            and intent.target_cell == candidate.current_cell
            and intent.current_cell != candidate.current_cell
        )
        for requester in sorted(
            upstream,
            key=lambda intent: intent.source_vehicle_id,
        ):
            inherited = chain_winner(requester, next_visited)
            if motion_intent_precedes(inherited, winner):
                winner = inherited
        return winner

    inherited = own
    for requester in sorted(requesters, key=lambda intent: intent.source_vehicle_id):
        requester = chain_winner(requester, frozenset((own.source_vehicle_id,)))
        if motion_intent_precedes(requester, inherited):
            inherited = requester
    if inherited is own:
        return own
    return replace(
        own,
        wait_ticks=max(own.wait_ticks, inherited.wait_ticks),
        task_sequence=min(own.task_sequence, inherited.task_sequence),
        task_age_ticks=max(own.task_age_ticks, inherited.task_age_ticks),
        priority_owner_id=inherited.priority_owner_id,
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


def corridor_descriptors_conflict(
    first: CorridorDescriptor,
    second: CorridorDescriptor,
) -> bool:
    first_horizontal = first.entry_cell[1] == first.exit_cell[1]
    second_horizontal = second.entry_cell[1] == second.exit_cell[1]
    if first_horizontal != second_horizontal:
        return False
    axis = 0 if first_horizontal else 1
    fixed = 1 - axis
    if first.entry_cell[fixed] != second.entry_cell[fixed]:
        return False
    first_interval = sorted((first.entry_cell[axis], first.exit_cell[axis]))
    second_interval = sorted((second.entry_cell[axis], second.exit_cell[axis]))
    return max(first_interval[0], second_interval[0]) <= min(
        first_interval[1], second_interval[1]
    )


def _corridor_progress(
    corridor: CorridorDescriptor,
    cell: tuple[int, int],
) -> int:
    axis = 0 if corridor.entry_cell[1] == corridor.exit_cell[1] else 1
    direction = 1 if corridor.exit_cell[axis] > corridor.entry_cell[axis] else -1
    return (cell[axis] - corridor.entry_cell[axis]) * direction


def _corridor_direction(corridor: CorridorDescriptor) -> tuple[int, int]:
    axis = 0 if corridor.entry_cell[1] == corridor.exit_cell[1] else 1
    direction = 1 if corridor.exit_cell[axis] > corridor.entry_cell[axis] else -1
    return axis, direction


def _front_corridor_waiter(
    winner: PeerMotionIntent,
    candidates: tuple[PeerMotionIntent, ...],
) -> PeerMotionIntent | None:
    """Return the deterministic front waiter approaching the winner's exit."""
    if winner.corridor is None:
        return None
    winner_axis, winner_direction = _corridor_direction(winner.corridor)
    waiters = []
    for candidate in candidates:
        if (
            candidate.source_vehicle_id == winner.source_vehicle_id
            or candidate.corridor is None
            or not corridor_descriptors_conflict(
                winner.corridor,
                candidate.corridor,
            )
        ):
            continue
        axis, direction = _corridor_direction(candidate.corridor)
        if axis != winner_axis or direction == winner_direction:
            continue
        waiters.append(
            (
                max(
                    0,
                    -_corridor_progress(
                        candidate.corridor,
                        candidate.current_cell,
                    ),
                ),
                candidate.source_vehicle_id,
                candidate,
            )
        )
    return None if not waiters else min(waiters, key=lambda item: item[:2])[2]


def _corridor_path_fixed_m(
    intent: PeerMotionIntent,
    peer: PeerVehicleState | None,
    resolution_m: float,
    fallback_position_m: tuple[float, float] | None = None,
) -> float:
    assert intent.corridor is not None
    axis, _ = _corridor_direction(intent.corridor)
    fixed_axis = 1 - axis
    if intent.target_cell is not None:
        return (intent.target_cell[fixed_axis] + 0.5) * resolution_m
    if peer is not None:
        return (peer.global_x_m, peer.global_y_m)[fixed_axis]
    if fallback_position_m is not None:
        return fallback_position_m[fixed_axis]
    return (intent.corridor.entry_cell[fixed_axis] + 0.5) * resolution_m


def _corridor_waiter_is_staged(
    winner: PeerMotionIntent,
    waiter: PeerMotionIntent,
    winner_state: PeerVehicleState | None,
    waiter_state: PeerVehicleState | None,
    resolution_m: float,
    winner_radius_m: float,
    winner_position_m: tuple[float, float],
) -> bool:
    """Require the waiter's current pose and remaining stage sweep off-axis."""
    if winner.corridor is None or waiter_state is None:
        return False
    axis, _ = _corridor_direction(winner.corridor)
    fixed_axis = 1 - axis
    path_fixed_m = _corridor_path_fixed_m(
        winner,
        winner_state,
        resolution_m,
        winner_position_m,
    )
    current_fixed_m = (
        waiter_state.global_x_m,
        waiter_state.global_y_m,
    )[fixed_axis]
    sweep_fixed_m = current_fixed_m
    if waiter.target_cell is not None:
        sweep_fixed_m = (
            waiter.target_cell[fixed_axis] + 0.5
        ) * resolution_m
    current_side = current_fixed_m - path_fixed_m
    sweep_side = sweep_fixed_m - path_fixed_m
    sweep_clearance_m = (
        0.0
        if current_side * sweep_side < 0.0
        else min(abs(current_side), abs(sweep_side))
    )
    required_center_clearance_m = (
        winner_radius_m
        + waiter_state.radius_m
        + math.sqrt(max(waiter_state.covariance[:2]))
        + AUTOMATIC_MINIMUM_CLEARANCE_M
    )
    return (
        abs(current_side) >= required_center_clearance_m - 1e-12
        and sweep_clearance_m >= required_center_clearance_m - 1e-12
    )


def _corridor_entry_gate_reached(
    corridor: CorridorDescriptor,
    point_m: tuple[float, float],
    resolution_m: float,
    vehicle: Vehicle,
) -> bool:
    """Start bounded braking before the vehicle footprint crosses the entry."""
    axis, direction = _corridor_direction(corridor)
    entry_face_m = (
        corridor.entry_cell[axis] * resolution_m
        if direction > 0
        else (corridor.entry_cell[axis] + 1) * resolution_m
    )
    linear_mps = abs(vehicle.body_velocities()[0])
    braking_distance_m = linear_mps**2 / (
        2 * vehicle.linear_deceleration_mps2
    )
    hold_center_m = entry_face_m - direction * (
        vehicle.radius + braking_distance_m
    )
    return direction * point_m[axis] >= direction * hold_center_m - 1e-12


def _rejoin_segment_stays_before_corridor_entry(
    corridor: CorridorDescriptor,
    start_m: tuple[float, float],
    end_m: tuple[float, float],
    resolution_m: float,
    vehicle_radius_m: float,
) -> bool:
    axis, direction = _corridor_direction(corridor)
    entry_face_m = (
        corridor.entry_cell[axis] * resolution_m
        if direction > 0
        else (corridor.entry_cell[axis] + 1) * resolution_m
    )
    footprint_limit_m = direction * entry_face_m - vehicle_radius_m
    return max(
        direction * start_m[axis],
        direction * end_m[axis],
    ) <= footprint_limit_m + 1e-12


def _corridor_passed(
    corridor: CorridorDescriptor,
    point_m: tuple[float, float],
    resolution_m: float,
    vehicle_radius_m: float,
) -> bool:
    """Require the whole collision footprint and safety margin past the exit."""
    axis = 0 if corridor.entry_cell[1] == corridor.exit_cell[1] else 1
    direction = 1 if corridor.exit_cell[axis] > corridor.entry_cell[axis] else -1
    far_face_m = (
        (corridor.exit_cell[axis] + 1) * resolution_m
        if direction > 0
        else corridor.exit_cell[axis] * resolution_m
    )
    release_center_m = far_face_m + direction * (
        vehicle_radius_m + AUTOMATIC_MINIMUM_CLEARANCE_M
    )
    return direction * point_m[axis] > direction * release_center_m


def _extend_corridor_release(
    corridor: CorridorDescriptor,
    peer_corridor: CorridorDescriptor,
) -> CorridorDescriptor:
    """Monotonically include the opposite peer's confirmed entry boundary."""
    axis = 0 if corridor.entry_cell[1] == corridor.exit_cell[1] else 1
    current_length = round(math.dist(corridor.entry_cell, corridor.exit_cell))
    if _corridor_progress(corridor, peer_corridor.entry_cell) <= current_length:
        return corridor
    exit_cell = list(corridor.exit_cell)
    exit_cell[axis] = peer_corridor.entry_cell[axis]
    return CorridorDescriptor(corridor.entry_cell, tuple(exit_cell))


def _effective_corridor_intent(
    intent: PeerMotionIntent,
    live_corridor_vehicle_ids: frozenset[str],
) -> PeerMotionIntent:
    """Drop inherited ownership after that owner leaves this corridor lease."""
    if intent.priority_owner_id in live_corridor_vehicle_ids:
        return intent
    return replace(
        intent,
        priority_owner_id=intent.source_vehicle_id,
        reserved=False,
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


def _global_coordination_path(
    anchor: AnchorSpec,
    local_path: tuple[tuple[int, int], ...],
    resolution_m: float,
    current_cell: tuple[int, int],
    target_cell: tuple[int, int] | None,
) -> tuple[tuple[int, int], ...]:
    transformed = []
    for gx, gy in local_path:
        global_x_m, global_y_m, _ = anchor.anchor_to_global(
            (gx + 0.5) * resolution_m,
            (gy + 0.5) * resolution_m,
            0.0,
        )
        cell = (
            math.floor(global_x_m / resolution_m),
            math.floor(global_y_m / resolution_m),
        )
        if not transformed or transformed[-1] != cell:
            transformed.append(cell)
    result = [current_cell]
    if target_cell is None or target_cell == current_cell:
        return tuple(result)
    result.append(target_cell)
    if target_cell in transformed:
        result.extend(transformed[transformed.index(target_cell) + 1 :])
    return tuple(result)


def _reservation_time_margin_s(vehicle: Vehicle) -> float:
    """Cover one publish lease plus acceleration/braking timing uncertainty."""
    speed = vehicle.linear_speed
    dynamic_margin_s = max(
        speed / (2 * vehicle.linear_acceleration_mps2),
        speed / (2 * vehicle.linear_deceleration_mps2),
    )
    return MOTION_INTENT_TTL_S + dynamic_margin_s


def _global_corridor(
    anchor: AnchorSpec,
    local_corridor: tuple[tuple[int, int], tuple[int, int]],
    resolution_m: float,
    current_cell: tuple[int, int],
) -> CorridorDescriptor | None:
    transformed = []
    for gx, gy in local_corridor:
        global_x_m, global_y_m, _ = anchor.anchor_to_global(
            (gx + 0.5) * resolution_m,
            (gy + 0.5) * resolution_m,
            0.0,
        )
        transformed.append(
            (
                math.floor(global_x_m / resolution_m),
                math.floor(global_y_m / resolution_m),
            )
        )
    first, second = transformed
    delta_x = abs(second[0] - first[0])
    delta_y = abs(second[1] - first[1])
    if delta_x and not delta_y:
        first = first[0], current_cell[1]
        second = second[0], current_cell[1]
    elif delta_y and not delta_x:
        first = current_cell[0], first[1]
        second = current_cell[0], second[1]
    else:
        return None
    return CorridorDescriptor(first, second)


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


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    length_squared = delta_x**2 + delta_y**2
    if length_squared <= 1e-12:
        return math.dist(point, start)
    projection = max(
        0.0,
        min(
            1.0,
            (
                (point[0] - start[0]) * delta_x
                + (point[1] - start[1]) * delta_y
            )
            / length_squared,
        ),
    )
    return math.dist(
        point,
        (
            start[0] + projection * delta_x,
            start[1] + projection * delta_y,
        ),
    )


def _intent_sweep_distance(
    point_m: tuple[float, float],
    intent: PeerMotionIntent,
    peer: PeerVehicleState | None,
    resolution_m: float,
) -> float:
    start = (
        (
            (intent.current_cell[0] + 0.5) * resolution_m,
            (intent.current_cell[1] + 0.5) * resolution_m,
        )
        if peer is None
        else (peer.global_x_m, peer.global_y_m)
    )
    endpoints = []
    if intent.target_cell is not None:
        endpoints.append(
            (
                (intent.target_cell[0] + 0.5) * resolution_m,
                (intent.target_cell[1] + 0.5) * resolution_m,
            )
        )
    if peer is not None and math.hypot(peer.vx_mps, peer.vy_mps) > 1e-12:
        endpoints.append(
            (
                start[0] + peer.vx_mps * PEER_CONFLICT_HORIZON_S,
                start[1] + peer.vy_mps * PEER_CONFLICT_HORIZON_S,
            )
        )
    return min(
        (
            _point_segment_distance(point_m, start, endpoint)
            for endpoint in endpoints
        ),
        default=math.dist(point_m, start),
    )


def _rejoin_segment_blocked_by_peer(
    start_m: tuple[float, float],
    end_m: tuple[float, float],
    vehicle: Vehicle,
    peers: tuple[PeerVehicleState, ...],
) -> bool:
    return any(
        _point_segment_distance(
            (peer.global_x_m, peer.global_y_m),
            start_m,
            end_m,
        )
        <= vehicle.radius
        + peer.radius_m
        + math.sqrt(max(peer.covariance[:2]))
        + AUTOMATIC_MINIMUM_CLEARANCE_M
        + 1e-12
        for peer in peers
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
        self._corridor: CorridorDescriptor | None = None
        self._corridor_reserved = False
        self._corridor_admission_confirmed = False
        self._corridor_claim_ticks = 0
        self._corridor_rejoin_target_m: tuple[float, float] | None = None
        self._coordination_wait_reason: str | None = None
        self._coordination_wait_owner_id: str | None = None
        self._known_coordination_peer_ids: set[str] = set()
        self._active_task_age_ticks = 0
        self._temporal_plan_generation = 0
        self._temporal_plan_signature: tuple[object, ...] | None = None
        self._temporal_trajectory: tuple[TimedCell, ...] = ()
        self._temporal_committed_until_s = 0.0
        self._temporal_goal_hold = False
        self._temporal_safety_margin_s = 0.0
        self._temporal_commit_deadline_s = 0.0

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
        CorridorDescriptor | None,
    ]:
        return (
            self._intent_target_m,
            self._reservation_wait_ticks,
            self._intent_priority_owner_id,
            self._intent_reserved,
            self._corridor,
        )

    @property
    def temporal_motion_intent(self) -> tuple[
        int | None,
        int,
        int,
        tuple[TimedCell, ...],
        float,
        bool,
        float,
    ]:
        return (
            (
                self._temporal_plan_generation
                if self._temporal_trajectory
                else None
            ),
            (
                self.active_mission.submitted_seq
                if self.active_mission is not None
                else (1 << 64) - 1
            ),
            self._active_task_age_ticks,
            self._temporal_trajectory,
            self._temporal_committed_until_s,
            self._temporal_goal_hold,
            self._temporal_safety_margin_s,
        )

    def planning_ignored_peer_ids(
        self,
        vehicle_id: str,
        peer_motion_intents: tuple[PeerMotionIntent, ...],
    ) -> frozenset[str]:
        """Return yielded corridor peers the confirmed owner may plan through."""
        if (
            self._corridor is None
            or not self._corridor_reserved
            or not self._corridor_admission_confirmed
            or self._intent_priority_owner_id != vehicle_id
        ):
            return frozenset()
        return frozenset(
            intent.source_vehicle_id
            for intent in peer_motion_intents
            if intent.corridor is not None
            and not intent.reserved
            and intent.priority_owner_id == vehicle_id
            and corridor_descriptors_conflict(self._corridor, intent.corridor)
        )

    def _live_corridor_lease_owner(
        self,
        vehicle_id: str | None,
        peer_motion_intents: tuple[PeerMotionIntent, ...],
    ) -> str | None:
        """Return the peer whose live matching lease this vehicle is yielding to."""
        owner_id = self._intent_priority_owner_id
        if (
            vehicle_id is None
            or self._corridor is None
            or self._corridor_reserved
            or owner_id is None
            or owner_id == vehicle_id
        ):
            return None
        for intent in peer_motion_intents:
            if (
                intent.source_vehicle_id == owner_id
                and intent.priority_owner_id == owner_id
                and intent.reserved
                and intent.corridor is not None
                and corridor_descriptors_conflict(
                    self._corridor,
                    intent.corridor,
                )
            ):
                return owner_id
        return None

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
        coordination_map: ObservedGrid | None = None,
        coordination_ready: bool | None = None,
        expected_peer_vehicle_ids: tuple[str, ...] = (),
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
        self._active_task_age_ticks = min(
            MAX_INTENT_WAIT_TICKS,
            self._active_task_age_ticks + 1,
        )

        persistent_map = coordination_map
        if persistent_map is None:
            persistent_grid = getattr(local_map, "persistent_grid", None)
            if callable(persistent_grid):
                persistent_map = persistent_grid()
        has_transient_obstacles = getattr(
            local_map,
            "has_transient_obstacles",
            None,
        )
        transient_active = (
            bool(has_transient_obstacles())
            if callable(has_transient_obstacles)
            else False
        )
        has_attributed_peer_obstacles = getattr(
            local_map,
            "has_attributed_peer_obstacles",
            None,
        )
        attributed_peer_active = (
            bool(has_attributed_peer_obstacles())
            if callable(has_attributed_peer_obstacles)
            else False
        )
        classify_no_path = getattr(
            self.navigation,
            "classify_no_path_against_persistent",
            None,
        )
        corridor_wait_owner = self._live_corridor_lease_owner(
            vehicle_id,
            peer_motion_intents,
        )
        if getattr(
            self.navigation,
            "static_no_path_probe_pending",
            False,
        ) is True:
            if persistent_map is None or not callable(classify_no_path):
                self.navigation.block("no_path", self.navigation.detail)
                self._finish_blocked(vehicle)
                return
            no_path_kind = classify_no_path(
                pose,
                persistent_map,
                local_map,
                transient_active=transient_active,
                attributed_peer_active=attributed_peer_active,
            )
            if no_path_kind == "static":
                self._finish_blocked(vehicle)
                return
            desired = (0.0, 0.0)
        elif (
            corridor_wait_owner is None
            and self._corridor_rejoin_target_m is None
        ):
            self._coordination_wait_reason = None
            self._coordination_wait_owner_id = None
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
                if (
                    self.navigation.reason == "no_path"
                    and persistent_map is not None
                    and callable(classify_no_path)
                ):
                    no_path_kind = classify_no_path(
                        pose,
                        persistent_map,
                        local_map,
                        transient_active=transient_active,
                        attributed_peer_active=attributed_peer_active,
                    )
                    if no_path_kind != "static":
                        desired = (0.0, 0.0)
                    else:
                        self._finish_blocked(vehicle)
                        return
                else:
                    self._finish_blocked(vehicle)
                    return
        else:
            # A live corridor lease is an explicit coordination wait, not a
            # navigation failure.  A staged waiter also keeps it frozen until
            # it has reversed over the already-validated segment; planning
            # from the side pocket can otherwise produce a terminal no_path.
            self._coordination_wait_reason = (
                None if corridor_wait_owner is None else "corridor_lease"
            )
            self._coordination_wait_owner_id = corridor_wait_owner
            desired = (0.0, 0.0)

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
            coordination_map=coordination_map,
            coordination_ready=coordination_ready,
            expected_peer_vehicle_ids=expected_peer_vehicle_ids,
        )
        corridor_rejoin_active = self._corridor_rejoin_target_m is not None

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
            if (
                corridor_rejoin_active
                and decision.reason == "safety_obstacle"
                and safety_scan_points is not None
                and any(point.dynamic for point in safety_scan_points)
            ):
                return
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

    def _schedule_temporal_motion(
        self,
        desired: tuple[float, float],
        *,
        own: PeerMotionIntent,
        vehicle: Vehicle,
        anchor: AnchorSpec,
        pose: PoseEstimate,
        local_map: ObservedGrid,
        now: float,
        peers: dict[str, PeerVehicleState],
        peer_motion_intents: tuple[PeerMotionIntent, ...],
        coordination_map: ObservedGrid | None,
    ) -> tuple[tuple[float, float], tuple[int, int] | None, PeerMotionIntent, bool]:
        current_cell, target_cell = own.current_cell, own.target_cell
        route_source = coordination_map or local_map
        route_method = getattr(self.navigation, "coordination_path_cells", None)
        local_path = (
            route_method(pose, route_source)
            if callable(route_method)
            else None
        )
        spatial_path = _global_coordination_path(
            anchor,
            local_path if isinstance(local_path, tuple) else (),
            local_map.resolution_m,
            current_cell,
            target_cell,
        )
        own_margin_s = _reservation_time_margin_s(vehicle)
        reservations = ReservationTable(
            local_map.resolution_m,
            own_radius_m=vehicle.radius,
            clearance_m=AUTOMATIC_MINIMUM_CLEARANCE_M,
        )
        higher_priority = tuple(
            intent
            for intent in peer_motion_intents
            if intent.target_cell != current_cell
            and motion_intent_precedes(intent, own)
        )
        for intent in sorted(
            higher_priority,
            key=lambda item: item.source_vehicle_id,
        ):
            peer = peers.get(intent.source_vehicle_id)
            reservations.add(
                intent.source_vehicle_id,
                intent.timed_trajectory,
                base_time_s=(
                    now
                    if intent.received_at_s is None
                    else intent.received_at_s
                ),
                radius_m=(
                    vehicle.radius
                    if peer is None
                    else peer.radius_m
                    + math.sqrt(max(peer.covariance[:2]))
                ),
                time_margin_s=intent.safety_time_margin_s,
                goal_hold=intent.goal_hold,
            )

        global_yaw_rad = anchor.anchor_to_global(
            pose.x_m,
            pose.y_m,
            pose.yaw_rad,
        )[2]

        def schedule(path: tuple[tuple[int, int], ...]) -> tuple[TimedCell, ...] | None:
            return prioritized_sipp(
                path,
                reservations,
                now_s=now,
                horizon_s=MOTION_PLAN_HORIZON_S,
                linear_speed_mps=vehicle.linear_speed,
                angular_speed_rps=vehicle.angular_speed,
                initial_yaw_rad=global_yaw_rad,
                time_margin_s=own_margin_s,
            )

        plan = schedule(spatial_path)
        if plan is None:
            detours = getattr(self.navigation, "coordination_detours", None)
            choices = detours(pose, local_map) if callable(detours) else ()
            for detour_m in choices if isinstance(choices, tuple) else ():
                detour_cell = _coordination_cell(
                    anchor,
                    detour_m,
                    local_map.resolution_m,
                )
                candidate = schedule((current_cell, detour_cell))
                if candidate is None:
                    continue
                self._invalidate_temporal_commit()
                plan = candidate
                target_cell = detour_cell
                desired = _intent_setpoint(detour_m, pose, vehicle)
                self._intent_target_m = detour_m
                own = replace(own, target_cell=target_cell)
                break

        task_sequence = own.task_sequence
        peer_plans = tuple(
            sorted(
                (
                    intent.source_vehicle_id,
                    intent.intent_generation,
                    intent.plan_generation,
                )
                for intent in higher_priority
            )
        )
        if plan is None:
            plan = (TimedCell(current_cell, 0.0, MOTION_PLAN_HORIZON_S),)
            target_cell = None
            own = replace(own, target_cell=None)
            self._intent_target_m = None
            goal_hold = False
            blocked = True
        else:
            goal_cell = (
                None
                if self.active_mission is None
                else (
                    math.floor(
                        self.active_mission.subgoals[self._subgoal_index][0]
                        / local_map.resolution_m
                    ),
                    math.floor(
                        self.active_mission.subgoals[self._subgoal_index][1]
                        / local_map.resolution_m
                    ),
                )
            )
            goal_hold = (
                goal_cell is not None
                and plan[-1].cell == goal_cell
                and len(plan) == len(spatial_path)
            )
            if len(plan) < 2:
                blocked = target_cell not in {None, current_cell}
            else:
                target_heading = math.atan2(
                    plan[1].cell[1] - current_cell[1],
                    plan[1].cell[0] - current_cell[0],
                )
                nominal_turn_s = abs(
                    math.atan2(
                        math.sin(target_heading - global_yaw_rad),
                        math.cos(target_heading - global_yaw_rad),
                    )
                ) / vehicle.angular_speed
                blocked = plan[0].leave_offset_s > nominal_turn_s + 1e-9

        signature = (
            task_sequence,
            tuple(item.cell for item in plan),
            blocked,
            goal_hold,
            peer_plans,
        )
        if signature != self._temporal_plan_signature:
            self._temporal_plan_generation += 1
            self._temporal_plan_signature = signature
        self._temporal_trajectory = plan
        self._temporal_goal_hold = goal_hold
        self._temporal_safety_margin_s = own_margin_s
        if blocked:
            self._temporal_commit_deadline_s = 0.0
            self._temporal_committed_until_s = 0.0
        else:
            if self._temporal_commit_deadline_s <= now:
                self._temporal_commit_deadline_s = (
                    now + MOTION_COMMIT_HORIZON_S
                )
            self._temporal_committed_until_s = min(
                self._temporal_commit_deadline_s - now,
                plan[-1].leave_offset_s,
            )
            own = replace(own, reserved=True)
        return desired, target_cell, own, blocked

    def _hold_for_temporal_reservation(
        self,
        own: PeerMotionIntent,
        peer_motion_intents: tuple[PeerMotionIntent, ...],
    ) -> tuple[float, float]:
        temporal_winner = next(
            (
                intent
                for intent in sorted(
                    peer_motion_intents,
                    key=_motion_intent_priority_key,
                )
                if motion_intent_precedes(intent, own)
            ),
            None,
        )
        self._yielding_for = (
            None if temporal_winner is None else temporal_winner.source_vehicle_id
        )
        self._yield_requires_intent = temporal_winner is not None
        self._intent_reserved = False
        self._coordination_wait_reason = "space_time_reservation"
        self._coordination_wait_owner_id = (
            None if temporal_winner is None else temporal_winner.priority_owner_id
        )
        self._reservation_wait_ticks = min(
            MAX_INTENT_WAIT_TICKS,
            self._reservation_wait_ticks + 1,
        )
        return 0.0, 0.0

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
        coordination_map: ObservedGrid | None = None,
        coordination_ready: bool | None = None,
        expected_peer_vehicle_ids: tuple[str, ...] = (),
    ) -> tuple[float, float]:
        if vehicle_id is None:
            self._clear_yield()
            return desired

        peers = {state.source_vehicle_id: state for state in peer_states}
        intents = {
            intent.source_vehicle_id: intent for intent in peer_motion_intents
        }
        self._known_coordination_peer_ids.update(expected_peer_vehicle_ids)
        self._known_coordination_peer_ids.update(peers)
        self._known_coordination_peer_ids.update(intents)
        self._known_coordination_peer_ids.discard(vehicle_id)
        fresh_intent_quorum = (
            coordination_ready is not False
            and self._known_coordination_peer_ids <= intents.keys()
        )
        temporal_quorum_required = (
            bool(expected_peer_vehicle_ids) or coordination_ready is not None
        )
        fresh_temporal_quorum = (
            coordination_ready is not False
            and self._known_coordination_peer_ids
            <= (intents.keys() & peers.keys())
        )
        motion_target = self.navigation.motion_target
        self._intent_target_m = motion_target
        self._intent_priority_owner_id = vehicle_id
        current_cell = _coordination_cell(
            anchor,
            (pose.x_m, pose.y_m),
            local_map.resolution_m,
        )
        global_x_m, global_y_m, _ = anchor.anchor_to_global(
            pose.x_m,
            pose.y_m,
            pose.yaw_rad,
        )
        if self._corridor is not None and _corridor_passed(
            self._corridor,
            (global_x_m, global_y_m),
            local_map.resolution_m,
            vehicle.radius,
        ):
            self._corridor = None
            self._corridor_reserved = False
            self._corridor_admission_confirmed = False
            self._corridor_claim_ticks = 0
            self._corridor_rejoin_target_m = None
            self._coordination_wait_reason = None
            self._coordination_wait_owner_id = None
        if self._corridor is None:
            corridor_source = coordination_map or local_map
            detect_corridor = getattr(self.navigation, "coordination_corridor", None)
            local_corridor = (
                None
                if detect_corridor is None
                else detect_corridor(pose, corridor_source)
            )
            if (
                isinstance(local_corridor, tuple)
                and len(local_corridor) == 2
                and all(
                    isinstance(cell, tuple)
                    and len(cell) == 2
                    and all(type(value) is int for value in cell)
                    for cell in local_corridor
                )
            ):
                self._corridor = _global_corridor(
                    anchor,
                    local_corridor,
                    local_map.resolution_m,
                    current_cell,
                )
                if self._corridor is not None:
                    self._invalidate_temporal_commit()
                self._corridor_admission_confirmed = False
                self._corridor_claim_ticks = 0
        if self._last_coordination_cell != current_cell:
            self._last_coordination_cell = current_cell
            if self._corridor is None:
                self._reservation_wait_ticks = 0
            self._reservation_hold_ticks = 0
            self._reservation_cells = None
        target_cell = (
            None
            if motion_target is None
            else _coordination_cell(anchor, motion_target, local_map.resolution_m)
        )
        corridor_peers: list[PeerMotionIntent] = []
        if self._corridor is not None:
            while True:
                corridor_peers = [
                    intent
                    for intent in peer_motion_intents
                    if intent.corridor is not None
                    and corridor_descriptors_conflict(
                        self._corridor,
                        intent.corridor,
                    )
                ]
                extended = self._corridor
                for peer_intent in corridor_peers:
                    assert peer_intent.corridor is not None
                    extended = _extend_corridor_release(
                        extended,
                        peer_intent.corridor,
                    )
                if extended == self._corridor:
                    break
                self._corridor = extended
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
            (
                self._corridor_reserved
                if self._corridor is not None
                else self._intent_reserved
                and now < self._temporal_commit_deadline_s
                and self._reservation_hold_ticks
                < PEER_RESERVATION_MAX_HOLD_TICKS
            ),
            self._corridor,
            task_sequence=(
                self.active_mission.submitted_seq
                if self.active_mission is not None
                else (1 << 64) - 1
            ),
            task_age_ticks=self._active_task_age_ticks,
        )
        corridor_peer_ids = {
            intent.source_vehicle_id for intent in corridor_peers
        }
        if self._corridor is not None:
            live_corridor_vehicle_ids = frozenset(
                corridor_peer_ids | {vehicle_id}
            )
            corridor_length = round(
                math.dist(
                    self._corridor.entry_cell,
                    self._corridor.exit_cell,
                )
            )
            inside = 0 <= _corridor_progress(self._corridor, current_cell) <= (
                corridor_length
            )
            winner = own
            if not inside:
                for peer_intent in sorted(
                    corridor_peers,
                    key=lambda intent: intent.source_vehicle_id,
                ):
                    peer_intent = _effective_corridor_intent(
                        peer_intent,
                        live_corridor_vehicle_ids,
                    )
                    if motion_intent_precedes(peer_intent, winner):
                        winner = peer_intent
            front_waiter = _front_corridor_waiter(
                winner,
                (own, *corridor_peers),
            )
            if winner is not own:
                self._corridor_reserved = False
                self._corridor_admission_confirmed = False
                self._corridor_claim_ticks = 0
                self._intent_reserved = False
                self._invalidate_temporal_commit()
                self._intent_priority_owner_id = winner.priority_owner_id
                self._coordination_wait_reason = (
                    "corridor_lease"
                    if winner.reserved
                    and winner.source_vehicle_id == winner.priority_owner_id
                    else "corridor_election"
                )
                self._coordination_wait_owner_id = winner.priority_owner_id
                self._reservation_wait_ticks = min(
                    MAX_INTENT_WAIT_TICKS,
                    self._reservation_wait_ticks + 1,
                )
                axis = (
                    0
                    if self._corridor.entry_cell[1]
                    == self._corridor.exit_cell[1]
                    else 1
                )
                fixed_axis = 1 - axis
                winner_state = peers.get(winner.source_vehicle_id)
                winner_path_fixed_m = _corridor_path_fixed_m(
                    winner,
                    winner_state,
                    local_map.resolution_m,
                )
                required_center_clearance_m = (
                    2 * vehicle.radius + AUTOMATIC_MINIMUM_CLEARANCE_M
                    if winner_state is None
                    else vehicle.radius
                    + winner_state.radius_m
                    + math.sqrt(max(winner_state.covariance[:2]))
                    + AUTOMATIC_MINIMUM_CLEARANCE_M
                )
                current_lateral_clearance_m = abs(
                    (global_x_m, global_y_m)[fixed_axis]
                    - winner_path_fixed_m
                )
                current_sweep_clearance_m = _intent_sweep_distance(
                    (global_x_m, global_y_m),
                    winner,
                    winner_state,
                    local_map.resolution_m,
                )
                current_clearance_score_m = min(
                    current_lateral_clearance_m,
                    current_sweep_clearance_m,
                )
                unavailable = {
                    cell
                    for intent in peer_motion_intents
                    for cell in (intent.current_cell, intent.target_cell)
                    if cell is not None
                }
                choices = []
                for detour_m in self.navigation.coordination_detours(
                    pose,
                    local_map,
                ):
                    detour_cell = _coordination_cell(
                        anchor,
                        detour_m,
                        local_map.resolution_m,
                    )
                    lateral_offset = abs(
                        detour_cell[fixed_axis]
                        - self._corridor.entry_cell[fixed_axis]
                    )
                    detour_global_x_m, detour_global_y_m, _ = (
                        anchor.anchor_to_global(*detour_m, 0.0)
                    )
                    detour_lateral_clearance_m = abs(
                        (detour_global_x_m, detour_global_y_m)[fixed_axis]
                        - winner_path_fixed_m
                    )
                    detour_sweep_clearance_m = _intent_sweep_distance(
                        (detour_global_x_m, detour_global_y_m),
                        winner,
                        winner_state,
                        local_map.resolution_m,
                    )
                    detour_clearance_score_m = min(
                        detour_lateral_clearance_m,
                        detour_sweep_clearance_m,
                    )
                    if (
                        detour_cell in unavailable
                        or _corridor_progress(self._corridor, detour_cell) > 0
                        or lateral_offset
                        <= abs(
                            current_cell[fixed_axis]
                            - self._corridor.entry_cell[fixed_axis]
                        )
                        or detour_clearance_score_m
                        <= current_clearance_score_m + 1e-12
                    ):
                        continue
                    choices.append(
                        (
                            -detour_clearance_score_m,
                            -lateral_offset,
                            detour_cell,
                            detour_m,
                        )
                    )
                staged_position_clear = (
                    current_lateral_clearance_m
                    >= required_center_clearance_m - 1e-12
                    and current_sweep_clearance_m
                    >= required_center_clearance_m - 1e-12
                )
                stage_until_clear = (
                    not staged_position_clear
                    and front_waiter is not None
                    and front_waiter.source_vehicle_id == vehicle_id
                )
                if choices and stage_until_clear:
                    _, _, target_cell, motion_target = min(choices)
                    if self._corridor_rejoin_target_m is None:
                        self._corridor_rejoin_target_m = (
                            pose.x_m,
                            pose.y_m,
                        )
                    self._intent_target_m = motion_target
                    return _intent_setpoint(motion_target, pose, vehicle)
                if staged_position_clear:
                    self._intent_target_m = None
                return 0.0, 0.0
            self._corridor_reserved = True
            self._intent_reserved = True
            self._intent_priority_owner_id = vehicle_id
            self._coordination_wait_reason = None
            self._coordination_wait_owner_id = None
            if inside:
                self._corridor_admission_confirmed = True
            elif not self._corridor_admission_confirmed:
                self._corridor_claim_ticks = min(
                    PEER_RESERVATION_MAX_HOLD_TICKS,
                    self._corridor_claim_ticks + 1,
                )
                acknowledged = (
                    bool(corridor_peers)
                    and all(
                        not intent.reserved
                        and intent.priority_owner_id == vehicle_id
                        for intent in corridor_peers
                    )
                    and fresh_intent_quorum
                )
                uncontested_announcement_complete = (
                    not corridor_peers
                    and fresh_intent_quorum
                    and self._corridor_claim_ticks >= PEER_YIELD_CLEAR_TICKS
                )
                self._corridor_admission_confirmed = (
                    acknowledged or uncontested_announcement_complete
                )
            own = replace(
                own,
                target_cell=target_cell,
                wait_ticks=self._reservation_wait_ticks,
                priority_owner_id=vehicle_id,
                reserved=True,
                corridor=self._corridor,
            )
            front_waiter_state = (
                None
                if front_waiter is None
                else peers.get(front_waiter.source_vehicle_id)
            )
            front_waiter_staged = (
                front_waiter is None
                or _corridor_waiter_is_staged(
                    own,
                    front_waiter,
                    None,
                    front_waiter_state,
                    local_map.resolution_m,
                    vehicle.radius,
                    (global_x_m, global_y_m),
                )
            )
            rejoin_target = self._corridor_rejoin_target_m
            rejoin_global_m = None
            rejoin_stays_before_entry = False
            if rejoin_target is not None:
                rejoin_global_x_m, rejoin_global_y_m, _ = anchor.anchor_to_global(
                    *rejoin_target,
                    0.0,
                )
                rejoin_global_m = rejoin_global_x_m, rejoin_global_y_m
                rejoin_stays_before_entry = (
                    _rejoin_segment_stays_before_corridor_entry(
                        self._corridor,
                        (global_x_m, global_y_m),
                        rejoin_global_m,
                        local_map.resolution_m,
                        vehicle.radius,
                    )
                )
            entry_gate_blocked = (
                not self._corridor_admission_confirmed
                or not front_waiter_staged
            ) and _corridor_entry_gate_reached(
                self._corridor,
                (global_x_m, global_y_m),
                local_map.resolution_m,
                vehicle,
            )
            if entry_gate_blocked and not rejoin_stays_before_entry:
                return 0.0, 0.0
            if rejoin_target is not None:
                assert rejoin_global_m is not None
                if _rejoin_segment_blocked_by_peer(
                    (global_x_m, global_y_m),
                    rejoin_global_m,
                    vehicle,
                    peer_states,
                ):
                    self._corridor_rejoin_target_m = None
                    self._intent_target_m = None
                    return 0.0, 0.0
                if math.dist(
                    (pose.x_m, pose.y_m),
                    rejoin_target,
                ) > CORRIDOR_REJOIN_TOLERANCE_M:
                    self._intent_target_m = rejoin_target
                    return _intent_setpoint(rejoin_target, pose, vehicle)
                self._corridor_rejoin_target_m = None
                self._intent_target_m = None
                return 0.0, 0.0
        if self._corridor is None:
            if temporal_quorum_required and not fresh_temporal_quorum:
                self._invalidate_temporal_commit()
                sync_signature = (
                    "reservation_sync",
                    (
                        self.active_mission.submitted_seq
                        if self.active_mission is not None
                        else (1 << 64) - 1
                    ),
                    current_cell,
                )
                if sync_signature != self._temporal_plan_signature:
                    self._temporal_plan_generation += 1
                    self._temporal_plan_signature = sync_signature
                self._temporal_trajectory = (
                    TimedCell(
                        current_cell,
                        0.0,
                        MOTION_PLAN_HORIZON_S,
                    ),
                )
                self._temporal_committed_until_s = MOTION_COMMIT_HORIZON_S
                self._temporal_goal_hold = False
                self._temporal_safety_margin_s = _reservation_time_margin_s(
                    vehicle
                )
                self._intent_target_m = None
                self._intent_reserved = False
                self._coordination_wait_reason = "reservation_sync"
                self._coordination_wait_owner_id = None
                self._reservation_wait_ticks = min(
                    MAX_INTENT_WAIT_TICKS,
                    self._reservation_wait_ticks + 1,
                )
                return 0.0, 0.0
            desired, target_cell, own, temporal_blocked = (
                self._schedule_temporal_motion(
                    desired,
                    own=own,
                    vehicle=vehicle,
                    anchor=anchor,
                    pose=pose,
                    local_map=local_map,
                    now=now,
                    peers=peers,
                    peer_motion_intents=peer_motion_intents,
                    coordination_map=coordination_map,
                )
            )
            if temporal_blocked:
                return self._hold_for_temporal_reservation(
                    own,
                    peer_motion_intents,
                )
            self._coordination_wait_reason = None
            self._coordination_wait_owner_id = None
            temporal_target_cell = target_cell
        else:
            temporal_target_cell = None
        requesters = [
            intent
            for intent in peer_motion_intents
            if intent.source_vehicle_id not in corridor_peer_ids
            and intent.target_cell == current_cell
            and intent.current_cell != current_cell
        ]
        inherited_from = None
        inherited_own = inherit_motion_priority(
            own,
            tuple(requesters),
            peer_motion_intents,
        )
        if inherited_own.priority_owner_id != own.priority_owner_id:
            inherited_from = next(
                (
                    requester
                    for requester in requesters
                    if requester.priority_owner_id
                    == inherited_own.priority_owner_id
                ),
                min(
                    requesters,
                    key=lambda requester: requester.source_vehicle_id,
                ),
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
                own = replace(own, target_cell=target_cell)
                break

        if self._corridor is None and target_cell != temporal_target_cell:
            self._invalidate_temporal_commit()
            desired, target_cell, own, temporal_blocked = (
                self._schedule_temporal_motion(
                    desired,
                    own=own,
                    vehicle=vehicle,
                    anchor=anchor,
                    pose=pose,
                    local_map=local_map,
                    now=now,
                    peers=peers,
                    peer_motion_intents=peer_motion_intents,
                    coordination_map=coordination_map,
                )
            )
            if temporal_blocked:
                return self._hold_for_temporal_reservation(
                    own,
                    peer_motion_intents,
                )

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
                        self._invalidate_temporal_commit()
                        self._intent_target_m = None
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
            if peer_intent.source_vehicle_id in corridor_peer_ids:
                continue
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

        if self._corridor is not None:
            self._intent_reserved = self._corridor_reserved
            return desired

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

    def _invalidate_temporal_commit(self) -> None:
        self._temporal_plan_signature = None
        self._temporal_trajectory = ()
        self._temporal_committed_until_s = 0.0
        self._temporal_goal_hold = False
        self._temporal_safety_margin_s = 0.0
        self._temporal_commit_deadline_s = 0.0

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
        self._corridor = None
        self._corridor_reserved = False
        self._corridor_admission_confirmed = False
        self._corridor_claim_ticks = 0
        self._corridor_rejoin_target_m = None
        self._coordination_wait_reason = None
        self._coordination_wait_owner_id = None
        self._known_coordination_peer_ids.clear()
        self._invalidate_temporal_commit()

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
            "coordination": {
                "state": (
                    "waiting"
                    if self._coordination_wait_reason is not None
                    else "reserved"
                    if self._corridor_reserved
                    and self._corridor_admission_confirmed
                    else "tentative"
                    if self._corridor_reserved
                    else "idle"
                ),
                "reason": self._coordination_wait_reason,
                "priority_owner_vehicle_id": (
                    self._coordination_wait_owner_id
                    if self._coordination_wait_reason is not None
                    else self._intent_priority_owner_id
                    if self._corridor_reserved
                    else None
                ),
            },
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
            self._active_task_age_ticks = 0
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
